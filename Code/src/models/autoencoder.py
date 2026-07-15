"""Sequence autoencoder architecture for protein token reconstruction."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.dataloader import EOS_IDX, VOCAB_SIZE
import warnings


class ProteinSequenceAutoencoder(nn.Module):
    """Autoencoder for integer-encoded protein sequences.

    Supports the existing CNN+GRU encoder/GRU decoder path and a transformer encoder/decoder path.
    The model consumes batches shaped ``(batch, sequence_length)`` and returns reconstruction logits shaped ``(batch, sequence_length, vocab_size)``.
    """

    def __init__(
        self,
        embedding_dim: int,
        cnn_out_channels: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int,
        kernel_size: int,
        bidirectional: bool = True,
        grad_clip: bool = True,
        dropout: float = 0.0,
        pad_idx: int = 0,
        bos_idx: int = 2,
        eos_idx: int = EOS_IDX,
        condition_decoder_on_latent: bool = True,
        layer_type: str = "gru",
        teacher_forcing_dropout_rate: float = 0.0,
        use_decoder_positional_embeddings: bool = False,
        max_decoder_positions: int = 1024,
        max_encoder_positions: int = 1024,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        # is_autoregressive: bool = False, # really should only be when training -> optional for validation and testing
    ) -> None:
        """Initialize the protein sequence autoencoder.

        Parameters
        ----------
        layer_type : str, optional
            Type of Encoder/Decoder layer to use, by default "gru"
        embedding_dim : int, optional
            Dimension of the token embeddings, by default 128
        hidden_dim : int, optional
            Number of hidden units in the GRU layers, by default 256
        latent_dim : int, optional
            Number of dimensions in compressed latent space, by default 128
        num_layers : int, optional
            _description_, by default 1
        dropout : float, optional
            _description_, by default 0.0
        pad_idx : int, optional
            _description_, by default 0
        bos_idx : int, optional
            _description_, by default 2
        condition_decoder_on_latent : bool, optional
            If True, concatenate the latent vector to every decoder input timestep.
        techer_forcing_dropout_rate : float, optional
            The probability of dropping decoder inputs during training to encourage the model not to rely too heavily on teacher forcing.

        Raises
        ------
        ValueError
            _description_
        """
        super().__init__()
        if layer_type not in {"gru", "transformer"}:
            raise ValueError("layer_type must be 'gru' or 'transformer'")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if not 0.0 <= teacher_forcing_dropout_rate <= 1.0:
            raise ValueError("teacher_forcing_dropout_rate must be between 0 and 1")
        if max_decoder_positions < 1:
            raise ValueError("max_decoder_positions must be at least 1")
        if max_encoder_positions < 1:
            raise ValueError("max_encoder_positions must be at least 1")
        if layer_type == "transformer" and embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads for transformer layers")
        if latent_dim >= hidden_dim:
            warnings.warn("latent_dim should ideally be smaller than hidden_dim for effective compression", UserWarning)
        
        self.layer_type = layer_type
        self.bidirectional = bidirectional
        self.encoder_num_directions = 2 if self.bidirectional else 1
        self.grad_clip = grad_clip

        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.vocab_size = VOCAB_SIZE
        self.embedding_dim = embedding_dim
        self.cnn_out_channels = cnn_out_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.condition_decoder_on_latent = condition_decoder_on_latent
        self.teacher_forcing_dropout_rate = teacher_forcing_dropout_rate
        self.use_decoder_positional_embeddings = use_decoder_positional_embeddings
        self.max_decoder_positions = max_decoder_positions
        self.max_encoder_positions = max_encoder_positions
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.pad_idx)
        if self.layer_type == "transformer":
            self.encoder_position_embedding = nn.Embedding(
                self.max_encoder_positions,
                self.embedding_dim,
            )
        if self.use_decoder_positional_embeddings or self.layer_type == "transformer":
            self.decoder_position_embedding = nn.Embedding(
                self.max_decoder_positions,
                self.embedding_dim,
            )

        if self.layer_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=self.num_heads,
                dim_feedforward=self.dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.num_layers,
                enable_nested_tensor=False,
            )
            self.attention_score = nn.Linear(self.embedding_dim, 1)
            self.to_latent = nn.Linear(self.embedding_dim, self.latent_dim)
            self.from_latent = nn.Linear(self.latent_dim, self.embedding_dim)

            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.embedding_dim,
                nhead=self.num_heads,
                dim_feedforward=self.dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)
            self.output = nn.Linear(self.embedding_dim, self.vocab_size)
        else:
            # TODO: experiment with adding a CNN layer before the GRU encoder to capture local patterns in the sequence
            self.cnn = nn.Conv1d(
                in_channels=self.embedding_dim,
                out_channels=self.cnn_out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2, # to maintain sequence length
                )
            
            
            self.encoder = nn.GRU(
                self.cnn_out_channels, # self.embedding_dim is no longer output
                self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=self.bidirectional,
                )

            # self.attention_score = nn.Linear(self.hidden_dim, 1)
            self.attention_score = nn.Linear(self.hidden_dim * self.encoder_num_directions, 1)
            
            self.to_latent = nn.Linear(self.hidden_dim * self.encoder_num_directions, self.latent_dim)
            self.from_latent = nn.Linear(self.latent_dim, self.hidden_dim * self.num_layers)
            decoder_input_dim = self.embedding_dim
            if self.condition_decoder_on_latent:
                decoder_input_dim += self.latent_dim

            self.decoder = nn.GRU(
                decoder_input_dim,
                self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=False,
            )
            
            self.output = nn.Linear(self.hidden_dim, self.vocab_size)

    def encode(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode token IDs into a latent vector."""
        if self.layer_type == "transformer":
            return self._encode_transformer(input_ids, lengths=lengths)
        return self._encode_gru(input_ids, lengths=lengths)

    def _encode_gru(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode token IDs with the CNN+GRU encoder."""
        embedded: torch.Tensor = self.embedding(input_ids)
        
        x = embedded.transpose(1, 2) # x: [batch, embedding_dim, seq_len]
        x: torch.Tensor = self.cnn(x)
        x = torch.relu(x) # x: [batch, cnn_out_channels, seq_len]
        x = x.transpose(1, 2) # x: [batch, seq_len, cnn_out_channels]
        
        if lengths is None:
            encoder_outputs, _ = self.encoder(x)
        else:
            packed = pack_padded_sequence(
                x,
                lengths.detach().cpu().clamp(min=1, max=input_ids.size(1)),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_outputs, _ = self.encoder(packed)
            encoder_outputs, _ = pad_packed_sequence(
                packed_outputs,
                batch_first=True,
                total_length=input_ids.size(1),
            )
            
        # encoder_outputs: [batch, seq_len, hidden_dim]
        attention_logits: torch.Tensor = self.attention_score(encoder_outputs).squeeze(-1)
        # attention_logits: [batch, seq_len]
        
        if lengths is not None:
            max_len = input_ids.size(1)
            mask = torch.arange(max_len, device=input_ids.device).unsqueeze(0) >= lengths.unsqueeze(1)
            attention_logits = attention_logits.masked_fill(mask, float("-inf"))
        
        attention_weights: torch.Tensor = torch.softmax(attention_logits, dim=1) # attention_weights: [batch, seq_len]
        
        context: torch.Tensor = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
            ).squeeze(1)
        
        return self.to_latent(context)

    def _encode_transformer(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode token IDs with a transformer encoder."""
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.max_encoder_positions:
            raise ValueError(
                f"Encoder sequence length {sequence_length} exceeds "
                f"max_encoder_positions={self.max_encoder_positions}"
            )

        position_ids = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        encoder_inputs = self.embedding(input_ids) + self.encoder_position_embedding(position_ids)
        padding_mask = self._padding_mask(input_ids, lengths=lengths)
        encoder_outputs = self.encoder(encoder_inputs, src_key_padding_mask=padding_mask)

        # TODO: use this attention/pooling for now, but not necessary for a transformer encoder to have more attention
        attention_logits = self.attention_score(encoder_outputs).squeeze(-1)
        attention_logits = attention_logits.masked_fill(padding_mask, float("-inf"))
        attention_weights = torch.softmax(attention_logits, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return self.to_latent(context)

    def _padding_mask(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a True-for-padding mask for transformer attention."""
        if lengths is None:
            return input_ids == self.pad_idx

        max_len = input_ids.size(1)
        clipped_lengths = lengths.to(input_ids.device).clamp(min=1, max=max_len)
        return torch.arange(max_len, device=input_ids.device).unsqueeze(0) >= clipped_lengths.unsqueeze(1)

    def _initial_decoder_hidden(self, latent: torch.Tensor) -> torch.Tensor:
        """Initialize the hidden state of the decoder from the latent vector.

        Parameters
        ----------
        latent : torch.Tensor
            The latent vector from the encoder.

        Returns
        -------
        torch.Tensor
            The initial hidden state for the decoder.
        """
        hidden: torch.Tensor = self.from_latent(latent)
        hidden = hidden.view(latent.size(0), self.num_layers, self.hidden_dim)
        return hidden.transpose(0, 1).contiguous()

    def _apply_teacher_forcing_token_dropout(
        self,
        token_embeddings: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Mask whole teacher-forced residue embeddings during training."""
        if not self.training or self.teacher_forcing_dropout_rate == 0.0:
            return token_embeddings

        residue_tokens = (
            (decoder_input_ids != self.pad_idx)
            & (decoder_input_ids != self.bos_idx)
            & (decoder_input_ids != self.eos_idx)
        )
        drop_tokens = (
            torch.rand(decoder_input_ids.shape, device=decoder_input_ids.device)
            < self.teacher_forcing_dropout_rate
        ) & residue_tokens

        return token_embeddings.masked_fill(drop_tokens.unsqueeze(-1), 0.0)

    def _decoder_inputs(
        self,
        decoder_input_ids: torch.Tensor,
        latent: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """Prepare decoder inputs by optionally conditioning on the latent vector.

        Parameters
        ----------
        decoder_input_ids : torch.Tensor
            The input token IDs for the decoder.
        latent : torch.Tensor
            The latent vector from the encoder.

        Returns
        -------
        torch.Tensor
            The prepared decoder inputs.
        """
        token_embeddings = self.embedding(decoder_input_ids)
        if self.use_decoder_positional_embeddings:
            batch_size, sequence_length = decoder_input_ids.shape
            max_position = position_offset + sequence_length
            if max_position > self.max_decoder_positions:
                raise ValueError(
                    f"Decoder position {max_position} exceeds "
                    f"max_decoder_positions={self.max_decoder_positions}"
                )

            position_ids = torch.arange(
                position_offset,
                max_position,
                device=decoder_input_ids.device,
            ).unsqueeze(0).expand(batch_size, -1)
            token_embeddings = token_embeddings + self.decoder_position_embedding(position_ids)

        token_embeddings = self._apply_teacher_forcing_token_dropout(
            token_embeddings,
            decoder_input_ids,
        )
        if not self.condition_decoder_on_latent:
            return token_embeddings

        latent_repeated = latent.unsqueeze(1).expand(
            -1,
            token_embeddings.size(1),
            -1,
        )
        return torch.cat([token_embeddings, latent_repeated], dim=-1)

    def _transformer_decoder_inputs(
        self,
        decoder_input_ids: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """Prepare transformer decoder token and positional embeddings."""
        batch_size, sequence_length = decoder_input_ids.shape
        max_position = position_offset + sequence_length
        if max_position > self.max_decoder_positions:
            raise ValueError(
                f"Decoder position {max_position} exceeds "
                f"max_decoder_positions={self.max_decoder_positions}"
            )

        position_ids = torch.arange(
            position_offset,
            max_position,
            device=decoder_input_ids.device,
        ).unsqueeze(0).expand(batch_size, -1)
        token_embeddings = self.embedding(decoder_input_ids) + self.decoder_position_embedding(position_ids)
        return self._apply_teacher_forcing_token_dropout(token_embeddings, decoder_input_ids)

    def _causal_mask(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        """Return a causal attention mask for transformer decoding."""
        mask = torch.full((sequence_length, sequence_length), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def decode(
        self,
        latent: torch.Tensor,
        decoder_input_ids: torch.Tensor | None = None,
        sequence_length: int | None = None,
    ) -> torch.Tensor:
        """Decode a latent vector into token logits.

        Passing ``decoder_input_ids`` enables teacher forcing. Without it, decoding is autoregressive for ``sequence_length`` steps.
        """
        # Optional teacher enforcing
        if decoder_input_ids is None:
            if sequence_length is None:
                raise ValueError("sequence_length is required without decoder_input_ids")
            # Call actual autoregressive decoding function which feeds predictions back in at each step
            return self.decode_autoregressive(latent, sequence_length)

        if self.layer_type == "transformer":
            return self._decode_transformer(latent, decoder_input_ids)
        return self._decode_gru(latent, decoder_input_ids)
    
    def _decode_gru(
        self,
        latent: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Decode teacher-forced token IDs with the GRU decoder."""
        hidden = self._initial_decoder_hidden(latent)
        decoder_inputs = self._decoder_inputs(decoder_input_ids, latent)
        decoded, _ = self.decoder(decoder_inputs, hidden)
        return self.output(decoded)

    def _decode_transformer(
        self,
        latent: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Decode teacher-forced token IDs with the transformer decoder."""
        decoder_inputs = self._transformer_decoder_inputs(decoder_input_ids)
        causal_mask = self._causal_mask(decoder_input_ids.size(1), decoder_input_ids.device)
        padding_mask = decoder_input_ids == self.pad_idx
        memory = self.from_latent(latent).unsqueeze(1)
        decoded = self.decoder(
            tgt=decoder_inputs,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )
        return self.output(decoded)

    def decode_autoregressive(
        self,
        latent: torch.Tensor,
        max_length: int,
    ) -> torch.Tensor:
        """Decode from latent vectors by feeding each prediction into the next step."""
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        if self.layer_type == "transformer":
            return self._decode_autoregressive_transformer(latent, max_length)
        return self._decode_autoregressive_gru(latent, max_length)

    def _initial_autoregressive_input(self, latent: torch.Tensor) -> torch.Tensor:
        """Return a one-token BOS input for autoregressive decoding."""
        return torch.full(
            (latent.size(0), 1),
            self.bos_idx,
            dtype=torch.long,
            device=latent.device,
        )

    def _initial_finished_mask(self, latent: torch.Tensor) -> torch.Tensor:
        """Return the per-sequence EOS tracking mask for autoregressive decoding."""
        return torch.zeros(latent.size(0), dtype=torch.bool, device=latent.device)

    def _autoregressive_next_tokens(
        self,
        step_logits: torch.Tensor,
        finished: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedily choose next tokens and pad sequences after EOS."""
        next_tokens = step_logits.argmax(dim=-1)
        finished = finished | (next_tokens == self.eos_idx)
        next_tokens = torch.where(
            finished,
            torch.full_like(next_tokens, self.pad_idx),
            next_tokens,
        )
        return next_tokens, finished

    def _decode_autoregressive_gru(
        self,
        latent: torch.Tensor,
        max_length: int,
    ) -> torch.Tensor:
        """Autoregressively decode latent vectors with the GRU decoder."""
        hidden = self._initial_decoder_hidden(latent)
        decoder_input_ids = self._initial_autoregressive_input(latent)
        finished = self._initial_finished_mask(latent)
        logits_by_step: list[torch.Tensor] = []

        for step in range(max_length):
            step_logits, hidden = self._decode_autoregressive_gru_step(
                latent=latent,
                decoder_input_ids=decoder_input_ids,
                hidden=hidden,
                position_offset=step,
            )
            logits_by_step.append(step_logits)

            next_tokens, finished = self._autoregressive_next_tokens(step_logits, finished)
            decoder_input_ids = next_tokens.unsqueeze(1)

        return torch.stack(logits_by_step, dim=1)

    def _decode_autoregressive_gru_step(
        self,
        latent: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        hidden: torch.Tensor,
        position_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one GRU autoregressive decoding step."""
        decoder_inputs = self._decoder_inputs(
            decoder_input_ids,
            latent,
            position_offset=position_offset,
        )
        decoded, hidden = self.decoder(decoder_inputs, hidden)
        return self.output(decoded[:, -1, :]), hidden

    def _decode_autoregressive_transformer(
        self,
        latent: torch.Tensor,
        max_length: int,
    ) -> torch.Tensor:
        """Autoregressively decode latent vectors with the transformer decoder."""
        decoder_input_ids = self._initial_autoregressive_input(latent)
        finished = self._initial_finished_mask(latent)
        logits_by_step: list[torch.Tensor] = []

        for _step in range(max_length):
            step_logits = self._decode_autoregressive_transformer_step(latent, decoder_input_ids)
            logits_by_step.append(step_logits)

            next_tokens, finished = self._autoregressive_next_tokens(step_logits, finished)
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens.unsqueeze(1)], dim=1)

        return torch.stack(logits_by_step, dim=1)

    def _decode_autoregressive_transformer_step(
        self,
        latent: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run one transformer autoregressive decoding step."""
        logits = self._decode_transformer(latent, decoder_input_ids)
        return logits[:, -1, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return reconstruction logits for ``input_ids``."""
        if decoder_input_ids is None:
            decoder_input_ids = input_ids
        latent = self.encode(input_ids, lengths=lengths)
        return self.decode(latent, decoder_input_ids=decoder_input_ids)

    @torch.no_grad()
    def reconstruct(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return greedy reconstructed token IDs. Useful for comparing input to output of model.

        Parameters
        ----------
        input_ids : torch.Tensor
            Original input token IDs to be reconstructed

        Returns
        -------
        torch.Tensor
            Greedy reconstructed token IDs
        """
        return self.forward(input_ids, lengths=lengths).argmax(dim=-1)
