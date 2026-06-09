"""Sequence autoencoder architecture for protein token reconstruction."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.dataloader import VOCAB_SIZE


class ProteinSequenceAutoencoder(nn.Module):
    """GRU autoencoder for integer-encoded protein sequences.

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
        dropout: float = 0.0,
        pad_idx: int = 0,
        bos_idx: int = 2,
        layer_type: str = "gru", # placeholder for future layer types
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

        Raises
        ------
        ValueError
            _description_
        """
        super().__init__()
        if layer_type != "gru":
            raise ValueError("Only layer_type='gru' is currently supported")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if latent_dim >= hidden_dim:
            raise Warning("latent_dim should ideally be smaller than hidden_dim for effective compression")
        
        self.bidirectional = bidirectional
        self.encoder_num_directions = 2 if self.bidirectional else 1

        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.vocab_size = VOCAB_SIZE
        self.embedding_dim = embedding_dim
        self.cnn_out_channels = cnn_out_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.pad_idx)
        
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
        
        # self.to_latent = nn.Linear(self.hidden_dim * self.encoder_num_directions, self.latent_dim)
        # trying a deeper mapping to latent space with nonlinearity and dropout to encourage better compression
        self.to_latent = nn.Sequential(
            nn.Linear(self.hidden_dim * self.encoder_num_directions, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, self.latent_dim)
        )
        self.from_latent = nn.Linear(self.latent_dim, self.hidden_dim * self.num_layers)
        self.decoder = nn.GRU(
            self.embedding_dim,
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
        embedded: torch.Tensor = self.embedding(input_ids)
        
        x = embedded.transpose(1, 2) # x: [batch, embedding_dim, seq_len]
        x: torch.Tensor = self.cnn(x)
        x = torch.relu(x) # x: [batch, cnn_out_channels, seq_len]
        x = x.transpose(1, 2) # x: [batch, seq_len, cnn_out_channels]
        
        if lengths is None:
            # _, hidden = self.encoder(embedded)
            # _, hidden = self.encoder(x)
            encoder_outputs, hidden = self.encoder(x)
        else:
            packed = pack_padded_sequence(
                # embedded,
                x,
                lengths.detach().cpu().clamp(min=1, max=input_ids.size(1)),
                batch_first=True,
                enforce_sorted=False,
            )
            # _, hidden = self.encoder(packed)
            packed_outputs, hidden = self.encoder(packed)
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
        
        # return self.to_latent(hidden[-1])
        return self.to_latent(context)

    def decode(
        self,
        latent: torch.Tensor,
        decoder_input_ids: torch.Tensor | None = None,
        sequence_length: int | None = None,
    ) -> torch.Tensor:
        """Decode a latent vector into token logits.

        Passing ``decoder_input_ids`` enables teacher forcing. Without it, the decoder receives repeated BOS tokens for ``sequence_length`` steps.
        """
        if decoder_input_ids is None:
            if sequence_length is None:
                raise ValueError("sequence_length is required without decoder_input_ids")
            decoder_input_ids = torch.full(
                (latent.size(0), sequence_length),
                self.bos_idx,
                dtype=torch.long,
                device=latent.device,
            )

        hidden = self.from_latent(latent)
        # hidden = hidden.view(latent.size(0), self.num_layers, self.hidden_dim)
        hidden = hidden.view(latent.size(0), self.num_layers, self.hidden_dim)
        hidden = hidden.transpose(0, 1).contiguous()
        decoded, _ = self.decoder(self.embedding(decoder_input_ids), hidden)
        return self.output(decoded)

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
