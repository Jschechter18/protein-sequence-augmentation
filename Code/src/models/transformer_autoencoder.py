"""Transformer sequence autoencoder architecture for protein token reconstruction."""

import torch
from torch import nn
from utils.dataloader import VOCAB_SIZE


class ProteinSequenceTransformerAutoencoder(nn.Module):
    """Transformer autoencoder for integer-encoded protein sequences.

    Consumes batches shaped (batch, sequence_length) and returns reconstruction
    logits shaped (batch, sequence_length, vocab_size).
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        # max_length: int  = 512,
        max_length: int  = 1024,
        pad_idx: int = 0,
        bos_idx: int = 2,
    ) -> None:
        super().__init__()

        if embedding_dim != hidden_dim:
            raise ValueError(
                "For this simple transformer implementation, embedding_dim "
                "must equal hidden_dim."
            )

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.vocab_size = VOCAB_SIZE
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx

        self.token_embedding = nn.Embedding(
            self.vocab_size,
            self.embedding_dim,
            padding_idx=self.pad_idx,
        )

        self.position_embedding = nn.Embedding(
            self.max_length,
            self.embedding_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            enable_nested_tensor=False,
        )

        self.to_latent = nn.Linear(self.hidden_dim, self.latent_dim)

        # Project latent vector into a small decoder memory.
        self.from_latent = nn.Linear(self.latent_dim, self.hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.num_layers,
        )

        self.output = nn.Linear(self.hidden_dim, self.vocab_size)

    def _add_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Add learned positional embeddings to token embeddings."""
        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.max_length:
            raise ValueError(
                f"sequence_length={sequence_length} exceeds max_length={self.max_length}"
            )

        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        ).unsqueeze(0)

        token_embedded = self.token_embedding(input_ids)
        position_embedded = self.position_embedding(positions)

        return token_embedded + position_embedded

    def _make_padding_mask(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return True where tokens should be ignored by attention."""
        if lengths is None:
            return input_ids == self.pad_idx

        batch_size, sequence_length = input_ids.shape

        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        ).unsqueeze(0)

        lengths = lengths.to(input_ids.device).unsqueeze(1)

        return positions >= lengths

    def _masked_mean_pool(
        self,
        encoded: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool token-level encoder outputs into one sequence-level vector."""
        nonpad_mask = (~padding_mask).unsqueeze(-1)

        summed = (encoded * nonpad_mask).sum(dim=1)
        counts = nonpad_mask.sum(dim=1).clamp(min=1)

        return summed / counts

    def encode(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode token IDs into a latent vector."""
        padding_mask = self._make_padding_mask(input_ids, lengths=lengths)

        embedded = self._add_positions(input_ids)

        encoded = self.encoder(
            embedded,
            src_key_padding_mask=padding_mask,
        )

        pooled = self._masked_mean_pool(encoded, padding_mask)

        return self.to_latent(pooled)

    def _causal_mask(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        """Prevent decoder positions from attending to future positions."""
        return torch.triu(
            torch.ones(sequence_length, sequence_length, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def decode(
        self,
        latent: torch.Tensor,
        decoder_input_ids: torch.Tensor | None = None,
        sequence_length: int | None = None,
    ) -> torch.Tensor:
        """Decode a latent vector into token logits.

        Passing decoder_input_ids enables teacher forcing.
        Without it, the decoder receives repeated BOS tokens.
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

        target_padding_mask = decoder_input_ids == self.pad_idx

        target_embedded = self._add_positions(decoder_input_ids)

        # Shape: (batch, 1, hidden_dim)
        # This is the memory that the decoder cross-attends to.
        memory = self.from_latent(latent).unsqueeze(1)

        target_sequence_length = decoder_input_ids.size(1)
        target_mask = self._causal_mask(
            target_sequence_length,
            decoder_input_ids.device,
        )

        decoded = self.decoder(
            tgt=target_embedded,
            memory=memory,
            tgt_mask=target_mask,
            tgt_key_padding_mask=target_padding_mask,
        )

        return self.output(decoded)

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return reconstruction logits for input_ids."""
        if decoder_input_ids is None:
            decoder_input_ids = input_ids

        latent = self.encode(input_ids, lengths=lengths)

        return self.decode(
            latent,
            decoder_input_ids=decoder_input_ids,
        )

    @torch.no_grad()
    def reconstruct(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return greedy reconstructed token IDs using teacher-forced inputs."""
        logits = self.forward(input_ids, lengths=lengths)
        return logits.argmax(dim=-1)
