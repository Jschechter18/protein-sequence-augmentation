"""Sequence autoencoder architecture for protein token reconstruction."""

import torch
from torch import nn


class ProteinSequenceAutoencoder(nn.Module):
    """GRU autoencoder for integer-encoded protein sequences.

    The model consumes batches shaped ``(batch, sequence_length)`` and returns reconstruction logits shaped ``(batch, sequence_length, vocab_size)``.
    """

    def __init__(
        self,
        layer_type: str = "gru", # placeholder for future layer types
        vocab_size: int = 24,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        pad_idx: int = 0,
        bos_idx: int = 2,
    ) -> None:
        """Initialize the protein sequence autoencoder.

        Parameters
        ----------
        layer_type : str, optional
            Type of Encoder/Decoder layer to use, by default "gru"
        embedding_dim : int, optional
            _description_, by default 64
        hidden_dim : int, optional
            _description_, by default 128
        latent_dim : int, optional
            Number of dimensions in compressed latent space, by default 64
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
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.vocab_size = 24
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.pad_idx)
        self.encoder = nn.GRU(
            self.embedding_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.to_latent = nn.Linear(self.hidden_dim, self.latent_dim)
        self.from_latent = nn.Linear(self.latent_dim, self.hidden_dim * self.num_layers)
        self.decoder = nn.GRU(
            self.embedding_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.output = nn.Linear(self.hidden_dim, self.vocab_size)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode token IDs into a latent vector."""
        _, hidden = self.encoder(self.embedding(input_ids))
        return self.to_latent(hidden[-1])

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
        hidden = hidden.view(latent.size(0), self.num_layers, self.hidden_dim)
        hidden = hidden.transpose(0, 1).contiguous()
        decoded, _ = self.decoder(self.embedding(decoder_input_ids), hidden)
        return self.output(decoded)

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return reconstruction logits for ``input_ids``."""
        if decoder_input_ids is None:
            decoder_input_ids = input_ids
        latent = self.encode(input_ids)
        return self.decode(latent, decoder_input_ids=decoder_input_ids)

    @torch.no_grad()
    def reconstruct(self, input_ids: torch.Tensor) -> torch.Tensor:
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
        return self.forward(input_ids).argmax(dim=-1)

