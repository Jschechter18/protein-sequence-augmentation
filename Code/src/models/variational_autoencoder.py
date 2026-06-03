# """Variational sequence autoencoder for protein token reconstruction."""

# import torch
# from torch import nn


# class ProteinSequenceVariationalAutoencoder(nn.Module):
#     """GRU variational autoencoder for integer-encoded protein sequences.

#     The model consumes batches shaped ``(batch, sequence_length)`` and returns ``(reconstruction_logits, mu, logvar)`` from ``forward``.
#     """

#     def __init__(
#         self,
#         vocab_size: int = 24,
#         embedding_dim: int = 64,
#         hidden_dim: int = 128,
#         latent_dim: int = 64,
#         num_layers: int = 1,
#         dropout: float = 0.0,
#         pad_idx: int = 0,
#         bos_idx: int = 2,
#     ) -> None:
#         super().__init__()
#         if num_layers < 1:
#             raise ValueError("num_layers must be at least 1")

#         rnn_dropout = dropout if num_layers > 1 else 0.0
#         self.vocab_size = vocab_size
#         self.hidden_dim = hidden_dim
#         self.latent_dim = latent_dim
#         self.num_layers = num_layers
#         self.pad_idx = pad_idx
#         self.bos_idx = bos_idx

#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
#         self.encoder = nn.GRU(
#             embedding_dim,
#             hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=rnn_dropout,
#         )
#         self.to_mu = nn.Linear(hidden_dim, latent_dim)
#         self.to_logvar = nn.Linear(hidden_dim, latent_dim)
#         self.from_latent = nn.Linear(latent_dim, hidden_dim * num_layers)
#         self.decoder = nn.GRU(
#             embedding_dim,
#             hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=rnn_dropout,
#         )
#         self.output = nn.Linear(hidden_dim, vocab_size)

#     def encode(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         """Encode token IDs into latent distribution parameters."""
#         _, hidden = self.encoder(self.embedding(input_ids))
#         final_hidden = hidden[-1]
#         return self.to_mu(final_hidden), self.to_logvar(final_hidden)

#     def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
#         """Sample a latent vector with the reparameterization trick."""
#         if not self.training:
#             return mu
#         std = torch.exp(0.5 * logvar)
#         return mu + torch.randn_like(std) * std

#     def decode(
#         self,
#         latent: torch.Tensor,
#         decoder_input_ids: torch.Tensor | None = None,
#         sequence_length: int | None = None,
#     ) -> torch.Tensor:
#         """Decode a latent vector into token logits."""
#         if decoder_input_ids is None:
#             if sequence_length is None:
#                 raise ValueError("sequence_length is required without decoder_input_ids")
#             decoder_input_ids = torch.full(
#                 (latent.size(0), sequence_length),
#                 self.bos_idx,
#                 dtype=torch.long,
#                 device=latent.device,
#             )

#         hidden = self.from_latent(latent)
#         hidden = hidden.view(latent.size(0), self.num_layers, self.hidden_dim)
#         hidden = hidden.transpose(0, 1).contiguous()
#         decoded, _ = self.decoder(self.embedding(decoder_input_ids), hidden)
#         return self.output(decoded)

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         decoder_input_ids: torch.Tensor | None = None,
#     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """Return reconstruction logits plus latent distribution parameters."""
#         if decoder_input_ids is None:
#             decoder_input_ids = input_ids
#         mu, logvar = self.encode(input_ids)
#         latent = self.reparameterize(mu, logvar)
#         logits = self.decode(latent, decoder_input_ids=decoder_input_ids)
#         return logits, mu, logvar

#     @staticmethod
#     def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
#         """Return mean KL divergence from N(mu, var) to N(0, 1)."""
#         return -0.5 * torch.mean(
#             torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
#         )

#     @torch.no_grad()
#     def reconstruct(self, input_ids: torch.Tensor) -> torch.Tensor:
#         """Return greedy reconstructed token IDs."""
#         logits, _, _ = self.forward(input_ids)
#         return logits.argmax(dim=-1)

#     @torch.no_grad()
#     def sample(
#         self,
#         num_sequences: int,
#         sequence_length: int,
#         device: torch.device | str | None = None,
#     ) -> torch.Tensor:
#         """Sample greedy token IDs from the prior."""
#         if device is None:
#             device = next(self.parameters()).device
#         latent = torch.randn(num_sequences, self.latent_dim, device=device)
#         logits = self.decode(latent, sequence_length=sequence_length)
#         return logits.argmax(dim=-1)


# # Convinient alias naming
# VariationalAutoencoder = ProteinSequenceVariationalAutoencoder
# VAE = ProteinSequenceVariationalAutoencoder


# TODO: The above code is a placeholder for the actual variational autoencoder implementation. The current implementation is a copy of the autoencoder with some modifications for the variational aspect.