import logging

import esm
import torch
import torch.nn as nn

from Code.src.utils.dataloader import BOS_IDX, EOS_IDX, PAD_IDX
from Code.src.models.autoencoder import ProteinSequenceAutoencoder

logger = logging.getLogger(__name__)

class LinearHead(nn.Module): #classifier 
    """Common output head used for every encoder benchmark."""
    def __init__(self, embedding_dim: int, num_classes: int = 2):
        super().__init__()    

        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # logits: [B, C]
        return self.linear(embeddings)

class MLPHead(nn.Module):
    """MLP head for protein sequence embeddings.

    Parameters
    ----------
    embedding_dim : int
        Dimension of the input embeddings.
    hidden_dim : int
        Dimension of the hidden layer.
    num_classes : int
        Number of output classes.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.mlp(embeddings)


class CNNHead(nn.Module):
    """CNN head for protein sequence embeddings.

    Parameters
    ----------
    embedding_dim : int
        Dimension of the input embeddings.
    num_filters : int
        Number of convolutional filters.
    kernel_size : int
        Size of the convolutional kernel.
    num_classes : int
        Number of output classes.
    """
    def __init__(self, embedding_dim: int, num_filters: int = 64, kernel_size: int = 3, num_classes: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError(
                "CNNHead expects residue-level embeddings with shape [B, L, D], "
                f"but received shape {tuple(embeddings.shape)}."
            )
        embeddings = embeddings.transpose(1, 2)
        features = torch.relu(self.conv(embeddings))
        pooled = self.pool(features).squeeze(-1)
        return self.fc(pooled)
    
class ESM2Embedding(nn.Module):
        """
        ESM-2 protein sequence encoder.

        Input:
            Raw protein sequences: list[str]

        Intermediate representation:
            Per-residue ESM-2 embeddings [B, L, 320]

        Output:
            Mean-pooled sequence embeddings [B, 320]
        """
        def __init__(self, model_name: str = "esm2_t6_8M_UR50D", device: str = "cpu"):
            super().__init__()
            self.device = device
            self.output_dim = 320
            self.repr_layer = 6
            self.is_frozen = False

            if model_name == "esm2_t6_8M_UR50D":
                self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
                self.model = self.model.to(self.device)
            else:
                raise ValueError(f"Unsupported ESM model name: {model_name}")
                
        def freeze_all_params(self):
            """
            Freeze all ESM-2 parameters. 
            """
            if self.model is not None:
                for param in self.model.parameters():
                    param.requires_grad = False
                self.model.eval()
                self.is_frozen = True

        def train(self, mode: bool = True):
            super().train(mode)
            if self.is_frozen:
                self.model.eval()
            return self

        def unfreeze_last_layers(self, num_layers: int = 1):
            """
            Unfreeze only the final transformer layer(s) of ESM-2.
            For esm2_t6_8M_UR50D, there are 6 transformer layers.

            Used in Stage 2:
                Earlier ESM-2 layers remain frozen to preserve general protein
                representations, while the final layer adapts to the specific
                classification task.

            Input:
                num_layers: number of final transformer layers to make trainable.

            Example:
                num_layers=1 means only the 6th/final layer is trainable for
                esm2_t6_8M_UR50D.
            """
            if self.model is None:
                return

            # Freeze everything first
            for param in self.model.parameters():
                param.requires_grad = False

            # Try to locate transformer layers in several possible attributes
            layers = None
            if hasattr(self.model, "layers"):
                layers = getattr(self.model, "layers")
            elif hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layers"):
                layers = getattr(self.model.encoder, "layers")

            if layers is None:
                logger.warning("Could not locate transformer layers on ESM model; leaving encoder frozen except layer norms if present.")
            else:
                try:
                    total_layers = len(layers)
                    # Unfreeze last N transformer layers
                    for layer in list(layers)[max(0, total_layers - num_layers):]:
                        for param in layer.parameters():
                            param.requires_grad = True
                    self.is_frozen = not any(
                        parameter.requires_grad for parameter in self.model.parameters()
                    )
                except Exception:
                    logger.warning("Failed to unfreeze specific layers; skipping.")


        def forward(self, sequences: list[str]) -> torch.Tensor:
            """
            Encode sequences using ESM-2
            Args:
                sequences: List of protein sequences
            Returns:
                Mean-pooled sequence embeddings with shape
                [batch_size, 320].
            """
            
            # alphabet may be None if esm not installed; guard for type checkers
            if self.alphabet is None:
                raise RuntimeError("ESM alphabet not available. Ensure 'esm' is installed and model loaded.")

            batch_converter = self.alphabet.get_batch_converter()

            _, _, batch_tokens = batch_converter([(str(i), seq) for i, seq in enumerate(sequences)])

            model_device = next(self.model.parameters()).device

            batch_tokens = batch_tokens.to(device=model_device, dtype=torch.long, non_blocking=True)
                # Run pretrained ESM-2 model
                #
                # Output shape:
                #     [batch_size, seq_len, 320]
            results = self.model(
                batch_tokens,
                repr_layers=[self.repr_layer],
                return_contacts=False,
            )
            embeddings = results["representations"][self.repr_layer]

            padding_idx = self.alphabet.padding_idx
            valid_mask = batch_tokens.ne(padding_idx)

            # Remove beginning and end tokens from pooling.
            valid_mask &= batch_tokens.ne(self.alphabet.cls_idx)
            valid_mask &= batch_tokens.ne(self.alphabet.eos_idx)

            valid_mask = valid_mask.unsqueeze(-1).to(embeddings.dtype)
            # valid_mask shape: [B, L, 1]

            summed_embeddings = (
                embeddings * valid_mask
            ).sum(dim=1)
            # [B, 320]

            sequence_lengths = valid_mask.sum(dim=1).clamp(min=1)
            # [B, 1]

            sequence_embeddings = summed_embeddings / sequence_lengths  # [B, 320]

            return sequence_embeddings

class TrainedAutoencoderEncoder(nn.Module):
    """
    Load a trained ProteinSequenceAutoencoder and expose only its encoder.

    Input:
        token_ids: [B, L]
        lengths: [B]

    Output:
        latent embeddings: [B, latent_dim]
    """

    def __init__(
        self,
        checkpoint_path: str,
        embedding_dim: int,
        cnn_out_channels: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int,
        kernel_size: int,
        device: str,
        freeze: bool = True,
    ):
        super().__init__()

        self.output_dim = latent_dim
        self.is_frozen = freeze

        self.autoencoder = ProteinSequenceAutoencoder(
            embedding_dim=embedding_dim,
            cnn_out_channels=cnn_out_channels,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        # Adjust this section after confirming how the partner saved it.
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.autoencoder.load_state_dict(state_dict)

        if freeze:
            for parameter in self.autoencoder.parameters():
                parameter.requires_grad = False
            self.autoencoder.eval()

        self.autoencoder.to(device)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.is_frozen:
            self.autoencoder.eval()
        return self

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        return self.autoencoder.encode(input_ids=token_ids,lengths=lengths,)

class RandomAutoencoderEncoder(nn.Module):
    """
    Expose a frozen, randomly initialized autoencoder encoder.

    Input:
        token_ids: [B, L]
        lengths: [B]

    Output:
        latent embeddings: [B, latent_dim]
    """

    def __init__(
        self,
        embedding_dim: int,
        cnn_out_channels: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int,
        kernel_size: int,
        device: str,
    ):
        super().__init__()

        self.output_dim = latent_dim

        self.autoencoder = ProteinSequenceAutoencoder(
            embedding_dim=embedding_dim,
            cnn_out_channels=cnn_out_channels,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
        )

        for parameter in self.autoencoder.parameters():
            parameter.requires_grad = False
        self.autoencoder.to(device)
        self.autoencoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.autoencoder.eval()
        return self

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        return self.autoencoder.encode(input_ids=token_ids,lengths=lengths,)

class CombinedAutoencoderESM2Encoder(nn.Module):
    """
    Combine embeddings from a trained autoencoder and ESM-2.

    Input:
        token_ids: [B, L]
        lengths: [B]
        sequences: list[str]

    Output:
        concatenated embeddings: [B, autoencoder_dim + esm_dim]
    """

    def __init__(
        self,
        # autoencoder_encoder: nn.Module,
        # esm_encoder: nn.Module,
        autoencoder_encoder: TrainedAutoencoderEncoder,
        esm_encoder: ESM2Embedding,
    ):
        super().__init__()
        self.autoencoder_encoder = autoencoder_encoder
        self.esm_encoder = esm_encoder
        self.autoencoder_normalization = nn.LayerNorm(
            autoencoder_encoder.output_dim, elementwise_affine=False
        )
        self.esm_normalization = nn.LayerNorm(
            esm_encoder.output_dim, elementwise_affine=False
        )
        self.output_dim = autoencoder_encoder.output_dim + esm_encoder.output_dim

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        sequences: list[str],
    ) -> torch.Tensor:
        autoencoder_embeddings = self.autoencoder_encoder(token_ids, lengths)
        esm_embeddings = self.esm_encoder(sequences)
        if autoencoder_embeddings.ndim != 2 or esm_embeddings.ndim != 2:
            raise ValueError(
                "CombinedAutoencoderESM2Encoder expects both encoders to return "
                "sequence-level embeddings with shape [B, D]."
            )
        if autoencoder_embeddings.size(0) != esm_embeddings.size(0):
            raise ValueError(
                "Autoencoder and ESM-2 embeddings must have the same batch size."
            )
        return torch.cat(
            [
                self.autoencoder_normalization(autoencoder_embeddings),
                self.esm_normalization(esm_embeddings),
            ],
            dim=-1,
        )

class ProteinSequenceClassifier(nn.Module):
    """Unified wrapper model combining specified embedding encoders with a linear head."""
    def __init__(
        self,
        embedding_type: str = "esm2",
        num_classes: int = 2,
        esm_model_name: str = "esm2_t6_8M_UR50D",
        head_type: str = "linear",
        autoencoder_checkpoint: str | None = None,
        autoencoder_embedding_dim: int = 128,
        autoencoder_cnn_channels: int = 128,
        autoencoder_hidden_dim: int = 256,
        autoencoder_latent_dim: int = 128,
        autoencoder_num_layers: int = 1,
        autoencoder_kernel_size: int = 3,
        device: str = "cpu",
        pad_idx: int = PAD_IDX,
    ):
        super().__init__()
        self.device = device
        self.embedding_type = embedding_type

        if embedding_type == "random_autoencoder":
            self.embedded_representation = RandomAutoencoderEncoder(
                embedding_dim=autoencoder_embedding_dim,
                cnn_out_channels=autoencoder_cnn_channels,
                hidden_dim=autoencoder_hidden_dim,
                latent_dim=autoencoder_latent_dim,
                num_layers=autoencoder_num_layers,
                kernel_size=autoencoder_kernel_size,
                device=self.device,
            ).to(self.device)
        elif embedding_type == "esm2":
            # CASE 2: Baseline 2: ESM-2 Encoder
            self.embedded_representation = ESM2Embedding(
                model_name=esm_model_name, device=self.device
            ).to(self.device)
            self.embedded_representation.freeze_all_params()
            logger.info("Using ESM-2 encoder on %s", next(self.embedded_representation.parameters()).device)

        elif embedding_type == "trained_autoencoder":
            # CASE 3: Trained Autoencoder Encoder
            if autoencoder_checkpoint is None:
                raise ValueError(
                    "--autoencoder_checkpoint is required "
                    "when embedding_type='trained_autoencoder'."
                )
            self.embedded_representation = TrainedAutoencoderEncoder(
                checkpoint_path=autoencoder_checkpoint,
                embedding_dim=autoencoder_embedding_dim,
                cnn_out_channels=autoencoder_cnn_channels,
                hidden_dim=autoencoder_hidden_dim,
                latent_dim=autoencoder_latent_dim,
                num_layers=autoencoder_num_layers,
                kernel_size=autoencoder_kernel_size,
                device=self.device,
                freeze=True,
            ).to(self.device)
            self.encoder_output_dim = self.embedded_representation.output_dim
        elif embedding_type == "autoencoder+esm2":
            # CASE 4: Combined Autoencoder + ESM-2 Encoder
            if autoencoder_checkpoint is None:
                raise ValueError(
                    "--autoencoder_checkpoint is required "
                    "when embedding_type='autoencoder+esm2'."
                )
            autoencoder_encoder = TrainedAutoencoderEncoder(
                checkpoint_path=autoencoder_checkpoint,
                embedding_dim=autoencoder_embedding_dim,
                cnn_out_channels=autoencoder_cnn_channels,
                hidden_dim=autoencoder_hidden_dim,
                latent_dim=autoencoder_latent_dim,
                num_layers=autoencoder_num_layers,
                kernel_size=autoencoder_kernel_size,
                device=self.device,
                freeze=True,
            ).to(self.device)
            esm_encoder = ESM2Embedding(
                model_name=esm_model_name, device=self.device
            ).to(self.device)
            esm_encoder.freeze_all_params()
            self.embedded_representation = CombinedAutoencoderESM2Encoder(
                autoencoder_encoder=autoencoder_encoder,
                esm_encoder=esm_encoder,
            )
        else:
            raise ValueError(f"Unsupported encoder type: {embedding_type}")

        self.output_dim = self.embedded_representation.output_dim

        if head_type == "linear":
            self.head = LinearHead(
                embedding_dim=self.output_dim,
                num_classes=num_classes,
            ).to(self.device)
        elif head_type == "mlp":
            self.head = MLPHead(
                embedding_dim=self.output_dim,
                hidden_dim=128,
                num_classes=num_classes,
            ).to(self.device)
        elif head_type == "cnn":
            raise ValueError(
                "head_type='cnn' is incompatible with the current sequence-level "
                "encoder outputs [B, D]; CNNHead requires residue-level [B, L, D]."
            )
        else:
            raise ValueError(f"Unsupported head type: {head_type}")
        self.pad_idx = pad_idx

    def forward(self, batch: dict) -> torch.Tensor:
        if self.embedding_type == "random_autoencoder":
            input_ids = batch["input_ids"].to(self.device).long()
            lengths = batch["length"].to(self.device).long()
            input_ids, lengths = self._add_autoencoder_special_tokens(input_ids, lengths)
            embeddings = self.embedded_representation(input_ids, lengths)
        elif self.embedding_type == "esm2":
            embeddings = self.embedded_representation(batch["sequence"])
        elif self.embedding_type == "trained_autoencoder":
            input_ids = batch["input_ids"].to(self.device).long()
            lengths = batch["length"].to(self.device).long()
            input_ids, lengths = self._add_autoencoder_special_tokens(input_ids, lengths)
            embeddings = self.embedded_representation(input_ids, lengths)
        elif self.embedding_type == "autoencoder+esm2":
            input_ids = batch["input_ids"].to(self.device).long()
            lengths = batch["length"].to(self.device).long()
            input_ids, lengths = self._add_autoencoder_special_tokens(input_ids, lengths)
            embeddings = self.embedded_representation(input_ids, lengths, batch["sequence"])
        else:
            raise ValueError(f"Unsupported embedding type: {self.embedding_type}")
        return self.head(embeddings)
    
    def _add_autoencoder_special_tokens(
        self, input_ids: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, padded_length = input_ids.shape
        framed_ids = torch.full(
            size=(batch_size, padded_length + 2),
            fill_value=self.pad_idx,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        framed_ids[:, 0] = BOS_IDX
        for i, length in enumerate(lengths.tolist()):
            framed_ids[i, 1 : length + 1] = input_ids[i, :length]
            framed_ids[i, length + 1] = EOS_IDX
        return framed_ids, lengths + 2
