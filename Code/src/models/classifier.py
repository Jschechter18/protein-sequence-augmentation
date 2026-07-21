import logging
from typing import List, Optional

import torch
import torch.nn as nn

from Code.src.utils.dataloader import VOCAB_SIZE, PAD_IDX
from Code.src.models.autoencoder import ProteinSequenceAutoencoder

logger = logging.getLogger(__name__)

class LinearHead(nn.Module): #classifier 
    """Common output head used for every encoder benchmark."""
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()    

        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # logits: [B, C]
        return self.linear(embeddings)
    
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
            try:
                import esm

                if model_name == "esm2_t6_8M_UR50D":

                    self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
                else:
                    raise ValueError(f"Unsupported ESM model name: {model_name}")
                
            except ImportError as error:
                raise ImportError( "The 'esm' package is required for embedding_type='esm2'.") from error

        def freeze_all_params(self):
            """
            Freeze all ESM-2 parameters. 
            """
            if self.model is not None:
                for param in self.model.parameters():
                    param.requires_grad = False 

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
            results = self.model(batch_tokens, repr_layers=[6])
            embeddings = results["representations"][6] #token representation

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

            sequence_embeddings = (summed_embeddings / sequence_lengths)  # [B, 320]

            return sequence_embeddings        

class CNNEmbedding(nn.Module):
    """
    Integer-encoded protein sequence -> CNN embedding.

    Input:
        input_ids: [B, L]

    Output:
        sequence embeddings: [B, output_dim]
    """
    def __init__(
        self,
        embedding_dim: int = 128,
        num_filters: int =64,
        kernel_sizes: Optional[List[int]] = None,):
        super().__init__()
        
        if kernel_sizes is None: 
            # Multiple kernel sizes let the CNN detect local residue patterns
            # at different scales:
            #   k=3: short motifs
            #   k=5: medium motifs
            #   k=7: longer local motifs

            kernel_sizes = [3, 5, 7]
        

        self.amino_embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=embedding_dim,
            padding_idx=PAD_IDX,
        )
        
        # Convolutional layers for different kernel sizes
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2,
                bias=False,
            )
            for k in kernel_sizes
        ])
        

        self.output_dim = num_filters * len(kernel_sizes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B,L]

        embeddings = self.amino_embedding(input_ids)  # [B, L, embedding_dim]
        embeddings = embeddings.transpose(1, 2)  # [B, embedding_dim, L]

        pooled_outputs = []

        for convolution in self.conv_layers:
            features = torch.relu(convolution(embeddings))  # [B, num_filters,, L]
            pooled = torch.amax(features, dim=2)             # [B, num_filters,]
            pooled_outputs.append(pooled)

        return torch.cat(pooled_outputs, dim=1)  # [B, F × kernels]

class AutoencoderEncoder(nn.Module):
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

        self.autoencoder.to(device)

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        return self.autoencoder.encode(input_ids=token_ids,lengths=lengths,)

class ProteinSequenceClassifier(nn.Module):
    """Unified wrapper model combining specified embedding encoders with a linear head."""
    def __init__(
        self,
        embedding_type: str = "esm2",
        num_classes: int = 2,
        esm_model_name: str = "facebook/esm2_t6_8M",
        cnn_embedding_dim: int = 128,
        cnn_num_filters: int = 128,
        autoencoder_checkpoint: str | None = None,
        autoencoder_embedding_dim: int = 128,
        autoencoder_cnn_channels: int = 128,
        autoencoder_hidden_dim: int = 256,
        autoencoder_latent_dim: int = 128,
        autoencoder_num_layers: int = 1,
        autoencoder_kernel_size: int = 3,
        device: str = "cpu",
        
    ):
        self.device = device
        super().__init__()
        self.embedding_type = embedding_type

        if embedding_type == "esm2":
            self.encoder = ESM2Embedding(model_name=esm_model_name)
            self.encoder = self.encoder.to(self.device) 
            logger.info(f"Using ESM-2 encoder: %s", next(self.encoder.parameters()).device)

            self.encoder_output_dim = 320  # ESM-2 embedding dimension

        elif embedding_type == "cnn":
            self.encoder = CNNEmbedding(
                embedding_dim=cnn_embedding_dim,
                num_filters = cnn_num_filters,
                kernel_sizes=[3, 5, 7],
            ).to(self.device)

            self.encoder_output_dim = self.encoder.output_dim

        elif embedding_type == "autoencoder":
            if autoencoder_checkpoint is None:
                raise ValueError(
                    "--autoencoder_checkpoint is required "
                    "when embedding_type='autoencoder'."
                )
            self.encoder = AutoencoderEncoder(
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
            self.encoder_output_dim = self.encoder.output_dim
        else:
            raise ValueError(
                f"Unsupported encoder type: {embedding_type}"
            )
        self.head = LinearHead(embedding_dim=self.encoder_output_dim,num_classes=num_classes,).to(self.device)

    def forward(self, batch: dict) -> torch.Tensor:
        if self.embedding_type == "esm2":
            embeddings = self.encoder(batch["sequence"])
        elif self.embedding_type == "cnn":
            input_ids = batch["input_ids"].to(self.device).long()
            embeddings = self.encoder(input_ids)
        elif self.embedding_type == "autoencoder":
            input_ids = batch["input_ids"].to(self.device).long()
            lengths = batch["length"].to(self.device).long()
            input_ids, lengths = self._add_autoencoder_special_tokens(input_ids, lengths)
            embeddings = self.encoder(input_ids, lengths)
        else:
            raise ValueError(f"Unsupported embedding type: {self.embedding_type}")
        return self.head(embeddings)
    
    def _add_autoencoder_special_tokens(
        self, input_ids: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from Code.src.utils.dataloader import BOS_IDX, EOS_IDX
        batch_size, padded_length = input_ids.shape
        framed_ids = torch.full(
            size=(batch_size, padded_length + 2),
            fill_value=PAD_IDX,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        framed_ids[:, 0] = BOS_IDX
        for i, length in enumerate(lengths.tolist()):
            framed_ids[i, 1 : length + 1] = input_ids[i, :length]
            framed_ids[i, length + 1] = EOS_IDX
        return framed_ids, lengths + 2