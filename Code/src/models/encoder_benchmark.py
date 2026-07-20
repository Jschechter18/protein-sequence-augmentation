import argparse
import json
import logging
import time

from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from Code.src.utils.dataloader import (create_dataloader, VOCAB_SIZE, PAD_IDX, BOS_IDX,EOS_IDX,)
from Code.src.models.autoencoder import ProteinSequenceAutoencoder


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="solubility",
                        choices=["solubility", "localization"])
    parser.add_argument("--data_dir", type=str, default="data/processed/peer")
    parser.add_argument("--results_dir", type=str, default="Code/results/encoder_benchmark")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--embedding_type", type=str, default="esm2", choices=["esm2","cnn","autoencoder"], help="Feature embedding used before the common linear output head.",)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--encoder_learning_rate", type=float, default=1e-3)
    parser.add_argument("--unfreeze_layers", type=int, default=0)
    parser.add_argument("--unfreeze_all_esm", action="store_true", help="Unfreeze the entire ESM-2 backbone.")
    parser.add_argument("--unfreeze_esm", action="store_true")
    parser.add_argument("--esm_learning_rate", type=float, default=1e-5)
    parser.add_argument("--esm_model_name", type=str, default="esm2_t6_8M_UR50D")
    parser.add_argument("--cnn_embedding_dim", type=int, default=128,)
    parser.add_argument("--cnn_num_filters", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=None, help="Number of classes for classification. If not provided, it will be inferred from the dataset.")

    parser.add_argument("--autoencoder_checkpoint", type=str, default=None)
    parser.add_argument("--autoencoder_embedding_dim", type=int, default=128)
    parser.add_argument("--autoencoder_cnn_channels", type=int,default=128)
    parser.add_argument("--autoencoder_hidden_dim", type=int, default=256)
    parser.add_argument("--autoencoder_latent_dim", type=int, default=128)
    parser.add_argument("--autoencoder_num_layers", type=int, default=1)
    parser.add_argument("--autoencoder_kernel_size", type=int, default=3)



    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--evaluate_test", action="store_true",help=("Reload the best validation checkpoint and evaluate " "once on the held-out test set after training."),)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

def get_stage_name(unfreeze_esm: bool, unfreeze_all_esm: bool, unfreeze_layers: int) -> str:
    if not unfreeze_esm and not unfreeze_all_esm:
        return "stage_0_frozen"
    if unfreeze_all_esm:
        return "stage_full"
    return f"stage{unfreeze_layers}_unfreeze_last{unfreeze_layers}"


def create_run_dir(results_dir: str, dataset: str, args) -> Path:

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.embedding_type == "esm2":
        stage = get_stage_name(
            args.unfreeze_esm,
            args.unfreeze_all_esm,
            args.unfreeze_layers,
    )
    elif args.embedding_type == "autoencoder":
        stage = "frozen_pretrained"
    else:
        stage = "trained_from_scratch"


    run_dir = (
        Path(results_dir)
        / dataset
        / args.embedding_type
        / stage
        / (f"{args.embedding_type}_{dataset}_{timestamp}")
    )

    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)

    
class LinearHead(nn.Module): #classifier 
    """Common output head used for every encoder benchmark."""
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()    

        self.linear = nn.Linear(embedding_dim, num_classes)
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # logits: [B, C]
        return self.linear(embeddings)



class ESM2Encoder(nn.Module):
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

class CNNEncoder(nn.Module):
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
        return self.autoencoder.encode(
            input_ids=token_ids,
            lengths=lengths,
        )    
        

class EmbeddingPipeline:

    def __init__(
        self,
        num_classes: int = 2,
        embedding_type: str = "esm2",
        encoder_learning_rate: float = 1e-3,
        device: str | None = None,
        esm_model_name: str = "facebook/esm2_t6_8M",
        run_dir: str | Path | None = None,
        checkpoint_dir: str | Path = "checkpoints",
        dataset: str = "solubility",
        learning_rate: float = 1e-3,
        unfreeze_esm: bool = False,
        unfreeze_layers: int = 1,
        unfreeze_all_esm: bool = False,
        esm_learning_rate: float = 1e-5,
        cnn_embedding_dim: int = 128,
        cnn_num_filters: int = 128,
        autoencoder_checkpoint: str | None = None,
        autoencoder_embedding_dim: int = 128,
        autoencoder_cnn_channels: int = 128,
        autoencoder_hidden_dim: int = 256,
        autoencoder_latent_dim: int = 128,
        autoencoder_num_layers: int = 1,
        autoencoder_kernel_size: int = 3,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.learning_rate = learning_rate
        self.encoder_learning_rate = encoder_learning_rate
        self.cnn_embedding_dim = cnn_embedding_dim
        self.cnn_num_filters = cnn_num_filters
        self.unfreeze_all_esm = unfreeze_all_esm
        self.autoencoder_checkpoint = autoencoder_checkpoint
        self.autoencoder_embedding_dim = autoencoder_embedding_dim
        self.autoencoder_cnn_channels = autoencoder_cnn_channels
        self.autoencoder_hidden_dim = autoencoder_hidden_dim
        self.autoencoder_latent_dim = autoencoder_latent_dim
        self.autoencoder_num_layers = autoencoder_num_layers
        self.autoencoder_kernel_size = autoencoder_kernel_size
        
        self.num_classes = num_classes
        self.dataset = dataset
        self.esm_learning_rate = esm_learning_rate
        self.unfreeze_layers = unfreeze_layers
        self.unfreeze_esm = unfreeze_esm     # unfreeze last N layers
        self.run_dir = Path(run_dir) if run_dir is not None else Path(".")
        self.run_dir.mkdir(parents=True, exist_ok=True)

    
        self.embedding_type = embedding_type

        if embedding_type == "esm2":
            self.encoder = ESM2Encoder(model_name=esm_model_name)
            self.encoder = self.encoder.to(self.device) 
            logger.info(f"Using ESM-2 encoder: %s", next(self.encoder.parameters()).device)

            self.encoder_output_dim = 320  # ESM-2 embedding dimension

        elif embedding_type == "cnn":
            self.encoder = CNNEncoder(
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

        self.checkpoint_dir = ( Path(checkpoint_dir) / "encoder_benchmark" / self.run_dir.name )

        self.checkpoint_dir.mkdir( parents=True, exist_ok=True,)


        if self.embedding_type == "esm2":
            # Begin with the full ESM-2 encoder frozen.
            self.encoder.freeze_all_params() # type: ignore

            if unfreeze_all_esm:
                logger.info(
                    "Unfreezing entire ESM-2 backbone."
                )

                for parameter in self.encoder.parameters():
                    parameter.requires_grad = True

            elif unfreeze_esm:
                logger.info(
                    "Unfreezing last %d ESM transformer layer(s).",
                    unfreeze_layers,
                )

                self.encoder.unfreeze_last_layers(
                    int(unfreeze_layers)
                ) # type: ignore
            
        encoder_parameters = [
            parameter
            for parameter in self.encoder.parameters()
            if parameter.requires_grad
            ]

        parameter_groups = [
            {
                "params": self.head.parameters(),
                "lr": learning_rate,
            }
        ]

        if encoder_parameters:
            parameter_groups.append({
                "params": encoder_parameters,
                "lr": (
                    esm_learning_rate
                    if embedding_type == "esm2"
                    else encoder_learning_rate
                ),
            })

        self.optimizer = optim.Adam(
            parameter_groups
        )

        trainable_encoder = sum(
            p.numel()
            for p in self.encoder.parameters()
            if p.requires_grad
        )

        trainable_classifier = sum(
            p.numel()
            for p in self.head.parameters()
            if p.requires_grad
        )

        logger.info(
            "Trainable %s encoder parameters: %s", self.embedding_type.upper(), f"{trainable_encoder:,}",
        )

        logger.info(
            "Trainable head parameters: %s", f"{trainable_classifier:,}"
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def add_autoencoder_special_tokens(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert classification token IDs into the format used
        during autoencoder training:

            residues -> BOS + residues + EOS

        Parameters
        ----------
        input_ids:
            Padded classification token IDs [B, L].

        lengths:
            Number of non-padding residue tokens [B].

        Returns
        -------
        framed_ids:
            Token IDs with BOS and EOS added [B, L + 2].

        framed_lengths:
            Original lengths plus two [B].
        """
        batch_size, padded_length = input_ids.shape

        framed_ids = torch.full(
            size=(batch_size, padded_length + 2),
            fill_value=PAD_IDX,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        # Every sequence begins with BOS.
        framed_ids[:, 0] = BOS_IDX

        for row_index, sequence_length in enumerate(
            lengths.tolist()
        ):
            # Copy the original amino-acid tokens after BOS.
            framed_ids[
                row_index,
                1 : sequence_length + 1,
            ] = input_ids[
                row_index,
                :sequence_length,
            ]

            # Place EOS immediately after the final residue.
            framed_ids[
                row_index,
                sequence_length + 1,
            ] = EOS_IDX

        framed_lengths = lengths + 2

        return framed_ids, framed_lengths


    def encode_batch(
        self,
        batch: Dict,
    ) -> torch.Tensor:

        if self.embedding_type == "esm2":
            sequences = batch["sequence"]

            return self.encoder(sequences)

        input_ids = (
            batch["input_ids"]
            .to(self.device)
            .long()
        )

        if self.embedding_type == "cnn":
            return self.encoder(input_ids)

        if self.embedding_type == "autoencoder":
            lengths = (
                batch["length"]
                .to(self.device)
                .long()
            )

            input_ids, lengths = (
                self.add_autoencoder_special_tokens(
                    input_ids=input_ids,
                    lengths=lengths,
                )
            )

            return self.encoder(
                input_ids,
                lengths,
            )

        raise ValueError(
            f"Unsupported encoder type: "
            f"{self.embedding_type}"
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.head.train()

        if any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.train()
        else:
            self.encoder.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc="Training")


        for batch in progress_bar:
            labels = batch["label"].to(self.device).long()
            
            # Encode sequences
            
            embeddings = self.encode_batch(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.head(embeddings)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            trainable_parameters = [
                parameter
                for parameter in list(self.encoder.parameters())
                + list(self.head.parameters())
                if parameter.requires_grad
            ]
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            progress_bar.set_postfix({
                'loss': total_loss / (total / labels.size(0)),
                'acc': correct / total
            })
        
        return {
            "loss": float(total_loss / len(train_loader)),
            "accuracy": float(accuracy_score(all_labels, all_preds)), 
            "f1": float(f1_score(all_labels, all_preds, average="macro")),
            "precision": float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
            "recall": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),   
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.encoder.eval()
        self.head.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                labels = batch["label"].to(self.device).long()
                
                # Encode sequences
                embeddings = self.encode_batch(batch)
                
                # Forward pass
                logits = self.head(embeddings)
                loss = self.criterion(logits, labels)
                
                # Metrics
                total_loss += loss.item()

                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return {
                "loss": float(total_loss / len(val_loader)),
                "accuracy": float(accuracy_score(all_labels, all_preds)),
                "f1": float(f1_score(all_labels, all_preds, average="macro")),
                "precision": float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
                "recall": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
            }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        early_stopping_patience: int = 5
    ) -> Dict[str, list]:
        """
        Train the model
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
        
        Returns:
            Dictionary with training history
        """
        history = {
            "epoch": [],
            "train_loss": [],
            "train_accuracy": [],
            "train_f1": [],
            "train_precision": [],
            "train_recall": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "val_precision": [],
            "val_recall": [],
        }
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            history["epoch"].append(epoch + 1)

            # Train
            train_metrics = self.train_epoch(train_loader)
            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["train_f1"].append(train_metrics["f1"])
            history["train_precision"].append(train_metrics["precision"])
            history["train_recall"].append(train_metrics["recall"])
    

            # Validate
            val_metrics = self.validate(val_loader)
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["f1"])
            history["val_precision"].append(val_metrics["precision"])
            history["val_recall"].append(val_metrics["recall"])
            
            
            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            # Early stopping
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.save_checkpoint()
            else:
                self.patience_counter += 1

                if self.patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
        self.save_final_checkpoint()
    
        pd.DataFrame(history).to_csv(
            self.run_dir / "training_history.csv",
            index=False
        )
        
        hyperparameters = {
            "model": (
                f"{self.embedding_type} encoder "
                "linear output head"
            ),
            "embedding_type": self.embedding_type,
            "device": self.device,
            "dataset": self.dataset,
            "num_classes": self.num_classes,
            "learning_rate": self.learning_rate,
            "epochs": epochs,
            "early_stopping_patience": early_stopping_patience,
            "encoder_output_dim": self.encoder_output_dim,
        }

        if self.embedding_type == "cnn":
            hyperparameters.update({
                "encoder_learning_rate": (
                    self.encoder_learning_rate
                ),
                "cnn_embedding_dim": (
                    self.cnn_embedding_dim
                ),
                "cnn_num_filters": (
                    self.cnn_num_filters
                ),
                "cnn_kernel_sizes": [3, 5, 7],
                "training_strategy": (
                    "trained_from_scratch"
                ),
            })

        elif self.embedding_type == "esm2":
            if self.unfreeze_all_esm:
                training_strategy = (
                    "full_backbone_unfrozen"
                )
            elif self.unfreeze_esm:
                training_strategy = (
                    f"last_{self.unfreeze_layers}_layers_unfrozen"
                )
            else:
                training_strategy = (
                    "frozen_backbone"
                )

            hyperparameters.update({
                "esm_model": "esm2_t6_8M_UR50D",
                "esm_learning_rate": (
                    self.esm_learning_rate
                ),
                "unfreeze_all_esm": (
                    self.unfreeze_all_esm
                ),
                "unfreeze_esm": (
                    self.unfreeze_esm
                ),
                "unfreeze_layers": (
                    self.unfreeze_layers
                    if self.unfreeze_esm
                    else 0
                ),
                "training_strategy": training_strategy,
            })

        elif self.embedding_type == "autoencoder":
            hyperparameters.update({
                "autoencoder_checkpoint": (
                    self.autoencoder_checkpoint
                ),
                "autoencoder_embedding_dim": (
                    self.autoencoder_embedding_dim
                ),
                "autoencoder_cnn_channels": (
                    self.autoencoder_cnn_channels
                ),
                "autoencoder_hidden_dim": (
                    self.autoencoder_hidden_dim
                ),
                "autoencoder_latent_dim": (
                    self.autoencoder_latent_dim
                ),
                "autoencoder_num_layers": (
                    self.autoencoder_num_layers
                ),
                "autoencoder_kernel_size": (
                    self.autoencoder_kernel_size
                ),
                "training_strategy": (
                    "frozen_pretrained"
                ),
            })
        history_json = {
            "hyperparameters": hyperparameters,

            "summary": {
                "best_val_loss": self.best_val_loss,
                "best_val_accuracy": max(
                    history["val_accuracy"]
                ),
                "best_val_f1": max(
                    history["val_f1"]
                ),
                "best_val_precision": max(
                    history["val_precision"]
                ),
                "best_val_recall": max(
                    history["val_recall"]
                ),

                "final_train_loss": (
                    history["train_loss"][-1]
                ),
                "final_train_accuracy": (
                    history["train_accuracy"][-1]
                ),
                "final_train_f1": (
                    history["train_f1"][-1]
                ),
                "final_train_precision": (
                    history["train_precision"][-1]
                ),
                "final_train_recall": (
                    history["train_recall"][-1]
                ),
                "final_val_loss": (
                    history["val_loss"][-1]
                ),
                "final_val_accuracy": (
                    history["val_accuracy"][-1]
                ),
                "final_val_f1": (
                    history["val_f1"][-1]
                ),
                "final_val_precision": (
                    history["val_precision"][-1]
                ),
                "final_val_recall": (
                    history["val_recall"][-1]
                ),

                "best_epoch": (
                    history["val_loss"]
                    .index(self.best_val_loss)
                    + 1
                ),
                "epochs_trained": len(
                    history["epoch"]
                ),
            },
        }
            
        save_json(history_json, self.run_dir / "history.json")

        return history
    def load_checkpoint(self) -> None:
        
        encoder_path = (self.checkpoint_dir / "best_encoder.pt")
        classifier_path = (self.checkpoint_dir/ "best_linear_head.pt")

        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_path}")
        
        if not classifier_path.exists():
            raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_path}")

        self.encoder.load_state_dict(
        torch.load(
            encoder_path,
            map_location=self.device,
        )
        )

        logger.info(f"Best encoder checkpoint loaded from {encoder_path}")

        self.head.load_state_dict(
        torch.load(
            classifier_path,
            map_location=self.device,
        )
        )

        logger.info(f"Best head checkpoint loaded from {classifier_path}")



    def evaluate_test(self, test_loader: DataLoader,) -> Dict[str, float]:
        """ 
        Evaluate the model on the test set.

        saves: test_predictions.csv & test_metrics.json
        """
        self.encoder.eval()
        self.head.eval()

        all_sequences = []
        all_true_labels = []
        all_predictions = []
        all_probabilities = []
        all_confidences = []
        all_sequence_lengths = []

        total_loss = 0.0
    

        with torch.inference_mode():
            for batch in tqdm(test_loader, desc="Test evaluation"):
                labels = (batch["label"].to(self.device).long())
                # Raw sequences -> ESM-2 residue representations.
                embeddings = self.encode_batch(batch)
                # ESM embeddings -> selected classifier head.
                logits = self.head(embeddings)

                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                probabilities = torch.softmax(logits, dim =1)
                confidence = torch.max(probabilities, dim=1).values

                if "sequence" in batch:
                    sequences = batch["sequence"]
                    lengths = [len(seq) for seq in sequences]
                else:
                    # For CNN/Autoencoder, fallback or flag as encoded
                    sequences = ["<encoded_sequence>"] * labels.size(0)
                    lengths = batch["length"].cpu().tolist() if "length" in batch else [0] * labels.size(0)

                all_sequences.extend(sequences)
                all_sequence_lengths.extend(
                    [len(seq) for seq in sequences]
                )

                all_true_labels.extend(
                labels.cpu().tolist()
                )

                all_predictions.extend(
                    torch.argmax(logits, dim=1).cpu().tolist()
                )

                all_confidences.extend(
                    confidence.cpu().tolist()
                )

                all_probabilities.extend(
                    probabilities.cpu().tolist()
                )

        true_labels = np.asarray(all_true_labels)
        predictions = np.asarray(all_predictions)
        probabilities = np.asarray(all_probabilities)

        test_metrics = {
            "test_loss": float(
                total_loss / len(test_loader)
            ),
            "test_accuracy": float(
                accuracy_score(
                    true_labels,
                    predictions,
                )
            ),
            "test_f1": float(
                f1_score(
                    true_labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "test_precision": float(
                precision_score(
                    true_labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "test_recall": float(
                recall_score(
                    true_labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "num_test_samples": int(
                len(true_labels)
            ),
        }

        prediction_data = {
            "sample_index": np.arange(
                len(true_labels)
            ),
            "sequence": all_sequences,
            "sequence_length": all_sequence_lengths,
            "true_label": true_labels,
            "predicted_label": predictions,
            "confidence": all_confidences,
            "correct": (
                true_labels == predictions
            ),
        }

        # Add one probability column for every class.
    
        for class_index in range(self.num_classes):
            prediction_data[
                f"probability_class_{class_index}"
            ] = probabilities[:, class_index]

        predictions_df = pd.DataFrame(
            prediction_data
        )

        predictions_path = (
            self.run_dir
            / "test_predictions.csv"
        )

        metrics_path = (
            self.run_dir
            / "test_metrics.json"
        )

        predictions_df.to_csv(
            predictions_path,
            index=False,
        )

        save_json(
            test_metrics,
            metrics_path,
        )

        logger.info(
            f"Test predictions saved to {predictions_path}"
        )

        logger.info(
            f"Test metrics saved to {metrics_path}"
        )

        logger.info(
            "Test Loss: %.4f | "
            "Test Accuracy: %.4f | "
            "Test F1: %.4f | "
            "Test Precision: %.4f | "
            "Test Recall: %.4f",
            test_metrics["test_loss"],
            test_metrics["test_accuracy"],
            test_metrics["test_f1"],
            test_metrics["test_precision"],
            test_metrics["test_recall"],
        )

        return test_metrics


    def save_checkpoint(self):
        encoder_path = (self.checkpoint_dir / "best_encoder.pt")

        head_path = (self.checkpoint_dir/ "best_linear_head.pt")

        torch.save(self.encoder.state_dict(), encoder_path,)

        torch.save(self.head.state_dict(), head_path,)

        logger.info(f"Best encoder checkpoint saved to {encoder_path}")
        logger.info(f"Best head checkpoint saved to {head_path}")

        

    def save_final_checkpoint(self):
        """Save final model state at end of training."""

        encoder_path = (self.checkpoint_dir / "final_encoder.pt")
        head_path = (self.checkpoint_dir / "final_linear_head.pt")

        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.head.state_dict(), head_path)

        logger.info(f"Final encoder checkpoint saved to {encoder_path}")
        logger.info(f"Final head checkpoint saved to {head_path}")

    def predict(self, sequences: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on new sequences
        
        Args:
            sequences: List of protein sequences
        
        Returns:
            predictions (class labels) and probabilities
        """
        self.head.eval()
        
        with torch.no_grad():
            embeddings = self.encoder(sequences)
            embeddings = embeddings.to(self.device)
            logits = self.head(embeddings)
            probs = torch.softmax(logits, dim=1)
        
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        probabilities = probs.cpu().numpy()
    
        
        return predictions, probabilities


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    run_dir = create_run_dir(args.results_dir, args.dataset, args)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if args.num_classes is None:
        args.num_classes = (
            10
            if args.dataset == "localization"
            else 2
        )

    if (
        args.embedding_type != "esm2"
        and (
            args.unfreeze_esm
            or args.unfreeze_all_esm
        )
    ):
        raise ValueError(
            "--unfreeze_esm and --unfreeze_all_esm "
            "can only be used with "
            "--embedding_type esm2."
        )
    
    if args.embedding_type == "esm2":
        data_encoding = "raw"
    else:
        data_encoding = "char"

    config = vars(args)
    config["run_dir"] = str(run_dir)
    config["device"] = device
    config["model"] = (f"{args.embedding_type} encoder " + "linear output head")
    config["encoding"] = data_encoding
    

    save_json(config, run_dir / "config.json")

    logger.info("Protein Encoder + Linear Head Benchmark")
    logger.info("embedding_type: %s", args.embedding_type)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Device: {device}")
    logger.info(f"Run directory: {run_dir}")
    
    strategy = None
        
    if args.embedding_type == "esm2":
        if args.unfreeze_all_esm:
            strategy = ("Full ESM-2 backbone unfrozen")
        elif args.unfreeze_esm:
            strategy = (
                f"Last {args.unfreeze_layers} "
                "ESM layer(s) unfrozen"
            )
        else:
            strategy = "Frozen ESM-2 backbone"

    elif args.embedding_type == "cnn":
        strategy = (
            "CNN encoder trained from scratch"
        )

    elif args.embedding_type == "autoencoder":
        strategy = (
            "Pretrained autoencoder frozen"
        )

    logger.info("Encoder training strategy: %s",strategy,)

    train_loader = create_dataloader(
        task=args.dataset,
        split="train",
        data_dir=args.data_dir,
        mode="classification",
        encoding=data_encoding,
        batch_size=args.batch_size,
        shuffle=True,
        use_cache=False,
    )

    val_loader = create_dataloader(
        task=args.dataset,
        split="valid",
        data_dir=args.data_dir,
        mode="classification",
        encoding=data_encoding,
        batch_size=args.batch_size,
        shuffle=False,
        use_cache=False,
    )
    test_loader = None

    if args.evaluate_test:
        test_loader = create_dataloader(
            task=args.dataset,
            split="test",
            data_dir=args.data_dir,
            mode="classification",
            encoding=data_encoding,
            batch_size=args.batch_size,
            shuffle=False,
            use_cache=False,
        )

    pipeline = EmbeddingPipeline(
        num_classes=args.num_classes,
        device=device,
        run_dir=run_dir,
        dataset=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        learning_rate=args.learning_rate,
        embedding_type=args.embedding_type,
        encoder_learning_rate=args.encoder_learning_rate,
        esm_model_name=args.esm_model_name,
        esm_learning_rate=args.esm_learning_rate,
        unfreeze_esm=args.unfreeze_esm,
        unfreeze_layers=args.unfreeze_layers,
        unfreeze_all_esm=args.unfreeze_all_esm,
        cnn_embedding_dim=args.cnn_embedding_dim,
        cnn_num_filters=args.cnn_num_filters,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        autoencoder_embedding_dim=args.autoencoder_embedding_dim,
        autoencoder_cnn_channels=args.autoencoder_cnn_channels,
        autoencoder_hidden_dim=args.autoencoder_hidden_dim,
        autoencoder_latent_dim=args.autoencoder_latent_dim,
        autoencoder_num_layers=args.autoencoder_num_layers,
        autoencoder_kernel_size=args.autoencoder_kernel_size,
    )

    start_time = time.time()

    pipeline.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
    )

    elapsed_time = time.time() -start_time
    logger.info(f"Training completed in {elapsed_time:.2f} seconds.")

    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    runtime_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

    config["total_runtime_seconds"] = elapsed_time
    config["total_runtime_formatted"] = runtime_str
    save_json(config, run_dir / "config.json")

    if args.evaluate_test:
        if test_loader is None:
            raise RuntimeError(
                "Test evaluation requested, but test_loader is None."
            )
        pipeline.load_checkpoint()

        pipeline.evaluate_test(test_loader)


if __name__ == "__main__":
    main()
