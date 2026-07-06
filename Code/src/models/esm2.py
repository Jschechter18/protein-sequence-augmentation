import argparse
import json
import logging

from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from Code.src.utils.dataloader import create_dataloader


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="solubility",
                        choices=["solubility", "localization"])
    parser.add_argument("--data_dir", type=str, default="data/processed/peer")
    parser.add_argument("--results_dir", type=str, default="Code/results/esm2")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--unfreeze_esm", action="store_true")
    parser.add_argument("--classifier_head", type=str, default="cnn", choices=["cnn","lstm","gru"])
    parser.add_argument("--unfreeze_layers", type=int, default=0)
    parser.add_argument("--esm_learning_rate", type=float, default=1e-5)
    parser.add_argument("--cnn_checkpoint", type=str, default= None)

    parser.add_argument("--esm_model_name", type=str, default="esm2_t6_8M_UR50D")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

def create_run_dir(results_dir, dataset):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(results_dir) / dataset / f"esm2_{dataset}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)

class ESM2Encoder(nn.Module):
    """ESM-2 protein sequence encoder
    Input:
        List[str]

        Example:
        [
            "MKTLLILAV",
            "AAAGGGVVV"
        ]

    Output:
        Tensor of shape:
            (batch_size, sequence_length, 320)
    Notes:
        Uses pretrained ESM-2 model.
        Model weights are frozen during training.
    
    """
    
    def __init__(self, model_name: str = "esm2_t6_8M_UR50D"):
        super().__init__()
        try:
            import esm # type: ignore

            if model_name == "esm2_t6_8M_UR50D":

                self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            else:
                raise ValueError(f"Unsupported ESM model name: {model_name}")
            
        except ImportError:
            logger.warning("ESM is not installed. Using random fallback embeddings.")
            self.model = None
            self.alphabet = None
    def freeze(self):
        """
        Freeze all ESM-2 parameters.

        Used in Stage 1:
            ESM-2 acts as a fixed feature extractor.
            Only the CNN classifier is updated during training.
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

        # Also unfreeze final layer norm if available
        #if hasattr(self.model, "emb_layer_norm_after"):
            #for param in self.model.emb_layer_norm_after.parameters():
                #param.requires_grad = True
    
    def forward(self, sequences: list) -> torch.Tensor:
        """
        Encode sequences using ESM-2
        Args:
            sequences: List of protein sequences
        Returns:
            Embeddings of shape (batch_size, seq_len, embedding_dim)
        """
        if self.model is None:
            # Fallback: return random embeddings
            max_len = max(len(seq) for seq in sequences)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return torch.randn(len(sequences), max_len, 320, device=device)
        
        batch_converter = self.alphabet.get_batch_converter() # type: ignore

        batch_labels, batch_strs, batch_tokens = batch_converter(
                [(str(i), seq) for i, seq in enumerate(sequences)]
            )

        batch_tokens = batch_tokens.to(next(self.model.parameters()).device)
            # Run pretrained ESM-2 model
            #
            # Output shape:
            #     [batch_size, seq_len, 320]
        results = self.model(batch_tokens, repr_layers=[6])
        embeddings = results["representations"][6]
        
        return embeddings
        

class CNN1DClassifier(nn.Module):
    """
    Multi-kernel 1D-CNN classifier applied to ESM-2 embeddings.

    Input to forward():
        x shape [B, L, 320]
        B = batch size
        L = sequence length
        320 = ESM embedding dimension

    Output from forward():
        logits shape [B, C]
        C = number of classes

    Architecture:
        ESM embeddings
            -> parallel Conv1D branches with kernel sizes 3, 5, 7
            -> batch norm + ReLU
            -> global max pooling
            -> concatenate branch outputs
            -> fully connected classifier
    """
    def __init__(
        self,
        input_dim: int = 320,
        num_classes: int = 2,
        kernel_sizes: list = None, # type: ignore
        num_filters: int = 64,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        if kernel_sizes is None: 
            # Multiple kernel sizes let the CNN detect local residue patterns
            # at different scales:
            #   k=3: short motifs
            #   k=5: medium motifs
            #   k=7: longer local motifs

            kernel_sizes = [3, 5, 7]
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Convolutional layers for different kernel sizes
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2
            )
            for k in kernel_sizes
        ])
        
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in kernel_sizes
        ])
        
        # Pooling and fully connected layers
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected head
        fc_input_dim = num_filters * len(kernel_sizes)
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Transpose to (batch_size, input_dim, seq_len) for Conv1d
        # Input from ESM-2:
        #     x shape [B, L, 320]
        #
        # PyTorch Conv1d expects:
        #     [B, channels, sequence_length]
        #
        # After transpose:
        #     x shape [B, 320, L]
        x = x.transpose(1, 2)
        
        # Apply convolutions with different kernel sizes
        conv_outputs = []

        for conv, bn in zip(self.conv_layers, self.bns):
            # Apply one convolution branch.
            # Input shape:  [B, 320, L]
            # Output shape: [B, 64, L] because out_channels=num_filters.
            out = conv(x)

            # Normalize each convolution branch to stabilize training.
            out = bn(out)

            # Nonlinear activation so the CNN can learn complex patterns.
            out = torch.relu(out)

            # Global max pooling to reduce sequence dimension.
            # Input shape:  [B, 64, L]
            # Output shape: [B, 64, 1]  
            out = self.global_pool(out)

            # Remove final singleton dimension.
            # Shape: [B, 64, 1] -> [B, 64]
            out = out.squeeze(-1)
            conv_outputs.append(out)

        
        # Concatenate outputs from all convolution branches
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        
        # Fully connected layers
        logits = self.fc(x)
        return logits

class RNNClassifier(nn.Module):
    """
    RNN classifier (LSTM or GRU) applied to ESM-2 embeddings.

    Input to forward():
        x shape [B, L, 320]
        B = batch size
        L = sequence length
        320 = ESM embedding dimension

    Output from forward():

        logits: Tensor of shape [B,C]
        B = batch size
        C = number of classes
    """

    def __init__(
        self,
        input_dim: int = 320,
        hidden_dim: int = 128,
        num_classes: int = 2,
        num_layers: int = 1,
        dropout_rate: float = 0.3,
        bidirectional: bool = True,
        rnn_type: str = "gru"  # or "lstm"
    ):
        super().__init__()

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        rnn_class = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_class(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(rnn_output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # RNN expects input shape [B, L, input_dim]
        # ESM-2 embeddings are already in this shape.
        rnn_out, hidden = self.rnn(x)

        if self.rnn_type == "lstm":
            # For LSTM, hidden is a tuple (h_n, c_n)
            hidden_state, cell_state = hidden
        else:
            hidden_state = hidden

        if self.bidirectional: 
            #last forward and backward hidden states
            forward_hidden = hidden_state[-2, :, :]
            backward_hidden = hidden_state[-1, :, :]
            last_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        else:
            last_hidden = hidden_state[-1, :, :]    
            
        # last_hidden shape: 
        # [B, hidden_dim * * 2] if bidirectional
        # [B, hidden_dim] if unidirectional
        last_hidden = self.dropout(last_hidden)

        logits = self.fc(last_hidden)
        return logits




class ESM2CNNPipeline:
    """
    End-to-end training pipeline combining:
        1. ESM-2 encoder
        2. 1D-CNN classifier
        3. optimizer/loss
        4. training loop
        5. validation loop
        6. checkpoint and metrics saving

    Stage 0:
        unfreeze_esm=False
        ESM-2 frozen, CNN trainable.

    Stage 1:
        unfreeze_esm=True
        final ESM-2 layer(s) trainable with low learning rate,
        CNN trainable with classifier learning rate.
    """
    def __init__(
        self,
        num_classes: int = 2,
        learning_rate: float = 1e-5,
        device: str | None = None,
        esm_model_name: str = "facebook/esm2_t6_8M",
        run_dir: str | Path | None = None,
        checkpoint_dir: str | Path = "checkpoints",
        unfreeze_esm: bool = False,
        unfreeze_layers: int = 1,
        esm_learning_rate: float = 1e-5,
        cnn_checkpoint: str | None = None,
        classifier_head: str = "cnn"  # or "lstm" or "gru"
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.esm_learning_rate = esm_learning_rate
        self.unfreeze_layers = unfreeze_layers
        self.unfreeze_esm = unfreeze_esm
        self.classifier_head = classifier_head
        self.run_dir = Path(run_dir) if run_dir is not None else Path(".")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        
        # Initialize pretrained ESM-2 encoder.
        # Input: raw sequence strings.
        # Output: embeddings [B, L, 320].
        self.encoder = ESM2Encoder(model_name=esm_model_name).to(self.device)

        # Initialize classifier.
        # Input: ESM embeddings [B, L, 320].
        # Output: class logits [B, num_classes].
        self.classifier_head = classifier_head

        if classifier_head == "cnn":
            self.classifier = CNN1DClassifier(
                input_dim=320,
                num_classes=num_classes
            ).to(self.device)

        elif classifier_head == "gru":
            self.classifier = RNNClassifier(
                input_dim=320,
                hidden_dim=128,
                num_classes=num_classes,
                num_layers=1,
                dropout_rate=0.3,
                bidirectional=True,
                rnn_type="gru"
            ).to(self.device)

        elif classifier_head == "lstm":
            self.classifier = RNNClassifier(
                input_dim=320,
                hidden_dim=128,
                num_classes=num_classes,
                num_layers=1,
                dropout_rate=0.3,
                bidirectional=True,
                rnn_type="lstm"
            ).to(self.device)

        else:
            raise ValueError(f"Unsupported classifier_head: {classifier_head}")

        self.checkpoint_dir = Path(checkpoint_dir)

        run_name = self.run_dir.name

        self.cnn_checkpoint_dir = (self.checkpoint_dir / "cnn" / run_name)
        self.esm_checkpoint_dir = (self.checkpoint_dir / "esm2" / run_name)

        self.cnn_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.esm_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if unfreeze_esm:
            self.encoder.unfreeze_last_layers(num_layers=unfreeze_layers)
        else:
            self.encoder.freeze()

        if cnn_checkpoint:
            checkpoint_path = Path(cnn_checkpoint)

            if checkpoint_path.exists():
                self.classifier.load_state_dict(
                    torch.load(checkpoint_path, map_location=self.device))
                logger.info(f"Loaded CNN checkpoint from {checkpoint_path}")

            else: 
                logger.warning(f"CNN checkpoint not found: {checkpoint_path}. Starting from scratch.")
        
        # Optimizer (only for classifier since encoder is frozen)
        self.optimizer = optim.Adam([
            {
                "params": self.classifier.parameters(),
                "lr": learning_rate
            },
            {
                "params": filter(lambda p: p.requires_grad, self.encoder.parameters()),
                "lr": esm_learning_rate
            }
        ])

        trainable_encoder = sum(
            p.numel()
            for p in self.encoder.parameters()
            if p.requires_grad
        )

        trainable_classifier = sum(
            p.numel()
            for p in self.classifier.parameters()
            if p.requires_grad
        )

        logger.info(
            f"Trainable ESM parameters: {trainable_encoder:,}"
        )

        logger.info(
            f"Trainable classifier parameters: {trainable_classifier:,}"
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_loss = float('inf')
        self.patience_counter = 0


    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.classifier.train()

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
            sequences = batch["sequence"]
            labels = batch["label"].to(self.device).long()
            
            # Encode sequences
            
            embeddings = self.encoder(sequences)
            embeddings = embeddings.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.classifier(embeddings)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 1.0)
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
        self.classifier.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                sequences = batch["sequence"]
                labels = batch["label"].to(self.device).long()
                
                # Encode sequences
                embeddings = self.encoder(sequences)
                embeddings = embeddings.to(self.device)
                
                # Forward pass
                logits = self.classifier(embeddings)
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
        
        history_json = {
            "hyperparameters": {
                "model": "ESM2 + CNN",
                "esm_model": "esm2_t6_8M_UR50D",
                "device": self.device,
                "dataset": self.run_dir.name,
                "num_classes": self.num_classes,
                "learning_rate": self.learning_rate,
                "esm_learning_rate": self.esm_learning_rate,
                "epochs": epochs,
                "early_stopping_patience": early_stopping_patience,
                "classifier_head": self.classifier_head,
                "unfreeze_esm": self.unfreeze_esm,
                "unfreeze_layers": self.unfreeze_layers if self.unfreeze_esm else 0,
                "fine_tuning_strategy":(f"last_{self.unfreeze_layers}_layers" if self.unfreeze_esm else "frozen_backbone"),
                "embedding_dim": 320,
                "cnn_num_filters": 64,
                "cnn_kernel_sizes": [3, 5, 7],
            },
            "summary": {
                "best_val_loss": self.best_val_loss,
                "best_val_accuracy": max(history["val_accuracy"]),
                "best_val_f1": max(history["val_f1"]),
                "best_val_precision": max(history["val_precision"]),
                "best_val_recall": max(history["val_recall"]),
                
                "final_train_loss": history["train_loss"][-1],
                "final_train_accuracy": history["train_accuracy"][-1],
                "final_train_f1": history["train_f1"][-1],
                "final_train_precision": history["train_precision"][-1],
                "final_train_recall": history["train_recall"][-1],
                "final_val_loss": history["val_loss"][-1],
                "final_val_accuracy": history["val_accuracy"][-1],
                "final_val_f1": history["val_f1"][-1],
                "final_val_precision": history["val_precision"][-1],
                "final_val_recall": history["val_recall"][-1],
                
                "best_epoch": history["val_loss"].index(self.best_val_loss) + 1,
                "epochs_trained": len(history["epoch"]),
                },

            "epochs": [
                {
                    "epoch": int(history["epoch"][i]),
                    "train_loss": history["train_loss"][i],
                    "train_accuracy": history["train_accuracy"][i],
                    "train_f1": history["train_f1"][i],
                    "train_precision": history["train_precision"][i],
                    "train_recall": history["train_recall"][i],
                    "val_loss": history["val_loss"][i],
                    "val_accuracy": history["val_accuracy"][i],
                    "val_f1": history["val_f1"][i],
                    "val_precision": history["val_precision"][i],
                    "val_recall": history["val_recall"][i]
                }
                for i in range(len(history["epoch"])) 
            ], 
            "train_loss": history["train_loss"],
            "train_scores": {
                "train_accuracy": history["train_accuracy"],
                "train_f1": history["train_f1"],
                "train_precision": history["train_precision"],
                "train_recall": history["train_recall"],
            },
            "val_loss": history["val_loss"],
            "val_scores": {
                "val_accuracy": history["val_accuracy"],
                "val_f1": history["val_f1"],
                "val_precision": history["val_precision"],
                "val_recall": history["val_recall"]
            }

        }
            
        save_json(history_json, self.run_dir / "history.json")

        return history
    
    def save_checkpoint(self):
        """Save CNN and ESM-2 checkpoints separately."""

        cnn_path = self.cnn_checkpoint_dir / "best_cnn.pt"
        esm_path = self.esm_checkpoint_dir / "best_esm.pt"

        torch.save(self.classifier.state_dict(), cnn_path)

        if self.encoder.model is not None:
            torch.save(self.encoder.model.state_dict(), esm_path)

        logger.info(f"CNN checkpoint saved to {cnn_path}")
        logger.info(f"ESM checkpoint saved to {esm_path}")

    def save_final_checkpoint(self):
        """Save final model state at end of training."""

        cnn_path = self.cnn_checkpoint_dir / "final_cnn.pt"
        esm_path = self.esm_checkpoint_dir / "final_esm.pt"

        torch.save(self.classifier.state_dict(), cnn_path)

        if self.encoder.model is not None:
            torch.save(self.encoder.model.state_dict(), esm_path)

        logger.info(f"Final CNN checkpoint saved to {cnn_path}")
        logger.info(f"Final ESM checkpoint saved to {esm_path}")   

    def predict(self, sequences: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on new sequences
        
        Args:
            sequences: List of protein sequences
        
        Returns:
            predictions (class labels) and probabilities
        """
        self.classifier.eval()
        
        with torch.no_grad():
            embeddings = self.encoder(sequences)
            embeddings = embeddings.to(self.device)
            logits = self.classifier(embeddings)
            probs = torch.softmax(logits, dim=1)
        
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        probabilities = probs.cpu().numpy()
    
        
        return predictions, probabilities


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    run_dir = create_run_dir(args.results_dir, args.dataset)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    config = vars(args)
    config["run_dir"] = str(run_dir)
    config["device"] = device
    config["model"] = "ESM2 + 1D CNN"
    config["encoding"] = "raw"
    config["classifier_head"] = args.classifier_head

    save_json(config, run_dir / "config.json")

    logger.info("ESM-2 + 1D-CNN Classification Pipeline")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Device: {device}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Classifier head: {self.classifier_head}") # type: ignore
    logger.info(f"Fine-tuning strategy: {'Frozen' if not self.unfreeze_esm else f'Last {self.unfreeze_layers} layer(s)'}") # type: ignore

    train_loader = create_dataloader(
        task=args.dataset,
        split="train",
        data_dir=args.data_dir,
        mode="classification",
        encoding="raw",
        batch_size=args.batch_size,
        shuffle=True,
        use_cache=False,
    )

    val_loader = create_dataloader(
        task=args.dataset,
        split="valid",
        data_dir=args.data_dir,
        mode="classification",
        encoding="raw",
        batch_size=args.batch_size,
        shuffle=False,
        use_cache=False,
    )

    pipeline = ESM2CNNPipeline(
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        device=device,
        esm_model_name=args.esm_model_name,
        run_dir=run_dir,
        checkpoint_dir=args.checkpoint_dir,
        unfreeze_esm=args.unfreeze_esm,
        unfreeze_layers=args.unfreeze_layers,
        esm_learning_rate=args.esm_learning_rate,
        cnn_checkpoint=args.cnn_checkpoint,
        classifier_head=args.classifier_head
    )

    pipeline.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
    )



if __name__ == "__main__":
    main()
