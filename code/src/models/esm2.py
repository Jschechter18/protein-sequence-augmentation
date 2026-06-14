import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict

import esm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm

from code.src.utils.dataloader import create_dataloader


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="solubility",
                        choices=["solubility", "localization"])
    parser.add_argument("--data_dir", type=str, default="data/processed/peer")
    parser.add_argument("--results_dir", type=str, default="code/results/esm2")

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
    run_dir = Path(results_dir) / f"esm2_{dataset}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)

class ESM2Encoder(nn.Module):
    """ESM-2 protein sequence encoder"""
    
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
            return torch.randn(len(sequences), max_len, 320)
        
        with torch.no_grad():
            batch_converter = self.alphabet.get_batch_converter() # type: ignore
            batch_labels, batch_strs, batch_tokens = batch_converter(
                [(str(i), seq) for i, seq in enumerate(sequences)]
            )

            batch_tokens = batch_tokens.to(next(self.model.parameters()).device)
            results = self.model(batch_tokens, repr_layers=[6])
            embeddings = results["representations"][6]
        
        return embeddings


class CNN1DClassifier(nn.Module):
    """1D-CNN classifier for protein sequences"""
    
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
        x = x.transpose(1, 2)
        
        # Apply convolutions with different kernel sizes
        conv_outputs = []

        for conv, bn in zip(self.conv_layers, self.bns):
            out = conv(x)
            out = bn(out)
            out = torch.relu(out)
            out = self.global_pool(out)
            out = out.squeeze(-1)
            conv_outputs.append(out)

        
        # Concatenate outputs from all convolution branches
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        
        # Fully connected layers
        logits = self.fc(x)
        return logits


class ESM2CNNPipeline:
    """Training pipeline combining ESM-2 encoder with 1D-CNN classifier"""
    
    def __init__(
        self,
        num_classes: int = 2,
        learning_rate: float = 1e-3,
        device: str | None = None,
        esm_model_name: str = "facebook/esm2_t6_8M",
        run_dir: str | Path | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.run_dir = Path(run_dir) if run_dir is not None else Path(".")
        self.run_dir.mkdir( parents=True, exist_ok=True)

        
        # Initialize encoder and classifier
        self.encoder = ESM2Encoder(model_name=esm_model_name).to(self.device)
        self.classifier = CNN1DClassifier(
            input_dim=320,
            num_classes=num_classes
        ).to(self.device)
        
        # Optimizer (only for classifier since encoder is frozen)
        self.optimizer = optim.Adam(
            self.classifier.parameters(),
            lr=learning_rate
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    # ==========
        config = {
            "model": "ESM2 + CNN",
            "esm_model": esm_model_name,
            "learning_rate": learning_rate,
            "num_classes": num_classes,
            "device": self.device
        }

        with open(
            self.run_dir / "config.json",
            "w"
        ) as f:
            json.dump(config, f, indent=4)
    # ==========

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")


        for batch in progress_bar:
            sequences = batch["sequence"]
            labels = batch["label"].to(self.device).long()
            
            # Encode sequences
            with torch.no_grad():
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
            
            progress_bar.set_postfix({
                'loss': total_loss / (total / labels.size(0)),
                'acc': correct / total
            })
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total    
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.classifier.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
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
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': correct / total
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
            "val_loss": [],
            "val_accuracy": []
        }
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            history["epoch"].append(epoch + 1)

            # Train
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            
            # Validate
            val_metrics = self.validate(val_loader)
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            
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
        pd.DataFrame(history).to_csv(
            self.run_dir / "training_history.csv",
            index=False
        )
        
        metrics = {
            "best_val_loss": float(min(history["val_loss"])),
            "best_val_accuracy": float(max(history["val_accuracy"])),
            "final_train_loss": float(history["train_loss"][-1]),
            "final_train_accuracy": float(history["train_accuracy"][-1]),
            "final_val_loss": float(history["val_loss"][-1]),
            "final_val_accuracy": float(history["val_accuracy"][-1]),
            "epochs_completed": len(history["epoch"]),
        }

        save_json(metrics, self.run_dir / "metrics.json")

        return history
    
    def save_checkpoint(self, filepath = None):
        """Save model checkpoint"""
        if filepath is None:
            filepath = self.run_dir / "best_model.pt"

        torch.save(self.classifier.state_dict(), filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_checkpoint(self, filepath=None):

        """Load model checkpoint"""
        if filepath is None:
            filepath = self.run_dir / "best_model.pt"

        self.classifier.load_state_dict(torch.load(filepath, map_location=self.device))
        logger.info(f"Model loaded from {filepath}")
    
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = vars(args)
    config["run_dir"] = str(run_dir)
    config["device"] = device
    config["model"] = "ESM2 + 1D CNN"
    config["encoding"] = "raw"

    save_json(config, run_dir / "config.json")

    logger.info("ESM-2 + 1D-CNN Classification Pipeline")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Device: {device}")
    logger.info(f"Run directory: {run_dir}")

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
    )

    pipeline.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
    )



if __name__ == "__main__":
    main()