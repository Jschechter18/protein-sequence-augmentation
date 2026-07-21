import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from Code.src.models.classifier import ProteinSequenceClassifier

logger = logging.getLogger(__name__)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)

class ProteinClassificationTrainingPipeline:
    def __init__(
        self,
        model: ProteinSequenceClassifier,
        num_classes: int = 2,
        device: str | None = None,
        run_dir: str | Path | None = None,
        checkpoint_dir: str | Path = "checkpoints",
        dataset: str = "solubility",
        learning_rate: float = 1e-3,
        encoder_learning_rate: float = 1e-3,
        esm_learning_rate: float = 1e-5,
        unfreeze_esm: bool = False,
        unfreeze_layers: int = 1,
        unfreeze_all_esm: bool = False,
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
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.encoder_learning_rate = encoder_learning_rate
        self.esm_learning_rate = esm_learning_rate
        self.unfreeze_esm = unfreeze_esm
        self.unfreeze_layers = unfreeze_layers
        self.unfreeze_all_esm = unfreeze_all_esm
        self.cnn_embedding_dim = cnn_embedding_dim
        self.cnn_num_filters = cnn_num_filters
        self.autoencoder_checkpoint = autoencoder_checkpoint
        self.autoencoder_embedding_dim = autoencoder_embedding_dim
        self.autoencoder_cnn_channels = autoencoder_cnn_channels
        self.autoencoder_hidden_dim = autoencoder_hidden_dim
        self.autoencoder_latent_dim = autoencoder_latent_dim
        self.autoencoder_num_layers = autoencoder_num_layers
        self.autoencoder_kernel_size = autoencoder_kernel_size

        self.run_dir = Path(run_dir) if run_dir is not None else Path(".")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = Path(checkpoint_dir) / "encoder_benchmark" / self.run_dir.name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Configure fine-tuning for ESM-2
        if self.model.embedding_type == "esm2":
            self.model.encoder.freeze_all_params() # type: ignore
            if unfreeze_all_esm:
                logger.info("Unfreezing entire ESM-2 backbone.")
                for param in self.model.encoder.parameters():
                    param.requires_grad = True
            elif unfreeze_esm:
                logger.info("Unfreezing last %d ESM transformer layer(s).", unfreeze_layers)
                self.model.encoder.unfreeze_last_layers(int(unfreeze_layers)) # type: ignore

        # Optimizer setup
        encoder_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
        parameter_groups = [{"params": self.model.head.parameters(), "lr": learning_rate}]
        if encoder_params:
            parameter_groups.append({
                "params": encoder_params,
                "lr": esm_learning_rate if self.model.embedding_type == "esm2" else encoder_learning_rate,
            })

        self.optimizer = optim.Adam(parameter_groups)
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            labels = batch["label"].to(self.device).long()
            self.optimizer.zero_grad()
            
            logits = self.model(batch)
            loss = self.criterion(logits, labels)
            loss.backward()

            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            progress_bar.set_postfix({'loss': total_loss / len(train_loader), 'acc': correct / total})

        return {
            "loss": float(total_loss / len(train_loader)),
            "accuracy": float(accuracy_score(all_labels, all_preds)),
            "f1": float(f1_score(all_labels, all_preds, average="macro")),
            "precision": float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
            "recall": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                labels = batch["label"].to(self.device).long()
                logits = self.model(batch)
                loss = self.criterion(logits, labels)

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

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 10, early_stopping_patience: int = 5):
        history = {k: [] for k in [
            "epoch", "train_loss", "train_accuracy", "train_f1", "train_precision", "train_recall",
            "val_loss", "val_accuracy", "val_f1", "val_precision", "val_recall"
        ]}

        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            history["epoch"].append(epoch + 1)

            t_metrics = self.train_epoch(train_loader)
            v_metrics = self.validate(val_loader)

            for key in ["loss", "accuracy", "f1", "precision", "recall"]:
                history[f"train_{key}"].append(t_metrics[key])
                history[f"val_{key}"].append(v_metrics[key])

            logger.info(
                f"Train Loss: {t_metrics['loss']:.4f}, Acc: {t_metrics['accuracy']:.4f} | "
                f"Val Loss: {v_metrics['loss']:.4f}, Acc: {v_metrics['accuracy']:.4f}"
            )

            if v_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = v_metrics['loss']
                self.patience_counter = 0
                self.save_checkpoint()
            else:
                self.patience_counter += 1
                if self.patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        self.save_final_checkpoint()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(self.run_dir / "training_history.csv", index=False)
        self._save_history_json(history, epochs, early_stopping_patience)

    def evaluate_test(self, test_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_sequences, all_true_labels, all_predictions, all_probabilities, all_confidences, all_sequence_lengths = [], [], [], [], [], []
        total_loss = 0.0

        with torch.inference_mode():
            for batch in tqdm(test_loader, desc="Test evaluation"):
                labels = batch["label"].to(self.device).long()
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                probabilities = torch.softmax(logits, dim=1)
                confidence = torch.max(probabilities, dim=1).values

                if "sequence" in batch:
                    sequences = batch["sequence"]
                    lengths = [len(seq) for seq in sequences]
                else:
                    sequences = ["<encoded_sequence>"] * labels.size(0)
                    lengths = batch["length"].cpu().tolist() if "length" in batch else [0] * labels.size(0)

                all_sequences.extend(sequences)
                all_sequence_lengths.extend(lengths)
                all_true_labels.extend(labels.cpu().tolist())
                all_predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
                all_confidences.extend(confidence.cpu().tolist())
                all_probabilities.extend(probabilities.cpu().tolist())

        true_labels = np.asarray(all_true_labels)
        predictions = np.asarray(all_predictions)
        probabilities = np.asarray(all_probabilities)

        test_metrics = {
            "test_loss": float(total_loss / len(test_loader)),
            "test_accuracy": float(accuracy_score(true_labels, predictions)),
            "test_f1": float(f1_score(true_labels, predictions, average="macro", zero_division=0)),
            "test_precision": float(precision_score(true_labels, predictions, average="macro", zero_division=0)),
            "test_recall": float(recall_score(true_labels, predictions, average="macro", zero_division=0)),
            "num_test_samples": int(len(true_labels)),
        }

        prediction_data = {
            "sample_index": np.arange(len(true_labels)),
            "sequence": all_sequences,
            "sequence_length": all_sequence_lengths,
            "true_label": true_labels,
            "predicted_label": predictions,
            "confidence": all_confidences,
            "correct": (true_labels == predictions),
        }
        for c in range(self.num_classes):
            prediction_data[f"probability_class_{c}"] = probabilities[:, c]

        pd.DataFrame(prediction_data).to_csv(self.run_dir / "test_predictions.csv", index=False)
        save_json(test_metrics, self.run_dir / "test_metrics.json")
        return test_metrics

    def save_checkpoint(self):
        encoder_path = self.checkpoint_dir / "best_encoder.pt"
        head_path = self.checkpoint_dir / "best_linear_head.pt"

        torch.save(self.model.encoder.state_dict(), encoder_path,)
        torch.save(self.model.head.state_dict(), head_path,)

        logger.info(f"Best encoder checkpoint saved to {encoder_path}")
        logger.info(f"Best head checkpoint saved to {head_path}")

    def save_final_checkpoint(self):
        encoder_path = (self.checkpoint_dir / "final_encoder.pt")
        head_path = (self.checkpoint_dir / "final_linear_head.pt")

        torch.save(self.model.encoder.state_dict(), encoder_path)
        torch.save(self.model.head.state_dict(), head_path)

        logger.info(f"Final encoder checkpoint saved to {encoder_path}")
        logger.info(f"Final head checkpoint saved to {head_path}")

    def load_checkpoint(self):
        encoder_path = (self.checkpoint_dir / "best_encoder.pt")
        classifier_path = (self.checkpoint_dir/ "best_linear_head.pt")

        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_path}")
        
        if not classifier_path.exists():
            raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_path}")

        self.model.encoder.load_state_dict(torch.load(encoder_path,map_location=self.device,))
        logger.info(f"Best encoder checkpoint loaded from {encoder_path}")

        self.model.head.load_state_dict(torch.load(classifier_path,map_location=self.device,))
        logger.info(f"Best head checkpoint loaded from {classifier_path}")

    def _save_history_json(self, history, epochs, patience):
        # Keeps your original schema cleanly isolated
        hyperparameters = {
            "model": f"{self.model.embedding_type} encoder linear output head",
            "embedding_type": self.model.embedding_type,
            "device": self.device,
            "dataset": self.dataset,
            "num_classes": self.num_classes,
            "learning_rate": self.learning_rate,
            "epochs": epochs,
            "early_stopping_patience": patience,
            "encoder_output_dim": self.model.encoder_output_dim,
        }
        save_json({"hyperparameters": hyperparameters, "summary": {"best_val_loss": self.best_val_loss}}, self.run_dir / "history.json")