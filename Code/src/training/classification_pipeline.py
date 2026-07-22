"""Training and evaluation utilities for protein sequence classifiers.

The pipeline deliberately treats a run directory as the unit of recovery.  Model,
optimizer, configuration, history, metrics, and predictions are kept together so
that an interrupted run can be resumed without reconstructing hidden state from
several unrelated files.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.optim import Adam, AdamW, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

METRIC_NAMES = (
    "loss",
    "accuracy",
    "positive_precision",
    "positive_recall",
    "positive_f1",
    # Compatibility aliases. These have positive-class, not macro, semantics.
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "balanced_accuracy",
    "auroc",
    "auprc",
    "num_samples",
)


def _temporary_path(path: Path) -> Path:
    """Return a closed temporary file in ``path``'s directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    return Path(temporary_name)


def _json_safe(value: Any) -> Any:
    """Convert common scientific Python values to strict JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        # JSON has no portable NaN representation.  In-memory metrics retain NaN;
        # persisted artifacts use null.
        return None
    return value


def save_json(obj: Any, path: str | Path) -> None:
    """Atomically save ``obj`` as standards-compliant JSON.

    This helper is intentionally public because the experiment entry point also
    uses it for ``config.json``.
    """

    target = Path(path)
    temporary = _temporary_path(target)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(obj), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _save_dataframe(frame: pd.DataFrame, path: Path) -> None:
    temporary = _temporary_path(path)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_torch(obj: Any, path: Path) -> None:
    temporary = _temporary_path(path)
    try:
        torch.save(obj, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _as_list(value: Any, expected_length: int) -> list[Any]:
    """Normalize a collated batch field without splitting strings into chars."""

    if torch.is_tensor(value):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        values = value.reshape(-1).tolist()
    elif isinstance(value, (str, bytes)):
        values = [value]
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        values = [value]
    if len(values) != expected_length:
        raise ValueError(
            f"Batch metadata has {len(values)} values for {expected_length} samples."
        )
    return values


class ProteinClassificationTrainingPipeline:
    """Train, checkpoint, resume, and evaluate a classifier.

    ``run_dir`` is required so a run can never silently write artifacts into the
    process working directory.  Encoder-related arguments remain accepted for
    compatibility with the existing experiment entry point.
    """

    CHECKPOINT_VERSION = 2

    def __init__(
        self,
        model: nn.Module,
        run_dir: str | Path,
        num_classes: int = 2,
        device: str | torch.device | None = None,
        checkpoint_dir: str | Path | None = None,
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
        optimizer_name: str = "adam",
        weight_decay: float = 0.0,
        max_grad_norm: float | None = 1.0,
        config: Mapping[str, Any] | None = None,
        run_config: Mapping[str, Any] | None = None,
        show_progress: bool = False,
    ) -> None:
        if not str(run_dir):
            raise ValueError("run_dir must be a non-empty path.")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if learning_rate <= 0 or encoder_learning_rate <= 0 or esm_learning_rate <= 0:
            raise ValueError("Learning rates must be positive.")
        if weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive or None.")

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.num_classes = int(num_classes)
        self.dataset = dataset
        self.learning_rate = float(learning_rate)
        self.encoder_learning_rate = float(encoder_learning_rate)
        self.esm_learning_rate = float(esm_learning_rate)
        self.weight_decay = float(weight_decay)
        self.max_grad_norm = max_grad_norm
        self.optimizer_name = optimizer_name.lower()
        self.show_progress = show_progress

        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None else self.run_dir
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.checkpoint_dir / "best_model.pt"
        self.last_checkpoint_path = self.checkpoint_dir / "last_model.pt"
        self.history_csv_path = self.run_dir / "history.csv"
        self.history_json_path = self.run_dir / "history.json"

        self._configure_encoder_trainability(
            unfreeze_esm=unfreeze_esm,
            unfreeze_layers=unfreeze_layers,
            unfreeze_all_esm=unfreeze_all_esm,
        )
        self.optimizer = self._create_optimizer()
        self.criterion = nn.CrossEntropyLoss()

        self.best_val_loss = float("inf")
        self.best_epoch: int | None = None
        self.current_epoch = 0
        self.patience_counter = 0
        self.history: list[dict[str, Any]] = []
        self._train_generator: torch.Generator | None = None

        embedding_type = getattr(self.model, "embedding_type", type(self.model).__name__)
        self.config: dict[str, Any] = {
            "dataset": dataset,
            "embedding_type": embedding_type,
            "num_classes": self.num_classes,
            "device": str(self.device),
            "learning_rate": self.learning_rate,
            "encoder_learning_rate": self.encoder_learning_rate,
            "esm_learning_rate": self.esm_learning_rate,
            "optimizer": self.optimizer_name,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "unfreeze_esm": unfreeze_esm,
            "unfreeze_layers": unfreeze_layers,
            "unfreeze_all_esm": unfreeze_all_esm,
            "cnn_embedding_dim": cnn_embedding_dim,
            "cnn_num_filters": cnn_num_filters,
            "autoencoder_checkpoint": autoencoder_checkpoint,
            "autoencoder_embedding_dim": autoencoder_embedding_dim,
            "autoencoder_cnn_channels": autoencoder_cnn_channels,
            "autoencoder_hidden_dim": autoencoder_hidden_dim,
            "autoencoder_latent_dim": autoencoder_latent_dim,
            "autoencoder_num_layers": autoencoder_num_layers,
            "autoencoder_kernel_size": autoencoder_kernel_size,
        }
        if config is not None:
            self.config.update(dict(config))
        if run_config is not None:
            self.config.update(dict(run_config))

    def _configure_encoder_trainability(
        self,
        *,
        unfreeze_esm: bool,
        unfreeze_layers: int,
        unfreeze_all_esm: bool,
    ) -> None:
        """Apply the legacy ESM fine-tuning controls when they are relevant."""

        embedding_type = getattr(self.model, "embedding_type", None)
        if unfreeze_esm and unfreeze_all_esm:
            raise ValueError("Choose either partial or complete ESM unfreezing, not both.")
        if embedding_type != "esm2":
            if unfreeze_esm or unfreeze_all_esm:
                raise ValueError(
                    "ESM unfreezing is supported only for embedding_type='esm2'; "
                    f"received {embedding_type!r}."
                )
            return
        encoder = getattr(self.model, "embedded_representation", None)
        if encoder is None:
            return
        if hasattr(encoder, "freeze_all_params"):
            encoder.freeze_all_params()
        if unfreeze_all_esm:
            logger.info("Unfreezing the complete ESM-2 encoder.")
            for parameter in encoder.parameters():
                parameter.requires_grad = True
            if hasattr(encoder, "is_frozen"):
                encoder.is_frozen = False
        elif unfreeze_esm:
            if unfreeze_layers < 1:
                raise ValueError("unfreeze_layers must be at least 1 when unfreeze_esm=True.")
            if not hasattr(encoder, "unfreeze_last_layers"):
                raise ValueError("The selected encoder does not support partial unfreezing.")
            encoder.unfreeze_last_layers(int(unfreeze_layers))

    def _create_optimizer(self) -> Optimizer:
        """Build non-overlapping parameter groups containing trainable tensors only."""

        head = getattr(self.model, "head", None)
        encoder = getattr(self.model, "embedded_representation", None)
        head_params = [
            parameter for parameter in head.parameters() if parameter.requires_grad
        ] if isinstance(head, nn.Module) else []
        encoder_params = [
            parameter for parameter in encoder.parameters() if parameter.requires_grad
        ] if isinstance(encoder, nn.Module) else []

        assigned = {id(parameter) for parameter in head_params + encoder_params}
        remaining_params = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in assigned
        ]
        parameter_groups: list[dict[str, Any]] = []
        if head_params or remaining_params:
            parameter_groups.append(
                {"params": head_params + remaining_params, "lr": self.learning_rate}
            )
        if encoder_params:
            encoder_lr = (
                self.esm_learning_rate
                if getattr(self.model, "embedding_type", None) == "esm2"
                else self.encoder_learning_rate
            )
            parameter_groups.append({"params": encoder_params, "lr": encoder_lr})
        if not parameter_groups:
            raise ValueError("The model has no trainable parameters.")

        optimizer_classes: dict[str, type[Optimizer]] = {
            "adam": Adam,
            "adamw": AdamW,
        }
        if self.optimizer_name not in optimizer_classes:
            raise ValueError("optimizer_name must be 'adam' or 'adamw'.")
        return optimizer_classes[self.optimizer_name](
            parameter_groups, weight_decay=self.weight_decay
        )

    @property
    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def _move_batch_to_device(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def _classification_labels(self, batch: Mapping[str, Any]) -> torch.Tensor:
        if "label" not in batch:
            raise KeyError("Every classification batch must contain 'label'.")
        raw_labels = batch["label"]
        if not torch.is_tensor(raw_labels):
            raise TypeError("Classification labels must be collated into a tensor.")
        if (
            raw_labels.dtype == torch.bool
            or torch.is_floating_point(raw_labels)
            or torch.is_complex(raw_labels)
        ):
            raise TypeError("Classification labels must use an integer tensor dtype.")
        labels = raw_labels.long().reshape(-1)
        if labels.numel() and (
            labels.min().item() < 0 or labels.max().item() >= self.num_classes
        ):
            raise ValueError(
                f"Classification labels must be in [0, {self.num_classes - 1}]."
            )
        return labels

    def _collect_split(
        self,
        loader: DataLoader,
        *,
        training: bool,
        description: str,
    ) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        self.model.train(training)
        total_loss = 0.0
        total_samples = 0
        labels_parts: list[np.ndarray] = []
        predictions_parts: list[np.ndarray] = []
        probability_parts: list[np.ndarray] = []

        iterator: Iterable[Mapping[str, Any]] = tqdm(
            loader,
            desc=description,
            disable=not self.show_progress,
            dynamic_ncols=True,
            leave=False,
        )
        context = torch.enable_grad() if training else torch.inference_mode()
        with context:
            for raw_batch in iterator:
                batch = self._move_batch_to_device(raw_batch)
                labels = self._classification_labels(batch)
                batch_size = int(labels.numel())

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(batch)
                if logits.ndim != 2 or logits.shape != (batch_size, self.num_classes):
                    raise ValueError(
                        "Classifier logits must have shape "
                        f"[{batch_size}, {self.num_classes}], got {tuple(logits.shape)}."
                    )
                loss = self.criterion(logits, labels)
                if training:
                    loss.backward()
                    if self.max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(
                            self.trainable_parameters, self.max_grad_norm
                        )
                    self.optimizer.step()

                probabilities = torch.softmax(logits.detach(), dim=1)
                predictions = probabilities.argmax(dim=1)
                total_loss += float(loss.detach().item()) * batch_size
                total_samples += batch_size
                if self.show_progress:
                    iterator.set_postfix(loss=f"{total_loss / total_samples:.4f}")
                labels_parts.append(labels.detach().cpu().numpy())
                predictions_parts.append(predictions.cpu().numpy())
                probability_parts.append(probabilities.cpu().numpy())

        if total_samples == 0:
            raise ValueError(f"The {description.lower()} loader produced no samples.")
        labels_array = np.concatenate(labels_parts).astype(np.int64, copy=False)
        predictions_array = np.concatenate(predictions_parts).astype(np.int64, copy=False)
        probabilities_array = np.concatenate(probability_parts).astype(np.float64, copy=False)
        metrics = self._calculate_metrics(
            labels_array,
            predictions_array,
            probabilities_array,
            loss=total_loss / total_samples,
        )
        return metrics, {
            "labels": labels_array,
            "predictions": predictions_array,
            "probabilities": probabilities_array,
        }

    def _calculate_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        *,
        loss: float,
    ) -> dict[str, float]:
        """Calculate metrics once over a complete split.

        Precision, recall, and F1 use class 1 as the positive class.  For a
        multiclass task they therefore remain one-vs-rest metrics for class 1.
        AUROC and AUPRC are undefined when that class is absent or is the only
        class in a split, and are returned as ``NaN`` in that case.
        """

        if probabilities.shape != (len(labels), self.num_classes):
            raise ValueError("Probability array does not match labels and num_classes.")
        positive_labels = labels == 1
        positive_predictions = predictions == 1
        present_classes = np.unique(labels)
        recalls_by_class = [
            float(np.mean(predictions[labels == label] == label))
            for label in present_classes
        ]
        if np.unique(positive_labels).size < 2:
            auroc = float("nan")
            auprc = float("nan")
        else:
            positive_probabilities = probabilities[:, 1]
            auroc = float(roc_auc_score(positive_labels, positive_probabilities))
            auprc = float(
                average_precision_score(positive_labels, positive_probabilities)
            )

        positive_precision = float(
            precision_score(positive_labels, positive_predictions, zero_division=0)
        )
        positive_recall = float(
            recall_score(positive_labels, positive_predictions, zero_division=0)
        )
        positive_f1 = float(
            f1_score(positive_labels, positive_predictions, zero_division=0)
        )
        return {
            "loss": float(loss),
            "accuracy": float(accuracy_score(labels, predictions)),
            "positive_precision": positive_precision,
            "positive_recall": positive_recall,
            "positive_f1": positive_f1,
            "precision": positive_precision,
            "recall": positive_recall,
            "f1": positive_f1,
            "macro_f1": float(
                f1_score(
                    labels,
                    predictions,
                    labels=list(range(self.num_classes)),
                    average="macro",
                    zero_division=0,
                )
            ),
            "balanced_accuracy": float(np.mean(recalls_by_class)),
            "auroc": auroc,
            "auprc": auprc,
            "num_samples": int(len(labels)),
        }

    def train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        metrics, _ = self._collect_split(
            train_loader, training=True, description="Training"
        )
        return metrics

    def validate(self, val_loader: DataLoader) -> dict[str, float]:
        metrics, _ = self._collect_split(
            val_loader, training=False, description="Validation"
        )
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        early_stopping_patience: int = 5,
        resume_from: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Train up to ``epochs`` total epochs and restore the best model."""

        if epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1.")
        self.config.update(
            {
                "epochs": int(epochs),
                "early_stopping_patience": int(early_stopping_patience),
            }
        )
        self._train_generator = getattr(train_loader, "generator", None)
        if resume_from is not None:
            self.load_checkpoint(resume_from, load_optimizer=True)

        if self.patience_counter >= early_stopping_patience:
            logger.info(
                "Checkpoint already met early-stopping patience at epoch %d; "
                "restoring its best model without training another epoch.",
                self.current_epoch,
            )

        for zero_based_epoch in range(self.current_epoch, epochs):
            if self.patience_counter >= early_stopping_patience:
                break
            epoch = zero_based_epoch + 1
            logger.info("Epoch %d/%d", epoch, epochs)
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            record: dict[str, Any] = {"epoch": epoch}
            record.update(
                {f"train_{name}": train_metrics[name] for name in METRIC_NAMES}
            )
            record.update({f"val_{name}": val_metrics[name] for name in METRIC_NAMES})
            self.history.append(record)
            self.current_epoch = epoch

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(self.best_checkpoint_path, epoch=epoch)
            else:
                self.patience_counter += 1

            self.save_checkpoint(self.last_checkpoint_path, epoch=epoch)
            self._save_history()
            logger.info(
                "Epoch %d: train loss %.4f, val loss %.4f, val accuracy %.4f",
                epoch,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["accuracy"],
            )
            if self.patience_counter >= early_stopping_patience:
                logger.info("Early stopping at epoch %d.", epoch)
                break

        if not self.best_checkpoint_path.exists():
            raise RuntimeError("Training completed without producing a best checkpoint.")
        self.load_checkpoint(self.best_checkpoint_path, load_optimizer=False)
        # load_checkpoint restores checkpoint history, which may stop at the best
        # epoch. Recover the complete record from last_model after restoring only
        # the desired model weights.
        last_bundle = self._read_checkpoint(self.last_checkpoint_path)
        self.history = list(last_bundle.get("history", self.history))
        self.current_epoch = int(last_bundle["epoch"])
        self.patience_counter = int(last_bundle.get("patience_counter", 0))
        self.best_val_loss = float(last_bundle["best_val_loss"])
        best_epoch = last_bundle.get("best_epoch")
        self.best_epoch = int(best_epoch) if best_epoch is not None else None
        self._save_history()
        return self.history

    def _checkpoint_bundle(self, epoch: int) -> dict[str, Any]:
        return {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": int(epoch),
            "best_val_loss": float(self.best_val_loss),
            "best_epoch": self.best_epoch,
            "patience_counter": int(self.patience_counter),
            "config": dict(self.config),
            "history": list(self.history),
            "rng_state": self._capture_rng_state(),
        }

    def _capture_rng_state(self) -> dict[str, Any]:
        """Capture randomness that affects an exactly resumed training run."""

        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if self._train_generator is not None:
            state["train_loader_generator"] = self._train_generator.get_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        if hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
            try:
                state["mps"] = torch.mps.get_rng_state()
            except RuntimeError:
                # MPS may be compiled in but unavailable on the current host.
                pass
        return state

    def _restore_rng_state(self, state: Mapping[str, Any] | None) -> None:
        """Restore checkpoint randomness, including the training sampler."""

        if not state:
            logger.warning(
                "Checkpoint has no RNG state; resumed training may differ from an "
                "uninterrupted run."
            )
            return
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        loader_state = state.get("train_loader_generator")
        if loader_state is not None and self._train_generator is not None:
            self._train_generator.set_state(loader_state.cpu())
        cuda_state = state.get("cuda")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([item.cpu() for item in cuda_state])
        mps_state = state.get("mps")
        if (
            mps_state is not None
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "set_rng_state")
        ):
            torch.mps.set_rng_state(mps_state.cpu())

    def save_checkpoint(
        self,
        path: str | Path | None = None,
        *,
        epoch: int | None = None,
    ) -> Path:
        """Atomically save a bundled checkpoint (defaults to ``best_model.pt``)."""

        target = Path(path) if path is not None else self.best_checkpoint_path
        checkpoint_epoch = self.current_epoch if epoch is None else epoch
        _save_torch(self._checkpoint_bundle(checkpoint_epoch), target)
        return target

    def save_final_checkpoint(self) -> Path:
        """Backward-compatible alias for saving the recoverable last checkpoint."""

        return self.save_checkpoint(self.last_checkpoint_path)

    def _read_checkpoint(self, path: str | Path) -> dict[str, Any]:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Classifier checkpoint not found: {source}")
        try:
            checkpoint = torch.load(
                source, map_location=self.device, weights_only=False
            )
        except TypeError:  # PyTorch versions predating weights_only
            checkpoint = torch.load(source, map_location=self.device)
        required = {
            "model_state_dict",
            "optimizer_state_dict",
            "epoch",
            "best_val_loss",
            "config",
        }
        if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
            missing = required.difference(checkpoint if isinstance(checkpoint, dict) else {})
            raise ValueError(f"Invalid classifier checkpoint; missing keys: {sorted(missing)}")
        if not isinstance(checkpoint["config"], Mapping):
            raise ValueError("Invalid classifier checkpoint; 'config' must be a mapping.")
        checkpoint_version = int(checkpoint.get("checkpoint_version", 0))
        if checkpoint_version > self.CHECKPOINT_VERSION:
            raise ValueError(
                f"Checkpoint version {checkpoint_version} is newer than supported "
                f"version {self.CHECKPOINT_VERSION}."
            )
        return checkpoint

    def load_checkpoint(
        self,
        path: str | Path | None = None,
        *,
        load_optimizer: bool = True,
        restore_rng: bool | None = None,
    ) -> dict[str, Any]:
        """Load a bundled checkpoint and return its metadata."""

        source = Path(path) if path is not None else self.best_checkpoint_path
        checkpoint = self._read_checkpoint(source)
        checkpoint_config = checkpoint.get("config", {})
        saved_fingerprint = checkpoint_config.get("resume_fingerprint")
        requested_fingerprint = self.config.get("resume_fingerprint")
        if (
            saved_fingerprint is not None
            and requested_fingerprint is not None
            and saved_fingerprint != requested_fingerprint
        ):
            raise ValueError(
                "Classifier checkpoint configuration does not match the requested run."
            )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(self.device)
        self.current_epoch = int(checkpoint["epoch"])
        self.best_val_loss = float(checkpoint["best_val_loss"])
        best_epoch = checkpoint.get("best_epoch")
        self.best_epoch = int(best_epoch) if best_epoch is not None else None
        self.patience_counter = int(checkpoint.get("patience_counter", 0))
        self.history = list(checkpoint.get("history", []))
        should_restore_rng = load_optimizer if restore_rng is None else restore_rng
        if should_restore_rng:
            self._restore_rng_state(checkpoint.get("rng_state"))
        return checkpoint

    def _save_history(self) -> None:
        frame = pd.DataFrame(self.history)
        _save_dataframe(frame, self.history_csv_path)
        save_json(
            {
                "config": self.config,
                "summary": {
                    "best_val_loss": self.best_val_loss,
                    "best_epoch": self.best_epoch,
                    "epochs_completed": self.current_epoch,
                },
                "history": self.history,
            },
            self.history_json_path,
        )

    def _save_history_json(
        self,
        history: Any | None = None,
        epochs: int | None = None,
        patience: int | None = None,
    ) -> None:
        """Backward-compatible wrapper around the complete history artifact."""

        del epochs, patience
        if isinstance(history, list):
            self.history = history
        self._save_history()

    def evaluate_test(self, test_loader: DataLoader) -> dict[str, float]:
        """Evaluate the current (normally best-restored) model and save artifacts."""

        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        labels_parts: list[np.ndarray] = []
        predictions_parts: list[np.ndarray] = []
        probability_parts: list[np.ndarray] = []
        sequences: list[str] = []
        sequence_lengths: list[int] = []
        sample_ids: list[Any] = []
        saw_sample_id = False

        iterator: Iterable[Mapping[str, Any]] = tqdm(
            test_loader,
            desc="Test evaluation",
            disable=not self.show_progress,
            dynamic_ncols=True,
            leave=False,
        )
        with torch.inference_mode():
            for raw_batch in iterator:
                batch = self._move_batch_to_device(raw_batch)
                labels = self._classification_labels(batch)
                batch_size = int(labels.numel())
                logits = self.model(batch)
                if logits.ndim != 2 or logits.shape != (batch_size, self.num_classes):
                    raise ValueError(
                        "Classifier logits must have shape "
                        f"[{batch_size}, {self.num_classes}], got {tuple(logits.shape)}."
                    )
                loss = self.criterion(logits, labels)
                probabilities = torch.softmax(logits, dim=1)
                predictions = probabilities.argmax(dim=1)

                total_loss += float(loss.item()) * batch_size
                total_samples += batch_size
                if self.show_progress:
                    iterator.set_postfix(loss=f"{total_loss / total_samples:.4f}")
                labels_parts.append(labels.cpu().numpy())
                predictions_parts.append(predictions.cpu().numpy())
                probability_parts.append(probabilities.cpu().numpy())

                if "sequence" in raw_batch:
                    batch_sequences = [
                        str(sequence)
                        for sequence in _as_list(raw_batch["sequence"], batch_size)
                    ]
                else:
                    batch_sequences = ["<encoded_sequence>"] * batch_size
                sequences.extend(batch_sequences)
                if "length" in raw_batch:
                    sequence_lengths.extend(
                        int(length)
                        for length in _as_list(raw_batch["length"], batch_size)
                    )
                else:
                    sequence_lengths.extend(len(sequence) for sequence in batch_sequences)

                id_key = next(
                    (key for key in ("sample_id", "idx", "id") if key in raw_batch),
                    None,
                )
                if id_key is not None:
                    saw_sample_id = True
                    sample_ids.extend(_as_list(raw_batch[id_key], batch_size))
                else:
                    sample_ids.extend([None] * batch_size)

        if total_samples == 0:
            raise ValueError("The test evaluation loader produced no samples.")
        labels_array = np.concatenate(labels_parts).astype(np.int64, copy=False)
        predictions_array = np.concatenate(predictions_parts).astype(np.int64, copy=False)
        probabilities_array = np.concatenate(probability_parts).astype(np.float64, copy=False)
        metrics = self._calculate_metrics(
            labels_array,
            predictions_array,
            probabilities_array,
            loss=total_loss / total_samples,
        )

        prediction_data: dict[str, Any] = {
            "sample_index": np.arange(total_samples),
            "sequence": sequences,
            "sequence_length": sequence_lengths,
            "true_label": labels_array,
            "predicted_label": predictions_array,
            "confidence": probabilities_array.max(axis=1),
            "correct": labels_array == predictions_array,
        }
        if saw_sample_id:
            prediction_data["sample_id"] = sample_ids
        for class_index in range(self.num_classes):
            prediction_data[f"probability_class_{class_index}"] = probabilities_array[
                :, class_index
            ]

        _save_dataframe(
            pd.DataFrame(prediction_data), self.run_dir / "test_predictions.csv"
        )
        save_json(metrics, self.run_dir / "metrics.json")
        return metrics
