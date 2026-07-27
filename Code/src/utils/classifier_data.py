"""Classifier dataset validation and dataloader construction."""

from __future__ import annotations

import os
import random
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from Code.src.models.autoencoder import ProteinSequenceAutoencoder
from Code.src.utils.dataloader import create_dataloader
from Code.src.utils.utils import set_random_seed


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before weights_only support.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Autoencoder checkpoint must be a mapping: {path}")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Autoencoder checkpoint has no valid model state: {path}")
    return state_dict


def resolve_split_source(config: Any, split: str) -> tuple[Path, bool]:
    task_dir = Path(config.data_dir) / config.dataset
    split_path = task_dir / f"{split}.csv"
    if split_path.is_file():
        return split_path, False
    combined_path = task_dir / f"{config.dataset}.csv"
    if combined_path.is_file():
        return combined_path, True
    raise FileNotFoundError(
        f"No data source found for {config.dataset!r} split {split!r}; looked for "
        f"{split_path} and {combined_path}."
    )


def read_classification_split(config: Any, split: str) -> pd.DataFrame:
    path, is_combined = resolve_split_source(config, split)
    frame = pd.read_csv(path)
    if is_combined:
        if "split" not in frame.columns:
            raise ValueError(f"Combined data file {path} has no 'split' column.")
        frame = frame.loc[frame["split"] == split]
    required = {"sequence", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Data file {path} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"The {config.dataset} {split} split is empty.")
    return frame


def validate_dataset_integrity(config: Any) -> None:
    """Reject malformed labels and exact sequence leakage before any run starts."""

    sequences_by_split: dict[str, set[str]] = {}
    for split in ("train", "valid", "test"):
        frame = read_classification_split(config, split)
        if frame["sequence"].isna().any():
            raise ValueError(f"The {config.dataset} {split} split has missing sequences.")
        normalized_sequences = frame["sequence"].astype(str).str.upper().str.strip()
        if normalized_sequences.eq("").any():
            raise ValueError(f"The {config.dataset} {split} split has empty sequences.")

        labels = pd.to_numeric(frame["label"], errors="coerce")
        if labels.isna().any() or not np.isfinite(labels.to_numpy(dtype=float)).all():
            raise ValueError(f"The {config.dataset} {split} split has invalid labels.")
        label_values = labels.to_numpy(dtype=float)
        if not np.equal(label_values, np.floor(label_values)).all():
            raise ValueError(
                f"The {config.dataset} {split} split has non-integer class labels."
            )
        if ((label_values < 0) | (label_values >= config.num_classes)).any():
            raise ValueError(
                f"The {config.dataset} {split} split has labels outside "
                f"[0, {config.num_classes - 1}]."
            )
        if "idx" in frame.columns and (
            frame["idx"].isna().any() or frame["idx"].duplicated().any()
        ):
            raise ValueError(
                f"The {config.dataset} {split} split must have unique, non-null idx values."
            )
        sequences_by_split[split] = set(normalized_sequences)

    for left, right in combinations(("train", "valid", "test"), 2):
        overlap = sequences_by_split[left].intersection(sequences_by_split[right])
        if overlap:
            raise ValueError(
                f"Detected {len(overlap)} exact normalized sequence(s) shared by the "
                f"{left} and {right} splits for {config.dataset}; refusing a leaky run."
            )


def validate_preflight(configs: list[Any]) -> None:
    if not configs:
        raise ValueError("At least one classifier run configuration is required.")
    validate_dataset_integrity(configs[0])
    representations = {config.representation for config in configs}
    uses_autoencoder_checkpoint = bool(
        representations & {"trained_autoencoder", "trained_autoencoder+esm2"}
    )
    if uses_autoencoder_checkpoint:
        checkpoint_values = {config.autoencoder_checkpoint for config in configs}
        if len(checkpoint_values) != 1 or None in checkpoint_values:
            raise ValueError("Trained-autoencoder runs require one explicit checkpoint")
        checkpoint_path = Path(next(iter(checkpoint_values)))  # type: ignore[arg-type]
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Autoencoder checkpoint not found: {checkpoint_path}")
        exemplar = configs[0]
        model = ProteinSequenceAutoencoder(
            embedding_dim=exemplar.autoencoder_embedding_dim,
            cnn_out_channels=exemplar.autoencoder_cnn_channels,
            hidden_dim=exemplar.autoencoder_hidden_dim,
            latent_dim=exemplar.autoencoder_latent_dim,
            num_layers=exemplar.autoencoder_num_layers,
            kernel_size=exemplar.autoencoder_kernel_size,
        )
        try:
            model.load_state_dict(load_state_dict(checkpoint_path), strict=True)
        except RuntimeError as error:
            raise ValueError(
                "Autoencoder architecture arguments do not match checkpoint "
                f"{checkpoint_path}: {error}"
            ) from error

    if representations & {"esm2", "trained_autoencoder+esm2"}:
        import esm

        if not hasattr(esm, "pretrained") or not hasattr(
            esm.pretrained, "esm2_t6_8M_UR50D"
        ):
            raise ImportError(
                "ESM-2 runs require the 'fair-esm' distribution. Remove the conflicting "
                "'esm' package and install the pinned requirements."
            )


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    set_random_seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False
    if deterministic:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False


def create_sequence_dataloader(
    config: Any,
    split: str,
    *,
    shuffle: bool,
    offset: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed + offset)
    return create_dataloader(
        task=config.dataset,
        split=split,
        data_dir=config.data_dir,
        mode="classification",
        encoding="char",
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        generator=generator,
        worker_init_fn=seed_worker,
        use_cache=config.use_cache,
    )
