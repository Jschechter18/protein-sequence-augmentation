"""Frozen classifier-embedding cache construction and loading."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset

from Code.src.models.classifier import ProteinSequenceClassifier
from Code.src.utils.dataloader import collate_sequence_batch
from Code.src.utils.experiment_artifacts import data_source_metadata, file_sha256

logger = logging.getLogger(__name__)

EMBEDDING_CACHE_SCHEMA_VERSION = 1


def _validate_embedding_payload(payload: dict[str, Any], *, source: str) -> None:
    embeddings = payload.get("embeddings")
    labels = payload.get("labels")
    lengths = payload.get("lengths")
    sequences = payload.get("sequences")
    sample_ids = payload.get("sample_ids")
    if not all(torch.is_tensor(value) for value in (embeddings, labels, lengths)):
        raise ValueError(f"{source} is missing tensor fields.")
    size = len(labels)
    if (
        embeddings.ndim != 2
        or len(embeddings) != size
        or len(lengths) != size
        or not isinstance(sequences, (list, tuple))
        or len(sequences) != size
        or not isinstance(sample_ids, (list, tuple))
        or len(sample_ids) != size
    ):
        raise ValueError(f"{source} contains inconsistent dimensions.")
    if not torch.isfinite(embeddings).all().item():
        bad_rows = (~torch.isfinite(embeddings)).any(dim=1).nonzero().flatten()
        preview = ", ".join(str(int(index)) for index in bad_rows[:10])
        suffix = "..." if len(bad_rows) > 10 else ""
        raise ValueError(
            f"{source} contains non-finite embeddings in row(s) {preview}{suffix}."
        )


class CachedEmbeddingDataset(Dataset):
    def __init__(self, payload: dict[str, Any]) -> None:
        _validate_embedding_payload(payload, source="Cached embedding payload")
        self.embeddings = payload["embeddings"].float()
        self.labels = payload["labels"].long()
        self.lengths = payload["lengths"].long()
        self.sequences = list(payload["sequences"])
        self.sample_ids = list(payload["sample_ids"])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "embedding": self.embeddings[index],
            "label": self.labels[index],
            "length": self.lengths[index],
            "sequence": self.sequences[index],
            "sample_id": self.sample_ids[index],
        }


def embedding_cache_metadata(
    config: Any,
    split: str,
    *,
    project_root: Path,
    resolve_split_source: Callable[[Any, str], tuple[Path, bool]],
) -> dict[str, Any]:
    uses_trained_autoencoder = config.representation in {
        "trained_autoencoder",
        "trained_autoencoder+esm2",
    }
    try:
        fair_esm_version = importlib.metadata.version("fair-esm")
    except importlib.metadata.PackageNotFoundError:
        fair_esm_version = None
    return {
        "schema_version": EMBEDDING_CACHE_SCHEMA_VERSION,
        "embedding_dtype": "float32",
        "encoder_evaluation_mode": True,
        "dataset": config.dataset,
        "split": split,
        "representation": config.representation,
        "encoder_seed": config.seed
        if config.representation == "random_autoencoder"
        else None,
        "source": data_source_metadata(config, resolve_split_source)[split],
        "autoencoder_checkpoint_sha256": (
            file_sha256(config.autoencoder_checkpoint)
            if uses_trained_autoencoder
            else None
        ),
        "autoencoder_architecture": {
            "layer_type": config.autoencoder_layer_type,
            "embedding_dim": config.autoencoder_embedding_dim,
            "cnn_channels": config.autoencoder_cnn_channels,
            "hidden_dim": config.autoencoder_hidden_dim,
            "latent_dim": config.autoencoder_latent_dim,
            "num_layers": config.autoencoder_num_layers,
            "kernel_size": config.autoencoder_kernel_size,
        }
        if "autoencoder" in config.representation
        else None,
        "esm_model_name": config.esm_model_name
        if "esm2" in config.representation
        else None,
        "esm_max_sequence_length": config.esm_max_sequence_length
        if "esm2" in config.representation
        else None,
        "encoder_source_sha256": {
            "autoencoder": file_sha256(project_root / "Code/src/models/autoencoder.py"),
            "classifier": file_sha256(project_root / "Code/src/models/classifier.py"),
            "dataloader": file_sha256(project_root / "Code/src/utils/dataloader.py"),
            "sequence_dataset": file_sha256(
                project_root / "Code/src/utils/sequence_dataset.py"
            ),
            "cache_pipeline": file_sha256(
                project_root / "Code/src/training/train_classifier.py"
            ),
            "embedding_cache": file_sha256(
                project_root / "Code/src/utils/embedding_cache.py"
            ),
        },
        "library_versions": {
            "torch": str(torch.__version__),
            "fair_esm": fair_esm_version if "esm2" in config.representation else None,
        },
    }


def embedding_cache_path(
    config: Any,
    split: str,
    *,
    project_root: Path,
    resolve_split_source: Callable[[Any, str], tuple[Path, bool]],
) -> tuple[Path, dict[str, Any]]:
    metadata = embedding_cache_metadata(
        config,
        split,
        project_root=project_root,
        resolve_split_source=resolve_split_source,
    )
    material = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(material.encode("utf-8")).hexdigest()
    seed_dir = (
        f"seed_{config.seed}"
        if config.representation == "random_autoencoder"
        else "shared"
    )
    path = (
        Path(config.embedding_cache_root)
        / config.dataset
        / config.representation
        / seed_dir
        / f"{split}_{fingerprint[:16]}.pt"
    )
    metadata["fingerprint"] = fingerprint
    return path, metadata


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_embedding_cache(
    config: Any,
    split: str,
    model: ProteinSequenceClassifier,
    path: Path,
    metadata: dict[str, Any],
    *,
    sequence_dataloader: Callable[..., DataLoader],
) -> None:
    split_offset = {"train": 0, "valid": 1, "test": 2}[split]
    loader = sequence_dataloader(
        config, split, shuffle=False, offset=split_offset
    )
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    lengths: list[torch.Tensor] = []
    sequences: list[str] = []
    sample_ids: list[Any] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            embeddings.append(model.encode(batch).detach().cpu())
            labels.append(batch["label"].detach().cpu().long().reshape(-1))
            lengths.append(batch["length"].detach().cpu().long().reshape(-1))
            sequences.extend(str(value) for value in batch["sequence"])
            sample_ids.extend(list(batch["sample_id"]))
    if not embeddings:
        raise ValueError(f"Cannot cache an empty {config.dataset} {split} split.")
    payload = {
        "metadata": metadata,
        "embeddings": torch.cat(embeddings),
        "labels": torch.cat(labels),
        "lengths": torch.cat(lengths),
        "sequences": sequences,
        "sample_ids": sample_ids,
    }
    atomic_torch_save(payload, path)
    logger.info("Saved frozen embeddings: %s", path)


def load_embedding_cache(
    path: Path,
    expected_metadata: dict[str, Any],
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("metadata") != expected_metadata:
        raise ValueError(f"Embedding cache metadata mismatch: {path}")
    _validate_embedding_payload(payload, source=f"Embedding cache {path}")
    return payload


def ensure_embedding_caches(
    config: Any,
    splits: list[str],
    *,
    cache_path: Callable[[Any, str], tuple[Path, dict[str, Any]]],
    build_cache: Callable[
        [Any, str, ProteinSequenceClassifier, Path, dict[str, Any]], None
    ],
    model_factory: Callable[..., ProteinSequenceClassifier],
) -> dict[str, dict[str, Any]]:
    cache_specs = {split: cache_path(config, split) for split in splits}
    payloads: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for split, (path, metadata) in cache_specs.items():
        if not path.is_file():
            missing.append(split)
            continue
        try:
            payloads[split] = load_embedding_cache(path, metadata)
            logger.info("Using cached frozen embeddings: %s", path)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            logger.warning("Rebuilding invalid embedding cache %s: %s", path, error)
            missing.append(split)
    if missing:
        encoder_model = model_factory(config, use_cached_embeddings=False)
        for split in missing:
            path, metadata = cache_specs[split]
            build_cache(config, split, encoder_model, path, metadata)
            payloads[split] = load_embedding_cache(path, metadata)
        del encoder_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return payloads


def cached_embedding_dataloader(
    config: Any,
    payload: dict[str, Any],
    *,
    shuffle: bool,
    offset: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed + offset)
    return DataLoader(
        CachedEmbeddingDataset(payload),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=config.pin_memory,
        persistent_workers=False,
        generator=generator,
        collate_fn=collate_sequence_batch,
    )


def create_run_dataloaders(
    config: Any,
    *,
    ensure_caches: Callable[[Any, list[str]], dict[str, dict[str, Any]]],
    sequence_dataloader: Callable[..., DataLoader],
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    if config.cache_embeddings:
        splits = ["train", "valid"]
        if config.evaluate_test:
            splits.append("test")
        payloads = ensure_caches(config, splits)
        return (
            cached_embedding_dataloader(
                config, payloads["train"], shuffle=True, offset=0
            ),
            cached_embedding_dataloader(
                config, payloads["valid"], shuffle=False, offset=1
            ),
            cached_embedding_dataloader(
                config, payloads["test"], shuffle=False, offset=2
            )
            if config.evaluate_test
            else None,
        )

    loaders = []
    for offset, (split, shuffle) in enumerate(
        (("train", True), ("valid", False), ("test", False))
    ):
        if split == "test" and not config.evaluate_test:
            loaders.append(None)
            continue
        loaders.append(
            sequence_dataloader(config, split, shuffle=shuffle, offset=offset)
        )
    return tuple(loaders)  # type: ignore[return-value]
