"""Train frozen protein-representation classifiers.

The module supports a single run and the Stage 1 Cartesian sweep. Run it from
the repository root with either of these equivalent commands::

    python -m Code.src.training.train_classifier --help
    python Code/src/training/train_classifier.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

# Make direct script execution behave like module execution.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from Code.src.models.autoencoder import ProteinSequenceAutoencoder
from Code.src.models.classifier import (
    CANONICAL_EMBEDDING_TYPES,
    ProteinSequenceClassifier,
    normalize_embedding_type,
)
from Code.src.training.classification_pipeline import (
    ProteinClassificationTrainingPipeline,
    save_json,
)
from Code.src.utils.dataloader import create_dataloader
from Code.src.utils.utils import set_random_seed

logger = logging.getLogger(__name__)

STAGE1_SEEDS = (42, 43, 44)
STAGE1_REPRESENTATIONS = (
    "random_autoencoder",
    "trained_autoencoder",
    "esm2",
    "trained_autoencoder+esm2",
)
HEAD_TYPES = ("linear", "mlp")
LEGACY_REPRESENTATIONS = ("autoencoder+esm2",)
DEFAULT_AE_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "autoencoder"
    / "solubility"
    / "v5"
    / "model_ae_solubility.pt"
)
FINGERPRINTED_SOURCE_FILES = (
    "Code/src/models/autoencoder.py",
    "Code/src/models/classifier.py",
    "Code/src/training/classification_pipeline.py",
    "Code/src/training/train_classifier.py",
    "Code/src/utils/dataloader.py",
    "Code/src/utils/sequence_dataset.py",
)


@dataclass(frozen=True)
class ClassifierRunConfig:
    dataset: str
    data_dir: str
    results_dir: str
    version: str
    representation: str
    head_type: str
    seed: int
    num_classes: int
    batch_size: int
    epochs: int
    early_stopping_patience: int
    learning_rate: float
    weight_decay: float
    encoder_learning_rate: float
    esm_learning_rate: float
    esm_model_name: str
    esm_max_sequence_length: int
    autoencoder_checkpoint: str | None
    autoencoder_embedding_dim: int
    autoencoder_cnn_channels: int
    autoencoder_hidden_dim: int
    autoencoder_latent_dim: int
    autoencoder_num_layers: int
    autoencoder_kernel_size: int
    unfreeze_esm: bool
    unfreeze_all_esm: bool
    unfreeze_layers: int
    max_grad_norm: float | None
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    use_cache: bool
    deterministic: bool
    evaluate_test: bool
    device: str
    mode: str

    @property
    def version_dir(self) -> str:
        return self.version if self.version.startswith("v") else f"v{self.version}"

    @property
    def run_dir(self) -> Path:
        return (
            Path(self.results_dir)
            / self.dataset
            / self.version_dir
            / self.representation
            / self.head_type
            / f"seed_{self.seed}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train protein sequence classifiers")
    parser.add_argument("--dataset", default="solubility", choices=["solubility", "localization"])
    parser.add_argument("--data_dir", default="data/processed/peer")
    parser.add_argument("--results_dir", default="Code/results/classifier")
    parser.add_argument("--version", default="1")
    parser.add_argument(
        "--embedding_type",
        "--representation",
        dest="embedding_type",
        default="esm2",
        choices=[*CANONICAL_EMBEDDING_TYPES, *LEGACY_REPRESENTATIONS],
    )
    parser.add_argument("--head_type", default="linear", choices=HEAD_TYPES)
    parser.add_argument("--num_classes", type=int, default=None)

    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--encoder_learning_rate", type=float, default=1e-3)
    parser.add_argument("--esm_learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--unfreeze_layers", type=int, default=0)
    parser.add_argument("--unfreeze_all_esm", action="store_true")
    parser.add_argument("--unfreeze_esm", action="store_true")
    parser.add_argument("--esm_model_name", default="esm2_t6_8M_UR50D")
    parser.add_argument(
        "--esm_max_sequence_length",
        type=int,
        default=1022,
        help="Maximum residues passed to ESM-2; longer sequences are truncated and the value is recorded.",
    )

    # These defaults match the current v5 solubility checkpoint and also define
    # the architecture of the matched random-autoencoder baseline.
    parser.add_argument("--autoencoder_checkpoint", default=str(DEFAULT_AE_CHECKPOINT))
    parser.add_argument("--autoencoder_embedding_dim", type=int, default=256)
    parser.add_argument("--autoencoder_cnn_channels", type=int, default=256)
    parser.add_argument("--autoencoder_hidden_dim", type=int, default=512)
    parser.add_argument("--autoencoder_latent_dim", type=int, default=512)
    parser.add_argument("--autoencoder_num_layers", type=int, default=2)
    parser.add_argument("--autoencoder_kernel_size", type=int, default=5)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Defaults to enabled on CUDA and disabled elsewhere.",
    )
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluate_test", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--sweep",
        "--run_experiment",
        dest="run_sweep",
        action="store_true",
        help="Run the Stage 1 representation/head/seed sweep.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=[*CANONICAL_EMBEDDING_TYPES, *LEGACY_REPRESENTATIONS],
        default=None,
    )
    parser.add_argument("--head_types", nargs="+", choices=HEAD_TYPES, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail_fast", action="store_true")

    args = parser.parse_args(argv)
    _validate_args(args, parser)
    return args


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    positive_ints = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "esm_max_sequence_length": args.esm_max_sequence_length,
        "autoencoder_embedding_dim": args.autoencoder_embedding_dim,
        "autoencoder_cnn_channels": args.autoencoder_cnn_channels,
        "autoencoder_hidden_dim": args.autoencoder_hidden_dim,
        "autoencoder_latent_dim": args.autoencoder_latent_dim,
        "autoencoder_num_layers": args.autoencoder_num_layers,
        "autoencoder_kernel_size": args.autoencoder_kernel_size,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            parser.error(f"--{name} must be positive")
    if args.num_workers < 0:
        parser.error("--num_workers must be non-negative")
    if args.num_classes is not None and args.num_classes < 2:
        parser.error("--num_classes must be at least 2")
    if args.learning_rate <= 0 or args.encoder_learning_rate <= 0 or args.esm_learning_rate <= 0:
        parser.error("learning rates must be positive")
    if args.weight_decay < 0:
        parser.error("--weight_decay must be non-negative")
    if args.max_grad_norm is not None and args.max_grad_norm <= 0:
        parser.error("--max_grad_norm must be positive")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.unfreeze_esm and args.unfreeze_all_esm:
        parser.error("--unfreeze_esm and --unfreeze_all_esm are mutually exclusive")
    if args.unfreeze_esm and args.unfreeze_layers < 1:
        parser.error("--unfreeze_layers must be positive with --unfreeze_esm")
    if not args.unfreeze_esm and args.unfreeze_layers:
        parser.error("--unfreeze_layers has no effect without --unfreeze_esm")
    if args.run_sweep and (args.unfreeze_esm or args.unfreeze_all_esm or args.unfreeze_layers):
        parser.error("Stage 1 sweep requires fully frozen encoders; unfreezing options are not allowed")
    if not args.run_sweep and (args.unfreeze_esm or args.unfreeze_all_esm):
        if normalize_embedding_type(args.embedding_type) != "esm2":
            parser.error("ESM unfreezing options are supported only with --representation esm2")


def _unique_normalized(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        canonical = normalize_embedding_type(value)
        if canonical not in result:
            result.append(canonical)
    return result


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_run_configs(args: argparse.Namespace, device: str | None = None) -> list[ClassifierRunConfig]:
    device = device or select_device()
    num_classes = args.num_classes or (10 if args.dataset == "localization" else 2)
    pin_memory = args.pin_memory if args.pin_memory is not None else device == "cuda"
    persistent_workers = bool(args.persistent_workers and args.num_workers > 0)

    if args.run_sweep:
        seeds = list(dict.fromkeys(args.seeds or STAGE1_SEEDS))
        representations = _unique_normalized(args.representations or STAGE1_REPRESENTATIONS)
        heads = list(dict.fromkeys(args.head_types or HEAD_TYPES))
        mode = "stage1_sweep"
    else:
        seeds = [args.seed]
        representations = [normalize_embedding_type(args.embedding_type)]
        heads = [args.head_type]
        mode = "single"

    configs: list[ClassifierRunConfig] = []
    for seed in seeds:
        for representation in representations:
            for head_type in heads:
                configs.append(
                    ClassifierRunConfig(
                        dataset=args.dataset,
                        data_dir=args.data_dir,
                        results_dir=args.results_dir,
                        version=str(args.version),
                        representation=representation,
                        head_type=head_type,
                        seed=seed,
                        num_classes=num_classes,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        early_stopping_patience=args.early_stopping_patience,
                        learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay,
                        encoder_learning_rate=args.encoder_learning_rate,
                        esm_learning_rate=args.esm_learning_rate,
                        esm_model_name=args.esm_model_name,
                        esm_max_sequence_length=args.esm_max_sequence_length,
                        autoencoder_checkpoint=args.autoencoder_checkpoint,
                        autoencoder_embedding_dim=args.autoencoder_embedding_dim,
                        autoencoder_cnn_channels=args.autoencoder_cnn_channels,
                        autoencoder_hidden_dim=args.autoencoder_hidden_dim,
                        autoencoder_latent_dim=args.autoencoder_latent_dim,
                        autoencoder_num_layers=args.autoencoder_num_layers,
                        autoencoder_kernel_size=args.autoencoder_kernel_size,
                        unfreeze_esm=args.unfreeze_esm,
                        unfreeze_all_esm=args.unfreeze_all_esm,
                        unfreeze_layers=args.unfreeze_layers,
                        max_grad_norm=args.max_grad_norm,
                        num_workers=args.num_workers,
                        pin_memory=pin_memory,
                        persistent_workers=persistent_workers,
                        use_cache=args.use_cache,
                        deterministic=args.deterministic,
                        evaluate_test=args.evaluate_test,
                        device=device,
                        mode=mode,
                    )
                )
    return configs


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
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


def _read_classification_split(
    config: ClassifierRunConfig, split: str
) -> pd.DataFrame:
    path, is_combined = _resolve_split_source(config, split)
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


def validate_dataset_integrity(config: ClassifierRunConfig) -> None:
    """Reject malformed labels and exact sequence leakage before any run starts."""

    sequences_by_split: dict[str, set[str]] = {}
    for split in ("train", "valid", "test"):
        frame = _read_classification_split(config, split)
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


def validate_preflight(configs: list[ClassifierRunConfig]) -> None:
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
            model.load_state_dict(_load_state_dict(checkpoint_path), strict=True)
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


def create_run_dataloaders(config: ClassifierRunConfig):
    loaders = []
    for offset, (split, shuffle) in enumerate(
        (("train", True), ("valid", False), ("test", False))
    ):
        if split == "test" and not config.evaluate_test:
            loaders.append(None)
            continue
        generator = torch.Generator()
        generator.manual_seed(config.seed + offset)
        loaders.append(
            create_dataloader(
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
        )
    return tuple(loaders)


def create_model(config: ClassifierRunConfig) -> ProteinSequenceClassifier:
    return ProteinSequenceClassifier(
        embedding_type=config.representation,
        num_classes=config.num_classes,
        esm_model_name=config.esm_model_name,
        esm_max_sequence_length=config.esm_max_sequence_length,
        head_type=config.head_type,
        autoencoder_checkpoint=config.autoencoder_checkpoint,
        autoencoder_embedding_dim=config.autoencoder_embedding_dim,
        autoencoder_cnn_channels=config.autoencoder_cnn_channels,
        autoencoder_hidden_dim=config.autoencoder_hidden_dim,
        autoencoder_latent_dim=config.autoencoder_latent_dim,
        autoencoder_num_layers=config.autoencoder_num_layers,
        autoencoder_kernel_size=config.autoencoder_kernel_size,
        device=config.device,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@lru_cache(maxsize=64)
def _cached_file_sha256(
    resolved_path: str,
    size_bytes: int,
    mtime_ns: int,
    ctime_ns: int,
) -> str:
    del size_bytes, mtime_ns, ctime_ns  # These values invalidate the cache key.
    digest = hashlib.sha256()
    with Path(resolved_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    resolved = file_path.resolve()
    stat = resolved.stat()
    return _cached_file_sha256(
        str(resolved), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    )


def _resolve_split_source(config: ClassifierRunConfig, split: str) -> tuple[Path, bool]:
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


def _data_source_metadata(config: ClassifierRunConfig) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for split in ("train", "valid", "test"):
        path, is_combined = _resolve_split_source(config, split)
        resolved = path.resolve()
        metadata[split] = {
            "path": str(resolved),
            "sha256": _file_sha256(resolved),
            "combined_file": is_combined,
            "combined_split_value": split if is_combined else None,
        }
    return metadata


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def _runtime_metadata() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for distribution in ("fair-esm", "numpy", "pandas", "scikit-learn", "torch"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda_version": torch.version.cuda,
    }


def _config_payload(config: ClassifierRunConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload.update(
        {
            "run_dir": str(config.run_dir),
            "autoencoder_checkpoint_sha256": _file_sha256(config.autoencoder_checkpoint),
            "data_sources": _data_source_metadata(config),
            "source_file_sha256": {
                relative_path: _file_sha256(PROJECT_ROOT / relative_path)
                for relative_path in FINGERPRINTED_SOURCE_FILES
            },
            "preprocessing": {
                "classification_encoding": "char",
                "autoencoder_special_tokens": "BOS+residues+EOS",
                "esm_long_sequence_policy": "truncate_right",
                "esm_max_sequence_length": config.esm_max_sequence_length,
            },
            "runtime": _runtime_metadata(),
            **_git_metadata(),
        }
    )
    exact_material = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["configuration_fingerprint"] = hashlib.sha256(
        exact_material.encode("utf-8")
    ).hexdigest()
    resume_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"epochs", "evaluate_test", "configuration_fingerprint"}
    }
    resume_material = json.dumps(resume_payload, sort_keys=True, separators=(",", ":"))
    payload["resume_fingerprint"] = hashlib.sha256(
        resume_material.encode("utf-8")
    ).hexdigest()
    return payload


def _validate_existing_config(
    existing: dict[str, Any],
    requested: dict[str, Any],
    *,
    for_resume: bool,
) -> None:
    fingerprint_name = "resume_fingerprint" if for_resume else "configuration_fingerprint"
    existing_fingerprint = existing.get(fingerprint_name)
    requested_fingerprint = requested[fingerprint_name]
    if existing_fingerprint != requested_fingerprint:
        action = "resume" if for_resume else "reuse"
        raise ValueError(
            f"Refusing to {action} {requested['run_dir']}: its saved configuration "
            "does not match the requested code, data, checkpoint, preprocessing, or "
            "hyperparameters. Use --overwrite to archive it and start a new run."
        )
    if for_resume and int(requested["epochs"]) < int(existing.get("epochs", 0)):
        raise ValueError(
            "A resumed run may preserve or extend its epoch budget, but may not "
            "reduce it. Use --overwrite for a shorter run."
        )


def _status_path(run_dir: Path) -> Path:
    return run_dir / "status.json"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return result


def _is_complete(run_dir: Path, evaluate_test: bool) -> bool:
    required = [run_dir / "config.json", run_dir / "history.csv", run_dir / "best_model.pt"]
    if evaluate_test:
        required.extend([run_dir / "metrics.json", run_dir / "test_predictions.csv"])
    status_path = _status_path(run_dir)
    if not status_path.is_file() or not all(path.is_file() for path in required):
        return False
    try:
        return _read_json(status_path).get("status") == "complete"
    except (OSError, ValueError, TypeError):
        return False


def _archive_run_dir(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = run_dir.with_name(f"{run_dir.name}.backup_{suffix}")
    counter = 1
    while archive.exists():
        archive = run_dir.with_name(f"{run_dir.name}.backup_{suffix}_{counter}")
        counter += 1
    run_dir.rename(archive)


def _attach_run_log(run_dir: Path) -> logging.Handler:
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def _row_from_metrics(
    config: ClassifierRunConfig,
    metrics: dict[str, Any] | None,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": config.dataset,
        "version": config.version_dir,
        "representation": config.representation,
        "head_type": config.head_type,
        "seed": config.seed,
        "status": status,
        "run_dir": str(config.run_dir),
        "error": error,
    }
    if metrics:
        row.update(metrics)
    return row


def run_one(
    config: ClassifierRunConfig,
    *,
    resume: bool,
    overwrite: bool,
    skip_completed: bool,
) -> dict[str, Any]:
    run_dir = config.run_dir
    payload = _config_payload(config)
    if _is_complete(run_dir, config.evaluate_test) and skip_completed and not overwrite:
        existing_config = _read_json(run_dir / "config.json")
        try:
            _validate_existing_config(existing_config, payload, for_resume=False)
        except ValueError:
            if not resume:
                raise
        else:
            logger.info("Skipping completed run: %s", run_dir)
            metrics = _read_json(run_dir / "metrics.json") if config.evaluate_test else None
            return _row_from_metrics(config, metrics, "complete")

    if run_dir.exists() and overwrite:
        _archive_run_dir(run_dir)
    elif run_dir.exists() and not resume:
        raise FileExistsError(
            f"Run directory already exists but is not a validated completed run: {run_dir}. "
            "Use --resume or --overwrite."
        )

    if run_dir.exists() and resume:
        existing_config_path = run_dir / "config.json"
        if existing_config_path.is_file():
            _validate_existing_config(
                _read_json(existing_config_path), payload, for_resume=True
            )
        elif any(run_dir.iterdir()):
            raise ValueError(
                f"Cannot validate resume compatibility because {existing_config_path} "
                "is missing. Use --overwrite to archive the incomplete directory."
            )
        last_checkpoint = run_dir / "last_model.pt"
        prior_training_artifacts = (
            run_dir / "best_model.pt",
            run_dir / "history.csv",
            run_dir / "history.json",
        )
        if not last_checkpoint.is_file() and any(
            path.exists() for path in prior_training_artifacts
        ):
            raise FileNotFoundError(
                f"Cannot safely resume {run_dir}: last_model.pt is missing even though "
                "training artifacts exist. Use --overwrite to archive and restart it."
            )

    run_dir.mkdir(parents=True, exist_ok=True)
    log_handler = _attach_run_log(run_dir)
    started = time.time()
    existing_status: dict[str, Any] = {}
    if _status_path(run_dir).is_file():
        try:
            existing_status = _read_json(_status_path(run_dir))
        except (OSError, ValueError, TypeError):
            existing_status = {}
    started_at = existing_status.get("started_at") or _utc_now()
    payload["started_at"] = started_at
    save_json(payload, run_dir / "config.json")
    save_json({"status": "running", "started_at": started_at}, _status_path(run_dir))

    try:
        configure_reproducibility(config.seed, config.deterministic)
        train_loader, val_loader, test_loader = create_run_dataloaders(config)
        model = create_model(config)
        pipeline = ProteinClassificationTrainingPipeline(
            model=model,
            num_classes=config.num_classes,
            device=config.device,
            run_dir=run_dir,
            dataset=config.dataset,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            encoder_learning_rate=config.encoder_learning_rate,
            esm_learning_rate=config.esm_learning_rate,
            unfreeze_esm=config.unfreeze_esm,
            unfreeze_layers=config.unfreeze_layers,
            unfreeze_all_esm=config.unfreeze_all_esm,
            max_grad_norm=config.max_grad_norm,
            run_config=payload,
        )

        resume_path = run_dir / "last_model.pt" if resume else None
        pipeline.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.epochs,
            early_stopping_patience=config.early_stopping_patience,
            resume_from=resume_path if resume_path and resume_path.is_file() else None,
        )

        metrics: dict[str, Any] = {}
        if config.evaluate_test:
            if test_loader is None:
                raise RuntimeError("Test evaluation requested but no test loader was created")
            metrics = pipeline.evaluate_test(test_loader)

        elapsed = time.time() - started
        payload["total_runtime_seconds"] = elapsed
        payload["completed_at"] = _utc_now()
        save_json(payload, run_dir / "config.json")
        save_json(
            {
                "status": "complete",
                "started_at": started_at,
                "completed_at": payload["completed_at"],
                "runtime_seconds": elapsed,
            },
            _status_path(run_dir),
        )
        return _row_from_metrics(config, metrics, "complete")
    except Exception as error:
        save_json(
            {
                "status": "failed",
                "started_at": started_at,
                "failed_at": _utc_now(),
                "runtime_seconds": time.time() - started,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
            _status_path(run_dir),
        )
        logger.exception("Run failed: %s", run_dir)
        raise
    finally:
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()


def _atomic_dataframe_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_summaries(configs: list[ClassifierRunConfig], rows: list[dict[str, Any]]) -> None:
    if not configs:
        return
    summary_root = Path(configs[0].results_dir) / configs[0].dataset / configs[0].version_dir
    summary_path = summary_root / "summary.csv"
    summary = pd.DataFrame(rows)
    if summary_path.is_file():
        try:
            existing = pd.read_csv(summary_path)
            summary = pd.concat([existing, summary], ignore_index=True, sort=False)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            logger.warning("Could not merge existing summary %s: %s", summary_path, error)
    identity_columns = ["dataset", "version", "representation", "head_type", "seed"]
    if not summary.empty and set(identity_columns).issubset(summary.columns):
        summary = summary.drop_duplicates(identity_columns, keep="last")
        summary = summary.sort_values(identity_columns, kind="stable").reset_index(drop=True)
    _atomic_dataframe_csv(summary, summary_path)

    completed = summary[summary["status"] == "complete"].copy()
    if completed.empty:
        _atomic_dataframe_csv(completed, summary_root / "aggregated_summary.csv")
        return
    identity = {
        "dataset",
        "version",
        "representation",
        "head_type",
        "seed",
        "status",
        "run_dir",
        "error",
    }
    metric_columns = [
        column
        for column in completed.columns
        if column not in identity and pd.api.types.is_numeric_dtype(completed[column])
    ]
    grouped = completed.groupby(["representation", "head_type"], dropna=False)
    aggregate = grouped.size().rename("num_seeds").reset_index()
    for metric in metric_columns:
        values = grouped[metric].agg(["mean", "std"]).reset_index()
        values = values.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"})
        aggregate = aggregate.merge(values, on=["representation", "head_type"], how="left")
    _atomic_dataframe_csv(aggregate, summary_root / "aggregated_summary.csv")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    configs = build_run_configs(args)
    validate_preflight(configs)

    if args.run_sweep:
        logger.info("Starting Stage 1 sweep with %d unique runs", len(configs))

    rows: list[dict[str, Any]] = []
    failures = 0
    for index, config in enumerate(configs, start=1):
        logger.info(
            "Run %d/%d: representation=%s head=%s seed=%d",
            index,
            len(configs),
            config.representation,
            config.head_type,
            config.seed,
        )
        try:
            rows.append(
                run_one(
                    config,
                    resume=args.resume,
                    overwrite=args.overwrite,
                    skip_completed=args.skip_completed,
                )
            )
        except Exception as error:
            failures += 1
            rows.append(_row_from_metrics(config, None, "failed", str(error)))
            if not args.run_sweep or args.fail_fast:
                save_summaries(configs, rows)
                raise

    save_summaries(configs, rows)
    if failures:
        raise RuntimeError(f"{failures} of {len(configs)} classifier runs failed")


if __name__ == "__main__":
    main()
