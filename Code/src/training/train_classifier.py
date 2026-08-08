"""Train protein-representation classifiers.

The module supports a single run and the Stage 1 Cartesian sweep. Run it from
the repository root with either of these equivalent commands::

    python -m Code.src.training.train_classifier --help
    python Code/src/training/train_classifier.py --help

Pipeline going forward:
1. Stage 1: Hyperparameter tuning for the best representation/head combination
python -m Code.src.training.train_classifier \
  --hp_tune \
  --representations \
    random_autoencoder \
    trained_autoencoder \
    trained_autoencoder+esm2
  --version <number>

2. Stage 2: Run the experiment using the from the best configuration (for each representation/head combination)
The final experiment should:
    1. Freeze the selected hyperparameters.
    2. Retrain each representation/head condition from scratch.
    3. Use seeds 42, 43, and 44.
    4. Continue using validation data for early stopping.
    5. Evaluate the clean test split once per final seeded run.
    6. Aggregate test metrics across the three seeds.

End-to-end sweep (all representation encoders trainable):
python -m Code.src.training.train_classifier \
  --end_to_end_sweep \
  --version <number> \
  --encoder_learning_rate 1e-4 \
  --esm_learning_rate 1e-5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

# Make direct script execution behave like module execution.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from Code.src.models.classifier import (
    CANONICAL_EMBEDDING_TYPES,
    CachedEmbeddingClassifier,
    ProteinSequenceClassifier,
    normalize_embedding_type,
)
from Code.src.training.classification_pipeline import (
    ProteinClassificationTrainingPipeline,
    save_json,
)
from Code.src.utils.classifier_data import (
    configure_reproducibility,
    create_sequence_dataloader as _create_sequence_dataloader_impl,
    load_state_dict as _load_state_dict,
    read_classification_split as _read_classification_split,
    resolve_split_source as _resolve_split_source,
    seed_worker,
    validate_dataset_integrity,
    validate_preflight,
)
from Code.src.utils.classifier_tuning_results import (
    select_tuning_hyperparameters,
    tuning_metrics_from_history,
)
from Code.src.utils.embedding_cache import (
    EMBEDDING_CACHE_SCHEMA_VERSION,
    CachedEmbeddingDataset,
    atomic_torch_save as _atomic_torch_save,
    build_embedding_cache as _build_embedding_cache_impl,
    cached_embedding_dataloader as _cached_embedding_dataloader,
    create_run_dataloaders as _create_run_dataloaders_impl,
    embedding_cache_metadata as _embedding_cache_metadata_impl,
    embedding_cache_path as _embedding_cache_path_impl,
    ensure_embedding_caches as _ensure_embedding_caches_impl,
    load_embedding_cache as _load_embedding_cache,
)
from Code.src.utils.experiment_artifacts import (
    archive_run_dir as _archive_run_dir,
    attach_run_log as _attach_run_log,
    _cached_file_sha256,
    config_payload as _config_payload_impl,
    data_source_metadata as _data_source_metadata_impl,
    file_sha256 as _file_sha256,
    git_metadata as _git_metadata_impl,
    is_complete as _is_complete,
    read_json as _read_json,
    runtime_metadata as _runtime_metadata,
    status_path as _status_path,
    utc_now as _utc_now,
    validate_existing_config as _validate_existing_config,
)

logger = logging.getLogger(__name__)

STAGE1_SEEDS = (42, 43, 44)
STAGE1_REPRESENTATIONS = (
    "random_autoencoder",
    "trained_autoencoder",
    "esm2",
    "trained_autoencoder+esm2",
)
TUNING_SEED = 42
TUNING_REPRESENTATIONS = STAGE1_REPRESENTATIONS
TUNING_LEARNING_RATES = (1e-4, 1e-5, 1e-6)
# TUNING_LEARNING_RATES = (1e-4,)
# TUNING_LEARNING_RATES = (1e-5,)
# TUNING_LEARNING_RATES = (1e-6,)
TUNING_WEIGHT_DECAYS = (0.0, 1e-4)
TUNING_MLP_DROPOUTS = (0.1, 0.3)
END_TO_END_TUNING_WEIGHT_DECAYS = (0.0, 1e-4)
END_TO_END_TUNING_DROPOUTS = {
    "random_autoencoder": ((0.0, 0.0), (0.2, 0.0)),
    "trained_autoencoder": ((0.0, 0.0), (0.1, 0.0), (0.2, 0.0)),
    "esm2": ((0.0, 0.0), (0.0, 0.1)),
    "trained_autoencoder+esm2": (
        (0.0, 0.0),
        (0.1, 0.05),
        (0.2, 0.1),
    ),
}
END_TO_END_TUNING_EPOCHS = 10
END_TO_END_TUNING_PATIENCE = 3
HEAD_TYPES = ("linear", "mlp")
# Tuning starts with these known winners so a partial --representations run still
# produces a complete selection file. Results from requested conditions replace
# their defaults below.
DEFAULT_SELECTED_HYPERPARAMETERS = {
    "linear": {
        "random_autoencoder": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "selection_source": "hardcoded_reuse",
        },
        "esm2": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "selection_source": "hardcoded_reuse",
        },
        "trained_autoencoder": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "selection_source": "hardcoded_default",
        },
        "trained_autoencoder+esm2": {
            "learning_rate": 3e-4,
            "weight_decay": 0.0,
            "selection_source": "hardcoded_default",
        },
    },
    "mlp": {
        "random_autoencoder": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "dropout": 0.1,
            "selection_source": "hardcoded_reuse",
        },
        "esm2": {
            "learning_rate": 3e-4,
            "weight_decay": 0.0,
            "dropout": 0.3,
            "selection_source": "hardcoded_reuse",
        },
        "trained_autoencoder": {
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "dropout": 0.1,
            "selection_source": "hardcoded_default",
        },
        "trained_autoencoder+esm2": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "dropout": 0.1,
            "selection_source": "hardcoded_default",
        },
    },
}
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
    "Code/src/utils/classifier_data.py",
    "Code/src/utils/classifier_tuning_results.py",
    "Code/src/utils/dataloader.py",
    "Code/src/utils/embedding_cache.py",
    "Code/src/utils/experiment_artifacts.py",
    "Code/src/utils/sequence_dataset.py",
)


@dataclass(frozen=True)
class ClassifierRunConfig:
    dataset: str
    data_dir: str
    results_dir: str
    checkpoint_root: str
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
    end_to_end: bool
    max_grad_norm: float | None
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    use_cache: bool
    deterministic: bool
    evaluate_test: bool
    device: str
    mode: str
    dropout: float | None
    autoencoder_embedding_dropout: float
    esm_embedding_dropout: float
    phase: str
    cache_embeddings: bool
    embedding_cache_root: str
    encoder_mode: str = "frozen"
    autoencoder_version: str | None = None
    autoencoder_layer_type: str = "gru"
    freeze_autoencoder: bool = True
    freeze_esm2: bool = True

    @property
    def version_dir(self) -> str:
        return self.version if self.version.startswith("v") else f"v{self.version}"

    @staticmethod
    def _format_trial_value(value: float) -> str:
        if value == 0:
            return "0"
        formatted = f"{value:.0e}"
        return formatted.replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")

    @property
    def trial_name(self) -> str:
        parts = [
            f"lr_{self._format_trial_value(self.learning_rate)}",
            f"wd_{self._format_trial_value(self.weight_decay)}",
        ]
        if self.head_type == "mlp":
            if self.dropout is None:
                raise ValueError("MLP runs require an explicit dropout value.")
            parts.append(f"do_{self.dropout:g}")
        if self.phase == "end_to_end_tuning":
            parts.extend(
                [
                    f"ae_do_{self.autoencoder_embedding_dropout:g}",
                    f"esm_do_{self.esm_embedding_dropout:g}",
                ]
            )
        parts.append(f"seed_{self.seed}")
        return "_".join(parts)

    @property
    def experiment_stage_dir(self) -> Path:
        mode_dir, stage_dir = {
            "tuning": ("frozen", "tuning"),
            "final": ("frozen", "final"),
            "end_to_end_tuning": ("unfrozen", "tuning"),
            "end_to_end": ("unfrozen", "final"),
        }[self.phase]
        return Path(mode_dir) / stage_dir

    @property
    def run_dir(self) -> Path:
        root = Path(self.results_dir) / self.dataset / self.version_dir
        if self.phase == "single" and self.encoder_mode != "frozen":
            return (
                root
                / self.encoder_mode
                / self.representation
                / self.head_type
                / f"seed_{self.seed}"
            )
        if self.phase in {"tuning", "final", "end_to_end", "end_to_end_tuning"}:
            leaf = (
                self.trial_name
                if self.phase in {"tuning", "end_to_end_tuning"}
                else f"seed_{self.seed}"
            )
            return (
                root
                / self.experiment_stage_dir
                / self.head_type
                / self.representation
                / leaf
            )
        return (
            root
            / self.representation
            / self.head_type
            / f"seed_{self.seed}"
        )

    @property
    def checkpoint_dir(self) -> Path:
        root = (
            Path(self.checkpoint_root).expanduser()
            / self.dataset
            / self.version_dir
        )
        if self.phase == "single" and self.encoder_mode != "frozen":
            return (
                root
                / self.encoder_mode
                / self.representation
                / self.head_type
                / f"seed_{self.seed}"
            )
        if self.phase in {"tuning", "final", "end_to_end", "end_to_end_tuning"}:
            leaf = (
                self.trial_name
                if self.phase in {"tuning", "end_to_end_tuning"}
                else f"seed_{self.seed}"
            )
            return (
                root
                / self.experiment_stage_dir
                / self.head_type
                / self.representation
                / leaf
            )
        return (
            root
            / self.representation
            / self.head_type
            / f"seed_{self.seed}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train protein sequence classifiers")
    parser.add_argument("--dataset", default="solubility", choices=["solubility", "localization"])
    parser.add_argument("--data_dir", default="data/processed/peer")
    parser.add_argument("--results_dir", default="Code/results/classifier")
    parser.add_argument(
        "--checkpoint_dir",
        dest="checkpoint_root",
        default=str(PROJECT_ROOT / "checkpoints" / "classifier"),
        help=(
            "Root for classifier model checkpoints; defaults to "
            "<project>/checkpoints/classifier. Task/version/run components are appended."
        ),
    )
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
    parser.add_argument(
        "--tuning_learning_rates",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional subset of classifier learning rates for --hp_tune. "
            "Use one value per EC2 instance to partition the tuning grid "
            "without editing source code."
        ),
    )
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--encoder_learning_rate", type=float, default=1e-3)
    parser.add_argument("--esm_learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--unfreeze_layers", type=int, default=0)
    parser.add_argument("--unfreeze_all_esm", action="store_true")
    parser.add_argument("--unfreeze_esm", action="store_true")
    parser.add_argument(
        "--freeze_autoencoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze autoencoder encoding parameters (enabled by default).",
    )
    parser.add_argument(
        "--freeze_esm2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze ESM-2 parameters (enabled by default).",
    )
    parser.add_argument(
        "--end_to_end",
        "--full_end_to_end",
        action="store_true",
        help=(
            "Train the classifier head, trained autoencoder encoder, and complete "
            "ESM-2 model jointly. Requires --representation trained_autoencoder+esm2."
        ),
    )
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
    parser.add_argument("--autoencoder_version", default=None)
    parser.add_argument(
        "--autoencoder_layer_type",
        choices=("gru", "lstm", "transformer"),
        default="gru",
    )
    parser.add_argument("--autoencoder_embedding_dim", type=int, default=256)
    parser.add_argument("--autoencoder_cnn_channels", type=int, default=256)
    parser.add_argument("--autoencoder_hidden_dim", type=int, default=512)
    parser.add_argument("--autoencoder_latent_dim", type=int, default=512)
    parser.add_argument("--autoencoder_num_layers", type=int, default=2)
    parser.add_argument("--autoencoder_kernel_size", type=int, default=5)
    parser.add_argument("--autoencoder_embedding_dropout", type=float, default=0.0)
    parser.add_argument("--esm_embedding_dropout", type=float, default=0.0)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument(
        "--final_epochs",
        type=int,
        default=10,
        help="Maximum epochs for the automatic final sweep after end-to-end tuning.",
    )
    parser.add_argument(
        "--final_early_stopping_patience",
        type=int,
        default=3,
        help="Early-stopping patience for the automatic final sweep.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Defaults to enabled on CUDA and disabled elsewhere.",
    )
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cache_embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Precompute and reuse frozen encoder outputs (enabled by default).",
    )
    parser.add_argument(
        "--embedding_cache_dir",
        dest="embedding_cache_root",
        default=str(PROJECT_ROOT / "data" / "processed" / "embeddings"),
        help="Root directory for fingerprinted frozen-embedding cache files.",
    )
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluate_test", action=argparse.BooleanOptionalAction, default=True)

    experiment_mode = parser.add_mutually_exclusive_group()
    experiment_mode.add_argument(
        "--sweep",
        "--run_experiment",
        dest="run_sweep",
        action="store_true",
        help="Run the Stage 1 representation/head/seed sweep.",
    )
    parser.add_argument(
        "--run_final_after_tuning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically run missing final seeded runs after successful end-to-end tuning.",
    )
    experiment_mode.add_argument(
        "--hp_tune",
        action="store_true",
        help="Run hyperparameter tuning for the classifier.",
    )
    parser.add_argument(
        "--include_unfrozen_tuning",
        "--include_end_to_end_tuning",
        action="store_true",
        help=(
            "With --hp_tune, run the frozen grid first and then the trainable-"
            "encoder grid using the frozen winners from this invocation. Requires "
            "exactly one --tuning_learning_rates value and does not run final sweeps."
        ),
    )
    experiment_mode.add_argument(
        "--end_to_end_sweep",
        action="store_true",
        help=(
            "Run the representation/head/seed sweep with every selected encoder "
            "trainable. Frozen-embedding caching is disabled automatically."
        ),
    )
    experiment_mode.add_argument(
        "--end_to_end_hp_tune",
        action="store_true",
        help=(
            "Tune representation-specific embedding dropout and weight decay with "
            "trainable encoders. Uses one seed, 10 epochs, patience 3, and no test evaluation."
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=[*CANONICAL_EMBEDDING_TYPES, *LEGACY_REPRESENTATIONS],
        default=None,
    )
    parser.add_argument("--head_types", nargs="+", choices=HEAD_TYPES, default=None)
    parser.add_argument(
        "--selected_hyperparameters",
        default=None,
        help=(
            "JSON file containing per-head/per-representation tuning winners. "
            "With --sweep, defaults to "
            "<results_dir>/<dataset>/v<version>/frozen/tuning/"
            "selected_hyperparameters.json "
            "when that file exists."
        ),
    )
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
        "final_epochs": args.final_epochs,
        "final_early_stopping_patience": args.final_early_stopping_patience,
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
    if args.tuning_learning_rates is not None and any(
        learning_rate <= 0 for learning_rate in args.tuning_learning_rates
    ):
        parser.error("--tuning_learning_rates values must be positive")
    if args.tuning_learning_rates is not None and any(
        learning_rate not in TUNING_LEARNING_RATES
        for learning_rate in args.tuning_learning_rates
    ):
        parser.error(
            "--tuning_learning_rates must be a subset of "
            f"{TUNING_LEARNING_RATES}"
        )
    if args.tuning_learning_rates is not None and len(
        set(args.tuning_learning_rates)
    ) != len(args.tuning_learning_rates):
        parser.error("--tuning_learning_rates values must be distinct")
    if args.tuning_learning_rates is not None and not args.hp_tune:
        parser.error("--tuning_learning_rates requires --hp_tune")
    if args.include_unfrozen_tuning and not args.hp_tune:
        parser.error("--include_unfrozen_tuning requires --hp_tune")
    if args.include_unfrozen_tuning and (
        args.tuning_learning_rates is None
        or len(args.tuning_learning_rates) != 1
    ):
        parser.error(
            "--include_unfrozen_tuning requires exactly one "
            "--tuning_learning_rates value"
        )
    if args.include_unfrozen_tuning and args.selected_hyperparameters is not None:
        parser.error(
            "--include_unfrozen_tuning cannot use --selected_hyperparameters; "
            "it consumes the frozen winners produced by the same invocation"
        )
    if args.weight_decay < 0:
        parser.error("--weight_decay must be non-negative")
    for name in ("autoencoder_embedding_dropout", "esm_embedding_dropout"):
        if not 0.0 <= getattr(args, name) < 1.0:
            parser.error(f"--{name} must be in the range [0, 1)")
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
    if args.end_to_end and (args.unfreeze_esm or args.unfreeze_all_esm):
        parser.error("--end_to_end cannot be combined with ESM-only unfreezing options")
    if args.run_sweep and (
        args.unfreeze_esm
        or args.unfreeze_all_esm
        or args.unfreeze_layers
        or args.end_to_end
        or not args.freeze_autoencoder
        or not args.freeze_esm2
    ):
        parser.error("Stage 1 sweep requires fully frozen encoders; unfreezing options are not allowed")
    if args.hp_tune and (
        args.unfreeze_esm
        or args.unfreeze_all_esm
        or args.unfreeze_layers
        or args.end_to_end
        or not args.freeze_autoencoder
        or not args.freeze_esm2
    ):
        parser.error(
            "Frozen hyperparameter tuning requires frozen encoders; use "
            "--end_to_end_hp_tune for trainable encoders."
        )
    if (args.end_to_end_sweep or args.end_to_end_hp_tune) and (
        args.end_to_end
        or args.unfreeze_esm
        or args.unfreeze_all_esm
        or args.unfreeze_layers
        or not args.freeze_autoencoder
        or not args.freeze_esm2
    ):
        parser.error(
            "end-to-end experiment modes cannot be combined with individual "
            "unfreezing options"
        )
    if not args.run_sweep and (args.unfreeze_esm or args.unfreeze_all_esm):
        if normalize_embedding_type(args.embedding_type) != "esm2":
            parser.error("ESM unfreezing options are supported only with --representation esm2")
    if args.end_to_end and normalize_embedding_type(args.embedding_type) != "trained_autoencoder+esm2":
        parser.error("--end_to_end requires --representation trained_autoencoder+esm2")
    if not any(
        (args.run_sweep, args.hp_tune, args.end_to_end_sweep, args.end_to_end_hp_tune)
    ):
        representation = normalize_embedding_type(args.embedding_type)
        if not args.freeze_autoencoder and "autoencoder" not in representation:
            parser.error(
                "--no-freeze_autoencoder requires an autoencoder representation"
            )
        if not args.freeze_esm2 and "esm2" not in representation:
            parser.error("--no-freeze_esm2 requires an ESM-2 representation")
        if (
            representation == "trained_autoencoder+esm2"
            and args.freeze_autoencoder != args.freeze_esm2
        ):
            parser.error(
                "The combined representation requires both encoders frozen or "
                "both encoders trainable."
            )


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


def _make_run_config(
    args: argparse.Namespace,
    *,
    device: str,
    representation: str,
    head_type: str,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float | None,
    evaluate_test: bool,
    mode: str,
    phase: str,
    autoencoder_embedding_dropout: float | None = None,
    esm_embedding_dropout: float | None = None,
    epochs: int | None = None,
    early_stopping_patience: int | None = None,
) -> ClassifierRunConfig:
    num_classes = args.num_classes or (10 if args.dataset == "localization" else 2)
    pin_memory = args.pin_memory if args.pin_memory is not None else device == "cuda"
    persistent_workers = bool(args.persistent_workers and args.num_workers > 0)
    train_all_encoders = bool(
        args.end_to_end or args.end_to_end_sweep or args.end_to_end_hp_tune
    )
    uses_autoencoder = "autoencoder" in representation
    uses_trained_autoencoder = representation in {
        "trained_autoencoder",
        "trained_autoencoder+esm2",
    }
    uses_esm2 = "esm2" in representation
    freeze_autoencoder = (
        not train_all_encoders if uses_autoencoder else True
    )
    freeze_esm2 = not train_all_encoders if uses_esm2 else True
    if uses_autoencoder and not train_all_encoders:
        freeze_autoencoder = args.freeze_autoencoder
    if uses_esm2 and not train_all_encoders:
        freeze_esm2 = args.freeze_esm2
    if uses_esm2 and (args.unfreeze_esm or args.unfreeze_all_esm):
        freeze_esm2 = False
    encoders_frozen = freeze_autoencoder and freeze_esm2
    encoder_mode = (
        "frozen"
        if encoders_frozen
        else "from_scratch"
        if representation == "random_autoencoder"
        else "fine_tuned"
    )
    autoencoder_version = None
    if uses_trained_autoencoder:
        autoencoder_version = args.autoencoder_version
        if (
            autoencoder_version is None
            and args.autoencoder_checkpoint is not None
        ):
            autoencoder_version = Path(args.autoencoder_checkpoint).expanduser().parent.name
    return ClassifierRunConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        checkpoint_root=args.checkpoint_root,
        version=str(args.version),
        representation=representation,
        encoder_mode=encoder_mode,
        head_type=head_type,
        seed=seed,
        num_classes=num_classes,
        batch_size=args.batch_size,
        epochs=args.epochs if epochs is None else epochs,
        early_stopping_patience=(
            args.early_stopping_patience
            if early_stopping_patience is None
            else early_stopping_patience
        ),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        encoder_learning_rate=args.encoder_learning_rate,
        esm_learning_rate=args.esm_learning_rate,
        esm_model_name=args.esm_model_name,
        esm_max_sequence_length=args.esm_max_sequence_length,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        autoencoder_version=autoencoder_version,
        autoencoder_layer_type=args.autoencoder_layer_type,
        autoencoder_embedding_dim=args.autoencoder_embedding_dim,
        autoencoder_cnn_channels=args.autoencoder_cnn_channels,
        autoencoder_hidden_dim=args.autoencoder_hidden_dim,
        autoencoder_latent_dim=args.autoencoder_latent_dim,
        autoencoder_num_layers=args.autoencoder_num_layers,
        autoencoder_kernel_size=args.autoencoder_kernel_size,
        unfreeze_esm=args.unfreeze_esm,
        unfreeze_all_esm=args.unfreeze_all_esm,
        unfreeze_layers=args.unfreeze_layers,
        end_to_end=(
            args.end_to_end or args.end_to_end_sweep or args.end_to_end_hp_tune
        ),
        freeze_autoencoder=freeze_autoencoder,
        freeze_esm2=freeze_esm2,
        max_grad_norm=args.max_grad_norm,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        use_cache=args.use_cache,
        deterministic=args.deterministic,
        evaluate_test=evaluate_test,
        device=device,
        mode=mode,
        dropout=dropout,
        autoencoder_embedding_dropout=(
            args.autoencoder_embedding_dropout
            if autoencoder_embedding_dropout is None
            else autoencoder_embedding_dropout
        ),
        esm_embedding_dropout=(
            args.esm_embedding_dropout
            if esm_embedding_dropout is None
            else esm_embedding_dropout
        ),
        phase=phase,
        cache_embeddings=(
            args.cache_embeddings
            and encoders_frozen
        ),
        embedding_cache_root=args.embedding_cache_root,
    )


def build_tuning_configs(
    args: argparse.Namespace,
    device: str | None = None,
) -> list[ClassifierRunConfig]:
    """Build the tuning grid without evaluating any candidate on the test set."""

    device = device or select_device()
    representations = _unique_normalized(
        args.representations or TUNING_REPRESENTATIONS
    )
    heads = list(dict.fromkeys(args.head_types or HEAD_TYPES))
    configs: list[ClassifierRunConfig] = []

    for head_type, representation in product(heads, representations):
        dropouts: tuple[float | None, ...]
        if head_type == "mlp":
            dropouts = TUNING_MLP_DROPOUTS
        else:
            dropouts = (None,)
        for learning_rate, weight_decay, dropout in product(
            args.tuning_learning_rates or TUNING_LEARNING_RATES,
            TUNING_WEIGHT_DECAYS,
            dropouts,
        ):
            configs.append(
                _make_run_config(
                    args,
                    device=device,
                    representation=representation,
                    head_type=head_type,
                    seed=TUNING_SEED,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    dropout=dropout,
                    evaluate_test=False,
                    mode="hp_tune",
                    phase="tuning",
                )
            )
    return configs


def build_end_to_end_tuning_configs(
    args: argparse.Namespace,
    device: str | None = None,
) -> list[ClassifierRunConfig]:
    """Build the compact end-to-end regularization grid without test evaluation."""

    device = device or select_device()
    representations = _unique_normalized(
        args.representations or STAGE1_REPRESENTATIONS
    )
    heads = list(dict.fromkeys(args.head_types or HEAD_TYPES))
    selected = _load_selected_hyperparameters(args)
    configs: list[ClassifierRunConfig] = []
    for head_type, representation in product(heads, representations):
        learning_rate, _, head_dropout, _, _ = _hyperparameters_for_condition(
            selected,
            head_type=head_type,
            representation=representation,
        )
        for (autoencoder_dropout, esm_dropout), weight_decay in product(
            END_TO_END_TUNING_DROPOUTS[representation],
            END_TO_END_TUNING_WEIGHT_DECAYS,
        ):
            configs.append(
                _make_run_config(
                    args,
                    device=device,
                    representation=representation,
                    head_type=head_type,
                    seed=TUNING_SEED,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    dropout=head_dropout,
                    evaluate_test=False,
                    mode="end_to_end_hp_tune",
                    phase="end_to_end_tuning",
                    autoencoder_embedding_dropout=autoencoder_dropout,
                    esm_embedding_dropout=esm_dropout,
                    epochs=END_TO_END_TUNING_EPOCHS,
                    early_stopping_patience=END_TO_END_TUNING_PATIENCE,
                )
            )
    return configs


def _selected_hyperparameters_path(args: argparse.Namespace) -> Path:
    if args.selected_hyperparameters is not None:
        return Path(args.selected_hyperparameters).expanduser()
    version_dir = (
        str(args.version)
        if str(args.version).startswith("v")
        else f"v{args.version}"
    )
    return (
        Path(args.results_dir)
        / args.dataset
        / version_dir
        / "frozen"
        / "tuning"
        / "selected_hyperparameters.json"
    )


def _load_selected_hyperparameters(
    args: argparse.Namespace,
) -> dict[str, dict[str, dict[str, float]]]:
    path = _selected_hyperparameters_path(args)
    if args.selected_hyperparameters is None and not path.is_file():
        legacy_path = (
            Path(args.results_dir)
            / args.dataset
            / (
                str(args.version)
                if str(args.version).startswith("v")
                else f"v{args.version}"
            )
            / "tuning"
            / "selected_hyperparameters.json"
        )
        if legacy_path.is_file():
            logger.warning(
                "Using legacy frozen tuning selection path: %s", legacy_path
            )
            path = legacy_path
    if not path.is_file():
        raise FileNotFoundError(
            "Final sweeps require selected tuning hyperparameters; file not found: "
            f"{path}. Run --hp_tune first or pass --selected_hyperparameters."
        )

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Selected hyperparameters must be a JSON object: {path}")
    return payload


def _hyperparameters_for_condition(
    selected: dict[str, Any],
    *,
    head_type: str,
    representation: str,
) -> tuple[float, float, float | None, float, float]:
    head_parameters = selected.get(head_type)
    if not isinstance(head_parameters, dict):
        raise ValueError(
            f"Selected hyperparameters have no valid {head_type!r} section."
        )
    parameters = head_parameters.get(representation)
    if parameters is None:
        raise ValueError(
            "Selected hyperparameters are missing the requested condition: "
            f"{head_type}/{representation}."
        )
    if not isinstance(parameters, dict):
        raise ValueError(
            "Selected hyperparameters for "
            f"{head_type}/{representation} must be a JSON object."
        )

    try:
        learning_rate = float(parameters["learning_rate"])
        weight_decay = float(parameters["weight_decay"])
        dropout = (
            float(parameters["dropout"]) if head_type == "mlp" else None
        )
        autoencoder_embedding_dropout = float(
            parameters.get("autoencoder_embedding_dropout", 0.0)
        )
        esm_embedding_dropout = float(parameters.get("esm_embedding_dropout", 0.0))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid selected hyperparameters for {head_type}/{representation}."
        ) from error
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError(
            f"Invalid selected hyperparameters for {head_type}/{representation}."
        )
    if dropout is not None and not 0.0 <= dropout < 1.0:
        raise ValueError(
            f"Invalid selected hyperparameters for {head_type}/{representation}."
        )
    if not 0.0 <= autoencoder_embedding_dropout < 1.0:
        raise ValueError(
            f"Invalid autoencoder embedding dropout for {head_type}/{representation}."
        )
    if not 0.0 <= esm_embedding_dropout < 1.0:
        raise ValueError(
            f"Invalid ESM embedding dropout for {head_type}/{representation}."
        )
    return (
        learning_rate,
        weight_decay,
        dropout,
        autoencoder_embedding_dropout,
        esm_embedding_dropout,
    )


def build_run_configs(
    args: argparse.Namespace,
    device: str | None = None,
) -> list[ClassifierRunConfig]:
    device = device or select_device()
    if args.hp_tune:
        return build_tuning_configs(args, device=device)
    if args.end_to_end_hp_tune:
        return build_end_to_end_tuning_configs(args, device=device)

    if args.run_sweep or args.end_to_end_sweep:
        seeds = list(dict.fromkeys(args.seeds or STAGE1_SEEDS))
        representations = _unique_normalized(
            args.representations or STAGE1_REPRESENTATIONS
        )
        heads = list(dict.fromkeys(args.head_types or HEAD_TYPES))
        mode = "end_to_end_sweep" if args.end_to_end_sweep else "stage1_sweep"
        phase = "end_to_end" if args.end_to_end_sweep else "final"
        selected = _load_selected_hyperparameters(args)
    else:
        seeds = [args.seed]
        representations = [normalize_embedding_type(args.embedding_type)]
        heads = [args.head_type]
        mode = "single"
        phase = "single"
        selected = {}

    configs: list[ClassifierRunConfig] = []
    for seed, representation, head_type in product(seeds, representations, heads):
        if args.run_sweep or args.end_to_end_sweep:
            (
                learning_rate,
                weight_decay,
                dropout,
                autoencoder_embedding_dropout,
                esm_embedding_dropout,
            ) = _hyperparameters_for_condition(
                selected,
                head_type=head_type,
                representation=representation,
            )
        else:
            learning_rate = args.learning_rate
            weight_decay = args.weight_decay
            dropout = 0.1 if head_type == "mlp" else None
            autoencoder_embedding_dropout = args.autoencoder_embedding_dropout
            esm_embedding_dropout = args.esm_embedding_dropout
        configs.append(
            _make_run_config(
                args,
                device=device,
                representation=representation,
                head_type=head_type,
                seed=seed,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                dropout=dropout,
                evaluate_test=args.evaluate_test,
                mode=mode,
                phase=phase,
                autoencoder_embedding_dropout=autoencoder_embedding_dropout,
                esm_embedding_dropout=esm_embedding_dropout,
            )
        )
    return configs


def _create_sequence_dataloader(
    config: ClassifierRunConfig,
    split: str,
    *,
    shuffle: bool,
    offset: int,
) -> DataLoader:
    return _create_sequence_dataloader_impl(
        config, split, shuffle=shuffle, offset=offset
    )


def _embedding_cache_metadata(
    config: ClassifierRunConfig,
    split: str,
) -> dict[str, Any]:
    return _embedding_cache_metadata_impl(
        config,
        split,
        project_root=PROJECT_ROOT,
        resolve_split_source=_resolve_split_source,
    )


def _embedding_cache_path(
    config: ClassifierRunConfig,
    split: str,
) -> tuple[Path, dict[str, Any]]:
    return _embedding_cache_path_impl(
        config,
        split,
        project_root=PROJECT_ROOT,
        resolve_split_source=_resolve_split_source,
    )


def _build_embedding_cache(
    config: ClassifierRunConfig,
    split: str,
    model: ProteinSequenceClassifier,
    path: Path,
    metadata: dict[str, Any],
) -> None:
    _build_embedding_cache_impl(
        config,
        split,
        model,
        path,
        metadata,
        sequence_dataloader=_create_sequence_dataloader,
    )


def _ensure_embedding_caches(
    config: ClassifierRunConfig,
    splits: list[str],
) -> dict[str, dict[str, Any]]:
    return _ensure_embedding_caches_impl(
        config,
        splits,
        cache_path=_embedding_cache_path,
        build_cache=_build_embedding_cache,
        model_factory=create_model,
    )


def create_run_dataloaders(config: ClassifierRunConfig):
    return _create_run_dataloaders_impl(
        config,
        ensure_caches=_ensure_embedding_caches,
        sequence_dataloader=_create_sequence_dataloader,
    )


def create_model(
    config: ClassifierRunConfig,
    *,
    use_cached_embeddings: bool | None = None,
) -> ProteinSequenceClassifier | CachedEmbeddingClassifier:
    use_cached_embeddings = (
        config.cache_embeddings
        if use_cached_embeddings is None
        else use_cached_embeddings
    )
    if use_cached_embeddings:
        embedding_dim = (
            config.autoencoder_latent_dim
            if config.representation in {"random_autoencoder", "trained_autoencoder"}
            else 320
        )
        if config.representation == "trained_autoencoder+esm2":
            embedding_dim = config.autoencoder_latent_dim + 320
        return CachedEmbeddingClassifier(
            embedding_type=config.representation,
            embedding_dim=embedding_dim,
            num_classes=config.num_classes,
            head_type=config.head_type,
            dropout=config.dropout if config.dropout is not None else 0.0,
            head_seed=config.seed,
            device=config.device,
        )
    return ProteinSequenceClassifier(
        embedding_type=config.representation,
        num_classes=config.num_classes,
        esm_model_name=config.esm_model_name,
        esm_max_sequence_length=config.esm_max_sequence_length,
        head_type=config.head_type,
        dropout=config.dropout if config.dropout is not None else 0.0,
        head_seed=config.seed,
        autoencoder_checkpoint=config.autoencoder_checkpoint,
        autoencoder_layer_type=config.autoencoder_layer_type,
        autoencoder_embedding_dim=config.autoencoder_embedding_dim,
        autoencoder_cnn_channels=config.autoencoder_cnn_channels,
        autoencoder_hidden_dim=config.autoencoder_hidden_dim,
        autoencoder_latent_dim=config.autoencoder_latent_dim,
        autoencoder_num_layers=config.autoencoder_num_layers,
        autoencoder_kernel_size=config.autoencoder_kernel_size,
        autoencoder_embedding_dropout=config.autoencoder_embedding_dropout,
        esm_embedding_dropout=config.esm_embedding_dropout,
        freeze_autoencoder=config.freeze_autoencoder,
        freeze_esm2=config.freeze_esm2,
        device=config.device,
    )


def _data_source_metadata(config: ClassifierRunConfig) -> dict[str, Any]:
    return _data_source_metadata_impl(config, _resolve_split_source)


def _git_metadata() -> dict[str, Any]:
    return _git_metadata_impl(PROJECT_ROOT)


def _config_payload(config: ClassifierRunConfig) -> dict[str, Any]:
    return _config_payload_impl(
        config,
        project_root=PROJECT_ROOT,
        fingerprinted_source_files=FINGERPRINTED_SOURCE_FILES,
        embedding_cache_schema_version=EMBEDDING_CACHE_SCHEMA_VERSION,
        embedding_cache_path=_embedding_cache_path,
        resolve_split_source=_resolve_split_source,
    )


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
        "encoder_mode": config.encoder_mode,
        "freeze_autoencoder": config.freeze_autoencoder,
        "freeze_esm2": config.freeze_esm2,
        "autoencoder_version": config.autoencoder_version,
        "head_type": config.head_type,
        "seed": config.seed,
        "phase": config.phase,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "dropout": config.dropout,
        "autoencoder_embedding_dropout": config.autoencoder_embedding_dropout,
        "esm_embedding_dropout": config.esm_embedding_dropout,
        "status": status,
        "run_dir": str(config.run_dir),
        "checkpoint_dir": str(config.checkpoint_dir),
        "error": error,
    }
    if metrics:
        row.update(metrics)
    return row


def _tuning_metrics_from_history(
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return tuning_metrics_from_history(history)


def _read_tuning_metrics(run_dir: Path) -> dict[str, Any]:
    history = pd.read_csv(
        run_dir / "history.csv", float_precision="round_trip"
    ).to_dict(orient="records")
    return _tuning_metrics_from_history(history)


def run_one(
    config: ClassifierRunConfig,
    *,
    resume: bool,
    overwrite: bool,
    skip_completed: bool,
) -> dict[str, Any]:
    run_dir = config.run_dir
    checkpoint_dir = config.checkpoint_dir
    payload = _config_payload(config)
    if (
        _is_complete(run_dir, config.evaluate_test, checkpoint_dir)
        and skip_completed
        and not overwrite
    ):
        existing_config = _read_json(run_dir / "config.json")
        try:
            _validate_existing_config(existing_config, payload, for_resume=False)
        except ValueError:
            if not resume:
                raise
        else:
            logger.info("Skipping completed run: %s", run_dir)
            if config.phase in {"tuning", "end_to_end_tuning"}:
                metrics = _read_tuning_metrics(run_dir)
            else:
                metrics = _read_json(run_dir / "metrics.json") if config.evaluate_test else None
            return _row_from_metrics(config, metrics, "complete")

    existing_locations = [path for path in (run_dir, checkpoint_dir) if path.exists()]
    if existing_locations and overwrite:
        for path in existing_locations:
            _archive_run_dir(path)
    elif existing_locations and not resume:
        raise FileExistsError(
            "Run artifacts already exist but do not form a validated completed run: "
            f"{', '.join(str(path) for path in existing_locations)}. "
            "Use --resume or --overwrite."
        )

    if existing_locations and resume:
        if not run_dir.is_dir():
            raise ValueError(
                f"Cannot validate resume compatibility because {run_dir} is missing. "
                "Use --overwrite to archive the orphaned checkpoint directory."
            )
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
        last_checkpoint = checkpoint_dir / "last_model.pt"
        prior_training_artifacts = (
            checkpoint_dir / "best_model.pt",
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
        # Cache construction initializes the frozen encoder only on a cache miss.
        # Reset before head creation so cache hits and misses produce identical heads.
        configure_reproducibility(config.seed, config.deterministic)
        model = create_model(config)
        pipeline = ProteinClassificationTrainingPipeline(
            model=model,
            num_classes=config.num_classes,
            device=config.device,
            run_dir=run_dir,
            checkpoint_dir=checkpoint_dir,
            dataset=config.dataset,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            encoder_learning_rate=config.encoder_learning_rate,
            esm_learning_rate=config.esm_learning_rate,
            unfreeze_esm=config.unfreeze_esm,
            unfreeze_layers=config.unfreeze_layers,
            unfreeze_all_esm=config.unfreeze_all_esm,
            end_to_end=config.end_to_end,
            freeze_autoencoder=config.freeze_autoencoder,
            freeze_esm2=config.freeze_esm2,
            max_grad_norm=config.max_grad_norm,
            run_config=payload,
            show_progress=True,
        )

        resume_path = checkpoint_dir / "last_model.pt" if resume else None
        history = pipeline.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.epochs,
            early_stopping_patience=config.early_stopping_patience,
            resume_from=resume_path if resume_path and resume_path.is_file() else None,
        )

        metrics: dict[str, Any] = {}
        if config.phase in {"tuning", "end_to_end_tuning"}:
            metrics = _tuning_metrics_from_history(history)
        elif config.evaluate_test:
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
    is_frozen_tuning = all(config.phase == "tuning" for config in configs)
    is_end_to_end_tuning = all(
        config.phase == "end_to_end_tuning" for config in configs
    )
    is_tuning = is_frozen_tuning or is_end_to_end_tuning
    if is_tuning:
        summary_root = (
            summary_root / "unfrozen" / "tuning"
            if is_end_to_end_tuning
            else summary_root / "frozen" / "tuning"
        )
        summary_path = summary_root / "tuning_results.csv"
    else:
        if all(config.phase == "end_to_end" for config in configs):
            summary_root = summary_root / "unfrozen" / "final"
        elif all(config.phase == "final" for config in configs):
            summary_root = summary_root / "frozen" / "final"
        elif all(
            config.phase == "single" and config.encoder_mode != "frozen"
            for config in configs
        ):
            summary_root = summary_root / configs[0].encoder_mode
        summary_path = summary_root / "summary.csv"
    summary = pd.DataFrame(rows)
    if summary_path.is_file():
        try:
            existing = pd.read_csv(summary_path)
            summary = pd.concat([existing, summary], ignore_index=True, sort=False)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            logger.warning("Could not merge existing summary %s: %s", summary_path, error)

    def legacy_encoder_state(row: pd.Series) -> tuple[str, bool | None, bool | None]:
        phase = str(row.get("phase", ""))
        representation = str(row.get("representation", ""))
        if phase in {"end_to_end", "end_to_end_tuning"}:
            mode = (
                "from_scratch"
                if representation == "random_autoencoder"
                else "fine_tuned"
            )
            return (
                mode,
                "autoencoder" not in representation,
                "esm2" not in representation,
            )
        if phase in {"final", "tuning"}:
            return "frozen", True, True
        return "unknown", None, None

    inferred_state = summary.apply(legacy_encoder_state, axis=1)
    state_defaults = {
        "encoder_mode": inferred_state.map(lambda state: state[0]),
        "freeze_autoencoder": inferred_state.map(lambda state: state[1]),
        "freeze_esm2": inferred_state.map(lambda state: state[2]),
    }
    for column, default in state_defaults.items():
        if column not in summary:
            summary[column] = default
        else:
            summary[column] = summary[column].fillna(default)
    identity_columns = [
        "dataset",
        "version",
        "representation",
        "encoder_mode",
        "freeze_autoencoder",
        "freeze_esm2",
        "head_type",
        "seed",
    ]
    if is_tuning:
        identity_columns.extend(
            [
                "learning_rate",
                "weight_decay",
                "dropout",
                "autoencoder_embedding_dropout",
                "esm_embedding_dropout",
            ]
        )
    if not summary.empty and set(identity_columns).issubset(summary.columns):
        summary = summary.drop_duplicates(identity_columns, keep="last")
        summary = summary.sort_values(identity_columns, kind="stable").reset_index(drop=True)
    _atomic_dataframe_csv(summary, summary_path)

    completed = summary[summary["status"] == "complete"].copy()
    if is_tuning:
        requested_conditions = {
            (config.head_type, config.representation) for config in configs
        }
        selected = select_tuning_hyperparameters(
            summary,
            requested_conditions=requested_conditions,
            is_end_to_end_tuning=is_end_to_end_tuning,
            default_selected=DEFAULT_SELECTED_HYPERPARAMETERS,
        )
        save_json(selected, summary_root / "selected_hyperparameters.json")
        return

    if completed.empty:
        _atomic_dataframe_csv(completed, summary_root / "aggregated_summary.csv")
        return
    identity = {
        "dataset",
        "version",
        "representation",
        "encoder_mode",
        "freeze_autoencoder",
        "freeze_esm2",
        "autoencoder_version",
        "head_type",
        "seed",
        "status",
        "run_dir",
        "checkpoint_dir",
        "error",
    }
    metric_columns = [
        column
        for column in completed.columns
        if column not in identity and pd.api.types.is_numeric_dtype(completed[column])
    ]
    group_columns = [
        "representation",
        "head_type",
        "encoder_mode",
        "freeze_autoencoder",
        "freeze_esm2",
    ]
    grouped = completed.groupby(group_columns, dropna=False)
    aggregate = grouped.size().rename("num_seeds").reset_index()
    for metric in metric_columns:
        values = grouped[metric].agg(["mean", "std"]).reset_index()
        values = values.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"})
        aggregate = aggregate.merge(values, on=group_columns, how="left")
    _atomic_dataframe_csv(aggregate, summary_root / "aggregated_summary.csv")


def _execute_configs(
    args: argparse.Namespace,
    configs: list[ClassifierRunConfig],
) -> None:
    validate_preflight(configs)

    if args.hp_tune:
        logger.info("Starting hyperparameter tuning with %d unique runs", len(configs))
    elif args.end_to_end_hp_tune:
        logger.info(
            "Starting end-to-end hyperparameter tuning with %d unique runs",
            len(configs),
        )
    elif args.run_sweep:
        logger.info("Starting Stage 1 sweep with %d unique runs", len(configs))
    elif args.end_to_end_sweep:
        logger.info("Starting end-to-end sweep with %d unique runs", len(configs))

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
            if (
                not args.run_sweep
                and not args.end_to_end_sweep
                and not args.hp_tune
                and not args.end_to_end_hp_tune
            ) or args.fail_fast:
                save_summaries(configs, rows)
                raise

    save_summaries(configs, rows)
    if failures:
        raise RuntimeError(f"{failures} of {len(configs)} classifier runs failed")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    configs = build_run_configs(args)
    print(f"Device: {configs[0].device}", flush=True)
    _execute_configs(args, configs)

    if args.hp_tune and args.include_unfrozen_tuning:
        end_to_end_args = argparse.Namespace(**vars(args))
        end_to_end_args.hp_tune = False
        end_to_end_args.end_to_end_hp_tune = True
        end_to_end_args.include_unfrozen_tuning = False
        end_to_end_args.run_final_after_tuning = False
        end_to_end_configs = build_run_configs(end_to_end_args)
        logger.info(
            "Frozen tuning complete; starting unfrozen tuning with %d unique runs",
            len(end_to_end_configs),
        )
        _execute_configs(end_to_end_args, end_to_end_configs)
        return

    if args.end_to_end_hp_tune and args.run_final_after_tuning:
        final_args = argparse.Namespace(**vars(args))
        final_args.end_to_end_hp_tune = False
        final_args.end_to_end_sweep = True
        final_args.epochs = args.final_epochs
        final_args.early_stopping_patience = args.final_early_stopping_patience
        final_args.selected_hyperparameters = str(
            Path(args.results_dir)
            / args.dataset
            / configs[0].version_dir
            / "unfrozen"
            / "tuning"
            / "selected_hyperparameters.json"
        )
        final_configs = build_run_configs(final_args)
        logger.info(
            "Tuning complete; starting automatic final sweep with %d seeded runs",
            len(final_configs),
        )
        final_rows: list[dict[str, Any]] = []
        final_failures = 0
        for index, config in enumerate(final_configs, start=1):
            logger.info(
                "Final run %d/%d: representation=%s head=%s seed=%d",
                index,
                len(final_configs),
                config.representation,
                config.head_type,
                config.seed,
            )
            try:
                final_rows.append(
                    run_one(
                        config,
                        resume=final_args.resume,
                        overwrite=final_args.overwrite,
                        skip_completed=final_args.skip_completed,
                    )
                )
            except Exception as error:
                final_failures += 1
                final_rows.append(
                    _row_from_metrics(config, None, "failed", str(error))
                )
                if final_args.fail_fast:
                    save_summaries(final_configs, final_rows)
                    raise
        save_summaries(final_configs, final_rows)
        if final_failures:
            raise RuntimeError(
                f"{final_failures} of {len(final_configs)} final classifier runs failed"
            )


if __name__ == "__main__":
    main()
