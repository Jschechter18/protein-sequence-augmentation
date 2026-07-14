from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional
from utils.hyperparameters import AutoencoderHyperparameters as AEParams

# LENGTH_QUARTILE_FILE_LABELS = {
#     "s": "short",
#     "ms": "medium_short",
#     "ml": "medium_long",
#     "l": "long",
# }

LENGTH_SPLIT_COUNTS = {
    "halves": 2,
    "thirds": 3,
    "quarters": 4,
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value")


def autoencoder_artifact_stem(
    model: str,
    task: str,
    length_options: Optional[str] = None,
    length_bin: int | None = None,
    is_overfit: bool = False,
    artifact_suffix: str | None = None,
) -> str:
    parts = ["model", model.lower()]
    if length_options is not None:
        if length_bin is None:
            raise ValueError("length_bin must be provided when length_options is set")
        split_count = LENGTH_SPLIT_COUNTS[length_options]
        parts.append(f"length_{length_bin}_of_{split_count}")
    parts.append(task)
    if is_overfit:
        parts.append("overfit")
    if artifact_suffix:
        parts.append(artifact_suffix)
    return "_".join(parts)


def autoencoder_checkpoint_dir(task: str, version: int | str) -> Path:
    version_name = str(version)
    if not version_name:
        raise ValueError("version must not be empty")
    version_dir = version_name if version_name.startswith("v") else f"v{version_name}"
    return PROJECT_ROOT / "checkpoints" / "autoencoder" / task / version_dir


def autoencoder_results_dir(task: str, version: int | str) -> Path:
    version_name = str(version)
    if not version_name:
        raise ValueError("version must not be empty")
    version_dir = version_name if version_name.startswith("v") else f"v{version_name}"
    return PROJECT_ROOT / "Code" / "results" / "autoencoder" / task / version_dir


def autoencoder_artifact_paths(
    model_type: str,
    task: str,
    version: int | str,
    length_options: str | None,
    length_bin: int | None,
    is_overfit: bool,
    artifact_suffix: str | None = None,
) -> tuple[Path, Path]:
    artifact_stem = autoencoder_artifact_stem(
        model_type,
        task,
        length_options,
        length_bin=length_bin,
        is_overfit=is_overfit,
        artifact_suffix=artifact_suffix,
    )
    checkpoint_dir = autoencoder_checkpoint_dir(task, version)
    results_dir = autoencoder_results_dir(task, version)
    checkpoint_path = checkpoint_dir / f"{artifact_stem}.pt"
    history_path = results_dir / f"{results_dir.name}_{artifact_stem}_history.json"
    return checkpoint_path, history_path


def _add_args(args: argparse.ArgumentParser) -> argparse.Namespace:
    args.add_argument('--model', type=str, default='AE', help='Model to train (default: AE)')
    args.add_argument('--task', type=str, default='localization', help='Task to perform (default: localization)')
    args.add_argument('--load_path', type=str, nargs='?', default=None, help='Path to load the best existing model checkpoint (optional)')
    args.add_argument('--version', type=int, default=None, help='Version identifier for this training run (used for checkpoint and history naming)') # should always be unique
    args.add_argument(
        '--overfit_batches',
        type=int,
        default=None,
        help='Debug mode: train and validate on this many batches from the training set.',
    )
    args.add_argument(
        '--overfit_epochs',
        type=int,
        default=None,
        help='Override num_epochs for an overfit/debug run.',
    )
    args.add_argument(
        '--overfit_learning_rate',
        type=float,
        default=None,
        help='Debug mode: override learning rate for overfit runs.',
    )
    args.add_argument(
        '--overfit_batch_size',
        type=int,
        default=None,
        help='Debug mode: override batch size before selecting overfit batches.',
    )
    args.add_argument(
        '--curriculum_epochs',
        type=int,
        default=0, # for length curriculum start with 5
        help='Use length curriculum for this many initial epochs. 0 disables curriculum.',
    )
    args.add_argument(
        '--curriculum_start_fraction',
        type=float,
        default=0.2,
        help='Fraction of shortest training examples to use in the first curriculum epoch.',
    )
    args.add_argument(
        '--length_options',
        type=str,
        default="thirds",
        choices=["quarters", "thirds", "halves"],
        help='Split training data by sequence length into this many bins. If not set, train on all lengths.',
    )
    args.add_argument(
        '--length_bin',
        type=int,
        default=2,
        help='1-indexed length bin to train on. Requires --length_options. For --length_options thirds, use 1, 2, or 3.',
    )
    args.add_argument(
        '--cumulative',
        type=_str_to_bool,
        nargs='?',
        const=True,
        default=True,
        help='If set, train on all sequences up to the specified length. Otherwise, train on sequences only in the specified length range.',
    )
    args.add_argument(
        '--sweep',
        type=_str_to_bool,
        nargs='?',
        const=True,
        default=False,
        help='Run the autoencoder latent_dim/teacher_forcing_dropout_rate sweep.',
    )
    args.add_argument(
        '--max_length',
        type=int,
        default=None,
        help='If set, only sequences with length <= max_length will be used. Ignored if --length_options is set.',
    )
    args.add_argument(
        '--use_decoder_positional_embeddings',
        type=_str_to_bool,
        nargs='?',
        const=True,
        default=False,
        help='Add learned absolute position embeddings to decoder token embeddings.',
    )
    args.add_argument(
        '--max_decoder_positions',
        type=int,
        default=1024,
        help='Maximum decoder sequence length supported by learned positional embeddings.',
    )
    
    return args.parse_args()


def add_and_validate_train_inputs():
    args = _add_args(argparse.ArgumentParser())
    
    if args.task != "localization" and args.task != "solubility":
        raise ValueError("Task only accepts 'localization' or 'solubility'")
    
    if args.model.upper() == "AE":
        hyperparams = AEParams()
    else:
        raise ValueError("Only --model AE is currently supported")

    if args.version is None:
        raise ValueError("--version is required so autoencoder checkpoints and histories are saved under versioned artifact directories")
    if args.version < 0:
        raise ValueError("--version must be non-negative")
    
    if args.sweep and args.load_path is not None:
        raise ValueError("--load_path is not supported with --sweep because swept latent dimensions may not match the checkpoint.")

    if args.length_options is None and args.length_bin is not None:
        raise ValueError("--length_bin requires --length_options")
    if args.length_options is not None:
        if args.length_bin is None:
            raise ValueError("--length_bin is required when --length_options is set")
        split_count = LENGTH_SPLIT_COUNTS[args.length_options]
        if not 1 <= args.length_bin <= split_count:
            raise ValueError(f"--length_bin must be between 1 and {split_count} for --length_options {args.length_options}")
    if args.max_length is not None and args.max_length <= 0:
        raise ValueError("--max_length must be positive")
    if args.max_decoder_positions <= 0:
        raise ValueError("--max_decoder_positions must be positive")

    hyperparams.use_decoder_positional_embeddings = args.use_decoder_positional_embeddings
    hyperparams.max_decoder_positions = args.max_decoder_positions
    
    if not args.sweep:
        # Make sure this checkpoint filename is unique to avoid overwriting previous runs.
        checkpoint_path, history_path = autoencoder_artifact_paths(
            args.model,
            args.task,
            args.version,
            args.length_options,
            length_bin=args.length_bin,
            is_overfit=(args.overfit_batches is not None),
        )

        if checkpoint_path.exists():
            raise ValueError(
                f"Checkpoint already exists for this run: {checkpoint_path}"
                )

        if history_path.exists():
            raise ValueError(
                f"Training history already exists for this run: {history_path}"
                )

    if args.curriculum_epochs < 0:
        raise ValueError("--curriculum_epochs must be non-negative")
    if not 0.0 < args.curriculum_start_fraction <= 1.0:
        raise ValueError("--curriculum_start_fraction must be in the range (0, 1]")
    if args.curriculum_start_fraction is not None and args.curriculum_epochs == 0:
        warnings.warn("--curriculum_fraction has no effect when --curriculum_epochs is 0")

    if args.overfit_batches is not None:
        hyperparams.dropout = 0.0
        if args.overfit_learning_rate is not None:
            if args.overfit_learning_rate <= 0:
                raise ValueError("--overfit_learning_rate must be positive")
            hyperparams.learning_rate = args.overfit_learning_rate
        if args.overfit_batch_size is not None:
            if args.overfit_batch_size <= 0:
                raise ValueError("--overfit_batch_size must be a positive integer")
            hyperparams.batch_size = args.overfit_batch_size

    if args.overfit_epochs is not None:
        if args.overfit_epochs <= 0:
            raise ValueError("--overfit_epochs must be a positive integer")
        hyperparams.num_epochs = args.overfit_epochs
    
    return args, hyperparams
