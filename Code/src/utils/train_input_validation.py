from __future__ import annotations

import argparse
import sys
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


def _cli_arg_provided(flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:])


def _validate_autoencoder_hyperparams(hyperparams: AEParams) -> None:
    if hyperparams.layer_type not in {"gru", "transformer"}:
        raise ValueError("layer_type must be 'gru' or 'transformer'")
    if hyperparams.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if hyperparams.num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if hyperparams.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 <= hyperparams.dropout < 1.0:
        raise ValueError("dropout must be in the range [0, 1)")
    if hyperparams.patience < 0:
        raise ValueError("patience must be non-negative")
    if hyperparams.lr_patience < 0:
        raise ValueError("lr_patience must be non-negative")
    if hyperparams.embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    if hyperparams.cnn_out_channels <= 0:
        raise ValueError("cnn_out_channels must be positive")
    if hyperparams.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if hyperparams.latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    if hyperparams.kernel_size <= 0:
        raise ValueError("kernel_size must be positive")
    if hyperparams.num_layers < 1:
        raise ValueError("num_layers must be at least 1")
    if not 0.0 <= hyperparams.teacher_forcing_dropout_rate <= 1.0:
        raise ValueError("teacher_forcing_dropout_rate must be in the range [0, 1]")
    if hyperparams.max_decoder_positions <= 0:
        raise ValueError("max_decoder_positions must be positive")
    if hyperparams.max_encoder_positions <= 0:
        raise ValueError("max_encoder_positions must be positive")
    if hyperparams.num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if hyperparams.dim_feedforward <= 0:
        raise ValueError("dim_feedforward must be positive")
    if not 0.0 < hyperparams.scheduler_factor < 1.0:
        raise ValueError("scheduler_factor must be in the range (0, 1)")

    uses_cnn_stem = (
        hyperparams.layer_type == "gru"
        or (
            hyperparams.layer_type == "transformer"
            and hyperparams.use_cnn_before_transformer
        )
    )
    if uses_cnn_stem and hyperparams.kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd when a CNN stem is used")

    if (
        hyperparams.layer_type == "transformer"
        and hyperparams.embedding_dim % hyperparams.num_heads != 0
    ):
        raise ValueError("embedding_dim must be divisible by num_heads for transformer layers")


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


def architecture_artifact_suffix(layer_type: str, artifact_suffix: str | None = None) -> str | None:
    """Include non-default architectures in artifact names."""
    suffix_parts = []
    if layer_type != "gru":
        suffix_parts.append(layer_type)
    if artifact_suffix:
        suffix_parts.append(artifact_suffix)
    return "_".join(suffix_parts) if suffix_parts else None


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


def validate_artifact_paths(paths: list[tuple[Path, Path]]) -> None:
    existing_paths = [
        path
        for checkpoint_path, history_path in paths
        for path in (checkpoint_path, history_path)
        if path.exists()
    ]
    if existing_paths:
        formatted_paths = "\n".join(str(path) for path in existing_paths)
        raise ValueError(f"Training artifacts already exist:\n{formatted_paths}")


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
    args.add_argument(
        '--layer_type',
        type=str,
        choices=["gru", "transformer"],
        default=None,
        help="Autoencoder architecture to train.",
    )
    args.add_argument(
        '--max_encoder_positions',
        type=int,
        default=1024,
        help='Maximum encoder sequence length supported by learned positional embeddings.',
    )
    parsed_args = args.parse_args()
    parsed_args.curriculum_start_fraction_provided = _cli_arg_provided("--curriculum_start_fraction")
    return parsed_args


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
    if args.load_path is not None and not Path(args.load_path).exists():
        raise ValueError(f"--load_path does not exist: {args.load_path}")

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
    if args.max_length is not None and args.length_options is not None:
        warnings.warn("--max_length is ignored when --length_options is set")
    if args.max_decoder_positions <= 0:
        raise ValueError("--max_decoder_positions must be positive")
    if args.max_encoder_positions <= 0:
        raise ValueError("--max_encoder_positions must be positive")

    if args.layer_type is not None:
        hyperparams.layer_type = args.layer_type

    hyperparams.use_decoder_positional_embeddings = args.use_decoder_positional_embeddings
    hyperparams.max_decoder_positions = args.max_decoder_positions
    hyperparams.max_encoder_positions = args.max_encoder_positions
    _validate_autoencoder_hyperparams(hyperparams)
    
    if not args.sweep:
        # Make sure this checkpoint filename is unique to avoid overwriting previous runs.
        artifact_suffix = hyperparams.layer_type if hyperparams.layer_type != "gru" else None
        checkpoint_path, history_path = autoencoder_artifact_paths(
            args.model,
            args.task,
            args.version,
            args.length_options,
            length_bin=args.length_bin,
            is_overfit=(args.overfit_batches is not None),
            artifact_suffix=artifact_suffix,
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
    if args.curriculum_start_fraction_provided and args.curriculum_epochs == 0:
        warnings.warn("--curriculum_start_fraction has no effect when --curriculum_epochs is 0")

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

    _validate_autoencoder_hyperparams(hyperparams)
    
    return args, hyperparams
