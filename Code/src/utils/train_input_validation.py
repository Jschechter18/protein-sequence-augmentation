import argparse
import warnings
from pathlib import Path
from typing import Optional
from utils.hyperparameters import AutoencoderHyperparameters as AEParams

LENGTH_QUARTILE_FILE_LABELS = {
    "s": "short",
    "ms": "medium_short",
    "ml": "medium_long",
    "l": "long",
}


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
    length_quartile: Optional[str] = None,
    is_overfit: bool = False,
    artifact_suffix: str | None = None,
) -> str:
    parts = ["model", model.lower()]
    if length_quartile is not None:
        parts.append(LENGTH_QUARTILE_FILE_LABELS[length_quartile])
    parts.append(task)
    if is_overfit:
        parts.append("overfit")
    if artifact_suffix:
        parts.append(artifact_suffix)
    return "_".join(parts)


def _add_args(args: argparse.ArgumentParser) -> argparse.Namespace:
    args.add_argument('--model', type=str, default='AE', help='Model to train (default: AE)')
    args.add_argument('--task', type=str, default='localization', help='Task to perform (default: localization)')
    args.add_argument('--load_path', type=str, nargs='?', default=None, help='Path to load the best existing model checkpoint (optional)')
    args.add_argument('--version', type=int, default=0, help='Version identifier for this training run (used for checkpoint and history naming)') # should always be unique
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
        '--length_quartile',
        type=str,
        default=None,
        choices=["s", "ms", "ml", "l"],
    )
    args.add_argument(
        '--cumulative_quartiles',
        type=_str_to_bool,
        nargs='?',
        const=True,
        default=False,
        help='If set, each length quartile includes all shorter quartiles (e.g., "ml" includes "s", "ms", and "ml"). Ignored if --length_quartile is not set.',
    )
    args.add_argument(
        '--max_length',
        type=int,
        default=None,
        help='If set, only sequences with length <= max_length will be used for training and validation. Ignored if --length_quartile is set.',
    )
    # args.add_argument(
    #     '--tuning',
    #     type=bool,
    #     default=False,
    # )
    args.add_argument(
        '--sweep',
        type=_str_to_bool,
        nargs='?',
        const=True,
        default=False,
        help='Run the autoencoder latent_dim/teacher_forcing_dropout_rate sweep.',
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
    
    if args.sweep and args.load_path is not None:
        raise ValueError("--load_path is not supported with --sweep because swept latent dimensions may not match the checkpoint.")

    if not args.sweep:
        # Make sure this checkpoint filename is unique to avoid overwriting previous runs.
        model_dir = "autoencoder" if args.model.upper() == "AE" else args.model.lower()
        version_dir = Path("checkpoints") / model_dir / args.task / f"v{args.version}"
        artifact_stem = autoencoder_artifact_stem(
            args.model,
            args.task,
            args.length_quartile,
            is_overfit=(args.overfit_batches is not None),
        )
        checkpoint_path = version_dir / f"{artifact_stem}.pt"

        if checkpoint_path.exists():
            raise ValueError(
                f"Checkpoint already exists for this run: {checkpoint_path}"
                )

        history_path = Path("history") / f"v{args.version}_{artifact_stem}_history.json"

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
