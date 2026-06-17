import argparse
import warnings
from pathlib import Path
from utils.hyperparameters import (AutoencoderHyperparameters as AEParams)


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
    
    return args.parse_args()

def add_and_validate_train_inputs():
    args = _add_args(argparse.ArgumentParser())
    
    if args.task != "localization" and args.task != "solubility":
        raise ValueError("Task only accepts 'localization' or 'solubility'")
    
    if args.model.upper() == "AE":
        hyperparams = AEParams()
    else:
        raise ValueError("Only --model AE is currently supported")
    
    # Make sure the version is unique for the task to avoid overwriting previous checkpoints and history
    model_dir = "autoencoder" if args.model.upper() == "AE" else args.model.lower()
    version_dir = Path("checkpoints") / model_dir / args.task / f"v{args.version}"

    if version_dir.exists():
        raise ValueError(
            f"Version v{args.version} has already been run for {args.task}: "
            f"{version_dir}"
            )
        
    history_dir = Path("Code/results") / model_dir / args.task / f"v{args.version}"

    # if version_dir.exists() or history_dir.exists():
    if history_dir.exists():
        raise ValueError(
            f"Version v{args.version} has already been run for {args.task}. "
            f"Found existing output at {version_dir if version_dir.exists() else history_dir}"
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