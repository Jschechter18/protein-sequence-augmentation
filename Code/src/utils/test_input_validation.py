import argparse
import sys
import warnings
from pathlib import Path
from utils.hyperparameters import (AutoencoderHyperparameters as AEParams)

LENGTH_QUARTILE_FILE_LABELS = {
    "s": "short",
    "ms": "medium_short",
    "ml": "medium_long",
    "l": "long",
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



def _add_args(args: argparse.ArgumentParser) -> argparse.Namespace:
    args.add_argument("--model", type=str, default="AE", choices=["AE", "ae"], help="Model to test.")
    args.add_argument("--task", type=str, default="solubility", choices=["localization", "solubility"], help="Task to test.")
    args.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint to test. Defaults to checkpoints/autoencoder/<task>/<version>/...",
    )
    args.add_argument("--version", type=str, default="v5", help="Checkpoint version directory to test.")
    args.add_argument(
        "--teacher_forcing",
        type=_str_to_bool,
        default=True,
        help="Deprecated: both teacher-forced and autoregressive evaluations are always run.",
    )
    args.add_argument(
        "--output_path",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "output_results.csv"),
        help="CSV path for decoder prediction results.",
    )
    args.add_argument(
        '--length_quartile',
        type=str,
        default=None,
        choices=["s", "ms", "ml", "l"],
    )
    args.add_argument(
        '--length_options',
        type=str,
        default=None,
        choices=["quarters", "thirds", "halves"],
        help='Split test data by sequence length into this many bins. Matches training --length_options.',
    )
    args.add_argument(
        '--length_bin',
        type=int,
        default=None,
        help='1-indexed length bin to test. Requires --length_options.',
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
    args.add_argument(
        '--cumulative',
        type=_str_to_bool,
        nargs='?',
        const=True,
        default=False,
        help='If set with --length_options, test on the selected length bin and all shorter bins.',
    )
    args.add_argument(
        '--latent_dim',
        type=int,
        default=None,
        help='Latent dimension suffix for selecting a swept autoencoder checkpoint.',
    )
    args.add_argument(
        '--teacher_forcing_dropout_rate',
        type=float,
        default=None,
        help='Teacher forcing dropout suffix for selecting a swept autoencoder checkpoint.',
    )
    args.add_argument(
        '--scheduler_factor',
        type=float,
        default=None,
        help='Scheduler factor suffix for selecting swept checkpoints that include _sf<factor>.',
    )
    args.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='Learning rate suffix for selecting swept checkpoints that include _lr<learning_rate>.',
    )
    args.add_argument(
        '--lr_patience',
        type=int,
        default=None,
        help='LR scheduler patience suffix for selecting swept checkpoints that include _lrp<patience>.',
    )

    parsed_args = args.parse_args()
    parsed_args.version_provided = any(
        arg == "--version" or arg.startswith("--version=")
        for arg in sys.argv[1:]
    )
    return parsed_args

def add_and_validate_test_inputs():
    # pass
    args = _add_args(argparse.ArgumentParser(description="Test an autoencoder checkpoint on protein sequences."))
    
    if args.model.upper() != "AE":
        raise ValueError("Only --model AE is currently supported")

    has_latent_dim = args.latent_dim is not None
    has_teacher_forcing_dropout = args.teacher_forcing_dropout_rate is not None
    if has_latent_dim != has_teacher_forcing_dropout:
        raise ValueError(
            "--latent_dim and --teacher_forcing_dropout_rate must be provided together."
        )
    if args.scheduler_factor is not None and not has_latent_dim:
        raise ValueError("--scheduler_factor requires --latent_dim and --teacher_forcing_dropout_rate.")
    if args.learning_rate is not None and not has_latent_dim:
        raise ValueError("--learning_rate requires --latent_dim and --teacher_forcing_dropout_rate.")
    if args.lr_patience is not None and not has_latent_dim:
        raise ValueError("--lr_patience requires --latent_dim and --teacher_forcing_dropout_rate.")

    if args.length_quartile is not None and args.length_options is not None:
        raise ValueError("--length_quartile and --length_options cannot be used together")
    if args.length_options is None and args.length_bin is not None:
        raise ValueError("--length_bin requires --length_options")
    if args.length_options is not None:
        if args.length_bin is None:
            raise ValueError("--length_bin is required when --length_options is set")
        split_counts = {"halves": 2, "thirds": 3, "quarters": 4}
        split_count = split_counts[args.length_options]
        if not 1 <= args.length_bin <= split_count:
            raise ValueError(f"--length_bin must be between 1 and {split_count} for --length_options {args.length_options}")
    
    return args
