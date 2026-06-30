import argparse
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
    # args.add_argument(
    #     "--checkpoint",
    #     type=str,
    #     default="Code/results/autoencoder/solubility/v5/solubility_ae_history.json",
    #     help="Path to the checkpoint to test. Defaults to checkpoints/<version>/model_<model>_<task>.pt when --version is set.",
    # )
    args.add_argument("--version", type=str, default="v5", help="Checkpoint version directory to test.")
    args.add_argument(
        "--teacher_forcing",
        type=_str_to_bool,
        default=True,
        help="Use teacher forcing during reconstruction evaluation.",
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

    return args.parse_args()

def add_and_validate_test_inputs():
    # pass
    args = _add_args(argparse.ArgumentParser(description="Test an autoencoder checkpoint on protein sequences."))
    
    if args.model.upper() != "AE":
        raise ValueError("Only --model AE is currently supported")
    
    return args