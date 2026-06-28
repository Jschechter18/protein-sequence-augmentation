"""
Post-process ESM-2 experiment results into standardized history.json files.

Expected old run structure:
    Code/results/esm2/<run_name>/
        config.json
        metrics.json
        training_history.csv

Optional new structure:
    Code/results/esm2/solubility/stage0_frozen/<run_name>/
    Code/results/esm2/solubility/stage1_unfreeze_last1/<run_name>/

Usage:
    python -m Code.scripts.post_processing

or:
    python Code/scripts/post_processing.py
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file as a dictionary."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    """Save a dictionary as a formatted JSON file."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def infer_stage(config: Dict[str, Any]) -> str:
    """
    Infer experiment stage from config values.

    Stage naming:
        stage0_frozen          = ESM-2 frozen
        stage1_unfreeze_last1 = last 1 ESM-2 layer unfrozen
        stage2_unfreeze_last2 = last 2 ESM-2 layers unfrozen
    """
    unfreeze_esm = config.get("unfreeze_esm", False)
    unfreeze_layers = int(config.get("unfreeze_layers", 0) or 0)

    if not unfreeze_esm or unfreeze_layers == 0:
        return "stage0_frozen"

    return f"stage{unfreeze_layers}_unfreeze_last{unfreeze_layers}"


def build_epoch_records(history_df: pd.DataFrame) -> list[Dict[str, Any]]:
    """
    Convert training_history.csv rows into a list of epoch-level dictionaries.

    Handles missing columns gracefully so older runs can still be migrated.
    """
    records = []

    for _, row in history_df.iterrows():
        epoch_record = {
            "epoch": int(row["epoch"]),
            "train_loss": float(row["train_loss"]),
            "train_accuracy": float(row["train_accuracy"]),
            "train_f1": float(row["train_f1"]) if "train_f1" in history_df.columns else None,
            "train_precision": float(row["train_precision"]) if "train_precision" in history_df.columns else None,
            "train_recall": float(row["train_recall"]) if "train_recall" in history_df.columns else None,
            "val_loss": float(row["val_loss"]),
            "val_accuracy": float(row["val_accuracy"]),
            "val_f1": float(row["val_f1"]) if "val_f1" in history_df.columns else None,
            "val_precision": float(row["val_precision"]) if "val_precision" in history_df.columns else None,
            "val_recall": float(row["val_recall"]) if "val_recall" in history_df.columns else None,
        }

        records.append(epoch_record)

    return records


def build_summary(metrics: Dict[str, Any], history_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build experiment-level summary metrics.

    If metrics.json already contains the value, use it.
    Otherwise, compute from training_history.csv when possible.
    """
    best_val_idx = history_df["val_loss"].idxmin()

    summary = {
        "best_epoch": int(history_df.loc[best_val_idx, "epoch"]), # type: ignore
        "best_val_loss": float(metrics.get("best_val_loss", history_df["val_loss"].min())),
        "best_val_accuracy": float(metrics.get("best_val_accuracy", history_df["val_accuracy"].max())),
        "best_val_f1": float(metrics.get("best_val_f1", history_df["val_f1"].max()))
        if "val_f1" in history_df.columns
        else metrics.get("best_val_f1"),
        "best_val_precision": float(metrics.get("best_val_precision", history_df["val_precision"].max()))
        if "val_precision" in history_df.columns
        else metrics.get("best_val_precision"),
        "best_val_recall": float(metrics.get("best_val_recall", history_df["val_recall"].max()))
        if "val_recall" in history_df.columns
        else metrics.get("best_val_recall"),

        "final_train_loss": float(metrics.get("final_train_loss", history_df["train_loss"].iloc[-1])),
        "final_train_accuracy": float(metrics.get("final_train_accuracy", history_df["train_accuracy"].iloc[-1])),
        "final_train_f1": float(metrics.get("final_train_f1", history_df["train_f1"].iloc[-1]))
        if "train_f1" in history_df.columns
        else metrics.get("final_train_f1"),
        "final_train_precision": float(metrics.get("final_train_precision", history_df["train_precision"].iloc[-1]))
        if "train_precision" in history_df.columns
        else metrics.get("final_train_precision"),
        "final_train_recall": float(metrics.get("final_train_recall", history_df["train_recall"].iloc[-1]))
        if "train_recall" in history_df.columns
        else metrics.get("final_train_recall"),

        "final_val_loss": float(metrics.get("final_val_loss", history_df["val_loss"].iloc[-1])),
        "final_val_accuracy": float(metrics.get("final_val_accuracy", history_df["val_accuracy"].iloc[-1])),
        "final_val_f1": float(metrics.get("final_val_f1", history_df["val_f1"].iloc[-1]))
        if "val_f1" in history_df.columns
        else metrics.get("final_val_f1"),
        "final_val_precision": float(metrics.get("final_val_precision", history_df["val_precision"].iloc[-1]))
        if "val_precision" in history_df.columns
        else metrics.get("final_val_precision"),
        "final_val_recall": float(metrics.get("final_val_recall", history_df["val_recall"].iloc[-1]))
        if "val_recall" in history_df.columns
        else metrics.get("final_val_recall"),

        "epochs_completed": int(metrics.get("epochs_completed", len(history_df))),
    }

    return summary


def build_history_json(
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    history_df: pd.DataFrame,
    run_name: str,
) -> Dict[str, Any]:
    """Build the new standardized history.json structure."""

    dataset = config.get("dataset", "unknown")
    unfreeze_esm = config.get("unfreeze_esm", False)
    unfreeze_layers = int(config.get("unfreeze_layers", 0) or 0)

    history_json = {
        "experiment": {
            "run_name": run_name,
            "dataset": dataset,
            "stage": infer_stage(config),
            "model_family": "ESM-2 + 1D-CNN",
        },

        "hyperparameters": {
            "batch_size": config.get("batch_size"),
            "num_epochs": config.get("epochs"),
            "learning_rate": config.get("learning_rate"),
            "esm_learning_rate": config.get("esm_learning_rate"),
            "early_stopping_patience": config.get("early_stopping_patience"),
            "seed": config.get("seed"),
            "num_classes": config.get("num_classes"),
            "data_dir": config.get("data_dir"),
            "results_dir": config.get("results_dir"),
            "checkpoint_dir": config.get("checkpoint_dir"),
        },

        "model": {
            "esm_model": config.get("esm_model_name", "esm2_t6_8M_UR50D"),
            "embedding_dim": 320,
            "classifier": "1D-CNN",
            "cnn_kernel_sizes": [3, 5, 7],
            "cnn_num_filters": 64,
            "cnn_dropout": 0.3,
            "unfreeze_esm": unfreeze_esm,
            "unfreeze_layers": unfreeze_layers,
            "fine_tuning_strategy": (
                f"last_{unfreeze_layers}_layers"
                if unfreeze_esm
                else "frozen_backbone"
            ),
        },

        "summary": build_summary(metrics, history_df),

        "epochs": build_epoch_records(history_df),

        "train_loss": history_df["train_loss"].tolist(),

        "train_scores": {
            "accuracy": history_df["train_accuracy"].tolist(),
            "f1": history_df["train_f1"].tolist() if "train_f1" in history_df.columns else [],
            "precision": history_df["train_precision"].tolist() if "train_precision" in history_df.columns else [],
            "recall": history_df["train_recall"].tolist() if "train_recall" in history_df.columns else [],
        },

        "val_loss": history_df["val_loss"].tolist(),

        "val_scores": {
            "accuracy": history_df["val_accuracy"].tolist(),
            "f1": history_df["val_f1"].tolist() if "val_f1" in history_df.columns else [],
            "precision": history_df["val_precision"].tolist() if "val_precision" in history_df.columns else [],
            "recall": history_df["val_recall"].tolist() if "val_recall" in history_df.columns else [],
        },
    }

    return history_json


def process_run(run_dir: Path, output_dir: Optional[Path] = None, move: bool = False) -> None:
    """
    Convert one run directory into the new history.json format.

    If output_dir is provided:
        - copy or move the run into output_dir before writing history.json.

    If output_dir is None:
        - write history.json directly inside the existing run folder.
    """
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    history_path = run_dir / "training_history.csv"

    if not config_path.exists() or not history_path.exists():
        print(f"Skipping {run_dir.name}: missing config.json or training_history.csv")
        return

    config = load_json(config_path)

    if metrics_path.exists():
        metrics = load_json(metrics_path)
    else:
        metrics = {}

    history_df = pd.read_csv(history_path)

    stage = infer_stage(config)
    dataset = config.get("dataset", "unknown")

    target_run_dir = run_dir

    if output_dir is not None:
        target_parent = output_dir / dataset / stage
        target_parent.mkdir(parents=True, exist_ok=True)

        target_run_dir = target_parent / run_dir.name

        if target_run_dir.exists():
            print(f"Target already exists, skipping copy/move: {target_run_dir}")
        else:
            if move:
                shutil.move(str(run_dir), str(target_run_dir))
            else:
                shutil.copytree(run_dir, target_run_dir)

    history_json = build_history_json(
        config=config,
        metrics=metrics,
        history_df=history_df,
        run_name=target_run_dir.name,
    )

    save_json(history_json, target_run_dir / "history.json")

    print(f"Created history.json for {target_run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert old ESM-2 run outputs into standardized history.json files."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="Code/results/esm2",
        help="Directory containing old ESM-2 run folders.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Optional new root directory for organized results. "
            "Example: Code/results/esm2. "
            "If provided, runs are copied/moved to output_dir/dataset/stage/run_name."
        ),
    )

    parser.add_argument(
        "--move",
        action="store_true",
        help="Move runs instead of copying them when output_dir is provided.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    run_dirs = [
    path
    for path in input_dir.rglob("*")
    if path.is_dir()
    and (path / "config.json").exists()
    and (path / "training_history.csv").exists()
]
    for run_dir in sorted(run_dirs):
        process_run(run_dir, output_dir=output_dir, move=args.move)


if __name__ == "__main__":
    main()

#python Code/scripts/post_processing.py --input_dir Code/results/esm2/solubility
    