"""Build the publication-facing autoencoder results table.

Run from any directory with:

    python3 Code/scripts/build_ae_final_results.py

The selected models are intentionally declared below rather than inferred from all
experiment runs. This keeps hyperparameter-sweep results out of the final table.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "Code" / "results" / "autoencoder" / "solubility"
AUTOENCODER_RESULTS_PATH = PROJECT_ROOT / "Code" / "results" / "tables" / "autoencoder_results.csv"
OUTPUT_PATH = PROJECT_ROOT / "Code" / "results" / "tables" / "ae_final_results.csv"


@dataclass(frozen=True)
class SelectedModel:
    display_name: str
    version: str
    distinguishing_feature: str
    training_data: str
    history_path: Path
    checkpoint_path: str


SELECTED_MODELS = (
    SelectedModel(
        display_name="Baseline 512 Latent GRU",
        version="v5",
        distinguishing_feature="Initial full-dataset baseline; length curriculum",
        training_data="Full dataset",
        history_path=RESULTS_ROOT / "v5" / "solubility_ae_history.json",
        checkpoint_path="checkpoints/autoencoder/solubility/v5/model_ae_solubility.pt",
    ),
    SelectedModel(
        display_name="Compact 256 Latent GRU",
        version="v12",
        distinguishing_feature="Lower-cost 2-layer, 512-hidden model",
        training_data="Cumulative length bins 1-2 of 3",
        history_path=RESULTS_ROOT
        / "v12"
        / "v12_model_ae_length_2_of_3_solubility_lr0p0003_num_layers2_hidden_dim512_history.json",
        checkpoint_path="checkpoints/autoencoder/solubility/v12/model_ae_length_2_of_3_solubility_lr0p0003_num_layers2_hidden_dim512.pt",
    ),
    SelectedModel(
        display_name="High-capacity 256 Latent GRU",
        version="v12",
        distinguishing_feature="3-layer, 1024-hidden model",
        training_data="Cumulative length bins 1-2 of 3",
        history_path=RESULTS_ROOT
        / "v12"
        / "v12_model_ae_length_2_of_3_solubility_lr0p0001_num_layers3_hidden_dim1024_history.json",
        checkpoint_path="checkpoints/autoencoder/solubility/v12/model_ae_length_2_of_3_solubility_lr0p0001_num_layers3_hidden_dim1024.pt",
    ),
    SelectedModel(
        display_name="Full-data 256 Latent GRU",
        version="v23",
        distinguishing_feature="Final model trained with all three length bins",
        training_data="Full dataset (length bins 1-3 of 3)",
        history_path=RESULTS_ROOT / "v23" / "v23_model_ae_length_3_of_3_solubility_history.json",
        checkpoint_path="checkpoints/autoencoder/solubility/v23/model_ae_length_3_of_3_solubility.pt",
    ),
    SelectedModel(
        display_name="Full-data 256 Latent LSTM",
        version="v26",
        distinguishing_feature="LSTM comparison with the full-data 256-latent GRU",
        training_data="Full dataset (length bins 1-3 of 3)",
        history_path=RESULTS_ROOT
        / "v26"
        / "v26_model_ae_length_3_of_3_solubility_lstm_history.json",
        checkpoint_path=(
            "checkpoints/autoencoder/solubility/v26/"
            "model_ae_length_3_of_3_solubility_lstm.pt"
        ),
    ),
    SelectedModel(
        display_name="Final 512 Latent AE",
        version="v27",
        distinguishing_feature="Final 512-dimensional latent model",
        training_data="Full dataset (length bins 1-3 of 3)",
        history_path=RESULTS_ROOT
        / "v27"
        / "v27_model_ae_length_3_of_3_solubility_history.json",
        checkpoint_path=(
            "checkpoints/autoencoder/solubility/v27/"
            "model_ae_length_3_of_3_solubility.pt"
        ),
    ),
)


FIELDNAMES = (
    "model",
    "version",
    "distinguishing_feature",
    "training_data",
    "learning_rate",
    "hidden_dim",
    "latent_dim",
    "num_layers",
    "teacher_forced_test_loss",
    "teacher_forced_test_accuracy",
)


def load_history(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Autoencoder history not found: {path}")
    with path.open(encoding="utf-8") as history_file:
        return json.load(history_file)


def load_teacher_forced_metrics(checkpoint_path: str) -> dict[str, float]:
    """Read teacher-forced test metrics from the aggregate results table."""
    if not AUTOENCODER_RESULTS_PATH.is_file():
        raise FileNotFoundError(f"Autoencoder results not found: {AUTOENCODER_RESULTS_PATH}")

    with AUTOENCODER_RESULTS_PATH.open(newline="", encoding="utf-8") as results_file:
        for row in csv.DictReader(results_file):
            if row.get("file name") == checkpoint_path:
                return {
                    "loss": float(row["test teacher force loss"]),
                    "accuracy": float(row["test teacher forced accuracy"]),
                }

    raise ValueError(f"No test metrics found for checkpoint: {checkpoint_path}")


def build_row(model: SelectedModel) -> dict[str, str | int | float]:
    history = load_history(model.history_path)
    hyperparameters = history.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        raise ValueError(f"Hyperparameters missing from {model.history_path}")
    metrics = load_teacher_forced_metrics(model.checkpoint_path)

    return {
        "model": model.display_name,
        "version": model.version,
        "distinguishing_feature": model.distinguishing_feature,
        "training_data": model.training_data,
        "learning_rate": hyperparameters["learning_rate"],
        "hidden_dim": hyperparameters["hidden_dim"],
        "latent_dim": hyperparameters["latent_dim"],
        "num_layers": hyperparameters["num_layers"],
        "teacher_forced_test_loss": f"{metrics['loss']:.6f}",
        "teacher_forced_test_accuracy": f"{metrics['accuracy']:.6f}",
    }


def main() -> None:
    rows = [build_row(model) for model in SELECTED_MODELS]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} selected autoencoder results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
