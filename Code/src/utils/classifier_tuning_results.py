"""Shared validation-metric selection logic for classifier tuning results."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import pandas as pd


def representation_learning_rate(
    representation: str,
    *,
    encoder_learning_rate: Any,
    esm_learning_rate: Any,
) -> float:
    """Return the learning rate used by a representation's trainable encoder."""

    encoder_rate = float(encoder_learning_rate)
    esm_rate = float(esm_learning_rate)
    if representation in {"random_autoencoder", "trained_autoencoder"}:
        rate = encoder_rate
    elif representation == "esm2":
        rate = esm_rate
    elif representation == "trained_autoencoder+esm2":
        rate = encoder_rate
    else:
        raise ValueError(f"Unsupported representation: {representation!r}")
    if rate <= 0:
        raise ValueError("Representation learning rate must be positive.")
    return rate


def tuning_metrics_from_history(
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the validation metrics used to rank one tuning trial."""

    if not history:
        raise ValueError("A completed tuning run must contain training history.")
    selection_record = max(
        history,
        key=lambda record: (
            float(record.get("val_f1", float("-inf"))),
            -float(record.get("val_loss", float("inf"))),
        ),
    )
    return {
        "selection_epoch": int(selection_record["epoch"]),
        "best_val_f1": float(selection_record["val_f1"]),
        "val_loss_at_selection": float(selection_record["val_loss"]),
        "val_accuracy_at_selection": float(selection_record["val_accuracy"]),
    }


def select_tuning_hyperparameters(
    summary: pd.DataFrame,
    *,
    requested_conditions: Iterable[tuple[str, str]],
    is_end_to_end_tuning: bool,
    default_selected: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Select one completed validation winner per head/representation pair."""

    selected: dict[str, dict[str, dict[str, Any]]]
    if is_end_to_end_tuning:
        selected = {}
    else:
        selected = {
            head_type: {
                representation: dict(parameters)
                for representation, parameters in representations.items()
            }
            for head_type, representations in (default_selected or {}).items()
        }

    requested = set(requested_conditions)
    for head_type, representation in requested:
        selected.get(head_type, {}).pop(representation, None)

    if summary.empty or "status" not in summary:
        return selected
    completed = summary[summary["status"] == "complete"].copy()
    if completed.empty or "best_val_f1" not in completed:
        return selected

    rank_columns = [
        "head_type",
        "representation",
        "best_val_f1",
        "val_loss_at_selection",
        "head_learning_rate" if is_end_to_end_tuning else "learning_rate",
        *(["representation_learning_rate"] if is_end_to_end_tuning else []),
        "weight_decay",
        "dropout",
        "autoencoder_embedding_dropout",
        "esm_embedding_dropout",
    ]
    ranked = completed.sort_values(
        rank_columns,
        ascending=[True, True, False, True] + [True] * (len(rank_columns) - 4),
        na_position="first",
        kind="stable",
    )
    winners = ranked.drop_duplicates(
        ["head_type", "representation"], keep="first"
    )
    for row in winners.to_dict(orient="records"):
        parameters: dict[str, Any] = {
            "weight_decay": float(row["weight_decay"]),
            "selection_seed": int(row["seed"]),
            "selection_epoch": int(row["selection_epoch"]),
            "best_val_f1": float(row["best_val_f1"]),
            "val_loss_at_selection": float(row["val_loss_at_selection"]),
        }
        if row["head_type"] == "mlp":
            parameters["dropout"] = float(row["dropout"])
        if is_end_to_end_tuning:
            parameters["head_learning_rate"] = float(row["head_learning_rate"])
            parameters["representation_learning_rate"] = float(
                row["representation_learning_rate"]
            )
            parameters["autoencoder_embedding_dropout"] = float(
                row["autoencoder_embedding_dropout"]
            )
            parameters["esm_embedding_dropout"] = float(
                row["esm_embedding_dropout"]
            )
        else:
            parameters["learning_rate"] = float(row["learning_rate"])
        selected.setdefault(str(row["head_type"]), {})[
            str(row["representation"])
        ] = parameters
    return selected
