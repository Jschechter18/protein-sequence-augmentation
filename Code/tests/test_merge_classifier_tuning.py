"""Tests for merging classifier tuning artifacts from multiple machines."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import pandas as pd
import pytest

from Code.scripts.merge_classifier_tuning import (
    TuningMergeError,
    merge_tuning_results,
)


def _write_trial(
    source: Path,
    *,
    learning_rate: float,
    val_f1: float,
    val_loss: float = 0.5,
    status: str = "complete",
    data_sha256: str = "same-data",
    directory_name: str | None = None,
    representation: str = "esm2",
    head_type: str = "linear",
    weight_decay: float = 0.0,
    dropout: float | None = None,
    phase: str = "tuning",
    encoder_learning_rate: float = 1e-3,
    esm_learning_rate: float = 1e-5,
) -> Path:
    trial_name = directory_name or (
        f"trial_{head_type}_{representation}_{learning_rate:g}_"
        f"{weight_decay:g}_{dropout}"
    )
    run_dir = source / head_type / representation / trial_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": "solubility",
        "version": "9",
        "phase": phase,
        "mode": "end_to_end_hp_tune" if phase == "end_to_end_tuning" else "hp_tune",
        "representation": representation,
        "head_type": head_type,
        "seed": 42,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "autoencoder_embedding_dropout": 0.0,
        "esm_embedding_dropout": 0.0,
        "encoder_mode": "frozen",
        "freeze_autoencoder": True,
        "freeze_esm2": True,
        "checkpoint_dir": f"/remote/checkpoints/{trial_name}",
        "num_classes": 2,
        "batch_size": 16,
        "epochs": 30,
        "early_stopping_patience": 5,
        "deterministic": True,
        "esm_model_name": "esm2_t6_8M_UR50D",
        "esm_max_sequence_length": 1022,
        "encoder_learning_rate": encoder_learning_rate,
        "esm_learning_rate": esm_learning_rate,
        "max_grad_norm": 1.0,
        "git_commit": "same-commit",
        "data_sources": {
            split: {"sha256": data_sha256}
            for split in ("train", "valid", "test")
        },
        "source_file_sha256": {
            "Code/src/training/train_classifier.py": "same-code"
        },
        "preprocessing": {
            "classification_encoding": "char",
            "autoencoder_special_tokens": "BOS+residues+EOS",
            "esm_long_sequence_policy": "truncate_right",
            "esm_max_sequence_length": 1022,
            "embedding_cache": {"schema_version": 1},
        },
        "runtime": {
            "packages": {"torch": "2.0"},
            "torch_cuda_version": "12.6",
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps({"status": status, "error": "failed" if status == "failed" else None}),
        encoding="utf-8",
    )
    if status == "complete":
        pd.DataFrame(
            [
                {"epoch": 1, "val_f1": val_f1 - 0.1, "val_loss": 0.6, "val_accuracy": 0.7},
                {"epoch": 2, "val_f1": val_f1, "val_loss": val_loss, "val_accuracy": 0.8},
            ]
        ).to_csv(run_dir / "history.csv", index=False)
        (run_dir / "history.json").write_text("{}\n", encoding="utf-8")
    return run_dir


def test_merge_rebuilds_global_results_and_is_idempotent(tmp_path: Path) -> None:
    sources = [tmp_path / f"ec2_{index}" for index in range(3)]
    for source, learning_rate, val_f1 in zip(
        sources,
        (1e-4, 1e-5, 1e-6),
        (0.70, 0.85, 0.75),
        strict=True,
    ):
        _write_trial(source, learning_rate=learning_rate, val_f1=val_f1)
        (source / "selected_hyperparameters.json").write_text(
            json.dumps({"wrong": "local winner"}), encoding="utf-8"
        )
        (source / "tuning_results.csv").write_text(
            "wrong,local\n1,2\n", encoding="utf-8"
        )

    output = tmp_path / "merged" / "tuning"
    manifest = merge_tuning_results(
        sources, output, expected_trials=3
    )

    results = pd.read_csv(output / "tuning_results.csv")
    selected = json.loads((output / "selected_hyperparameters.json").read_text())
    assert len(results) == 3
    assert results["learning_rate"].tolist() == pytest.approx([1e-6, 1e-5, 1e-4])
    assert selected["linear"]["esm2"]["learning_rate"] == pytest.approx(1e-5)
    assert selected["linear"]["esm2"]["best_val_f1"] == pytest.approx(0.85)
    assert manifest["num_copied_trials"] == 3
    assert (output / "linear" / "esm2" / "lr_1e-5_wd_0_seed_42").is_dir()
    assert all((source / "selected_hyperparameters.json").is_file() for source in sources)

    rerun = merge_tuning_results(sources, output, expected_trials=3)
    assert rerun["num_copied_trials"] == 0


def test_end_to_end_merge_preserves_head_and_representation_rates(
    tmp_path: Path,
) -> None:
    sources = [tmp_path / f"ec2_{index}" for index in range(3)]
    for source, representation_rate, val_f1 in zip(
        sources,
        (1e-4, 1e-5, 1e-6),
        (0.70, 0.85, 0.75),
        strict=True,
    ):
        _write_trial(
            source,
            learning_rate=1e-4,
            encoder_learning_rate=representation_rate,
            esm_learning_rate=representation_rate,
            phase="end_to_end_tuning",
            representation="trained_autoencoder+esm2",
            val_f1=val_f1,
        )

    output = tmp_path / "merged" / "tuning"
    merge_tuning_results(
        sources,
        output,
        phase="end_to_end_tuning",
        expected_trials=3,
        expected_head_learning_rate=1e-4,
        require_tied_representation_learning_rates=True,
    )

    results = pd.read_csv(output / "tuning_results.csv")
    assert sorted(results["representation_learning_rate"].tolist()) == pytest.approx(
        [1e-6, 1e-5, 1e-4]
    )
    selected = json.loads((output / "selected_hyperparameters.json").read_text())
    winner = selected["linear"]["trained_autoencoder+esm2"]
    assert winner["head_learning_rate"] == pytest.approx(1e-4)
    assert winner["representation_learning_rate"] == pytest.approx(1e-5)
    assert "learning_rate" not in winner
    assert (
        output
        / "linear"
        / "trained_autoencoder+esm2"
        / "lr_1e-4_wd_0_rep_lr_1e-5_ae_do_0_esm_do_0_seed_42"
    ).is_dir()


def test_merge_rejects_conflicting_duplicate_completed_trials(tmp_path: Path) -> None:
    sources = [tmp_path / "ec2_1", tmp_path / "ec2_2"]
    _write_trial(sources[0], learning_rate=1e-4, val_f1=0.7)
    _write_trial(sources[1], learning_rate=1e-4, val_f1=0.8)

    with pytest.raises(TuningMergeError, match="Conflicting completed attempts"):
        merge_tuning_results(sources, tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


def test_merge_ignores_archives_and_requires_complete_trials(tmp_path: Path) -> None:
    source = tmp_path / "ec2"
    _write_trial(source, learning_rate=1e-4, val_f1=0.7)
    _write_trial(
        source,
        learning_rate=1e-4,
        val_f1=0.9,
        status="failed",
        directory_name="trial.backup_20260801_120000",
    )
    _write_trial(source, learning_rate=1e-5, val_f1=0.8, status="running")

    with pytest.raises(TuningMergeError, match="incomplete trials"):
        merge_tuning_results([source], tmp_path / "strict", expected_trials=2)

    manifest = merge_tuning_results(
        [source],
        tmp_path / "partial",
        expected_trials=2,
        require_complete=False,
    )
    assert manifest["num_attempts"] == 2
    assert manifest["status_counts"] == {"complete": 1, "running": 1}
    partial_results = pd.read_csv(tmp_path / "partial" / "tuning_results.csv")
    running_row = partial_results[partial_results["status"] == "running"].iloc[0]
    assert not bool(running_row["artifacts_copied"])
    assert pd.isna(running_row["run_dir"])
    selected = json.loads(
        (tmp_path / "partial" / "selected_hyperparameters.json").read_text()
    )
    assert selected["linear"]["esm2"]["learning_rate"] == pytest.approx(1e-4)

    _write_trial(source, learning_rate=1e-5, val_f1=0.8, status="complete")
    completed_manifest = merge_tuning_results(
        [source], tmp_path / "partial", expected_trials=2
    )
    assert completed_manifest["num_copied_trials"] == 1
    assert completed_manifest["status_counts"] == {"complete": 2}


def test_merge_rejects_provenance_mismatch_by_default(tmp_path: Path) -> None:
    sources = [tmp_path / "ec2_1", tmp_path / "ec2_2"]
    _write_trial(sources[0], learning_rate=1e-4, val_f1=0.7)
    _write_trial(
        sources[1],
        learning_rate=1e-5,
        val_f1=0.8,
        data_sha256="different-data",
    )

    with pytest.raises(TuningMergeError, match="incompatible provenance"):
        merge_tuning_results(sources, tmp_path / "merged")

    with pytest.raises(TuningMergeError, match="incompatible provenance"):
        merge_tuning_results(
            sources,
            tmp_path / "still_rejected",
            allow_partition_source_mismatch=True,
        )


def test_partition_source_override_is_narrow_and_audited(tmp_path: Path) -> None:
    sources = [tmp_path / "ec2_1", tmp_path / "ec2_2"]
    _write_trial(sources[0], learning_rate=1e-4, val_f1=0.7)
    second = _write_trial(sources[1], learning_rate=1e-5, val_f1=0.8)
    second_config_path = second / "config.json"
    second_config = json.loads(second_config_path.read_text())
    second_config["git_commit"] = "partition-commit"
    second_config["source_file_sha256"][
        "Code/src/training/train_classifier.py"
    ] = "partitioned-driver"
    second_config_path.write_text(json.dumps(second_config), encoding="utf-8")

    manifest = merge_tuning_results(
        sources,
        tmp_path / "merged",
        allow_partition_source_mismatch=True,
    )

    assert manifest["partition_source_audit"]["git_commits"] == [
        "partition-commit",
        "same-commit",
    ]
    assert manifest["partition_source_audit"]["train_classifier_sha256"] == [
        "partitioned-driver",
        "same-code",
    ]


def test_duplicate_choice_is_independent_of_input_order(tmp_path: Path) -> None:
    first = tmp_path / "a_source"
    second = tmp_path / "b_source"
    first_run = _write_trial(first, learning_rate=1e-4, val_f1=0.7)
    second_run = _write_trial(second, learning_rate=1e-4, val_f1=0.7)
    (first_run / "run.log").write_text("first\n", encoding="utf-8")
    (second_run / "run.log").write_text("second\n", encoding="utf-8")
    output = tmp_path / "merged"

    merge_tuning_results([second, first], output)
    rerun = merge_tuning_results([first, second], output)

    assert rerun["num_copied_trials"] == 0
    copied_log = next(output.glob("linear/esm2/*/run.log"))
    assert copied_log.read_text() == "first\n"


def test_merge_rejects_symlinked_output_escape(tmp_path: Path) -> None:
    source = tmp_path / "ec2"
    _write_trial(source, learning_rate=1e-4, val_f1=0.7)
    output = tmp_path / "merged"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    (output / "linear").symlink_to(outside, target_is_directory=True)

    with pytest.raises(TuningMergeError, match="escapes"):
        merge_tuning_results([source], output)
    assert list(outside.iterdir()) == []


def test_full_grid_validation_checks_coordinates_not_only_count(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ec2"
    representations = (
        "random_autoencoder",
        "trained_autoencoder",
        "esm2",
        "trained_autoencoder+esm2",
    )
    learning_rates = (1e-4, 1e-5, 1e-6)
    for head_type, representation, learning_rate, weight_decay in product(
        ("linear", "mlp"),
        representations,
        learning_rates,
        (0.0, 1e-4),
    ):
        dropouts = (0.1, 0.3) if head_type == "mlp" else (None,)
        for dropout in dropouts:
            _write_trial(
                source,
                learning_rate=learning_rate,
                val_f1=0.7,
                representation=representation,
                head_type=head_type,
                weight_decay=weight_decay,
                dropout=dropout,
            )

    valid = merge_tuning_results(
        [source], tmp_path / "valid", expect_full_grid=True
    )
    assert valid["num_unique_trials"] == 72

    config_path = next(source.rglob("config.json"))
    config = json.loads(config_path.read_text())
    config["learning_rate"] = 1e-3
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(TuningMergeError, match="does not match"):
        merge_tuning_results(
            [source],
            tmp_path / "wrong_but_still_72",
            expected_trials=72,
            expect_full_grid=True,
        )
