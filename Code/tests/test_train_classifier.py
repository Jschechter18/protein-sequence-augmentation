"""Focused tests for the classifier experiment entrypoint."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from Code.src.training import train_classifier as train


def _sweep_args(results_dir: Path, *extra: str):
    return train.parse_args(
        ["--sweep", "--results_dir", str(results_dir), *extra]
    )


def _write_tiny_splits(data_dir: Path) -> None:
    task_dir = data_dir / "solubility"
    task_dir.mkdir(parents=True)
    contents = "idx,sequence,label\n0,ACD,0\n1,EFG,1\n"
    for split in ("train", "valid", "test"):
        (task_dir / f"{split}.csv").write_text(contents, encoding="utf-8")


def _write_trainable_tiny_splits(data_dir: Path) -> None:
    task_dir = data_dir / "solubility"
    task_dir.mkdir(parents=True)
    split_sequences = {
        "train": ("ACDE", "FGHIK", "LMNP", "QRSTV"),
        "valid": ("WYAC", "DEFGH", "IKLM", "NPQRS"),
        "test": ("TVWY", "CDEFG", "HIKL", "MNPQR"),
    }
    for split, sequences in split_sequences.items():
        contents = "idx,sequence,label\n" + "".join(
            f"{index},{sequence},{index % 2}\n"
            for index, sequence in enumerate(sequences)
        )
        (task_dir / f"{split}.csv").write_text(contents, encoding="utf-8")


def test_default_stage1_sweep_has_24_unique_balanced_configs(tmp_path: Path) -> None:
    configs = train.build_run_configs(_sweep_args(tmp_path), device="cpu")
    identities = {
        (config.representation, config.head_type, config.seed) for config in configs
    }

    assert len(configs) == 24
    assert len(identities) == 24

    seeds_by_condition: dict[tuple[str, str], set[int]] = defaultdict(set)
    for config in configs:
        seeds_by_condition[(config.representation, config.head_type)].add(config.seed)

    assert set(seeds_by_condition) == {
        (representation, head)
        for representation in train.STAGE1_REPRESENTATIONS
        for head in train.HEAD_TYPES
    }
    assert all(seeds == set(train.STAGE1_SEEDS) for seeds in seeds_by_condition.values())


def test_sweep_normalizes_aliases_and_removes_duplicate_axes(tmp_path: Path) -> None:
    args = _sweep_args(
        tmp_path,
        "--representations",
        "autoencoder+esm2",
        "trained_autoencoder+esm2",
        "esm2",
        "esm2",
        "--head_types",
        "linear",
        "linear",
        "--seeds",
        "42",
        "42",
        "43",
    )
    configs = train.build_run_configs(args, device="cpu")

    assert len(configs) == 4
    assert {config.representation for config in configs} == {
        "trained_autoencoder+esm2",
        "esm2",
    }
    assert len(
        {(config.representation, config.head_type, config.seed) for config in configs}
    ) == len(configs)


def test_run_directory_uses_stable_experiment_layout(tmp_path: Path) -> None:
    args = train.parse_args(
        [
            "--results_dir",
            str(tmp_path),
            "--dataset",
            "localization",
            "--version",
            "7",
            "--representation",
            "autoencoder+esm2",
            "--head_type",
            "mlp",
            "--seed",
            "44",
        ]
    )
    config = train.build_run_configs(args, device="cpu")[0]

    assert config.run_dir == (
        tmp_path
        / "localization"
        / "v7"
        / "trained_autoencoder+esm2"
        / "mlp"
        / "seed_44"
    )


def test_complete_run_requires_status_and_every_expected_artifact(tmp_path: Path) -> None:
    config = train.build_run_configs(
        train.parse_args(["--results_dir", str(tmp_path)]), device="cpu"
    )[0]
    run_dir = config.run_dir
    run_dir.mkdir(parents=True)

    assert not train._is_complete(run_dir, evaluate_test=True)

    for filename in (
        "config.json",
        "history.csv",
        "best_model.pt",
        "metrics.json",
        "test_predictions.csv",
    ):
        (run_dir / filename).write_text("{}", encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps({"status": "running"}), encoding="utf-8"
    )
    assert not train._is_complete(run_dir, evaluate_test=True)

    (run_dir / "status.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    assert train._is_complete(run_dir, evaluate_test=True)

    (run_dir / "test_predictions.csv").unlink()
    assert not train._is_complete(run_dir, evaluate_test=True)
    assert train._is_complete(run_dir, evaluate_test=False)


def test_overwrite_archives_existing_run_instead_of_deleting_it(tmp_path: Path) -> None:
    run_dir = tmp_path / "esm2" / "linear" / "seed_42"
    run_dir.mkdir(parents=True)
    (run_dir / "only-copy.txt").write_text("preserve me", encoding="utf-8")

    train._archive_run_dir(run_dir)

    assert not run_dir.exists()
    archives = list(run_dir.parent.glob("seed_42.backup_*"))
    assert len(archives) == 1
    assert (archives[0] / "only-copy.txt").read_text(encoding="utf-8") == "preserve me"


def test_aggregate_summary_reports_completed_seed_mean_and_sample_std(
    tmp_path: Path,
) -> None:
    configs = train.build_run_configs(
        _sweep_args(
            tmp_path,
            "--representations",
            "esm2",
            "--head_types",
            "linear",
            "--seeds",
            "42",
            "43",
            "44",
        ),
        device="cpu",
    )
    rows = [
        train._row_from_metrics(configs[0], {"accuracy": 0.6, "loss": 2.0}, "complete"),
        train._row_from_metrics(configs[1], {"accuracy": 0.8, "loss": 4.0}, "complete"),
        train._row_from_metrics(configs[2], None, "failed", "intentional"),
    ]

    train.save_summaries(configs, rows)

    summary_root = tmp_path / "solubility" / "v1"
    summary = pd.read_csv(summary_root / "summary.csv")
    aggregate = pd.read_csv(summary_root / "aggregated_summary.csv")
    assert len(summary) == 3
    assert len(aggregate) == 1
    assert aggregate.loc[0, "num_seeds"] == 2
    assert aggregate.loc[0, "accuracy_mean"] == pytest.approx(0.7)
    assert aggregate.loc[0, "accuracy_std"] == pytest.approx(2**0.5 / 10)
    assert aggregate.loc[0, "loss_mean"] == pytest.approx(3.0)
    assert aggregate.loc[0, "loss_std"] == pytest.approx(2**0.5)


def test_sweep_continues_after_a_run_failure_and_saves_all_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempted_seeds: list[int] = []
    saved_rows: list[dict[str, object]] = []

    monkeypatch.setattr(train, "validate_preflight", lambda configs: None)

    def fake_run_one(config, **kwargs):
        del kwargs
        attempted_seeds.append(config.seed)
        if config.seed == 42:
            raise RuntimeError("intentional failure")
        return train._row_from_metrics(config, {"accuracy": 0.75}, "complete")

    def fake_save_summaries(configs, rows):
        del configs
        saved_rows[:] = rows

    monkeypatch.setattr(train, "run_one", fake_run_one)
    monkeypatch.setattr(train, "save_summaries", fake_save_summaries)

    with pytest.raises(RuntimeError, match="1 of 2 classifier runs failed"):
        train.main(
            [
                "--sweep",
                "--results_dir",
                str(tmp_path),
                "--representations",
                "esm2",
                "--head_types",
                "linear",
                "--seeds",
                "42",
                "43",
            ]
        )

    assert attempted_seeds == [42, 43]
    assert [row["status"] for row in saved_rows] == ["failed", "complete"]


def test_run_fingerprint_rejects_changed_config_but_allows_epoch_extension(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    _write_tiny_splits(data_dir)
    args = train.parse_args(
        [
            "--results_dir",
            str(tmp_path / "results"),
            "--data_dir",
            str(data_dir),
            "--representation",
            "random_autoencoder",
            "--epochs",
            "2",
        ]
    )
    config = train.build_run_configs(args, device="cpu")[0]
    original = train._config_payload(config)

    train._validate_existing_config(original, original, for_resume=False)
    extended = train._config_payload(replace(config, epochs=4))
    train._validate_existing_config(original, extended, for_resume=True)

    changed_learning_rate = train._config_payload(
        replace(config, learning_rate=config.learning_rate * 2)
    )
    with pytest.raises(ValueError, match="Refusing to resume"):
        train._validate_existing_config(
            original, changed_learning_rate, for_resume=True
        )


def test_preflight_rejects_exact_sequence_leakage(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_tiny_splits(data_dir)
    config = train.build_run_configs(
        train.parse_args(
            [
                "--data_dir",
                str(data_dir),
                "--results_dir",
                str(tmp_path / "results"),
                "--representation",
                "random_autoencoder",
            ]
        ),
        device="cpu",
    )[0]

    with pytest.raises(ValueError, match="shared by the train and valid splits"):
        train.validate_dataset_integrity(config)


def test_resume_refuses_training_artifacts_without_last_checkpoint(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    _write_trainable_tiny_splits(data_dir)
    config = train.build_run_configs(
        train.parse_args(
            [
                "--data_dir",
                str(data_dir),
                "--results_dir",
                str(tmp_path / "results"),
                "--representation",
                "random_autoencoder",
            ]
        ),
        device="cpu",
    )[0]
    config.run_dir.mkdir(parents=True)
    (config.run_dir / "config.json").write_text(
        json.dumps(train._config_payload(config)), encoding="utf-8"
    )
    (config.run_dir / "best_model.pt").touch()

    with pytest.raises(FileNotFoundError, match="last_model.pt is missing"):
        train.run_one(
            config, resume=True, overwrite=False, skip_completed=False
        )


def test_summary_updates_merge_distinct_subset_runs(tmp_path: Path) -> None:
    configs = train.build_run_configs(
        _sweep_args(
            tmp_path,
            "--representations",
            "esm2",
            "--head_types",
            "linear",
            "--seeds",
            "42",
            "43",
        ),
        device="cpu",
    )
    train.save_summaries(
        [configs[0]],
        [train._row_from_metrics(configs[0], {"accuracy": 0.6}, "complete")],
    )
    train.save_summaries(
        [configs[1]],
        [train._row_from_metrics(configs[1], {"accuracy": 0.8}, "complete")],
    )

    root = tmp_path / "solubility" / "v1"
    summary = pd.read_csv(root / "summary.csv")
    aggregate = pd.read_csv(root / "aggregated_summary.csv")
    assert summary["seed"].tolist() == [42, 43]
    assert aggregate.loc[0, "num_seeds"] == 2
    assert aggregate.loc[0, "accuracy_mean"] == pytest.approx(0.7)


def test_tiny_random_autoencoder_entrypoint_creates_complete_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    _write_trainable_tiny_splits(data_dir)
    monkeypatch.setattr(train, "select_device", lambda: "cpu")

    train.main(
        [
            "--data_dir",
            str(data_dir),
            "--results_dir",
            str(results_dir),
            "--representation",
            "random_autoencoder",
            "--head_type",
            "linear",
            "--autoencoder_checkpoint",
            str(tmp_path / "unused.pt"),
            "--autoencoder_embedding_dim",
            "4",
            "--autoencoder_cnn_channels",
            "4",
            "--autoencoder_hidden_dim",
            "4",
            "--autoencoder_latent_dim",
            "2",
            "--autoencoder_num_layers",
            "1",
            "--autoencoder_kernel_size",
            "3",
            "--batch_size",
            "2",
            "--epochs",
            "1",
            "--early_stopping_patience",
            "1",
            "--no-use_cache",
        ]
    )

    run_dir = (
        results_dir
        / "solubility"
        / "v1"
        / "random_autoencoder"
        / "linear"
        / "seed_42"
    )
    assert json.loads((run_dir / "status.json").read_text())["status"] == "complete"
    assert {
        "config.json",
        "metrics.json",
        "history.csv",
        "best_model.pt",
        "last_model.pt",
        "test_predictions.csv",
        "run.log",
    }.issubset(path.name for path in run_dir.iterdir())
    assert len(pd.read_csv(run_dir / "test_predictions.csv")) == 4
