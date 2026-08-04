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
    tuning_root = results_dir / "solubility" / "v1" / "tuning"
    tuning_root.mkdir(parents=True, exist_ok=True)
    selected_path = tuning_root / "selected_hyperparameters.json"
    if not selected_path.exists():
        selected_path.write_text(
            json.dumps(
                {
                    head: {
                        representation: {
                            "learning_rate": 1e-3,
                            "weight_decay": 0.0,
                            **({"dropout": 0.1} if head == "mlp" else {}),
                        }
                        for representation in train.STAGE1_REPRESENTATIONS
                    }
                    for head in train.HEAD_TYPES
                }
            ),
            encoding="utf-8",
        )
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


def test_classifier_tuning_search_space_constants() -> None:
    assert train.TUNING_SEED == 42
    assert train.TUNING_REPRESENTATIONS == train.STAGE1_REPRESENTATIONS
    assert train.TUNING_LEARNING_RATES == (1e-4, 1e-5, 1e-6)
    assert train.TUNING_WEIGHT_DECAYS == (0.0, 1e-4)
    assert train.TUNING_MLP_DROPOUTS == (0.1, 0.3)


def test_classifier_checkpoints_default_to_project_checkpoint_tree() -> None:
    args = train.parse_args([])

    assert Path(args.checkpoint_root) == (
        train.PROJECT_ROOT / "checkpoints" / "classifier"
    )


def test_end_to_end_flag_builds_combined_uncached_config() -> None:
    args = train.parse_args(
        [
            "--representation",
            "trained_autoencoder+esm2",
            "--end_to_end",
            "--no-cache_embeddings",
        ]
    )

    config = train.build_run_configs(args, device="cpu")[0]
    assert config.end_to_end is True
    assert config.cache_embeddings is False
    assert config.encoder_mode == "fine_tuned"
    assert config.freeze_autoencoder is False
    assert config.freeze_esm2 is False


def test_end_to_end_flag_rejects_wrong_representation_and_disables_cache() -> None:
    with pytest.raises(SystemExit):
        train.parse_args(["--representation", "esm2", "--end_to_end"])

    args = train.parse_args(
        ["--representation", "trained_autoencoder+esm2", "--end_to_end"]
    )
    assert train.build_run_configs(args, device="cpu")[0].cache_embeddings is False


@pytest.mark.parametrize(
    ("representation", "freeze_argument", "encoder_mode"),
    [
        ("random_autoencoder", "--no-freeze_autoencoder", "from_scratch"),
        ("trained_autoencoder", "--no-freeze_autoencoder", "fine_tuned"),
        ("esm2", "--no-freeze_esm2", "fine_tuned"),
    ],
)
def test_standalone_trainable_modes_are_explicit_and_uncached(
    tmp_path: Path,
    representation: str,
    freeze_argument: str,
    encoder_mode: str,
) -> None:
    common_args = [
        "--representation",
        representation,
        "--results_dir",
        str(tmp_path / "results"),
        "--checkpoint_dir",
        str(tmp_path / "checkpoints"),
    ]
    args = train.parse_args([*common_args, freeze_argument])

    config = train.build_run_configs(args, device="cpu")[0]
    frozen_config = train.build_run_configs(
        train.parse_args(common_args), device="cpu"
    )[0]

    assert config.encoder_mode == encoder_mode
    assert config.cache_embeddings is False
    assert config.run_dir != frozen_config.run_dir
    assert config.checkpoint_dir != frozen_config.checkpoint_dir
    assert encoder_mode in config.run_dir.parts
    if "autoencoder" in representation:
        assert config.freeze_autoencoder is False
    if "esm2" in representation:
        assert config.freeze_esm2 is False


def test_combined_trainable_mode_uses_both_explicit_freeze_flags() -> None:
    args = train.parse_args(
        [
            "--representation",
            "trained_autoencoder+esm2",
            "--no-freeze_autoencoder",
            "--no-freeze_esm2",
        ]
    )

    config = train.build_run_configs(args, device="cpu")[0]

    assert config.encoder_mode == "fine_tuned"
    assert config.freeze_autoencoder is False
    assert config.freeze_esm2 is False
    assert config.cache_embeddings is False


@pytest.mark.parametrize(
    "freeze_argument",
    ["--no-freeze_autoencoder", "--no-freeze_esm2"],
)
def test_combined_partial_freezing_is_not_generated(
    freeze_argument: str,
) -> None:
    with pytest.raises(SystemExit):
        train.parse_args(
            [
                "--representation",
                "trained_autoencoder+esm2",
                freeze_argument,
            ]
        )


def test_v27_checkpoint_metadata_and_layer_type_remain_configurable() -> None:
    checkpoint = Path("checkpoints/autoencoder/solubility/v27/final.pt")
    args = train.parse_args(
        [
            "--representation",
            "trained_autoencoder",
            "--autoencoder_checkpoint",
            str(checkpoint),
            "--autoencoder_layer_type",
            "lstm",
        ]
    )

    config = train.build_run_configs(args, device="cpu")[0]

    assert config.autoencoder_checkpoint == str(checkpoint)
    assert config.autoencoder_version == "v27"
    assert config.autoencoder_layer_type == "lstm"
    assert config.encoder_mode == "frozen"
    row = train._row_from_metrics(config, None, "complete")
    assert row["autoencoder_version"] == "v27"
    assert row["encoder_mode"] == "frozen"
    assert row["freeze_autoencoder"] is True
    assert row["freeze_esm2"] is True


def test_embedding_cache_identity_is_seeded_only_for_random_encoder(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    _write_trainable_tiny_splits(data_dir)
    args = train.parse_args(
        [
            "--data_dir",
            str(data_dir),
            "--embedding_cache_dir",
            str(tmp_path / "embeddings"),
            "--representation",
            "random_autoencoder",
        ]
    )
    random_42 = train.build_run_configs(args, device="cpu")[0]
    random_43 = replace(random_42, seed=43)
    trained_42 = replace(random_42, representation="trained_autoencoder")
    trained_43 = replace(trained_42, seed=43)

    assert train._embedding_cache_path(random_42, "train")[0] != (
        train._embedding_cache_path(random_43, "train")[0]
    )
    assert train._embedding_cache_path(trained_42, "train")[0] == (
        train._embedding_cache_path(trained_43, "train")[0]
    )
    assert train._embedding_cache_path(random_42, "train")[0] != (
        train._embedding_cache_path(random_42, "valid")[0]
    )
    original_train_path = train._embedding_cache_path(random_42, "train")[0]
    train_csv = data_dir / "solubility" / "train.csv"
    train_csv.write_text(
        train_csv.read_text(encoding="utf-8") + "4,VWYAC,0\n",
        encoding="utf-8",
    )
    assert train._embedding_cache_path(random_42, "train")[0] != original_train_path


def test_embedding_cache_round_trip_preserves_examples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = train.build_run_configs(
        train.parse_args(
            [
                "--embedding_cache_dir",
                str(tmp_path),
                "--representation",
                "random_autoencoder",
            ]
        ),
        device="cpu",
    )[0]
    batch = {
        "input_ids": pd.Series(dtype=object),  # Unused by the fake encoder.
        "label": train.torch.tensor([0, 1]),
        "length": train.torch.tensor([3, 4]),
        "sequence": ["ACD", "EFGH"],
        "sample_id": [10, 11],
    }

    class FakeEncoder:
        def eval(self):
            return self

        def encode(self, ignored_batch):
            assert ignored_batch is batch
            return train.torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    monkeypatch.setattr(
        train,
        "_create_sequence_dataloader",
        lambda *args, **kwargs: [batch],
    )
    path = tmp_path / "cache.pt"
    metadata = {"fingerprint": "test"}

    train._build_embedding_cache(
        config, "train", FakeEncoder(), path, metadata
    )
    payload = train._load_embedding_cache(path, metadata)
    dataset = train.CachedEmbeddingDataset(payload)

    assert len(dataset) == 2
    assert dataset[1]["embedding"].tolist() == [3.0, 4.0]
    assert dataset[1]["label"].item() == 1
    assert dataset[1]["sequence"] == "EFGH"
    assert dataset[1]["sample_id"] == 11


def test_cached_random_autoencoder_path_builds_embeddings_and_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    _write_trainable_tiny_splits(data_dir)
    args = train.parse_args(
        [
            "--data_dir",
            str(data_dir),
            "--embedding_cache_dir",
            str(tmp_path / "embeddings"),
            "--representation",
            "random_autoencoder",
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
            "--num_workers",
            "0",
            "--no-evaluate_test",
            "--no-use_cache",
        ]
    )
    config = train.build_run_configs(args, device="cpu")[0]
    train.configure_reproducibility(config.seed, config.deterministic)
    raw_model = train.create_model(config, use_cached_embeddings=False)
    expected_head_state = {
        name: value.detach().clone()
        for name, value in raw_model.head.state_dict().items()
    }
    raw_valid_loader = train._create_sequence_dataloader(
        config, "valid", shuffle=False, offset=1
    )
    raw_model.eval()
    with train.torch.inference_mode():
        expected_valid_embeddings = train.torch.cat(
            [raw_model.encode(batch).cpu() for batch in raw_valid_loader]
        )
    del raw_model
    train.configure_reproducibility(config.seed, config.deterministic)

    train_loader, valid_loader, test_loader = train.create_run_dataloaders(config)
    train.configure_reproducibility(config.seed, config.deterministic)
    model = train.create_model(config)
    batch = next(iter(train_loader))

    assert isinstance(model, train.CachedEmbeddingClassifier)
    assert all(
        train.torch.equal(model.head.state_dict()[name], expected)
        for name, expected in expected_head_state.items()
    )
    assert model(batch).shape == (2, 2)
    cached_valid_embeddings = train.torch.cat(
        [cached_batch["embedding"] for cached_batch in valid_loader]
    )
    assert cached_valid_embeddings.shape[1] == 2
    assert train.torch.equal(cached_valid_embeddings, expected_valid_embeddings)
    assert test_loader is None
    assert len(list((tmp_path / "embeddings").rglob("*.pt"))) == 2

    def fail_if_encoder_is_rebuilt(*args, **kwargs):
        raise AssertionError("A valid embedding cache should be reused.")

    monkeypatch.setattr(train, "create_model", fail_if_encoder_is_rebuilt)
    reused_train, reused_valid, reused_test = train.create_run_dataloaders(config)
    assert next(iter(reused_train))["embedding"].shape[1] == 2
    assert next(iter(reused_valid))["embedding"].shape[1] == 2
    assert reused_test is None


def test_sweep_and_hp_tune_are_mutually_exclusive() -> None:
    assert train.parse_args(["--sweep"]).run_sweep is True
    assert train.parse_args(["--hp_tune"]).hp_tune is True
    assert train.parse_args(["--end_to_end_sweep"]).end_to_end_sweep is True

    with pytest.raises(SystemExit):
        train.parse_args(["--sweep", "--hp_tune"])
    with pytest.raises(SystemExit):
        train.parse_args(["--sweep", "--end_to_end_sweep"])


def test_default_tuning_grid_matches_declared_search_space(tmp_path: Path) -> None:
    args = train.parse_args(["--hp_tune", "--results_dir", str(tmp_path)])
    configs = train.build_run_configs(args, device="cpu")

    linear = [config for config in configs if config.head_type == "linear"]
    mlp = [config for config in configs if config.head_type == "mlp"]

    assert len(configs) == 72
    assert len(linear) == 24
    assert len(mlp) == 48
    assert {config.representation for config in configs} == set(
        train.TUNING_REPRESENTATIONS
    )
    assert {
        (config.learning_rate, config.weight_decay, config.dropout)
        for config in linear
    } == {
        (learning_rate, weight_decay, None)
        for learning_rate in train.TUNING_LEARNING_RATES
        for weight_decay in train.TUNING_WEIGHT_DECAYS
    }
    assert {
        (config.learning_rate, config.weight_decay, config.dropout)
        for config in mlp
    } == {
        (learning_rate, weight_decay, dropout)
        for learning_rate in train.TUNING_LEARNING_RATES
        for weight_decay in train.TUNING_WEIGHT_DECAYS
        for dropout in train.TUNING_MLP_DROPOUTS
    }
    assert all(config.seed == train.TUNING_SEED for config in configs)
    assert all(config.mode == "hp_tune" for config in configs)
    assert all(config.phase == "tuning" for config in configs)
    assert all(config.evaluate_test is False for config in configs)


def test_tuning_grid_respects_requested_heads_and_representations(
    tmp_path: Path,
) -> None:
    args = train.parse_args(
        [
            "--hp_tune",
            "--results_dir",
            str(tmp_path),
            "--representations",
            "trained_autoencoder",
            "esm2",
            "--head_types",
            "mlp",
        ]
    )

    configs = train.build_tuning_configs(args, device="cpu")

    assert len(configs) == 24
    assert {config.representation for config in configs} == {
        "trained_autoencoder",
        "esm2",
    }
    assert {config.head_type for config in configs} == {"mlp"}
    assert {config.dropout for config in configs} == {0.1, 0.3}


def test_tuning_paths_are_versioned_and_unique(tmp_path: Path) -> None:
    args = train.parse_args(
        [
            "--hp_tune",
            "--results_dir",
            str(tmp_path / "results"),
            "--checkpoint_dir",
            str(tmp_path / "checkpoints"),
            "--embedding_cache_dir",
            str(tmp_path / "embeddings"),
            "--version",
            "3",
            "--representations",
            "esm2",
            "--head_types",
            "mlp",
        ]
    )
    config = train.build_tuning_configs(args, device="cpu")[0]

    relative = (
        Path("solubility")
        / "v3"
        / "tuning"
        / "mlp"
        / "esm2"
        / "lr_1e-4_wd_0_do_0.1_seed_42"
    )
    assert config.run_dir == tmp_path / "results" / relative
    assert config.checkpoint_dir == tmp_path / "checkpoints" / relative
    assert len({candidate.run_dir for candidate in train.build_tuning_configs(args)}) == 12


def test_tuning_summaries_and_selected_hyperparameters_are_saved_separately(
    tmp_path: Path,
) -> None:
    args = train.parse_args(
        [
            "--hp_tune",
            "--results_dir",
            str(tmp_path),
            "--representations",
            "esm2",
            "--head_types",
            "mlp",
        ]
    )
    configs = train.build_tuning_configs(args, device="cpu")[:2]
    rows = [
        train._row_from_metrics(
            configs[0],
            {
                "selection_epoch": 2,
                "best_val_f1": 0.7,
                "val_loss_at_selection": 0.5,
                "val_accuracy_at_selection": 0.75,
            },
            "complete",
        ),
        train._row_from_metrics(
            configs[1],
            {
                "selection_epoch": 3,
                "best_val_f1": 0.8,
                "val_loss_at_selection": 0.4,
                "val_accuracy_at_selection": 0.8,
            },
            "complete",
        ),
    ]

    train.save_summaries(configs, rows)

    tuning_root = tmp_path / "solubility" / "v1" / "tuning"
    results = pd.read_csv(tuning_root / "tuning_results.csv")
    selected = json.loads(
        (tuning_root / "selected_hyperparameters.json").read_text()
    )
    assert len(results) == 2
    assert not (tmp_path / "solubility" / "v1" / "summary.csv").exists()
    assert selected["mlp"]["esm2"] == {
        "learning_rate": configs[1].learning_rate,
        "weight_decay": configs[1].weight_decay,
        "dropout": configs[1].dropout,
        "selection_seed": 42,
        "selection_epoch": 3,
        "best_val_f1": 0.8,
        "val_loss_at_selection": 0.4,
    }


def test_partial_tuning_output_includes_hardcoded_unselected_conditions(
    tmp_path: Path,
) -> None:
    args = train.parse_args(
        [
            "--hp_tune",
            "--results_dir",
            str(tmp_path),
            "--representations",
            "trained_autoencoder",
            "--head_types",
            "linear",
        ]
    )
    config = train.build_tuning_configs(args, device="cpu")[0]
    row = train._row_from_metrics(
        config,
        {
            "selection_epoch": 2,
            "best_val_f1": 0.7,
            "val_loss_at_selection": 0.5,
            "val_accuracy_at_selection": 0.75,
        },
        "complete",
    )

    train.save_summaries([config], [row])

    selected = json.loads(
        (
            tmp_path
            / "solubility"
            / "v1"
            / "tuning"
            / "selected_hyperparameters.json"
        ).read_text()
    )
    assert selected["linear"]["random_autoencoder"] == {
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "selection_source": "hardcoded_reuse",
    }
    assert selected["mlp"]["esm2"] == {
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "dropout": 0.3,
        "selection_source": "hardcoded_reuse",
    }
    assert "trained_autoencoder" in selected["linear"]
    assert "selection_source" not in selected["linear"]["trained_autoencoder"]

    final_configs = train.build_run_configs(_sweep_args(tmp_path), device="cpu")
    final_by_condition = {
        (config.head_type, config.representation, config.seed): config
        for config in final_configs
    }
    assert len(final_configs) == 24
    assert final_by_condition[("linear", "random_autoencoder", 42)].learning_rate == (
        pytest.approx(1e-3)
    )
    assert final_by_condition[("mlp", "esm2", 42)].dropout == pytest.approx(0.3)
    assert final_by_condition[("linear", "trained_autoencoder", 42)].learning_rate == (
        pytest.approx(config.learning_rate)
    )


def test_failed_requested_tuning_condition_does_not_use_hardcoded_default(
    tmp_path: Path,
) -> None:
    args = train.parse_args(
        [
            "--hp_tune",
            "--results_dir",
            str(tmp_path),
            "--representations",
            "trained_autoencoder",
            "--head_types",
            "linear",
        ]
    )
    config = train.build_tuning_configs(args, device="cpu")[0]
    failed_row = train._row_from_metrics(config, None, "failed", "test failure")

    train.save_summaries([config], [failed_row])

    selected = json.loads(
        (
            tmp_path
            / "solubility"
            / "v1"
            / "tuning"
            / "selected_hyperparameters.json"
        ).read_text()
    )
    assert "trained_autoencoder" not in selected["linear"]
    assert "random_autoencoder" in selected["linear"]
    assert "trained_autoencoder" in selected["mlp"]


def test_default_stage1_sweep_has_24_unique_balanced_configs(tmp_path: Path) -> None:
    configs = train.build_run_configs(_sweep_args(tmp_path), device="cpu")
    identities = {
        (config.representation, config.head_type, config.seed) for config in configs
    }

    assert len(configs) == 24
    assert len(identities) == 24
    assert all(config.encoder_mode == "frozen" for config in configs)
    assert all(config.freeze_autoencoder for config in configs)
    assert all(config.freeze_esm2 for config in configs)
    assert all(config.cache_embeddings for config in configs)
    assert all(not config.end_to_end for config in configs)

    seeds_by_condition: dict[tuple[str, str], set[int]] = defaultdict(set)
    for config in configs:
        seeds_by_condition[(config.representation, config.head_type)].add(config.seed)

    assert set(seeds_by_condition) == {
        (representation, head)
        for representation in train.STAGE1_REPRESENTATIONS
        for head in train.HEAD_TYPES
    }
    assert all(seeds == set(train.STAGE1_SEEDS) for seeds in seeds_by_condition.values())


def test_default_end_to_end_sweep_has_24_uncached_trainable_configs(
    tmp_path: Path,
) -> None:
    args = _sweep_args(tmp_path)
    args.run_sweep = False
    args.end_to_end_sweep = True

    configs = train.build_run_configs(args, device="cpu")

    assert len(configs) == 24
    assert all(config.end_to_end for config in configs)
    assert all(not config.cache_embeddings for config in configs)
    assert all(config.encoder_mode != "frozen" for config in configs)
    assert all(
        config.freeze_autoencoder is ("autoencoder" not in config.representation)
        for config in configs
    )
    assert all(
        config.freeze_esm2 is ("esm2" not in config.representation)
        for config in configs
    )
    assert all(config.mode == "end_to_end_sweep" for config in configs)
    assert all(config.phase == "end_to_end" for config in configs)
    assert all(
        config.run_dir
        == (
            tmp_path
            / "solubility"
            / "v1"
            / "end_to_end"
            / config.head_type
            / config.representation
            / f"seed_{config.seed}"
        )
        for config in configs
    )


def test_end_to_end_tuning_uses_compact_regularization_grid(tmp_path: Path) -> None:
    args = _sweep_args(tmp_path)
    args.run_sweep = False
    args.end_to_end_hp_tune = True

    configs = train.build_run_configs(args, device="cpu")

    assert len(configs) == 40
    counts = defaultdict(int)
    for config in configs:
        counts[(config.representation, config.head_type)] += 1
    assert counts == {
        ("random_autoencoder", "linear"): 4,
        ("random_autoencoder", "mlp"): 4,
        ("trained_autoencoder", "linear"): 6,
        ("trained_autoencoder", "mlp"): 6,
        ("esm2", "linear"): 4,
        ("esm2", "mlp"): 4,
        ("trained_autoencoder+esm2", "linear"): 6,
        ("trained_autoencoder+esm2", "mlp"): 6,
    }
    assert all(config.epochs == 10 for config in configs)
    assert all(config.early_stopping_patience == 3 for config in configs)
    assert all(config.seed == 42 for config in configs)
    assert all(config.end_to_end for config in configs)
    assert all(not config.evaluate_test for config in configs)
    assert all(not config.cache_embeddings for config in configs)
    assert all(config.phase == "end_to_end_tuning" for config in configs)
    assert all("end_to_end/tuning" in str(config.run_dir) for config in configs)


def test_successful_end_to_end_tuning_automatically_runs_final_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _sweep_args(tmp_path)
    args.run_sweep = False
    args.end_to_end_hp_tune = True
    tuning_config = train.build_run_configs(args, device="cpu")[0]
    final_config = replace(
        tuning_config,
        phase="end_to_end",
        mode="end_to_end_sweep",
        evaluate_test=True,
    )
    built_args = []
    run_phases = []

    def fake_build(run_args, device=None):
        built_args.append(run_args)
        return [tuning_config] if run_args.end_to_end_hp_tune else [final_config]

    monkeypatch.setattr(train, "parse_args", lambda _argv: args)
    monkeypatch.setattr(train, "build_run_configs", fake_build)
    monkeypatch.setattr(train, "validate_preflight", lambda _configs: None)
    monkeypatch.setattr(
        train,
        "run_one",
        lambda config, **_kwargs: (
            run_phases.append(config.phase)
            or train._row_from_metrics(config, {"accuracy": 0.5}, "complete")
        ),
    )
    monkeypatch.setattr(train, "save_summaries", lambda _configs, _rows: None)

    train.main([])

    assert run_phases == ["end_to_end_tuning", "end_to_end"]
    assert len(built_args) == 2
    automatic_final_args = built_args[1]
    assert automatic_final_args.end_to_end_sweep is True
    assert automatic_final_args.end_to_end_hp_tune is False
    assert automatic_final_args.epochs == 10
    assert automatic_final_args.early_stopping_patience == 3
    assert automatic_final_args.selected_hyperparameters.endswith(
        "end_to_end/tuning/selected_hyperparameters.json"
    )


def test_end_to_end_sweep_summary_is_separate_from_frozen_summary(
    tmp_path: Path,
) -> None:
    args = _sweep_args(
        tmp_path,
        "--representations",
        "esm2",
        "--head_types",
        "linear",
        "--seeds",
        "42",
    )
    args.run_sweep = False
    args.end_to_end_sweep = True
    configs = train.build_run_configs(args, device="cpu")

    train.save_summaries(
        configs,
        [
            train._row_from_metrics(
                configs[0], {"accuracy": 0.75, "loss": 0.5}, "complete"
            )
        ],
    )

    summary_root = tmp_path / "solubility" / "v1" / "end_to_end"
    assert (summary_root / "summary.csv").is_file()
    assert (summary_root / "aggregated_summary.csv").is_file()
    assert not (tmp_path / "solubility" / "v1" / "summary.csv").exists()


def test_sweep_uses_selected_hyperparameters_for_each_condition(
    tmp_path: Path,
) -> None:
    tuning_root = tmp_path / "solubility" / "v1" / "tuning"
    tuning_root.mkdir(parents=True)
    selected = {
        "linear": {
            "esm2": {
                "learning_rate": 3e-4,
                "weight_decay": 1e-4,
            },
            "trained_autoencoder": {
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
            },
        },
        "mlp": {
            "trained_autoencoder": {
                "learning_rate": 1e-4,
                "weight_decay": 0.0,
                "dropout": 0.3,
            },
            "esm2": {
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "dropout": 0.1,
            },
        },
    }
    (tuning_root / "selected_hyperparameters.json").write_text(
        json.dumps(selected), encoding="utf-8"
    )
    args = _sweep_args(
        tmp_path,
        "--representations",
        "esm2",
        "trained_autoencoder",
        "--head_types",
        "linear",
        "mlp",
        "--seeds",
        "42",
    )

    configs = train.build_run_configs(args, device="cpu")
    by_condition = {
        (config.head_type, config.representation): config for config in configs
    }

    esm2_linear = by_condition[("linear", "esm2")]
    assert esm2_linear.learning_rate == pytest.approx(3e-4)
    assert esm2_linear.weight_decay == pytest.approx(1e-4)
    assert esm2_linear.dropout is None

    trained_mlp = by_condition[("mlp", "trained_autoencoder")]
    assert trained_mlp.learning_rate == pytest.approx(1e-4)
    assert trained_mlp.weight_decay == pytest.approx(0.0)
    assert trained_mlp.dropout == pytest.approx(0.3)

    assert by_condition[("linear", "trained_autoencoder")].learning_rate == pytest.approx(
        args.learning_rate
    )
    assert by_condition[("mlp", "esm2")].dropout == pytest.approx(0.1)


def test_explicit_selected_hyperparameter_path_must_exist(tmp_path: Path) -> None:
    args = _sweep_args(
        tmp_path,
        "--selected_hyperparameters",
        str(tmp_path / "missing.json"),
    )

    with pytest.raises(FileNotFoundError, match="Final sweeps require selected"):
        train.build_run_configs(args, device="cpu")


def test_sweep_rejects_a_missing_selected_condition(tmp_path: Path) -> None:
    selected_path = tmp_path / "selected.json"
    selected_path.write_text(
        json.dumps(
            {
                "linear": {
                    "esm2": {
                        "learning_rate": 1e-3,
                        "weight_decay": 0.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    args = _sweep_args(
        tmp_path,
        "--representations",
        "esm2",
        "--head_types",
        "mlp",
        "--selected_hyperparameters",
        str(selected_path),
    )

    with pytest.raises(ValueError, match="no valid 'mlp' section"):
        train.build_run_configs(args, device="cpu")


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
            "--checkpoint_dir",
            str(tmp_path / "checkpoints"),
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
    assert config.checkpoint_dir == (
        tmp_path
        / "checkpoints"
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
    checkpoint_dir = tmp_path / "checkpoints"
    run_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)

    assert not train._is_complete(run_dir, evaluate_test=True, checkpoint_dir=checkpoint_dir)

    for filename in (
        "config.json",
        "history.csv",
        "metrics.json",
        "test_predictions.csv",
    ):
        (run_dir / filename).write_text("{}", encoding="utf-8")
    (checkpoint_dir / "best_model.pt").write_text("{}", encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps({"status": "running"}), encoding="utf-8"
    )
    assert not train._is_complete(run_dir, evaluate_test=True, checkpoint_dir=checkpoint_dir)

    (run_dir / "status.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    assert train._is_complete(run_dir, evaluate_test=True, checkpoint_dir=checkpoint_dir)

    (run_dir / "test_predictions.csv").unlink()
    assert not train._is_complete(run_dir, evaluate_test=True, checkpoint_dir=checkpoint_dir)
    assert train._is_complete(run_dir, evaluate_test=False, checkpoint_dir=checkpoint_dir)


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
    _sweep_args(tmp_path)
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
                "--checkpoint_dir",
                str(tmp_path / "checkpoints"),
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
                "--checkpoint_dir",
                str(tmp_path / "checkpoints"),
                "--representation",
                "random_autoencoder",
            ]
        ),
        device="cpu",
    )[0]
    config.run_dir.mkdir(parents=True)
    config.checkpoint_dir.mkdir(parents=True)
    (config.run_dir / "config.json").write_text(
        json.dumps(train._config_payload(config)), encoding="utf-8"
    )
    (config.checkpoint_dir / "best_model.pt").touch()

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


def test_trainable_single_run_does_not_replace_frozen_summary(
    tmp_path: Path,
) -> None:
    common_args = [
        "--results_dir",
        str(tmp_path),
        "--representation",
        "esm2",
    ]
    frozen = train.build_run_configs(
        train.parse_args(common_args), device="cpu"
    )[0]
    fine_tuned = train.build_run_configs(
        train.parse_args([*common_args, "--no-freeze_esm2"]), device="cpu"
    )[0]

    train.save_summaries(
        [frozen],
        [train._row_from_metrics(frozen, {"accuracy": 0.5}, "complete")],
    )
    train.save_summaries(
        [fine_tuned],
        [train._row_from_metrics(fine_tuned, {"accuracy": 0.8}, "complete")],
    )

    root = tmp_path / "solubility" / "v1"
    frozen_summary = pd.read_csv(root / "summary.csv")
    fine_tuned_summary = pd.read_csv(root / "fine_tuned" / "summary.csv")
    assert frozen_summary["encoder_mode"].tolist() == ["frozen"]
    assert fine_tuned_summary["encoder_mode"].tolist() == ["fine_tuned"]


def test_legacy_end_to_end_summary_infers_trainable_encoder_state(
    tmp_path: Path,
) -> None:
    args = _sweep_args(
        tmp_path,
        "--representations",
        "trained_autoencoder+esm2",
        "--head_types",
        "linear",
        "--seeds",
        "42",
    )
    args.run_sweep = False
    args.end_to_end_sweep = True
    config = train.build_run_configs(args, device="cpu")[0]
    summary_root = tmp_path / "solubility" / "v1" / "end_to_end"
    summary_root.mkdir(parents=True)
    legacy_row = train._row_from_metrics(
        config, {"accuracy": 0.5}, "complete"
    )
    for field in (
        "encoder_mode",
        "freeze_autoencoder",
        "freeze_esm2",
        "autoencoder_version",
    ):
        legacy_row.pop(field)
    pd.DataFrame([legacy_row]).to_csv(summary_root / "summary.csv", index=False)

    train.save_summaries(
        [config],
        [train._row_from_metrics(config, {"accuracy": 0.8}, "complete")],
    )

    summary = pd.read_csv(summary_root / "summary.csv")
    aggregate = pd.read_csv(summary_root / "aggregated_summary.csv")
    assert len(summary) == 1
    assert summary.loc[0, "encoder_mode"] == "fine_tuned"
    assert not bool(summary.loc[0, "freeze_autoencoder"])
    assert not bool(summary.loc[0, "freeze_esm2"])
    assert len(aggregate) == 1
    assert aggregate.loc[0, "accuracy_mean"] == pytest.approx(0.8)


def test_tiny_random_autoencoder_entrypoint_creates_complete_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
            "--checkpoint_dir",
            str(tmp_path / "checkpoints"),
            "--embedding_cache_dir",
            str(tmp_path / "embeddings"),
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
            "--num_workers",
            "0",
            "--epochs",
            "1",
            "--early_stopping_patience",
            "1",
            "--no-use_cache",
        ]
    )

    assert "Device: cpu" in capsys.readouterr().out

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
        "test_predictions.csv",
        "run.log",
    }.issubset(path.name for path in run_dir.iterdir())
    assert not (run_dir / "best_model.pt").exists()
    assert not (run_dir / "last_model.pt").exists()
    checkpoint_dir = (
        tmp_path
        / "checkpoints"
        / "solubility"
        / "v1"
        / "random_autoencoder"
        / "linear"
        / "seed_42"
    )
    assert {"best_model.pt", "last_model.pt"}.issubset(
        path.name for path in checkpoint_dir.iterdir()
    )
    assert len(pd.read_csv(run_dir / "test_predictions.csv")) == 4
    assert len(list((tmp_path / "embeddings").rglob("*.pt"))) == 3
