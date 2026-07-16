import sys
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from training import train_autoencoder
from utils.dataloader import VOCAB_SIZE, collate_sequence_batch
from utils.hyperparameters import (
    AutoencoderHyperparameters as Params,
    sweep_search_space_for_layer,
)
from utils import train_input_validation
from utils.train_input_validation import autoencoder_artifact_paths


class _FastTqdm:
    def __init__(self, iterable, **_kwargs):
        self.iterable = iterable

    def __iter__(self):
        yield from self.iterable

    def set_postfix(self, **_kwargs):
        pass


class _RecordingScheduler:
    def __init__(self):
        self.metrics = []

    def step(self, metric):
        self.metrics.append(metric)

    def state_dict(self):
        return {}


class _RecordingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(VOCAB_SIZE))
        self.calls = []
        self.autoregressive_calls = []

    def forward(self, input_ids, decoder_input_ids=None, lengths=None):
        self.calls.append(
            {
                "input_ids": input_ids.detach().clone(),
                "decoder_input_ids": decoder_input_ids.detach().clone(),
                "lengths": lengths.detach().clone(),
            }
        )
        batch_size, sequence_length = decoder_input_ids.shape
        return self.logits.view(1, 1, -1).expand(batch_size, sequence_length, -1)

    def encode(self, input_ids, lengths=None):
        return input_ids.float()

    def decode_autoregressive(self, latent, max_length):
        self.autoregressive_calls.append(
            {
                "latent": latent.detach().clone(),
                "max_length": max_length,
            }
        )
        batch_size = latent.size(0)
        return self.logits.view(1, 1, -1).expand(batch_size, max_length, -1)


class _LengthDataset(Dataset):
    def __init__(self, lengths):
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        length = self.lengths[idx]
        ids = torch.tensor([2] + [4] * (length - 2) + [3], dtype=torch.long)
        return {
            "input_ids": ids,
            "target_ids": ids.clone(),
            "length": torch.tensor(length, dtype=torch.long),
            "sequence": "A" * length,
        }


def _batch(ids):
    input_ids = torch.tensor(ids, dtype=torch.long)
    lengths = (input_ids != train_autoencoder.PAD_IDX).sum(dim=1)
    return {
        "input_ids": input_ids,
        "target_ids": input_ids.clone(),
        "length": lengths,
    }


def _configure_mock_main_pipeline(monkeypatch, tmp_path):
    calls = {
        "artifact_suffixes": [],
    }

    class DummyFullTrainLoader:
        dataset = object()

    def fake_create_dataloader(**kwargs):
        calls.setdefault("dataloader_kwargs", []).append(kwargs)
        if kwargs["split"] == train_autoencoder.TRAIN_SPLIT and "loader_type" not in kwargs:
            return DummyFullTrainLoader()
        return [kwargs["split"]]

    def fake_artifact_paths(
        model_type,
        task,
        version,
        length_options,
        length_bin=None,
        is_overfit=False,
        artifact_suffix=None,
    ):
        calls["artifact_suffixes"].append(artifact_suffix)
        artifact_label = artifact_suffix or "default"
        return (
            tmp_path / f"{model_type}_{task}_{version}_{artifact_label}.pt",
            tmp_path / f"{model_type}_{task}_{version}_{artifact_label}_history.json",
        )

    def fake_train(
        model_type,
        train_dataloader,
        val_dataloader,
        hyperparams,
        **kwargs,
    ):
        calls["train"] = {
            "model_type": model_type,
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "hyperparams": hyperparams,
            "kwargs": kwargs,
        }
        return object(), {}

    monkeypatch.setattr(train_autoencoder, "create_dataloader", fake_create_dataloader)
    monkeypatch.setattr(
        train_autoencoder,
        "compute_train_length_boundaries",
        lambda _dataset, num_bins: list(range(num_bins + 1)),
    )
    monkeypatch.setattr(train_autoencoder, "autoencoder_artifact_paths", fake_artifact_paths)
    monkeypatch.setattr(train_input_validation, "autoencoder_artifact_paths", fake_artifact_paths)
    monkeypatch.setattr(train_autoencoder, "train", fake_train)
    monkeypatch.setattr(train_autoencoder, "test", lambda *_args, **_kwargs: None)

    return calls


def test_autoencoder_artifact_paths_split_checkpoint_and_history_dirs():
    checkpoint_path, history_path = autoencoder_artifact_paths(
        "AE",
        "solubility",
        7,
        "thirds",
        length_bin=2,
        is_overfit=False,
        artifact_suffix="latent256_tfd0p45",
    )

    assert checkpoint_path.parts[-6:] == (
        "protein-sequence-augmentation",
        "checkpoints",
        "autoencoder",
        "solubility",
        "v7",
        "model_ae_length_2_of_3_solubility_latent256_tfd0p45.pt",
    )
    assert history_path.parts[-7:] == (
        "protein-sequence-augmentation",
        "Code",
        "results",
        "autoencoder",
        "solubility",
        "v7",
        "v7_model_ae_length_2_of_3_solubility_latent256_tfd0p45_history.json",
    )


def test_model_definition_builds_autoencoder_optimizer_and_scheduler(monkeypatch):
    created = {}

    class FakeAutoencoder(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))
            created["kwargs"] = kwargs

        def to(self, device):
            created["device"] = device
            return self

    monkeypatch.setattr(train_autoencoder, "AE", FakeAutoencoder)
    monkeypatch.setattr(train_autoencoder, "device", torch.device("cpu"))

    hyperparams = replace(
        Params(),
        embedding_dim=8,
        cnn_out_channels=9,
        hidden_dim=16,
        latent_dim=4,
        num_layers=2,
        kernel_size=3,
        bidirectional=False,
        dropout=0.2,
        learning_rate=0.123,
    )

    model, optimizer, scheduler = train_autoencoder.model_definition("ae", hyperparams)

    assert isinstance(model, FakeAutoencoder)
    assert created["kwargs"] == {
        "layer_type": "gru",
        "embedding_dim": 8,
        "cnn_out_channels": 9,
        "hidden_dim": 16,
        "latent_dim": 4,
        "num_layers": 2,
        "kernel_size": 3,
        "bidirectional": False,
        "dropout": 0.2,
        "pad_idx": train_autoencoder.PAD_IDX,
        "bos_idx": train_autoencoder.BOS_IDX,
        "condition_decoder_on_latent": True,
        "teacher_forcing_dropout_rate": hyperparams.teacher_forcing_dropout_rate,
        "use_decoder_positional_embeddings": hyperparams.use_decoder_positional_embeddings,
        "max_decoder_positions": hyperparams.max_decoder_positions,
        "max_encoder_positions": hyperparams.max_encoder_positions,
        "num_heads": hyperparams.num_heads,
        "dim_feedforward": hyperparams.dim_feedforward,
        "use_cnn_before_transformer": hyperparams.use_cnn_before_transformer,
    }
    assert created["device"] == torch.device("cpu")
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.123)
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_model_definition_uses_hyperparameter_layer_type_by_default(monkeypatch):
    created = {}

    class FakeAutoencoder(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))
            created["kwargs"] = kwargs

        def to(self, device):
            return self

    monkeypatch.setattr(train_autoencoder, "AE", FakeAutoencoder)

    hyperparams = replace(Params(), layer_type="transformer")

    train_autoencoder.model_definition("ae", hyperparams)

    assert created["kwargs"]["layer_type"] == "transformer"


def test_model_definition_layer_type_override_wins(monkeypatch):
    created = {}

    class FakeAutoencoder(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))
            created["kwargs"] = kwargs

        def to(self, device):
            return self

    monkeypatch.setattr(train_autoencoder, "AE", FakeAutoencoder)

    hyperparams = replace(Params(), layer_type="transformer")

    train_autoencoder.model_definition("ae", hyperparams, layer_type="gru")

    assert created["kwargs"]["layer_type"] == "gru"


def test_sweep_search_space_depends_on_layer_type() -> None:
    assert sweep_search_space_for_layer("gru") == {
        "learning_rate": (1e-4, 3e-4),
        "num_layers": (2, 3),
        "hidden_dim": (512, 1024),
    }
    assert sweep_search_space_for_layer("transformer") == {
        "learning_rate": (1e-4, 3e-4),
        "num_layers": (2, 3),
        "num_heads": (4, 8),
        "dim_feedforward": (512, 1024),
    }

    with pytest.raises(ValueError, match="Unsupported layer_type"):
        sweep_search_space_for_layer("cnn")


def test_train_uses_shifted_decoder_inputs_and_validation_loss(monkeypatch):
    model = _RecordingAutoencoder()
    scheduler = _RecordingScheduler()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    monkeypatch.setattr(train_autoencoder, "device", torch.device("cpu"))
    monkeypatch.setattr(train_autoencoder, "tqdm", _FastTqdm)
    monkeypatch.setattr(train_autoencoder.torch, "save", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        train_autoencoder,
        "model_definition",
        lambda _model_type, _hyperparams, **_kwargs: (model, optimizer, scheduler),
    )

    train_batch = _batch([[2, 4, 5, 0], [2, 6, 0, 0]])
    valid_batch = _batch([[2, 7, 0, 0]])
    before = model.logits.detach().clone()

    returned_model, history = train_autoencoder.train(
        "ae",
        [train_batch],
        [valid_batch],
        Params(num_epochs=1),
        version=0,
    )

    assert returned_model is model
    assert len(model.calls) == 2
    assert model.calls[0]["decoder_input_ids"].tolist() == [[2, 4, 5], [2, 6, 0]]
    assert model.calls[1]["decoder_input_ids"].tolist() == [[2, 7, 0]]
    assert not torch.equal(model.logits.detach(), before)
    assert len(scheduler.metrics) == 1
    assert scheduler.metrics[0] == pytest.approx(history["val_loss"][0])
    assert history["autoregressive_val"] == []


def test_train_runs_autoregressive_validation_every_10_epochs(monkeypatch):
    model = _RecordingAutoencoder()
    scheduler = _RecordingScheduler()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    monkeypatch.setattr(train_autoencoder, "device", torch.device("cpu"))
    monkeypatch.setattr(train_autoencoder, "tqdm", _FastTqdm)
    monkeypatch.setattr(train_autoencoder.torch, "save", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        train_autoencoder,
        "model_definition",
        lambda _model_type, _hyperparams, **_kwargs: (model, optimizer, scheduler),
    )

    train_batch = _batch([[2, 4, 5, 0], [2, 6, 0, 0]])
    valid_batch = _batch([[2, 7, 0, 0]])

    _, history = train_autoencoder.train(
        "ae",
        [train_batch],
        [valid_batch],
        Params(num_epochs=10, patience=20),
        version=0,
    )

    assert len(model.autoregressive_calls) == 1
    assert model.autoregressive_calls[0]["max_length"] == 3
    assert history["autoregressive_val"] == [
        {
            "epoch": 10,
            "loss": pytest.approx(history["epochs"][9]["autoregressive_val_loss"]),
            "accuracy": pytest.approx(history["epochs"][9]["autoregressive_val_accuracy"]),
        }
    ]


def test_length_curriculum_uses_shortest_examples_first():
    dataloader = DataLoader(
        _LengthDataset([8, 3, 6, 4]),
        batch_size=4,
        shuffle=False,
        collate_fn=collate_sequence_batch,
    )

    curriculum_loader, subset_size, fraction = train_autoencoder.make_length_curriculum_dataloader(
        dataloader,
        epoch=0,
        curriculum_epochs=3,
        start_fraction=0.5,
        num_workers=0,
    )
    batch = next(iter(curriculum_loader))

    assert subset_size == 2
    assert fraction == pytest.approx(0.5)
    assert sorted(batch["length"].tolist()) == [3, 4]


def test_make_overfit_dataloaders_uses_same_training_subset():
    dataloader = DataLoader(
        _LengthDataset([3, 4, 5, 6, 7]),
        batch_size=2,
        shuffle=False,
        collate_fn=collate_sequence_batch,
    )

    train_loader, val_loader = train_autoencoder.make_overfit_dataloaders(
        dataloader,
        num_batches=2,
    )

    assert len(train_loader.dataset) == 4
    assert len(val_loader.dataset) == 4
    assert train_loader.batch_size == dataloader.batch_size
    assert val_loader.batch_size == dataloader.batch_size
    assert sorted(val_loader.dataset[i]["length"].item() for i in range(len(val_loader.dataset))) == [3, 4, 5, 6]


def test_make_overfit_dataloaders_rejects_invalid_batch_count():
    dataloader = DataLoader(
        _LengthDataset([3, 4]),
        batch_size=2,
        shuffle=False,
        collate_fn=collate_sequence_batch,
    )

    with pytest.raises(ValueError, match="overfit_batches"):
        train_autoencoder.make_overfit_dataloaders(dataloader, num_batches=0)


def test_main_validates_args_and_starts_gru_autoencoder_training(monkeypatch, tmp_path):
    calls = _configure_mock_main_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_autoencoder.py",
            "--task",
            "solubility",
            "--version",
            "999999",
            "--curriculum_epochs",
            "3",
            "--curriculum_start_fraction",
            "0.4",
        ],
    )

    train_autoencoder.main()

    assert [call["split"] for call in calls["dataloader_kwargs"]] == [
        train_autoencoder.TRAIN_SPLIT,
        train_autoencoder.TRAIN_SPLIT,
        train_autoencoder.VALID_SPLIT,
        "test",
    ]
    assert calls["train"]["model_type"] == "ae"
    assert calls["train"]["train_dataloader"] == [train_autoencoder.TRAIN_SPLIT]
    assert calls["train"]["val_dataloader"] == [train_autoencoder.VALID_SPLIT]
    assert isinstance(calls["train"]["hyperparams"], Params)
    assert calls["train"]["hyperparams"].layer_type == "gru"
    assert calls["train"]["kwargs"]["layer_type"] == "gru"
    assert calls["train"]["kwargs"]["artifact_suffix"] is None
    assert calls["artifact_suffixes"] == [None, None]
    assert calls["train"]["kwargs"]["curriculum_epochs"] == 3
    assert calls["train"]["kwargs"]["curriculum_start_fraction"] == pytest.approx(0.4)


def test_main_cli_layer_type_override_starts_transformer_pipeline(monkeypatch, tmp_path):
    calls = _configure_mock_main_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_autoencoder.py",
            "--task",
            "solubility",
            "--version",
            "999999",
            "--layer_type",
            "transformer",
        ],
    )

    train_autoencoder.main()

    assert calls["train"]["hyperparams"].layer_type == "transformer"
    assert calls["train"]["kwargs"]["layer_type"] == "transformer"
    assert calls["train"]["kwargs"]["artifact_suffix"] == "transformer"
    assert calls["artifact_suffixes"] == ["transformer", "transformer"]


def test_main_uses_hyperparameter_layer_type_when_cli_omits_it(monkeypatch, tmp_path):
    calls = _configure_mock_main_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(train_input_validation, "AEParams", lambda: replace(Params(), layer_type="transformer"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_autoencoder.py",
            "--task",
            "solubility",
            "--version",
            "999999",
        ],
    )

    train_autoencoder.main()

    assert calls["train"]["hyperparams"].layer_type == "transformer"
    assert calls["train"]["kwargs"]["layer_type"] == "transformer"
    assert calls["train"]["kwargs"]["artifact_suffix"] == "transformer"
    assert calls["artifact_suffixes"] == ["transformer", "transformer"]


def test_main_cli_layer_type_overrides_hyperparameter_default(monkeypatch, tmp_path):
    calls = _configure_mock_main_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(train_input_validation, "AEParams", lambda: replace(Params(), layer_type="transformer"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_autoencoder.py",
            "--task",
            "solubility",
            "--version",
            "999999",
            "--layer_type",
            "gru",
        ],
    )

    train_autoencoder.main()

    assert calls["train"]["hyperparams"].layer_type == "gru"
    assert calls["train"]["kwargs"]["layer_type"] == "gru"
    assert calls["train"]["kwargs"]["artifact_suffix"] is None
    assert calls["artifact_suffixes"] == [None, None]


def test_main_sweep_uses_transformer_search_space_and_artifact_suffixes(monkeypatch, tmp_path):
    calls = _configure_mock_main_pipeline(monkeypatch, tmp_path)
    train_calls = []

    def fake_train(
        model_type,
        train_dataloader,
        val_dataloader,
        hyperparams,
        **kwargs,
    ):
        train_calls.append(
            {
                "model_type": model_type,
                "hyperparams": hyperparams,
                "kwargs": kwargs,
            }
        )
        return object(), {}

    monkeypatch.setattr(train_autoencoder, "train", fake_train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_autoencoder.py",
            "--task",
            "solubility",
            "--version",
            "999999",
            "--layer_type",
            "transformer",
            "--sweep",
        ],
    )

    train_autoencoder.main()

    assert len(train_calls) == 16
    assert {call["hyperparams"].layer_type for call in train_calls} == {"transformer"}
    assert {call["kwargs"]["layer_type"] for call in train_calls} == {"transformer"}
    assert all(call["kwargs"]["artifact_suffix"].startswith("transformer_") for call in train_calls)
    assert any("num_heads4" in call["kwargs"]["artifact_suffix"] for call in train_calls)
    assert any("dim_feedforward1024" in call["kwargs"]["artifact_suffix"] for call in train_calls)
    assert calls["artifact_suffixes"] == [call["kwargs"]["artifact_suffix"] for call in train_calls]


def test_main_validation_errors(monkeypatch, tmp_path):
    _configure_mock_main_pipeline(monkeypatch, tmp_path)

    monkeypatch.setattr(sys, "argv", ["train_autoencoder.py", "--model", "CNN"])
    with pytest.raises(ValueError, match="Only --model AE"):
        train_autoencoder.main()

    monkeypatch.setattr(sys, "argv", ["train_autoencoder.py", "--task", "stability"])
    with pytest.raises(ValueError, match="localization' or 'solubility"):
        train_autoencoder.main()

    monkeypatch.setattr(
        sys,
        "argv",
        ["train_autoencoder.py", "--version", "999999", "--curriculum_start_fraction", "0"],
    )
    with pytest.raises(ValueError, match="curriculum_start_fraction"):
        train_autoencoder.main()

    missing_checkpoint = tmp_path / "missing.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_autoencoder.py",
            "--version",
            "999999",
            "--load_path",
            str(missing_checkpoint),
        ],
    )
    with pytest.raises(ValueError, match="--load_path does not exist"):
        train_autoencoder.main()


def test_main_warns_when_max_length_is_ignored_by_length_options(monkeypatch, tmp_path):
    _configure_mock_main_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_autoencoder.py",
            "--task",
            "solubility",
            "--version",
            "999999",
            "--max_length",
            "128",
        ],
    )

    with pytest.warns(UserWarning, match="--max_length is ignored"):
        train_autoencoder.main()


def test_main_rejects_invalid_transformer_hyperparameters(monkeypatch, tmp_path):
    _configure_mock_main_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(
        train_input_validation,
        "AEParams",
        lambda: replace(Params(), embedding_dim=30, num_heads=8),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_autoencoder.py",
            "--task",
            "solubility",
            "--version",
            "999999",
            "--layer_type",
            "transformer",
        ],
    )

    with pytest.raises(ValueError, match="embedding_dim must be divisible by num_heads"):
        train_autoencoder.main()


def test_main_rejects_even_kernel_size_when_cnn_stem_is_used(monkeypatch, tmp_path):
    _configure_mock_main_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(
        train_input_validation,
        "AEParams",
        lambda: replace(Params(), kernel_size=4),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_autoencoder.py",
            "--task",
            "solubility",
            "--version",
            "999999",
        ],
    )

    with pytest.raises(ValueError, match="kernel_size must be odd"):
        train_autoencoder.main()


def test_main_allows_even_kernel_size_for_transformer_without_cnn_stem(monkeypatch, tmp_path):
    calls = _configure_mock_main_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(
        train_input_validation,
        "AEParams",
        lambda: replace(
            Params(),
            kernel_size=4,
            layer_type="transformer",
            use_cnn_before_transformer=False,
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_autoencoder.py",
            "--task",
            "solubility",
            "--version",
            "999999",
        ],
    )

    train_autoencoder.main()

    assert calls["train"]["hyperparams"].kernel_size == 4
    assert calls["train"]["hyperparams"].layer_type == "transformer"
