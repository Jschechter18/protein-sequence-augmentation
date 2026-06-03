import sys
from dataclasses import replace

import pytest
import torch
from torch import nn

from training import train_autoencoder
from utils.hyperparameters import AutoencoderHyperParameters as Params


class _FastTqdm:
    def __init__(self, iterable, **_kwargs):
        self.iterable = iterable
        self.n = 0

    def __iter__(self):
        for item in self.iterable:
            self.n += 1
            yield item

    def set_postfix(self, **_kwargs):
        pass


class _RecordingScheduler:
    def __init__(self):
        self.metrics = []

    def step(self, metric):
        self.metrics.append(metric)


class _RecordingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(train_autoencoder.VOCAB_SIZE))
        self.calls = []

    def forward(self, input_ids, decoder_input_ids=None):
        self.calls.append(
            {
                "input_ids": input_ids.detach().clone(),
                "decoder_input_ids": decoder_input_ids.detach().clone(),
            }
        )
        batch_size, sequence_length = decoder_input_ids.shape
        return self.logits.view(1, 1, -1).expand(batch_size, sequence_length, -1)


def test_model_definition_builds_autoencoder_optimizer_and_scheduler(monkeypatch):
    created = {}

    class FakeAutoencoder(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))
            created["kwargs"] = kwargs
            created["device_before_to"] = None

        def to(self, device):
            created["device_before_to"] = device
            return self

    monkeypatch.setattr(train_autoencoder, "AE", FakeAutoencoder)
    monkeypatch.setattr(train_autoencoder, "device", torch.device("cpu"))

    hyperparams = replace(
        Params(),
        embedding_dim=8,
        hidden_dim=16,
        latent_dim=4,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.123,
    )

    model, optimizer, scheduler = train_autoencoder.model_definition(hyperparams)

    assert isinstance(model, FakeAutoencoder)
    assert created["kwargs"] == {
        "embedding_dim": 8,
        "hidden_dim": 16,
        "latent_dim": 4,
        "num_layers": 2,
        "dropout": 0.2,
        "pad_idx": train_autoencoder.PAD_IDX,
        "bos_idx": train_autoencoder.BOS_IDX,
    }
    assert created["device_before_to"] == torch.device("cpu")
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.123)
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_train_uses_shifted_decoder_inputs_ignores_padding_and_updates_model(monkeypatch):
    model = _RecordingAutoencoder()
    scheduler = _RecordingScheduler()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    monkeypatch.setattr(train_autoencoder, "device", torch.device("cpu"))
    monkeypatch.setattr(train_autoencoder, "tqdm", _FastTqdm)
    monkeypatch.setattr(
        train_autoencoder,
        "model_definition",
        lambda _hyperparams: (model, optimizer, scheduler),
    )

    batch = {
        "input_ids": torch.tensor([[2, 4, 5, 0], [2, 6, 0, 0]], dtype=torch.long),
        "target_ids": torch.tensor([[2, 4, 5, 0], [2, 6, 0, 0]], dtype=torch.long),
    }
    before = model.logits.detach().clone()

    returned_model = train_autoencoder.train([batch], Params(num_epochs=1))

    assert returned_model is model
    assert len(model.calls) == 1
    assert model.calls[0]["input_ids"].tolist() == batch["input_ids"].tolist()
    assert model.calls[0]["decoder_input_ids"].tolist() == [[2, 4, 5], [2, 6, 0]]
    assert not torch.equal(model.logits.detach(), before)
    assert len(scheduler.metrics) == 1
    assert scheduler.metrics[0] == pytest.approx(
        nn.CrossEntropyLoss(ignore_index=train_autoencoder.PAD_IDX)(
            torch.zeros(6, train_autoencoder.VOCAB_SIZE),
            torch.tensor([4, 5, 0, 6, 0, 0]),
        ).item()
    )


def test_train_validates_and_steps_scheduler_with_validation_loss(monkeypatch):
    model = _RecordingAutoencoder()
    scheduler = _RecordingScheduler()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    monkeypatch.setattr(train_autoencoder, "device", torch.device("cpu"))
    monkeypatch.setattr(train_autoencoder, "tqdm", _FastTqdm)
    monkeypatch.setattr(
        train_autoencoder,
        "model_definition",
        lambda _hyperparams: (model, optimizer, scheduler),
    )

    train_batch = {
        "input_ids": torch.tensor([[2, 4, 5, 0]], dtype=torch.long),
        "target_ids": torch.tensor([[2, 4, 5, 0]], dtype=torch.long),
    }
    valid_batch = {
        "input_ids": torch.tensor([[2, 6, 0]], dtype=torch.long),
        "target_ids": torch.tensor([[2, 6, 0]], dtype=torch.long),
    }

    train_autoencoder.train(
        [train_batch],
        Params(num_epochs=1),
        val_dataloader=[valid_batch],
    )

    assert len(model.calls) == 2
    assert model.calls[1]["input_ids"].tolist() == valid_batch["input_ids"].tolist()
    assert model.calls[1]["decoder_input_ids"].tolist() == [[2, 6]]
    assert len(scheduler.metrics) == 1
    assert scheduler.metrics[0] == pytest.approx(
        nn.CrossEntropyLoss(ignore_index=train_autoencoder.PAD_IDX)(
            model.calls[1]["decoder_input_ids"].new_zeros(
                2, train_autoencoder.VOCAB_SIZE, dtype=torch.float32
            ),
            torch.tensor([6, 0]),
        ).item()
    )


def test_main_validates_args_and_starts_autoencoder_training(monkeypatch):
    calls = {}

    def fake_create_dataloader(**kwargs):
        calls.setdefault("dataloader_kwargs", []).append(kwargs)
        return [kwargs["split"]]

    monkeypatch.setattr(sys, "argv", ["train_autoencoder.py", "--task", "solubility"])
    monkeypatch.setattr(train_autoencoder, "create_dataloader", fake_create_dataloader)
    monkeypatch.setattr(
        train_autoencoder,
        "train",
        lambda dataloader, hyperparams, val_dataloader=None: calls.update(
            {
                "dataloader": dataloader,
                "val_dataloader": val_dataloader,
                "hyperparams": hyperparams,
            }
        ),
    )

    train_autoencoder.main()

    assert calls["dataloader_kwargs"] == [
        {
            "task": "solubility",
            "split": train_autoencoder.TRAIN_SPLIT,
            "mode": "autoencoder",
            "batch_size": Params().batch_size,
            "shuffle": Params().shuffle,
            "num_workers": train_autoencoder.num_workers,
        },
        {
            "task": "solubility",
            "split": train_autoencoder.VALID_SPLIT,
            "mode": "autoencoder",
            "batch_size": Params().batch_size,
            "shuffle": False,
            "num_workers": train_autoencoder.num_workers,
        },
    ]
    assert calls["dataloader"] == [train_autoencoder.TRAIN_SPLIT]
    assert calls["val_dataloader"] == [train_autoencoder.VALID_SPLIT]
    assert isinstance(calls["hyperparams"], Params)

    monkeypatch.setattr(sys, "argv", ["train_autoencoder.py", "--model", "CNN"])
    with pytest.raises(ValueError, match="Only --model AE"):
        train_autoencoder.main()

    monkeypatch.setattr(sys, "argv", ["train_autoencoder.py", "--task", "stability"])
    with pytest.raises(ValueError, match="localization' or 'solubility"):
        train_autoencoder.main()
