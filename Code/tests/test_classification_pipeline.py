import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from Code.src.training.classification_pipeline import (
    METRIC_NAMES,
    ProteinClassificationTrainingPipeline,
)


class BatchLogitModel(nn.Module):
    """Return supplied logits while retaining a trainable parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(2))

    def forward(self, batch):
        return batch["logits"] + self.offset


class TinyClassifier(nn.Module):
    embedding_type = "tiny"

    def __init__(self) -> None:
        super().__init__()
        self.embedded_representation = nn.Linear(2, 3)
        self.head = nn.Linear(3, 2)
        for parameter in self.embedded_representation.parameters():
            parameter.requires_grad = False

    def forward(self, batch):
        return self.head(torch.tanh(self.embedded_representation(batch["features"])))


class ScalarClassifier(nn.Module):
    embedding_type = "tiny"

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(1, 2, bias=False)

    def forward(self, batch):
        return self.head(batch["features"])


def make_logit_loader(labels=(0, 1, 1), batch_size=2):
    logits = ([3.0, 0.0], [0.0, 2.0], [2.0, 1.0])[: len(labels)]
    records = [
        {
            "logits": torch.tensor(logit),
            "label": label,
            "sequence": sequence,
            "length": len(sequence),
            "sample_id": f"sample-{index}",
        }
        for index, (logit, label, sequence) in enumerate(
            zip(logits, labels, ("AA", "CCC", "DDDD"))
        )
    ]
    return DataLoader(records, batch_size=batch_size, shuffle=False)


def complete_metrics(loss: float) -> dict[str, float]:
    metrics = {name: 0.0 for name in METRIC_NAMES}
    metrics.update({"loss": loss, "num_samples": 2})
    return metrics


def test_validation_uses_sample_weighted_loss_and_whole_split_metrics(tmp_path):
    pipeline = ProteinClassificationTrainingPipeline(BatchLogitModel(), tmp_path)
    loader = make_logit_loader(batch_size=2)

    metrics = pipeline.validate(loader)

    logits = torch.tensor([[3.0, 0.0], [0.0, 2.0], [2.0, 1.0]])
    labels = torch.tensor([0, 1, 1])
    expected_loss = F.cross_entropy(logits, labels, reduction="sum").item() / 3
    assert metrics["loss"] == pytest.approx(expected_loss)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["positive_precision"] == metrics["precision"]
    assert metrics["positive_recall"] == metrics["recall"]
    assert metrics["positive_f1"] == metrics["f1"]
    assert metrics["macro_f1"] == pytest.approx(2 / 3)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["num_samples"] == 3


def test_one_class_scores_are_nan_in_memory_and_null_in_json(tmp_path):
    pipeline = ProteinClassificationTrainingPipeline(BatchLogitModel(), tmp_path)
    loader = make_logit_loader(labels=(0, 0), batch_size=2)

    metrics = pipeline.evaluate_test(loader)

    assert math.isnan(metrics["auroc"])
    assert math.isnan(metrics["auprc"])
    persisted = json.loads((tmp_path / "metrics.json").read_text())
    assert persisted["auroc"] is None
    assert persisted["auprc"] is None


def test_fractional_classification_labels_are_rejected(tmp_path):
    records = [
        {"logits": torch.tensor([1.0, 0.0]), "label": 0.5},
        {"logits": torch.tensor([0.0, 1.0]), "label": 1.0},
    ]
    loader = DataLoader(records, batch_size=2)
    pipeline = ProteinClassificationTrainingPipeline(BatchLogitModel(), tmp_path)

    with pytest.raises(TypeError, match="integer tensor dtype"):
        pipeline.validate(loader)


def test_optimizer_contains_only_trainable_parameters(tmp_path):
    model = TinyClassifier()
    pipeline = ProteinClassificationTrainingPipeline(
        model,
        tmp_path,
        optimizer_name="adamw",
        weight_decay=0.01,
        max_grad_norm=None,
    )

    optimized_ids = {
        id(parameter)
        for group in pipeline.optimizer.param_groups
        for parameter in group["params"]
    }
    expected_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    frozen_ids = {
        id(parameter)
        for parameter in model.embedded_representation.parameters()
    }
    assert optimized_ids == expected_ids
    assert optimized_ids.isdisjoint(frozen_ids)
    assert all(group["weight_decay"] == pytest.approx(0.01) for group in pipeline.optimizer.param_groups)


def test_one_epoch_training_writes_complete_recoverable_artifacts(tmp_path):
    records = [
        {"features": torch.tensor([0.0, 0.0]), "label": 0},
        {"features": torch.tensor([0.0, 1.0]), "label": 0},
        {"features": torch.tensor([1.0, 0.0]), "label": 1},
        {"features": torch.tensor([1.0, 1.0]), "label": 1},
    ]
    loader = DataLoader(records, batch_size=3, shuffle=False)
    pipeline = ProteinClassificationTrainingPipeline(
        TinyClassifier(),
        tmp_path,
        learning_rate=0.05,
        run_config={"seed": 42, "representation": "tiny"},
    )

    history = pipeline.fit(loader, loader, epochs=1, early_stopping_patience=1)

    assert len(history) == 1
    assert (tmp_path / "best_model.pt").is_file()
    assert (tmp_path / "last_model.pt").is_file()
    assert (tmp_path / "history.csv").is_file()
    assert (tmp_path / "history.json").is_file()
    history_frame = pd.read_csv(tmp_path / "history.csv")
    assert len(history_frame) == 1
    assert {"train_auroc", "val_auprc", "val_balanced_accuracy"}.issubset(
        history_frame.columns
    )
    history_json = json.loads((tmp_path / "history.json").read_text())
    assert len(history_json["history"]) == 1
    assert history_json["summary"]["best_epoch"] == 1

    checkpoint = torch.load(
        tmp_path / "last_model.pt", map_location="cpu", weights_only=False
    )
    assert {
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "best_val_loss",
        "config",
    }.issubset(checkpoint)
    assert checkpoint["epoch"] == 1
    assert checkpoint["config"]["seed"] == 42


def test_fit_restores_best_weights_and_bundles_last_state(tmp_path):
    pipeline = ProteinClassificationTrainingPipeline(ScalarClassifier(), tmp_path)
    validation_losses = iter((0.1, 0.5))

    def fake_train_epoch(_loader):
        with torch.no_grad():
            pipeline.model.head.weight.fill_(pipeline.current_epoch + 1)
        return complete_metrics(0.2)

    def fake_validate(_loader):
        return complete_metrics(next(validation_losses))

    pipeline.train_epoch = fake_train_epoch
    pipeline.validate = fake_validate
    pipeline.fit([], [], epochs=2, early_stopping_patience=2)

    assert torch.all(pipeline.model.head.weight == 1)
    assert pipeline.best_epoch == 1
    assert len(pipeline.history) == 2

    loaded = ProteinClassificationTrainingPipeline(ScalarClassifier(), tmp_path)
    loaded.load_checkpoint(load_optimizer=False)
    assert torch.all(loaded.model.head.weight == 1)
    assert loaded.current_epoch == 1

    loaded.load_checkpoint(tmp_path / "last_model.pt", load_optimizer=True)
    assert torch.all(loaded.model.head.weight == 2)
    assert loaded.current_epoch == 2
    assert loaded.best_epoch == 1
    assert len(loaded.history) == 2

    def resumed_train_epoch(_loader):
        with torch.no_grad():
            loaded.model.head.weight.fill_(loaded.current_epoch + 1)
        return complete_metrics(0.1)

    loaded.train_epoch = resumed_train_epoch
    loaded.validate = lambda _loader: complete_metrics(0.05)
    resumed_history = loaded.fit(
        [],
        [],
        epochs=3,
        early_stopping_patience=2,
        resume_from=tmp_path / "last_model.pt",
    )
    assert len(resumed_history) == 3
    assert loaded.current_epoch == 3
    assert loaded.best_epoch == 3
    assert torch.all(loaded.model.head.weight == 3)


def test_test_predictions_preserve_ids_sequences_lengths_and_probabilities(tmp_path):
    pipeline = ProteinClassificationTrainingPipeline(BatchLogitModel(), tmp_path)

    metrics = pipeline.evaluate_test(make_logit_loader(batch_size=2))

    predictions = pd.read_csv(tmp_path / "test_predictions.csv")
    expected_probabilities = torch.softmax(
        torch.tensor([[3.0, 0.0], [0.0, 2.0], [2.0, 1.0]]), dim=1
    ).numpy()
    assert metrics["num_samples"] == 3
    assert predictions["sample_id"].tolist() == ["sample-0", "sample-1", "sample-2"]
    assert predictions["sequence"].tolist() == ["AA", "CCC", "DDDD"]
    assert predictions["sequence_length"].tolist() == [2, 3, 4]
    np.testing.assert_allclose(
        predictions[["probability_class_0", "probability_class_1"]].to_numpy(),
        expected_probabilities,
    )


def test_resumed_training_matches_uninterrupted_training(tmp_path):
    records = [
        {"features": torch.tensor([float(index % 2)]), "label": index % 2}
        for index in range(12)
    ]

    def initialize(run_dir: Path):
        random.seed(91)
        np.random.seed(91)
        torch.manual_seed(91)
        generator = torch.Generator().manual_seed(177)
        loader = DataLoader(
            records, batch_size=3, shuffle=True, generator=generator
        )
        validation_loader = DataLoader(records, batch_size=4, shuffle=False)
        pipeline = ProteinClassificationTrainingPipeline(
            ScalarClassifier(), run_dir, learning_rate=0.02
        )
        return pipeline, loader, validation_loader

    uninterrupted, train_loader, validation_loader = initialize(
        tmp_path / "uninterrupted"
    )
    uninterrupted.fit(
        train_loader, validation_loader, epochs=3, early_stopping_patience=3
    )
    uninterrupted_last = torch.load(
        tmp_path / "uninterrupted" / "last_model.pt",
        map_location="cpu",
        weights_only=False,
    )

    first_part, train_loader, validation_loader = initialize(tmp_path / "resumed")
    first_part.fit(
        train_loader, validation_loader, epochs=1, early_stopping_patience=3
    )
    resumed, train_loader, validation_loader = initialize(tmp_path / "resumed")
    resumed.fit(
        train_loader,
        validation_loader,
        epochs=3,
        early_stopping_patience=3,
        resume_from=tmp_path / "resumed" / "last_model.pt",
    )
    resumed_last = torch.load(
        tmp_path / "resumed" / "last_model.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert uninterrupted_last["history"] == resumed_last["history"]
    for name, tensor in uninterrupted_last["model_state_dict"].items():
        assert torch.equal(tensor, resumed_last["model_state_dict"][name])


def test_checkpoint_rejects_mismatched_resume_fingerprint(tmp_path):
    original = ProteinClassificationTrainingPipeline(
        ScalarClassifier(), tmp_path, run_config={"resume_fingerprint": "original"}
    )
    original.save_checkpoint(tmp_path / "checkpoint.pt", epoch=0)
    changed = ProteinClassificationTrainingPipeline(
        ScalarClassifier(), tmp_path, run_config={"resume_fingerprint": "changed"}
    )

    with pytest.raises(ValueError, match="does not match"):
        changed.load_checkpoint(tmp_path / "checkpoint.pt")
