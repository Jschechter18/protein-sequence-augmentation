import sys
from pathlib import Path

from testing import test_autoencoder


def test_default_checkpoint_path_uses_training_quartile_naming() -> None:
    path = test_autoencoder.default_checkpoint_path("ae", "solubility", "6", "ml")

    assert path == (
        test_autoencoder.PROJECT_ROOT
        / "checkpoints"
        / "autoencoder"
        / "solubility"
        / "v6"
        / "model_ae_medium_long_solubility.pt"
    )


def test_main_passes_quartile_loader_options_to_test_dataloader(monkeypatch, tmp_path: Path) -> None:
    calls = []
    boundaries = [0, 1, 2, 3, 4]

    class DummyLoader:
        dataset = object()

    def fake_create_dataloader(**kwargs):
        calls.append(kwargs)
        return DummyLoader()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_autoencoder.py",
            "--task",
            "solubility",
            "--version",
            "6",
            "--length_quartile",
            "ml",
            "--cumulative_quartiles",
            "True",
            "--output_path",
            str(tmp_path / "outputs.csv"),
        ],
    )
    monkeypatch.setattr(test_autoencoder, "create_dataloader", fake_create_dataloader)
    monkeypatch.setattr(test_autoencoder, "compute_train_length_boundaries", lambda _dataset: boundaries)
    monkeypatch.setattr(test_autoencoder, "model_definition", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(test_autoencoder, "test", lambda **_kwargs: None)

    test_autoencoder.main()

    assert len(calls) == 2
    assert calls[0]["split"] == "train"
    assert calls[0]["mode"] == "autoencoder"
    assert calls[1]["split"] == "test"
    assert calls[1]["loader_type"] == "quartile"
    assert calls[1]["quartile_name"] == "ml"
    assert calls[1]["cumulative"] is True
    assert calls[1]["length_boundaries"] == boundaries
