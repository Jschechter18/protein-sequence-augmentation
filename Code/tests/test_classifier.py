from types import SimpleNamespace

import pytest
import torch
from torch import nn

import Code.src.models.classifier as classifier_module
from Code.src.models.autoencoder import ProteinSequenceAutoencoder
from Code.src.models.classifier import (
    CombinedAutoencoderESM2Encoder,
    LinearHead,
    MLPHead,
    ProteinSequenceClassifier,
    RandomAutoencoderEncoder,
    TrainedAutoencoderEncoder,
)


class StubEncoder(nn.Module):
    def __init__(self, output_dim: int, batch_size: int):
        super().__init__()
        self.output_dim = output_dim
        self.batch_size = batch_size

    def forward(self, *args):
        return torch.randn(self.batch_size, self.output_dim)


def autoencoder_kwargs():
    return {
        "embedding_dim": 4,
        "cnn_out_channels": 4,
        "hidden_dim": 8,
        "latent_dim": 3,
        "num_layers": 1,
        "kernel_size": 3,
        "device": "cpu",
    }


def test_random_autoencoder_is_frozen_and_stays_in_eval_mode():
    encoder = RandomAutoencoderEncoder(**autoencoder_kwargs())

    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    encoder.train()
    assert not encoder.autoencoder.training


def test_frozen_trained_autoencoder_stays_in_eval_mode(tmp_path):
    kwargs = autoencoder_kwargs()
    model = ProteinSequenceAutoencoder(**{key: value for key, value in kwargs.items() if key != "device"})
    checkpoint = tmp_path / "autoencoder.pt"
    torch.save(model.state_dict(), checkpoint)
    encoder = TrainedAutoencoderEncoder(checkpoint_path=str(checkpoint), freeze=True, **kwargs)

    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    encoder.train()
    assert not encoder.autoencoder.training


def test_frozen_esm_stays_in_eval_mode(monkeypatch):
    model = nn.Sequential(nn.Linear(2, 2), nn.Dropout())
    alphabet = object()
    monkeypatch.setattr(
        classifier_module.esm,
        "pretrained",
        SimpleNamespace(esm2_t6_8M_UR50D=lambda: (model, alphabet)),
        raising=False,
    )
    classifier = ProteinSequenceClassifier(embedding_type="esm2")
    encoder = classifier.embedded_representation

    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    classifier.train()
    assert not encoder.model.training


@pytest.mark.parametrize("head_class", [LinearHead, MLPHead])
def test_sequence_level_heads_accept_two_dimensional_embeddings(head_class):
    head = head_class(embedding_dim=5, num_classes=2)
    assert head(torch.randn(4, 5)).shape == (4, 2)


def test_classifier_rejects_cnn_head_for_sequence_level_encoders():
    with pytest.raises(ValueError, match="residue-level"):
        ProteinSequenceClassifier(
            embedding_type="random_autoencoder",
            head_type="cnn",
            autoencoder_embedding_dim=4,
            autoencoder_cnn_channels=4,
            autoencoder_hidden_dim=8,
            autoencoder_latent_dim=3,
        )


def test_combined_encoder_uses_declared_dimensions():
    encoder = CombinedAutoencoderESM2Encoder(
        StubEncoder(output_dim=3, batch_size=2),
        StubEncoder(output_dim=5, batch_size=2),
    )

    output = encoder(torch.ones(2, 4), torch.tensor([4, 4]), ["AA", "CC"])

    assert encoder.output_dim == 8
    assert output.shape == (2, 8)


def test_combined_encoder_rejects_mismatched_batch_sizes():
    encoder = CombinedAutoencoderESM2Encoder(
        StubEncoder(output_dim=3, batch_size=2),
        StubEncoder(output_dim=5, batch_size=1),
    )

    with pytest.raises(ValueError, match="same batch size"):
        encoder(torch.ones(2, 4), torch.tensor([4, 4]), ["AA", "CC"])
