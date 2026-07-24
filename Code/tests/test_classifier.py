from types import SimpleNamespace

import pytest
import torch
from torch import nn

import Code.src.models.classifier as classifier_module
from Code.src.models.autoencoder import ProteinSequenceAutoencoder
from Code.src.models.classifier import (
    CombinedAutoencoderESM2Encoder,
    ESM2Embedding,
    LinearHead,
    MLPHead,
    ProteinSequenceClassifier,
    RandomAutoencoderEncoder,
    TrainedAutoencoderEncoder,
)


class FakeESMAlphabet:
    padding_idx = 0
    cls_idx = 1
    eos_idx = 2

    def __init__(self):
        self.converted_sequences = []

    def get_batch_converter(self):
        def convert(items):
            self.converted_sequences = [sequence for _, sequence in items]
            max_length = max(len(sequence) for sequence in self.converted_sequences)
            tokens = torch.full((len(items), max_length + 2), self.padding_idx)
            for index, sequence in enumerate(self.converted_sequences):
                tokens[index, 0] = self.cls_idx
                tokens[index, 1 : len(sequence) + 1] = 4
                tokens[index, len(sequence) + 1] = self.eos_idx
            return None, None, tokens

        return convert


class FakeESMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, batch_tokens, repr_layers, return_contacts):
        del return_contacts
        embeddings = batch_tokens.to(torch.float32).unsqueeze(-1).expand(-1, -1, 320)
        return {"representations": {repr_layers[0]: embeddings}}


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


def classifier_autoencoder_kwargs():
    return {
        "autoencoder_embedding_dim": 4,
        "autoencoder_cnn_channels": 4,
        "autoencoder_hidden_dim": 8,
        "autoencoder_latent_dim": 3,
        "autoencoder_num_layers": 1,
        "autoencoder_kernel_size": 3,
    }


@pytest.fixture
def autoencoder_checkpoint(tmp_path):
    kwargs = autoencoder_kwargs()
    model = ProteinSequenceAutoencoder(
        **{key: value for key, value in kwargs.items() if key != "device"}
    )
    checkpoint = tmp_path / "autoencoder.pt"
    torch.save(model.state_dict(), checkpoint)
    return checkpoint


@pytest.fixture
def fake_esm(monkeypatch):
    alphabet = FakeESMAlphabet()
    monkeypatch.setattr(
        classifier_module,
        "esm",
        SimpleNamespace(
            pretrained=SimpleNamespace(
                esm2_t6_8M_UR50D=lambda: (FakeESMModel(), alphabet)
            )
        ),
    )
    return alphabet


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


def test_mlp_head_applies_requested_dropout() -> None:
    head = MLPHead(embedding_dim=16, dropout=0.3)

    dropout_layers = [
        layer for layer in head.modules() if isinstance(layer, torch.nn.Dropout)
    ]
    assert len(dropout_layers) == 1
    assert dropout_layers[0].p == pytest.approx(0.3)


@pytest.mark.parametrize("dropout", [-0.1, 1.0])
def test_mlp_head_rejects_invalid_dropout(dropout: float) -> None:
    with pytest.raises(ValueError, match="dropout"):
        MLPHead(embedding_dim=16, dropout=dropout)


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


@pytest.mark.parametrize(
    ("embedding_type", "expected_output_dim"),
    [
        ("random_autoencoder", 3),
        ("trained_autoencoder", 3),
        ("esm2", 320),
        ("trained_autoencoder+esm2", 323),
    ],
)
def test_canonical_representations_dispatch_and_forward(
    embedding_type,
    expected_output_dim,
    autoencoder_checkpoint,
    fake_esm,
):
    del fake_esm
    classifier = ProteinSequenceClassifier(
        embedding_type=embedding_type,
        head_type="mlp",
        autoencoder_checkpoint=str(autoencoder_checkpoint),
        **classifier_autoencoder_kwargs(),
    )
    batch = {
        "input_ids": torch.tensor([[4, 5, 6], [7, 8, 0]]),
        "length": torch.tensor([3, 2]),
        "sequence": ["ACD", "EF"],
    }

    logits = classifier(batch)

    assert classifier.embedding_type == embedding_type
    assert classifier.head_type == "mlp"
    assert classifier.encoder_output_dim == expected_output_dim
    assert classifier.output_dim == expected_output_dim
    assert logits.shape == (2, 2)
    assert all(
        not parameter.requires_grad
        for parameter in classifier.embedded_representation.parameters()
    )
    assert all(parameter.requires_grad for parameter in classifier.head.parameters())


def test_legacy_combined_alias_is_normalized_immediately(
    autoencoder_checkpoint,
    fake_esm,
):
    del fake_esm
    classifier = ProteinSequenceClassifier(
        embedding_type="autoencoder+esm2",
        autoencoder_checkpoint=str(autoencoder_checkpoint),
        **classifier_autoencoder_kwargs(),
    )

    assert classifier.embedding_type == "trained_autoencoder+esm2"
    assert classifier.encoder_output_dim == 323


def test_combined_frozen_encoders_stay_in_eval_mode_after_classifier_train(
    autoencoder_checkpoint,
    fake_esm,
):
    del fake_esm
    classifier = ProteinSequenceClassifier(
        embedding_type="trained_autoencoder+esm2",
        autoencoder_checkpoint=str(autoencoder_checkpoint),
        **classifier_autoencoder_kwargs(),
    )

    classifier.train()

    combined = classifier.embedded_representation
    assert not combined.autoencoder_encoder.autoencoder.training
    assert not combined.esm_encoder.model.training


def test_incompatible_esm_package_raises_clear_import_error(monkeypatch):
    monkeypatch.setattr(classifier_module, "esm", SimpleNamespace())

    with pytest.raises(ImportError, match=r"fair-esm==2\.0\.0"):
        ESM2Embedding()


def test_esm_truncates_sequences_before_batch_conversion(fake_esm):
    encoder = ESM2Embedding(esm_max_sequence_length=3)

    output = encoder(["ACDEFG", "HI"])

    assert fake_esm.converted_sequences == ["ACD", "HI"]
    assert output.shape == (2, 320)
    # Fake residue embeddings equal 4, while CLS/EOS/padding equal 1/2/0.
    # An exact value of 4 verifies all three special-token types are excluded.
    assert torch.equal(output, torch.full((2, 320), 4.0))


def test_autoencoder_special_tokens_are_added_once_with_correct_lengths():
    classifier = ProteinSequenceClassifier(
        embedding_type="random_autoencoder",
        **classifier_autoencoder_kwargs(),
    )
    input_ids = torch.tensor([[4, 5, 6], [7, 8, 0]])

    framed_ids, framed_lengths = classifier._add_autoencoder_special_tokens(
        input_ids,
        torch.tensor([3, 2]),
    )

    assert framed_ids.tolist() == [[2, 4, 5, 6, 3], [2, 7, 8, 3, 0]]
    assert framed_lengths.tolist() == [5, 4]


@pytest.mark.parametrize(
    ("input_ids", "lengths", "error_type", "message"),
    [
        (torch.tensor([4, 5]), torch.tensor([2]), ValueError, "input_ids must have shape"),
        (
            torch.tensor([[4, 5]]),
            torch.tensor([[2]]),
            ValueError,
            "lengths must have shape",
        ),
        (
            torch.tensor([[4, 5], [6, 0]]),
            torch.tensor([2]),
            ValueError,
            "one value for each row",
        ),
        (
            torch.tensor([[4, 5]]),
            torch.tensor([-1]),
            ValueError,
            "non-negative",
        ),
        (
            torch.tensor([[4, 5]]),
            torch.tensor([3]),
            ValueError,
            "cannot exceed",
        ),
        (
            torch.tensor([[4.0, 5.0]]),
            torch.tensor([2]),
            TypeError,
            "input_ids must use an integer dtype",
        ),
        (
            torch.tensor([[4, 5]]),
            torch.tensor([2.0]),
            TypeError,
            "lengths must use an integer dtype",
        ),
    ],
)
def test_autoencoder_special_token_validation(
    input_ids,
    lengths,
    error_type,
    message,
):
    classifier = ProteinSequenceClassifier(
        embedding_type="random_autoencoder",
        **classifier_autoencoder_kwargs(),
    )

    with pytest.raises(error_type, match=message):
        classifier._add_autoencoder_special_tokens(input_ids, lengths)
