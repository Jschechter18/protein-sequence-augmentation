from pathlib import Path
import pickle

import pytest
import torch

from utils.sequence_dataset import BOS_IDX, EOS_IDX, PAD_IDX, UNK_IDX, SequenceDataset

from .test_utils.test_helpers import write_csv


def test_getitem_returns_classification_char_tensors(tmp_path: Path) -> None:
    write_csv(tmp_path, {"sequence": ["acdx"], "label": [1]})

    dataset = SequenceDataset(
        task="toy",
        split="train",
        data_dir=tmp_path,
        max_length=6,
        use_cache=False,
    )
    item = dataset[0]

    assert len(dataset) == 1
    assert item["sequence"] == "ACDX"
    assert item["input_ids"].tolist() == [4, 5, 6, UNK_IDX, PAD_IDX, PAD_IDX]
    assert item["length"].item() == 4
    assert item["label"].item() == 1
    assert item["label"].dtype == torch.long


def test_encode_sequence_truncates_pads_and_tracks_effective_length(tmp_path: Path) -> None:
    write_csv(tmp_path, {"sequence": ["ACD"], "label": [0]})
    dataset = SequenceDataset(
        task="toy",
        split="train",
        data_dir=tmp_path,
        max_length=4,
        use_cache=False,
    )

    token_ids, length = dataset._encode_sequence("ACDEFG")

    assert token_ids == [4, 5, 6, 7]
    assert length == 4


def test_default_max_length_preserves_full_sequence(tmp_path: Path) -> None:
    write_csv(tmp_path, {"sequence": ["ACDEFG"], "label": [0]})
    dataset = SequenceDataset(
        task="toy",
        split="train",
        data_dir=tmp_path,
        use_cache=False,
    )

    item = dataset[0]

    assert item["input_ids"].tolist() == [4, 5, 6, 7, 8, 9]
    assert item["length"].item() == 6


def test_autoencoder_encoding_adds_special_tokens_and_clones_target(
    tmp_path: Path,
) -> None:
    write_csv(tmp_path, {"sequence": ["ACDE"], "label": [0]})

    dataset = SequenceDataset(
        task="toy",
        split="train",
        data_dir=tmp_path,
        mode="autoencoder",
        max_length=5,
        use_cache=False,
    )
    item = dataset[0]

    assert item["input_ids"].tolist() == [BOS_IDX, 4, 5, 6, EOS_IDX]
    assert item["target_ids"].tolist() == item["input_ids"].tolist()
    assert item["target_ids"] is not item["input_ids"]
    assert "label" not in item


def test_raw_encoding_returns_truncated_sequence_without_token_ids(tmp_path: Path) -> None:
    write_csv(tmp_path, {"sequence": ["ACDEFG"], "label": [1]})

    dataset = SequenceDataset(
        task="toy",
        split="train",
        data_dir=tmp_path,
        encoding="raw",
        max_length=3,
        use_cache=False,
    )
    item = dataset[0]

    assert item["sequence"] == "ACD"
    assert item["label"].item() == 1
    assert item["label"].dtype == torch.long
    assert item["length"].item() == 3
    assert "input_ids" not in item


def test_cache_key_changes_with_preprocessing_options(tmp_path: Path) -> None:
    write_csv(tmp_path, {"sequence": ["ACD"], "label": [0]})

    base = SequenceDataset(
        task="toy",
        split="train",
        data_dir=tmp_path,
        max_length=5,
        use_cache=False,
    )
    changed = SequenceDataset(
        task="toy",
        split="train",
        data_dir=tmp_path,
        max_length=6,
        use_cache=False,
    )

    assert base._cache_key().endswith(".pkl")
    assert base._cache_key() != changed._cache_key()
    assert "toy_train_classification_char_5_sequence_label_False" in base._cache_key()


def test_cache_file_is_created_and_loaded(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    write_csv(tmp_path, {"sequence": ["ACD"], "label": [1]})
    cache_dir = tmp_path / "cache"

    first = SequenceDataset(
        task="toy",
        split="train",
        data_dir=tmp_path,
        cache_dir=str(cache_dir),
        max_length=4,
        use_cache=True,
    )
    cache_path = cache_dir / first._cache_key()

    assert cache_path.exists()
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    assert payload["metadata"]["num_examples"] == 1
    assert payload["examples"] == first.examples

    write_csv(tmp_path, {"sequence": ["YYYY"], "label": [0]})
    second = SequenceDataset(
        task="toy",
        split="train",
        data_dir=tmp_path,
        cache_dir=str(cache_dir),
        max_length=4,
        use_cache=True,
    )

    assert second.examples == first.examples
    assert "[cache] Loaded 1 examples" in capsys.readouterr().out


def test_combined_csv_is_filtered_by_split(tmp_path: Path) -> None:
    write_csv(
        tmp_path,
        {
            "sequence": ["AAA", "CCC"],
            "label": [1, 0],
            "split": ["train", "test"],
        },
        filename="toy.csv",
    )

    dataset = SequenceDataset(
        task="toy",
        split="test",
        data_dir=tmp_path,
        max_length=None,
        use_cache=False,
    )

    assert len(dataset) == 1
    assert dataset[0]["sequence"] == "CCC"


def test_missing_label_column_raises_value_error(tmp_path: Path) -> None:
    write_csv(tmp_path, {"sequence": ["ACD"], "target": [1]})

    with pytest.raises(ValueError, match="missing required columns"):
        SequenceDataset(
            task="toy",
            split="train",
            data_dir=tmp_path,
            use_cache=False,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "invalid"}, "mode must be"),
        ({"encoding": "invalid"}, "encoding must be"),
        ({"mode": "autoencoder", "encoding": "raw"}, "requires encoding='char'"),
        ({"mode": "autoencoder", "max_length": 1}, "max_length >= 2"),
    ],
)
def test_invalid_modes_and_encodings_raise(
    tmp_path: Path,
    kwargs: dict,
    message: str,
) -> None:
    write_csv(tmp_path, {"sequence": ["ACD"], "label": [0]})

    with pytest.raises(ValueError, match=message):
        SequenceDataset(
            task="toy",
            split="train",
            data_dir=tmp_path,
            use_cache=False,
            **kwargs,
        )
