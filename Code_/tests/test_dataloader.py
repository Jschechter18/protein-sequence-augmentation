from pathlib import Path

import pandas as pd
import pytest
import torch

from utils.dataloader import (
    BOS_IDX,
    EOS_IDX,
    PAD_IDX,
    SequenceDataset,
    create_dataloader,
)


def _write_split_csv(root: Path, task: str = "toy", split: str = "train") -> None:
    """Write a CSV file for a given task and split.

    Parameters
    ----------
    root : Path
        The root directory where the task directory will be created.
    task : str, optional
        The name of the task, by default "toy"
    split : str, optional
        The name of the split, by default "train"
    """
    task_dir = root / task
    task_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "sequence": ["ACD", "ACDEFG"],
            "label": [1, 0],
        }
    ).to_csv(task_dir / f"{split}.csv", index=False)


def test_sequence_dataset_loads_split_csv(tmp_path: Path) -> None:
    _write_split_csv(tmp_path)

    dataset = SequenceDataset(
        task="toy",
        split="train",
        data_dir=tmp_path,
        max_length=5,
        use_cache=False,
    )

    assert len(dataset) == 2
    item = dataset[0]
    assert item["input_ids"].tolist() == [4, 5, 6, PAD_IDX, PAD_IDX]
    assert item["length"].item() == 3
    assert item["label"].dtype == torch.long


def test_create_dataloader_batches_examples(tmp_path: Path) -> None:
    _write_split_csv(tmp_path)

    loader = create_dataloader(
        task="toy",
        split="train",
        data_dir=tmp_path,
        batch_size=2,
        max_length=5,
        use_cache=False,
    )
    batch = next(iter(loader))

    assert batch["input_ids"].shape == (2, 5)
    assert batch["label"].tolist() == [1, 0]
    assert batch["sequence"] == ["ACD", "ACDEFG"]


def test_char_batches_pad_when_max_length_is_none(tmp_path: Path) -> None:
    _write_split_csv(tmp_path)

    loader = create_dataloader(
        task="toy",
        split="train",
        data_dir=tmp_path,
        batch_size=2,
        max_length=None,
        use_cache=False,
    )
    batch = next(iter(loader))

    assert batch["input_ids"].shape == (2, 6)
    assert batch["input_ids"][0].tolist() == [4, 5, 6, PAD_IDX, PAD_IDX, PAD_IDX]
    assert batch["length"].tolist() == [3, 6]


def test_create_dataloader_defaults_to_dynamic_batch_padding(tmp_path: Path) -> None:
    _write_split_csv(tmp_path)

    loader = create_dataloader(
        task="toy",
        split="train",
        data_dir=tmp_path,
        batch_size=2,
        use_cache=False,
    )
    batch = next(iter(loader))

    assert batch["input_ids"].shape == (2, 6)
    assert batch["input_ids"][0].tolist() == [4, 5, 6, PAD_IDX, PAD_IDX, PAD_IDX]
    assert batch["length"].tolist() == [3, 6]


def test_autoencoder_adds_special_tokens(tmp_path: Path) -> None:
    _write_split_csv(tmp_path)

    dataset = SequenceDataset(
        task="toy",
        split="train",
        data_dir=tmp_path,
        mode="autoencoder",
        max_length=6,
        use_cache=False,
    )
    item = dataset[0]

    assert item["input_ids"].tolist() == [BOS_IDX, 4, 5, 6, EOS_IDX, PAD_IDX]
    assert item["target_ids"].tolist() == item["input_ids"].tolist()


def test_raw_encoding_batches_sequences(tmp_path: Path) -> None:
    _write_split_csv(tmp_path)

    loader = create_dataloader(
        task="toy",
        split="train",
        data_dir=tmp_path,
        encoding="raw",
        max_length=None,
        batch_size=2,
        use_cache=False,
    )
    batch = next(iter(loader))

    assert batch["sequence"] == ["ACD", "ACDEFG"]
    assert batch["length"].tolist() == [3, 6]


def test_autoencoder_rejects_too_short_max_length(tmp_path: Path) -> None:
    _write_split_csv(tmp_path)

    with pytest.raises(ValueError, match="max_length >= 2"):
        SequenceDataset(
            task="toy",
            split="train",
            data_dir=tmp_path,
            mode="autoencoder",
            max_length=1,
            use_cache=False,
        )


def test_dataloader_handles_esm2_raw_classification_case(tmp_path: Path) -> None:
    _write_split_csv(tmp_path)

    loader = create_dataloader(
        task="toy",
        split="train",
        data_dir=tmp_path,
        mode="classification",
        encoding="raw",
        max_length=None,
        batch_size=2,
        shuffle=True,
        use_cache=False,
    )
    batch = next(iter(loader))

    assert set(batch) == {"sequence", "label", "length"}
    assert sorted(batch["sequence"]) == ["ACD", "ACDEFG"]
    assert sorted(batch["label"].tolist()) == [0, 1]
    assert sorted(batch["length"].tolist()) == [3, 6]


def test_dataloader_handles_1d_cnn_classification_case(tmp_path: Path) -> None:
    _write_split_csv(tmp_path)

    loader = create_dataloader(
        task="toy",
        split="train",
        data_dir=tmp_path,
        mode="classification",
        encoding="char",
        max_length=8,
        batch_size=2,
        shuffle=False,
        use_cache=False,
    )
    batch = next(iter(loader))

    assert set(batch) == {"input_ids", "label", "length", "sequence"}
    assert batch["input_ids"].shape == (2, 8)
    assert batch["input_ids"].dtype == torch.long
    assert batch["input_ids"].tolist() == [
        [4, 5, 6, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX],
        [4, 5, 6, 7, 8, 9, PAD_IDX, PAD_IDX],
    ]
    assert batch["label"].tolist() == [1, 0]
    assert batch["length"].tolist() == [3, 6]
    assert batch["sequence"] == ["ACD", "ACDEFG"]


def test_dataloader_handles_autoencoder_case(tmp_path: Path) -> None:
    _write_split_csv(tmp_path)

    loader = create_dataloader(
        task="toy",
        split="train",
        data_dir=tmp_path,
        mode="autoencoder",
        encoding="char",
        max_length=8,
        batch_size=2,
        shuffle=False,
        use_cache=False,
    )
    batch = next(iter(loader))

    expected = [
        [BOS_IDX, 4, 5, 6, EOS_IDX, PAD_IDX, PAD_IDX, PAD_IDX],
        [BOS_IDX, 4, 5, 6, 7, 8, 9, EOS_IDX],
    ]
    assert set(batch) == {"input_ids", "target_ids", "length", "sequence"}
    assert batch["input_ids"].shape == (2, 8)
    assert batch["input_ids"].dtype == torch.long
    assert batch["input_ids"].tolist() == expected
    assert batch["target_ids"].tolist() == expected
    assert batch["target_ids"].data_ptr() != batch["input_ids"].data_ptr()
    assert batch["length"].tolist() == [5, 8]
    assert batch["sequence"] == ["ACD", "ACDEFG"]
