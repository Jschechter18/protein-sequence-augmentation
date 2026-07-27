from pathlib import Path

import torch

from utils.dataloader import (
    BOS_IDX,
    EOS_IDX,
    LengthAwareBatchSampler,
    PAD_IDX,
    SequenceDataset,
    create_dataloader,
)
from .test_utils.test_helpers import write_csv, write_split_csv


def test_sequence_dataset_loads_split_csv(tmp_path: Path) -> None:
    write_split_csv(tmp_path)

    dataset = SequenceDataset(
        task="toy",
        split="train",
        data_dir=tmp_path,
        use_cache=False,
    )

    assert len(dataset) == 2
    item = dataset[0]
    assert item["input_ids"].tolist() == [4, 5, 6]
    assert item["length"].item() == 3
    assert item["label"].dtype == torch.long


def test_create_dataloader_batches_examples(tmp_path: Path) -> None:
    write_split_csv(tmp_path)

    loader = create_dataloader(
        task="toy",
        split="train",
        data_dir=tmp_path,
        batch_size=2,
        use_cache=False,
    )
    batch = next(iter(loader))

    assert batch["input_ids"].shape == (2, 6)
    assert batch["label"].tolist() == [1, 0]
    assert batch["sequence"] == ["ACD", "ACDEFG"]
    assert batch["sample_id"] == [0, 1]


def test_char_batches_pad_to_longest_sequence_in_batch(tmp_path: Path) -> None:
    write_split_csv(tmp_path)

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


def test_create_dataloader_defaults_to_dynamic_batch_padding(tmp_path: Path) -> None:
    write_split_csv(tmp_path)

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
    write_split_csv(tmp_path)

    dataset = SequenceDataset(
        task="toy",
        split="train",
        data_dir=tmp_path,
        mode="autoencoder",
        use_cache=False,
    )
    item = dataset[0]

    assert item["input_ids"].tolist() == [BOS_IDX, 4, 5, 6, EOS_IDX]
    assert item["target_ids"].tolist() == item["input_ids"].tolist()


def test_raw_encoding_batches_sequences(tmp_path: Path) -> None:
    write_split_csv(tmp_path)

    loader = create_dataloader(
        task="toy",
        split="train",
        data_dir=tmp_path,
        encoding="raw",
        batch_size=2,
        use_cache=False,
    )
    batch = next(iter(loader))

    assert batch["sequence"] == ["ACD", "ACDEFG"]
    assert batch["length"].tolist() == [3, 6]


def test_dataloader_handles_esm2_raw_classification_case(tmp_path: Path) -> None:
    write_split_csv(tmp_path)

    loader = create_dataloader(
        task="toy",
        split="train",
        data_dir=tmp_path,
        mode="classification",
        encoding="raw",
        batch_size=2,
        shuffle=True,
        use_cache=False,
    )
    batch = next(iter(loader))

    assert set(batch) == {"sequence", "label", "length", "sample_id"}
    assert sorted(batch["sequence"]) == ["ACD", "ACDEFG"]
    assert sorted(batch["label"].tolist()) == [0, 1]
    assert sorted(batch["length"].tolist()) == [3, 6]


def test_dataloader_handles_1d_cnn_classification_case(tmp_path: Path) -> None:
    write_split_csv(tmp_path)

    loader = create_dataloader(
        task="toy",
        split="train",
        data_dir=tmp_path,
        mode="classification",
        encoding="char",
        batch_size=2,
        shuffle=False,
        use_cache=False,
    )
    batch = next(iter(loader))

    assert set(batch) == {"input_ids", "label", "length", "sequence", "sample_id"}
    assert batch["input_ids"].shape == (2, 6)
    assert batch["input_ids"].dtype == torch.long
    assert batch["input_ids"].tolist() == [
        [4, 5, 6, PAD_IDX, PAD_IDX, PAD_IDX],
        [4, 5, 6, 7, 8, 9],
    ]
    assert batch["label"].tolist() == [1, 0]
    assert batch["length"].tolist() == [3, 6]
    assert batch["sequence"] == ["ACD", "ACDEFG"]
    assert batch["sample_id"] == [0, 1]


def test_dataloader_handles_autoencoder_case(tmp_path: Path) -> None:
    write_split_csv(tmp_path)

    loader = create_dataloader(
        task="toy",
        split="train",
        data_dir=tmp_path,
        mode="autoencoder",
        encoding="char",
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


def _collect_lengths(loader) -> list[int]:
    lengths = []
    for batch in loader:
        lengths.extend(batch["length"].tolist())
    return lengths


def _collect_sample_ids(loader) -> list[int]:
    sample_ids = []
    for batch in loader:
        sample_ids.extend(batch["sample_id"])
    return sample_ids


def test_generator_makes_shuffle_order_reproducible(tmp_path: Path) -> None:
    write_csv(
        tmp_path,
        {
            "idx": [10, 20, 30, 40],
            "sequence": ["A", "AC", "ACD", "ACDE"],
            "label": [0, 1, 0, 1],
        },
    )
    first_generator = torch.Generator().manual_seed(123)
    second_generator = torch.Generator().manual_seed(123)

    first = create_dataloader(
        task="toy",
        split="train",
        data_dir=tmp_path,
        batch_size=2,
        shuffle=True,
        use_cache=False,
        generator=first_generator,
    )
    second = create_dataloader(
        task="toy",
        split="train",
        data_dir=tmp_path,
        batch_size=2,
        shuffle=True,
        use_cache=False,
        generator=second_generator,
    )

    assert _collect_sample_ids(first) == _collect_sample_ids(second)


def test_length_aware_batching_is_reproducible_and_covers_dataset(
    tmp_path: Path,
) -> None:
    sequences = ["A" * length for length in range(1, 41)]
    write_csv(
        tmp_path,
        {
            "idx": list(range(len(sequences))),
            "sequence": sequences,
            "label": [0] * len(sequences),
        },
    )

    def make_loader(seed: int):
        return create_dataloader(
            task="toy",
            split="train",
            data_dir=tmp_path,
            batch_size=4,
            shuffle=True,
            use_cache=False,
            generator=torch.Generator().manual_seed(seed),
            length_aware_batching=True,
            length_pool_size_multiplier=5,
        )

    first = make_loader(123)
    second = make_loader(123)

    assert isinstance(first.batch_sampler, LengthAwareBatchSampler)
    assert _collect_sample_ids(first) == _collect_sample_ids(second)
    assert sorted(_collect_sample_ids(make_loader(123))) == list(range(40))


def test_length_aware_batching_reduces_within_batch_length_spread(
    tmp_path: Path,
) -> None:
    sequences = ["A" * length for length in range(1, 65)]
    write_csv(
        tmp_path,
        {
            "idx": list(range(len(sequences))),
            "sequence": sequences,
            "label": [0] * len(sequences),
        },
    )
    loader = create_dataloader(
        task="toy",
        split="train",
        data_dir=tmp_path,
        batch_size=4,
        use_cache=False,
        generator=torch.Generator().manual_seed(123),
        length_aware_batching=True,
        length_pool_size_multiplier=16,
    )

    batch_spreads = [
        max(batch["length"].tolist()) - min(batch["length"].tolist())
        for batch in loader
    ]

    assert max(batch_spreads) <= 3


def test_length_aware_batching_supports_filtered_subset(tmp_path: Path) -> None:
    write_csv(
        tmp_path,
        {
            "idx": list(range(8)),
            "sequence": ["A" * length for length in range(1, 9)],
            "label": [0] * 8,
        },
    )
    loader = create_dataloader(
        task="toy",
        split="train",
        data_dir=tmp_path,
        batch_size=2,
        use_cache=False,
        loader_type="max_length",
        max_length=4,
        length_aware_batching=True,
    )

    assert isinstance(loader.batch_sampler, LengthAwareBatchSampler)
    assert sorted(_collect_sample_ids(loader)) == [0, 1, 2, 3]


def test_persistent_workers_is_disabled_without_workers(tmp_path: Path) -> None:
    write_split_csv(tmp_path)

    loader = create_dataloader(
        task="toy",
        split="train",
        data_dir=tmp_path,
        use_cache=False,
        num_workers=0,
        persistent_workers=True,
    )

    assert loader.persistent_workers is False


def test_filtered_loader_preserves_worker_and_generator_settings(tmp_path: Path) -> None:
    write_split_csv(tmp_path)
    generator = torch.Generator().manual_seed(123)

    def initialize_worker(worker_id: int) -> None:
        del worker_id

    loader = create_dataloader(
        task="toy",
        split="train",
        data_dir=tmp_path,
        use_cache=False,
        num_workers=1,
        generator=generator,
        worker_init_fn=initialize_worker,
        persistent_workers=True,
        loader_type="max_length",
        max_length=3,
    )

    assert loader.generator is generator
    assert loader.worker_init_fn is initialize_worker
    assert loader.persistent_workers is True


def test_create_dataloader_can_filter_by_max_length(tmp_path: Path) -> None:
    write_csv(
        tmp_path,
        {
            "sequence": ["A", "AC", "ACD", "ACDE"],
            "label": [0, 1, 0, 1],
        },
    )

    loader = create_dataloader(
        task="toy",
        split="train",
        data_dir=tmp_path,
        batch_size=4,
        shuffle=False,
        use_cache=False,
        loader_type="max_length",
        max_length=2,
    )

    assert _collect_lengths(loader) == [1, 2]


def test_create_dataloader_requires_max_length_for_max_length_loader(tmp_path: Path) -> None:
    write_split_csv(tmp_path)

    try:
        create_dataloader(
            task="toy",
            split="train",
            data_dir=tmp_path,
            use_cache=False,
            loader_type="max_length",
        )
    except ValueError as exc:
        assert "max_length must be specified" in str(exc)
    else:
        raise AssertionError("Expected max_length loader to require max_length")


def test_create_dataloader_can_filter_by_single_quartile(tmp_path: Path) -> None:
    write_csv(
        tmp_path,
        {
            "sequence": ["A", "AC", "ACD", "ACDE"],
            "label": [0, 1, 0, 1],
        },
    )

    loader = create_dataloader(
        task="toy",
        split="train",
        data_dir=tmp_path,
        batch_size=4,
        shuffle=False,
        use_cache=False,
        loader_type="quartile",
        quartile_name="ms",
    )

    assert _collect_lengths(loader) == [2]


def test_create_dataloader_can_filter_by_cumulative_quartile(tmp_path: Path) -> None:
    write_csv(
        tmp_path,
        {
            "sequence": ["A", "AC", "ACD", "ACDE"],
            "label": [0, 1, 0, 1],
        },
    )

    loader = create_dataloader(
        task="toy",
        split="train",
        data_dir=tmp_path,
        batch_size=4,
        shuffle=False,
        use_cache=False,
        loader_type="quartile",
        quartile_name="ml",
        cumulative=True,
    )

    assert _collect_lengths(loader) == [1, 2, 3]
