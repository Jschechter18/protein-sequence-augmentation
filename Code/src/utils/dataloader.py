"""Protein sequence dataloader utility function

Supports multiple tasks (e.g., localization, solubility), encoding modes (char, raw), and model types (classification, autoencoder)
"""
from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from .sequence_dataset import SequenceDataset



# ---------------------------------------------------------------------------
# Data directory
DATA_DIR = Path("data/processed/peer")
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Standard 20 amino acids in alphabetical order.
_STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"

VOCAB: dict[str, int] = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
}
for _aa in _STANDARD_AAS:
    VOCAB[_aa] = len(VOCAB)

PAD_IDX = VOCAB["<PAD>"]
UNK_IDX = VOCAB["<UNK>"]
BOS_IDX = VOCAB["<BOS>"]
EOS_IDX = VOCAB["<EOS>"]

VOCAB_SIZE = len(VOCAB)

# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def _pad_1d_tensors(tensors: list[torch.Tensor], pad_value: int = PAD_IDX) -> torch.Tensor:
    """Pad a list of 1D tensors to the longest length in the batch

    Parameters
    ----------
    tensors : list[torch.Tensor]
        _description_
    pad_value : int, optional
        _description_, by default PAD_IDX

    Returns
    -------
    torch.Tensor
        _description_
    """
    max_len = max(tensor.size(0) for tensor in tensors)
    padded = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype)
    for idx, tensor in enumerate(tensors):
        padded[idx, : tensor.size(0)] = tensor
    return padded


def collate_sequence_batch(batch: list[dict]) -> dict:
    """Collect sequence examples, padding variable-length token tensors if needed

    Parameters
    ----------
    batch : list[dict]
        _description_

    Returns
    -------
    dict
        Key value of sequences
    """
    collated: dict = {}
    keys = batch[0].keys()

    for key in keys:
        values = [item[key] for item in batch]

        if key in {"input_ids", "target_ids"}:
            collated[key] = _pad_1d_tensors(values)
        elif isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        elif isinstance(values[0], str):
            collated[key] = values
        else:
            collated[key] = values

    return collated


# ---------------------------------------------------------------------------
# Sequence length bin utilities
# ---------------------------------------------------------------------------


QUARTILES = ["s", "ms", "ml", "l"]
LENGTH_SPLIT_COUNTS = {
    "halves": 2,
    "thirds": 3,
    "quarters": 4,
}

def get_length(dataset, idx):
    # Your SequenceDataset stores preprocessed examples with "_length"
    if hasattr(dataset, "examples"):
        return int(dataset.examples[idx]["_length"])

    item = dataset[idx]
    length = item["length"]
    return int(length.item() if hasattr(length, "item") else length)


def compute_train_length_boundaries(train_dataset, num_bins: int = 4):
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    lengths = np.array([get_length(train_dataset, i) for i in range(len(train_dataset))])
    return np.quantile(lengths, np.linspace(0.0, 1.0, num_bins + 1))


def length_bin_indices(dataset, boundaries, bin_index: int, cumulative=False):
    if bin_index < 1 or bin_index >= len(boundaries):
        raise ValueError(f"bin_index must be between 1 and {len(boundaries) - 1}")

    lower = boundaries[bin_index - 1]
    upper = boundaries[bin_index]

    indices = []
    for i in range(len(dataset)):
        length = get_length(dataset, i)
        if cumulative:
            keep = length <= upper
        elif bin_index == 1:
            keep = lower <= length <= upper
        else:
            keep = lower < length <= upper

        if keep:
            indices.append(i)

    return indices


def make_length_bin_loader(base_loader, boundaries, bin_index: int, shuffle, cumulative=False):
    indices = length_bin_indices(base_loader.dataset, boundaries, bin_index, cumulative)
    subset = Subset(base_loader.dataset, indices)

    return DataLoader(
        subset,
        batch_size=base_loader.batch_size,
        shuffle=shuffle,
        num_workers=base_loader.num_workers,
        pin_memory=base_loader.pin_memory,
        collate_fn=base_loader.collate_fn,
    )


def quartile_indices(dataset, boundaries, quartile_name, cumulative=False):
    q = QUARTILES.index(quartile_name)
    lower = boundaries[q]
    upper = boundaries[q + 1]

    indices = []
    for i in range(len(dataset)):
        length = get_length(dataset, i)
        if cumulative:
            keep = length <= upper
        elif q == 0:
            keep = lower <= length <= upper
        else:
            keep = lower < length <= upper

        if keep:
            indices.append(i)

    return indices


def make_quartile_loader(base_loader, boundaries, quartile_name, shuffle, cumulative=False):
    indices = quartile_indices(base_loader.dataset, boundaries, quartile_name, cumulative)
    subset = Subset(base_loader.dataset, indices)

    return DataLoader(
        subset,
        batch_size=base_loader.batch_size,
        shuffle=shuffle,
        num_workers=base_loader.num_workers,
        pin_memory=base_loader.pin_memory,
        collate_fn=base_loader.collate_fn,
    )
    
def make_max_length_loader(base_loader, max_length, shuffle):
    indices = [i for i in range(len(base_loader.dataset)) if get_length(base_loader.dataset, i) <= max_length]
    subset = Subset(base_loader.dataset, indices)

    return DataLoader(
        subset,
        batch_size=base_loader.batch_size,
        shuffle=shuffle,
        num_workers=base_loader.num_workers,
        pin_memory=base_loader.pin_memory,
        collate_fn=base_loader.collate_fn,
    )

def create_dataloader(
    task: str = "solubility",
    split: str = "train",
    data_dir: str | Path = DATA_DIR,
    mode: str = "classification",
    encoding: str = "char",
    cache_dir: str | None = None,
    use_cache: bool = True,
    seq_col: str = "sequence",
    label_col: str = "label",
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    loader_type: str | None = None,
    max_length: int | None = None,
    quartile_name: str | None = None,
    length_options: str | None = None,
    length_bin: int | None = None,
    cumulative: bool = False,
    length_boundaries: np.ndarray | None = None,
) -> DataLoader[SequenceDataset]:
    """Create a DataLoader for a protein sequence dataset.

    Parameters
    ----------
    data_dir:
        Root data directory. See :class:`ProteinSequenceDataset`.
    task:
        Task name (e.g., ``"localization"``, ``"solubility"``). Defaults to ``"solubility"`` so autoencoder callers only need to set ``mode`` when using the default training data.
    split:
        Data split — one of ``"train"``, ``"valid"``, or ``"test"``. Defaults to ``"train"``.
    mode:
        ``"classification"`` or ``"autoencoder"``.
    encoding:
        ``"char"`` for integer-encoded sequences, ``"raw"`` for raw strings.
    cache_dir:
        Cache directory. Defaults to ``{data_dir}/cache``.
    use_cache:
        Whether to use the pickle cache.
    seq_col:
        Sequence column name in the CSV.
    label_col:
        Label column name in the CSV.
    batch_size:
        Number of samples per batch.
    shuffle:
        Whether to shuffle the data each epoch.
    num_workers:
        Number of DataLoader worker processes.
    pin_memory:
        Whether to pin memory (allows for faster data transfers btw CPU and GPU).
    loader_type:
        Optional sequence-length filter type. Supports ``"max_length"`` and
        ``"quartile"``. ``None`` returns the full dataset.
    max_length:
        Maximum sequence length to keep when ``loader_type="max_length"``.
    quartile_name:
        Length quartile to keep when ``loader_type="quartile"``. One of
        ``"s"``, ``"ms"``, ``"ml"``, or ``"l"``.
    length_options:
        Length split scheme to use when ``loader_type="length_bin"``. One of
        ``"halves"``, ``"thirds"``, or ``"quarters"``.
    length_bin:
        1-indexed length bin to keep when ``loader_type="length_bin"``.
    cumulative:
        If true, length filtering keeps the selected bin and all shorter bins.
    length_boundaries:
        Optional precomputed length boundaries to use for length-bin filtering.
        Pass train-set boundaries when filtering validation or test splits.

    Returns
    -------
    DataLoader
        A PyTorch DataLoader whose batches are dicts. String fields (e.g., ``"sequence"``) are collected into lists by the default collate function; tensor fields are stacked into batched tensors.

    Examples
    --------
    Classification with char encoding:

        loader = create_dataloader(
            data_dir="data/processed/peer", task="localization", split="train",
            mode="classification", encoding="char", batch_size=32, shuffle=True,
        )

    Autoencoder:

        loader = create_dataloader(
            mode="autoencoder", encoding="char", batch_size=32, shuffle=True,
        )

    ESM-2 (raw strings, tokenize later):

        loader = create_dataloader(
            data_dir="data/processed/peer", task="localization", split="train",
            mode="classification", encoding="raw",
            batch_size=16, shuffle=True, use_cache=False,
        )
    """
    dataset = SequenceDataset(
        task=task,
        split=split,
        data_dir=data_dir,
        mode=mode,
        encoding=encoding,
        cache_dir=cache_dir,
        use_cache=use_cache,
        seq_col=seq_col,
        label_col=label_col,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_sequence_batch, # The custom collate function keeps raw strings as lists and pads variable-length token tensors.
    )
    
    

    if loader_type == "max_length":
        if max_length is None:
            raise ValueError("max_length must be specified when loader_type is 'max_length'")
        dataloader = make_max_length_loader(dataloader, max_length, shuffle)
    elif loader_type == "quartile":
        if quartile_name is None:
            raise ValueError("quartile_name must be specified when loader_type is 'quartile'")
        if quartile_name not in QUARTILES:
            raise ValueError(f"quartile_name must be one of {QUARTILES}")
        boundaries = length_boundaries
        if boundaries is None:
            boundaries = compute_train_length_boundaries(dataset)
        dataloader = make_quartile_loader(dataloader, boundaries, quartile_name, shuffle, cumulative=cumulative)
    elif loader_type == "length_bin":
        if length_options is None:
            raise ValueError("length_options must be specified when loader_type is 'length_bin'")
        if length_options not in LENGTH_SPLIT_COUNTS:
            raise ValueError(f"length_options must be one of {list(LENGTH_SPLIT_COUNTS)}")
        if length_bin is None:
            raise ValueError("length_bin must be specified when loader_type is 'length_bin'")
        num_bins = LENGTH_SPLIT_COUNTS[length_options]
        if not 1 <= length_bin <= num_bins:
            raise ValueError(f"length_bin must be between 1 and {num_bins} for {length_options}")
        boundaries = length_boundaries
        if boundaries is None:
            boundaries = compute_train_length_boundaries(dataset, num_bins=num_bins)
        dataloader = make_length_bin_loader(dataloader, boundaries, length_bin, shuffle, cumulative=cumulative)
    elif loader_type is not None:
        raise ValueError("loader_type must be one of None, 'max_length', 'quartile', or 'length_bin'")
    
    return dataloader
