"""Protein sequence dataloader utility function

Supports multiple tasks (e.g., localization, solubility), encoding modes (char, raw), and model types (classification, autoencoder)
"""
from pathlib import Path
import torch
from torch.utils.data import DataLoader
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
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_sequence_batch, # The custom collate function keeps raw strings as lists and pads variable-length token tensors.
    )
