"""Protein sequence dataset and dataloader utilities.

Supports multiple tasks (e.g., localization, solubility), encoding modes
(char, raw), and model types (classification, autoencoder).

Expected data layout::

    {data_dir}/{task}/{split}.csv          # split-specific files (preferred)
    {data_dir}/{task}/{task}.csv           # combined file with a 'split' column

Each CSV must contain at least a sequence column and a label column.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data directory
DATA_DIR = "data/processed/peer"
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SequenceDataset(Dataset):
    """PyTorch Dataset for protein sequence classification and autoencoder tasks.

    Parameters
    ----------
    data_dir:
        Root directory under which task data lives.
        Expected layout: ``{data_dir}/{task}/{split}.csv``.
    task:
        Task name (e.g., ``"localization"``, ``"solubility"``).
    split:
        Data split — one of ``"train"``, ``"val"``, or ``"test"``.
    mode:
        ``"classification"`` or ``"autoencoder"``.
    encoding:
        ``"char"`` for integer-encoded sequences, ``"raw"`` for raw strings.
    max_length:
        Pad/truncate encoded sequences to this length. ``None`` means no
        truncation or padding.
    cache_dir:
        Directory for pickle cache files. Defaults to ``{data_dir}/cache``.
    use_cache:
        Whether to load/save processed examples from/to a pickle cache.
    seq_col:
        CSV column name for sequences.
    label_col:
        CSV column name for labels.
    """

    def __init__(
        self,
        task: str,
        split: str,
        mode: str = "classification",
        encoding: str = "char",
        max_length: int | None = 512,
        cache_dir: str | None = None,
        use_cache: bool = True,
        seq_col: str = "sequence",
        label_col: str = "label",
    ) -> None:
        if mode not in ("classification", "autoencoder"):
            raise ValueError(
                f"mode must be 'classification' or 'autoencoder', got {mode!r}"
            )
        if encoding not in ("char", "raw"):
            raise ValueError(f"encoding must be 'char' or 'raw', got {encoding!r}")
        if mode == "autoencoder" and encoding != "char":
            raise ValueError("autoencoder mode requires encoding='char'")

        self.task = task
        self.split = split
        self.mode = mode
        self.encoding = encoding
        self.max_length = max_length
        self.seq_col = seq_col
        self.label_col = label_col
        self.use_cache = use_cache

        # BOS/EOS tokens frame the reconstruction target in autoencoder mode.
        self.add_special_tokens: bool = mode == "autoencoder"

        self.vocab = VOCAB

        # Resolve cache directory (create lazily on first write).
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DATA_DIR / "cache"

        self.examples: list[dict[str, Any]] = self._load_examples()
        
    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        label_dtype = torch.long if isinstance(ex["_label"], int) else torch.float32
        label_tensor = torch.tensor(ex["_label"], dtype=label_dtype)

        if self.encoding == "char":
            input_ids = torch.tensor(ex["_token_ids"], dtype=torch.long)
            length = torch.tensor(ex["_length"], dtype=torch.long)

            if self.mode == "classification":
                return {
                    "input_ids": input_ids,
                    "label": label_tensor,
                    "length": length,
                    "sequence": ex["_sequence"],
                }
            else:  # autoencoder — target is identical to input for now
                return {
                    "input_ids": input_ids,
                    "target_ids": input_ids.clone(), # deep copy for decoder output
                    "length": length,
                    "sequence": ex["_sequence"],
                }

        else:  # raw
            return {
                "sequence": ex["_sequence"],
                "label": label_tensor,
                "length": torch.tensor(ex["_length"], dtype=torch.long),
            }

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self) -> str:
        """Build a deterministic, filesystem-safe cache filename.

        The filename encodes all preprocessing-relevant arguments so that
        different configurations never collide.
        """
        parts = "_".join(
            [
                self.task,
                self.split,
                self.mode,
                self.encoding,
                str(self.max_length),
                self.seq_col,
                self.label_col,
                str(self.add_special_tokens),
            ]
        )
        # Append a short hash to guard against overly long or
        # special-character filenames.
        digest = hashlib.md5(parts.encode()).hexdigest()[:8]
        safe = parts.replace("/", "-").replace(" ", "_")
        return f"{safe}_{digest}.pkl"

    def _cache_metadata(self) -> dict[str, Any]:
        """Return the metadata dict stored inside each cache file."""
        return {
            "task": self.task,
            "split": self.split,
            "mode": self.mode,
            "encoding": self.encoding,
            "max_length": self.max_length,
            "seq_col": self.seq_col,
            "label_col": self.label_col,
            "add_special_tokens": self.add_special_tokens,
            "vocab": self.vocab,
        }

    def _cache_is_valid(self, cached_meta: dict[str, Any]) -> bool:
        """Return True when cached metadata exactly matches the current config."""
        for key, val in self._cache_metadata().items():
            if cached_meta.get(key) != val:
                return False
        return True

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _find_csv(self) -> tuple[Path, bool]:
        """Locate the CSV file for this task/split.

        Returns the path and a boolean indicating whether the file is a
        combined dataset (True) that needs to be filtered by split column.

        Raises
        ------
        FileNotFoundError
            If neither a split-specific file nor a combined file is found.
        """
        task_dir = self.data_dir / self.task
        split_file = task_dir / f"{self.split}.csv"
        combined_file = task_dir / f"{self.task}.csv"

        if split_file.exists():
            return split_file, False
        if combined_file.exists():
            return combined_file, True

        raise FileNotFoundError(
            f"No CSV found for task={self.task!r}, split={self.split!r}.\n"
            f"  Looked for:\n    {split_file}\n    {combined_file}"
        )

    def _load_dataframe(self) -> pd.DataFrame:
        """Load the CSV and return rows for the requested split."""
        csv_path, is_combined = self._find_csv()
        df = pd.read_csv(csv_path)

        # Filter combined files to the requested split.
        if is_combined:
            if "split" not in df.columns:
                raise ValueError(
                    f"Combined file {csv_path} has no 'split' column to filter by."
                )
            df = df[df["split"] == self.split].reset_index(drop=True)

        # Validate required columns.
        missing = {self.seq_col, self.label_col} - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV {csv_path} is missing required columns: {missing}. "
                f"Available: {list(df.columns)}"
            )

        # Drop rows with missing sequences.
        before = len(df)
        df = df.dropna(subset=[self.seq_col]).reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            logger.warning("Dropped %d rows with missing sequences.", dropped)

        return df

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode_sequence(self, sequence: str) -> tuple[list[int], int]:
        """Encode a single amino acid sequence into integer token IDs.

        Parameters
        ----------
        sequence:
            Raw (already-normalized) amino acid string.

        Returns
        -------
        token_ids:
            Padded/truncated list of integer IDs of length ``max_length``
            (or the natural length when ``max_length`` is ``None``).
        effective_length:
            Number of real (non-padding) tokens, i.e., the sequence length
            after optional truncation (including BOS/EOS when applicable).
        """
        if self.add_special_tokens:
            # Reserve 2 positions for BOS and EOS within max_length.
            content_limit = (self.max_length - 2) if self.max_length is not None else None
            seq = sequence[:content_limit] if content_limit is not None else sequence
            tokens = [BOS_IDX] + [self.vocab.get(aa, UNK_IDX) for aa in seq] + [EOS_IDX]
        else:
            seq = sequence[:self.max_length] if self.max_length is not None else sequence
            tokens = [self.vocab.get(aa, UNK_IDX) for aa in seq]

        effective_length = len(tokens)

        # Pad to max_length with <PAD> tokens.
        if self.max_length is not None and len(tokens) < self.max_length:
            tokens = tokens + [PAD_IDX] * (self.max_length - len(tokens))

        return tokens, effective_length

    # ------------------------------------------------------------------
    # Processing pipeline
    # ------------------------------------------------------------------

    def _build_examples(self) -> list[dict[str, Any]]:
        """Load the CSV and produce the list of processed example dicts.

        Returns
        -------
        list[dict[str, Any]]
            List of processed example dictionaries.
        """
        df = self._load_dataframe()
        examples: list[dict[str, Any]] = []

        for _, row in df.iterrows():
            raw_seq: str = str(row[self.seq_col]).upper().strip()

            # Convert label to int when possible; fall back to float for
            # continuous regression targets.
            raw_label = row[self.label_col]
            try:
                label: int | float = int(raw_label)
            except (ValueError, TypeError):
                label = float(raw_label)

            if self.encoding == "char":
                token_ids, length = self._encode_sequence(raw_seq)
                example: dict[str, Any] = {
                    "_token_ids": token_ids,
                    "_length": length,
                    "_label": label,
                    "_sequence": raw_seq,
                }
            else:  # raw — no integer encoding
                seq = raw_seq[:self.max_length] if self.max_length is not None else raw_seq
                example = {
                    "_sequence": seq,
                    "_length": len(seq),
                    "_label": label,
                }

            examples.append(example)

        return examples

    def _load_examples(self) -> list[dict[str, Any]]:
        """Return processed examples, using the pickle cache when available."""
        if not self.use_cache:
            return self._build_examples()

        # Ensure the cache directory exists before attempting any read/write.
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / self._cache_key()

        # --- Try loading from an existing cache file ---
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as fh:
                    cached = pickle.load(fh)

                if self._cache_is_valid(cached["metadata"]):
                    n = len(cached["examples"])
                    print(f"[cache] Loaded {n} examples from cache: {cache_path.name}")
                    return cached["examples"]
                else:
                    # Config changed — discard stale cache and rebuild.
                    print("[cache] Metadata mismatch — rebuilding cache.")
            except Exception as exc:  # noqa: BLE001
                # Corrupted or incompatible pickle; rebuild silently.
                logger.warning("[cache] Failed to load cache (%s) — rebuilding.", exc)
                print("[cache] Failed to load cache — rebuilding.")

        # --- Build from the source CSV ---
        print(f"[cache] Building examples for task={self.task!r}, split={self.split!r} ...")
        examples = self._build_examples()

        # Persist the processed examples (not the Dataset object) to disk.
        meta = self._cache_metadata()
        meta["num_examples"] = len(examples)
        payload = {"metadata": meta, "examples": examples}

        try:
            with open(cache_path, "wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[cache] Saved cache: {cache_path.name}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[cache] Could not write cache: %s", exc)

        return examples

    


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def create_dataloader(
    task: str,
    split: str,
    mode: str = "classification",
    encoding: str = "char",
    max_length: int | None = 512,
    cache_dir: str | None = None,
    use_cache: bool = True,
    seq_col: str = "sequence",
    label_col: str = "label",
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a DataLoader for a protein sequence dataset.

    Parameters
    ----------
    data_dir:
        Root data directory. See :class:`ProteinSequenceDataset`.
    task:
        Task name (e.g., ``"localization"``, ``"solubility"``).
    split:
        Data split — one of ``"train"``, ``"val"``, or ``"test"``.
    mode:
        ``"classification"`` or ``"autoencoder"``.
    encoding:
        ``"char"`` for integer-encoded sequences, ``"raw"`` for raw strings.
    max_length:
        Max sequence length (tokens). ``None`` for no truncation.
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
        Whether to pin memory (recommended when using a GPU).

    Returns
    -------
    DataLoader
        A PyTorch DataLoader whose batches are dicts. String fields (e.g.,
        ``"sequence"``) are collected into lists by the default collate
        function; tensor fields are stacked into batched tensors.

    Examples
    --------
    Classification with char encoding::

        loader = create_dataloader(
            data_dir="data", task="localization", split="train",
            mode="classification", encoding="char", batch_size=32, shuffle=True,
        )

    Autoencoder::

        loader = create_dataloader(
            data_dir="data", task="solubility", split="train",
            mode="autoencoder", encoding="char", batch_size=32, shuffle=True,
        )

    ESM-2 (raw strings, tokenize later)::

        loader = create_dataloader(
            data_dir="data", task="localization", split="train",
            mode="classification", encoding="raw", max_length=None,
            batch_size=16, shuffle=True, use_cache=False,
        )
    """
    dataset = SequenceDataset(
        task=task,
        split=split,
        mode=mode,
        encoding=encoding,
        max_length=max_length,
        cache_dir=cache_dir,
        use_cache=use_cache,
        seq_col=seq_col,
        label_col=label_col,
    )

    # The default collate_fn handles dicts containing:
    #   - tensors  → stacked into batched tensors
    #   - strings  → collected into a list
    # A custom collate_fn is not required because char-encoded sequences are
    # already padded to a uniform length, and raw strings are handled natively.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
