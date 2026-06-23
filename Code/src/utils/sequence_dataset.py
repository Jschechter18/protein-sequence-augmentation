"""Protein sequence Dataset utility class

Wraps the torch Dataset class for protein sequencing usecase.
"""

import hashlib
from pathlib import Path
import pickle
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Any

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed/peer")

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Standard 20 amino acids in alphabetical order
_STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"

VOCAB: dict[str, int] = {
    "<PAD>": 0, # padding token for variable-length sequences
    "<UNK>": 1, # unknown token for the ambiguous amino acid characters
    "<BOS>": 2, # beginning-of-sequence token for autoencoder mode
    "<EOS>": 3, # end-of-sequence token for autoencoder mode
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
        Data split — one of ``"train"``, ``"valid"``, or ``"test"``.
    mode:
        ``"classification"`` or ``"autoencoder"``.
    encoding:
        ``"char"`` for integer-encoded sequences, ``"raw"`` for raw strings.
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
        data_dir: str | Path = DATA_DIR,
        mode: str = "classification",
        encoding: str = "char",
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
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.encoding = encoding
        self.seq_col = seq_col
        self.label_col = label_col
        self.use_cache = use_cache

        # BOS/EOS tokens frame the reconstruction target in autoencoder mode
        self.add_special_tokens: bool = mode == "autoencoder"

        self.vocab = VOCAB

        # Resolve cache directory (create lazily on first write)
        self.cache_dir = (
            Path(cache_dir) if cache_dir is not None else self.data_dir / "cache"
        )

        self.examples: list[dict[str, Any]] = self._load_examples()
        
    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
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
        """Build a cache filename.

        The filename encodes all preprocessing-relevant arguments so that different configurations never collide.
        """
        parts = "_".join(
            [
                self.task,
                self.split,
                self.mode,
                self.encoding,
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

    def _cache_metadata(self) -> dict:
        """Return the metadata dict stored inside each cache file.

        Returns
        -------
        dict
            Metadata dictionary containing the configuration used to create the cache.
        """
    
        return {
            "task": self.task,
            "split": self.split,
            "mode": self.mode,
            "encoding": self.encoding,
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

        Returns the path and a boolean indicating whether the file is a combined dataset (True) that needs to be filtered by split column.

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
        """Load the CSV and return rows for the requested split.

        Returns
        -------
        pd.DataFrame
            Dataframe of the dataset that was pulled from the csv

        Raises
        ------
        ValueError
            If can't find the split
        ValueError
            If csv is missing designated column
        """
        csv_path, is_combined = self._find_csv()
        df = pd.read_csv(csv_path)

        # Filter combined files to the requested split.
        if is_combined:
            if "split" not in df.columns:
                raise ValueError(
                    f"Combined file {csv_path} has no 'split' column to filter by."
                )
            df = df[df["split"] == self.split].reset_index(drop=True)

        # Validate required columns
        missing = {self.seq_col, self.label_col} - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV {csv_path} is missing required columns: {missing}. "
                f"Available: {list(df.columns)}"
            )

        # Drop rows with missing sequences
        # before = len(df)
        # df = df.dropna(subset=[self.seq_col]).reset_index(drop=True)
        # dropped = before - len(df)
        # if dropped:
        #     logger.warning("Dropped %d rows with missing sequences.", dropped)

        return df

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode_sequence(self, sequence: str) -> tuple[list[int], int]:
        """Encode a single amino acid sequence into integer token IDs

        Parameters
        ----------
        sequence:
            Raw (already-normalized) amino acid string.

        Returns
        -------
        token_ids:
            List of integer IDs for the full sequence.
        effective_length:
            Number of real tokens, including BOS/EOS when applicable.
        """
        if self.add_special_tokens:
            tokens = [BOS_IDX] + [self.vocab.get(aa, UNK_IDX) for aa in sequence] + [EOS_IDX]
        else:
            tokens = [self.vocab.get(aa, UNK_IDX) for aa in sequence]

        effective_length = len(tokens)

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

            # Convert label to int when possible; fall back to float for continuous regression targets.
            raw_label = row[self.label_col]
            try:
                label: int | float = int(raw_label)
            except (ValueError, TypeError):
                label = float(raw_label)

            if self.encoding == "char":
                token_ids, length = self._encode_sequence(raw_seq)
                example: dict = {
                    "_token_ids": token_ids,
                    "_length": length,
                    "_label": label,
                    "_sequence": raw_seq,
                }
            else: # raw — no integer encoding -> this is the case for ESM-based models that take raw strings as input and handle this internally
                seq = raw_seq
                example = {
                    "_sequence": seq,
                    "_length": len(seq),
                    "_label": label,
                }

            examples.append(example)

        return examples

    def _load_examples(self) -> list[dict[str, Any]]:
        """Return processed examples, using the pickle cache when available

        Returns
        -------
        list[dict[str, Any]]
            _description_
        """
        if not self.use_cache:
            # Loading for first time
            return self._build_examples()

        # Ensure the cache directory exists before attempting any read/write.
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / self._cache_key()

        # Try loading from an existing cache file
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as fh:
                    cached = pickle.load(fh)

                if self._cache_is_valid(cached["metadata"]):
                    n = len(cached["examples"])
                    print(f"[cache] Loaded {n} examples from cache: {cache_path.name}")
                    return cached["examples"]
                else:
                    # Config changed — discard stale cache and rebuild
                    print("[cache] Metadata mismatch — rebuilding cache.")
            except Exception as exc:
                # Corrupted or incompatible pickle; rebuild silently
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

    
