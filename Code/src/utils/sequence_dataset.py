"""Protein sequence Dataset utility class

Wraps the torch Dataset class for protein sequencing usecase.
"""

import hashlib
import logging
import math
import os
import pickle
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed/peer")

# Increment these values whenever the serialized payload shape or sequence
# preprocessing behavior changes. Including both values in cache identity keeps
# old caches from being treated as compatible after a code change.
CACHE_SCHEMA_VERSION = 2
PREPROCESSING_VERSION = 2


@lru_cache(maxsize=128)
def _cached_file_sha256(
    resolved_path: str,
    size_bytes: int,
    mtime_ns: int,
    ctime_ns: int,
) -> str:
    """Hash a source file once for a given filesystem fingerprint.

    The stat values are intentionally part of the function signature: they make
    the LRU entry stale whenever the file is rewritten while avoiding repeated
    reads of large, unchanged CSV files across dataset instances.
    """
    del size_bytes, mtime_ns, ctime_ns  # Used as cache-key components.
    digest = hashlib.sha256()
    with Path(resolved_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

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
                    "sample_id": ex["_sample_id"],
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
                "sample_id": ex["_sample_id"],
            }

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self) -> str:
        """Build a cache filename.

        The filename encodes all preprocessing-relevant arguments so that different configurations never collide.
        """
        source = self._source_fingerprint()
        parts = "_".join(
            [
                self.task,
                self.split,
                self.mode,
                self.encoding,
                self.seq_col,
                self.label_col,
                str(self.add_special_tokens),
                f"schema{CACHE_SCHEMA_VERSION}",
                f"preprocessing{PREPROCESSING_VERSION}",
                f"source-{source['sha256']}",
            ]
        )
        # Append a short hash to guard against overly long or
        # special-character filenames.
        # The compact suffix also distinguishes identical files located in
        # different source roots when callers share a cache directory.
        digest_material = f"{parts}_{source['path']}"
        digest = hashlib.sha256(digest_material.encode()).hexdigest()[:12]
        safe = parts.replace("/", "-").replace(" ", "_")
        return f"{safe}_{digest}.pkl"

    def _source_fingerprint(self) -> dict[str, str | int]:
        """Return stable provenance and content identity for the source CSV."""
        cached = getattr(self, "_source_fingerprint_value", None)
        if cached is not None:
            return cached

        csv_path, _ = self._find_csv()
        resolved_path = csv_path.resolve()
        stat = resolved_path.stat()
        fingerprint: dict[str, str | int] = {
            "path": str(resolved_path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _cached_file_sha256(
                str(resolved_path),
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            ),
        }
        self._source_fingerprint_value = fingerprint
        return fingerprint

    def _cache_metadata(self) -> dict:
        """Return the metadata dict stored inside each cache file.

        Returns
        -------
        dict
            Metadata dictionary containing the configuration used to create the cache.
        """
    
        return {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "preprocessing_version": PREPROCESSING_VERSION,
            "task": self.task,
            "split": self.split,
            "mode": self.mode,
            "encoding": self.encoding,
            "seq_col": self.seq_col,
            "label_col": self.label_col,
            "add_special_tokens": self.add_special_tokens,
            "vocab": self.vocab,
            "source_csv": self._source_fingerprint(),
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
            # Preserve the original RangeIndex: it is the stable source-row
            # fallback when the CSV does not provide an explicit ``idx``.
            df = df[df["split"] == self.split]

        # Validate required columns
        missing = {self.seq_col, self.label_col} - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV {csv_path} is missing required columns: {missing}. "
                f"Available: {list(df.columns)}"
            )

        if df[self.seq_col].isna().any():
            raise ValueError(f"CSV {csv_path} contains missing sequences.")
        normalized_sequences = df[self.seq_col].astype(str).str.strip()
        if normalized_sequences.eq("").any():
            raise ValueError(f"CSV {csv_path} contains empty sequences.")

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

        for source_row_index, row in df.iterrows():
            raw_seq: str = str(row[self.seq_col]).upper().strip()

            sample_id: Any = source_row_index
            if "idx" in df.columns and pd.notna(row["idx"]):
                sample_id = row["idx"]
            # Convert NumPy scalar IDs to ordinary Python values so cached
            # examples and collated batches have a portable representation.
            if hasattr(sample_id, "item"):
                sample_id = sample_id.item()

            # Convert label to int when possible; fall back to float for continuous regression targets.
            raw_label = row[self.label_col]
            try:
                numeric_label = float(raw_label)
            except (ValueError, TypeError) as error:
                raise ValueError(
                    f"Invalid numeric label {raw_label!r} at source row "
                    f"{source_row_index}."
                ) from error
            if not math.isfinite(numeric_label):
                raise ValueError(
                    f"Invalid numeric label {raw_label!r} at source row "
                    f"{source_row_index}."
                )
            label: int | float = (
                int(numeric_label) if numeric_label.is_integer() else numeric_label
            )

            if self.encoding == "char":
                token_ids, length = self._encode_sequence(raw_seq)
                example: dict = {
                    "_token_ids": token_ids,
                    "_length": length,
                    "_label": label,
                    "_sequence": raw_seq,
                    "_sample_id": sample_id,
                }
            else: # raw — no integer encoding -> this is the case for ESM-based models that take raw strings as input and handle this internally
                seq = raw_seq
                example = {
                    "_sequence": seq,
                    "_length": len(seq),
                    "_label": label,
                    "_sample_id": sample_id,
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

        temporary_path: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{cache_path.name}.",
                suffix=".tmp",
                dir=self.cache_dir,
            )
            os.close(descriptor)
            temporary_path = Path(temporary_name)
            with temporary_path.open("wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temporary_path, cache_path)
            print(f"[cache] Saved cache: {cache_path.name}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[cache] Could not write cache: %s", exc)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        return examples

    
