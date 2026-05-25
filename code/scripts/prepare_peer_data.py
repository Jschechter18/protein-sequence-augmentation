#!/usr/bin/env python
"""Prepare official PEER benchmark splits for localization and solubility.

This script intentionally mirrors the official PEER/TorchDrug data path:
1) validate that the official PEER benchmark repository is available,
2) download official task archives if LMDB files are missing,
3) read the official LMDB split files (train/valid/test),
4) normalize records into simple CSVs for downstream research code,
5) emit metadata with split sizes and label distributions.

The output is deterministic with respect to the source archives and preserves
the official split boundaries from the benchmark.
"""

import csv
import hashlib
import json
import math
import os
import pickle
import tarfile
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import lmdb  # type: ignore[import-not-found]
except ImportError:
    lmdb = None


REPO_ROOT = Path(__file__).resolve().parents[2]
PEER_REPO = REPO_ROOT / "external" / "PEER_Benchmark"
RAW_ROOT = REPO_ROOT / "data" / "raw" / "peer"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed" / "peer"
EXTRACT_ROOT = RAW_ROOT / "extracted"

DATASETS = {
    "localization": {
        "peer_dataset_class": "SubcellularLocalization",
        "peer_config": PEER_REPO / "config" / "single_task" / "BERT" / "subloc_BERT.yaml",
        "url": "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/subcellular_localization.tar.gz",
        "md5": "37cb6138b8d4603512530458b7c8a77d",
        "archive_name": "subcellular_localization.tar.gz",
        "task_dir": "subcellular_localization",
        "split_basename": "subcellular_localization",
        "target_field": "localization",
        "output_dir": PROCESSED_ROOT / "localization",
    },
    "solubility": {
        "peer_dataset_class": "Solubility",
        "peer_config": PEER_REPO / "config" / "single_task" / "BERT" / "solubility_BERT.yaml",
        "url": "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/solubility.tar.gz",
        "md5": "8a8612b7bfa2ed80375db6e465ccf77e",
        "archive_name": "solubility.tar.gz",
        "task_dir": "solubility",
        "split_basename": "solubility",
        "target_field": "solubility",
        "output_dir": PROCESSED_ROOT / "solubility",
    },
}

SPLITS = ("train", "valid", "test")


def main() -> None:
    """Run the end-to-end PEER data preparation workflow.

    Raises
    ------
    ImportError
        _description_
    """
    
    
    if lmdb is None:
        raise ImportError(
            "The `lmdb` package is required to prepare PEER data. "
            "Run `python -m pip install lmdb` or use `bash scripts/setup_peer_data.sh`."
        )

    # Ensure we are anchored to an official PEER checkout before touching data.
    validate_peer_checkout()
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)

    # Top-level metadata captures provenance and per-dataset summaries.
    metadata = {
        "source_repo_path": str(PEER_REPO),
        "timestamp": utc_now(),
        "datasets": {},
    }

    for dataset_name, config in DATASETS.items():
        print(f"[PEER] Preparing {dataset_name} from official class {config['peer_dataset_class']}")
        lmdb_paths = ensure_dataset_lmdbs(dataset_name, config)
        dataset_metadata = export_dataset(dataset_name, config, lmdb_paths)
        metadata["datasets"][dataset_name] = dataset_metadata

    metadata_path = PROCESSED_ROOT / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[PEER] Wrote metadata summary to {metadata_path}")


def validate_peer_checkout() -> None:
    """Confirm the expected official PEER repository/config files exist.

    Raises
    ------
    FileNotFoundError
        _description_
    FileNotFoundError
        _description_
    """
        
    if not PEER_REPO.exists():
        raise FileNotFoundError(
            f"Expected the official PEER repo at {PEER_REPO}, but it was not found. "
            "Run `bash scripts/setup_peer_data.sh` to clone it first."
        )

    missing_configs = [
        str(config["peer_config"])
        for config in DATASETS.values()
        if not config["peer_config"].exists()
    ]
    if missing_configs:
        raise FileNotFoundError(
            "The PEER checkout does not contain the expected benchmark config files:\n"
            + "\n".join(missing_configs)
            + "\nPlease verify external/PEER_Benchmark is a valid checkout of the official repository."
        )


def ensure_dataset_lmdbs(dataset_name: str, config: dict) -> dict:
    """Return split->LMDB paths, downloading/extracting official archives as needed.


    Parameters
    ----------
    dataset_name : str
        _description_
    config : dict
        _description_

    Returns
    -------
    dict
        _description_

    Raises
    ------
    FileNotFoundError
        _description_
    """
    # Fast path for repeated runs: if split files are already present, skip network I/O.
    discovered = discover_split_lmdbs(config)
    if all(discovered.values()):
        print(f"[PEER] Found existing official LMDB files for {dataset_name}")
        return discovered

    archive_path = RAW_ROOT / config["archive_name"]
    if archive_path.exists():
        print(f"[PEER] Reusing existing archive {archive_path}")
    else:
        print(f"[PEER] Downloading {dataset_name} archive from official PEER dataset host")
        download_file(config["url"], archive_path)

    verify_md5(archive_path, config["md5"])
    print(f"[PEER] Extracting {archive_path} into {EXTRACT_ROOT}")
    safe_extract_tar(archive_path, EXTRACT_ROOT)

    # Re-discover after extraction so we fail loudly if archive layout changed.
    discovered = discover_split_lmdbs(config)
    missing = [split for split, path in discovered.items() if path is None]
    if missing:
        expected = [f"{config['split_basename']}_{split}.lmdb" for split in SPLITS]
        raise FileNotFoundError(
            f"Could not find the expected LMDB files for {dataset_name} after extraction. "
            f"Missing splits: {missing}. Expected basenames: {expected}. "
            f"Searched under {RAW_ROOT}."
        )
    return discovered


def discover_split_lmdbs(config: dict) -> dict:
    """Locate each split LMDB path for a dataset config.

    We first check the canonical extraction location and then fall back to a
    recursive search under data/raw/peer to tolerate minor directory changes.

    Parameters
    ----------
    config : dict
        _description_

    Returns
    -------
    dict
        _description_
    """
    
    split_paths = {}
    for split in SPLITS:
        basename = f"{config['split_basename']}_{split}.lmdb"
        expected = EXTRACT_ROOT / config["task_dir"] / basename
        if expected.exists():
            split_paths[split] = expected
            continue

        matches = sorted(RAW_ROOT.rglob(basename))
        split_paths[split] = matches[0] if matches else None
    return split_paths


def export_dataset(dataset_name: str, config: dict, lmdb_paths: dict) -> dict:
    """Export all official splits for one dataset and build metadata summary.

    Parameters
    ----------
    dataset_name : str
        _description_
    config : dict
        _description_
    lmdb_paths : dict
        _description_

    Returns
    -------
    dict
        _description_
    """
    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    split_sizes = {}
    label_counts = {}

    for split in SPLITS:
        # Parse official records, normalize schema, and enforce quality checks.
        rows = load_split_rows(dataset_name, split, lmdb_paths[split], config["target_field"])
        validate_rows(dataset_name, split, rows)
        csv_path = output_dir / f"{split}.csv"
        write_csv(csv_path, rows)

        split_sizes[split] = len(rows)
        counts = Counter(row["label"] for row in rows)
        label_counts[split] = {str(key): value for key, value in sorted(counts.items(), key=lambda item: str(item[0]))}

        print(f"[PEER] {dataset_name} {split}: {len(rows)} samples")
        print(f"[PEER] {dataset_name} {split} label counts: {label_counts[split]}")

    return {
        "dataset_name": dataset_name,
        "peer_dataset_class": config["peer_dataset_class"],
        "source_repo_path": str(PEER_REPO),
        "source_config": str(config["peer_config"]),
        "source_url": config["url"],
        "timestamp": utc_now(),
        "split_sizes": split_sizes,
        "label_counts_per_split": label_counts,
        "raw_lmdb_paths": {split: str(path) for split, path in lmdb_paths.items()},
    }


def load_split_rows(dataset_name: str, split: str, lmdb_path: Path, target_field: str) -> list:
    """Read one LMDB split and return normalized rows for CSV export.

    Expected LMDB structure follows TorchDrug ProteinDataset.load_lmdbs:
    - key `num_examples` stores the split size,
    - keys `0..num_examples-1` store pickled dict items,
    - `primary` is sequence and `target_field` is the benchmark label.

    Parameters
    ----------
    dataset_name : str
        _description_
    split : str
        _description_
    lmdb_path : Path
        _description_
    target_field : str
        _description_

    Returns
    -------
    list
        _description_

    Raises
    ------
    FileNotFoundError
        _description_
    KeyError
        _description_
    KeyError
        _description_
    """
    if not lmdb_path.exists():
        raise FileNotFoundError(f"Expected LMDB file for {dataset_name} {split} at {lmdb_path}, but it does not exist.")

    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False)
    rows = []
    try:
        with env.begin(write=False) as txn:
            raw_num_examples = txn.get(b"num_examples")
            if raw_num_examples is None:
                raise KeyError(f"LMDB {lmdb_path} does not contain the `num_examples` key.")
            num_examples = pickle.loads(raw_num_examples)

            for idx in range(num_examples):
                raw_item = txn.get(str(idx).encode("utf-8"))
                if raw_item is None:
                    raise KeyError(f"LMDB {lmdb_path} is missing record {idx}.")

                item = pickle.loads(raw_item)
                # Normalize to consistent text sequence and scalar label values.
                sequence = normalize_sequence(item.get("primary"))
                label = normalize_label(item.get(target_field))

                rows.append(
                    {
                        "idx": idx,
                        "sequence": sequence,
                        "label": label,
                        "split": split,
                        "dataset": dataset_name,
                    }
                )
    finally:
        env.close()

    return rows


def validate_rows(dataset_name: str, split: str, rows: list) -> None:
    """Apply required integrity checks before writing split CSV files."""
    if not rows:
        raise ValueError(f"Official PEER split {dataset_name}/{split} is empty, which should never happen.")

    missing_sequences = [row["idx"] for row in rows if not row["sequence"]]
    if missing_sequences:
        raise ValueError(
            f"Found {len(missing_sequences)} missing sequences in {dataset_name}/{split}. "
            f"Example indices: {missing_sequences[:5]}"
        )

    missing_labels = [row["idx"] for row in rows if is_missing_label(row["label"])]
    if missing_labels:
        raise ValueError(
            f"Found {len(missing_labels)} missing labels in {dataset_name}/{split}. "
            f"Example indices: {missing_labels[:5]}"
        )


def write_csv(path: Path, rows: list) -> None:
    """Write normalized rows with a stable, explicit column order.


    Parameters
    ----------
    path : Path
        _description_
    rows : list
        _description_
    """
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["idx", "sequence", "label", "split", "dataset"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[PEER] Wrote {path}")


def download_file(url: str, destination: Path) -> None:
    """Download a remote file with chunked streaming to limit memory usage.


    Parameters
    ----------
    url : str
        _description_
    destination : Path
        _description_
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as output_handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output_handle.write(chunk)


def verify_md5(path: Path, expected_md5: str) -> None:
    """Verify the downloaded archive matches the official checksum.


    Parameters
    ----------
    path : Path
        _description_
    expected_md5 : str
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    actual_md5 = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            actual_md5.update(chunk)
    digest = actual_md5.hexdigest()
    if digest != expected_md5:
        raise ValueError(f"MD5 mismatch for {path}. Expected {expected_md5}, found {digest}.")


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    """Safely extract a gzipped tar archive into destination.

    We validate every member path to avoid path traversal and use the tarfile
    extraction filter when available (Python >=3.12 behavior).
    

    Parameters
    ----------
    archive_path : Path
        _description_
    destination : Path
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = destination / member.name
            if not is_within_directory(destination, member_path):
                raise ValueError(f"Refusing to extract unsafe path {member.name} from {archive_path}")
        try:
            archive.extractall(destination, filter="data")
        except TypeError:
            archive.extractall(destination)


def is_within_directory(directory: Path, target: Path) -> bool:
    """Return True if target resolves within directory.


    Parameters
    ----------
    directory : Path
        _description_
    target : Path
        _description_

    Returns
    -------
    bool
        _description_
    """
    directory = directory.resolve()
    target = target.resolve()
    try:
        target.relative_to(directory)
    except ValueError:
        return False
    return True


def normalize_sequence(value) -> str:
    """Convert sequence field to a stripped string, preserving empty as ''.


    Parameters
    ----------
    value : _type_
        _description_

    Returns
    -------
    str
        _description_
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).strip()


def normalize_label(value):
    """Convert target values to plain Python scalars and map NaN to None.

    Parameters
    ----------
    value : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except ValueError:
            pass

    if isinstance(value, bytes):
        value = value.decode("utf-8")

    if isinstance(value, float) and math.isnan(value):
        return None

    return value


def is_missing_label(value) -> bool:
    """Return True when a label is None or blank text.


    Parameters
    ----------
    value : _type_
        _description_

    Returns
    -------
    bool
        _description_
    """
    return value is None or (isinstance(value, str) and not value.strip())


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp for metadata snapshots.


    Returns
    -------
    str
        _description_
    """
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()