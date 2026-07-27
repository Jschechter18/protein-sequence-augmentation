"""Run metadata, fingerprinting, and artifact lifecycle helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

import torch


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@lru_cache(maxsize=64)
def _cached_file_sha256(
    resolved_path: str,
    size_bytes: int,
    mtime_ns: int,
    ctime_ns: int,
) -> str:
    del size_bytes, mtime_ns, ctime_ns
    digest = hashlib.sha256()
    with Path(resolved_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    resolved = file_path.resolve()
    stat = resolved.stat()
    return _cached_file_sha256(
        str(resolved), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    )


def data_source_metadata(
    config: Any,
    resolve_split_source: Callable[[Any, str], tuple[Path, bool]],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for split in ("train", "valid", "test"):
        path, is_combined = resolve_split_source(config, split)
        resolved = path.resolve()
        metadata[split] = {
            "path": str(resolved),
            "sha256": file_sha256(resolved),
            "combined_file": is_combined,
            "combined_split_value": split if is_combined else None,
        }
    return metadata


def git_metadata(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def runtime_metadata() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for distribution in ("fair-esm", "numpy", "pandas", "scikit-learn", "torch"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda_version": torch.version.cuda,
    }


def config_payload(
    config: Any,
    *,
    project_root: Path,
    fingerprinted_source_files: Iterable[str],
    embedding_cache_schema_version: int,
    embedding_cache_path: Callable[[Any, str], tuple[Path, dict[str, Any]]],
    resolve_split_source: Callable[[Any, str], tuple[Path, bool]],
) -> dict[str, Any]:
    payload = asdict(config)
    payload.update(
        {
            "run_dir": str(config.run_dir),
            "checkpoint_dir": str(config.checkpoint_dir),
            "autoencoder_checkpoint_sha256": file_sha256(
                config.autoencoder_checkpoint
            ),
            "data_sources": data_source_metadata(config, resolve_split_source),
            "source_file_sha256": {
                relative_path: file_sha256(project_root / relative_path)
                for relative_path in fingerprinted_source_files
            },
            "preprocessing": {
                "classification_encoding": "char",
                "autoencoder_special_tokens": "BOS+residues+EOS",
                "esm_long_sequence_policy": "truncate_right",
                "esm_max_sequence_length": config.esm_max_sequence_length,
                "embedding_cache": {
                    "enabled": config.cache_embeddings,
                    "root": config.embedding_cache_root,
                    "schema_version": embedding_cache_schema_version,
                    "files": {
                        split: str(embedding_cache_path(config, split)[0])
                        for split in (
                            ["train", "valid", "test"]
                            if config.evaluate_test
                            else ["train", "valid"]
                        )
                    }
                    if config.cache_embeddings
                    else {},
                },
            },
            "runtime": runtime_metadata(),
            **git_metadata(project_root),
        }
    )
    exact_material = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["configuration_fingerprint"] = hashlib.sha256(
        exact_material.encode("utf-8")
    ).hexdigest()
    resume_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"epochs", "evaluate_test", "configuration_fingerprint"}
    }
    resume_material = json.dumps(resume_payload, sort_keys=True, separators=(",", ":"))
    payload["resume_fingerprint"] = hashlib.sha256(
        resume_material.encode("utf-8")
    ).hexdigest()
    return payload


def validate_existing_config(
    existing: dict[str, Any],
    requested: dict[str, Any],
    *,
    for_resume: bool,
) -> None:
    fingerprint_name = "resume_fingerprint" if for_resume else "configuration_fingerprint"
    existing_fingerprint = existing.get(fingerprint_name)
    requested_fingerprint = requested[fingerprint_name]
    if existing_fingerprint != requested_fingerprint:
        action = "resume" if for_resume else "reuse"
        raise ValueError(
            f"Refusing to {action} {requested['run_dir']}: its saved configuration "
            "does not match the requested code, data, checkpoint, preprocessing, or "
            "hyperparameters. Use --overwrite to archive it and start a new run."
        )
    if for_resume and int(requested["epochs"]) < int(existing.get("epochs", 0)):
        raise ValueError(
            "A resumed run may preserve or extend its epoch budget, but may not "
            "reduce it. Use --overwrite for a shorter run."
        )


def status_path(run_dir: Path) -> Path:
    return run_dir / "status.json"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return result


def is_complete(
    run_dir: Path,
    evaluate_test: bool,
    checkpoint_dir: Path | None = None,
) -> bool:
    checkpoint_dir = checkpoint_dir or run_dir
    required = [
        run_dir / "config.json",
        run_dir / "history.csv",
        checkpoint_dir / "best_model.pt",
    ]
    if evaluate_test:
        required.extend([run_dir / "metrics.json", run_dir / "test_predictions.csv"])
    run_status_path = status_path(run_dir)
    if not run_status_path.is_file() or not all(path.is_file() for path in required):
        return False
    try:
        return read_json(run_status_path).get("status") == "complete"
    except (OSError, ValueError, TypeError):
        return False


def archive_run_dir(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = run_dir.with_name(f"{run_dir.name}.backup_{suffix}")
    counter = 1
    while archive.exists():
        archive = run_dir.with_name(f"{run_dir.name}.backup_{suffix}_{counter}")
        counter += 1
    run_dir.rename(archive)


def attach_run_log(run_dir: Path) -> logging.Handler:
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(handler)
    return handler
