"""Merge classifier tuning trials collected from multiple machines.

The script treats each trial's config/status/history files as authoritative. It
does not consume per-machine tuning_results.csv or selected_hyperparameters.json
files because those contain only local subsets of the search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from Code.src.training.train_classifier import (
    DEFAULT_SELECTED_HYPERPARAMETERS,
    HEAD_TYPES,
    TUNING_LEARNING_RATES,
    TUNING_MLP_DROPOUTS,
    TUNING_REPRESENTATIONS,
    TUNING_SEED,
    TUNING_WEIGHT_DECAYS,
)
from Code.src.utils.classifier_tuning_results import (
    select_tuning_hyperparameters,
    tuning_metrics_from_history,
)


class TuningMergeError(ValueError):
    """Raised when tuning artifacts cannot be merged safely."""


@dataclass(frozen=True)
class TrialCandidate:
    source: str
    source_root: Path
    run_dir: Path
    config: dict[str, Any]
    status_payload: dict[str, Any]
    status: str
    metrics: dict[str, Any] | None
    history_sha256: str | None
    identity: tuple[Any, ...]
    relative_destination: Path


IDENTITY_FIELDS = (
    "dataset",
    "version",
    "phase",
    "representation",
    "encoder_mode",
    "freeze_autoencoder",
    "freeze_esm2",
    "autoencoder_version",
    "head_type",
    "seed",
    "learning_rate",
    "weight_decay",
    "dropout",
    "autoencoder_embedding_dropout",
    "esm_embedding_dropout",
)

PROVENANCE_FIELDS = (
    "dataset",
    "version",
    "phase",
    "mode",
    "seed",
    "num_classes",
    "batch_size",
    "epochs",
    "early_stopping_patience",
    "deterministic",
    "autoencoder_checkpoint_sha256",
    "autoencoder_embedding_dim",
    "autoencoder_cnn_channels",
    "autoencoder_hidden_dim",
    "autoencoder_latent_dim",
    "autoencoder_num_layers",
    "autoencoder_kernel_size",
    "autoencoder_layer_type",
    "esm_model_name",
    "esm_max_sequence_length",
    "encoder_learning_rate",
    "esm_learning_rate",
    "max_grad_norm",
    "git_commit",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise TuningMergeError(f"Could not read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TuningMergeError(f"Expected a JSON object in {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_version(value: Any) -> str:
    version = str(value)
    return version if version.startswith("v") else f"v{version}"


def _format_trial_value(value: float) -> str:
    if value == 0:
        return "0"
    formatted = f"{value:.0e}"
    return formatted.replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")


def _canonical_trial_name(config: dict[str, Any]) -> str:
    head_type = str(config.get("head_type"))
    parts = [
        f"lr_{_format_trial_value(float(config['learning_rate']))}",
        f"wd_{_format_trial_value(float(config['weight_decay']))}",
    ]
    if head_type == "mlp":
        if config.get("dropout") is None:
            raise TuningMergeError("An MLP tuning config is missing dropout")
        parts.append(f"do_{float(config['dropout']):g}")
    if config.get("phase") == "end_to_end_tuning":
        parts.extend(
            [
                f"ae_do_{float(config['autoencoder_embedding_dropout']):g}",
                f"esm_do_{float(config['esm_embedding_dropout']):g}",
            ]
        )
    parts.append(f"seed_{int(config['seed'])}")
    return "_".join(parts)


def _encoder_state(config: dict[str, Any]) -> tuple[str, bool, bool]:
    phase = str(config.get("phase"))
    representation = str(config.get("representation"))
    default_mode = (
        "frozen"
        if phase == "tuning"
        else "from_scratch"
        if representation == "random_autoencoder"
        else "fine_tuned"
    )
    return (
        str(config.get("encoder_mode", default_mode)),
        bool(config.get("freeze_autoencoder", phase == "tuning")),
        bool(config.get("freeze_esm2", phase == "tuning")),
    )


def _candidate_identity(config: dict[str, Any]) -> tuple[Any, ...]:
    encoder_mode, freeze_autoencoder, freeze_esm2 = _encoder_state(config)
    values = {
        "dataset": str(config["dataset"]),
        "version": _normalized_version(config["version"]),
        "phase": str(config["phase"]),
        "representation": str(config["representation"]),
        "encoder_mode": encoder_mode,
        "freeze_autoencoder": freeze_autoencoder,
        "freeze_esm2": freeze_esm2,
        "autoencoder_version": config.get("autoencoder_version"),
        "head_type": str(config["head_type"]),
        "seed": int(config["seed"]),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "dropout": (
            None if config.get("dropout") is None else float(config["dropout"])
        ),
        "autoencoder_embedding_dropout": float(
            config.get("autoencoder_embedding_dropout", 0.0)
        ),
        "esm_embedding_dropout": float(
            config.get("esm_embedding_dropout", 0.0)
        ),
    }
    return tuple(values[field] for field in IDENTITY_FIELDS)


def _read_completed_history(run_dir: Path) -> tuple[dict[str, Any], str]:
    history_path = run_dir / "history.csv"
    if not history_path.is_file():
        raise TuningMergeError(
            f"Trial is marked complete but history.csv is missing: {run_dir}"
        )
    try:
        history = pd.read_csv(history_path, float_precision="round_trip")
    except (OSError, pd.errors.ParserError) as error:
        raise TuningMergeError(f"Could not read {history_path}: {error}") from error
    required = {"epoch", "val_f1", "val_loss", "val_accuracy"}
    missing = required.difference(history.columns)
    if history.empty or missing:
        detail = "empty" if history.empty else f"missing columns {sorted(missing)}"
        raise TuningMergeError(f"Invalid completed history {history_path}: {detail}")
    for column in required:
        values = pd.to_numeric(history[column], errors="coerce")
        if not all(math.isfinite(float(value)) for value in values):
            raise TuningMergeError(
                f"Invalid completed history {history_path}: {column} is not finite"
            )
    metrics = tuning_metrics_from_history(history.to_dict(orient="records"))
    return metrics, _sha256(history_path)


def _discover_source(
    source_root: Path, source: str, phase: str
) -> list[TrialCandidate]:
    candidates: list[TrialCandidate] = []
    for config_path in sorted(source_root.rglob("config.json")):
        relative_parts = config_path.relative_to(source_root).parts
        if any(
            part.startswith(".") or ".backup_" in part
            for part in relative_parts
        ):
            continue
        config = _read_json(config_path)
        if config.get("phase") != phase:
            continue
        required = {
            "dataset",
            "version",
            "representation",
            "head_type",
            "seed",
            "learning_rate",
            "weight_decay",
        }
        missing = required.difference(config)
        if missing:
            raise TuningMergeError(
                f"Tuning config {config_path} is missing fields {sorted(missing)}"
            )
        if config["head_type"] not in {"linear", "mlp"}:
            raise TuningMergeError(
                f"Unsupported head_type in {config_path}: {config['head_type']!r}"
            )
        representation = str(config["representation"])
        if Path(representation).name != representation or representation in {".", ".."}:
            raise TuningMergeError(
                f"Invalid representation in {config_path}: {representation!r}"
            )
        run_dir = config_path.parent
        status_path = run_dir / "status.json"
        status_payload = _read_json(status_path) if status_path.is_file() else {}
        status = str(status_payload.get("status", "incomplete"))
        if status not in {"complete", "failed", "running"}:
            status = "incomplete"
        metrics: dict[str, Any] | None = None
        history_sha256: str | None = None
        if status == "complete":
            metrics, history_sha256 = _read_completed_history(run_dir)
        relative_destination = (
            Path(str(config["head_type"]))
            / str(config["representation"])
            / _canonical_trial_name(config)
        )
        candidates.append(
            TrialCandidate(
                source=source,
                source_root=source_root,
                run_dir=run_dir,
                config=config,
                status_payload=status_payload,
                status=status,
                metrics=metrics,
                history_sha256=history_sha256,
                identity=_candidate_identity(config),
                relative_destination=relative_destination,
            )
        )
    if not candidates:
        raise TuningMergeError(
            f"No {phase!r} trial configs were found under {source_root}"
        )
    return candidates


def _provenance(
    config: dict[str, Any], *, ignore_partition_driver: bool = False
) -> dict[str, Any]:
    data_sources = {
        split: {
            "sha256": metadata.get("sha256"),
            "combined_file": metadata.get("combined_file"),
            "combined_split_value": metadata.get("combined_split_value"),
        }
        for split, metadata in config.get("data_sources", {}).items()
        if isinstance(metadata, dict)
    }
    preprocessing = config.get("preprocessing", {})
    embedding_cache = preprocessing.get("embedding_cache", {})
    runtime = config.get("runtime", {})
    result = {field: config.get(field) for field in PROVENANCE_FIELDS}
    source_hashes = dict(config.get("source_file_sha256", {}))
    if ignore_partition_driver:
        result.pop("git_commit", None)
        source_hashes = {
            path: digest
            for path, digest in source_hashes.items()
            if not str(path).endswith("Code/src/training/train_classifier.py")
        }
    result.update(
        {
            "version": _normalized_version(config.get("version")),
            "data_sources": data_sources,
            "source_file_sha256": source_hashes,
            "preprocessing": {
                "classification_encoding": preprocessing.get(
                    "classification_encoding"
                ),
                "autoencoder_special_tokens": preprocessing.get(
                    "autoencoder_special_tokens"
                ),
                "esm_long_sequence_policy": preprocessing.get(
                    "esm_long_sequence_policy"
                ),
                "esm_max_sequence_length": preprocessing.get(
                    "esm_max_sequence_length"
                ),
                "embedding_cache_schema_version": embedding_cache.get(
                    "schema_version"
                ),
            },
            "package_versions": runtime.get("packages", {}),
            "python_version": runtime.get("python"),
            "platform": runtime.get("platform"),
            "torch_cuda_version": runtime.get("torch_cuda_version"),
        }
    )
    return result


def _json_digest(payload: Any) -> str:
    material = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _validate_provenance(
    candidates: list[TrialCandidate], *, allow_partition_source_mismatch: bool
) -> list[str]:
    signatures: dict[str, tuple[dict[str, Any], TrialCandidate]] = {}
    for candidate in candidates:
        provenance = _provenance(
            candidate.config,
            ignore_partition_driver=allow_partition_source_mismatch,
        )
        signatures.setdefault(_json_digest(provenance), (provenance, candidate))
    if len(signatures) <= 1:
        return sorted(signatures)

    examples = [candidate for _, candidate in signatures.values()]
    baseline = _provenance(
        examples[0].config,
        ignore_partition_driver=allow_partition_source_mismatch,
    )
    differing_fields: set[str] = set()
    for candidate in examples[1:]:
        other = _provenance(
            candidate.config,
            ignore_partition_driver=allow_partition_source_mismatch,
        )
        differing_fields.update(
            field for field in baseline if baseline[field] != other[field]
        )
    raise TuningMergeError(
        "Input trials have incompatible provenance in fields "
        f"{sorted(differing_fields)}. Use the same code, data, checkpoints, and "
        "environment on every instance."
    )


def _deduplicate(
    candidates: list[TrialCandidate],
) -> tuple[list[TrialCandidate], dict[tuple[Any, ...], list[str]]]:
    grouped: dict[tuple[Any, ...], list[TrialCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.identity].append(candidate)

    selected: list[TrialCandidate] = []
    duplicate_sources: dict[tuple[Any, ...], list[str]] = {}
    for identity, attempts in grouped.items():
        attempts = sorted(attempts, key=lambda attempt: str(attempt.run_dir.resolve()))
        completed = [attempt for attempt in attempts if attempt.status == "complete"]
        if len(completed) > 1:
            history_hashes = {attempt.history_sha256 for attempt in completed}
            if len(history_hashes) != 1:
                locations = [str(attempt.run_dir) for attempt in completed]
                raise TuningMergeError(
                    "Conflicting completed attempts have the same trial identity: "
                    f"{locations}"
                )
            chosen = completed[0]
        elif completed:
            chosen = completed[0]
        else:
            chosen = attempts[0]
        selected.append(chosen)
        duplicate_sources[identity] = sorted({attempt.source for attempt in attempts})

    destinations: dict[Path, TrialCandidate] = {}
    for candidate in selected:
        prior = destinations.get(candidate.relative_destination)
        if prior is not None and prior.identity != candidate.identity:
            raise TuningMergeError(
                "Distinct trial identities map to the same destination: "
                f"{candidate.relative_destination}"
            )
        destinations[candidate.relative_destination] = candidate
    selected.sort(
        key=lambda candidate: tuple(str(value) for value in candidate.identity)
    )
    return selected, duplicate_sources


def _validate_full_frozen_grid(candidates: list[TrialCandidate]) -> None:
    expected: set[tuple[Any, ...]] = set()
    for head_type in HEAD_TYPES:
        dropouts: tuple[float | None, ...] = (
            TUNING_MLP_DROPOUTS if head_type == "mlp" else (None,)
        )
        for representation in TUNING_REPRESENTATIONS:
            for learning_rate in TUNING_LEARNING_RATES:
                for weight_decay in TUNING_WEIGHT_DECAYS:
                    for dropout in dropouts:
                        expected.add(
                            (
                                head_type,
                                representation,
                                TUNING_SEED,
                                learning_rate,
                                weight_decay,
                                dropout,
                                0.0,
                                0.0,
                                "frozen",
                                True,
                                True,
                            )
                        )

    actual: set[tuple[Any, ...]] = set()
    for candidate in candidates:
        config = candidate.config
        encoder_mode, freeze_autoencoder, freeze_esm2 = _encoder_state(config)
        actual.add(
            (
                str(config["head_type"]),
                str(config["representation"]),
                int(config["seed"]),
                float(config["learning_rate"]),
                float(config["weight_decay"]),
                (
                    None
                    if config.get("dropout") is None
                    else float(config["dropout"])
                ),
                float(config.get("autoencoder_embedding_dropout", 0.0)),
                float(config.get("esm_embedding_dropout", 0.0)),
                encoder_mode,
                freeze_autoencoder,
                freeze_esm2,
            )
        )

    if actual == expected and len(candidates) == len(expected):
        return
    missing = sorted(expected.difference(actual), key=str)
    unexpected = sorted(actual.difference(expected), key=str)
    raise TuningMergeError(
        "Frozen tuning grid does not match the declared 72-trial Cartesian grid: "
        f"missing={len(missing)}, unexpected={len(unexpected)}, "
        f"duplicate_coordinates={len(candidates) - len(actual)}. "
        f"Missing examples={missing[:3]}; unexpected examples={unexpected[:3]}"
    )


def _tree_digest(root: Path) -> str:
    if root.is_symlink():
        raise TuningMergeError(f"Symlinked trial directories are not supported: {root}")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise TuningMergeError(
                f"Symlinks are not supported in trial artifacts: {path}"
            )
        digest.update(str(relative).encode("utf-8"))
        if path.is_file():
            digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _copy_trials(candidates: Iterable[TrialCandidate], output_dir: Path) -> int:
    candidates = list(candidates)
    resolved_output = output_dir.resolve()
    expected_destinations = {
        (output_dir / candidate.relative_destination).resolve()
        for candidate in candidates
    }
    escaped_destinations = [
        destination
        for destination in expected_destinations
        if not destination.is_relative_to(resolved_output)
    ]
    if escaped_destinations:
        raise TuningMergeError(
            "A symlinked output path escapes the requested output directory: "
            f"{escaped_destinations[0]}"
        )
    if output_dir.is_dir():
        for config_path in output_dir.rglob("config.json"):
            relative_parts = config_path.relative_to(output_dir).parts
            if any(
                part.startswith(".") or ".backup_" in part
                for part in relative_parts
            ):
                continue
            if config_path.parent.resolve() not in expected_destinations:
                raise TuningMergeError(
                    "Output contains a trial not present in the inputs: "
                    f"{config_path.parent}"
                )

    pending: list[tuple[TrialCandidate, Path]] = []
    for candidate in candidates:
        destination = output_dir / candidate.relative_destination
        source_digest = _tree_digest(candidate.run_dir)
        if destination.exists():
            if not destination.is_dir() or _tree_digest(destination) != source_digest:
                raise TuningMergeError(
                    f"Refusing to overwrite conflicting destination {destination}"
                )
            continue
        pending.append((candidate, destination))

    copied = 0
    for candidate, destination in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(
            tempfile.mkdtemp(prefix=".tuning_merge_", dir=destination.parent)
        )
        temporary_trial = temporary_root / destination.name
        try:
            shutil.copytree(candidate.run_dir, temporary_trial)
            os.replace(temporary_trial, destination)
            copied += 1
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)
    return copied


def _row(
    candidate: TrialCandidate,
    output_dir: Path,
    sources: list[str],
) -> dict[str, Any]:
    config = candidate.config
    encoder_mode, freeze_autoencoder, freeze_esm2 = _encoder_state(config)
    merged_run_dir = output_dir / candidate.relative_destination
    row: dict[str, Any] = {
        "dataset": str(config["dataset"]),
        "version": _normalized_version(config["version"]),
        "representation": str(config["representation"]),
        "encoder_mode": encoder_mode,
        "freeze_autoencoder": freeze_autoencoder,
        "freeze_esm2": freeze_esm2,
        "autoencoder_version": config.get("autoencoder_version"),
        "head_type": str(config["head_type"]),
        "seed": int(config["seed"]),
        "phase": str(config["phase"]),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "dropout": config.get("dropout"),
        "autoencoder_embedding_dropout": config.get(
            "autoencoder_embedding_dropout", 0.0
        ),
        "esm_embedding_dropout": config.get("esm_embedding_dropout", 0.0),
        "status": candidate.status,
        "run_dir": str(merged_run_dir) if candidate.status == "complete" else None,
        "artifacts_copied": candidate.status == "complete",
        "checkpoint_dir": config.get("checkpoint_dir"),
        "error": candidate.status_payload.get("error"),
        "sources": ";".join(sources),
        "source_run_dir": str(candidate.run_dir),
    }
    if candidate.metrics is not None:
        row.update(candidate.metrics)
    return row


def _atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary_name, index=False)
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def merge_tuning_results(
    input_dirs: Iterable[str | Path],
    output_dir: str | Path,
    *,
    phase: str = "tuning",
    expected_trials: int | None = None,
    expect_full_grid: bool = False,
    require_complete: bool = True,
    allow_partition_source_mismatch: bool = False,
) -> dict[str, Any]:
    """Merge trial trees and rebuild the global tuning summary and winners."""

    if phase not in {"tuning", "end_to_end_tuning"}:
        raise TuningMergeError(f"Unsupported tuning phase: {phase!r}")
    sources = [Path(path).expanduser().resolve() for path in input_dirs]
    if not sources:
        raise TuningMergeError("At least one input directory is required")
    for source in sources:
        if not source.is_dir():
            raise TuningMergeError(f"Input directory does not exist: {source}")
    destination_root = Path(output_dir).expanduser()
    resolved_destination = destination_root.resolve()
    for source in sources:
        if resolved_destination == source or resolved_destination.is_relative_to(
            source
        ):
            raise TuningMergeError(
                f"Output directory must not be inside an input directory: {source}"
            )

    all_candidates: list[TrialCandidate] = []
    source_counts: dict[str, int] = {}
    for index, source_root in enumerate(sources, start=1):
        source = f"source_{index}"
        discovered = _discover_source(source_root, source, phase)
        source_counts[source] = len(discovered)
        all_candidates.extend(discovered)

    datasets = {str(candidate.config["dataset"]) for candidate in all_candidates}
    versions = {
        _normalized_version(candidate.config["version"])
        for candidate in all_candidates
    }
    if len(datasets) != 1 or len(versions) != 1:
        raise TuningMergeError(
            "All inputs must have one dataset/version; found "
            f"datasets={sorted(datasets)}, versions={sorted(versions)}"
        )

    provenance_signatures = _validate_provenance(
        all_candidates,
        allow_partition_source_mismatch=allow_partition_source_mismatch,
    )
    candidates, duplicate_sources = _deduplicate(all_candidates)
    if expect_full_grid:
        if phase != "tuning":
            raise TuningMergeError(
                "Full frozen-grid validation is available only for phase='tuning'"
            )
        _validate_full_frozen_grid(candidates)
    if expected_trials is not None and len(candidates) != expected_trials:
        raise TuningMergeError(
            f"Expected {expected_trials} unique trials, found {len(candidates)}"
        )
    incomplete = [
        candidate for candidate in candidates if candidate.status != "complete"
    ]
    if require_complete and incomplete:
        counts = Counter(candidate.status for candidate in incomplete)
        raise TuningMergeError(
            "Refusing to publish winners with incomplete trials: "
            + ", ".join(
                f"{status}={count}" for status, count in sorted(counts.items())
            )
        )

    copied = _copy_trials(
        [candidate for candidate in candidates if candidate.status == "complete"],
        destination_root,
    )
    rows = [
        _row(candidate, destination_root, duplicate_sources[candidate.identity])
        for candidate in candidates
    ]
    summary = pd.DataFrame(rows)
    sort_columns = list(IDENTITY_FIELDS)
    summary = summary.sort_values(sort_columns, kind="stable", na_position="first")
    requested_conditions = {
        (str(candidate.config["head_type"]), str(candidate.config["representation"]))
        for candidate in candidates
    }
    selected = select_tuning_hyperparameters(
        summary,
        requested_conditions=requested_conditions,
        is_end_to_end_tuning=phase == "end_to_end_tuning",
        default_selected=DEFAULT_SELECTED_HYPERPARAMETERS,
    )

    status_counts = Counter(candidate.status for candidate in candidates)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": next(iter(datasets)),
        "version": next(iter(versions)),
        "phase": phase,
        "input_dirs": [str(source) for source in sources],
        "source_trial_counts": source_counts,
        "num_attempts": len(all_candidates),
        "num_unique_trials": len(candidates),
        "num_copied_trials": copied,
        "status_counts": dict(sorted(status_counts.items())),
        "expected_trials": expected_trials,
        "expected_full_frozen_grid": expect_full_grid,
        "require_complete": require_complete,
        "partition_source_mismatch_allowed": allow_partition_source_mismatch,
        "partition_source_audit": {
            "git_commits": sorted(
                {
                    str(candidate.config.get("git_commit"))
                    for candidate in all_candidates
                }
            ),
            "train_classifier_sha256": sorted(
                {
                    str(digest)
                    for candidate in all_candidates
                    for path, digest in candidate.config.get(
                        "source_file_sha256", {}
                    ).items()
                    if str(path).endswith("Code/src/training/train_classifier.py")
                }
            ),
        },
        "provenance_signatures": provenance_signatures,
    }
    _atomic_csv(summary, destination_root / "tuning_results.csv")
    _atomic_json(selected, destination_root / "selected_hyperparameters.json")
    _atomic_json(manifest, destination_root / "merge_manifest.json")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge classifier tuning outputs collected from multiple machines."
    )
    parser.add_argument(
        "--input_dir",
        "--input-dir",
        action="append",
        required=True,
        help="Collected result tree from one instance; repeat for each instance.",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        required=True,
        help=(
            "Canonical tuning directory to create, such as "
            ".../v4/frozen/tuning or .../v4/unfrozen/tuning."
        ),
    )
    parser.add_argument(
        "--phase",
        choices=("tuning", "end_to_end_tuning"),
        default="tuning",
    )
    parser.add_argument(
        "--expected_trials",
        "--expected-trials",
        type=int,
        default=None,
        help="Fail unless this many unique trials are present (72 for the full frozen grid).",
    )
    parser.add_argument(
        "--expect_full_grid",
        "--expect-full-grid",
        action="store_true",
        help="Require the exact declared 72-trial frozen Cartesian grid.",
    )
    parser.add_argument(
        "--allow_incomplete",
        "--allow-incomplete",
        action="store_true",
        help="Write a partial audit/selection even when trials are not complete.",
    )
    parser.add_argument(
        "--allow_partition_source_mismatch",
        "--allow-partition-source-mismatch",
        action="store_true",
        help=(
            "Ignore only Git commit and train_classifier.py hash differences "
            "after auditing manually edited learning-rate partitions."
        ),
    )
    args = parser.parse_args(argv)
    if args.expected_trials is not None and args.expected_trials <= 0:
        parser.error("--expected_trials must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        manifest = merge_tuning_results(
            args.input_dir,
            args.output_dir,
            phase=args.phase,
            expected_trials=args.expected_trials,
            expect_full_grid=args.expect_full_grid,
            require_complete=not args.allow_incomplete,
            allow_partition_source_mismatch=args.allow_partition_source_mismatch,
        )
    except TuningMergeError as error:
        raise SystemExit(f"error: {error}") from error
    print(
        f"Merged {manifest['num_unique_trials']} trials into {args.output_dir} "
        f"({manifest['num_copied_trials']} copied)."
    )


if __name__ == "__main__":
    main()
