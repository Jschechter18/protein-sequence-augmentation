"""
Test script for evaluating a trained autoencoder checkpoint.

Usage:
    python Code/src/testing/test_autoencoder.py --model AE --task solubility --checkpoint checkpoints/autoencoder/solubility/<version>/model_ae_solubility.pt  --teacher_forcing True
    
    python Code/src/testing/test_autoencoder.py --model AE --task solubility --version 6 --length_quartile ml --cumulative_quartiles True
"""

import csv
import sys
from pathlib import Path
from torch.utils.data import DataLoader

import torch
import torch.nn as nn

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.autoencoder import ProteinSequenceAutoencoder as AE
from utils.dataloader import (
    BOS_IDX,
    EOS_IDX,
    LENGTH_SPLIT_COUNTS,
    PAD_IDX,
    VOCAB,
    compute_train_length_boundaries,
    create_dataloader,
)
from utils.hyperparameters import AutoencoderHyperparameters as AEParams, autoencoder_sweep_suffix
from utils.test_input_validation import add_and_validate_test_inputs
from utils.train_input_validation import autoencoder_artifact_paths, autoencoder_artifact_stem

IDX_TO_TOKEN = {idx: token for token, idx in VOCAB.items()}
LENGTH_QUARTILE_FILE_LABELS = {
    "s": "short",
    "ms": "medium_short",
    "ml": "medium_long",
    "l": "long",
}

# ----------------------------------------------------------------------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps") # for macOS with Apple Silicon -> just for when you test locally
else:
    device = torch.device("cpu")

print()
print(f"Using device: {device}")
print()

# ----------------------------------------------------------------------------------------------------------------
def _version_dir_name(version: str | int | None) -> str | None:
    if version is None:
        return None
    version_name = str(version)
    return version_name if version_name.startswith("v") else f"v{version_name}"


def _format_sweep_value(value: int | float) -> str:
    if isinstance(value, float):
        value_label = f"{value:g}"
    else:
        value_label = str(value)
    return value_label.replace(".", "p").replace("-", "m")


def _sweep_suffix_component(name: str, value: int | float) -> str:
    labels = {
        "latent_dim": "latent",
        "teacher_forcing_dropout_rate": "tfd",
        "learning_rate": "lr",
        "lr_patience": "lrp",
        "scheduler_factor": "sf",
    }
    return f"{labels[name]}{_format_sweep_value(value)}"


def sweep_artifact_suffix(args) -> str | None:
    if args.latent_dim is None:
        return None

    components = [
        _sweep_suffix_component("latent_dim", args.latent_dim),
        _sweep_suffix_component("teacher_forcing_dropout_rate", args.teacher_forcing_dropout_rate),
    ]
    if args.learning_rate is not None:
        components.append(_sweep_suffix_component("learning_rate", args.learning_rate))
    if args.lr_patience is not None:
        components.append(_sweep_suffix_component("lr_patience", args.lr_patience))
    if args.scheduler_factor is not None:
        components.append(_sweep_suffix_component("scheduler_factor", args.scheduler_factor))

    return "_".join(components)


def legacy_sweep_artifact_suffix(args) -> str | None:
    if args.latent_dim is None:
        return None
    return autoencoder_sweep_suffix(
        args.latent_dim,
        args.teacher_forcing_dropout_rate,
        args.scheduler_factor,
    )


def sweep_checkpoint_match_pattern(
    model_type: str,
    task: str,
    length_options: str | None,
    length_bin: int | None,
) -> str:
    artifact_stem = autoencoder_artifact_stem(
        model_type,
        task,
        length_options,
        length_bin=length_bin,
        artifact_suffix=None,
    )
    return f"{artifact_stem}*.pt"


def resolve_matching_checkpoint(
    candidates: list[Path],
    expected_path: Path,
    artifact_suffix: str | None,
) -> Path | None:
    suffix_components = artifact_suffix.split("_") if artifact_suffix else []
    existing_candidates = sorted(
        path
        for path in candidates
        if path.is_file() and all(component in path.stem for component in suffix_components)
    )
    if len(existing_candidates) == 1:
        print(
            f"Exact checkpoint not found: {expected_path}\n"
            f"Using matching swept checkpoint: {existing_candidates[0]}"
        )
        return existing_candidates[0]
    if len(existing_candidates) > 1:
        formatted_candidates = "\n".join(f"  - {path}" for path in existing_candidates)
        raise FileNotFoundError(
            f"Multiple checkpoints matched the requested sweep arguments.\n"
            f"Expected exact path: {expected_path}\n"
            f"Matching checkpoints:\n{formatted_candidates}\n"
            "Add --learning_rate, --lr_patience, --scheduler_factor, or pass --checkpoint."
        )
    return None


def version_checkpoint_path(
    model_type: str,
    task: str,
    version: str | int | None,
    length_options: str | None = None,
    length_bin: int | None = None,
    artifact_suffix: str | None = None,
    fallback_artifact_suffix: str | None = None,
) -> Path:
    if version is None:
        raise ValueError("--version is required unless --checkpoint is provided")

    result_checkpoint_path, _ = autoencoder_artifact_paths(
        model_type,
        task,
        version,
        length_options,
        length_bin,
        is_overfit=False,
        artifact_suffix=artifact_suffix,
    )
    if result_checkpoint_path.is_file():
        return result_checkpoint_path

    fallback_result_checkpoint_path = None
    if fallback_artifact_suffix and fallback_artifact_suffix != artifact_suffix:
        fallback_result_checkpoint_path, _ = autoencoder_artifact_paths(
            model_type,
            task,
            version,
            length_options,
            length_bin,
            is_overfit=False,
            artifact_suffix=fallback_artifact_suffix,
        )
        if fallback_result_checkpoint_path.is_file():
            print(
                f"New sweep checkpoint not found: {result_checkpoint_path}\n"
                f"Falling back to legacy sweep checkpoint: {fallback_result_checkpoint_path}"
            )
            return fallback_result_checkpoint_path

    version_dir = _version_dir_name(version)
    legacy_checkpoint_dir = PROJECT_ROOT / "checkpoints" / "autoencoder" / task / version_dir
    checkpoint_name = f"{autoencoder_artifact_stem(model_type, task, length_options, length_bin=length_bin, artifact_suffix=artifact_suffix)}.pt"
    legacy_checkpoint_path = legacy_checkpoint_dir / checkpoint_name
    if legacy_checkpoint_path.is_file():
        print(
            f"Result checkpoint not found: {result_checkpoint_path}\n"
            f"Falling back to legacy checkpoint: {legacy_checkpoint_path}"
        )
        return legacy_checkpoint_path

    fallback_legacy_checkpoint_path = None
    if fallback_artifact_suffix and fallback_artifact_suffix != artifact_suffix:
        fallback_checkpoint_name = f"{autoencoder_artifact_stem(model_type, task, length_options, length_bin=length_bin, artifact_suffix=fallback_artifact_suffix)}.pt"
        fallback_legacy_checkpoint_path = legacy_checkpoint_dir / fallback_checkpoint_name
        if fallback_legacy_checkpoint_path.is_file():
            print(
                f"New sweep checkpoint not found: {result_checkpoint_path}\n"
                f"Falling back to legacy sweep checkpoint: {fallback_legacy_checkpoint_path}"
            )
            return fallback_legacy_checkpoint_path

    matching_checkpoint = resolve_matching_checkpoint(
        list(result_checkpoint_path.parent.glob(
            sweep_checkpoint_match_pattern(
                model_type,
                task,
                length_options,
                length_bin,
            )
        ))
        + list(legacy_checkpoint_dir.glob(
            sweep_checkpoint_match_pattern(
                model_type,
                task,
                length_options,
                length_bin,
            )
        )),
        result_checkpoint_path,
        artifact_suffix,
    )
    if matching_checkpoint is not None:
        return matching_checkpoint

    candidates = sorted(result_checkpoint_path.parent.glob(f"model_{model_type.lower()}*.pt"))
    candidates.extend(sorted(legacy_checkpoint_dir.glob(f"model_{model_type.lower()}*.pt")))
    if len(candidates) == 1:
        print(
            f"Exact checkpoint not found: {result_checkpoint_path}\n"
            f"Using the only available checkpoint for this version: {candidates[0]}"
        )
        return candidates[0]
    if candidates:
        formatted_candidates = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(
            f"No checkpoint matched the requested arguments for {task} {_version_dir_name(version)}.\n"
            f"Expected: {result_checkpoint_path}\n"
            f"Available checkpoints:\n{formatted_candidates}\n"
            "Provide the length/sweep arguments that identify one checkpoint, for example:\n"
            f"  --task {task} --version {version} --length_options halves --length_bin 1 "
            "--latent_dim 128 --teacher_forcing_dropout_rate 0.3\n"
            "Or pass an explicit checkpoint with --checkpoint."
        )

    raise FileNotFoundError(
        f"Checkpoint not found: {result_checkpoint_path}\n"
        f"Also checked legacy path: {legacy_checkpoint_path}"
    )


def legacy_quartile_checkpoint_path(
    model_type: str,
    task: str,
    version: str | int | None,
    length_quartile: str,
    artifact_suffix: str | None = None,
    fallback_artifact_suffix: str | None = None,
) -> Path:
    if version is None:
        raise ValueError("--version is required unless --checkpoint is provided")

    version_dir = _version_dir_name(version)
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "autoencoder" / task / version_dir
    artifact_stem = autoencoder_artifact_stem(
        model_type,
        task,
        artifact_suffix=artifact_suffix,
    )
    base_path = checkpoint_dir / f"{artifact_stem}.pt"
    quartile_label = LENGTH_QUARTILE_FILE_LABELS[length_quartile]
    quartile_path = checkpoint_dir / f"model_{model_type.lower()}_{quartile_label}_{task}.pt"
    if quartile_path.is_file():
        return quartile_path
    if base_path.is_file():
        print(
            f"Quartile-specific checkpoint not found: {quartile_path}\n"
            f"Falling back to base checkpoint: {base_path}"
        )
        return base_path
    if fallback_artifact_suffix and fallback_artifact_suffix != artifact_suffix:
        fallback_artifact_stem = autoencoder_artifact_stem(
            model_type,
            task,
            artifact_suffix=fallback_artifact_suffix,
        )
        fallback_base_path = checkpoint_dir / f"{fallback_artifact_stem}.pt"
        if fallback_base_path.is_file():
            print(
                f"New sweep checkpoint not found: {base_path}\n"
                f"Falling back to legacy sweep checkpoint: {fallback_base_path}"
            )
            return fallback_base_path
    return quartile_path


def default_checkpoint_path(
    model_type: str,
    task: str,
    version: str | int | None,
    length_quartile: str,
) -> Path:
    """Backward-compatible wrapper for older quartile checkpoint tests."""
    return legacy_quartile_checkpoint_path(model_type, task, version, length_quartile)


def checkpoint_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint

    raise TypeError("Checkpoint must be a state_dict or contain 'model_state_dict'.")


def rnn_num_layers(state_dict: dict[str, torch.Tensor], prefix: str) -> int:
    layer_ids: set[int] = set()

    for key in state_dict:
        if not key.startswith(f"{prefix}.weight_ih_l"):
            continue

        layer_name = key.removeprefix(f"{prefix}.weight_ih_l").removesuffix("_reverse")
        if layer_name.isdigit():
            layer_ids.add(int(layer_name))

    return max(layer_ids) + 1 if layer_ids else 1


def ae_params_from_state_dict(
    hyperparams: AEParams,
    state_dict: dict[str, torch.Tensor],
) -> dict:
    params = hyperparams.__dict__.copy()
    params.update(
        {
            "embedding_dim": state_dict["embedding.weight"].shape[1],
            "cnn_out_channels": state_dict["cnn.weight"].shape[0],
            "hidden_dim": state_dict["encoder.weight_hh_l0"].shape[1],
            "latent_dim": state_dict["to_latent.weight"].shape[0],
            "num_layers": rnn_num_layers(state_dict, "encoder"),
            "kernel_size": state_dict["cnn.weight"].shape[2],
            "bidirectional": "encoder.weight_ih_l0_reverse" in state_dict,
            "condition_decoder_on_latent": (
                state_dict["decoder.weight_ih_l0"].shape[1]
                == state_dict["embedding.weight"].shape[1] + state_dict["to_latent.weight"].shape[0]
            ),
            "pad_idx": PAD_IDX,
            "bos_idx": BOS_IDX,
        }
    )

    for training_only_param in (
        "learning_rate",
        "batch_size",
        "num_epochs",
        "shuffle",
        "patience",
        "lr_patience",
        "scheduler_factor",
    ):
        params.pop(training_only_param, None)

    return params


def model_definition(
    model_type: str,
    hyperparams: AEParams,
    checkpoint_path: str | Path,
) -> AE:
    model_type = model_type.lower()
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint_state_dict(checkpoint)

    if model_type == "ae":
        model = AE(**ae_params_from_state_dict(hyperparams, state_dict)).to(device)
    else:
        raise ValueError("Only --model AE is currently supported")

    model.load_state_dict(state_dict)
    return model


def decode_token_ids(token_ids: torch.Tensor) -> str:
    decoded_tokens: list[str] = []

    for token_id in token_ids.tolist():
        if token_id in {PAD_IDX, BOS_IDX}:
            continue
        if token_id == EOS_IDX:
            break

        decoded_tokens.append(IDX_TO_TOKEN.get(token_id, "<UNK>"))

    return "".join(decoded_tokens)


REQUIRED_RESULTS_COLUMNS = [
    "version",
    "file name",
    "test teacher force loss",
    "test autoregressive loss",
    "test teacher forced accuracy",
    "test autoregressive accuracy",
]


def output_path_for_mode(output_path: str | Path, mode: str) -> Path:
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}_{mode}{output_path.suffix}")


def infer_version(version: str | int | None, checkpoint_path: Path) -> str:
    if version is not None:
        return _version_dir_name(version) or str(version)

    for parent in [checkpoint_path.parent, *checkpoint_path.parents]:
        if parent.name.startswith("v") and parent.name[1:].isdigit():
            return parent.name

    raise ValueError("Could not infer version from checkpoint path; pass --version explicitly.")


def append_autoencoder_results(
    results_csv_path: str | Path,
    version: str,
    checkpoint_path: Path,
    teacher_forced_metrics: dict[str, float],
    autoregressive_metrics: dict[str, float],
) -> None:
    results_csv_path = Path(results_csv_path)
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    fieldnames = REQUIRED_RESULTS_COLUMNS.copy()
    if results_csv_path.exists():
        with results_csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                raise ValueError(f"Existing results CSV has no header: {results_csv_path}")
            fieldnames = []
            seen_fields: set[str] = set()
            for field in reader.fieldnames:
                if field in seen_fields:
                    continue
                fieldnames.append(field)
                seen_fields.add(field)
            for column in REQUIRED_RESULTS_COLUMNS:
                if column not in fieldnames:
                    fieldnames.append(column)
            rows = list(reader)

    result_row = {
        "version": version,
        "file name": str(checkpoint_path),
        "test teacher force loss": f"{teacher_forced_metrics['loss']:.6f}",
        "test autoregressive loss": f"{autoregressive_metrics['loss']:.6f}",
        "test teacher forced accuracy": f"{teacher_forced_metrics['accuracy']:.6f}",
        "test autoregressive accuracy": f"{autoregressive_metrics['accuracy']:.6f}",
    }

    matching_row = next(
        (
            row
            for row in rows
            if row.get("version") == version and row.get("file name") == str(checkpoint_path)
        ),
        None,
    )
    if matching_row is None:
        rows.append(result_row)
        print(f"Appending test results to: {results_csv_path}")
    else:
        matching_row.update(result_row)
        print(f"Updating existing test results row in: {results_csv_path}")

    with results_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


# ----------------------------------------------------------------------------------------------------------------

def test(
    model: AE,
    dataloader: DataLoader,
    output_path: str | Path,
    teacher_forcing: bool = True,
) -> dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file, torch.no_grad():
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "example_index",
                "target_sequence",
                "predicted_sequence",
                "target_length",
                "predicted_length",
                "token_accuracy",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        example_index = 0

        for batch in dataloader:
            inputs: torch.Tensor = batch["input_ids"].to(device)
            lengths: torch.Tensor = batch["length"].to(device)
            targets: torch.Tensor = batch["target_ids"].to(device)
            input_sequences: list[str] = batch["sequence"]

            decoder_inputs = inputs[:, :-1] if teacher_forcing else None
            targets = targets[:, 1:]
            
            outputs: torch.Tensor = model(inputs, decoder_input_ids=decoder_inputs, lengths=lengths) if teacher_forcing else model.decode_autoregressive(
                model.encode(inputs, lengths=lengths),
                max_length=targets.size(1),
            )

            loss: torch.Tensor = loss_fn(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

            predictions: torch.Tensor = outputs.argmax(dim=-1)
            non_pad_tokens: torch.Tensor = targets != PAD_IDX
            batch_tokens = non_pad_tokens.sum().item()

            total_loss += loss.item() * batch_tokens
            total_correct += (predictions[non_pad_tokens] == targets[non_pad_tokens]).sum().item()
            total_tokens += batch_tokens

            for row_idx, _ in enumerate(input_sequences):
                target_ids = targets[row_idx].detach().cpu()
                predicted_ids = predictions[row_idx].detach().cpu()
                row_non_pad_tokens = target_ids != PAD_IDX
                row_tokens = row_non_pad_tokens.sum().item()
                row_correct = (
                    predicted_ids[row_non_pad_tokens] == target_ids[row_non_pad_tokens]
                ).sum().item()
                row_accuracy = row_correct / row_tokens if row_tokens > 0 else 0.0
                target_sequence = decode_token_ids(target_ids)
                predicted_sequence = decode_token_ids(predicted_ids)

                writer.writerow(
                    {
                        "example_index": example_index,
                        "target_sequence": target_sequence,
                        "predicted_sequence": predicted_sequence,
                        "target_length": len(target_sequence),
                        "predicted_length": len(predicted_sequence),
                        "token_accuracy": f"{row_accuracy:.6f}",
                    }
                )
                example_index += 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

    mode_label = "teacher-forced" if teacher_forcing else "autoregressive"
    print(f"{mode_label.capitalize()} Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}, ")
    print(f"Saved decoder outputs to: {output_path}")
    return {"loss": avg_loss, "accuracy": accuracy}


def evaluate_checkpoint(
    args,
    model_type: str,
    hyperparams: AEParams,
    checkpoint_path: Path,
    length_options: str | None,
    length_bin: int | None,
    output_path: str | Path,
) -> None:
    print(f"Loading checkpoint: {checkpoint_path}")

    try:
        model = model_definition(model_type, hyperparams, checkpoint_path=checkpoint_path)
    except FileNotFoundError as error:
        print(error)
        sys.exit(1)

    if length_options is not None:
        loader_type = "length_bin"
    elif args.length_quartile is not None:
        loader_type = "quartile"
    elif args.max_length is not None:
        loader_type = "max_length"
    else:
        loader_type = None

    length_boundaries = None
    if loader_type in {"quartile", "length_bin"}:
        train_dataloader = create_dataloader(
            task=args.task,
            split="train",
            mode="autoencoder",
            batch_size=hyperparams.batch_size,
            shuffle=False,
        )
        num_bins = LENGTH_SPLIT_COUNTS[length_options] if loader_type == "length_bin" else 4
        if loader_type == "length_bin":
            length_boundaries = compute_train_length_boundaries(train_dataloader.dataset, num_bins=num_bins)
        else:
            length_boundaries = compute_train_length_boundaries(train_dataloader.dataset)

    test_dataloader = create_dataloader(
        task=args.task,
        split="test",
        mode="autoencoder",
        batch_size=hyperparams.batch_size,
        shuffle=False,
        loader_type=loader_type,
        max_length=args.max_length,
        quartile_name=args.length_quartile,
        length_options=length_options,
        length_bin=length_bin,
        cumulative=args.cumulative if loader_type == "length_bin" else args.cumulative_quartiles,
        length_boundaries=length_boundaries,
    )

    teacher_forced_metrics = test(
        model=model,
        dataloader=test_dataloader,
        output_path=output_path_for_mode(output_path, "teacher_forced"),
        teacher_forcing=True,
    )
    autoregressive_metrics = test(
        model=model,
        dataloader=test_dataloader,
        output_path=output_path_for_mode(output_path, "autoregressive"),
        teacher_forcing=False,
    )
    if teacher_forced_metrics is None or autoregressive_metrics is None:
        return

    results_csv_path = PROJECT_ROOT / "Code" / "results" / "tables" / "autoencoder_results.csv"
    append_autoencoder_results(
        results_csv_path=results_csv_path,
        version=infer_version(args.version if args.version_provided else None, checkpoint_path),
        checkpoint_path=checkpoint_path,
        teacher_forced_metrics=teacher_forced_metrics,
        autoregressive_metrics=autoregressive_metrics,
    )
    print(f"Saved aggregate test results to: {results_csv_path}")


def main():
    args = add_and_validate_test_inputs()

    model_type = args.model.lower()
    hyperparams = AEParams()
    artifact_suffix = sweep_artifact_suffix(args)
    fallback_artifact_suffix = legacy_sweep_artifact_suffix(args)

    try:
        if args.checkpoint is not None:
            checkpoint_path = Path(args.checkpoint)
        elif args.length_quartile is not None:
            checkpoint_path = legacy_quartile_checkpoint_path(
                model_type,
                args.task,
                args.version,
                args.length_quartile,
                artifact_suffix=artifact_suffix,
                fallback_artifact_suffix=fallback_artifact_suffix,
            )
        else:
            checkpoint_path = version_checkpoint_path(
                model_type,
                args.task,
                args.version,
                args.length_options,
                args.length_bin,
                artifact_suffix=artifact_suffix,
                fallback_artifact_suffix=fallback_artifact_suffix,
            )
    except FileNotFoundError as error:
        print(error)
        sys.exit(1)

    evaluate_checkpoint(
        args=args,
        model_type=model_type,
        hyperparams=hyperparams,
        checkpoint_path=checkpoint_path,
        length_options=args.length_options,
        length_bin=args.length_bin,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
