"""
Test script for evaluating a trained autoencoder checkpoint.

Usage:
    python Code/src/testing/test_autoencoder.py --model AE --task solubility --checkpoint checkpoints/v2/model_ae_solubility.pt  --teacher_forcing True
"""

import argparse
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
from utils.dataloader import BOS_IDX, EOS_IDX, PAD_IDX, VOCAB, create_dataloader
from utils.hyperparameters import AutoencoderHyperparameters as AEParams

IDX_TO_TOKEN = {idx: token for token, idx in VOCAB.items()}

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
def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError("Expected a boolean value.")


def default_checkpoint_path(model_type: str, task: str, version: str | None) -> Path:
    model_slug = model_type.lower()
    checkpoint_name = f"model_{model_slug}_{task}.pt"
    checkpoint_dir = PROJECT_ROOT / "checkpoints"

    if version:
        return checkpoint_dir / version / checkpoint_name

    return checkpoint_dir / checkpoint_name


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


# ----------------------------------------------------------------------------------------------------------------

def test(
    model: AE,
    dataloader: DataLoader,
    output_path: str | Path,
    teacher_forcing: bool = True,
) -> None:
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

            if teacher_forcing:
                outputs: torch.Tensor = model(inputs, decoder_input_ids=decoder_inputs, lengths=lengths)
            else:
                outputs = model.decode_autoregressive(
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

    print(
        f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}, "
    )
    print(f"Saved decoder outputs to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test an autoencoder checkpoint on protein sequences.")
    parser.add_argument("--model", type=str, default="AE", choices=["AE", "ae"], help="Model to test.")
    parser.add_argument("--task", type=str, default="solubility", choices=["localization", "solubility"], help="Task to test.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint to test. Defaults to checkpoints/<version>/model_<model>_<task>.pt when --version is set.",
    )
    parser.add_argument("--version", type=str, default="v1", help="Checkpoint version directory to test.")
    parser.add_argument(
        "--teacher_forcing",
        type=str_to_bool,
        default=True,
        help="Use teacher forcing during reconstruction evaluation.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "output_results.csv"),
        help="CSV path for decoder prediction results.",
    )

    args = parser.parse_args()

    model_type = args.model.lower()
    hyperparams = AEParams()
    checkpoint_path = args.checkpoint or default_checkpoint_path(model_type, args.task, args.version)
    print(f"Loading checkpoint: {checkpoint_path}")

    model = model_definition(model_type, hyperparams, checkpoint_path=checkpoint_path)
    
    test(
        model=model,
        dataloader=create_dataloader(
            task=args.task,
            split="test",
            mode="autoencoder",
            batch_size=hyperparams.batch_size,
            shuffle=False,
        ),
        output_path=args.output_path,
        teacher_forcing=args.teacher_forcing,
    )


if __name__ == "__main__":
    main()
