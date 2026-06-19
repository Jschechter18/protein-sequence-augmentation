"""Autoencoder training script.

To run script:

# GRU AE architecture:
python Code/src/training/train_autoencoder.py --model AE --task localization
python Code/src/training/train_autoencoder.py --model AE --task solubility



python Code/src/training/train_autoencoder.py --model AE --task solubility --curriculum_epochs 5 --curriculum_start_fraction 0.2 --version <version>



# Length quartile training:
python Code/src/training/train_autoencoder.py --model AE --task solubility --version <version> --length_quartile l

"""
import json
import copy
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.autoencoder import ProteinSequenceAutoencoder as AE
from utils.dataloader import BOS_IDX, PAD_IDX, create_dataloader, compute_train_length_boundaries, make_quartile_loader
from utils.hyperparameters import (AutoencoderHyperparameters as AEParams)
from utils.utils import load_training_checkpoint, make_token_weights
from utils.curriculum import make_length_curriculum_dataloader
from utils.train_input_validation import add_and_validate_train_inputs, autoencoder_artifact_stem

import warnings

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

num_workers = 4 if torch.cuda.is_available() else 0
# ----------------------------------------------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# ----------------------------------------------------------------------------------------------------------------

TRAIN_SPLIT = "train"
VALID_SPLIT = "valid"

def model_definition(model_type: str, hyperparams: AEParams) -> tuple[AE, torch.optim.Adam, ReduceLROnPlateau]:
    if model_type == "ae":
        model = AE(
            embedding_dim=hyperparams.embedding_dim,
            cnn_out_channels=hyperparams.cnn_out_channels,
            hidden_dim=hyperparams.hidden_dim,
            latent_dim=hyperparams.latent_dim,
            num_layers=hyperparams.num_layers,
            kernel_size=hyperparams.kernel_size,
            bidirectional=hyperparams.bidirectional,
            dropout=hyperparams.dropout,
            pad_idx=PAD_IDX,
            bos_idx=BOS_IDX,
            condition_decoder_on_latent=hyperparams.condition_decoder_on_latent,
            teacher_forcing_dropout_rate=hyperparams.teacher_forcing_dropout_rate,
        ).to(device)
    else:
        raise ValueError(f"Model type {model_type} not supported.")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.learning_rate)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=hyperparams.lr_patience)
    
    return model, optimizer, scheduler
# ----------------------------------------------------------------------------------------------------------------
def validate(model: AE, dataloader: DataLoader, loss_fn: nn.CrossEntropyLoss) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_ids"].to(device)
            lengths = batch["length"].to(device)
            targets = batch["target_ids"].to(device)
            
            targets = targets[:, 1:]
            
            outputs = model(inputs, decoder_input_ids=inputs[:, :-1], lengths=lengths)
            # outputs = model.decode_autoregressive(
            #     model.encode(inputs, lengths=lengths),
            #     max_length=targets.size(1),
            # )
            loss = loss_fn(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            
            predictions = outputs.argmax(dim=-1)
            non_pad_tokens = targets != PAD_IDX
            batch_tokens = non_pad_tokens.sum().item()
            total_loss += loss.item() * batch_tokens
            total_correct += (predictions[non_pad_tokens] == targets[non_pad_tokens]).sum().item()
            total_tokens += batch_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return {"loss": avg_loss, "accuracy": accuracy}
# ----------------------------------------------------------------------------------------------------------------

def save_training_history(history: dict, history_path: str | Path) -> None:
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = history_path.with_suffix(f"{history_path.suffix}.tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=float)

    tmp_path.replace(history_path)


def train(
    model_type: str,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    hyperparams: AEParams,
    version: int,
    is_overfit: bool = False,
    load_path: str | None = None,
    task: str = "solubility",
    history_path: str | Path | None = None,
    curriculum_epochs: int = 0,
    curriculum_start_fraction: float = 0.3,
    length_quartile: str | None = None,
) -> tuple[AE, dict]:

    model, optimizer, scheduler = model_definition(model_type, hyperparams)
    start_epoch = 0
    best_val_loss = float("inf")
    best_state_dict = None
    if load_path is not None:
        print(f"Checkpoint path provided: {load_path}")
        start_epoch, best_val_loss = load_training_checkpoint(
            model,
            optimizer,
            scheduler,
            load_path,
            device,
        )
        best_state_dict = copy.deepcopy(model.state_dict())
        print(f"Resuming training from epoch {start_epoch + 1}.")
        
    loss_fn = nn.CrossEntropyLoss(weight=make_token_weights(device), ignore_index=PAD_IDX)
    epochs_without_improvement = 0
    history = {
        "hyperparameters": asdict(hyperparams),
        "curriculum": {
            "type": "length" if curriculum_epochs > 0 else "none",
            "epochs": curriculum_epochs,
            "start_fraction": curriculum_start_fraction,
        },
        "epochs": [],
        "train_loss": [],
        "train_scores": {
            "accuracy": [],
            "f1": [],
        },
        "val_loss": [],
        "val_scores": {
            "accuracy": [],
        },
    }
    if history_path is not None:
        save_training_history(history, history_path)
    
    if start_epoch >= hyperparams.num_epochs:
        print(
            f"Checkpoint already completed {start_epoch} epoch(s); "
            f"num_epochs is {hyperparams.num_epochs}, so no additional training will run."
        )

    for epoch in range(start_epoch, hyperparams.num_epochs):
        
        model.train()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        epoch_targets = []
        epoch_predictions = []

        epoch_train_dataloader, curriculum_examples, curriculum_fraction = make_length_curriculum_dataloader(
            train_dataloader,
            epoch,
            curriculum_epochs,
            curriculum_start_fraction,
            num_workers=num_workers,
        )
        desc = f"Epoch {epoch + 1}/{hyperparams.num_epochs}"
        if curriculum_epochs > 0 and curriculum_fraction < 1.0:
            desc += f" ({curriculum_examples} shortest examples)"

        progress_bar = tqdm(
            epoch_train_dataloader,
            desc=desc,
            unit="batch",
        )

        for batch in progress_bar:
            inputs = batch["input_ids"].to(device)
            lengths = batch["length"].to(device)
            targets = batch["target_ids"].to(device)
            
            targets = targets[:, 1:]
            
            optimizer.zero_grad()
            outputs = model(inputs, decoder_input_ids=inputs[:, :-1], lengths=lengths)
            loss: torch.Tensor = loss_fn(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            loss.backward()
            if hyperparams.grad_clip:
                clip_grad_norm_(model.parameters(), max_norm=1.0) # Testing out gradient clipping to see if it helps with training stability
            optimizer.step()
            
            predictions = outputs.argmax(dim=-1)
            non_pad_tokens = targets != PAD_IDX
            batch_tokens = non_pad_tokens.sum().item()
            total_loss += loss.item() * batch_tokens
            total_correct += (predictions[non_pad_tokens] == targets[non_pad_tokens]).sum().item()
            total_tokens += batch_tokens
            epoch_targets.append(targets[non_pad_tokens].detach().cpu())
            epoch_predictions.append(predictions[non_pad_tokens].detach().cpu())

            running_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
            running_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
            progress_bar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_accuracy:.4f}")
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        if epoch_targets:
            all_targets = torch.cat(epoch_targets)
            all_predictions = torch.cat(epoch_predictions)
            accuracy = accuracy_score(all_targets, all_predictions)
            f1 = f1_score(all_targets, all_predictions, average='weighted')
        else:
            accuracy = 0.0
            f1 = 0.0
        print(f"Epoch [{epoch+1}/{hyperparams.num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
            "train_f1": f1,
            "curriculum_fraction": curriculum_fraction,
            "curriculum_examples": curriculum_examples,
        }
        history["train_loss"].append(avg_loss)
        history["train_scores"]["accuracy"].append(accuracy)
        history["train_scores"]["f1"].append(f1)
        

        val_metrics = validate(model, val_dataloader, loss_fn)
        val_loss = val_metrics["loss"]
        scheduler.step(val_loss)
        epoch_info["val_loss"] = val_loss
        epoch_info["val_accuracy"] = val_metrics["accuracy"]
        history["val_loss"].append(val_loss)
        history["val_scores"]["accuracy"].append(val_metrics["accuracy"])
        history["epochs"].append(epoch_info)
        if history_path is not None:
            save_training_history(history, history_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            checkpoint_dir = Path(__file__).resolve().parents[3] / f"checkpoints/autoencoder/{task}/v{version}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_stem = autoencoder_artifact_stem(
                model_type,
                task,
                length_quartile,
                is_overfit=is_overfit,
            )
            checkpoint_path = checkpoint_dir / f"{checkpoint_stem}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
            }, checkpoint_path)
        else:
            epochs_without_improvement += 1
            print(
                f"No validation loss improvement for "
                f"{epochs_without_improvement}/{hyperparams.patience} epoch(s)."
            )
            if epochs_without_improvement >= hyperparams.patience:
                print(
                    f"Early stopping after {epoch + 1} epochs. "
                    f"Best validation loss: {best_val_loss:.4f}"
                )
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        
    return model, history

def test(model: AE, dataloader: DataLoader) -> None:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    all_targets: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            inputs: torch.Tensor = batch["input_ids"].to(device)
            lengths: torch.Tensor = batch["length"].to(device)
            targets: torch.Tensor = batch["target_ids"].to(device)

            targets = targets[:, 1:]

            outputs: torch.Tensor = model(inputs, decoder_input_ids=inputs[:, :-1], lengths=lengths)
            loss: torch.Tensor = loss_fn(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

            predictions: torch.Tensor = outputs.argmax(dim=-1)
            non_pad_tokens: torch.Tensor = targets != PAD_IDX
            batch_tokens = non_pad_tokens.sum().item()

            total_loss += loss.item() * batch_tokens
            total_correct += (predictions[non_pad_tokens] == targets[non_pad_tokens]).sum().item()
            total_tokens += batch_tokens

            all_targets.append(targets[non_pad_tokens].detach().cpu())
            all_predictions.append(predictions[non_pad_tokens].detach().cpu())

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

    print(
        f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}, "
    )

def make_overfit_dataloaders(
    train_dataloader: DataLoader,
    num_batches: int,
) -> tuple[DataLoader, DataLoader]:
    """Create train/validation loaders over the same tiny training subset.

    Parameters
    ----------
    train_dataloader : DataLoader
        Original training dataloader
    num_batches : int
        _description_

    Returns
    -------
    tuple[DataLoader, DataLoader]
        _description_

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    """
    # Input validation for the overfit test
    if num_batches <= 0:
        raise ValueError("--overfit_batches must be a positive integer")

    batch_size = train_dataloader.batch_size
    if batch_size is None:
        raise ValueError("batch_size cannot be None")

    num_examples = min(
        num_batches * batch_size,
        len(train_dataloader.dataset),
    )
    subset = Subset(train_dataloader.dataset, range(num_examples))

    train_subset_loader = DataLoader(
        subset,
        batch_size=train_dataloader.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=train_dataloader.pin_memory,
        collate_fn=train_dataloader.collate_fn,
    )
    val_subset_loader = DataLoader(
        subset,
        batch_size=train_dataloader.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=train_dataloader.pin_memory,
        collate_fn=train_dataloader.collate_fn,
    )

    return train_subset_loader, val_subset_loader

def main():
    args, hyperparams = add_and_validate_train_inputs()

    
    train_dataloader = create_dataloader(task=args.task, split=TRAIN_SPLIT, mode="autoencoder",
                                batch_size=hyperparams.batch_size, shuffle=hyperparams.shuffle,
                                num_workers=num_workers)
    val_dataloader = create_dataloader(task=args.task, split=VALID_SPLIT, mode="autoencoder",
                                    batch_size=hyperparams.batch_size, shuffle=False,
                                    num_workers=num_workers)
    test_dataloader = create_dataloader(task=args.task, split="test", mode="autoencoder",
                                    batch_size=hyperparams.batch_size, shuffle=False,
                                    num_workers=num_workers)
    
    if args.length_quartile is not None:
        # Call the utility functions for length boundaries
        boundaries = compute_train_length_boundaries(train_dataloader.dataset)

        train_dataloader = make_quartile_loader(
            train_dataloader, boundaries, args.length_quartile, shuffle=True
        )
        val_dataloader = make_quartile_loader(
            val_dataloader, boundaries, args.length_quartile, shuffle=False
        )
        test_dataloader = make_quartile_loader(
            test_dataloader, boundaries, args.length_quartile, shuffle=False
        )
    
    if args.overfit_batches is not None:
        train_dataloader, val_dataloader = make_overfit_dataloaders(
            train_dataloader,
            args.overfit_batches,
        )
        print(
            f"Overfit debug mode: training and validating on "
            f"{len(train_dataloader.dataset)} examples "
            f"(batch_size={hyperparams.batch_size}, dropout={hyperparams.dropout}, "
            f"learning_rate={hyperparams.learning_rate})."
        )

    if args.curriculum_epochs > 0:
        print(
            f"Length curriculum enabled: starting with "
            f"{args.curriculum_start_fraction:.0%} of the shortest training examples "
            f"and reaching the full training set over {args.curriculum_epochs} epoch(s)."
        )
    
    history_dir = Path(__file__).resolve().parents[3] / "history"
    history_stem = autoencoder_artifact_stem(
        args.model,
        args.task,
        args.length_quartile,
        is_overfit=(args.overfit_batches is not None),
    )
    history_path = history_dir / f"v{args.version}_{history_stem}_history.json"

    print()
    model, history = train(
        args.model.lower(),
        train_dataloader,
        val_dataloader,
        hyperparams,
        version=args.version,
        is_overfit=(args.overfit_batches is not None),
        task=args.task,
        load_path=args.load_path,
        history_path=history_path,
        curriculum_epochs=args.curriculum_epochs,
        curriculum_start_fraction=args.curriculum_start_fraction,
        length_quartile=args.length_quartile,
    )

    print(f"Saved training history to: {history_path}")

    if args.overfit_batches is None:
        test(model, test_dataloader)
    else:
        print("Skipping full test set evaluation in overfit debug mode.")
    


if __name__ == "__main__":
    main()


# TODO:
# - Important next steps:
# 1. Per the protein vae paper, we should consider using a **KL divergence loss** (in addition to our cross entropy) to regularize the latent space -> this would require some changes to the architecture to output a mean and variance vector for the latent space, and then sampling from that distribution during training. This could help encourage the model to learn a more structured latent space, which could improve generalization and interpolation between sequences.
# 2. When we do this, we should also implement dropout for the teacher forcing during training. We can start small at like 0.1, but then set it as a hyperparameter that can be tuned -> the paper set it up to 45% dropout for that dropout

# Other goals:
# 1. Better curriculum where we gradually add longer sequences for training -> idea is a smoother rampup has a better effect rather than slowly increasing which improves short faster but not long

# Once we are done with these improvements, we will call what we have our baselines:
# - Start hyperparameter tuning with the baselines we have
