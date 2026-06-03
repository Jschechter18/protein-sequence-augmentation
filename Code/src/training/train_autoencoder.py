"""Autoencoder training script."""
import argparse
import json
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, cohen_kappa_score
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.autoencoder import ProteinSequenceAutoencoder as AE
from utils.dataloader import BOS_IDX, PAD_IDX, VOCAB_SIZE, create_dataloader
from utils.hyperparameters import AutoencoderHyperParameters as Params

# ----------------------------------------------------------------------------------------------------------------
TRIAL = 'ae_benchmark'
# ----------------------------------------------------------------------------------------------------------------


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps") # for macOS with Apple Silicon -> just for when you test locally
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
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

def model_definition(hyperparams: Params) -> tuple[AE, torch.optim.Adam, ReduceLROnPlateau]:
    model = AE(
        embedding_dim=hyperparams.embedding_dim,
        hidden_dim=hyperparams.hidden_dim,
        latent_dim=hyperparams.latent_dim,
        num_layers=hyperparams.num_layers,
        dropout=hyperparams.dropout,
        pad_idx=PAD_IDX,
        bos_idx=BOS_IDX,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.learning_rate)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=hyperparams.patience)
    
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
            
            decoder_inputs = inputs[:, :-1]
            targets = targets[:, 1:]
            
            outputs = model(inputs, decoder_input_ids=decoder_inputs, lengths=lengths)
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

def train(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    hyperparams: Params,
) -> tuple[AE, dict]:
    model, optimizer, scheduler = model_definition(hyperparams)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0
    history = {
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
    
    for epoch in range(hyperparams.num_epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        epoch_targets = []
        epoch_predictions = []

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{hyperparams.num_epochs}",
            unit="batch",
        )

        for batch in progress_bar:
            inputs = batch["input_ids"].to(device)
            lengths = batch["length"].to(device)
            targets = batch["target_ids"].to(device)
            
            decoder_inputs = inputs[:, :-1]
            targets = targets[:, 1:]
            
            optimizer.zero_grad()
            outputs = model(inputs, decoder_input_ids=decoder_inputs, lengths=lengths)
            loss = loss_fn(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            loss.backward()
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
        }
        history["train_loss"].append(avg_loss)
        history["train_scores"]["accuracy"].append(accuracy)
        history["train_scores"]["f1"].append(f1)
        
        # if val_dataloader is None:
        #     scheduler.step(avg_loss)
        #     history["epochs"].append(epoch_info)
        #     continue

        val_metrics = validate(model, val_dataloader, loss_fn)
        val_loss = val_metrics["loss"]
        scheduler.step(val_loss)
        epoch_info["val_loss"] = val_loss
        epoch_info["val_accuracy"] = val_metrics["accuracy"]
        history["val_loss"].append(val_loss)
        history["val_scores"]["accuracy"].append(val_metrics["accuracy"])
        history["epochs"].append(epoch_info)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            checkpoint_dir = Path(__file__).resolve().parents[3] / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"model_{TRIAL}.pt"
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= hyperparams.patience:
                print(
                    f"Early stopping after {epoch + 1} epochs. "
                    f"Best validation loss: {best_val_loss:.4f}"
                )
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        
    # return {"model": model, "history": history}
    return model, history

def test(model: AE, dataloader: DataLoader) -> None:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_ids"].to(device)
            lengths = batch["length"].to(device)
            targets = batch["target_ids"].to(device)

            decoder_inputs = inputs[:, :-1]
            targets = targets[:, 1:]

            outputs = model(inputs, decoder_input_ids=decoder_inputs, lengths=lengths)
            loss = loss_fn(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

            predictions = outputs.argmax(dim=-1)
            non_pad_tokens = targets != PAD_IDX
            batch_tokens = non_pad_tokens.sum().item()

            total_loss += loss.item() * batch_tokens
            total_correct += (predictions[non_pad_tokens] == targets[non_pad_tokens]).sum().item()
            total_tokens += batch_tokens

            all_targets.append(targets[non_pad_tokens].detach().cpu())
            all_predictions.append(predictions[non_pad_tokens].detach().cpu())

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

    if total_tokens > 0:
        y_true = torch.cat(all_targets).numpy()
        y_pred = torch.cat(all_predictions).numpy()
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        kappa = cohen_kappa_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, zero_division=0)
    else:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        kappa = 0.0
        report = "No non-padding tokens found in test set."

    print(
        f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, "
        f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
        f"F1: {f1:.4f}, Cohen's Kappa: {kappa:.4f}"
    )
    print("Classification Report:")
    print(report)

def main():
    hyperparams = Params()
    args = argparse.ArgumentParser(description='Train an autoencoder on protein sequences.')
    args.add_argument('--model', type=str, default='AE', help='Model to train (default: AE)')
    args.add_argument('--task', type=str, default='localization', help='Task to perform (default: localization)')
    
    args = args.parse_args()
    if args.model.upper() != "AE":
        raise ValueError("Only --model AE is currently supported")
    if args.task != "localization" and args.task != "solubility":
        raise ValueError("Task only accepts 'localization' or 'solubility'")
    
    train_dataloader = create_dataloader(task=args.task, split=TRAIN_SPLIT, mode="autoencoder",
                                   batch_size=hyperparams.batch_size, shuffle=hyperparams.shuffle,
                                   num_workers=num_workers, max_length=hyperparams.max_len)
    val_dataloader = create_dataloader(task=args.task, split=VALID_SPLIT, mode="autoencoder",
                                       batch_size=hyperparams.batch_size, shuffle=False,
                                       num_workers=num_workers, max_length=hyperparams.max_len)
    test_dataloader = create_dataloader(task=args.task, split="test", mode="autoencoder",
                                       batch_size=hyperparams.batch_size, shuffle=False,
                                       num_workers=num_workers, max_length=hyperparams.max_len)
    
    
    model, history = train(train_dataloader, val_dataloader, hyperparams)
    
    history_dir = Path(__file__).resolve().parents[3] / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"ae_{args.task}_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=float)

    print(f"Saved training history to: {history_path}")
        
    test(model, test_dataloader)
    
    
    # TODO: output results to correct location once that is ready


if __name__ == "__main__":
    main()


# TODO:
# - Output results to a file in addition to printing to console
