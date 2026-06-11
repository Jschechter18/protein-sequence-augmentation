import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from models.autoencoder import ProteinSequenceAutoencoder as AE
from torch.utils.data import Dataset, DataLoader, Subset


def load_training_checkpoint(
    model: AE,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    load_path: str | Path,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint_path = Path(load_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            print("Checkpoint has no optimizer state; starting optimizer from scratch.")
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            print("Checkpoint has no scheduler state; starting scheduler from scratch.")

        start_epoch = int(checkpoint.get("epoch", 0))
        best_val_loss = float(checkpoint.get("val_loss", float("inf")))
        return start_epoch, best_val_loss

    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
        print("Loaded state_dict-only checkpoint; optimizer and scheduler start from scratch.")
        return 0, float("inf")

    raise TypeError("Checkpoint must be a state_dict or contain 'model_state_dict'.")

