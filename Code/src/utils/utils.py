import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from models.autoencoder import ProteinSequenceAutoencoder as AE
from torch.utils.data import Dataset, DataLoader, Subset
from utils.dataloader import VOCAB, VOCAB_SIZE


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


def make_token_weights(device):
    """Create a tensor of token weights for amino acids.

    Parameters
    ----------
    device : torch.device
        The device to which the tensor will be moved.

    Returns
    -------
    torch.Tensor
        A tensor of token weights.
    """
    weights = torch.ones(VOCAB_SIZE, dtype=torch.float32)

    aa_weights = {
        "A": 0.79, "C": 1.97, "D": 0.93, "E": 0.84, "F": 1.14,
        "G": 0.86, "H": 1.24, "I": 0.93, "K": 0.92, "L": 0.73,
        "M": 1.43, "N": 1.11, "P": 1.05, "Q": 1.14, "R": 0.95,
        "S": 0.89, "T": 0.99, "V": 0.87, "W": 2.00, "Y": 1.24,
    }

    for aa, weight in aa_weights.items():
        weights[VOCAB[aa]] = weight

    return weights.to(device)