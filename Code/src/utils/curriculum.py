import torch
from torch.utils.data import Dataset, DataLoader, Subset


def _example_length(dataset: Dataset, index: int) -> int:
    """Return sequence length for a dataset item without assuming one dataset type."""
    if isinstance(dataset, Subset):
        return _example_length(dataset.dataset, dataset.indices[index])

    examples = getattr(dataset, "examples", None)
    if examples is not None:
        return int(examples[index]["_length"])

    length = dataset[index]["length"]
    return int(length.item() if isinstance(length, torch.Tensor) else length)

def _curriculum_fraction(
    epoch: int,
    curriculum_epochs: int,
    start_fraction: float,
) -> float:
    if curriculum_epochs <= 0:
        return 1.0
    if curriculum_epochs == 1:
        return start_fraction if epoch == 0 else 1.0

    progress = min(epoch, curriculum_epochs - 1) / (curriculum_epochs - 1)
    return min(1.0, start_fraction + (1.0 - start_fraction) * progress)


def make_length_curriculum_dataloader(
    train_dataloader: DataLoader,
    epoch: int,
    curriculum_epochs: int,
    start_fraction: float,
    num_workers: int = 2,
) -> tuple[DataLoader, int, float]:
    """Build an epoch dataloader using the shortest sequences first."""
    if curriculum_epochs <= 0:
        try:
            num_examples = len(train_dataloader.dataset)
        except AttributeError:
            num_examples = len(train_dataloader)
        return train_dataloader, num_examples, 1.0

    fraction = _curriculum_fraction(epoch, curriculum_epochs, start_fraction)
    dataset = train_dataloader.dataset
    num_examples = len(dataset)

    if fraction >= 1.0:
        return train_dataloader, num_examples, fraction

    subset_size = max(1, int(round(num_examples * fraction)))
    sorted_indices = sorted(range(num_examples), key=lambda idx: _example_length(dataset, idx))
    subset = Subset(dataset, sorted_indices[:subset_size])

    return (
        DataLoader(
            subset,
            batch_size=train_dataloader.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=train_dataloader.pin_memory,
            collate_fn=train_dataloader.collate_fn,
        ),
        subset_size,
        fraction,
    )