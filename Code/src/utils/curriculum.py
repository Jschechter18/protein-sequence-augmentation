import torch
from torch.utils.data import Dataset, DataLoader, Subset

from .dataloader import (
    LengthAwareBatchSampler,
    dataloader_batch_size,
    make_length_aware_dataloader,
)


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
    length_aware_sampler = train_dataloader.batch_sampler
    generator = train_dataloader.generator
    if isinstance(length_aware_sampler, LengthAwareBatchSampler):
        generator = length_aware_sampler.generator

    curriculum_loader = DataLoader(
            subset,
            batch_size=dataloader_batch_size(train_dataloader),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=train_dataloader.pin_memory,
            collate_fn=train_dataloader.collate_fn,
            generator=generator,
            worker_init_fn=train_dataloader.worker_init_fn,
            persistent_workers=(
                train_dataloader.persistent_workers if num_workers > 0 else False
            ),
        )
    if isinstance(length_aware_sampler, LengthAwareBatchSampler):
        curriculum_loader = make_length_aware_dataloader(
            curriculum_loader,
            pool_size_multiplier=(
                length_aware_sampler.pool_size
                // length_aware_sampler.batch_size
            ),
        )

    return (
        curriculum_loader,
        subset_size,
        fraction,
    )
    
    
    
def make_gradual_length_curriculum_dataloader():
    # TODO:
    # - implement a curriculum that gradually adds longer sequences for training -> idea is a smoother rampup has a better effect
    # reasoning is that the curriculum training seems to have shown that we have more effectively trained shorter sequences but longer sequences have not improved
    pass
