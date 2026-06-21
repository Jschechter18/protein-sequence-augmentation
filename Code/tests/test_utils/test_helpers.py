from pathlib import Path

import pandas as pd


def write_split_csv(root: Path, task: str = "toy", split: str = "train") -> None:
    """Write a CSV file for a given task and split."""
    task_dir = root / task
    task_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "sequence": ["ACD", "ACDEFG"],
            "label": [1, 0],
        }
    ).to_csv(task_dir / f"{split}.csv", index=False)


def write_csv(
    root: Path,
    rows: dict[str, list],
    task: str = "toy",
    filename: str = "train.csv",
) -> Path:
    task_dir = root / task
    task_dir.mkdir(parents=True, exist_ok=True)
    csv_path = task_dir / filename
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path
