# Scripts

Run commands from the repository root unless noted otherwise.

## `setup_peer_data.sh`

Sets up the official PEER benchmark data used by the project.

What it does:

- Creates required `external/`, `data/raw/peer/`, and `data/processed/peer/` directories.
- Clones or updates `external/PEER_Benchmark`.
- Installs/import-checks `lmdb`.
- Runs `Code/scripts/prepare_peer_data.py`.

Usage:

```bash
bash Code/scripts/setup_peer_data.sh
```

Optional: choose a Python executable with `PYTHON`:

```bash
PYTHON=.venv/bin/python bash Code/scripts/setup_peer_data.sh
```

Outputs:

- `data/processed/peer/localization/{train,valid,test}.csv`
- `data/processed/peer/solubility/{train,valid,test}.csv`
- `data/processed/peer/metadata.json`

## `prepare_peer_data.py`

Prepares official PEER localization and solubility splits from LMDB files into CSV files.

Usually run through `setup_peer_data.sh`, because it requires the PEER checkout and `lmdb`.

Direct usage:

```bash
python Code/scripts/prepare_peer_data.py
```

This script downloads missing official task archives, verifies checksums, extracts LMDBs, exports CSV splits, and writes metadata.

## `post_processing.py`

Converts older ESM-2 experiment result folders into standardized `history.json` files.

Default usage:

```bash
python Code/scripts/post_processing.py
```

By default, it scans:

```text
Code/results/esm2
```

Useful options:

```bash
python Code/scripts/post_processing.py \
  --input_dir Code/results/esm2/solubility
```

Copy organized runs into a new output root:

```bash
python Code/scripts/post_processing.py \
  --input_dir Code/results/esm2 \
  --output_dir Code/results/esm2
```

Move instead of copy:

```bash
python Code/scripts/post_processing.py \
  --input_dir Code/results/esm2 \
  --output_dir Code/results/esm2 \
  --move
```

Expected run inputs:

- `config.json`
- `training_history.csv`
- optional `metrics.json`

Output:

- `history.json` written into each processed run directory.

## `merge_classifier_tuning.py`

Combines classifier hyperparameter trials run on separate machines, copies the
unique trial artifacts into one canonical tuning tree, and rebuilds the global
`tuning_results.csv` and `selected_hyperparameters.json` from validation
histories. It deliberately ignores the partial summary files produced on each
machine.

Partition the full frozen tuning grid by learning rate while keeping the same
commit on all three EC2 instances:

```bash
# EC2 instance 1
python -m Code.src.training.train_classifier \
  --hp_tune --version 4 --tuning_learning_rates 1e-4

# EC2 instance 2
python -m Code.src.training.train_classifier \
  --hp_tune --version 4 --tuning_learning_rates 1e-5

# EC2 instance 3
python -m Code.src.training.train_classifier \
  --hp_tune --version 4 --tuning_learning_rates 1e-6
```

Each one-rate instance runs 24 trials; the merged full grid has 72 trials. Use
the same dataset files, autoencoder checkpoint, environment, and Git commit on
all instances.

Collect each result tree into a separate local staging directory. Preserve the
trailing slash on the remote tuning directory:

```bash
mkdir -p collected/classifier-v4/ec2-1 \
  collected/classifier-v4/ec2-2 \
  collected/classifier-v4/ec2-3

rsync -az ubuntu@ec2-1:/path/to/repo/Code/results/classifier/solubility/v4/tuning/ \
  collected/classifier-v4/ec2-1/
rsync -az ubuntu@ec2-2:/path/to/repo/Code/results/classifier/solubility/v4/tuning/ \
  collected/classifier-v4/ec2-2/
rsync -az ubuntu@ec2-3:/path/to/repo/Code/results/classifier/solubility/v4/tuning/ \
  collected/classifier-v4/ec2-3/
```

Then merge and require full coverage:

```bash
python Code/scripts/merge_classifier_tuning.py \
  --input_dir collected/classifier-v4/ec2-1 \
  --input_dir collected/classifier-v4/ec2-2 \
  --input_dir collected/classifier-v4/ec2-3 \
  --output_dir Code/results/classifier/solubility/v4/tuning \
  --expect_full_grid
```

Start the final three-seed sweep from the merged winner file:

```bash
python -m Code.src.training.train_classifier --sweep --version 4
```

Outputs:

- Canonical trial directories under `<head>/<representation>/<trial>/`
- `tuning_results.csv` rebuilt from every trial's `status.json` and `history.csv`
- `selected_hyperparameters.json` ready for the final seeded sweep
- `merge_manifest.json` recording inputs, counts, statuses, and provenance hashes

With `--expect_full_grid`, the merge checks every expected representation, head,
learning rate, weight decay, dropout, and seed—not just the total of 72. The
merge also fails before publishing results if any discovered trial is not
complete, duplicate completed trials disagree, or code/data/environment
provenance differs. Archived `.backup_*` trials are ignored. Tuning checkpoints
are stored in a separate checkpoint tree and are not needed to choose
hyperparameters; transfer them separately only if they must be archived or
resumed.

If jobs were already partitioned by manually editing `TUNING_LEARNING_RATES`,
the three saved source hashes will differ. Compare those source files first. If
the only difference is the assigned search-rate constant, rerun the merge with
`--allow_partition_source_mismatch`. This narrow exception still requires the
data, checkpoints, environment, and all other fingerprinted source files to
match; the distinct driver hashes and commits remain recorded in
`merge_manifest.json`. New runs should use the CLI partition shown above so this
exception is unnecessary.

## `download_checkpoints.sh`

Downloads any saved checkpoints to local repository.

## `test_aes.zsh`

Runs autoencoder testing for every swept length-bin checkpoint in a version directory.

```bash
./Code/scripts/download_checkpoints.sh
```

Default usage runs solubility `v9`:

```zsh
./Code/scripts/test_aes.zsh
```

Run another version:

```zsh
./Code/scripts/test_aes.zsh v8
```

Run another task/version:

```zsh
./Code/scripts/test_aes.zsh v8 solubility
```

Run cumulative length-bin checkpoints:

```zsh
./Code/scripts/test_aes.zsh v9 solubility cumulative
```

What it expects:

- Checkpoints under `checkpoints/autoencoder/<task>/<version>/`
- Checkpoint names like:

```text
model_ae_length_1_of_2_solubility_latent128_tfd0p3.pt
```

The script parses checkpoint names to recover:

- `--length_options`
- `--length_bin`
- `--latent_dim`
- `--teacher_forcing_dropout_rate`

The script cannot infer whether a run used cumulative length bins from the checkpoint filename. Pass `cumulative` as the third argument when the models were trained with `--cumulative`.

Outputs:

- Per-checkpoint decoder outputs under `outputs/autoencoder/<version>/`
- Aggregate metrics appended or updated in `Code/results/tables/autoencoder_results.csv`

Notes:

- The script automatically changes to the repository root.
- It uses `.venv/bin/python` if available; otherwise it uses `python`.
- It is a `zsh` script, so run it with `zsh` or execute it directly as shown above.
