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

## `test_aes.zsh`

Runs autoencoder testing for every swept length-bin checkpoint in a version directory.

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
