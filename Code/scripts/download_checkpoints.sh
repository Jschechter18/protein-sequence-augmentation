#!/usr/bin/env bash
set -euo pipefail

# Google Drive folder containing checkpoint folders
DRIVE_FOLDER_ID="1vE6wXgCHVCZ-yhsc2QH8Vlw9u1hd8eda"
DRIVE_FOLDER_URL="https://drive.google.com/drive/folders/${DRIVE_FOLDER_ID}"

# Local destination, relative to repo root
DEST_DIR="Code/results/autoencoder/solubility/checkpoints"

mkdir -p "$DEST_DIR"

echo "Downloading checkpoint folders from Google Drive..."
echo "Source: $DRIVE_FOLDER_URL"
echo "Destination: $DEST_DIR"

# gdown handles Google Drive folders recursively.
# It also handles Google Drive's large-file confirmation pages.
if ! command -v gdown >/dev/null 2>&1; then
    echo "gdown not found. Installing gdown..."
    python3 -m pip install --user gdown
fi

python3 -m gdown --folder "$DRIVE_FOLDER_URL" --output "$DEST_DIR"

echo "Download complete."
echo "Checkpoint folders saved under: $DEST_DIR"