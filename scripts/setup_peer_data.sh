#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PEER_REPO_DIR="$REPO_ROOT/external/PEER_Benchmark"
PYTHON_BIN="${PYTHON:-python}"

echo "[PEER] Starting PEER data setup"
echo "[PEER] Repository root: $REPO_ROOT"

echo "[PEER] Creating required directories"
mkdir -p \
  "$REPO_ROOT/external" \
  "$REPO_ROOT/data/raw/peer" \
  "$REPO_ROOT/data/processed/peer/localization" \
  "$REPO_ROOT/data/processed/peer/solubility"

if [ -d "$PEER_REPO_DIR/.git" ]; then
  echo "[PEER] Official PEER repository already exists. Pulling latest changes"
  git -C "$PEER_REPO_DIR" pull --ff-only || echo "[PEER] Warning: unable to pull latest changes. Continuing with existing checkout"
elif [ -d "$PEER_REPO_DIR" ]; then
  echo "[PEER] Directory $PEER_REPO_DIR already exists and is not a git checkout. Reusing it as-is"
else
  echo "[PEER] Cloning official PEER benchmark repository"
  git clone https://github.com/DeepGraphLearning/PEER_Benchmark.git "$PEER_REPO_DIR"
fi

echo "[PEER] Ensuring minimal Python dependency is available"
if ! "$PYTHON_BIN" -c "import lmdb" >/dev/null 2>&1; then
  echo "[PEER] Installing lmdb"
  "$PYTHON_BIN" -m pip install lmdb
else
  echo "[PEER] lmdb already available"
fi

echo "[PEER] Preparing official localization and solubility splits"
cd "$REPO_ROOT"
"$PYTHON_BIN" scripts/prepare_peer_data.py

echo "[PEER] Setup complete"