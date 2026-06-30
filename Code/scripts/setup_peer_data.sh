#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PEER_REPO_DIR="$REPO_ROOT/external/PEER_Benchmark"
PYTHON_BIN="${PYTHON:-python}"

check_lmdb_import() {
  "$PYTHON_BIN" -c "import lmdb" >/dev/null 2>&1
}

install_lmdb() {
  echo "[PEER] Installing lmdb"

  "$PYTHON_BIN" -m pip uninstall -y lmdb >/dev/null 2>&1 || true
  "$PYTHON_BIN" -m pip install cffi

  if "$PYTHON_BIN" -m pip install --no-cache-dir --force-reinstall --only-binary=:all: lmdb && check_lmdb_import; then
    return 0
  fi

  if ! command -v gcc >/dev/null 2>&1; then
    cat <<'EOF'
[PEER] Unable to install lmdb from a prebuilt wheel, and gcc is not available.
[PEER] On Ubuntu, install the build tools and rerun this script:

  sudo apt-get update
  sudo apt-get install -y build-essential python3-dev liblmdb-dev

EOF
    return 1
  fi

  echo "[PEER] Binary lmdb install did not import cleanly. Building from source with local compiler"
  if "$PYTHON_BIN" -m pip install --no-cache-dir --force-reinstall --no-binary=lmdb lmdb && check_lmdb_import; then
    return 0
  fi

  OS_NAME="$(uname -s)"
  if [ "$OS_NAME" = "Darwin" ]; then
    cat <<'EOF'
[PEER] lmdb is still not importable after reinstalling from source.
[PEER] On macOS, install the compiler tools and LMDB headers, then rerun this script:

  xcode-select --install
  brew install lmdb

EOF
    return 1
  fi

  cat <<'EOF'
[PEER] lmdb is still not importable after reinstalling from source.
[PEER] On Ubuntu, this usually means the LMDB development headers are missing.
[PEER] Install them and rerun this script:

  sudo apt-get update
  sudo apt-get install -y build-essential python3-dev liblmdb-dev

EOF
  return 1
}

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
if ! check_lmdb_import; then
  install_lmdb
else
  echo "[PEER] lmdb already available"
fi

echo "[PEER] Preparing official localization and solubility splits"
cd "$REPO_ROOT"
"$PYTHON_BIN" Code/scripts/prepare_peer_data.py

echo "[PEER] Setup complete"
