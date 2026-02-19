#!/usr/bin/env bash
# ==========================================================================
# post-create.sh -- Runs automatically when Codespace is created
#
# Installs system dependencies and prepares the environment.
# Does NOT build ns-3 (takes too long for post-create).
# Run ./setup.sh manually after Codespace starts.
# ==========================================================================

set -euo pipefail

echo "============================================================"
echo "CCOD WiFi RL -- Codespace Post-Create Setup"
echo "============================================================"

# System packages for ns-3.40
echo "[1/3] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential cmake ninja-build \
    g++ \
    libzmq3-dev libprotobuf-dev protobuf-compiler \
    pkg-config wget tar bzip2 \
    libsqlite3-dev libxml2-dev \
    2>&1 | tail -3

echo "[2/3] Installing Python packages..."
pip install --upgrade pip -q
pip install -q \
    torch --index-url https://download.pytorch.org/whl/cpu \
    numpy matplotlib tqdm pyzmq gymnasium gym protobuf pandas

echo "[3/3] Done with post-create setup."
echo ""
echo "============================================================"
echo "Next steps:"
echo "  ./setup.sh               # Download ns-3.40 and build (~5-10 min)"
echo "  ./run_experiments.sh     # Run experiments"
echo "============================================================"
