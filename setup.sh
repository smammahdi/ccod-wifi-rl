#!/usr/bin/env bash
# ==========================================================================
# setup.sh -- One-shot setup for CCOD WiFi RL experiments on Ubuntu
#
# This script:
#   1. Installs system dependencies (cmake, g++, protobuf, zmq, python3)
#   2. Downloads ns-3.40 source
#   3. Copies project code (simulation, opengym module) into ns-3 tree
#   4. Applies the PHY idle-state fix patch
#   5. Builds ns-3 with opengym
#   6. Creates Python virtual environment with all dependencies
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# After setup, run experiments with:
#   ./run_experiments.sh
# ==========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS3_VERSION="3.40"
NS3_DIR="$SCRIPT_DIR/ns3"
NS3_URL="https://www.nsnam.org/releases/ns-allinone-${NS3_VERSION}.tar.bz2"
VENV_DIR="$NS3_DIR/ns-3-dev/venv"

echo "============================================================"
echo "CCOD WiFi RL - Setup Script"
echo "============================================================"
echo "Script directory:  $SCRIPT_DIR"
echo "ns-3 directory:    $NS3_DIR"
echo ""

# ------------------------------------------------------------------
# Step 1: System dependencies
# ------------------------------------------------------------------
echo "[1/6] Installing system dependencies..."

if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        build-essential cmake ninja-build \
        g++ python3 python3-dev python3-pip python3-venv \
        libzmq3-dev libprotobuf-dev protobuf-compiler \
        pkg-config wget tar bzip2 \
        libsqlite3-dev libxml2-dev \
        git
elif command -v brew &>/dev/null; then
    echo "macOS detected -- using Homebrew"
    brew install cmake ninja protobuf zeromq pkg-config wget
else
    echo "ERROR: Neither apt-get nor brew found. Install dependencies manually."
    exit 1
fi

echo "   System dependencies installed."

# ------------------------------------------------------------------
# Step 2: Download ns-3.40
# ------------------------------------------------------------------
echo ""
echo "[2/6] Setting up ns-3.40..."

if [ -d "$NS3_DIR/ns-3-dev" ]; then
    echo "   ns-3 directory already exists, skipping download."
else
    echo "   Downloading ns-allinone-${NS3_VERSION}..."
    cd "$SCRIPT_DIR"
    if [ ! -f "ns-allinone-${NS3_VERSION}.tar.bz2" ]; then
        wget -q --show-progress "$NS3_URL"
    fi
    echo "   Extracting..."
    tar xjf "ns-allinone-${NS3_VERSION}.tar.bz2"
    mv "ns-allinone-${NS3_VERSION}" "$NS3_DIR"
    # The actual ns-3 source is inside ns-allinone-X.XX/ns-X.XX/
    # Rename to ns-3-dev for consistency
    if [ -d "$NS3_DIR/ns-${NS3_VERSION}" ] && [ ! -d "$NS3_DIR/ns-3-dev" ]; then
        mv "$NS3_DIR/ns-${NS3_VERSION}" "$NS3_DIR/ns-3-dev"
    fi
    echo "   ns-3.40 extracted to $NS3_DIR/ns-3-dev"
fi

NS3_ROOT="$NS3_DIR/ns-3-dev"

# ------------------------------------------------------------------
# Step 3: Copy project code into ns-3 tree
# ------------------------------------------------------------------
echo ""
echo "[3/6] Copying project code into ns-3 tree..."

# Copy opengym module to contrib/
if [ ! -d "$NS3_ROOT/contrib/opengym" ]; then
    mkdir -p "$NS3_ROOT/contrib"
fi
cp -R "$SCRIPT_DIR/ns3-opengym/" "$NS3_ROOT/contrib/opengym/"
echo "   Copied ns3-opengym -> contrib/opengym/"

# Copy simulation code to scratch/linear-mesh/
mkdir -p "$NS3_ROOT/scratch/linear-mesh"
# Copy C++ simulation files
cp "$SCRIPT_DIR/scratch/linear-mesh/cw.cc" "$NS3_ROOT/scratch/linear-mesh/"
cp "$SCRIPT_DIR/scratch/linear-mesh/scenario.h" "$NS3_ROOT/scratch/linear-mesh/"
# Copy Python files
cp "$SCRIPT_DIR/scratch/linear-mesh/preprocessor.py" "$NS3_ROOT/scratch/linear-mesh/"
cp "$SCRIPT_DIR/scratch/linear-mesh/CW_data.csv" "$NS3_ROOT/scratch/linear-mesh/"
# Copy agents directory
cp -R "$SCRIPT_DIR/scratch/linear-mesh/agents" "$NS3_ROOT/scratch/linear-mesh/"
# Copy experiments directory
cp -R "$SCRIPT_DIR/scratch/linear-mesh/experiments" "$NS3_ROOT/scratch/linear-mesh/"
echo "   Copied simulation code -> scratch/linear-mesh/"

# ------------------------------------------------------------------
# Step 4: Apply PHY patch
# ------------------------------------------------------------------
echo ""
echo "[4/6] Applying PHY idle-state fix..."

PHY_FILE="$NS3_ROOT/src/wifi/model/phy-entity.cc"
if grep -q "aborting stale reception" "$PHY_FILE" 2>/dev/null; then
    echo "   Patch already applied, skipping."
else
    # Apply the patch manually since the patch file format may vary
    # Find the "case WifiPhyState::IDLE:" line and add the fix after it
    PATCH_APPLIED=false

    # Try using patch command first
    if patch -p1 --dry-run -d "$NS3_ROOT" < "$SCRIPT_DIR/patches/phy-entity-idle-fix.patch" &>/dev/null; then
        patch -p1 -d "$NS3_ROOT" < "$SCRIPT_DIR/patches/phy-entity-idle-fix.patch"
        PATCH_APPLIED=true
    fi

    if [ "$PATCH_APPLIED" = false ]; then
        # Manual patching with sed
        echo "   patch command failed, applying manually with sed..."
        # Find line number of "case WifiPhyState::IDLE:"
        LINE=$(grep -n "case WifiPhyState::IDLE:" "$PHY_FILE" | head -1 | cut -d: -f1)
        if [ -n "$LINE" ]; then
            # Insert the fix after the IDLE case line
            sed -i "${LINE}a\\
        // Workaround for ns-3.40 race condition: PHY can be IDLE but still\\
        // have a stale currentEvent when many stations cause overlapping\\
        // receptions.  Properly abort the stale reception before proceeding.\\
        if (m_wifiPhy->m_currentEvent)\\
        {\\
            NS_LOG_DEBUG(\"IDLE but currentEvent not null -- aborting stale reception\");\\
            AbortCurrentReception(FRAME_CAPTURE_PACKET_SWITCH);\\
        }" "$PHY_FILE"
            PATCH_APPLIED=true
        fi
    fi

    if [ "$PATCH_APPLIED" = true ]; then
        echo "   PHY patch applied successfully."
    else
        echo "   WARNING: Could not apply PHY patch. Please apply manually."
        echo "   See patches/phy-entity-idle-fix.patch"
    fi
fi

# ------------------------------------------------------------------
# Step 5: Build ns-3
# ------------------------------------------------------------------
echo ""
echo "[5/6] Building ns-3 (this may take several minutes)..."

cd "$NS3_ROOT"

# Configure
./ns3 configure --enable-examples --enable-tests 2>&1 | tail -5

# Build (use all available cores)
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "   Building with $NPROC parallel jobs..."
cmake --build cmake-cache -j"$NPROC" 2>&1 | tail -10

# Verify the simulation binary exists
BINARY="$NS3_ROOT/build/scratch/linear-mesh/ns3.40-cw-default"
if [ -f "$BINARY" ]; then
    echo "   Build successful! Binary: $BINARY"
else
    echo "   WARNING: Binary not found at expected path."
    echo "   Looking for alternatives..."
    find "$NS3_ROOT/build/scratch" -name "*cw*" -type f 2>/dev/null
fi

# ------------------------------------------------------------------
# Step 6: Python virtual environment
# ------------------------------------------------------------------
echo ""
echo "[6/6] Setting up Python virtual environment..."

if [ -d "$VENV_DIR" ]; then
    echo "   venv already exists, updating packages..."
else
    python3 -m venv "$VENV_DIR"
    echo "   Created venv at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel -q

# Core dependencies
pip install -q \
    torch \
    numpy \
    matplotlib \
    tqdm \
    pyzmq \
    gymnasium \
    gym \
    protobuf \
    pandas

# Install ns3gym package (from opengym model directory)
NS3GYM_DIR="$NS3_ROOT/contrib/opengym/model/ns3gym"
if [ -d "$NS3GYM_DIR" ]; then
    pip install -e "$NS3GYM_DIR" -q
    echo "   Installed ns3gym package"
else
    echo "   WARNING: ns3gym directory not found at $NS3GYM_DIR"
fi

echo ""
echo "============================================================"
echo "Setup complete!"
echo ""
echo "  ns-3 root:      $NS3_ROOT"
echo "  Python venv:    $VENV_DIR"
echo "  Simulation:     $NS3_ROOT/scratch/linear-mesh/"
echo ""
echo "To run experiments:"
echo "  ./run_experiments.sh"
echo ""
echo "Or manually:"
echo "  source $VENV_DIR/bin/activate"
echo "  cd $NS3_ROOT/scratch/linear-mesh"
echo "  python -u -m experiments.run_all"
echo "============================================================"
