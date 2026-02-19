#!/usr/bin/env bash
# ==========================================================================
# sync.sh -- Re-copy project code into ns-3 tree after git pull
#
# Use this after pulling changes to update the ns-3 tree without
# re-downloading or rebuilding everything.
#
# Usage:
#   ./sync.sh            # Copy Python + C++ files
#   ./sync.sh --rebuild  # Copy files AND rebuild ns-3 (if C++ changed)
# ==========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS3_ROOT="$SCRIPT_DIR/ns3/ns-3-dev"
WORK_DIR="$NS3_ROOT/scratch/linear-mesh"

if [ ! -d "$NS3_ROOT" ]; then
    echo "ERROR: ns-3 not found. Run ./setup.sh first."
    exit 1
fi

echo "Syncing project code into ns-3 tree..."

# Copy simulation code
cp "$SCRIPT_DIR/scratch/linear-mesh/cw.cc" "$WORK_DIR/"
cp "$SCRIPT_DIR/scratch/linear-mesh/scenario.h" "$WORK_DIR/"
cp "$SCRIPT_DIR/scratch/linear-mesh/preprocessor.py" "$WORK_DIR/"
cp "$SCRIPT_DIR/scratch/linear-mesh/CW_data.csv" "$WORK_DIR/"

# Copy agents
mkdir -p "$WORK_DIR/agents/"
cp -R "$SCRIPT_DIR/scratch/linear-mesh/agents/"* "$WORK_DIR/agents/"

# Copy experiments
mkdir -p "$WORK_DIR/experiments/"
cp -R "$SCRIPT_DIR/scratch/linear-mesh/experiments/"* "$WORK_DIR/experiments/"

# Copy opengym module
cp -R "$SCRIPT_DIR/ns3-opengym/"* "$NS3_ROOT/contrib/opengym/"

echo "Files synced."

if [ "${1:-}" = "--rebuild" ]; then
    echo "Rebuilding ns-3..."
    cd "$NS3_ROOT"
    NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    cmake --build cmake-cache -j"$NPROC" 2>&1 | tail -10
    echo "Build complete."
fi

echo "Done."
