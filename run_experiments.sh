#!/usr/bin/env bash
# ==========================================================================
# run_experiments.sh -- Run CCOD WiFi RL experiments
#
# Prerequisites: run setup.sh first
#
# Usage:
#   ./run_experiments.sh                  # Full paper experiments (all)
#   ./run_experiments.sh quick            # Quick pipeline test
#   ./run_experiments.sh beb              # BEB + Lookup baselines only
#   ./run_experiments.sh ddpg             # DDPG training only
#   ./run_experiments.sh conv             # Convergence scenario only
#   ./run_experiments.sh extended         # Extended (up to 100 stations)
#   ./run_experiments.sh plot             # Generate plots only
#   ./run_experiments.sh plot extended    # Generate plots including extended
# ==========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS3_ROOT="$SCRIPT_DIR/ns3/ns-3-dev"
VENV_DIR="$NS3_ROOT/venv"
WORK_DIR="$NS3_ROOT/scratch/linear-mesh"

# ------------------------------------------------------------------
# Validate setup
# ------------------------------------------------------------------
if [ ! -d "$NS3_ROOT" ]; then
    echo "ERROR: ns-3 directory not found at $NS3_ROOT"
    echo "Please run ./setup.sh first."
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Python venv not found at $VENV_DIR"
    echo "Please run ./setup.sh first."
    exit 1
fi

BINARY="$NS3_ROOT/build/scratch/linear-mesh/ns3.40-cw-default"
if [ ! -f "$BINARY" ]; then
    echo "ERROR: Simulation binary not found at $BINARY"
    echo "Please run ./setup.sh first, or rebuild with:"
    echo "  cd $NS3_ROOT && cmake --build cmake-cache -j\$(nproc)"
    exit 1
fi

# ------------------------------------------------------------------
# Activate virtual environment
# ------------------------------------------------------------------
source "$VENV_DIR/bin/activate"
export NS3_DIR="$NS3_ROOT"

cd "$WORK_DIR"

echo "============================================================"
echo "CCOD WiFi RL Experiments"
echo "============================================================"
echo "ns-3 root:    $NS3_ROOT"
echo "Working dir:  $WORK_DIR"
echo "Python:       $(which python)"
echo ""

# ------------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------------
MODE="${1:-all}"
EXTRA="${2:-}"

case "$MODE" in
    quick)
        echo "Mode: QUICK (pipeline test)"
        echo ""
        python -u -m experiments.run_all --quick 2>&1 | tee results/experiment_quick_log.txt
        echo ""
        echo "Generating plots..."
        python -m experiments.plot_results
        ;;
    beb)
        echo "Mode: BEB + Lookup baselines"
        echo ""
        python -u -m experiments.run_all --beb-only 2>&1 | tee results/experiment_beb_log.txt
        ;;
    ddpg)
        echo "Mode: DDPG training"
        echo ""
        python -u -m experiments.run_all --ddpg-only 2>&1 | tee results/experiment_ddpg_log.txt
        ;;
    conv)
        echo "Mode: Convergence scenario"
        echo ""
        python -u -m experiments.run_all --conv-only 2>&1 | tee results/experiment_conv_log.txt
        ;;
    extended)
        echo "Mode: Extended experiments (up to 100 stations)"
        echo ""
        python -u -m experiments.run_all --extended 2>&1 | tee results/experiment_extended_log.txt
        echo ""
        echo "Generating extended plots..."
        python -m experiments.plot_results --extended
        ;;
    plot)
        echo "Mode: Plot generation only"
        echo ""
        if [ "$EXTRA" = "extended" ]; then
            python -m experiments.plot_results --extended
        else
            python -m experiments.plot_results
        fi
        ;;
    all)
        echo "Mode: FULL paper experiments"
        echo ""
        python -u -m experiments.run_all 2>&1 | tee results/experiment_full_log.txt
        echo ""
        echo "Generating plots..."
        python -m experiments.plot_results
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo ""
        echo "Usage:"
        echo "  ./run_experiments.sh              # Full paper experiments"
        echo "  ./run_experiments.sh quick         # Quick pipeline test"
        echo "  ./run_experiments.sh beb           # BEB + Lookup only"
        echo "  ./run_experiments.sh ddpg          # DDPG only"
        echo "  ./run_experiments.sh conv          # Convergence only"
        echo "  ./run_experiments.sh extended       # Extended (100 stations)"
        echo "  ./run_experiments.sh plot           # Generate plots"
        echo "  ./run_experiments.sh plot extended  # Extended plots"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "Done! Results are in: $WORK_DIR/results/"
echo "Figures are in:       $WORK_DIR/results/figures/"
echo "============================================================"
