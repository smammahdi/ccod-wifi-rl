#!/usr/bin/env bash
# ==========================================================================
# collect_results.sh -- Copy results from ns-3 tree back to repo & push
#
# Usage:
#   ./collect_results.sh              # Copy results to repo
#   ./collect_results.sh --push       # Copy + commit + push to GitHub
#   ./collect_results.sh --push -m "msg"  # Custom commit message
# ==========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS3_ROOT="$SCRIPT_DIR/ns3/ns-3-dev"
WORK_DIR="$NS3_ROOT/scratch/linear-mesh"
SRC_RESULTS="$WORK_DIR/results"
DST_RESULTS="$SCRIPT_DIR/results"
LOGS_DIR="$SCRIPT_DIR/logs"

# ---- Copy results ----
if [ ! -d "$SRC_RESULTS" ]; then
    echo "No results directory found at: $SRC_RESULTS"
    echo "Run experiments first: ./run_parallel.sh all"
    exit 1
fi

mkdir -p "$DST_RESULTS"
echo "Copying results from ns-3 tree to repo..."
cp -R "$SRC_RESULTS/"* "$DST_RESULTS/" 2>/dev/null || true

# Count what we got
N_JSON=$(find "$DST_RESULTS" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
N_FIG=$(find "$DST_RESULTS" -name "*.pdf" -o -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
echo "  Copied: $N_JSON JSON files, $N_FIG figure files"

# ---- Copy any logs ----
if [ -d "$WORK_DIR/logs" ]; then
    mkdir -p "$LOGS_DIR"
    cp -R "$WORK_DIR/logs/"* "$LOGS_DIR/" 2>/dev/null || true
    N_LOG=$(find "$LOGS_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "  Copied: $N_LOG log files"
fi

# ---- Summary ----
echo ""
echo "Results in repo: $DST_RESULTS/"
[ -d "$LOGS_DIR" ] && echo "Logs in repo:    $LOGS_DIR/"
echo ""

# List result files
echo "--- Result Files ---"
ls -la "$DST_RESULTS/"*.json 2>/dev/null || echo "  (no JSON files)"
echo ""
if [ -d "$DST_RESULTS/figures" ]; then
    echo "--- Figures ---"
    ls -la "$DST_RESULTS/figures/" 2>/dev/null || echo "  (no figures)"
    echo ""
fi

# ---- Optional: push to git ----
if [ "${1:-}" = "--push" ]; then
    cd "$SCRIPT_DIR"
    MSG="${3:-results: $(date '+%Y-%m-%d %H:%M') experiment outputs}"
    if [ "${2:-}" = "-m" ] && [ -n "${3:-}" ]; then
        MSG="$3"
    fi
    
    git add results/ logs/ 2>/dev/null || git add results/
    git commit -m "$MSG" || { echo "Nothing new to commit."; exit 0; }
    git push
    echo ""
    echo "Pushed to GitHub! Pull locally with: git pull"
fi
