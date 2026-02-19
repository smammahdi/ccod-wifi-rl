#!/usr/bin/env bash
# ==========================================================================
# run_parallel.sh -- Run experiments with FULL CPU utilization
#
# Directly launches one ns-3 + Python process per station count using
# bash background jobs (&). No Python multiprocessing overhead.
#
# Usage:
#   ./run_parallel.sh beb          # BEB baselines (all stations in parallel)
#   ./run_parallel.sh lookup       # Lookup baselines (all stations parallel)
#   ./run_parallel.sh ddpg [N]     # DDPG training (N parallel, default 4)
#   ./run_parallel.sh baselines    # BEB + Lookup (both parallel)
#   ./run_parallel.sh all [N]      # Everything (baselines parallel, DDPG N-way)
#   ./run_parallel.sh extended     # Extended station counts
# ==========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS3_ROOT="$SCRIPT_DIR/ns3/ns-3-dev"
VENV_DIR="$NS3_ROOT/venv"
WORK_DIR="$NS3_ROOT/scratch/linear-mesh"
RESULTS="$WORK_DIR/results"
TMP="$RESULTS/_parallel_tmp"

# Validate
if [ ! -d "$NS3_ROOT" ] || [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Run ./setup.sh first"
    exit 1
fi

source "$VENV_DIR/bin/activate"
export NS3_DIR="$NS3_ROOT"
cd "$WORK_DIR"
mkdir -p "$RESULTS" "$TMP"

# Log everything to file
LOGFILE="$RESULTS/run_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOGFILE") 2>&1
echo "Log file: $LOGFILE"

# Verify run_single.py exists
if [ ! -f "$WORK_DIR/experiments/run_single.py" ]; then
    echo "ERROR: experiments/run_single.py not found in ns-3 tree."
    echo "Run: git pull && ./sync.sh"
    exit 1
fi

# Station counts
STATIONS_PAPER="5 10 15 20 25 30 35 40 45"
STATIONS_EXTENDED="5 10 15 20 25 30 35 40 45 60 80 100"
SIM_TIME=60
EPISODES=15

MODE="${1:-all}"
DDPG_PARALLEL="${2:-4}"

# Port allocation: direct increment, no subshell
PORT=8000

echo "============================================================"
echo "PARALLEL EXPERIMENT RUNNER"
echo "============================================================"
echo "Mode: $MODE"
echo "CPUs available: $(nproc)"
echo ""

# ------------------------------------------------------------------
# BEB: launch ALL station counts at once
# ------------------------------------------------------------------
run_beb() {
    local stations="$1"
    local prefix="$2"
    echo "--- BEB Baseline (parallel) ---"
    local pids=()
    for n in $stations; do
        PORT=$((PORT + 1))
        local p=$PORT
        local out="$TMP/beb_${n}.json"
        echo "  Launching BEB n=$n on port $p"
        python -u -m experiments.run_single beb "$n" "$p" "$SIM_TIME" "$out" &
        pids+=($!)
    done
    echo "  Waiting for ${#pids[@]} BEB jobs..."
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    echo "  BEB complete. Aggregating..."
    python -u -c "
import json, os, glob, sys
thr, fair = {}, {}
prefix = '$prefix'
for f in sorted(glob.glob('$TMP/beb_*.json')):
    n = int(os.path.basename(f).replace('beb_','').replace('.json',''))
    d = json.load(open(f))
    thr[n] = d['throughput']
    fair[n] = d['fairness']
    print(f'  n={n}: thr={d[\"throughput\"]:.2f} Mbps, fair={d[\"fairness\"]:.4f}')
json.dump(thr, open('$RESULTS/' + prefix + 'beb.json','w'), indent=2)
json.dump(fair, open('$RESULTS/' + prefix + 'beb_fairness.json','w'), indent=2)
print(f'  Saved {prefix}beb.json, {prefix}beb_fairness.json')
"
}

# ------------------------------------------------------------------
# Lookup: launch ALL station counts at once
# ------------------------------------------------------------------
run_lookup() {
    local stations="$1"
    local prefix="$2"
    echo "--- Lookup Baseline (parallel) ---"
    local pids=()
    for n in $stations; do
        PORT=$((PORT + 1))
        local p=$PORT
        local out="$TMP/lookup_${n}.json"
        echo "  Launching Lookup n=$n on port $p"
        python -u -m experiments.run_single lookup "$n" "$p" "$SIM_TIME" "$out" &
        pids+=($!)
    done
    echo "  Waiting for ${#pids[@]} Lookup jobs..."
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    echo "  Lookup complete. Aggregating..."
    python -u -c "
import json, os, glob, sys
thr, fair = {}, {}
prefix = '$prefix'
for f in sorted(glob.glob('$TMP/lookup_*.json')):
    n = int(os.path.basename(f).replace('lookup_','').replace('.json',''))
    d = json.load(open(f))
    thr[n] = d['throughput']
    fair[n] = d['fairness']
    print(f'  n={n}: thr={d[\"throughput\"]:.2f} Mbps, fair={d[\"fairness\"]:.4f}')
json.dump(thr, open('$RESULTS/' + prefix + 'lookup.json','w'), indent=2)
json.dump(fair, open('$RESULTS/' + prefix + 'lookup_fairness.json','w'), indent=2)
print(f'  Saved {prefix}lookup.json, {prefix}lookup_fairness.json')
"
}

# ------------------------------------------------------------------
# DDPG: run N station counts in parallel (episodes are sequential per station)
# ------------------------------------------------------------------
run_ddpg() {
    local stations="$1"
    local prefix="$2"
    local max_par="${3:-4}"
    echo "--- DDPG Training (${max_par}-way parallel) ---"

    local station_arr=($stations)
    local total=${#station_arr[@]}
    local idx=0
    local running=0
    local pids=()
    local nwifis=()

    while [ $idx -lt $total ] || [ $running -gt 0 ]; do
        # Launch new jobs up to max_par
        while [ $idx -lt $total ] && [ $running -lt $max_par ]; do
            local n=${station_arr[$idx]}
            PORT=$((PORT + 1))
            local p=$PORT
            # Reserve enough ports for all episodes
            PORT=$((PORT + EPISODES))
            local out="$TMP/ddpg_${n}.json"
            echo "  Launching DDPG n=$n (port_base=$p, $EPISODES episodes)"
            python -u -m experiments.run_single ddpg "$n" "$p" "$SIM_TIME" "$EPISODES" "$out" &
            pids+=($!)
            nwifis+=($n)
            idx=$((idx + 1))
            running=$((running + 1))
        done

        # Wait for any one to finish
        if [ $running -gt 0 ]; then
            local new_pids=()
            local new_nwifis=()
            local found_done=false
            while ! $found_done; do
                sleep 5
                local tmp_pids=()
                local tmp_nwifis=()
                for i in "${!pids[@]}"; do
                    if kill -0 "${pids[$i]}" 2>/dev/null; then
                        tmp_pids+=("${pids[$i]}")
                        tmp_nwifis+=("${nwifis[$i]}")
                    else
                        echo "  DDPG n=${nwifis[$i]} finished (pid=${pids[$i]})"
                        running=$((running - 1))
                        found_done=true
                    fi
                done
                pids=("${tmp_pids[@]+"${tmp_pids[@]}"}")
                nwifis=("${tmp_nwifis[@]+"${tmp_nwifis[@]}"}")
            done
        fi
    done

    echo "  All DDPG training complete. Aggregating..."
    python -u -c "
import json, os, glob
results, cw_hist, thr_hist, fair_hist = {}, {}, {}, {}
prefix = '$prefix'
for f in sorted(glob.glob('$TMP/ddpg_*.json')):
    n = int(os.path.basename(f).replace('ddpg_','').replace('.json',''))
    d = json.load(open(f))
    results[n] = d['final_throughput']
    thr_hist[n] = d['thr_history']
    cw_hist[n] = d['cw_history']
    fair_hist[n] = d['fair_history']
    print(f'  n={n}: final_thr={d[\"final_throughput\"]:.2f} Mbps')
json.dump(results, open('$RESULTS/' + prefix + 'ddpg.json','w'), indent=2)
json.dump(thr_hist, open('$RESULTS/' + prefix + 'ddpg_thr_history.json','w'), indent=2)
json.dump(cw_hist, open('$RESULTS/' + prefix + 'ddpg_cw_history.json','w'), indent=2)
json.dump(fair_hist, open('$RESULTS/' + prefix + 'ddpg_fair_history.json','w'), indent=2)
print(f'  Saved {prefix}ddpg*.json')
"
}

# ------------------------------------------------------------------
# Convergence: BEB baseline + DDPG
# ------------------------------------------------------------------
run_convergence() {
    echo "--- Convergence Experiment ---"
    python -u -m experiments.run_all --conv-only
}

# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
run_plots() {
    local extra="${1:-}"
    echo "--- Generating Plots ---"
    if [ "$extra" = "extended" ]; then
        python -m experiments.plot_results --extended
    else
        python -m experiments.plot_results
    fi
}

# ------------------------------------------------------------------
# Main dispatch
# ------------------------------------------------------------------
case "$MODE" in
    beb)
        run_beb "$STATIONS_PAPER" "static_"
        ;;
    lookup)
        run_lookup "$STATIONS_PAPER" "static_"
        ;;
    baselines)
        run_beb "$STATIONS_PAPER" "static_"
        run_lookup "$STATIONS_PAPER" "static_"
        ;;
    ddpg)
        run_ddpg "$STATIONS_PAPER" "static_" "$DDPG_PARALLEL"
        ;;
    conv)
        run_convergence
        ;;
    all)
        run_beb "$STATIONS_PAPER" "static_"
        run_lookup "$STATIONS_PAPER" "static_"
        run_ddpg "$STATIONS_PAPER" "static_" "$DDPG_PARALLEL"
        run_convergence
        run_plots
        ;;
    extended)
        run_beb "$STATIONS_EXTENDED" "extended_"
        run_lookup "$STATIONS_EXTENDED" "extended_"
        run_ddpg "$STATIONS_EXTENDED" "extended_" "$DDPG_PARALLEL"
        run_plots "extended"
        ;;
    plot)
        run_plots "${2:-}"
        ;;
    *)
        echo "Usage: $0 {beb|lookup|baselines|ddpg|conv|all|extended|plot}"
        exit 1
        ;;
esac

# Cleanup tmp
rm -rf "$TMP"

echo ""
echo "============================================================"
echo "Done! Results: $RESULTS/"
echo "Figures: $RESULTS/figures/"
echo "============================================================"
