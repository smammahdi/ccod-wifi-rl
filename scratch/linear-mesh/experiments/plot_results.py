"""
Plot generation for CCOD paper reproduction and extensions.

Reads JSON result files in results/ and produces matplotlib figures
equivalent to Figures 2-6 in the paper, plus fairness and training plots.

Usage:
    cd scratch/linear-mesh
    python -m experiments.plot_results
    python -m experiments.plot_results --extended   # Include extended results
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, STEP_TIME

FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(filename):
    """Load a JSON file from RESULTS_DIR. Returns None if not found."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"  [SKIP] {filename} not found")
        return None
    with open(path) as f:
        return json.load(f)


def sorted_items(d):
    """Return items sorted by integer key."""
    return sorted(d.items(), key=lambda x: int(x[0]))


def savefig(fig, name):
    """Save figure as PDF and PNG."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {name}.pdf / {name}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Throughput vs. Station Count (Static Topology)
# ---------------------------------------------------------------------------

def plot_fig2(prefix="static_"):
    """Throughput vs station count for BEB, Lookup, DDPG."""
    print("\nFigure 2: Throughput vs Station Count")

    beb = load_json(f"{prefix}beb.json")
    lookup = load_json(f"{prefix}lookup.json")
    ddpg = load_json(f"{prefix}ddpg.json")

    if not any([beb, lookup, ddpg]):
        print("  No data available, skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    if beb:
        xs, ys = zip(*sorted_items(beb))
        ax.plot([int(x) for x in xs], ys, "s-", label="BEB (Standard 802.11)",
                color="tab:blue", markersize=6)

    if lookup:
        xs, ys = zip(*sorted_items(lookup))
        ax.plot([int(x) for x in xs], ys, "^-", label="Lookup Table",
                color="tab:orange", markersize=6)

    if ddpg:
        xs, ys = zip(*sorted_items(ddpg))
        ax.plot([int(x) for x in xs], ys, "o-", label="DDPG",
                color="tab:green", markersize=6)

    ax.set_xlabel("Number of Stations")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("Fig. 2: Throughput vs Number of Stations (Static Topology)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Auto determine xticks from data
    all_xs = set()
    for d in [beb, lookup, ddpg]:
        if d:
            all_xs.update(int(k) for k in d.keys())
    if all_xs:
        ax.set_xticks(sorted(all_xs))

    label = "extended" if prefix == "extended_" else "paper"
    savefig(fig, f"fig2_throughput_vs_stations_{label}")


# ---------------------------------------------------------------------------
# Figure 3: Mean CW per Training Round
# ---------------------------------------------------------------------------

def plot_fig3(prefix="static_"):
    """Mean CW across training rounds for selected station counts."""
    print("\nFigure 3: Mean CW per Training Round")

    cw_hist = load_json(f"{prefix}ddpg_cw_history.json")
    if not cw_hist:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot CW learning for multiple station counts
    for key, cws in sorted_items(cw_hist):
        rounds = list(range(1, len(cws) + 1))
        ax.plot(rounds, cws, "o-", label=f"{key} stations", markersize=4)

    ax.set_xlabel("Training Round")
    ax.set_ylabel("Mean Contention Window")
    ax.set_title("Fig. 3: CW Convergence per Training Round")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    label = "extended" if prefix == "extended_" else "paper"
    savefig(fig, f"fig3_cw_learning_{label}")


# ---------------------------------------------------------------------------
# Figure 4 + 5: Dynamic scenario -- CW and throughput over time
# ---------------------------------------------------------------------------

def plot_fig4_fig5():
    """CW and throughput over time in dynamic (convergence) scenario."""
    print("\nFigure 4-5: Dynamic Scenario (CW and Throughput over time)")

    ddpg_conv = load_json("convergence_ddpg.json")
    beb_conv = load_json("convergence_beb.json")

    if not ddpg_conv and not beb_conv:
        print("  No data available, skipping.")
        return

    # --- Figure 4: CW over time ---
    if ddpg_conv and "timesteps" in ddpg_conv:
        ts = ddpg_conv["timesteps"]
        times = [d["time"] for d in ts]
        cws = [d["cw"] for d in ts]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times, cws, linewidth=0.5, color="tab:green", alpha=0.7)

        if len(cws) > 50:
            window = min(100, len(cws) // 5)
            smoothed = np.convolve(cws, np.ones(window) / window, mode="valid")
            ax.plot(times[window // 2: window // 2 + len(smoothed)],
                    smoothed, color="tab:green", linewidth=2, label="DDPG (smoothed)")
            ax.legend()

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Contention Window")
        ax.set_title("Fig. 4: CW Adaptation in Dynamic Topology")
        ax.grid(True, alpha=0.3)
        savefig(fig, "fig4_dynamic_cw")

    # --- Figure 5: Throughput over time ---
    fig, ax = plt.subplots(figsize=(10, 4))

    if beb_conv and "timesteps" in beb_conv:
        ts = beb_conv["timesteps"]
        times = [d["time"] for d in ts]
        thrs = [d["throughput_mbps"] for d in ts]

        if len(thrs) > 50:
            window = min(100, len(thrs) // 5)
            smoothed = np.convolve(thrs, np.ones(window) / window, mode="valid")
            ax.plot(times[window // 2: window // 2 + len(smoothed)],
                    smoothed, color="tab:blue", linewidth=2, label="BEB")

    if ddpg_conv and "timesteps" in ddpg_conv:
        ts = ddpg_conv["timesteps"]
        times = [d["time"] for d in ts]
        thrs = [d["throughput_mbps"] for d in ts]

        if len(thrs) > 50:
            window = min(100, len(thrs) // 5)
            smoothed = np.convolve(thrs, np.ones(window) / window, mode="valid")
            ax.plot(times[window // 2: window // 2 + len(smoothed)],
                    smoothed, color="tab:green", linewidth=2, label="DDPG")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("Fig. 5: Instantaneous Throughput (Dynamic Topology)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, "fig5_dynamic_throughput")


# ---------------------------------------------------------------------------
# Figure 6: Throughput vs Station Count (Dynamic)
# ---------------------------------------------------------------------------

def plot_fig6():
    """Throughput vs station count from dynamic runs."""
    print("\nFigure 6: Throughput vs Final Station Count (Dynamic)")

    ddpg_conv = load_json("convergence_ddpg.json")
    beb_conv = load_json("convergence_beb.json")

    if not ddpg_conv and not beb_conv:
        print("  No data available, skipping.")
        return

    def extract_per_station_throughput(data):
        """Group timestep data by station count and compute mean throughput."""
        if not data or "timesteps" not in data:
            return {}
        groups = {}
        for d in data["timesteps"]:
            n = int(d["stations"])
            if n not in groups:
                groups[n] = []
            groups[n].append(d["throughput_mbps"])
        return {n: np.mean(vs) for n, vs in groups.items()}

    fig, ax = plt.subplots(figsize=(8, 5))

    if beb_conv:
        beb_groups = extract_per_station_throughput(beb_conv)
        if beb_groups:
            xs, ys = zip(*sorted(beb_groups.items()))
            ax.plot(xs, ys, "s-", label="BEB", color="tab:blue", markersize=6)

    if ddpg_conv:
        ddpg_groups = extract_per_station_throughput(ddpg_conv)
        if ddpg_groups:
            xs, ys = zip(*sorted(ddpg_groups.items()))
            ax.plot(xs, ys, "o-", label="DDPG", color="tab:green", markersize=6)

    ax.set_xlabel("Number of Stations")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("Fig. 6: Throughput vs Station Count (Dynamic Scenario)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, "fig6_dynamic_final")


# ---------------------------------------------------------------------------
# Fairness comparison
# ---------------------------------------------------------------------------

def plot_fairness(prefix="static_"):
    """Jain's fairness index vs station count for all methods."""
    print("\nFairness: Jain's Fairness Index vs Station Count")

    beb_fair = load_json(f"{prefix}beb_fairness.json")
    lookup_fair = load_json(f"{prefix}lookup_fairness.json")
    ddpg_fair_hist = load_json(f"{prefix}ddpg_fair_history.json")

    if not any([beb_fair, lookup_fair, ddpg_fair_hist]):
        print("  No fairness data available, skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    if beb_fair:
        xs, ys = zip(*sorted_items(beb_fair))
        ax.plot([int(x) for x in xs], ys, "s-", label="BEB",
                color="tab:blue", markersize=6)

    if lookup_fair:
        xs, ys = zip(*sorted_items(lookup_fair))
        ax.plot([int(x) for x in xs], ys, "^-", label="Lookup Table",
                color="tab:orange", markersize=6)

    if ddpg_fair_hist:
        # Use the last round (operational) fairness for each station count
        ddpg_fair = {k: v[-1] for k, v in ddpg_fair_hist.items() if v}
        if ddpg_fair:
            xs, ys = zip(*sorted_items(ddpg_fair))
            ax.plot([int(x) for x in xs], ys, "o-", label="DDPG",
                    color="tab:green", markersize=6)

    ax.set_xlabel("Number of Stations")
    ax.set_ylabel("Jain's Fairness Index")
    ax.set_title("Fairness Comparison: Jain's Index vs Station Count")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    label = "extended" if prefix == "extended_" else "paper"
    savefig(fig, f"fairness_vs_stations_{label}")


# ---------------------------------------------------------------------------
# Training Curves (Throughput per round)
# ---------------------------------------------------------------------------

def plot_training_curves(prefix="static_"):
    """Throughput per training round for each station count."""
    print("\nTraining Curves")

    thr_hist = load_json(f"{prefix}ddpg_thr_history.json")
    if not thr_hist:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for key, vals in sorted_items(thr_hist):
        rounds = list(range(1, len(vals) + 1))
        ax.plot(rounds, vals, "o-", label=f"{key} stations", markersize=4)

    ax.set_xlabel("Training Round")
    ax.set_ylabel("Mean Throughput (Mbps)")
    ax.set_title("Training Progress: Throughput per Round")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    label = "extended" if prefix == "extended_" else "paper"
    savefig(fig, f"training_curves_{label}")


# ===================================================================
# Main
# ===================================================================

def main():
    extended = "--extended" in sys.argv

    print("=" * 60)
    print("CCOD Paper -- Plot Generation")
    print(f"Reading results from: {RESULTS_DIR}")
    print(f"Saving figures to:    {FIGURES_DIR}")
    print("=" * 60)

    # Paper results (prefix "static_")
    plot_fig2("static_")
    plot_fig3("static_")
    plot_fig4_fig5()
    plot_fig6()
    plot_fairness("static_")
    plot_training_curves("static_")

    # Extended results (if available)
    if extended:
        print("\n--- Extended Plots ---")
        plot_fig2("extended_")
        plot_fig3("extended_")
        plot_fairness("extended_")
        plot_training_curves("extended_")

    print("\nDone.")


if __name__ == "__main__":
    main()
