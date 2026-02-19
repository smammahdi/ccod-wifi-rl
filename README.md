# CCOD WiFi RL: Contention Window Optimization with Deep Reinforcement Learning

Reproduction and extension of:

> Wydmanski & Szott, "Contention Window Optimization in IEEE 802.11ax Networks
> with Deep Reinforcement Learning," IEEE WCNC 2021

**Student**: Shah Mohammad Abdul Mannan (2105056)
**Course**: CSE 322 -- Computer Networks Sessional, BUET

## Overview

This project uses deep reinforcement learning (DDPG) to dynamically optimize
the 802.11 contention window (CW) parameter, replacing the standard Binary
Exponential Backoff (BEB) mechanism. The agent observes collision probability
history and outputs a CW value that maximizes network throughput.

## Repository Structure

```
.
├── setup.sh                        # Install deps, download ns-3.40, build
├── run_experiments.sh              # Run experiments
├── requirements.txt                # Python dependencies
├── patches/
│   └── phy-entity-idle-fix.patch   # PHY fix for ns-3.40
├── ns3-opengym/                    # ns3-gym C++ module (contrib/opengym)
│   ├── CMakeLists.txt
│   ├── model/                      # C++ model + Python ns3gym package
│   ├── helper/
│   └── examples/
└── scratch/linear-mesh/            # Main project code
    ├── cw.cc                       # ns-3 simulation (802.11ax WiFi)
    ├── scenario.h                  # Static and convergence scenarios
    ├── preprocessor.py             # Observation preprocessor (windowed stats)
    ├── CW_data.csv                 # Lookup table data
    ├── agents/
    │   ├── teacher.py              # Training loop
    │   ├── loggers.py              # CSV logging
    │   ├── ddpg/                   # DDPG agent (LSTM Actor-Critic)
    │   └── dqn/                    # DQN agent
    └── experiments/
        ├── config.py               # All experiment parameters
        ├── run_all.py              # Experiment runner
        └── plot_results.py         # Plot generation
```

## Quick Start

### 1. Setup (one-time, ~10 min)

```bash
git clone <repo-url> ccod-wifi-rl
cd ccod-wifi-rl
chmod +x setup.sh run_experiments.sh
./setup.sh
```

This downloads ns-3.40, copies the code, applies patches, builds, and sets up
the Python environment.

### 2. Run Experiments

```bash
# Quick test (5 min)
./run_experiments.sh quick

# Full paper experiments (2+ hours)
./run_experiments.sh

# Extended experiments with 100 stations
./run_experiments.sh extended

# Individual experiment types
./run_experiments.sh beb          # BEB + Lookup baselines
./run_experiments.sh ddpg         # DDPG training
./run_experiments.sh conv         # Dynamic topology convergence

# Generate plots only
./run_experiments.sh plot
./run_experiments.sh plot extended
```

### 3. Results

Results are saved to `ns3/ns-3-dev/scratch/linear-mesh/results/`:
- `static_beb.json` -- BEB throughput per station count
- `static_lookup.json` -- Lookup table throughput
- `static_ddpg.json` -- DDPG final throughput
- `static_*_fairness.json` -- Jain's fairness index
- `convergence_*.json` -- Dynamic topology data
- `figures/` -- Generated plots (PDF + PNG)

## Experiments

### Paper Reproduction (Figures 2-6)

| Experiment | Description | Stations |
|------------|-------------|----------|
| BEB Baseline | Standard 802.11 backoff | 5-45 |
| Lookup Table | Fixed optimal CW per station count | 5-45 |
| DDPG Static | RL-optimized CW per station count | 5-45 |
| Convergence | Dynamic topology (5 -> max stations) | max |

### Extensions

- **100 stations**: Tests scalability beyond the paper
- **Fairness**: Jain's fairness index tracked for all methods
- **Algorithm comparison**: DDPG vs DQN vs PPO (planned)

## Technical Notes

### ns-3.40 Compatibility

The original RLinWiFi code targeted ns-3.29. Six API fixes were applied for
ns-3.40 compatibility (see `patches/` and code comments in `cw.cc`):

1. `YansWifiPhyHelper::Default()` removed
2. `WIFI_PHY_STANDARD_80211ax_5GHZ` renamed
3. GuardInterval moved from PHY to HeConfiguration
4. ChannelWidth replaced by ChannelSettings string
5. `RegularWifiMac` merged into `WifiMac`
6. `PhyTxBegin` callback signature changed

### PHY Fix

A race condition in ns-3.40 causes assertion failures when many stations
(>25) cause overlapping receptions. The fix in `patches/phy-entity-idle-fix.patch`
properly aborts stale receptions using `AbortCurrentReception()`.

## Development Workflow

1. Make changes on local machine
2. `git push`
3. On remote machine: `git pull`
4. Re-copy changed files: `./setup.sh` (idempotent)
5. Rebuild if C++ changed: `cd ns3/ns-3-dev && cmake --build cmake-cache -j$(nproc)`
6. Run experiments: `./run_experiments.sh`

## References

- [RLinWiFi](https://github.com/wwydmanski/RLinWiFi) -- Original implementation
- [ns3-gym](https://github.com/tkn-tub/ns3-gym) -- ns-3 OpenAI Gym interface
- Paper: Wydmanski & Szott, WCNC 2021
