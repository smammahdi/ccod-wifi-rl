"""
Experiment configuration for CCOD reproduction and extension.

Base Paper: "Contention Window Optimization in IEEE 802.11ax Networks with Deep
Reinforcement Learning" (Wydmanski & Szott, WCNC 2021)

Extension: Up to 100 stations, fairness evaluation, PPO comparison.

All experiment parameters centralized here.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# NS3_DIR is set by setup.sh via environment variable, or defaults to
# traversing up from scratch/linear-mesh/ to the ns-3-dev root.
NS3_DIR = os.environ.get(
    "NS3_DIR",
    os.path.dirname(os.path.dirname(BASE_DIR)),
)

# ---------------------------------------------------------------------------
# Simulation parameters (from paper Section IV)
# ---------------------------------------------------------------------------
SIM_TIME = 60           # seconds per round
STEP_TIME = 0.01        # 10 ms interaction period
HISTORY_LENGTH = 300    # observation history

# ---------------------------------------------------------------------------
# DRL hyperparameters (from paper Table I)
# ---------------------------------------------------------------------------
LR_ACTOR = 4e-4
LR_CRITIC = 4e-3
BATCH_SIZE = 32
GAMMA = 0.7
TAU = 1e-3
REPLAY_BUFFER_MULTIPLIER = 4   # buffer_size = multiplier * steps_per_ep
UPDATE_EVERY = 1

# Neural network architecture: LSTM(8) -> Dense(128) -> Dense(64)
ACTOR_LAYERS = [8, 128, 64]
CRITIC_LAYERS = [8, 128, 64]

# ---------------------------------------------------------------------------
# Training parameters
# ---------------------------------------------------------------------------
EPISODE_COUNT = 15      # 14 training rounds + 1 operational round (paper)
NOISE_OFF_FRACTION = 0.8  # noise turned off after 80% of episodes

# ---------------------------------------------------------------------------
# Static topology experiment (Fig. 2, Fig. 3)
# ---------------------------------------------------------------------------
# Paper counts:  5, 10, 15, 20, 25, 30, 35, 40, 45, 50
# nWifi=50 excluded on ns-3.40: produces anomalous throughput due to a
# PHY simulation artifact that persists despite the stale-event fix.
STATION_COUNTS = [5, 10, 15, 20, 25, 30, 35, 40, 45]

# Extended counts for project (beyond paper scope)
EXTENDED_STATION_COUNTS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 80, 100]

# ---------------------------------------------------------------------------
# Look-up table CW values (from CW_data.csv / paper description)
# CW = 2^x - 1 where x is chosen per station count.
# These are determined by simulation to give best performance.
# ---------------------------------------------------------------------------
LOOKUP_TABLE = {
    5:    31,     # 2^5 - 1
    10:   63,     # 2^6 - 1
    15:   63,     # 2^6 - 1
    20:  127,     # 2^7 - 1
    25:  127,     # 2^7 - 1
    30:  255,     # 2^8 - 1
    35:  255,     # 2^8 - 1
    40:  511,     # 2^9 - 1
    45:  511,     # 2^9 - 1
    50: 1023,     # 2^10 - 1
    60: 1023,     # 2^10 - 1  (extended)
    80: 1023,     # 2^10 - 1  (extended)
    100: 1023,    # 2^10 - 1  (extended)
}

# ---------------------------------------------------------------------------
# Quick mode (for testing the pipeline before full runs)
# ---------------------------------------------------------------------------
QUICK_MODE = False
QUICK_SIM_TIME = 10
QUICK_EPISODE_COUNT = 3
QUICK_STATION_COUNTS = [5, 15, 30, 45]
