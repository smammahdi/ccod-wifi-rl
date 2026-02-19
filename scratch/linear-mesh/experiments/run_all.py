"""
Comprehensive experiment runner for CCOD paper reproduction.

Reproduces all experiments from:
  "Contention Window Optimization in IEEE 802.11ax Networks with Deep
   Reinforcement Learning" (Wydmanski & Szott, WCNC 2021)

Experiments:
  1. BEB baseline   -- standard 802.11 (dryRun mode) for various station counts
  2. Lookup table   -- fixed optimal CW per station count
  3. DDPG training  -- static topology with DDPG agent
  4. Convergence    -- dynamic topology where stations increase from 5 to max

Usage:
    cd scratch/linear-mesh
    python -u -m experiments.run_all                # Full paper experiments
    python -u -m experiments.run_all --quick        # Quick test run
    python -u -m experiments.run_all --beb-only     # Only BEB + Lookup baselines
    python -u -m experiments.run_all --ddpg-only    # Only DDPG training
    python -u -m experiments.run_all --conv-only    # Only convergence scenario
    python -u -m experiments.run_all --extended     # Extended (up to 100 stations)
"""

import sys
import os
import time
import json
import subprocess
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ns3gym import ns3env
from config import (
    NS3_DIR, RESULTS_DIR, LOGS_DIR,
    SIM_TIME, STEP_TIME, HISTORY_LENGTH,
    LR_ACTOR, LR_CRITIC, BATCH_SIZE, GAMMA, TAU,
    REPLAY_BUFFER_MULTIPLIER, UPDATE_EVERY,
    ACTOR_LAYERS, CRITIC_LAYERS,
    EPISODE_COUNT, NOISE_OFF_FRACTION,
    STATION_COUNTS, EXTENDED_STATION_COUNTS, LOOKUP_TABLE,
    QUICK_MODE, QUICK_SIM_TIME, QUICK_EPISODE_COUNT, QUICK_STATION_COUNTS,
)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
# Binary name depends on ns-3 version; ns-3.40 builds to this path
BINARY = os.path.join(NS3_DIR, "build", "scratch", "linear-mesh",
                      "ns3.40-cw-default")
_PORT_COUNTER = 8000   # Start high to avoid conflicts


def _next_port():
    """Get a unique port to avoid collisions between runs."""
    global _PORT_COUNTER
    _PORT_COUNTER += 1
    return _PORT_COUNTER


def _print(msg):
    """Print with flush for real-time output."""
    print(msg, flush=True)


# ===================================================================
# Utility: launch ns-3 + connect ns3gym
# ===================================================================

def launch_simulation(n_wifi, sim_time, port, scenario="basic",
                      agent_type="continuous", dry_run=False, seed=-1):
    """Launch ns-3 binary and return the subprocess handle."""
    cmd = [
        BINARY,
        f"--simTime={sim_time}",
        f"--nWifi={n_wifi}",
        f"--envStepTime={STEP_TIME}",
        f"--historyLength={HISTORY_LENGTH}",
        f"--scenario={scenario}",
        f"--agentType={agent_type}",
        f"--openGymPort={port}",
        f"--dryRun={'1' if dry_run else '0'}",
    ]
    if seed >= 0:
        cmd.append(f"--seed={seed}")

    proc = subprocess.Popen(
        cmd, cwd=NS3_DIR,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return proc


def connect_env(port, retries=3):
    """Connect to the ns-3 simulation with retry logic."""
    time.sleep(2)  # Give ns-3 time to start and bind
    for attempt in range(retries):
        try:
            env = ns3env.Ns3Env(
                port=port, stepTime=STEP_TIME, startSim=0,
                simSeed=0, simArgs={}, debug=False,
            )
            return env
        except Exception as e:
            if attempt < retries - 1:
                _print(f"    Connection attempt {attempt+1} failed, retrying...")
                time.sleep(2)
            else:
                raise


def parse_info(info_str):
    """Parse the info string: 'sentMb|CW|stationCount|fairness'."""
    try:
        parts = str(info_str).split("|")
        return {
            "throughput_mbps": float(parts[0]) / STEP_TIME,
            "cw": float(parts[1]),
            "stations": float(parts[2]),
            "fairness": float(parts[3]) if parts[3] != "inf" else 1.0,
        }
    except (ValueError, IndexError):
        return None


def cleanup(env, proc):
    """Safely close environment and kill process."""
    try:
        if env is not None:
            env.close()
    except Exception:
        pass
    try:
        if proc is not None:
            proc.kill()
            proc.wait(timeout=10)
    except Exception:
        pass
    time.sleep(1)


def run_single_episode(n_wifi, sim_time, port, action_fn, dry_run=False,
                       scenario="basic", collect_after=None):
    """Generic single-episode runner.

    action_fn: callable(obs, step) -> action array
    collect_after: skip first N steps for data collection (default: HISTORY_LENGTH)

    Returns: list of parsed info dicts
    """
    if collect_after is None:
        collect_after = HISTORY_LENGTH

    proc = None
    env = None
    data = []

    try:
        proc = launch_simulation(n_wifi, sim_time, port, scenario=scenario,
                                 dry_run=dry_run)
        env = connect_env(port)
        obs = env.reset()

        steps = int(sim_time / STEP_TIME) + HISTORY_LENGTH
        for step in range(steps):
            action = action_fn(obs, step)
            obs, reward, done, info = env.step(action)

            if step > collect_after:
                parsed = parse_info(info)
                if parsed:
                    parsed["step"] = step
                    parsed["time"] = (step - collect_after) * STEP_TIME
                    data.append(parsed)

            if done:
                break

    except Exception as e:
        _print(f"    ERROR: {e}")
        traceback.print_exc()
    finally:
        cleanup(env, proc)

    return data


def summarize_episode(data):
    """Compute summary statistics from episode data."""
    if not data:
        return {"throughput_mbps": 0, "fairness": 0, "mean_cw": 0}
    throughputs = [d["throughput_mbps"] for d in data]
    fairnesses = [d["fairness"] for d in data]
    cws = [d["cw"] for d in data]
    return {
        "throughput_mbps": float(np.mean(throughputs)),
        "fairness": float(np.mean(fairnesses)),
        "mean_cw": float(np.mean(cws)),
    }


# ===================================================================
# Experiment 1: BEB Baseline
# ===================================================================

def run_beb_experiment(station_counts, sim_time):
    """Run BEB baseline for each station count.

    Returns:
        throughput_results: {n: throughput_mbps}
        fairness_results: {n: jain_fairness_index}
    """
    _print("\n" + "=" * 70)
    _print("EXPERIMENT: BEB Baseline (Standard 802.11)")
    _print("=" * 70)

    throughput_results = {}
    fairness_results = {}

    for n_wifi in station_counts:
        port = _next_port()
        _print(f"\n  nWifi={n_wifi}, port={port} ...")

        def action_fn(obs, step):
            return np.array([3.0])  # Dummy action, ignored in dryRun mode

        data = run_single_episode(n_wifi, sim_time, port, action_fn, dry_run=True)
        summary = summarize_episode(data)

        throughput_results[n_wifi] = summary["throughput_mbps"]
        fairness_results[n_wifi] = summary["fairness"]
        _print(f"  -> Throughput: {summary['throughput_mbps']:.2f} Mbps, "
               f"Fairness: {summary['fairness']:.4f} ({len(data)} samples)")

    return throughput_results, fairness_results


# ===================================================================
# Experiment 2: Lookup Table Baseline
# ===================================================================

def run_lookup_experiment(station_counts, sim_time):
    """Run lookup table baseline: fixed optimal CW per station count.

    Returns:
        throughput_results: {n: throughput_mbps}
        fairness_results: {n: jain_fairness_index}
    """
    _print("\n" + "=" * 70)
    _print("EXPERIMENT: Lookup Table Baseline")
    _print("=" * 70)

    throughput_results = {}
    fairness_results = {}

    for n_wifi in station_counts:
        target_cw = LOOKUP_TABLE.get(n_wifi, 63)
        # CW = 2^(a+4) - 1 => a = log2(CW+1) - 4
        action_val = np.log2(target_cw + 1) - 4
        action_val = float(np.clip(action_val, 0, 6))

        port = _next_port()
        _print(f"\n  nWifi={n_wifi}, CW={target_cw}, action={action_val:.2f}, port={port} ...")

        def action_fn(obs, step, a=action_val):
            return np.array([a])

        data = run_single_episode(n_wifi, sim_time, port, action_fn, dry_run=False)
        summary = summarize_episode(data)

        throughput_results[n_wifi] = summary["throughput_mbps"]
        fairness_results[n_wifi] = summary["fairness"]
        _print(f"  -> Throughput: {summary['throughput_mbps']:.2f} Mbps, "
               f"Fairness: {summary['fairness']:.4f}")

    return throughput_results, fairness_results


# ===================================================================
# Experiment 3: DDPG Training (Static Topology)
# ===================================================================

def run_ddpg_static(station_counts, sim_time, episode_count):
    """Train DDPG agent on static topology for each station count.

    Returns:
        results: {n: final_throughput}
        cw_history: {n: [mean_cw_per_round]}
        thr_history: {n: [mean_thr_per_round]}
        fair_history: {n: [mean_fairness_per_round]}
    """
    _print("\n" + "=" * 70)
    _print("EXPERIMENT: DDPG Training (Static Topology)")
    _print("=" * 70)

    from agents.ddpg.agent import Agent, Config
    from preprocessor import Preprocessor

    steps_per_ep = int(sim_time / STEP_TIME) + HISTORY_LENGTH
    preprocess = Preprocessor(False).preprocess

    results = {}
    cw_history = {}
    thr_history = {}
    fair_history = {}

    for n_wifi in station_counts:
        _print(f"\n  --- Training DDPG for {n_wifi} stations ---")

        config = Config(
            buffer_size=REPLAY_BUFFER_MULTIPLIER * steps_per_ep,
            batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU,
            lr_actor=LR_ACTOR, lr_critic=LR_CRITIC,
            update_every=UPDATE_EVERY,
        )
        agent = Agent(
            HISTORY_LENGTH, action_size=1, config=config,
            actor_layers=ACTOR_LAYERS, critic_layers=CRITIC_LAYERS,
        )

        round_throughputs = []
        round_cws = []
        round_fairnesses = []

        for episode in range(episode_count):
            add_noise = episode < int(episode_count * NOISE_OFF_FRACTION)
            port = _next_port()

            proc = None
            env = None
            ep_throughputs = []
            ep_cws = []
            ep_fairnesses = []

            try:
                proc = launch_simulation(n_wifi, sim_time, port, dry_run=False)
                env = connect_env(port)
                obs = env.reset()
                obs = preprocess(np.reshape(obs, (-1, 1, 1)))

                last_obs = None
                for step in range(steps_per_ep):
                    actions = agent.act(np.array(obs, dtype=np.float32), add_noise)
                    # Flatten for env.step (ns3gym expects 1D array)
                    action_flat = np.array([float(actions.flat[0])])
                    next_obs, reward, done, info = env.step(action_flat)
                    next_obs = preprocess(np.reshape(next_obs, (-1, 1, 1)))

                    # Train (except on last episode = operational phase)
                    if (last_obs is not None and step > HISTORY_LENGTH
                            and episode < episode_count - 1):
                        agent.step(obs, actions, np.array([reward]),
                                   next_obs, np.array([done]), 2)

                    if step > HISTORY_LENGTH:
                        parsed = parse_info(info)
                        if parsed:
                            ep_throughputs.append(parsed["throughput_mbps"])
                            ep_cws.append(parsed["cw"])
                            ep_fairnesses.append(parsed["fairness"])

                    last_obs = obs
                    obs = next_obs

                    if done:
                        break

            except Exception as e:
                _print(f"    ERROR in round {episode+1}: {e}")
            finally:
                cleanup(env, proc)
                agent.reset()

            mean_thr = np.mean(ep_throughputs) if ep_throughputs else 0
            mean_cw = np.mean(ep_cws) if ep_cws else 0
            mean_fair = np.mean(ep_fairnesses) if ep_fairnesses else 0
            round_throughputs.append(float(mean_thr))
            round_cws.append(float(mean_cw))
            round_fairnesses.append(float(mean_fair))

            noise_str = "NOISE" if add_noise else "NO_NOISE"
            _print(f"    Round {episode+1}/{episode_count}: "
                   f"thr={mean_thr:.2f} Mbps, CW={mean_cw:.0f}, "
                   f"fair={mean_fair:.4f} [{noise_str}]")

        results[n_wifi] = round_throughputs[-1] if round_throughputs else 0
        cw_history[n_wifi] = round_cws
        thr_history[n_wifi] = round_throughputs
        fair_history[n_wifi] = round_fairnesses

    return results, cw_history, thr_history, fair_history


# ===================================================================
# Experiment 4: Dynamic Topology (Convergence)
# ===================================================================

def run_ddpg_convergence(sim_time, episode_count, n_wifi=None):
    """Train DDPG on convergence scenario (stations increase 5->max)."""
    if n_wifi is None:
        n_wifi = max(STATION_COUNTS)

    _print("\n" + "=" * 70)
    _print(f"EXPERIMENT: DDPG Convergence (Dynamic Topology, nWifi={n_wifi})")
    _print("=" * 70)

    from agents.ddpg.agent import Agent, Config
    from preprocessor import Preprocessor

    steps_per_ep = int(sim_time / STEP_TIME) + HISTORY_LENGTH
    preprocess = Preprocessor(False).preprocess

    config = Config(
        buffer_size=REPLAY_BUFFER_MULTIPLIER * steps_per_ep,
        batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU,
        lr_actor=LR_ACTOR, lr_critic=LR_CRITIC,
        update_every=UPDATE_EVERY,
    )
    agent = Agent(
        HISTORY_LENGTH, action_size=1, config=config,
        actor_layers=ACTOR_LAYERS, critic_layers=CRITIC_LAYERS,
    )

    all_timestep_data = []
    round_throughputs = []

    for episode in range(episode_count):
        add_noise = episode < int(episode_count * NOISE_OFF_FRACTION)
        port = _next_port()

        proc = None
        env = None
        ep_data = []

        try:
            proc = launch_simulation(n_wifi, sim_time, port,
                                     scenario="convergence", dry_run=False)
            env = connect_env(port)
            obs = env.reset()
            obs = preprocess(np.reshape(obs, (-1, 1, 1)))

            last_obs = None
            for step in range(steps_per_ep):
                actions = agent.act(np.array(obs, dtype=np.float32), add_noise)
                action_flat = np.array([float(actions.flat[0])])
                next_obs, reward, done, info = env.step(action_flat)
                next_obs = preprocess(np.reshape(next_obs, (-1, 1, 1)))

                if (last_obs is not None and step > HISTORY_LENGTH
                        and episode < episode_count - 1):
                    agent.step(obs, actions, np.array([reward]),
                               next_obs, np.array([done]), 2)

                if step > HISTORY_LENGTH:
                    parsed = parse_info(info)
                    if parsed:
                        parsed["time"] = (step - HISTORY_LENGTH) * STEP_TIME
                        ep_data.append(parsed)

                last_obs = obs
                obs = next_obs

                if done:
                    break

        except Exception as e:
            _print(f"    ERROR in round {episode+1}: {e}")
        finally:
            cleanup(env, proc)
            agent.reset()

        ep_thr = np.mean([d["throughput_mbps"] for d in ep_data]) if ep_data else 0
        round_throughputs.append(float(ep_thr))
        _print(f"  Round {episode+1}/{episode_count}: thr={ep_thr:.2f} Mbps")

        if episode == episode_count - 1:
            all_timestep_data = ep_data

    return all_timestep_data, round_throughputs


def run_beb_convergence(sim_time, n_wifi=None):
    """BEB baseline for convergence scenario."""
    if n_wifi is None:
        n_wifi = max(STATION_COUNTS)

    _print(f"\n  Running BEB convergence baseline (nWifi={n_wifi}) ...")
    port = _next_port()

    def action_fn(obs, step):
        return np.array([3.0])

    return run_single_episode(n_wifi, sim_time, port,
                              action_fn, dry_run=True,
                              scenario="convergence")


# ===================================================================
# Save results
# ===================================================================

def save_results(results_dict, filename):
    """Save experiment results to JSON file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "w") as f:
        def default_serializer(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)
        json.dump(results_dict, f, indent=2, default=default_serializer)
    _print(f"  Saved: {filepath}")


# ===================================================================
# Main
# ===================================================================

def main():
    quick = "--quick" in sys.argv or QUICK_MODE
    beb_only = "--beb-only" in sys.argv
    ddpg_only = "--ddpg-only" in sys.argv
    conv_only = "--conv-only" in sys.argv
    extended = "--extended" in sys.argv

    if quick:
        station_counts = QUICK_STATION_COUNTS
        sim_time = QUICK_SIM_TIME
        episode_count = QUICK_EPISODE_COUNT
        _print("*** QUICK MODE: reduced parameters ***")
    elif extended:
        station_counts = EXTENDED_STATION_COUNTS
        sim_time = SIM_TIME
        episode_count = EPISODE_COUNT
        _print("*** EXTENDED MODE: up to 100 stations ***")
    else:
        station_counts = STATION_COUNTS
        sim_time = SIM_TIME
        episode_count = EPISODE_COUNT

    _print(f"Station counts: {station_counts}")
    _print(f"Simulation time: {sim_time}s, Episodes: {episode_count}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Determine prefix for result files (to separate paper vs extended)
    prefix = "extended_" if extended else "static_"
    run_all = not (beb_only or ddpg_only or conv_only)

    # ------ BEB Baseline ------
    if run_all or beb_only:
        beb_thr, beb_fair = run_beb_experiment(station_counts, sim_time)
        save_results(beb_thr, f"{prefix}beb.json")
        save_results(beb_fair, f"{prefix}beb_fairness.json")

    # ------ Lookup Table Baseline ------
    if run_all or beb_only:
        lookup_thr, lookup_fair = run_lookup_experiment(station_counts, sim_time)
        save_results(lookup_thr, f"{prefix}lookup.json")
        save_results(lookup_fair, f"{prefix}lookup_fairness.json")

    # ------ DDPG Static ------
    if run_all or ddpg_only:
        ddpg_results, cw_hist, thr_hist, fair_hist = run_ddpg_static(
            station_counts, sim_time, episode_count)
        save_results(ddpg_results, f"{prefix}ddpg.json")
        save_results(cw_hist, f"{prefix}ddpg_cw_history.json")
        save_results(thr_hist, f"{prefix}ddpg_thr_history.json")
        save_results(fair_hist, f"{prefix}ddpg_fair_history.json")

    # ------ Convergence ------
    if run_all or conv_only:
        n_conv = max(station_counts)
        beb_conv = run_beb_convergence(sim_time, n_wifi=n_conv)
        save_results({"timesteps": beb_conv}, "convergence_beb.json")

        ddpg_conv_data, ddpg_conv_thr = run_ddpg_convergence(
            sim_time, episode_count, n_wifi=n_conv)
        save_results({
            "timesteps": ddpg_conv_data,
            "round_throughputs": ddpg_conv_thr,
        }, "convergence_ddpg.json")

    _print("\n" + "=" * 70)
    _print("ALL EXPERIMENTS COMPLETE")
    _print(f"Results saved in: {RESULTS_DIR}")
    _print("=" * 70)


if __name__ == "__main__":
    main()
