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
# Parallel launcher: run run_single.py as separate OS processes
# ===================================================================

def _run_parallel_singles(mode, station_counts, sim_time, max_workers,
                          episode_count=None):
    """Launch multiple run_single.py processes in parallel.

    Returns dict of {n_wifi: parsed_json_result}.
    """
    python = sys.executable
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp_dir = os.path.join(RESULTS_DIR, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Prepare jobs: (n_wifi, port, output_file, cmd)
    jobs = []
    for n_wifi in station_counts:
        port = _next_port()
        out_file = os.path.join(tmp_dir, f"{mode}_{n_wifi}.json")

        if mode == "ddpg":
            # DDPG needs port_base with enough room for episodes
            # Reserve ports: port_base .. port_base + episode_count
            for _ in range(episode_count - 1):
                _next_port()
            cmd = [python, "-u", "-m", "experiments.run_single",
                   mode, str(n_wifi), str(port), str(sim_time),
                   str(episode_count), out_file]
        else:
            cmd = [python, "-u", "-m", "experiments.run_single",
                   mode, str(n_wifi), str(port), str(sim_time), out_file]

        jobs.append((n_wifi, cmd, out_file))

    # Launch up to max_workers at a time
    results = {}
    active = []  # (n_wifi, proc, out_file)
    pending = list(jobs)

    while pending or active:
        # Fill up worker slots
        while pending and len(active) < max_workers:
            n_wifi, cmd, out_file = pending.pop(0)
            proc = subprocess.Popen(
                cmd, cwd=base_dir,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            active.append((n_wifi, proc, out_file))

        # Check for completed processes
        still_active = []
        for n_wifi, proc, out_file in active:
            ret = proc.poll()
            if ret is not None:
                # Process finished
                stdout = proc.stdout.read().decode("utf-8", errors="replace")
                if stdout.strip():
                    _print(stdout.strip())
                if ret == 0 and os.path.exists(out_file):
                    with open(out_file) as f:
                        results[n_wifi] = json.load(f)
                else:
                    _print(f"  WARNING: {mode} n={n_wifi} exited with code {ret}")
                    results[n_wifi] = None
            else:
                still_active.append((n_wifi, proc, out_file))
        active = still_active

        if active:
            time.sleep(1)

    # Cleanup tmp files
    for n_wifi, cmd, out_file in jobs:
        if os.path.exists(out_file):
            os.unlink(out_file)
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    return results


# ===================================================================
# Experiment 1: BEB Baseline
# ===================================================================

def _run_beb_single(n_wifi, sim_time, port):
    """Run BEB for a single station count. Designed to be called in parallel."""
    def action_fn(obs, step):
        return np.array([3.0])
    data = run_single_episode(n_wifi, sim_time, port, action_fn, dry_run=True)
    summary = summarize_episode(data)
    return n_wifi, summary, len(data)


def run_beb_experiment(station_counts, sim_time, max_workers=1):
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

    if max_workers > 1:
        _print(f"  Launching {len(station_counts)} parallel BEB simulations "
               f"(max {max_workers} at once)")
        results = _run_parallel_singles("beb", station_counts, sim_time, max_workers)
        for n_wifi, res in sorted(results.items()):
            if res:
                throughput_results[n_wifi] = res["throughput"]
                fairness_results[n_wifi] = res["fairness"]
    else:
        for n_wifi in station_counts:
            port = _next_port()
            _print(f"\n  nWifi={n_wifi}, port={port} ...")
            n_wifi, summary, n_samples = _run_beb_single(n_wifi, sim_time, port)
            throughput_results[n_wifi] = summary["throughput_mbps"]
            fairness_results[n_wifi] = summary["fairness"]
            _print(f"  -> Throughput: {summary['throughput_mbps']:.2f} Mbps, "
                   f"Fairness: {summary['fairness']:.4f} ({n_samples} samples)")

    return throughput_results, fairness_results


# ===================================================================
# Experiment 2: Lookup Table Baseline
# ===================================================================

def _run_lookup_single(n_wifi, sim_time, port, action_val, target_cw):
    """Run Lookup for a single station count. Designed for parallel use."""
    def action_fn(obs, step, a=action_val):
        return np.array([a])
    data = run_single_episode(n_wifi, sim_time, port, action_fn, dry_run=False)
    summary = summarize_episode(data)
    return n_wifi, summary, target_cw, action_val


def run_lookup_experiment(station_counts, sim_time, max_workers=1):
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

    if max_workers > 1:
        _print(f"  Launching {len(station_counts)} parallel Lookup simulations "
               f"(max {max_workers} at once)")
        results = _run_parallel_singles("lookup", station_counts, sim_time, max_workers)
        for n_wifi, res in sorted(results.items()):
            if res:
                throughput_results[n_wifi] = res["throughput"]
                fairness_results[n_wifi] = res["fairness"]
    else:
        for n_wifi in station_counts:
            target_cw = LOOKUP_TABLE.get(n_wifi, 63)
            action_val = np.log2(target_cw + 1) - 4
            action_val = float(np.clip(action_val, 0, 6))
            port = _next_port()
            _print(f"\n  nWifi={n_wifi}, CW={target_cw}, action={action_val:.2f}, port={port} ...")
            n_wifi, summary, _, _ = _run_lookup_single(n_wifi, sim_time, port, action_val, target_cw)
            throughput_results[n_wifi] = summary["throughput_mbps"]
            fairness_results[n_wifi] = summary["fairness"]
            _print(f"  -> Throughput: {summary['throughput_mbps']:.2f} Mbps, "
                   f"Fairness: {summary['fairness']:.4f}")

    return throughput_results, fairness_results


# ===================================================================
# Experiment 3: DDPG Training (Static Topology)
# ===================================================================

def run_ddpg_static(station_counts, sim_time, episode_count, max_workers=1):
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

    results = {}
    cw_history = {}
    thr_history = {}
    fair_history = {}

    # Parallel: run different station counts as separate processes
    # Each process trains its own agent independently
    ddpg_workers = min(max_workers, len(station_counts)) if max_workers > 1 else 1
    if ddpg_workers > 4:
        # Limit DDPG parallelism to avoid GPU memory issues
        ddpg_workers = 4

    if ddpg_workers > 1:
        _print(f"  Launching {len(station_counts)} DDPG trainings in parallel "
               f"(max {ddpg_workers} at once)")
        par_results = _run_parallel_singles(
            "ddpg", station_counts, sim_time, ddpg_workers,
            episode_count=episode_count)
        for n_wifi, res in sorted(par_results.items()):
            if res:
                results[n_wifi] = res["final_throughput"]
                thr_history[n_wifi] = res["thr_history"]
                cw_history[n_wifi] = res["cw_history"]
                fair_history[n_wifi] = res["fair_history"]
    else:
        from agents.ddpg.agent import Agent, Config
        from preprocessor import Preprocessor
        steps_per_ep = int(sim_time / STEP_TIME) + HISTORY_LENGTH
        preprocess = Preprocessor(False).preprocess

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

    # Parallel workers: --parallel N (default 1 = sequential)
    max_workers = 1
    for arg in sys.argv:
        if arg.startswith("--parallel"):
            if "=" in arg:
                max_workers = int(arg.split("=")[1])
            else:
                idx = sys.argv.index(arg)
                if idx + 1 < len(sys.argv):
                    max_workers = int(sys.argv[idx + 1])
    if max_workers < 1:
        max_workers = 1

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
    if max_workers > 1:
        _print(f"Parallel workers: {max_workers}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Determine prefix for result files (to separate paper vs extended)
    prefix = "extended_" if extended else "static_"
    run_all = not (beb_only or ddpg_only or conv_only)

    # ------ BEB Baseline ------
    if run_all or beb_only:
        beb_thr, beb_fair = run_beb_experiment(station_counts, sim_time, max_workers)
        save_results(beb_thr, f"{prefix}beb.json")
        save_results(beb_fair, f"{prefix}beb_fairness.json")

    # ------ Lookup Table Baseline ------
    if run_all or beb_only:
        lookup_thr, lookup_fair = run_lookup_experiment(station_counts, sim_time, max_workers)
        save_results(lookup_thr, f"{prefix}lookup.json")
        save_results(lookup_fair, f"{prefix}lookup_fairness.json")

    # ------ DDPG Static ------
    if run_all or ddpg_only:
        ddpg_results, cw_hist, thr_hist, fair_hist = run_ddpg_static(
            station_counts, sim_time, episode_count, max_workers)
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
