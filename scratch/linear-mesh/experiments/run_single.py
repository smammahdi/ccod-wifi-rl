"""
Run a single experiment for one station count.

This is a standalone script designed to be called in parallel from run_all.py.
Each invocation spawns one ns-3 simulation and produces one result file.

Usage:
    python -m experiments.run_single beb 5 8001 60 results/beb_5.json
    python -m experiments.run_single lookup 15 8002 60 results/lookup_15.json
    python -m experiments.run_single ddpg 30 8003 60 15 results/ddpg_30.json
"""

import sys
import os
import json
import time
import subprocess
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ns3gym import ns3env
from config import (
    NS3_DIR, STEP_TIME, HISTORY_LENGTH,
    LR_ACTOR, LR_CRITIC, BATCH_SIZE, GAMMA, TAU,
    REPLAY_BUFFER_MULTIPLIER, UPDATE_EVERY,
    ACTOR_LAYERS, CRITIC_LAYERS, NOISE_OFF_FRACTION,
    LOOKUP_TABLE,
)

BINARY = os.path.join(NS3_DIR, "build", "scratch", "linear-mesh",
                      "ns3.40-cw-default")


def launch_simulation(n_wifi, sim_time, port, scenario="basic",
                      agent_type="continuous", dry_run=False):
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
    return subprocess.Popen(cmd, cwd=NS3_DIR,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def connect_env(port, retries=3):
    time.sleep(2)
    for attempt in range(retries):
        try:
            return ns3env.Ns3Env(
                port=port, stepTime=STEP_TIME, startSim=0,
                simSeed=0, simArgs={}, debug=False,
            )
        except Exception:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                raise


def parse_info(info_str):
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


def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    def default_serializer(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=default_serializer)


# -------------------------------------------------------------------
# BEB: single station count
# -------------------------------------------------------------------
def run_beb(n_wifi, sim_time, port):
    proc = None
    env = None
    data = []
    try:
        proc = launch_simulation(n_wifi, sim_time, port, dry_run=True)
        env = connect_env(port)
        obs = env.reset()
        steps = int(sim_time / STEP_TIME) + HISTORY_LENGTH
        for step in range(steps):
            obs, reward, done, info = env.step(np.array([3.0]))
            if step > HISTORY_LENGTH:
                parsed = parse_info(info)
                if parsed:
                    data.append(parsed)
            if done:
                break
    except Exception as e:
        print(f"  [BEB n={n_wifi}] ERROR: {e}", flush=True)
    finally:
        cleanup(env, proc)

    if data:
        thr = float(np.mean([d["throughput_mbps"] for d in data]))
        fair = float(np.mean([d["fairness"] for d in data]))
    else:
        thr, fair = 0.0, 0.0
    return {"throughput": thr, "fairness": fair, "samples": len(data)}


# -------------------------------------------------------------------
# Lookup: single station count
# -------------------------------------------------------------------
def run_lookup(n_wifi, sim_time, port):
    target_cw = LOOKUP_TABLE.get(n_wifi, 63)
    action_val = float(np.clip(np.log2(target_cw + 1) - 4, 0, 6))

    proc = None
    env = None
    data = []
    try:
        proc = launch_simulation(n_wifi, sim_time, port, dry_run=False)
        env = connect_env(port)
        obs = env.reset()
        steps = int(sim_time / STEP_TIME) + HISTORY_LENGTH
        for step in range(steps):
            obs, reward, done, info = env.step(np.array([action_val]))
            if step > HISTORY_LENGTH:
                parsed = parse_info(info)
                if parsed:
                    data.append(parsed)
            if done:
                break
    except Exception as e:
        print(f"  [Lookup n={n_wifi}] ERROR: {e}", flush=True)
    finally:
        cleanup(env, proc)

    if data:
        thr = float(np.mean([d["throughput_mbps"] for d in data]))
        fair = float(np.mean([d["fairness"] for d in data]))
    else:
        thr, fair = 0.0, 0.0
    return {"throughput": thr, "fairness": fair, "cw": target_cw,
            "samples": len(data)}


# -------------------------------------------------------------------
# DDPG: single station count, full training
# -------------------------------------------------------------------
def run_ddpg(n_wifi, sim_time, port_base, episode_count):
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

    round_throughputs = []
    round_cws = []
    round_fairnesses = []

    for episode in range(episode_count):
        add_noise = episode < int(episode_count * NOISE_OFF_FRACTION)
        port = port_base + episode

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
            print(f"  [DDPG n={n_wifi} ep={episode+1}] ERROR: {e}", flush=True)
        finally:
            cleanup(env, proc)
            agent.reset()

        mean_thr = float(np.mean(ep_throughputs)) if ep_throughputs else 0
        mean_cw = float(np.mean(ep_cws)) if ep_cws else 0
        mean_fair = float(np.mean(ep_fairnesses)) if ep_fairnesses else 0
        round_throughputs.append(mean_thr)
        round_cws.append(mean_cw)
        round_fairnesses.append(mean_fair)

        noise_str = "NOISE" if add_noise else "NO_NOISE"
        print(f"  [DDPG n={n_wifi}] Round {episode+1}/{episode_count}: "
              f"thr={mean_thr:.2f}, CW={mean_cw:.0f}, fair={mean_fair:.4f} "
              f"[{noise_str}]", flush=True)

    return {
        "final_throughput": round_throughputs[-1] if round_throughputs else 0,
        "thr_history": round_throughputs,
        "cw_history": round_cws,
        "fair_history": round_fairnesses,
    }


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    if len(sys.argv) < 5:
        print("Usage:")
        print("  python -m experiments.run_single beb <n_wifi> <port> <sim_time> <output>")
        print("  python -m experiments.run_single lookup <n_wifi> <port> <sim_time> <output>")
        print("  python -m experiments.run_single ddpg <n_wifi> <port_base> <sim_time> <episodes> <output>")
        sys.exit(1)

    mode = sys.argv[1]
    n_wifi = int(sys.argv[2])
    port = int(sys.argv[3])
    sim_time = float(sys.argv[4])

    if mode == "beb":
        output = sys.argv[5]
        result = run_beb(n_wifi, sim_time, port)
        print(f"  [BEB n={n_wifi}] thr={result['throughput']:.2f} Mbps, "
              f"fair={result['fairness']:.4f} ({result['samples']} samples)", flush=True)
        save_json(result, output)

    elif mode == "lookup":
        output = sys.argv[5]
        result = run_lookup(n_wifi, sim_time, port)
        print(f"  [Lookup n={n_wifi}] thr={result['throughput']:.2f} Mbps, "
              f"fair={result['fairness']:.4f} CW={result['cw']}", flush=True)
        save_json(result, output)

    elif mode == "ddpg":
        episodes = int(sys.argv[5])
        output = sys.argv[6]
        result = run_ddpg(n_wifi, sim_time, port, episodes)
        print(f"  [DDPG n={n_wifi}] final_thr={result['final_throughput']:.2f} Mbps",
              flush=True)
        save_json(result, output)

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
