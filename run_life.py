"""One life: a childhood in a scarce or rich world (or no childhood), then the adult world.

    python run_life.py --childhood scarce --seed 0
    python run_life.py --childhood rich --seed 0
    python run_life.py --childhood none --seed 0

With the same seed, the three conditions start from the same untrained network
and later move into the same adult world, so they can be compared seed by seed.
The move is not announced: the agent simply finds itself somewhere else.

Phases written to the CSV:
  child   learning in the childhood world
  frozen  the agent right after childhood, acting in the adult world without learning
  adult   learning goes on in the adult world
"""
import argparse
import copy
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from agent import Agent
from world import ADULT, N_ACTIONS, OBS_DIM, RICH, SCARCE, Worlds

CHILDHOODS = {"scarce": SCARCE, "rich": RICH, "none": None}
COLUMNS = ["phase", "step", "wellbeing", "collapses_per_1k", "starved_per_1k", "thirst_per_1k", "froze_per_1k", "energy", "water", "temp",
           "carry", "carry_sated", "carry_sated_normal", "carry_sated_drought",
           "cold_day", "cold_day_normal", "cold_day_drought", "cold_night", "drought_share", "drinks_per_100", "meals_per_100"]


def mean_where(values, mask):
    return float(values[mask].mean()) if mask.any() else float("nan")


def summarise(infos, rewards):
    get = lambda k: np.stack([i[k] for i in infos])
    carry, sated, cold = get("carry"), get("sated"), get("on_cold")
    night, drought = get("night"), get("drought")
    day = ~night
    return {
        "wellbeing": float(rewards.mean()),
        "collapses_per_1k": float(get("collapse").mean() * 1000),
        "starved_per_1k": float(get("starved").mean() * 1000),
        "thirst_per_1k": float(get("thirst").mean() * 1000),
        "froze_per_1k": float(get("froze").mean() * 1000),
        "energy": float(get("energy").mean()),
        "water": float(get("water").mean()),
        "temp": float(get("temp").mean()),
        "carry": float(carry.mean()),
        "carry_sated": mean_where(carry, sated),
        "carry_sated_normal": mean_where(carry, sated & ~drought),
        "carry_sated_drought": mean_where(carry, sated & drought),
        "cold_day": mean_where(cold, day),
        "cold_day_normal": mean_where(cold, day & ~drought),
        "cold_day_drought": mean_where(cold, day & drought),
        "cold_night": mean_where(cold, night),
        "drought_share": float(drought.mean()),
        "drinks_per_100": float(get("drank").mean() * 100),
        "meals_per_100": float(get("ate").mean() * 100),
    }


def live(agent, worlds, total_steps, phase, write, learn=True, rollout=128):
    """Run the agent in `worlds` for `total_steps` agent-steps, one CSV row per rollout."""
    obs = worlds.observe()
    done = 0
    shape = (rollout, worlds.n)
    while done < total_steps:
        b_obs = np.zeros(shape + (OBS_DIM,), np.float32)
        b_act, b_logp, b_val, b_rew, b_done = (np.zeros(shape, np.float32) for _ in range(5))
        infos = []
        for t in range(rollout):
            a, logp, v = agent.act(obs)
            b_obs[t], b_act[t], b_logp[t], b_val[t] = obs, a, logp, v
            obs, r, collapsed, info = worlds.step(a)
            b_rew[t], b_done[t] = r, collapsed
            infos.append(info)
        if learn:
            agent.learn(b_obs, b_act, b_logp, b_val, b_rew, b_done, agent.value(obs))
        done += rollout * worlds.n
        write({"phase": phase, "step": done, **summarise(infos, b_rew)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--childhood", choices=CHILDHOODS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--child-steps", type=int, default=3_000_000)
    p.add_argument("--frozen-steps", type=int, default=100_000)
    p.add_argument("--adult-steps", type=int, default=3_000_000)
    p.add_argument("--worlds", type=int, default=32)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--out", default="runs")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{args.childhood}_s{args.seed}"
    child_cfg = CHILDHOODS[args.childhood]
    meta = {"args": vars(args), "childhood": asdict(child_cfg) if child_cfg else None, "adult": asdict(ADULT)}
    (out / f"{name}.json").write_text(json.dumps(meta, indent=2))

    agent = Agent(OBS_DIM, N_ACTIONS, lr=args.lr, gamma=args.gamma)
    started = time.time()
    with open(out / f"{name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        write = lambda row: (writer.writerow(row), f.flush())

        if child_cfg is not None:
            child_world = Worlds(child_cfg, args.worlds, np.random.default_rng([args.seed, 0]))
            live(agent, child_world, args.child_steps, "child", write)
            print(f"{name}: childhood done in {time.time() - started:.0f}s", flush=True)
        torch.save(agent.net.state_dict(), out / f"{name}_after_childhood.pt")

        # the same agent, unchanged, in a fresh set of adult worlds
        frozen = copy.deepcopy(agent)
        test_world = Worlds(ADULT, args.worlds, np.random.default_rng([args.seed, 2]))
        live(frozen, test_world, args.frozen_steps, "frozen", write, learn=False)

        adult_world = Worlds(ADULT, args.worlds, np.random.default_rng([args.seed, 1]))
        live(agent, adult_world, args.adult_steps, "adult", write)
        torch.save(agent.net.state_dict(), out / f"{name}_end.pt")
    print(f"{name}: done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
