"""One life: a childhood in a scarce or rich world (or no childhood), then the adult world.

    python run_life.py --childhood scarce --seed 0
    python run_life.py --childhood rich --seed 0
    python run_life.py --childhood none --seed 0

With the same seed, the three conditions start from the same untrained network
and later move into the same adult world, so they can be compared seed by seed.
The move is not announced: the agent simply finds itself somewhere else.

Phases written to the CSV:
  child        learning in the childhood world
  frozen_home  (version 1) the agent right after childhood, without learning, in its own kind of
               world but on maps it has never seen
  frozen       the agent right after childhood, acting in the adult world without learning
  adult        learning goes on in the adult world

Version 1 (--new-map-every N --branches ...): every world gets a fresh layout every N steps of its
clock, and one childhood is followed by one adult life in each listed adult world. Each branch has
its own CSV, <childhood>_s<seed>_<branch>.csv, which repeats the childhood rows at its head.
"""
import argparse
import copy
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from agent import Agent, Net
from world import ADULT, ADULT_POORER, ADULT_RICHER, N_ACTIONS, OBS_DIM, HUNGRY, RICH, SCARCE, UNRELIABLE, Worlds

CHILDHOODS = {"scarce": SCARCE, "rich": RICH, "adult": ADULT, "unreliable": UNRELIABLE, "hungry": HUNGRY, "none": None}
# version 1: adult worlds a life can branch into after one childhood; the number fixes each branch's
# random seed, so a branch does not depend on which other branches run in the same process
BRANCHES = {"standard": (0, ADULT), "richer": (1, ADULT_RICHER), "poorer": (2, ADULT_POORER)}
COLUMNS = ["phase", "step", "wellbeing", "collapses_per_1k", "starved_per_1k", "thirst_per_1k", "froze_per_1k",
           "collapses_per_1k_normal", "collapses_per_1k_drought", "energy", "water", "temp",
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
        "collapses_per_1k_normal": mean_where(get("collapse"), ~drought) * 1000,
        "collapses_per_1k_drought": mean_where(get("collapse"), drought) * 1000,
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
    p.add_argument("--scarce-regrow", type=float, help="food regrowth of the scarce childhood world (default as in world.py)")
    p.add_argument("--reset", choices=["no", "last"], default="no",
                   help="'last' re-initialises the second hidden layer and both heads at the move")
    p.add_argument("--new-map-every", type=int, default=0,
                   help="version 1: a fresh layout every N steps of each world's clock (0 = never, as in version 0)")
    p.add_argument("--new-map-offset", type=int, default=0,
                   help="version 1: ... at the steps where the clock modulo N equals this")
    p.add_argument("--branches", nargs="+", choices=list(BRANCHES),
                   help="version 1: after one childhood, one adult life in each of these adult worlds")
    p.add_argument("--wave-on", type=int, help="experiment 2: mean length of a harvest in the unreliable childhood")
    p.add_argument("--wave-off", type=int, help="experiment 2: mean length of an empty spell in the unreliable childhood")
    p.add_argument("--out", default="runs")
    args = p.parse_args()
    if args.branches:
        return main_v1(args)

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{args.childhood}_s{args.seed}"
    child_cfg = CHILDHOODS[args.childhood]
    if args.childhood == "scarce" and args.scarce_regrow is not None:
        child_cfg = replace(child_cfg, regrow=args.scarce_regrow)
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

        if args.reset == "last":
            # keep the first layer, give everything after it a fresh start (the same fresh
            # weights for every childhood with this seed), and forget the optimiser's history
            torch.manual_seed(args.seed + 10_000)
            fresh = Net(OBS_DIM, N_ACTIONS)
            for part in ("policy", "value"):
                getattr(agent.net, part).load_state_dict(getattr(fresh, part).state_dict())
            agent.net.body[2].load_state_dict(fresh.body[2].state_dict())
            agent.opt = torch.optim.Adam(agent.net.parameters(), lr=args.lr, eps=1e-5)

        # the same agent, unchanged, in a fresh set of adult worlds
        frozen = copy.deepcopy(agent)
        test_world = Worlds(ADULT, args.worlds, np.random.default_rng([args.seed, 2]))
        live(frozen, test_world, args.frozen_steps, "frozen", write, learn=False)

        adult_world = Worlds(ADULT, args.worlds, np.random.default_rng([args.seed, 1]))
        live(agent, adult_world, args.adult_steps, "adult", write)
        torch.save(agent.net.state_dict(), out / f"{name}_end.pt")
    print(f"{name}: done in {time.time() - started:.0f}s", flush=True)


def reset_last(agent, args):
    """Keep the first layer, give everything after it a fresh start (the same fresh weights for every
    childhood with this seed), and forget the optimiser's history."""
    torch.manual_seed(args.seed + 10_000)
    fresh = Net(OBS_DIM, N_ACTIONS)
    for part in ("policy", "value"):
        getattr(agent.net, part).load_state_dict(getattr(fresh, part).state_dict())
    agent.net.body[2].load_state_dict(fresh.body[2].state_dict())
    agent.opt = torch.optim.Adam(agent.net.parameters(), lr=args.lr, eps=1e-5)


def main_v1(args):
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    maps = lambda cfg: replace(cfg, new_map_every=args.new_map_every, new_map_offset=args.new_map_offset)
    name = f"{args.childhood}_s{args.seed}"
    child_cfg = CHILDHOODS[args.childhood]
    if args.childhood == "scarce" and args.scarce_regrow is not None:
        child_cfg = replace(child_cfg, regrow=args.scarce_regrow)
    if args.childhood == "unreliable" and args.wave_on:
        child_cfg = replace(child_cfg, wave_on=args.wave_on, wave_off=args.wave_off)
    child_cfg = maps(child_cfg) if child_cfg is not None else None
    branches = sorted(args.branches, key=lambda b: BRANCHES[b][0])
    meta = {"args": vars(args), "childhood": asdict(child_cfg) if child_cfg else None,
            "branches": {b: asdict(maps(BRANCHES[b][1])) for b in branches}}
    (out / f"{name}.json").write_text(json.dumps(meta, indent=2))

    torch.manual_seed(args.seed)
    agent = Agent(OBS_DIM, N_ACTIONS, lr=args.lr, gamma=args.gamma)
    started = time.time()
    head = []          # childhood and home-check rows, written at the top of every branch file
    if child_cfg is not None:
        live(agent, Worlds(child_cfg, args.worlds, np.random.default_rng([args.seed, 0])),
             args.child_steps, "child", head.append)
        print(f"{name}: childhood done in {time.time() - started:.0f}s", flush=True)
    torch.save(agent.net.state_dict(), out / f"{name}_after_childhood.pt")
    if child_cfg is not None:
        # the same agent, unchanged, in its own kind of world but on maps it has never seen
        torch.manual_seed(args.seed + 20_000)
        live(copy.deepcopy(agent), Worlds(child_cfg, args.worlds, np.random.default_rng([args.seed, 3])),
             args.frozen_steps, "frozen_home", head.append, learn=False)
    if args.reset == "last":
        reset_last(agent, args)

    for b in branches:
        k, cfg = BRANCHES[b]
        cfg = maps(cfg)
        torch.manual_seed(args.seed + 30_000 + 1000 * k)
        a = copy.deepcopy(agent)          # network and optimiser, as they were at the move
        with open(out / f"{name}_{b}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, COLUMNS)
            writer.writeheader()
            writer.writerows(head)
            write = lambda row: (writer.writerow(row), f.flush())
            live(copy.deepcopy(a), Worlds(cfg, args.worlds, np.random.default_rng([args.seed, 2])),
                 args.frozen_steps, "frozen", write, learn=False)
            live(a, Worlds(cfg, args.worlds, np.random.default_rng([args.seed, 1])), args.adult_steps, "adult", write)
        torch.save(a.net.state_dict(), out / f"{name}_{b}_end.pt")
        print(f"{name}: {b} done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
