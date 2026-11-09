"""Where in the network does childhood leave its trace? An exploratory analysis of the networks saved
at the move in version 1 and its replication. Nothing learns here, and nothing was registered in
advance: the results are reported as exploration.

    .venv/bin/python trace_v1.py --parallel 17
    .venv/bin/python trace_v1.py --seeds 100 --frozen-steps 4096 --out trace_test     # to test the script

The network (agent.py) is: first hidden layer, second hidden layer, then the action head and the
value head. All childhoods of one seed start from the same untrained network, saved as
none_s<seed>_after_childhood.pt.

1. Similarity by layer. The three twins of a seed (scarce, rich and adult childhood) and their shared
   untrained start get the same observations, and the linear CKA (Kornblith et al., 2019) between
   their activations is computed for each layer. 1 means the same representation up to rotation and
   scale; lower means the twins came apart there.
2. Chimeras. A network is built from the lower layers of one twin and the upper layers of the other,
   cut after the first hidden layer ("cut1": only the first layer is the donor's) or after the second
   ("cut2": both hidden layers are the donor's, only the heads are the other twin's). Every chimera
   and both intact twins go through the frozen check (100k steps without learning) in each adult
   world, with the same maps and the same random numbers as the frozen check of version 1. If a
   chimera behaves like the twin that gave its lower layers, the trace is there; if it behaves like
   the twin that gave its upper layers, the trace is above the cut. A chimera whose wellbeing falls
   well below both twins shows that the layers do not fit together, and its cut is not read.

Writes <out>/similarity.csv, <out>/chimeras.csv and <out>/summary.txt.
"""
import argparse
import copy
import itertools
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from agent import Agent
from analyze import paired
from run_life import BRANCHES, live
from world import N_ACTIONS, OBS_DIM, Worlds

TWINS = ["scarce", "rich", "adult"]
LAYERS = ["hidden1", "hidden2", "actions", "value"]
NEW_MAP_EVERY, NEW_MAP_OFFSET = 1000, 500
SOURCES = {**{s: "runs_v1" for s in range(100, 120)}, **{s: "runs_v1_rep" for s in range(120, 140)}}
CUTS = {"cut1": ["body.0."], "cut2": ["body.0.", "body.2."]}
READ = {          # what to read in each adult world (the measures that differed between childhoods)
    "standard": ["wellbeing", "collapses_per_1k", "cold_day", "carry_sated"],
    "richer": ["wellbeing", "collapses_per_1k_normal", "thirst_per_1k", "meals_per_100", "drinks_per_100"],
    "poorer": ["wellbeing", "collapses_per_1k", "starved_per_1k"],
}


def load(seed, who):
    return torch.load(Path(SOURCES[seed]) / f"{who}_s{seed}_after_childhood.pt", weights_only=True)


def chimera(low, high, cut):
    """Lower layers (up to the cut) from `low`, everything above from `high`."""
    return {k: (low if any(k.startswith(p) for p in CUTS[cut]) else high)[k].clone() for k in high}


def activations(state, obs):
    agent = Agent(OBS_DIM, N_ACTIONS)
    agent.net.load_state_dict(state)
    with torch.no_grad():
        h1 = agent.net.body[:2](obs)
        h2 = agent.net.body[2:](h1)
        return {"hidden1": h1, "hidden2": h2, "actions": agent.net.policy(h2), "value": agent.net.value(h2)}


def cka(x, y):
    x, y = x - x.mean(0), y - y.mean(0)
    return float((y.T @ x).norm() ** 2 / ((x.T @ x).norm() * (y.T @ y).norm()))


def similarity(seed, n_steps=48, n_worlds=32):
    """Observations met by each twin in the standard adult world (maps unlike those of any check),
    pooled, then the CKA between every two networks of the seed at every layer."""
    obs = []
    for k, who in enumerate(TWINS):
        agent = Agent(OBS_DIM, N_ACTIONS)
        agent.net.load_state_dict(load(seed, who))
        w = Worlds(replace(BRANCHES["standard"][1], new_map_every=NEW_MAP_EVERY, new_map_offset=NEW_MAP_OFFSET),
                   n_worlds, np.random.default_rng([seed, 50 + k]))
        torch.manual_seed(seed + 50_000 + k)
        o = w.observe()
        for _ in range(n_steps):
            obs.append(o.copy())
            o = w.step(agent.act(o)[0])[0]
    obs = torch.as_tensor(np.concatenate(obs))
    acts = {who: activations(load(seed, who), obs) for who in TWINS + ["none"]}
    rows = []
    for a, b in itertools.combinations(TWINS + ["none"], 2):
        for layer in LAYERS:
            rows.append({"seed": seed, "pair": f"{a}-{b}", "layer": layer, "cka": cka(acts[a][layer], acts[b][layer])})
    return rows


def frozen_check(job):
    seed, name, state, branch, steps = job
    torch.set_num_threads(1)
    k, cfg = BRANCHES[branch]
    agent = Agent(OBS_DIM, N_ACTIONS)
    agent.net.load_state_dict(state)
    rows = []
    # as in run_life.main_v1: the torch seed is set right before the check, then the same worlds
    torch.manual_seed(seed + 30_000 + 1000 * k)
    live(agent, Worlds(replace(cfg, new_map_every=NEW_MAP_EVERY, new_map_offset=NEW_MAP_OFFSET), 32,
                       np.random.default_rng([seed, 2])), steps, "frozen", rows.append, learn=False)
    d = pd.DataFrame(rows)
    return {"seed": seed, "network": name, "world": branch,
            **{m: float(np.nanmean(d[m])) for ms in READ.values() for m in ms}}


def jobs(seed, steps):
    nets = {f"intact {who}": load(seed, who) for who in TWINS}
    for low, high in [("scarce", "rich"), ("rich", "scarce")]:
        for cut in CUTS:
            nets[f"{cut}: low {low}, high {high}"] = chimera(load(seed, low), load(seed, high), cut)
    return [(seed, name, state, b, steps) for name, state in nets.items() for b in READ]


def summarise(sim, chim, rng):
    out = ["SIMILARITY BY LAYER (linear CKA, mean over seeds, 95% CI)"]
    for pair, g in sim.groupby("pair", sort=False):
        cells = []
        for layer in LAYERS:
            v = g[g.layer == layer].cka.to_numpy()
            boot = rng.choice(v, (10_000, len(v))).mean(1)
            cells.append(f"{layer} {v.mean():.3f} [{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}]")
        out.append(f"  {pair:14s} " + "   ".join(cells))

    out.append("\nCHIMERAS, frozen check. 'position' = where the chimera lies between the twin that gave its")
    out.append("upper layers (0) and the twin that gave its lower layers (1), from the means over seeds.")
    seeds = sorted(chim.seed.unique())
    get = lambda name, world, m: chim[(chim.network == name) & (chim.world == world)].set_index("seed").loc[seeds, m].to_numpy()
    for world, measures in READ.items():
        out.append(f"\n  {world} adult world")
        for m in measures:
            s, r = get("intact scarce", world, m), get("intact rich", world, m)
            out.append(f"    {m:24s} intact scarce {s.mean():7.3f}   intact rich {r.mean():7.3f}")
            for cut in CUTS:
                for low, high in [("scarce", "rich"), ("rich", "scarce")]:
                    c = get(f"{cut}: low {low}, high {high}", world, m)
                    lo, hi = (s, r) if low == "scarce" else (r, s)
                    pos = (c.mean() - hi.mean()) / (lo.mean() - hi.mean()) if lo.mean() != hi.mean() else float("nan")
                    st = paired(c, hi, rng)
                    out.append(f"      {cut} low {low:6s} high {high:6s} {c.mean():7.3f}  position {pos:+.2f}   "
                               f"chimera - upper twin {st['mean_difference']:+.3f} [{st['ci95'][0]:+.3f}, {st['ci95'][1]:+.3f}]")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 140)))
    p.add_argument("--frozen-steps", type=int, default=100_000)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--out", default="trace_v1")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    torch.set_num_threads(1)

    sim = pd.DataFrame([r for s in args.seeds for r in similarity(s)])
    sim.to_csv(out / "similarity.csv", index=False)
    print(f"similarity done for {len(args.seeds)} seeds", flush=True)

    todo = [j for s in args.seeds for j in jobs(s, args.frozen_steps)]
    rows = []
    with Pool(args.parallel) as pool:
        for i, r in enumerate(pool.imap_unordered(frozen_check, todo), 1):
            rows.append(r)
            if i % 20 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] frozen checks done", flush=True)
    chim = pd.DataFrame(rows).sort_values(["seed", "network", "world"])
    chim.to_csv(out / "chimeras.csv", index=False)

    text = summarise(sim, chim, np.random.default_rng(0))
    (out / "summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
