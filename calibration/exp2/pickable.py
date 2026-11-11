"""How often is there food that can be picked up? A check made after experiment 2, NOT pre-registered.

An agent can pick up a piece only from a patch that holds at least one bite (0.2). This script runs the
calibration agents of the hungry, the scarce and the unreliable childhood (seeds 900 and 901, the networks
saved at the end of childhood) for 3,000 steps without learning in their own kind of world, on the maps of
the home check, and prints the share of time in which at least one such patch exists.

    .venv/bin/python calibration/exp2/pickable.py > calibration/exp2/pickable.txt
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from agent import Net                                                 # noqa: E402
from world import BITE, N_ACTIONS, OBS_DIM, WorldConfig, Worlds      # noqa: E402

STEPS, N = 3000, 32
torch.set_num_threads(1)
for label, name in [("hungry (0.00045)", "hungry_0.00045/scarce"), ("scarce", "scarce/scarce"),
                    ("unreliable", "long/unreliable")]:
    for seed in (900, 901):
        cfg = WorldConfig(**json.loads((HERE / f"{name}_s{seed}.json").read_text())["childhood"])
        net = Net(OBS_DIM, N_ACTIONS)
        net.load_state_dict(torch.load(HERE / f"{name}_s{seed}_after_childhood.pt"))
        net.eval()
        torch.manual_seed(seed + 20_000)
        worlds = Worlds(cfg, N, np.random.default_rng([seed, 3]))
        obs = worlds.observe()
        some, count = [], []
        for _ in range(STEPS):
            with torch.no_grad():
                actions = torch.distributions.Categorical(logits=net(torch.as_tensor(obs))[0]).sample().numpy()
            obs = worlds.step(actions)[0]
            full = worlds.food.reshape(N, -1) >= BITE
            some.append(full.any(1))
            count.append(full.sum(1))
        print(f"{label:18s} seed {seed}: at least one patch with a piece {100 * np.mean(some):5.1f}% of the time, "
              f"{np.mean(count):.2f} such patches on average")
