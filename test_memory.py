"""Checks of the agent with memory (run from the repository root: .venv/bin/python test_memory.py).

1. Replaying a rollout the way learn() does gives exactly the values and action probabilities the agent had
   when it acted, also across collapses, and ends with the same memory.
2. The check is sensitive: a replay that does not empty the memory at collapses gives different values.
3. The same observation with a different memory gives a different output, and a collapse empties the memory.
"""
from dataclasses import replace

import numpy as np
import torch

from agent import MemoryAgent
from world import ADULT, N_ACTIONS, OBS_DIM, Worlds

ROLLOUT, N = 128, 32
torch.set_num_threads(1)
torch.manual_seed(0)
agent = MemoryAgent(OBS_DIM, N_ACTIONS)
worlds = Worlds(replace(ADULT, new_map_every=1000, new_map_offset=500), N, np.random.default_rng(0))
obs = worlds.observe()
agent.begin(N)

collapses = 0
for k in range(3):
    start = agent.memory.clone()
    b_obs = np.zeros((ROLLOUT, N, OBS_DIM), np.float32)
    b_act, b_logp, b_val, b_done = (np.zeros((ROLLOUT, N), np.float32) for _ in range(4))
    for t in range(ROLLOUT):
        a, logp, v = agent.act(obs)
        b_obs[t], b_act[t], b_logp[t], b_val[t] = obs, a, logp, v
        obs, _, collapsed, _ = worlds.step(a)
        agent.after_step(collapsed)
        b_done[t] = collapsed
    with torch.no_grad():
        encoded = agent.net.encode(torch.as_tensor(b_obs))
        keep = torch.as_tensor(1 - b_done)
        memory, no_reset = start, start
        logits, values, values_no_reset = [], [], []
        for t in range(ROLLOUT):
            l, x, memory = agent.net.step(encoded[t], memory)
            logits.append(l)
            values.append(x)
            memory = memory * keep[t, :, None]
            _, y, no_reset = agent.net.step(encoded[t], no_reset)
            values_no_reset.append(y)
        logp = torch.distributions.Categorical(logits=torch.stack(logits)).log_prob(torch.as_tensor(b_act).long())
    d_logp = float(np.abs(logp.numpy() - b_logp).max())
    d_val = float(np.abs(torch.stack(values).numpy() - b_val).max())
    d_wrong = float(np.abs(torch.stack(values_no_reset).numpy() - b_val).max())
    collapses += int(b_done.sum())
    print(f"rollout {k}: {int(b_done.sum())} collapses, replay differs by {d_logp:.1e} (log-probabilities) and "
          f"{d_val:.1e} (values); without emptying the memory at collapses by {d_wrong:.1e}")
    assert d_logp < 1e-5 and d_val < 1e-5 and torch.allclose(memory, agent.memory, atol=1e-6)
    if b_done.sum() and k > 0:
        assert d_wrong > 1e-3
assert collapses > 0, "the check needs collapses"

with torch.no_grad():
    o = torch.as_tensor(obs)
    assert float((agent.net(o, torch.zeros(N, 128))[0] - agent.net(o, agent.memory)[0]).abs().max()) > 0
agent.after_step(np.array([True] + [False] * (N - 1)))
assert float(agent.memory[0].abs().sum()) == 0.0 and float(agent.memory[1].abs().sum()) > 0
print("all checks passed")
