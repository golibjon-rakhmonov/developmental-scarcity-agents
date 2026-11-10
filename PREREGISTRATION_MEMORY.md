# Pre-registration: robustness to an agent with memory

Written after version 1, its replications, experiment 2, the network-size check and the registration of
experiment 3, and before any run with seeds 220-239. The commit that adds this file fixes the code, the
settings and the analysis. It is pushed to the private GitHub repository before the first run.

    .venv/bin/python run_v1.py --parallel 17 --seeds $(seq 220 239) --childhoods scarce rich adult --memory --out runs_memory
    .venv/bin/python analyze_v1_rep.py --runs runs_memory --seeds $(seq 220 239)

## Why

Every registered run so far used an agent without memory: what it does depends only on what it senses at
that step. Such an agent cannot notice, within its life, that the world around it has changed; it can only
relearn. An agent with memory could. A result that holds only for an agent without memory says little
about learning agents in general, so this check repeats version 1 with an agent that has one.

## What changes

Only the network and how it is trained. The first hidden layer is as before (128 tanh units). The second
hidden layer is a recurrent GRU cell of 128 units whose state is carried from step to step (`MemoryNet`
and `MemoryAgent` in `agent.py`, `--memory`). The memory is empty when the agent enters a new set of
worlds (at the start of childhood, of the home check, of each frozen check and of each adult life) and
after a collapse, when the agent wakes with a fresh body. PPO has the same settings as before (learning
rate, discount, GAE, clip, 4 epochs, 4 minibatches, entropy bonus, rollouts of 128 steps); it is trained
through time over each whole rollout, and a minibatch is 8 of the 32 worlds with their whole rollouts.
The network has about 134,000 weights (51,000 without memory; 135,000 in the 256-unit network of the
network-size check).

Everything else is as in version 1: the scarce, rich and adult childhoods, the three adult worlds, maps
that change every 1,000 steps at clock 500, the home check, the frozen checks, 3M childhood steps, 3M adult
steps. Without `--memory` the code gives exactly the lives of version 1 (checked on three short lives:
identical tables and identical networks).

All of this was fixed before the calibration below.

## Predictions and tests

The 26 predictions of `PREREGISTRATION_V1_REPLICATION.md`, tested with `analyze_v1_rep.py` unchanged:
two-sided Wilcoxon signed-rank tests on paired differences, Holm within each family, equivalence margins of
+-0.15 (frozen check) and +-0.05 (end of adult life).

Known before this run: "scarce minus adult, collapses in the poorer world, no difference" failed in both
replications and at 64 units in the same direction (scarce collapsed less). It is kept unchanged so that
the analysis stays the same; a further failure in that direction is the same finding.

## How the results will be read

| Pattern | Conclusion |
|---|---|
| The collapse predictions of families B, D and E and the cold-ground predictions of families A and B hold | The mismatch and the cold-ground habit do not depend on the agent having no memory. |
| The collapse predictions of family B hold in sign but some of B, D, E fail | Memory weakens the mismatch; reported test by test. |
| The collapse predictions of family B do not hold | With memory, a childhood does not leave the mismatch found in version 1. |

Food carried and the "no difference" predictions are reported as they come. Each prediction is reported
whether it holds or not. The size of the mismatch against the agents without memory of version 1 and its
replications (different seeds, so not paired) is reported as descriptive.

## Calibration

Seed 900 only, childhood, home check and frozen checks, no adult life: scarce, rich and adult childhood
with `--memory` (`calibration/memory/`). Its only purpose is to check that the agent with memory learns;
nothing was tuned on it. It learns: wellbeing at the end of childhood was 0.969, 0.981 and 0.956 (scarce,
rich, adult), against 0.959, 0.976 and 0.945 for the agent without memory with the same seed
(`calibration/exp2/scarce`, `calibration/exp3`).

**Directions already seen** (one seed, recorded so they are not taken for blind predictions): collapses per
1,000 steps in the richer world outside droughts were 2.99 (scarce) and 0.14 (rich); in the poorer world
0.53 and 1.55. The rich agent spent the least daytime on cold ground (0.005, against 0.062 and 0.018).

`test_memory.py` checks the training of the agent with memory: replaying a rollout in learning gives
exactly the values and action probabilities the agent had when it acted, also across collapses. Two lives
with the same seed are identical.

## Order of steps

1. This file and the code are committed in one commit.
2. The commit is pushed to the private GitHub repository.
3. Its hash is written into the report.
4. Only then `run_v1.py` is started.

## Exclusions

None. Every life is reported. A life that stops for a technical reason is rerun with the same seed and
settings.
