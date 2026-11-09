# Pre-registration: version 1

Written after version 0 (pilot on seeds 0-9, robustness check on seeds 10-29) and two rounds of
review, and before any run with seeds 100-119. The commit that adds this file fixes the code, the
settings and the analysis below. It is pushed to the private GitHub repository
`golibjon-rakhmonov/developmental-scarcity-agents` before the first run, so the push time in the
repository's activity log dates it independently of this computer. Anything decided later is
reported as exploratory. Version 0 results stay as they are and are reported as version 0.

## Why a version 1

The review of version 0 found four problems.

1. **The agents learned their maps.** A layout was drawn once, when a world was created, so each of
   the 32 bodies lived its whole childhood on one map. Run without learning, collapses per 1,000
   steps rose from 0.04 to 0.55 (scarce) and from 0.12 to 1.61 (rich) when only the map changed and
   the food stayed the same (seeds 10-29, 100k steps). The frozen check mixed two things: a new food
   regime and an unfamiliar map.
2. **The scarce childhood was not hungry.** Energy stayed around 0.87 and fell below 0.5 about 1% of
   the time. It is a childhood with rare food and long searches.
3. **The adult world was closer to the scarce childhood.** With droughts, the adult world adds
   0.0028 food per patch per step on average: 2.8 times the scarce childhood, a seventh of the rich
   one.
4. **No control for the amount of training.** Agents without a childhood trained 3.1M steps, the
   others 6.1M.

## What changes

- **Maps keep changing.** Every world gets a fresh layout every 1,000 steps of its own clock, at the
  steps where the clock modulo 1,000 equals 500 (`--new-map-every 1000 --new-map-offset 500`), in
  childhood, in both checks and in adult life. So each drought (1,000 steps) contains exactly one map
  change, in its middle, and each stretch without drought (2,000 steps) two. Worlds start at
  different points of their clock, so they do not all change at once. Layouts come from their own
  random generator, so two agents with the same seed see the same sequence of maps whatever they do.
- **Food moves with the map.** At a map change, the food on the old cold patches goes, shuffled, to
  the new cold patches, and the food on the old warm patches to the new warm ones (there are always
  four of each). A map change neither adds nor removes food, and food an agent left on cold ground
  stays on cold ground. The agent keeps its body and is put on warm ground.
- **A childhood in the adult world.** A third childhood, `adult`: 3M steps in the standard adult
  world. It has the same amount of training as the other two, in the world the agent will live in.
- **Mirror adult worlds.** Besides the standard adult world, two adult worlds with the same droughts:
  one richer and one poorer.
- **A home check.** Right after childhood, 100k steps without learning in the agent's own kind of
  world, on maps it has never seen (`frozen_home`). Then, as before, 100k steps without learning in
  the adult world (`frozen`).
- **Branches.** One childhood is followed by three adult lives, one in each adult world. Each starts
  from the network and optimiser exactly as they were at the move.
- **Collapses by drought phase.** Collapses outside droughts and during droughts are now recorded
  apart (`collapses_per_1k_normal`, `collapses_per_1k_drought`), as food carried and time on cold
  ground already were.

| World | Regrowth, normal | Regrowth, drought | Droughts | Mean regrowth |
|---|---|---|---|---|
| scarce childhood | 0.001 | - | none | 0.001 |
| rich childhood | 0.02 | - | none | 0.02 |
| adult childhood = standard adult world | 0.004 | 0.0005 | 1,000 of every 3,000 steps | 0.0028 |
| richer adult world | 0.04 | 0.0005 | 1,000 of every 3,000 steps | 0.0268 |
| poorer adult world | 0.0008 | 0.0003 | 1,000 of every 3,000 steps | 0.00063 |

The poorer world is poorer than both childhoods in both phases. The richer world is richer than both
childhoods only outside droughts: during a drought it is poorer than even the scarce childhood. Its
two phases are therefore tested apart (family B).

Everything else is as in version 0: the 11x11 map, the body, the senses, the reward, the network and
PPO with the same settings, 3M childhood steps, 100k steps per check, 3M adult steps, 32 worlds.

## What this version tests, and what it does not

The scarce childhood is a childhood with **rare food**, not a hungry one: in calibration the agent
kept its energy around 0.86 and its wellbeing at 0.97. Version 1 asks what rare food leaves behind.
A really hungry childhood is a separate experiment, after version 1.

## Design

- Childhoods: scarce, rich, adult, none.
- Seeds 100-119: 20 per childhood. Within a seed, all childhoods start from the same network, see
  the same sequence of maps in each kind of world, and move into the same adult worlds.
- 80 processes (4 childhoods x 20 seeds), each with three adult branches: 240 adult lives.
- `run_v1.py` runs everything with the defaults of `run_life.py` and `world.py` at this commit.

## Measures

The same three as in version 0, measured in the frozen check (mean over its rows):

1. **Food carried while full** (`carry_sated`): pieces carried while energy is above 0.8.
2. **Share of daytime on cold ground** (`cold_day`). Not called "risk".
3. **Collapses per 1,000 steps** (`collapses_per_1k`).

Each also exists outside droughts and during droughts (`_normal`, `_drought`).

## Tests

All tests are two-sided; no direction is fixed in advance. Each is a Wilcoxon signed-rank test on
the paired differences (one pair per seed), reported with the mean difference, its bootstrap 95%
confidence interval (10,000 resamples of seeds) and the matched-pairs rank-biserial correlation.
A difference counts as found if it is significant after Holm correction within its family, alpha
0.05. `analyze_v1.py` does all of this.

**Family A, frozen check in the standard adult world (9 tests).** Scarce minus rich, scarce minus
adult, rich minus adult, on each of the three measures.

**Family B, scarce minus rich in the mirror adult worlds, frozen check (9 tests).** The three
measures in the richer world outside droughts, the three in the richer world during droughts, and
the three in the poorer world (both phases together). The comparison that decides between the
explanations is the richer world outside droughts and the poorer world; the richer world during
droughts is reported as its own line.

**Family C, end of adult life in the standard adult world (3 tests).** Collapses in the last 10% of
adult life: scarce minus rich, scarce minus adult, rich minus adult.

## How the results will be read

Written before the run, so that no story is fitted afterwards. Each row is read per measure; the
version 0 claim rested on collapses, so collapses come first. "Found" means significant after Holm.

| Pattern | Conclusion |
|---|---|
| Scarce minus rich is not found in family A | The version 0 difference was mostly the agents knowing their maps. |
| Scarce minus rich is found in A, and has the same sign, found, in the richer world outside droughts and in the poorer world (B) | The difference belongs to the childhood itself, not to how close the childhood was to the adult world. |
| Scarce minus rich in the richer world outside droughts is found with the opposite sign to the poorer world | The difference comes from how close each childhood is to the world the agent lives in. |
| Scarce is not better than the adult childhood (scarce minus adult in A and C not found, or found in favour of adult) | No hardening: the version 0 advantage of scarce agents was extra training in a similar world. |
| Rich is worse than the adult childhood (rich minus adult found) while scarce and adult do not differ | Abundance spoils: the effect is a cost of a rich childhood, not a benefit of a scarce one. |
| Any other mix, or B not found | Undecided between the explanations; reported as such. |

## Descriptive, without tests

- The map check: collapses at the end of childhood, in the home check and in the frozen check, by
  childhood. With changing maps the first two should be close.
- Means by childhood for every measure, in both drought phases, at the end of childhood, in both
  checks and at the end of adult life; the course of adult life in 200k-step blocks; the agents
  without a childhood.
- Late collapses in the mirror worlds.

## Calibration

Seeds 900 and above only. Everything is in `calibration/`, committed before this file.

- `calibration/v0`: the version 0 calibration runs, added after the fact (see its README).
- `calibration/v1`: the first version 1 calibration (commit 65b90fa). Maps changed at clock 0
  modulo 1,000, and a new map got fresh food drawn uniformly between 0 and 1 on every patch.
- After it, the second review found two problems, and both were fixed before this file:
  - the fresh food gave food to agents that eat patches clean and took it from agents that leave food
    behind, most strongly in the poorer world; food now moves with the map, within each class of
    patches;
  - every drought started and ended exactly on a map change, so collapses during droughts would mix
    with searching on an unfamiliar map. **The offset of 500 steps was added after the first version
    1 calibration, for this reason.** The 1,000-step interval and the two mirror worlds were set
    before any calibration and not changed.
- `calibration/v1b`: the same four calibration lives with both fixes. Agents still learn
  (wellbeing 0.956-0.975 at the end of childhood). The map gap is gone or nearly so: collapses per
  1,000 steps at the end of childhood and in the home check were 0.01 and 0.01 (scarce), 0.27 and
  0.32 (rich), 0.22 and 0.30 (adult). An agent learning from scratch gets down to about 0.6
  collapses per 1,000 steps within 1.5M steps in both mirror worlds. The scarce agent kept its
  energy around 0.86-0.87. Details in `calibration/v1b/README.md`.

**A direction already seen.** In the first version 1 calibration, collapses per 1,000 steps in the
frozen check in the standard adult world were 0.12 for the adult childhood (seed 902), 0.25 for the
scarce childhood and 0.88 for the rich one (both seed 901). This is not even a pair: the adult
childhood had a different seed. It is recorded here as an order already seen, as seed 904 was in
version 0. In the second calibration the same
check gave 0.20 (scarce, seed 901), 0.34 (adult, seed 902) and 0.61 (rich, seed 901). The rich
childhood was the worst both times; the scarce and the adult childhood changed places between the
two rounds.

## Order of steps

1. This file and the code are committed in one commit.
2. The commit is pushed to the private GitHub repository.
3. Its hash is written into the report.
4. Only then `run_v1.py` is started.

## Exclusions

None. Every life is reported. A life that stops for a technical reason is rerun with the same seed
and settings; runs are deterministic, so the rerun gives the same file.

## Left for later

Directional predictions for a replication on new seeds, after this round. A really hungry childhood.
Checking the literature cited in the report. Making the repository public and registering on OSF,
before the paper.
