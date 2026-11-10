# childhood-scarcity

Pilot for one question: does growing up with too little food leave a lasting mark on how a
learning agent behaves later, once it lives in the same world as everyone else?

## The world

An 11x11 grid with eight food patches, two water cells, two shelters and a band of cold ground
along one edge. Half of the food grows on the cold ground. Day and night alternate every 50 steps,
and at night the cold ground becomes dangerous.

Food regrows after it is eaten. How fast it regrows is the only thing that differs between the
two childhoods:

| World | Regrowth per patch per step | Droughts |
|---|---|---|
| scarce childhood | 0.001 | none |
| rich childhood | 0.02 | none |
| adult world | 0.004 | 1,000 of every 3,000 steps, regrowth 0.0005 |

## The agent

The agent gets no coordinates, no clock, no step counter and no label for the world it is in.
It senses:

- a 7x7 window around itself: food, water, cold ground, shelter, edge of the world;
- a faint smell of water and of food, where it stands and one step in each direction;
- its body: energy, water, temperature, how much food it carries, how warm the ground is.

It can move, rest, eat, drink, or pick up food to carry (up to five pieces, each one costs a
little energy to carry).

The reward is its wellbeing at each step: 1 when energy, water and temperature are where the body
wants them, lower the further they drift. Nothing rewards food as such. If energy or water reaches
zero, or the body gets too cold, the agent collapses: it gets nothing for that step and wakes up
somewhere else on warm ground with a fresh body. What it has learned stays, so a collapse costs it
the wellbeing it would have had, not its memory.

Learning is PPO with a small two-layer network. 32 copies of the world run side by side and all of
them update the same network, so one network is one individual.

## A life

1. Childhood: 3 million steps in a scarce or a rich world (or none, for the baseline).
2. The move: the agent is put into the adult world without any signal.
3. Frozen check: 100,000 steps in the adult world with learning switched off. This is the clean
   trace of childhood.
4. Adult life: 3 million more steps in the adult world, learning on.

With the same seed, all three conditions start from the same untrained network and move into the
same adult world, so they can be compared seed by seed.

## Running

    python3.12 -m venv .venv            # needs Python 3.11 or newer; the python3 that ships with macOS is 3.9
    .venv/bin/pip install -r requirements.txt

    .venv/bin/python run_life.py --childhood scarce --seed 0      # one life
    PYTHON=.venv/bin/python ./run_pilot.sh 0 9 8                  # seeds 0-9, 8 lives at a time

Watching:

    .venv/bin/python curves.py --live                              # learning curves while lives run
    .venv/bin/python watch.py runs/scarce_s0_after_childhood.pt     # one agent, live, in the adult world
    .venv/bin/python watch.py runs/rich_s0_end.pt --world rich      # or back in a rich world
    .venv/bin/python summary.py                                    # table by condition and stage

## Robustness check

After the pilot: a replication on seeds 10-29, four variations (longer and shorter childhood,
harsher and milder scarcity) and a layer reset at the move. The plan is in
`PREREGISTRATION_ROBUSTNESS.md`.

    .venv/bin/python run_robust.py          # all 160 lives, results in runs_robust/
    .venv/bin/python analyze_robust.py      # the pre-registered tests

## Version 1

A review of version 0 found that every body lived its whole childhood on one map, so the agents
learned their maps by heart, and that the scarce childhood was closer to the adult world than the
rich one. Version 1 changes three things; the plan is in `PREREGISTRATION_V1.md`.

- Every world gets a fresh layout every 1,000 steps of its clock, halfway between drought edges
  (`--new-map-every 1000 --new-map-offset 500`). Food moves with the map: amounts on the old cold
  patches go to the new cold patches, warm to warm. Two agents with the same seed still see the same
  sequence of maps.
- A third childhood, `adult`: 3M steps in the adult world itself, the same amount of training.
- After one childhood the agent lives three adult lives, in the standard adult world and in two
  mirror worlds, one richer and one poorer than both childhoods. A home check (100k steps without
  learning in its own kind of world, on maps it has never seen) comes before them.

    .venv/bin/python run_v1.py --parallel $(( $(sysctl -n hw.ncpu) - 1 ))     # 80 processes, results in runs_v1/
    .venv/bin/python analyze_v1.py

The replication with directional predictions (`PREREGISTRATION_V1_REPLICATION.md`) runs the same
lives on seeds 120-139:

    .venv/bin/python run_v1.py --parallel 17 --seeds $(seq 120 139) --out runs_v1_rep
    .venv/bin/python analyze_v1_rep.py

Without `--new-map-every` and `--branches`, `run_life.py` lives exactly the same lives as in
version 0; its CSV only has two more columns (collapses outside droughts and during droughts).

Where in the network the trace of childhood lives (an exploratory analysis of the networks saved at the
move, no learning): `trace_v1.py`, results in `trace_v1/`.

## Experiment 2

`PREREGISTRATION_EXP2.md`. Two more childhoods: a hungry one (steady, 0.00045 food per patch per step)
and an unreliable one (0.001 on average, but in harvests and empty spells of random length), matched on
wellbeing in calibration (`calibration/exp2/`). Seeds 140-169, five childhoods, 150 processes:

    .venv/bin/python run_v1.py --parallel 17 --seeds $(seq 140 169) --childhoods scarce hungry unreliable rich adult --out runs_exp2
    .venv/bin/python analyze_exp2.py
    .venv/bin/python analyze_v1_rep.py --runs runs_exp2 --seeds $(seq 140 169)     # second replication of version 1

## Robustness to network size

`PREREGISTRATION_NETSIZE.md`. Version 1 again with 64 and with 256 units per hidden layer instead of 128,
seeds 170-189, scarce, rich and adult childhood:

    .venv/bin/python run_v1.py --parallel 17 --seeds $(seq 170 189) --childhoods scarce rich adult --hidden 64 --out runs_net64
    .venv/bin/python run_v1.py --parallel 17 --seeds $(seq 170 189) --childhoods scarce rich adult --hidden 256 --out runs_net256
    .venv/bin/python analyze_v1_rep.py --runs runs_net64 --seeds $(seq 170 189)
    .venv/bin/python analyze_v1_rep.py --runs runs_net256 --seeds $(seq 170 189)

## Experiment 3

`PREREGISTRATION_EXP3.md`. One more childhood, `varied`: plentiful spells (food as in the rich childhood)
and lean spells (food as in the scarce childhood) by turns, 5,000 steps each on average. It tests whether
living through both makes an agent do well across adult worlds. Seeds 190-219, four childhoods, 120
processes:

    .venv/bin/python run_v1.py --parallel 17 --seeds $(seq 190 219) --childhoods scarce rich adult varied --out runs_exp3
    .venv/bin/python analyze_exp3.py
    .venv/bin/python analyze_v1_rep.py --runs runs_exp3 --seeds $(seq 190 219)     # third replication of version 1

## Robustness to an agent with memory

`PREREGISTRATION_MEMORY.md`. Version 1 again with an agent that has a memory: the second hidden layer is
a recurrent GRU cell (`--memory`), trained through time. Seeds 220-239, scarce, rich and adult childhood,
60 processes (each takes several times longer than without memory):

    .venv/bin/python run_v1.py --parallel 17 --seeds $(seq 220 239) --childhoods scarce rich adult --memory --out runs_memory
    .venv/bin/python analyze_v1_rep.py --runs runs_memory --seeds $(seq 220 239)

## Output

For each life, in `runs/`:

- `<childhood>_s<seed>.csv`: one row per 4,096 steps with wellbeing, collapses and their causes,
  food carried (overall, while full, during droughts), time on cold ground, drinks and meals;
- `<childhood>_s<seed>.json`: the settings of that life;
- `<childhood>_s<seed>_after_childhood.pt` and `<childhood>_s<seed>_end.pt`: the network at the
  move and at the end.

## Seeds

Seeds 900 and above were used to calibrate the world. Seeds 0-9 are kept for the pilot itself and
are run only after `PREREGISTRATION.md` is committed. Seeds 10-29 belong to the version 0 robustness
check, seeds 100-119 to version 1, seeds 120-139 to its replication, seeds 140-169 to experiment 2, seeds 170-189 to the network-size check,
seeds 190-219 to experiment 3, seeds 220-239 to the check with memory.
The calibration runs are in `calibration/`.
