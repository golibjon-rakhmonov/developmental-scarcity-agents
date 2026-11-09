"""Grid world for the pilot.

A world is a square grid with food patches, water, cold ground and shelters.
Food regrows after it is eaten; how fast it regrows is the one thing that
differs between a scarce and a rich childhood.

The agent never gets coordinates, the time, a step counter or a label for the
world it is in. It gets a 7x7 window around itself, a faint smell of water and
food (stronger closer to them, felt where it stands and one step in each
direction), and how its own body feels: hunger, thirst, body temperature, how
much food it carries and how warm the ground under it is.

Many worlds run side by side (one row per world in every array), so one call
to step() moves every agent once.
"""
from dataclasses import dataclass

import numpy as np

UP, DOWN, LEFT, RIGHT, REST, EAT, DRINK, PICK = range(8)
N_ACTIONS = 8
MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

VIEW = 7
R = VIEW // 2
N_CHANNELS = 5        # food, water, cold ground, shelter, outside the world
N_BODY = 5            # energy, water, temperature, food carried, warmth of the ground
N_SMELL = 10          # water and food smell, here and one step up, down, left, right
OBS_DIM = N_CHANNELS * VIEW * VIEW + N_BODY + N_SMELL
SMELL_RANGE = 4.0     # smell fades by e every this many steps of distance

# body
BITE = 0.2            # food eaten or picked up in one action
CARRY_MAX = 5         # pieces of food the agent can carry
ENERGY_COST = 0.003   # per step, at rest
MOVE_COST = 0.001     # extra per step when moving
CARRY_COST = 0.0004   # extra per step for each piece carried
WATER_COST = 0.004
DRINK_GAIN = 0.3
T_SET = 0.5           # body temperature the agent is built to keep
T_RANGE = 0.35        # deviation that counts as fully uncomfortable
T_COLLAPSE = 0.15     # below this the agent collapses
HEAT_RATE = 0.05      # how fast the body moves toward the temperature it would settle at
BODY_HEAT = 0.2       # the body settles this much above the ground temperature

# ground temperature (before body heat)
WARM_DAY, WARM_NIGHT = 0.3, 0.2
COLD_DAY, COLD_NIGHT = 0.1, -0.1
SHELTER = 0.3

START_ENERGY = START_WATER = 0.8


@dataclass(frozen=True)
class WorldConfig:
    size: int = 11
    n_patches: int = 8
    cold_patch_share: float = 0.5    # share of food patches on cold ground
    regrow: float = 0.004            # food added to a patch per step, up to 1.0
    n_water: int = 2
    n_shelters: int = 2
    day_length: int = 100            # one day and one night
    drought_every: int = 0           # 0 means no droughts
    drought_length: int = 0
    drought_regrow: float = 0.0
    new_map_every: int = 0           # a fresh layout every this many steps of a world's clock; 0 = never (version 0)
    new_map_offset: int = 0          # ... when the clock modulo new_map_every equals this

    def mean_regrow(self):
        """Food added per patch per step, averaged over the drought cycle."""
        if self.drought_every == 0:
            return self.regrow
        dry = self.drought_length / self.drought_every
        return (1 - dry) * self.regrow + dry * self.drought_regrow


SCARCE = WorldConfig(regrow=0.001)
RICH = WorldConfig(regrow=0.02)
ADULT = WorldConfig(regrow=0.004, drought_every=3000, drought_length=1000, drought_regrow=0.0005)
# mirror adult worlds for version 1: same droughts, richer than both childhoods or poorer than both
ADULT_RICHER = WorldConfig(regrow=0.04, drought_every=3000, drought_length=1000, drought_regrow=0.0005)
ADULT_POORER = WorldConfig(regrow=0.0008, drought_every=3000, drought_length=1000, drought_regrow=0.0003)


class Worlds:
    """`n` independent worlds with the same rules, each with one agent."""

    def __init__(self, config, n, rng):
        self.cfg, self.n, self.rng = config, n, rng
        s = config.size
        self.food = np.zeros((n, s, s), np.float32)
        self.patch = np.zeros((n, s, s), bool)
        self.water = np.zeros((n, s, s), bool)
        self.cold = np.zeros((n, s, s), bool)
        self.shelter = np.zeros((n, s, s), bool)
        for i in range(n):
            self._lay_out(i)
        self.food[self.patch] = rng.uniform(0, 1, self.patch.sum())

        # fixed channels, padded so the window never runs off the grid
        pad = lambda a, v: np.pad(a, ((0, 0), (R, R), (R, R)), constant_values=v)
        inside = np.ones((n, s, s), bool)
        self.static = np.stack([pad(self.water, 0), pad(self.cold, 0), pad(self.shelter, 0),
                                pad(~inside, 1)], 1).astype(np.float32)
        self.food_padded = np.zeros((n, s + 2 * R, s + 2 * R), np.float32)

        # smell spreads from every source and fades with walking distance
        ys, xs = np.divmod(np.arange(s * s), s)
        dist = np.abs(ys[:, None] - ys[None]) + np.abs(xs[:, None] - xs[None])
        self.spread = np.exp(-dist / SMELL_RANGE).astype(np.float32)
        self.water_smell = (self.water.reshape(n, -1).astype(np.float32) @ self.spread).reshape(n, s, s) / config.n_water

        # each world starts at a different point of its day and drought cycle
        self.clock = rng.integers(0, 10_000, n)
        self.pos = np.zeros((n, 2), int)
        self.energy = np.zeros(n, np.float32)
        self.water_level = np.zeros(n, np.float32)
        self.temp = np.zeros(n, np.float32)
        self.carry = np.zeros(n, int)
        self._wake(np.ones(n, bool))
        # the agent can drink standing on water or next to it
        wp = np.pad(self.water, ((0, 0), (1, 1), (1, 1)))
        self.near_water = wp[:, 1:-1, 1:-1] | wp[:, :-2, 1:-1] | wp[:, 2:, 1:-1] | wp[:, 1:-1, :-2] | wp[:, 1:-1, 2:]
        self._rows = np.arange(n)
        # version 1: layouts keep changing, drawn from their own generator so that two agents with the
        # same seed see the same sequence of maps whatever they do (collapses draw from self.rng)
        if config.new_map_every:
            self.map_rng = np.random.default_rng(rng.integers(2**63))

    def _lay_out(self, i, rng=None):
        s, cfg = self.cfg.size, self.cfg
        rng = self.rng if rng is None else rng
        band = s // 3
        side = rng.integers(4)               # which edge of the map is cold
        cold = np.zeros((s, s), bool)
        if side == 0: cold[:band] = True
        elif side == 1: cold[-band:] = True
        elif side == 2: cold[:, :band] = True
        else: cold[:, -band:] = True
        cold_cells = np.flatnonzero(cold)
        warm_cells = np.flatnonzero(~cold)
        n_cold = int(round(cfg.n_patches * cfg.cold_patch_share))
        picks_cold = rng.choice(cold_cells, n_cold, replace=False)
        picks_warm = rng.choice(warm_cells, cfg.n_patches - n_cold + cfg.n_water + cfg.n_shelters, replace=False)
        patch = np.zeros(s * s, bool); patch[picks_cold] = True
        patch[picks_warm[:cfg.n_patches - n_cold]] = True
        rest = picks_warm[cfg.n_patches - n_cold:]
        water = np.zeros(s * s, bool); water[rest[:cfg.n_water]] = True
        shelter = np.zeros(s * s, bool); shelter[rest[cfg.n_water:]] = True
        self.cold[i], self.patch[i] = cold, patch.reshape(s, s)
        self.water[i], self.shelter[i] = water.reshape(s, s), shelter.reshape(s, s)

    def _derive(self):
        """Recompute everything that follows from the layouts (after some worlds got a new map)."""
        n, s = self.n, self.cfg.size
        pad = lambda a, v: np.pad(a, ((0, 0), (R, R), (R, R)), constant_values=v)
        inside = np.ones((n, s, s), bool)
        self.static = np.stack([pad(self.water, 0), pad(self.cold, 0), pad(self.shelter, 0),
                                pad(~inside, 1)], 1).astype(np.float32)
        self.water_smell = (self.water.reshape(n, -1).astype(np.float32) @ self.spread).reshape(n, s, s) / self.cfg.n_water
        wp = np.pad(self.water, ((0, 0), (1, 1), (1, 1)))
        self.near_water = wp[:, 1:-1, 1:-1] | wp[:, :-2, 1:-1] | wp[:, 2:, 1:-1] | wp[:, 1:-1, :-2] | wp[:, 1:-1, 2:]

    def _new_maps(self, which):
        """Give the worlds in `which` a fresh layout and put their agents somewhere on warm ground. The food
        moves with the map: the amounts on the old cold patches go, shuffled, to the new cold patches and the
        amounts on the old warm patches to the new warm ones, so a new map neither adds nor removes food, and
        food left on cold ground stays on cold ground. The body (energy, water, temperature, food carried) is
        kept: the agent has moved, not woken up."""
        s, rng = self.cfg.size, self.map_rng
        for i in np.flatnonzero(which):
            old_cold = self.food[i][self.patch[i] & self.cold[i]]
            old_warm = self.food[i][self.patch[i] & ~self.cold[i]]
            self._lay_out(i, rng)
            self.food[i] = 0
            new_cold, new_warm = self.patch[i] & self.cold[i], self.patch[i] & ~self.cold[i]
            assert new_cold.sum() == len(old_cold) and new_warm.sum() == len(old_warm)
            self.food[i][new_cold] = rng.permutation(old_cold)
            self.food[i][new_warm] = rng.permutation(old_warm)
            free = np.flatnonzero(~self.cold[i] & ~self.water[i])
            c = rng.choice(free)
            self.pos[i] = (c // s, c % s)
        self._derive()

    def _wake(self, who):
        """Agents in `who` start (or restart after a collapse) on warm ground with a fresh body."""
        s = self.cfg.size
        for i in np.flatnonzero(who):
            free = np.flatnonzero(~self.cold[i] & ~self.water[i])
            c = self.rng.choice(free)
            self.pos[i] = (c // s, c % s)
        self.energy[who] = START_ENERGY
        self.water_level[who] = START_WATER
        self.temp[who] = T_SET
        self.carry[who] = 0

    def night(self):
        return (self.clock % self.cfg.day_length) >= self.cfg.day_length // 2

    def drought(self):
        cfg = self.cfg
        if cfg.drought_every == 0:
            return np.zeros(self.n, bool)
        return (self.clock % cfg.drought_every) >= cfg.drought_every - cfg.drought_length

    def _ground_temp(self):
        y, x = self.pos[:, 0], self.pos[:, 1]
        cold = self.cold[self._rows, y, x]
        shelter = self.shelter[self._rows, y, x]
        night = self.night()
        t = np.where(cold, np.where(night, COLD_NIGHT, COLD_DAY), np.where(night, WARM_NIGHT, WARM_DAY))
        return np.where(shelter, SHELTER, t).astype(np.float32)

    def observe(self):
        self.food_padded[:, R:-R, R:-R] = self.food
        ys = self.pos[:, 0][:, None] + np.arange(VIEW)           # window rows in padded coordinates
        xs = self.pos[:, 1][:, None] + np.arange(VIEW)
        b = self._rows[:, None, None]
        food = self.food_padded[b, ys[:, :, None], xs[:, None, :]]
        fixed = self.static[b, :, ys[:, :, None], xs[:, None, :]]  # (n, 7, 7, 4)
        grid = np.concatenate([food[:, None], fixed.transpose(0, 3, 1, 2)], 1)
        body = np.stack([self.energy, self.water_level, self.temp,
                         self.carry / CARRY_MAX, self._ground_temp()], 1)
        s = self.cfg.size
        food_smell = (self.food.reshape(self.n, -1) @ self.spread).reshape(self.n, s, s) / 2
        smell = np.concatenate([self._around(self.water_smell), self._around(food_smell)], 1)
        return np.concatenate([grid.reshape(self.n, -1), body, smell], 1).astype(np.float32)

    def _around(self, field):
        """A field's value where the agent stands and one step up, down, left and right
        (off the map counts as where it stands)."""
        s = self.cfg.size
        y, x = self.pos[:, 0], self.pos[:, 1]
        out = [field[self._rows, y, x]]
        for dy, dx in MOVES:
            out.append(field[self._rows, np.clip(y + dy, 0, s - 1), np.clip(x + dx, 0, s - 1)])
        return np.stack(out, 1)

    def wellbeing(self):
        """1 when every need is met, lower the further the body is from its set points."""
        drive = ((1 - self.energy) ** 2 + (1 - self.water_level) ** 2
                 + np.minimum(np.abs(self.temp - T_SET) / T_RANGE, 1) ** 2) / 3
        return 1 - drive

    def step(self, actions):
        n, rows = self.n, self._rows
        moving = actions < 4
        new = np.clip(self.pos + MOVES[np.where(moving, actions, 0)], 0, self.cfg.size - 1)
        self.pos = np.where(moving[:, None], new, self.pos)
        y, x = self.pos[:, 0], self.pos[:, 1]
        here = self.food[rows, y, x]

        # eat: from the ground if there is food here, otherwise from what is carried
        eat = actions == EAT
        from_ground = eat & (here > 0.01)
        from_carry = eat & ~from_ground & (self.carry > 0)
        bite = np.where(from_ground, np.minimum(BITE, here), 0) + np.where(from_carry, BITE, 0)
        self.food[rows, y, x] -= np.where(from_ground, np.minimum(BITE, here), 0)
        self.carry -= from_carry
        self.energy += bite

        # pick up a piece to carry
        pick = (actions == PICK) & (self.food[rows, y, x] >= BITE) & (self.carry < CARRY_MAX)
        self.food[rows, y, x] -= np.where(pick, BITE, 0)
        self.carry += pick

        drink = (actions == DRINK) & self.near_water[rows, y, x]
        self.water_level += np.where(drink, DRINK_GAIN, 0)

        # the body uses energy and water, and drifts toward the temperature of the ground
        ground = self._ground_temp()
        self.energy -= ENERGY_COST + MOVE_COST * moving + CARRY_COST * self.carry
        self.water_level -= WATER_COST
        self.temp += HEAT_RATE * (ground + BODY_HEAT - self.temp)
        np.clip(self.energy, 0, 1, out=self.energy)
        np.clip(self.water_level, 0, 1, out=self.water_level)

        drought = self.drought()
        rate = np.where(drought, self.cfg.drought_regrow, self.cfg.regrow).astype(np.float32)
        self.food += self.patch * rate[:, None, None]
        np.clip(self.food, 0, 1, out=self.food)
        self.clock += 1

        collapse = (self.energy <= 0) | (self.water_level <= 0) | (self.temp < T_COLLAPSE)
        reward = np.where(collapse, 0.0, self.wellbeing()).astype(np.float32)

        info = {
            "collapse": collapse,
            "starved": collapse & (self.energy <= 0),
            "thirst": collapse & (self.water_level <= 0),
            "froze": collapse & (self.temp < T_COLLAPSE),
            "carry": self.carry.copy(),
            "sated": self.energy > 0.8,
            "on_cold": self.cold[rows, y, x],
            "night": self.night(),
            "drought": drought,
            "energy": self.energy.copy(),
            "water": self.water_level.copy(),
            "temp": self.temp.copy(),
            "ate": bite > 0,
            "picked": pick,
            "drank": drink,
        }
        if collapse.any():
            self._wake(collapse)
        if self.cfg.new_map_every:
            switch = self.clock % self.cfg.new_map_every == self.cfg.new_map_offset
            if switch.any():
                self._new_maps(switch)
        return self.observe(), reward, collapse, info
