"""Inputs of the observation: Setup and states [B, N] from recorded battles (tools/nn/gamedata.py),
from the simulator (tools/nn/sim/state.py) and made up (tests, timing).

Several battles make one batch: units are padded to the largest army pair, time to the longest
battle (a finished battle repeats its last sample). The recordings have no per-side visibility
(an empty flat map: every unit is taken as visible); fatigue `fat` is the loader's state index.
"""
import json
from dataclasses import dataclass

import numpy as np

from tools.nn import gamedata
from tools.nn.model.observation import Setup

# MP Crossroads (flat), the arena's map: the minimap frame (docs/en/game/maps/crossroads-flat.md).
CROSSROADS = (-768.0, 768.0, -800.0, 736.0)
EMPIRE = "wh_main_emp_empire"
SKAVEN = "wh2_main_skv_skaven"


def factions(battle, runs_dir=None):
    """(faction of side 1, faction of side 2) from the run's manifest."""
    path = (runs_dir or gamedata.RUNS) / battle.run / "manifest.json"
    f = {}
    if path.exists():
        f = json.loads(path.read_text(encoding="utf-8"))["config"].get("factions") or {}
    return f.get("own", EMPIRE), f.get("enemy", EMPIRE)


def attacker(battle):
    """Side 1 attacks when its AI attacks; 'defend' and 'hold' leave the attack to side 2."""
    return 1 if battle.own_ai == "attack" else 2


def from_sim(state, factions=None):
    """The simulator's State -> (Setup, state dict of tensors [B, N]).

    The State has no factions: `factions` [B] of (side 1, side 2); default Empire vs Skaven.
    """
    B = state.B
    h = float(state.bounds)
    setup = Setup(keys=[list(k) for k in state.keys], side=state.u["side"].cpu().numpy(),
                  bounds=np.tile(np.array([-h, h, -h, h], np.float32), (B, 1)),
                  factions=factions or [(EMPIRE, SKAVEN)] * B, attacker=state.attacker.cpu().numpy())
    return setup, state.observation()


@dataclass
class Recorded:
    setup: Setup
    t: np.ndarray        # [T, B] seconds
    fields: dict         # name -> [T, B, N]
    length: np.ndarray   # [B] samples per battle

    def state(self, ti):
        """The state at step ti: dict of [B, N] arrays, t [B]."""
        s = {k: v[ti] for k, v in self.fields.items()}
        s["t"] = self.t[ti]
        return s

    @property
    def steps(self):
        return self.t.shape[0]


def batch(battles, runs_dir=None, bounds=CROSSROADS):
    """Recorded battles -> Recorded (units padded, time padded with the last sample)."""
    B = len(battles)
    N = max(len(b.names) for b in battles)
    T = max(len(b.t) for b in battles)
    keys = [list(b.keys) + [""] * (N - len(b.keys)) for b in battles]
    side = np.zeros((B, N), dtype=np.int64)
    for i, b in enumerate(battles):
        side[i, :len(b.side)] = b.side
    setup = Setup(keys=keys, side=side, bounds=np.tile(np.asarray(bounds, np.float32), (B, 1)),
                  factions=[factions(b, runs_dir) for b in battles],
                  attacker=np.array([attacker(b) for b in battles]))
    names = gamedata.FLOAT_FIELDS + gamedata.BOOL_FIELDS
    names += tuple(k for k in ("fat",) if all(k in b.f for b in battles))
    fields = {}
    for name in names + ("target",):
        dtype = bool if name in gamedata.BOOL_FIELDS else (np.int64 if name == "target" else np.float64)
        fill = -1 if name == "target" else 0
        out = np.full((T, B, N), fill, dtype=dtype)
        for i, b in enumerate(battles):
            v = b.target if name == "target" else b.f[name]
            n = v.shape[1]
            out[:len(b.t), i, :n] = v
            out[len(b.t):, i, :n] = v[-1]
        fields[name] = out
    t = np.zeros((T, B))
    for i, b in enumerate(battles):
        t[:len(b.t), i] = b.t
        t[len(b.t):, i] = b.t[-1] + np.arange(1, T - len(b.t) + 1)
    return Recorded(setup, t, fields, np.array([len(b.t) for b in battles]))


def synthetic(batch=1, own=20, enemy=20, seed=0, bounds=CROSSROADS, keys=None):
    """A made-up battle for tests and timing: (Setup, state) with `own` units of side 1 facing
    `enemy` units of side 2 across 350 m, random health, morale, movement. Keys from the passports."""
    from tools.nn.model import passport
    rng = np.random.default_rng(seed)
    pool = keys or list(passport.load())
    N = own + enemy
    ks = [list(rng.choice(pool, N)) for _ in range(batch)]
    side = np.array([[1] * own + [2] * enemy] * batch)
    setup = Setup(keys=ks, side=side, bounds=np.tile(np.asarray(bounds, np.float32), (batch, 1)),
                  factions=[("wh_main_emp_empire", "wh2_main_skv_skaven")] * batch, attacker=np.ones(batch, int))
    men0 = setup.men0
    x = np.where(side == 1, -175.0, 175.0) + rng.normal(0, 20, (batch, N))
    z = rng.uniform(-150, 150, (batch, N))
    ms = rng.integers(1, 8, (batch, N)).astype(float)
    state = {
        "t": np.full(batch, 30.0), "x": x, "z": z, "b": np.where(side == 1, 90.0, 270.0) + rng.normal(0, 10, (batch, N)),
        "men": np.floor(men0 * rng.uniform(0.2, 1, (batch, N))), "hp": rng.uniform(0.2, 1, (batch, N)),
        "mp": rng.uniform(-1, 1.5, (batch, N)), "ms": ms, "a": setup.ammo0 * rng.uniform(0, 1, (batch, N)),
        "k": rng.integers(0, 50, (batch, N)).astype(float), "ox": x + rng.normal(0, 30, (batch, N)),
        "oz": z + rng.normal(0, 30, (batch, N)), "target": np.full((batch, N), -1),
    }
    for f in ("r", "s", "w", "m", "mv", "f", "fire", "lf", "rf", "bf"):
        state[f] = rng.random((batch, N)) < 0.3
    state["w"], state["r"], state["s"] = ms == 5, ms >= 6, ms == 7
    return setup, state
