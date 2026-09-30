"""Recorded battles of the arena (build/nn-arena/runs/*/events.jsonl) as arrays: data for training.

One battle -> Battle: times t (s), and per field an array [T, N] in the order of
the run's units (side 1's, then side 2's, as its manifest lists them; the mirror
arena: 7 + 7 = 14), plus the run's arena, own_ai, enemy role and result. Only numpy.

    python -m tools.nn.gamedata                  # list the arena's runs
"""
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tools import config as project

RUNS = project.BUILD / "nn-arena" / "runs"
# The mirror arena (config/nn/arena.json): used when a manifest does not list its units.
SLOTS = ("lord", "spear_1", "spear_2", "spear_3", "spear_4", "archer_1", "archer_2")
NAMES = tuple(f"own_{s}" for s in SLOTS) + tuple(f"enemy_{s}" for s in SLOTS)
FLOAT_FIELDS = ("x", "z", "b", "men", "hp", "mp", "ms", "a", "k", "ox", "oz")
BOOL_FIELDS = ("r", "s", "w", "m", "mv", "f", "fire", "lf", "rf", "bf")
# `fat` is recorded as a string; the array holds its index here (NaN: unknown).
FATIGUE_LEVELS = ("threshold_fresh", "threshold_active", "threshold_winded", "threshold_tired",
                  "threshold_very_tired", "threshold_exhausted")


@dataclass
class Battle:
    run: str
    own_ai: str
    enemy_role: str
    result: dict
    t: np.ndarray
    f: dict = field(default_factory=dict)       # field -> [T, N]; fat: the fatigue state's index
    target: np.ndarray = None                   # [T, N] index of the current target (-1: none)
    arena: str = "arena"                        # config/nn/arena.json, or a name in config/nn/arenas.json
    names: tuple = NAMES                        # [N] script names: side 1's units, then side 2's
    keys: tuple = ()                            # [N] unit keys (main_units)
    side: np.ndarray = None                     # [N] 1 or 2

    @property
    def winner(self):
        return self.result.get("winner", 0)


def load(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["config"]
    listed = cfg.get("units") or {}
    specs = [(1, u) for u in listed.get("own", [])] + [(2, u) for u in listed.get("enemy", [])]
    names = tuple(u["script_name"] for _, u in specs) or NAMES
    keys = tuple(u.get("key", "") for _, u in specs)
    side = np.array([s for s, _ in specs] or [1] * 7 + [2] * 7)
    samples, result = [], {}
    for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if '"nn_sample"' in line or '"nn_final"' in line:
            samples.append(json.loads(line))
        elif '"event":"result"' in line:
            result = json.loads(line)
    index = {n: i for i, n in enumerate(names)}
    T, N = len(samples), len(names)
    t = np.array([s["t"] / 1000 for s in samples])
    arrays = {k: np.full((T, N), np.nan) for k in FLOAT_FIELDS}
    arrays.update({k: np.zeros((T, N), dtype=bool) for k in BOOL_FIELDS})
    arrays["fat"] = np.full((T, N), np.nan)
    fatigue = {name: float(i) for i, name in enumerate(FATIGUE_LEVELS)}
    target = np.full((T, N), -1, dtype=int)
    for ti, s in enumerate(samples):
        for u in s["units"]:
            i = index[u["n"]]
            for k in FLOAT_FIELDS:
                v = u.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    arrays[k][ti, i] = v
            for k in BOOL_FIELDS:
                arrays[k][ti, i] = bool(u.get(k))
            target[ti, i] = index.get(u.get("t") or "", -1)
            arrays["fat"][ti, i] = fatigue.get(u.get("fat"), np.nan)
    return Battle(run=run_dir.name, own_ai=cfg.get("own_ai", "?"), enemy_role=cfg.get("enemy_role", "?"),
                  result=result, t=t, f=arrays, target=target, arena=cfg.get("arena", "arena"),
                  names=names, keys=keys, side=side)


FAIR = 1   # battle_difficulty 1 = Normal: the only difficulty our battles count at (user, 30.09.2026)


def difficulty(run_dir):
    """The battle difficulty a run was played at (launch.json); runs before the launcher set it
    were played at the user's Very Hard (3)."""
    launch = Path(run_dir) / "launch.json"
    if launch.exists():
        text = launch.read_text(encoding="utf-8-sig")
        value = json.loads(text).get("battle_difficulty")
        if value is not None:
            return int(value)
    return 3


def runs(own_ai=None, since=None, fair_only=True, arena=None, root=None):
    """Arena runs with a result, oldest first (optionally only one own_ai, one arena, only from a
    run name on). Only fair runs (Normal difficulty) unless fair_only is false. Runs of the dropped
    battle series (series.json: several battles in one load) are not standard runs and are left out."""
    out = []
    for d in sorted(Path(root or RUNS).glob("*")):
        if since and d.name < since:
            continue
        if not (d / "events.jsonl").exists() or not (d / "manifest.json").exists():
            continue
        if (d / "series.json").exists():
            continue
        if fair_only and difficulty(d) != FAIR:
            continue
        cfg = json.loads((d / "manifest.json").read_text(encoding="utf-8"))["config"]
        if own_ai and cfg.get("own_ai") != own_ai:
            continue
        if arena and cfg.get("arena", "arena") != arena:
            continue
        if '"event":"result"' not in (d / "events.jsonl").read_text(encoding="utf-8"):
            continue
        out.append(d)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for d in runs():
        b = load(d)
        r = b.result
        print(f"{b.run} {b.arena:24} own={b.own_ai:6} enemy={b.enemy_role:6} {r.get('status')} winner={b.winner} "
              f"t={len(b.t)}s men {r.get('side_1_men')}/{r.get('side_2_men')}")


if __name__ == "__main__":
    main()
