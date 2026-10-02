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
# For the army collapse and morale rules (recorded from 02.10.2026; NaN in older runs and where the
# game gave nothing), into Battle.f but not part of the simulator's state (FLOAT_FIELDS):
# sv = unit:strategic_value(), pcr / phr = CCO PercentCasualtiesRecently / PercentHpLostRecently
# (men / HP lost in the last 4 s). The battle's balance of power and the morale text: Battle.bop, .mge.
MORALE_FIELDS = ("sv", "pcr", "phr")
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
    bop: np.ndarray = None                      # [T] CCO BattleRoot.BalanceOfPowerPercent (NaN: not recorded)
    bop_side: int = 0                           # the side bop is for (the player's alliance; 0: unknown)
    mge: np.ndarray = None                      # [T, N] CCO MoraleGreatestEffect: localised text or None

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
    arrays.update({k: np.full((T, N), np.nan) for k in MORALE_FIELDS})
    mge = np.full((T, N), None, dtype=object)
    bop = np.full(T, np.nan)
    bop_side = 0
    for ti, s in enumerate(samples):
        if _number(s.get("bop")):
            bop[ti] = s["bop"]
        if s.get("bop_side") in (1, 2):
            bop_side = s["bop_side"]
        for u in s["units"]:
            i = index[u["n"]]
            for k in MORALE_FIELDS:
                if _number(u.get(k)):
                    arrays[k][ti, i] = u[k]
            if isinstance(u.get("mge"), str):
                mge[ti, i] = u["mge"]
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
                  names=names, keys=keys, side=side, bop=bop, bop_side=bop_side, mge=mge)


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def balance(b, side):
    """[T] the balance of power for `side` (1 or 2): Battle.bop as recorded for bop_side, else its
    complement (the bar's two parts taken to add up to 1, or to 100 when recorded in percent).
    All NaN when not recorded."""
    if b.bop is None or b.bop_side not in (1, 2):
        return np.full(len(b.t), np.nan)
    if side == b.bop_side:
        return b.bop.copy()
    top = 100.0 if np.nanmax(np.append(b.bop, 0.0)) > 1.0 else 1.0
    return top - b.bop


def morale_coverage(b):
    """Share of the recorded seconds (unit-seconds) where each of the collapse and morale values
    was read: {bop, sv, pcr, phr, mge}. 0 for a run recorded before them."""
    def share(a):
        return float(a.mean()) if a is not None and a.size else 0.0
    out = {"bop": share(np.isfinite(b.bop)) if b.bop is not None else 0.0}
    for k in MORALE_FIELDS:
        out[k] = share(np.isfinite(b.f[k])) if k in b.f else 0.0
    out["mge"] = share(b.mge != None) if b.mge is not None else 0.0  # noqa: E711
    return out


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
