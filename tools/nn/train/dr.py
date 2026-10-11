"""Domain randomisation of the simulator's uncertain numbers (config/nn/dr.json; run.py --dr; off by default).

A number the simulator could not take exactly from the game (measured in a few probes, fitted, estimated, or with a
known residual gap) gets a range [lo, hi] around its sim.json value; every training battle draws, at its start, one
value of each such number inside its range, so the network does not learn one possibly-wrong value. The centre never
moves: half the battles draw below it (uniform in [lo, centre]), half above (uniform in [centre, hi]) - the centre is
the median. Both sides of a battle get the same value (a property of the game).

* scalar knobs (tools/nn/sim/knobs.py SCALARS): the drawn value goes into the state as u["k_<name>"] [B, N]; the
  simulator reads it there (knobs.get) and the sim.json number where it is missing (off: nothing is added).
* per-unit knobs ("fields"): the unit field itself (aim_s, quiet_s, ...) of every unit whose value is a listed
  centre takes the battle's draw for that centre.

On top of tools/nn/train/randomise.py (damage, speed, leadership +-15 % and +-5 % a side; start places +-5 m), called
right after it. The network does not see the draws (its input takes the passports from config/nn/units.json).
"""
import json
from dataclasses import dataclass
from pathlib import Path

from tools.nn.sim import knobs
from tools.nn.sim.params import CONFIG

DEFAULT = CONFIG / "dr.json"
TOL = 1e-3          # a unit's field "is" a listed centre within this (the fields are float32)


@dataclass(frozen=True)
class Scalar:
    name: str
    centre: float
    lo: float
    hi: float


@dataclass(frozen=True)
class Field:
    name: str
    field: str
    entries: tuple       # ((centre, lo, hi), ...)


@dataclass(frozen=True)
class Config:
    scalars: tuple
    fields: tuple
    path: str = ""

    def text(self):
        parts = [f"{s.name} {s.centre:g} [{s.lo:g}-{s.hi:g}]" for s in self.scalars]
        parts += [f"{f.field} " + ", ".join(f"{c:g} [{lo:g}-{hi:g}]" for c, lo, hi in f.entries) for f in self.fields]
        return f"domain randomisation ({self.path}): " + "; ".join(parts)


def _check(name, c, lo, hi):
    if not lo <= c <= hi:
        raise ValueError(f"dr {name}: the range [{lo}, {hi}] does not hold the centre {c}")


def load(path=None, params=None):
    """Config from a dr.json (default config/nn/dr.json); scalar centres from params' sim.json (default: load())."""
    if params is None:
        from tools.nn.sim.params import load as load_params
        params = load_params()
    path = Path(path or DEFAULT)
    raw = json.loads(path.read_text(encoding="utf-8"))
    scalars = []
    for name, k in (raw.get("scalars") or {}).items():
        if k.get("off"):
            continue
        if name not in knobs.SCALARS:
            raise ValueError(f"dr: unknown scalar knob {name!r} (tools/nn/sim/knobs.py SCALARS)")
        c = knobs.centre(params.sim, name)
        _check(name, c, float(k["lo"]), float(k["hi"]))
        scalars.append(Scalar(name, c, float(k["lo"]), float(k["hi"])))
    fields = []
    for name, k in (raw.get("fields") or {}).items():
        if k.get("off"):
            continue
        entries = tuple((float(c), float(lo), float(hi)) for c, lo, hi in k["by_value"])
        for c, lo, hi in entries:
            _check(f"{name} {c}", c, lo, hi)
        fields.append(Field(name, k["field"], entries))
    return Config(tuple(scalars), tuple(fields), str(path))


def split(c, lo, hi, q):
    """The value at quantile q (tensor, 0-1): below the centre for q < 0.5 (uniform in [lo, c]), above it otherwise
    (uniform in [c, hi]); the centre is the median."""
    import torch
    return torch.where(q < 0.5, c + (lo - c) * (1 - 2 * q), c + (hi - c) * (2 * q - 1))


def apply(st, rows, cfg, generator=None):
    """Draw every knob of cfg for the battles where rows [B] is true, in place; call right after they (re)start
    (after randomise.apply). -> {knob name (a field: "field@centre"): [B] values drawn (the old where rows is false)}."""
    import torch
    u = st.u
    B, N = u["side"].shape
    out = {}
    for s in cfg.scalars:
        q = torch.rand(B, device=st.device, generator=generator)
        v = split(s.centre, s.lo, s.hi, q)
        key = knobs.PREFIX + s.name
        old = u.get(key)
        if old is None:
            old = torch.full((B, N), s.centre, device=st.device)
        u[key] = torch.where(rows[:, None], v[:, None].expand(B, N), old)
        out[s.name] = u[key][:, 0]
    for f in cfg.fields:
        x = u[f.field]
        # who holds which centre - from the field before any entry changes it (a draw may land on another centre)
        masks = [(x - c).abs() < TOL for c, _, _ in f.entries]
        new = x
        for (c, lo, hi), m in zip(f.entries, masks):
            q = torch.rand(B, device=st.device, generator=generator)
            v = split(c, lo, hi, q)
            new = torch.where(m & rows[:, None], v[:, None].to(x.dtype).expand(B, N), new)
            out[f"{f.field}@{c:g}"] = torch.where(rows, v, torch.full_like(v, c))
        u[f.field] = new
    return out
