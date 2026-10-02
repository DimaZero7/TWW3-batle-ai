"""A unit's innate effects as the network sees them (docs/en/training/model.md "Innate effects"):
per effect of the catalogue (config/nn/effects.json `order`, append-only) two numbers, appended at
the end of the unit's token: `owned` (the unit has it: both sides know it, it is on the unit's card)
and `on` (it acts now). Pairs, so that a new effect appends two columns at the token's end and an
older network loads with zero weights for them (encoder.pad_inputs).

`on` comes from the simulator's bitmask `fx_on` (tools/nn/sim/effects.py) when the state has it. A
recorded battle or the game (the companion) has none: there it is worked out from what the token
already shows - health, the morale state, melee, own morale - by the same predicates; a timed effect
(Strength of the Penitent) whose timer is not known counts as not on. Own units: always; enemies:
while seen (the game lists a seen unit's active effects); the critic's full view: all.
Works on numpy arrays and torch tensors.
"""
import json
from functools import lru_cache

import numpy as np

from tools import config as project

CATALOGUE = project.ROOT / "config" / "nn" / "effects.json"


@lru_cache(maxsize=4)
def load(path=None):
    return json.loads((path or CATALOGUE).read_text(encoding="utf-8"))


def keys(doc=None):
    return list((doc or load())["order"])


SIZE = 2 * len(keys())
NAMES = tuple(f"fx_{k}_{w}" for k in keys() for w in ("owned", "on"))


def owned(unit_keys, doc=None):
    """[n, E] float32: the effects each unit owns (padding "": none)."""
    doc = doc or load()
    order = keys(doc)
    idx = {k: i for i, k in enumerate(order)}
    out = np.zeros((len(unit_keys), len(order)), np.float32)
    for i, u in enumerate(unit_keys):
        for e in (doc["units"].get(u) or ()) if u else ():
            out[i, idx[e]] = 1.0
    return out


def _rows(doc):
    """Per effect (needs, off_when, timed) predicate lists, in order."""
    return [(list(e.get("needs") or ()), list(e.get("off_when") or ()), bool(e.get("timed")))
            for e in (doc["effects"].get(k, {}) for k in keys(doc))]


def observed_predicates(m, state, like):
    """{predicate: [B, N] bool} from observed fields (hp, ms, m, mp); a predicate not known: None."""
    hp = m.nan_to_num(state["hp"] * 1.0, nan=1.0)
    ms = m.nan_to_num(state["ms"] * 1.0, nan=0.0)
    melee = state["m"] > 0.5 if m is np else state["m"].bool()
    mp = state.get("mp")
    mp = None if mp is None else m.nan_to_num(mp * 1.0, nan=1.0)
    return {"in_melee": melee, "out_of_melee": ~melee, "losing_melee": None,
            "morale_below_half": None if mp is None else mp < 0.5, "not_wavering": ms < 5,
            "hp_below_half": hp < 0.5, "hp_below_quarter": hp < 0.25}


def active(m, state, own_bits, doc=None):
    """[B, N, E] bool: the owned effects on now (module doc). own_bits [B, N, E] (0/1)."""
    doc = doc or load()
    E = len(keys(doc))
    have = own_bits > 0.5
    fx = state.get("fx_on")
    if fx is not None:
        bit = np.arange(E) if m is np else m.arange(E, device=fx.device)
        on = ((fx[..., None] >> bit) & 1) > 0
        return on & have
    pred = observed_predicates(m, state, own_bits)
    cols = []
    for needs, off, timed in _rows(doc):
        ok = have[..., len(cols)]
        if timed or any(pred.get(p) is None for p in needs):
            ok = ok & False
        for p in needs:
            if pred.get(p) is not None:
                ok = ok & pred[p]
        for p in off:
            if pred.get(p) is not None:
                ok = ok & ~pred[p]
        cols.append(ok)
    return m.stack(cols, -1) if cols else have[..., :0]


def features(m, state, own_bits, visible, doc=None):
    """[B, N, 2E] float: per effect (owned, on); `on` only where visible [B, N] (own units, seen enemies)."""
    on = active(m, state, own_bits, doc) & visible[..., None]
    on = on.astype(np.float32) if m is np else on.float()
    own = own_bits.astype(np.float32) if m is np else own_bits.float()
    return m.stack([own, on], -1).reshape(*own.shape[:-1], 2 * own.shape[-1])
