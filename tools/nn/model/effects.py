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
    """[n, E] float32: the effects each unit owns (padding "": none); for a contact effect (kind contact: poison, the
    censer's contagion, suppressing fire) the units whose weapons lay it (the catalogue's inflicts)."""
    doc = doc or load()
    order = keys(doc)
    idx = {k: i for i, k in enumerate(order)}
    out = np.zeros((len(unit_keys), len(order)), np.float32)
    for i, u in enumerate(unit_keys):
        for e in ((doc["units"].get(u) or []) + ((doc.get("inflicts") or {}).get(u) or [])) if u else ():
            out[i, idx[e]] = 1.0
    return out


def contact(doc=None):
    """[E] bool: the catalogue's contact effects (on a unit hit by a weapon that lays it; not owned by the unit)."""
    doc = doc or load()
    return np.array([(doc["effects"].get(k) or {}).get("kind") == "contact" for k in keys(doc)], bool)


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
    """[B, N, E] bool: the owned effects on now (module doc). own_bits [B, N, E] (0/1). A contact effect is on
    wherever the state's bitmask says (the unit hit by it does not own it); without a bitmask it is not known: off
    (the companion reads it from the unit card's phases, exchange.effects_on)."""
    doc = doc or load()
    E = len(keys(doc))
    have = own_bits > 0.5
    hit = contact(doc)
    fx = state.get("fx_on")
    if fx is not None:
        bit = np.arange(E) if m is np else m.arange(E, device=fx.device)
        on = ((fx[..., None] >> bit) & 1) > 0
        mask = (hit if m is np else m.as_tensor(hit, device=fx.device)) | have
        return on & mask
    pred = observed_predicates(m, state, own_bits)
    cols = []
    for needs, off, timed in _rows(doc):
        ok = have[..., len(cols)]
        if timed or hit[len(cols)] or any(pred.get(p) is None for p in needs):
            ok = ok & False
        for p in needs:
            if pred.get(p) is not None:
                ok = ok & pred[p]
        for p in off:
            if pred.get(p) is not None:
                ok = ok & ~pred[p]
        cols.append(ok)
    return m.stack(cols, -1) if cols else have[..., :0]


def features(m, state, own_bits, visible, doc=None, on=None):
    """[B, N, 2E] float: per effect (owned, on); `on` only where visible [B, N] (own units, seen enemies). on: active()
    already worked out (None: here)."""
    on = (active(m, state, own_bits, doc) if on is None else on) & visible[..., None]
    on = on.astype(np.float32) if m is np else on.float()
    own = own_bits.astype(np.float32) if m is np else own_bits.float()
    return m.stack([own, on], -1).reshape(*own.shape[:-1], 2 * own.shape[-1])


# What the effects on now do, as numbers (the token's FX_STATS, observation.py; 23.10.2026): per (stat, how) of the
# catalogue's effects' stats - the same stats and scales as an ability's passport (abilities.STATS): a multiplier as
# value - 1 (the effects on multiply), an addition / its scale (they add). The stats an effect lays on the unit itself:
# `on` self (an innate effect of the unit: Frenzy, Strength in Numbers, the owner's Hold the Line) or hit (a contact
# effect on the unit: poison, contagion, suppression); an aura's friends get theirs through RECEIVED (abilities.received:
# the simulator's fx_rx). Fixed and append-only (a new stat of the catalogue: test_nn_fx_stats_input fails until it
# is appended here).
FX_STATS = (("scalar_speed", "mult"), ("stat_melee_attack", "add"), ("stat_melee_defence", "add"),
            ("stat_melee_damage_base", "mult"), ("stat_melee_damage_ap", "mult"), ("stat_charge_bonus", "mult"),
            ("stat_morale", "add"), ("stat_resistance_physical", "add"), ("scalar_missile_damage_base", "mult"),
            ("scalar_missile_damage_ap", "mult"))
FX_STAT_NAMES = tuple(f"fxs_{s}_{h}" for s, h in FX_STATS)
# Then (25.10.2026, line g43; the token's FX_STATS2 after FIGHT: older checkpoints load with zero weights) the stats the
# new units' effects lay: the Snare Net's charge speed (on the enemies it slows: their RECEIVED; here only an own one), the
# shield breaker's block chance, Blackpowder Discipline's accuracy and reload skill.
FX_STATS2 = (("scalar_charge_speed", "mult"), ("stat_missile_block_chance", "add"), ("stat_accuracy", "add"),
             ("stat_reloading", "add"))
FX_STAT2_NAMES = tuple(f"fxs_{s}_{h}" for s, h in FX_STATS2)
FX_ALL = FX_STATS + FX_STATS2
FX_MULT = np.array([h == "mult" for _, h in FX_ALL])
_TORCH = {}


@lru_cache(maxsize=1)
def stat_table():
    """[E, len(FX_ALL)] float32: per effect of the catalogue (keys() order) the log of each multiplier and each
    addition / its scale (abilities.STATS) it lays on the unit itself (FX_STATS' note), FX_STATS then FX_STATS2."""
    from tools.nn.model import abilities
    doc = load()
    scale = {(s, h): float(k) for s, h, k in abilities.STATS}
    col = {sh: j for j, sh in enumerate(FX_ALL)}
    out = np.zeros((len(keys(doc)), len(FX_ALL)), np.float32)
    for i, k in enumerate(keys(doc)):
        for e in (doc["effects"].get(k) or {}).get("stats") or ():
            on = set(e.get("on") or ())
            j = col.get((e["stat"], e["how"]))
            if j is None or not on & {"self", "hit"}:
                continue
            v = float(e["value"])
            out[i, j] += np.log(max(v, 1e-6)) if e["how"] == "mult" else v / scale[(e["stat"], e["how"])]
    return out


def stats_on(m, on):
    """[B, N, len(FX_ALL)] float: the stats of the effects on now (on [B, N, E] bool, active() and the visibility
    already applied): multipliers multiply (value - 1), additions add (/ scale). numpy or torch."""
    if m is np:
        tab, mult = stat_table(), FX_MULT
        val = on.astype(np.float32) @ tab
    else:
        key = str(on.device)
        if key not in _TORCH:
            _TORCH[key] = (m.as_tensor(stat_table(), device=on.device), m.as_tensor(FX_MULT, device=on.device))
        tab, mult = _TORCH[key]
        val = on.float() @ tab
    return m.where(mult, m.exp(val) - 1, val)
