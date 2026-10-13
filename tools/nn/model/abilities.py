"""Static features of an ability from its passport (config/nn/abilities.json): what both sides know
before battle (a unit's abilities are on its card, the enemy's too). Only numpy.

An ability is described by what it does, never by its key: times, reach, targets and its effects on
allies (the owner and his friends) and on enemies, one number per stat of the game's database
(STATS) and a flag per attribute it gives (ATTRIBUTES). A stat or attribute not listed counts for
nothing; a new ability with listed stats needs only its passport. Numbers are scaled to about -1..1:
multipliers as value - 1, additions divided by a typical size.

Slots (layout()): SLOTS per unit, every ability of the unit, none dropped: the actives and the game-fired ones (with
timers) in the first ACTIVE slots, in the simulator's order (tools/nn/sim/abilities.py ordered(): the network's slot k
< ACTIVE is the simulator's slot k), the passives (auras first) in the rest. The ability head chooses among the first
ACTIVE only; a passive is never pressed.

What a slot carries (observation.py): its state, STATE numbers a unit and decision (Obs.abil), and the ability's index
into the passports' table (keys() order, Obs.abil_idx; -1 none). The passport (STATIC numbers) is not stored per unit
and step: full() puts it back from table() / torch_table() on the device just before the networks read it. The full
features of a slot: DYNAMIC, STATIC, DYNAMIC_NEW (new columns last: an older network loads with zero weights for them).
"""
import json
from functools import lru_cache

import numpy as np

from tools import config as project
from tools.nn.model import passport as unit_passport
from tools.nn.sim.abilities import SLOTS as SIM_SLOTS
from tools.nn.sim.abilities import ordered

PASSPORTS = project.ROOT / "config" / "nn" / "abilities.json"

# (stat, how, scale): how = mult -> value - 1 (multiplied over effects); add -> sum / scale
STATS = (
    ("scalar_speed", "mult", 1), ("scalar_charge_speed", "mult", 1),
    ("stat_melee_attack", "add", 50), ("stat_melee_attack", "mult", 1),
    ("stat_melee_defence", "add", 50), ("stat_melee_defence", "mult", 1),
    ("stat_melee_damage_base", "mult", 1), ("stat_melee_damage_ap", "mult", 1),
    ("stat_charge_bonus", "mult", 1), ("stat_charge_bonus", "add", 50),
    ("stat_morale", "add", 50), ("stat_armour", "add", 100),
    ("stat_resistance_all", "add", 100), ("stat_resistance_physical", "add", 100),
    ("stat_resistance_missile", "add", 100), ("stat_resistance_magic", "add", 100),
    ("stat_resistance_flame", "add", 100), ("stat_weakness_flame", "add", 100),
    ("scalar_missile_damage_base", "mult", 1), ("scalar_missile_damage_ap", "mult", 1),
    ("stat_reloading", "add", 100), ("stat_accuracy", "add", 100), ("scalar_missile_range", "mult", 1),
    ("stat_missile_block_chance", "add", 100), ("stat_bonus_vs_large", "add", 50),
    ("stat_bonus_vs_infantry", "add", 50), ("stat_mass", "mult", 1), ("stat_damage_reflection", "add", 100),
)
ATTRIBUTES = ("immune_to_psychology", "unbreakable", "rampage", "stalk", "causes_terror", "causes_fear",
              "unspottable", "charge_defense", "cannot_die", "silenced", "strider", "invulnerable",
              "fatigue_immune", "devastating_flanker", "charge_reflection")
GROUPS = ("allies", "enemies")     # allies: the owner and his friends (phase targets self or friends)
HEAD = ("passive", "active_s", "recharge_s", "uses_limited", "uses", "range_m", "self_cast", "targets_own",
        "friendly_all", "friendly_n", "enemy_all", "enemy_n", "target_self", "target_friends", "target_enemies")
# When the game fires it and when it is off (02.10.2026; appended last so that older checkpoints load
# with zero weights for them, encoder.py): auto = a timed passive the game fires by itself (never by
# order), its recharge context (recharge_when: its recharge runs only while this holds; the column names
# when_* are kept so that trained networks keep their weights) and the conditions that hold it off
# (off_when), from the database.
WHEN = (("auto", None), ("when_losing_melee", "losing_melee_combat"), ("when_in_melee", "engaged_in_melee"),
        ("off_out_of_melee", "out_of_melee"), ("off_morale_below_half", "morale_is_lower_than_half_of_base_morale"),
        ("off_not_wavering", "morale_is_higher_than_wavering"), ("off_health_below_half", "health_below_50%_base"),
        ("other_condition", None))
# Then (13.10.2026, appended last: older checkpoints load with zero weights for them) WHEN2 - the condition that turns
# a passive on, not among WHEN (Wounded / single_entity: its recharge runs only below 25 % health, then it is on), and
# VIGOUR - the vigour it gives its owner a second while active: -fatigue_change_ratio x 100 (the database's
# special_ability_phases; abilities.json vigour_per_s): + restores vigour (Foe-Seeker +1: 1 % of the maximum fatigue
# a second taken off, as the simulator's fatigue.py does), - tires.
WHEN2 = (("when_health_below_quarter", "health_below_25%"),)
VIGOUR = ("vigour",)
STATIC_NAMES = HEAD + tuple(f"{g}_{s}_{h}" for g in GROUPS for s, h, _ in STATS) + tuple(
    f"{g}_{a}" for g in GROUPS for a in ATTRIBUTES) + tuple(n for n, _ in WHEN) + tuple(n for n, _ in WHEN2) + VIGOUR
STATIC = len(STATIC_NAMES)
# The state of a slot in battle (tools/nn/model/observation.py): owned, ready to use, seconds until
# ready / 60, seconds active left / 30, active now (its timer runs: an active or game-fired one). Enemies: owned and
# active now (while seen). Then DYNAMIC_NEW (13.10.2026, the full features' last column): on now - its effect acts now
# (an active or game-fired one: active now; a passive: its condition holds, the innate effect's `on`,
# tools/nn/model/effects.py: Wounded once the unit fell below 25 %), own units and seen enemies.
DYNAMIC_NAMES = ("owned", "ready", "recharge_left", "active_left", "active_now")
DYNAMIC_NEW = ("on_now",)
DYNAMIC = len(DYNAMIC_NAMES)
STATE = DYNAMIC + len(DYNAMIC_NEW)            # the numbers a slot carries per unit and decision (Obs.abil)
SIZE = DYNAMIC + STATIC + len(DYNAMIC_NEW)    # the full features of a slot (full())
INDEX = {n: i for i, n in enumerate(DYNAMIC_NAMES + STATIC_NAMES + DYNAMIC_NEW)}
STATE_INDEX = {n: i for i, n in enumerate(DYNAMIC_NAMES + DYNAMIC_NEW)}
OLD_SIZE = SIZE - len(WHEN2) - len(VIGOUR) - len(DYNAMIC_NEW)    # the full features before 13.10.2026
RECHARGE, ACTIVE_S = 60.0, 30.0
# The slots: SLOTS a unit, the first ACTIVE for the actives and game-fired ones (the head's choices), the rest for
# the passives. OLD_SLOTS: the slots a network saw before 13.10.2026 (the simulator's first 3; encoder.py).
SLOTS, ACTIVE, OLD_SLOTS = 30, 15, 3


def _group(on):
    return [g for g, hit in (("allies", "self" in on or "friends" in on), ("enemies", "enemies" in on)) if hit]


def features(p):
    """The passport of one ability (a dict from abilities.json) as a list of floats."""
    t = p.get("targets") or {}
    uses, friendly, enemy = p.get("uses", -1), p.get("friendly_units", 0), p.get("enemy_units", 0)
    passive = bool(p.get("passive"))
    out = [float(passive), 0.0 if passive else max(p.get("active_s", 0), 0) / 60,
           max(p.get("recharge_s", 0), 0) / 120, float(uses > 0), max(uses, 0) / 5,
           max(p.get("range_m", 0), 0) / 100, float(bool(p.get("self_cast"))), float(bool(p.get("targets_own"))),
           float(friendly == -1), max(friendly, 0) / 10, float(enemy == -1), max(enemy, 0) / 10,
           float(bool(t.get("self"))), float(bool(t.get("friends"))), float(bool(t.get("enemies")))]
    mult = {(g, s, h): 1.0 for g in GROUPS for s, h, _ in STATS if h == "mult"}
    add = {(g, s, h): 0.0 for g in GROUPS for s, h, _ in STATS if h == "add"}
    for e in p.get("effects") or ():
        for g in _group(e.get("on") or ()):
            k = (g, e["stat"], e["how"])
            if k in mult:
                mult[k] *= float(e["value"])
            elif k in add:
                add[k] += float(e["value"])
    for g in GROUPS:
        for s, h, scale in STATS:
            out.append(mult[(g, s, h)] - 1 if h == "mult" else add[(g, s, h)] / scale)
    given = {(g, a["attribute"]) for a in p.get("attributes") or () for g in _group(a.get("on") or ())}
    for g in GROUPS:
        out += [float((g, a) in given) for a in ATTRIBUTES]
    conds = set(p.get("recharge_when") or ()) | set(p.get("off_when") or ())
    known = {c for _, c in WHEN + WHEN2 if c}
    out += [float(bool(p.get("auto")))] + [float(c in conds) for n, c in WHEN[1:-1]] + [float(bool(conds - known))]
    out += [float(c in conds) for _, c in WHEN2]
    out += [-float(p.get("vigour_per_s", 0.0)) * 100]
    return out


assert len(features({})) == STATIC


@lru_cache(maxsize=4)
def load(path=None):
    return json.loads((path or PASSPORTS).read_text(encoding="utf-8"))["abilities"]


def keys(passports=None):
    """The passports' order: the table's rows (the simulator's abilities.keys: sorted)."""
    return sorted(passports or load())


@lru_cache(maxsize=1)
def _table():
    passports = load()
    return np.asarray([features(passports[k]) for k in keys(passports)] + [[0.0] * STATIC], np.float32)


def table(passports=None):
    """[A + 1, STATIC] float32: every passport's features in keys() order; the last row (index -1) is no ability."""
    if passports is None:
        return _table()
    return np.asarray([features(passports[k]) for k in keys(passports)] + [[0.0] * STATIC], np.float32)


_TORCH = {}


def torch_table(device):
    """table() as a tensor on the device (made once per device)."""
    import torch
    key = str(device)
    if key not in _TORCH:
        _TORCH[key] = torch.as_tensor(table(), device=device)
    return _TORCH[key]


def warm(device):
    """The device's tables made now, outside torch.compile (a compiled observation or decision then reads them as
    constants instead of tracing their json and Python loops)."""
    import torch
    device = torch.zeros(0, device=device).device          # "cuda" -> "cuda:0", as a tensor's device reads
    torch_table(device)
    for what in ("allies", "enemies", "mask"):
        _received_torch(what, device)


def full(state, idx, tab=None):
    """The full features [..., SIZE] of slots: their state [..., STATE] (Obs.abil) and the passport of their ability
    idx [...] (-1: none, the table's zero row). numpy or torch (tab: the table on the device; None: torch_table)."""
    if isinstance(state, np.ndarray):
        tab = table() if tab is None else tab
        return np.concatenate([state[..., :DYNAMIC], tab[np.asarray(idx)], state[..., DYNAMIC:]], -1).astype(np.float32)
    import torch
    tab = torch_table(state.device) if tab is None else tab
    return torch.cat([state[..., :DYNAMIC], tab.to(state.dtype)[idx], state[..., DYNAMIC:]], -1)


def legacy(owned):
    """[..., SLOTS] bool: the owned slots a network saw before 13.10.2026 - the first OLD_SLOTS of the unit's
    abilities in the simulator's order (its actives, then its passives). owned [..., SLOTS] bool (numpy or torch)."""
    if isinstance(owned, np.ndarray):
        n_act = owned[..., :ACTIVE].sum(-1, keepdims=True)
        k = np.arange(SLOTS)
        rank = np.where(k < ACTIVE, k + 0 * n_act, k - ACTIVE + n_act)
    else:
        import torch
        n_act = owned[..., :ACTIVE].sum(-1, keepdim=True)
        k = torch.arange(SLOTS, device=owned.device)
        rank = torch.where(k < ACTIVE, k + 0 * n_act, k - ACTIVE + n_act)
    return owned & (rank < OLD_SLOTS)


def layout(unit_key, units=None, passports=None):
    """The unit's SLOTS ability keys ("" = empty): its actives and game-fired ones from slot 0 (the simulator's order),
    its passives from slot ACTIVE. None dropped (more than fit: an error)."""
    units = units or unit_passport.load()
    passports = passports or load()
    own = ordered(unit_key, units, passports)
    act = [k for k in own if not passports[k].get("passive")]
    pas = [k for k in own if passports[k].get("passive")]
    if len(act) > ACTIVE or len(pas) > SLOTS - ACTIVE:
        raise ValueError(f"{unit_key}: {len(act)} actives, {len(pas)} passives: the slots hold {ACTIVE} and "
                         f"{SLOTS - ACTIVE} (tools/nn/model/abilities.py)")
    return act + [""] * (ACTIVE - len(act)) + pas + [""] * (SLOTS - ACTIVE - len(pas))


def slots(unit_keys, units=None, passports=None):
    """For units by key: (idx [n, SLOTS] int64 - the ability's row in table(), -1 none; usable [n, SLOTS] bool - an
    active self-cast ability the network may order; fx [n, SLOTS] int64 - a passive's innate effect in
    config/nn/effects.json order (its `on`: tools/nn/model/effects.py), -1 none). Padding ("") and units without
    abilities give none."""
    from tools.nn.model import effects
    units = units or unit_passport.load()
    passports = passports or load()
    rows = {k: i for i, k in enumerate(keys(passports))}
    fx_order = {k: i for i, k in enumerate(effects.keys())}
    n = len(unit_keys)
    idx = np.full((n, SLOTS), -1, np.int64)
    usable = np.zeros((n, SLOTS), bool)
    fx = np.full((n, SLOTS), -1, np.int64)
    for i, key in enumerate(unit_keys):
        if not key:
            continue
        for k, a in enumerate(layout(key, units, passports)):
            if a:
                p = passports[a]
                idx[i, k] = rows[a]
                usable[i, k] = bool(p.get("self_cast")) and not p.get("passive")
                if p.get("passive"):
                    fx[i, k] = fx_order.get(a, -1)
    return idx, usable, fx


def sim_fits(idx):
    """Whether every unit's timed abilities fit the simulator's SIM_SLOTS timer slots (idx [..., SLOTS]): a setup the
    simulator plays must (the compiled networks read its first SIM_SLOTS active slots only, encoder.py)."""
    return not (np.asarray(idx)[..., SIM_SLOTS:ACTIVE] >= 0).any()


MULT_MASK = np.array([h == "mult" for _, h, _ in STATS])
RECEIVED_NAMES = tuple(f"rx_{s}_{h}" for s, h, _ in STATS)


def received(rx, side, idx):
    """[..., len(STATS)] the stat bonuses other units' abilities lay on each unit now (the token's RECEIVED,
    observation.py), scaled as features(): multipliers multiply (value - 1), additions add (/ scale). rx [B, N] int64:
    bitmask over keys() of the abilities of others on the unit (the simulator's ab_rx; the companion's from the unit
    card); each counts its passport's allies group when a unit of the unit's side owns it, else its enemies group (an
    enemy's on it). side [B, N]; idx [B, N, SLOTS] (Setup.abil_idx). numpy or torch."""
    if isinstance(rx, np.ndarray):
        m, dev = np, None
        allies, enemies = _received_table("allies"), _received_table("enemies")
        A = allies.shape[0]
        bit = np.arange(A)
        owned = (idx[..., None] == bit).any(-2)                                           # [B, N, A]
        same = lambda s: (owned & (side == s)[..., None]).any(-2, keepdims=True)
    else:
        import torch
        m, dev = torch, rx.device
        allies, enemies = _received_torch("allies", dev), _received_torch("enemies", dev)
        A = allies.shape[0]
        bit = torch.arange(A, device=dev)
        owned = (idx[..., None] == bit).any(-2)
        same = lambda s: (owned & (side == s)[..., None]).any(-2, keepdim=True)
    on = ((rx[..., None] >> bit) & 1) > 0                                                  # [B, N, A]
    friend = ((side == 1)[..., None] & same(1)) | ((side == 2)[..., None] & same(2))
    f32 = (lambda a: a.astype(np.float32)) if m is np else (lambda a: a.float())
    val = f32(on & friend) @ allies + f32(on & ~friend) @ enemies                        # [B, N, S]
    mask = MULT_MASK if m is np else _received_torch("mask", dev)
    return f32(m.where(mask, m.exp(val) - 1, val))


@lru_cache(maxsize=2)
def _received_table(group):
    """[A, S] float32 of the passports' group (allies / enemies): the log of each multiplier, each addition / scale."""
    passports = load()
    out = np.zeros((len(passports), len(STATS)), np.float32)
    for a, k in enumerate(keys(passports)):
        f = features(passports[k])
        for j, (s, h, _) in enumerate(STATS):
            v = f[INDEX[f"{group}_{s}_{h}"] - DYNAMIC]
            out[a, j] = np.log(max(v + 1.0, 1e-6)) if h == "mult" else v
    return out


def _received_torch(what, device):
    import torch
    key = (what, str(device))
    if key not in _TORCH:
        _TORCH[key] = torch.as_tensor(MULT_MASK if what == "mask" else _received_table(what), device=device)
    return _TORCH[key]
