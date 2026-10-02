"""Lord abilities cast by order or by the game's AI (docs/en/training/simulator.md). Every number comes
from the ability passports (config/nn/abilities.json, the game's database); config/nn/sim.json
`abilities` says which of them the simulator models (`model`), when the game's AI fires an active one
(`triggers`) and near_m. Innate abilities - passive ones and those the game fires by itself (passport
`auto`: Strength of the Penitent) - are innate effects (tools/nn/sim/effects.py): this module leaves
their slots alone (effects.py shows a timed one's timers in its slot).

Who fires an active ability:
* a side played by the game's AI (STATIC `ai`) by a rule (an assumption, not measured): `melee`
  when its owner is in melee, `near` when a standing enemy is within near_m of him, `waver` when a
  friend within the ability's range wavers or routs;
* any unit when ordered (Orders.ability: the slot to use, -1 none; the network's side) - only a
  self-cast ability (passport self_cast: used on the owner, no target to choose), ready and not
  active. A side the network plays should have `ai` false, or its lord also fires by the rule.
It then lasts active_s and is ready again recharge_s after it ends. Switching off (passport
off_when): out_of_melee ends an active one at once (recharge from then).

Effects (the passport's effects the simulator has a number for, SIM_STATS): on the owner himself
(the phase targets self), on his side's units within range_m (targets friends) and on enemies
within range_m (targets enemies). Multipliers multiply, additions add.

Slots: a unit shows up to SLOTS abilities (slot_keys): active ones first, then passives that reach
other units, then the rest, each group by key. The network's input (tools/nn/model/abilities.py)
uses the same slots (passives included: the ability bar shows them). Only json here and in
slot_keys(); the tensors need torch.
"""
try:
    import torch
except ImportError:          # slots_of() is used without torch (params)
    torch = None

SLOTS = 3
TRIGGERS = {"passive": 0, "melee": 1, "near": 2, "waver": 3, "losing": 4, "ready": 5}
AUTO_TRIGGERS = {"losing_melee_combat": "losing", "engaged_in_melee": "melee"}
OFF = ("out_of_melee", "morale_is_lower_than_half_of_base_morale", "morale_is_higher_than_wavering",
       "health_below_50%_base")
GROUPS = ("self", "friends", "enemies")
STATS = ("speed", "charge_speed", "attack", "defence", "damage", "ap", "charge", "leadership", "resist_physical")
MULT = ("speed", "charge_speed", "damage", "ap", "charge")
# (stat, how) of the game's database -> the simulator's stat
SIM_STATS = {("scalar_speed", "mult"): "speed", ("scalar_charge_speed", "mult"): "charge_speed",
             ("stat_melee_attack", "add"): "attack", ("stat_melee_defence", "add"): "defence",
             ("stat_melee_damage_base", "mult"): "damage", ("stat_melee_damage_ap", "mult"): "ap",
             ("stat_charge_bonus", "mult"): "charge", ("stat_morale", "add"): "leadership",
             ("stat_resistance_physical", "add"): "resist_physical"}
HEAD = ("active_s", "recharge_s", "passive", "trigger", "range_m", "self_cast", "modelled", "auto") + tuple(
    f"off_{f}" for f in OFF)
COLS = HEAD + tuple(f"{g}_{s}" for g in GROUPS for s in STATS)
COL = {c: i for i, c in enumerate(COLS)}


def keys(params):
    """Ability keys in table order (the passports, sorted)."""
    return sorted(params.abilities)


def slot_keys(unit_key, units, passports):
    """The unit's abilities in its SLOTS slots ("" = empty): actives, then passives reaching others."""
    own = [k for k in (units.get(unit_key) or {}).get("abilities") or () if k in passports]

    def order(k):
        p = passports[k]
        reach = p["targets"].get("friends") or p["targets"].get("enemies")
        return (bool(p["passive"]), bool(p["passive"]) and not reach, k)
    own = sorted(own, key=order)[:SLOTS]
    return own + [""] * (SLOTS - len(own))


def slots_of(params, unit_key):
    """Indices of the unit's abilities in the table (-1: none), SLOTS of them."""
    names = keys(params)
    return [names.index(k) if k else -1 for k in slot_keys(unit_key, params.units, params.abilities)]


def effects(passport):
    """{(group, stat): value} of the stats the simulator models; multipliers multiplied, additions summed."""
    out = {}
    for e in passport["effects"]:
        stat = SIM_STATS.get((e["stat"], e["how"]))
        if stat is None:
            continue
        for g in e["on"]:
            k = (g, stat)
            if stat in MULT:
                out[k] = out.get(k, 1.0) * float(e["value"])
            else:
                out[k] = out.get(k, 0.0) + float(e["value"])
    return out


def row(params, key):
    """One ability's row of the table (COLS)."""
    p = params.abilities[key]
    cal = params.sim["abilities"]
    auto = bool(p.get("auto"))
    if p["passive"]:
        trigger = "passive"
    elif auto:
        trigger = next((AUTO_TRIGGERS[c] for c in p.get("auto_when") or () if c in AUTO_TRIGGERS), "ready")
    else:
        trigger = cal["triggers"].get(key, cal["default_trigger"])
    eff = effects(p)
    out = [float(p["active_s"]), float(p["recharge_s"]), float(p["passive"]), float(TRIGGERS[trigger]),
           float(p["range_m"]), float(p["self_cast"]) * float(not auto), float(key in cal["model"]), float(auto)]
    out += [float(f in (p.get("off_when") or ())) for f in OFF]
    for g in GROUPS:
        for s in STATS:
            out.append(eff.get((g, s), 1.0 if s in MULT else 0.0))
    return out


def table(params, device):
    """[A + 1, COLS]: every ability; the last row (index -1) is no ability."""
    rows = [row(params, k) for k in keys(params)]
    empty = [0.0] * len(HEAD) + [1.0 if s in MULT else 0.0 for _ in GROUPS for s in STATS]
    return torch.tensor(rows + [empty], device=device)


def apply(u, params, dt, standing, engaged, dist, same_side, use=None):
    """Advance the timers, fire abilities (the game's AI by its rule, the network by `use` [B, N]:
    the slot to fire, -1 none) and lay their effects on u (in place). Returns the fields it changed
    with their old values, for restore()."""
    T = table(params, u["men"].device)
    near_m = float(params.sim["abilities"]["near_m"])
    eye = torch.eye(u["men"].shape[1], dtype=torch.bool, device=u["men"].device)[None]
    present = u["side"] > 0
    foe_near = (~same_side & standing[:, None, :] & (dist <= near_m)).any(2)
    shaky = (u["w"] | u["r"]) & (u["men"] > 0)
    # losing the melee: HP taken / dealt recently at the morale rule's "losing" ratio (morale.combat_points)
    lose_ratio = float(params.sim["morale"]["combat_ratio"]["slightly"])
    losing = engaged & ((u["taken"] + 1.0) >= lose_ratio * (u["dealt"] + 1.0))
    # conditions that hold an effect off (OFF)
    L = u["leadership"].clamp(min=1)
    off_now = {"out_of_melee": ~engaged, "morale_is_lower_than_half_of_base_morale": u["morale"] < 0.5 * L,
               "morale_is_higher_than_wavering": ~(u["w"] | u["r"]), "health_below_50%_base": u["hp"] < 0.5}
    zero = torch.zeros_like(u["men"])
    log = {s: zero.clone() for s in MULT}
    add = {s: zero.clone() for s in STATS if s not in MULT}
    use = torch.full_like(u["ab0"], -1) if use is None else use
    for k in range(SLOTS):
        r = T[u[f"ab{k}"]]                          # [B, N, COLS]; -1 picks the empty last row
        c = lambda name: r[..., COL[name]]
        passive, trig, rng = c("passive") > 0, c("trigger"), c("range_m")
        # innate (passive or fired by the game itself): tools/nn/sim/effects.py
        has = (u[f"ab{k}"] >= 0) & ~passive & ~(c("auto") > 0)
        on = (u[f"ab{k}_on"] - dt).clamp(min=0)
        cd = (u[f"ab{k}_cd"] - dt).clamp(min=0)
        friends = same_side & ~eye & present[:, None, :] & (dist <= rng[:, :, None])     # owner i -> unit j
        enemies = ~same_side & present[:, None, :] & (rng[:, :, None] > 0) & (dist <= rng[:, :, None])
        friend_shaky = (friends & shaky[:, None, :]).any(2)
        want = torch.where(trig == 1, engaged, torch.where(trig == 2, foe_near, torch.where(
            trig == 3, friend_shaky, torch.where(trig == 4, losing, trig == 5))))
        off = torch.zeros_like(engaged)
        for f in OFF:
            off = off | ((c(f"off_{f}") > 0) & off_now[f])
        ready = standing & has & ~passive & (on <= 0) & (cd <= 0) & ~off
        ordered = (use == k) & (c("self_cast") > 0)
        fire = ready & ((u["ai"] & want) | ordered)
        on = torch.where(fire, c("active_s"), on)
        cd = torch.where(fire, c("active_s") + c("recharge_s"), cd)
        # switched off while active: it ends now and recharges from now
        ended = (on > 0) & off
        cd = torch.where(ended, torch.minimum(cd, c("recharge_s").clamp(min=0)), cd)
        on = torch.where(ended, torch.zeros_like(on), on)
        u[f"ab{k}_on"] = torch.where(has, on, u[f"ab{k}_on"])
        u[f"ab{k}_cd"] = torch.where(has, cd, u[f"ab{k}_cd"])
        active = has & standing & (c("modelled") > 0) & (on > 0)
        reach = {"self": eye & active[:, :, None], "friends": friends & active[:, :, None],
                 "enemies": enemies & active[:, :, None]}
        for g in GROUPS:
            give = reach[g].float()
            for s in STATS:
                v = c(f"{g}_{s}")
                if s in MULT:
                    log[s] = log[s] + torch.einsum("bij,bi->bj", give, torch.log(v.clamp(min=1e-6)))
                else:
                    add[s] = add[s] + torch.einsum("bij,bi->bj", give, v)
    old = {k: u[k] for k in ("walk", "run", "charge_speed", "damage", "ap_damage", "charge_bonus", "attack",
                              "defence", "morale_bonus", "resist_physical")}
    speed = torch.exp(log["speed"])
    # scalar_charge_speed, when an ability gives it, sets the charge speed; else scalar_speed scales it too
    charge_speed = torch.exp(torch.where(log["charge_speed"].abs() > 1e-9, log["charge_speed"], log["speed"]))
    u["walk"], u["run"] = u["walk"] * speed, u["run"] * speed
    u["charge_speed"] = u["charge_speed"] * charge_speed
    u["damage"] = u["damage"] * torch.exp(log["damage"])
    u["ap_damage"] = u["ap_damage"] * torch.exp(log["ap"])
    u["charge_bonus"] = u["charge_bonus"] * torch.exp(log["charge"])
    u["attack"] = u["attack"] + add["attack"]
    u["defence"] = u["defence"] + add["defence"]
    u["morale_bonus"] = u["morale_bonus"] + add["leadership"]
    # resistances add up to a 90 % cap (docs/en/game/mechanics/missiles.md)
    u["resist_physical"] = (u["resist_physical"] + add["resist_physical"] / 100).clamp(max=0.9)
    return old


def restore(u, old):
    u.update(old)


def set_rule(u, by_rule):
    """Who fires abilities by the game-AI rule, per battle and side: by_rule [B, 2] bool (side 1,
    side 2). A side a network plays must be False (it fires by order), whichever side it is:
    scenario.build sets `ai` for side 2 by default. Call it again after battles restart from a bank
    (their rows bring the bank's `ai`). In place."""
    side = u["side"]
    rule = by_rule.to(torch.bool)
    u["ai"] = ((side == 1) & rule[:, 0:1]) | ((side == 2) & rule[:, 1:2])


def timers(u):
    """The abilities' state as the observation reads it: {"ab{k}_on", "ab{k}_cd"} [B, N] (s)."""
    return {f"ab{k}_{t}": u[f"ab{k}_{t}"] for k in range(SLOTS) for t in ("on", "cd")}

