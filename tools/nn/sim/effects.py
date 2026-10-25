"""Innate unit effects: one mechanism for every attribute and passive or game-fired ability
(docs/en/training/simulator.md "Innate effects"). The catalogue is config/nn/effects.json
(python -m tools.nn.effects, from the game's database); nothing here knows a unit or an effect by
its key.

Each unit owns a set of effects (STATIC `fx`, a bitmask over the catalogue's order). Every step,
for every owned effect the catalogue marks `modelled` (and config/nn/sim.json effects.off does not
leave out):
* its conditions are evaluated (PREDICATES, from the unit's state at the step's start): it is on
  while all its `needs` hold and none of its `off_when` does; an attribute is always on;
* a timed one (Strength of the Penitent) fires by itself as soon as it is ready and no off_when
  holds (in melee: the recordings, 53 of 54 first fires at the contact, without losses too), lasts
  active_s, ends at once while an off_when holds; after a fire its recharge_s runs only while its
  `recharge_needs` hold (the database's recharge context losing_melee_combat, CA hotfix 6.2.2
  "recharges when losing"; as the game acts on it: not winning its melee - out of melee too - by the
  morale's fight balance, u cmb > 0 winning: the probe T-E1 - winning flagellants stall 70 s, losing
  and even ones recharge in exactly 3 s; the recordings - 249 of 270 gaps in a continuous melee
  exactly 3 s, the 21 long ones while dealing more than taking, and after a fire ended by leaving
  melee the next one exactly 3 s later out of melee too, 17 of 17 with 1.5-3.5 s out); the initial
  recharge (initial_s) runs always
  (timers fxt{j}_on / fxt{j}_cd / fxt{j}_used for the unit's timed effects, STATIC fxt{j}: which);
* while on, its stat modifiers lie on the owner (multipliers multiply, additions add) and, for an
  aura (range_m > 0, stats on friends), on the friends within range_m of a living owner - routing
  too, and on routing friends (Hold the Line from a routing General: 1312 of 1312 recorded s, on
  routing friends 2017 of 2017, build/effects/spec.md 3.4); its rules (RULES: unbreakable,
  expendable, ...) set the unit's rule flags for the step.
An effect lies on a unit that is alive (routing too: Scurry Away! speeds a rout); a timed one fires
only for a standing unit. apply() returns the old values; restore() puts them back at the step's end.
`fx_on` (INTERNAL) keeps the bitmask of the effects on in the last step (the network's input): an
effect the simulator does not act on (Hide (forest): no woods) still counts as on while its
conditions hold, as in the game.

The ability bar (ab{k}_on / ab{k}_cd, tools/nn/sim/abilities.py) of a slot holding a timed effect's
ability shows that effect's timers, as before this module (the observation reads them).

Contact effects (the catalogue's kind "contact": poison, the censer's contagion, suppressing fire): a unit
does not own them; a weapon lays one on the enemy it hits (STATIC mfx: its blows' effect, sfx: its
projectiles', indices in the order, -1 none). lay() keeps a timer per contact effect of the catalogue
(INTERNAL cfx{j}_s, j over contact_keys): a step whose hits of such a weapon on a unit land (at least
one: 1 - exp(-expected hits), a hash draw as morale's) starts it at the effect's duration_s (10 s, the
database), each hit again; friendly fire lays none; different contact effects stack, the same one
refreshes (CA, Update 8.1). While its timer runs it lies on the unit like an own effect (apply), and
fx_on shows it.

Only json in the table builders; the tensors need torch.
"""
try:
    import torch
except ImportError:          # the table's layout is importable without torch
    torch = None

from tools.nn.sim.abilities import MULT, SIM_STATS, SLOTS, STATS

# The effects' stats: the abilities' and the missile damage (poison: x0.85, scalar_missile_damage_base / _ap); since
# 25.10.2026 (line g43) the shield's block chance (the shield breaker's -24: percentage points off the chance, at least 0 -
# how the game adds it is not described: an assumption), the accuracy (Blackpowder Discipline's +40, added to land_units
# accuracy + marksmanship, at most 100) and the reload skill (its +10: the reload is max(4 s, base x (100 - skill) %), the
# game's rule, base the database's reload_s; the measured cycle scales with it)
E_STATS = STATS + ("m_damage", "m_ap", "shield", "accuracy", "reloading")
E_MULT = MULT + ("m_damage", "m_ap")
E_SIM_STATS = {**SIM_STATS, ("scalar_missile_damage_base", "mult"): "m_damage",
               ("scalar_missile_damage_ap", "mult"): "m_ap", ("stat_missile_block_chance", "add"): "shield",
               ("stat_accuracy", "add"): "accuracy", ("stat_reloading", "add"): "reloading"}
RELOAD_MIN_S = 4.0  # the game's floor of a reload cut by the reload skill (max(4 s, base x (100 - skill) %))
CONTACT_SLOTS = 4   # contact effects the catalogue may hold (INTERNAL cfx0_s .. cfx3_s)

PREDICATES =("in_melee", "out_of_melee", "losing_melee", "morale_below_half", "not_wavering", "hp_below_half",
              "hp_below_quarter", "moving", "climbing")
# The catalogue's rules the simulator acts on -> the unit field that carries the flag for the step.
RULES = {"unbreakable": "unbreakable", "expendable": "expendable", "encourages": "encourages",
         "charge_reflection": "reflect", "fire_while_moving": "fire_move", "fatigue_immune": "fatigue_immune",
         "stalk": "stalk"}
TIMERS = 2          # timed effects a unit can carry (fxt0, fxt1)
MAX_EFFECTS = 62    # the bitmask is an int64
# self: the owner; friends: an aura's (range_m); enemies (25.10.2026, the Snare Net): the enemies whose centre is within
# range_m of a living owner's - one owner's or several (the same effect does not stack: an assumption, not described),
# its multipliers ADDED to the unit's other multipliers of the stat (x0.9 with poison's x0.85: x0.75; the game adds its
# stat multipliers), scalar_speed scaling the charge speed and scalar_charge_speed on top (the net's charge x0.81)
GROUPS = ("self", "friends", "enemies")
# A list to append each step's {effect key: [B, N] bool - on the unit, its own or an owner's aura} to, or None
# (training: off); the probes' simulator twins read it (tools/nn/morale_probe.py).
RECEIVED = None
HEAD = ("modelled", "timed", "active_s", "recharge_s", "range_m", "ability", "contact", "duration_s", "delay_s")
COLS = HEAD + tuple(f"{w}_{p}" for w in ("need", "off", "fire", "rcg") for p in PREDICATES) + tuple(
    f"rule_{r}" for r in RULES) + tuple(f"{g}_{s}" for g in GROUPS for s in E_STATS)
COL = {c: i for i, c in enumerate(COLS)}


def order(params):
    """The catalogue's effect keys in their (append-only) order."""
    return list(params.effects.get("order") or ())


def index(params):
    return {k: i for i, k in enumerate(order(params))}


def modelled(params, key):
    """The simulator acts on the effect: the catalogue says it can (`modelled`) and sim.json effects.off does
    not leave it out (a calibration switch, with its why)."""
    e = (params.effects.get("effects") or {}).get(key) or {}
    return bool(e.get("modelled")) and key not in ((params.sim.get("effects") or {}).get("off") or ())


def contact_keys(params):
    """[(slot j, index in the order, key)] of the catalogue's contact effects the simulator acts on (modelled), in the
    order's order: the j-th has the timer cfx{j}_s."""
    eff = params.effects.get("effects") or {}
    out = [(i, k) for i, k in enumerate(order(params)) if (eff.get(k) or {}).get("kind") == "contact"
           and modelled(params, k)]
    assert len(out) <= CONTACT_SLOTS, f"{len(out)} contact effects: the state holds {CONTACT_SLOTS}"
    return [(j, i, k) for j, (i, k) in enumerate(out)]


def laid_by(params, key):
    """(melee, missile): the indices in the order of the contact effects a unit key's blows and projectiles lay (the
    passport's melee.contact_phase, missile.contact_phase; -1 none or not modelled)."""
    u = params.units.get(key) or {}
    idx = {k: i for _, i, k in contact_keys(params)}
    return (idx.get((u.get("melee") or {}).get("contact_phase"), -1),
            idx.get((u.get("missile") or {}).get("contact_phase"), -1))


def unit_effects(params, key):
    """The effect keys a unit owns: the catalogue's link (generated from the database rows); a unit not
    in it (not yet rebuilt) gets those of its attributes and innate abilities the catalogue knows."""
    links = params.effects.get("units") or {}
    if key in links:
        return list(links[key])
    u = params.units.get(key) or {}
    known = set(order(params))
    own = list(u.get("attributes") or ())
    own += [a for a in u.get("abilities") or () if (params.abilities or {}).get(a, {}).get("passive")
            or (params.abilities or {}).get(a, {}).get("auto")]
    return [k for k in own if k in known]


def static(params, key):
    """STATIC fields of a unit from its effects: `fx` (bitmask), fxt{j} (its timed effects, -1 none) and the
    rule flags of its effects that hold always (attributes and passives without conditions)."""
    idx = index(params)
    assert len(idx) <= MAX_EFFECTS, f"{len(idx)} effects: the bitmask holds {MAX_EFFECTS}"
    own = unit_effects(params, key)
    eff = params.effects.get("effects") or {}
    bits = 0
    timed = []
    rules = {f: False for f in RULES.values()}
    for k in own:
        bits |= 1 << idx[k]
        e = eff.get(k) or {}
        if e.get("timed") and modelled(params, k):
            timed.append(idx[k])
        if modelled(params, k) and not e.get("timed") and not e.get("needs") and not e.get("off_when"):
            for r in e.get("rules") or ():
                if r in RULES:
                    rules[RULES[r]] = True
    assert len(timed) <= TIMERS, f"{key}: {len(timed)} timed effects, the state holds {TIMERS}"
    timed += [-1] * (TIMERS - len(timed))
    mfx, sfx = laid_by(params, key)
    # a delayed effect (delay_s: its first phases lay nothing - Blackpowder Discipline): one a unit, its clock fxd_s
    delayed = [idx[k] for k in own if float((eff.get(k) or {}).get("delay_s") or 0.0) > 0 and modelled(params, k)]
    assert len(delayed) <= 1, f"{key}: {len(delayed)} delayed effects, the state holds 1"
    return {"fx": bits, **{f"fxt{j}": t for j, t in enumerate(timed)}, **rules, "mfx": mfx, "sfx": sfx,
            "fxd": delayed[0] if delayed else -1}


def row(params, key):
    """One effect's row of the table (COLS)."""
    e = params.effects["effects"][key]
    ab_keys = sorted(params.abilities or {})
    t = e.get("timed") or {}
    out = {c: 0.0 for c in COLS}
    contact = e.get("kind") == "contact"
    out.update(modelled=float(modelled(params, key)), timed=float(bool(t)),
               active_s=float(t.get("active_s", 0)), recharge_s=float(max(t.get("recharge_s", 0), 0)),
               range_m=float(e.get("range_m") or 0.0),
               ability=float(ab_keys.index(key)) if key in ab_keys else -1.0,
               contact=float(contact), duration_s=float(e.get("duration_s") or 0.0),
               delay_s=float(e.get("delay_s") or 0.0))
    for w, preds in (("need", e.get("needs") or ()), ("off", e.get("off_when") or ()),
                     ("fire", t.get("fires_when") or ()), ("rcg", t.get("recharge_needs") or ())):
        for p in preds:
            out[f"{w}_{p}"] = 1.0
    for r in e.get("rules") or ():
        if r in RULES:
            out[f"rule_{r}"] = 1.0
    for g in GROUPS:
        for s in E_STATS:
            out[f"{g}_{s}"] = 1.0 if s in E_MULT else 0.0
    # (sim.json effects.stats_off: stats of an effect the game does not show acting - a measurement, with its why)
    stats_off = set(((params.sim.get("effects") or {}).get("stats_off") or {}).get(key) or ())
    for st in e.get("stats") or ():
        s = E_SIM_STATS.get((st["stat"], st["how"]))
        if s is None or st["stat"] in stats_off:
            continue
        # (a contact effect's stats lie on the unit hit: "self" of the unit carrying it)
        # (sim_value: the catalogue's number for the simulator where it is not the database's alone - the Snare Net's
        # charge 0.81 = its speed 0.9 x its charge 0.9, tools/nn/effects.py)
        v = float(st.get("sim_value", st["value"]))
        for g in ("self",) if contact else st.get("on") or ():
            if g in GROUPS:
                k = f"{g}_{s}"
                out[k] = out[k] * v if s in E_MULT else out[k] + v
    return [out[c] for c in COLS]


def table(params, device):
    """[E + 1, COLS]: every effect of the catalogue; the last row (index -1) is none."""
    empty = [0.0] * len(COLS)
    for g in GROUPS:
        for s in E_MULT:
            empty[COL[f"{g}_{s}"]] = 1.0
    empty[COL["ability"]] = -1.0
    # an effect in the order that no unit owns any more (the order is append-only) has an empty row
    rows = [row(params, k) if k in params.effects["effects"] else empty for k in order(params)]
    return torch.tensor(rows + [empty], device=device)


def hexes(params):
    """Indices of the effects that reach enemies (range_m > 0 and a stat on enemies: the Snare Net), modelled."""
    out = []
    for i, k in enumerate(order(params)):
        e = params.effects["effects"].get(k) or {}
        if modelled(params, k) and (e.get("range_m") or 0) > 0 and any(
                "enemies" in (s.get("on") or ()) and s.get("sim") for s in e.get("stats") or ()):
            out.append(i)
    return out


def auras(params):
    """Indices of the effects that reach friends (range_m > 0 and a stat on friends), modelled."""
    out = []
    for i, k in enumerate(order(params)):
        e = params.effects["effects"].get(k) or {}
        if modelled(params, k) and (e.get("range_m") or 0) > 0 and any(
                "friends" in (s.get("on") or ()) and s.get("sim") for s in e.get("stats") or ()):
            out.append(i)
    return out


def predicates(u, engaged, params):
    """{predicate: [B, N] bool} from the unit's state at the step's start."""
    L = u["leadership"].clamp(min=1)
    ratio = float(params.sim["morale"]["combat_ratio"]["slightly"])
    # the database's losing_melee_combat as the game acts on it: not winning its melee, out of melee too (module doc;
    # winning = the morale's fight balance of the last step above 0: u cmb, morale.combat_points - HP dealt / taken
    # recently, the database's ratios and the 10 % gate - one rule of winning and losing)
    winning = (u["cmb"] > 0) if "cmb" in u else ((u["dealt"] + 1.0) >= ratio * (u["taken"] + 1.0))
    return {"in_melee": engaged, "out_of_melee": ~engaged, "losing_melee": ~(engaged & winning),
            "morale_below_half": u["morale"] < 0.5 * L,
            "not_wavering": ~(u["w"] | u["r"]),
            "hp_below_half": u["hp"] < 0.5,
            # below a quarter: once fallen there, for the effect's initial cooldown (Wounds: 5 s, the database's
            # initial_recharge; the recordings: on 5-6 s after, to the end - Update 2.0), never back (u low_s)
            "hp_below_quarter": (u["low_s"] >= quarter_delay(params)) if "low_s" in u else u["hp"] < 0.25,
            # moving: the observed flag (the game's IsMoving); climbing or manning on a platform: never on our maps
            "moving": u["mv"], "climbing": torch.zeros_like(u["mv"])}


def quarter_delay(params):
    """Seconds below a quarter of health before an effect that needs it comes on: the largest initial cooldown
    (abilities.json initial_s) of the effects that need hp_below_quarter (Wounds: 5)."""
    out = 0.0
    for k in order(params):
        e = (params.effects.get("effects") or {}).get(k) or {}
        if "hp_below_quarter" in (e.get("needs") or ()):
            out = max(out, float((params.abilities.get(k) or {}).get("initial_s") or 0.0))
    return out


def owned(u, E):
    """[B, N, E] bool: the effects each unit owns (its bitmask)."""
    bit = torch.arange(E, device=u["fx"].device)
    return ((u["fx"][..., None] >> bit) & 1) > 0


def apply(u, params, dt, standing, engaged, dist, same_side):
    """Advance the timed effects, lay the effects that are on onto u (in place) and set the rule flags.
    Returns the fields it changed with their old values, for restore()."""
    if not params.effects or not order(params):
        return {}
    T = table(params, u["men"].device)
    E = T.shape[0] - 1
    c = lambda name: T[:E, COL[name]]                                   # [E]
    own = owned(u, E) & (u["side"] > 0)[..., None]
    alive = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"]
    pred = predicates(u, engaged, params)
    P = torch.stack([pred[p] for p in PREDICATES], -1)                  # [B, N, Pn]
    need = T[:E, [COL[f"need_{p}"] for p in PREDICATES]] > 0            # [E, Pn]
    offc = T[:E, [COL[f"off_{p}"] for p in PREDICATES]] > 0
    fire = T[:E, [COL[f"fire_{p}"] for p in PREDICATES]] > 0
    rcg = T[:E, [COL[f"rcg_{p}"] for p in PREDICATES]] > 0
    Pf = P.float()
    unmet = torch.einsum("bnp,ep->bne", (~P).float(), need.float()) > 0
    off = torch.einsum("bnp,ep->bne", Pf, offc.float()) > 0
    # on: owned, alive, its conditions hold (whether the simulator acts on it or not: fx_on, the network's
    # input, says what holds in the game); the stats and rules below are those of the modelled ones
    on = own & alive[..., None] & ~unmet & ~off & (c("timed") <= 0)
    # timed effects: the unit's timer slots
    for j in range(TIMERS):
        e = u[f"fxt{j}"]
        has = (e >= 0) & own.gather(2, e.clamp(min=0)[..., None]).squeeze(-1)
        r = T[e]                                                        # [B, N, COLS]; -1: the empty row
        was_on = u[f"fxt{j}_on"] > 0
        used = u[f"fxt{j}_used"] if f"fxt{j}_used" in u else torch.zeros_like(was_on)
        t_on = (u[f"fxt{j}_on"] - dt).clamp(min=0)
        # the cooldown (active time + recharge) runs while it is on, before its first fire (the initial recharge),
        # and after a fire only while its recharge context holds (recharge_needs)
        r_ok = ~((~P) & rcg[e.clamp(min=0)]).any(-1)
        runs = was_on | ~used | r_ok
        t_cd = torch.where(runs, (u[f"fxt{j}_cd"] - dt).clamp(min=0), u[f"fxt{j}_cd"])
        want = ~((~P) & fire[e.clamp(min=0)]).any(-1)                   # fires_when (the database: none)
        e_off = off.gather(2, e.clamp(min=0)[..., None]).squeeze(-1)
        e_unmet = unmet.gather(2, e.clamp(min=0)[..., None]).squeeze(-1)
        ready = standing & has & (r[..., COL["modelled"]] > 0) & (t_on <= 0) & (t_cd <= 0) & ~e_off & ~e_unmet
        go = ready & want
        t_on = torch.where(go, r[..., COL["active_s"]], t_on)
        t_cd = torch.where(go, r[..., COL["active_s"]] + r[..., COL["recharge_s"]], t_cd)
        if f"fxt{j}_used" in u:
            u[f"fxt{j}_used"] = used | go
        # switched off while active: it ends now and recharges from now
        ended = (t_on > 0) & e_off
        t_cd = torch.where(ended, torch.minimum(t_cd, r[..., COL["recharge_s"]]), t_cd)
        t_on = torch.where(ended | ~alive, torch.zeros_like(t_on), t_on)
        u[f"fxt{j}_on"], u[f"fxt{j}_cd"] = t_on, t_cd
        live = has & alive & (t_on > 0)
        on = on | (torch.nn.functional.one_hot(e.clamp(min=0), E + 1)[..., :E].bool() & live[..., None])
        # the ability bar of the slot holding this effect's ability shows its timers
        ab = r[..., COL["ability"]].long()
        for k in range(SLOTS):
            match = has & (ab >= 0) & (u[f"ab{k}"] == ab)
            u[f"ab{k}_on"] = torch.where(match, t_on, u[f"ab{k}_on"])
            u[f"ab{k}_cd"] = torch.where(match, t_cd, u[f"ab{k}_cd"])
    # a delayed effect (delay_s: Blackpowder Discipline's 10 s): on once its conditions have held delay_s without a break
    on = delayed(u, params, T, E, on, dt)
    # contact effects: on while their timer runs (lay(); a living unit)
    on = on | contact_on(u, params, E, alive)
    bit = torch.arange(E, device=u["fx"].device)
    u["fx_on"] = (on.long() << bit).sum(-1)
    on = on & (c("modelled") > 0)
    onf = on.float()
    log = {s: torch.einsum("bne,e->bn", onf, torch.log(c(f"self_{s}").clamp(min=1e-6))) for s in E_MULT}
    add = {s: torch.einsum("bne,e->bn", onf, c(f"self_{s}")) for s in E_STATS if s not in E_MULT}
    eye = torch.eye(u["men"].shape[1], dtype=torch.bool, device=u["men"].device)[None]
    present = u["side"] > 0
    got = {}
    ab_keys = sorted(params.abilities or {})
    rx = torch.zeros_like(u["fx"]) if "fx_rx" in u else None
    for i in auras(params):
        # from a living owner, routing too, to friends in range, routing too (spec 3.4)
        give = (same_side & ~eye & present[:, None, :] & (dist <= T[i, COL["range_m"]])
                & (on[..., i] & alive)[:, :, None]).float()             # owner i -> friend j
        if RECEIVED is not None:
            got[i] = give.sum(1) > 0
        key = order(params)[i]
        if rx is not None and key in ab_keys:
            # the aura's ability on a friend: the network's "effects on me from others" (abilities.py ab_rx)
            rx = rx | ((give.sum(1) > 0).long() << ab_keys.index(key))
        for s in E_STATS:
            v = T[i, COL[f"friends_{s}"]]
            if s in E_MULT:
                log[s] = log[s] + give.sum(1) * torch.log(v.clamp(min=1e-6))
            else:
                add[s] = add[s] + give.sum(1) * v
    # effects on enemies (GROUPS: the Snare Net): on the enemies within range_m of a living owner, once whatever the owners;
    # their multipliers add to the others' (hexm: value - 1 summed)
    hexm = {s: torch.zeros_like(u["men"]) for s in E_MULT}
    for i in hexes(params):
        give = (~same_side & present[:, None, :] & (dist <= T[i, COL["range_m"]])
                & (on[..., i] & alive)[:, :, None])                     # owner i -> enemy j
        got_i = (give.any(1) & present).float()
        if RECEIVED is not None:
            got[i] = got_i > 0
        key = order(params)[i]
        if rx is not None and key in ab_keys:
            rx = rx | ((got_i > 0).long() << ab_keys.index(key))
        for s in E_STATS:
            v = T[i, COL[f"enemies_{s}"]]
            if s in E_MULT:
                hexm[s] = hexm[s] + got_i * (v - 1)
            else:
                add[s] = add[s] + got_i * v
    fields = ("walk", "run", "charge_speed", "damage", "ap_damage", "charge_bonus", "attack", "defence",
              "morale_bonus", "resist_physical", "m_damage", "m_ap") + tuple(RULES.values()) + tuple(
        f for f in ("shield", "acc", "reload") if f in u)
    old = {k: u[k] for k in fields}
    mult = lambda s: (torch.exp(log[s]) + hexm[s]).clamp(min=0.05)
    u["m_damage"] = u["m_damage"] * mult("m_damage")
    u["m_ap"] = u["m_ap"] * mult("m_ap")
    speed = mult("speed")
    # scalar_charge_speed, when an effect gives it, sets the charge speed; else scalar_speed scales it too - an enemy's
    # (the Snare Net: its catalogue charge 0.81, the game's 0.9 x 0.9) as the others, added to them
    own_c = torch.exp(torch.where(log["charge_speed"].abs() > 1e-9, log["charge_speed"], log["speed"]))
    charge_speed = (own_c + torch.where(hexm["charge_speed"].abs() > 1e-9, hexm["charge_speed"],
                                        hexm["speed"])).clamp(min=0.05)
    u["walk"], u["run"] = u["walk"] * speed, u["run"] * speed
    u["charge_speed"] = u["charge_speed"] * charge_speed
    u["damage"] = u["damage"] * mult("damage")
    u["ap_damage"] = u["ap_damage"] * mult("ap")
    u["charge_bonus"] = u["charge_bonus"] * mult("charge")
    # the shield's block chance (the shield breaker: -24 percentage points, at least 0), the accuracy (at most 100) and
    # the reload skill (E_STATS' note): only where an effect gives one (the rest keep their numbers to the bit)
    if "shield" in u:
        u["shield"] = torch.where(add["shield"] != 0, (u["shield"] + add["shield"] / 100).clamp(0, 1), u["shield"])
    if "acc" in u:
        u["acc"] = torch.where(add["accuracy"] != 0, (u["acc"] + add["accuracy"]).clamp(0, 100), u["acc"])
    if "reload" in u and "reload_db" in u:
        base = u["reload_db"]
        cut = (base * (1 - add["reloading"] / 100)).clamp(min=RELOAD_MIN_S) / base.clamp(min=RELOAD_MIN_S)
        u["reload"] = torch.where((add["reloading"] != 0) & (base > 0), u["reload"] * cut, u["reload"])
    u["attack"] = u["attack"] + add["attack"]
    u["defence"] = u["defence"] + add["defence"]
    u["morale_bonus"] = u["morale_bonus"] + add["leadership"]
    # resistances add up to a 90 % cap (docs/en/game/mechanics/missiles.md)
    u["resist_physical"] = (u["resist_physical"] + add["resist_physical"] / 100).clamp(max=0.9)
    for r, f in RULES.items():
        u[f] = torch.einsum("bne,e->bn", onf, c(f"rule_{r}")) > 0
    if rx is not None:
        u["fx_rx"] = rx
    if RECEIVED is not None:
        keys = order(params)
        RECEIVED.append({keys[i]: (on[..., i] | got.get(i, torch.zeros_like(on[..., i]))).clone()
                         for i in range(E) if bool(on[..., i].any()) or i in got})
    return old


def restore(u, old):
    u.update(old)


def refresh_on(u, params):
    """At the step's end: `fx_on` (the network's input) from the state the network observes now - the
    untimed effects by their conditions on the step's final state (in melee = `m`, as the observation
    works them out from a recording; tools/nn/model/effects.py), the timed ones as their timers say.
    apply() acts on the conditions at the step's start; without this the network saw an effect one
    step late whenever a condition changed within the step (a unit wavering, health crossing a half)."""
    if not params.effects or not order(params):
        return
    T = table(params, u["men"].device)
    E = T.shape[0] - 1
    timed = T[:E, COL["timed"]] > 0
    own = owned(u, E) & (u["side"] > 0)[..., None]
    alive = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"]
    pred = predicates(u, u["m"], params)
    P = torch.stack([pred[p] for p in PREDICATES], -1)
    need = T[:E, [COL[f"need_{p}"] for p in PREDICATES]] > 0
    offc = T[:E, [COL[f"off_{p}"] for p in PREDICATES]] > 0
    unmet = torch.einsum("bnp,ep->bne", (~P).float(), need.float()) > 0
    off = torch.einsum("bnp,ep->bne", P.float(), offc.float()) > 0
    bit = torch.arange(E, device=u["fx"].device)
    was = ((u["fx_on"][..., None] >> bit) & 1) > 0
    on = torch.where(timed, was & alive[..., None], own & alive[..., None] & ~unmet & ~off)
    on = delayed(u, params, T, E, on, None)
    on = on | contact_on(u, params, E, alive)
    u["fx_on"] = (on.long() << bit).sum(-1)


def card(u, params):
    """{stat: [B, N]} of the effects on now (fx_on) as the unit card shows them: multipliers for the E_MULT stats,
    additions for the others; their own stats only (no aura given, no ability, no fatigue). The probes' twins
    (tools/nn/unit_probe.py) compare the card's speed, weapon damage, melee attack and leadership with the game's."""
    if not params.effects or not order(params):
        return {}
    T = table(params, u["men"].device)
    E = T.shape[0] - 1
    bit = torch.arange(E, device=u["fx"].device)
    on = ((((u["fx_on"][..., None] >> bit) & 1) > 0) & (T[:E, COL["modelled"]] > 0)).float()
    out = {}
    for s in E_STATS:
        v = T[:E, COL[f"self_{s}"]]
        out[s] = (torch.exp(torch.einsum("bne,e->bn", on, torch.log(v.clamp(min=1e-6)))) if s in E_MULT
                  else torch.einsum("bne,e->bn", on, v))
    return out


def delayed(u, params, T, E, on, dt):
    """on [B, N, E] with the unit's delayed effect (STATIC fxd, the catalogue's delay_s > 0: Blackpowder Discipline) on
    only once its conditions have held delay_s without a break: its clock fxd_s (INTERNAL) runs while they hold (dt:
    advance it, apply(); None: read it, refresh_on()) and is 0 when they break. Without the clock (an older state) a
    delayed effect is off."""
    if "fxd" not in u:
        return on
    d_i = u["fxd"]
    has = d_i >= 0
    pick = torch.nn.functional.one_hot(d_i.clamp(min=0), E + 1)[..., :E].bool() & has[..., None]
    if "fxd_s" not in u:
        return on & ~pick
    cond = (on & pick).any(-1)
    if dt is not None:
        u["fxd_s"] = torch.where(cond, u["fxd_s"] + dt, torch.zeros_like(u["fxd_s"]))
    ready = cond & (u["fxd_s"] >= T[d_i, COL["delay_s"]] - 1e-6)
    return on & ~(pick & ~ready[..., None])


def contact_on(u, params, E, alive):
    """[B, N, E] bool: the contact effects on each living unit now (their timers cfx{j}_s, lay())."""
    on = torch.zeros(u["men"].shape + (E,), dtype=torch.bool, device=u["men"].device)
    for j, i, _ in contact_keys(params):
        if f"cfx{j}_s" in u:
            on[..., i] = (u[f"cfx{j}_s"] > 0) & alive
    return on


def lay(u, params, dt, hp_melee, mhit, hp_missile, shit, enemy, t):
    """The contact effects laid this step (module doc), in place: each timer cfx{j}_s runs down by dt, then starts at
    the effect's duration_s on every unit an enemy weapon of it hit this step. hp_melee / hp_missile [B, i, j]: HP i's
    blows / projectiles took from j this step, mhit / shit [B, i, j] the HP a hit; enemy [B, i, j]. Hits land as
    Poisson counts of the expected ones: at least one with 1 - exp(-hits) (a hash draw per battle, unit and step,
    morale._hash_uniform, so a replay draws the same)."""
    keys = contact_keys(params)
    if not keys or "cfx0_s" not in u:
        return
    from tools.nn.sim import morale
    B, N = u["men"].shape
    k = torch.round(t / dt).long() if isinstance(t, torch.Tensor) else torch.full(
        (B,), round(float(t) / dt), dtype=torch.long, device=u["men"].device)
    ids = morale.battle_ids(u)
    n_m = torch.where(mhit > 0, hp_melee / mhit.clamp(min=1e-6), torch.zeros_like(hp_melee)) * enemy
    n_s = torch.where(shit > 0, hp_missile / shit.clamp(min=1e-6), torch.zeros_like(hp_missile)) * enemy
    T = table(params, u["men"].device)
    for j, i, _ in keys:
        hits = ((u["mfx"] == i).float()[:, :, None] * n_m).sum(1) + ((u["sfx"] == i).float()[:, :, None] * n_s).sum(1)
        draw = morale._hash_uniform(B, N, k, u["men"].device, salt=52817 + 131 * j, ids=ids)
        hit = (hits > 0) & (draw < 1 - torch.exp(-hits))
        left = (u[f"cfx{j}_s"] - dt).clamp(min=0)
        u[f"cfx{j}_s"] = torch.where(hit, T[i, COL["duration_s"]].expand_as(left), left)


def timers(u):
    """The timed effects' state: {"fxt{j}_on", "fxt{j}_cd"} [B, N] (s)."""
    return {f"fxt{j}_{t}": u[f"fxt{j}_{t}"] for j in range(TIMERS) for t in ("on", "cd")}
