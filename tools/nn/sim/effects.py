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

Only json in the table builders; the tensors need torch.
"""
try:
    import torch
except ImportError:          # the table's layout is importable without torch
    torch = None

from tools.nn.sim.abilities import MULT, SIM_STATS, STATS

PREDICATES = ("in_melee", "out_of_melee", "losing_melee", "morale_below_half", "not_wavering", "hp_below_half",
              "hp_below_quarter")
# The catalogue's rules the simulator acts on -> the unit field that carries the flag for the step.
RULES = {"unbreakable": "unbreakable", "expendable": "expendable", "encourages": "encourages",
         "charge_reflection": "reflect", "fire_while_moving": "fire_move", "fatigue_immune": "fatigue_immune"}
TIMERS = 2          # timed effects a unit can carry (fxt0, fxt1)
MAX_EFFECTS = 62    # the bitmask is an int64
GROUPS = ("self", "friends")
# A list to append each step's {effect key: [B, N] bool - on the unit, its own or an owner's aura} to, or None
# (training: off); the probes' simulator twins read it (tools/nn/morale_probe.py).
RECEIVED = None
HEAD = ("modelled", "timed", "active_s", "recharge_s", "range_m", "ability")
COLS = HEAD + tuple(f"{w}_{p}" for w in ("need", "off", "fire", "rcg") for p in PREDICATES) + tuple(
    f"rule_{r}" for r in RULES) + tuple(f"{g}_{s}" for g in GROUPS for s in STATS)
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
    return {"fx": bits, **{f"fxt{j}": t for j, t in enumerate(timed)}, **rules}


def row(params, key):
    """One effect's row of the table (COLS)."""
    e = params.effects["effects"][key]
    ab_keys = sorted(params.abilities or {})
    t = e.get("timed") or {}
    out = {c: 0.0 for c in COLS}
    out.update(modelled=float(modelled(params, key)), timed=float(bool(t)),
               active_s=float(t.get("active_s", 0)), recharge_s=float(max(t.get("recharge_s", 0), 0)),
               range_m=float(e.get("range_m") or 0.0),
               ability=float(ab_keys.index(key)) if key in ab_keys else -1.0)
    for w, preds in (("need", e.get("needs") or ()), ("off", e.get("off_when") or ()),
                     ("fire", t.get("fires_when") or ()), ("rcg", t.get("recharge_needs") or ())):
        for p in preds:
            out[f"{w}_{p}"] = 1.0
    for r in e.get("rules") or ():
        if r in RULES:
            out[f"rule_{r}"] = 1.0
    for g in GROUPS:
        for s in STATS:
            out[f"{g}_{s}"] = 1.0 if s in MULT else 0.0
    for st in e.get("stats") or ():
        s = SIM_STATS.get((st["stat"], st["how"]))
        if s is None:
            continue
        for g in st.get("on") or ():
            if g in GROUPS:
                k = f"{g}_{s}"
                out[k] = out[k] * float(st["value"]) if s in MULT else out[k] + float(st["value"])
    return [out[c] for c in COLS]


def table(params, device):
    """[E + 1, COLS]: every effect of the catalogue; the last row (index -1) is none."""
    empty = [0.0] * len(COLS)
    for g in GROUPS:
        for s in MULT:
            empty[COL[f"{g}_{s}"]] = 1.0
    empty[COL["ability"]] = -1.0
    # an effect in the order that no unit owns any more (the order is append-only) has an empty row
    rows = [row(params, k) if k in params.effects["effects"] else empty for k in order(params)]
    return torch.tensor(rows + [empty], device=device)


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
            "hp_below_quarter": (u["low_s"] >= quarter_delay(params)) if "low_s" in u else u["hp"] < 0.25}


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
        for k in range(3):
            match = has & (ab >= 0) & (u[f"ab{k}"] == ab)
            u[f"ab{k}_on"] = torch.where(match, t_on, u[f"ab{k}_on"])
            u[f"ab{k}_cd"] = torch.where(match, t_cd, u[f"ab{k}_cd"])
    bit = torch.arange(E, device=u["fx"].device)
    u["fx_on"] = (on.long() << bit).sum(-1)
    on = on & (c("modelled") > 0)
    onf = on.float()
    log = {s: torch.einsum("bne,e->bn", onf, torch.log(c(f"self_{s}").clamp(min=1e-6))) for s in MULT}
    add = {s: torch.einsum("bne,e->bn", onf, c(f"self_{s}")) for s in STATS if s not in MULT}
    eye = torch.eye(u["men"].shape[1], dtype=torch.bool, device=u["men"].device)[None]
    present = u["side"] > 0
    got = {}
    for i in auras(params):
        # from a living owner, routing too, to friends in range, routing too (spec 3.4)
        give = (same_side & ~eye & present[:, None, :] & (dist <= T[i, COL["range_m"]])
                & (on[..., i] & alive)[:, :, None]).float()             # owner i -> friend j
        if RECEIVED is not None:
            got[i] = give.sum(1) > 0
        for s in STATS:
            v = T[i, COL[f"friends_{s}"]]
            if s in MULT:
                log[s] = log[s] + give.sum(1) * torch.log(v.clamp(min=1e-6))
            else:
                add[s] = add[s] + give.sum(1) * v
    fields = ("walk", "run", "charge_speed", "damage", "ap_damage", "charge_bonus", "attack", "defence",
              "morale_bonus", "resist_physical") + tuple(RULES.values())
    old = {k: u[k] for k in fields}
    speed = torch.exp(log["speed"])
    # scalar_charge_speed, when an effect gives it, sets the charge speed; else scalar_speed scales it too
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
    for r, f in RULES.items():
        u[f] = torch.einsum("bne,e->bn", onf, c(f"rule_{r}")) > 0
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
    u["fx_on"] = (on.long() << bit).sum(-1)


def timers(u):
    """The timed effects' state: {"fxt{j}_on", "fxt{j}_cd"} [B, N] (s)."""
    return {f"fxt{j}_{t}": u[f"fxt{j}_{t}"] for j in range(TIMERS) for t in ("on", "cd")}
