"""The new units in the game before the simulator knows them (build/newunits/plan.md 4): the Ratling Gun, the Plague
Monks, the Plague Censer Bearers, the Wolf Rats (plain and poisoned) and the Halberdiers. The lanes run on the melee probe
(tools/nn/charge_probe.py, src/entries/charge_probe.lua: its plans by these names); one mechanic a plan:

  ratling   the gun's fire (R1): it shoots held fearless swordsmen 60 / 100 / 135 m off (front to front), shield spearmen
            100 m off, the swordsmen 100 m off with half its crew killed before the go, and the lone General 100 m off (few
            hits: the whole ammunition's cadence); fire at will, one ranged attack at a walk, RATLING_S - 2 battles of 6
            lanes, rotated. Measures: crew and ammunition at the go, the first shot, bullets a second and a man, the bursts
            (ammo drops between gaps of 1+ s), the time to empty, HP the target lost per bullet, the target's phases (fx:
            suppressing fire?) and card speed;
  suppress  suppressing fire (R2): swordsmen attack the gun from SUPPRESS_GAP m at a walk / at a run, the gun aimed at them
            (fire at will, its skirmish mode off) or silent (never ordered: the control); the swordsmen's speed under fire and
            without, their card speed - 2 battles of 4 lanes (the gun fires on in melee: the phase never ends there);
  suppress2 the phase's life (R2): the same walk / run with the gun falling silent (fire at will off, a halt) 25 / 15 s
            after the go; the phase after its last shot, the speed back - 1 battle of 4 lanes;
  censer    the censer's contagion, the monks' frenzy, the halberds (C1, H1): censer bearers -> swordsmen, monks ->
            swordsmen (no contact effect: the control), censers -> shield spearmen, swordsmen -> held censers, censers
            leaving 20 s after contact with the swordsmen halting (the effect's life after the last blow), the two pairs again
            with the swordsmen keeping their morale (the rout), halberdiers -> stormvermin with shields; from 30 m at a run,
            60 s - 2 battles of 8 lanes on a grid, rotated;
  wolfrats  the beasts' pace and charge (W1, W2): a run 150 m aside and back (speed, acceleration, the turn), a charge from
            120 m on swordsmen (the charge speed), from 40 m on braced spearmen, on swordsmen facing away (the rear), on the
            General, a recharge (30 m out, 3 s, in again), the chase of routed swordsmen (half their men killed at
            placement, routed 15 s after contact) and the same chase by clanrats (the control) - 2 battles of 8 lanes;
  wolfpoison the poison (W3): poisoned wolf rats -> swordsmen, plain ones (the control), the poisoned leaving 15 s after
            contact with the swordsmen halting or chasing them, the plain leaving chased (the control) - 2 battles of 5 lanes.

All the units are fearless (but the t_morale targets); every lane's rows carry the morale fields with the active phases (fx)
and the unit card's values (cs: leadership, speed, melee attack / defence, weapon and missile damage).

    python -m tools.nn.charge_probe plan --plan ratling            # the battles
    python -m tools.nn.charge_probe run --plan ratling --speed 10   # build + launch each, in turn
    python -m tools.nn.unit_probe [runs...]                         # the five plans' tables (the game's numbers)

Units by short name (tools/nn/charge_probe.UNITS) or any Empire / Skaven key with a passport (config/nn/units.json,
data/units/probe-units.json); the plans' units are this module's constants (RATLING, CENSER, ...).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tools.nn import charge_probe as cp

RATLING, MONKS, CENSER, WOLF, WOLFP, HALB = "ratling", "monks", "censer", "wolfrat", "wolfratp", "halb"
GRID = cp.RETARGET2_PLACES          # 10 places 290+ m apart in x, 400 m in z
FX = dict(morale_rows=True, card_rows=True)
RATLING_S = 200          # the game: a burst of 18 a man every ~12 s (7 s of fire, 5 of reload), 15 bursts: ~180 s
# (kind, target, gap front to front m, share of the crew killed at placement)
RATLING_LANES = (("d60", "swords", 60, None), ("d100", "swords", 100, None), ("d135", "swords", 135, None),
                 ("shield100", "spearsh", 100, None), ("half100", "swords", 100, 0.5), ("lord100", "general", 100, None))
SUPPRESS_GAP = 120
SUPPRESS2_STOP = {"walk_stop25": (True, 25), "run_stop15": (False, 15)}     # kind -> (walk, the gun silent after s)
HITSLOW_SHOOTERS = ("hgun", "xbow")      # the hitslow plan's shooters (no contact effect): handguns flat, bolts arcing
HITSLOW_RUNNER = "wh2_main_skv_inf_clanrats_0"     # ... and the runners at them: clanrats without shields
PACE_M, PACE_BACK_S = 150, 20
CHASE = dict(t_morale=True, at_rout="none", after_rout_s=45, max_s=200, rout_at_s=15,
             damage={"t": {"method": "kill", "share": 0.5}})


def _placed(base, k):
    """The lanes rotated by k on the grid's places; a pace lane runs towards the middle of the field."""
    out = []
    for spec, place in zip(cp.rotate(list(base), k), GRID):
        spec = dict(spec, place=place)
        if spec.get("kind") == "pace":
            dx = -PACE_M if place[0] > 0 else PACE_M
            spec["steps"] = [{"at_s": 1, "kind": "move", "dx": dx, "dz": 0, "run": True},
                             {"at_s": PACE_BACK_S, "kind": "move", "dx": 0, "dz": 0, "run": True}]
        out.append(spec)
    return out


def battles(plan):
    """[[lane spec]] of a new units' plan (tools/nn/charge_probe.battles hands these over)."""
    if plan == "ratling":
        def r(kind, target, gap, cut):
            extra = {"damage": {"a": {"method": "kill", "share": cut}}} if cut else {}
            return cp.lane(RATLING, target, "shoot", "hold", gap_m=gap, fight_s=RATLING_S, max_s=RATLING_S, answer=False,
                           kind=kind, **FX, **extra)
        base = [r(*x) for x in RATLING_LANES]
        return [_placed(base, k) for k in (0, 3)]
    if plan == "suppress":
        def s(kind, walk, fire):
            return cp.lane("swords", RATLING, "attack_walk" if walk else "attack_run", "skirmish" if fire else "hold",
                           gap_m=SUPPRESS_GAP, fight_s=15, max_s=200, answer=False, kind=kind, t_skirmish=False, **FX)
        base = [s("walk_fire", True, True), s("walk_quiet", True, False), s("run_fire", False, True),
                s("run_quiet", False, False)]
        return [_placed(base, k) for k in (0, 2)]
    if plan == "hitslow":
        # (12.10.2026) the suppress plan's walk / run with clanrats without shields walking / running at handgunners
        # (flat fire, no contact effect) and crossbowmen (arcing): do hits alone slow a unit as the gun's do beyond its
        # x0.7 (the gun's walk 0.91 m/s, run 1.29 against 1.05 / 2.10 with x0.7)? quiet: the handgunners never fire
        def hs(kind, shooter, walk, fire=True):
            return cp.lane(HITSLOW_RUNNER, shooter, "attack_walk" if walk else "attack_run",
                           "skirmish" if fire else "hold", gap_m=SUPPRESS_GAP, fight_s=15, max_s=200, answer=False,
                           kind=kind, t_skirmish=False, **FX)
        base = [hs(f"{gait}_{s}", s, gait == "walk") for s in HITSLOW_SHOOTERS for gait in ("walk", "run")]
        base += [hs(f"{gait}_quiet", "hgun", gait == "walk", False) for gait in ("walk", "run")]
        return [_placed(base, k) for k in (0, 3)]
    if plan == "suppress2":
        def s2(kind):
            walk, stop = SUPPRESS2_STOP[kind]
            return cp.lane("swords", RATLING, "attack_walk" if walk else "attack_run", "skirmish", gap_m=SUPPRESS_GAP,
                           fight_s=5, max_s=150, answer=False, kind=kind, t_skirmish=False, t_stop_fire_s=stop, **FX)
        return [_placed([s2(k) for k in SUPPRESS2_STOP] * 2, 0)]
    if plan == "censer":
        def c(kind, a, t, mode="attack_run", tmode="stand", **kw):
            return cp.lane(a, t, mode, tmode, **dict(dict(gap_m=30, fight_s=60, kind=kind, **FX), **kw))
        leave = dict(answer=False, recharge_after_s=20, back_m=60, recharge_max_s=10 ** 6, t_at_leave="halt")
        rout = dict(t_morale=True, at_rout="halt", after_rout_s=10)
        base = [c("censer", CENSER, "swords"), c("monks", MONKS, "swords"), c("censer_shield", CENSER, "spearsh"),
                c("held_censer", "swords", CENSER, tmode="hold", answer=False),
                c("censer_leave", CENSER, "swords", "withdraw", "both", **leave),
                c("censer_morale", CENSER, "swords", **rout), c("monks_morale", MONKS, "swords", **rout),
                c("halb", HALB, "svsh")]
        return [_placed(base, k) for k in (0, 4)]
    if plan == "wolfrats":
        def w(kind, a, t, mode="attack_run", tmode="stand", **kw):
            return cp.lane(a, t, mode, tmode, **dict(dict(gap_m=40, fight_s=40, kind=kind, **FX), **kw))
        base = [w("pace", WOLF, "swords", "script", "hold", gap_m=60, answer=False, men_all_s=45, max_s=45),
                w("charge120", WOLF, "swords", gap_m=120),
                w("braced", WOLF, "spear", tmode="hold", answer=False),
                w("rear", WOLF, "swords", tmode="rear", answer=False),
                w("general", WOLF, "general"),
                w("recharge", WOLF, "swords", "recharge", "hold", answer=False, recharge_after_s=15, back_m=30,
                  pause_s=3, recharge_max_s=30, after2_s=20, max_s=100),
                w("chase", WOLF, "swords", fight_s=200, **CHASE),
                w("chase_clanrat", "clanrat", "swords", fight_s=200, **CHASE)]
        return [_placed(base, k) for k in (0, 4)]
    if plan == "wolfpoison":
        def p(kind, a, leave=None):
            if leave is None:
                return cp.lane(a, "swords", "attack_run", "stand", gap_m=30, fight_s=40, kind=kind, **FX)
            extra = dict(t_at_leave="halt") if leave == "halt" else {}
            return cp.lane(a, "swords", "withdraw", "both", gap_m=30, fight_s=55, answer=False, recharge_after_s=15,
                           back_m=80, recharge_max_s=10 ** 6, kind=kind, **FX, **extra)
        base = [p("poison", WOLFP), p("plain", WOLF), p("poison_leave_halt", WOLFP, "halt"),
                p("poison_leave_chase", WOLFP, "chase"), p("plain_leave_chase", WOLF, "chase")]
        return [_placed(base, k) for k in (0, 2)]
    raise ValueError(plan)


# ---------------------------------------------------------------- the recordings

def _t(lane):
    return np.array([x[0] for x in lane["samples"]])


def _row(sample, who):
    return sample[{"a": 1, "tg": 2}[who]] or {}


def _fx(lane, who):
    """[set of phase keys] a sample."""
    return [set(filter(None, (_row(x, who).get("fx") or "").split(","))) for x in lane["samples"]]


def _card(lane, who, key):
    return np.array([np.nan if (_row(x, who).get("cs") or {}).get(key) is None else float(_row(x, who)["cs"][key])
                     for x in lane["samples"]])


def _speed(lane, who, window_s=2.0):
    """The unit's speed (m/s) over the window_s before each sample (from its centre's places: the engine moves a unit in
    steps, a 0.5 s difference is noise)."""
    t = _t(lane)
    x, z = cp.series(lane["samples"], who, "x"), cp.series(lane["samples"], who, "z")
    v = np.full(len(t), np.nan)
    ok = np.isfinite(x) & np.isfinite(z)
    if ok.sum() < 2:
        return v
    for i, ti in enumerate(t):
        if ti - window_s >= t[ok][0] and ok[i]:
            x0, z0 = np.interp(ti - window_s, t[ok], x[ok]), np.interp(ti - window_s, t[ok], z[ok])
            v[i] = float(np.hypot(x[i] - x0, z[i] - z0) / window_s)
    return v


def _rate(t, v, lo, hi):
    """v lost a second from lo to hi (NaN outside the recording)."""
    a, b = cp.at(t, v, lo), cp.at(t, v, hi)
    return (a - b) / (hi - lo) if np.isfinite(a) and np.isfinite(b) and hi > lo else np.nan


def _phase_t(lane, name):
    return next((p["t"] / 1000 for p in lane.get("phases", []) if p["phase"] == name), None)


def _med(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else None


def effects(lane, who, since=None):
    """{phase key: (first s, last s)} the unit carried (from the lane's go; since: only keys not on before it)."""
    t, sets = _t(lane), _fx(lane, who)
    before = set().union(*[s for ti, s in zip(t, sets) if since is not None and ti < since - 1e-6]) if since else set()
    out = {}
    for ti, s in zip(t, sets):
        for k in s - before:
            out[k] = (out.get(k, (float(ti),))[0], float(ti))
    return out


def _new_effect(lane, who, since, part):
    """(key, first s, last s) of the first phase key containing part (or any new one, part None) the unit got after
    `since`, else None."""
    fx = effects(lane, who, since)
    keys = sorted((k for k in fx if part is None or part in k), key=lambda k: fx[k][0])
    return (keys[0],) + fx[keys[0]] if keys else None


def _card_change(lane, who, key, c, lo=0.0, hi=30.0):
    """(card value before c, the extreme after it within lo..hi s: the min, or the max when it went up) or None."""
    t, v = _t(lane), _card(lane, who, key)
    pre, post = v[(t < c) & np.isfinite(v)], v[(t >= c + lo) & (t <= c + hi) & np.isfinite(v)]
    if not len(pre) or not len(post):
        return None
    base = float(pre[-1])
    lo_v, hi_v = float(post.min()), float(post.max())
    return base, (hi_v if base - lo_v < hi_v - base else lo_v)


def ratling_measure(lane):
    spec, s = lane["spec"], lane["samples"]
    t = _t(lane)
    out = {"kind": spec.get("kind"), "run": lane["run"]}
    ammo, men = cp.series(s, "a", "ammo"), cp.series(s, "a", "men")
    hp, tmen = cp.series(s, "tg", "hp"), cp.series(s, "tg", "men")
    ok = np.isfinite(ammo)
    if not ok.any():
        return out
    out["ammo0"], out["men0"] = float(ammo[ok][0]), float(men[0])
    out["ammo_per_man"] = out["ammo0"] / out["men0"] if out["men0"] else None
    a0, t0 = _row(s[0], "a"), _row(s[0], "tg")
    out["d0_m"] = float(np.hypot(a0["x"] - t0["x"], a0["z"] - t0["z"])) if None not in (a0.get("x"), t0.get("x")) else None
    drop = -np.diff(np.where(ok, ammo, np.nan))           # bullets fired in each half second
    fired = np.nan_to_num(drop) > 0
    if not fired.any():
        return out
    ft = t[1:][fired]
    first, last = float(ft[0]), float(ft[-1])
    bullets = float(np.nansum(drop[fired]))
    out.update(first_shot_s=first, last_shot_s=last, bullets=bullets, fire_s=last - first + 0.5)
    out["bullets_per_s"] = bullets / out["fire_s"]
    out["bullets_per_man_s"] = out["bullets_per_s"] / out["men0"] if out["men0"] else None
    empty = t[ok & (ammo <= 0)]
    out["empty_s"] = float(empty[0]) if len(empty) else None
    # bursts: fired samples with gaps of 1+ s between them
    starts, sizes, ends = [ft[0]], [0.0], [ft[0]]
    for ti, d in zip(ft, drop[fired]):
        if ti - ends[-1] > 1.0 + 1e-6:
            starts.append(ti), sizes.append(0.0), ends.append(ti)
        sizes[-1] += float(d)
        ends[-1] = ti
    out["bursts"] = len(starts)
    out["burst_bullets"] = float(np.median(sizes))
    out["burst_s"] = float(np.median(np.array(ends) - np.array(starts) + 0.5))
    out["burst_every_s"] = float(np.median(np.diff(starts))) if len(starts) > 1 else None
    stop = min(float(t[-1]), last + 2.0)          # the last bullets' flight and the health readout
    lost = cp.at(t, hp, first - 0.5) - cp.at(t, hp, stop)
    out["tg_hp_lost"] = lost
    out["tg_men_lost"] = cp.at(t, tmen, first - 0.5) - cp.at(t, tmen, stop)
    out["hp_per_bullet"] = lost / bullets if bullets else None
    out["tg_hp_per_s"] = lost / (stop - first + 0.5)
    out["dd_end"] = _row(s[-1], "a").get("dd")
    out["tg_dead"] = bool(np.nanmin(tmen) == 0) if np.isfinite(tmen).any() else None
    sup = _new_effect(lane, "tg", first, None)
    if sup:
        out["tg_fx"], out["tg_fx_first_s"] = sup[0], sup[1] - first
    spd = _card_change(lane, "tg", "scalar_speed", first, 0, RATLING_S)
    if spd:
        out["tg_card_speed"], out["tg_card_speed_fx"] = spd
    return out


def suppress_measure(lane):
    spec, s = lane["spec"], lane["samples"]
    t = _t(lane)
    out = {"kind": spec.get("kind"), "run": lane["run"]}
    c = lane["contacts"].get(1)
    out["contact_s"] = c
    end = c if c is not None else float(t[-1])
    d = np.hypot(cp.series(s, "a", "x") - cp.series(s, "tg", "x"), cp.series(s, "a", "z") - cp.series(s, "tg", "z"))
    v = _speed(lane, "a")
    on = np.array([any("suppress" in k for k in fx) for fx in _fx(lane, "a")])
    way = (t >= 4) & (t < end) & (d > 50)            # on the way, before the last 30 m sprint (the charge)
    out["speed_way"] = _med(v[way])
    out["speed_way_fx"] = _med(v[way & on])
    out["speed_way_nofx"] = _med(v[way & ~on])
    out["fx_share_way"] = float(on[way].mean()) if way.any() else None
    keys = set().union(*_fx(lane, "a"))
    out["a_fx"] = ",".join(sorted(keys)) or None
    spd = _card(lane, "a", "scalar_speed")
    if np.isfinite(spd).any():
        out["card_speed0"] = float(spd[np.isfinite(spd)][0])
        out["card_speed_fx"] = _med(spd[on])
        out["card_speed_min"] = float(np.nanmin(spd))
    ammo = cp.series(s, "tg", "ammo")
    drop = np.nan_to_num(np.diff(ammo)) < 0
    if drop.any():
        last = float(t[1:][drop][-1])
        out["shots"] = float(np.nansum(-np.diff(ammo)[drop]))
        out["last_shot_s"] = last
        if on.any():
            # the phase's last sample after the gun's last shot (negative: gone before it); still on at the end -> no end
            last_on = float(t[on].max())
            out["fx_after_last_shot_s"] = last_on - last
            out["fx_ended"] = last_on < float(t[-1]) - 1e-6
    hp = cp.series(s, "a", "hp")
    out["a_hp_lost_way"] = (cp.at(t, hp, 0.0) - cp.at(t, hp, end)) / cp.at(t, hp, 0.0) if np.isfinite(hp).any() else None
    return out


WIN = ((0, 5), (5, 15), (15, 30), (30, 60))


def _fight(lane, out, c, stop=None):
    """HP lost a second by both sides in WIN after the contact c (to stop)."""
    t = _t(lane)
    stop = float(t[-1]) if stop is None else stop
    for who in ("tg", "a"):
        hp = cp.series(lane["samples"], who, "hp")
        for lo, hi in WIN:
            if c + hi <= stop + 1e-6:
                out[f"{who}_hp_{lo}_{hi}"] = _rate(t, hp, c + lo, c + hi)


def censer_measure(lane):
    spec = lane["spec"]
    out = {"kind": spec.get("kind"), "run": lane["run"]}
    c = lane["contacts"].get(1)
    out["contact_s"] = c
    if c is None:
        return out
    t = _t(lane)
    leave = _phase_t(lane, "out")
    _fight(lane, out, c, leave)
    for who in ("tg", "a"):
        fx = effects(lane, who, c)
        out[f"{who}_fx_new"] = ",".join(sorted(fx)) or None
    hit = _new_effect(lane, "tg", c, "censer") or _new_effect(lane, "tg", c, None)
    if hit:
        out["tg_fx"], out["tg_fx_first_s"] = hit[0], hit[1] - c
    for who, key in (("tg", "stat_morale"), ("a", "stat_melee_attack"), ("a", "stat_weapon_damage"),
                     ("tg", "stat_melee_attack")):
        ch = _card_change(lane, who, key, c, 0, 30)
        if ch:
            out[f"{who}_{key}"], out[f"{who}_{key}_fight"] = ch
    mp = cp.series(lane["samples"], "tg", "mp")
    sel = (t >= c + 5) & (t <= c + 30)
    out["tg_mp_5_30"] = _med(mp[sel])
    for dt in (5, 10):
        out[f"tg_mp_{dt}"] = cp.at(t, mp, c + dt)
    rout = _phase_t(lane, "rout")
    out["rout_s"] = rout - c if rout is not None else None
    if leave is not None:
        m = cp.series(lane["samples"], "tg", "m")
        inside = (t >= leave) & (m > 0.5)
        clear = float(t[inside].max()) if inside.any() else leave
        out["leave_clear_s"] = clear - leave
        if hit and hit[2] > clear:
            # the phase's last sample after the last melee sample; still on at the lane's end -> no end seen
            out["tg_fx_after_clear_s"] = hit[2] - clear
            out["tg_fx_still_on"] = hit[2] >= float(t[-1]) - 1e-6
    return out


def wolfrats_measure(lane):
    spec = lane["spec"]
    kind = spec.get("kind")
    out = {"kind": kind, "run": lane["run"]}
    t, v = _t(lane), _speed(lane, "a")
    if kind == "pace":
        # per leg (the order at lo): the cruise speed (the median of the samples above half the leg's top: the centre
        # moves in steps, a 2 s peak is noise), the seconds to 90 % of it, the leg's time to a stop and its metres
        x, z = cp.series(lane["samples"], "a", "x"), cp.series(lane["samples"], "a", "z")
        for leg, (lo, hi) in enumerate(((1, PACE_BACK_S), (PACE_BACK_S, float(t[-1]) + 1)), 1):
            sel = (t >= lo) & (t < hi) & np.isfinite(v)
            if sel.sum() < 3:
                continue
            tv, vv = t[sel], v[sel]
            cruise = float(np.median(vv[vv > 0.5 * vv.max()]))
            reach = tv[vv >= 0.9 * cruise]
            out[f"leg{leg}_cruise"] = cruise
            out[f"leg{leg}_to90_s"] = float(reach[0]) - lo if len(reach) else None
            v05 = _speed(lane, "a", 0.5)[sel]          # a stop: no step in the half second
            moving = tv[v05 > 0.5]
            if len(moving):
                end = float(moving[-1])
                out[f"leg{leg}_s"] = end - lo
                out[f"leg{leg}_m"] = float(np.hypot(cp.at(t, x, end) - cp.at(t, x, lo), cp.at(t, z, end) - cp.at(t, z, lo)))
        return out
    if kind == "recharge":
        r = cp.recontact_measure(lane)
        for k in ("first_tg_0_3", "first_tg_10_20", "re_tg_0_3", "re_tg_10_20", "first_a_0_3", "re_a_0_3", "out_s"):
            out[k] = r.get(k)
        return out
    m = cp.measure(lane)
    for k in ("contact_s", "speed_peak30", "speed_last30", "speed_30_60", "approach_s", "tg_hp_0_1", "tg_hp_0_2",
              "tg_hp_0_5", "tg_hp_5_15", "tg_hp_steady", "a_hp_0_5", "a_hp_steady", "a_near_0_5", "a_near_5_15"):
        out[k] = m.get(k)
    c = m.get("contact_s")
    if c is not None:
        before = (t < c) & np.isfinite(v)
        out["speed_peak_way"] = float(v[before].max()) if before.any() else None
    if kind and kind.startswith("chase"):
        rout = _phase_t(lane, "rout")
        out["rout_s"] = rout - c if rout is not None and c is not None else None
        if rout is not None:
            hp, men = cp.series(lane["samples"], "tg", "hp"), cp.series(lane["samples"], "tg", "men")
            stop = min(float(t[-1]), rout + 45)
            out["chase_s"] = stop - rout
            out["chase_tg_hp_per_s"] = _rate(t, hp, rout, stop)
            out["chase_tg_men_lost"] = cp.at(t, men, rout) - cp.at(t, men, stop)
            out["chase_tg_men_left"] = cp.at(t, men, stop)
    return out


def wolfpoison_measure(lane):
    spec = lane["spec"]
    out = {"kind": spec.get("kind"), "run": lane["run"]}
    c = lane["contacts"].get(1)
    out["contact_s"] = c
    if c is None:
        return out
    t = _t(lane)
    leave = _phase_t(lane, "out")
    _fight(lane, out, c, leave)
    fx = effects(lane, "tg", c)
    out["tg_fx_new"] = ",".join(sorted(fx)) or None
    hit = _new_effect(lane, "tg", c, "poison")
    if hit:
        out["tg_fx"], out["tg_fx_first_s"] = hit[0], hit[1] - c
    for key in ("scalar_speed", "stat_weapon_damage", "stat_melee_attack", "stat_missile_damage_over_time"):
        ch = _card_change(lane, "tg", key, c, 0, 15)
        if ch:
            out[f"tg_{key}"], out[f"tg_{key}_fight"] = ch
    if leave is not None:
        m = cp.series(lane["samples"], "tg", "m")
        inside = (t >= leave) & (m > 0.5)
        clear = float(t[inside].max()) if inside.any() else leave
        out["leave_clear_s"] = clear - leave
        v = _speed(lane, "tg")
        a_m = cp.series(lane["samples"], "a", "m")
        chase = (t >= clear + 2) & (t <= clear + 12) & ~(m > 0.5) & ~(a_m > 0.5)
        on = np.array([any("poison" in k for k in s) for s in _fx(lane, "tg")])
        out["tg_speed_chase"] = _med(v[chase])
        out["tg_speed_chase_fx"] = _med(v[chase & on])
        out["tg_speed_chase_nofx"] = _med(v[chase & ~on])
        if hit and hit[2] > clear:
            out["tg_fx_after_clear_s"] = hit[2] - clear
            out["tg_fx_still_on"] = hit[2] >= float(t[-1]) - 1e-6
    return out


MEASURE = {"ratling": ratling_measure, "suppress": suppress_measure, "suppress2": suppress_measure,
           "hitslow": suppress_measure, "censer": censer_measure,
           "wolfrats": wolfrats_measure, "wolfpoison": wolfpoison_measure}


def summary(rows):
    """{(plan, kind): {field: (mean, min, max, n) for numbers, the distinct values for texts}}."""
    groups = {}
    for r in rows:
        groups.setdefault((r["plan"], r["kind"]), []).append(r)
    out = {}
    for key, rs in groups.items():
        row = {"n": len(rs)}
        for f in dict.fromkeys(k for r in rs for k in r):
            if f in ("plan", "kind", "run", "battle", "lane"):
                continue
            vals = [r[f] for r in rs if r.get(f) is not None]
            nums = [float(x) for x in vals if isinstance(x, (int, float)) and not isinstance(x, bool)
                    and np.isfinite(float(x))]
            if nums and len(nums) == len(vals):
                row[f] = (float(np.mean(nums)), min(nums), max(nums), len(nums))
            elif vals:
                row[f] = sorted({str(x) for x in vals})
        out[key] = row
    return out


def _fmt(v):
    if isinstance(v, tuple):
        mean, lo, hi, n = v
        p = 0 if abs(mean) >= 100 else 2 if abs(mean) < 10 else 1
        return f"{mean:.{p}f}" if lo == hi else f"{mean:.{p}f} ({lo:.{p}f}-{hi:.{p}f})"
    return "/".join(v)


# The numbers the simulator's twin is checked on (report --sim: game | sim side by side), per plan
KEY = {"ratling": ("first_shot_s", "bullets_per_s", "bullets_per_man_s", "bursts", "burst_bullets", "burst_s",
                   "burst_every_s", "hp_per_bullet", "tg_hp_per_s", "tg_men_lost", "tg_fx_first_s", "tg_card_speed_fx"),
       "suppress": ("contact_s", "speed_way", "speed_way_fx", "fx_share_way", "card_speed_fx", "shots",
                    "fx_after_last_shot_s"),
       "suppress2": ("contact_s", "speed_way", "speed_way_fx", "speed_way_nofx", "fx_share_way", "shots",
                     "fx_after_last_shot_s"),
       "hitslow": ("contact_s", "speed_way", "a_hp_lost_way", "shots"),
       "censer": ("contact_s", "tg_hp_0_5", "tg_hp_5_15", "tg_hp_15_30", "a_hp_0_5", "a_hp_5_15", "tg_fx_first_s",
                  "a_stat_melee_attack", "a_stat_weapon_damage", "tg_mp_5", "tg_mp_10", "tg_mp_5_30", "rout_s",
                  "tg_fx_after_clear_s", "leave_clear_s"),
       "wolfrats": ("leg1_cruise", "leg1_to90_s", "leg2_cruise", "contact_s", "speed_peak30", "speed_last30",
                    "speed_peak_way", "tg_hp_0_1", "tg_hp_0_2", "tg_hp_0_5", "tg_hp_5_15", "tg_hp_steady", "a_hp_0_5",
                    "a_hp_steady", "first_tg_0_3", "re_tg_0_3", "chase_tg_hp_per_s", "chase_tg_men_lost"),
       "wolfpoison": ("contact_s", "tg_hp_0_5", "tg_hp_5_15", "tg_hp_15_30", "a_hp_0_5", "a_hp_5_15", "a_hp_15_30",
                      "tg_fx_first_s", "tg_scalar_speed_fight", "tg_stat_weapon_damage_fight", "tg_fx_after_clear_s",
                      "leave_clear_s", "tg_speed_chase")}


def measure_lanes(lanes):
    rows = []
    for lane in lanes:
        if lane.get("plan") in MEASURE:
            r = MEASURE[lane["plan"]](lane)
            r.update(plan=lane["plan"], battle=lane.get("battle"), lane=lane["spec"]["name"])
            rows.append(r)
    return rows


def twin(lanes, device="cpu", copies=4, params=None):
    """The simulator on the recorded lanes (tools/nn/charge_probe.sim_lanes: each lane its own battle, its units at
    their places at the go, the same scripted orders), `copies` a lane, in the game's shape with each lane's plan."""
    out = []
    for plan in MEASURE:
        mine = [ln for ln in lanes if ln.get("plan") == plan]
        if not mine:
            continue
        for ln, s in zip([ln for ln in mine for _ in range(copies)],
                         cp.sim_lanes(mine, params=params, device=device, copies=copies)):
            s.update(plan=plan, battle=ln.get("battle"))
            out.append(s)
    return out


def _side_by_side(game, sim):
    """Prints game | sim of the KEY numbers per plan and kind (means; the game's min-max)."""
    for plan, keys in KEY.items():
        kinds = sorted({k for p, k in game if p == plan}, key=str)
        if not kinds:
            continue
        print(f"== {plan} (game | sim)")
        for kind in kinds:
            g, s = game.get((plan, kind), {}), sim.get((plan, kind), {})
            cells = []
            for f in keys:
                if f not in g and f not in s:
                    continue
                gs = _fmt(g[f]) if f in g else "-"
                ss = _fmt((s[f][0], s[f][0], s[f][0], 1)) if isinstance(s.get(f), tuple) else (
                    "/".join(s[f]) if f in s else "-")
                cells.append(f"{f} {gs} | {ss}")
            print(f"  {kind}: " + "; ".join(cells))


def report(run_dirs, out=None, sim=False, device="cpu", copies=4):
    """The five plans' lanes in the runs: one row per lane, the means (min-max) per plan and kind; prints the tables,
    writes build/charge-probe/newunits.json (not in Git). sim: the simulator's twin on the same lanes beside them
    (newunits_sim.json)."""
    lanes = [ln for d in run_dirs for ln in cp.load_run(d) if ln.get("plan") in MEASURE]
    rows = measure_lanes(lanes)
    table = summary(rows)
    for plan in MEASURE:
        keys = [k for k in table if k[0] == plan]
        if not keys:
            continue
        print(f"== {plan}")
        for key in sorted(keys, key=lambda k: str(k[1])):
            r = table[key]
            print(f"  {key[1]} (n {r['n']}): " + "; ".join(f"{f} {_fmt(v)}" for f, v in r.items() if f != "n"))
    result = {"lanes": rows, "summary": [dict({k: v for k, v in r.items()}, plan=p, kind=k) for (p, k), r in table.items()]}
    if sim:
        srows = measure_lanes(twin(lanes, device=device, copies=copies))
        stable = summary(srows)
        _side_by_side(table, stable)
        result["sim"] = {"lanes": srows, "summary": [dict(r, plan=p, kind=k) for (p, k), r in stable.items()]}
    out = out or cp.ROOT / ("newunits_sim.json" if sim else "newunits.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(out)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path, help="run folders (default: build/charge-probe/runs/*)")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sim", action="store_true", help="the simulator's twin on the same lanes beside the game's "
                                                           "(torch; game | sim per plan and kind)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--copies", type=int, default=4)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    report(args.runs or cp.runs(), out=args.out, sim=args.sim, device=args.device, copies=args.copies)
    return 0


if __name__ == "__main__":
    sys.exit(main())
