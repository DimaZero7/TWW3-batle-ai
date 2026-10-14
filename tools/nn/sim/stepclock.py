"""The simulator's hidden clocks rebuilt from a recorded battle (numpy only): what tools/nn/sim/stepreplay.py puts into
the simulator at each window start besides the observed state (docs/en/training/simulator.md, "The step replay").

A recording (tools/nn/gamedata.Battle, one row a second) shows the place, men, health, morale, the flags and the target
of every unit, not the clocks the simulator keeps for itself (tools/nn/sim/state.py INTERNAL). Each is rebuilt here from
the rows up to the window start, by the simulator's own rule applied to the recorded history:

    lost_hist         the HP lost each 0.5 s step over the last 60 s (morale's recent / extended casualties: a row's loss
                      split over its two steps); a unit leaving the map is not a loss (its men go to 0, its health stays)
    taken, shot_taken the decaying sums of HP taken in melee / from missiles (morale.recent_s): a row's loss is melee when
                      the unit's melee flag is on (this row or the last), else missile
    dealt, shot_dealt the same of HP dealt: each loss shared out over the enemies the recording names as attacking it
                      (target = it; in melee: in melee themselves, else the nearest enemy in melee within NEAR_M; missile:
                      firing at it) - an estimate, the recording does not say who struck whom
    waver_s           seconds of the current unbroken wavering (the w flag)
    rout_count, rout_s, rally_s     routs begun so far, seconds since the current rout began, since the last rally
    contact_s, out_s, melee_s, ran_in   the melee clock (the m flag of a standing unit; gaps under contact.reset_s keep
                      it), seconds out of contact, seconds in melee over the battle, came into the fight moving
    runup, charge     run-up (m at contact charge.min_speed_share of the run speed or faster towards the target / the
                      nearest enemy) and the charge left (1 at a contact under an attack order after min_runup_m, down over
                      charge_decay_duration)
    vx, vz            velocity (the last row's way; 0 where the game's moving flag is off)
    rpin_s, rpin_d    the rout's exit of a formation that routed from melee (contact.rout_pin_s): its clock and way run
    under_fire_s      seconds since the last HP lost while an enemy fired at it
    low_s, dead_s, gone, gone_s     since health first under a quarter, since killed, left the map, since
    unready           the share of a shooter's men still reloading (from its last ammunition drop)
    lord_dead_s       [2] seconds since each side's lord died, shattered or left (-1 alive)
    aim, aim_tgt      a shooter's seconds standing still out of melee with ammunition, and the target it fires at
                      (the fire flag on that target this row and the last; -1 none)

Abilities (casts): ability_rows() reads the run's nn_effects events (every change of the effects on a unit) - a lord's
active ability appearing on him is a cast; its timers at a row follow from the passport (active_s, recharge_s).
"""
import json
from pathlib import Path

import numpy as np

NEAR_M = 40.0        # an enemy in melee this near (centres) counts as striking a unit the recording names no attacker of
DEAD_HP = 0.003      # health share at or below which a unit with no men is dead, not gone


def side4(rel, front_deg, rear_deg):
    """0 front, 1 right flank, 2 rear, 3 left flank of a unit, from an angle off its facing (radians; any wrap) - the
    simulator's geometry.side4."""
    a = (np.asarray(rel) + np.pi) % (2 * np.pi) - np.pi
    d = np.abs(a)
    s = np.where(d <= np.deg2rad(front_deg), 0, np.where(d >= np.deg2rad(rear_deg), 2, 1))
    return np.where((s == 1) & (a < 0), 3, s)


def _ffill(a):
    """[T, n] float with NaN rows carried forward (a unit no longer listed keeps its last value); leading NaN -> 0."""
    a = np.array(a, float)
    for k in range(1, len(a)):
        bad = ~np.isfinite(a[k])
        a[k, bad] = a[k - 1, bad]
    return np.nan_to_num(a, nan=0.0)


def status(g):
    """(alive, dead, gone) [T, n] bool from the recording: dead = health 0; gone = no men left with health kept (a router
    that left the map - the game takes its men off, the health stays)."""
    men = _ffill(g.f["men"])
    hp = _ffill(g.f["hp"])
    dead = (hp <= DEAD_HP)
    gone = (men <= 0) & ~dead
    gone = np.maximum.accumulate(gone, axis=0)
    dead = np.maximum.accumulate(dead & ~gone, axis=0)
    alive = ~dead & ~gone
    return alive, dead, gone


def losses(g, hp0):
    """[T, n] HP (absolute) lost in each row (row 0: none); none once gone or dead."""
    hp = _ffill(g.f["hp"])
    _, dead, gone = status(g)
    out = np.zeros_like(hp)
    out[1:] = np.clip(hp[:-1] - hp[1:], 0, None) * np.asarray(hp0, float)[None, :]
    out[1:] = np.where(gone[1:] & gone[:-1], 0.0, out[1:])
    return out


def clocks(g, rows, info, cal, hist=True):
    """The hidden clocks at the given rows (window starts): {name: [len(rows), n]} (lord_dead_s: [len(rows), 2]).
    info: per recorded unit numpy arrays hp0 (HP at the start), run (m/s), men0, reload (s; 0 none), range (m).
    cal: {"dt", "fade_s" (morale.recent_s), "hist_s" (extended window), "reset_s", "min_share", "min_runup", "decay"
    (charge_decay_duration), "rout_pin_s"}. hist False: no lost_hist (its [rows, K, n] is large for every row)."""
    f = g.f
    T, n = len(g.t), len(g.names)
    side = np.asarray(g.side)
    enemy = side[:, None] != side[None, :]
    x, z = _ffill(f["x"]), _ffill(f["z"])
    bear = _ffill(f["b"]) if "b" in f else np.zeros_like(x)
    hp = _ffill(f["hp"])
    a = _ffill(f["a"])
    m = np.asarray(f["m"], bool)
    r = np.asarray(f["r"], bool)
    s = np.asarray(f["s"], bool)
    w = np.asarray(f["w"], bool)
    fire = np.asarray(f["fire"], bool)
    mv = np.asarray(f["mv"], bool) if "mv" in f else np.zeros_like(fire)
    tgt = np.asarray(g.target, int)
    alive, dead, gone = status(g)
    loss = losses(g, info["hp0"])
    hp0 = np.asarray(info["hp0"], float)
    run = np.asarray(info["run"], float)
    men0 = np.asarray(info["men0"], float)
    reload = np.asarray(info["reload"], float)
    lord = np.asarray(info.get("lord", np.zeros(n, bool)), bool)
    dt = float(cal["dt"])
    per_row = int(round(1.0 / dt))
    K = int(round(float(cal["hist_s"]) / dt))
    fade = float(np.exp(-1.0 / float(cal["fade_s"])))
    want = {int(k): j for j, k in enumerate(rows)}
    names = ("taken", "shot_taken", "dealt", "shot_dealt", "waver_s", "rout_count", "rout_s", "rally_s", "contact_s",
             "out_s", "melee_s", "ran_in", "runup", "charge", "vx", "vz", "rpin_s", "rpin_d", "under_fire_s", "low_s",
             "dead_s", "gone", "gone_s", "unready", "recent", "extended", "aim", "aim_tgt")
    out = {k: np.zeros((len(rows), n)) for k in names}
    out["lost_hist"] = np.zeros((len(rows), K if hist else 0, n))
    out["lord_dead_s"] = np.full((len(rows), 2), -1.0)
    # running values
    taken = np.zeros(n); shot_taken = np.zeros(n); dealt = np.zeros(n); shot_dealt = np.zeros(n)
    wav = np.zeros(n); rcount = np.zeros(n); rout_at = np.full(n, -1.0); rally_at = np.full(n, -1e9)
    contact = np.zeros(n); out_s = np.zeros(n); melee_s = np.zeros(n); ran_in = np.zeros(n, bool)
    runup = np.zeros(n); charge_at = np.full(n, -1e9); vx = np.zeros(n); vz = np.zeros(n)
    pin_at = np.full(n, -1.0); pin_d = np.zeros(n); fired_at = np.full(n, -1e9); low_at = np.full(n, -1.0)
    dead_at = np.full(n, -1.0); gone_at = np.full(n, -1.0); vol_at = np.full(n, -1e9)
    lord_at = np.full(2, -1.0)
    still = np.zeros(n)
    on_t = np.zeros(n)
    for k in range(T):
        eng = m[k] & ~r[k] & alive[k]
        if k > 0:
            dx, dz = x[k] - x[k - 1], z[k] - z[k - 1]
            # (only where the game shows it moving: its place jitters ~0.3 m/s as men shift, and a standing shooter
            # with that speed is moving to the simulator - its aim back to 0)
            on_move = alive[k] & mv[k]
            vx, vz = np.where(on_move, dx, 0.0), np.where(on_move, dz, 0.0)
            # losses: melee or missile; who dealt them
            L = loss[k]
            mel = m[k] | m[k - 1]
            taken = taken * fade + np.where(mel, L, 0.0)
            shot_taken = shot_taken * fade + np.where(mel, 0.0, L)
            dealt = dealt * fade
            shot_dealt = shot_dealt * fade
            d = np.hypot(x[k][:, None] - x[k][None, :], z[k][:, None] - z[k][None, :])
            for j in np.flatnonzero(L > 0):
                on_j = enemy[:, j] & alive[k] & ((tgt[k] == j) | (tgt[k - 1] == j))
                if mel[j]:
                    att = on_j & (m[k] | m[k - 1])
                    if not att.any():
                        near = enemy[:, j] & alive[k] & m[k] & ~r[k] & (d[:, j] <= NEAR_M)
                        if near.any():
                            att = np.zeros(n, bool)
                            att[np.argmin(np.where(near, d[:, j], np.inf))] = True
                    if att.any():
                        dealt = dealt + att * (L[j] / att.sum())
                else:
                    att = on_j & (fire[k] | fire[k - 1])
                    if att.any():
                        shot_dealt = shot_dealt + att * (L[j] / att.sum())
                        fired_at[j] = k
            # rout and rally edges
            rise = r[k] & ~r[k - 1] & alive[k]
            fall = ~r[k] & r[k - 1] & alive[k]
            rcount = rcount + rise
            rout_at = np.where(rise, k, rout_at)
            rally_at = np.where(fall, k, rally_at)
            # the rout's exit: a formation that routed from melee
            pin = rise & (m[k - 1] & ~r[k - 1]) & (men0 > 1)
            pin_at = np.where(pin, k, pin_at)
            pin_d = np.where(pin, 0.0, pin_d + np.hypot(dx, dz))
            # the run-up and the charge (towards the recorded target, else the nearest standing enemy)
            foe = enemy & alive[k][None, :] & ~r[k][None, :]
            dd = np.where(foe, d, np.inf)
            near_j = dd.argmin(1)
            to = np.where(tgt[k - 1] >= 0, tgt[k - 1], near_j)
            has = (tgt[k - 1] >= 0) | np.isfinite(dd.min(1))
            ux, uz = x[k - 1][to] - x[k - 1], z[k - 1][to] - z[k - 1]
            v_to = np.where(has, (dx * ux + dz * uz) / np.maximum(np.hypot(ux, uz), 1e-6), 0.0)
            fast = (v_to >= float(cal["min_share"]) * run) & (v_to > 0.1)
            fresh = eng & (contact <= 0)
            j_of = np.where(tgt[k] >= 0, tgt[k], tgt[k - 1])
            attack = j_of >= 0
            # the target already fighting (another of ours) the row before: no charge (charge.free_target_only); with
            # charge.blocked_side only when one of ours in melee within NEAR_M of it stands on the charger's side of it
            # (the recording does not say who touches whom)
            jj = np.clip(j_of, 0, None)
            busy = attack & m[k - 1][jj] & ~r[k - 1][jj]
            if cal.get("blocked_side"):
                xj, zj = x[k - 1][jj], z[k - 1][jj]
                bj = np.deg2rad(bear[k - 1][jj])
                s_i = side4(np.arctan2(x[k - 1] - xj, z[k - 1] - zj) - bj, cal["front_deg"], cal["rear_deg"])
                qx, qz = x[k - 1][None, :] - xj[:, None], z[k - 1][None, :] - zj[:, None]
                s_q = side4(np.arctan2(qx, qz) - bj[:, None], cal["front_deg"], cal["rear_deg"])
                ours = (side[:, None] == side[None, :]) & ~np.eye(n, dtype=bool)
                q_on = (m[k - 1] & ~r[k - 1] & alive[k - 1])[None, :]
                block = ours & q_on & (np.hypot(qx, qz) <= NEAR_M) & (s_q == s_i[:, None])
                busy = busy & block.any(1)
            ran = runup + np.where(fast, v_to, 0.0)
            charge_now = fresh & attack & (ran >= float(cal["min_runup"])) & ~busy
            charge_at = np.where(charge_now, k, charge_at)
            ran_in = np.where(fresh, np.hypot(dx, dz) > 0.3, ran_in)
            runup = np.where(eng, 0.0, np.where(fast, runup + v_to, 0.0))
            # a shooter's volley: its ammunition fell
            vol_at = np.where((a[k] < a[k - 1] - 1e-6) & (reload > 0), k, vol_at)
        # a shooter's aim: seconds standing still out of melee with ammunition (the simulator's aim clock), and the
        # target it fires at (the flag on, the same target: rows in a row)
        can = (reload > 0) & alive[k] & ~r[k] & ~m[k] & ~mv[k] & (a[k] > 0)
        still = np.where(can, still + (1.0 if k > 0 else 0.0), 0.0)
        same = fire[k] & (tgt[k] >= 0) & ((tgt[k] == tgt[k - 1]) if k > 0 else True)
        on_t = np.where(same & can, on_t + 1.0, 0.0)
        # the melee clock
        melee_s = melee_s + eng
        out_s = np.where(eng, 0.0, out_s + (1.0 if k > 0 else 0.0))
        contact = np.where(eng, contact + 1.0, np.where(out_s < float(cal["reset_s"]), contact, 0.0))
        ran_in = ran_in & (contact > 0)
        wav = np.where(w[k] & ~r[k] & alive[k], wav + 1, 0)
        low_at = np.where((low_at < 0) & (hp[k] < 0.25) & ~gone[k], k, low_at)
        dead_at = np.where((dead_at < 0) & dead[k], k, dead_at)
        gone_at = np.where((gone_at < 0) & gone[k], k, gone_at)
        for sd in (1, 2):
            fell = lord & (side == sd) & (dead[k] | gone[k] | s[k])
            if lord_at[sd - 1] < 0 and fell.any():
                lord_at[sd - 1] = k
        if k in want:
            j = want[k]
            o = out
            o["taken"][j], o["shot_taken"][j], o["dealt"][j], o["shot_dealt"][j] = taken, shot_taken, dealt, shot_dealt
            o["waver_s"][j] = np.maximum(wav - 0.5, 0.0) * (wav > 0)
            o["rout_count"][j] = rcount
            o["rout_s"][j] = np.where(r[k] & (rout_at >= 0), k - rout_at, 0.0)
            o["rally_s"][j] = np.where(rally_at > -1e8, k - rally_at, 1e6)
            o["contact_s"][j], o["out_s"][j], o["melee_s"][j] = contact, out_s, melee_s
            o["ran_in"][j] = ran_in
            o["runup"][j] = runup
            o["charge"][j] = np.clip(1.0 - (k - charge_at - 0.5 * dt) / float(cal["decay"]), 0.0, 1.0)
            o["vx"][j], o["vz"][j] = vx, vz
            pin_on = r[k] & (pin_at >= 0) & (k - pin_at < float(cal["rout_pin_s"]))
            o["rpin_s"][j] = np.where(pin_on, np.maximum(k - pin_at, 1e-6), 0.0)
            o["rpin_d"][j] = np.where(pin_on, pin_d, 0.0)
            o["under_fire_s"][j] = np.where(fired_at > -1e8, k - fired_at, 1e6)
            o["low_s"][j] = np.where(low_at >= 0, k - low_at + dt, 0.0)
            o["dead_s"][j] = np.where(dead_at >= 0, k - dead_at + dt, 0.0)
            o["gone"][j] = gone[k]
            o["gone_s"][j] = np.where(gone_at >= 0, k - gone_at + dt, 0.0)
            o["aim"][j] = still
            o["aim_tgt"][j] = np.where((on_t > 0) & fire[k], tgt[k], -1)
            o["unready"][j] = np.where(reload > 0, np.clip(1.0 - (k - vol_at) / np.maximum(reload, 1e-6), 0.0, 1.0), 0.0)
            past = np.repeat(loss[max(0, k - K // per_row + 1):k + 1][::-1], per_row, axis=0)[:K] / per_row
            if hist:
                o["lost_hist"][j, :len(past)] = past
            o["recent"][j] = past[:int(round(float(cal.get("recent_window_s", 4.0)) / dt))].sum(0)
            o["extended"][j] = past.sum(0)
            o["lord_dead_s"][j] = np.where(lord_at >= 0, k - lord_at + dt, -1.0)
    return out


def effect_rows(run_dir, g):
    """{key: [T, n] bool}: the effects on each unit at each row, from the run's nn_effects events (each event lists every
    effect on the unit after a change); empty without them."""
    path = Path(run_dir) / "events.jsonl"
    t = np.asarray(g.t, float)
    idx = {nm: i for i, nm in enumerate(g.names)}
    ev = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if '"nn_effects"' not in line:
            continue
        e = json.loads(line)
        if e.get("u") in idx:
            row = min(int(np.searchsorted(t, float(e.get("t", 0)) / 1000.0 - 1e-6)), len(t) - 1)
            fx = e.get("fx") or []
            ev.append((row, idx[e["u"]], set(fx) if isinstance(fx, list) else set(fx)))
    ev.sort(key=lambda r: r[0])
    out = {}
    cur = [set() for _ in g.names]
    T = len(t)
    k = 0
    keys = sorted({key for _, _, fx in ev for key in fx})
    for key in keys:
        out[key] = np.zeros((T, len(g.names)), bool)
    for i in range(T):
        while k < len(ev) and ev[k][0] <= i:
            cur[ev[k][1]] = ev[k][2]
            k += 1
        for u, fx in enumerate(cur):
            for key in fx:
                out[key][i, u] = True
    return out


def casts(on):
    """[T, n] bool (an ability on its owner) -> [T, n] bool: the rows it came on (a cast)."""
    on = np.asarray(on, bool)
    c = on.copy()
    c[1:] &= ~on[:-1]
    return c


def ability_timers(cast, rows, active_s, recharge_s, initial_s=0.0):
    """(on, cd) [len(rows), n] seconds at each row (the simulator's ab_on / ab_cd before the row's steps): from the last
    cast at or before the row; none yet: on 0, cd = initial_s - row (clamped)."""
    cast = np.asarray(cast, bool)
    T, n = cast.shape
    last = np.where(cast, np.arange(T)[:, None], -1)
    last = np.maximum.accumulate(last, axis=0)
    rows = np.asarray(rows, int)
    lr = last[rows]
    since = rows[:, None] - lr
    on = np.where(lr >= 0, np.clip(active_s - since, 0.0, None), 0.0)
    cd = np.where(lr >= 0, np.clip(active_s + recharge_s - since, 0.0, None),
                  np.clip(initial_s - rows[:, None], 0.0, None) * np.ones((1, n)))
    return on, cd
