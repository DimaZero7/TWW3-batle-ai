"""The unit script's exam of execution (METHOD.md 'Уровень', «Личный уровень — без сети»): a simple drill-generator
per skill of stage 0 (net/unit/scripts.SKILLS), the measure of its execution, the orders to the game a unit a minute
(net/common/reissue.py). 'hold' is drill U2 (u2.py: its exec and head_on); here counter, attack, withdraw.

A skirmish: my one unit (drill_hold.roster: the 16 melee infantry of the Empire and the Skaven) against two enemy
slots of our gold (u2.GOLD_TOL) - A the cause's enemy, B the decoy or away (AWAY_M behind, holding); the ground the
flat field or Kislev (u2.MAPS, laid at a random place and turn as U2's); a decision a second; the order's pace any /
walk / run. The generator changes all but the cause; half the skirmishes have no cause (NO_CAUSE).
    counter   my unit in its frame (FRAME_M) facing its course; A starts START_M ahead within SIDE_DEG of the course
              and walks or runs at the frame (u2.walker / drill_go.to_area: it strikes us within its GRAB_M). No
              cause: A holds far away. End: CONTACT_S after our first contact with A, or LIMIT_S.
    attack    A (the main target) TARGET_M away in any direction, holding or coming (sim nearest); with the cause B
              stands across the way (DECOY_AT of it, DECOY_OFF_M aside) and attacks us (nearest); no cause: B away.
              End: A routs, dies or leaves, or the limit (the way at a walk x 1.5 + 60 s).
    withdraw  my unit faces A, which attacks it (nearest) from MELEE_M; the frame BACK_M behind, its course ours; the
              order is 'hold' (its frame where it starts) until our first contact + ORDER_S, then 'withdraw'. No cause: no A (the order at
              ORDER_S). End: LIMIT_S.
Measures (each named for what it counts):
    counter   charged      at our first contact with A: our charge on (state charge > 0.5 within 2 s of it), A in our
                           front sector (contact.front_deg) and our centre in the frame when A came within its reach
                           (scripts.CHARGE_M + its run x LEAD_S): the skill done - stood, then hit with a run on the last
                           metres; with the cause only. waited / charge_on / front - its parts
              stayed       no cause: our centre never left the frame (no charge at nothing)
    attack    reached      our unit in contact with A (state hit_sec either way) before the end; reached_decoy /
                           reached_free - with / without the decoy across the way; first_on_target - our first
                           contact of all was with A
    withdraw  in_frame     our centre in the frame, our unit standing (not routing) at the end; reach_s - seconds
                           from the order to the frame (NaN never); lost - the share of our men lost from the order
              (all) orders_min - new orders to the game a unit a minute (the gate); t_end
    python -m net.unit.skills --skills counter,attack,withdraw --players unit,stand --seeds 2 --n 256 --out f.json
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from sim import battle, scenario
from sim import orders as O
from net.unit import scripts as SC
from net.unit import u2

FRAME_M = (48.0, 80.0)
START_M = (120.0, 400.0)
SIDE_DEG = 60.0
TARGET_M = (100.0, 350.0)
DECOY_AT = (0.3, 0.6)
DECOY_OFF_M = (0.0, 30.0)
MELEE_M = 25.0
BACK_M = (80.0, 250.0)
ORDER_S = (3.0, 10.0)
NO_CAUSE = 0.5
AWAY_M = 3000.0
CONTACT_S = 3.0
LIMIT_S = 240.0
SEED = 6301
SKILL_DRILLS = ("counter", "attack", "withdraw")
HOLD_E, WALK_E, RUN_E, NEAREST_E = range(4)          # the enemy slot's script


def specs(skill, n, gen):
    from net.unit import drill_hold as DH
    pp = u2._passports()
    R = DH.roster()
    u = lambda a, b: a + (b - a) * float(torch.rand(1, generator=gen))       # noqa: E731
    pick = lambda xs: xs[int(torch.randint(len(xs), (1,), generator=gen))]    # noqa: E731
    out = []
    for _ in range(n):
        ours, theirs = pick(("emp", "skv")), pick(("emp", "skv"))
        me = pick(R[ours])
        g = float(pp[me]["multiplayer_cost"])
        ok = [k for k in R[theirs] if abs(float(pp[k]["multiplayer_cost"]) - g) <= u2.GOLD_TOL * g] or \
            [min(R[theirs], key=lambda k: abs(float(pp[k]["multiplayer_cost"]) - g))]
        foe, decoy = pick(ok), pick(ok)
        cause = float(torch.rand(1, generator=gen)) >= NO_CAUSE
        fw, fd = u(*FRAME_M), u(*FRAME_M)
        bf = u(0.0, 360.0)
        h = math.radians(bf)
        fwd = lambda d, off=0.0: (d * math.sin(h + off), d * math.cos(h + off))       # noqa: E731
        away = fwd(-AWAY_M)
        s = {"skill": skill, "me": me, "ours": ours, "theirs": theirs, "foe": foe, "decoy": decoy, "cause": cause,
             "pace": pick(u2.PACES), "map": u2._pick(gen, u2.MAPS), "limit": LIMIT_S, "order_s": 0.0,
             "b": bf, "frame": (0.0, 0.0, bf, fw, fd), "B": (*away, bf, HOLD_E)}
        s["x"], s["z"] = 0.0, 0.0
        if skill == "counter":
            th = h + math.radians(u(-SIDE_DEG, SIDE_DEG))
            d = u(*START_M)
            ax, az = d * math.sin(th), d * math.cos(th)
            beh = pick((WALK_E, RUN_E))
            s["A"] = (ax, az, math.degrees(math.atan2(-ax, -az)) % 360.0, beh) if cause else (*fwd(AWAY_M), bf, HOLD_E)
        elif skill == "attack":
            th = u(0.0, 2 * math.pi)
            d = u(*TARGET_M)
            ax, az = d * math.sin(th), d * math.cos(th)
            s["b"] = u(0.0, 360.0)
            s["A"] = (ax, az, math.degrees(math.atan2(-ax, -az)) % 360.0, pick((HOLD_E, NEAREST_E)))
            if cause:
                k, off = u(*DECOY_AT), u(*DECOY_OFF_M) * pick((-1.0, 1.0))
                bx, bz = ax * k + off * math.cos(th), az * k - off * math.sin(th)
                s["B"] = (bx, bz, math.degrees(math.atan2(-bx, -bz)) % 360.0, NEAREST_E)
            s["frame"] = (0.0, 0.0, s["b"], fw, fd)
            s["limit"] = d / float(pp[me]["speed"]["walk"]) * 1.5 + 60.0
        else:                                                   # withdraw
            back = u(*BACK_M)
            s["x"], s["z"] = fwd(back)
            ax, az = fwd(back + MELEE_M)
            s["A"] = (ax, az, (bf + 180.0) % 360.0, NEAREST_E) if cause else (*fwd(AWAY_M), bf, HOLD_E)
            s["order_s"] = u(*ORDER_S)
        out.append(s)
    return out


def build(sp, P, device):
    from net.unit import drill_hold as DH
    armies = []
    for s in sp:
        mine = {"key": s["me"], "x": s["x"], "z": s["z"], "b": s["b"]}
        th = [{"key": s[k], "x": s[e][0], "z": s[e][1], "b": s[e][2]} for k, e in (("foe", "A"), ("decoy", "B"))]
        armies.append({"attacker": 2, "sides": {1: {"faction": DH.FACTIONS[s["ours"]], "units": [mine]},
                                                2: {"faction": DH.FACTIONS[s["theirs"]], "units": th}}})
    st = scenario.build(armies, P, device=device)
    B, N = st.B, st.N
    side = st.u["side"]
    own = torch.zeros((B, N), dtype=torch.bool, device=device)
    own[:, 0] = True
    foes = (side[0] == 2).nonzero().flatten().tolist()
    col = lambda i: torch.tensor([s["frame"][i] for s in sp], device=device,           # noqa: E731
                                 dtype=torch.float32)[:, None].expand(B, N).clone()
    area = {"x": col(0), "z": col(1), "b": col(2), "front": col(3), "depth": col(4), "on": own.clone()}
    # the Kislev skirmishes laid as U2's (the slots far away out of the place: side 0 for realmap)
    lay = [dict(s, enemy=s["cause"] or s["skill"] == "attack") for s in sp]
    sd = st.u["side"].clone()
    away = torch.tensor([[s[e][3] == HOLD_E and abs(s[e][0]) + abs(s[e][1]) > AWAY_M / 2 for e in ("A", "B")]
                         for s in sp], device=device)
    for k, j in enumerate(foes):
        st.u["side"][:, j] = torch.where(away[:, k], torch.zeros_like(sd[:, j]), sd[:, j])
    st.u["side"][:, foes[0]] = sd[:, foes[0]]               # (u2._lay leaves the first enemy out by its own flag)
    u2._lay(st, area, lay)
    st.u["side"] = sd
    return st, own, area, foes


def enemy_orders(st, area, beh):
    """The enemy's orders: each slot's script (beh [B, N] HOLD_E / WALK_E / RUN_E / NEAREST_E)."""
    from net.unit import drill_go as DG
    from sim import scripts as opponents
    o = opponents.hold(st)
    o = O.merge(o, u2.walker(st, area), beh == WALK_E)
    o = O.merge(o, DG.to_area(st, area), beh == RUN_E)
    return O.merge(o, opponents.nearest(st), beh == NEAREST_E)


def _touch(st, i, j):
    hs = st.u["hit_sec"].view(st.B, st.N, st.N)
    return (hs[:, i, j] > 0) | (hs[:, j, i] > 0)


def play(sp, player, P, device, compile=False, record=()):
    """{measure: [B] (host)} of the skirmishes sp (one skill) with my unit played by player (scripts.py)."""
    from net.matchup import duel as DU
    from net.operative import lower as LW
    from net.unit import ckpt as UC
    from net.unit import net as Nt
    skill = sp[0]["skill"]
    assert all(s["skill"] == skill for s in sp)
    st, own, area, (A, Bd) = build(sp, P, device)
    B, N = st.B, st.N
    u = st.u
    setup = DU._SideSetup(st, P, device, 1, UC.cards(st, device).float())
    units = LW.Units(setup, player, gate=True)
    units.area = {**{k: area[k].clone() for k in ("x", "z", "b", "front", "depth")}, "sb": area["b"].clone(),
                  "sw": torch.zeros((B, N), device=device), "on": own.clone()}
    pace = {"any": Nt.PACE_ANY, "slow": Nt.PACE_SLOW, "fast": Nt.PACE_FAST}
    units.pace = torch.tensor([pace[s["pace"]] for s in sp], device=device)[:, None].expand(B, N).clone()
    code = SC.SKILLS.index(skill)
    order_s = torch.tensor([s["order_s"] for s in sp], device=device)
    cause = torch.tensor([s["cause"] for s in sp], device=device)
    units.prio = torch.full((B, N), -1, dtype=torch.long, device=device)
    units.prio[:, 0] = A
    beh = torch.full((B, N), HOLD_E, dtype=torch.long, device=device)
    beh[:, A] = torch.tensor([s["A"][3] for s in sp], device=device)
    beh[:, Bd] = torch.tensor([s["B"][3] for s in sp], device=device)
    limit = torch.tensor([float(s["limit"]) for s in sp], device=device)
    a0 = {k: area[k][:, 0] for k in ("x", "z", "b", "front", "depth")}
    start = {"x": u["x"][:, :1].expand(B, N).clone(), "z": u["z"][:, :1].expand(B, N).clone()}
    from net.unit import decode as Dc
    from net.unit import view as V
    from sim import geometry
    nan = lambda: torch.full((B,), float("nan"), device=device)          # noqa: E731
    f0 = lambda: torch.zeros(B, dtype=torch.bool, device=device)         # noqa: E731
    m = {"t_contact": nan(), "t_reachA": nan(), "waited": f0(), "charge_on": f0(), "front": f0(), "left": f0(),
         "first_A": f0(), "first_any": nan(), "reached": f0(), "t_order": nan(), "reach_s": nan(), "men_o": nan(),
         "in_end": f0(), "over": f0()}
    first_c = f0()
    t_first = nan()
    sim_step = battle.stepper(device, compile)
    steps = round(1.0 / P.dt)
    st.done = torch.zeros(B, dtype=torch.bool, device=device)
    front_rad = math.radians(P.sim["contact"]["front_deg"])
    frames = {b: [] for b in record}
    for sec in range(int(limit.max()) + 2):
        for b in frames:
            if not bool(st.done[b]):
                from game.vision import record as RC
                frames[b].append(RC.sim_frame(st, b))
        if bool(st.done.all()):
            break
        u = st.u
        t = st.t.float()
        # the tactician's order: the skill (withdraw: hold until its moment)
        if skill == "withdraw":
            due = torch.where(cause, t_first + order_s, order_s)
            now = torch.isfinite(due) & (t >= due)
            m["t_order"] = torch.where(now & torch.isnan(m["t_order"]), t, m["t_order"])
            men = u["men"][:, 0].float() / u["men0"][:, 0].float().clamp(min=1)
            m["men_o"] = torch.where(now & torch.isnan(m["men_o"]), men, m["men_o"])
            units.skill[:, 0] = torch.where(now, torch.full_like(units.skill[:, 0], SC.WITHDRAW),
                                            torch.full_like(units.skill[:, 0], SC.HOLD))
            for k in ("x", "z"):                  # (before the order: hold where it starts - its frame there)
                units.area[k] = torch.where(now[:, None], area[k], start[k])
        else:
            units.skill[:, 0] = code
        lands = units.second(st)
        for j in range(steps):
            was = st.done.clone()
            sim_step(st, units.orders(st, j, lands, lambda s_: enemy_orders(s_, area, beh)), P)
            u = st.u
            live = ~was
            out0 = u["r"][:, 0] | (u["men"][:, 0] <= 0) | u["gone"][:, 0]
            tA, tB = _touch(st, 0, A), _touch(st, 0, Bd)
            t = st.t.float()
            new_c = live & (tA | tB) & ~first_c
            m["first_A"] = torch.where(new_c, tA, m["first_A"])
            t_first = torch.where(new_c, t, t_first)
            first_c = first_c | new_c
            m["reached"] = m["reached"] | (live & tA)
            inside = Dc.signed_dist(u["x"][:, 0], u["z"][:, 0], a0) >= 0
            m["left"] = m["left"] | (live & ~inside)
            if skill == "counter":
                raw = V.raw(st, P.sim["formation"]["spacing_m"], missile=P.sim["missile"])
                gap = geometry.rect_gap(*(raw[..., V.I[k]] for k in ("x", "z", "b", "front", "depth")))[:, 0, A]
                reach = SC.CHARGE_M + u["run"][:, A] * SC.LEAD_S
                came = live & (gap <= reach) & torch.isnan(m["t_reachA"])
                m["waited"] = torch.where(came, inside, m["waited"])
                m["t_reachA"] = torch.where(came, t, m["t_reachA"])
                newA = live & tA & torch.isnan(m["t_contact"])
                if bool(newA.any()):
                    dx, dz = u["x"][:, A] - u["x"][:, 0], u["z"][:, A] - u["z"][:, 0]
                    rel = torch.remainder(torch.atan2(dx, dz) - torch.deg2rad(u["b"][:, 0]) + math.pi,
                                          2 * math.pi) - math.pi
                    m["front"] = torch.where(newA, rel.abs() <= front_rad, m["front"])
                    m["t_contact"] = torch.where(newA, t, m["t_contact"])
                inwin = live & torch.isfinite(m["t_contact"]) & (t - m["t_contact"] <= 2.0)
                m["charge_on"] = m["charge_on"] | (inwin & (u["charge"][:, 0] > 0.5))
                end = torch.isfinite(m["t_contact"]) & (t - m["t_contact"] >= CONTACT_S)
            elif skill == "attack":
                end = u["r"][:, A] | (u["men"][:, A] <= 0) | u["gone"][:, A]
            else:
                ordered = torch.isfinite(m["t_order"])
                got = live & ordered & inside & ~out0 & torch.isnan(m["reach_s"])
                m["reach_s"] = torch.where(got, t - m["t_order"], m["reach_s"])
                end = f0()
            over = live & (out0 | end | (t >= limit) | st.done)
            m["in_end"] = torch.where(over, inside & ~out0, m["in_end"])
            st.done = st.done | over
        units.end_second()
    u = st.u
    g = units.gate.per_minute(own)
    fl = lambda v: v.float()                                             # noqa: E731
    na = lambda v, ok: torch.where(ok, fl(v), torch.full_like(fl(v), float("nan")))     # noqa: E731
    res = {"orders_min": g, "t_end": st.t.float()}
    if skill == "counter":
        ok = m["waited"] & m["charge_on"] & m["front"] & torch.isfinite(m["t_contact"])
        res.update(charged=na(ok, cause), waited=na(m["waited"], cause), charge_on=na(m["charge_on"], cause),
                   front=na(m["front"] & torch.isfinite(m["t_contact"]), cause),
                   contact=na(torch.isfinite(m["t_contact"]), cause), stayed=na(~m["left"], ~cause))
    elif skill == "attack":
        res.update(reached=fl(m["reached"]), reached_decoy=na(m["reached"], cause), reached_free=na(m["reached"], ~cause),
                   first_on_target=fl(first_c & m["first_A"]))
    else:
        men = u["men"][:, 0].float() / u["men0"][:, 0].float().clamp(min=1)
        res.update(in_frame=fl(m["in_end"]), reach_s=m["reach_s"], lost=(m["men_o"] - men),
                   in_frame_melee=na(m["in_end"], cause), in_frame_free=na(m["in_end"], ~cause))
    out = {k: v.detach().cpu() for k, v in res.items()}
    if frames:
        out["_frames"] = frames
    return out


def _mean(v):
    v = v.float()
    v = v[torch.isfinite(v)]
    return float(v.mean()) if v.numel() else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", default=",".join(SKILL_DRILLS))
    ap.add_argument("--players", default="unit,stand")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--compile", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from sim.params import load
    P = load()
    res = {}
    for skill in a.skills.split(","):
        for name in a.players.split(","):
            rows = []
            for s in range(a.seeds):
                sp = specs(skill, a.n, torch.Generator().manual_seed(SEED + 100 * SKILL_DRILLS.index(skill) + s))
                torch.manual_seed(3000 + s)
                t0 = time.time()
                mm = play(sp, SC.player(name), P, a.device, compile=bool(a.compile))
                rows.append({k: _mean(v) for k, v in mm.items()})
                print(json.dumps({"skill": skill, "player": name, "seed": s, "wall_s": round(time.time() - t0, 1),
                                  **{k: round(v, 4) for k, v in rows[-1].items()}}), flush=True)
            keys = rows[0].keys()
            res.setdefault(skill, {})[name] = {
                k: (round(sum(r[k] for r in rows) / len(rows), 4),
                    round(float(torch.tensor([r[k] for r in rows]).std()), 4) if len(rows) > 1 else float("nan"))
                for k in keys}
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    main()
