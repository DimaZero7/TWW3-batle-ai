"""Drill U2 'stand where ordered, facing where ordered' by the method (METHOD.md 'Рецепт навыка'; design/unit #u2, the
table decided 09.11) - the first drill learned from scratch; the executor of the way 'hold'. Its exam: exam_u2.py.

Words. Situation: the order gives the unit its frame - its place and the course it faces (where the enemy comes from) -
and the unit stands out of it; an enemy infantry walks to the frame. The stupid move: come into the frame and stand
side-on to the enemy (run2 in the game: a new course every second, the formation never re-forms), or stay where it
is. The needed move: straight into the frame, there face its course and stand; in contact strike the one touching.

The skirmish (the page's table): my unit one of the 16 melee infantry of the Empire and the Skaven (drill_hold.roster);
its frame FRAME_M x FRAME_M, turned at random, its course at the enemy; the start out of the frame NEAR_M or FAR_M
from its edge, beside or behind it (OUT_SIDE_DEG: not between it and the enemy, who would meet us there before we are
in - the cause is 'time to get in and turn'), facing anywhere; the order 'hold' at a pace any / walk / run (PACES);
the ground the flat field or Kislev (MAPS, at a random place and turn: realmap.places); a decision a second; the end
at the first contact, else LIMIT_S (far starts LIMIT_FAR_S). The generator changes everything but the cause (the
frame looks where the enemy comes from, and there is time to get in and turn): half the skirmishes have no enemy
(NO_ENEMY: the recipe's half without the cause); in the rest an infantry of our gold (GOLD_TOL, either faction) walks
to the frame (drill_go.to_area at a walk: it strikes us when within its GRAB_M) from the frame's front sector
(ENEMY_ARC_DEG), coming LEAD_S after the needed move can stand ready there.

Measures (Meter):
  execution (the reward: per second) - the formation in the frame (all four corners of its box) and the course
          within FACE_OK_DEG of the frame's: 1, else 0. The exam: exec - the share of such seconds before the contact
          from the second the needed move can be ready (the spec's ready_s); exec_all - from the start; reached /
          reach_s - once in it so and when; in_frame_contact - the formation in the frame at the first contact (0: no
          contact); and the orders to the game a minute (net/common/reissue.py)
  usefulness (the judge) - met head-on: at the first contact the enemy's centre lies in our front sector (the
          simulator's melee sectors, contact.front_deg 45: the rule the side a contact began on is labelled by - state
          hit_sec); 0 when on our flank or rear, or the enemy got to the frame and no contact came by the end;
          skirmishes with an enemy only. Standing in the frame facing its course is for exactly this: the blow
          comes into the front - in the game a contact on the flank or rear brings the charge's bonus and the morale
          hit (game/manual.md); no trade and no outcome (tmp/decisions.md 5).

    specs(n, gen) -> [spec]   plain numbers (no torch state)
    play(sp, player, P, device, greedy=False) -> {measure: [B]}   a batch played: my unit by a scripted player
              (net/unit/scripts.py) or the network (operative/lower.Units: the latency, the order gate, the leash), the
              enemy walking to the frame (or holding far away: no enemy)
"""
import math
import types

import numpy as np
import torch

from sim import battle, scenario
from sim import geometry
from sim import orders as O
from net.unit import decode as Dc

FRAME_M = (48.0, 80.0)    # the frame's front and depth (design/unit #u2)
NEAR_M = (10.0, 120.0)    # our centre this far out of the frame's nearest edge (a near start) ...
FAR_M = (120.0, 700.0)    # ... a far one
FAR_SHARE = 0.5
OUT_SIDE_DEG = (60.0, 300.0)   # our side of the frame, from its course (module doc)
PACES = ("any", "slow", "fast")
NO_ENEMY = 0.5
GOLD_TOL = 0.10           # (net/config/stage0.json general.gold_tol: the stage's armies equal in gold)
ENEMY_ARC_DEG = 45.0      # the enemy starts within this of the frame's course, seen from its centre (the cause)
LEAD_S = (8.0, 30.0)      # the enemy comes to the frame this long after the needed move stands ready there (the cause)
AWAY_M = 3000.0           # no enemy: it holds this far behind our start (the simulator needs a standing side 2)
FACE_OK_DEG = 45.0        # 'facing by the frame' (the page: the course within 45 deg of the frame's)
IN_TOL_M = 0.5            # a corner this far out still counts as in
LIMIT_S, LIMIT_FAR_S = 180.0, 700.0
MAPS = (("flat", 0.5), ("ksl_plains_01", 0.5))
SEED = 5201               # the exam's first generator seed


def _passports():
    from net.unit import drill_hold as DH
    return DH._passports()


def _ready_s(dist, p, pace):
    """Seconds the needed move takes to stand ready: the walk to the centre at the order's pace (any: a run beyond
    scripts.RUN_M) + the re-form and a half turn (sim.json turn: face_delay_s 3.9 + 180 deg at formation_deg_s 80)."""
    from net.unit import scripts as SC
    run = pace == "fast" or (pace == "any" and dist > SC.RUN_M)
    return dist / p["speed"]["run" if run else "walk"] + 3.9 + 180.0 / 80.0


def _pick(gen, pairs):
    names, w = zip(*pairs)
    r = float(torch.rand(1, generator=gen)) * sum(w)
    for n, x in zip(names, w):
        r -= x
        if r <= 0:
            return n
    return names[-1]


def specs(n, gen):
    """n skirmishes of the generator (module doc); gen a torch.Generator."""
    from net.unit import drill_hold as DH
    pp = _passports()
    R = DH.roster()
    out = []
    u = lambda a, b: a + (b - a) * float(torch.rand(1, generator=gen))       # noqa: E731
    for _ in range(n):
        ours = "emp" if float(torch.rand(1, generator=gen)) < 0.5 else "skv"
        me = R[ours][int(torch.randint(len(R[ours]), (1,), generator=gen))]
        theirs = "emp" if float(torch.rand(1, generator=gen)) < 0.5 else "skv"
        g = float(pp[me]["multiplayer_cost"])
        ok = [k for k in R[theirs] if abs(float(pp[k]["multiplayer_cost"]) - g) <= GOLD_TOL * g]
        if not ok:
            ok = [min(R[theirs], key=lambda k: abs(float(pp[k]["multiplayer_cost"]) - g))]
        foe = ok[int(torch.randint(len(ok), (1,), generator=gen))]
        fw, fd = u(*FRAME_M), u(*FRAME_M)
        bf = u(0.0, 360.0)                                    # the frame's course (a world bearing)
        far = float(torch.rand(1, generator=gen)) < FAR_SHARE
        psi = math.radians(u(*OUT_SIDE_DEG))                  # our side of the frame, in its axes
        e = u(*(FAR_M if far else NEAR_M))
        half = abs(math.sin(psi)) * fw / 2 + abs(math.cos(psi)) * fd / 2
        s, f = (e + half) * math.sin(psi), (e + half) * math.cos(psi)
        edge = math.hypot(max(abs(s) - fw / 2, 0.0), max(abs(f) - fd / 2, 0.0))
        h = math.radians(bf)
        x, z = s * math.cos(h) + f * math.sin(h), -s * math.sin(h) + f * math.cos(h)
        pace = PACES[int(torch.randint(len(PACES), (1,), generator=gen))]
        ready = _ready_s(math.hypot(s, f), pp[me], pace)
        enemy = float(torch.rand(1, generator=gen)) >= NO_ENEMY
        lead = u(*LEAD_S)
        th = math.radians(bf + u(-ENEMY_ARC_DEG, ENEMY_ARC_DEG))
        de = fd / 2 + pp[foe]["speed"]["walk"] * (ready + lead)
        ex, ez = (de * math.sin(th), de * math.cos(th)) if enemy else \
            (x - AWAY_M * math.sin(h), z - AWAY_M * math.cos(h))
        out.append({"me": me, "foe": foe, "ours": ours, "theirs": theirs, "far": far, "enemy": enemy,
                    "pace": pace, "map": _pick(gen, MAPS), "x": x, "z": z, "b": u(0.0, 360.0), "edge": edge,
                    "frame": (0.0, 0.0, bf, fw, fd), "ex": ex, "ez": ez,
                    "eb": (math.degrees(math.atan2(-ex, -ez))) % 360.0, "ready_s": ready, "lead_s": lead,
                    "limit": LIMIT_FAR_S if far else LIMIT_S, "gold": (g, float(pp[foe]["multiplayer_cost"]))})
    return out


def build(sp, P, device):
    """(st, own [B, N], area {x, z, b, front, depth, on} [B, N]); the Kislev skirmishes laid on the map."""
    from net.unit import drill_hold as DH
    armies = []
    for s in sp:
        mine = {"key": s["me"], "x": s["x"], "z": s["z"], "b": s["b"]}
        theirs = {"key": s["foe"], "x": s["ex"], "z": s["ez"], "b": s["eb"]}
        armies.append({"attacker": 2, "sides": {1: {"faction": DH.FACTIONS[s["ours"]], "units": [mine]},
                                                2: {"faction": DH.FACTIONS[s["theirs"]], "units": [theirs]}}})
    st = scenario.build(armies, P, device=device)
    B, N = st.B, st.N
    own = torch.zeros((B, N), dtype=torch.bool, device=device)
    own[:, 0] = True
    col = lambda i: torch.tensor([s["frame"][i] for s in sp], device=device,           # noqa: E731
                                 dtype=torch.float32)[:, None].expand(B, N).clone()
    area = {"x": col(0), "z": col(1), "b": col(2), "front": col(3), "depth": col(4), "on": own.clone()}
    _lay(st, area, sp)
    return st, own, area


def _lay(st, area, sp, seed=7310):
    """The skirmishes whose map is not flat laid on it at a random place and turn (realmap.places: our unit, the frame
    and the enemy coming on free ground; an enemy holding far away is left out of the place)."""
    from net.unit import realmap
    u = dict(st.u)
    H = st.N // 2
    away = torch.tensor([not s["enemy"] for s in sp], device=st.device)
    side = u["side"].clone()
    side[:, H] = torch.where(away, torch.zeros_like(side[:, H]), side[:, H])
    u["side"] = side
    view = types.SimpleNamespace(u=u, B=st.B)
    for name in sorted({s["map"] for s in sp if s["map"] != "flat"}):
        realmap.activate(name, exam=True)
        sel = np.array([s["map"] == name for s in sp])
        gen = torch.Generator().manual_seed(seed)
        px, pz, deg, _ = realmap.places({"st": view, "area": area}, gen, np.zeros(len(sp), np.int64))
        realmap.lay(st, np.where(sel, 0, -1), px, pz, deg)
        realmap.deactivate()


def walker(st, area):
    """The enemy's script: drill_go.to_area (to the frame's centre; within GRAB_M of it, or in melee, it strikes the
    nearest of ours near it or fighting it) at a walk - only its strike runs."""
    from net.unit import drill_go as DG
    go = DG.to_area(st, area)
    go.run = go.run & (go.kind == O.ATTACK)
    return go


def in_frame(u, area, spacing):
    """([B, N], [B, N]) U2's execution of every unit: its formation in its frame (all four corners of its box - its
    front x depth at its bearing, geometry.dims - within IN_TOL_M of the frame) and its course within FACE_OK_DEG of
    the frame's; area {x, z, b, front, depth} [B, N] (or broadcasting to it)."""
    front, depth = geometry.dims(u, spacing)
    b = torch.deg2rad(u["b"])
    fx, fz, rx, rz = torch.sin(b), torch.cos(b), torch.cos(b), -torch.sin(b)
    a = {k: area[k] for k in ("x", "z", "b", "front", "depth")}
    ins = torch.ones_like(u["x"], dtype=torch.bool)
    for ka in (-0.5, 0.5):
        for kd in (-0.5, 0.5):
            cx = u["x"] + rx * ka * front + fx * kd * depth
            cz = u["z"] + rz * ka * front + fz * kd * depth
            ins = ins & (Dc.signed_dist(cx, cz, a) >= -IN_TOL_M)
    off = (torch.remainder(u["b"] - a["b"] + 180.0, 360.0) - 180.0).abs()
    return ins, off <= FACE_OK_DEG


class Ender:
    """After every step: over at the first contact of the two units (state hit_sec either way), at the rout, death or
    leaving of either, or at the skirmish's limit."""

    def __init__(self, sp, device):
        B = len(sp)
        self.limit = torch.tensor([float(s["limit"]) for s in sp], device=device)
        self.over = torch.zeros(B, dtype=torch.bool, device=device)

    def after_step(self, st, touch):
        u = st.u
        H = st.N // 2
        out = u["r"] | (u["men"] <= 0) | u["gone"]
        new = ~self.over & (touch | out[:, 0] | out[:, H] | (st.t >= self.limit) | st.done)
        self.over = self.over | new
        st.done = self.over.clone()


class Meter:
    """The measures per step (module doc) of our unit (slot 0) against the enemy (slot N // 2)."""

    def __init__(self, st, area, sp, P):
        B, dev = st.B, st.device
        self.area = {k: area[k][:, 0] for k in ("x", "z", "b", "front", "depth")}
        self.dt = float(P.dt)
        self.spacing = P.sim["formation"]["spacing_m"]
        self.front = math.radians(P.sim["contact"]["front_deg"])
        self.ready = torch.tensor([float(s["ready_s"]) for s in sp], device=dev)
        self.enemy = torch.tensor([bool(s["enemy"]) for s in sp], device=dev)
        z = lambda v=0.0: torch.full((B,), v, device=dev)       # noqa: E731
        self.ok_s, self.n_s, self.ok_all, self.n_all = z(), z(), z(), z()
        self.reach_t, self.t_contact, self.label = z(float("nan")), z(float("nan")), z(float("nan"))
        self.in_contact = z(float("nan"))

    def touch(self, st):
        B, N = st.B, st.N
        hs = st.u["hit_sec"].view(B, N, N)
        return (hs[:, 0, N // 2] > 0) | (hs[:, N // 2, 0] > 0)

    def ok(self, st, facing=True):
        """[B] our formation in the frame (its box's corners) and (facing) within FACE_OK_DEG of the frame's course."""
        ins, face = in_frame(st.u, {k: v[:, None] for k, v in self.area.items()}, self.spacing)
        return ins[:, 0] & (face[:, 0] | (not facing))

    def step(self, st, was_done, touch):
        u = st.u
        B, N = st.B, st.N
        H = N // 2
        live = ~was_done & (u["men"][:, 0] > 0) & ~u["gone"][:, 0] & ~u["r"][:, 0]
        ok = self.ok(st)
        scored = live & ~u["m"][:, 0] & ~touch
        after = scored & (st.t >= self.ready)
        self.ok_all += (scored & ok).float() * self.dt
        self.n_all += scored.float() * self.dt
        self.ok_s += (after & ok).float() * self.dt
        self.n_s += after.float() * self.dt
        self.reach_t = torch.where(live & ok & torch.isnan(self.reach_t), st.t.float(), self.reach_t)
        first = ~was_done & touch & torch.isnan(self.t_contact)
        if bool(first.any()):
            dx, dz = u["x"][:, H] - u["x"][:, 0], u["z"][:, H] - u["z"][:, 0]
            rel = torch.remainder(torch.atan2(dx, dz) - torch.deg2rad(u["b"][:, 0]) + math.pi, 2 * math.pi) - math.pi
            self.t_contact = torch.where(first, st.t.float(), self.t_contact)
            self.label = torch.where(first, (rel.abs() > self.front).float(), self.label)    # 0 front, 1 not
            self.in_contact = torch.where(first, self.ok(st, facing=False).float(), self.in_contact)

    def result(self, st, gate=None):
        nan = lambda v: torch.full_like(v, float("nan"))       # noqa: E731
        met = torch.where(torch.isfinite(self.label), (self.label == 0).float(), torch.zeros_like(self.label))
        out = {"head_on": torch.where(self.enemy, met, nan(met)),
               "contact": torch.where(self.enemy, torch.isfinite(self.t_contact).float(), nan(met)),
               "exec": torch.where(self.n_s > 0, self.ok_s / self.n_s.clamp(min=1e-6), nan(self.n_s)),
               "exec_all": torch.where(self.n_all > 0, self.ok_all / self.n_all.clamp(min=1e-6), nan(self.n_all)),
               "reached": torch.isfinite(self.reach_t).float(), "reach_s": self.reach_t,
               "in_frame_contact": torch.where(self.enemy, torch.nan_to_num(self.in_contact, 0.0), nan(met)),
               "t_contact": self.t_contact, "t_end": st.t.float().clone()}
        if gate is not None:
            own = torch.zeros_like(st.u["side"], dtype=torch.bool)
            own[:, 0] = True
            out["orders_min"] = gate.per_minute(own)
            out["raw_orders_min"] = gate.per_minute(own, raw=True)
        return out


def play(sp, player, P, device, greedy=False, still=True, record=(), compile=None):
    """{measure: [B] (host)} of the skirmishes sp with my unit played by player (a scripted one, net/unit/scripts.py, or
    the unit network: its rules of the chain - still / aim - as in the game, lower.Units); record: skirmishes whose
    frames (game/vision/record.sim_frame, a second each until its end) come back in out["_frames"] {b: [frame]}."""
    from net.matchup import duel as DU
    from net.operative import lower as LW
    from net.unit import ckpt as UC
    from net.unit import net as Nt
    from sim import scripts as opponents
    st, own, area = build(sp, P, device)
    B, N = st.B, st.N
    setup = DU._SideSetup(st, P, device, 1, UC.cards(st, device).float())
    scripted = hasattr(player, "intend")
    units = LW.Units(setup, player, unit_greedy=greedy, still=still and not scripted, aim=still and not scripted,
                     gate=True)
    units.area = {**{k: area[k].clone() for k in ("x", "z", "b", "front", "depth")}, "sb": area["b"].clone(),
                  "sw": torch.zeros((B, N), device=device), "on": own.clone()}
    pace = {"any": Nt.PACE_ANY, "slow": Nt.PACE_SLOW, "fast": Nt.PACE_FAST}
    units.pace = torch.tensor([pace[s["pace"]] for s in sp], device=device)[:, None].expand(B, N).clone()
    coming = torch.tensor([bool(s["enemy"]) for s in sp], device=device)[:, None].expand(B, N)
    ender = Ender(sp, device)
    meter = Meter(st, area, sp, P)
    sim_step = battle.stepper(device, compile)
    steps = round(1.0 / P.dt)
    st.done = torch.zeros(B, dtype=torch.bool, device=device)
    frames = {b: [] for b in record}

    def enemy(s_):
        return O.merge(opponents.hold(s_), walker(s_, area), coming)
    for _ in range(int(max(s["limit"] for s in sp)) + 2):
        if frames:
            from game.vision import record as RC
            for b in frames:
                if not bool(st.done[b]):
                    frames[b].append(RC.sim_frame(st, b))
        if bool(st.done.all()):
            break
        lands = units.second(st)
        for j in range(steps):
            was = st.done.clone()
            o = units.orders(st, j, lands, enemy)
            sim_step(st, o, P)
            touch = meter.touch(st)
            ender.after_step(st, touch)
            meter.step(st, was, touch)
        units.end_second()
    out = {k: v.detach().cpu() for k, v in meter.result(st, units.gate).items()}
    if frames:
        out["_frames"] = frames
    return out
