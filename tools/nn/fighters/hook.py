"""The fighters lab inside the simulator's step, without changing a line of it (docs/en/training/fighters_lab.md).

The simulator (tools/nn/sim) keeps everything a unit does as a whole - orders, movement of the unit's centre, morale
and routs, fatigue, abilities, when a unit fires and how many shots. The lab takes over, for the State it was built
for (scenario.build while installed):
  contact      geometry.contact_limit: two units touch when a man of one is within the formed combat distance (+ the
               other man's radius) of a man of the other - the men decide, not the rectangles
  melee        melee.strikes: the HP each unit deals each other comes from the men's blows (fighters/melee.py); the
               first strike, the opening wave and the front's fill come out of the men's clocks and places
  missiles     missile.volley: the simulator's shots, loosed by the men, each projectile traced to the man it hits
               (fighters/missile.py)
  losses       morale.casualty_windows: what the step takes from each unit (the simulator's scaling - a leaver's x1.25 -
               included) is laid on the men struck; their health, the dead
  after the step  the unit's men and health are the men's; the men walk to their places or close on the man they
               fight (move.py)
Everything else of the step runs as it is. Install with `with installed(lab=...):`; the State built inside gets its
men (HOOK.F); a copy of it (st.clone(), the step replay's probe step) runs the plain simulator.
Needs torch."""
import contextlib
import json
import math
from pathlib import Path

import torch

from tools.nn.fighters import cells, knock as FK, melee as FM, missile as FS, move as MV, state as S

LAB = json.loads((Path(__file__).with_name("lab.json")).read_text(encoding="utf-8"))


class Near:
    """The step's neighbour pass: the men's places, the cell table, each man's neighbours and their distances."""

    def __init__(self, F, u, lab):
        self.px, self.pz = S.positions(F, u)
        live = S.alive(F)
        cc = lab["cells"]
        self.table = cells.Table(self.px, self.pz, live, float(cc["cell_m"]), int(cc["hash"]))
        self.cand = self.table.query(self.px, self.pz, int(cc["cap"]))
        self.cand = torch.where(self.cand == torch.arange(F.M, device=F.hp.device)[None, :, None],
                                torch.full_like(self.cand, -1), self.cand)
        cx, cz = cells.gather2(self.px, self.cand), cells.gather2(self.pz, self.cand)
        self.d = torch.sqrt((cx - self.px[:, :, None]) ** 2 + (cz - self.pz[:, :, None]) ** 2 + 1e-9)
        del cx, cz
        self.c_uid = cells.gather2(F.uid, self.cand)


class Hook:
    def __init__(self, lab=None, params=None, seed=0):
        self.lab = lab or LAB
        self.params = params
        self.seed = seed
        self.st = None
        self.F = None
        self.on = False
        self.near = None
        self.contact = None
        self.listeners = []          # f(st, F) after every step of the hooked State
        self.stats = {"traced": 0.0, "wanted": 0.0, "steps": 0}
        self.orig = {}
        self.landings = None

    # ---- the State ----
    def adopt(self, st, params):
        self.st = st
        self.params = params
        self.F = S.build(st, params, seed=self.seed, start_phase=bool(self.lab['melee'].get('start_phase')))
        self.x_last, self.z_last = st.u["x"].clone(), st.u["z"].clone()

    def _pre(self, st, params, dt):
        u = st.u
        F = self.F
        # an outside reset (the step replay's window start: the recording's places, men and health)
        jump = (torch.sqrt((u["x"] - self.x_last) ** 2 + (u["z"] - self.z_last) ** 2) > 6.0) & (u["side"] > 0)
        db = S.wrap_deg(u["b"] - F.b_last)
        if bool(jump.any()):
            S.rotate_offsets(F, u, db, jump)
            F.b_last = torch.where(jump, u["b"], F.b_last)
            F.tgt = torch.where(jump.gather(1, F.uid), torch.full_like(F.tgt, -1), F.tgt)
        if bool(S.differs(F, u).any()):
            back = S.conform(F, u)
            if bool(back.any()):
                sa, sf = S.slots(F, u)
                bx, bz = S.to_world(u, F, sa, sf)
                F.dx = torch.where(back, bx, F.dx)
                F.dz = torch.where(back, bz, F.dz)
        F.pend = torch.zeros_like(F.pend)
        self.near = Near(F, u, self.lab)
        # the men's contact: unit pairs with a man of each within reach (formed combat distance + the other's radius)
        B, N = u["men"].shape
        live = S.alive(F) & ~(F.down > 0)            # (a man lying keeps nobody in contact)
        n = self.near
        combat = float(params.battle.get("entity_action_attack_formed_combat_distance", 2.5))
        r_c = cells.gather2(S.ugather(u, F, "radius"), n.cand)
        side_m = S.ugather(u, F, "side")
        enemy = cells.gather2(side_m, n.cand) != side_m[:, :, None]
        touch = live[:, :, None] & (n.cand >= 0) & cells.gather2(live, n.cand) & enemy & (n.d <= combat + r_c)
        pair = (F.uid[:, :, None] * N + n.c_uid).reshape(B, -1)
        c = torch.zeros((B, N * N), device=u["men"].device).scatter_add(1, pair, touch.reshape(B, -1).float())
        c = c.reshape(B, N, N) > 0
        self.contact = c | c.transpose(1, 2)

    def _post(self, st, params, dt):
        u = st.u
        F = self.F
        S.write_back(F, u)
        # a man lying (knock.py) stays where he fell while his unit moves on
        lying = (F.down > 0) & S.alive(F)
        F.dx = torch.where(lying, F.dx - (u["x"] - self.x_last).gather(1, F.uid), F.dx)
        F.dz = torch.where(lying, F.dz - (u["z"] - self.z_last).gather(1, F.uid), F.dz)
        MV.step(F, u, params, self.lab, dt, self.near)
        FK.tick(F, dt)
        self.x_last, self.z_last = u["x"].clone(), u["z"].clone()
        self.stats["steps"] += 1
        for f in self.listeners:
            f(st, F)

    # ---- the patched functions ----
    def step(self, st, orders, params=None, dt=None):
        if st is not self.st or self.F is None:
            return self.orig["step"](st, orders, params, dt)
        params = params or self.params
        dt = dt or params.dt
        self._pre(st, params, dt)
        self.on = True
        try:
            out = self.orig["step"](st, orders, params, dt)
        finally:
            self.on = False
        self._post(st, params, dt)
        return out

    def contact_limit(self, u, pw, ccal, rules=None, spacing=1.5, engage=True):
        lim = self.orig["contact_limit"](u, pw, ccal, rules, spacing, engage)
        if not self.on or not engage or u is not self.st.u:
            return lim
        return torch.where(self.contact, torch.full_like(lim, 1e6), torch.full_like(lim, -1e6))

    def strikes(self, u, pw, contact, params, charge_now, first=False, ran=None):
        out = self.orig["strikes"](u, pw, contact, params, charge_now, first=first, ran=ran)
        if not self.on or u is not self.st.u:
            return out
        dt = float(params.dt)
        hp, men = FM.resolve(self.F, u, params, self.lab, contact, pw, charge_now, dt, self.near)
        rate = hp / dt
        if first:
            return rate, out[1], out[2], men, torch.zeros_like(rate)
        return rate, out[1], out[2], men

    def volley(self, u, pw, target, dt, params, contact=None, clear=None, loaded=None, catch=None):
        shots, hp, hit = self.orig["volley"](u, pw, target, dt, params, contact=contact, clear=clear, loaded=loaded,
                                             catch=catch)
        if not self.on or u is not self.st.u:
            return shots, hp, hit
        hp_f, traced, wanted = FS.fire(self.F, u, params, self.lab, shots, target, self.near, dt)
        self.stats["traced"] += traced
        self.stats["wanted"] += wanted
        return shots, hp_f, hit

    def casualty_windows(self, u, taken, params, dt):
        if self.on and u is self.st.u:
            F = self.F
            N = u["men"].shape[1]
            pend_u = S.unit_sum(F, F.pend, N)
            scale = torch.where(pend_u > 0, taken / pend_u.clamp(min=1e-9), torch.zeros_like(taken))
            hp = F.hp - F.pend * scale.gather(1, F.uid)
            # what no man took (another source: a spell's bombardment) falls on one living man a unit at random
            rest = torch.where(pend_u <= 0, taken, torch.zeros_like(taken))
            if bool((rest > 0).any()):
                live = S.alive(F)
                pick = FS._random_alive(F, live, torch.arange(N, device=taken.device).expand_as(taken), F.gen)
                hp = hp.scatter_add(1, pick.clamp(min=0), -torch.where(pick >= 0, rest, torch.zeros_like(rest)))
            F.hp = torch.where(F.valid, hp.clamp(min=0), F.hp)
            F.pend = torch.zeros_like(F.pend)
        return self.orig["casualty_windows"](u, taken, params, dt)

    def magic_strikes(self, u, params, dt):
        out = self.orig["magic_strikes"](u, params, dt)
        if self.on and u is self.st.u:
            self.landings = out
        return out

    def bombard(self, u, params, dt, front, depth, noise=False, extra=None):
        self.landings = None
        out = self.orig["bombard"](u, params, dt, front, depth, noise=noise, extra=extra)
        if not self.on or u is not self.st.u:
            return out
        hp, hits = FK.bombard(self.F, u, params, self.lab, self.landings, float(dt))
        return hp, torch.zeros_like(hp), hits

    def build(self, armies, params=None, device="cpu", per_side=None):
        st = self.orig["build"](armies, params, device=device, per_side=per_side)
        from tools.nn.sim.params import load
        self.adopt(st, params or load())
        return st


HOOK = None


@contextlib.contextmanager
def installed(lab=None, params=None, seed=0):
    """Patch the simulator's functions for the State built inside (module docstring); yields the Hook."""
    global HOOK
    from tools.nn.sim import battle, geometry, magic, melee, missile, morale, scenario
    h = Hook(lab, params, seed)
    h.orig = {"step": battle.step, "contact_limit": geometry.contact_limit, "strikes": melee.strikes,
              "volley": missile.volley, "casualty_windows": morale.casualty_windows, "build": scenario.build,
              "magic_strikes": magic.strikes, "bombard": magic.bombard}
    magic.strikes = h.magic_strikes
    magic.bombard = h.bombard
    battle.step = h.step
    geometry.contact_limit = h.contact_limit
    melee.strikes = h.strikes
    missile.volley = h.volley
    morale.casualty_windows = h.casualty_windows
    scenario.build = h.build
    # battle.run's stepper compiles battle.step on CUDA; the lab's step runs eager (its Python control flow)
    had = battle._COMPILED.get("step")
    battle._COMPILED["step"] = h.step
    HOOK = h
    try:
        yield h
    finally:
        battle.step = h.orig["step"]
        geometry.contact_limit = h.orig["contact_limit"]
        melee.strikes = h.orig["strikes"]
        missile.volley = h.orig["volley"]
        morale.casualty_windows = h.orig["casualty_windows"]
        scenario.build = h.orig["build"]
        magic.strikes = h.orig["magic_strikes"]
        magic.bombard = h.orig["bombard"]
        if had is None:
            battle._COMPILED.pop("step", None)
        else:
            battle._COMPILED["step"] = had
        HOOK = None


__all__ = ["installed", "Hook", "Near", "LAB", "math"]
