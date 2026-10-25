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
               included) is laid on the men struck; their health, the dead. The rectangle's kill rules that cut the
               step's HP by its own dead (kills.missile_front) are off in the men's step (own_kills)
  after the step  the unit's men and health are the men's; the men walk to their places or close on the man they
               fight (move.py)
Everything else of the step runs as it is. Install with `with installed(lab=...):`; a State built inside gets its
men (st.men, the last one also HOOK.F); a copy of it (st.clone(), the step replay's probe step) runs the plain
simulator. Training (tools/nn/train/men.py; installed(adopt=False)): nothing built is adopted, the training gives its
states their men itself (Hook.attach) - every State with men steps with them (st.men: a batch, its ability branches,
an evaluation), any other plainly (compiled as battle.stepper on CUDA).
Speed: on CUDA the whole step - the simulator's and the men's - runs compiled (torch.compile, as battle.stepper; fixed
sizes, the men's draws by a counter hash, state.rnd); an outside reset (the step replay's window start) is looked for
before it, eager (one sync a step). installed(compile=False): eager.
Needs torch."""
import contextlib
import json
import math
from pathlib import Path

import torch

from tools.nn.fighters import cells, knock as FK, melee as FM, missile as FS, move as MV, scans as SC, state as S
from tools.nn.sim.compiled import compiled

LAB = json.loads((Path(__file__).with_name("lab.json")).read_text(encoding="utf-8"))
COMPILE_MEN = 100_000        # Hook._runner: the batch's men (battles x places) from which the step is compiled
DEAD_FRAC = 1e-5             # casualty_windows: a man left under this share of his health is dead (float residue)


class Near:
    """The step's neighbour pass: the men's places and the cell table; each man's neighbours (cand [B, M, K], -1 none,
    himself left out), their distances d and units c_uid - built at their first use (the GPU kernels, kernels.py,
    read the table and never build them)."""

    def __init__(self, F, u, lab):
        self.px, self.pz = S.positions(F, u)
        live = S.alive(F)
        cc = lab["cells"]
        self.cap = int(cc["cap"])
        self.uid = F.uid
        self.table = cells.Table(self.px, self.pz, live, float(cc["cell_m"]), int(cc["hash"]), cap=self.cap)
        self._cand = None

    def _lists(self):
        if self._cand is None:
            cand = self.table.query(self.px, self.pz, self.cap)
            M = self.px.shape[1]
            cand = torch.where(cand == torch.arange(M, device=cand.device)[None, :, None], torch.full_like(cand, -1),
                               cand)
            cx, cz = cells.gather2(self.px, cand), cells.gather2(self.pz, cand)
            self._d = torch.sqrt((cx - self.px[:, :, None]) ** 2 + (cz - self.pz[:, :, None]) ** 2 + 1e-9)
            self._c_uid = cells.gather2(self.uid, cand)
            self._cand = cand

    @property
    def cand(self):
        self._lists()
        return self._cand

    @property
    def d(self):
        self._lists()
        return self._d

    @property
    def c_uid(self):
        self._lists()
        return self._c_uid


def contact_pairs(F, lo, hi, N):
    """[B, N, N] the unit pairs with a man of each within reach of a man of the other, from each man's lowest and
    highest enemy unit within his reach (lo [B, M], N none; hi, -1 none: scans.contact_scan) - two scatters a man,
    not one a neighbour (a scatter over [B, M, K] is ~60 M contended atomics a step). Exact while a man touches at most
    two enemy units at once; a third between them is still found by its own men (the pairs are made symmetric)."""
    B = F.B
    some = hi >= 0
    dump = torch.full_like(lo, N * N)
    i_lo = torch.where(some, F.uid * N + lo.clamp(max=N - 1), dump)
    i_hi = torch.where(some, F.uid * N + hi.clamp(min=0), dump)
    one = torch.ones_like(lo, dtype=torch.float32)
    c = torch.zeros((B, N * N + 1), device=lo.device).scatter_add(1, i_lo, one).scatter_add(1, i_hi, one)
    c = c[:, :N * N].reshape(B, N, N) > 0
    return c | c.transpose(1, 2)


class Hook:
    def __init__(self, lab=None, params=None, seed=0, shape=False, compile=None, adopt=True, outside=True):
        self.lab = lab or LAB
        self.params = params
        self.seed = seed
        self.compile = compile       # None: compiled on CUDA (battle.stepper's rule)
        self.compiled = None
        self.st = None
        self.F = None
        self.on = False
        self.near = None
        self.contact = None
        self.listeners = []          # f(st, F) after every step of the hooked State
        self.traced = self.wanted = 0.0      # projectiles (0-dim tensors once the men shoot: read through stats)
        self.steps = 0
        self.orig = {}
        self.landings = None
        self.on_reset = None         # f(F, u, row) at a step replay's window start (places.FromRecording)
        self.shape = shape           # the units' shape for the network after every step (u shape / shape_ok); "lazy":
        #                              only when asked (shape_now: before an observation - training, tools/nn/train/men.py)
        self.shape_fn = None         # shape_now's shape_of (compiled on CUDA for a big batch)
        self.reset_every = 10
        self.last_reset = None
        self.adopt_built = adopt     # a State scenario.build makes gets its men (off in training: men.py attaches them)
        self.outside = outside       # look for an outside reset before each step (_reset; training: none, men.py)
        self.plain = None            # the plain simulator's step for a State without men (compiled on CUDA)
        self._own = {}               # id(params) -> (params, its copy for the men's step: own_kills)

    @property
    def stats(self):
        return {"traced": float(self.traced), "wanted": float(self.wanted), "steps": self.steps}

    # ---- the State ----
    def adopt(self, st, params):
        self.params = params
        self.last_reset = None
        self.attach(st, S.build(st, params, seed=self.seed, start_phase=bool(self.lab['melee'].get('start_phase'))))

    def attach(self, st, F):
        """st steps with the men F from now on (st.men; the hook's current State and men): their units' last places
        (F.x_last, z_last: _reset, _post) and the units' shape (shape=True) from them now."""
        st.men = F
        F.x_last, F.z_last = st.u["x"].clone(), st.u["z"].clone()
        self.st, self.F = st, F
        if self.shape is True:
            self._shape(st, F)

    def _reset(self, st):
        """An outside reset (the step replay's window start: the recording's places, men and health) - eager, before the
        (compiled) step: one sync a step."""
        u = st.u
        F = self.F
        jump = (torch.sqrt((u["x"] - F.x_last) ** 2 + (u["z"] - F.z_last) ** 2) > 6.0) & (u["side"] > 0)
        dif = S.differs(F, u)
        any_jump, any_dif = torch.stack([jump.any(), dif.any()]).tolist()
        if any_jump:
            db = S.wrap_deg(u["b"] - F.b_last)
            S.rotate_offsets(F, u, db, jump)
            F.b_last = torch.where(jump, u["b"], F.b_last)
            F.tgt = torch.where(jump.gather(1, F.uid), torch.full_like(F.tgt, -1), F.tgt)
        if any_dif:
            back = S.conform(F, u)
            if bool(back.any()):
                sa, sf = S.slots(F, u)
                bx, bz = S.to_world(u, F, sa, sf)
                F.dx = torch.where(back, bx, F.dx)
                F.dz = torch.where(back, bz, F.dz)
        if self.on_reset is not None:
            # the step replay's window start (st.t back on a whole recording row, every reset_every rows): the
            # recorded soldiers' places (places.py) - here, eager (a Python decision on st.t, data-dependent sizes)
            t = float(st.t.min())
            if abs(t - round(t)) < 1e-6 and round(t) % self.reset_every == 0 and round(t) != self.last_reset:
                self.last_reset = round(t)
                self.on_reset(F, u, int(round(t)))

    def _pre(self, st, params, dt):
        u = st.u
        F = self.F
        F.pend = torch.zeros_like(F.pend)
        self.near = Near(F, u, self.lab)
        # the men's contact: unit pairs with a man of each within reach (formed combat distance + the other's radius)
        B, N = u["men"].shape
        live = S.alive(F) & ~(F.down > 0)            # (a man lying keeps nobody in contact)
        combat = float(params.battle.get("entity_action_attack_formed_combat_distance", 2.5))
        lo, hi = SC.contact_scan(self.near, live, F.uid, u["side"], u["radius"], combat)
        self.contact = contact_pairs(F, lo, hi, N)

    def _post(self, st, params, dt):
        u = st.u
        F = self.F
        S.write_back(F, u)
        # a man lying (knock.py) stays where he fell while his unit moves on
        lying = (F.down > 0) & S.alive(F)
        F.dx = torch.where(lying, F.dx - (u["x"] - F.x_last).gather(1, F.uid), F.dx)
        F.dz = torch.where(lying, F.dz - (u["z"] - F.z_last).gather(1, F.uid), F.dz)
        # a man fighting an enemy man within reach is not carried by his unit's centre either (lab.json move.hold)
        held = MV.held(F, u, self.lab)
        if held is not None:
            F.dx = torch.where(held, F.dx - (u["x"] - F.x_last).gather(1, F.uid), F.dx)
            F.dz = torch.where(held, F.dz - (u["z"] - F.z_last).gather(1, F.uid), F.dz)
        # a man stopped by his own (lab.json move.stop_own) is not carried either: he does not pass through them
        blk = MV.blocked_own(F, u, self.lab, self.near) if self.near is not None else None
        if blk is not None:
            F.dx = torch.where(blk, F.dx - (u["x"] - F.x_last).gather(1, F.uid), F.dx)
            F.dz = torch.where(blk, F.dz - (u["z"] - F.z_last).gather(1, F.uid), F.dz)
        F.blk = blk
        MV.step(F, u, params, self.lab, dt, self.near)
        FK.tick(F, dt)
        F.tick = F.tick + 1
        if self.shape is True:
            self._shape(st, F)
        F.x_last, F.z_last = u["x"].clone(), u["z"].clone()
        self.near = None

    def _inner(self, st, orders, params, dt):
        """The step with the men (self.on set by the caller): compiled whole on CUDA."""
        self._pre(st, params, dt)
        out = self.orig["step"](st, orders, params, dt)
        self._post(st, params, dt)
        return out

    def _runner(self, st):
        # compiled: on CUDA for a batch of COMPILE_MEN men or more (a compile takes ~5 min - a small batch, the probes'
        # lanes, runs eager sooner)
        use = self.compile if self.compile is not None else (torch.device(st.device).type == "cuda"
                                                             and self.F.B * self.F.M >= COMPILE_MEN)
        if not use:
            return self._inner
        if self.compiled is None:
            self.compiled = compiled(self._inner)
        return self.compiled

    def _shape(self, st, F=None):
        """The units' shape from their men (fighters/shape.unit_features) into the state: u shape [B, N, 25],
        shape_ok [B, N] (a unit with living men) - the observation's SHAPE (tools/nn/model/observation.py)."""
        u = st.u
        u["shape"], u["shape_ok"] = shape_of(u, self.F if F is None else F)

    def shape_now(self, st):
        """With shape="lazy": the units' shape of st's men now into its state (as _shape after every step; the network
        reads it only at a decision, every second step at the game's cadence - each skipped one saves a sort of all the
        men and a contact scan). The same function, compiled on its own on CUDA for a batch of COMPILE_MEN men or more.
        Nothing without men or with shape True / False."""
        F = getattr(st, "men", None)
        if self.shape != "lazy" or F is None:
            return
        if self.shape_fn is None:
            use = self.compile if self.compile is not None else torch.device(st.device).type == "cuda"
            self.shape_fn = compiled(shape_of) if use else shape_of
        fn = self.shape_fn if F.B * F.M >= COMPILE_MEN else shape_of
        st.u["shape"], st.u["shape_ok"] = fn(st.u, F)

    # ---- the patched functions ----
    def _plain(self, st):
        """The plain simulator's step of a State without men: as before the hook (adopt: the lab's own runs, eager), or
        in training compiled on CUDA as battle.stepper (unless compile=False)."""
        use = self.compile if self.compile is not None else torch.device(st.device).type == "cuda"
        if self.adopt_built or not use:
            return self.orig["step"]
        if self.plain is None:
            self.plain = compiled(self.orig["step"])
        return self.plain

    def step(self, st, orders, params=None, dt=None):
        F = getattr(st, "men", None)
        if F is None:
            return self._plain(st)(st, orders, params, dt)
        self.st, self.F = st, F
        params = own_kills(params or self.params, self._own)
        dt = dt or params.dt
        if self.outside:
            self._reset(st)
        self.on = True
        try:
            out = self._runner(st)(st, orders, params, dt)
        finally:
            self.on = False
        self.steps += 1
        for f in self.listeners:
            f(st, self.F)
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

    def volley(self, u, pw, target, dt, params, contact=None, clear=None, loaded=None, catch=None, info=None):
        # (info: the rectangles' penetration factor - the men's projectiles trace their own, fighters/missile.py)
        shots, hp, hit = self.orig["volley"](u, pw, target, dt, params, contact=contact, clear=clear, loaded=loaded,
                                             catch=catch, info=info)
        if not self.on or u is not self.st.u:
            return shots, hp, hit
        hp_f, traced, wanted = FS.fire(self.F, u, params, self.lab, shots, target, self.near, dt)
        self.traced = self.traced + traced
        self.wanted = self.wanted + wanted
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
            live = S.alive(F)
            pick = FS._random_alive(F, live, torch.arange(N, device=taken.device).expand_as(taken), None,
                                    u01=S.rnd(F, N, 301))
            hp = hp.scatter_add(1, pick.clamp(min=0), -torch.where((pick >= 0) & (rest > 0), rest,
                                                                   torch.zeros_like(rest)))
            # a blow that took all a man had leaves him at 0 up to float rounding (the overkill share, the scale ~1):
            # under DEAD_FRAC of a man's health he is dead. Without it ~1 % of the living stood on with under 1e-6 of
            # their health (none between 1e-6 and 1e-4), how many depending on the rounding: the compiled step left 7 %
            # fewer dead for the same HP than the eager one, the eager ~9 % fewer than with the rule
            hp = torch.where(hp < DEAD_FRAC * S.ugather(u, F, "hp_man"), torch.zeros_like(hp), hp)
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
        if self.adopt_built:
            from tools.nn.sim.params import load
            self.adopt(st, params or load())
        return st


# the simulator's kill rules that take HP back from the step's damage by the rectangle's own count of who dies
# (own_kills): off in the men's step, where every man takes at most his health already (fighters/missile.py, melee.py)
RECT_KILLS = (("kills", "missile_front"),)


def own_kills(params, cache=None):
    """params for the men's step: the rectangle's kill rules that cut the step's HP by its own dead (RECT_KILLS) off.
    kills.missile_front lays a pair's projectile hits on the rectangle's front rank and takes the hits beyond its dead men
    back out of the HP taken; the men already lose the HP beyond a man's health, so the cut came on top: casualty_windows
    scales every man's pending HP down by it and a man struck for all his health stood on with a sliver (the ratform
    probe lanes: the dead's share of the lost HP 0.59 -> 0.01, HP a bullet on clanrats 6.83 -> 4.31, game 6.60). The
    same Params object back when nothing is on; the copy is kept (cache: id -> (params, copy)) - a stable object for the
    compiled step."""
    if not any((params.sim.get(sec) or {}).get(key) for sec, key in RECT_KILLS):
        return params
    hit = cache.get(id(params)) if cache is not None else None
    if hit is not None and hit[0] is params:
        return hit[1]
    out = params
    for sec, key in RECT_KILLS:
        if (out.sim.get(sec) or {}).get(key):
            out = out.with_cal(sec, **{key: 0})
    if cache is not None:
        cache[id(params)] = (params, out)
    return out


def shape_of(u, F):
    """(shape [B, N, 25], shape_ok [B, N]) of the units from their men F (shape.unit_features; ok: a unit with living
    men)."""
    from tools.nn.fighters import shape
    px, pz = S.positions(F, u)
    live = S.alive(F)
    N = u["men"].shape[1]
    return (shape.unit_features(px, pz, F.uid, live, S.ugather(u, F, "side").long(), u["b"], u["men0"], u["hp"], N),
            (S.unit_count(F, live, N) > 0) & (u["side"] > 0))


HOOK = None


@contextlib.contextmanager
def installed(lab=None, params=None, seed=0, shape=False, compile=None, adopt=True, outside=True):
    """Patch the simulator's functions for the State built inside (module docstring); yields the Hook. shape: the units'
    shape for the network after every step (Hook._shape: the state's shape / shape_ok, observation SHAPE); "lazy": only
    when asked (Hook.shape_now; training asks before every observation, tools/nn/train/men.py); compile:
    the step compiled (None: on CUDA for a batch of COMPILE_MEN men or more, Hook._runner). adopt: a State built
    inside gets its men (False, training: the training attaches them, Hook.attach, and any other State steps plainly,
    compiled on CUDA); outside: look for an outside reset of the units' places, men and health before each step
    (Hook._reset; training keeps its men in step itself, tools/nn/train/men.py)."""
    global HOOK
    from tools.nn.sim import battle, geometry, magic, melee, missile, morale, scenario
    h = Hook(lab, params, seed, shape=shape, compile=compile, adopt=adopt, outside=outside)
    h.orig = {"step": battle.step, "contact_limit": geometry.contact_limit, "strikes": melee.strikes,
              "volley": missile.volley, "casualty_windows": morale.casualty_windows, "build": scenario.build,
              "magic_strikes": magic.strikes, "bombard": magic.bombard}
    magic.strikes = h.magic_strikes
    # (eager inside the compiled step, as the simulator's own: the landings' sizes depend on the data)
    magic.bombard = torch.compiler.disable(h.bombard)
    battle.step = h.step
    geometry.contact_limit = h.contact_limit
    melee.strikes = h.strikes
    missile.volley = h.volley
    morale.casualty_windows = h.casualty_windows
    scenario.build = h.build
    # battle.run's stepper compiles battle.step on CUDA; the lab's step compiles itself (Hook._runner)
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


__all__ = ["installed", "Hook", "Near", "LAB", "shape_of", "own_kills", "math"]
