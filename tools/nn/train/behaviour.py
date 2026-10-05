"""What every unit does in a batch of battles, per decision (docs/en/training/training.md).

facts() reads the simulator's state after a step and says, per unit [B, N]:

    melee          standing and in melee
    flanked        ... and struck from the flank or rear now (the simulator's flank_hit >= 1;
                   the game: a flank attack costs the defender ~x1.74 losses)
    missile_melee  a missile unit (range > 0, not a lord) standing in melee
    flank_attack   standing, in melee, its target a standing enemy it reaches from that enemy's
                   flank or rear (the angle off the target's facing, contact.front_deg / rear_deg)
    crowded        one of more than `crowd` own units on the same enemy (its attack order, or the
                   enemy it fights) while another standing enemy strikes an own unit in flank or rear;
                   as the share of the excess, (n - crowd) / n, so the whole pile weighs n - crowd
    aim            the enemy slot it attacks or fights (-1 none)

The test protocol's metrics (Tracker, tools/nn/train/test5.py) read it.

Liveliness (Tracker, measured only; docs/en/training/training.md "Liveliness"): units that change
their orders too often look artificial. From the order in force after each step (order_kind,
order_target, ox / oz), per role:

    order_changes_per_min    real changes a standing unit-minute: a new kind, a new attack target or a
                             move / withdraw point more than MOVE_M away (as reward.order_changes; the
                             simulator's own ATTACK -> HOLD when the target dies is not a change)
    target_switches_per_min  attack-target switches while the old target still stands (reward.retargets)
    flips_per_min            back-and-forth: a change back to the order before the previous one
                             (A -> B -> A, a point within MOVE_M of it) within FLIP_S of that change
    move_jitter_m            mean distance between successive points of a unit that keeps moving
                             (move / withdraw -> move / withdraw, more than REPEAT_M apart: the bridge
                             does not give a nearer one in the game); move_repoints_per_min of moving time
    free_changes_per_min     order changes a minute out of melee (standing)
    twitch_share             units with at least FREE_MIN_S out of melee that change orders there
                             TWITCH_PER_MIN times a minute or more (a unit in a battle counts once)
    engine_switches_per_min  the unit's own current target (the simulator's `target`: whom it fights or
                             shoots) changing between whole seconds, as the game's recordings sample it
                             (tools/nn/gate.py: the same for the game's AI, a reference);
                             opp_engine_switches_per_min: the opponent script's units
"""
import math

import torch

from tools.nn.sim import geometry
from tools.nn.sim import orders as O

CROWD = 2          # more own units than this on one enemy is a pile
ABILITY_SLOTS = 3
MOVE_M = 10.0          # = reward.Weights.order_move_m: a nearer new point is not an order change
REPEAT_M = 5.0         # = src/apps/bridge/services.lua REPEAT_M: a nearer point is not given in the game
FLIP_S = 10.0          # A -> B -> A within this many seconds is a flip
TWITCH_PER_MIN = 6.0   # order changes a minute out of melee that make a unit "twitching"
FREE_MIN_S = 30.0      # ... over at least this many seconds out of melee


def standing_mask(u):
    return (u["side"] > 0) & (u["men"] > 0) & ~u["gone"] & ~u["r"]


def _gather(x, idx):
    return x.gather(1, idx.clamp(min=0))


def facts(st, params, crowd=CROWD):
    u = st.u
    side = u["side"]
    stand = standing_mask(u)
    melee = stand & u["m"]
    cc = params.sim["contact"]

    # Whom each unit fights, and from which side of that enemy.
    tgt = u["target"]
    t_ok = (tgt >= 0) & (_gather(side, tgt) != side) & (_gather(side, tgt) > 0) & _gather(stand, tgt)
    dx = _gather(u["x"], tgt) - u["x"]
    dz = _gather(u["z"], tgt) - u["z"]
    rel = geometry.wrap(torch.atan2(dx, dz) + math.pi - _gather(u["b"], tgt) * geometry.DEG)
    sector = geometry.sector(rel, cc["front_deg"], cc["rear_deg"])
    flank_attack = melee & t_ok & (sector >= 1)

    # Piles: own units on one enemy (an attack order counts before contact), while another flanks us.
    attack = (u["order_kind"] == O.ATTACK) & (u["order_target"] >= 0)
    aim = torch.where(attack, u["order_target"], torch.where(melee & (tgt >= 0), tgt, torch.full_like(tgt, -1)))
    aim = torch.where(stand & (aim >= 0) & (_gather(side, aim) != side), aim, torch.full_like(aim, -1))
    B, N = aim.shape
    count = torch.zeros(B, N + 1, device=aim.device).scatter_add_(1, aim + 1, torch.ones(B, N, device=aim.device))
    on_mine = _gather(count[:, 1:], aim) * (aim >= 0).float()              # [B, N] units on my aim, me included
    flankers = [(flank_attack & (side == s)).sum(1, keepdim=True) for s in (1, 2)]   # side s's units in our flank
    enemy_flankers = torch.where(side == 1, flankers[1], flankers[0])
    other = enemy_flankers - (_gather(flank_attack, aim) & (aim >= 0)).long()   # not the one we pile on
    crowded = torch.where((aim >= 0) & (on_mine > crowd) & (other > 0) & stand,
                          (on_mine - crowd) / on_mine.clamp(min=1), torch.zeros_like(on_mine))

    missile = (u["range"] > 0) & (u["ammo0"] > 0) & ~u["lord"]
    return {"melee": melee, "flanked": melee & (u["flank_hit"] >= 1), "missile_melee": melee & missile,
            "missile": stand & missile, "flank_attack": flank_attack, "crowded": crowded, "aim": aim}


def ability_marks(u):
    """[B, N, SLOTS] the abilities' cooldowns: a use sets its cooldown up (tools/nn/sim/abilities.py)."""
    return torch.stack([u.get(f"ab{k}_cd", torch.zeros_like(u["men"])) for k in range(ABILITY_SLOTS)], -1)


def track(st, params, mine, live, sums, marks, mem=None):
    """Tracker.update: adds this step to sums (in place) -> the abilities' new marks. mem: the
    liveliness memory (lively_memory, updated in place) or None (no liveliness)."""
    f = facts(st, params)
    dt = params.dt
    m = mine & live[:, None]
    add = {"steps": live.float(),
           "melee_s": (f["melee"] & m).float().sum(1) * dt,
           "flanked_s": (f["flanked"] & m).float().sum(1) * dt,
           "missile_melee_s": (f["missile_melee"] & m).float().sum(1) * dt,
           "missile_s": (f["missile"] & m).float().sum(1) * dt,
           "flank_attack_s": (f["flank_attack"] & m).float().sum(1) * dt,
           "crowd_steps": ((f["crowded"] > 0) & m).any(1).float(),
           "crowded_units": (f["crowded"] * m.float()).sum(1)}
    new = ability_marks(st.u)
    add["abilities"] = ((new > marks + 1e-3) & m[..., None]).float().sum((1, 2))
    if mem is not None:
        add.update(lively(st, mine, live, mem, dt))
    for k, v in add.items():
        sums[k] += v
    return new


def lively_memory(st):
    """The liveliness memory [B, N] (and [B]): the order in force, the one before the last change and
    when it changed, the unit's current target at the last whole second; per-unit sums out of melee."""
    u = st.u
    z = torch.zeros_like(u["x"])
    return {"k0": u["order_kind"].clone(), "t0": u["order_target"].clone(), "x0": u["ox"].clone(),
            "z0": u["oz"].clone(), "pk": torch.full_like(u["order_kind"], -1),
            "pt": torch.full_like(u["order_target"], -1), "px": z.clone(), "pz": z.clone(),
            "tc": torch.full_like(z, -1e9), "eng": torch.full_like(u["target"], -1),
            "sec": torch.floor(st.t).clone(), "free_s": z.clone(), "free_ch": z.clone()}


def lively(st, mine, live, mem, dt):
    """This step's liveliness counts {name: [B]} over mine's standing units (opp_*: the other side's);
    updates mem in place (no reads back from the GPU: it runs compiled, tools/nn/train/rollout.py fast)."""
    u = st.u
    stand = standing_mask(u)
    alive = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"]
    run = live[:, None]
    m = mine & run & stand
    k, tg, ox, oz = u["order_kind"], u["order_target"], u["ox"], u["oz"]
    k0, t0 = mem["k0"], mem["t0"]
    point = (k == O.MOVE) | (k == O.WITHDRAW)
    point0 = (k0 == O.MOVE) | (k0 == O.WITHDRAW)
    d = torch.sqrt((ox - mem["x0"]) ** 2 + (oz - mem["z0"]) ** 2)
    auto = (k0 == O.ATTACK) & (k == O.HOLD) & ~_gather(alive, t0)       # the simulator's own: the target died
    change = m & ~auto & ((k != k0) | ((k == O.ATTACK) & (tg != t0)) | (point & point0 & (d > MOVE_M)))
    switch = m & (k0 == O.ATTACK) & (k == O.ATTACK) & (t0 >= 0) & (tg != t0) & _gather(stand, t0)
    back = torch.sqrt((ox - mem["px"]) ** 2 + (oz - mem["pz"]) ** 2) <= MOVE_M
    same = (k == mem["pk"]) & torch.where(k == O.ATTACK, tg == mem["pt"], torch.where(point, back, torch.ones_like(back)))
    t = st.t[:, None].expand_as(d)
    flip = change & same & (t - mem["tc"] <= FLIP_S)
    repoint = m & point & point0 & (d > REPEAT_M)
    free = m & ~u["m"]
    # the unit's current target at whole seconds (the game's recordings sample every second)
    sec = torch.floor(st.t)
    tick = (sec > mem["sec"]) & live
    eng_now = torch.where(stand, u["target"], torch.full_like(u["target"], -1))
    eng_sw = tick[:, None] & stand & (eng_now >= 0) & (mem["eng"] >= 0) & (eng_now != mem["eng"])
    opp = (u["side"] > 0) & ~mine & run & stand
    f = (lambda x: x.float().sum(1))
    out = {"unit_s": f(m) * dt, "changes": f(change), "switches": f(switch), "flips": f(flip),
           "move_s": f(m & point) * dt, "repoints": f(repoint), "repoint_m": (d * repoint.float()).sum(1),
           "eng_switches": f(eng_sw & mine & run), "eng_s": f(m) * dt,
           "opp_eng_switches": f(eng_sw & opp), "opp_unit_s": f(opp) * dt}
    mem["free_s"] += free.float() * dt
    mem["free_ch"] += (change & free).float()
    mem["pk"].copy_(torch.where(change, k0, mem["pk"]))
    mem["pt"].copy_(torch.where(change, t0, mem["pt"]))
    mem["px"].copy_(torch.where(change, mem["x0"], mem["px"]))
    mem["pz"].copy_(torch.where(change, mem["z0"], mem["pz"]))
    mem["tc"].copy_(torch.where(change, t, mem["tc"]))
    mem["k0"].copy_(k)
    mem["t0"].copy_(tg)
    mem["x0"].copy_(ox)
    mem["z0"].copy_(oz)
    mem["eng"].copy_(torch.where(tick[:, None], eng_now, mem["eng"]))
    mem["sec"].copy_(torch.where(tick, sec, mem["sec"]))
    return out


class Tracker:
    """Per battle [B] sums over the learner's units (mine [B, N]) while the battle runs: seconds in
    melee, flanked, missile units in melee, flank attacks; decisions with a pile; ability uses."""
    FIELDS = ("steps", "melee_s", "flanked_s", "missile_melee_s", "missile_s", "flank_attack_s", "crowd_steps",
              "crowded_units", "abilities")
    LIVELY = ("unit_s", "changes", "switches", "flips", "move_s", "repoints", "repoint_m", "eng_switches", "eng_s",
              "opp_eng_switches", "opp_unit_s")

    def __init__(self, st, params, mine, wrap=None, lively=True):
        """wrap: e.g. tools/nn/train/rollout.py fast (compiles the update on CUDA). lively: also the
        liveliness counts (the module's docstring)."""
        self.params = params
        self.mine = mine
        self.dt = params.dt
        self.sums = {k: torch.zeros(st.B, device=st.device) for k in self.FIELDS + (self.LIVELY if lively else ())}
        self.marks = ability_marks(st.u)
        self.mem = lively_memory(st) if lively else None
        self._track = wrap(track) if wrap else track

    def update(self, st, live):
        """After a step; live [B]: the battle was running before it."""
        self.marks = self._track(st, self.params, self.mine, live, self.sums, self.marks, self.mem)

    def summary(self, sel):
        """Metrics over the battles sel [B] (numpy bool)."""
        s = {k: v.cpu().numpy()[sel] for k, v in self.sums.items()}
        n = max(1, int(sel.sum()))
        melee = max(1e-9, float(s["melee_s"].sum()))
        steps = max(1e-9, float(s["steps"].sum()))
        minutes = steps * self.dt / 60
        return {"missile_melee_s": float(s["missile_melee_s"].sum() / n),
                "missile_melee_share": float(s["missile_melee_s"].sum() / max(1e-9, s["missile_s"].sum())),
                "flanked_share": float(s["flanked_s"].sum() / melee),
                "flank_attack_share": float(s["flank_attack_s"].sum() / melee),
                "crowding_share": float(s["crowd_steps"].sum() / steps),
                "crowded_units_per_min": float(s["crowded_units"].sum() / minutes),
                "abilities_per_battle": float(s["abilities"].sum() / n), **self.lively(sel, s)}

    def lively(self, sel, s):
        """The liveliness metrics (the module's docstring) over the battles sel; {} without them."""
        if self.mem is None:
            return {}

        def per(k, d, scale=60.0):
            den = float(s[d].sum())
            return float(s[k].sum()) / den * scale if den > 0 else None
        free_s = self.mem["free_s"].cpu().numpy()[sel]
        free_ch = self.mem["free_ch"].cpu().numpy()[sel]
        long = free_s >= FREE_MIN_S
        rate = free_ch[long] / (free_s[long] / 60)
        return {"order_changes_per_min": per("changes", "unit_s"), "target_switches_per_min": per("switches", "unit_s"),
                "flips_per_min": per("flips", "unit_s"), "move_jitter_m": per("repoint_m", "repoints", 1.0),
                "move_repoints_per_min": per("repoints", "move_s"),
                "free_changes_per_min": float(free_ch.sum() / (free_s.sum() / 60)) if free_s.sum() > 0 else None,
                "twitch_share": float((rate >= TWITCH_PER_MIN).mean()) if long.any() else None,
                "engine_switches_per_min": per("eng_switches", "eng_s"),
                "opp_engine_switches_per_min": per("opp_eng_switches", "opp_unit_s")}


FATIGUE_LEVELS = 6        # the simulator's u["fat"]: fresh, active, winded, tired, very tired, exhausted
TIRED = 3                 # tired and worse
FAR_M = 150.0             # "far from the fight": no standing enemy this near (about a bow's range), before first contact


def far_from_fight(u, touched):
    """[B, N] units far from the fight: their battle had no melee yet (touched [B] 0 / 1) and no standing
    enemy is within FAR_M."""
    stand = standing_mask(u)
    dx = u["x"][:, :, None] - u["x"][:, None, :]
    dz = u["z"][:, :, None] - u["z"][:, None, :]
    enemy = (u["side"][:, :, None] != u["side"][:, None, :]) & stand[:, None, :]
    near = (enemy & (dx * dx + dz * dz < FAR_M * FAR_M)).any(2)
    return ~near & (touched[:, None] < 0.5)


def fatigue_step(st, mine, live, sums, dt):
    """Fatigue.update: adds this step's standing unit-seconds of mine's units by fatigue state, and of those
    far from the fight under a move / withdraw order, running or not (in place)."""
    u = st.u
    m = mine & live[:, None] & standing_mask(u)
    level = u["fat"].long().clamp(0, FATIGUE_LEVELS - 1)
    hot = torch.nn.functional.one_hot(level, FATIGUE_LEVELS).float() * m[..., None].float()
    sums["fat_s"] += hot.sum(1) * dt
    moving = m & ((u["order_kind"] == O.MOVE) | (u["order_kind"] == O.WITHDRAW)) & far_from_fight(u, sums["touched"])
    sums["far_move_s"] += moving.float().sum(1) * dt
    sums["far_run_s"] += (moving & u["order_run"].bool()).float().sum(1) * dt
    contact = (u["m"] & standing_mask(u)).any(1).float()
    sums["touched"] += contact * (1 - sums["touched"])


class Fatigue:
    """Per battle [B, 6] the learner's standing unit-seconds in each fatigue state (the simulator's
    u["fat"]) and its run share far from the fight (measured only: the "fatigue" metric profile,
    tools/nn/train/profiles.py)."""

    def __init__(self, st, params, mine, wrap=None):
        self.dt = params.dt
        self.mine = mine
        z = (lambda: torch.zeros(st.B, device=st.device))
        self.sums = {"fat_s": torch.zeros(st.B, FATIGUE_LEVELS, device=st.device), "far_move_s": z(),
                     "far_run_s": z(), "touched": z()}
        self._step = wrap(fatigue_step) if wrap else fatigue_step

    def update(self, st, live):
        self._step(st, self.mine, live, self.sums, self.dt)

    def summary(self, sel):
        """{fatigue_shares: [6] shares of the standing unit-time by state, fatigue_tired_share (tired or
        worse), fatigue_exhausted_share, run_far_share: the share of the unit-time under a move / withdraw
        order far from the fight (far_from_fight) that runs} over the battles sel [B] (numpy bool); None
        without time."""
        s = self.sums["fat_s"].cpu().numpy()[sel].sum(0)
        tot = float(s.sum())
        move = float(self.sums["far_move_s"].cpu().numpy()[sel].sum())
        run = (float(self.sums["far_run_s"].cpu().numpy()[sel].sum()) / move) if move > 0 else None
        if tot <= 0:
            return {"fatigue_shares": None, "fatigue_tired_share": None, "fatigue_exhausted_share": None,
                    "run_far_share": run}
        return {"fatigue_shares": [round(float(x) / tot, 4) for x in s],
                "fatigue_tired_share": float(s[TIRED:].sum()) / tot, "fatigue_exhausted_share": float(s[-1]) / tot,
                "run_far_share": run}
