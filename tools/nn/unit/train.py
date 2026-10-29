"""The unit level's own small training loop (build/diag_g49/plan_unit_net.md step 6): decisions -> simulator steps ->
reward -> PPO (tools/nn/train/ppo.py gae / policy_loss / explained_variance). Not run.py: none of its chain of heads,
general, ability branches or pools is needed here.

    bash tools/nn/dock.sh tools.nn.unit.train --battles 256 --seconds 120 --updates 5 [--device cuda --gpu-duty 0.7]

The smoke battle (the code works; not a drill): Empire spearmen (the network, one unit) against clanrat spearmen under
the script `nearest`, 150 m apart face to face; the spearmen's area a 100 x 100 m square round their start; the order:
any pace, in place. A decision a second (two simulator steps of 0.5 s), the orders at once (no latency here), the leash
every step. The reward: the side's reward.step (gold trade, the end) - no term for the area (a ban, not a penalty).
Stored per decision: only the raw state (view.raw) and the choices; the views are rebuilt in the update (a stored view
would not fit, plan section 3).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tools.nn.model import passport
from tools.nn.sim import battle, scenario
from tools.nn.sim import orders as O
from tools.nn.sim.params import load
from tools.nn.train import cadence, opponents, ppo, reward
from tools.nn.unit import decode as Dc
from tools.nn.unit import net as Nt
from tools.nn.unit import view as V

ME, FOE = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_clanrat_spearmen_0"
KIND = "unit_v1"
FACE_CHANGE_DEG = 22.5   # a facing order this far from the one in force is a change (main training has no facing)
WIDTH_CHANGE_M = 1.0     # a width order this far (m, snapped) from the one in force is a change: any file (h >= 1.4 m)
AREA_REWARD = 0.001     # per second the area is ours / the enemy's (U2): 150 s ~ 0.15 of the budget, the trade ~0.4


class Duty:
    """Keep the GPU busy at most `duty` of the time (the user may play on it): after each piece of work, sync and sleep
    (1 - duty) / duty of its length (as --gpu-duty in the main training)."""

    def __init__(self, device, duty=1.0):
        self.cuda = torch.device(device).type == "cuda"
        self.duty = duty
        self.t0 = time.time()
        self.slept = 0.0

    def start(self):
        self.t0 = time.time()

    def end(self):
        if self.cuda:
            torch.cuda.synchronize()
        if self.cuda and self.duty < 1.0:
            s = (time.time() - self.t0) * (1 - self.duty) / self.duty
            time.sleep(s)
            self.slept += s


def smoke_battles(B, P, device, gap_m=150.0, area_m=100.0):
    """B copies of the smoke battle: the state, the network's slots (own [B, N]) and their areas."""
    desc = {"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire", "units": [{"key": ME, "x": 0.0, "z": -gap_m / 2,
                                                                                       "b": 0.0}]},
                                     2: {"faction": "wh2_main_skv_skaven", "units": [{"key": FOE, "x": 0.0, "z": gap_m / 2,
                                                                                       "b": 180.0}]}}}
    st = scenario.build([desc] * B, P, device=device)
    own = torch.zeros((B, st.N), dtype=torch.bool, device=device)
    own[:, 0] = True
    area = {"x": torch.zeros((B, st.N), device=device), "z": torch.full((B, st.N), -gap_m / 2, device=device),
            "b": torch.zeros((B, st.N), device=device), "front": torch.full((B, st.N), area_m, device=device),
            "depth": torch.full((B, st.N), area_m, device=device), "on": own.clone()}
    return st, own, area


def cards(st, device):
    return torch.from_numpy(np.stack([passport.table(r) for r in st.keys])).to(device)


def history(raws, d, T=V.T):
    """[T, ..., RAW] the T raw snapshots before decision d of a stored sequence raws [D, ...] (newest first; before the
    start: empty, alive 0)."""
    out = torch.zeros((T,) + raws.shape[1:], device=raws.device)
    for k in range(T):
        if d - 1 - k >= 0:
            out[k] = raws[d - 1 - k]
    return out


def order_changes(new, old, bi, ii, move_m=reward.Weights.order_move_m, face_deg=FACE_CHANGE_DEG, widths=None,
                  width_m=WIDTH_CHANGE_M):
    """[R] bool: the network's own order (decode.to_orders, before the leash) really changes from its previous one, as
    the main training counts it (reward.order_changes: another kind, another attack target, a move / withdraw point
    more than move_m from the previous one) and a facing face_deg or more from the previous facing (or one given / taken
    away), and a width order (not 'keep') whose width differs from the one in force by more than width_m (widths:
    (in force, ordered) [R] - widths_of; None: widths not counted). The leash's replacements are the decoder's, not the
    network's: they are never compared."""
    k, k0 = new.kind[bi, ii], old.kind[bi, ii]
    new_kind = k != k0
    new_target = (k == O.ATTACK) & (new.target[bi, ii] != old.target[bi, ii])
    point = (k == O.MOVE) | (k == O.WITHDRAW)
    moved = point & ((new.x[bi, ii] - old.x[bi, ii]) ** 2 + (new.z[bi, ii] - old.z[bi, ii]) ** 2 > move_m ** 2)
    f, f0 = new.face[bi, ii], old.face[bi, ii]
    turned = ((f >= 0) != (f0 >= 0)) | ((f >= 0) & (f0 >= 0) & ((torch.remainder(f - f0 + 180, 360) - 180).abs() >= face_deg))
    wide = torch.zeros_like(turned)
    if widths is not None:
        force, goal = widths
        wide = (new.width[bi, ii] >= 0) & ((goal - force).abs() > width_m)
    return (k != O.KEEP) & (new_kind | new_target | moved | turned | wide)


def widths_of(st, P, intended, bi, ii):
    """([R], [R]) the width in force of each network unit (the one being formed, else its own) and the width its order
    intended would form (reform.snap: whole files within its limits), m - order_changes' widths."""
    from tools.nn.sim import reform
    u = st.u
    rf = P.sim["formation"].get("reform") or {}
    force = torch.where(u["wnew"] >= 0, u["wnew"], u["width"])
    goal = reform.snap(u, intended.width.clamp(min=0), rf, P.sim["formation"]["spacing_m"])
    return force[bi, ii], goal[bi, ii]


def prio_now(raw_d, prio, bi, ii):
    """[R] the main target named by each unit's order (drill U8: prio [B, N] the slot, -1 none) while it is still there
    (alive, not gone: view.raw 'alive'), else -1; raw_d [B, N, RAW] the raw state now, prio None: none anywhere."""
    if prio is None:
        return torch.full_like(ii, -1)
    p = prio[bi, ii]
    alive = raw_d[bi, p.clamp(min=0), V.I["alive"]] > 0
    return torch.where((p >= 0) & alive, p, torch.full_like(p, -1))


def views(raws, d_idx, b_idx, i_idx, card, area, cfg, everyone=False, prio=None, wlim=None):
    """The views of the samples (d_idx, b_idx, i_idx: decision, battle, the network's unit's slot) rebuilt from the stored
    raw states - one viewer a sample; v["bi"] is set to the battles (to_orders writes the orders there). prio [B, N]
    (drill U8): the order's main target - its row's 'my_target' flag; v["prio"] [M] its slot now (prio_now).
    wlim: decode.width_limits [B, N] - v["width_ok"] [M, WIDTHS], the width head's ban (a line wider than the area's
    front; decode.width_ok); None: no ban. v["shooter"] [M]: the viewer has a missile weapon (obedience_parts)."""
    now = raws[d_idx, b_idx]                                              # [M, N, RAW]
    T = V.T
    hist = torch.zeros((T,) + now.shape, device=now.device)
    for k in range(T):
        d = d_idx - 1 - k
        ok = d >= 0
        hist[k] = torch.where(ok[:, None, None], raws[d.clamp(min=0), b_idx], torch.zeros((), device=now.device))
    ar = {k: a[b_idx] for k, a in area.items()}
    M = now.shape[0]
    m_idx = torch.arange(M, device=now.device)
    who = torch.zeros(now.shape[:2], dtype=torch.bool, device=now.device)
    who[m_idx, i_idx] = True
    p = prio_now(now, None if prio is None else prio[b_idx], m_idx, i_idx)
    tgt = torch.full(now.shape[:2], -1, dtype=torch.long, device=now.device)
    tgt[m_idx, i_idx] = p
    # the map's cell: infantry cfg cell_m, a shooter its range / view.RANGE_CELLS (its range circle on the map)
    cm = V.cell_size(now[m_idx, i_idx], cfg["cell_m"])
    v = V.build(now, hist, card[b_idx], who, order_target=tgt, area=ar, cell_m=cm, bounds=cfg["bounds"],
                everyone=everyone)
    v["bi"] = b_idx
    v["prio"] = p
    v["cell_m"] = cm
    v["shooter"] = now[m_idx, i_idx, V.I["range"]] > 0
    if wlim is not None:
        v["width_ok"] = Dc.width_ok(wlim[0][b_idx, i_idx], wlim[1][b_idx, i_idx],
                                    None if area is None else area["front"][b_idx, i_idx])
    return v


def viewer_ok(st, bi, ii):
    """[R] the network's units still deciding: alive, not gone, not routing (a router's orders are ignored)."""
    u = st.u
    return (u["men"][bi, ii] > 0) & ~u["gone"][bi, ii] & ~u["r"][bi, ii]


def smoke_setup(cfg, P, device, gen=None):
    st, own, area = smoke_battles(cfg["battles"], P, device)
    return {"st": st, "own": own, "area": area, "ender": None}


def hold_setup(cfg, P, device, gen=None, sp=None):
    """Drill U1 (drill_hold.py): cfg battles random skirmishes (or the given specs sp)."""
    from tools.nn.unit import drill_hold as DH
    sp = sp if sp is not None else DH.specs(cfg["battles"], gen or torch.Generator())
    st, own, area = DH.build(sp, P, device)
    return {"st": st, "own": own, "area": area, "ender": DH.Ender(st.B, device), "specs": sp}


def go_setup(cfg, P, device, gen=None, sp=None):
    """Drill U2 (drill_go.py): reach the area and hold it; the enemy goes to the area too."""
    from tools.nn.unit import drill_go as DG
    sp = sp if sp is not None else DG.specs(cfg["battles"], gen or torch.Generator())
    st, own, area, pace = DG.build(sp, P, device)
    return {"st": st, "own": own, "area": area, "ender": DG.Ender(sp, device), "specs": sp, "pace": pace,
            "script": lambda s_: DG.to_area(s_, area)}


def orders_now(raw_d, area, pace, arrived, bi, ii, prio=None):
    """[R] orders (net.order_of) of the network's units (battle bi, slot ii) from the raw state raw_d [B, N, RAW]; prio
    [B, N] the main target named (drill U8: has_target while it is there, prio_now)."""
    return Nt.order_of(raw_d[bi, ii], {k: v[bi, ii] for k, v in area.items() if k != "on"}, pace[bi], arrived[bi],
                       has_target=None if prio is None else prio_now(raw_d, prio, bi, ii) >= 0)


def obedience_parts(v, act):
    """Drill U8, every reading of a decision ([R] bool each): allowed / obeyed - the order's main target (v["prio"]) is
    now among the allowed targets (v["target_ok"]: infantry - in its area, a shooter - in its sector and range) and,
    for a shooter (v["shooter"]), it is not itself in melee (it cannot fire there) and the target does not rout (its
    order is met: the exam's 'down'); obeyed - the network's target is it; contested / obeyed_contested - allowed with
    another enemy allowed too (a real choice: alone, the only target is taken ~always); allowed_old / obeyed_old - the
    rule before (no melee and rout exceptions), for the comparison. build/diag_g49/u9/duel_why.json: of the duel's
    misses 5 % in melee, 20 % at a routing shooter, 59 % our own choice of their infantry."""
    p = v["prio"]
    is_p = (v["slot"] == p[:, None]) & (p[:, None] >= 0)
    old = (is_p & v["target_ok"]).any(1)
    me_melee = v["rows"][:, 0, V.ROW.index("m")] > 0
    p_rout = (is_p & (v["rows"][..., V.ROW.index("r")] > 0)).any(1)
    shooter = v.get("shooter", torch.zeros_like(old))
    allowed = old & ~(shooter & (me_melee | p_rout))
    other = (v["target_ok"] & ~is_p & (v["slot"] >= 0)).any(1)
    K = v["slot"].shape[1]
    t = act["target"]
    chosen = torch.where(t < K, v["slot"].gather(1, t.clamp(max=K - 1)[:, None])[:, 0], torch.full_like(t, -1))
    hit = chosen == p
    return {"allowed": allowed, "obeyed": allowed & hit, "contested": allowed & other,
            "obeyed_contested": allowed & other & hit, "allowed_old": old, "obeyed_old": old & hit}


def obedience(v, act):
    """Drill U8: (allowed [R], obeyed [R]) of obedience_parts - the cost's and the exam's main reading."""
    o = obedience_parts(v, act)
    return o["allowed"], o["obeyed"]


def multi_setup(cfg, P, device, gen=None, sp=None):
    """Drill U3 (drill_multi.py): hold the area against two enemies; the branches' kit of four variants."""
    from tools.nn.unit import branch as Br
    from tools.nn.unit import drill_multi as DM
    sp = sp if sp is not None else DM.specs(cfg["battles"], gen or torch.Generator())
    st, own, area = DM.build(sp, P, device)
    return {"st": st, "own": own, "area": area, "ender": DM.Ender(st.B, device), "specs": sp, "kit": Br.KIT_MULTI}


def pair_setup(cfg, P, device, gen=None, sp=None):
    """Drill U4 (drill_pair.py): two of our units side by side, each with its own area, both commanded by the network
    (the side's reward shared by both)."""
    from tools.nn.unit import drill_pair as DP
    sp = sp if sp is not None else DP.specs(cfg["battles"], gen or torch.Generator())
    st, own, area = DP.build(sp, P, device)
    return {"st": st, "own": own, "area": area, "ender": DP.Ender(st.B, device), "specs": sp}


def shoot_setup(cfg, P, device, gen=None, sp=None):
    """Drill U5 (drill_shoot.py): a shooter in its position with a fire sector; one infantry enemy goes at it or past."""
    from tools.nn.unit import drill_go as DG
    from tools.nn.unit import drill_shoot as DS
    sp = sp if sp is not None else DS.specs(cfg["battles"], gen or torch.Generator())
    st, own, area, goal = DS.build(sp, P, device)
    return {"st": st, "own": own, "area": area, "ender": DS.Ender(st.B, device), "specs": sp,
            "script": lambda s_: DG.to_area(s_, goal, grab_m=DS.GRAB_M)}


def withdraw_setup(cfg, P, device, gen=None, sp=None):
    """Drill U6 (drill_withdraw.py): in melee, ordered to a new area behind or aside at a random pace."""
    from tools.nn.unit import drill_withdraw as DW
    sp = sp if sp is not None else DW.specs(cfg["battles"], gen or torch.Generator())
    st, own, area, pace = DW.build(sp, P, device)
    return {"st": st, "own": own, "area": area, "ender": DW.Ender(sp, st), "specs": sp, "pace": pace}


def shoot2_setup(cfg, P, device, gen=None, sp=None):
    """Drill U5b (drill_shoot2.py): two shooters side by side / in file, or a shooter behind our infantry; 1-2 enemies
    going at our front unit."""
    from tools.nn.unit import drill_go as DG
    from tools.nn.unit import drill_pair as DP
    from tools.nn.unit import drill_shoot2 as D2
    sp = sp if sp is not None else D2.specs(cfg["battles"], gen or torch.Generator())
    st, own, area, goal = D2.build(sp, P, device)
    return {"st": st, "own": own, "area": area, "ender": DP.Ender(st.B, device), "specs": sp,
            "script": lambda s_: DG.to_area(s_, goal, grab_m=D2.GRAB_M)}


def rout_setup(cfg, P, device, gen=None, sp=None):
    """Drill U7 (drill_rout.py): hold the area against an enemy breaking in melee or fleeing across our area."""
    from tools.nn.unit import drill_rout as DR
    sp = sp if sp is not None else DR.specs(cfg["battles"], gen or torch.Generator())
    st, own, area = DR.build(sp, P, device)
    return {"st": st, "own": own, "area": area, "ender": DR.Ender(st), "specs": sp, "script": DR.script}


def prio_setup(cfg, P, device, gen=None, sp=None):
    """Drill U8 (drill_prio.py): a shooters' duel or infantry against two; the order names a main target in 70 % of the
    skirmishes (prio [B, N])."""
    from tools.nn.unit import drill_multi as DM
    from tools.nn.unit import drill_prio as DQ
    sp = sp if sp is not None else DQ.specs(cfg["battles"], gen or torch.Generator())
    st, own, area, goal, prio, duel = DQ.build(sp, P, device)
    return {"st": st, "own": own, "area": area, "ender": DM.Ender(st.B, device), "specs": sp, "prio": prio,
            "script": lambda s_: DQ.script(s_, goal, duel)}


def width_setup(cfg, P, device, gen=None, sp=None):
    """Drill U9 (drill_width.py): hold an area 20-80 m wide against one enemy or a line of two of random widths; our
    line starts at a random width."""
    from tools.nn.unit import drill_multi as DM
    from tools.nn.unit import drill_width as DW
    sp = sp if sp is not None else DW.specs(cfg["battles"], gen or torch.Generator())
    st, own, area = DW.build(sp, P, device)
    return {"st": st, "own": own, "area": area, "ender": DM.Ender(st.B, device, DW.LIMIT_S), "specs": sp}


def shootw_setup(cfg, P, device, gen=None, sp=None):
    """Drill U5c (drill_shootw.py): the shooter's width - into a gap between our infantry, at a slant, or in the open;
    1-2 enemies (infantry coming, shooters standing)."""
    from tools.nn.unit import drill_multi as DM
    from tools.nn.unit import drill_shootw as DW5
    sp = sp if sp is not None else DW5.specs(cfg["battles"], gen or torch.Generator())
    st, own, area, goal = DW5.build(sp, P, device)
    stand = DW5.shooter_slots(sp, st)
    return {"st": st, "own": own, "area": area, "ender": DM.Ender(st.B, device, DW5.LIMIT_S), "specs": sp,
            "script": lambda s_: DW5.script(s_, goal, stand)}


def height_setup(cfg, P, device, gen=None, sp=None):
    """Drill U10 (drill_height.py, a real map only): hold an area with a rise in it / a shooter's position on a slope,
    the enemy's infantry from above, below or across; the drill chooses its places on the map."""
    from tools.nn.unit import drill_height as DT
    sp = sp if sp is not None else DT.specs(cfg["battles"], gen or torch.Generator())
    st, own, area, goal, places = DT.build(sp, P, device, gen)
    shoot = torch.tensor([s["kind"] == "shoot" for s in sp], device=device)
    return {"st": st, "own": own, "area": area, "ender": DH_Ender(st.B, device), "specs": sp, "map_place": places,
            "script": lambda s_: DT.script(s_, goal, shoot)}


def DH_Ender(B, device):                   # noqa: N802  (drill_hold's end rule)
    from tools.nn.unit import drill_hold as DH
    return DH.Ender(B, device)


def _on_map(fn):
    """A drill's setup laid on the active real map (realmap.apply: each skirmish at a random place and turn of it, its
    own coordinates kept; the exams' fixed places); as it is without --map."""
    def setup(cfg, P, device, gen=None, **kw):
        from tools.nn.unit import realmap
        return realmap.apply(fn(cfg, P, device, gen, **kw), gen)
    setup.__name__, setup.__doc__ = fn.__name__, fn.__doc__
    return setup


smoke_setup, hold_setup, go_setup, multi_setup, pair_setup = (_on_map(f) for f in (
    smoke_setup, hold_setup, go_setup, multi_setup, pair_setup))
shoot_setup, withdraw_setup, shoot2_setup, rout_setup, prio_setup, width_setup, shootw_setup = (
    _on_map(f) for f in (shoot_setup, withdraw_setup, shoot2_setup, rout_setup, prio_setup, width_setup, shootw_setup))

SETUPS = {"smoke": smoke_setup, "hold": hold_setup, "go": go_setup, "multi": multi_setup, "hold2": pair_setup,
          "shoot1": shoot_setup, "withdraw": withdraw_setup, "shoot2": shoot2_setup, "rout": rout_setup,
          "prio": prio_setup, "width": width_setup, "shootw": shootw_setup, "height": height_setup}
MIX = ("hold", "go", "multi", "hold2", "shoot1", "withdraw", "shoot2", "rout", "prio", "width", "shootw", "height")   # --drill mix: one a batch, in turn (height: on a --map only)
MAP_ONLY = ("height",)     # drills that need a real map (--map): left out of a flat mix
# per second the named main target is allowed and the network's target is not it (drill U8). The reference network
# (u7/run2) disobeys ~32 of ~83 allowed seconds a skirmish: 0.02 costs it ~0.6 against the reward's |sum| ~0.9 a
# skirmish (the advantages' spread 0.17 a decision); on the CPU, 6 updates of 128 skirmishes from it: 0.005 moved the
# exam's obedience 0.615 -> 0.629, 0.02 -> 0.713 (build/diag_g49/u8_cpu)
PRIO_COST = 0.02


def rollout(net, critic, cfg, P, device, duty, setup, watch=None, greedy=False, branch=None):
    """One batch of battles played by the network (its unit) against the script `nearest`; the orders land after the
    game's latency (tools/nn/train/cadence.py: 0.36 s on average - one 0.5 s step late with probability 0.72), the
    leash every step. critic None: no values (the exam). watch(st, before_hp, step_was_done): the exam's metrics.
    branch {share, weight, lead_s, horizon_s} (drill U1, branch.py): the share of the battles copied at the approach and
    played on as 'stand' and 'attack'; the network's behaviour at the contact matching the better one earns weight x
    their trade difference at the contact's decision. Per battle results in stats["_branch"]."""
    st, own, area, ender = setup["st"], setup["own"], setup["area"], setup["ender"]
    B = st.B
    bi, ii = own.nonzero(as_tuple=True)              # the network's units: R viewers (one a battle in U1-U3)
    R = bi.shape[0]
    card = cards(st, device)
    spacing = P.sim["formation"]["spacing_m"]
    D = int(cfg["seconds"])
    steps = round(1.0 / P.dt)
    cad = cadence.Cadence()
    leash = Dc.Leash(B, st.N, device)
    raws = torch.zeros((D, B, st.N, len(V.RAW)), device=device)
    store = {k: torch.zeros((D, R), dtype=torch.long, device=device) for k in Nt.HEADS}
    lp_old = torch.zeros((D, R), device=device)
    val = torch.zeros((D, R), device=device)
    rew = torch.zeros((D, R), device=device)
    done = torch.zeros((D, R), dtype=torch.bool, device=device)
    valid = torch.zeros((D, R), dtype=torch.bool, device=device)
    stats = {"bans_broken": 0, "leash_s": 0.0, "out_free_s": 0.0, "out_pushed_s": 0.0, "nan": 0, "order_changes": 0,
             "entropy": {k: 0.0 for k in Nt.HEADS}, "n_dec": 0}
    pace = setup.get("pace", torch.full((B,), Nt.PACE_ANY, dtype=torch.long, device=device))
    script = setup.get("script", opponents.nearest)
    prio = setup.get("prio")                  # drill U8: [B, N] the main target each unit's order names, -1 none
    if prio is not None:
        zb = lambda: torch.zeros(B, device=device)       # noqa: E731
        pr = {"allowed_s": zb(), "obeyed_s": zb(), "allowed_melee_s": zb(), "obeyed_melee_s": zb(),
              "contested_s": zb(), "obeyed_contested_s": zb(), "allowed_old_s": zb(), "obeyed_old_s": zb()}
    # the width head's choices (decode.width_limits: the men at the start, the same all the skirmish) and its ban
    wlim = Dc.width_limits(st, P)
    arrived_d = torch.ones((D, B), dtype=torch.bool, device=device)
    d_all = torch.arange(D, device=device)
    t_sim = 0.0
    sim_step = battle.stepper(device)          # compiled on CUDA (~10x), plain on the CPU
    in_force = O.hold(B, st.N, device)        # the orders in force until the new ones land
    from tools.nn.unit import realmap
    meter = realmap.Meter(B, device) if realmap.active() else None     # (on a real map: the exams' terrain readings)
    if branch is not None:
        from tools.nn.unit import branch as Br
        assert R == B and bool((ii == 0).all()), "branches: one network unit (slot 0) a battle"
        H = st.N // 2
        will = torch.rand(B, device=device) < branch["share"]
        branched = torch.zeros(B, dtype=torch.bool, device=device)
        snaps = []
        contact_d = torch.full((B,), -1, dtype=torch.long, device=device)
        beh_d = torch.full((B,), -1, dtype=torch.long, device=device)     # the decision that settled the behaviour
        beh = torch.full((B,), Br.OTHER, dtype=torch.long, device=device)
        brace = Br.bracing_deg(P)
        kit = setup.get("kit", Br.KIT1)
    for d in range(D):
        duty.start()
        valid[d] = ~st.done[bi] & viewer_ok(st, bi, ii)
        raws[d] = V.raw(st, spacing, leash=leash.on, missile=P.sim["missile"])
        with torch.no_grad():
            dd = d_all[d].expand(R)
            v = views(raws, dd, bi, ii, card, area, cfg, prio=prio, wlim=wlim)
            arrived = getattr(ender, "arrived", None)
            arrived = torch.ones(B, dtype=torch.bool, device=device) if arrived is None else arrived.clone()
            arrived_d[d] = arrived
            order = orders_now(raws[d], area, pace, arrived, bi, ii, prio=prio)
            lg = net(v, order, v["cell_m"])
            act = Nt.sample(lg, greedy=greedy)
            lp, ent = Nt.log_prob(lg, act)
            if critic is not None:
                vc = views(raws, dd, bi, ii, card, area, cfg, everyone=True, prio=prio)
                val[d] = critic(vc, order, vc["cell_m"])
        live = valid[d]
        if prio is not None:
            # obedience (drill U8): the named target allowed now and the network's target another one (or nobody)
            op = {k_: m_ & live for k_, m_ in obedience_parts(v, act).items()}
            allow, obey = op["allowed"], op["obeyed"]
            melee = st.u["m"][bi, ii]
            for k_, m_ in (("allowed_s", allow), ("obeyed_s", obey), ("allowed_melee_s", allow & melee),
                           ("obeyed_melee_s", obey & melee), ("contested_s", op["contested"]),
                           ("obeyed_contested_s", op["obeyed_contested"]), ("allowed_old_s", op["allowed_old"]),
                           ("obeyed_old_s", op["obeyed_old"])):
                pr[k_].index_add_(0, bi, m_.float())
            disobey = (allow & ~obey).float()
        stats["n_dec"] += int(live.sum())
        for k in Nt.HEADS:
            dk = torch.distributions.Categorical(logits=lg[k])
            p = dk.probs
            e = -(torch.where(p > 0, p * dk.logits, torch.zeros_like(p))).sum(1)
            stats["entropy"][k] += float(e[live].sum())
        stats["nan"] += int(sum(torch.isnan(lg[k]).sum() for k in Nt.HEADS))
        # the bans (the network's masks): never a cell or a target outside them
        stats["bans_broken"] += int((~v["cells_ok"].flatten(1).gather(1, act["where"][:, None])[:, 0] & live).sum())
        t_ok = torch.cat([v["target_ok"], torch.ones((R, 1), dtype=torch.bool, device=device)], 1)
        stats["bans_broken"] += int((~t_ok.gather(1, act["target"][:, None])[:, 0] & live).sum())
        stats["bans_broken"] += int((~v["width_ok"].gather(1, act["width"][:, None])[:, 0] & live).sum())
        for k in Nt.HEADS:
            store[k][d] = act[k]
        lp_old[d] = lp
        intended = Dc.to_orders(act, v, raws[d], v["cell_m"], B, st.N, wlim=wlim)
        wid = widths_of(st, P, intended, bi, ii)               # (the width in force before this decision's orders)
        lands = cad.delays(P.dt, B, device)                   # [B] the step this decision's orders land at
        r = torch.zeros(B, device=device)
        t0 = time.time()
        for j in range(steps):
            was_done = st.done.clone()
            now_in = lands <= j
            mine = O.Orders(*(torch.where(now_in[:, None], getattr(intended, f), getattr(in_force, f)) for f in O.FIELDS))
            o = O.merge(script(st), leash.apply(st, mine, area, own), own)
            before = reward.measure(st)
            hp0 = st.u["hp_abs"][:, 0].clone()
            sim_step(st, o, P)
            if ender is not None:
                finished = ender.after_step(st, spacing, area)
                st.winner = torch.where(st.done, st.winner, torch.zeros_like(st.winner))
            else:
                finished = st.done & ~was_done
            r = r + torch.where(was_done, torch.zeros_like(r),
                                reward.step(before, reward.measure(st), finished, st.winner)[:, 0])
            if cfg.get("area_reward"):
                # the area's signal (drill_go.whose_area): + per second ours; - per second the enemy's once we have
                # arrived (before, the march is the order's - its pace, the way - not the unit's choice: 0); 0 empty
                from tools.nn.unit import drill_go as DG
                wa = DG.whose_area(st, area)
                arr = getattr(ender, "arrived", None)
                if arr is not None:
                    wa = torch.where((wa == DG.THEIRS) & ~arr, torch.zeros_like(wa), wa)
                r = r + torch.where(was_done, torch.zeros_like(r), cfg["area_reward"] * P.dt * wa.float())
            if branch is not None:
                m = will & ~branched & ~was_done & Br.moment(st, spacing, branch["lead_s"])
                if bool(m.any()):
                    idx = m.nonzero()[:, 0]
                    snaps.append((idx, Br.take(st, idx)))
                    branched = branched | m
                first = (contact_d < 0) & Br.contact(st) & ~was_done
                if bool(first.any()):
                    contact_d = torch.where(first, d, contact_d)
                    beh_d = torch.where(first, d, beh_d)
                    beh = torch.where(first, kit.behaviour(st, brace), beh)
                if hasattr(kit, "track") and bool(branched.any()):
                    nb = kit.track(st, beh, spacing)
                    beh_d = torch.where((nb != beh) & branched & ~was_done, d, beh_d)
                    beh = torch.where(branched & ~was_done, nb, beh)
            if watch is not None:
                watch(st, hp0, was_done, area, d * steps + j + 1)
            if meter is not None:
                meter.step(st, bi, ii, P, was_done)
            u = st.u
            out = Dc.signed_dist(u["x"], u["z"], {k: a for k, a in area.items() if k != "on"}) < -0.5
            alive = own & ~was_done[:, None] & (u["men"] > 0)
            stats["leash_s"] += float((leash.on & alive).sum()) * P.dt
            pushed = u["m"] | u["r"]
            stats["out_free_s"] += float((out & alive & ~pushed).sum()) * P.dt
            stats["out_pushed_s"] += float((out & alive & pushed).sum()) * P.dt
        # the order changes' cost (the main training's: reward.order_change, divided by the side's network units; the
        # first decision's orders are no change - the order in force at the start is the drill's)
        ch = order_changes(intended, in_force, bi, ii, widths=wid) & valid[d] if d > 0 else torch.zeros(
            R, dtype=torch.bool, device=device)
        stats["order_changes"] += int(ch.sum())
        n_units = torch.zeros(B, device=device).index_add_(0, bi, torch.ones(R, device=device))
        side_ch = torch.zeros(B, device=device).index_add_(0, bi, ch.float())
        r = r - float(cfg.get("order_change", 0.0)) * side_ch / n_units.clamp(min=1)
        if prio is not None:
            # the cost of disobeying (drill U8): per second of the decision, divided by the side's network units as above
            side_dis = torch.zeros(B, device=device).index_add_(0, bi, disobey)
            r = r - float(cfg.get("prio_cost", 0.0)) * steps * P.dt * side_dis / n_units.clamp(min=1)
        in_force = intended
        t_sim += time.time() - t0
        rew[d] = r[bi]                      # the side's reward, shared by its network units
        # the drill's time limit is a cut, not an end: no 'done' there, the critic's value of the last state goes on;
        # a unit that routed or died (the battle going on) ends its own trajectory
        cut = (ender.why == 3) if ender is not None else torch.zeros_like(st.done)
        done[d] = valid[d] & ~(~st.done[bi] & viewer_ok(st, bi, ii)) & ~(st.done[bi] & cut[bi])
        duty.end()
        if bool(st.done.all()):
            D = d + 1
            break
    if branch is not None:
        tb = time.time()
        Vn = len(kit.variants)
        info = {"branched": branched, "contact_d": contact_d, "beh_d": beh_d, "behaviour": beh, "names": kit.names(),
                "trades": torch.full((B, Vn), float("nan"), device=device),
                "best": torch.full((B,), -1, dtype=torch.long, device=device)}
        if snaps:
            idx = torch.cat([i for i, _ in snaps])
            snap = Br.cat([x for _, x in snaps])
            rep = {}
            # (pad the branches to a fixed size - twice the expected share - so the compiled step keeps its shape)
            rows = max(1, int(round(B * min(1.0, 2 * branch["share"]))))
            info["trades"][idx] = kit.play(snap, {k: v[idx] for k, v in area.items()}, P, sim_step,
                                           branch["horizon_s"], rows=rows, report=rep)
            ok = branched & (contact_d >= 0) & (contact_d < D)
            bon, best = Br.bonus_of(beh, torch.nan_to_num(info["trades"]), branch["weight"])
            bon = torch.where(ok, bon, torch.zeros_like(bon))
            rew[beh_d.clamp(0, D - 1), torch.arange(B, device=device)] += bon
            info["best"] = torch.where(branched, best, torch.full_like(best, -1))
            _, margin = Br.best_of(torch.nan_to_num(info["trades"]))
            n = max(int(ok.sum()), 1)
            stand_v = [v for v, (k_, _) in enumerate(kit.variants) if k_ == "stand"]
            braced = rep["braced"][:, stand_v][rep["contact"][:, stand_v]]
            stats["branch"] = {"share": round(float(branched.float().mean()), 3),
                               "best": {nm: round(float((best[branched] == v).float().mean()), 3)
                                        for v, nm in enumerate(kit.names())},
                               "margin": round(float(margin[branched].mean()), 4),
                               "beh": {nm: round(float((beh[ok] == v).float().sum()) / n, 3)
                                       for v, nm in enumerate(kit.names())},
                               "chose_best": round(float((beh[ok] == best[ok]).float().sum()) / n, 3),
                               "bonus_mean": round(float(bon.sum()) / B, 4), "weight": branch["weight"],
                               "stand_branch_braced": round(float(braced.float().mean()), 3) if braced.numel() else None,
                               "seconds": round(time.time() - tb, 1)}
        if Vn == 2:                         # (U1's names in the exam)
            info["t_stand"], info["t_attack"] = info["trades"][:, Br.STAND], info["trades"][:, Br.ATTACK]
        stats["_branch"] = info
    last_value = torch.zeros(R, device=device)
    if critic is not None:
        with torch.no_grad():
            nxt = V.raw(st, spacing, leash=leash.on, missile=P.sim["missile"])
            last = torch.cat([raws[:D], nxt[None]])
            vc = views(last, torch.full((R,), D, device=device), bi, ii, card, area, cfg, everyone=True, prio=prio)
            cut = (ender.why == 3) if ender is not None else torch.zeros_like(st.done)
            arr = getattr(ender, "arrived", None)
            order = orders_now(nxt, area, pace, torch.ones(B, dtype=torch.bool, device=device) if arr is None else arr,
                               bi, ii, prio=prio)
            going = (~st.done[bi] & viewer_ok(st, bi, ii)) | (st.done[bi] & cut[bi])
            last_value = torch.where(going, critic(vc, order, vc["cell_m"]), torch.zeros(R, device=device))
    batch = {"raws": raws[:D], "act": {k: v_[:D] for k, v_ in store.items()}, "lp": lp_old[:D], "value": val[:D],
             "reward": rew[:D], "done": done[:D], "valid": valid[:D], "last_value": last_value, "card": card, "own": own,
             "area": area, "pace": pace, "arrived": arrived_d[:D], "bi": bi, "ii": ii, "prio": prio, "wlim": wlim}
    if prio is not None:
        a_s, o_s = float(pr["allowed_s"].sum()), float(pr["obeyed_s"].sum())
        am_s, om_s = float(pr["allowed_melee_s"].sum()), float(pr["obeyed_melee_s"].sum())
        c_s, oc_s = float(pr["contested_s"].sum()), float(pr["obeyed_contested_s"].sum())
        a0_s, o0_s = float(pr["allowed_old_s"].sum()), float(pr["obeyed_old_s"].sum())
        stats["prio"] = {"allowed_s": round(a_s / B, 1), "obedience": round(o_s / a_s, 3) if a_s else None,
                         "obedience_contested": round(oc_s / c_s, 3) if c_s else None,
                         "obedience_old": round(o0_s / a0_s, 3) if a0_s else None,
                         "obedience_melee": round(om_s / am_s, 3) if am_s else None,
                         "obedience_free": round((o_s - om_s) / (a_s - am_s), 3) if a_s - am_s else None}
        stats["_prio"] = pr
    n = max(stats.pop("n_dec"), 1)
    stats["entropy"] = {k: round(v_ / n, 4) for k, v_ in stats["entropy"].items()}
    stats.update({"decisions": D, "battles": B, "win1": float((st.winner == 1).float().mean()),
                  "ended": float(st.done.float().mean()), "reward_mean": float(rew[:D][valid[:D]].sum() / R),
                  "sim_s": round(t_sim, 2)})
    if meter is not None:
        stats["terrain"] = meter.result()
    if ender is not None:
        stats["end"] = {w: float((ender.why == k).float().mean()) for k, w in ((1, "we_routed"), (2, "enemy_left"),
                                                                              (3, "limit"), (0, "open"))}
    return batch, stats


def update(net, critic, opt, batch, cfg, device, duty):
    adv, ret = ppo.gae(batch["reward"], batch["value"], batch["done"], batch["last_value"], cfg["gamma"], cfg["lam"])
    d_idx, r_idx = batch["valid"].nonzero(as_tuple=True)
    M = d_idx.shape[0]
    a_all = adv[d_idx, r_idx]
    a_all = (a_all - a_all.mean()) / (a_all.std() + 1e-8)
    r_all = ret[d_idx, r_idx]
    out = {"pi": [], "v": [], "ent": {k: [] for k in Nt.HEADS}, "clip": [], "value_pred": [], "ret": []}
    for _ in range(cfg["epochs"]):
        perm = torch.randperm(M, device=device)
        for i in range(0, M, cfg["minibatch"]):
            duty.start()
            j = perm[i:i + cfg["minibatch"]]
            di, ri = d_idx[j], r_idx[j]
            bi, si = batch["bi"][ri], batch["ii"][ri]
            pq = batch.get("prio")
            v = views(batch["raws"], di, bi, si, batch["card"], batch["area"], cfg, prio=pq, wlim=batch.get("wlim"))
            vc = views(batch["raws"], di, bi, si, batch["card"], batch["area"], cfg, everyone=True, prio=pq)
            order = Nt.order_of(batch["raws"][di, bi, si], {k: v[bi, si] for k, v in batch["area"].items() if k != "on"},
                                batch["pace"][bi], batch["arrived"][di, bi],
                                has_target=None if pq is None else v["prio"] >= 0)
            lg = net(v, order, v["cell_m"])
            act = {k: batch["act"][k][di, ri] for k in Nt.HEADS}
            lp, ent = Nt.log_prob(lg, act)
            pl, clip = ppo.policy_loss(lp[:, None], batch["lp"][di, ri][:, None], a_all[j], torch.ones_like(lp[:, None]),
                                       cfg["clip"])
            value = critic(vc, order, vc["cell_m"])
            vl = F.mse_loss(value, r_all[j])
            loss = pl + cfg["vf"] * vl - cfg["entropy"] * ent.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(net.parameters()) + list(critic.parameters()), 1.0)
            opt.step()
            out["pi"].append(float(pl.detach()))
            out["v"].append(float(vl.detach()))
            out["clip"].append(float(clip))
            for k in Nt.HEADS:
                d_ = torch.distributions.Categorical(logits=lg[k])
                p = d_.probs
                out["ent"][k].append(float(-(torch.where(p > 0, p * d_.logits, torch.zeros_like(p))).sum(1).mean()))
            out["value_pred"].append(value.detach())
            out["ret"].append(r_all[j])
            duty.end()
    ev = float(ppo.explained_variance(torch.cat(out["value_pred"]), torch.cat(out["ret"])))
    return {"policy_loss": float(np.mean(out["pi"])), "value_loss": float(np.mean(out["v"])),
            "clip": float(np.mean(out["clip"])), "explained_var": ev, "samples": M,
            "entropy": {k: round(float(np.mean(v)), 4) for k, v in out["ent"].items()}}


def save(path, net, critic, opt, cfg, updates=0):
    torch.save({"kind": KIND, "net": net.state_dict(), "critic": critic.state_dict(), "opt": opt.state_dict(),
                "cfg": cfg, "updates": updates}, path)


def restore(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("kind") != KIND:
        raise ValueError(f"not a {KIND} checkpoint: {path}")
    net, critic = Nt.UnitNet().to(device), Nt.Critic().to(device)
    ck["padded"] = False
    for mod, key in ((net, "net"), (critic, "critic")):
        sd = dict(ck[key])
        own = mod.state_dict()
        w = sd["trunk.core.net.0.weight"]
        want = mod.trunk.core.net[0].weight.shape[1]
        if w.shape[1] < want:
            # a checkpoint from before the order's later fields (the fire sector, U5): their columns zero - the old
            # network's outputs unchanged for any order - inserted before the last column (the cell size)
            new = torch.zeros((w.shape[0], want), dtype=w.dtype, device=w.device)
            new[:, :w.shape[1] - 1] = w[:, :-1]
            new[:, -1] = w[:, -1]
            sd["trunk.core.net.0.weight"] = new
            ck["padded"] = True
        w = sd["trunk.map.c1.weight"]
        want = own["trunk.map.c1.weight"].shape[1]
        if w.shape[1] < want:
            # a checkpoint from before the shooter's map layers (zmap.LAYERS: reach, fire_arc - the last ones): their
            # input channels zero, the old map features unchanged
            new = torch.zeros((w.shape[0], want) + tuple(w.shape[2:]), dtype=w.dtype, device=w.device)
            new[:, :w.shape[1]] = w
            sd["trunk.map.c1.weight"] = new
            ck["padded"] = True
        for name, col in (("trunk.rows.state.weight", None), ("trunk.memory.gru.weight_ih_l0", -1)):
            # a checkpoint from before the row's later columns (view.ROW: dh, the ground's height - the last): their
            # input columns zero; the memory's GRU takes the row and the snapshot's mask last, so its new columns go
            # before that one
            if name in sd and sd[name].shape[1] < own[name].shape[1]:
                w = sd[name]
                new = torch.zeros(own[name].shape, dtype=w.dtype, device=w.device)
                if col is None:
                    new[:, :w.shape[1]] = w
                else:
                    new[:, :w.shape[1] - 1] = w[:, :-1]
                    new[:, -1] = w[:, -1]
                sd[name] = new
                ck["padded"] = True
        for name, p in own.items():
            if name not in sd and name.startswith("width."):
                # a checkpoint from before the width head: its start (net.init_width: zero weights, 'keep' first)
                sd[name] = p
                ck["padded"] = True
        mod.load_state_dict(sd)
    return net, critic, ck


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--drill", default="smoke", choices=sorted(SETUPS) + ["mix"],
                    help="mix: U1-U9 (hold / go / multi / hold2 / shoot1 / withdraw / shoot2 / rout / prio / width / shootw) in turn, one drill a batch - equal shares of the "
                         "skirmishes; a batch of mixed drills would wait for its longest skirmish (go's up to 700 s)")
    ap.add_argument("--battles", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=None, help="decisions a batch at most (smoke 120, hold 180)")
    ap.add_argument("--updates", type=int, default=5, help="at most this many updates")
    ap.add_argument("--minutes", type=float, default=0.0, help="> 0: no new update after this many minutes")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init", default=None, help="go on from this checkpoint (latest.pt of a run): weights, Adam, counter")
    ap.add_argument("--exam", action="store_true", help="the drill's exam at the end (exam_hold.py / exam_go.py) -> "
                    "exam.json")
    ap.add_argument("--branch-share", type=float, default=0.0, help="drill hold: the share of battles branched at the "
                    "approach (stand / attack, branch.py); 0 off")
    ap.add_argument("--branch-bonus", type=float, default=1.0, help="the bonus's weight x the branches' trade difference")
    ap.add_argument("--branch-decay", type=int, default=20, help="the weight falls linearly to 0 over this many updates of this run")
    ap.add_argument("--branch-lead-s", type=float, default=4.0)
    ap.add_argument("--branch-horizon-s", type=float, default=60.0)
    ap.add_argument("--order-change-cost", type=float, default=reward.Weights.order_change,
                    help="the cost of a real order change of the network (kind, target, a point 10 m or more away, a "
                         "facing 22.5 deg or more away, another width; divided by the side's network units; the leash's "
                         "are free), as "
                         "the main training's reward.order_change (0.001)")
    ap.add_argument("--prio-cost", type=float, default=PRIO_COST,
                    help="drill U8 (prio, in the mix too): the cost a second while the main target the order names is "
                         "allowed (infantry: in the area; a shooter: in its sector and range, the shooter not in melee "
                         "and the target not routing: obedience_parts) and the network's target is another one or "
                         "nobody (divided by the side's network units)")
    ap.add_argument("--area-reward", type=float, default=None, help="per second the area is ours +W, the enemy's -W (after our arrival) "
                    "(drill_go.whose_area); default 0.001 for --drill go (~0.15 of the budget over 150 s), else 0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gpu-duty", type=float, default=1.0)
    ap.add_argument("--out", default="build/diag_g49/smoke")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map", default=None, help="lay every skirmish at a random place and turn of this real map "
                    "(build/maps/<name>: heights, ground, trunks; tools/nn/unit/realmap.py); the --exam at the end on "
                    "its fixed places of the map")
    a = ap.parse_args(argv)
    if a.map:
        from tools.nn.unit import realmap
        realmap.activate(a.map)
    torch.manual_seed(a.seed)
    P = load()
    drills = list(MIX) if a.drill == "mix" else [a.drill]
    if not a.map:
        assert "height" not in drills or a.drill == "mix", "--drill height runs on a real map only (--map)"
        drills = [d_ for d_ in drills if d_ not in MAP_ONLY]          # (the mix without a map: its flat drills)

    def cfg_of(drill):
        """The batch's settings for one drill (mix: the area's signal off unless --area-reward says)."""
        area = a.area_reward if a.area_reward is not None else (AREA_REWARD if a.drill == "go" else 0.0)
        return {"battles": a.battles,
                "seconds": a.seconds or {"hold": 180.0, "go": 700.0, "multi": 180.0, "hold2": 180.0,
                                         "shoot1": 180.0, "withdraw": 400.0, "shoot2": 180.0,
                                         "rout": 125.0, "prio": 180.0, "width": 180.0, "shootw": 180.0, "height": 180.0}.get(drill, 120.0),
                "epochs": a.epochs, "minibatch": a.minibatch, "cell_m": 8.0, "bounds": None, "gamma": 0.9994,
                "lam": 0.9025, "clip": 0.2, "vf": 0.5, "entropy": 0.01, "drill": drill, "area_reward": area,
                "order_change": a.order_change_cost, "prio_cost": a.prio_cost}
    cfg = cfg_of(drills[0])
    dev = a.device
    net, critic = Nt.UnitNet().to(dev), Nt.Critic().to(dev)
    opt = torch.optim.Adam(list(net.parameters()) + list(critic.parameters()), lr=a.lr)
    done_updates = 0
    if a.init:
        net, critic, ck = restore(a.init, dev)
        opt = torch.optim.Adam(list(net.parameters()) + list(critic.parameters()), lr=a.lr)
        if not ck["padded"]:
            opt.load_state_dict(ck["opt"])          # (a padded network starts a fresh Adam: its moments do not fit)
        done_updates = int(ck.get("updates", 0))
    duty = Duty(dev, a.gpu_duty)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(a.seed + 1000 * done_updates)
    log = []
    start = time.time()
    for k in range(a.updates):
        if a.minutes > 0 and time.time() - start > a.minutes * 60:
            break
        t0 = time.time()
        drill = drills[k % len(drills)]
        cfg = cfg_of(drill)
        br = None
        if a.branch_share > 0 and drill in ("hold", "multi"):
            from tools.nn.unit import branch as Br
            # (the decay counts this run's updates: a run going on from a checkpoint starts the bonus afresh)
            br = {"share": a.branch_share, "weight": Br.weight_at(a.branch_bonus, k, a.branch_decay),
                  "lead_s": a.branch_lead_s, "horizon_s": a.branch_horizon_s}
        batch, rs = rollout(net, critic, cfg, P, dev, duty, SETUPS[drill](cfg, P, dev, gen), branch=br)
        rs.pop("_branch", None)
        rs.pop("_prio", None)
        t1 = time.time()
        us = update(net, critic, opt, batch, cfg, dev, duty)
        t2 = time.time()
        done_updates += 1
        row = {"update": done_updates, "drill": drill, **rs, **us, "rollout_s": round(t1 - t0, 1), "update_s": round(t2 - t1, 1),
               "battle_s_per_s": round(float(batch["valid"].sum()) / max(t1 - t0, 1e-9), 1),
               "slept_s": round(duty.slept, 1)}
        log.append(row)
        print(json.dumps(row), flush=True)
        save(out / "latest.pt", net, critic, opt, cfg, done_updates)
    path = out / "latest.pt"
    save(path, net, critic, opt, cfg, done_updates)
    net2, critic2, _ = restore(path, dev)
    # the checkpoint reads back: the same output bit for bit
    st, own, area = smoke_battles(4, P, dev)
    raws = V.raw(st, P.sim["formation"]["spacing_m"])[None]
    v = views(raws, torch.zeros(4, dtype=torch.long, device=dev), torch.arange(4, device=dev),
              torch.zeros(4, dtype=torch.long, device=dev), cards(st, dev), area, cfg)
    o, cm = Nt.order_vec(4, has_area=True, device=dev), v["cell_m"]
    with torch.no_grad():
        same = all(torch.equal(net.eval()(v, o, cm)[k], net2.eval()(v, o, cm)[k]) for k in Nt.HEADS)
    net.train()
    print(json.dumps({"checkpoint_same": same, "updates": done_updates,
                      "minutes": round((time.time() - start) / 60, 1)}), flush=True)
    (out / "log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    if a.exam:
        from tools.nn.unit import (exam_go, exam_hold, exam_multi, exam_pair, exam_prio, exam_rout, exam_shoot,
                                   exam_height, exam_shoot2, exam_shootw, exam_width, exam_withdraw)
        exams = {"hold": exam_hold, "go": exam_go, "multi": exam_multi, "hold2": exam_pair, "shoot1": exam_shoot,
                 "withdraw": exam_withdraw, "shoot2": exam_shoot2, "rout": exam_rout, "prio": exam_prio,
                 "width": exam_width, "shootw": exam_shootw, "height": exam_height}
        assert set(MIX) <= set(exams), "every drill of the mix has its exam"
        if a.map:
            realmap.activate(a.map, exam=True)          # (the exams' fixed places)
        if a.drill == "mix":                       # every drill's, one after another
            for d_ in MIX:
                if d_ in MAP_ONLY and not a.map:
                    continue
                exams[d_].run(net, P, dev, duty, out / f"exam_{d_}.json")
        else:
            exams.get(a.drill, exam_hold).run(net, P, dev, duty, out / "exam.json")
    return log, same


if __name__ == "__main__":
    main()
