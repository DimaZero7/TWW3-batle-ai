"""Drill T0-2 'strike the pinned one': the first training (imitation) and the exam after by the move marks
(tmp/tactic_slice/design_attack_v23.md sections 7-8; review_attack_v23 O - the first training runs on the simulator at
HEAD and is judged by the move marks M3, M4, M5, M7; the outcome N - C, TEACH - C is stated, not judged).

    NetBuffer   the hook of tags.Record for heads.TacticNet2 (input D2): per group-tick with a head-free unit what
                inputs.build needs (the rows' raw state, the parcel, the tiles, the areas of the frames in force, the
                bounds; the battle's passport cards once a battle) - the map windows are rebuilt in each update
                (rebuild) - and the rest of the network's input as the level gave it (heads.net_inputs: cols, own, g_row,
                g_on; the command; the walk fed with the labeller's choices: open, pair) with the label, its weight's
                facts (head-free, dropped / don't-care label, kinds, trivial, near2) and the don't-care keys. A tick
                whose head-free units are all trivial keeps is kept one in loss.trivial_every with that weight (as
                learn.Buffer). check - the first ticks with the level's own input (a network driving) rebuilt and
                compared
    Disturb     the teacher with its battles' disturbances (section 7 'Who drives'): a free unit sent at a random open
                enemy within orders.reach_m, or one of the teacher's new attack orders held back 2-6 ticks
    Marks       the read of a set (section 8 'followed'): path N - the network drives (greedy), the labeller labels its
                states; path T - TEACH drives, the network is asked on its states with the walk fed with the teacher's
                choices. Per head-free unit-tick: the label, the network's pick (0 keep, 1 the label, 2 a don't-care key,
                3 a false attack), uncaused (the pick's target not pinned at touch bands.touch_m top or farther than
                bands.reach_m top); path N also the orders held after each tick (target, side) - 'followed by target'
    marks(...)  M3's table of a set (followed on N's path by target, pooled / by family x half; on the teacher's path
                by unit; false attacks a head-free unit-tick, by taught kind, by recipe; no-cause scenes with a false
                attack; uncaused), M4 (N's attacks on YES labels at the labeller's target)
    train(a)    `learn train --out DIR [--rounds N] [--no-general]`: rounds of section 7 (config tactic.json
                train.learn), V read after each round, the stop when M3 holds two rounds running
    after(a)    `learn after --ckpt PATH --out DIR [--vision]`: the exam after on S, H, G (M3, M4, M5 by axis, M7; N - C
                and TEACH - C at the battle's end stated), three records for game.vision

    DOCK_NAME=agent-s6 DOCK_CPUS=8 DOCK_MEM=6g bash ops/dock.sh net.tactic.learn train --out weights/tactic/attack1 \
        --log tmp/tactic_slice/v23/stage6/train --sim e0afc53
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from net.stage0 import battle as BA
from net.stage0 import cfg as scfg
from net.stage0 import specs as SP
from net.tactic import cfg as tcfg
from net.tactic import exam_pin as EP
from net.tactic import heads as HD
from net.tactic import inputs as TIN
from net.tactic import learn as LN
from net.tactic import orders as OR
from net.tactic import parcel as PC
from net.tactic import pin as PN
from net.tactic import tags as TG
from net.tactic import uzone as U
from net.tactic import zone as ZN
from net.unit import scripts as SC

GMAX = LN.GMAX
A_KEYS = ("x", "z", "b", "front", "depth")
P_KEYS = ("mine", "target", "cx", "cz", "sin", "cos", "line", "group")
T_KEYS = ("cx", "cz", "heading", "rows", "cols", "mask")
NET_KEYS = ("raw", "command", "cols", "own", "g_row", "g_on", "open", "pair", "label", "dc", "bounds") + tuple(
    "p_" + k for k in P_KEYS) + tuple("t_" + k for k in T_KEYS) + tuple("a_" + k for k in A_KEYS + ("on",))
# drill T0-2 (side=True): own() of 19, the S2 row columns, the side label (approach, -1 none) and the side head's columns
# at the label's target (heads.side_feats + held) - learn.side_label_inputs
SIDE_KEYS = ("s2", "ap", "x_k")
# drill T0-2's side marks on V (config tactic.json drills.side.marks): followed per class on the teacher's path,
# false flank on the no-cause scenes S2N
SIDE_MARKS = tcfg()["drills"]["side"]["marks"]
STOP_FROM = int(SIDE_MARKS.get("stop_from_round", 4))      # (drill T0-2: no early stop before this round)


def L():
    return tcfg()["train"]["learn"]


def _bands():
    return scfg()["drills"]["attack"]["bands"]


# drill T0-2: a label whose teacher approach is LEFT / RIGHT is not dropped for 'through own' (the walk goes round P; the
# teacher's answer is definite) - in the pointer's and the side's loss and in the validation reads; FRONT keeps T0-1's
# exclusion, the key-band part stays (tmp/astra/t02_review/out/REVIEW.md finding 1). False only for a scratch check.
KEEP_FLANK_THROUGH_OWN = True


def drop_of(T, ap=None):
    """[R, G] the dropped / don't-care labels of a tick's tags T: T0-1's dropped | dc_label; with ap [R, G] (drill T0-2:
    the teacher's approach of each label, -1 none) a flank label's through_own part is lifted."""
    if ap is None or not KEEP_FLANK_THROUGH_OWN:
        return T["dropped"] | T["dc_label"]
    flank = (ap == OR.LEFT) | (ap == OR.RIGHT)
    return T["dropped"] | T["dc_band"] | (T["dc_thr"] & ~flank)


def done_rows(v):
    """[R] bool: the view's battles already over (sim st.done; REVIEW2 finding 3 - net/matchup/duel.py keeps stepping a
    chunk until its last battle ends, and a finished battle's frozen state is seen again at every later tick: its rows
    are kept out of the buffer and the marks)."""
    st = v.sim
    dn = getattr(st, "done", None) if st is not None else None
    if dn is None:
        return torch.zeros(v.idx.shape, dtype=torch.bool, device=v.fx.s.device)
    return dn.to(v.idx.device)[v.idx].to(v.fx.s.device)


def open_at(opn, k):
    """[.., G] bool: the row k (-1 none: False) open in opn [.., G, K] - the pointer's mask of the walk the execution
    uses at that slot (REVIEW2 finding 2: a side label whose own pointer label is closed there is impossible)."""
    return opn.gather(-1, k.clamp(min=0)[..., None])[..., 0] & (k >= 0)


def group_of_meta(m):
    if m.get("kind") == "general":
        return "general"
    return "S2" if m.get("recipe") in LN.SIDE_RECIPES else "T0-1 scenes"


# ------------------------------------------------------------------ the network's input rebuilt
def rebuild(t, card, K):
    """inputs.build of M stored group-ticks t (NetBuffer's keys, one device): each its own battle (parcel.bi = its
    index), card [M, N, 69] the passport cards of their battles."""
    raw = t["raw"]
    M = raw.shape[0]
    dev = raw.device
    p = PC.Parcel(torch.arange(M, device=dev), *(t["p_" + k] for k in P_KEYS))
    tiles = ZN.Tiles(*(t["t_" + k] for k in T_KEYS))
    areas = {k: t["a_" + k] for k in A_KEYS}
    areas["on"] = t["a_on"]
    bd = t["bounds"]
    nan = torch.isnan(bd)
    assert bool(nan.all()) or not bool(nan.any()), "bounds: a flat battle's square and none in one batch"
    return TIN.build(raw, card, p, tiles, areas, bounds=None if bool(nan.all()) else bd, rows=K)


def _diff(a, b):
    """The largest difference of two inputs.build dicts (nested), inf when a shape or an integer tensor differs."""
    if isinstance(a, dict):
        return max((_diff(a[k], b[k]) for k in a), default=0.0)
    if a.shape != b.shape:
        return float("inf")
    if a.is_floating_point():
        return float((a.float() - b.float()).abs().max()) if a.numel() else 0.0
    return 0.0 if torch.equal(a, b) else float("inf")


class NetBuffer:
    """(module doc) ticks: a list of dicts (CPU); cards {battle id: [N, 69]}; meta by battle id."""

    def __init__(self, check=0, every=None, side=False):
        self.every = int(LN._c()["loss"]["trivial_every"] if every is None else every)
        self.pending, self.ticks, self.meta, self.cards, self.pend_card = [], [], [], [], {}
        self.n_triv = 0
        self.check, self.checked, self.check_max = int(check), 0, 0.0
        self.NK = None
        self.side = bool(side)
        self.pend_done, self.done_stats = {}, {}     # rows of finished battles left out: {group: [rows seen, left out]}

    def hook(self, net_driven):
        return lambda v, lab, T: self(v, T, net_driven, lab)

    def __call__(self, v, T, net_driven, lab=None):
        dn = done_rows(v)
        h0 = T["head_free"].any(1)
        for b, a_, d_ in zip(v.idx.tolist(), h0.tolist(), (h0 & dn).tolist()):
            c = self.pend_done.setdefault(b, [0, 0])
            c[0] += int(a_)
            c[1] += int(d_)
        head = T["head_free"] & ~dn[:, None]
        sel = head.any(1)
        if not bool(sel.any()):
            return
        triv = sel & ~(head & ~T["trivial"]).any(1)
        tw = torch.ones(sel.shape, device=sel.device)
        for r in triv.nonzero()[:, 0].tolist():
            self.n_triv += 1
            if self.n_triv % self.every:
                sel[r] = False
            else:
                tw[r] = float(self.every)
        if not bool(sel.any()):
            return
        fx = v.fx
        assert v.raw is not None and v.card is not None and v.parcel is not None, "the level gave no raw / card / parcel"
        nk = (int(v.raw.shape[1]), int(fx.K))
        self.NK = self.NK or nk
        assert self.NK == nk, f"the battles' slots / rows changed: {self.NK} -> {nk}"
        R, G = fx.R, fx.G
        dev = fx.s.device
        if self.side:
            x = HD.net_inputs3(v)
            si = LN.side_label_inputs(v, lab, T)
            wi = si["wi"]
        else:
            x = HD.net_inputs(v)
            wi = HD.walk_inputs(fx, T["k"])
        label = torch.where(T["attack"], T["k"], torch.full_like(T["k"], -1))
        kinds = torch.zeros((R, G), dtype=torch.int32, device=dev)
        for i, n in enumerate(TG.KINDS):
            kinds |= T["kinds"][n].int() << i
        p, tl = v.parcel, v.tiles
        if v.ctx.prev is not None:
            ar = U.world(v.ctx.prev, tl)
        else:
            z = torch.zeros((R, G), device=dev)
            ar = {k: z for k in A_KEYS}
            ar["on"] = torch.zeros((R, G), dtype=torch.bool, device=dev)
        bd = v.bounds
        if bd is None:
            bounds = torch.full((R,), float("nan"), device=dev)
        elif torch.is_tensor(bd) and bd.numel() > 1:
            bounds = bd.float().to(dev)[p.bi]
        else:
            bounds = torch.full((R,), float(bd), device=dev)
        st = {"raw": v.raw.float(), "command": v.command.float(), "cols": x["cols"], "own": x["own"],
              "g_row": x["g_row"], "g_on": x["g_on"], "open": wi["open"], "pair": wi["pair"], "label": label,
              "dc": T["dc_keys"], "bounds": bounds, "head": head,
              "drop": drop_of(T, si["ap"] if self.side else None), "kinds": kinds,
              "trivial": T["trivial"], "near2": T["tags"]["near2"] & T["attack"]}
        if self.side:
            # (dc_thr: the labels T0-1 drops as through own - kept to read the F1 share with and without the lift)
            st.update(s2=x["s2"], ap=si["ap"], x_k=si["x_k"], dc_thr=T["dc_thr"])
        for k in P_KEYS:
            st["p_" + k] = getattr(p, k)
        for k in T_KEYS:
            st["t_" + k] = getattr(tl, k)
        for k in A_KEYS + ("on",):
            st["a_" + k] = ar[k]
        if self.check > 0 and v.inp is not None:
            self.check -= 1
            self.checked += 1
            with torch.no_grad():
                got = rebuild({k: st[k] for k in NET_KEYS}, v.card.float(), fx.K)
            self.check_max = max(self.check_max, _diff({k: v.inp[k] for k in got}, got))
        out = {"b": v.idx[sel].cpu(), "sec": torch.full((int(sel.sum()),), int(round(float(v.now))), dtype=torch.int16),
               "tw": tw[sel].cpu(), "net": torch.full((int(sel.sum()),), bool(net_driven))}
        for k, t in st.items():
            t = t[sel]
            if t.dim() >= 2 and t.shape[1] == G and (k in ("own", "g_row", "g_on", "open", "pair", "label", "dc",
                                                           "head", "drop", "kinds", "trivial", "near2", "ap", "x_k",
                                                           "dc_thr")
                                                      or k.startswith("a_")):
                t = LN._pad_g(t, -1 if k in ("label", "ap") else 0)
            out[k] = t.cpu()
        for r in sel.nonzero()[:, 0].tolist():
            b = int(v.idx[r])
            if b not in self.pend_card:
                self.pend_card[b] = v.card[r].float().cpu()
        self.pending.append(out)

    def flush(self, ii, specs, kind, net_driven, disturbed=()):
        base = len(self.meta)
        for j, i in enumerate(ii):
            s = specs[i]
            self.meta.append({"kind": kind, "recipe": s.get("recipe", "G"), "twin_of": s.get("twin_of"),
                              "held": s.get("held"), "n": len(s["mine"]), "net": bool(net_driven),
                              "disturbed": i in disturbed,
                              **({"start": s["axes"].get("start")} if isinstance(s.get("axes"), dict) else {})})
            c = self.pend_card.get(j)
            self.cards.append(c)
            n = self.pend_done.get(j, [0, 0])
            g = self.done_stats.setdefault(group_of_meta(self.meta[-1]), [0, 0])
            g[0], g[1] = g[0] + n[0], g[1] + n[1]
        for t in self.pending:
            keep = t["b"] < len(ii)
            if not bool(keep.any()):
                continue
            t = {k: x[keep] for k, x in t.items()}
            t["b"] = t["b"].long() + base
            self.ticks.append(t)
        self.pending, self.pend_card, self.pend_done = [], {}, {}

    def data(self):
        d = {k: torch.cat([t[k] for t in self.ticks], 0) for k in self.ticks[0]}
        some = next(c for c in self.cards if c is not None)
        d["cards"] = torch.stack([some * 0 if c is None else c for c in self.cards], 0)
        return d


# ------------------------------------------------------------------ who drives
class Disturb:
    """(module doc) the teacher; plans {Tacticians row: {kind: send | hold, left: events}}; holds kept per row."""
    needs_input = False

    def __init__(self, teacher, plans, seed):
        self.t, self.plans = teacher, plans
        self.name, self.allow = getattr(teacher, "name", "TEACH"), getattr(teacher, "allow", (SC.HOLD, SC.ATTACK))
        self.rnd = random.Random(seed)
        self.n = {"send": 0, "hold": 0, "held_ticks": 0}
        for p in plans.values():
            p.setdefault("hold", {})

    @property
    def last_tags(self):
        return self.t.last_tags

    def __call__(self, v):
        out = self.t(v)
        rows = v.idx.tolist()
        mine = [(r, self.plans[b]) for r, b in enumerate(rows) if b in self.plans]
        if not mine:
            return out
        c = L()["disturb"]
        fx = v.fx
        o = v.order
        skill, tgt, side = out["skill"].clone(), out["target"].clone(), out["side"].clone()
        new = ((skill == SC.ATTACK) & (tgt >= 0) & ((tgt != o["tgt"]) | (o["skill"] != SC.ATTACK))).cpu()
        reach = float(OR.c_orders()["reach_m"])
        cand = None

        def revert(r, g):
            skill[r, g], tgt[r, g], side[r, g] = o["skill"][r, g], o["tgt"][r, g], o["side"][r, g]
        for r, pl in mine:
            for g in list(pl["hold"]):
                if bool(new[r, g]):
                    revert(r, g)
                    self.n["held_ticks"] += 1
                pl["hold"][g] -= 1
                if pl["hold"][g] <= 0:
                    del pl["hold"][g]
            if pl["left"] <= 0:
                continue
            if pl["kind"] == "hold":
                gs = [g for g in range(fx.G) if bool(new[r, g]) and g not in pl["hold"]]
                if gs and self.rnd.random() < float(c["p_hold"]):
                    g = self.rnd.choice(gs)
                    revert(r, g)
                    pl["hold"][g] = self.rnd.randint(*c["hold_ticks"]) - 1
                    pl["left"] -= 1
                    self.n["hold"] += 1
                    self.n["held_ticks"] += 1
            elif self.rnd.random() < float(c["p_send"]):
                if cand is None:
                    w = OR.Walk(fx)
                    opn = torch.stack([w.open(g) for g in range(fx.G)], 1)                  # [R, G, K]
                    unit = fx.free & fx.able & ~v.melee & ~fx.striking & ~new.to(fx.s.device)
                    cand = ((unit[..., None] & opn & fx.valid[:, None, :] & (fx.g_dist <= reach)).cpu(),
                            fx.slot.cpu(), fx.g_side.cpu())
                ok, slot, gside = cand
                gs = [g for g in range(fx.G) if bool(ok[r, g].any())]
                if gs:
                    g = self.rnd.choice(gs)
                    k = self.rnd.choice(ok[r, g].nonzero()[:, 0].tolist())
                    skill[r, g] = SC.ATTACK
                    tgt[r, g] = int(slot[r, k])
                    side[r, g] = int(gside[r, g, k])
                    pl["left"] -= 1
                    self.n["send"] += 1
        return {**out, "skill": skill, "target": tgt, "side": side}


def _kw(scene):
    return {"battles": BA.scene_battles, "enemy": BA.SceneEnemy} if scene else {}


def play(name, specs, scene, seeds, chunk, a, P, make_driver, hook_of, on_flush, needs_input=None, watch=None,
         labeller="TEACH"):
    """exam_pin.run_player under tags.Record: the driver make_driver(ii) (ii the chunk's spec indices, play_tactic's
    order by limit), the labeller (TEACH; drill T0-2: TEACH2) built under its own name, hook_of(driver) the Record's
    hook; on_flush(ii) after each chunk; needs_input - set on the Record (the teacher-path read: the network asked on
    the teacher's states)."""
    order = sorted(range(len(specs)), key=lambda i: int(specs[i]["limit"]))
    chunks = [order[i:i + chunk] for i in range(0, len(order), chunk)]
    state = [0]
    drivers = []

    def make():
        ii = chunks[state[0]]
        state[0] += 1
        drv = make_driver(ii)
        drivers.append(drv)
        rec = TG.Record(drv, EP.build_under(labeller), hook=hook_of(drv))
        if needs_input is not None:
            rec.needs_input = bool(needs_input)
        return rec
    res = EP.run_player(name, specs, seeds, chunk, a, P, make=make, on_chunk=lambda ii, w, s: on_flush(ii),
                        watch_cls=watch, full=True, by_limit=True, **_kw(scene))
    res["recipes"] = [s.get("recipe", "G") for s in specs]
    return res, drivers


# ------------------------------------------------------------------ the reads (section 8)
class Marks:
    """(module doc) path 'N' or 'T'; rows per head-free unit-tick (CPU), holds {(battle, sec): {target x 4 + side}}."""

    def __init__(self, net, path):
        self.net, self.path = net, path
        self.pending, self.rows, self.meta = [], [], []
        self.hold_pending, self.holds = [], {}
        self.pick_side = {}          # (the chunk's battle row, unit) -> (target slot, side class at the pick); path N
        self.done_cnt = [0, 0]       # head-free group-ticks seen, of finished battles left out (REVIEW2 finding 3)

    def hook(self, drv):
        return lambda v, lab, T: self(v, T, drv, lab)

    def __call__(self, v, T, drv, lab=None):
        fx = v.fx
        dn = done_rows(v)
        h0 = T["head_free"].any(1)
        self.done_cnt[0] += int(h0.sum())
        self.done_cnt[1] += int((h0 & dn).sum())
        head = T["head_free"] & ~dn[:, None]
        now = int(round(float(v.now)))
        side = isinstance(self.net, HD.TacticNet3) and lab is not None and "approach" in lab
        nap = None
        if self.path == "T":
            with torch.no_grad():
                if side:
                    o = self.net.act(v, greedy=True, given=T["k"], given_ap=lab["approach"])
                    nap = o["approach"]
                else:
                    o = self.net.act(v, greedy=True, given=T["k"])
                k = o["k"]
        else:
            k = drv.last["k"]
            if side:
                nap = drv.last["approach"]
            o = v.order
            att = k >= 0
            kc = k.clamp(min=0)
            pt = torch.where(att, fx.slot_of_row(k), o["tgt"])
            ps = torch.where(att, fx.g_side.gather(2, kc[..., None])[..., 0], o["side"])
            if side:
                # (T0-1's M3 'followed by target' reads the side class the striker stood on at its pick; a flank order
                # in force carries its approach in o['side'] (pin.NetPlayer) - the class at the pick is kept here per
                # battle and unit, so M3 means on the side path what it meant on T0-1's - REVIEW.md finding 14)
                ps = ps.clone()
                for r_, g_ in (~att & (o["skill"] == SC.ATTACK) & (o["tgt"] >= 0)).nonzero().tolist():
                    m_ = self.pick_side.get((int(v.idx[r_]), g_))
                    if m_ is not None and m_[0] == int(o["tgt"][r_, g_]):
                        ps[r_, g_] = m_[1]
                for r_, g_ in att.nonzero().tolist():
                    self.pick_side[(int(v.idx[r_]), g_)] = (int(pt[r_, g_]), int(ps[r_, g_]))
            pk = torch.where(att, torch.full_like(o["skill"], SC.ATTACK), o["skill"])
            code = torch.where((pk == SC.ATTACK) & (pt >= 0) & fx.able & ~dn[:, None], pt * 4 + ps.clamp(min=0),
                               torch.full_like(pt, -1))
            self.hold_pending.append((v.idx.cpu(), now, code.cpu()))
        if not bool(head.any()):
            return
        b = _bands()
        label = torch.where(T["attack"], T["k"], torch.full_like(T["k"], -1))
        k = torch.where(head, k, torch.full_like(k, -1))
        kc = k.clamp(min=0)
        dcp = T["dc_keys"].gather(2, kc[..., None])[..., 0]
        code = torch.where(k < 0, 0, torch.where(k == label, 1, torch.where(dcp, 2, 3)))
        pin = PN.pinned(fx, touch=float(b["touch_m"][1]))
        far = fx.g_dist.gather(2, kc[..., None])[..., 0] > float(b["reach_m"][1])
        unc = (k >= 0) & (~pin.gather(1, kc) | far)
        lc = label.clamp(min=0)
        kinds = torch.zeros(head.shape, dtype=torch.int32, device=head.device)
        for i, n in enumerate(TG.KINDS):
            kinds |= T["kinds"][n].int() << i
        r, g = head.nonzero(as_tuple=True)
        row = {"b": v.idx[r].cpu(), "sec": torch.full((r.numel(),), now, dtype=torch.int32),
               "att": (label >= 0)[r, g].cpu(),
               "drop": drop_of(T, torch.where(label >= 0, lab["approach"], torch.full_like(label, -1))
                               if side else None)[r, g].cpu(),
               "kinds": kinds[r, g].cpu(), "near2": (T["tags"]["near2"] & T["attack"])[r, g].cpu(),
               "code": code[r, g].to(torch.int8).cpu(), "unc": unc[r, g].cpu(),
               "e": fx.slot_of_row(label)[r, g].cpu(),
               "es": fx.g_side.gather(2, lc[..., None])[..., 0][r, g].cpu()}
        if side:
            # the side marks (drill T0-2): the label's approach, the network's at the label's pair (path T: asked on it;
            # path N: where its pick is the label's target, else -1)
            lap = torch.where(label >= 0, lab["approach"], torch.full_like(label, -1))
            ok_n = (code == 1) if self.path == "N" else (label >= 0)
            si = LN.side_label_inputs(v, lab, T)
            sb = LN.side_band(si["x_k"], lap)                                        # (the rule's threshold)
            sc = (label >= 0) & ~open_at(si["wi"]["open"], label)                    # (REVIEW2 finding 2)
            row.update(lap=lap[r, g].cpu(), nap=torch.where(ok_n, nap, torch.full_like(nap, -1))[r, g].cpu(),
                       sband=sb[r, g].cpu(), sclosed=sc[r, g].cpu())
        self.pending.append(row)

    def flush(self, ii, specs, kind):
        base = len(self.meta)
        for i in ii:
            s = specs[i]
            self.meta.append({"kind": kind, "recipe": s.get("recipe", "G"), "twin_of": s.get("twin_of"),
                              "held": s.get("held"), "n": len(s["mine"]), "spec": i})
        for t in self.pending:
            keep = t["b"] < len(ii)
            t = {k: x[keep] for k, x in t.items()}
            t["b"] = t["b"].long() + base
            self.rows.append(t)
        for idx, now, code in self.hold_pending:
            for r, b in enumerate(idx.tolist()):
                if b < len(ii):
                    self.holds[(b + base, now)] = set(x for x in code[r].tolist() if x >= 0)
        self.pending, self.hold_pending, self.pick_side = [], [], {}

    def data(self):
        return {k: torch.cat([t[k] for t in self.rows], 0) for k in self.rows[0]}


def cell_of(m, setname):
    if setname == "H":
        return "H:" + str(m.get("held"))
    if m["kind"] == "general":
        return "G"
    return LN.family_of(m)


def marks(mN, mT, setname):
    """(module doc) M3's and M4's table of a set from its two reads (the same specs in the same order)."""
    sh = LN.share
    step = int(tcfg()["decide_s"])
    out = {}
    meta = mN.meta
    dN = mN.data()
    cellN = [cell_of(m, setname) for m in meta]
    ok = ~dN["drop"]
    lab = dN["att"] & ok
    idx = lab.nonzero()[:, 0].tolist()
    fol = torch.zeros_like(lab)
    bl, sl, el, sd = dN["b"].tolist(), dN["sec"].tolist(), dN["e"].tolist(), dN["es"].tolist()
    for i in idx:
        c = el[i] * 4 + sd[i]
        fol[i] = any(c in mN.holds.get((bl[i], sl[i] + d * step), ()) for d in (-1, 0, 1, 2))
    n2 = dN["near2"]
    # what the network did instead at the unfollowed labels (a read, not a mark)
    by_b = {}
    for (b_, s_), cs in mN.holds.items():
        by_b.setdefault(b_, []).append((s_, cs))
    codes = dN["code"].tolist()
    cats = {}
    for i in idx:
        if bool(fol[i]):
            continue
        c, s0 = el[i] * 4 + sd[i], sl[i]
        if codes[i] in (2, 3):
            cat = "attacked another target"
        elif codes[i] == 1:
            cat = "attacked the label's target, not held after"
        elif any(any(x // 4 == el[i] for x in cs) for s_, cs in by_b.get(bl[i], ()) if s0 - step <= s_ <= s0 + 2 * step):
            cat = "kept; a unit of another side class held it in the window"
        elif any(c in cs for s_, cs in by_b.get(bl[i], ()) if s_ > s0 + 2 * step):
            cat = "kept; a unit of the side took it later than two ticks"
        else:
            cat = "kept; nobody of the side took it later"
        h = "YES" if bool(n2[i]) else "NO"
        cats.setdefault(cat, {"NO": 0, "YES": 0})[h] += 1
    out["unfollowed_N"] = {"labels_unfollowed": len(idx) - int(fol.sum()), "by_what": cats}

    def followed(f, d, cells, lab_):
        res = {"pooled": sh(int((f & lab_).sum()), int(lab_.sum())),
               "NO": sh(int((f & lab_ & ~d["near2"]).sum()), int((lab_ & ~d["near2"]).sum())),
               "YES": sh(int((f & lab_ & d["near2"]).sum()), int((lab_ & d["near2"]).sum())), "cells": {},
               "cells_all": {}}
        cb = [cells[b] for b in d["b"].tolist()]
        for c in sorted(set(cb)):
            s = torch.tensor([x == c for x in cb], dtype=torch.bool)
            res["cells"][c] = {h: sh(int((f & lab_ & s & m).sum()), int((lab_ & s & m).sum()))
                               for h, m in (("NO", ~d["near2"]), ("YES", d["near2"]))}
            res["cells_all"][c] = sh(int((f & lab_ & s).sum()), int((lab_ & s).sum()))
        return res

    def false(d, cells):
        okd = ~d["drop"]
        fal = okd & (d["code"].long() == 3)
        keep = okd & ~d["att"]
        res = {"tick": sh(int(fal.sum()), int(okd.sum())), "by_kind": {}, "by_recipe": {}}
        for i, k in enumerate(TG.KINDS):
            kb = ((d["kinds"] >> i) & 1).bool() & keep
            res["by_kind"][k] = dict(sh(int((fal & kb).sum()), int(kb.sum())), taught=k in TG.TAUGHT)
        rb = [cells[b] for b in d["b"].tolist()]
        for c in sorted(set(rb)):
            s = torch.tensor([x == c for x in rb], dtype=torch.bool)
            res["by_recipe"][c] = sh(int((fal & s).sum()), int((okd & s).sum()))
        return res
    recN = [m["recipe"] if setname != "H" else cell_of(m, setname) for m in meta]
    out["followed_N"] = followed(fol, dN, cellN, lab)
    out["false_N"] = false(dN, recN)
    if setname == "H":
        out["false_N"]["by_recipe_scene"] = false(dN, [m["recipe"] for m in meta])["by_recipe"]
    pick = dN["code"].long() > 0
    out["uncaused"] = sh(int((dN["unc"] & pick).sum()), int(pick.sum()))
    y = lab & n2 & pick
    out["M4"] = sh(int((y & (dN["code"].long() == 1)).sum()), int(y.sum()))
    # no-cause scenes: no full-weight label on N's path; with a false attack
    B = len(meta)
    has_lab = torch.zeros(B, dtype=torch.bool)
    has_lab[dN["b"][lab]] = True
    has_fal = torch.zeros(B, dtype=torch.bool)
    has_fal[dN["b"][ok & (dN["code"].long() == 3)]] = True
    nc = torch.tensor([m["kind"] == "scene" for m in meta]) & ~has_lab
    out["nocause_false"] = sh(int((nc & has_fal).sum()), int(nc.sum()))
    # per battle (the vision picks)
    def per_b(sel):
        c = torch.zeros(B, dtype=torch.long)
        c.index_add_(0, dN["b"][sel], torch.ones(int(sel.sum()), dtype=torch.long))
        return c.tolist()
    out["per_battle"] = {"uncaused": per_b(dN["unc"] & pick), "false": per_b(ok & (dN["code"].long() == 3)),
                         "yes_labels": per_b(lab & n2), "labels": per_b(lab), "nocause": nc.tolist()}
    if mT is not None:
        dT = mT.data()
        okT = ~dT["drop"]
        labT = dT["att"] & okT
        cellT = [cell_of(m, setname) for m in mT.meta]
        out["followed_T"] = followed(dT["code"].long() == 1, dT, cellT, labT)
        out["false_T"] = false(dT, [m["recipe"] if setname != "H" else cell_of(m, setname) for m in mT.meta])
    sN, sT = side_marks(mN), (side_marks(mT) if mT is not None else None)
    if sN is not None or sT is not None:
        out["side_N"], out["side_T"] = sN, sT
    return out


def side_marks(m):
    """Drill T0-2's side marks of a Marks read (its rows with lap / nap; None without): per class front / left / right
    followed (the network's approach = the label's, of the labels it was read on) - pooled, on S2 scenes, on S2N, on the
    other scenes, on G; false flank - front labels the network sided left / right, same groups."""
    if not m.rows or "lap" not in m.rows[0]:
        return None
    d = m.data()
    gr = [LN.side_group(x) for x in m.meta]
    gb = [gr[int(b)].split(" held")[0] for b in d["b"].tolist()]
    ok0 = ~d["drop"] & (d["lap"] >= 0) & (d["nap"] >= 0)
    sb = d["sband"] if "sband" in d else torch.zeros_like(ok0)
    scl = d["sclosed"] if "sclosed" in d else torch.zeros_like(ok0)
    ok = ok0 & ~sb & ~scl          # (the labels at the rule's threshold and those the execution's mask closes out)
    lap, nap = d["lap"].long(), d["nap"].long()
    out = {}
    for name in ("all", "S2", "S2N", "T0-1 scenes", "G"):
        grp = torch.tensor([name == "all" or x == name for x in gb], dtype=torch.bool)
        sel = ok & grp
        res = {c: LN.share(int((sel & (lap == i) & (nap == i)).sum()), int((sel & (lap == i)).sum()))
               for i, c in enumerate(OR.APPROACHES)}
        res["false_flank"] = LN.share(int((sel & (lap == 0) & (nap > 0)).sum()), int((sel & (lap == 0)).sum()))
        res["banded"] = LN.share(int((ok0 & grp & sb).sum()), int((ok0 & grp).sum()))
        res["closed"] = LN.share(int((ok0 & grp & scl).sum()), int((ok0 & grp).sum()))
        out[name] = res
    return out


def side_ok(sm, M=None):
    """(passed, fails) of the side marks on V's teacher path (SIDE_MARKS); a class or the S2N front labels with no
    observation are 'not judged' = not passed (REVIEW.md finding 14)."""
    M = M or SIDE_MARKS
    if sm is None:
        return False, ["no side marks"]
    fails = []
    mn = int(M.get("min_n", 1))
    for c in OR.APPROACHES:
        x = sm["all"][c]
        if x["n"] < mn or x["share"] is None:
            fails.append(f"side followed T {c} not judged (n {x['n']} < {mn})")
        elif x["share"] < float(M["followed_t"]):
            fails.append(f"side followed T {c} {x['share']} (n {x['n']})")
    x = sm["S2N"]["false_flank"]
    if x["n"] < mn or x["share"] is None:
        fails.append(f"false flank S2N not judged (n {x['n']} < {mn})")
    elif x["share"] > float(M["false_flank"]):
        fails.append(f"false flank S2N {x['share']} (n {x['n']})")
    return not fails, fails


def m3(t, M=None):
    """(passed, fails) of M3's lines (config train.learn.marks) on a set's table."""
    M = M or L()["marks"]
    nmin = int(M["cell_min_n"])
    fails = []
    f = t["followed_N"]["pooled"]
    if f["share"] is None or f["share"] < float(M["followed_n"]):
        fails.append(f"followed N pooled {f['share']} (n {f['n']})")
    for c, x in t["followed_N"]["cells"].items():
        for h in ("NO", "YES"):
            if x[h]["n"] >= nmin and x[h]["share"] < float(M["followed_cell"]):
                fails.append(f"followed N {c} {h} {x[h]['share']} (n {x[h]['n']})")
    if "followed_T" in t:
        f = t["followed_T"]["pooled"]
        if f["share"] is None or f["share"] < float(M["followed_t"]):
            fails.append(f"followed T pooled {f['share']} (n {f['n']})")
    x = t["false_N"]["tick"]
    if x["share"] is not None and x["share"] > float(M["false_tick"]):
        fails.append(f"false N a tick {x['share']} (n {x['n']})")
    for k, x in t["false_N"]["by_kind"].items():
        if x["taught"] and x["n"] >= nmin and x["share"] > float(M["false_kind"]):
            fails.append(f"false N kind {k} {x['share']} (n {x['n']})")
    x = t["nocause_false"]
    if x["n"] and x["share"] > float(M["nocause_false"]):
        fails.append(f"no-cause scenes with a false attack {x['share']} (n {x['n']})")
    x = t["uncaused"]
    if x["n"] and x["share"] > float(M["uncaused"]):
        fails.append(f"uncaused {x['share']} (n {x['n']})")
    return not fails, fails


def brief(t):
    """The numbers of M3 in one dict (the rounds table)."""
    g = lambda x: x["share"]                                                          # noqa: E731
    out = {"fol_N": g(t["followed_N"]["pooled"]), "fol_N_n": t["followed_N"]["pooled"]["n"],
           "fol_N_NO": g(t["followed_N"]["NO"]), "fol_N_YES": g(t["followed_N"]["YES"]),
           "fol_T": g(t["followed_T"]["pooled"]) if "followed_T" in t else None,
           "false_tick": g(t["false_N"]["tick"]),
           "false_kind": {k: g(x) for k, x in t["false_N"]["by_kind"].items() if x["taught"]},
           "false_A5L": g(t["false_N"]["by_recipe"].get("A5L", {"share": None})),
           "false_N2": g(t["false_N"]["by_recipe"].get("N2", {"share": None})),
           "nocause_false": g(t["nocause_false"]), "uncaused": g(t["uncaused"]), "M4": g(t["M4"])}
    out["cells_N"] = {c: [x["NO"]["share"], x["YES"]["share"]] for c, x in t["followed_N"]["cells"].items()}
    n = lambda x: x["n"]                                                              # noqa: E731
    out["n"] = {"fol_N": n(t["followed_N"]["pooled"]), "fol_N_NO": n(t["followed_N"]["NO"]),
                "fol_N_YES": n(t["followed_N"]["YES"]),
                "fol_T": n(t["followed_T"]["pooled"]) if "followed_T" in t else None,
                "false_tick": n(t["false_N"]["tick"]), "nocause_false": n(t["nocause_false"]), "uncaused": n(t["uncaused"]),
                "M4": n(t["M4"]), "false_kind": {k: n(x) for k, x in t["false_N"]["by_kind"].items() if x["taught"]}}
    for p in ("side_T", "side_N"):
        if t.get(p):
            out[p] = {grp: {c: x["share"] for c, x in v.items()} for grp, v in t[p].items()}
            out["n"][p] = {grp: {c: x["n"] for c, x in v.items()} for grp, v in t[p].items()}
    return out


def minus_v0(bv, b0, key="fol_N"):
    """{diff, two_errors, n, n0}: a round's V share minus V0's (brief dicts), two errors by two proportions (the two
    reads' labels are not the same states - N drives - so no pairing by tick)."""
    if not bv or not b0 or bv.get(key) is None or b0.get(key) is None:
        return None
    p1, p0 = float(bv[key]), float(b0[key])
    n1, n0 = int(bv["n"][key]), int(b0["n"][key])
    se = math.sqrt(max(p1 * (1 - p1), 1e-12) / max(n1, 1) + max(p0 * (1 - p0), 1e-12) / max(n0, 1))
    return {"diff": round(p1 - p0, 4), "two_errors": round(2 * se, 4), "n": n1, "n0": n0}


def read(net, specs, scene, seeds, chunk, a, P, kind, mN=None, mT=None, watch=None, paths="NT", teacher="TEACH"):
    """The two reads of specs (module doc) into Marks mN / mT (new ones when None): ({path: player file}, mN, mT);
    teacher - the labeller and path T's driver (drill T0-2: TEACH2)."""
    mN = mN or Marks(net, "N")
    mT = mT or Marks(net, "T")
    files = {}
    net.eval()
    if "N" in paths:
        files["N"], _ = play("N", specs, scene, seeds, chunk, a, P, lambda ii: PN.NetPlayer(net, greedy=True), mN.hook,
                             lambda ii: mN.flush(ii, specs, kind), needs_input=True, watch=watch, labeller=teacher)
    if "T" in paths:
        files["T"], _ = play(teacher, specs, scene, seeds, chunk, a, P, lambda ii: EP.build_under(teacher), mT.hook,
                             lambda ii: mT.flush(ii, specs, kind), needs_input=True, watch=watch, labeller=teacher)
    return files, mN, mT


# ------------------------------------------------------------------ training
def _limit(specs, s):
    for x in specs:
        x["limit"] = min(int(x["limit"]), int(s))
        if x.get("end_s") is not None:
            x["end_s"] = min(float(x["end_s"]), float(s))
    return specs


def s2_specs(seed0, n):
    """n S2 scenes of the seeds seed0.. and their n S2N twins (drill T0-2, scenes.side_scene)."""
    from net.tactic import scenes as SN
    return [SN.side_scene(seed0 + i) for i in range(n)] + [SN.side_scene(seed0 + i, twin=True) for i in range(n)]


def v_specs(smoke=0, side=False):
    """(scenes, general battles) of the validation set V (config drills.attack.sets.V; its general battles on G's
    trained sizes, Random(train.learn.seeds.V_general)); side - + V's S2 / S2N scenes (learn.SIDE_LEARN)."""
    from net.tactic import scenes as SN
    c = scfg()["drills"]["attack"]["sets"]
    sc = SN.set_specs("V")
    if side:
        sc += s2_specs(int(LN.SIDE_LEARN["seeds"]["V"]), int(LN.SIDE_LEARN["V_n"]) if not smoke else 2)
    G = c["G"]
    gen = SP.drill_generals(int(c["V"]["general"]), SP.seeded(int(L()["seeds"]["V_general"])), sizes=G["sizes"],
                            limit=int(G["limit"]))
    if smoke:
        sc, gen = _limit(sc[::max(1, len(sc) // smoke)][:smoke] + (sc[-4:] if side else []), 200), _limit(gen[:smoke], 200)
    return sc, gen


def v_read(net, a, P, chunk, smoke=0, side=False):
    sc, gen = v_specs(smoke, side)
    sd = L()["seeds"]
    t0 = int(sd["V_tseed"])
    mN, mT = Marks(net, "N"), Marks(net, "T")
    teacher = "TEACH2" if side else "TEACH"
    ch = min(chunk, len(sc))
    read(net, sc, True, LN._seeds(t0, len(sc), ch), ch, a, P, "scene", mN, mT, teacher=teacher)
    ch = min(chunk, len(gen))
    read(net, gen, False, LN._seeds(t0 + 50, len(gen), ch), ch, a, P, "general", mN, mT, teacher=teacher)
    out = marks(mN, mT, "V")
    out["done_rows"] = {"N": list(mN.done_cnt), "T": list(mT.done_cnt)}
    if side:
        # M3 apart (REVIEW2 'Stopping'): T0-1's scene types and the general battles (retention), and the S2 / S2N scenes
        out["parts"] = {nm: marks(subset(mN, f), subset(mT, f), "V") for nm, f in (
            ("retention", lambda m: m["recipe"] not in LN.SIDE_RECIPES), ("S2", lambda m: m["recipe"] in LN.SIDE_RECIPES))}
    return out


def subset(m, keep):
    """A Marks read restricted to the battles whose meta passes keep (the others' rows out, their meta kind 'out')."""
    s = Marks(m.net, m.path)
    s.meta = [x if keep(x) else {**x, "kind": "out"} for x in m.meta]
    ok = torch.tensor([keep(x) for x in m.meta], dtype=torch.bool)
    s.rows = [{k: v[ok[t["b"]]] for k, v in t.items()} for t in m.rows]
    s.rows = [t for t in s.rows if t["b"].numel()] or [{k: v[:0] for k, v in m.rows[0].items()}]
    s.holds = m.holds
    return s


def losses(net, t, inp, w):
    """(pointer loss, side loss or None) of a minibatch t (NetBuffer's keys on the device) with its input inp: the
    pointer's cross-entropy (heads.loss) and, for a TacticNet3, the side head's on the attack labels with an approach
    (weight: the label's own w - the change ticks' weight, as T0-1's)."""
    side = isinstance(net, HD.TacticNet3)
    g_row = t["g_row"].long()
    if side:
        enc = net.enc(inp, t["command"], t["cols"], t["own"], g_row, t["g_on"], t["s2"])
    else:
        enc = net.enc(inp, t["command"], t["cols"], t["own"], g_row, t["g_on"])
    lg = net.ptr.logits_all(enc["cards"], t["own"], g_row, t["pair"], t["open"])
    lp, _ = HD.loss(lg, t["label"].long(), w, t["dc"])
    if not side:
        return lp, None
    cards = enc["cards"]
    W = cards.shape[-1]
    lab = t["label"].long()
    ap = t["ap"].long()
    cg = cards.gather(1, g_row[..., None].expand(-1, -1, W))
    ck = cards.gather(1, lab.clamp(min=0)[..., None].expand(-1, -1, W))
    sl = net.side(cg, t["own"], ck, LN.at_label(t["pair"], lab), t["x_k"].float())     # [M, G, 3]
    m = side_mask(t, w)
    ls, _ = HD.side_loss(sl[m], ap[m], w[m])
    return lp, ls


def side_mask(t, w):
    """[M, G] the side labels that count: an attack label with an approach and weight, not at the rule's threshold
    (learn.side_band, config drills.side.bands), and open for the pointer under the execution's mask of its slot (the
    stored walk fed with the teacher's targets and approaches - REVIEW2 finding 2)."""
    lab, ap = t["label"].long(), t["ap"].long()
    return (lab >= 0) & (ap >= 0) & (w > 0) & ~LN.side_band(t["x_k"], ap) & open_at(t["open"], lab)


def side_counts(d, w):
    """The side labels of a round's buffer (weight > 0): all, at the threshold, closed for the pointer, counting."""
    lab, ap = d["label"].long(), d["ap"].long()
    base = (lab >= 0) & (ap >= 0) & (w > 0)
    bd = LN.side_band(d["x_k"], ap) & base
    cl = ~open_at(d["open"], lab) & base
    return {"side_labels": int(base.sum()), "at_threshold": int(bd.sum()), "closed_for_pointer": int(cl.sum()),
            "closed_share": round(int(cl.sum()) / max(int(base.sum()), 1), 4),
            "counting": int((base & ~bd & ~cl).sum())}


def m3_split(tv_parts):
    """(passed, fails) of M3 on V's two parts (T0-1's scene types + general = retention; S2 / S2N)."""
    fails = []
    for nm, t in tv_parts.items():
        ok, f = m3(t)
        fails += [f"[{nm}] {x}" for x in f]
    return not fails, fails


def fit(net, opt, d, w, dev, K, log, gen, passes, mb):
    """Section 7's update: passes over the buffer's group-ticks with weight in minibatches of mb; the input rebuilt
    (rebuild), the pointer on the stored walk (fed with the labeller's choices). Returns the mean loss a pass. A
    TacticNet3 (drill T0-2): loss = pointer + side (losses); the pass -1 - the losses before any update; the history
    {loss, pointer, side} a pass."""
    use = (w.sum(1) > 0).nonzero()[:, 0]
    hist = []
    c = L()
    side = isinstance(net, HD.TacticNet3)
    keys = NET_KEYS + (SIDE_KEYS if side else ())
    hs = {"pointer": [], "side": []}
    for p in range(-1 if side else 0, passes):
        net.train(p >= 0)
        perm = use[torch.randperm(use.numel(), generator=gen)] if p >= 0 else use
        tot, tp, ts, nb, t0 = 0.0, 0.0, 0.0, 0, time.time()
        for i in range(0, perm.numel(), mb):
            idx = perm[i:i + mb]
            t = {k: d[k][idx].to(dev) for k in keys}
            card = d["cards"][d["b"][idx]].to(dev)
            with torch.no_grad():
                inp = rebuild(t, card, K)
            with torch.set_grad_enabled(p >= 0):
                lp, ls = losses(net, t, inp, w[idx].to(dev))
                loss = lp if ls is None else lp + ls
            if p >= 0:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([q for q in net.parameters() if q.requires_grad], float(c["max_grad"]))
                opt.step()
            tot += float(loss.detach())
            tp += float(lp.detach())
            ts += 0.0 if ls is None else float(ls.detach())
            nb += 1
        if p >= 0:
            hist.append(round(tot / max(nb, 1), 5))
        row = {"pass": p, "loss": round(tot / max(nb, 1), 5), "minibatches": nb, "min": round((time.time() - t0) / 60, 1)}
        if side:
            hs["pointer"].append(round(tp / max(nb, 1), 5))
            hs["side"].append(round(ts / max(nb, 1), 5))
            row.update(pointer=hs["pointer"][-1], side=hs["side"][-1])
        log(row)
    net.eval()
    return (hist, hs) if side else hist


def seed_ranges(rounds):
    """{"train": {name: [lo, hi)}, "kept": {name: [lo, hi)}} - the scene seeds (an S2 scene and its S2N twin share one
    seed: twins are the same ranges) and the general battles' Random seeds the training draws in `rounds` rounds, and
    those of the sets it must never see (V, the label probe's, the exams'); a fixed set of recipes is taken whole
    (seed + 10000 x its number of recipes), a superset of its seeds."""
    st = scfg()["drills"]
    at = st["attack"]
    sd = L()["seeds"]
    SL, SP_ = LN.SIDE_LEARN, LN.SIDE_PROBE
    nrec = len([r for r in at["mix"] if r != "E_on_top"]) + 3
    tr, kept = {}, {}
    for k in range(rounds):
        s0 = int(sd["scenes"]) + int(sd["round_step"]) * k
        tr[f"T0-1 scenes r{k + 1}"] = (s0, s0 + 10000 * nrec)
        s2 = int(SL["seeds"]["s2"]) + int(SL["seeds"]["round_step"]) * k
        tr[f"S2 + S2N twins r{k + 1}"] = (s2, s2 + int(SL["s2_round"]))
    tr["general (Random seeds)"] = (int(sd["general"]), int(sd["general"]) + rounds)
    sets = at["sets"]
    for nm in ("S", "V", "F"):
        kept[f"T0-1 {nm}"] = (int(sets[nm]["seed"]), int(sets[nm]["seed"]) + 10000 * len(sets[nm]["counts"]))
    kept["T0-1 H"] = (int(sets["H"]["seed"]), int(sets["H"]["seed"]) + 10000 * len(sets["H"]["axes"]))
    kept["V S2 + S2N"] = (int(SL["seeds"]["V"]), int(SL["seeds"]["V"]) + int(SL["V_n"]))
    kept["probe S2 train"] = (int(SP_["seeds"]["train"]), int(SP_["seeds"]["train"]) + int(SP_["n_train"]))
    kept["probe S2 eval"] = (int(SP_["seeds"]["eval"]), int(SP_["seeds"]["eval"]) + int(SP_["n_eval"]))
    kept["probe held axes"] = (int(SP_["seeds"]["held"]), int(SP_["seeds"]["held"]) + 10000 * len(st["side"]["sets"]["held"]["axes"]))
    pr = at["probe"]["seeds"]
    kept["T0-1 probe scenes (2 rounds)"] = (int(pr["scenes"]), int(pr["scenes"]) + int(pr["round_step"]) + 10000 * nrec)
    kept["T0-1 probe S'"] = (int(pr["S"]), int(pr["S"]) + 10000 * len(sets["S"]["counts"]))
    ss = st["side"]["sets"]
    kept["exam taught"] = (int(ss["taught"]["seed"]), int(ss["taught"]["seed"]) + int(ss["taught"]["n"]))
    kept["exam held"] = (int(ss["held"]["seed"]), int(ss["held"]["seed"]) + 10000 * len(ss["held"]["axes"]))
    kept["general V (Random)"] = (int(sd["V_general"]), int(sd["V_general"]) + 1)
    for nm in ("seed", "held_seed"):
        kept[f"general G {nm} (Random)"] = (int(sets["G"][nm]), int(sets["G"][nm]) + 1)
    kept["general probe (Random)"] = (int(pr["general"]), int(pr["general"]) + 2)
    for nm in ("G", "G_held"):
        kept[f"general probe {nm} (Random)"] = (int(pr[nm]), int(pr[nm]) + 1)
    return {"train": tr, "kept": kept}


def seed_check(rounds, log=None):
    """seed_ranges asserted: no training range meets a kept one (REVIEW.md finding 4); printed once into the log."""
    rg = seed_ranges(rounds)
    bad = [(a, b) for a, (lo, hi) in rg["train"].items() for b, (lo2, hi2) in rg["kept"].items() if lo < hi2 and lo2 < hi]
    if log is not None:
        log({"seed_ranges": {k: {n: list(x) for n, x in v.items()} for k, v in rg.items()}, "rounds": rounds,
             "overlaps": bad})
    assert not bad, f"training seeds meet a kept set: {bad}"
    return rg


def side_round(r, smoke=0):
    """Drill T0-2's scenes of training round r (learn.SIDE_LEARN: s2_round S2 seeds + their S2N twins)."""
    s = LN.SIDE_LEARN
    n = int(s["s2_round"]) if not smoke else max(1, smoke // 2)
    return s2_specs(int(s["seeds"]["s2"]) + int(s["seeds"]["round_step"]) * r, n)


def collect(r, net, a, P, chunk, no_general, smoke, log, side=False):
    """Round r's buffer (section 7 'Round', 'Who drives'); side (drill T0-2): the mix + the side scenes (side_round),
    TEACH2 drives (with the disturbances) and labels."""
    c = L()
    sd = c["seeds"]
    n_sc = float(scfg()["drills"]["attack"]["sets"]["round"]["scenes"])
    n_g = float(scfg()["drills"]["attack"]["sets"]["round"]["general"])
    scale, gscale = (1.0, 1.0) if not smoke else (smoke / n_sc, smoke / n_g)
    scenes, gen = LN.round_specs(r, scale, gscale, seeds=sd)
    if smoke:
        scenes, gen = _limit(scenes[:smoke], 200), _limit(gen[:smoke], 200)
    if side:
        scenes = scenes + (_limit(side_round(r, smoke), 200) if smoke else side_round(r))
    teacher = "TEACH2" if side else "TEACH"
    buf = NetBuffer(check=64 if smoke else 32, side=side)
    rnd = random.Random(int(sd["tseed"]) * 1000 + r)
    dc = c["disturb"]
    stats = {"battles_teacher": 0, "battles_net": 0, "disturbed": 0, "send": 0, "hold": 0, "held_ticks": 0}
    parts = [("scene", scenes, True, 0)] + ([] if no_general else [("general", gen, False, 50)])
    for kind, sp, scene, off in parts:
        if r == 0 and not smoke:
            sp_t, sp_n = sp, []
        else:
            cut = int(round(len(sp) * (1.0 - float(c["net_share"]))))
            perm = list(range(len(sp)))
            rnd.shuffle(perm)
            sp_t, sp_n = [sp[i] for i in sorted(perm[:cut])], [sp[i] for i in sorted(perm[cut:])]
        dis = set(rnd.sample(range(len(sp_t)), int(round(float(dc["share"]) * len(sp_t)))))
        plans = {i: {"kind": rnd.choice(("send", "hold")), "left": rnd.randint(*dc["events"])} for i in dis}
        dseed = rnd.randrange(1 << 30)
        stats["battles_teacher"] += len(sp_t)
        stats["battles_net"] += len(sp_n)
        stats["disturbed"] += len(dis)
        t0 = time.time()
        if sp_t:
            ch = min(chunk, len(sp_t))
            _, drv = play(teacher, sp_t, scene, LN._seeds(int(sd["tseed"]) + 100 * r + off, len(sp_t), ch), ch, a, P,
                          lambda ii: Disturb(EP.build_under(teacher),
                                             {j: copy.deepcopy(plans[i]) for j, i in enumerate(ii) if i in plans},
                                             dseed + ii[0]),
                          lambda d_: buf.hook(False), lambda ii: buf.flush(ii, sp_t, kind, False, dis),
                          labeller=teacher)
            for d_ in drv:
                for k in ("send", "hold", "held_ticks"):
                    stats[k] += d_.n[k]
        if sp_n:
            net.eval()
            ch = min(chunk, len(sp_n))
            play("N", sp_n, scene, LN._seeds(int(sd["tseed"]) + 100 * r + off + 25, len(sp_n), ch), ch, a, P,
                 lambda ii: PN.NetPlayer(net, greedy=True), lambda d_: buf.hook(True),
                 lambda ii: buf.flush(ii, sp_n, kind, True), labeller=teacher)
        log({"collect": kind, "round": r + 1, "teacher": len(sp_t), "net": len(sp_n), "disturbed": len(dis),
             "group_ticks_total": sum(t["b"].shape[0] for t in buf.ticks), "min": round((time.time() - t0) / 60, 1)})
    stats.update(rebuild_checked=buf.checked, rebuild_max_diff=buf.check_max)
    assert buf.check_max <= 1e-3, f"the rebuilt input differs from the level's: {buf.check_max}"
    return buf, stats


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def train(a):
    from sim.params import load
    P = load()
    c = L()
    out, logd = Path(a.out), Path(a.log or a.out)
    out.mkdir(parents=True, exist_ok=True)
    logd.mkdir(parents=True, exist_ok=True)
    logf = open(logd / "train.log", "a", encoding="utf-8")

    def log(x):
        s = json.dumps(x) if not isinstance(x, str) else x
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    dev = a.device
    ar = SimpleNamespace(device=dev, sim=a.sim)
    torch.manual_seed(int(c["seeds"]["tseed"]))
    side = a.drill == "side"
    if side:
        # drill T0-2: the network of T0-1 (tactic_v2) in a TacticNet3, the new layers at their start (ckpt.from_v2)
        from net.tactic import ckpt as TK
        net, _ = TK.from_v2(a.init, dev)
    else:
        net = HD.TacticNet2().to(dev)
    opt = torch.optim.Adam([q for q in net.parameters() if q.requires_grad], lr=float(c["lr"]))
    gen = torch.Generator().manual_seed(int(c["seeds"]["tseed"]))
    rounds = min(int(a.rounds or c["rounds"]), int(c["rounds"]))
    chunk = int(c["chunk"])
    passes = int(a.passes or c["passes"])
    log({"start": time.strftime("%H:%M"), "rounds_max": rounds, "no_general": a.no_general, "device": dev, "sim": a.sim,
         "sim_key": EP.sim_key(), "reach_m": OR.c_orders()["reach_m"], "taught": TG.TAUGHT, "cols": HD.d2_cols()[0],
         "smoke": a.smoke, "learn": {k: v for k, v in c.items() if not k.startswith("_")}, "drill": a.drill,
         "passes": passes, **({"init": a.init, "side_learn": LN.SIDE_LEARN, "side_marks": SIDE_MARKS} if side else {})})
    table, passed, shares, final = [], [], [], None
    r0 = 0
    key = run_key(a, side, rounds, passes)
    if side:
        seed_check(int(c["rounds"]), log)
        log({"stop_rule": f"from round {STOP_FROM} on, two rounds running with ALL of: T0-1's M3 on V's retention part "
                          "(T0-1's scene types + general) AND on its S2 / S2N part (imitate.m3 on each: followed on N's "
                          "and T's path, false attacks, no-cause, uncaused - config train.learn.marks) AND the side marks "
                          "on V's teacher path (imitate.side_ok: front / left / right followed >= drills.side.marks."
                          "followed_t each, false flank on S2N <= false_flank, each with n >= drills.side.marks.min_n, "
                          "fewer = not judged = no stop); final.pt = the last round's weights either way"})
    ckr = None
    if a.resume:
        # the same training continued in a new process (the earlier one cut), said explicitly: the network, the
        # optimiser, the minibatch generator and torch's random state from the checkpoint of round from_round - 1
        ckr = torch.load(a.resume, map_location=dev, weights_only=False)
        r0 = int(a.from_round) - 1
        assert int(ckr.get("round", r0)) == r0, (ckr.get("round"), r0)
        src = str(a.resume)
    else:
        # the keeper restarts the same command after a crash: the latest complete checkpoint of THIS run in --out (its
        # run_key) is taken up; a checkpoint of another run there is refused (REVIEW.md finding 7)
        cks = sorted(out.glob("round_[0-9][0-9].pt"))
        if cks:
            ckr = torch.load(cks[-1], map_location=dev, weights_only=False)
            if ckr.get("run_key") != key:
                raise SystemExit(f"{cks[-1]}: a checkpoint of another run (run_key {ckr.get('run_key')} != {key}) in "
                                 f"--out {out} - give a clean folder or --resume explicitly")
            r0 = int(ckr["round"])
            src = str(cks[-1])
    if ckr is not None:
        net.load_state_dict(ckr["net"])
        opt.load_state_dict(ckr["opt"])
        if ckr.get("gen_state") is not None:
            gen.set_state(ckr["gen_state"].cpu())
        else:
            gen = torch.Generator().manual_seed(int(c["seeds"]["tseed"]) + 1000 * (r0 + 1))
        if ckr.get("rng") is not None:
            torch.set_rng_state(ckr["rng"].cpu())
        if ckr.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([x.cpu() for x in ckr["cuda_rng"]])
        lf = logd / "train.log"
        for line in (lf.read_text(encoding="utf-8", errors="replace").splitlines() if lf.exists() else []):
            if line.startswith('{"ROUND"'):
                try:
                    row = json.loads(line)["ROUND"]
                except (ValueError, KeyError):
                    # (a torn last line of a crashed process: dropped, the row is rebuilt from its checkpoint below)
                    log({"TORN_ROUND_LINE_DROPPED": line[:80]})
                    continue
                if row["round"] <= r0:
                    table = [x for x in table if x["round"] != row["round"]] + [row]
        have = {x["round"] for x in table}
        for rr in range(1, r0 + 1):
            if rr in have:
                continue
            # the checkpoint of round rr is complete but its ROUND line never reached the log: its row from the
            # checkpoint, written once now (no duplicate line, no double count)
            p = out / f"round_{rr:02d}.pt"
            row = torch.load(p, map_location="cpu", weights_only=False).get("row") if p.exists() else None
            assert row is not None, f"round {rr}: no ROUND line in {lf} and no row in {p}"
            log({"ROUND_FROM_CKPT": rr})
            log({"ROUND": row})
            table.append(row)
        table.sort(key=lambda x: x["round"])
        assert [x["round"] for x in table] == list(range(1, r0 + 1)), [x["round"] for x in table]
        for x in table:
            passed.append(bool(x["stop_ok"]) if "stop_ok" in x else
                          (bool(x["M3_V"]) and (not side or bool(x.get("side_V")))))
            if x["label_share_net_states"] is not None:
                shares.append(x["label_share_net_states"])
        log({"RESUME": src, "from_round": r0 + 1, "rounds_read": len(table), "passed_v": passed,
             "minibatch_generator": "restored" if ckr.get("gen_state") is not None else
             int(c["seeds"]["tseed"]) + 1000 * (r0 + 1)})
        if len(passed) >= 2 and passed[-1] and passed[-2] and (not side or r0 >= STOP_FROM):
            final = out / f"round_{r0:02d}.pt"
            log({"STOP": f"(restored) M3 and the side marks held on V at rounds {r0 - 1} and {r0}"})
    t_all = time.time()
    if side and r0 == 0 and not a.no_v and not (logd / "v_round_00.json").exists():
        # (the start's marks on V - the reference a round's M3 is compared with; written once, a restart skips it)
        tv0 = v_read(net, ar, P, chunk, a.smoke, side=True)
        (logd / "v_round_00.json").write_text(json.dumps(tv0, indent=1), encoding="utf-8")
        log({"V0": brief(tv0), "M3_V0": m3(tv0)[0], "side_V0": side_ok(tv0.get("side_T"))})
    b0 = None
    if side and (logd / "v_round_00.json").exists():
        b0 = brief(json.loads((logd / "v_round_00.json").read_text(encoding="utf-8")))
    for r in range(r0, rounds if final is None else r0):
        t0 = time.time()
        buf, st = collect(r, net, ar, P, chunk, a.no_general, a.smoke, log, side=side)
        d, meta = buf.data(), buf.meta
        w, winfo = LN.weights(d, meta)
        item = d["head"] & ~d["drop"]
        att = item & (d["label"] >= 0)
        tw = d["tw"].float()[:, None]
        on_net = d["net"][:, None]
        share_net = float((tw * (att & on_net)).sum() / (tw * (d["head"] & on_net)).sum().clamp(min=1e-9)) \
            if bool(d["net"].any()) else None
        nbytes = sum(t.numel() * t.element_size() for t in d.values())
        log({"round": r + 1, "buffer_group_ticks": int(d["head"].shape[0]), "unit_ticks_head_free": int(d["head"].sum()),
             "attack_labels": int(att.sum()), "attack_labels_net_states": int((att & on_net).sum()),
             "label_share_net_states": share_net, "bytes": nbytes, **st})
        log({"weights": winfo})
        extra = {}
        if side:
            aps = d["ap"][att & (d["ap"] >= 0)]
            extra["side_labels"] = {n: int((aps == i).sum()) for i, n in enumerate(OR.APPROACHES)}
            extra["flank_weighted"] = flank_weight_share(d, meta, w)
            extra["side_count"] = side_counts(d, w)
            extra["done_rows"] = {k: {"rows": v[0], "left_out": v[1], "share": round(v[1] / max(v[0], 1), 4)}
                                  for k, v in buf.done_stats.items()}
            log({"side_labels": extra["side_labels"], "flank_weighted": extra["flank_weighted"],
                 "side_count": extra["side_count"], "done_rows": extra["done_rows"]})
        hist = fit(net, opt, d, w, dev, buf.NK[1], log, gen, passes, int(c["minibatch"]))
        if side:
            hist, extra["losses"] = hist
        del d, w, buf
        ck = out / f"round_{r + 1:02d}.pt"
        if a.no_v:
            tv = None
            ok, fails, sok, sfails = False, ["V not read (--no-v)"], False, []
        else:
            tv = v_read(net, ar, P, chunk, a.smoke, side=side)
            ok, fails = m3(tv)
            sok, sfails = side_ok(tv.get("side_T")) if side else (True, [])
        if side and tv is not None:
            pok, pfails = m3_split(tv["parts"])
            stop_ok = pok and sok
            fails = pfails
        else:
            stop_ok = ok and sok
        bv = brief(tv) if tv is not None else None
        if side:
            mv = {"M3 fol_N minus V0": minus_v0(bv, b0), "M3 fol_T minus V0": minus_v0(bv, b0, "fol_T")}
            log({"round": r + 1, "V_n": bv["n"] if bv else None, **mv,
                 "M3_parts": {k: brief(x) for k, x in tv["parts"].items()} if tv is not None else None,
                 "done_rows_V": tv.get("done_rows") if tv is not None else None})
        row = {"round": r + 1, "drove": "teacher" if (r == 0 and not a.smoke) else "teacher + network (half)",
               "battles": st["battles_teacher"] + st["battles_net"], "labels": int(att.sum()),
               "label_share_net_states": share_net, "loss": hist, "V": bv, "M3_V": ok, "fails": fails + sfails,
               "ckpt": str(ck), "min": round((time.time() - t0) / 60, 1),
               "stop_ok": stop_ok, **({"side_V": sok, **extra, **mv} if side else {})}
        if tv is not None:
            (logd / f"v_round_{r + 1:02d}.json").write_text(json.dumps(tv, indent=1), encoding="utf-8")
        state = {"row": row, "run_key": key, "gen_state": gen.get_state(), "rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        if side:
            TK_save3(ck, net, opt, a, round=r + 1, marks_v=bv, passed_v=ok, side_v=sok, **state)
        else:
            TK_save(ck, net, opt, a, round=r + 1, marks_v=bv, passed_v=ok, **state)
        passed.append(stop_ok)
        table.append(row)
        log({"ROUND": row})
        if share_net is not None:
            shares.append(share_net)
            if len(shares) >= 3 and shares[-3] < shares[-2] < shares[-1]:
                log("!!!!!! LOUD: the share of attack labels on the network's own states grew three rounds running "
                    f"{[round(x, 5) for x in shares[-3:]]} - look at two battles (design section 7); going on")
        if len(passed) >= 2 and passed[-1] and passed[-2] and (not side or r + 1 >= STOP_FROM):
            final = ck
            log({"STOP": f"M3{' and the side marks' if side else ''} held on V at rounds {r} and {r + 1}"})
            break
    if final is None:
        final = out / f"round_{len(table):02d}.pt"
        log({"NO_STOP": f"{len(table)} rounds without M3 two rounds running", "fails_last": table[-1]["fails"]})
    part = out / "final.pt.part"
    shutil.copyfile(final, part)
    os.replace(part, out / "final.pt")
    held = len(passed) >= 2 and passed[-1] and passed[-2] and (not side or len(passed) >= STOP_FROM)
    res = {"rounds": table, "final": str(final), "final_copy": str(out / "final.pt"), "sha256": _sha(out / "final.pt"),
           "stopped": held, "minutes": round((time.time() - t_all) / 60, 1)}
    (logd / "train.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log({"done": res["final"], "sha256": res["sha256"], "stopped": res["stopped"], "minutes": res["minutes"]})
    log({"STOP_CONDITION_HELD": held, "last_round_stop_ok": passed[-1] if passed else None,
         "note": "final.pt = the last round's weights; converged only if STOP_CONDITION_HELD"})
    logf.close()
    return res


def run_key(a, side, rounds, passes):
    """sha256 (16) of what makes a training run itself: the drill, its start's weights, rounds, passes, the draw's
    options and the configs it reads - a restart of the same command has the same key."""
    from net.tactic import cfg as tc
    x = {"drill": a.drill, "init": _sha(a.init) if side else None, "rounds": rounds, "passes": passes, "smoke": a.smoke,
         "no_general": a.no_general, "no_v": a.no_v, "sim": a.sim, "sim_key": EP.sim_key(), "learn": L(),
         "side": tc()["drills"].get("side"), "orders": tc()["orders"], "stage0": scfg()["drills"]}
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:16]


def flank_weight_share(d, meta, w):
    """Acceptance line 1 of the T0-2 fix (REVIEW.md finding 1): of the flank-labelled attack ticks (teacher approach left
    / right) the share with loss weight > 0 - all and on the scenes whose start is second_row; and the same with T0-1's
    through-own exclusion put back (dc_thr in the drop) - the number before the fix, from the same buffer."""
    lab, ap = d["label"] >= 0, d["ap"]
    fl = d["head"] & lab & ((ap == OR.LEFT) | (ap == OR.RIGHT))
    sr = torch.tensor([m.get("start") == "second_row" for m in meta], dtype=torch.bool)[d["b"]][:, None] & fl
    out = {}
    w0, _ = LN.weights({**d, "drop": d["drop"] | (d["dc_thr"] & lab)}, meta)
    for nm, ww in (("with_fix", w), ("without_fix", w0)):
        out[nm] = {"all": LN.share(int((fl & (ww > 0)).sum()), int(fl.sum())),
                   "second_row": LN.share(int((sr & (ww > 0)).sum()), int(sr.sum()))}
    return out


def TK_save(path, net, opt, a, **more):
    from net.tactic import ckpt as TK
    TK.save_v2(path, net, opt, ["T0-2"], unit_commit=a.unit, sim_commit=a.sim, **more)


def TK_save3(path, net, opt, a, **more):
    # (the journal's ids: T0-1 'бить связанного' - the start's weights, a.init -, and T0-2 'сторона удара')
    from net.tactic import ckpt as TK
    TK.save_v3(path, net, opt, ["T0-1", "T0-2"], unit_commit=a.unit, sim_commit=a.sim, init=a.init,
               init_sha256=_sha(a.init), **more)


# ------------------------------------------------------------------ the exam after
def _same_file(a_, b_):
    bad = [k for k in b_["per_battle"] if not EP.same(b_["per_battle"][k], a_["per_battle"].get(k))]
    return {"per_battle_differ": bad, "hp_series": EP.same(a_["hp_series"], b_["hp_series"]),
            "order_log": EP.same(EP._log_key(a_["order_log"]), EP._log_key(b_.get("order_log", []))),
            "bit_for_bit": not bad and EP.same(a_["hp_series"], b_["hp_series"])
            and EP.same(EP._log_key(a_["order_log"]), EP._log_key(b_.get("order_log", [])))}


def m7(fN, fT):
    pb = lambda f: f["per_battle"]["game_orders_min"]                                # noqa: E731
    d = EP.paired(pb(fN), pb(fT))
    return {"N": EP._mean(pb(fN)), "TEACH": EP._mean(pb(fT)), "N - TEACH": d,
            "passed": bool(d and d["diff"] <= float(L()["marks"]["m7"]))}


def m5(tS, tH):
    """M5 by held-out axis (followed on N's path, false attacks a tick) against S's."""
    M = L()["marks"]
    fS, xS = tS["followed_N"]["pooled"], tS["false_N"]["tick"]
    out = {}
    for ax in sorted(tH["false_N"]["by_recipe"]):
        fa = tH["followed_N"]["cells_all"].get(ax)
        xa = tH["false_N"]["by_recipe"][ax]
        row = {"followed": fa, "false_tick": xa}
        if fa and fa["n"] and fS["n"]:
            se = math.sqrt(fS["share"] * (1 - fS["share"]) / fS["n"] + fa["share"] * (1 - fa["share"]) / fa["n"])
            row["S_minus_axis"] = round(fS["share"] - fa["share"], 4)
            row["two_errors"] = round(2 * se, 4)
            row["followed_ok"] = fa["share"] >= float(M["m5_followed"]) and row["S_minus_axis"] <= row["two_errors"]
        if xa["n"] and xS["n"]:
            row["false_ok"] = xa["share"] <= 2 * xS["share"] + (xa["two_errors"] or 0.0)
        row["passed"] = bool(row.get("followed_ok", False) and row.get("false_ok", True))
        out[ax] = row
    return out


def after(a):
    from sim.params import load
    from net.tactic import ckpt as TK
    from net.tactic import scenes as SN
    P = load()
    D = Path(a.out)
    D.mkdir(parents=True, exist_ok=True)
    logf = open(D / "after.log", "a", encoding="utf-8")

    def log(x):
        s = json.dumps(x) if not isinstance(x, str) else x
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    dev = a.device
    ar = SimpleNamespace(device=dev, sim=a.sim)
    net, _, ck = TK.restore(a.ckpt, dev)
    net.eval()
    chunk = int(L()["chunk"])
    sS, tS, chS = EP.s_set(chunk, "S")
    H = SN.held_specs()
    sG, tG, chG = EP.g_set()
    sets = {"S": (sS, tS, chS, True), "H": (H, LN._seeds(int(L()["seeds"]["H_tseed"]), len(H), chunk), chunk, True),
            "G": (sG, tG, chG, False)}
    if a.smoke:
        sets = {k: (_limit(copy.deepcopy(sp[::max(1, len(sp) // a.smoke)][:a.smoke]), 200), sd[:1],
                    min(a.smoke, ch), sc) for k, (sp, sd, ch, sc) in sets.items()}
    log({"start": time.strftime("%H:%M"), "ckpt": a.ckpt, "sha256": _sha(a.ckpt), "round": ck.get("round"),
         "sim": a.sim, "sim_key": EP.sim_key(), "sets": {k: len(v[0]) for k, v in sets.items()}})
    tabs, files, mks = {}, {}, {}
    for sname, (sp, sd, ch, scene) in sets.items():
        t0 = time.time()
        fl, mN, mT = read(net, sp, scene, sd, ch, ar, P, "scene" if scene else "general", watch=EP.HoleWatch)
        for p, f in fl.items():
            (D / f"{sname}_{'N' if p == 'N' else 'TEACH'}.json").write_text(json.dumps(f), encoding="utf-8")
        files[sname] = fl
        t = marks(mN, mT, sname)
        ok, fails = m3(t)
        t["M3"] = {"passed": ok, "fails": fails}
        t["M4_passed"] = bool(t["M4"]["n"] and t["M4"]["share"] >= float(L()["marks"]["m4"]))
        t["M7"] = m7(fl["N"], fl["T"])
        tabs[sname] = t
        mks[sname] = mN
        log({"set": sname, "M3": t["M3"], "brief": brief(t), "M4": t["M4"], "M7": t["M7"],
             "min": round((time.time() - t0) / 60, 1)})
    tabs["M5"] = m5(tabs["S"], tabs["H"])
    # C on S (the generator changed after stage 3b) and G (stage 3b's file when it is this simulator's and these battles)
    stated = {}
    for sname in ("G", "S"):
        sp, sd, ch, scene = sets[sname]
        ref = Path(a.g_ref) / "G_C.json" if sname == "G" and a.g_ref else None
        fC = None
        if ref is not None and ref.exists() and not a.smoke:
            x = json.loads(ref.read_text(encoding="utf-8"))
            if x.get("sim_key") == EP.sim_key() and x.get("meta") == EP.meta_of(sp):
                fC = x
                log({"C": sname, "reused": str(ref)})
            tref = Path(a.g_ref) / "G_TEACH.json"
            if tref.exists():
                y = json.loads(tref.read_text(encoding="utf-8"))
                log({"bitcheck_TEACH_G_against": str(tref), **_same_file(files["G"]["T"], y)})
                tabs["bitcheck_TEACH_G"] = _same_file(files["G"]["T"], y)
        if fC is None:
            fC = EP.watched_run("C", sp, sd, ch, ar, P, scene=scene)
            (D / f"{sname}_C.json").write_text(json.dumps(fC), encoding="utf-8")
        fN, fT = files[sname]["N"], files[sname]["T"]
        hp = lambda f: f["per_battle"]["hp_trade"]                                   # noqa: E731
        ao = lambda f: EP._mean(f["per_battle"]["attack_orders"])                    # noqa: E731
        stated[sname] = {"N - C": EP.paired(hp(fN), hp(fC)), "TEACH - C": EP.paired(hp(fT), hp(fC)),
                         "N - TEACH": EP.paired(hp(fN), hp(fT)),
                         "attack_orders_battle": {"N": ao(fN), "TEACH": ao(fT), "C": ao(fC)}}
        if sname == "S":
            cause = [b for b, n in enumerate(tabs["S"]["per_battle"]["labels"]) if n > 0]
            stated[sname]["N - C, scenes with a label on N's path"] = EP.paired(hp(fN), hp(fC), cause)
        log({"stated": sname, **stated[sname]})
    tabs["stated"] = stated
    # the records for the eyes
    (D / "after.json").write_text(json.dumps(tabs, indent=1), encoding="utf-8")
    if a.vision:
        try:
            tabs["vision"] = vision_picks(net, tabs, sets, ar, P, D, log)
        except Exception as e:                                                       # (the tables are written already)
            tabs["vision"] = {"error": repr(e)}
            log({"vision_error": repr(e)})
        (D / "after.json").write_text(json.dumps(tabs, indent=1), encoding="utf-8")
    log({"done": str(D / "after.json")})
    logf.close()
    return tabs


def _seed_of(specs, seeds, chunk, b):
    order = sorted(range(len(specs)), key=lambda i: int(specs[i]["limit"]))
    return seeds[order.index(b) // chunk]


def vision_picks(net, tabs, sets, a, P, D, log):
    """Three battles for game.vision with N driving (design M11): the most uncaused attacks (S or G), a no-cause scene
    with a false attack (S; if any), a YES battle of G (the most YES labels); each alone, the torch seed of its chunk
    in the exam (one battle alone does not replay the batch's random stream bit for bit)."""
    picks = []
    best = max(((s, b, n) for s in ("S", "G") for b, n in enumerate(tabs[s]["per_battle"]["uncaused"])),
               key=lambda x: x[2], default=None)
    if best and best[2] > 0:
        picks.append(("most_uncaused", best[0], best[1], best[2]))
    pb = tabs["S"]["per_battle"]
    nc = [(b, pb["false"][b]) for b in range(len(pb["false"])) if pb["nocause"][b] and pb["false"][b] > 0]
    if nc:
        b, n = max(nc, key=lambda x: x[1])
        picks.append(("nocause_false", "S", b, n))
    pg = tabs["G"]["per_battle"]["yes_labels"]
    if pg and max(pg) > 0:
        b = max(range(len(pg)), key=lambda i: pg[i])
        picks.append(("G_yes", "G", b, pg[b]))
    out = []
    for why, sname, b, n in picks:
        sp, sd, ch, scene = sets[sname]
        outd = D / "vision" / f"{why}_{sname}_{b}"
        res = EP.vision_record("N", sp[b], scene, [_seed_of(sp, sd, ch, b)], a, P, lambda: PN.NetPlayer(net, greedy=True),
                               outd)
        out.append({"why": why, "set": sname, "battle": b, "count": n, "recipe": sp[b].get("recipe", "G"),
                    "dir": str(outd), "attack_orders": sum(res["per_battle"]["attack_orders"])})
        log({"vision": out[-1]})
    return out


# ------------------------------------------------------------------ drill T0-2: the exam after (METHOD step 6)
SIDE_PLAYERS = ("C", "TEACH", "TEACH2", "N", "N1")
SIDE_PAIRS = (("N", "TEACH"), ("N", "TEACH2"), ("N", "C"), ("N", "N1"), ("N1", "TEACH"), ("TEACH2", "TEACH"),
              ("TEACH2", "C"), ("TEACH", "C"))


def net_side_run(net, specs, seeds, chunk, a, P, scene=True):
    """exam_side.side_run with the network as the player (pin.NetPlayer greedy) - the same watch, the same file."""
    from net.tactic import exam_side as ES
    ex = {"strikes": []}

    def look(ii, watch, side_):
        ex["strikes"] += [[ii[r[0]]] + r[1:] for r in watch.strikes if r[0] < len(ii)]
    res = EP.run_player("N", specs, seeds, chunk, a, P, make=lambda: PN.NetPlayer(net, greedy=True), on_chunk=look,
                        watch_cls=ES.SideWatch, full=True, by_limit=True, **_kw(scene))
    res.update(strikes=ex["strikes"])
    return res


def side_comer(mt, pl):
    """exam_side.comer_scenes + the battles where a flank walk of the network (N) met a comer (scripts.Unit.comer)."""
    from net.tactic import exam_side as ES
    cm = ES.comer_scenes(mt, pl)
    seen = {int(r[0]) for p in ("N",) if p in pl for r in pl[p].get("strikes", []) if r[4] > 0 and len(r) > 9 and r[9]}
    return {**cm, "seen_N": sorted(seen), "all": sorted(set(cm["all"]) | seen)}


def side_table(D, names):
    """exam_side.table's reads with the network's players and pairs (SIDE_PLAYERS, SIDE_PAIRS) and side_comer."""
    from net.tactic import exam_side as ES
    out = {}
    for nm in names:
        files = {p: D / f"{nm}_{p}.json" for p in SIDE_PLAYERS if (D / f"{nm}_{p}.json").exists()}
        if "C" not in files:
            continue
        pl = {p: json.loads(f.read_text(encoding="utf-8")) for p, f in files.items()}
        mt = json.loads((D / f"{nm}_meta.json").read_text(encoding="utf-8"))
        R = {p: EP.reads(d, pl["C"]["per_battle"]["t_contact"]) for p, d in pl.items()}
        cm = side_comer(mt, pl)
        tab = {"battles": len(mt), "comer_scenes": cm, "pairs": {}, "orders_min": {}, "exec": {}, "n_battles": {}}
        for lab, sel in ES.groups_of(mt, nm, set(cm["all"])):
            tab["n_battles"][lab] = len(sel)
            for x, y in SIDE_PAIRS:
                if x in pl and y in pl:
                    tab["pairs"].setdefault(f"{x} - {y}", {})[lab] = {
                        "i_300": EP.paired(R[x]["i"], R[y]["i"], sel), "iii_end": EP.paired(R[x]["iii"], R[y]["iii"], sel),
                        "orders": EP.paired(pl[x]["per_battle"]["game_orders_min"],
                                            pl[y]["per_battle"]["game_orders_min"], sel)}
            for p, d in pl.items():
                tab["orders_min"].setdefault(p, {})[lab] = EP._mean([d["per_battle"]["game_orders_min"][b] for b in sel])
                if "strikes" in d:
                    tab["exec"].setdefault(p, {})[lab] = ES._flank_exec(d, sel)
        out[nm] = tab
    return out


def pool_all(parts):
    """exam_side._pool of parts, None (not judged) when any part is missing (REVIEW2 finding 6)."""
    from net.tactic import exam_side as ES
    return None if any(p is None for p in parts) else ES._pool(parts)


def _piece(passed, **x):
    """A verdict piece: passed True / False, None = not judged (a missing or undefined number)."""
    return {**x, "passed": passed, "status": {True: "passed", False: "failed", None: "not judged"}[passed]}


def line_c_of(out, player):
    """exam_side's line c (first contact on the target's flank) for player - exam_side's own definition: its line_c
    when it has one, else its lines() read on a copy where the player's execution stands in TEACH2's place."""
    from net.tactic import exam_side as ES
    if hasattr(ES, "line_c"):
        return ES.line_c(out, player)
    o2 = {nm: ({**t, "exec": {**t["exec"], "TEACH2": t["exec"].get(player, {})}} if isinstance(t, dict) and "exec" in t
               else t) for nm, t in out.items() if nm != "lines"}
    return ES.lines(o2)["c"]


def side_lines(out, t01=None):
    """METHOD step 6's verdict of drill T0-2 (REVIEW.md finding 3): six pieces, each must be present and hold - a
    missing piece is 'not judged' = not passed. The comer battles out everywhere (exam_side.comer_scenes).
      i    closes: N - TEACH on the cause scenes (taught + held pooled) >= half of TEACH2 - TEACH (and TEACH2 - TEACH > 0)
      ii   executes: N's flank orders a cause scene >= half of TEACH2's, and N's first contact on the flank (exam_side's
           line c, its own definition) >= 0.75
      iii  no harm: N - TEACH on the no-cause scenes (pooled) and on G not below -2 errors
      iv   held: (N - TEACH) held cause - taught cause not below -2 errors
      v    T0-1 retained: t01_compare (N3 - C on T0-1's sets against the reference's N - C) passed
      vi   orders: N's orders to the game a unit-minute <= 1.2 on every set (taught, held, G)"""
    from net.tactic import exam_side as ES
    X = " [-comer]"
    g = lambda nm, lab, pair: ES._get(out, nm, lab, pair=pair)                       # noqa: E731
    res = {}
    # i
    nt = pool_all([g("taught", "cause" + X, "N - TEACH"), g("held", "cause" + X, "N - TEACH")])
    tt = pool_all([g("taught", "cause" + X, "TEACH2 - TEACH"), g("held", "cause" + X, "TEACH2 - TEACH")])
    ok = None if not (nt and tt and tt["diff"] > 0) else nt["diff"] >= 0.5 * tt["diff"]
    res["i_closes"] = _piece(ok, rule="N - TEACH >= 0.5 x (TEACH2 - TEACH) on the S2 cause scenes (taught + held pooled, "
                                      "comer out); TEACH2 - TEACH <= 0 or missing: not judged",
                             N_TEACH=nt, TEACH2_TEACH=tt,
                             share_closed=round(nt["diff"] / tt["diff"], 3) if nt and tt and tt["diff"] else None)
    # ii
    def per_scene(p):
        fo, nb = 0, 0
        for nm in ("taught", "held"):
            e = out.get(nm, {}).get("exec", {}).get(p, {}).get("cause" + X)
            n = out.get(nm, {}).get("n_battles", {}).get("cause" + X)
            if e is None or not n:
                return None
            fo, nb = fo + e["flank_orders"], nb + n
        return {"flank_orders": fo, "cause_scenes": nb, "a_scene": round(fo / nb, 4)}
    fN, fT = per_scene("N"), per_scene("TEACH2")
    cN = line_c_of(out, "N") if all(nm in out for nm in ("taught", "held", "G")) else None
    a_ok = None if not (fN and fT and fT["a_scene"] > 0) else fN["a_scene"] >= 0.5 * fT["a_scene"]
    c_ok = None if not cN or cN.get("share") is None else cN["share"] >= 0.75
    ok = False if (a_ok is False or c_ok is False) else (None if (a_ok is None or c_ok is None) else True)
    res["ii_executes"] = _piece(ok, rule="N's flank orders a cause scene (comer out) >= 0.5 x TEACH2's, and N's first "
                                         "contact on the flank (exam_side line c) >= 0.75",
                                flank_orders_N=fN, flank_orders_TEACH2=fT,
                                orders_ok={True: "passed", False: "failed", None: "not judged"}[a_ok],
                                line_c_N=cN, line_c_ok={True: "passed", False: "failed", None: "not judged"}[c_ok])
    # iii
    nn = pool_all([g("taught", "no cause" + X, "N - TEACH"), g("held", "no cause" + X, "N - TEACH")])
    gt = g("G", "all" + X, "N - TEACH")
    ok = None if not (nn and gt) else (nn["diff"] >= -nn["two_errors"] and gt["diff"] >= -gt["two_errors"])
    res["iii_no_harm"] = _piece(ok, rule="N - TEACH >= -2 errors on the no-cause scenes (taught + held pooled) and on G "
                                         "(comer out)", no_cause=nn, G=gt,
                                G_N_N1=g("G", "all" + X, "N - N1"), G_N_TEACH2=g("G", "all" + X, "N - TEACH2"))
    # iv
    t, h = g("taught", "cause" + X, "N - TEACH"), g("held", "cause" + X, "N - TEACH")
    if t and h:
        e2 = round(math.sqrt(t["two_errors"] ** 2 + h["two_errors"] ** 2), 4)
        res["iv_held"] = _piece(h["diff"] - t["diff"] >= -e2, rule="(N - TEACH) held cause - taught cause >= -2 errors",
                                taught=t, held=h, gap=round(h["diff"] - t["diff"], 4), two_errors=e2,
                                by_axis={lab: x for lab, x in out["held"]["pairs"].get("N - TEACH", {}).items()
                                         if lab.startswith("held ") and lab.endswith(", cause" + X)})
    else:
        res["iv_held"] = _piece(None, rule="(N - TEACH) held cause - taught cause >= -2 errors", taught=t, held=h)
    # v
    if t01 is None:
        res["v_t01"] = _piece(None, rule="T0-1 retained (t01_compare)", note="--t01-new / --t01-ref not given")
    else:
        res["v_t01"] = _piece(t01.get("passed"), rule="T0-1 retained (t01_compare)", **{k: v for k, v in t01.items()
                                                                                       if k != "passed"})
    # vi
    om = {nm: out.get(nm, {}).get("orders_min", {}).get("N", {}).get("all" + X) for nm in ("taught", "held", "G")}
    known = all(isinstance(v, (int, float)) and math.isfinite(v) for v in om.values())
    res["vi_orders"] = _piece(None if not known else all(v <= 1.2 for v in om.values()),
                              rule="N's orders to the game a unit-minute <= 1.2 (METHOD: 1.5 x human 0.8) on taught, held, "
                                   "G (comer out)", by_set=om)
    res["comer_alone"] = {nm: g(nm, "comer", "N - TEACH") for nm in ("taught", "held", "G")}
    pieces = ("i_closes", "ii_executes", "iii_no_harm", "iv_held", "v_t01", "vi_orders")
    res["passed"] = all(res[p]["passed"] is True for p in pieces)
    res["verdict"] = "passed" if res["passed"] else "NOT passed"
    res["reasons"] = [f"{p}: {res[p]['status']}" for p in pieces if res[p]["passed"] is not True]
    return res


def t01_compare(new_dir, ref_dir):
    """Piece v: T0-1 retained - `learn after --ckpt` (T0-1's exam) of the new network (new_dir) against T0-1's network
    (ref_dir, tmp/drill1_w/before): on each of T0-1's sets with an outcome (S, G) the paired N3 - N1 of the battle's
    health trade (the same battles: the files' meta and simulator must agree, else not judged) >= -2 errors; T0-1's
    marks (followed on N's path, false attacks a tick) printed beside. A missing folder, file or set: not judged."""
    nd, rd = Path(new_dir), Path(ref_dir)
    res = {"new": str(nd), "ref": str(rd), "sets": {}}
    if nd.resolve() == rd.resolve():
        return {**res, "passed": None, "note": "--t01-new and --t01-ref are the same folder: refused (REVIEW2 finding 6)"}
    try:
        a_, b_ = (json.loads((Path(x) / "after.json").read_text(encoding="utf-8")) for x in (nd, rd))
    except (OSError, ValueError) as e:
        return {**res, "passed": None, "note": f"after.json unreadable: {e!r}"}
    for s in ("S", "G"):
        fa, fb = nd / f"{s}_N.json", rd / f"{s}_N.json"
        if not (fa.exists() and fb.exists()):
            res["sets"][s] = {"passed": None, "note": "no N file"}
            continue
        x, y = (json.loads(f.read_text(encoding="utf-8")) for f in (fa, fb))
        if x.get("sim_key") != y.get("sim_key") or x.get("meta") != y.get("meta"):
            res["sets"][s] = {"passed": None, "note": "the files' simulator or battles differ: the reference is to be "
                                                      "replayed on this simulator", "sim_key": [x.get("sim_key"), y.get("sim_key")]}
            continue
        hp = lambda f: f["per_battle"]["hp_trade"]                                   # noqa: E731
        d = EP.paired(hp(x), hp(y))
        r = {"N3 - N1": d, "N3 - C": (a_.get("stated", {}).get(s) or {}).get("N - C"),
             "N1 - C": (b_.get("stated", {}).get(s) or {}).get("N - C"),
             "passed": None if d is None else d["diff"] >= -d["two_errors"]}
        if s in a_ and s in b_:
            r["marks"] = {"followed_N": [a_[s]["followed_N"]["pooled"]["share"], b_[s]["followed_N"]["pooled"]["share"]],
                          "false_tick": [a_[s]["false_N"]["tick"]["share"], b_[s]["false_N"]["tick"]["share"]]}
        res["sets"][s] = r
    st = [x["passed"] for x in res["sets"].values()]
    res["passed"] = False if False in st else (None if None in st else True)
    return res


def after_side(a):
    """`learn after --drill side --ckpt PATH --out DIR [--exam-ref DIR] [--t01-new DIR --t01-ref DIR] [--n N]`: the
    network N (tactic_v3) on exam_side's sets (taught, held, G) beside TEACH2, TEACH, C (reused from --exam-ref when
    that file is this simulator's and these battles, else played) and N1 (T0-1's network, --init); exam_side's splits,
    the comer battles apart; METHOD step 6's lines (side_lines)."""
    from sim.params import load
    from net.tactic import ckpt as TK
    from net.tactic import exam_side as ES
    P = load()
    D = Path(a.out)
    D.mkdir(parents=True, exist_ok=True)
    ar = SimpleNamespace(device=a.device, sim=a.sim)
    net, _, ck = TK.restore(a.ckpt, a.device)
    net.eval()
    n1, _, _ = TK.restore(a.init, a.device)
    n1.eval()
    sa = SimpleNamespace(n=a.n, chunk=a.chunk)
    shas = {"N": _sha(a.ckpt), "N1": _sha(a.init)}
    names = [x for x in a.sets.split(",") if x]
    players = [x for x in a.players.split(",") if x]
    print(json.dumps({"after_side": a.ckpt, "kind": ck.get("kind"), "round": ck.get("round"), "sim_key": EP.sim_key()}),
          flush=True)
    for nm, (sp, sd, ch, scene) in ES.sets_of(names, sa).items():
        (D / f"{nm}_meta.json").write_text(json.dumps(ES.meta(sp)), encoding="utf-8")
        for name in players:
            f = D / f"{nm}_{name}.json"
            sha = shas.get(name)
            if f.exists():
                # (an earlier result in this folder is taken only when it is this simulator's, these battles and - for
                # N / N1 - these weights; anything else is refused, never silently reused: REVIEW.md finding 7)
                x = json.loads(f.read_text(encoding="utf-8"))
                same = x.get("sim_key") == EP.sim_key() and x.get("meta") == EP.meta_of(sp) and (
                    sha is None or x.get("ckpt_sha256") == sha)
                if not same:
                    raise SystemExit(f"{f}: an earlier result of other weights, battles or simulator (ckpt_sha256 "
                                     f"{x.get('ckpt_sha256')} vs {sha}) - give a clean --out")
                print(json.dumps({"set": nm, "player": name, "kept": str(f), "ckpt_sha256": sha}), flush=True)
                continue
            if a.exam_ref and name in ES.PLAYERS:
                ref = Path(a.exam_ref) / f"{nm}_{name}.json"
                if ref.exists():
                    x = json.loads(ref.read_text(encoding="utf-8"))
                    if x.get("sim_key") == EP.sim_key() and x.get("meta") == EP.meta_of(sp):
                        f.write_text(json.dumps(x), encoding="utf-8")
                        print(json.dumps({"set": nm, "player": name, "reused": str(ref)}), flush=True)
                        continue
            t0 = time.time()
            if name in ("N", "N1"):
                res = net_side_run(net if name == "N" else n1, sp, sd, ch, ar, P, scene=scene)
                res["ckpt_sha256"] = sha
            else:
                res = ES.side_run(name, sp, sd, ch, ar, P, scene=scene)
            f.write_text(json.dumps(res), encoding="utf-8")
            print(json.dumps({"set": nm, "player": name, "battles": len(sp), "min": round((time.time() - t0) / 60, 1)}),
                  flush=True)
    out = side_table(D, names)
    t01 = t01_compare(a.t01_new, a.t01_ref) if a.t01_new and a.t01_ref else None
    out["lines"] = side_lines(out, t01)
    (D / "after_side.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for nm, tab in out.items():
        if nm == "lines":
            continue
        print(f"\n## set {nm} ({tab['battles']} battles) - (i) C's first contact + 300 s | (iii) end | orders")
        for pair, rows in tab["pairs"].items():
            if pair.startswith(("N", "TEACH2 - TEACH")):
                for lab, r in rows.items():
                    if lab in ("all", "cause", "no cause", "all [-comer]", "cause [-comer]", "no cause [-comer]", "comer"):
                        print(f"  {pair} {lab}: {EP.fmt(r['i_300'])} | {EP.fmt(r['iii_end'])} | {EP.fmt(r['orders'])}")
        print("  orders a unit-minute (all):", {p: v.get("all") for p, v in tab["orders_min"].items()})
    print("\nlines:", json.dumps(out["lines"], indent=1))
    ln = out["lines"]
    for p in ("i_closes", "ii_executes", "iii_no_harm", "iv_held", "v_t01", "vi_orders"):
        x = {k: v for k, v in ln[p].items() if k not in ("rule", "by_axis")}
        print(f"PIECE {p}: {ln[p]['status']} - {json.dumps(x)}", flush=True)
    print(f"VERDICT: {ln['verdict']}" + (f" - {'; '.join(ln['reasons'])}" if ln["reasons"] else ""), flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("train", "after"))
    ap.add_argument("--out", required=True, help="train: the checkpoints' folder; after: the exam's folder")
    ap.add_argument("--log", default="", help="train: the logs' folder (default --out)")
    ap.add_argument("--rounds", type=int, default=0, help="train: at most this many rounds (config train.learn.rounds)")
    ap.add_argument("--no-general", action="store_true", help="train: the scenes-only run (no general-battle batch)")
    ap.add_argument("--resume", default="", help="train: a round's checkpoint to continue from (net and optimiser)")
    ap.add_argument("--from-round", type=int, default=0, help="train with --resume: the first round to run (1-based)")
    ap.add_argument("--ckpt", default="", help="after: the tactic_v2 checkpoint")
    ap.add_argument("--vision", action="store_true", help="after: three records for game.vision")
    ap.add_argument("--g-ref", default="tmp/tactic_slice/v23/stage3b/G",
                    help="after: stage 3b's G_C.json / G_TEACH.json (C reused when this simulator's; TEACH compared)")
    ap.add_argument("--sim", default="", help="the simulator's commit (said by the caller)")
    ap.add_argument("--unit", default="", help="the unit script's commit (said by the caller)")
    ap.add_argument("--smoke", type=int, default=0, help="n scenes + n general battles a part, 200 s, tiny reads")
    ap.add_argument("--drill", choices=("attack", "side"), default="attack",
                    help="attack: T0-1 'бить связанного'; side: T0-2 'сторона удара' (tactic_v3 from --init, TEACH2)")
    ap.add_argument("--init", default="weights/tactic/attack1/final.pt",
                    help="side: the tactic_v2 checkpoint the training starts from; after: T0-1's network N1")
    ap.add_argument("--passes", type=int, default=0, help="train: passes a round (config train.learn.passes)")
    ap.add_argument("--no-v", action="store_true", help="train: no V read after the rounds (a plumbing run)")
    ap.add_argument("--sets", default="taught,held,G", help="after --drill side: exam_side's sets")
    ap.add_argument("--players", default=",".join(SIDE_PLAYERS), help="after --drill side: the players")
    ap.add_argument("--exam-ref", default="", help="after --drill side: exam_side's folder (C, TEACH, TEACH2 reused)")
    ap.add_argument("--n", type=int, default=0, help="after --drill side: a smaller set (exam_side --n)")
    ap.add_argument("--chunk", type=int, default=256, help="after --drill side: the scenes' chunk")
    ap.add_argument("--t01-new", default="", help="after --drill side: `learn after --ckpt` folder of this network")
    ap.add_argument("--t01-ref", default="tmp/drill1_w/before", help="after --drill side: the same of T0-1's network (tmp/drill1_w/before)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    if a.what == "after" and a.drill == "side":
        return after_side(a)
    return train(a) if a.what == "train" else after(a)


if __name__ == "__main__":
    main()
