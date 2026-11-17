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


def L():
    return tcfg()["train"]["learn"]


def _bands():
    return scfg()["drills"]["attack"]["bands"]


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

    def __init__(self, check=0, every=None):
        self.every = int(LN._c()["loss"]["trivial_every"] if every is None else every)
        self.pending, self.ticks, self.meta, self.cards, self.pend_card = [], [], [], [], {}
        self.n_triv = 0
        self.check, self.checked, self.check_max = int(check), 0, 0.0
        self.NK = None

    def hook(self, net_driven):
        return lambda v, lab, T: self(v, T, net_driven)

    def __call__(self, v, T, net_driven):
        head = T["head_free"]
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
              "dc": T["dc_keys"], "bounds": bounds, "head": head, "drop": T["dropped"] | T["dc_label"], "kinds": kinds,
              "trivial": T["trivial"], "near2": T["tags"]["near2"] & T["attack"]}
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
                                                           "head", "drop", "kinds", "trivial", "near2")
                                                      or k.startswith("a_")):
                t = LN._pad_g(t, -1 if k == "label" else 0)
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
                              "disturbed": i in disturbed})
            c = self.pend_card.get(j)
            self.cards.append(c)
        for t in self.pending:
            keep = t["b"] < len(ii)
            if not bool(keep.any()):
                continue
            t = {k: x[keep] for k, x in t.items()}
            t["b"] = t["b"].long() + base
            self.ticks.append(t)
        self.pending, self.pend_card = [], {}

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


def play(name, specs, scene, seeds, chunk, a, P, make_driver, hook_of, on_flush, needs_input=None, watch=None):
    """exam_pin.run_player under tags.Record: the driver make_driver(ii) (ii the chunk's spec indices, play_tactic's
    order by limit), the labeller TEACH built under its own name, hook_of(driver) the Record's hook; on_flush(ii)
    after each chunk; needs_input - set on the Record (the teacher-path read: the network asked on TEACH's states)."""
    order = sorted(range(len(specs)), key=lambda i: int(specs[i]["limit"]))
    chunks = [order[i:i + chunk] for i in range(0, len(order), chunk)]
    state = [0]
    drivers = []

    def make():
        ii = chunks[state[0]]
        state[0] += 1
        drv = make_driver(ii)
        drivers.append(drv)
        rec = TG.Record(drv, EP.build_under("TEACH"), hook=hook_of(drv))
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

    def hook(self, drv):
        return lambda v, lab, T: self(v, T, drv)

    def __call__(self, v, T, drv):
        fx = v.fx
        head = T["head_free"]
        now = int(round(float(v.now)))
        if self.path == "T":
            with torch.no_grad():
                k = self.net.act(v, greedy=True, given=T["k"])["k"]
        else:
            k = drv.last["k"]
            o = v.order
            att = k >= 0
            kc = k.clamp(min=0)
            pt = torch.where(att, fx.slot_of_row(k), o["tgt"])
            ps = torch.where(att, fx.g_side.gather(2, kc[..., None])[..., 0], o["side"])
            pk = torch.where(att, torch.full_like(o["skill"], SC.ATTACK), o["skill"])
            code = torch.where((pk == SC.ATTACK) & (pt >= 0) & fx.able, pt * 4 + ps.clamp(min=0), torch.full_like(pt, -1))
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
        self.pending.append({"b": v.idx[r].cpu(), "sec": torch.full((r.numel(),), now, dtype=torch.int32),
                             "att": (label >= 0)[r, g].cpu(), "drop": (T["dropped"] | T["dc_label"])[r, g].cpu(),
                             "kinds": kinds[r, g].cpu(), "near2": (T["tags"]["near2"] & T["attack"])[r, g].cpu(),
                             "code": code[r, g].to(torch.int8).cpu(), "unc": unc[r, g].cpu(),
                             "e": fx.slot_of_row(label)[r, g].cpu(),
                             "es": fx.g_side.gather(2, lc[..., None])[..., 0][r, g].cpu()})

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
        self.pending, self.hold_pending = [], []

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
    return out


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
    return out


def read(net, specs, scene, seeds, chunk, a, P, kind, mN=None, mT=None, watch=None, paths="NT"):
    """The two reads of specs (module doc) into Marks mN / mT (new ones when None): ({path: player file}, mN, mT)."""
    mN = mN or Marks(net, "N")
    mT = mT or Marks(net, "T")
    files = {}
    net.eval()
    if "N" in paths:
        files["N"], _ = play("N", specs, scene, seeds, chunk, a, P, lambda ii: PN.NetPlayer(net, greedy=True), mN.hook,
                             lambda ii: mN.flush(ii, specs, kind), needs_input=True, watch=watch)
    if "T" in paths:
        files["T"], _ = play("TEACH", specs, scene, seeds, chunk, a, P, lambda ii: EP.build_under("TEACH"), mT.hook,
                             lambda ii: mT.flush(ii, specs, kind), needs_input=True, watch=watch)
    return files, mN, mT


# ------------------------------------------------------------------ training
def _limit(specs, s):
    for x in specs:
        x["limit"] = min(int(x["limit"]), int(s))
        if x.get("end_s") is not None:
            x["end_s"] = min(float(x["end_s"]), float(s))
    return specs


def v_specs(smoke=0):
    """(scenes, general battles) of the validation set V (config drills.attack.sets.V; its general battles on G's
    trained sizes, Random(train.learn.seeds.V_general))."""
    from net.tactic import scenes as SN
    c = scfg()["drills"]["attack"]["sets"]
    sc = SN.set_specs("V")
    G = c["G"]
    gen = SP.drill_generals(int(c["V"]["general"]), SP.seeded(int(L()["seeds"]["V_general"])), sizes=G["sizes"],
                            limit=int(G["limit"]))
    if smoke:
        sc, gen = _limit(sc[::max(1, len(sc) // smoke)][:smoke], 200), _limit(gen[:smoke], 200)
    return sc, gen


def v_read(net, a, P, chunk, smoke=0):
    sc, gen = v_specs(smoke)
    sd = L()["seeds"]
    t0 = int(sd["V_tseed"])
    mN, mT = Marks(net, "N"), Marks(net, "T")
    ch = min(chunk, len(sc))
    read(net, sc, True, LN._seeds(t0, len(sc), ch), ch, a, P, "scene", mN, mT)
    ch = min(chunk, len(gen))
    read(net, gen, False, LN._seeds(t0 + 50, len(gen), ch), ch, a, P, "general", mN, mT)
    return marks(mN, mT, "V")


def fit(net, opt, d, w, dev, K, log, gen, passes, mb):
    """Section 7's update: passes over the buffer's group-ticks with weight in minibatches of mb; the input rebuilt
    (rebuild), the pointer on the stored walk (fed with the labeller's choices). Returns the mean loss a pass."""
    use = (w.sum(1) > 0).nonzero()[:, 0]
    net.train()
    hist = []
    c = L()
    for p in range(passes):
        perm = use[torch.randperm(use.numel(), generator=gen)]
        tot, nb, t0 = 0.0, 0, time.time()
        for i in range(0, perm.numel(), mb):
            idx = perm[i:i + mb]
            t = {k: d[k][idx].to(dev) for k in NET_KEYS}
            card = d["cards"][d["b"][idx]].to(dev)
            with torch.no_grad():
                inp = rebuild(t, card, K)
            enc = net.enc(inp, t["command"], t["cols"], t["own"], t["g_row"].long(), t["g_on"])
            lg = net.ptr.logits_all(enc["cards"], t["own"], t["g_row"].long(), t["pair"], t["open"])
            loss, _ = HD.loss(lg, t["label"].long(), w[idx].to(dev), t["dc"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([q for q in net.parameters() if q.requires_grad], float(c["max_grad"]))
            opt.step()
            tot += float(loss.detach())
            nb += 1
        hist.append(round(tot / max(nb, 1), 5))
        log({"pass": p, "loss": hist[-1], "minibatches": nb, "min": round((time.time() - t0) / 60, 1)})
    net.eval()
    return hist


def collect(r, net, a, P, chunk, no_general, smoke, log):
    """Round r's buffer (section 7 'Round', 'Who drives')."""
    c = L()
    sd = c["seeds"]
    n_sc = float(scfg()["drills"]["attack"]["sets"]["round"]["scenes"])
    n_g = float(scfg()["drills"]["attack"]["sets"]["round"]["general"])
    scale, gscale = (1.0, 1.0) if not smoke else (smoke / n_sc, smoke / n_g)
    scenes, gen = LN.round_specs(r, scale, gscale, seeds=sd)
    if smoke:
        scenes, gen = _limit(scenes[:smoke], 200), _limit(gen[:smoke], 200)
    buf = NetBuffer(check=64 if smoke else 32)
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
            _, drv = play("TEACH", sp_t, scene, LN._seeds(int(sd["tseed"]) + 100 * r + off, len(sp_t), ch), ch, a, P,
                          lambda ii: Disturb(EP.build_under("TEACH"),
                                             {j: copy.deepcopy(plans[i]) for j, i in enumerate(ii) if i in plans},
                                             dseed + ii[0]),
                          lambda d_: buf.hook(False), lambda ii: buf.flush(ii, sp_t, kind, False, dis))
            for d_ in drv:
                for k in ("send", "hold", "held_ticks"):
                    stats[k] += d_.n[k]
        if sp_n:
            net.eval()
            ch = min(chunk, len(sp_n))
            play("N", sp_n, scene, LN._seeds(int(sd["tseed"]) + 100 * r + off + 25, len(sp_n), ch), ch, a, P,
                 lambda ii: PN.NetPlayer(net, greedy=True), lambda d_: buf.hook(True),
                 lambda ii: buf.flush(ii, sp_n, kind, True))
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
    net = HD.TacticNet2().to(dev)
    opt = torch.optim.Adam([q for q in net.parameters() if q.requires_grad], lr=float(c["lr"]))
    gen = torch.Generator().manual_seed(int(c["seeds"]["tseed"]))
    rounds = min(int(a.rounds or c["rounds"]), int(c["rounds"]))
    chunk = int(c["chunk"])
    log({"start": time.strftime("%H:%M"), "rounds_max": rounds, "no_general": a.no_general, "device": dev, "sim": a.sim,
         "sim_key": EP.sim_key(), "reach_m": OR.c_orders()["reach_m"], "taught": TG.TAUGHT, "cols": HD.d2_cols()[0],
         "smoke": a.smoke, "learn": {k: v for k, v in c.items() if not k.startswith("_")}})
    table, passed, shares, final = [], [], [], None
    r0 = 0
    if a.resume:
        # the same training continued in a new process (the earlier one cut): the network and the optimiser from the
        # checkpoint of round from_round - 1, the rounds before it read back from this log folder (their ROUND rows and
        # V reads); the rounds from from_round on draw the seeds they would have had (round_specs / collect by r). The
        # minibatch order's generator cannot be replayed: it starts anew at tseed + 1000 x from_round (said in the log)
        ckr = torch.load(a.resume, map_location=dev, weights_only=False)
        net.load_state_dict(ckr["net"])
        opt.load_state_dict(ckr["opt"])
        r0 = int(a.from_round) - 1
        assert int(ckr.get("round", r0)) == r0, (ckr.get("round"), r0)
        gen = torch.Generator().manual_seed(int(c["seeds"]["tseed"]) + 1000 * (r0 + 1))
        for line in (logd / "train.log").read_text(encoding="utf-8").splitlines():
            if line.startswith('{"ROUND"'):
                row = json.loads(line)["ROUND"]
                if row["round"] <= r0:
                    table = [x for x in table if x["round"] != row["round"]] + [row]
        table.sort(key=lambda x: x["round"])
        assert [x["round"] for x in table] == list(range(1, r0 + 1)), [x["round"] for x in table]
        for x in table:
            passed.append(bool(x["M3_V"]))
            if x["label_share_net_states"] is not None:
                shares.append(x["label_share_net_states"])
        log({"RESUME": str(a.resume), "from_round": r0 + 1, "rounds_read": len(table), "passed_v": passed,
             "minibatch_generator_seed": int(c["seeds"]["tseed"]) + 1000 * (r0 + 1)})
    t_all = time.time()
    for r in range(r0, rounds):
        t0 = time.time()
        buf, st = collect(r, net, ar, P, chunk, a.no_general, a.smoke, log)
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
        hist = fit(net, opt, d, w, dev, buf.NK[1], log, gen, int(c["passes"]), int(c["minibatch"]))
        del d, w, buf
        ck = out / f"round_{r + 1:02d}.pt"
        tv = v_read(net, ar, P, chunk, a.smoke)
        ok, fails = m3(tv)
        passed.append(ok)
        TK_save(ck, net, opt, a, round=r + 1, marks_v=brief(tv), passed_v=ok)
        row = {"round": r + 1, "drove": "teacher" if (r == 0 and not a.smoke) else "teacher + network (half)",
               "battles": st["battles_teacher"] + st["battles_net"], "labels": int(att.sum()),
               "label_share_net_states": share_net, "loss": hist, "V": brief(tv), "M3_V": ok, "fails": fails,
               "ckpt": str(ck), "min": round((time.time() - t0) / 60, 1)}
        table.append(row)
        log({"ROUND": row})
        (logd / f"v_round_{r + 1:02d}.json").write_text(json.dumps(tv, indent=1), encoding="utf-8")
        if share_net is not None:
            shares.append(share_net)
            if len(shares) >= 3 and shares[-3] < shares[-2] < shares[-1]:
                log("!!!!!! LOUD: the share of attack labels on the network's own states grew three rounds running "
                    f"{[round(x, 5) for x in shares[-3:]]} - look at two battles (design section 7); going on")
        if len(passed) >= 2 and passed[-1] and passed[-2]:
            final = ck
            log({"STOP": f"M3 held on V at rounds {r} and {r + 1}"})
            break
    if final is None:
        final = out / f"round_{len(table):02d}.pt"
        log({"NO_STOP": f"{len(table)} rounds without M3 two rounds running", "fails_last": table[-1]["fails"]})
    shutil.copyfile(final, out / "final.pt")
    res = {"rounds": table, "final": str(final), "final_copy": str(out / "final.pt"), "sha256": _sha(out / "final.pt"),
           "stopped": len(passed) >= 2 and passed[-1] and passed[-2], "minutes": round((time.time() - t_all) / 60, 1)}
    (logd / "train.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log({"done": res["final"], "sha256": res["sha256"], "stopped": res["stopped"], "minutes": res["minutes"]})
    logf.close()
    return res


def TK_save(path, net, opt, a, **more):
    from net.tactic import ckpt as TK
    TK.save_v2(path, net, opt, ["T0-2"], unit_commit=a.unit, sim_commit=a.sim, **more)


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
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    return train(a) if a.what == "train" else after(a)


if __name__ == "__main__":
    main()
