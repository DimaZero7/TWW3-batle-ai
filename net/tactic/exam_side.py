"""Drill T0-2 'side of the strike' - the exam before training (METHOD.md recipe step 4; tmp/drill2/design_merged.md,
the lines in its run's report): three players on the same battles, paired by battle -
    TEACH2  the drill's teacher (pin.SideTeacher: T0-1's TEACH + the side of each new attack) with the unit script's
            walk round the flank (unit/scripts.Unit, orders.side)
    TEACH   T0-1's teacher - this drill's dumb move: the straight attack
    C       the level's default
    TEACH2_ON  TEACH2 with the unit script's cancel on a comer on (config side.cancel_comer is off: the game's probe) -
            the sensitivity row (not in the default run: the cancel is decided off)
the battles where an enemy comes at the hammer after the order (comer_scenes) apart: every group also '[-comer]', the
lines b and e on those (tmp/drill2/design_merged.md THE GAME NIGHT: the simulator is grossly wrong there);
on the sets (scenes.side_set; config drills.side.sets): taught (S2 cause scenes + their S2N twins, no axis held), held
(each held axis at its held-out value, cause + twin), G (T0-1's set G, exam_pin.g_set). The judge (as T0-1): hp_trade
paired by battle at C's first contact + 300 s (exam_pin.reads (i)) and at the end ((iii)); splits by cause / no cause,
the target's width (narrow < WIDE_M), blocker, the hammer's start, line, heading, flip, held axis. The execution of the
unit script (SideWatch, every flank order a player gave): reached P (its walk ended 'arrived'), the walk's end by phase
(scripts.PHASES: arrived, timeout, a fallback's fact), the first contact with its target on the target's flank (sim
hit_sec 2) / front / rear / none; orders to the game a unit-minute (watch's game_orders_min). Line c: first contact on
the flank over ALL flank orders that did not fall back, line d: the orders a unit-minute - both without the comer
battles; line g: each held axis binds in all its scenes (scenes.side_binds).
    python -m net.tactic.exam_side run --sets taught,held,G --players C,TEACH,TEACH2 --out tmp/t02/exam [--n 64]
    python -m net.tactic.exam_side table --out tmp/t02/exam
    --g-ref DIR: G's C and TEACH taken from DIR/G_C.json, G_TEACH.json (exam_pin / learn after files of the same
    battles and sim_key) instead of played
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from net.stage0 import battle as BA
from net.tactic import exam_pin as EP
from net.unit import scripts as SC

WIDE_M = 37.0
PLAYERS = ("C", "TEACH", "TEACH2", "TEACH2_ON")
DEFAULT = ("C", "TEACH", "TEACH2")      # (TEACH2_ON off the default run: the cancel on a comer is decided - off)
PAIRS = (("TEACH2", "TEACH"), ("TEACH2", "C"), ("TEACH", "C"), ("TEACH2_ON", "TEACH"), ("TEACH2_ON", "TEACH2"))
BASE = {"TEACH2_ON": "TEACH2"}          # TEACH2_ON: TEACH2 with the unit script's cancel on a comer on (scripts.CANCEL_COMER)


class SideWatch(EP.HoleWatch):
    """HoleWatch + every attack order my units hold (its target, its second, its approach): strikes [(b, my slot,
    target, second, approach, end phase, end second, first contact side, contact second, comer seen on the walk,
    the walk's length at the order m, its time guard s)] - the end phase (flank
    orders only): the first phase of the unit script other than walk after the order (scripts.PHASES; 'replaced' - the
    order changed before; None - still walking at the play's end or a front order); the first contact side - the side
    of the TARGET the striker's contact began on (sim hit_sec [B, striker, struck] - sim/battle.py: held[b, i, j], j's
    side seen from j; 1 front, 2 flank, 3 rear), 0 when the target touched the striker first without the striker's
    own contact on it that second, None never. (exam_pin.SceneWatch's contacts read hit_sec[b, target, striker]: the
    striker's own side.)"""

    def _init(self, st):
        super()._init(st)
        B, N, dev = self.B, self.N, self.dev
        self.s_t = torch.full((B, N), -1, dtype=torch.long, device=dev)
        self.s_born = torch.full((B, N), -1.0, device=dev)
        self.s_idx = torch.full((B, N), -1, dtype=torch.long, device=dev)       # the open record (contact not yet)
        self.w_idx = torch.full((B, N), -1, dtype=torch.long, device=dev)       # the open walk (flank orders)
        self.strikes = []

    def __call__(self, sec, st):
        super().__call__(sec, st)
        tac = getattr(self.chain, "tac", None)
        if tac is None:
            return
        o = tac.orders_of()
        live = ~st.done[:, None]
        att = o["on"] & (o["skill"] == SC.ATTACK) & (o["tgt"] >= 0) & self.mine & live
        same = att & (o["tgt"] == self.s_t) & (o["born"] == self.s_born)
        new = att & ~same
        gone = (self.w_idx >= 0) & live & ~same
        for b, i in gone.nonzero().tolist():
            r = self.strikes[int(self.w_idx[b, i])]
            if r[5] is None:
                r[5], r[6] = "replaced", int(sec)
        self.w_idx = torch.where(gone, torch.full_like(self.w_idx, -1), self.w_idx)
        self.s_idx = torch.where(live & ~same, torch.full_like(self.s_idx, -1), self.s_idx)
        for b, i in new.nonzero().tolist():
            k = len(self.strikes)
            ap = int(o["approach"][b, i])
            self.strikes.append([b, i, int(o["tgt"][b, i]), int(sec), ap, None, None, None, None, False, None, None])
            self.s_idx[b, i] = k
            if ap > 0:
                self.w_idx[b, i] = k
        self.s_t = torch.where(att, o["tgt"], torch.full_like(self.s_t, -1))
        self.s_born = torch.where(att, o["born"], torch.full_like(self.s_born, -1.0))
        # the first contact of each open record with its target
        hs = st.u["hit_sec"].view(self.B, self.N, self.N)
        tc = self.s_t.clamp(min=0)
        on_t = hs.gather(2, tc[..., None])[..., 0]                                     # [b, i, t_i]: t's side
        by_t = hs.gather(1, tc[:, None, :])[:, 0, :]                                   # [b, t_i, i]
        hit = (self.s_idx >= 0) & ((on_t > 0) | (by_t > 0))
        for b, i in hit.nonzero().tolist():
            r = self.strikes[int(self.s_idx[b, i])]
            r[7], r[8] = int(on_t[b, i]), int(sec)
            self.s_idx[b, i] = -1
        tr = getattr(self.chain.units.net, "trace", None)
        if tr is None:
            return
        un = self.chain.units
        wd, ws = getattr(un, "walk_d", None), getattr(un, "walk_s", None)
        if wd is not None:
            for b, i in (self.w_idx >= 0).nonzero().tolist():
                r = self.strikes[int(self.w_idx[b, i])]
                r[10], r[11] = round(float(wd[b, i]), 1), round(float(ws[b, i]), 1)
        cm = getattr(self.chain.units.net, "comer", None)
        if cm is not None:
            for b, i in ((self.w_idx >= 0) & cm).nonzero().tolist():
                self.strikes[int(self.w_idx[b, i])][9] = True
        done = (self.w_idx >= 0) & (tr != SC.PHASES.index("walk")) & (tr != 0)
        for b, i in done.nonzero().tolist():
            r = self.strikes[int(self.w_idx[b, i])]
            if r[5] is None:
                r[5], r[6] = SC.PHASES[int(tr[b, i])], int(sec)
            self.w_idx[b, i] = -1

def _make(name):
    def make():
        return EP.build_under(BASE.get(name, name))
    return make


def side_run(name, specs, seeds, chunk, a, P, scene=True):
    """The player's file (exam_pin.run_player through SideWatch) + contacts and flanks by spec index."""
    ex = {"strikes": []}

    def look(ii, watch, side):
        ex["strikes"] += [[ii[r[0]]] + r[1:] for r in watch.strikes if r[0] < len(ii)]
    kw = {"battles": BA.scene_battles, "enemy": BA.SceneEnemy} if scene else {}
    SC.CANCEL_COMER = True if name == "TEACH2_ON" else None
    try:
        res = EP.run_player(name, specs, seeds, chunk, a, P, make=_make(name), on_chunk=look, watch_cls=SideWatch,
                            full=True, by_limit=True, **kw)
    finally:
        SC.CANCEL_COMER = None
    res.update(strikes=ex["strikes"])
    return res


def meta(specs):
    from net.tactic import scenes as SN
    out = []
    for s in specs:
        ax = s.get("axes", {})
        out.append({"recipe": s.get("recipe", "G"), "cause": s.get("s2_cause"), "held": s.get("held"),
                    "width": ax.get("width_m"), "blocker": ax.get("blocker"), "start": ax.get("start"),
                    "line": ax.get("line"), "heading": ax.get("heading_deg"), "flip": (s.get("s2") or {}).get("flip"),
                    "worn_anvil": ax.get("worn_anvil"),
                    "binds": SN.side_binds(s) if s.get("family") == "S2" else None})
    return out


def sets_of(names, a):
    from net.stage0 import cfg as scfg
    from net.tactic import scenes as SN
    c = scfg()["drills"]["side"]["sets"]
    out = {}
    for nm in names:
        if nm == "G":
            sp, sd, ch = EP.g_set()
            if a.n:
                sp = sp[:a.n]
                sd = sd[:(len(sp) + ch - 1) // ch]
            out[nm] = (sp, sd, ch, False)
            continue
        sp = SN.side_set(nm, n=a.n or None)
        ch = a.chunk
        t0 = int(c["tseed"]) + (100 if nm == "held" else 0)
        out[nm] = (sp, [t0 + k for k in range((len(sp) + ch - 1) // ch)], ch, True)
    return out


def run(a):
    from sim.params import load
    P = load()
    D = Path(a.out)
    D.mkdir(parents=True, exist_ok=True)
    ar = SimpleNamespace(device=a.device, sim=a.sim)
    for nm, (sp, sd, ch, scene) in sets_of(a.sets.split(","), a).items():
        (D / f"{nm}_meta.json").write_text(json.dumps(meta(sp)), encoding="utf-8")
        for name in a.players.split(","):
            f = D / f"{nm}_{name}.json"
            if f.exists():
                print(json.dumps({"set": nm, "player": name, "kept": str(f)}), flush=True)
                continue
            if nm == "G" and a.g_ref and name in ("C", "TEACH") and (Path(a.g_ref) / f"G_{name}.json").exists():
                ref = Path(a.g_ref) / f"G_{name}.json"
                x = json.loads(ref.read_text(encoding="utf-8"))
                if x.get("sim_key") == EP.sim_key() and x.get("meta") == EP.meta_of(sp):
                    f.write_text(json.dumps(x), encoding="utf-8")
                    print(json.dumps({"set": nm, "player": name, "reused": str(ref)}), flush=True)
                    continue
            t0 = time.time()
            res = side_run(name, sp, sd, ch, ar, P, scene=scene)
            f.write_text(json.dumps(res), encoding="utf-8")
            print(json.dumps({"set": nm, "player": name, "battles": len(sp), "min": round((time.time() - t0) / 60, 1)}),
                  flush=True)
    return table(D, a.sets.split(","))


# ------------------------------------------------------------------ the tables
SIDE_NAMES = {0: "struck_first", 1: "front", 2: "flank", 3: "rear"}
KEPT = ("arrived", "timeout", "attacking", None)


def _shares(rows, n=None):
    n = len(rows) if n is None else n
    c = {}
    for r in rows:
        k = "none" if r[7] is None else SIDE_NAMES.get(int(r[7]), "none")
        c[k] = c.get(k, 0) + 1
    return {k: round(v / n, 4) for k, v in sorted(c.items())} if n else {}


def _flank_exec(d, sel=None):
    """The execution of a player's attack orders (module doc) over the battles sel (None: all): flank orders - reached
    P, the walk's end by phase, fell back by fact, the first contact's side of the target (all and of those that did
    not fall back: KEPT ends); straight orders - their first contact's side (the dumb move's own read)."""
    S = None if sel is None else set(sel)
    rows = [r for r in d.get("strikes", []) if S is None or r[0] in S]
    fl = [r for r in rows if r[4] > 0]
    st = [r for r in rows if r[4] == 0]
    n = len(fl)
    ends = {}
    for r in fl:
        ends[str(r[5])] = ends.get(str(r[5]), 0) + 1
    q = lambda x, m: round(x / m, 4) if m else None                                 # noqa: E731
    fb = {k: v for k, v in ends.items() if k.startswith("fb_")}
    kept = [r for r in fl if r[5] in KEPT]
    kc = [r for r in kept if r[7] is not None and int(r[7]) > 0]
    def q50(v, p):
        v = sorted(v)
        return round(v[min(len(v) - 1, int(p * len(v)))], 1) if v else None
    grp = {"arrived": [r for r in fl if r[5] == "arrived"], "timed_out": [r for r in fl if r[5] == "timeout"],
           "fell_back": [r for r in fl if str(r[5]).startswith("fb_")],
           "other": [r for r in fl if r[5] not in ("arrived", "timeout") and not str(r[5]).startswith("fb_")]}
    walks = {}
    for k, v in list(grp.items()) + [("all", fl)]:
        ln = [r[10] for r in v if len(r) > 10 and r[10] is not None]
        du = [r[6] - r[3] for r in v if r[6] is not None]
        walks[k] = {"n": len(v), "share": q(len(v), n), "len_m_p50": q50(ln, 0.5), "len_m_p90": q50(ln, 0.9),
                    "dur_s_p50": q50(du, 0.5), "dur_s_p90": q50(du, 0.9)}
    by_len = {}
    for lo, hi in ((0, 40), (40, 60), (60, 80), (80, 100), (100, 130), (130, 1e9)):
        v = [r for r in fl if r[5] in KEPT and r[7] is not None and int(r[7]) > 0 and len(r) > 10 and r[10] is not None
             and lo <= r[10] < hi]
        by_len[f"{lo}-{hi if hi < 1e9 else 'inf'}"] = {"kept_contact": len(v), "flank": q(sum(int(r[7]) == 2 for r in v), len(v)),
                                                     "timed_out": q(sum(r[5] == "timeout" for r in v), len(v))}
    return {"walks": walks, "c_by_walk_len": by_len, "attack_orders": len(rows), "flank_orders": n, "flank_share": q(n, len(rows)),
            "reached_P": q(ends.get("arrived", 0), n), "ends": ends, "fell_back": q(sum(fb.values()), n),
            "fell_back_by": {k: q(v, n) for k, v in fb.items()}, "first_contact": _shares(fl),
            "not_fallen_back": len(kept), "kept_contact": len(kc), "kept_flank": sum(1 for r in kc if int(r[7]) == 2),
            "first_flank": sum(1 for r in fl if r[7] is not None and int(r[7]) == 2),
            "replaced": sum(1 for r in fl if r[5] == "replaced"),
            "first_contact_of_not_fallen": _shares(kept), "straight_orders": len(st), "straight_first_contact": _shares(st)}

def _groups(mt, nm):
    B = len(mt)
    if nm == "G":
        return [("all", list(range(B)))]
    g = [("all", list(range(B))), ("cause", [b for b in range(B) if mt[b]["cause"]]),
         ("no cause", [b for b in range(B) if not mt[b]["cause"]])]
    for lab, f in (("narrow", lambda x: x["width"] < WIDE_M), ("wide", lambda x: x["width"] >= WIDE_M)):
        g.append((f"cause, {lab}", [b for b in range(B) if mt[b]["cause"] and f(mt[b])]))
        g.append((f"no cause, {lab}", [b for b in range(B) if not mt[b]["cause"] and f(mt[b])]))
    for key in ("blocker", "start", "line", "heading", "worn_anvil"):
        for v in sorted({str(x[key]) for x in mt}):
            g.append((f"cause, {key} {v}", [b for b in range(B) if mt[b]["cause"] and str(mt[b][key]) == v]))
    for v in sorted({str(x["flip"]) for x in mt if not x["cause"]}):
        g.append((f"no cause, flip {v}", [b for b in range(B) if not mt[b]["cause"] and str(mt[b]["flip"]) == v]))
    if nm == "held":
        for v in sorted({str(x["held"]) for x in mt}):
            for cz in (True, False):
                g.append((f"held {v}, {'cause' if cz else 'no cause'}",
                          [b for b in range(B) if str(mt[b]["held"]) == v and bool(mt[b]["cause"]) == cz]))
    return [(k, v) for k, v in g if v]


def comer_scenes(mt, pl):
    """The battles where an enemy comes at the hammer AFTER the order (tmp/drill2/design_merged.md, THE GAME NIGHT: the
    simulator is grossly wrong there - reported apart, not in lines b and e): the generator's (blocker 'comer': an enemy
    set off at the hammer after the order's time) and the seen (a flank walk of TEACH2 or TEACH2_ON with another enemy
    within cancel_clear_m of its way left - scripts.Unit.comer)."""
    gen = {b for b, x in enumerate(mt) if x.get("blocker") == "comer"}
    seen = {int(r[0]) for p in ("TEACH2", "TEACH2_ON") if p in pl
            for r in pl[p].get("strikes", []) if r[4] > 0 and len(r) > 9 and r[9]}
    return {"generator": sorted(gen), "seen": sorted(seen), "all": sorted(gen | seen)}


def groups_of(mt, nm, comer):
    """_groups, each also without the comer battles ('<label> [-comer]'), and the comer battles alone."""
    g = _groups(mt, nm)
    out = list(g) + [(f"{lab} [-comer]", [b for b in sel if b not in comer]) for lab, sel in g]
    cz = sorted(comer)
    out += [("comer", cz)]
    if nm != "G":
        out += [("comer, cause", [b for b in cz if mt[b]["cause"]]), ("comer, no cause", [b for b in cz if not mt[b]["cause"]])]
    return [(k, v) for k, v in out if v]


def table(D, names):
    D = Path(D)
    out = {}
    for nm in names:
        files = {p: D / f"{nm}_{p}.json" for p in PLAYERS if (D / f"{nm}_{p}.json").exists()}
        if "C" not in files:
            continue
        pl = {p: json.loads(f.read_text(encoding="utf-8")) for p, f in files.items()}
        mf = D / f"{nm}_meta.json"
        mt = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else [{"cause": None}] * len(pl["C"]["hp_series"])
        anchor = pl["C"]["per_battle"]["t_contact"]
        R = {p: EP.reads(d, anchor) for p, d in pl.items()}
        cm = comer_scenes(mt, pl)
        tab = {"battles": len(mt), "cut_i": {p: R[p]["cut_i"] for p in R}, "comer_scenes": cm, "pairs": {},
               "orders_min": {}, "attack_orders_battle": {}, "exec": {}, "binds": binds_of(mt)}
        for lab, sel in groups_of(mt, nm, set(cm["all"])):
            for x, y in PAIRS:
                if x in pl and y in pl:
                    tab["pairs"].setdefault(f"{x} - {y}", {})[lab] = {
                        "i_300": EP.paired(R[x]["i"], R[y]["i"], sel), "iii_end": EP.paired(R[x]["iii"], R[y]["iii"], sel),
                        "orders": EP.paired(pl[x]["per_battle"]["game_orders_min"],
                                            pl[y]["per_battle"]["game_orders_min"], sel)}
            for p, d in pl.items():
                v = [d["per_battle"]["game_orders_min"][b] for b in sel]
                tab["orders_min"].setdefault(p, {})[lab] = EP._mean(v)
                ao = d["per_battle"].get("attack_orders")
                if ao:
                    tab["attack_orders_battle"].setdefault(p, {})[lab] = EP._mean([ao[b] for b in sel])
            for p, d in pl.items():
                if "strikes" in d:
                    tab["exec"].setdefault(p, {})[lab] = _flank_exec(d, sel)
        out[nm] = tab
    out["lines"] = lines(out)
    (D / "exam_side.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print_table(out)
    return out


def binds_of(mt):
    """{held axis: {scenes, binds}} from a set's meta (scenes.side_binds; the scenes with a held axis only)."""
    out = {}
    for x in mt:
        if x.get("held") is None or x.get("binds") is None:
            continue
        r = out.setdefault(str(x["held"]), {"scenes": 0, "binds": 0})
        r["scenes"] += 1
        r["binds"] += int(bool(x["binds"]))
    return out


def _get(out, nm, lab, read="i_300", pair="TEACH2 - TEACH"):
    return out.get(nm, {}).get("pairs", {}).get(pair, {}).get(lab, {}).get(read)


def _pool(parts):
    """A pooled paired read of several sets' groups: the battles' differences together (from each part's mean, error
    and n: the pooled mean and its standard error)."""
    parts = [p for p in parts if p]
    if not parts:
        return None
    n = sum(p["n"] for p in parts)
    m = sum(p["diff"] * p["n"] for p in parts) / n
    # (each part's variance of the differences from its two errors: sd^2 = (e / 2)^2 n; pooled about m)
    ss = sum(((p["two_errors"] / 2) ** 2 * p["n"]) * (p["n"] - 1) + p["n"] * (p["diff"] - m) ** 2 for p in parts)
    sd = math.sqrt(ss / max(n - 1, 1))
    return {"diff": round(m, 4), "two_errors": round(2 * sd / math.sqrt(n), 4), "n": n}


def line_c(out, player="TEACH2"):
    """Line c of a player (drill T0-2; imitate's verdict reads the network's through it): first contact of the hammer on
    the target's flank / ALL its flank orders that did not fall back (no contact and struck first count as not flank),
    the comer battles out (exec groups 'all [-comer]' of the sets taught, held, G); >= 0.75. Also the old read (with a
    contact only), the read over all flank orders with the fallbacks, and the count with the comer battles."""
    X = " [-comer]"

    def ex_of(lab):
        e = [out[nm]["exec"].get(player, {}).get(lab) for nm in ("taught", "held", "G") if nm in out]
        return [x for x in e if x]
    q = lambda a, b: round(a / b, 4) if b else None                                 # noqa: E731
    ex = ex_of("all" + X)
    fl = sum(e["kept_flank"] for e in ex)
    nk = sum(e["not_fallen_back"] for e in ex)
    kc = sum(e["kept_contact"] for e in ex)
    nf = sum(e["flank_orders"] for e in ex)
    fa = sum(e.get("first_flank", 0) for e in ex)
    rp = sum(e.get("replaced", 0) for e in ex)
    return {"rule": "first contact of the hammer on the target's flank / ALL flank orders that did not fall back "
                    "(no contact and struck first counted as not flank), the comer battles out; >= 0.75",
            "player": player, "flank": fl, "n": nk, "share": q(fl, nk), "passed": bool(nk and fl / nk >= 0.75),
            "old_read_with_contact": {"flank": fl, "n": kc, "share": q(fl, kc)},
            "all_flank_orders_with_fallbacks": {"flank": fa, "n": nf, "share": q(fa, nf)},
            "with_comer": {"flank": sum(e["kept_flank"] for e in ex_of("all")),
                           "n": sum(e["not_fallen_back"] for e in ex_of("all"))},
            # (REVIEW2 finding 1: the walks that ended 'replaced' before a recorded fallback put in the denominator -
            # printed beside the binding read, no gate; the upper bound credits every first flank contact)
            "with_replaced": {"flank": fl, "n": nk + rp, "share": q(fl, nk + rp), "replaced": rp},
            "with_replaced_upper": {"flank": fa, "n": nk + rp, "share": q(fa, nk + rp)}}


def lines(out):
    """The lines written in advance (the run's brief, a-e; a is tmp/t02/regress.py's): passed or not, with the numbers."""
    res = {}
    X = " [-comer]"
    cause = _pool([_get(out, "taught", "cause" + X), _get(out, "held", "cause" + X)])
    nocause = _pool([_get(out, "taught", "no cause" + X), _get(out, "held", "no cause" + X)])
    g = _get(out, "G", "all" + X)
    res["b"] = {"rule": "TEACH2 - TEACH > 2 errors on S2 cause scenes (pooled); >= -2 errors on no-cause scenes and on G;"
                        " the comer battles out (comer_scenes)",
                "cause": cause, "no_cause": nocause, "G": g,
                "passed": bool(cause and cause["diff"] > cause["two_errors"]
                               and all(x is None or x["diff"] >= -x["two_errors"] for x in (nocause, g))),
                "with_comer": {"cause": _pool([_get(out, "taught", "cause"), _get(out, "held", "cause")]),
                               "no_cause": _pool([_get(out, "taught", "no cause"), _get(out, "held", "no cause")]),
                               "G": _get(out, "G", "all")},
                "comer_alone": {nm: _get(out, nm, "comer") for nm in ("taught", "held", "G")},
                "TEACH2_ON - TEACH": {"cause": _pool([_get(out, "taught", "cause", pair="TEACH2_ON - TEACH"),
                                                      _get(out, "held", "cause", pair="TEACH2_ON - TEACH")]),
                                      "comer": {nm: _get(out, nm, "comer", pair="TEACH2_ON - TEACH")
                                                for nm in ("taught", "held", "G")},
                                      "G": _get(out, "G", "all", pair="TEACH2_ON - TEACH")}}
    res["c"] = line_c(out)
    om = {nm: out[nm]["orders_min"].get("TEACH2", {}).get("all" + X) for nm in ("taught", "held", "G") if nm in out}
    res["d"] = {"rule": "TEACH2's orders to the game a unit-minute <= 1.2, the comer battles out", "by_set": om,
                "with_comer": {nm: out[nm]["orders_min"].get("TEACH2", {}).get("all") for nm in ("taught", "held", "G")
                               if nm in out},
                "passed": bool(om) and all(v is not None and v <= 1.2 for v in om.values())}
    bd = out.get("held", {}).get("binds") or {}
    res["g"] = {"rule": "every held axis binds in 100 % of its scenes", "by_axis": bd,
                "passed": bool(bd) and all(v["binds"] == v["scenes"] for v in bd.values())}
    t, h = _get(out, "taught", "cause" + X), _get(out, "held", "cause" + X)
    if t and h:
        gap = round(h["diff"] - t["diff"], 4)
        e2 = round(math.sqrt(t["two_errors"] ** 2 + h["two_errors"] ** 2), 4)
        res["e"] = {"rule": "held cause - taught cause (TEACH2 - TEACH) not below -2 errors; the comer battles out",
                    "taught": t, "held": h,
                    "gap": gap, "two_errors": e2, "passed": gap >= -e2,
                    "by_axis": {lab: _get(out, "held", lab) for lab in out["held"]["pairs"].get("TEACH2 - TEACH", {})
                                if lab.startswith("held ") and lab.endswith(", cause" + X)}}
    return res


def print_table(out):
    f = EP.fmt
    for nm, tab in out.items():
        if nm == "lines":
            continue
        print(f"\n## set {nm} ({tab['battles']} battles) - (i) C's first contact + 300 s | (iii) end | orders")
        for pair, rows in tab["pairs"].items():
            print(f"  {pair}")
            for lab, r in rows.items():
                print(f"    {lab}: {f(r['i_300'])} | {f(r['iii_end'])} | {f(r['orders'])}")
        print("  orders a unit-minute (all):", {p: v.get("all") for p, v in tab["orders_min"].items()})
        for p, e in tab["exec"].items():
            if e.get("all"):
                print(f"  exec {p} (all):", json.dumps(e["all"]))
    print("\nlines:", json.dumps(out["lines"], indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "table"))
    ap.add_argument("--sets", default="taught,held,G")
    ap.add_argument("--players", default=",".join(DEFAULT))
    ap.add_argument("--out", default="tmp/t02/exam")
    ap.add_argument("--n", type=int, default=0, help="a smaller set: n S2 scenes a part (G: the first n battles)")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sim", default="")
    ap.add_argument("--g-ref", default="")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a)
    return table(a.out, a.sets.split(","))


if __name__ == "__main__":
    main()
