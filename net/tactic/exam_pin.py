"""Drill T0-2 'strike the pinned one' (tmp/tactic_slice/design_attack_v23.md sections 1-2, 5): the players of
tactic/pin.py through the tactic level (stage0/battle.play_tactic, watch.PinWatch) on general battles, paired by battle.

    players   the pilot's 768 general battles (--set pilot: seeds 0-2 of the pilot's generator - stage0/specs.generals on
              SP.seeded(9100 + seed), the enemy drawn anew among the scripts with random.Random(9200 + seed), as
              tmp/tactic_slice/pilot/run.general_specs; the battles of a seed one chunk, its torch seed 2000 + seed), each
              player's file <out stem>_<player>.json {per_battle, hp_series, order_log (watch.PinWatch.ORDER_LOG; b the
              set's battle), meta (enemy, sizes), sim (the simulator's commit as --sim states it, sim_key the hash of its
              files)}; --ref name=path[@commit],...: the run-time assert that the player's file equals that reference bit
              for bit (per_battle on the reference's keys, hp_series, order_log) - only where the reference is keyed to
              this simulator (its sim_key, or @commit = --sim), else it says so and does not compare; a missing file is
              said too. Then the table of every player file of that stem in <out>
    table     the table only: the judge's three reads (design section 2) - (i) the anchor player's first contact + 300 s
              (--anchor, default C), (ii) each player's own first contact + 300 s, (iii) health at the end; orders to the
              game, attack orders by cause, within 10 s of the battle's first contact, the line's advance (walk-out);
              pairs (--pairs a-b,...) by the three reads and orders, by enemy script and by size
    labeller  the proving run of the stateless labeller (design section 1; review_attack_v23 A.3): the --driver (default
              TCw_back_co, asserted against --ref) with the --labeller (default TEACH) answering the same view at every
              tick - unit-ticks of able units of mine whose skill or target differ, and the new attack orders one gives
              and the other not (by the battle's first contact, the pin's age, the army's size) - then the labeller
              played (or read from --ref under its name); the rule's numbers: labeller - driver at 750 s within +-0.006
              and its two-error band holds 0, orders a unit-minute within +-0.01
A player's name: a key of pin.PLAYERS, optionally with ':holes' (config orders.hole_fits off - the old rule, a frame on a
hole must move), ':line' (orders.zone_fit off - a line zone on the units' middle), ':norout' (orders.end_on_rout off),
':nocap' (orders.cap_all off), ':mt' (orders.teach.mt on) and ':noreach' (orders.teach.come_reach_m null: TEACH with
come_own's reach the counter's). "Two errors" = paired().

    DOCK_NAME=agent-t2 DOCK_CPUS=8 DOCK_MEM=6g bash ops/dock.sh net.tactic.exam_pin players --players K,C --sim 1925110 \
        --ref K=tmp/tactic_slice/v2/stage1e/r/p_K.json@1925110 --out tmp/attack/s2/p.json
"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch

from net import ROOT
from net.stage0 import battle as BA
from net.stage0 import cfg as scfg
from net.stage0 import specs as SP
from net.stage0 import watch as W
from net.tactic import cfg as tcfg
from net.tactic import pin as PN

EVENTS = ("attack_orders", "intercepted", "pin_lost", "gone_before", "resumed_on_rallied", "resumed_way_m", "reached",
          "unpinned_attacks", "two_on_side", "melee_changes", "melee_blocked", "rejected", "forced_first",
          "forced_hole", "forced_zone", "hole_kept", "let_go", "changes", "ended_on_rout", "striker_routed",
          "ord_pinned", "ord_chasing", "ord_approaching", "ord_standing", "ord_other", "inter_nearer",
          "order60_inter_n", "order60_inter_sum", "order60_clean_n", "order60_clean_sum") + tuple(
          f"fc_{c}_{w}" for c in ("mine_charging", "enemy_onto_standing", "both_moving", "neither")
          for w in ("n", "n30", "s30", "n60", "s60"))
MEANS = ("hp_trade", "trade", "game_orders_min", "routs_own", "routs_enemy", "t_contact", "zone_worst_m") + EVENTS
WINDOW_S = 300
TSEED = 2000
SCRIPTS = ("human", "nearest", "hold_shoot")
SIZES = ((2, 4), (5, 8))
CELL_CATS = PN.CELL_CATS


def pilot_specs(n, seed, limit=0):
    """The pilot's general battles of a seed (module doc); limit: each battle's length, s (0: the stage's)."""
    shares = {k: v for k, v in scfg()["general"]["enemies"].items() if k != "chain"}
    sp = SP.generals(n, SP.seeded(9100 + seed))
    rnd = random.Random(9200 + seed)
    for x in sp:
        if x["enemy"] == "chain" or x["enemy"] not in shares:
            x["enemy"] = rnd.choices(list(shares), weights=list(shares.values()))[0]
        if limit:
            x["limit"] = int(limit)
    return sp


def pilot_set(n=256, seeds=(0, 1, 2), limit=750):
    """(specs, the chunks' torch seeds, chunk) of the pilot set: the seeds' battles in order, a seed's n battles one
    chunk with torch seed TSEED + seed (the runs of stages 1d-1f)."""
    sp = []
    for s in seeds:
        sp += pilot_specs(n, s, limit)
    return sp, [TSEED + s for s in seeds], n


def sim_key():
    """sha256 over the simulator's files (sim/ less evidence/, tests/, __pycache__): the key a result file carries."""
    h = hashlib.sha256()
    root = ROOT / "sim"
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if not p.is_file() or rel.startswith(("evidence/", "tests/")) or "__pycache__" in rel:
            continue
        h.update(rel.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _variant(name):
    base, *opts = name.split(":")
    o = tcfg()["orders"]
    saved = dict(o)
    saved["teach"] = dict(o["teach"])
    o["hole_fits"] = "holes" not in opts and bool(saved["hole_fits"])
    o["zone_fit"] = "line" not in opts and bool(saved["zone_fit"])
    o["end_on_rout"] = "norout" not in opts and bool(saved.get("end_on_rout", True))
    o["cap_all"] = "nocap" not in opts and bool(saved.get("cap_all", True))
    if "mt" in opts:
        o["teach"] = {**o["teach"], "mt": True}
    if "noreach" in opts:
        o["teach"] = {**o["teach"], "come_reach_m": None}
    return base, saved


def _restore(saved):
    o = tcfg()["orders"]
    o.update(saved)


def meta_of(specs):
    return [{"enemy": x["enemy"], "n": len(x["mine"]), "n_theirs": len(x["theirs"])} for x in specs]


def run_player(name, specs, seeds, chunk, a, P, make=None, on_chunk=None, watch_cls=None, chunk_args=False):
    """The player's file (module doc) on specs, chunk k under torch seed seeds[k]; make() - the player factory (None:
    pin.player of the name's base); on_chunk(ii, watch, side) - the caller's look at each chunk after it; watch_cls -
    a watch.PinWatch subclass; chunk_args - make(ii) and watch_cls(..., ii=) get the chunk's spec indices
    (battle.play_tactic)."""
    base, saved = _variant(name)
    log = []

    def keep(ii, watch, side):
        log.extend([ii[int(r[0])]] + list(r[1:]) for r in watch.order_log if int(r[0]) < len(ii))
        if on_chunk is not None:
            on_chunk(ii, watch, side)
    try:
        t0 = time.time()
        m, st = BA.play_tactic(specs, make or (lambda: PN.player(base)), P, a.device,
                               hygiene=base not in PN.HYGIENE_OFF, chunk=chunk,
                               compile=None if a.device.startswith("cuda") else False,
                               watch_cls=watch_cls or W.PinWatch, seeds=seeds, on_chunk=keep, chunk_args=chunk_args)
        print(json.dumps({"player": name, "battles": len(specs), "wall_s": st["wall_s"],
                          "min": round((time.time() - t0) / 60, 1),
                          **{k: round(float(m[k][torch.isfinite(m[k])].mean()), 4) for k in MEANS if k in m}}),
              flush=True)
    finally:
        _restore(saved)
    log.sort(key=lambda r: (r[0], r[2], r[1]))
    return {"wall_s": round(st["wall_s"], 1), "seeds": list(seeds),
            "per_battle": {k: [round(float(x), 4) for x in v.tolist()] for k, v in m.items() if v.dim() == 1},
            "hp_series": [[round(float(x), 4) for x in r] for r in m["hp_series"].tolist()],
            "order_log_cols": list(W.PinWatch.ORDER_LOG), "order_log": [[round(x, 4) for x in r] for r in log],
            "meta": meta_of(specs), "sim": getattr(a, "sim", ""), "sim_key": sim_key()}


def same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def _log_key(L):
    return sorted(L, key=lambda r: [x if x == x else -1e18 for x in r])


def ref_path(spec):
    """(path, commit) of a reference 'path[@commit]' (a commit: what follows the last @ when it has no '.' or '/')."""
    head, at, tail = spec.rpartition("@")
    if at and tail and not any(c in tail for c in "./\\"):
        return head, tail
    return spec, ""


def check_ref(name, res, spec, sim_commit):
    """The run-time assert of a player's file against its reference (module doc). spec 'path[@commit]'. Returns
    {against, compared, bit_for_bit | why}."""
    path, key = ref_path(spec)
    p = Path(path)
    if not p.exists():
        out = {"against": path, "compared": False, "why": "the reference file is absent"}
        print(json.dumps({"ref": name, **out}), flush=True)
        return out
    ref = json.loads(p.read_text(encoding="utf-8"))
    if ref.get("sim_key"):
        ok_sim, keyed = ref["sim_key"] == res["sim_key"], f"sim_key {ref['sim_key']}"
    else:
        ok_sim, keyed = bool(key) and key == sim_commit, f"commit {key or '?'}"
    if not ok_sim:
        out = {"against": path, "compared": False,
               "why": f"the reference is keyed to {keyed}, this run to commit {sim_commit or '?'} / sim_key "
                      f"{res['sim_key']}: another simulator"}
        print(json.dumps({"ref": name, **out}), flush=True)
        return out
    bad = [k for k in ref["per_battle"] if not same(ref["per_battle"][k], res["per_battle"].get(k))]
    hs = same(ref["hp_series"], res["hp_series"])
    lg = same(_log_key(ref.get("order_log", [])), _log_key(res["order_log"]))
    ok = hs and lg and not bad
    out = {"against": path, "compared": True, "keyed": keyed, "bit_for_bit": ok, "per_battle_differ": bad[:10],
           "hp_series": hs, "order_log": lg}
    print(json.dumps({"ref": name, **out}), flush=True)
    assert ok, f"{name} does not reproduce {path} bit for bit"
    return out


# ------------------------------------------------------------------ the reads and the tables
def paired(a, b, sel=None):
    """{diff, two_errors, n}: the paired difference a - b over the battles sel (finite pairs; None: all) - mean and two
    standard errors (the drill's 'two errors' everywhere)."""
    idx = range(len(a)) if sel is None else sel
    d = [a[i] - b[i] for i in idx if math.isfinite(a[i]) and math.isfinite(b[i])]
    n = len(d)
    if n < 2:
        return None
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    return {"diff": round(m, 4), "two_errors": round(2 * sd / math.sqrt(n), 4), "n": n}


def at(d, anchor, w=WINDOW_S):
    """[B] plain health at anchor + w (the series' step, capped at its last point) and the share of battles whose
    window passes the end of the series."""
    step = W.PinWatch.SERIES_S
    out, cut = [], 0
    for b, ser in enumerate(d["hp_series"]):
        i = int(round((anchor[b] + w) / step))
        cut += i > len(ser) - 1
        out.append(ser[min(i, len(ser) - 1)])
    return out, cut / max(len(d["hp_series"]), 1)


def reads(d, anchor):
    """The judge's three reads of a player file: (i) at the anchor's first contact + 300 s, (ii) at its own, (iii) the
    end (hp_trade)."""
    i, cut_i = at(d, anchor)
    ii, cut_ii = at(d, d["per_battle"]["t_contact"])
    return {"i": i, "ii": ii, "iii": d["per_battle"]["hp_trade"], "cut_i": round(cut_i, 4), "cut_ii": round(cut_ii, 4)}


def _mean(v):
    v = [x for x in v if isinstance(x, (int, float)) and math.isfinite(x)]
    return round(sum(v) / len(v), 4) if v else float("nan")


def groups(meta):
    out = [(en, [b for b, x in enumerate(meta) if x["enemy"] == en]) for en in SCRIPTS]
    out += [(f"{lo}-{hi} units", [b for b, x in enumerate(meta) if lo <= x["n"] <= hi]) for lo, hi in SIZES]
    return [(k, v) for k, v in out if v]


def player_row(d, rd):
    """The table row of a player file: the three reads, orders, attack orders by cause, within 10 s of the battle's
    first contact, at a target whose nearest unit of mine in its path is not fighting (cell), the walk-out."""
    pb = d["per_battle"]
    B = len(d["hp_series"])
    tc = pb["t_contact"]
    row = {"iii_end": _mean(rd["iii"]), "i_anchor300": _mean(rd["i"]), "ii_own300": _mean(rd["ii"]),
           "i_cut": rd["cut_i"], "ii_cut": rd["cut_ii"]}
    for k in MEANS:
        if k in pb:
            row[k] = _mean(pb[k])
    ao = sum(pb.get("attack_orders", [0]))
    for k in ("intercepted", "pin_lost", "gone_before", "resumed_on_rallied", "reached", "unpinned_attacks"):
        if k in pb and ao:
            row[k + "_share"] = round(sum(pb[k]) / ao, 4)
    for k in ("advance_0", "advance_60"):
        if k in pb:
            row[k] = _mean(pb[k])
    if "order_log" in d:
        ci = {k: i for i, k in enumerate(d["order_log_cols"])}
        rows = d["order_log"]
        row["attacks_battle"] = round(len(rows) / B, 3)
        for c, cn in ((1, "pin"), (2, "second"), (3, "other")):
            row[f"by_cause_{cn}"] = round(sum(1 for r in rows if int(r[ci["cause"]]) == c) / B, 3)
        rel = [r[ci["sec"]] - (tc[int(r[ci["b"]])] - 1) for r in rows]
        row["within10_of_contact"] = round(sum(1 for x in rel if 0 <= x < 10) / B, 3)
        row["before_contact"] = round(sum(1 for x in rel if x < 0) / B, 3)
        row["at_nearest_not_fighting"] = round(sum(1 for r in rows if int(r[ci["cell"]]) // 5 in (1, 3))
                                               / max(len(rows), 1), 4)
    row["wall_s"] = d.get("wall_s")
    return row


def by_cause(d):
    """Attack orders by cause and cell (the stage 1e split): share, a battle, the pair's trade and the striker's loss
    60 s on; first tick after the battle's first contact / later / before."""
    ci = {k: i for i, k in enumerate(d["order_log_cols"])}
    tc = d["per_battle"]["t_contact"]
    B = len(d["hp_series"])
    out = {}
    for c, cn in ((1, "pin"), (2, "second"), (3, "other")):
        rs = [r for r in d["order_log"] if int(r[ci["cause"]]) == c]
        if not rs:
            continue
        cell = {"a_battle": round(len(rs) / B, 3)}
        tests = [(CELL_CATS[k], (lambda r, k=k: int(r[ci["cell"]]) // 5 == k)) for k in range(len(CELL_CATS))]
        tests += [("first_tick", lambda r: 0 <= r[ci["sec"]] - (tc[int(r[ci["b"]])] - 1) < 5),
                  ("later", lambda r: r[ci["sec"]] - (tc[int(r[ci["b"]])] - 1) >= 5),
                  ("before_contact", lambda r: r[ci["sec"]] - (tc[int(r[ci["b"]])] - 1) < 0)]
        for lab, f in tests:
            q = [r for r in rs if f(r)]
            if q:
                cell[lab] = {"share": round(len(q) / len(rs), 3), "trade60": _mean([r[ci["trade60"]] for r in q]),
                             "loss60": _mean([r[ci["loss60"]] for r in q])}
        out[cn] = cell
    return out


def table(files, anchor_name="C", pairs=None):
    """{players, pairs, by_group, by_cause} of the player files {name: path} (module doc); the anchor's first contact
    from its file (absent: each player's own)."""
    pl = {n: json.loads(Path(f).read_text(encoding="utf-8")) for n, f in files.items()}
    anchor = pl[anchor_name]["per_battle"]["t_contact"] if anchor_name in pl else None
    R = {n: reads(d, anchor if anchor is not None else d["per_battle"]["t_contact"]) for n, d in pl.items()}
    out = {"anchor": anchor_name if anchor is not None else "own", "players": {}, "pairs": {}, "by_group": {},
           "by_cause": {}}
    for n, d in pl.items():
        out["players"][n] = player_row(d, R[n])
        if "order_log" in d:
            out["by_cause"][n] = by_cause(d)
    names = list(pl)
    want = [tuple(p.split("-", 1)) for p in pairs] if pairs else [(x, y) for x in names for y in names if x != y]
    meta = next((d["meta"] for d in pl.values() if "meta" in d), None)
    for x, y in want:
        if x not in pl or y not in pl:
            continue
        row = {k: paired(R[x][k], R[y][k]) for k in ("iii", "i", "ii")}
        for k in ("game_orders_min", "routs_own", "routs_enemy", "trade"):
            if k in pl[x]["per_battle"] and k in pl[y]["per_battle"]:
                row[k] = paired(pl[x]["per_battle"][k], pl[y]["per_battle"][k])
        out["pairs"][f"{x} - {y}"] = row
        if meta is not None and pairs:
            out["by_group"][f"{x} - {y}"] = {lab: {k: paired(R[x][k], R[y][k], sel) for k in ("iii", "i", "ii")}
                                            for lab, sel in groups(meta)}
    return out


def fmt(p):
    return "-" if p is None else f"{p['diff']:+.4f} ± {p['two_errors']:.4f} ({p['n']})"


def print_table(res):
    cols = ("iii_end", "i_anchor300", "ii_own300", "game_orders_min", "attacks_battle", "by_cause_pin",
            "by_cause_second", "by_cause_other", "within10_of_contact", "at_nearest_not_fighting", "advance_0",
            "advance_60", "t_contact")
    print(f"| player | {' | '.join(cols)} |  (anchor of (i): {res['anchor']})")
    for n, r in res["players"].items():
        print(f"| {n} | " + " | ".join(str(r.get(c, "-")) for c in cols) + " |")
    print("pairs: (iii) end | (i) anchor + 300 | (ii) own + 300 | orders a unit-minute")
    for k, r in res["pairs"].items():
        print(f"  {k}: {fmt(r['iii'])} | {fmt(r['i'])} | {fmt(r['ii'])} | {fmt(r.get('game_orders_min'))}")
        for lab, g in res["by_group"].get(k, {}).items():
            print(f"     {lab}: {fmt(g['iii'])} | {fmt(g['i'])} | {fmt(g['ii'])}")


# ------------------------------------------------------------------ the labeller's proving run
class Shadow:
    """The driver's output, and the labeller asked on the same view (stateless: the call order does not matter);
    counts of unit-ticks (able units of mine) whose skill or target differ, by battle of the chunk."""

    def __init__(self, driver, labeller):
        self.main, self.shadow = driver, labeller
        self.allow = getattr(driver, "allow", (0, 2))
        self.name = getattr(driver, "name", "driver")
        self.n, self.diff = {}, {}
        self.only = {"driver": [], "labeller": []}     # new attack orders one gives and the other not: (b, sec, mt(e))

    @property
    def last_tags(self):
        return self.main.last_tags

    def __call__(self, v):
        out = self.main(v)
        o2 = self.shadow(v)
        fx = v.fx
        able = fx.able
        d = able & ((out["skill"] != o2["skill"]) | (out["target"] != o2["target"]))
        rows = v.idx.tolist()
        for b, na, nd in zip(rows, able.sum(1).tolist(), d.sum(1).tolist()):
            self.n[b] = self.n.get(b, 0) + na
            self.diff[b] = self.diff.get(b, 0) + nd
        o = v.order
        new = lambda x: able & (x["skill"] == 2) & (x["target"] >= 0) & ((x["target"] != o["tgt"]) | (o["skill"] != 2))  # noqa
        n1, n2 = new(out), new(o2)
        now = int(round(float(v.now)))
        for who, m, x in (("driver", n1 & ~(n2 & (out["target"] == o2["target"])), out),
                          ("labeller", n2 & ~(n1 & (out["target"] == o2["target"])), o2)):
            if bool(m.any()):
                k = fx.row_of_slot(torch.where(m, x["target"], torch.full_like(x["target"], -1)))
                mt = fx.mt.gather(1, k.clamp(min=0))
                for r, g in m.nonzero().tolist():
                    self.only[who].append((rows[r], now, int(mt[r, g])))
        return out


def _split(only, meta, tc):
    """New attack orders one player gives on the view and the other not, by the battle's first contact (the first
    10 s after it / later / before), by the pin's age (mt(e) 2 fresh / >= 3 late) and by my army's size."""
    out = {"n": len(only), "a_battle": round(len(only) / max(len(meta), 1), 4)}
    for b, sec, mt in only:
        rel = sec - (tc[b] - 1)
        for k in ("contact:" + ("before" if rel < 0 else "first10s" if rel < 10 else "later"),
                  "pin:" + ("fresh" if mt <= 2 else "late"),
                  "size:" + next(f"{lo}-{hi}" for lo, hi in SIZES if lo <= meta[b]["n"] <= hi)):
            out[k] = out.get(k, 0) + 1
    return out


def labeller(a, P, stem, refs):
    """The proving run (module doc): --driver with --labeller on the same view, then --labeller played; returns the
    numbers and the rule's verdict."""
    specs, seeds, chunk = pilot_set(a.n, range(a.seeds), a.limit)
    drv, lab = a.driver, a.labeller
    cnt = {"unit_ticks": 0, "differ": 0, "only": {"driver": [], "labeller": []}}

    o = tcfg()["orders"]
    plain = {**o, "teach": dict(o["teach"])}           # the config before run_player lays the driver's variant

    def built(name):
        # each player under its OWN name's options, laid on the plain config: make() is called inside run_player, where
        # the driver's variant is in force (a labeller built there would silently take the driver's options)
        outer = {**o, "teach": dict(o["teach"])}
        o.update({**plain, "teach": dict(plain["teach"])})
        base, _ = _variant(name)
        try:
            return PN.player(base)
        finally:
            o.update(outer)

    def make():
        return Shadow(built(drv), built(lab))

    def look(ii, watch, side):
        sh = side.player
        cnt["unit_ticks"] += sum(sh.n.values())
        cnt["differ"] += sum(sh.diff.values())
        for k, v in sh.only.items():
            cnt["only"][k] += [(ii[b], s, m) for b, s, m in v if b < len(ii)]
    res = run_player(drv, specs, seeds, chunk, a, P, make=make, on_chunk=look)
    res["selfcheck"] = check_ref(drv, res, refs[drv], a.sim) if drv in refs else {"compared": False, "why": "no --ref"}
    stem(drv + "@shadow").write_text(json.dumps(res), encoding="utf-8")
    if lab in refs:
        res2 = json.loads(Path(ref_path(refs[lab])[0]).read_text(encoding="utf-8"))
        print(json.dumps({"labeller": lab, "played": "read from --ref", "file": refs[lab]}), flush=True)
    else:
        res2 = run_player(lab, specs, seeds, chunk, a, P)
        stem(lab).write_text(json.dumps(res2), encoding="utf-8")
    meta, tc = res["meta"], res["per_battle"]["t_contact"]
    B = len(meta)
    d = paired(res2["per_battle"]["hp_trade"], res["per_battle"]["hp_trade"])
    o = paired(res2["per_battle"]["game_orders_min"], res["per_battle"]["game_orders_min"])
    ab = paired(res2["per_battle"]["attack_orders"], res["per_battle"]["attack_orders"])
    ok = (abs(d["diff"]) <= 0.006 and abs(d["diff"]) <= d["two_errors"]) and abs(o["diff"]) <= 0.01
    return {"driver": drv, "labeller": lab, "labeller_minus_driver_750": d, "orders_unit_min": o,
            "attack_orders_battle": ab, "unit_ticks": cnt["unit_ticks"], "unit_ticks_differ": cnt["differ"],
            "share_differ": round(cnt["differ"] / max(cnt["unit_ticks"], 1), 6),
            "on_one_view": {"driver_only": _split(cnt["only"]["driver"], meta, tc),
                            "labeller_only": _split(cnt["only"]["labeller"], meta, tc)}, "battles": B,
            "rule": "|diff| <= 0.006, the two-error band holds 0, orders within +-0.01 a unit-minute", "passed": ok}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("players", "table", "labeller"))
    ap.add_argument("--set", default="pilot", choices=("pilot",))
    ap.add_argument("--players", default="K,C")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--limit", type=int, default=750, help="each battle's length, s (0: the stage's 600)")
    ap.add_argument("--ref", default="", help="bit-for-bit references: name=path[@sim commit],...")
    ap.add_argument("--sim", default="", help="the simulator's commit this run is on (git log -1 -- sim, said by the caller)")
    ap.add_argument("--driver", default="TCw_back_co", help="labeller: the player that drives")
    ap.add_argument("--labeller", default="TEACH", help="labeller: the player asked on the same view, then played")
    ap.add_argument("--anchor", default="C", help="read (i): this player's first contact + 300 s")
    ap.add_argument("--pairs", default="", help="a-b,...: the pairs of the table (with groups); none: every pair")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="tmp/attack/s2/p.json")
    a = ap.parse_args(argv)
    from sim.params import load
    P = load()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = lambda n: out.with_name(out.stem + "_" + n.replace(":", "@") + ".json")       # noqa: E731
    refs = dict(x.split("=", 1) for x in a.ref.split(",") if x)
    t0 = time.time()
    extra = {}
    if a.what == "players":
        specs, seeds, chunk = pilot_set(a.n, range(a.seeds), a.limit)
        for name in a.players.split(","):
            res = run_player(name, specs, seeds, chunk, a, P)
            if name in refs:
                res["selfcheck"] = check_ref(name, res, refs[name], a.sim)
            stem(name).write_text(json.dumps(res), encoding="utf-8")
    elif a.what == "labeller":
        extra["labeller"] = labeller(a, P, stem, refs)
        print(json.dumps(extra["labeller"]), flush=True)
    files = {p.stem[len(out.stem) + 1:].replace("@", ":"): str(p) for p in sorted(out.parent.glob(out.stem + "_*.json"))}
    res = table(files, a.anchor, [x for x in a.pairs.split(",") if x])
    res.update(extra)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print_table(res)


if __name__ == "__main__":
    main()
