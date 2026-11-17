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
    scenes    --check: stage 2c's first slice (design section 4 checks 1, 2, 3, 5, 9, 10, 12, 13): --n-scenes A1 scenes
              (tactic/scenes, seeds from config drills.attack.seeds) and their N1 twins played by TEACH with the
              recorder (tactic/tags.Record: the labeller TEACH built under its own name, asked on the same view), then
              the pilot's battles with shuffled slots (specs.shuffle_slots) by TEACH (recorded) and C; <out dir>/
              check.json. --vision A1:<seed>,N1:<seed>,G:<battle>: records for game.vision (<out dir>/vision/<name>/)
    exam      stage 3, the exam before training (design section 5; review C, D, I): --set G (config drills.attack.sets.G:
              1,024 fresh general battles on the trained sizes + 256 on the held ones, slots shuffled) or S (the scenes'
              set S, as `scenes --check --scene-set S` plays it) or F (S's recipes and counts on fresh seeds, review L)
              - each of --players (C,TEACH,TEACH:taught) through
              HoleWatch (the first contact of each strike, my line's widest hole every 5 s), TEACH recorded
              (tags.Record: S's cause scenes = those with a full-weight label); <out dir>/<set>_<player>.json, then
              exam_table: B5, B8, B11, events, by script and size (G) / by family (S), the strikes' contact sides, the
              hole 30 s after each strike against C's at the same second; <out dir>/exam_<set>.json
A player's name: a key of pin.PLAYERS, optionally with ':holes' (config orders.hole_fits off - the old rule, a frame on a
hole must move), ':line' (orders.zone_fit off - a line zone on the units' middle), ':norout' (orders.end_on_rout off),
':nocap' (orders.cap_all off), ':mt' (orders.teach.mt on), ':noreach' (orders.teach.come_reach_m null: TEACH with
come_own's reach the counter's), ':taught' (orders.teach.no_through on: TEACH giving only the attack orders the loss
carries at full weight, review I.2) and ':reach<m>' (orders.reach_m = m: a reference run keyed to its reach, e.g.
TEACH:reach150). "Two errors" = paired().

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
    if "taught" in opts:
        o["teach"] = {**o["teach"], "no_through": True}
    for x in opts:                       # ':reach<m>' - orders.reach_m pinned to m (a reference keyed to its reach)
        if x.startswith("reach") and x[5:].replace(".", "", 1).isdigit():
            o["reach_m"] = float(x[5:])
    return base, saved


def _restore(saved):
    o = tcfg()["orders"]
    o.update(saved)


def meta_of(specs):
    return [{"enemy": x["enemy"], "n": len(x["mine"]), "n_theirs": len(x["theirs"])} for x in specs]


def run_player(name, specs, seeds, chunk, a, P, make=None, on_chunk=None, watch_cls=None, chunk_args=False, **play):
    """The player's file (module doc) on specs, chunk k under torch seed seeds[k]; make() - the player factory (None:
    pin.player of the name's base); on_chunk(ii, watch, side) - the caller's look at each chunk after it; watch_cls -
    a watch.PinWatch subclass; chunk_args - make(ii) and watch_cls(..., ii=) get the chunk's spec indices
    (battle.play_tactic); play - more of play_tactic's arguments (battles, enemy, full, by_limit)."""
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
                               watch_cls=watch_cls or W.PinWatch, seeds=seeds, on_chunk=keep, chunk_args=chunk_args,
                               **play)
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


# ------------------------------------------------------------------ the scenes (stage 2c's first slice: A1, N1, G)
def build_under(name):
    """A player under its OWN name's options laid on the plain config (never a driver's variant in force)."""
    o = tcfg()["orders"]
    outer = {**o, "teach": dict(o["teach"])}
    base, saved = _variant(name)
    try:
        return PN.player(base)
    finally:
        _restore(saved)
        o.update(outer)


class SceneWatch(W.PinWatch):
    """PinWatch + each attack order's first contact with its target: the side of the target the contact began on
    (sim hit_sec: 1 front, 2 flank, 3 rear) - contacts (b, my slot, target slot, second, side)."""

    def _init(self, st):
        super()._init(st)
        self.c_tgt = torch.full((self.B, self.N), -1, dtype=torch.long, device=self.dev)
        self.c_done = torch.zeros((self.B, self.N), dtype=torch.bool, device=self.dev)
        self.contacts = []

    def __call__(self, sec, st):
        super().__call__(sec, st)
        tac = getattr(self.chain, "tac", None)
        if tac is None:
            return
        o = tac.orders_of()
        att = o["on"] & (o["skill"] == 2) & (o["tgt"] >= 0) & self.mine
        t = torch.where(att, o["tgt"], torch.full_like(o["tgt"], -1))
        self.c_done &= t == self.c_tgt
        self.c_tgt = t
        hs = st.u["hit_sec"].view(self.B, self.N, self.N)
        tc = t.clamp(min=0)
        sd = hs.gather(1, tc[:, None, :])[:, 0, :]                                    # hs[b, t, i]: the side of t
        back = hs.gather(2, tc[..., None])[..., 0]                                    # hs[b, i, t]: i touched by t
        new = att & ~self.c_done & ((sd > 0) | (back > 0)) & ~st.done[:, None]
        for b, i in new.nonzero().tolist():
            self.contacts.append((b, i, int(t[b, i]), int(sec), int(sd[b, i])))
        self.c_done |= new


def line_holes(w, st, hole_m=50.0):
    """([B] the widest lateral gap between my standing units - all depths, across the start's axis my centroid -> the
    enemy's; 0 under two units -, [B] the number of gaps over hole_m) for a PinWatch w (branches.FWatch's measure)."""
    from sim import geometry
    u = st.u
    front, _ = geometry.dims(u, w.spacing)
    ok = w.mine & ~u["r"] & (u["men"] > 0) & ~u["gone"]
    fx_, fz_ = w.fwd
    s = u["x"] * fz_[:, None] - u["z"] * fx_[:, None]
    big = 1e9
    lo = torch.where(ok, s - front / 2, torch.full_like(s, big))
    hi = torch.where(ok, s + front / 2, torch.full_like(s, -big))
    lo, order = lo.sort(1)
    reach = hi.gather(1, order).cummax(1).values
    gaps = torch.where(lo[:, 1:] < big / 2, lo[:, 1:] - reach[:, :-1], torch.full_like(lo[:, 1:], -big))
    g = gaps.amax(1).clamp(min=0.0)
    return torch.where(ok.sum(1) >= 2, g, torch.zeros_like(g)), (gaps > hole_m).float().sum(1)


class HoleWatch(SceneWatch):
    """SceneWatch + my line's widest hole (line_holes) every SERIES_S s from second 0: hole_series [T][B]."""

    def _init(self, st):
        super()._init(st)
        self.hole_series = []

    def __call__(self, sec, st):
        super().__call__(sec, st)
        if sec % self.SERIES_S == 0:
            self.hole_series.append(line_holes(self, st)[0].tolist())


def g_set():
    """(specs, torch seeds, chunk) of set G (module doc; config drills.attack.sets.G)."""
    c = scfg()["drills"]["attack"]["sets"]["G"]
    lim = int(c["limit"])
    sp = SP.drill_generals(int(c["n"]), SP.seeded(int(c["seed"])), sizes=c["sizes"], limit=lim)
    sp += SP.drill_generals(int(c["n_held"]), SP.seeded(int(c["held_seed"])), sizes=c["sizes_held"], limit=lim)
    ch = int(c["chunk"])
    return sp, [int(c["tseed"]) + k for k in range((len(sp) + ch - 1) // ch)], ch


def s_set(chunk=256, name="S"):
    """(specs, torch seeds, chunk) of set S as `scenes --check --scene-set S` plays it (TSEED + 200 + k); name F: the
    fresh-seed set F of config drills.attack.sets.F (its own tseed + k)."""
    from net.tactic import scenes as SN
    sp = SN.set_specs(name)
    t0 = TSEED + 200 if name == "S" else int(scfg()["drills"]["attack"]["sets"][name]["tseed"])
    return sp, [t0 + k for k in range((len(sp) + chunk - 1) // chunk)], chunk


def watched_run(name, specs, seeds, chunk, a, P, scene=False, record=False):
    """The player's file (run_player) through HoleWatch with full / by_limit batches, + contacts (b, my slot, target,
    second, side), holes {b: series} and, with record, rec (tags.Record's labels and counts: TEACH built under its own
    name, the labeller the same)."""
    from net.tactic import tags as TG
    ex = {"contacts": [], "holes": {}, "rec": {"cnt": {}, "labels": []}}

    def make():
        return TG.Record(build_under(name), build_under(name))

    def look(ii, watch, side):
        hs = list(zip(*watch.hole_series))
        for b in range(min(len(ii), len(hs))):
            ex["holes"][ii[b]] = [round(x, 2) for x in hs[b]]
        ex["contacts"] += [(ii[b], i, t, s, sd) for b, i, t, s, sd in watch.contacts if b < len(ii)]
        if record:
            rec = side.player
            for b, c in rec.cnt.items():
                if b < len(ii):
                    ex["rec"]["cnt"][ii[b]] = c
            ex["rec"]["labels"] += [{**x, "b": ii[x["b"]]} for x in rec.labels if x["b"] < len(ii)]
    kw = {"battles": BA.scene_battles, "enemy": BA.SceneEnemy} if scene else {}
    res = run_player(name, specs, seeds, chunk, a, P, make=make if record else None, on_chunk=look,
                     watch_cls=HoleWatch, full=True, by_limit=True, **kw)
    res.update(contacts=ex["contacts"], holes={str(b): v for b, v in ex["holes"].items()},
               recipes=[s.get("recipe", "G") for s in specs])
    if record:
        res["rec"] = ex["rec"]
    return res


def _strike_rows(d):
    """[(b, my slot, target, second)] of a player file's attack orders (order_log)."""
    ci = {k: i for i, k in enumerate(d["order_log_cols"])}
    return [(int(r[ci["b"]]), int(r[ci["slot"]]), int(r[ci["target"]]), float(r[ci["sec"]])) for r in d["order_log"]]


def strike_sides(d):
    """{(b, slot, target, sec): side} - each strike's first contact (1 front, 2 flank, 3 rear; absent: none)."""
    by = {}
    for b, i, t, s, sd in d.get("contacts", []):
        by.setdefault((int(b), int(i), int(t)), []).append((int(s), int(sd)))
    out = {}
    for b, i, t, s in _strike_rows(d):
        c = [x for x in by.get((b, i, t), []) if x[0] >= s]
        if c:
            out[(b, i, t, s)] = min(c)[1]
    return out


def side_shares(d, sel=None):
    S = None if sel is None else set(sel)
    rows = [r for r in _strike_rows(d) if S is None or r[0] in S]
    sd = strike_sides(d)
    n = max(len(rows), 1)
    out = {"strikes": len(rows)}
    for k, v in (("front", 1), ("flank", 2), ("rear", 3)):
        out[k] = round(sum(1 for r in rows if sd.get(r) == v) / n, 4)
    out["no_contact"] = round(sum(1 for r in rows if r not in sd) / n, 4)
    return out


def holes_after(d, dc, sel=None, after=30):
    """My line's widest hole `after` s after each strike of d against dc's at the same second (paired by strike), all
    and by the strike's contact side."""
    S = None if sel is None else set(sel)
    hd, hc = d.get("holes", {}), dc.get("holes", {})
    sd = strike_sides(d)
    pairs = {"all": ([], [])}
    for r in _strike_rows(d):
        b, s = r[0], r[3]
        if S is not None and b not in S:
            continue
        x, y = hd.get(str(b)), hc.get(str(b))
        k = int(round((s + after) / W.PinWatch.SERIES_S))
        if not x or not y or k >= min(len(x), len(y)):
            continue
        for lab in ("all", {1: "front", 2: "flank", 3: "rear"}.get(sd.get(r), "no_contact")):
            pairs.setdefault(lab, ([], []))
            pairs[lab][0].append(x[k])
            pairs[lab][1].append(y[k])
    return {lab: {"mean": _mean(a), "mean_C": _mean(c), "diff": paired(a, c)} for lab, (a, c) in pairs.items()}


def full_label_battles(d):
    return sorted({x["b"] for x in d.get("rec", {}).get("labels", []) if not x["dc_label"] and not x["dropped"]})


def exam(a, P, D):
    """`exam_pin exam --set G|S` (module doc): the players' files, then exam_table."""
    sname = a.set
    specs, seeds, chunk = g_set() if sname == "G" else s_set(a.chunk, sname)
    for name in a.players.split(","):
        f = D / f"{sname}_{name.replace(':', '@')}.json"
        if f.exists():
            print(json.dumps({"exam": sname, "player": name, "kept": str(f)}), flush=True)
            continue
        res = watched_run(name, specs, seeds, chunk, a, P, scene=sname != "G", record=name == "TEACH")
        f.write_text(json.dumps(res), encoding="utf-8")
    return exam_table(D, sname)


def exam_table(D, sname):
    """The stage-3 table of set sname from <D>/<set>_<player>.json (module doc)."""
    names = [n for n in ("C", "TEACH", "TEACH:taught") if (D / f"{sname}_{n.replace(':', '@')}.json").exists()]
    files = {n: str(D / f"{sname}_{n.replace(':', '@')}.json") for n in names}
    pairs = [p for p in ("TEACH-C", "TEACH:taught-C", "TEACH:taught-TEACH") if all(x in names for x in p.split("-", 1))]
    res = table(files, "C", pairs)
    pl = {n: json.loads(Path(f).read_text(encoding="utf-8")) for n, f in files.items()}
    meta = pl["C"]["meta"]
    B = len(meta)
    anchor = pl["C"]["per_battle"]["t_contact"]
    R = {n: reads(d, anchor) for n, d in pl.items()}
    grp = []
    if sname == "G":
        held = set(scfg()["drills"]["attack"]["sets"]["G"]["sizes_held"])
        grp += [("trained sizes", [b for b in range(B) if meta[b]["n"] not in held]),
                ("held sizes", [b for b in range(B) if meta[b]["n"] in held])]
    else:
        rc = pl["C"]["recipes"]
        cause = set(full_label_battles(pl["TEACH"])) if "TEACH" in pl else set()
        grp += [("cause scenes", sorted(cause)), ("no-cause scenes", [b for b in range(B) if b not in cause])]
        grp += [(f"family {r}", [b for b in range(B) if rc[b] == r]) for r in dict.fromkeys(rc)]
        grp += [(f"family {r}, cause scenes", [b for b in range(B) if rc[b] == r and b in cause])
                for r in dict.fromkeys(rc)]
    out = {"set": sname, "battles": B, "players": res["players"], "pairs": res["pairs"], "by_group": res["by_group"]}
    for p in pairs:
        x, y = p.split("-", 1)
        g = out["by_group"].setdefault(f"{x} - {y}", {})
        for lab, sel in grp:
            if sel:
                g[lab] = {k: paired(R[x][k], R[y][k], sel) for k in ("iii", "i", "ii")}
                g[lab]["orders"] = paired(pl[x]["per_battle"]["game_orders_min"], pl[y]["per_battle"]["game_orders_min"],
                                          sel)
    # B5
    tc = out["pairs"].get("TEACH - C", {}).get("iii")
    bg = out["by_group"].get("TEACH - C", {})
    if sname == "G":
        subs = {k: bg[k]["iii"] for k in list(SCRIPTS) + [f"{lo}-{hi} units" for lo, hi in SIZES] if k in bg}
        out["B5"] = {"rule": "TEACH - C at (iii) above two errors on G; above zero against each script and in both size "
                             "classes (read: the point above zero; the two-error band printed)",
                     "all": tc, "subs": subs,
                     "passed": bool(tc and tc["diff"] > tc["two_errors"] and all(v and v["diff"] > 0 for v in subs.values())),
                     "subs_band_above_zero": {k: bool(v and v["diff"] - v["two_errors"] > 0) for k, v in subs.items()}}
    else:
        cs = bg.get("cause scenes", {}).get("iii")
        out["B5"] = {"rule": "TEACH - C at (iii) above two errors on S's cause scenes (a full-weight teacher order)",
                     "cause_scenes": cs, "passed": bool(cs and cs["diff"] > cs["two_errors"])}
    pr = out["players"]
    out["B8"] = {n: pr[n].get("game_orders_min") for n in pr}
    out["B8"]["rule"] = "TEACH <= 1.2 a unit-minute (METHOD's mark; printed - cannot fail today)"
    out["B8"]["passed"] = bool(pr.get("TEACH", {}).get("game_orders_min", 9) <= 1.2)
    c = pr["C"]
    out["B11_C_before"] = {k: c.get(k) for k in ("iii_end", "i_anchor300", "ii_own300", "hp_trade", "trade",
                                                 "game_orders_min", "routs_own", "routs_enemy", "t_contact")}
    out["events"] = {n: {k: pr[n].get(k) for k in ("attacks_battle", "reached_share", "intercepted_share",
                                                   "pin_lost_share", "within10_of_contact", "before_contact")}
                     for n in pr}
    sel_all = [None] + ([sorted(full_label_battles(pl["TEACH"]))] if sname != "G" and "TEACH" in pl else [])
    out["contact_sides"] = {n: {"all": side_shares(pl[n])} for n in pl if n != "C"}
    out["holes_30s"] = {n: {"all": holes_after(pl[n], pl["C"])} for n in pl if n != "C"}
    if len(sel_all) > 1:
        for n in out["contact_sides"]:
            out["contact_sides"][n]["cause scenes"] = side_shares(pl[n], sel_all[1])
            out["holes_30s"][n]["cause scenes"] = holes_after(pl[n], pl["C"], sel_all[1])
    if sname != "G":
        rc = pl["C"]["recipes"]
        for n in out["contact_sides"]:
            out["contact_sides"][n]["by_family"] = {r: side_shares(pl[n], [b for b in range(B) if rc[b] == r])
                                                    for r in dict.fromkeys(rc)}
    (D / f"exam_{sname}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print_exam(out)
    return out


def print_exam(out):
    print(f"\n## exam before, set {out['set']} ({out['battles']} battles)")
    print_table({"anchor": "C", "players": out["players"], "pairs": out["pairs"], "by_group": out["by_group"]})
    for k in ("B5", "B8", "B11_C_before", "events", "contact_sides", "holes_30s"):
        print(f"{k}: {json.dumps(out[k])}")


def men_of(s):
    """The men of a scene at its start, both sides with lords."""
    from net.tactic import scenes as SN
    m = sum(SN._men0(scfg()["lords"][s["fac"][k]]) for k in (1, 2))
    for side in ("mine", "theirs"):
        for key, w in zip(s[side], s["worn"][side]):
            m += SN._men0(key) if w is None else math.ceil(SN._men0(key) * w)
    return m


def scene_specs(n, seed0):
    from net.tactic import scenes as SN
    a1 = SN.scenes("A1", n, seed0)
    n1 = SN.scenes("N1", n, seed0)
    return a1, n1


def shuffled_pilot(n, seeds, limit):
    sp, ts, chunk = pilot_set(n, seeds, limit)
    base = int(scfg()["drills"]["attack"]["seeds"]["pilot_shuffle"])
    for i, x in enumerate(sp):
        SP.shuffle_slots(x, random.Random(base + i))
    return sp, ts, chunk


def _spearman(a, b):
    import numpy as np
    if len(a) < 2:
        return None
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def slot_rank(specs):
    """The mean over battles of the rank correlation of my slot index with the place along the row (zone axes)."""
    v = []
    for s in specs:
        if s.get("places_mine"):
            pl = s["places_mine"]
        else:
            pl = SP.rows(s["mine_zone"], len(s["mine"]))
            if s.get("perm") is not None:
                pl = [pl[p] for p in s["perm"]]
        x = [SP.local(s["mine_zone"], p[0], p[1])[0] for p in pl]
        r = _spearman(list(range(len(x))), x)
        if r is not None:
            v.append(r)
    return round(sum(v) / max(len(v), 1), 4), len(v)


def record_run(name, specs, seeds, chunk, a, P, scene=False):
    """(the player's file, its recorder's counts / labels / first frames by spec, contacts): TEACH drives with the
    labeller TEACH (built under its own name) asked on the same view (tags.Record)."""
    from net.tactic import tags as TG
    got = {"cnt": {}, "labels": [], "first": {}, "contacts": []}

    def make():
        return TG.Record(build_under(name), build_under(name))

    def look(ii, watch, side):
        rec = side.player
        for b, c in rec.cnt.items():
            if b < len(ii):
                got["cnt"][ii[b]] = c
        got["labels"] += [{**x, "b": ii[x["b"]]} for x in rec.labels if x["b"] < len(ii)]
        got["first"].update({ii[b]: x for b, x in rec.first.items() if b < len(ii)})
        got["contacts"] += [(ii[b], i, t, s, sd) for b, i, t, s, sd in watch.contacts if b < len(ii)]
    kw = {"battles": BA.scene_battles, "enemy": BA.SceneEnemy} if scene else {}
    res = run_player(name, specs, seeds, chunk, a, P, make=make, on_chunk=look, watch_cls=SceneWatch, full=True,
                     by_limit=True, **kw)
    return res, got


def kinds_table(got, sel):
    from net.tactic import tags as TG
    cols = TG.Record.COUNTS
    tot = [0] * len(cols)
    for b in sel:
        for j, x in enumerate(got["cnt"].get(b, [0] * len(cols))):
            tot[j] += x
    t = dict(zip(cols, tot))
    keep = max(t["keep"], 1)
    out = {"head_free": t["head_free"], "attack_labels": t["attack"], "keeps": t["keep"],
           "trivial_keep_share": round(t["trivial"] / keep, 4), "dropped_share_of_head_free":
           round(t["dropped"] / max(t["head_free"], 1), 4), "taught_keep_share": round(t["taught"] / keep, 4),
           "keep_with_dc_key_share": round(t["dc_key_keep"] / keep, 4),
           "band_key_keep_share": round(t["band_key_keep"] / keep, 4),
           "attack_labels_dc_share": round(t["dc_label"] / max(t["attack"], 1), 4)}
    out["kinds_share_of_keeps"] = {k: round(t[k] / keep, 4) for k in TG.KINDS}
    return out


def label_marks(got, sel):
    """Checks 3, 9 (sides), 13 over the attack labels of the battles sel."""
    S = set(sel)
    L = [x for x in got["labels"] if x["b"] in S]
    first = {}
    for x in sorted(L, key=lambda x: (x["sec"], x["slot"])):
        first.setdefault(x["b"], x)
    lr = [x["side"] for x in L if x["side"] in (1, 2)]
    n = max(len(L), 1)
    return {"attack_labels": len(L), "scenes_with_label": len(first),
            "first_state_no_share": round(sum(1 for x in first.values() if not x["near2"]) / max(len(first), 1), 4),
            "left_share_of_left_right": round(sum(1 for s in lr if s == 1) / max(len(lr), 1), 4),
            "axis_share": round(sum(1 for x in L if x["side"] == 0) / n, 4),
            "through_own_share": round(sum(x["through_own"] for x in L) / n, 4),
            "near2_at_i_share": round(sum(x["near2_at_i"] for x in L) / n, 4),
            "near2_share": round(sum(x["near2"] for x in L) / n, 4),
            "late_share": round(sum(x["late"] for x in L) / n, 4),
            "dc_label_share": round(sum(x["dc_label"] for x in L) / n, 4)}


def contact_sides(got, sel):
    S = set(sel)
    c = [x for x in got["contacts"] if x[0] in S]
    n = max(len(c), 1)
    return {"contacts": len(c), **{k: round(sum(1 for x in c if x[4] == v) / n, 4)
                                   for k, v in (("front", 1), ("flank", 2), ("rear", 3), ("unknown", 0))}}


def scenes_check(a, P, D):
    """`exam_pin scenes --check` of this slice (design section 4 checks 1, 2, 3, 5, 9, 10, 12, 13)."""
    from net.tactic import scenes as SN
    ck = scfg()["drills"]["attack"]["checks"]
    t0 = time.time()
    a1, n1 = scene_specs(a.n_scenes, int(scfg()["drills"]["attack"]["seeds"][a.seeds_key]))
    gen_s = round(time.time() - t0, 1)
    nvals = SN.check_bands(a1 + n1)
    specs = a1 + n1
    chunk = a.chunk
    seeds = [TSEED + 100 + k for k in range((len(specs) + chunk - 1) // chunk)]
    res, got = record_run("TEACH", specs, seeds, chunk, a, P, scene=True)
    (D / "scenes_TEACH.json").write_text(json.dumps({**res, "rec": got, "specs_meta": [
        {k: s.get(k) for k in ("recipe", "seed", "roles", "n1_plan", "axes", "map", "east", "plan_theirs")}
        for s in specs]}),
        encoding="utf-8")
    nA = len(a1)
    iA, iN = list(range(nA)), list(range(nA, len(specs)))
    has = {x["b"] for x in got["labels"]}
    fr = got["first"]
    bad = [b for b in range(len(specs)) if b not in fr or fr[b][0] > ck["frame_m"] or fr[b][1] > ck["frame_deg"]]
    out = {"generated": {"A1": nA, "N1": len(iN), "seconds": gen_s, "values_checked": nvals,
                         "n1_plans": {k: sum(1 for s in n1 if s["n1_plan"] == k) for k in ("point", "halt")}},
           "c1_men_max": max(men_of(s) for s in specs), "c1_men_rule": "asserted at start (TacticSide, step.men_max)",
           "c1_first_frame_worst": [round(max(fr[b][0] for b in fr), 3), round(max(fr[b][1] for b in fr), 3)],
           "c1_dropped_share": round(len(bad) / len(specs), 4),
           "c2_yield_A1": round(sum(1 for b in iA if b in has) / nA, 4),
           "c2_yield_N1_none": round(sum(1 for b in iN if b not in has) / len(iN), 4),
           "c2_N1_with_order_by_plan": {k: sum(1 for j, b in enumerate(iN) if b in has and n1[j]["n1_plan"] == k)
                                        for k in ("point", "halt")},
           "c2_N1_halt_by_gap": {f"{lo}-{hi}": [sum(1 for j, b in enumerate(iN) if n1[j]["n1_plan"] == "halt"
                                                       and lo <= n1[j]["plan_theirs"][0]["gap"] < hi and b in has),
                                                   sum(1 for j in range(len(iN)) if n1[j]["n1_plan"] == "halt"
                                                       and lo <= n1[j]["plan_theirs"][0]["gap"] < hi)]
                                 for lo, hi in ((20, 30), (30, 40), (40, 50), (50, 60.01))},
           "A1": {**label_marks(got, iA), "contacts": contact_sides(got, iA), "kinds": kinds_table(got, iA)},
           "N1": {**label_marks(got, iN), "contacts": contact_sides(got, iN), "kinds": kinds_table(got, iN)},
           "c9_slot_rank": {"A1": slot_rank(a1), "N1": slot_rank(n1)}, "c10": "asserted (scenes.check_bands)"}
    if a.no_g:
        (D / "check_scenes.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(json.dumps(out, indent=1), flush=True)
        return out
    # G: the pilot's 768 with shuffled slots - TEACH (recorded) and C (check 12)
    gsp, gseeds, gchunk = shuffled_pilot(a.n, range(a.seeds), a.limit)
    out["c9_slot_rank"]["G"] = slot_rank(gsp)
    rT, gT = record_run("TEACH", gsp, gseeds, gchunk, a, P)
    rC = run_player("C", gsp, gseeds, gchunk, a, P)
    (D / "g_TEACH.json").write_text(json.dumps({**rT, "rec": gT}), encoding="utf-8")
    (D / "g_C.json").write_text(json.dumps(rC), encoding="utf-8")
    iG = list(range(len(gsp)))
    out["G"] = {**label_marks(gT, iG), "contacts": contact_sides(gT, iG), "kinds": kinds_table(gT, iG)}
    d = paired(rT["per_battle"]["hp_trade"], rC["per_battle"]["hp_trade"])
    ref = (0.0646, 0.0077)
    e = math.sqrt(d["two_errors"] ** 2 + ref[1] ** 2)
    out["c12_teach_minus_c_shuffled"] = {**d, "unshuffled": list(ref), "difference": round(d["diff"] - ref[0], 4),
                                         "two_errors_of_difference": round(e, 4),
                                         "passed": abs(d["diff"] - ref[0]) <= e}
    out["minutes"] = round((time.time() - t0) / 60, 1)
    (D / "check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1), flush=True)
    return out


def scenes_vision(a, P, D):
    """Records for game.vision (game/vision/README.md: record.sim_frame every second -> record.from_sim ->
    vision.render): --vision A1:<seed>,N1:<seed>,G:<the shuffled pilot's battle>; each played alone on this device,
    TEACH driving; <D>/vision/<name>/ gets the strip, the text, orders.json (the order log) and spec.json."""
    from net.tactic import scenes as SN
    want = [x.split(":") for x in a.vision.split(",") if x]
    gsp = None
    for kind, ident in want:
        if kind == "G":
            if gsp is None:
                gsp = shuffled_pilot(a.n, range(a.seeds), a.limit)[0]
            sp, scene = gsp[int(ident)], False
            seeds = [TSEED + int(ident) // a.n]
        else:
            sp, scene = SN.make(kind, int(ident)), True
            seeds = [TSEED + 100]
        vision_record("TEACH", sp, scene, seeds, a, P, lambda: build_under("TEACH"), D / "vision" / f"{kind}_{ident}")


def vision_record(name, spec, scene, seeds, a, P, make, outd):
    """One battle (spec) played alone on a.device by the player make() (run_player's name), recorded every second for
    game.vision (record.sim_frame -> record.from_sim -> vision.render) into outd: the strip, the text, orders.json (the
    order log) and spec.json. Returns the player's file."""
    from game import vision
    from game.vision import record as RC
    frames = []

    class VisWatch(SceneWatch):
        def __call__(self, sec, st):
            super().__call__(sec, st)
            if not bool(st.done[0]):
                frames.append(RC.sim_frame(st, 0))
    kw = {"battles": BA.scene_battles, "enemy": BA.SceneEnemy} if scene else {}
    res = run_player(name, [spec], seeds, 1, a, P, make=make, watch_cls=VisWatch, **kw)
    outd = Path(outd)
    bt = RC.from_sim(frames, list(range(len(frames))), map_name=None, name=outd.name)
    r = vision.render(bt, outd)
    (outd / "orders.json").write_text(json.dumps({"cols": res["order_log_cols"], "rows": res["order_log"]}),
                                      encoding="utf-8")
    (outd / "spec.json").write_text(json.dumps(spec, default=str), encoding="utf-8")
    ci = {k: i for i, k in enumerate(res["order_log_cols"])}
    first = min((r_[ci["sec"]] for r_ in res["order_log"]), default=None)
    print(json.dumps({"vision": str(outd), "first_attack_order_s": first, "t_contact": res["per_battle"]["t_contact"],
                      **r.get("tokens", {})}), flush=True)
    return res


def _rate(xs):
    xs = list(xs)
    return round(sum(xs) / max(len(xs), 1), 4)


def mix_of(yields, cnt, cause, easy):
    """The recipe shares of a training round, set once from S's yields (design section 4): the cause recipes in S's
    proportions, the twins (N1, N2, A4L, A5L) as one block in S's proportions, its mass k such that the scenes without
    an order are half of all but E1-E3 - (sum_cause w nc + k sum_twin w nc) / (C + k T) = 0.5; infeasible when the
    twins' own no-cause rate is under one half."""
    nc = {r: (1.0 - y) if r in cause else y for r, y in yields.items() if r not in easy}
    tw = [r for r in nc if r not in cause]
    C, T = sum(cnt[r] for r in cause if r in nc), sum(cnt[r] for r in tw)
    a_c = sum(cnt[r] * nc[r] for r in cause if r in nc)
    a_t = sum(cnt[r] * nc[r] for r in tw) / max(T, 1)
    if a_t <= 0.5:
        return {"feasible": False, "twin_nocause_rate": round(a_t, 4), "cause_nocause_rate": round(a_c / max(C, 1), 4)}
    k = (0.5 * C - a_c) / ((a_t - 0.5) * T)
    mix = {r: cnt[r] for r in cause if r in nc}
    mix.update({r: k * cnt[r] for r in tw})
    sm = sum(mix.values())
    return {"feasible": True, "shares": {r: round(v / sm, 4) for r, v in mix.items()},
            "nocause_expected": round(sum(mix[r] * nc[r] for r in mix) / sm, 4)}


def set_check(a, P, D):
    """`exam_pin scenes --check --set S`: design section 4 checks 1-10, 13 on set S (one TEACH run, recorded), with
    review B / C / E / G / I; sets H, V and one training round's draw generated and band-checked, not played; check 12
    is slice 1's (r/check.json). Every recipe's kinds table and contact sides. <out dir>/check_S.json."""
    from net.tactic import scenes as SN
    ck = scfg()["drills"]["attack"]["checks"]
    quota = scfg()["drills"]["attack"]["quota"]
    t0 = time.time()
    only = [x for x in a.only.split(",") if x]
    specs = SN.set_specs("S", seed0=a.set_seed or None, counts={r: a.per for r in only} if only else None)
    gen_s = round(time.time() - t0, 1)
    nvals = SN.check_bands(specs)
    t1 = time.time()
    if only:
        H, V, R = [], [], []
        nH = nV = nR = 0
    else:
        H = SN.held_specs()
        nH = SN.check_bands(H)
        V = SN.set_specs("V")
        nV = SN.check_bands(V)
    rnd_n = int(scfg()["drills"]["attack"]["sets"]["round"]["scenes"])
    cnt = scfg()["drills"]["attack"]["sets"]["S"]["counts"]
    tot = sum(cnt.values())
    if not only:
        rcounts = {k: max(1, round(v * rnd_n / tot)) for k, v in cnt.items()}
        R = SN.set_specs("round", counts=rcounts)
        nR = SN.check_bands(R)
    gen2 = round(time.time() - t1, 1)
    chunk = a.chunk
    seeds = [TSEED + 200 + k for k in range((len(specs) + chunk - 1) // chunk)]
    res, got = record_run("TEACH", specs, seeds, chunk, a, P, scene=True)
    (D / "S_TEACH.json").write_text(json.dumps({**res, "rec": got, "specs_meta": [
        {k: x.get(k) for k in ("recipe", "seed", "roles", "n1_plan", "a6", "a2_mode", "twin_of", "end_s", "limit",
                               "axes", "map", "east")} for x in specs]}), encoding="utf-8")
    recs = list(dict.fromkeys(x["recipe"] for x in specs))
    by = {r: [i for i, x in enumerate(specs) if x["recipe"] == r] for r in recs}
    full = [x for x in got["labels"] if not x["dc_label"] and not x["dropped"]]
    has = {x["b"] for x in full}                       # (ruling 2: a cause scene holds a FULL-WEIGHT attack label)
    fr = got["first"]
    bad = [b for b in range(len(specs)) if b not in fr or fr[b][0] > ck["frame_m"] or fr[b][1] > ck["frame_deg"]]
    first = {}
    for x in sorted(full, key=lambda x: (x["sec"], x["slot"])):
        first.setdefault(x["b"], x)
    cause = ("A1", "A2", "A3", "A4", "A5", "A6")
    twins = ("N1", "N2", "A4L", "A5L")
    easy = ("E1", "E2", "E3")
    out = {"generated": {"S": len(specs), "seconds": gen_s, "values": nvals, "H": len(H), "H_values": nH, "V": len(V),
                         "V_values": nV, "round": len(R), "round_values": nR, "seconds_H_V_round": gen2},
           "c1_men_max": max(men_of(x) for x in specs), "c1_first_frame_worst":
               [round(max(fr[b][0] for b in fr), 3), round(max(fr[b][1] for b in fr), 3)],
           "c1_dropped_share": round(len(bad) / len(specs), 4), "c10": "asserted (S, H, V, round)"}
    out["c2_yield"] = {r: (_rate(b in has for b in by[r]) if r in cause else _rate(b not in has for b in by[r]))
                       for r in recs}
    out["c2_yield_kind"] = {r: ("with an order" if r in cause else "without an order") for r in recs}
    non_e = [b for r in recs if r not in easy for b in by[r]]
    out["c2_nocause_share"] = _rate(b not in has for b in non_e)
    out["c3_first_state_no"] = {r: _rate(not first[b]["near2"] for b in by[r] if b in first) for r in recs}
    out["c3_no_states_S"] = sum(1 for b in first if not first[b]["near2"])
    out["c3_yes_states_S"] = sum(1 for b in first if first[b]["near2"])
    if "N2" in by:
        bases = {}
        for b in by["N2"]:
            t = specs[b].get("twin_of", ("?",))[0]
            bases.setdefault(t, [0, 0])
            bases[t][0] += b not in has
            bases[t][1] += 1
        out["c2_N2_by_base_without_order"] = bases
    L = got["labels"]
    rec_of = {b: specs[b]["recipe"] for b in range(len(specs))}
    out["c4_late"] = {"S": _rate(x["late"] for x in L),
                      **{r: _rate(x["late"] for x in L if rec_of[x["b"]] == r) for r in ("A4", "A5")}}
    kt = kinds_table(got, list(range(len(specs))))
    out["c5_kinds_S"] = {k: kt["kinds_share_of_keeps"][k] for k in quota}
    out["c5_quota"] = quota
    multi = [x for x in L if len(specs[x["b"]]["theirs"]) >= 2]
    out["c6_pointer_not_nearest"] = _rate(not x["nearest_e"] for x in multi)
    out["c7_split_attack_labels"] = {"through_own": _rate(x["through_own"] for x in L),
                                     "threshold_band_key": _rate(x["dc_label"] and not x["through_own"] for x in L),
                                     "dropped_tick_band": _rate(x["dropped"] for x in L),
                                     "any": _rate(x["dropped"] or x["dc_label"] for x in L)}
    out["c7_split_keeps"] = {"band_key": kt["band_key_keep_share"], "dropped_tick_band":
                             kt["dropped_share_of_head_free"], "any_dc_key": kt["keep_with_dc_key_share"],
                             **{k: kt["kinds_share_of_keeps"][k] for k in ("n1", "n2w", "n3", "n4", "n5", "n6", "n9L")}}
    hk = [got["cnt"][b] for b in got["cnt"]]
    from net.tactic import tags as TG
    ci = {k: i for i, k in enumerate(TG.Record.COUNTS)}
    nk = sum(c_[ci["hp_keep_n"]] for c_ in hk)
    out["c8_health"] = {"labels_hp_target": _rate(x["hp_e"] for x in L), "labels_hp_pinning": _rate(x["hp_pin"] for x in L),
                        "taught_keeps_hp_key": round(sum(c_[ci["hp_e_keep"]] for c_ in hk) / max(nk, 1), 4),
                        "taught_keeps_hp_pinning": round(sum(c_[ci["hp_pin_keep"]] for c_ in hk) / max(nk, 1), 4),
                        "taught_keeps_n": nk}
    lr = [x["side"] for x in L if x["side"] in (1, 2)]
    out["c9"] = {"left_share": _rate(sd == 1 for sd in lr), "slot_rank": slot_rank(specs)}
    out["c12"] = "slice 1 (tmp/tactic_slice/v23/stage2c/r/check.json): TEACH - C shuffled +0.0610 +- 0.0082, passed"
    out["c13"] = {"through_own": _rate(x["through_own"] for x in L), "near2_at_i": _rate(x["near2_at_i"] for x in L)}
    out["by_recipe"] = {r: {**label_marks(got, by[r]), "contacts": contact_sides(got, by[r]),
                            "kinds": kinds_table(got, by[r]), "scenes": len(by[r])} for r in recs}
    # the recipe shares of a training round, set once from the yields: the cause recipes and the late twins in S's
    # proportions, the N1 / N2 block scaled so that the no-cause scenes are half of all but E1-E3
    if not only:
        out["mix"] = mix_of(out["c2_yield"], cnt, cause, easy)
    out["minutes"] = round((time.time() - t0) / 60, 1)
    (D / ("check_S.json" if not only else "check_" + "_".join(only) + ".json")).write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1), flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("players", "table", "labeller", "scenes", "exam"))
    ap.add_argument("--check", action="store_true", help="scenes: the checks of stage 2c's slice")
    ap.add_argument("--vision", default="", help="scenes: A1:<seed>,N1:<seed>,G:<battle> records for game.vision")
    ap.add_argument("--n-scenes", type=int, default=256)
    ap.add_argument("--seeds-key", default="a1", help="scenes: the seeds of config drills.attack.seeds")
    ap.add_argument("--scene-set", default="", help="scenes --check: S - the whole generator on set S (stage 2c)")
    ap.add_argument("--only", default="", help="scenes --check --scene-set S: these recipes only, --per each (mending)")
    ap.add_argument("--per", type=int, default=128)
    ap.add_argument("--set-seed", type=int, default=0, help="scenes --check --scene-set S: another seed (mending draws)")
    ap.add_argument("--no-g", action="store_true", help="scenes --check: the scenes only (no shuffled pilot battles)")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--set", default="pilot", choices=("pilot", "G", "S", "F"), help="players: pilot; exam: G, S or F")
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
    elif a.what == "scenes":
        D = out.parent
        if a.check:
            (set_check if a.scene_set == "S" else scenes_check)(a, P, D)
        if a.vision:
            scenes_vision(a, P, D)
        return
    elif a.what == "exam":
        if a.players == "K,C":
            a.players = "C,TEACH,TEACH:taught"
        exam(a, P, out.parent)
        return
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
