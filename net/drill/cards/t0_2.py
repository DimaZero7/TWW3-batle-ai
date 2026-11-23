"""Card T0-2 'сторона удара' (side of the strike): the approach stage pin.choose_sides over T0-1's target stage
(tmp/engine/design_merged.md section 3). Its exam's strata, execution read and pieces live here (moved from
net/tactic/exam_side.py and imitate.py, which go after (e)); the engine's exam / verdict run them.

    BEFORE, AFTER     the exam tables' views (players, pairs, anchor C, strata groups_of, comer, exec_of, binds_of):
                      before = the teacher's exam (C, TEACH, TEACH2[, TEACH2_ON]), after = + N, N1
    lines_before(out) the exam before's lines b, c, d, g, e (exam_side.lines, written before the run)
    pieces_after(out, t01)   the legacy verdict's six pieces i-vi (imitate.side_lines, written before T0-2's training)
    t01_compare(new, ref)    piece v's read (imitate.t01_compare)
    value_gate_of(out)   the value gate decided on the teacher's exam before the training (verdict before)
    m2_pieces(out, gate, probe, t01)   the M2 pieces (design section 6, brief_A2 Part 6), each {id, kind, status, ...}:
                      binds (every held axis), orders, no harm (G, no-cause; N - N1 on G), the stored value gate ->
                      'half the gap' (paired per battle) or 'not worse than the teacher', the held gap, T0-1's
                      retention, skill rate, execution on its declared denominator, the probe (a gate); every gate at
                      GATE_WINDOW, the other windows printed
    card()            the card's data for the run (cards/__init__.py)
"""
import json
import math
from pathlib import Path
from types import SimpleNamespace

from net.drill import exam as EX

WIDE_M = 37.0                                            # (net/tactic/exam_side.py WIDE_M)
SIDE_NAMES = {0: "struck_first", 1: "front", 2: "flank", 3: "rear"}
KEPT = ("arrived", "timeout", "attacking", None)
X = " [-comer]"
N_MIN = 100
_N_MIN_WHY = ("a stratum's paired battles below this: NOT JUDGED (M2 BLOCKED) - the smallest T0-2 stratum the legacy "
              "verdict judged was held cause [-comer] 179 (tmp/t02/after_side), a half of it still gives two errors "
              "~0.005 on its spread")
NOHARM_LB = -0.005                                        # (design section 6: paired lower bound >= -0.005)
ORDERS_MAX = 1.2                                          # (design section 6: 1.5 x the human's 0.8)
LINE_C = 0.75                                             # (exam_side line c)
VALUE_GATE = (4.0, 0.02)                                  # (design section 3 Card.value_gate: errors, absolute)


# ------------------------------------------------------------------ strata and execution (from exam_side)
def _shares(rows, n=None):
    n = len(rows) if n is None else n
    c = {}
    for r in rows:
        k = "none" if r[7] is None else SIDE_NAMES.get(int(r[7]), "none")
        c[k] = c.get(k, 0) + 1
    return {k: round(v / n, 4) for k, v in sorted(c.items())} if n else {}


def exec_of(d, sel=None):
    """The execution of a player's attack orders over the battles sel (exam_side._flank_exec)."""
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


def groups_of(mt, nm, comer):
    g = _groups(mt, nm)
    out = list(g) + [(f"{lab}{X}", [b for b in sel if b not in comer]) for lab, sel in g]
    cz = sorted(comer)
    out += [("comer", cz)]
    if nm != "G":
        out += [("comer, cause", [b for b in cz if mt[b]["cause"]]), ("comer, no cause", [b for b in cz if not mt[b]["cause"]])]
    return [(k, v) for k, v in out if v]


def comer_before(mt, pl):
    """The battles where an enemy comes at the hammer after the order (exam_side.comer_scenes)."""
    gen = {b for b, x in enumerate(mt) if x.get("blocker") == "comer"}
    seen = {int(r[0]) for p in ("TEACH2", "TEACH2_ON") if p in pl
            for r in pl[p].get("strikes", []) if r[4] > 0 and len(r) > 9 and r[9]}
    return {"generator": sorted(gen), "seen": sorted(seen), "all": sorted(gen | seen)}


def comer_after(mt, pl):
    """comer_before + the network's flank walks that met a comer (imitate.side_comer)."""
    cm = comer_before(mt, pl)
    seen = {int(r[0]) for p in ("N",) if p in pl for r in pl[p].get("strikes", []) if r[4] > 0 and len(r) > 9 and r[9]}
    return {**cm, "seen_N": sorted(seen), "all": sorted(set(cm["all"]) | seen)}


def binds_of(mt):
    out = {}
    for x in mt:
        if x.get("held") is None or x.get("binds") is None:
            continue
        r = out.setdefault(str(x["held"]), {"scenes": 0, "binds": 0})
        r["scenes"] += 1
        r["binds"] += int(bool(x["binds"]))
    return out


BEFORE = SimpleNamespace(PLAYERS=("C", "TEACH", "TEACH2", "TEACH2_ON"), ANCHOR="C",
                         PAIRS=(("TEACH2", "TEACH"), ("TEACH2", "C"), ("TEACH", "C"), ("TEACH2_ON", "TEACH"),
                                ("TEACH2_ON", "TEACH2")),
                         groups_of=groups_of, comer=comer_before, exec_of=exec_of, binds_of=binds_of)
AFTER = SimpleNamespace(PLAYERS=("C", "TEACH", "TEACH2", "N", "N1"), ANCHOR="C",
                        PAIRS=(("N", "TEACH"), ("N", "TEACH2"), ("N", "C"), ("N", "N1"), ("N1", "TEACH"),
                               ("TEACH2", "TEACH"), ("TEACH2", "C"), ("TEACH", "C")),
                        groups_of=groups_of, comer=comer_after, exec_of=exec_of, binds_of=binds_of)


def _get(out, nm, lab, read="i_300", pair="TEACH2 - TEACH"):
    return out.get(nm, {}).get("pairs", {}).get(pair, {}).get(lab, {}).get(read)


def line_c(out, player="TEACH2"):
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
            "player": player, "flank": fl, "n": nk, "share": q(fl, nk), "passed": bool(nk and fl / nk >= LINE_C),
            "old_read_with_contact": {"flank": fl, "n": kc, "share": q(fl, kc)},
            "all_flank_orders_with_fallbacks": {"flank": fa, "n": nf, "share": q(fa, nf)},
            "with_comer": {"flank": sum(e["kept_flank"] for e in ex_of("all")),
                           "n": sum(e["not_fallen_back"] for e in ex_of("all"))},
            "with_replaced": {"flank": fl, "n": nk + rp, "share": q(fl, nk + rp), "replaced": rp},
            "with_replaced_upper": {"flank": fa, "n": nk + rp, "share": q(fa, nk + rp)}}


# ------------------------------------------------------------------ the exam before (lines b-g)
def lines_before(out):
    pool = EX.pool
    res = {}
    cause = pool([_get(out, "taught", "cause" + X), _get(out, "held", "cause" + X)])
    nocause = pool([_get(out, "taught", "no cause" + X), _get(out, "held", "no cause" + X)])
    g = _get(out, "G", "all" + X)
    res["b"] = {"rule": "TEACH2 - TEACH > 2 errors on S2 cause scenes (pooled); >= -2 errors on no-cause scenes and on G;"
                        " the comer battles out (comer_scenes)",
                "cause": cause, "no_cause": nocause, "G": g,
                "passed": bool(cause and cause["diff"] > cause["two_errors"]
                               and all(x is None or x["diff"] >= -x["two_errors"] for x in (nocause, g))),
                "with_comer": {"cause": pool([_get(out, "taught", "cause"), _get(out, "held", "cause")]),
                               "no_cause": pool([_get(out, "taught", "no cause"), _get(out, "held", "no cause")]),
                               "G": _get(out, "G", "all")},
                "comer_alone": {nm: _get(out, nm, "comer") for nm in ("taught", "held", "G")},
                "TEACH2_ON - TEACH": {"cause": pool([_get(out, "taught", "cause", pair="TEACH2_ON - TEACH"),
                                                     _get(out, "held", "cause", pair="TEACH2_ON - TEACH")]),
                                      "comer": {nm: _get(out, nm, "comer", pair="TEACH2_ON - TEACH")
                                                for nm in ("taught", "held", "G")},
                                      "G": _get(out, "G", "all", pair="TEACH2_ON - TEACH")}}
    res["c"] = line_c(out)
    om = {nm: out[nm]["orders_min"].get("TEACH2", {}).get("all" + X) for nm in ("taught", "held", "G") if nm in out}
    res["d"] = {"rule": "TEACH2's orders to the game a unit-minute <= 1.2, the comer battles out", "by_set": om,
                "with_comer": {nm: out[nm]["orders_min"].get("TEACH2", {}).get("all") for nm in ("taught", "held", "G")
                               if nm in out},
                "passed": bool(om) and all(v is not None and v <= ORDERS_MAX for v in om.values())}
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


# ------------------------------------------------------------------ the exam after (legacy pieces i-vi)
def _piece(passed, **x):
    return {**x, "passed": passed, "status": {True: "passed", False: "failed", None: "not judged"}[passed]}


PIECES_AFTER = ("i_closes", "ii_executes", "iii_no_harm", "iv_held", "v_t01", "vi_orders")
SKILL = ("i_closes", "ii_executes")                    # (the skill pieces NEG must fail)


def pieces_after(out, t01=None):
    pool_all = EX.pool_all
    g = lambda nm, lab, pair: _get(out, nm, lab, pair=pair)                         # noqa: E731
    res = {}
    nt = pool_all([g("taught", "cause" + X, "N - TEACH"), g("held", "cause" + X, "N - TEACH")])
    tt = pool_all([g("taught", "cause" + X, "TEACH2 - TEACH"), g("held", "cause" + X, "TEACH2 - TEACH")])
    ok = None if not (nt and tt and tt["diff"] > 0) else nt["diff"] >= 0.5 * tt["diff"]
    res["i_closes"] = _piece(ok, rule="N - TEACH >= 0.5 x (TEACH2 - TEACH) on the S2 cause scenes (taught + held pooled, "
                                      "comer out); TEACH2 - TEACH <= 0 or missing: not judged",
                             N_TEACH=nt, TEACH2_TEACH=tt,
                             share_closed=round(nt["diff"] / tt["diff"], 3) if nt and tt and tt["diff"] else None)

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
    cN = line_c(out, "N") if all(nm in out for nm in ("taught", "held", "G")) else None
    a_ok = None if not (fN and fT and fT["a_scene"] > 0) else fN["a_scene"] >= 0.5 * fT["a_scene"]
    c_ok = None if not cN or cN.get("share") is None else cN["share"] >= LINE_C
    ok = False if (a_ok is False or c_ok is False) else (None if (a_ok is None or c_ok is None) else True)
    res["ii_executes"] = _piece(ok, rule="N's flank orders a cause scene (comer out) >= 0.5 x TEACH2's, and N's first "
                                         "contact on the flank (exam_side line c) >= 0.75",
                                flank_orders_N=fN, flank_orders_TEACH2=fT,
                                orders_ok={True: "passed", False: "failed", None: "not judged"}[a_ok],
                                line_c_N=cN, line_c_ok={True: "passed", False: "failed", None: "not judged"}[c_ok])
    nn = pool_all([g("taught", "no cause" + X, "N - TEACH"), g("held", "no cause" + X, "N - TEACH")])
    gt = g("G", "all" + X, "N - TEACH")
    ok = None if not (nn and gt) else (nn["diff"] >= -nn["two_errors"] and gt["diff"] >= -gt["two_errors"])
    res["iii_no_harm"] = _piece(ok, rule="N - TEACH >= -2 errors on the no-cause scenes (taught + held pooled) and on G "
                                         "(comer out)", no_cause=nn, G=gt,
                                G_N_N1=g("G", "all" + X, "N - N1"), G_N_TEACH2=g("G", "all" + X, "N - TEACH2"))
    t, h = g("taught", "cause" + X, "N - TEACH"), g("held", "cause" + X, "N - TEACH")
    if t and h:
        e2 = round(math.sqrt(t["two_errors"] ** 2 + h["two_errors"] ** 2), 4)
        res["iv_held"] = _piece(h["diff"] - t["diff"] >= -e2, rule="(N - TEACH) held cause - taught cause >= -2 errors",
                                taught=t, held=h, gap=round(h["diff"] - t["diff"], 4), two_errors=e2,
                                by_axis={lab: x for lab, x in out["held"]["pairs"].get("N - TEACH", {}).items()
                                         if lab.startswith("held ") and lab.endswith(", cause" + X)})
    else:
        res["iv_held"] = _piece(None, rule="(N - TEACH) held cause - taught cause >= -2 errors", taught=t, held=h)
    if t01 is None:
        res["v_t01"] = _piece(None, rule="T0-1 retained (t01_compare)", note="--t01-new / --t01-ref not given")
    else:
        # (the legacy line as before: the M2 windows of t01_compare - windows, anchor, passed_at - left out)
        t1 = {**t01, "sets": {s: {k: v for k, v in x.items() if k not in ("windows", "anchor", "passed_at", "marks_gate")}
                              for s, x in (t01.get("sets") or {}).items()}} if "sets" in t01 else t01
        res["v_t01"] = _piece(t01.get("passed"), rule="T0-1 retained (t01_compare)", **{k: v for k, v in t1.items()
                                                                                       if k != "passed"})
    om = {nm: out.get(nm, {}).get("orders_min", {}).get("N", {}).get("all" + X) for nm in ("taught", "held", "G")}
    known = all(isinstance(v, (int, float)) and math.isfinite(v) for v in om.values())
    res["vi_orders"] = _piece(None if not known else all(v <= ORDERS_MAX for v in om.values()),
                              rule="N's orders to the game a unit-minute <= 1.2 (METHOD: 1.5 x human 0.8) on taught, held, "
                                   "G (comer out)", by_set=om)
    res["comer_alone"] = {nm: g(nm, "comer", "N - TEACH") for nm in ("taught", "held", "G")}
    res["passed"] = all(res[p]["passed"] is True for p in PIECES_AFTER)
    res["verdict"] = "passed" if res["passed"] else "NOT passed"
    res["reasons"] = [f"{p}: {res[p]['status']}" for p in PIECES_AFTER if res[p]["passed"] is not True]
    return res


_T01_ANCHOR_WHY = ("T0-1's exam files carry no C arm on G, so the + 300 / + 450 s windows are anchored on the reference "
                   "network's own first contact: one anchor a battle for both arms, independent of the new network; "
                   "'passed' stays the end read the legacy piece v reads, 'passed_at' feeds M2")


def t01_compare(new_dir, ref_dir):
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
        d = EX.paired(hp(x), hp(y))
        anc = y["per_battle"]["t_contact"]
        win = {str(w): EX.paired(EX.at(x, anc, w)[0], EX.at(y, anc, w)[0]) for w in (300, GATE_WINDOW)}
        win["end"] = d
        r = {"N3 - N1": d, "windows": win, "anchor": "the reference N1's first contact (" + _T01_ANCHOR_WHY + ")",
             "passed_at": {w: None if win[w] is None else win[w]["diff"] >= -win[w]["two_errors"]
                           for w in (str(GATE_WINDOW), "end")},
             "N3 - C": (a_.get("stated", {}).get(s) or {}).get("N - C"),
             "N1 - C": (b_.get("stated", {}).get(s) or {}).get("N - C"),
             "passed": None if d is None else d["diff"] >= -d["two_errors"]}
        if s in a_ and s in b_:
            r["marks"] = {"followed_N": [a_[s]["followed_N"]["pooled"]["share"], b_[s]["followed_N"]["pooled"]["share"]],
                          "false_tick": [a_[s]["false_N"]["tick"]["share"], b_[s]["false_N"]["tick"]["share"]]}
            # (brief_A3 V2: the follow and false rates GATE against the parent - not lower / not higher beyond two
            # errors of both, each at n >= cards.MARK_MIN_N)
            from net.drill.cards import MARK_MIN_N
            from net.drill.validate import compare
            mg = {}
            slim = lambda c: {k: v for k, v in c.items() if k != "bb"}                  # noqa: E731
            for nm, k1, k2, op in (("followed_N", "followed_N", "pooled", ">="), ("false_tick", "false_N", "tick", "<=")):
                c, c0 = a_[s][k1][k2], b_[s][k1][k2]
                if min(int(c.get("n") or 0), int(c0.get("n") or 0)) < MARK_MIN_N or c.get("share") is None:
                    mg[nm] = {"passed": None, "new": slim(c), "parent": slim(c0)}
                    continue
                # (Astra engine_code3 finding 10: judged by battle with the stop's own function - both cells carry
                # their battle counts (imitate.after writes them), the same roster, MIN_BATTLES each; else NOT JUDGED)
                if not (c.get("bb") and c0.get("bb")):
                    mg[nm] = {"passed": None, "new": slim(c), "parent": slim(c0),
                              "note": "no battle counts in after.json (a legacy exam): replay it"}
                    continue
                r_ = compare(op, c, c0)
                mg[nm] = {"passed": None if r_["unjudged"] else not r_["worse"], "new": slim(c), "parent": slim(c0),
                          "two_errors": None if r_["e"] is None else round(r_["e"], 6), "op": op, "how": r_["how"],
                          "note": r_["unjudged"]}
            r["marks_gate"] = mg
        res["sets"][s] = r
    st = [x["passed"] for x in res["sets"].values()]
    res["passed"] = False if False in st else (None if None in st else True)
    return res


# ------------------------------------------------------------------ M2 (design section 6; brief_A2 Part 6)
GATE_WINDOW = 450
_GATE_WINDOW_WHY = ("the declared window of every M2 gate: C's first contact + 450 s (design section 0: the flank "
                    "trick's cost opens at + 180..360 s, the round-1 M2 decided its value gate there); + 300 s and the "
                    "battle's end are printed beside, never gated")
EXEC_DENOMINATOR = "without_replaced"
_EXEC_WHY = ("execution's gate: first contact on the flank / the flank orders that did not fall back, the replaced "
             "walks and the fallbacks out - the denominator the teacher's exam read (exam_side line c); 'with replaced' "
             "and its upper estimate are printed beside as a known limit")
COMBOS = {"N - (TEACH + TEACH2) / 2": {"N": 1.0, "TEACH": -0.5, "TEACH2": -0.5}}
_COMBOS_WHY = "half the gap paired per battle: (N - TEACH) - 0.5 (TEACH2 - TEACH) >= 0, its own two errors"


def held_axes():
    from net.tactic import scenes as SN
    return tuple(SN.cs()["sets"]["held"]["axes"])


def groups_of_m2(mt, nm, comer):
    """groups_of with the held set's unbound battles out (a held battle scores only where its axis binds)."""
    if nm == "held":
        keep = [b for b, x in enumerate(mt) if x.get("binds") is True]
        sub = [mt[b] for b in keep]
        cz = {keep.index(b) for b in comer if b in keep}
        return [(lab, [keep[i] for i in sel]) for lab, sel in groups_of(sub, nm, cz)]
    return groups_of(mt, nm, comer)


AFTER_M2 = SimpleNamespace(**{**vars(AFTER), "groups_of": groups_of_m2, "COMBOS": COMBOS})


def _m2(pid, kind, status, **x):
    return {"id": pid, "kind": kind, "status": status, **x}


def _lb(p):
    return None if p is None else round(p["diff"] - p["two_errors"], 4)


def _small(*ps):
    return any(p is None or p["n"] < N_MIN for p in ps)


SKILL_WINDOWS = (GATE_WINDOW,)
SAFETY_WINDOWS = (GATE_WINDOW, "end")
_SAFETY_WINDOWS_WHY = ("the coordinator's decision 22.11 (T0-2's lesson, journal row 50bc02b: every piece passed at "
                       "+ 300 s and the loss showed only at the battle's end): a skill piece gates at + 450 s; a safety "
                       "piece (no harm, retention) must pass at + 450 s AND at the end, each window its own line, "
                       "+ 300 s printed beside, never gated")


def _gate_at(ps, line, lb_of):
    """(status, double?) of one window's parts: each part's n >= N_MIN BEFORE pooling (else NOT JUDGED); the interval
    straddling the line -> 'double' (n < 1,024) / 'NOT DECIDED'."""
    p = EX.pool_all(ps)
    if _small(*ps):
        return "NOT JUDGED", []
    if not EX.straddles(p["diff"], p["two_errors"], line):
        return ("pass" if lb_of(p) >= line else "fail"), []
    # (brief_A3 V3: 'at the cap' per set - a straddling read doubles every part below the cap, not the pooled n)
    low = [i for i, x in enumerate(ps) if x["n"] < EX.CAP]
    return ("double" if low else "NOT DECIDED"), low


def _brief(p, lb_of=_lb):
    return None if p is None else {"diff": p["diff"], "two_errors": p["two_errors"], "n": p["n"], "lower_bound": lb_of(p)}


def _gated(pid, kind, parts, line, lb_of=_lb, **more):
    """One M2 line a gated window (kind 'skill': SKILL_WINDOWS, 'safety': SAFETY_WINDOWS), id '<pid>@<window>'; the
    windows not gated (+ 300 s) printed beside on every line."""
    beside = {str(w): _brief(EX.pool_all(ps), lb_of) for w, ps in parts.items()
              if w not in (SAFETY_WINDOWS if kind == "safety" else SKILL_WINDOWS)}
    out = []
    for w in (SAFETY_WINDOWS if kind == "safety" else SKILL_WINDOWS):
        ps = parts[w]
        st, low = _gate_at(ps, line, lb_of)
        sets = more.get("cells") or [(s, None) for s in (more.get("sets") or [])]
        dbl = sorted({sets[i][0] for i in low if i < len(sets)}) if low else []
        out.append(_m2(f"{pid}@{w}", kind, st, line=line, window=w, read=_brief(EX.pool_all(ps), lb_of),
                       parts_n=[None if x is None else x["n"] for x in ps], beside=beside,
                       **({"double": dbl} if dbl else {}),
                       **{k: v for k, v in more.items() if k not in ("sets", "cells")}))
    return out


def value_gate_of(out, parent_teacher="TEACH", teacher="TEACH2"):
    """The value gate decided on the teacher's exam BEFORE the training (stored as value_gate.json beside its files):
    teacher - parent teacher on the cause scenes (comer out, taught + held pooled) at GATE_WINDOW >= VALUE_GATE[0]
    standard errors and >= VALUE_GATE[1] -> 'half_the_gap', else 'not_worse_than_teacher'; NOT JUDGED when small."""
    tp = f"{teacher} - {parent_teacher}"
    ps = [_get(out, nm, "cause" + X, read=f"i_{GATE_WINDOW}", pair=tp) for nm in ("taught", "held")]
    g = EX.pool_all(ps)
    base = {"card": "T0-2", "window": GATE_WINDOW, "pair": tp, "read": g, "rule": f"{tp} >= {VALUE_GATE[0]} standard "
            f"errors and >= {VALUE_GATE[1]} -> half_the_gap, else not_worse_than_teacher"}
    if _small(*ps):
        return {**base, "kind": "NOT JUDGED"}
    ok = g["diff"] >= VALUE_GATE[0] * g["two_errors"] / 2 and g["diff"] >= VALUE_GATE[1]
    return {**base, "kind": "half_the_gap" if ok else "not_worse_than_teacher"}


def _windows():
    return (300, GATE_WINDOW, "end")


def m2_pieces(out, arm="N", parent_teacher="TEACH", teacher="TEACH2", gate=None, probe=None, t01=None):
    """The M2 pieces of T0-2's exam after (module doc); a missing read, set or axis, or a part's n < N_MIN: NOT
    JUDGED; an undecided interval: double / NOT DECIDED."""
    res = []

    def rd(nm, lab, pair, w):
        return _get(out, nm, lab, read="iii_end" if w == "end" else f"i_{w}", pair=pair)

    def parts(pair, cells):
        return {w: [rd(s, lab, pair, w) for s, lab in cells] for w in _windows()}
    sets_of = lambda cells: sorted({s for s, _ in cells})                              # noqa: E731
    # binds: every declared held axis present, each bound in >= 90 % of its scenes (binds.MAX_UNBOUND)
    from net.drill.binds import MAX_UNBOUND
    bd = out.get("held", {}).get("binds") or {}
    miss = [a for a in held_axes() if a not in bd or not bd[a]["scenes"]]
    if miss:
        res.append(_m2("binds", "safety", "NOT JUDGED", missing_axes=miss, by_axis=bd))
    else:
        bad = {a: v for a, v in bd.items() if (v["scenes"] - v["binds"]) / max(v["scenes"], 1) > MAX_UNBOUND}
        res.append(_m2("binds", "safety", "fail" if bad else "pass", by_axis=bd, over=bad,
                       note="the unbound held battles are out of every M2 read (groups_of_m2)"))
    # orders a unit-minute
    om = {nm: out.get(nm, {}).get("orders_min", {}).get(arm, {}).get("all" + X) for nm in ("taught", "held", "G")}
    known = all(isinstance(v, (int, float)) and math.isfinite(v) for v in om.values())
    res.append(_m2("orders", "safety", "NOT JUDGED" if not known else
                   ("pass" if all(v <= ORDERS_MAX for v in om.values()) else "fail"), by_set=om, line=ORDERS_MAX))
    # no harm on G and no-cause against the parent teacher, and (a) against the parent network N1 on G
    G = [("G", "all" + X)]
    NC = [("taught", "no cause" + X), ("held", "no cause" + X)]
    CZ = [("taught", "cause" + X), ("held", "cause" + X)]
    pr = f"{arm} - {parent_teacher}"
    res.extend(_gated("no_harm_G", "safety", parts(pr, G), NOHARM_LB, pair=pr, sets=sets_of(G), cells=G))
    res.extend(_gated("no_harm_nocause", "safety", parts(pr, NC), NOHARM_LB, pair=pr, sets=sets_of(NC), cells=NC))
    res.extend(_gated("no_harm_vs_parent_G", "safety", parts(f"{arm} - N1", G), NOHARM_LB, pair=f"{arm} - N1",
                      sets=sets_of(G), cells=G, note="the stack with the skill not worse than the stack without it (METHOD 6)"))
    # the value gate, decided on the teacher's exam before the training (value_gate.json)
    kind = (gate or {}).get("kind")
    if kind not in ("half_the_gap", "not_worse_than_teacher"):
        res.append(_m2("value_gate", "safety", "NOT JUDGED", gate=gate,
                       note="no value gate decided before the training (verdict before --dir <teacher's exam>), or "
                            f"an unknown kind {kind!r}"))
    else:
        res.append(_m2("value_gate", "report", "reported", gate=gate))
        if kind == "half_the_gap":
            cn = "N - (TEACH + TEACH2) / 2"
            cp = {w: [(out.get(s, {}).get("combos", {}).get(cn, {}).get(lab) or {}).get(
                "iii_end" if w == "end" else f"i_{w}") for s, lab in CZ] for w in _windows()}
            res.extend(_gated("half_the_gap", "skill", cp, 0.0, pair=cn, sets=sets_of(CZ), cells=CZ,
                              rule=f"{cn} lower bound >= 0 on the cause scenes (paired per battle)"))
        else:
            res.extend(_gated("safety_vs_teacher", "safety", parts(f"{arm} - {teacher}", CZ), NOHARM_LB,
                              pair=f"{arm} - {teacher}", sets=sets_of(CZ), cells=CZ))
    # (b) the taught / held gap within noise
    def gap_of(w):
        t, h = rd("taught", "cause" + X, pr, w), rd("held", "cause" + X, pr, w)
        if _small(t, h):
            return None, {"taught": _brief(t), "held": _brief(h)}
        e2 = round(math.sqrt(t["two_errors"] ** 2 + h["two_errors"] ** 2), 4)
        return round(h["diff"] - t["diff"], 4), {"taught": _brief(t), "held": _brief(h), "two_errors": e2}
    g300 = gap_of(300)
    for w in SAFETY_WINDOWS:
        gap, x = gap_of(w)
        res.append(_m2(f"held_gap@{w}", "safety", "NOT JUDGED" if gap is None else (
            "pass" if gap >= -x["two_errors"] else "fail"), window=w, gap=gap, **x,
            beside={"300": {"gap": g300[0], **g300[1]}}, rule=f"({pr}) held cause - taught cause >= -2 errors"))
    # (c) retention of T0-1 against the parent on the same battles (t01_compare: S and G, N3 - N1 >= -2 errors)
    sets_ = (t01 or {}).get("sets") or {}
    for w in SAFETY_WINDOWS:
        st_ = [(v.get("passed_at") or {}).get(str(w)) for v in sets_.values()]
        if t01 is None or t01.get("passed") is None or not st_ or None in st_:
            res.append(_m2(f"retention_t0_1@{w}", "safety", "NOT JUDGED", window=w, t01=t01,
                           note="--t01-new / --t01-ref not given, not comparable, or no read at this window"))
            continue
        res.append(_m2(f"retention_t0_1@{w}", "safety", "fail" if False in st_ else "pass", window=w,
                       sets={k: _brief((v.get("windows") or {}).get(str(w))) for k, v in sets_.items()},
                       beside={"300": {k: _brief((v.get("windows") or {}).get("300")) for k, v in sets_.items()}},
                       rule="N3 - N1 >= -2 errors on S and on G (t01_compare)"))
    # retention's marks: T0-1's follow and false rates against the parent on S and G (gated, brief_A3 V2)
    mk = {k: (v.get("marks_gate") or {}) for k, v in sets_.items()}
    st_ = [x.get("passed") for g in mk.values() for x in g.values()] if mk else []
    res.append(_m2("retention_marks_t0_1", "safety", "NOT JUDGED" if (t01 is None or not st_ or None in st_ or
                   len(st_) < 2 * len(mk)) else ("fail" if False in st_ else "pass"), sets=mk,
                   rule="T0-1 followed (N, pooled) not lower and false a tick not higher than the parent's beyond two "
                        "errors, on S and on G"))

    # the skill used: flank orders a cause scene (comer out) >= half the teacher's; an arm without strikes in its
    # files: NOT JUDGED (never zero orders)
    def rate(p):
        fo, nb = 0, 0
        for nm in ("taught", "held"):
            e = out.get(nm, {}).get("exec", {}).get(p, {}).get("cause" + X)
            n = out.get(nm, {}).get("n_battles", {}).get("cause" + X)
            if not n or e is None or n < N_MIN:          # (each set's cause scenes >= N_MIN BEFORE pooling, V3)
                return None
            fo, nb = fo + e["flank_orders"], nb + n
        return {"flank_orders": fo, "cause_scenes": nb, "a_scene": round(fo / nb, 4)}
    rN, rT = rate(arm), rate(teacher)
    st = "NOT JUDGED" if not (rN and rT) or rT["cause_scenes"] < N_MIN or rT["a_scene"] <= 0 else (
        "pass" if rN["a_scene"] >= 0.5 * rT["a_scene"] else "fail")
    res.append(_m2("skill_rate", "skill", st, arm_rate=rN, teacher_rate=rT, rule="flank orders a cause scene >= 0.5 x "
                   "the teacher's (comer out)"))
    # (d) execution: the declared denominator gates, the other printed beside
    have = all(nm in out and out[nm].get("exec", {}).get(arm, {}).get("all" + X) is not None
               for nm in ("taught", "held", "G"))
    if have:
        c = line_c(out, arm)
        a_ = c["share"]
        # (brief_A3 V3: each set's denominator >= N_MIN BEFORE pooling; the per-set n printed)
        per = {nm: out[nm]["exec"][arm]["all" + X]["not_fallen_back"] for nm in ("taught", "held", "G")}
        st = "fail" if c["n"] == 0 and c["with_replaced"]["n"] == 0 else (
            "NOT JUDGED" if a_ is None or min(per.values()) < N_MIN else ("pass" if a_ >= LINE_C else "fail"))
        res.append(_m2("execution", "skill", st, denominator=EXEC_DENOMINATOR, share=a_, line=LINE_C, n=c["n"], n_set=per,
                       known_limit={"with_replaced": c["with_replaced"]["share"],
                                    "with_replaced_upper": c["with_replaced_upper"]["share"],
                                    "replaced": c["with_replaced"]["replaced"]}))
    else:
        res.append(_m2("execution", "skill", "NOT JUDGED", note="an arm's strikes missing in a set"))
    # the label probe: a gate (METHOD step 4) - its result file {verdict: {pass, n_ok}}
    res.append(probe_piece(probe))
    return res


def probe_piece(path):
    f = Path(path) if path else None
    if f is None or not f.is_file():
        return _m2("probe", "safety", "NOT JUDGED", file=str(f), note="no probe result file")
    x = json.loads(f.read_text(encoding="utf-8")) or {}
    if "verdict" not in x and "passed" in x:
        # (brief_A3 V6: the legacy `net.tactic.learn probe --side` output - exam H's starts, its own masks - refused)
        return _m2("probe", "safety", "NOT JUDGED", file=str(f), legacy_passed=x.get("passed"),
                   note="a legacy probe file (top-level 'passed': net.tactic.learn probe --side - exam starts, not "
                        "masks.of): refused; run python -m net.drill.levels.tactic_probe --card t0_2")
    v = x.get("verdict") or {}
    if v.get("n_ok") and v.get("pass") is True:
        return _m2("probe", "safety", "pass", file=str(f), verdict=v)
    # (METHOD step 4: a probe not passed means the drill is to be fixed, not the network judged - BLOCKED, not FAILED)
    return _m2("probe", "safety", "NOT JUDGED", file=str(f), verdict=v,
               note="the label probe " + ("failed: fix the drill, do not judge the network (METHOD 4)"
                                          if v.get("n_ok") else "too small (n_ok false)"))


# ------------------------------------------------------------------ the label probe (levels/tactic_probe.py)
PROBE = {"train": ("probe S2 train",), "held": ("probe S2 eval", "probe held axes"), "held_axis_n": 32,
         "followed": 0.90, "false_flank": 0.03, "erased_chance": 0.5, "erased_over": 0.10,
         "cause": {"x_k": "every column (cols.SIDE_PAIR + cols.SIDE_ROW at the label's pair + held)",
                   "cols": "the cols.SIDE_PAIR / SIDE_ROW columns among cols.D2 (none today: D2 carries no S2 "
                           "column - the network reads them through s2 and x_k only)",
                   "s2": "every column (cols.side_rows: lateral clearance left / right, the anvil's off-facing angle)"}}
_PROBE_WHY = ("METHOD step 4's label probe; lines followed >= 0.90 and S2N false flank <= 0.03 - the legacy probe's "
              "(net/config/tactic.json drills.side.probe.lines); the starts: the registry's probe_train / probe_held "
              "entries (S2 + S2N twins of each seed; 32 a held axis as the legacy n_held); the cause = whether the "
              "target's flank is open to the hammer - the side head's x_k (way clearance to enemies / own, extra path, "
              "|gP| a flank, lateral clearance, the anvil's off-facing angle, held) and the same facts as row columns "
              "(SIDE_ROW in cols, s2); erased: a cause-blind model sees the two twins of a pair alike, so on the pairs "
              "whose first labels differ it repeats at most one of two (chance 0.5), + 0.10 the margin the coordinator "
              "set (22.11)")


# ------------------------------------------------------------------ the card (cards/__init__.py)
S2N_FALSE = "S2N false flank (N: every pick, a flank on a wrong target counted)"
MARK_MIN = {S2N_FALSE: {"N": 30}}
_MARK_MIN_WHY = ("brief_A3 R3 (the coordinator's decision 22.11): path N's S2N cell had 37-44 labels in rounds 1-4 of "
                 "the legacy run (tmp/t02/train/train.log:13-44) - at 50 never judged; 30 there (its two-error band "
                 "carries the n), every other mark cards.MARK_MIN_N")


def marks(v):
    """The stop's marks on V (levels/tactic_v.side_marks_engine): followed per approach class - path N's denominator
    is every label with an approach, its wrong-target and KEEP picks counted as not followed ('missing' beside) - and
    the S2N false flank."""
    se = v.get("side_engine") or {}
    T, N = se.get("T") or {}, se.get("N") or {}
    g = lambda d, grp, k: (d.get(grp) or {}).get(k)                                   # noqa: E731
    out = {f"followed {c} (N: every pick)": {"T": g(T, "all", c), "N": g(N, "all", c), "N_missing": g(N, "all", "missing")}
           for c in ("front", "left", "right")}
    out[S2N_FALSE] = {"T": g(T, "S2N", "false_flank"), "N": g(N, "S2N", "false_flank")}
    return out


def retention(v):
    """T0-1's marks on V's retention part (the T0-1 scenes and G): followed N pooled, false N a tick."""
    r = (v.get("parts") or {}).get("retention") or {}
    return {"followed_N": (">=", (r.get("followed_N") or {}).get("pooled")),
            "false_N": ("<=", (r.get("false_N") or {}).get("tick"))}


def card():
    from net.drill.cards import Card
    from net.tactic import imitate as IM
    M = IM.SIDE_MARKS
    return Card(
        id="T0-2", words="side of the strike", cause="a flank of the target open to my hammer", level="tactic",
        parent="t0_1", stage="approach", teacher="TEACH2", disturb=None, arms={"TEACH": "T0-1's teacher",
        "TEACH2": "+ pin.choose_sides", "C": "the stack without the tactic", "N": "the network", "N1": "the parent"},
        below={"unit": "the unit scripts"}, opponent=("the scene enemy", "the general battles' AI stand-in"),
        scenes=None, sets={}, mass=1.0, classes=None, band_share_max=0.15,
        masks_changed={"lift": "through own lifted for a LEFT / RIGHT approach (imitate.py:91-97)"},
        marks=marks, marks_lines={**{f"followed {c} (N: every pick)": (">=", float(M["followed_t"]))
                                     for c in ("front", "left", "right")},
                                  S2N_FALSE: ("<=", float(M["false_flank"]))},
        value_gate=VALUE_GATE, files=("net/drill/levels/tactic.py",),
        level_args={"drill": "side", "tags": ["T0-1", "T0-2"]},
        stages={"teacher": "TEACH2 (disturbed)", "scenes": "learn.round_specs + imitate.side_round", "general": True,
                "net_share": "train.learn.net_share from round 2"},
        masks={"pointer": "tactic_pointer", "approach": "tactic_approach"},
        census={"pointer": {"attack": 50, "keep": 50}, "approach": {"front": 50, "left": 50, "right": 50}},
        dont_care_max=DONT_CARE_MAX, masks_moved_max=MASKS_MOVED_MAX, retention=retention,
        exec_denominator=EXEC_DENOMINATOR, mark_min=MARK_MIN)


DONT_CARE_MAX = {"pointer": {"attack": 0.35, "keep": 0.20}, "approach": {"front": 0.55, "left": 0.25, "right": 0.25}}
_DONT_CARE_WHY = ("round 1 of the T0-2 training on the GPU (tmp/engine/a/gpu_r1/run.log census): don't-care shares "
                  "attack 0.225, keep 0.093, approach front 0.410 (through own on FRONT, by design), left 0.105, right "
                  "0.107 - the maxima ~1.5 x those, so a mask that removes far more than by design halts")
MASKS_MOVED_MAX = 1000
_MOVED_WHY = ("round 1 (gpu_r1): the lift moved (0.95 - 0.8367) x 4,256 ~ 482 flank labels of the whole mixture "
              "(flank_weighted with / without the fix); on the T0-1 scenes alone fewer - twice that is the declared "
              "maximum")
