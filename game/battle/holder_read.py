"""The game probe plan `holder` (tmp/op1/design_merged.md 'The kill check'): the starts and the report.

    python -m game.battle.holder_read --make        build/holder/<h01..h24, r01..r24>.json (--arena-json), battles.json
    bash game/battle/chain.sh tactic h01             (OP=holder:M | holder:D: the scripted operative net/operative/opside)
    bash game/battle/chain.sh tactic r01             (arm R, no OP: the reference - the same start, the game's AI attacks)
    python -m game.battle.holder_read <runs...> --out <json>    the report of the recorded runs

Starts: 24 = 2 maps (flat mp_crossroads_flat, ksl_plains_01) x 6 armies (4-8 infantry + lord a side, gold matched as
game/battle/aivai.py: net/stage0/specs._sides) x 2 mirrors (1: the two armies and factions swap sides); ours attack, the
game's AI defends. h = 1 + map x 12 + army x 2 + mirror. rNN - the same start with the roles swapped (the design's last
section): the game's AI attacks, ours defend with the chain alone (arm R; E = the enemy's infantry).

Per battle (the run's events.jsonl nn_sample a second, companion.jsonl's op notes, the arena's keys): arm, start, our
move (the first zone move), E (the op note's piece; none - the enemy's infantry), E's first move (its centroid advanced
e_move_m toward ours from its place at our move: s, the least centre gap our member - E's unit, m), metres walked by
each side's centroid (no lords) to the first contact and to the end, the first contact (a unit of each side in melee),
the share of our members standing (< still_ms for >= still_s before their first melee), routs a side, the result,
orders a unit-minute (game/battle/tactic_read.orders), the whole-field trade over [30, 630 s] and [30, 330 s] (enemy
lost - own lost, shares of each side's start health: hp x the key's hp_total, lords in, men 0 = all lost; a battle that
ended before the window's end: its last health carried forward - counted), no_sally / halt_stood (the op's flags). Then
the paired D - M by start (SE = sd / sqrt(n)), the plan's three lines with their verdicts and line 2R (better - R).
"""
import argparse
import contextlib
import io
import json
import math
import random
import statistics as stt
import sys
from pathlib import Path

import game

ROOT = game.ROOT
OUT = game.BUILD / "holder"
MAPS = ("flat", "ksl_plains_01")
SIZES = (4, 5, 6, 7, 8, 6)
SEED0 = 501
WINDOWS = ((30.0, 630.0), (30.0, 330.0))
N_PAIRS = 24
STAND_WIN_S, STAND_RUN_S = 20.0, 3.0   # ('received from a stand': the amendment in tmp/op1/design_merged.md)
E_MOVE_FAR_M = 40.0      # (line 1: E came out while our front >= 40 m from his - the amendment; front to front)
WALK_STEP_S = 5.0       # (the centroid's path in 5 s steps: the samples' jitter a second adds ~0.2 m/s standing)
E_MOVE_RUN_S = 3.0      # (E's first move: a standing unit of E faster than comes_ms for 3 s in a row - the review 9)
SAMPLE_MAX_S = 1.5      # (... in consecutive ~1 s samples: a longer interval breaks the run - the second review)


def conf():
    return json.loads((ROOT / "net/config/operative.json").read_text(encoding="utf-8"))["o1_probe"]


def starts():
    """[(name, map, army k, mirror)] the 24 starts (module doc)."""
    out = []
    for mi, mp in enumerate(MAPS):
        for k in range(len(SIZES)):
            for mir in (0, 1):
                out.append((f"h{1 + mi * 12 + k * 2 + mir:02d}", mp, k, mir))
    return out


def ref_name(h):
    """hNN -> rNN: the reference arm's start (the same place, armies and map; the game's AI attacks)."""
    return "r" + h[1:]


PLANNED = tuple(s[0] for s in starts())          # (the pairs count only these 24 starts, by name)


def make():
    from net.stage0 import specs as S
    from game.battle import aivai as AV
    OUT.mkdir(parents=True, exist_ok=True)
    allb = {}
    for name, mp, k, mir in starts():
        seed, n = SEED0 + k, SIZES[k]
        rnd = random.Random(seed)
        fac = ("emp", "skv") if seed % 2 else ("skv", "emp")
        f1, f2, mine, theirs = S._sides(rnd, n, fac=fac)
        if mir:
            f1, f2, mine, theirs = f2, f1, theirs, mine
        mz, ez = S.zone(0), S.zone(1)
        sp = {"fac": {1: f1, 2: f2}, "mine": mine, "theirs": theirs, "map": mp, "mine_zone": mz, "enemy_zone": ez,
              "places": {1: S.rows(mz, len(mine)), 2: S.rows(ez, len(theirs))}, "gold": (S.gold(mine), S.gold(theirs))}
        units = AV.world_units(sp)
        arena = AV.arena_of(name, sp, units)
        arena["name"], arena["note"] = f"holder_{name}", "plan holder: stage 0 armies, ours attack, the game's AI defends"
        (OUT / f"{name}.json").write_text(json.dumps(arena, indent=1), encoding="utf-8")
        allb[name] = {"seed": seed, "army": k, "mirror": mir, "map": mp, "role": "defend", "fac": sp["fac"],
                      "gold": sp["gold"], "units": {str(s): v for s, v in units.items()}}
        # the reference arm R (the last section of the design): the same start, only the role swapped
        rn = ref_name(name)
        arena["name"], arena["note"] = f"holder_{rn}", "plan holder, arm R: stage 0 armies, the game's AI attacks, ours defend"
        (OUT / f"{rn}.json").write_text(json.dumps(arena, indent=1), encoding="utf-8")
        allb[rn] = dict(allb[name], role="attack")
        print(name, rn, mp, "army", k, "mirror", mir, sp["fac"], len(mine), len(theirs), [round(g) for g in sp["gold"]])
    (OUT / "battles.json").write_text(json.dumps(allb, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ the report

def _jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def _cen(us):
    return (sum(u["x"] for u in us) / len(us), sum(u["z"] for u in us) / len(us)) if us else None


def _front_gap(us, ours, theirs, spc):
    """The front-to-front gap: our and their formations' footprints (files = ow / spacing h, ranks = men / files x
    spacing v, turned by b) projected on the centroids' axis; E's least edge - our largest."""
    if not ours or not theirs:
        return None
    oc, ec = _cen([us[n] for n in ours]), _cen([us[n] for n in theirs])
    L = math.hypot(ec[0] - oc[0], ec[1] - oc[1]) or 1.0
    ux, uz = (ec[0] - oc[0]) / L, (ec[1] - oc[1]) / L

    def ext(u, n):
        h, v = spc.get(n, (1.5, 1.6))
        men = max(int(u.get("men") or 1), 1)
        files = min(men, max(1, math.floor(float(u.get("ow") or h) / h + 1e-4)))
        depth = math.ceil(men / files) * v
        b = math.radians(u["b"])
        return (abs(ux * math.sin(b) + uz * math.cos(b)) * depth + abs(ux * math.cos(b) - uz * math.sin(b)) * files * h) / 2
    along = lambda u: (u["x"] - oc[0]) * ux + (u["z"] - oc[1]) * uz          # noqa: E731
    return min(along(us[n]) - ext(us[n], n) for n in theirs) - max(along(us[n]) + ext(us[n], n) for n in ours)


def battle(run):
    """The row of one run (module doc)."""
    from game.battle import tactic_read as TRD
    c = conf()
    run = Path(run)
    man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    src = Path(str(man["config"]["army"]["from_file"]).replace("\\", "/"))
    start = src.stem
    UN = json.loads((ROOT / "sim/config/units.json").read_text(encoding="utf-8"))["units"]
    hp_tot, spc = {}, {}
    for side in ("own", "enemy"):
        for u in (man["config"]["units"].get(side) or []):
            hp_tot[u["script_name"]] = float(UN.get(u["key"], {}).get("hp_total") or 1.0)
            sp_ = UN.get(u["key"], {}).get("spacing") or {}
            spc[u["script_name"]] = (float(sp_.get("h") or 1.5), float(sp_.get("v") or 1.6))
    ev = _jsonl(run / "events.jsonl")
    comp = TRD.companion_rows(run)
    st0 = next((r["chain"]["start"] for r in comp if "start" in r["chain"]), {})
    notes = [n for r in comp for n in r["chain"].get("op") or []]
    arm = (st0.get("op") or ("R" if start.startswith("r") else "off")).split(":")[-1]   # (rNN without op: arm R)
    res = next((d for d in ev if d.get("event") == "result"), {})
    fr = [(d["t"] / 1000.0, {u["n"]: u for u in d["units"]}) for d in ev if d.get("event") == "nn_sample"]
    fin = next((d for d in ev if d.get("event") == "nn_final"), None)
    if fin is not None and fin.get("units") and (not fr or fin["t"] / 1000.0 > fr[-1][0]):
        fr.append((fin["t"] / 1000.0, {u["n"]: u for u in fin["units"]}))       # (the record's last snapshot)
    fr.sort(key=lambda x: x[0])
    names = list(fr[0][1])
    mine = st0.get("mine") or [n for n in names if n.startswith("own_u")]
    foes = [n for n in names if n.startswith("enemy_") and not n.endswith("_lord")]
    move = next((n for n in notes if n.get("zone") is not None), None)
    t_move = move["sec"] if move else None
    E = (move or {}).get("E") or next((n["E"] for n in notes if n.get("E")), None) or foes   # (D never moved: the decision's E)
    E = [e if isinstance(e, str) else names[e] for e in E]
    row = {"run": run.name, "start": start, "arm": arm, "status": res.get("status"), "winner": res.get("winner"),
           "end_s": round(res["duration_model_ms"] / 1000, 1) if res.get("duration_model_ms") else fr[-1][0],
           "our_move_s": t_move, "E": E, "op": [{k: n.get(k) for k in ("sec", "act", "gap_front", "e_close", "comer")}
                                               for n in notes if n.get("act")]}
    row["crashed"] = not res               # (no result event: the battle did not end - out of the pairs)
    row["no_sally"] = any(n.get("no_sally") for n in notes)
    row["halt_stood"] = any(n.get("halt_stood") for n in notes)
    up = lambda u: u is not None and (u.get("men") or 0) > 0 and not u.get("r")          # noqa: E731
    # contact
    ours_all = [n for n in names if n.startswith("own_")]
    contact = next((t for t, us in fr if any(us[n].get("m") for n in ours_all if n in us)
                    and any(us[n].get("m") for n in names if n.startswith("enemy_") and n in us)), None)
    row["contact_s"] = contact
    row["contact_lord_only"] = contact is not None and not any(
        (dict(fr)[contact].get(n) or {}).get("m") for n in mine)
    # E's first move: the first second a standing unit of E closes on our standing members' centroid faster than
    # comes_ms for >= E_MOVE_RUN_S s in a row (closing, not any speed: a 5 m shuffle at 1.3 m/s at 30 s in run 052545)
    ts = [t for t, _ in fr]
    first = None
    run_ = {}
    oc_ = [_cen([us[n] for n in mine if up(us.get(n))]) for _, us in fr]     # (closing: toward our standing centroid)
    for i in range(1, len(fr)):
        dt = ts[i] - ts[i - 1]
        for n in E:
            a_, b_ = fr[i - 1][1].get(n), fr[i][1].get(n)
            fast = False
            if up(a_) and up(b_) and 0 < dt <= SAMPLE_MAX_S and oc_[i] is not None:   # (a sampling hole breaks the run)
                wx, wz = oc_[i][0] - a_["x"], oc_[i][1] - a_["z"]
                wn = math.hypot(wx, wz) or 1.0
                fast = ((b_["x"] - a_["x"]) * wx + (b_["z"] - a_["z"]) * wz) / wn / dt > float(c["comes_ms"])
            st_ = run_.get(n)
            run_[n] = (st_ if st_ is not None else i - 1) if fast else None
            if fast and ts[i] - ts[run_[n]] >= E_MOVE_RUN_S - 1e-6:
                first = run_[n]
                break
        if first is not None:
            break
    row["e_move_s"] = row["e_move_gap_m"] = row["e_move_gap_centres_m"] = None
    if first is not None:
        us = fr[first][1]
        row["e_move_s"] = ts[first]
        g_ = _front_gap(us, [n for n in mine if up(us.get(n))], [n for n in E if up(us.get(n))], spc)
        row["e_move_gap_m"], row["e_move_gap_exact_m"] = (None, None) if g_ is None else (round(g_, 1), g_)   # (line 1 gates the exact)
        row["e_move_gap_centres_m"] = round(min(math.hypot(us[a_]["x"] - us[b_]["x"], us[a_]["z"] - us[b_]["z"])
                                                for a_ in mine if up(us.get(a_)) for b_ in E if up(us.get(b_))), 1)
    row["e_move_after_ours_s"] = None if first is None or t_move is None else round(ts[first] - t_move, 1)
    # metres walked: the sum of the units' own paths (no lords) in WALK_STEP_S steps, routing stretches out
    for side, us_n in (("own", mine), ("enemy", foes)):
        path, to_c = 0.0, None
        last = {}
        seg = lambda u, l: math.hypot(u["x"] - l[1], u["z"] - l[2])          # noqa: E731
        for t, us in fr:
            at_c = contact is not None and to_c is None and t >= contact
            part = 0.0                     # (the stretches since each unit's last step, up to the contact sample)
            for n in us_n:
                u = us.get(n)
                if not up(u):
                    if n in last and u is not None and u.get("x") is not None:
                        path += seg(u, last[n])          # (the stretch up to the rout sample)
                    last.pop(n, None)
                    continue
                if n not in last:
                    last[n] = (t, u["x"], u["z"])
                elif t - last[n][0] >= WALK_STEP_S:
                    path += seg(u, last[n])
                    last[n] = (t, u["x"], u["z"])
                elif at_c:
                    part += seg(u, last[n])
            if at_c:
                to_c = path + part
        us = fr[-1][1]
        path += sum(seg(us[n], l) for n, l in last.items() if n in us)     # (the stretch up to the end sample)
        row[f"{side}_walked_m"] = round(path, 1)
        row[f"{side}_walked_to_contact_m"] = None if to_c is None else round(to_c, 1)
    # standing at the first melee
    stand = tot = 0
    for n in mine:
        k = next((i for i, (t, us) in enumerate(fr) if (us.get(n) or {}).get("m")), None)
        if k is None:
            continue
        tot += 1
        lo = [i for i in range(k + 1) if ts[i] >= ts[k] - float(c["still_s"])]
        sp = []
        for i in lo[1:]:
            a, b = fr[i - 1][1].get(n), fr[i][1].get(n)
            dt = ts[i] - ts[i - 1]
            if a and b and dt > 0:
                sp.append(math.hypot(b["x"] - a["x"], b["z"] - a["z"]) / dt)
        stand += bool(sp) and max(sp) < float(c["still_ms"]) and ts[k] - ts[lo[0]] >= float(c["still_s"]) - 1.01
    row["first_melee_members"], row["standing5_share"] = tot, (round(stand / tot, 3) if tot else None)
    # 'received from a stand' (the amendment of 21.11): >= STAND_RUN_S consecutive seconds under still_ms within the
    # STAND_WIN_S before the member's first melee
    rec_n = 0
    for n in mine:
        k = next((i for i, (t, us) in enumerate(fr) if (us.get(n) or {}).get("m")), None)
        if k is None:
            continue
        run_s, best = 0.0, 0.0
        for i in range(1, k + 1):
            if ts[i - 1] < ts[k] - STAND_WIN_S - 1e-6:          # (the whole interval inside the window)
                continue
            a, b = fr[i - 1][1].get(n), fr[i][1].get(n)
            dt = ts[i] - ts[i - 1]
            slow = bool(a and b and dt > 0 and math.hypot(b["x"] - a["x"], b["z"] - a["z"]) / dt < float(c["still_ms"]))
            run_s = run_s + dt if slow else 0.0
            best = max(best, run_s)
        rec_n += best >= STAND_RUN_S - 1e-6
    row["stand_share"], row["stand_n"] = (round(rec_n / tot, 3) if tot else None), rec_n
    # our members moving (> still_ms) at E's first move (not gating)
    row["moving_at_e_move"] = None
    if row["e_move_s"] is not None:
        k = next(i for i, t in enumerate(ts) if t >= row["e_move_s"])
        live = [n for n in mine if up(fr[k][1].get(n))]
        if k > 0 and live:
            mv = 0
            for n in live:
                a, b = fr[k - 1][1].get(n), fr[k][1].get(n)
                dt = ts[k] - ts[k - 1]
                mv += bool(a and b and dt > 0 and math.hypot(b["x"] - a["x"], b["z"] - a["z"]) / dt > float(c["still_ms"]))
            row["moving_at_e_move"] = round(mv / len(live), 3)
    # routs
    for side, pre in (("own", "own_"), ("enemy", "enemy_")):
        n_r, prev = 0, {}
        for t, us in fr:
            for nm, u in us.items():
                if nm.startswith(pre):
                    if u.get("r") and not prev.get(nm):
                        n_r += 1
                    prev[nm] = bool(u.get("r"))
        row[f"{side}_routs"] = n_r
    # orders a unit-minute
    with contextlib.redirect_stdout(io.StringIO()):
        o = TRD.orders(run, comp)
    row["orders_per_unit_min"] = round(o["n"] / max(o["min"], 1e-9), 3)
    # the trade
    dead = {}

    def health(us, pre):
        h = 0.0
        for nm, u in us.items():
            if not nm.startswith(pre):
                continue
            if (u.get("men") or 0) <= 0:
                dead[nm] = True
            if dead.get(nm):
                continue
            h += float(u.get("hp") or 0.0) * hp_tot.get(nm, 1.0)
        return h
    H = [(t, health(us, "own_"), health(us, "enemy_")) for t, us in fr]
    h0o, h0e = H[0][1], H[0][2]

    def at(T):
        return next(((o_, e_) for t, o_, e_ in H if t >= T), (H[-1][1], H[-1][2]))
    row["ended_before"] = {}
    for a, b in WINDOWS:
        oa, ea = at(a)
        ob, eb = at(b)
        key = f"{int(a)}_{int(b)}"
        row[f"trade_{key}"] = round((ea - eb) / max(h0e, 1e-9) - (oa - ob) / max(h0o, 1e-9), 4)
        row[f"own_lost_{key}"] = round((oa - ob) / max(h0o, 1e-9), 4)
        row[f"enemy_lost_{key}"] = round((ea - eb) / max(h0e, 1e-9), 4)
        row["ended_before"][key] = H[-1][0] < b
    return row


def _mse(v):
    if not v:
        return None, None
    m = sum(v) / len(v)
    se = (stt.stdev(v) / math.sqrt(len(v))) if len(v) > 1 else float("nan")
    return m, se


def summary(rows):
    """The arms, the paired D - M and the plan's lines (module doc). A crashed run (no result event or an unreadable
    record) is out; a repeated start:arm - the latest run (by run id) wins everywhere; a start with a missing arm is out
    of the pairs and counted; a stalled or no-contact battle stays in. The lines are FINAL only with N_PAIRS complete
    pairs (all 48 battles in), else PROVISIONAL - and with fewer than 2 pairs no teacher is named. Only the planned
    starts count (h01..h24 arms D, M; r01..r24 arm R), each by its name. Line 2R pairs the better attack arm of line 2
    with R by start; FINAL with 24 complete triples (hNN D, hNN M, rNN R)."""
    plan = {**{(h, a): h for h in PLANNED for a in ("D", "M")}, **{(ref_name(h), "R"): h for h in PLANNED}}
    crashed = [r for r in rows if r.get("crashed")]
    good = sorted((r for r in rows if not r.get("crashed")), key=lambda r: r["run"])
    off_plan = [r["run"] for r in good if (r["start"], r["arm"]) not in plan]
    good = [r for r in good if (r["start"], r["arm"]) in plan]
    last = {}
    for r in good:
        last[(r["start"], r["arm"])] = r
    rows = list(last.values())
    by = {}
    for r in rows:
        by.setdefault(r["arm"], []).append(r)
    out = {"n": {a: len(v) for a, v in by.items()}, "arms": {}, "crashed": [r["run"] for r in crashed],
           "duplicates_dropped": len(good) - len(rows), "off_plan": off_plan}
    for a, v in by.items():
        m, se = _mse([r["trade_30_630"] for r in v])
        ss = [r["stand_share"] for r in v if r["stand_share"] is not None]
        out["arms"][a] = {"trade_30_630": m, "se": se, "trade_30_330": _mse([r["trade_30_330"] for r in v])[0],
                          "contact": sum(r["contact_s"] is not None for r in v), "n": len(v),
                          "stand_share_battles": (sum(ss) / len(ss)) if ss else None,
                          "stand_share": (sum(r["stand_n"] for r in v) / max(sum(r["first_melee_members"] for r in v), 1)
                                          if ss else None),         # (pooled over the members, not a mean of battles)
                          "ended_before_630": sum(r["ended_before"]["30_630"] for r in v),
                          "stalled": sum(r["status"] == "stalled" for r in v),
                          "no_sally": sum(r["no_sally"] for r in v), "halt_stood": sum(r["halt_stood"] for r in v),
                          "orders_per_unit_min": _mse([r["orders_per_unit_min"] for r in v])[0]}
    pairs = {}
    for r in rows:
        pairs.setdefault(plan[(r["start"], r["arm"])], {})[r["arm"]] = r
    for r in crashed:
        h = next((v for (st, _), v in plan.items() if st == r.get("start")), None)
        if h:
            pairs.setdefault(h, {}).setdefault("crashed", set()).add("R" if r["start"].startswith("r") else "DM")
    both = [p for p in pairs.values() if "D" in p and "M" in p]
    triples = sum("D" in p and "M" in p and "R" in p for p in pairs.values())
    out["pairs"], out["triples"] = len(both), triples
    out["pairs_dropped"] = sum(("D" in p or "M" in p or "DM" in p.get("crashed", ())) and not ("D" in p and "M" in p)
                               for p in pairs.values())
    for key in ("30_630", "30_330"):
        d = [p["D"][f"trade_{key}"] - p["M"][f"trade_{key}"] for p in both]
        m, se = _mse(d)
        out[f"D-M_{key}"] = {"mean": m, "se": se, "n": len(d), "positive": sum(x > 0 for x in d)}
    final = len(both) >= N_PAIRS
    tag = "FINAL" if final else f"PROVISIONAL ({len(both)} of {N_PAIRS} pairs)"
    lines = []
    D = by.get("D", [])
    n_d = len(D)
    gx = lambda r: r.get("e_move_gap_exact_m", r["e_move_gap_m"])          # noqa: E731  (the exact, not the rounded)
    far = sum(gx(r) is not None and gx(r) >= E_MOVE_FAR_M for r in D)
    gaps = [r["e_move_gap_m"] for r in D if r["e_move_gap_m"] is not None]
    med = stt.median(gaps) if gaps else float("nan")
    ns = sum(r["no_sally"] for r in D)
    if not n_d:
        v = "no D battle read"
    elif far >= 20:
        v = f"HALT on 'he comes' stays; stand-off = the median gap at E's first move {med:.0f} m"
    else:
        v = f"the teacher's distance = the median gap at E's first move {med:.0f} m; no sally {ns} of {n_d}" + \
            (" -> Attack(E) queued as O0-2" if ns >= 3 else "")
    lines.append(("1 sally", f"{tag}: in D, E's first move at a front gap >= {E_MOVE_FAR_M:.0f} m in {far} of {n_d} "
                             f"(need 20 of 24): {v}"))
    arms = out["arms"]
    cand = [a for a in ("D", "M") if a in arms and arms[a]["trade_30_630"] is not None]
    if cand:
        best = max(cand, key=lambda a: arms[a]["trade_30_630"])
        b = arms[best]
        if b["n"] < 2:
            verdict = "no verdict (n < 2)"
            rng = ""
        else:
            lo, hi = b["trade_30_630"] - 2 * b["se"], b["trade_30_630"] + 2 * b["se"]
            verdict = "GO" if b["contact"] >= 22 and lo >= -0.05 else "DROP" if hi < -0.15 else "FREEZE"
            rng = f" [{lo:+.3f}, {hi:+.3f}]"
        lines.append(("2 worth", f"{tag}: better arm {best}: contact {b['contact']} of {b['n']} (need 22 of 24; "
                                 f"stalled {b['stalled']}), trade {b['trade_30_630']:+.3f}{rng}: {verdict}"))
    # line 2R (the design's last section): paired by start, (the better attack arm of line 2) - R
    if cand and "R" in arms:
        rp = [p for p in pairs.values() if best in p and "R" in p]
        r_drop = sum(("R" in p or "R" in p.get("crashed", ())) and not (best in p and "R" in p) for p in pairs.values())
        t2 = "FINAL" if triples >= N_PAIRS else f"PROVISIONAL ({triples} of {N_PAIRS} triples)"
        for key in ("30_630", "30_330"):
            d = [p[best][f"trade_{key}"] - p["R"][f"trade_{key}"] for p in rp]
            m, se = _mse(d)
            out[f"{best}-R_{key}"] = {"mean": m, "se": se, "n": len(d), "positive": sum(x > 0 for x in d),
                                      "dropped": r_drop}
        bR = out[f"{best}-R_30_630"]
        if bR["n"] >= 2:
            lo, hi = bR["mean"] - 2 * bR["se"], bR["mean"] + 2 * bR["se"]
            v = ("GO (advancing costs no more than 0.05 of trade against being attacked)" if lo >= -0.05 else
                 "NOT WORTH (the next step is how to attack, not the teacher)" if hi < -0.05 else
                 "GO-with-cost (the teacher ADVANCE / KEEP, the cost in the journal)")
            lines.append(("2R reference", f"{t2}: {best} - R {bR['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}] over {bR['n']} "
                                          f"pairs ({bR['positive']} positive; dropped: {r_drop}; R alone "
                                          f"{arms['R']['trade_30_630']:+.3f}, contact {arms['R']['contact']} of "
                                          f"{arms['R']['n']}): {v}"))
        else:
            lines.append(("2R reference", f"{t2}: {best} - R over {bR['n']} pairs (dropped: {r_drop}): no verdict (n < 2)"))
    dm = out["D-M_30_630"]
    if dm["n"] >= 2:
        lo, hi = dm["mean"] - 2 * dm["se"], dm["mean"] + 2 * dm["se"]
        sd = arms.get("D", {}).get("stand_share")
        if lo >= 0.03:
            v = "the teacher is D"
        elif hi < 0.03:
            v = "the teacher is M (HALT removed)"
        else:
            v = "between: " + ("D (its stand share %.2f >= 0.75)" % sd if sd is not None and sd >= 0.75
                               else "M (D's stand share %s < 0.75)" % (None if sd is None else round(sd, 2)))
        lines.append(("3 form", f"{tag}: D - M {dm['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}] over {dm['n']} pairs "
                                f"({dm['positive']} positive; pairs dropped: {out['pairs_dropped']}): {v}"))
    else:
        lines.append(("3 form", f"{tag}: D - M {dm['mean'] if dm['mean'] is None else round(dm['mean'], 3)} over "
                                f"{dm['n']} pairs (pairs dropped: {out['pairs_dropped']}): no teacher chosen (n < 2)"))
    out["lines"] = dict(lines)
    out["note"] = (f"health carried forward from the battle's end in {sum(r['ended_before']['30_630'] for r in rows)} "
                   f"of {len(rows)} battles over [30, 630 s], {sum(r['ended_before']['30_330'] for r in rows)} over "
                   f"[30, 330 s]; SE with n < 2 is nan")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.make:
        make()
        return 0
    rows = []
    for r in a.runs:
        p = Path(r)
        if not p.exists():
            p = game.BUILD / "nn-arena" / "runs" / r
        try:
            rows.append(battle(p))
        except (Exception, SystemExit) as e:   # (a crashed run: no result or an unreadable record - out of the pairs)
            st = None
            try:
                man = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
                st = Path(str(man["config"]["army"]["from_file"]).replace(chr(92), "/")).stem
            except Exception:
                pass
            print(f"crashed run {p.name} ({st}): {type(e).__name__}: {e}")
            rows.append({"run": p.name, "start": st, "arm": None, "crashed": True})
    hdr = ("run", "start", "arm", "our_move_s", "e_move_s", "e_move_gap_m", "own_walked_m", "enemy_walked_m",
           "contact_s", "stand_share", "standing5_share", "moving_at_e_move", "e_move_after_ours_s", "own_routs", "enemy_routs", "status", "winner", "end_s",
           "orders_per_unit_min", "trade_30_630", "trade_30_330", "no_sally", "halt_stood")
    for r in rows:
        if r.get("crashed"):
            continue
        print(" ".join(f"{k}={r[k]}" for k in hdr))
        print("   op: " + "; ".join(f"{n['sec']:.0f} {n['act']} gap {n['gap_front']} close {n['e_close']}"
                                     + (" comer" if n["comer"] else "") for n in r["op"]))
    s = summary(rows)
    print(json.dumps({k: v for k, v in s.items() if k != "lines"}, indent=1, default=str))
    for k, v in s["lines"].items():
        print(f"LINE {k}: {v}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"rows": rows, "summary": s}, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
