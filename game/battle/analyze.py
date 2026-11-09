"""One chain battle in the game (companion --chain): step time, orders reaching the units, units in their areas, contact,
outcome. python -m game.battle.analyze <run> [--json out.json]  (host python, no torch)"""
import json
import math
import statistics
import sys
from pathlib import Path

from game.battle import costs, run_dir


def q(v):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    return {"n": len(v), "median": round(statistics.median(v), 1), "p90": round(v[int(0.9 * (len(v) - 1))], 1),
            "max": round(v[-1], 1)}


def outside_m(x, z, a):
    cx, cz, b, front, depth = a
    r = math.radians(b)
    dx, dz = x - cx, z - cz
    s = dx * math.cos(r) - dz * math.sin(r)
    f = dx * math.sin(r) + dz * math.cos(r)
    ex, ez = abs(s) - front / 2, abs(f) - depth / 2
    return math.hypot(max(ex, 0), max(ez, 0))


def main(run, out=None):
    run = run_dir(run)
    ev = [json.loads(line) for line in (run / "events.jsonl").open(encoding="utf-8")]
    comp = [json.loads(line) for line in (run / "companion.jsonl").open(encoding="utf-8")]
    moves = [r for r in comp if "move" in r and "chain" in r]
    dep = [r for r in comp if r.get("event") == "deploy"]
    res = {"run": run.name}
    if dep:
        d = dep[0]
        res["strategist"] = {"errors": d["errors"], "groups": [g["units"] for g in d["groups"]], "verb": d["verb"],
                             "task": d["task"]}
    op = [r for r in moves if r["chain"].get("operative")]
    tac_s = [r for r in moves if r["chain"]["ms"]["tac_ms"] > 0.5]
    res["step_ms"] = {"think_all": q([r["think_ms"] for r in moves]), "turn_all": q([r["turn_ms"] for r in moves]),
                      "think_operative_second": q([r["think_ms"] for r in op]),
                      "operative_ms": q([r["chain"]["ms"]["op_ms"] for r in op]),
                      "tactic_ms": q([r["chain"]["ms"]["tac_ms"] for r in tac_s]),
                      "unit_ms": q([r["chain"]["ms"]["unit_ms"] for r in moves]),
                      "raw_ms": q([r["chain"]["ms"]["raw_ms"] for r in moves])}
    no = [e for e in ev if e["event"] == "nn_orders"]
    res["game_wait"] = {"wait_model_ms": q([e.get("wait_model_ms") for e in no]),
                        "wait_real_ms": q([e.get("wait_real_ms") for e in no]),
                        "late_answers": sum(1 for e in no if e.get("lag", 0) > 0),
                        "missed": sum(1 for e in ev if e["event"] == "nn_miss"), "answers": len(no),
                        "moves": len(moves)}
    samples = [e for e in ev if e["event"] == "nn_sample"]
    status, kinds = {}, {}
    for o in no:
        for x in o["orders"]:
            status[x["status"]] = status.get(x["status"], 0) + 1
            kinds[x["k"]] = kinds.get(x["k"], 0) + 1
    # a move's point against the game's ordered_position at the next sample
    d_ord, si = [], 0
    for o in no:
        while si < len(samples) and samples[si]["model_ms"] <= o["model_ms"]:
            si += 1
        if si >= len(samples):
            break
        nxt = {u["n"]: u for u in samples[si]["units"]}
        for x in o["orders"]:
            if x["k"] == "move" and x["status"] == "given":
                u = nxt.get(x["u"])
                if u and u.get("ox") is not None and not u.get("r"):
                    d_ord.append(math.hypot(u["ox"] - x["x"], u["oz"] - x["z"]))
    res["orders"] = {"given_by_status": status, "kinds": kinds, "move_point_vs_ordered_position_m": q(d_ord),
                     "move_point_within_2m": round(sum(d < 2 for d in d_ord) / max(len(d_ord), 1), 3)}
    by_t = {round(s["t"] / 1000): {u["n"]: u for u in s["units"]} for s in samples if "t" in s}
    ins, far, per, after20 = [], [], {}, []
    for r in moves:
        c = r["chain"]
        sec = c["sec"]
        gone = {o["unit"] for o in r["orders"] if o.get("out")}
        for n, a in c["areas"].items():
            if n in gone:
                continue
            i = c["inside"][n]
            ins.append(i)
            per.setdefault(n, []).append(i)
            if sec >= 20:
                after20.append(i)
            u = (by_t.get(sec) or {}).get(n)
            if u and u.get("x") is not None:
                far.append(outside_m(u["x"], u["z"], a))
    res["areas"] = {"inside_share": round(sum(ins) / max(len(ins), 1), 3),
                    "inside_share_after_20s": round(sum(after20) / max(len(after20), 1), 3),
                    "by_unit": {n.removeprefix("own_"): round(sum(v) / len(v), 2) for n, v in per.items()},
                    "outside_m_when_out": q([f for f in far if f > 0.5]),
                    "area_front_m": q([a[3] for r in moves for a in r["chain"]["areas"].values()]),
                    "leash_unit_seconds": sum(len(r["chain"].get("leash") or []) for r in moves),
                    "tactic_fell": moves[-1]["chain"]["tactic_fell"] if moves else None,
                    "parcel_errors": moves[-1]["chain"]["parcel_errors"] if moves else None}
    res["operative"] = {"decisions": len(op),
                        "tasks": sum(g["task"] for r in op for g in r["chain"]["operative"]),
                        "reforms": sum(g["reformed"] for r in op for g in r["chain"]["operative"]),
                        "groups_first": [g["units"] for g in op[0]["chain"]["operative"]] if op else None,
                        "groups_last": [g["units"] for g in op[-1]["chain"]["operative"]] if op else None,
                        "zone_centres_last": [(g["cx"], g["cz"]) for g in op[-1]["chain"]["operative"]] if op else None}
    first_m = None
    for s in samples:
        if any(u.get("m") and u.get("side") == 1 for u in s["units"]):
            first_m = round(s["t"] / 1000)
            break
    result = [e for e in ev if e["event"] == "result"][0]
    final = [e for e in ev if e["event"] == "nn_final"][0]
    m = json.loads((run / "manifest.json").read_text(encoding="utf-8"))["config"]["units"]
    keys = {u["script_name"]: u["key"] for side in ("own", "enemy") for u in m[side]}
    db = costs()
    gold = {1: 0.0, 2: 0.0}
    for u in final["units"]:
        c = float((db.get(keys.get(u["n"], "")) or {}).get("multiplayer_cost") or 0)
        gold[u["side"]] += c * (1 - float(u.get("hp") or 0))
    first = samples[0]
    res["outcome"] = {"status": result["status"], "winner": result["winner"],
                      "t_s": round(result["duration_model_ms"] / 1000), "first_contact_s": first_m,
                      "men_start_end": {s: [sum(u["men"] for u in first["units"] if u["side"] == s),
                                            result[f"side_{s}_men"]] for s in (1, 2)},
                      "standing": [result["side_1_standing_units"], result["side_2_standing_units"]],
                      "gold_lost": {k: round(v) for k, v in gold.items()}, "trade_gold": round(gold[2] - gold[1])}
    txt = json.dumps(res, indent=1, ensure_ascii=False)
    if out:
        Path(out).write_text(txt, encoding="utf-8")
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(main(a[0], a[2] if len(a) > 2 and a[1] == "--json" else None), indent=1, ensure_ascii=False))
