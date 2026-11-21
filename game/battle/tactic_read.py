"""The technical check of a battle of the chain 'tactic_v2 only' (bridge/companion/tactic_chain.py; game/battle/chain.sh
tactic): did the level's 'attack e' reach the game, and is the level's input the twin's. No outcome judged.

    python -m game.battle.tactic_read <run> [--ticks N]       (torch: bash ops/dock.sh game.battle.tactic_read <run>)

1. attacks   each 'attack' the companion sent (companion.jsonl: its orders, the level's new attacks by name) against the
             bridge's nn_orders and nn_rally (the order given: status; 'by rally' - the bridge re-gave a rallied
             unit's attack) and the game's own unit state (nn_sample: within REACH_S the
             unit's target 't' is that enemy, or it is in melee within NEAR_M of it)
   orders    orders given to the game per unit-minute (all, attack), in melee, and of those cutting a live fight
2. replay    the logged documents of the first N ticks fed to a fresh TacticChain (the same weights): its input (rows,
             D2's columns, own, the pair numbers, the masks) and attacks equal the logged ones - the input the game's
             companion computed is the level's input of those documents
3. twin      the simulator's twin of the first document (sim/scenario at the document's places, facings and ordered
             widths 'ow' (without them two enemies at a sub-metre tie swap rows), the map's
             heights when it is captured, else flat; net/stage0/battle.TacticSide with the same network): tick 0's input
             against the logged one - the largest gap of each part, the rows' worst columns
"""
import argparse
import json
import math
import sys
from pathlib import Path

from game.battle import run_dir

REACH_S = 20.0           # an attack given reaches the game when the unit takes the target within 20 s (a 100 m walk)
NEAR_M = 40.0
CUT_M = 35.0             # the old target still at the unit (centres; two fronts' half widths, ~15 + 15 m, and a margin)
WEIGHTS = "weights/tactic/attack1/final.pt"
FACTIONS = {"emp": "wh_main_emp_empire", "skv": "wh2_main_skv_skaven"}


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def companion_rows(run):
    p = run / "companion.jsonl"
    if not p.exists():
        cands = sorted((run.parent.parent / "companion").glob("*.jsonl")) if run.parent.name == "runs" else []
        raise SystemExit(f"no {p} (watch.ps1 copies the companion's log there; latest logs: {cands[-1:] or 'none'})")
    return [r for r in _jsonl(p) if "chain" in r]


def attacks(run, rows):
    ev = _jsonl(run / "events.jsonl")
    given = [(d["t"], o) for d in ev if d.get("event") == "nn_orders" for o in d.get("orders") or [] if o.get("k") == "attack"]
    given += [(d["t"], {"u": d["u"], "tg": d.get("tg"), "status": f"{d.get('status')} by rally", "rally": True})
              for d in ev if d.get("event") == "nn_rally" and d.get("k") == "attack"]   # a rallied unit's order re-given
    given.sort(key=lambda x: x[0])
    used = set()        # a re-give by rally answers ONE choice: the first unanswered one it matches
    samples = [d for d in ev if d.get("event") == "nn_sample"]
    level = [(r["chain"]["sec"], a, b) for r in rows for a, b in (r["chain"].get("tactic") or {}).get("attacks", [])]
    sent = [(r["chain"]["sec"], o["unit"], o["target"]) for r in rows for o in r["orders"] if o["kind"] == "attack"
            and not o.get("out")]
    print(f"attacks: the level chose {len(level)} new attacks, the companion sent {len(sent)} attack lines, "
          f"the bridge gave {len(given)} attack orders")
    ok = 0
    for sec, u, tg in level:
        gi = [i for i, (t, o) in enumerate(given) if o["u"] == u and o.get("tg") == tg and t >= sec * 1000 - 1500
              and i not in used]
        g = [given[i][1] for i in gi]
        st = g[0].get("status") if g else "not given"
        took = None
        t0 = g and given[gi[0]][0]
        if g and g[0].get("rally"):
            used.add(gi[0])
        for s in samples:
            if not g or s["t"] < t0 or s["t"] > t0 + REACH_S * 1000:
                continue
            row = {x["n"]: x for x in s["units"]}
            me, e = row.get(u), row.get(tg)
            if me is None or e is None:
                continue
            if me.get("t") == tg or (me.get("m") and math.hypot(me["x"] - e["x"], me["z"] - e["z"]) <= NEAR_M):
                took = (s["t"] - t0) / 1000
                break
        ok += took is not None
        print(f"  s {sec:4d}  {u} -> {tg}: bridge {st}, " + (f"taken after {took:.0f} s" if took is not None
                                                            else f"NOT taken within {REACH_S:.0f} s"))
    print(f"attacks reached the game: {ok}/{len(level)}")
    return ok, len(level)


def orders(run, rows):
    """Orders given to the game per unit-minute of the level's units (nn_orders and nn_rally with a status other than
    kept / unknown_target, the bridge's re-gives nn_stall and nn_reaim - a reaim is an attack; a minute counts while the
    unit has men and does not rout), all kinds and attacks; those given in melee (the last nn_sample before: 'm'), and of
    them the attacks that changed the unit's target while the old one had men, did not rout and stood within CUT_M (cut
    a live fight: the unit stops striking back for ~25 s, game/manual.md 'не дёргать')."""
    import bisect
    ev = _jsonl(run / "events.jsonl")
    start = next((r["chain"]["start"] for r in rows if "start" in r["chain"]), {})
    mine = start.get("mine") or sorted({o["u"] for d in ev if d.get("event") == "nn_orders" for o in d.get("orders") or []
                                         if o["u"].startswith("own_u")})
    samples = [d for d in ev if d.get("event") == "nn_sample"]
    ts = [s["t"] for s in samples]
    by = [{x["n"]: x for x in s["units"]} for s in samples]
    live = {u: 0.0 for u in mine}
    for i in range(len(samples) - 1):
        for u in mine:
            x = by[i].get(u)
            if x and (x.get("men") or 0) > 0 and not x.get("r"):
                live[u] += (ts[i + 1] - ts[i]) / 60000
    skip = ("kept", "unknown_target")       # not an order the game got
    given = [(d["t"], o["u"], o["k"], o.get("tg"), False) for d in ev if d.get("event") == "nn_orders"
             for o in d.get("orders") or [] if o.get("status") not in skip]
    given += [(d["t"], d["u"], d["k"], d.get("tg"), True) for d in ev if d.get("event") == "nn_rally"
              and d.get("status") not in skip]
    given += [(d["t"], d["u"], d["k"], d.get("tg"), False) for d in ev if d.get("event") == "nn_stall"]
    given += [(d["t"], d["u"], "attack", d.get("tg"), False) for d in ev if d.get("event") == "nn_reaim"]
    c = {u: dict(n=0, att=0, rally=0, melee=0, cut=0) for u in mine}
    for t, u, k, tg, ra in sorted(given, key=lambda x: x[0]):
        if u not in c:
            continue
        c[u]["n"] += 1
        c[u]["att"] += k == "attack"
        c[u]["rally"] += ra
        i = bisect.bisect_left(ts, t) - 1        # the last sample strictly before the order
        me = by[i].get(u) if i >= 0 else None
        if not me or not me["m"]:
            continue
        c[u]["melee"] += 1
        old = by[i].get(me["t"]) if me["t"] else None
        if k == "attack" and old is not None and me["t"] != tg and old["men"] > 0 and not old["r"] \
                and math.hypot(me["x"] - old["x"], me["z"] - old["z"]) <= CUT_M:
            c[u]["cut"] += 1
    print("orders to the game (nn_orders, nn_rally, nn_stall, nn_reaim) per unit-minute standing:")
    tot = dict(n=0, att=0, rally=0, melee=0, cut=0, min=0.0)
    for u in mine:
        m, x = live[u], c[u]
        for k in x:
            tot[k] += x[k]
        tot["min"] += m
        print(f"  {u:8s} {m:5.2f} min: {x['n']:3d} orders ({x['rally']} by rally) {x['n'] / max(m, 1e-9):5.2f}/min, "
              f"attack {x['att'] / max(m, 1e-9):5.2f}/min; in melee {x['melee']}, cut a live fight {x['cut']}")
    m = max(tot["min"], 1e-9)
    print(f"  side     {tot['min']:5.2f} min: {tot['n']:3d} orders ({tot['rally']} by rally) {tot['n'] / m:5.2f}/min, "
          f"attack {tot['att'] / m:5.2f}/min; in melee {tot['melee']} of {tot['n']}, cut a live fight {tot['cut']}")
    return tot


def _gap(a, b):
    import torch
    a, b = torch.as_tensor(a, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32)
    if a.shape != b.shape:
        return float("inf")
    return float((a - b).abs().max()) if a.numel() else 0.0


def _dump(rec):
    import torch
    return {k: v[0].tolist() if v.dtype != torch.bool else v[0].int().tolist() for k, v in rec.items()}


def replay(rows, net, n):
    from bridge.companion import tactic_chain as TCH
    comp = TCH.TacticChain(net, dump_ticks=n)
    worst, k = {}, 0
    for r in rows:
        c = r["chain"]
        if "doc" not in c:
            break
        comp.decide(c["doc"])
        mine = comp.extra.get("dump")
        if "dump" in c:
            assert mine is not None, f"second {c['sec']}: the replay did not tick"
            for key, v in c["dump"].items():
                worst[key] = max(worst.get(key, 0.0), _gap(v, mine[key]))
            same = c["tactic"]["attacks"] == comp.extra["tactic"]["attacks"]
            worst["attacks_differ"] = worst.get("attacks_differ", 0) + (not same)
            k += 1
    print(f"replay of {k} ticks: largest gaps " + ", ".join(f"{a} {b:.2g}" for a, b in worst.items()))
    return worst


def twin(rows, net):
    import torch
    from bridge.companion import tactic_chain as TCH
    from bridge.companion import unit_input as UI
    from net.stage0 import battle as SB
    from net.tactic import pin as PN
    from net.tactic import raw as TR
    from net.tactic import rows as RW
    from net.unit import ckpt as T
    from sim import scenario, terrain
    from sim.params import load
    first = next((r["chain"] for r in rows if "dump" in r["chain"]), None)
    if first is None:
        print("twin: no dumped tick in the log")
        return None
    doc = first["doc"]
    sides = [int(u["side"]) for u in doc["units"]]
    if sides != sorted(sides):
        print("twin: the document is not in side order (side 1 first): not compared")
        return None
    P = load()
    spec = {"attacker": 2, "sides": {}}
    for u in doc["units"]:
        s = int(u["side"])
        fac = next((f for tag, f in FACTIONS.items() if f"_{tag}_" in u["key"]), None)
        if fac is None:
            print(f"twin: no faction for {u['key']}")
            return None
        side = spec["sides"].setdefault(s, {"faction": fac, "units": []})
        side["units"].append({"key": u["key"], "x": u["x"], "z": u["z"], "b": u["b"],
                              **({"width": u["ow"]} if u.get("men", 0) > 1 and u.get("ow") else {}),
                              **({"general": True} if u["n"].endswith("_lord") else {})})
    st = scenario.build([spec], P)
    folder = UI.map_folder(doc.get("map"))
    if folder is not None:
        zero = torch.tensor([0.0])
        terrain.set_map(st, UI.load_map(folder), place=(zero, zero.clone(), zero.clone()))
    card = T.cards(st, "cpu")[:1]
    me = SB.TacticSide(PN.NetPlayer(net, greedy=True))
    me.start(st, P, "cpu", card, None)
    me.tick(st, TR.Seer(P).raw(st))
    a = _dump(TCH.tick_record(me.tac))
    b = first["dump"]
    gaps = {k: _gap(a[k], b[k]) for k in b if k in a}
    print(f"twin (tick 0, {'map ' + folder.name if folder else 'flat'}): " + ", ".join(f"{k} {v:.2g}" for k, v in gaps.items()))
    if gaps.get("rows", 0) > 1e-3 and gaps["rows"] != float("inf"):
        d = (torch.tensor(a["rows"]) - torch.tensor(b["rows"])).abs().amax(0)
        top = d.argsort(descending=True)[:8].tolist()
        print("  rows' worst columns: " + ", ".join(f"{RW.ROW[i]} {float(d[i]):.3g}" for i in top if d[i] > 1e-3))
    return gaps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--ticks", type=int, default=12, help="the replay's ticks (the companion dumps 12)")
    ap.add_argument("--weights", default=WEIGHTS)
    a = ap.parse_args(argv)
    run = run_dir(a.run)
    rows = companion_rows(run)
    print(f"run {run}: {len(rows)} decisions; start {next((r['chain'].get('start') for r in rows if 'start' in r['chain']), None)}")
    attacks(run, rows)
    orders(run, rows)
    from net.tactic import ckpt as TC
    net, _, ck = TC.restore(a.weights, "cpu")
    assert ck.get("kind") == TC.KIND2, ck.get("kind")
    net.eval()
    replay(rows, net, a.ticks)
    twin(rows, net)
    return 0


if __name__ == "__main__":
    sys.exit(main())
