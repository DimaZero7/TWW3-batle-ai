"""The imitation set from a person's live battles (torch): data/label/human_set/samples.pt in the labelling record's
format (record.load reads it; record.rebuild rebuilds an operative sample's input bit for bit) and summary.json.

    strategist   each battle's deployment sample as the live record wrote it (game/label/live_deploy.py: the input
                 and the person's placement read as the strategist's choices; act {} with the errors when refused)
    operative    each window through the bridge (bridge_infer.infer + bridge_check.check): the raw input of the window,
                 the bridge's groups of the window before as the last groups, its groups now (members, the target's
                 seed, the area's centre as the goal) as the walk's choices - or 'keep'; act {} where today's operative
                 bans them, with the errors
    unit_order   each unit of each order command of the game (commands.jsonl: Attack Unit, Move Orientation Width): what
                 maps one to one onto the unit network's output - the target (slot), the facing (64 sectors from the
                 unit's bearing then), the run (ordered fast), the ordered width (m), the ordered point - and what is
                 missing (the cell of the area: the tactician's frame)
Every sample carries "battle" (the record's folder name).

    python -m game.label.digest_set data/label/<date>_live ... [--out data/label/human_set]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch

import game as project
from game.label import bridge_check as BC
from game.label import bridge_infer as BI
from game.label import digest as DG
from game.label import digest_export as DX
from game.label import record as RC

FACES = 64


def place_dims(smp, rec):
    """{name: (front, depth)} of the person's formations at the deployment: human_places' own when the record has them
    (game/label/live_deploy.py writes front / depth since 08.11), else the simulator's geometry.dims of each unit at
    his width and its men (the record's matchup) - the same rule live_deploy.formations_of uses."""
    places = (smp.get("human_places") or {}).get("units") or {}
    out = {n: (float(p["front"]), float(p["depth"])) for n, p in places.items()
           if p.get("front") is not None and p.get("depth") is not None}
    todo = [n for n in places if n not in out and places[n].get("width")]
    if not todo:
        return out
    from sim import geometry, scenario
    from sim.params import load
    P = load()
    men = {u["n"]: u.get("men") for u in rec.matchup.get("units", [])}
    fac = (rec.matchup.get("factions") or {})
    units = [{"key": rec.key_of[n], "x": float(places[n]["x"]), "z": float(places[n]["z"]),
              "b": float(places[n].get("b") or 0.0), "width": float(places[n]["width"]),
              **({"men": float(men[n])} if men.get(n) else {})} for n in todo]
    foe = next(n for n in rec.names if rec.side_of.get(n) == 2)
    army = {"attacker": 2, "sides": {1: {"faction": fac.get("own"), "units": units},
                                     2: {"faction": fac.get("enemy"), "units": [{"key": rec.key_of[foe], "x": 0.0,
                                                                                 "z": 500.0}]}}}
    st = scenario.build([army], P, per_side=len(units))
    front, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
    for i, n in enumerate(todo):
        out[n] = (float(front[0, i]), float(depth[0, i]))
    return out


def strategist_bridge(smp, rec):
    """The deployment sample with the person's groups made fit (the bridge for the strategist): his Ctrl+G groups, his
    units in none join the nearest group (by centre) with room, then the two nearest groups merge while over the
    strategist's G; the rest as the live record reads a deployment (live_deploy.edit_of: centre, mean front as the turn,
    line / square, every unit's exact place with its formation's front and depth - place_dims) and the strategist's walk
    (strategist.act_of, walk_of). Returns a new sample (source 'bridge')."""
    import math
    from game.label import human as HU
    from game.label import live_deploy as LD
    from game.label import strategist as LS
    from net.strategist import cfg as scfg
    G, cap = int(scfg()["groups"]), BI.MAX_GROUP
    places = (smp.get("human_places") or {}).get("units") or {}
    groups = {int(k): [n for n in v if n in places] for k, v in (smp.get("human_groups") or {}).get("own", {}).items()}
    groups = {k: v for k, v in groups.items() if v}
    pos = {n: (float(p["x"]), float(p["z"])) for n, p in places.items()}
    for n in pos:
        if any(n in v for v in groups.values()):
            continue
        best = min(((math.dist(pos[n], BI._centre(v, pos)), k) for k, v in groups.items() if len(v) < cap),
                   default=None)
        if best is None:
            groups[max(groups, default=0) + 1] = [n]
        else:
            groups[best[1]].append(n)
    merged = BI._merge_to(list(groups.values()), pos, G)
    groups = {i + 1: g for i, g in enumerate(merged)}
    names = smp.get("names") or rec.names
    doc = {"units": [{"n": n, "side": rec.side_of.get(n, 2), "men": 1 if n in places else 0,
                      "x": (places.get(n) or {}).get("x"), "z": (places.get(n) or {}).get("z"),
                      "b": (places.get(n) or {}).get("b"), "ow": (places.get(n) or {}).get("width")} for n in names]}
    inp, ctx, _ = LS.rebuild(smp)
    edit, raw, _ = LD.edit_of(doc, groups, float(ctx.base[0]), dims=place_dims(smp, rec))
    act, errs = {}, []
    try:
        act = LS.act_of(edit, inp, ctx.axes, inp["own"].device)
        _, errs = LS.walk_of(act, inp, ctx)
    except HU.Refused as e:
        errs = list(e.errors)
    out = dict(smp)
    out.update({"act": act if not errs else {}, "errors": errs, "source": "bridge", "edit": edit,
                "bridge_groups": groups})
    return out


def unit_commands(rec):
    slot_of = {n: i for i, n in enumerate(rec.names)}
    out = []
    for c in BI.battle_commands(rec):
        if c.get("name") not in BI.ORDER_COMMANDS:
            continue
        sec = float(c["sec"])
        st = rec.state(int(sec))
        for u in c.get("units") or []:
            o = (c.get("ordered") or {}).get(u) or {}
            r = st.get(u) or {}
            face = None
            if o.get("ob") is not None and r.get("b") is not None:
                face = int(round(((float(o["ob"]) - float(r["b"])) % 360.0) / (360.0 / FACES))) % FACES
            t = c.get("target") if c["name"] == "Attack Unit" else None
            out.append({"format": RC.FORMAT, "level": "unit_order", "mode": "commands", "where": "game", "sec": sec,
                        "name": u, "slot": slot_of.get(u), "key": rec.key_of.get(u),
                        "row": {k: r.get(k) for k in ("x", "z", "b", "men", "hp", "mv", "f", "m", "ow")},
                        "order": {"kind": "attack" if t else "move", "target": t,
                                  "point": [o.get("ox"), o.get("oz")] if not t else None, "bearing": o.get("ob"),
                                  "width_m": o.get("ow"), "run": bool(o.get("fast"))},
                        "act_partial": {"target_slot": slot_of.get(t) if t else None, "face": None if t else face,
                                        "pace": 1 if o.get("fast") else 0, "width_m": None if t else o.get("ow")},
                        "missing": ["where: a cell of the unit's area (the tactician's frame) - the person sets no area",
                                    "the view: the unit's rows and its zone map are built round the area",
                                    "width: the network's 8 steps need the simulator's width limits of the unit"]})
    return out


def battle(folder):
    rec = DG.Record(folder)
    name = rec.folder.name
    wins = BI.infer(rec)
    ops = []
    BC.check(rec, wins, samples=ops)
    strat = [s for s in RC.load(rec.folder) if s.get("level") == "strategist"] if (rec.folder / "samples.pt").exists() \
        else []
    strat = [strategist_bridge(x, rec) for x in strat]
    units = unit_commands(rec)
    for s in ops + strat + units:
        s["battle"] = name
    for s in ops[:3]:
        RC.rebuild(s)                                       # (the input builds back)
    n = {"strategist": {"samples": len(strat), "with_act": sum(1 for s in strat if s.get("act") and any(
             torch.is_tensor(v) for v in s["act"].values()))},
         "operative": {"samples": len(ops), "with_act": sum(1 for s in ops if s["act"]),
                       "human": sum(1 for s in ops if s["source"] == "human"),
                       "keep": sum(1 for s in ops if s["source"] == "keep")},
         "unit_order": {"samples": len(units), "attack": sum(1 for u in units if u["order"]["kind"] == "attack"),
                        "move": sum(1 for u in units if u["order"]["kind"] == "move")}}
    return strat + ops + units, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--out", default=str(project.DATA / "label" / "human_set"))
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    allS, summary = [], {}
    for f in a.folders:
        S, n = battle(f)
        allS += S
        summary[Path(f).name] = n
    tot = Counter()
    for n in summary.values():
        for lvl, v in n.items():
            tot[lvl + " samples"] += v["samples"]
            tot[lvl + " with act"] += v.get("with_act", v["samples"] if lvl == "unit_order" else 0)
    summary["total"] = dict(tot)
    summary["tactic"] = {"samples": 0, "missing": DX.MISSING_TACTIC}
    torch.save(allS, out / "samples.pt")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary["total"], ensure_ascii=False))


if __name__ == "__main__":
    main()
