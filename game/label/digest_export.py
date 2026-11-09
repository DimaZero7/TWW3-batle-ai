"""The training export of a live labelling record (torch): <out>/samples.pt in the labelling record's format (record.load
reads it, record.rebuild rebuilds the operative's input) and <out>/export.json (the counts and what is missing).

    operative    the record's own samples (game/label/live.py: the window's raw input, the person's groups before it,
                 his groups as the walk's choices - act - or 'keep'); each is rebuilt here once (record.rebuild) so the
                 export holds only samples whose input builds; a window the walk refused keeps act {} and its errors
                 (a label 'the groups as they were', not the walk's)
    unit_order   a unit's order the person gave (a change of its ordered point, target or bearing, seen in the orders
                 stream), against the unit network's output (net/unit/net.py HEADS where, face, target, pace,
                 width): one to one - the target (the slot; the network's pointer is that slot's row in its view),
                 the facing (64 sectors from the unit's front: the ordered bearing less its bearing; the run's events
                 only), the pace (moving fast or not - the game's is_moving_fast, a state, not the order itself),
                 the width in metres (the network's 8 steps need the simulator's width limits); missing - the cell
                 ('where' is a cell of the unit's area, the tactician's frame: the person sets no area) and so the view
                 (the rows and the zone map are built round the area)
    strategist   the deployment's sample (game/label/live_deploy.py: the strategist's input of the game's deployment
                 document, the person's groups and places as the walk's choices), rebuilt once (strategist.rebuild)
    tactic       none: the tactician's output is each unit's frame (a 48 / 64 / 80 m step of width and depth and a cell
                 of the group's near window) inside the group's zone; the person sets neither the zone (the record puts
                 it where the units stand) nor frames - his orders are points, targets and widths
"""
import json
from pathlib import Path

import torch

from game.label import record as RC

FACES = 64
MISSING_UNIT = ["where: a cell of the unit's area (the tactician's frame) - the person sets no area",
                "the view: the unit's rows and its zone map are built round the area",
                "width: the network's 8 steps need the simulator's width limits of the unit (decode.width_limits)",
                "pace: the game's is_moving_fast is the state, the order's run flag is not recorded"]
MISSING_TACTIC = ["the group's zone: the person does not set it (the record puts it where the units stand)",
                  "the frames: the person orders points, targets and widths, not 48 / 64 / 80 m areas in the zone",
                  "the target of the group: the enemy group (Ctrl+G on the cards) is there, the zone is not"]


def operative(rec):
    f = rec.folder / "samples.pt"
    if not f.exists():
        return [], {"samples": 0}
    out, n = [], {"samples": 0, "human": 0, "keep": 0, "with_act": 0, "refused": 0, "rebuilt": 0}
    for s in RC.load(rec.folder):
        if s.get("level") != "operative":
            continue
        RC.rebuild(s)                                       # (raises when the input does not build)
        n["rebuilt"] += 1
        n["samples"] += 1
        n[s.get("source", "human")] = n.get(s.get("source", "human"), 0) + 1
        if s.get("act"):
            n["with_act"] += 1
        else:
            n["refused"] += 1
        out.append(s)
    return out, n


def strategist(rec):
    """The deployment's strategist sample (game/label/live_deploy.py), rebuilt once here (strategist.rebuild)."""
    f = rec.folder / "samples.pt"
    if not f.exists():
        return [], {"samples": 0}
    from game.label import strategist as LS
    out, n = [], {"samples": 0, "with_act": 0, "refused": 0}
    for s in RC.load(rec.folder):
        if s.get("level") != "strategist":
            continue
        LS.rebuild(s)                                       # (raises when the input does not build)
        n["samples"] += 1
        n["with_act" if s.get("act") else "refused"] += 1
        out.append(s)
    if out:
        n["missing"] = out[0].get("missing")
    return out, n


def unit_orders(rec):
    """One sample a change of a unit's order (module doc)."""
    slot_of = {n: i for i, n in enumerate(rec.names)}
    out = []
    seen = {}
    for sec in sorted(rec.orders):
        st = rec.states.get(sec) or {}
        for n, r in rec.orders[sec].items():
            if (r.get("men") or 0) <= 0 or r.get("r"):
                continue
            full = st.get(n) or {}
            ob = full.get("ob")
            key = (r.get("ox"), r.get("oz"), r.get("t") or "", ob)
            if seen.get(n) == key:
                continue
            first = n not in seen
            seen[n] = key
            if first:
                continue                                    # (the deployment's places are no order)
            t = r.get("t") or ""
            face = None
            if ob is not None and r.get("b") is not None:
                face = int(round(((float(ob) - float(r["b"])) % 360.0) / (360.0 / FACES))) % FACES
            out.append({"format": RC.FORMAT, "level": "unit_order", "mode": "live", "where": "game", "sec": sec,
                        "name": n, "slot": slot_of.get(n), "key": rec.key_of.get(n),
                        "row": {k: r.get(k) for k in ("x", "z", "b", "men", "mv", "f", "m", "ow")},
                        "order": {"kind": "attack" if t else "move", "point": [r.get("ox"), r.get("oz")],
                                  "target": t or None, "bearing": ob, "width_m": r.get("ow")},
                        "act_partial": {"target_slot": slot_of.get(t) if t else None, "face": face,
                                        "pace": 1 if r.get("f") else 0, "width_m": r.get("ow")},
                        "missing": MISSING_UNIT})
    return out


def export(rec, out):
    out = Path(out)
    ops, n_op = operative(rec)
    strat, n_st = strategist(rec)
    units = unit_orders(rec)
    torch.save(strat + ops + units, out / "samples.pt")
    info = {"strategist": n_st, "operative": n_op, "unit_order": {"samples": len(units),
                                              "with_target": sum(1 for u in units if u["act_partial"]["target_slot"] is not None),
                                              "with_face": sum(1 for u in units if u["act_partial"]["face"] is not None),
                                              "missing": MISSING_UNIT},
            "tactic": {"samples": 0, "missing": MISSING_TACTIC}}
    (out / "export.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info
