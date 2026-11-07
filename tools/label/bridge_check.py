"""The bridge's decisions through today's operative (torch): which of its bans stop them, and the ones the live record's
own windows hit (the person's Ctrl+G groups). An analysis for changing the level - the operative's code is not touched.

    check(rec, wins)    per bridge window with a live sample (its raw input): the previous window's inferred groups as
                        the operative's last groups (live.operative_groups), the window's groups as the edit (members, the
                        seed when the enemy is seen, the goal at the area's centre, a line when the area is twice as wide
                        as deep), human.op_act + op_walk + op_parcels -> ok or the errors, named by the ban
    ctrlg(rec)          the same for the person's Ctrl+G groups as the live recorder takes them: why its windows failed
    python -m tools.label.bridge_check build/label/<date>_live   -> <record>/digest/bridge_check.json (with what-ifs:
                        the same decisions with one ban lifted in memory - zone_sep_m 0, spread_m off)
"""
import json
import sys
from collections import Counter
from pathlib import Path

import torch

from tools.label import bridge_infer as BI
from tools.label import digest as DG
from tools.label import human as HU
from tools.label import record as RC
from tools.label.live import operative_groups
from tools.nn.operative import cfg as ocfg
from tools.nn.operative import inputs as OIN
from tools.nn.operative import net as ONN

BANS = (("group_number", "номер 0-"), ("unit_in_no_group", "есть группа"), ("comp_full_or_far", "не может войти в группу"),
        ("zone_sep_or_field", "сюда поставить нельзя"), ("turn", "Поворот зоны"), ("target_not_seen", "не видимый враг"),
        ("target_not_compact", "слишком широкая"), ("target_split", "Цель разорвана"),
        ("zone_far_from_group", "слишком далеко от группы"), ("walk_mismatch", "выбор закрыт правилами"),
        ("too_many_in_group", "можно не больше"))


def ban_of(msg):
    return next((k for k, p in BANS if p in msg), "other: " + msg[:60])


def _samples(rec):
    return {int(s["sec"]): s for s in RC.load(rec.folder) if s.get("level") == "operative"}


def _ctx(s, groups):
    raw = s["raw"]
    from tools.nn.operative import frame as FR
    from tools.nn.operative import groups as GR  # noqa: F401
    a = s["axes"]
    axes = FR.Axes(a["cx"], a["cz"], a["heading"], dict(a["field"]))
    inp = OIN.build(raw, s["card"], s["side"], axes, groups, s["command"], bounds=s["bounds"], K=s["K"])
    return inp, ONN.ctx_of(inp, raw, axes, groups), axes


def fill(groups, rows, raw, slot_of, cap=10):
    """The groups with every own unit of the window's rows in one: a unit the bridge left out (routing, shattered, or
    dying in the window - the operative's rows keep every living own unit) joins the nearest group with room."""
    from tools.nn.tactic import raw as TR
    I = TR.I
    out = {k: list(v) for k, v in groups.items()}
    have = {n for v in out.values() for n in v}
    pos = lambda n: (float(raw[0, slot_of[n], I["x"]]), float(raw[0, slot_of[n], I["z"]]))  # noqa: E731
    for n in rows:
        if n in have:
            continue
        best = None
        for k, v in out.items():
            if len(v) >= cap or not v:
                continue
            cx = sum(pos(m)[0] for m in v) / len(v)
            cz = sum(pos(m)[1] for m in v) / len(v)
            d = (cx - pos(n)[0]) ** 2 + (cz - pos(n)[1]) ** 2
            if best is None or d < best[0]:
                best = (d, k)
        if best is not None:
            out[best[1]].append(n)
            have.add(n)
    return out


def _edit(rec, w, slot_of, seen, groups):
    out = []
    for k, names in sorted(groups.items()):
        ar = w.area.get(k) or {}
        seed = w.seed.get(k)
        t = {"seed": slot_of[seed] if seed in slot_of and slot_of[seed] in seen else -1,
             "goal": [ar["cx"], ar["cz"]] if ar else None,
             "form": "line" if ar and ar["front"] > 2 * ar["depth"] else "square", "turn": 0, "verb": "hold",
             "pace": "any"}
        out.append({"g": int(k) - 1, "members": [slot_of[n] for n in names], "task": t})
    return {"groups": out}


def run_walk(edit, inp, ctx, axes, raw):
    try:
        act, plan, _ = HU.op_act(edit, ctx, axes)
        out, errs = HU.op_walk(inp, ctx, act, plan)
        _, _, perrs = HU.op_parcels(out, raw, axes)
        return act, errs + perrs
    except HU.Refused as e:
        return None, e.errors


def check(rec, wins, samples=None):
    """samples: a list to append the windows' operative samples to (the record's format: the raw input, the bridge's
    previous groups, the walk's act or {} with the errors) - tools/label/digest_set.py."""
    S = _samples(rec)
    slot_of = {n: i for i, n in enumerate(rec.names)}
    G = int(ocfg()["groups"])
    rows, bans, ok, prev = [], Counter(), 0, None
    for w in wins:
        s = S.get(w.sec)
        if s is None:
            prev = w
            continue
        raw = s["raw"]
        # (the previous window's inferred groups, as the live recorder builds the person's: own k -> group k - 1)
        from tools.nn.operative import frame as FR
        a = s["axes"]
        axes = FR.Axes(a["cx"], a["cz"], a["heading"], dict(a["field"]))
        names = rec.names
        probe = operative_groups({}, {}, slot_of, raw, axes, G)
        _, ctx0, _ = _ctx(s, probe)
        own_rows = [names[sl] for sl, o in zip(HU.op_rows(ctx0)[0], HU.op_rows(ctx0)[1]) if o]
        before = fill(prev.groups, own_rows, raw, slot_of) if prev else {}
        now = fill(w.groups, own_rows, raw, slot_of)
        prev_g = operative_groups({int(k): v for k, v in before.items()},
                                  {int(k): v for k, v in (prev.targets if prev else {}).items()}, slot_of, raw, axes, G)
        inp, ctx, axes = _ctx(s, prev_g)
        seen = {sl for sl, e in zip(*HU.op_rows(ctx)[::2]) if e}
        edit = {"keep": True} if (w.decision == "keep" and prev is not None and before == now) else             _edit(rec, w, slot_of, seen, now)
        act, errs = run_walk(edit, inp, ctx, axes, raw)
        kinds = sorted({ban_of(e) for e in errs})
        bans.update(kinds)
        ok += not errs
        rows.append({"sec": w.sec, "decision": w.decision, "ok": not errs, "bans": kinds, "errors": errs[:3]})
        if samples is not None:
            samples.append({"format": RC.FORMAT, "level": "operative", "mode": "bridge", "where": "game", "sec": w.sec,
                            "raw": raw, "groups": RC.groups_dict(prev_g), "card": s["card"], "side": s["side"],
                            "command": s["command"], "axes": s["axes"], "bounds": s["bounds"], "K": s["K"],
                            "act": act if (act is not None and not errs) else {},
                            "source": "keep" if edit.get("keep") else "human", "errors": errs,
                            "bridge": {"groups": now, "targets": w.targets, "seed": w.seed, "area": w.area,
                                       "why": w.why}})
        prev = w
    return {"windows": len(rows), "ok": ok, "bans": dict(bans), "rows": rows}


def ctrlg(rec):
    """The live record's own windows: why they were refused (its errors, by ban) and the same groups re-checked here."""
    bans = Counter()
    for w in rec.windows:
        for e in set(w.get("errors") or []):
            bans[ban_of(e)] += 1
    cg = BI.ctrlg_of(rec)
    G = int(ocfg()["groups"])
    return {"windows": len(rec.windows), "refused": sum(1 for w in rec.windows if w.get("errors")),
            "bans": dict(bans), "ctrlg_groups": len(cg), "operative_groups": G,
            "over_G": {str(k): v for k, v in cg.items() if k > G}}


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    folder = Path((argv or sys.argv[1:])[0])
    rec = DG.Record(folder)
    wins = BI.infer(rec)
    torch.set_num_threads(4)
    res = {"bridge": check(rec, wins), "ctrlg": ctrlg(rec)}
    # what-if (an analysis, in memory only): the same decisions with a ban of today's operative lifted at a time
    whatif = {}
    for key, val in (("zone_sep_m", 0.0), ("spread_m", 1e6)):
        old = ocfg()[key]
        ocfg()[key] = val
        try:
            r = check(rec, wins)
            whatif[f"{key}={val}"] = {"ok": r["ok"], "bans": r["bans"]}
        finally:
            ocfg()[key] = old
    res["whatif"] = whatif
    out = rec.folder / "digest"
    out.mkdir(exist_ok=True)
    (out / "bridge_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in res.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
