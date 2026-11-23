"""Held binds (tmp/engine/design_merged.md section 4): per held axis, whether a battle's held-out value is what the axis
names in it - measured live after snapping or at the event (O0-1's sally_m), not trusted from the draw; unbound
battles are listed and excluded; over 10 % unbound halts.

    of(specs, bind, live=None)   {axis: {battles, bound, unbound [battle index], share_unbound}} and the excluded set;
                bind(spec) -> True / False / None (no held axis) - the card's spec-level check (scenes.side_binds,
                scenes_o1.binds); live {battle index: {axis: (value, want, tolerance)}} - a measured value overrides:
                bound only when |value - want| <= tolerance (exam_o1.held_binds: the gap at the level's first tick
                within 20 m of the held value)
    check(res)   halts (BindHalt) when an axis's unbound share > MAX_UNBOUND
"""
MAX_UNBOUND = 0.10        # (design section 4: halt over 10 %)


class BindHalt(RuntimeError):
    pass


def of(specs, bind, live=None):
    live = live or {}
    out, excluded = {}, set()
    for i, sp in enumerate(specs):
        b = bind(sp)
        if b is None:
            continue
        ax = str(sp.get("held"))
        ok = bool(b)
        if i in live and ax in live[i]:
            v, want, tol = live[i][ax]
            ok = ok and v is not None and abs(float(v) - float(want)) <= float(tol)
        r = out.setdefault(ax, {"battles": 0, "bound": 0, "unbound": []})
        r["battles"] += 1
        r["bound"] += int(ok)
        if not ok:
            r["unbound"].append(i)
            excluded.add(i)
    for r in out.values():
        r["share_unbound"] = round(len(r["unbound"]) / max(r["battles"], 1), 4)
    return {"axes": out, "excluded": sorted(excluded)}


def check(res, max_unbound=MAX_UNBOUND):
    bad = {a: r["share_unbound"] for a, r in res["axes"].items() if r["share_unbound"] > max_unbound}
    if bad:
        raise BindHalt(f"held axes unbound over {max_unbound}: {bad}")
    return res
