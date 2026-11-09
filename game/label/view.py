"""What the page draws (torch -> plain lists).

    units_info(sess)   per slot, once a battle: key, side, the Russian name (data/label_icons/names_ru.json: the game's
                       own, game/label/gameart.py), the kind (lord / infantry / shooters / cavalry / monster / artillery),
                       men at the start, the men's spacing (across, along), whether a card picture exists, gold
    snapshot(sess)     one second: every unit [slot, side, x, z, b, front, depth, men, hp, routing, alive, seen, ammo
                       (share left), melee, moving, firing], the operative's groups (members, target, zone with its
                       tiles), the units' frames from their tacticians; men: with the fighters lab each man's place
                       {slot: [x0, z0, x1, z1, ...]} (decimetres), else None (the page draws the formation's files and
                       ranks from front, depth and men)
    for_page(snap)     the snapshot as the person may see it: the enemies hidden from my side (vis 0) left out
"""
import json
from functools import lru_cache

import game as project
from game.label import pool as PL

ICONS = project.DATA / "label_icons"


@lru_cache(maxsize=1)
def names_ru():
    f = ICONS / "names_ru.json"
    if not f.exists():
        return {}
    return {k: v for k, v in json.loads(f.read_text(encoding="utf-8"))["names"].items() if v}


def kind_of(key, general=False, category=""):
    if general or "_cha_" in key:
        return "lord"
    if "_mon_" in key:
        return "monster"
    if category == "artillery" or "_art_" in key:
        return "artillery"
    if category == "cavalry" or "_cav_" in key:
        return "cavalry"
    if category == "inf_ranged":
        return "shooters"
    return "infantry"


def unit_entry(key):
    """{name, kind, cost, men, range, icon} of a unit key (any faction of the pool)."""
    p = PL.passports().get(key, {})
    e = next((u for f in PL.factions().values() for u in f["units"] if u["key"] == key), {})
    return {"key": key, "name": names_ru().get(key, key), "kind": kind_of(key, e.get("general", False),
                                                                        p.get("category", "")),
            "cost": p.get("multiplayer_cost"), "men": p.get("men"), "range": e.get("range"),
            "icon": f"/icons/{key}.png" if (ICONS / f"{key}.png").exists() else None,
            # the formation's numbers for drawing it before the battle (sim/geometry.dims: files = width / h, ranks
            # = men / files x v; artillery one rank; a single body 2 radii)
            "h": float((p.get("spacing") or {}).get("h") or 1.5), "v": float((p.get("spacing") or {}).get("v") or 1.5),
            "radius": float(p.get("radius_m") or 1.0), "one_rank": bool(p.get("engine")), "width": e.get("width")}


def units_info(sess):
    st = sess.b.st
    u = st.u
    out = []
    for s in range(st.N):
        if int(u["side"][0, s]) <= 0:
            out.append(None)
            continue
        key = st.keys[0][s]
        e = unit_entry(key)
        e.update({"slot": s, "side": int(u["side"][0, s]), "men0": int(u["men0"][0, s]),
                  "h": round(float(u["sp_h"][0, s]), 2) if "sp_h" in u and float(u["sp_h"][0, s]) > 0 else 1.5,
                  "v": round(float(u["sp_v"][0, s]), 2) if "sp_v" in u and float(u["sp_v"][0, s]) > 0 else 1.5,
                  "ammo0": float(u["ammo0"][0, s])})
        out.append(e)
    return out


def _r(v, k=1):
    return round(float(v), k)


def men_of(st):
    """{slot: [x, z, ...] in decimetres} of the living men - the fighters lab's (not carried over): None."""
    return None


def snapshot(sess):
    from net.unit import view as V
    st = sess.b.st
    raw = V.raw(st, sess.P.sim["formation"]["spacing_m"])[0]
    I = V.I
    u = st.u
    units = []
    for s in range(st.N):
        if float(raw[s, I["side"]]) <= 0:
            continue
        a0 = float(u["ammo0"][0, s])
        units.append([s, int(raw[s, I["side"]]), _r(raw[s, I["x"]]), _r(raw[s, I["z"]]), _r(raw[s, I["b"]]),
                      _r(raw[s, I["front"]]), _r(raw[s, I["depth"]]), int(round(float(u["men"][0, s]))),
                      _r(raw[s, I["hp"]], 3), int(float(raw[s, I["r"]]) > 0), int(float(raw[s, I["alive"]]) > 0),
                      int(float(raw[s, I["vis"]]) > 0), _r(float(u["a"][0, s]) / a0, 2) if a0 > 0 else None,
                      int(float(raw[s, I["m"]]) > 0), int(float(raw[s, I["mv"]]) > 0), int(float(raw[s, I["fire"]]) > 0)])
    groups, frames = groups_frames(sess.groups, sess.tac)
    return {"t": sess.sec, "units": units, "groups": groups, "frames": frames, "men": men_of(st)}


def groups_frames(g, tac):
    """(groups, frames) of a snapshot: the operative's groups (members, target, zone with its tiles) and the units' frames
    from their tacticians (operative/lower.Tacticians) - the simulator's battle and the game's chain alike."""
    from net.operative import step as SP
    groups = []
    for k in range(g.G):
        if not bool(g.on[0, k]):
            continue
        front, depth = SP.size(g.n0[0, k:k + 1], g.line[0, k:k + 1])
        row = k                                           # (B = 1: the tacticians' row of group k)
        tl = []
        if bool(tac.active[row]):
            r = tac.tiles(tac.bi.new_tensor([row])).rects()
            for t in range(r["x"].shape[1]):
                if float(r["front"][0, t]) > 0:
                    tl.append([_r(r["x"][0, t]), _r(r["z"][0, t])])
        groups.append({"g": k, "members": (g.member[0] == k).nonzero()[:, 0].tolist(),
                       "target": g.target[0, k].nonzero()[:, 0].tolist(), "cx": _r(g.cx[0, k]), "cz": _r(g.cz[0, k]),
                       "heading": _r(g.heading[0, k]), "front": _r(front[0]), "depth": _r(depth[0]),
                       "line": bool(g.line[0, k]), "tiles": tl, "tile_m": 80.0})
    w = tac.areas()
    frames = []
    for row in range(tac.R):
        for s in w["on"][row].nonzero()[:, 0].tolist():
            frames.append([s, row, _r(w["x"][row, s]), _r(w["z"][row, s]), _r(w["b"][row, s]), _r(w["front"][row, s]),
                           _r(w["depth"][row, s])])
    return groups, frames


def for_page(snap):
    """The enemies my side does not see are left out (side 2 with vis 0), their men too."""
    out = dict(snap)
    keep = {u[0] for u in snap["units"] if u[1] == 1 or u[11]}
    out["units"] = [u for u in snap["units"] if u[0] in keep]
    if snap.get("men") is not None:
        out["men"] = {k: v for k, v in snap["men"].items() if k in keep}
    return out


def for_replay(snaps, every=5):
    """The snapshots as replay.json keeps them: the men only every `every` seconds (their size)."""
    return [s if s.get("men") is None or s["t"] % every == 0 else {**s, "men": None} for s in snaps]


__all__ = ["snapshot", "groups_frames", "for_page", "units_info", "unit_entry", "kind_of", "names_ru"]
