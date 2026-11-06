"""What the page draws (torch -> plain lists).

    snapshot(sess)    one second: every unit (slot, side, x, z, b, front, depth, men, hp, routing, alive, seen), the
                      operative's groups (members, target, zone with its tiles), the units' frames from their tacticians
    for_page(snap, sess)   the snapshot as the person may see it: the enemies hidden from my side (vis 0) left out
"""
from tools.nn.operative import step as SP
from tools.nn.unit import view as V

UNIT_KEYS = ("x", "z", "b", "front", "depth", "men", "hp", "r", "alive", "vis", "side")


def _r(v, k=1):
    return round(float(v), k)


def snapshot(sess):
    st = sess.b.st
    raw = V.raw(st, sess.P.sim["formation"]["spacing_m"])[0]
    I = V.I
    units = []
    for s in range(st.N):
        if float(raw[s, I["side"]]) <= 0:
            continue
        units.append([s, int(raw[s, I["side"]]), _r(raw[s, I["x"]]), _r(raw[s, I["z"]]), _r(raw[s, I["b"]]),
                      _r(raw[s, I["front"]]), _r(raw[s, I["depth"]]), int(raw[s, I["men"]]), _r(raw[s, I["hp"]], 3),
                      int(float(raw[s, I["r"]]) > 0), int(float(raw[s, I["alive"]]) > 0),
                      int(float(raw[s, I["vis"]]) > 0)])
    g = sess.groups
    tac = sess.tac
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
    return {"t": sess.sec, "units": units, "groups": groups, "frames": frames}


def for_page(snap):
    """The enemies my side does not see are left out (side 2 with vis 0)."""
    out = dict(snap)
    out["units"] = [u for u in snap["units"] if u[1] == 1 or u[11]]
    return out
