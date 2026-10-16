"""The shape of a unit from its soldiers' places (numpy, no torch): the same numbers for the game's probe recordings
(charge_probe 'probe_men': every man's place once a second, decimetres) and for the fighters lab's own men.

Per second of a pair of units (a, its enemy e), each a [n, 2] array of the men's places (m) and the unit's bearing (deg):
  near(r)   men of a within r m of their nearest enemy man (r = 3.15: the database's formed combat distance 2.5 m + a
            man's radius 0.65 - the reach the step replay and the probes count damage by)
  gap       the front's gap: the median nearest-enemy distance of a's men whose nearest enemy is within 6 m (the men in
            the fight; the probes: 1.1-1.4 m at the contact, ~3 m after 20 s)
  front_gap the median nearest-enemy distance of the 15 men nearest the enemy (the front rank's standing distance)
  depth_m   the men's depth along the unit's facing, (p97 - p3) / 0.94 (build/formdepth: ranks = depth / v + 1)
  width_m   the same across it
  wrap      the share of a's men level with or behind the enemy's front men, seen from the enemy (in its frame along its
            facing towards a: closer to its centre than its front men's p90 less 1.5 m) and beside its body - men that
            flowed round the enemy (the game's 'drop' in a long melee)
  contact   any man of a within 3.15 m of a man of e
Routers: along / across the flight (p90 - p10 of the men's places along the router's own displacement over the next
seconds, and across it): the mob's stretch."""
import math

import numpy as np

REACH_M = 3.15


def xy(flat):
    """[n, 2] metres from a probe's flat decimetre list."""
    a = np.asarray(flat, float)
    if a.size < 2:
        return np.zeros((0, 2))
    return a.reshape(-1, 2) / 10.0


def frame(p, cx, cz, bearing_deg):
    """(across, along) of points p [n, 2] in a frame at (cx, cz) facing bearing (deg: facing (sin b, cos b))."""
    b = math.radians(bearing_deg)
    fx, fz = math.sin(b), math.cos(b)
    rx, rz = p[:, 0] - cx, p[:, 1] - cz
    return rx * fz - rz * fx, rx * fx + rz * fz


def nearest(a, e):
    """[n] distance from each man of a to his nearest man of e (inf when e is empty)."""
    if len(a) == 0:
        return np.zeros(0)
    if len(e) == 0:
        return np.full(len(a), np.inf)
    d = np.sqrt(((a[:, None, :] - e[None, :, :]) ** 2).sum(-1))
    return d.min(1)


def extent(v):
    if len(v) < 3:
        return np.nan
    lo, hi = np.percentile(v, [3, 97])
    return float((hi - lo) / 0.94)


def pair_second(a, e, b_a, b_e, radii=(2.5, REACH_M)):
    """One second's numbers of unit a against its enemy e (module docstring)."""
    out = {}
    if len(a) == 0 or len(e) == 0:
        return out
    d = nearest(a, e)
    for r in radii:
        out[f"near_{r:g}"] = int((d <= r).sum())
    out["contact"] = bool((d <= REACH_M).any())
    close = d[d <= 6.0]
    out["gap"] = float(np.median(close)) if len(close) else np.nan
    srt = np.sort(d)
    out["front_gap"] = float(np.median(srt[:15])) if len(srt) >= 15 else np.nan
    ca, cz = a.mean(0)
    acr, alo = frame(a, ca, cz, b_a)
    out["depth_m"] = extent(alo)
    out["width_m"] = extent(acr)
    # wrap: in e's frame facing towards a's centre
    ce = e.mean(0)
    to_a = math.degrees(math.atan2(ca - ce[0], cz - ce[1]))
    ea, el = frame(e, ce[0], ce[1], to_a)
    aa, al = frame(a, ce[0], ce[1], to_a)
    front = np.percentile(el, 90)
    half_w = np.percentile(np.abs(ea), 95)
    beside = (al < front - 1.5) & (np.abs(aa) <= half_w + 4.0)
    out["wrap"] = float(beside.mean())
    out["men"] = len(a)
    return out


def router_second(p_now, p_later):
    """The routing mob's stretch along its flight and across it (m, p90 - p10): the flight is the mob's centre's way from
    now to `p_later` (its places some seconds on)."""
    if len(p_now) < 5 or len(p_later) < 5:
        return {}
    c0, c1 = p_now.mean(0), p_later.mean(0)
    v = c1 - c0
    n = float(np.hypot(*v))
    if n < 1.0:
        return {}
    ux, uz = v / n
    rx, rz = p_now[:, 0] - c0[0], p_now[:, 1] - c0[1]
    along = rx * ux + rz * uz
    across = rx * uz - rz * ux
    q = lambda w: float(np.percentile(w, 90) - np.percentile(w, 10))
    return {"along": q(along), "across": q(across), "men": len(p_now)}


def lane_series(lane, who="a", other="tg", until_s=240.0):
    """Per second after the first contact: [(t - contact, pair_second)] of lane's unit `who` against `other` from its
    'men' rows (charge_probe.load_run's lane: men [(t, a flat, tg flat)]; samples give the bearings)."""
    c = (lane.get("contacts") or {}).get(1)
    if c is None or not lane.get("men"):
        return []
    s = lane["samples"]
    ts = np.array([x[0] for x in s])
    col = {"a": 1, "tg": 2}
    bear = {w: np.array([np.nan if (x[col[w]] or {}).get("b") is None else float(x[col[w]]["b"]) for x in s])
            for w in ("a", "tg")}
    out = []
    for t, fa, ft in lane["men"]:
        if t < c - 0.01 or t > c + until_s:
            continue
        k = int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1))
        P = {"a": xy(fa), "tg": xy(ft)}
        ba, be = bear[who][k], bear[other][k]
        if not (np.isfinite(ba) and np.isfinite(be)):
            continue
        r = pair_second(P[who], P[other], ba, be)
        if r:
            out.append((t - c, r))
    return out
