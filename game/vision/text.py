"""The battle as text for an LLM (host python): the measures table, the events, the groups (the chain), and the
character map (ascii_frames, the exam's variant B).

lat / fwd are metres against our infantry line at the start (events.Facts.rel): lat + = our right, fwd + = towards the
enemy. Times are battle seconds.
"""
import math
import re

import numpy as np

from game.vision import events as E
from game.vision.record import ROLE_CODE

S = E.short


def tokens(text):
    """A rough count of Claude's text tokens (no tokenizer here): words ~4 letters a token, digits ~3 a token, every
    other symbol one token, runs of one symbol ~4 a token. Checked against nothing exact - an estimate."""
    n = 0
    for m in re.finditer(r"[A-Za-z]+|[А-Яа-яЁё]+|\d+|(.)\1{2,}|\S", text):
        s = m.group(0)
        if s[0].isdigit():
            n += math.ceil(len(s) / 3)
        elif s[0].isalpha():
            n += math.ceil(len(s) / (4 if s.isascii() else 2.5))
        elif len(s) > 2:
            n += math.ceil(len(s) / 4)
        else:
            n += 1
    return n


def _t(v):
    return "-" if v is None else str(v)


def _rout(fx, n):
    """The second the unit routed or died (a rout flagged only in the last frame counts: the army's collapse)."""
    s, T = fx.rout.get(n), fx.B.t[-1]
    return s if s is not None and (s < T or (fx.B.frames[-1].get(n) or {}).get("r")) else None


def _r(fx, n):
    s = _rout(fx, n)
    return "end" if s is not None and s >= fx.B.t[-1] else _t(s)


def header(fx):
    B = fx.B
    T, F = B.t[-1], B.frames
    men = [round(100 * sum((F[-1].get(n) or {}).get("men", 0) for n in S_) / max(1, sum(F[0][n]["men"] for n in S_)))
           for S_ in (fx.own, fx.en)]
    win = {B.own: "OURS", None: "none"}.get(B.winner, "ENEMY")
    ro = [sum(1 for n in S_ if _rout(fx, n) is not None) for S_ in (fx.own, fx.en)]
    return [f"BATTLE {B.name} ({B.kind}, map {B.map or 'flat'}): end {T}s, winner {win}; men left ours {men[0]}% "
            f"enemy {men[1]}%; routed/dead ours {ro[0]}/{len(fx.own)}, enemy {ro[1]}/{len(fx.en)}",
            f"axes: lat + = our right, fwd + = toward enemy, m from our start infantry line (front {fx.line_lat[0]:.0f}.."
            f"{fx.line_lat[1]:+.0f}); roles L lord S sniper A art M missile F fast X monster I inf"]


def ground_text(fx):
    """Heights at the start: our infantry, the enemy, the lowest ground on the way between their middles."""
    B = fx.B
    if not B.map:
        return ["GROUND flat (no captured map)"]
    _ground(B.map)
    h = _G[B.map][2]

    def hh(x, z):
        return float(h[min(max(int(round(z + 512)), 0), h.shape[0] - 1), min(max(int(round(x + 512)), 0), h.shape[1] - 1)])
    f = B.frames[0]
    own = [f[n] for n in fx.own if B.role[n] == "inf"] or [f[n] for n in fx.own]
    en = [f[n] for n in fx.en]
    mo = [sum(u[k] for u in own) / len(own) for k in ("x", "z")]
    me = [sum(u[k] for u in en) / len(en) for k in ("x", "z")]
    ho = float(np.median([hh(u["x"], u["z"]) for u in own]))
    he = float(np.median([hh(u["x"], u["z"]) for u in en]))
    way = [hh(mo[0] + (me[0] - mo[0]) * k / 40, mo[1] + (me[1] - mo[1]) * k / 40) for k in range(41)]
    lo = min(way)
    return [f"GROUND at start: enemy {he - ho:+.0f} m against our infantry; lowest between {lo - ho:+.0f} m at "
            f"{100 * way.index(lo) / 40:.0f}% of the way to the enemy"]


def seen(fx, u, v=100.0, muzzle=1.5, aim=0.8):
    """Share of the ground 120-240 m ahead of the unit (+-100 m across, the enemy's side of the view) that a low-arc
    bullet (v m/s, from muzzle m over the ground to aim m over it) reaches without hitting the ground (the snipers'
    'behind a hill' check of tmp/old-build/diag_forest/render.py)."""
    B = fx.B
    if not B.map:
        return 1.0
    _ground(B.map)
    h = _G[B.map][2]

    def hh(x, z):
        return h[np.clip(np.round(z + 512).astype(int), 0, h.shape[0] - 1), np.clip(np.round(x + 512).astype(int), 0, h.shape[1] - 1)]
    ok = n = 0
    g = 9.81
    for ahead in (120, 180, 240):
        for side in range(-100, 101, 25):
            tx = u["x"] + ahead * B.fwd[0] + side * B.rgt[0]
            tz = u["z"] + ahead * B.fwd[1] + side * B.rgt[1]
            L = math.hypot(tx - u["x"], tz - u["z"])
            y0 = float(hh(np.array(u["x"]), np.array(u["z"]))) + muzzle
            dh = float(hh(np.array(tx), np.array(tz))) + aim - y0
            A = g * L * L / (2 * v * v)
            tan = (L - math.sqrt(max(L * L - 4 * A * (A + dh), 0))) / (2 * A)
            k = np.linspace(0.02, 0.995, 120)
            xs = L * k
            y = y0 + tan * xs - g * (1 + tan * tan) / (2 * v * v) * xs * xs
            ok += not bool((hh(u["x"] + (tx - u["x"]) * k, u["z"] + (tz - u["z"]) * k) > y).any())
            n += 1
    return ok / n


def own_table(fx):
    B, F, T = fx.B, fx.B.frames, fx.B.t[-1]
    out = ["OURS name role | start lat,fwd forest% height | shots 0-60/0-150/all | melee | rout | orders/min | walk m",
           "(snipers: 'sees N%' - ground 120-240 m ahead their low-arc bullet reaches over the terrain)"]
    lv = _heights(fx)
    for n in sorted(fx.own, key=lambda n: fx.rel(F[0][n])[0]):
        u = F[0][n]
        la, fw = fx.rel(u)
        sh = fx.shots[n]
        o = fx.orders.get(n, [0, 1])
        shots = f"{_t(sh[60])}/{_t(sh[150])}/{_t(sh[T])}" if B.pp[n].get("missile") else "-"
        if B.role[n] == "sn":
            shots = f"sees {100 * seen(fx, u):.0f}% | " + shots
        out.append(f"{S(n)} {ROLE_CODE[B.role[n]]} | {la:+.0f},{fw:+.0f} f{_forest(fx, n, u)} h{lv[n]:+.0f} | {shots} | "
                   f"{_t(fx.melee.get(n))} | {_r(fx, n)} | {60 * o[0] / max(o[1], 1):.0f} | {fx.walk.get(n, 0):.0f}")
    return out


def enemy_table(fx):
    """The enemy's snipers, artillery, fast units, monsters and lord one a row; the rest a row a role."""
    B, F, T = fx.B, fx.B.frames, fx.B.t[-1]
    out = ["ENEMY name role | start lat,fwd | shots 0-150/all | melee | rout | men start>end"]
    rest = {}
    for n in sorted(fx.en, key=lambda n: fx.rel(F[0][n])[0]):
        r = B.role[n]
        if r in ("inf", "sh"):
            rest.setdefault(r, []).append(n)
            continue
        la, fw = fx.rel(F[0][n])
        sh = fx.shots[n]
        shots = f"{_t(sh[150])}/{_t(sh[T])}" if B.pp[n].get("missile") else "-"
        out.append(f"{S(n)} {ROLE_CODE[r]} | {la:+.0f},{fw:+.0f} | {shots} | {_t(fx.melee.get(n))} | "
                   f"{_r(fx, n)} | {F[0][n]['men']}>{(F[-1].get(n) or {}).get('men', 0)}")
    for r, ns in rest.items():
        lat = [fx.rel(F[0][n])[0] for n in ns]
        mel = [fx.melee[n] for n in ns if n in fx.melee]
        ro = [_rout(fx, n) for n in ns if _rout(fx, n) is not None]
        m0 = sum(F[0][n]["men"] for n in ns)
        m1 = sum((F[-1].get(n) or {}).get("men", 0) for n in ns)
        shots = f" shots 0-150 {sum(fx.shots[n][150] or 0 for n in ns)}" if r == "sh" else ""
        out.append(f"{ROLE_CODE[r]} x{len(ns)} lat {min(lat):+.0f}..{max(lat):+.0f}{shots}; first melee {_t(min(mel) if mel else None)}; "
                   f"routed {len(ro)}" + (f" first {min(ro)}" if ro else "") + f"; men {m0}>{m1}")
    return out


def events_text(fx):
    B, F, T = fx.B, fx.B.frames, fx.B.t[-1]
    out = ["EVENTS"]
    if fx.contact is not None:
        i = B.at(fx.contact)
        who = [n for n in fx.own if (F[i].get(n) or {}).get("m")]
        out.append(f"{fx.contact} contact: " + ", ".join(f"{S(a)} vs {S((F[i].get(a) or {}).get('t') or '?')}" for a in who))
    for s, n, m in fx.fast_first:
        la, fw = fx.rel(F[B.at(s)][n])
        v = F[B.at(s - 30)].get(m)
        came = f"; it was at {fx.rel(v)[0]:+.0f},{fx.rel(v)[1]:+.0f} 30 s before ({fx.flank(fx.rel(v)[0])})" if v else ""
        out.append(f"{s} FAST {S(m)} reaches our {ROLE_CODE[B.role[n]]} {S(n)} at lat {la:+.0f} ({fx.flank(la)}) "
                   f"fwd {fw:+.0f}{came}")
    for n in fx.own:
        if B.role[n] == "lord" and n in fx.melee:
            out.append(f"{fx.melee[n]} our lord in melee ({sum(1 for f in F if (f.get(n) or {}).get('m'))} s in all)")
    for side, S_, k in (("our", fx.own, 12), ("enemy", fx.en, 8)):
        rr = sorted((_rout(fx, n), n) for n in S_ if _rout(fx, n) is not None)
        early = [(s, n) for s, n in rr if s < T]
        last = len(rr) - len(early)
        out.append(f"{side} routs: " + (", ".join(f"{s} {S(n)}" for s, n in early[:k]) or "none before the end")
                   + (f" ... ({len(early)} in all)" if len(early) > k else "")
                   + (f"; {last} more only at the end ({T}s, the army's collapse)" if last else ""))
    mv = []
    for n in fx.own:                                              # where the routing went in their first 20 s
        s = _rout(fx, n)
        if s is None or s >= T or fx.dead.get(n) == s:
            continue
        a, b = F[B.at(s)].get(n), F[B.at(s + 20)].get(n)
        if B.alive(a) and B.alive(b):
            mv.append((fx.rel(b)[0] - fx.rel(a)[0], fx.rel(b)[1] - fx.rel(a)[1]))
    if mv:
        out.append(f"our routing units, first 20 s: {sum(1 for x in mv if x[1] < -10)} of {len(mv)} went back; mean "
                   f"fwd {sum(x[1] for x in mv) / len(mv):+.0f} m, sideways {sum(abs(x[0]) for x in mv) / len(mv):.0f} m")
    return out


def line_text(fx):
    B, T = fx.B, fx.B.t[-1]
    out = ["LINE front row = all but artillery within 40 m of the infantry's median fwd"]
    for s in sorted({0, fx.contact or 0, (fx.contact or 0) + 30}):
        if s > T:
            continue
        i = B.at(s)
        g, row = fx.gaps(i), fx.row(i)
        out.append(f"{s}s {len(row)} units lat {row[0][0]:+.0f}..{row[-1][0]:+.0f}; holes >{fx.gap_m:.0f} m: "
                   + ("; ".join(f"{w} m at lat {la:+d} ({fx.flank(la)}) {S(a)}|{S(b)}" for w, la, a, b in g) or "none")
                   if row else f"{s}s no front row")
    if fx.contact is not None:
        tr, n_ = fx.turned(B.at(fx.contact))
        out.append(f"{fx.contact}s facing: {len(tr)} of {n_} standing units turned >45 deg off their nearest enemy"
                   + (": " + " ".join(f"{S(n)} {d}" for n, d in tr[:8]) + (" ..." if len(tr) > 8 else "") if tr else ""))
    adv = [f"{s}s {fx.line_fwd(B.at(s)):+.0f}" for s in (30, 60, 120, 180) if s <= T and fx.line_fwd(B.at(s)) is not None]
    out.append("infantry line fwd: " + ", ".join(adv))
    c, by = fx.churn()
    out.append(f"ORDERS new per unit-minute (target change or point >8 m): {c} (" +
               ", ".join(f"{k} {v}" for k, v in by.items()) + "); a calm human ~1")
    return out


def groups_text(fx):
    B = fx.B
    if not B.groups:
        return ["GROUPS: none recorded (not the chain)"]
    verbs = {}
    for _, gs in B.groups:
        for g in gs:
            verbs[g["verb"]] = verbs.get(g["verb"], 0) + 1
    mv = sum(g.get("moved", 0) for _, gs in B.groups for g in gs)
    rf = sum(g.get("reformed", 0) for _, gs in B.groups for g in gs)
    out = [f"GROUPS (decided every 15 s, {len(B.groups)} decisions): verbs {verbs}; zone moved {mv}x, re-formed {rf}x; "
           f"our units inside their areas {fx.inside_share()}"]
    for g in B.groups[0][1]:
        la, fw = fx.rel({"x": g["cx"], "z": g["cz"]})
        out.append(f" g{g['g']} {g['verb']} zone {la:+.0f},{fw:+.0f}: " + " ".join(S(n) for n in g["units"]))
    return out


def table(fx):
    """The measures and the events (~1.6-2 k tokens): ground, ours, the enemy, events, line, orders, groups."""
    return "\n".join(header(fx) + ground_text(fx) + own_table(fx) + enemy_table(fx) + events_text(fx) + line_text(fx) + groups_text(fx))


def _forest(fx, n, u):
    """Forest share of the unit's box at the start, % (sim/maps ground)."""
    B = fx.B
    if not B.map:
        return 0
    g, names = _ground(B.map)
    if "forest" not in names:
        return 0
    code = names.index("forest")
    fr, dp = B.dims(n, u)
    b = math.radians(u.get("b") or 0)
    k = tot = 0
    for a in np.linspace(-fr / 2, fr / 2, 7):
        for c in np.linspace(-dp / 2, dp / 2, 4):
            x = u["x"] - a * math.cos(b) + c * math.sin(b)
            z = u["z"] + a * math.sin(b) + c * math.cos(b)
            k += g[min(max(int(round(z + 512)), 0), g.shape[0] - 1), min(max(int(round(x + 512)), 0), g.shape[1] - 1)] == code
            tot += 1
    return round(100 * k / tot)


_G = {}


def _ground(name):
    if name not in _G:
        import json

        from sim.maps import MAPS
        d = np.load(MAPS / name / "heights.npz")
        names = json.loads((MAPS / name / "meta.json").read_text(encoding="utf-8")).get("ground_names") or []
        _G[name] = (d["ground"] if "ground" in d.files else np.zeros((1, 1), int), names, d["height"])
    return _G[name][:2]


def _heights(fx):
    """{unit: height at the start - the infantry's median height, m}."""
    B = fx.B
    if not B.map:
        return {n: 0 for n in B.role}
    _ground(B.map)
    h = _G[B.map][2]

    def hh(u):
        return float(h[min(max(int(round(u["z"] + 512)), 0), h.shape[0] - 1), min(max(int(round(u["x"] + 512)), 0), h.shape[1] - 1)])
    base = float(np.median([hh(B.frames[0][n]) for n in fx.own if B.role[n] == "inf"] or [0]))
    return {n: hh(B.frames[0][n]) - base for n in B.role if n in B.frames[0]}


def ascii_frames(fx, frames, cols=48, rows=18):
    """Character maps of the chosen frames (one crop): terrain '.' plain, '^' high (top quarter), '_' low (bottom
    quarter), 'T' forest, '#' cannot stand; ours upper case by role (I inf, S, M, A, F, X, L), the enemy lower case;
    in melee: ours '*', enemy '+'; routing: ours '!', enemy '?'; a group's zone (the chain) its digit."""
    from game.vision import draw
    B = fx.B
    l0, l1, f0, f1 = draw.crop(fx, [s for s, _ in frames])
    cw, rh = (l1 - l0) / cols, (f1 - f0) / rows
    base = [["."] * cols for _ in range(rows)]
    if B.map:
        g, names = _ground(B.map)
        h = _G[B.map][2]
        hs = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                la, fw = l0 + (c + 0.5) * cw, f1 - (r + 0.5) * rh
                x, z = la * B.rgt[0] + fw * B.fwd[0], la * B.rgt[1] + fw * B.fwd[1]
                iz, ix = min(max(int(round(z + 512)), 0), 1024), min(max(int(round(x + 512)), 0), 1024)
                hs[r, c] = h[iz, ix]
                if names and g[iz, ix] == names.index("forest"):
                    base[r][c] = "T"
        q1, q3 = np.percentile(hs, 25), np.percentile(hs, 75)
        for r in range(rows):
            for c in range(cols):
                if base[r][c] == ".":
                    base[r][c] = "^" if hs[r, c] >= q3 else ("_" if hs[r, c] <= q1 else ".")
    out = [f"MAP {B.name}: {cols}x{rows} chars, 1 char = {cw:.0f} m across x {rh:.0f} m deep; top = toward enemy, "
           f"left = our left; '.' plain '^' high '_' low 'T' forest; ours UPPER (I S M A F X L), enemy lower; "
           f"melee ours '*' enemy '+'; routing ours '!' enemy '?'; digit = a group's zone"]
    for sec, labels in frames:
        grid = [row[:] for row in base]
        f = B.frames[B.at(sec)]
        gs = [g for t, g in B.groups if t <= sec]
        if gs:
            for g in gs[-1]:
                la, fw = B.view(g["cx"], g["cz"])
                c, r = int((la - l0) / cw), int((f1 - fw) / rh)
                if 0 <= r < rows and 0 <= c < cols:
                    grid[r][c] = str(g["g"] % 10)
        for n, u in f.items():
            if not B.alive(u):
                continue
            own = B.side[n] == B.own
            ch = ROLE_CODE[B.role[n]]
            ch = ch if own else ch.lower()
            if u.get("r"):
                ch = "!" if own else "?"
            elif u.get("m"):
                ch = "*" if own else "+"
            la, fw = B.view(u["x"], u["z"])
            wid = B.dims(n, u)[0]
            k = max(1, int(round(wid / cw)))
            c0, r = int((la - l0) / cw - k / 2 + 0.5), int((f1 - fw) / rh)
            if 0 <= r < rows:
                for c in range(c0, c0 + k):
                    if 0 <= c < cols:
                        grid[r][c] = ch
        st = [sum(1 for n, u in f.items() if B.alive(u) and not u.get("r") and (B.side[n] == B.own) == o)
              for o in (True, False)]
        out.append(f"t={sec}s {', '.join(labels)} | standing ours {st[0]} enemy {st[1]}")
        out += ["".join(row) for row in grid]
    return "\n".join(out)
