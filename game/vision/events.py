"""What happened in a battle, as events and measures (host python): the frames' times for the strip and the text.

The view's coordinates (record.Battle.view): lat - metres to our right, fwd - metres towards the enemy; both against our
infantry line at the start (lat from the middle of its front, fwd from its median rank).
"""
import math
import statistics

GAP_M = 50.0          # a hole: neighbours of the front row (within ROW_M of the infantry's median fwd) this far apart
ROW_M = 40.0
FAST_REACH_M = 30.0   # a fast enemy in melee this near our missile unit in melee (or targeting each other)
ORDER_M = 8.0         # a new order: the target changes or the order point moves this far (the person's ~1 a minute)


def short(n):
    return n.removeprefix("own_").removeprefix("enemy_")


class Facts:
    def __init__(self, B, gap_m=GAP_M):
        self.B, self.gap_m = B, gap_m
        F = B.frames
        own0 = [B.view(u["x"], u["z"]) for n, u in F[0].items() if B.side[n] == B.own and B.role[n] == "inf"]
        if not own0:
            own0 = [B.view(u["x"], u["z"]) for n, u in F[0].items() if B.side[n] == B.own]
        lo, hi = min(p[0] for p in own0), max(p[0] for p in own0)
        self.lat0, self.fwd0 = (lo + hi) / 2, statistics.median(p[1] for p in own0)
        self.line_lat = (lo - self.lat0, hi - self.lat0)
        self.own = [n for n in B.role if B.side[n] == B.own]
        self.en = [n for n in B.role if B.side[n] != B.own]
        self._scan()

    def rel(self, u):
        r, f = self.B.view(u["x"], u["z"])
        return r - self.lat0, f - self.fwd0

    def flank(self, lat):
        lo, hi = self.line_lat
        third = (hi - lo) / 3
        if lat < lo - 15:
            return "left, beyond the line"
        if lat > hi + 15:
            return "right, beyond the line"
        return "left" if lat < lo + third else ("right" if lat > hi - third else "centre")

    def _scan(self):
        B, F, T = self.B, self.B.frames, self.B.t
        self.melee, self.rout, self.dead, self.contact = {}, {}, {}, None
        self.fast_hits, self.fire_s, self.walk, self.orders = [], {}, {}, {}
        self.shots = {}
        units = list(B.role)
        for i, s in enumerate(T):
            f = F[i]
            for n in units:
                u = f.get(n)
                if not u:
                    continue
                if (u.get("men") or 0) <= 0:
                    self.dead.setdefault(n, s)
                    self.rout.setdefault(n, s)
                    continue
                if u.get("m"):
                    self.melee.setdefault(n, s)
                    if self.contact is None and B.side[n] == B.own:
                        self.contact = s
                if u.get("r"):
                    self.rout.setdefault(n, s)
                if u.get("fire"):
                    self.fire_s[n] = self.fire_s.get(n, 0) + 1
                if i:
                    p = F[i - 1].get(n)
                    if p:
                        self.walk[n] = self.walk.get(n, 0.0) + math.hypot(u["x"] - p["x"], u["z"] - p["z"])
                        new = ((u.get("t") or "") != (p.get("t") or "") and (u.get("t") or "") != "") or \
                            math.hypot((u.get("ox") or 0) - (p.get("ox") or 0), (u.get("oz") or 0) - (p.get("oz") or 0)) > ORDER_M
                        c = self.orders.setdefault(n, [0, 0])
                        c[0] += new
                        c[1] += 1
            for n in self.own:                                    # fast enemies on our missile units
                if B.role[n] not in ("sn", "sh", "art"):
                    continue
                u = f.get(n)
                if not B.alive(u) or not u.get("m"):
                    continue
                for m in self.en:
                    v = f.get(m)
                    if B.role[m] != "fast" or not B.alive(v):
                        continue
                    d = math.hypot(u["x"] - v["x"], u["z"] - v["z"])
                    if (v.get("m") and (v.get("t") == n or d < FAST_REACH_M)) or u.get("t") == m:
                        self.fast_hits.append((s, n, m))
        first = {}
        for s, n, m in self.fast_hits:
            first.setdefault(n, (s, m))
        self.fast_first = sorted((s, n, m) for n, (s, m) in first.items())
        a0 = {n: (F[0].get(n) or {}).get("a") or 0 for n in units}
        for n in units:
            by = {}
            for lim in (60, 150, T[-1]):
                i = B.at(min(lim, T[-1]))
                by[lim] = a0[n] - ((F[i].get(n) or {}).get("a") or 0) if B.alive(F[i].get(n)) or lim == T[-1] else None
            self.shots[n] = by
        # the shots of a unit gone: its last alive ammunition
        for n in units:
            last = None
            for i in range(len(T)):
                if B.alive(F[i].get(n)):
                    last = F[i][n].get("a") or 0
            if last is not None:
                self.shots[n][T[-1]] = a0[n] - last

    def churn(self, side=None):
        """New orders a unit a minute (alive units' seconds), all and by role."""
        side = self.B.own if side is None else side
        tot, by = [0, 0], {}
        for n, (a, b) in self.orders.items():
            if self.B.side[n] != side:
                continue
            tot[0] += a
            tot[1] += b
            r = by.setdefault(self.B.role[n], [0, 0])
            r[0] += a
            r[1] += b
        return round(60 * tot[0] / max(tot[1], 1), 1), {k: round(60 * a / max(b, 1), 1) for k, (a, b) in by.items()}

    def row(self, i):
        """Our front row at frame i: [(lat, fwd, name, front_m)] standing, not artillery, within ROW_M of the infantry."""
        B, f = self.B, self.B.frames[i]
        p = [(self.rel(f[n]), n) for n in self.own
             if B.role[n] != "art" and B.alive(f.get(n)) and not f[n].get("r")]
        inf = [q[1] for q, n in p if B.role[n] == "inf"] or [q[1] for q, n in p]
        if not inf:
            return []
        med = statistics.median(inf)
        return sorted((q[0], q[1], n, B.dims(n, f[n])[0]) for q, n in p if abs(q[1] - med) < ROW_M)

    def gaps(self, i):
        """Holes of the front row at frame i: [(width m, lat of the hole's middle, left unit, right unit)]."""
        r = self.row(i)
        out = []
        for (l1, _, n1, w1), (l2, _, n2, w2) in zip(r, r[1:]):
            g = (l2 - w2 / 2) - (l1 + w1 / 2)
            if g > self.gap_m:
                out.append((round(g), round((l1 + l2) / 2), n1, n2))
        return out

    def turned(self, i, deg=45.0):
        """Our standing units at frame i facing more than deg off their nearest standing enemy: [(name, off deg)]."""
        B, f = self.B, self.B.frames[i]
        ens = [v for m, v in f.items() if B.side[m] != B.own and B.alive(v) and not v.get("r")]
        out, n_ = [], 0
        for n in self.own:
            u = f.get(n)
            if not B.alive(u) or u.get("r") or not ens:
                continue
            v = min(ens, key=lambda v: math.hypot(v["x"] - u["x"], v["z"] - u["z"]))
            a = math.degrees(math.atan2(v["x"] - u["x"], v["z"] - u["z"]))
            off = abs(((u.get("b") or 0) - a + 180) % 360 - 180)
            n_ += 1
            if off > deg:
                out.append((n, round(off)))
        return out, n_

    def line_fwd(self, i):
        B, f = self.B, self.B.frames[i]
        p = [self.rel(f[n])[1] for n in self.own if B.role[n] == "inf" and B.alive(f.get(n)) and not f[n].get("r")]
        return statistics.median(p) if p else None

    def group_moves(self):
        """The operative's decisions: [(sec, group, verb, moved, reformed, n units, zone lat / fwd)]."""
        out = []
        for sec, gs in self.B.groups:
            for g in gs:
                r, f = self.B.view(g["cx"], g["cz"])
                out.append((sec, g["g"], g["verb"], g.get("moved"), g.get("reformed"), len(g["units"]),
                            round(r - self.lat0), round(f - self.fwd0)))
        return out

    def inside_share(self):
        """Share of our units' seconds inside the area the chain gave them (companion chain.inside; None: no chain)."""
        v = [x for d in self.B.inside.values() for x in d.values()]
        return round(sum(v) / len(v), 2) if v else None

    def timeline(self):
        """[(sec, priority, text)]: the battle's moments (lower priority first when frames are few)."""
        B, ev = self.B, []
        ev.append((0, 0, "start"))
        if self.contact is not None:
            ev.append((self.contact, 1, "contact"))
        orouts = sorted((s, n) for n, s in self.rout.items() if n in self.own and s < B.t[-1])
        if orouts:
            ev.append((orouts[0][0], 1, f"rout1 {short(orouts[0][1])}"))
        for s, n, m in self.fast_first[:1]:
            ev.append((s, 1, f"fast>{short(n)}"))
        lords = [n for n in self.own if B.role[n] == "lord"]
        if lords and lords[0] in self.melee:
            ev.append((self.melee[lords[0]], 2, "lord melee"))
        if self.contact is not None:
            i0 = B.at(self.contact)
            for i in range(i0, len(B.t)):
                if B.t[i] > self.contact + 90:
                    break
                g = self.gaps(i)
                if g and max(x[0] for x in g) >= 1.5 * self.gap_m:
                    ev.append((B.t[i], 3, f"hole {max(x[0] for x in g)} m"))
                    break
        for sec, gs in B.groups:
            if sec > 0 and any(g.get("reformed") for g in gs):
                ev.append((sec, 3, "regroup"))
                break
        ev.append((B.t[-1], 0, "end"))
        return sorted(ev)


def pick_frames(fx, n=8, every=45):
    """Frame seconds for the strip: the timeline's moments (merged within 8 s), then evenly in the longest spans."""
    B = fx.B
    ev = fx.timeline()
    chosen = {}
    for s, pr, txt in sorted(ev, key=lambda e: (e[1], e[0])):
        near = [c for c in chosen if abs(c - s) <= 8]
        if near:
            chosen[near[0]].append(txt)
        elif len(chosen) < n:
            chosen[s] = [txt]
    while len(chosen) < n:
        ts = sorted(chosen)
        span = max(zip(ts, ts[1:]), key=lambda p: p[1] - p[0])
        if span[1] - span[0] < min(every, 20):
            break
        mid = B.t[B.at((span[0] + span[1]) / 2)]
        if mid in chosen:
            break
        chosen[mid] = []
    return sorted(chosen.items())
