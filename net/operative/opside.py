"""The scripted operative of the game probe plan `holder` (tmp/op1/design_merged.md, 'The kill check' and its
amendments; the review tmp/astra/holder_review/out/REVIEW.md): over the tactic level's one side (net/tactic/side.Side)
it moves the own group's zone through the existing path lower.Tacticians.apply(groups, task, moved=True) and leads the
hidden lord. The numbers: net/config/operative.json o1_probe (ASSUMPTIONS of the kill check). The simulator's side and
the bridge's companion (bridge/companion/tactic_chain.py, BAI_OP_SCRIPT=holder:M|D) call it alike.

    OpScript(arm, names=None)      arm 'M' | 'D'
    step(side, raw, card, sec, units=None) -> {b: (x, z, heading)} the lord's MOVEs of this tick, or {}
              raw [B, N, len(tactic raw.RAW)], card [B, N, 69]; units: lower.Units (written after a zone change)
    lord_orders(out, hide, lord)   the lord's rows of the orders out set to those MOVEs (at a run, facing E)
    notes                          the decisions not yet logged (pop_notes())

Every decide_s from start_s, per battle (B = 1 on the bridge):
    E        the enemy piece (operative/teacher.pieces in the zone's axes; no lord) nearest the members' centroid C,
             chosen once at the first decision and kept (reselected once if none of it stands); the axis u = C -> E's
             standing centroid; E_close = E's standing units' mean velocity toward C; a comer - an opposing standing
             unit coming at a standing member (the unit script's Facts.comes) within comer_gap_m edge to edge;
             quiet = E_close < still_ms, no comer, no member in melee
    edges    a unit's footprint (front x depth turned by its facing) projected on u; E's front = the least projected
             edge of its standing units; our true front = the largest of our standing members' (logged each tick)
    line     a RIGID move of the zone: turned to face along u, then shifted so that the standing members' frames'
             centroid lies on the line through E's centroid along u and the members' front (their frames' f + their
             depth / 2: they stand in the frames facing u) is gap m from E's front - M: the 'taken' bound (the frames' front TAKEN_M + taken_margin_m short of E's nearest centre: the
             unit script's reflex 'taken' holds a unit whose frame an enemy holds, so gap 0 cannot be walked); D: l60_m,
             never nearer than that bound. Computed once; the members' 'last' order (the unit script's memory,
             lower.Units.intended) set to KEEP ('no order yet': under 'counter' a unit out of its frame after a HOLD
             stands where it is); the arrival's point - the centroid of the standing members' new frames
    HALT     the zone turned to face u (kept at its turn if the members' places do not fit its tiles so) and placed on
             the standing members; each member's frame centred where it stands
             (tactic Tacticians.fr in the zone's axes); their 'last' order a HOLD, so the script's HOLD (another kind
             than the MOVE in force: the 8 m re-issue gate passes it) reaches the game this second
    hygiene  no zone change while a member is in melee
    M        at start_s: the line ('march'); nothing more
    D        at start_s, if not E_close >= comes_ms and no comer (else the next tick): the line at l60_m ('go'); then:
             E_close >= comes_ms or a comer - before the arrival HALT ('halt'), after it 'met'; the standing members'
             centroid within arrive_m of the arrival's point - 'at'; no_sally_s of UNINTERRUPTED quiet (from the last
             observation that was not quiet - melee, a comer, E closing, a missing observation, the arrival or the
             halt) in 'at' / 'halt' / 'met' - the line as M ('march', once): no_sally when E never came, else halt_stood
    lord     each tick from start_s: lord_back_m behind the standing members' centroid along u and never ahead of the
             rearmost member; a MOVE only when that point moved lord_regive_m from the last one sent; none in melee
"""
import math

import torch

from net.operative import cfg
from net.tactic import raw as TR

ARMS = ("M", "D")


def conf():
    return cfg()["o1_probe"]


def parse(spec):
    """'holder:M' | 'holder:D' -> 'M' | 'D'; None / '' -> None."""
    if not spec:
        return None
    plan, _, arm = str(spec).partition(":")
    if plan != "holder" or arm not in ARMS:
        raise ValueError(f"op script {spec!r}: holder:M or holder:D")
    return arm


def blocked_of(terrain_path):
    """(blocked [rows, cols] uint8, x0, z0, step) of the captured map's 'cannot stand' grid (sim/maps/<name>/passable.npz,
    the battle's terrain path as the document's 'map'), None: no captured map (the flat battle) or no grid."""
    try:
        import numpy as np
        from bridge.companion import unit_input as UI
        folder = UI.map_folder(terrain_path)
        if folder is None or not (folder / "passable.npz").exists():
            return None
        d = np.load(folder / "passable.npz")
        return d["blocked"], float(d["x0"]), float(d["z0"]), float(d["step"])
    except Exception:
        return None


def extent(front, depth, b, ux, uz):
    """The half extent along the unit vector (ux, uz) of footprints front x depth facing b (degrees)."""
    h = b * TR.DEG
    fwd = (torch.sin(h) * ux + torch.cos(h) * uz).abs()
    right = (torch.cos(h) * ux - torch.sin(h) * uz).abs()
    return front / 2 * right + depth / 2 * fwd


class OpScript:
    def __init__(self, arm, names=None, B=1, blocked=None):
        assert arm in ARMS, arm
        self.arm, self.names, self.c = arm, names, conf()
        self.blocked = blocked             # (blocked_of: the map's 'cannot stand' grid; None - none)
        self.phase = ["wait"] * B
        self.goal = [None] * B
        self.ref = [None] * B              # (the quiet clock's start: the last observation that was not quiet)
        self.E = [None] * B
        self.resel = [False] * B
        self.came = [False] * B
        self.marched = [False] * B
        self.lord_pt = [None] * B
        self.flags = [dict(no_sally=False, halt_stood=False) for _ in range(B)]
        self.next_t = float(self.c["start_s"])
        self.notes = []

    def pop_notes(self):
        n, self.notes = self.notes, []
        return n

    def _name(self, i):
        return self.names[i] if self.names is not None and 0 <= i < len(self.names) else int(i)

    def _pick(self, side, raw, ef, C, b):
        from net.operative import teacher as TE
        I = TR.I
        x, z = raw[b, :, I["x"]], raw[b, :, I["z"]]
        tac = side.tac
        try:
            ax = type("Axes", (), {})()
            ax.cx, ax.cz, ax.heading = tac.cx[b:b + 1], tac.cz[b:b + 1], tac.heading[b:b + 1]
            ps = TE.pieces(raw[b:b + 1], ax, ef[None])[0]
        except Exception:                       # (no pieces: E's standing infantry as one piece)
            ps = []
        if not ps:
            return ef.nonzero()[:, 0].tolist()
        cen = [torch.stack([x[p].mean(), z[p].mean()]) for p in ps]
        k = min(range(len(ps)), key=lambda i: float(((cen[i] - C) ** 2).sum()))
        return list(ps[k])

    def _facts(self, side, raw, card, b):
        """The tick's facts of battle b (module doc) or None (no standing member or no standing E)."""
        from net.tactic import side as TS
        from net.unit import scripts as SC
        I = TR.I
        tac = side.tac
        S = raw[b]
        ok = (S[:, I["alive"]] > 0) & ~(S[:, I["r"]] > 0)
        mem = tac.mine[b] & ok
        if not bool(mem.any()):
            return None
        own = int(round(float(S[mem.nonzero()[0, 0], I["side"]])))
        x, z = S[:, I["x"]], S[:, I["z"]]
        foe = ok & (S[:, I["side"]].round().long() != own) & (S[:, I["side"]] > 0)
        lord = card[b, :, TS.LORD_CARD] > 0.5 if card is not None else torch.zeros_like(foe)
        ef = foe & ~lord
        C = torch.stack([x[mem].mean(), z[mem].mean()])
        if self.E[b] is None and bool(ef.any()):
            self.E[b] = self._pick(side, raw, ef, C, b)
        E = torch.zeros_like(ef)
        if self.E[b] is not None:
            E[self.E[b]] = True
        E = E & ef
        if not bool(E.any()) and not self.resel[b] and bool(ef.any()):
            self.resel[b] = True
            self.E[b] = self._pick(side, raw, ef, C, b)
            E = torch.zeros_like(ef)
            E[self.E[b]] = True
            E = E & ef
        if not bool(E.any()):
            return None
        Ec = torch.stack([x[E].mean(), z[E].mean()])
        d = Ec - C
        u = d / max(float(torch.sqrt((d ** 2).sum())), 1e-6)
        ux, uz = float(u[0]), float(u[1])
        h = math.degrees(math.atan2(ux, uz)) % 360.0
        along = (x - C[0]) * ux + (z - C[1]) * uz
        fr, dp, bb = S[:, I["front"]], S[:, I["depth"]], S[:, I["b"]]
        ext = extent(fr, dp, bb, ux, uz)
        e_front = float((along - ext)[E].min())
        e_cmin = float(along[E].min())
        our_front = float((along + ext)[mem].max())
        tx, tz = C[0] - x, C[1] - z
        tn = torch.sqrt(tx * tx + tz * tz).clamp(min=1e-6)
        close = (S[:, I["vx"]] * tx + S[:, I["vz"]] * tz) / tn
        e_close = float(close[E].mean())
        f = SC.Facts.__new__(SC.Facts)
        f.raw = raw[b:b + 1, :, :TR.BASE]
        comes = f.comes()[0]                                                   # [i, j] j comes at i
        dx, dz = x[None, :] - x[:, None], z[None, :] - z[:, None]               # [i, j] from i to j
        dn = torch.sqrt(dx * dx + dz * dz).clamp(min=1e-6)
        ei = extent(fr[:, None], dp[:, None], bb[:, None], dx / dn, dz / dn)
        ej = extent(fr[None, :], dp[None, :], bb[None, :], dx / dn, dz / dn)
        near = (dn - ei - ej) <= float(self.c["comer_gap_m"])
        comer = bool((comes & near & mem[:, None] & foe[None, :]).any())
        melee = bool((S[:, I["m"]] > 0)[mem].any())
        a_E = (x - Ec[0]) * ux + (z - Ec[1]) * uz
        e_front_E, e_cmin_E = float((a_E - ext)[E].min()), float(a_E[E].min())
        return dict(C=C, u=(ux, uz), h=h, mem=mem, E=E, e_front=e_front, e_cmin=e_cmin, gap=e_front - our_front,
                    Ec=(float(Ec[0]), float(Ec[1])), e_front_E=e_front_E, e_cmin_E=e_cmin_E,
                    e_close=e_close, comer=comer, melee=melee, dep=dp, x=x, z=z, along=along)

    # -- the zone's moves
    def _line(self, side, fc, b, gap):
        """(cx, cz, heading) of the rigid move (module doc 'line'): the zone turned to face u, the standing members'
        frames' centroid put on the line through E's centroid along u, L m short of it - L sets the gap (or the
        'taken' bound, whichever is farther)."""
        from net.unit import scripts as SC
        tac = side.tac
        ux, uz = fc["u"]
        mem = fc["mem"]
        s_, f_ = tac.fr["s"][b][mem], tac.fr["f"][b][mem]
        hd = tac.fr["hd"][b][mem]
        ms, mf = float(s_.mean()), float(f_.mean())
        ex, ez = fc["Ec"]
        a_front = float((f_ - mf + fc["dep"][mem] / 2).max())       # (our front ahead of the frames' centroid)
        a_frame = float((f_ - mf + hd).max())                        # (the frames' front ahead of it)
        L_line = gap - fc["e_front_E"] + a_front
        L_bound = a_frame - fc["e_cmin_E"] + SC.TAKEN_M + float(self.c["taken_margin_m"])
        L = L_bound if gap <= 0 else max(L_line, L_bound)
        h = fc["h"] * TR.DEG
        c, sn = math.cos(h), math.sin(h)
        # a frame centre on 'cannot stand' ground: the game drops a MOVE there silently (h13, run 20261121-055845: two
        # of four units never set off) - the whole line steps back along u by blocked_step_m until none is
        for _ in range(int(self.c["blocked_steps"]) + 1):
            fx, fz = ex - L * ux, ez - L * uz                        # (the frames' centroid's new place)
            cx, cz = fx - (ms * c + mf * sn), fz - (-ms * sn + mf * c)
            if not self._on_blocked(cx, cz, c, sn, s_, f_):
                break
            L += float(self.c["blocked_step_m"])
        self.last_L = L
        return cx, cz, fc["h"]

    def _on_blocked(self, cx, cz, c, sn, s_, f_):
        """Any frame centre (s_, f_ in the zone (cx, cz) turned c, sn) within blocked_r_m of 'cannot stand' ground."""
        if self.blocked is None:
            return False
        B, x0, z0, st = self.blocked
        r = int(math.ceil(float(self.c["blocked_r_m"]) / st))
        for s1, f1 in zip(s_.tolist(), f_.tolist()):
            x, z = cx + s1 * c + f1 * sn, cz - s1 * sn + f1 * c
            ix, iz = int(round((x - x0) / st)), int(round((z - z0) / st))
            if 0 <= ix < B.shape[1] and 0 <= iz < B.shape[0] and                     B[max(iz - r, 0):iz + r + 1, max(ix - r, 0):ix + r + 1].max() > 0:
                return True
        return False

    def _stop(self, side, fc, b):
        """(cx, cz, heading) of the HALT's zone and the members' frames {slot: (s, f)} where they stand: the zone
        faces u if the members' places fit in its tiles there (within 1 m), else it keeps its turn."""
        from net.tactic import cfg as tcfg
        tac = side.tac
        sl = fc["mem"].nonzero()[:, 0].tolist()
        px = [float(fc["x"][i]) for i in sl]
        pz = [float(fc["z"][i]) for i in sl]
        so = [float(tac.fr["s"][b, i]) for i in sl]
        fo = [float(tac.fr["f"][b, i]) for i in sl]
        ms, mf = sum(so) / len(sl), sum(fo) / len(sl)
        tile = float(tcfg()["tile_m"])
        hz, dz_ = float(tac.cols[b]) * tile / 2, float(tac.rows[b]) * tile / 2
        best = None
        for hdeg in (fc["h"], float(tac.heading[b])):
            h = hdeg * TR.DEG
            c, sn = math.cos(h), math.sin(h)
            cx = sum(px) / len(sl) - (ms * c + mf * sn)
            cz = sum(pz) / len(sl) - (-ms * sn + mf * c)
            fr, worst = {}, 0.0
            for i, X, Zz in zip(sl, px, pz):
                dx, dz = X - cx, Zz - cz
                s1, f1 = dx * c - dz * sn, dx * sn + dz * c
                hw, hd = float(tac.fr["hw"][b, i]), float(tac.fr["hd"][b, i])
                worst = max(worst, abs(s1) + hw - hz, abs(f1) + hd - dz_, 0.0)
                fr[i] = (s1, f1)
            if best is None or worst < best[2] - 1e-6:
                best = ((cx, cz, hdeg), fr, worst)
            if worst <= 1.0:
                break
        return best[0], best[1]

    def _apply(self, side, raw, b, zone, units, frames=None):
        """The zone of battle b through lower.Tacticians.apply(moved=True); frames {slot: (s, f)} (a HALT) written in
        the zone's axes; the members' 'last' order KEEP (a move) or HOLD (a HALT); the arrival's point."""
        from net.tactic import cfg as tcfg
        from sim import orders as O
        tac = side.tac
        B = tac.B
        g = side.groups0.clone()
        g.member = torch.where(tac.mine.view(B, tac.G, tac.N)[:, 0], torch.zeros_like(g.member),
                               torch.full_like(g.member, -1))
        g.cx, g.cz, g.heading = (tac.cx.view(B, tac.G).clone(), tac.cz.view(B, tac.G).clone(),
                                 tac.heading.view(B, tac.G).clone())
        task = torch.zeros((B, tac.G), dtype=torch.bool, device=tac.dev)
        g.cx[b, 0], g.cz[b, 0], g.heading[b, 0] = zone
        task[b, 0] = True
        task = task & g.on
        tac.apply(g, task, task.clone(), side.hidden(raw))
        if frames:
            tile = float(tcfg()["tile_m"])
            hz, dz_ = float(tac.cols[b]) * tile / 2, float(tac.rows[b]) * tile / 2
            for i, (s, f) in frames.items():
                hw, hd = float(tac.fr["hw"][b, i]), float(tac.fr["hd"][b, i])
                tac.fr["s"][b, i] = max(-hz + hw, min(hz - hw, s))           # (kept inside the zone)
                tac.fr["f"][b, i] = max(-dz_ + hd, min(dz_ - hd, f))
                tac.fr_on[b, i] = True
        if units is None:
            return
        units.write(tac)
        S = raw[b]
        ok = (S[:, TR.I["alive"]] > 0) & ~(S[:, TR.I["r"]] > 0)
        mem = tac.mine[b] & ok
        kind = O.HOLD if frames else O.KEEP
        units.intended.kind[b] = torch.where(mem, torch.full_like(units.intended.kind[b], kind), units.intended.kind[b])
        if bool(mem.any()):
            self.goal[b] = torch.stack([units.area["x"][b][mem].mean(), units.area["z"][b][mem].mean()])

    def step(self, side, raw, card, sec, units=None):
        """(module doc)."""
        if sec < self.next_t:
            return {}
        while self.next_t <= sec:
            self.next_t += float(self.c["decide_s"])
        c = self.c
        lord = {}
        for b, ph in enumerate(self.phase):
            fc = self._facts(side, raw, card, b)
            if fc is None:
                self.ref[b] = float(sec)
                self.notes.append({"sec": float(sec), "b": b, "arm": self.arm, "act": "unseen", "phase": ph})
                continue
            come = fc["e_close"] >= float(c["comes_ms"]) or fc["comer"]
            quiet = fc["e_close"] < float(c["still_ms"]) and not fc["comer"] and not fc["melee"]
            if not quiet or self.ref[b] is None:
                self.ref[b] = float(sec)
            self.came[b] = self.came[b] or (come and ph != "wait")
            act, zone, frames = None, None, None
            if fc["melee"]:
                act = "hygiene"
            elif ph == "wait":
                if self.arm == "M":
                    zone, act = self._line(side, fc, b, 0.0), "march"
                elif come:
                    act = "wait"
                else:
                    zone, act = self._line(side, fc, b, float(c["l60_m"])), "go"
            elif ph == "go":
                if come:
                    zone, frames = self._stop(side, fc, b)
                    act = "halt"
                elif self.goal[b] is not None and \
                        float(torch.sqrt(((fc["C"] - self.goal[b]) ** 2).sum())) <= float(c["arrive_m"]):
                    act = "at"
            elif ph in ("at", "halt", "met"):
                if come and ph != "met":
                    act = "met"
                elif quiet and not self.marched[b] and sec - self.ref[b] >= float(c["no_sally_s"]) \
                        and (ph != "halt" or c.get("halt_resume")):
                    zone, act = self._line(side, fc, b, 0.0), "march"
                    self.marched[b] = True
                    self.flags[b]["no_sally" if not self.came[b] else "halt_stood"] = True
            if act in ("halt", "at"):
                self.ref[b] = float(sec)
            if act in ("go", "halt", "at", "march", "met"):
                self.phase[b] = act
            if zone is not None:
                self._apply(side, raw, b, zone, units, frames)
            # the lord (module doc)
            li = side.hide[b].nonzero()[:, 0].tolist()
            lact = None
            if li:
                I = TR.I
                L = raw[b, li[0]]
                if float(L[I["alive"]]) > 0 and not float(L[I["r"]]) > 0 and not float(L[I["m"]]) > 0:
                    ux, uz = fc["u"]
                    a_c = 0.0                                 # (along u from C)
                    rear = float(fc["along"][fc["mem"]].min())
                    a = min(a_c - float(c["lord_back_m"]), rear)
                    px, pz = float(fc["C"][0]) + a * ux, float(fc["C"][1]) + a * uz
                    lp = self.lord_pt[b]
                    if lp is None:
                        lp = (float(L[I["x"]]), float(L[I["z"]]))
                    if math.hypot(px - lp[0], pz - lp[1]) > float(c["lord_regive_m"]):
                        self.lord_pt[b] = (px, pz)
                        lord[b] = (px, pz, fc["h"])
                        lact = [round(px, 1), round(pz, 1)]
                else:
                    lact = "lord in melee / out"
            self.notes.append({"sec": float(sec), "b": b, "arm": self.arm, "act": act, "phase": self.phase[b],
                               "C": [round(float(v), 1) for v in fc["C"]], "gap_front": round(fc["gap"], 1),
                               "e_close": round(fc["e_close"], 2), "comer": fc["comer"], "melee": fc["melee"],
                               "quiet_s": round(float(sec) - self.ref[b], 1),
                               "E": [self._name(i) for i in (self.E[b] or [])],
                               "zone": None if zone is None else [round(float(v), 1) for v in zone],
                               "halt_frames": None if not frames else {self._name(i): [round(v, 1) for v in sf]
                                                                       for i, sf in frames.items()},
                               "lord": lact, **self.flags[b]})
        return lord

    @staticmethod
    def lord_orders(out, hide, lord):
        """(module doc) out sim.orders.Orders [B, N]; hide [B, N] the lord's rows; lord {b: (x, z, heading)}."""
        from sim import orders as O
        m = torch.zeros_like(hide)
        x, z, f = out.x.clone(), out.z.clone(), out.face.clone()
        for b, (lx, lz, h) in lord.items():
            m[b] = hide[b]
            x[b] = torch.where(hide[b], torch.full_like(x[b], lx), x[b])
            z[b] = torch.where(hide[b], torch.full_like(z[b], lz), z[b])
            f[b] = torch.where(hide[b], torch.full_like(f[b], h), f[b])
        out.kind = torch.where(m, torch.full_like(out.kind, O.MOVE), out.kind)
        out.target = torch.where(m, torch.full_like(out.target, -1), out.target)
        out.run = torch.where(m, torch.ones_like(out.run), out.run)
        out.x, out.z, out.face = x, z, f
        return out
