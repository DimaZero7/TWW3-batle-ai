"""Drill O0-1 'advance on the holder': the operative level's interface for this drill (tmp/op1/design_merged.md FINAL,
tmp/op1/design_opus.md 3; the numbers net/config/operative.json o1). One own group (all units but the lord); every
decide_s from start_s the level answers KEEP or Hold(line) with a pointer into the code's short list; the code only turns
the pointer into a zone (net/operative/opside.py's geometry, the zone moved through lower.Tacticians.apply(moved=True))
and decides nothing.

    E          the enemy piece (operative/teacher.pieces, no lord) nearest the members' centroid C, picked anew each tick
    POINTERS   deploy (the start's zone: the order in force before any decision), L0 (here: the zone placed on the
               members where they stand - opside's HALT form), LS (the stand-off line: the zone moved rigidly so that
               our front is standoff_m from E's front on the axis C -> E, facing E), LM (the dumb move's line: onto E -
               opside's arm M bound; not in the level's list)
    Level(B, att, blocked)   the batch's state: the order in force per battle (skill, pointer, zone, goal, born), the
               mission bit att [B], moved [B] (the zone moved once: the lord follows)
    level.facts(side, raw, card, b, sec)   the tick's facts of battle b (None: no standing member or no E) - the
               teacher's scalars (att, sec, melee, comer, e_close_max, gap, ls_same) and the level's input row
    level.apply(side, raw, b, ptr, f, units, sec)   the pointer's zone given (KEEP is no call)
    level.lord(side, raw, b, f)   the lord's order of the tick after a zone move: ('move', x, z, h) | ('hold',) | None
    INPUT      the names of the level's input row (row()); MOTION - its motion columns (the leakage check zeroes them)
Hygiene (a mask, not a choice): no zone change while a member is in melee.
"""
import math

import torch

from net.operative import cfg
from net.operative import opside as OPS
from net.tactic import raw as TR

POINTERS = ("deploy", "L0", "LS", "LM")
SKILLS = ("hold", "attack", "reserve", "flank")      # (the head's four; only hold open in this drill)
INPUT = ("att", "t", "own_n", "own_melee", "e_n", "gap", "e_right", "e_fwd", "u_sin", "u_cos",
         "sk_hold", "sk_attack", "sk_reserve", "sk_flank", "p_deploy", "p_L0", "p_LS", "p_LM",
         "order_dist", "order_age", "order_to_LS", "to_LS",
         "own_v_mean", "own_v_max", "e_close_mean", "e_close_max", "e_v_mean", "e_v_max", "comer")
MOTION = ("own_v_mean", "own_v_max", "e_close_mean", "e_close_max", "e_v_mean", "e_v_max", "comer")
PLACE_M = 1024.0      # (operative.json place_m: positions / 1024 m)
SPEED_MS = 3.0        # (speeds / 3 m/s: the infantry's run, sim/config/units.json speed.run)


def conf():
    return cfg()["o1"]


def geo_conf():
    """opside's numbers with the drill's own laid over (the comer's reach, the lord's; the stand-off is the level's
    own standoff_m - opside's l60_m stays the game probe's arm D)."""
    c = dict(OPS.conf())
    o = conf()
    c.update(comer_gap_m=o["comer_gap_m"], comes_ms=o["comes_ms"], still_ms=o["still_ms"],
             lord_back_m=o["lord_back_m"], lord_regive_m=o["lord_regive_m"])
    return c


class Level:
    def __init__(self, B, att, blocked=None, visible=None):
        self.B = B
        # (the teacher's predicates on the seen foes only - the network's selection, design_merged 5; config
        # o1.visible, default False: the legacy teacher of the 22.11 exams)
        self.visible = bool(conf().get("visible", False)) if visible is None else bool(visible)
        self.att = [int(a) for a in att]
        self.geo = OPS.OpScript("D", B=B)
        self.geo.c = geo_conf()
        self.blocked = list(blocked) if blocked is not None else [None] * B
        self.order = [None] * B
        self.moved = [False] * B
        self.lord_pt = [None] * B
        self.lord_held = [False] * B
        self.contact = [False] * B          # (a member was seen in melee at a tick since the zone's first move)
        self.c = conf()

    def _order0(self, side, b, C):
        tac = side.tac
        return {"skill": 0, "ptr": "deploy", "zone": (float(tac.cx[b]), float(tac.cz[b]), float(tac.heading[b])),
                "goal": (float(C[0]), float(C[1])), "born": 0.0}

    def facts(self, side, raw, card, b, sec, visible=None):
        """(module doc) -> dict or None. visible None: the level's own (self.visible)."""
        g = self.geo
        g.blocked = self.blocked[b]
        g.E[b], g.resel[b] = None, False                 # (E picked anew: the level keeps no memory but its order)
        fc = g._facts(side, raw, card, b, visible=self.visible if visible is None else bool(visible))
        if fc is None:
            return None
        if self.order[b] is None:
            self.order[b] = self._order0(side, b, fc["C"])
        I = TR.I
        S = raw[b]
        C = fc["C"]
        mem, E = fc["mem"], fc["E"]
        vx, vz = S[:, I["vx"]], S[:, I["vz"]]
        sp = torch.sqrt(vx * vx + vz * vz)
        tx, tz = C[0] - fc["x"], C[1] - fc["z"]
        tn = torch.sqrt(tx * tx + tz * tz).clamp(min=1e-6)
        close = (vx * tx + vz * tz) / tn
        ls = g._line(side, fc, b, float(self.c["standoff_m"]))
        L = g.last_L
        ex, ez = fc["Ec"]
        ux, uz = fc["u"]
        gls = (ex - L * ux, ez - L * uz)                   # (the members' frames' centroid on LS)
        o = self.order[b]
        d_ols = math.hypot(o["zone"][0] - ls[0], o["zone"][1] - ls[1])
        f = {"att": self.att[b], "sec": float(sec), "melee": bool(fc["melee"]), "comer": bool(fc["comer"]),
             "e_close_max": float(close[E].max()), "e_close_mean": float(close[E].mean()), "gap": float(fc["gap"]),
             "ls_same": o["ptr"] == "LS" and d_ols <= float(self.c["same_line_m"]),
             "fc": fc, "ls": ls, "gls": gls}
        # the level's input row (INPUT)
        tac = side.tac
        zh = float(tac.heading[b]) * TR.DEG
        dx, dz = ex - float(C[0]), ez - float(C[1])
        e_r, e_f = dx * math.cos(zh) - dz * math.sin(zh), dx * math.sin(zh) + dz * math.cos(zh)
        rel = fc["h"] * TR.DEG - zh
        sk = [0.0] * 4
        sk[o["skill"]] = 1.0
        pt = [0.0] * 4
        pt[POINTERS.index(o["ptr"])] = 1.0
        d_go = math.hypot(float(C[0]) - o["goal"][0], float(C[1]) - o["goal"][1])
        d_ls = math.hypot(float(C[0]) - gls[0], float(C[1]) - gls[1])
        k = PLACE_M
        row = [float(self.att[b]), min(float(sec) / 600.0, 2.0), float(mem.sum()) / 10.0, float(fc["melee"]),
               float(E.sum()) / 10.0, f["gap"] / k, e_r / k, e_f / k, math.sin(rel), math.cos(rel), *sk, *pt,
               d_go / k, min((float(sec) - o["born"]) / 60.0, 5.0), d_ols / k, d_ls / k,
               float(sp[mem].mean()) / SPEED_MS, float(sp[mem].max()) / SPEED_MS,
               f["e_close_mean"] / SPEED_MS, f["e_close_max"] / SPEED_MS,
               float(sp[E].mean()) / SPEED_MS, float(sp[E].max()) / SPEED_MS, float(fc["comer"])]
        assert len(row) == len(INPUT)
        f["row"] = row
        return f

    def apply(self, side, raw, b, ptr, f, units, sec):
        """The pointer's zone (module doc) through opside's _apply; the order in force set (born = sec)."""
        g = self.geo
        g.blocked = self.blocked[b]
        fc = f["fc"]
        frames = None
        if ptr == "LS":
            zone = f["ls"]
        elif ptr == "LM":
            zone = g._line(side, fc, b, 0.0)
        elif ptr == "L0":
            zone, frames = g._stop(side, fc, b)
        else:
            raise ValueError(ptr)
        g._apply(side, raw, b, zone, units, frames)
        goal = g.goal[b]
        gl = (float(goal[0]), float(goal[1])) if goal is not None else (float(fc["C"][0]), float(fc["C"][1]))
        self.order[b] = {"skill": 0, "ptr": ptr, "zone": tuple(float(v) for v in zone), "goal": gl, "born": float(sec)}
        self.moved[b] = True

    def lord(self, side, raw, b, f):
        """The lord's order of this tick after the zone's first move, at most one a tick: ('move', x, z, heading) -
        lord_back_m behind C along u, never ahead of the rearmost member - while the group walks (no member seen in
        melee since the move) and the point moved lord_regive_m from the last one sent, or at once when he stands
        ahead of the rearmost member; after the group's first contact no re-give, only ('hold',) once if a stale MOVE
        left him ahead of the rearmost member; None - nothing (also while he is in melee, routs or is gone)."""
        if not self.moved[b] or f is None:
            return None
        if f["melee"]:
            self.contact[b] = True
        li = side.hide[b].nonzero()[:, 0].tolist()
        if not li:
            return None
        I = TR.I
        L = raw[b, li[0]]
        if not float(L[I["alive"]]) > 0 or float(L[I["r"]]) > 0 or float(L[I["m"]]) > 0:
            return None
        fc = f["fc"]
        ux, uz = fc["u"]
        rear = float(fc["along"][fc["mem"]].min())
        ahead = (float(L[I["x"]]) - float(fc["C"][0])) * ux + (float(L[I["z"]]) - float(fc["C"][1])) * uz > rear
        if self.contact[b]:
            if ahead and self.lord_pt[b] is not None and not self.lord_held[b]:
                self.lord_held[b] = True
                return ("hold",)
            return None
        a = min(-float(self.c["lord_back_m"]), rear)
        px, pz = float(fc["C"][0]) + a * ux, float(fc["C"][1]) + a * uz
        lp = self.lord_pt[b] or (float(L[I["x"]]), float(L[I["z"]]))
        if math.hypot(px - lp[0], pz - lp[1]) <= float(self.c["lord_regive_m"]) and not ahead:
            return None
        self.lord_pt[b] = (px, pz)
        return ("move", px, pz, fc["h"])
