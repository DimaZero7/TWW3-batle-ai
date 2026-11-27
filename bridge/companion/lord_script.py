"""The lord script of the lord line's series T (tmp/lordline/PLAN.md "After critic 2"): OUR lord in the game by fixed
rules, "as the game's AI does". A lane instrument, OFF by default: the chain 'tactic_v2 only' (tactic_chain.py) imports
and runs it only with BAI_LORD_SCRIPT=T (watch.ps1 -LordScript T). No network sees the lord: the tactic level's input is
not touched, the script replaces the lord's order line (after the Gate) and adds at most ONE `ability` line a reply. It
prices the bundle (post + fight + step-out + buttons at 1 s) - never a label for a learned head.

It reads only the state document the bridge writes, one decision a battle second; an enemy row whose v is not true is
never read. Distances are centre to centre, strictly below the radius (the game's collector: horizontal squared distance
< r^2, tmp/astra/abil_re/out/REPORT.md Q1).

    lord      exactly one own row (side = own) named *_lord of a known loadout (LOADOUT) - else ValueError at the start
    heading   of the LINE, not of the units' facings: the normal of the principal axis of the standing infantry's centres,
              its sign toward the seen enemies' centroid (none seen: the sign nearest the last heading; a bunched line -
              eigenvalue ratio < 2 - keeps the last heading)
    pair      the front cluster = the foremost group of >= 2 standing infantry along the heading (groups split where the
              forward gap exceeds RANK_M) with any strays ahead of it (no such group: all of them); the pair = its two middle units by lateral order
              (odd: the middle one and its neighbour of the lower slot number; one unit: that one). Kept by name while
              both stand within KEEP_M (forward) of the front cluster's mean; a one-unit pair is re-picked as soon as a
              front cluster of two exists
    post      POST_BACK_M behind the midpoint of the pair's two centres along the heading, and never less than
              POST_MIN_M behind the rearmost centre of the front cluster. MOVE when the post moved more than REORDER_M
              from the last point sent (no MOVE when he already stands within REORDER_M of it); a MOVE not reached in
              STUCK_S while he stands still is logged 'stuck'
    run-in    while the pair advances (this reply or an earlier one gave a pair unit ATTACK, or the pair's midpoint went
              forward faster than ADV_MPS in each of the last ADV_TICKS decisions) the post leads by LEAD_M along the
              heading, the MOVE runs and is re-sent every decision the post moved more than ADV_REORDER_M (and once
              more when the run-in ends: a led point is never kept); a re-sent point within REORDER_M of the last one
              carries the heading +- FACE_NUDGE_DEG (alternately), so the bridge's same-order filter (services.lua
              changed: a point within REPEAT_M and a facing within 1 deg) passes it
    fight    from the line's first contact (the first own infantry unit in melee): ATTACK the nearest enemy in
              WITNESSED melee with our infantry - a seen, standing enemy with m true and a standing own infantry unit
              with m true, either one targeting the other with centres < LINK_M, or no target link (and neither
              naming another opponent) and centres < NEAR_M; a target link without both flags, or both flags at NEAR_M..LINK_M without a link, is 'ambiguous'
              (logged, not attacked). The latched target is dropped at once out of sight / routed / dead, and after
              GRACE_S when neither our infantry nor the lord himself is in witnessed melee with it (no lone chase);
              dropped with no own infantry left: MOVE to the last post (none: HOLD), never the old ATTACK kept
    buttons   one press a reply at most, by priority: Verminous Valour once due at the step-out; Stand Your Ground -
              >= SYG_MIN own infantry < AURA_M in melee AT THE PRESS (its recipients are fixed then); Rally - >= 1 own
              infantry < AURA_M wavering; Foe-Seeker / Deadly Onslaught - the lord himself in melee.
              Requested (this reply, move N) -> called (the bridge's abilities_used of that key at a battle ms in
              [N's t, N's t + 1 s)) -> took (a phase of the key ABSENT on a card read at the request - the lord's or a
              recipient's < AURA_M for an aura button - and PRESENT on that same unit's card within CONFIRM_S model s
              of the call; the deadline is checked first; took['late'] False - the smoke's line). Only a took starts
              the cooldown: ready again at the call + passport active_s + recharge_s + READY_PAD_S (the game refused
              all 37 of 114 presses made at exactly call + active + recharge, none later: tmp/gate_press/
              evidence_draft.md); a newly readable card is not a new phase.
              Not took -> 'unknown', the call still watched until active_s and retried only after RETRY_S and that
              window (one phase never answers two calls of a key): a phase seen
              later is a took with its real delay (late True), the cooldown from its call, and the misses requested
              inside that real cooldown are struck. A miss = not called, or called with a readable card and no new
              phase; an attempt whose cards were never read after the call is not a miss. The TRIES-th miss of a key
              stops its requests; once no call of it is still watched it is the brief's stop rule: self.failed set
              (the run fails; the reader stops the series). No readable card at all (the lord's, nor for an aura
              button a recipient's) at an eligible tick: no press, self.failed set; the lord's alone unread - proved on
              the recipients. The battle's first press: its cards logged at +1, +2, +3 s (note first_reads)
    step-out  health (the bridge's unary share) below STEP_OUT_HP, a finite reading only -> WITHDRAW to OUT_BACK_M behind
              the nearest standing own infantry along the heading; re-chosen when that unit routs or dies, or with no
              PROGRESS_M of progress in RESEND_S (at most RESENDS times, then 'step_failed': he is left as he stands);
              'out' only when he is seen within ARRIVED_M of the point; no return in this series
    hygiene   no new order while he is in melee (m true or unreadable), except the step-out
What the reply does not deliver (a lord marked out by the chain) is undone by commit(False): no order or press is
taken as sent.

_why: AURA_M 35 - the passports' range_m (sim/config/abilities.json, the game's database; centre to centre: sim/evidence/
effectsimpl/compare.md T-E2; strict <: abil_re Q1); STEP_OUT_HP 0.35 - the orchestrator's margin above the single
entity's 25 %, not probed; POST_BACK_M 20 - the owner's battles 15-19 m, the AI's 20-22 m (tmp/lordline/lord_inventory.md
3, 4); OUT_BACK_M 30, SYG_MIN 2, TRIES 3 - tmp/lordline/BRIEF_0a.md; CONFIRM_S 2 - its acceptance line; REORDER_M 5 -
services.lua REPEAT_M; RANK_M 15, KEEP_M 25, POST_MIN_M 10, LINK_M 60 (two 30-40 m wide units in contact at their ends),
NEAR_M 25, GRACE_S 5, RETRY_S 5, ARRIVED_M 10, PROGRESS_M 5, RESEND_S 6, RESENDS 3, STUCK_S 20 - this script's choices.
The run-in: the T smokes 27.11 (runs 20261127-114023 / -114252) - the pair ran in at 3-4 m/s (charge 3.8-4.2), the lord
walked to a point re-sent only past REORDER_M and fell to 26-34 m behind in the last 4 s; a 1-D replay of those profiles
(tmp/close0a/kin.py; the general's run 3.4 m/s, acceleration 2, deceleration 4 m/s2 - the passport) keeps 21.7-23.6 m
only with the onset at the pair's ATTACK and a re-send every second. LEAD_M 4.5 - 2-3 s of the pair's run (7-12 m)
capped so the point sent stays >= 15 m behind the pair (the band's near edge); ADV_MPS 1.5 x ADV_TICKS 2 - above
the standing pair's jitter (a single second of 1.0-1.1 m/s, sometimes 2.0, the 27.11 re-smoke p02 T 20261127-145136:
with 1.0 x 1 it started false run-ins that left the lord at the led point, 13.3-14.8 m behind); a led point is
replaced as soon as the run-in ends; ADV_REORDER_M 1 - above that jitter's 0.3 m; FACE_NUDGE_DEG 1.5 - above services.lua FACE_SAME_DEG 1.
"""
import copy
import math
import os

from net.cards import abilities as model_abilities
from net.cards import passport
from sim import abilities as sim_abilities

ARMS = ("T",)
POST_BACK_M, POST_MIN_M = 20.0, 10.0
REORDER_M = 5.0
RUN_FAR_M = 15.0
AURA_M = 35.0
SYG_MIN = 2
STEP_OUT_HP = 0.35
OUT_BACK_M = 30.0
RANK_M, KEEP_M = 15.0, 25.0
LINK_M, NEAR_M = 60.0, 25.0
GRACE_S = 5.0
CONFIRM_S = 2.0
RETRY_S = 5.0
TRIES = 3
ARRIVED_M = 10.0
PROGRESS_M = 5.0
RESEND_S = 6.0
RESENDS = 3
STUCK_S = 20.0
ADV_MPS, ADV_TICKS, LEAD_M, ADV_REORDER_M, FACE_NUDGE_DEG = 1.5, 2, 4.5, 1.0, 1.5
READY_PAD_S = 1.0       # a button is ready again at call + active_s + recharge_s + this (module doc 'buttons')
SYG = "wh_main_character_abilities_stand_your_ground"
RALLY = "wh_main_character_abilities_rally"
FOE_SEEKER = "wh_main_character_abilities_foe_seeker"
ONSLAUGHT = "wh_main_character_abilities_deadly_onslaught"
VALOUR = "wh2_main_character_abilities_verminous_valour"
LOADOUT = {"wh_main_emp_cha_general_0": {FOE_SEEKER, SYG},
           "wh2_main_skv_cha_warlord_0": {VALOUR, ONSLAUGHT, RALLY}}
DECISIONS = ("kind", "attacking", "last_point", "move_t", "target", "lost_t", "step", "out", "valour_due", "btn",
             "last_post", "first_req", "nudge", "led")
READS_S = 3.0           # the first press's card reads logged at +1, +2, +3 s of the call (the review's smoke trace)


def arm_of(value):
    """BAI_LORD_SCRIPT's value -> the arm or None (unset / empty: off)."""
    if value in (None, ""):
        return None
    if value not in ARMS:
        raise ValueError(f"BAI_LORD_SCRIPT={value!r}: one of {ARMS} or unset")
    return value


def fin(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def fwd(b):
    a = math.radians(float(b))
    return math.sin(a), math.cos(a)


def bearing_of(dx, dz):
    return math.degrees(math.atan2(dx, dz)) % 360


def angle(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def mean_bearing(bs):
    s = sum(math.sin(math.radians(b)) for b in bs)
    c = sum(math.cos(math.radians(b)) for b in bs)
    return math.degrees(math.atan2(s, c)) % 360


def dist(a, b):
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["z"]) - float(b["z"]))


def placed(u):
    """A row with men and a finite place."""
    return u is not None and fin(u.get("men")) and u["men"] > 0 and fin(u.get("x")) and fin(u.get("z"))


def standing(u):
    return placed(u) and u.get("r") is not True and u.get("s") is not True


def slot_no(name):
    digits = "".join(c for c in name if c.isdigit())
    return int(digits) if digits else 10 ** 6


def phases(row):
    """The phases on a row's card (set), None when not read (Lua omits fx; JSON gives an empty list as {})."""
    fx = row.get("fx") if row else None
    if fx is None:
        return None
    if isinstance(fx, dict):
        return set(fx.values()) if fx else set()
    return set(fx) if isinstance(fx, list) else None


def line_heading(inf, foes, last):
    """The line's heading (module doc) or last."""
    if len(inf) < 2:
        if foes and inf:
            cx = sum(float(e["x"]) for e in foes) / len(foes) - float(inf[0]["x"])
            cz = sum(float(e["z"]) for e in foes) / len(foes) - float(inf[0]["z"])
            return bearing_of(cx, cz) if (cx or cz) else last
        return last
    mx = sum(float(u["x"]) for u in inf) / len(inf)
    mz = sum(float(u["z"]) for u in inf) / len(inf)
    sxx = sum((float(u["x"]) - mx) ** 2 for u in inf)
    szz = sum((float(u["z"]) - mz) ** 2 for u in inf)
    sxz = sum((float(u["x"]) - mx) * (float(u["z"]) - mz) for u in inf)
    tr, det = sxx + szz, sxx * szz - sxz * sxz
    disc = math.sqrt(max(tr * tr / 4 - det, 0.0))
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    if l1 <= 1e-6 or (l2 > 1e-6 and l1 / l2 < 2.0):
        return last                                   # (bunched: no line axis)
    th = 0.5 * math.atan2(2 * sxz, sxx - szz)        # the axis (cos th, sin th) in (x, z)
    h = bearing_of(-math.sin(th), math.cos(th))      # its normal
    if foes:
        cx = sum(float(e["x"]) for e in foes) / len(foes) - mx
        cz = sum(float(e["z"]) for e in foes) / len(foes) - mz
        fx, fz = fwd(h)
        if fx * cx + fz * cz < 0:
            h = (h + 180) % 360
    elif last is not None and angle(h, last) > 90:
        h = (h + 180) % 360
    elif last is None:
        b = [float(u["b"]) for u in inf if fin(u.get("b"))]
        if b and angle(h, mean_bearing(b)) > 90:
            h = (h + 180) % 360
    return h


def frame(rows, heading):
    """{name: (forward, lateral)} of rows about their centroid along heading."""
    fx, fz = fwd(heading)
    mx = sum(float(u["x"]) for u in rows) / len(rows)
    mz = sum(float(u["z"]) for u in rows) / len(rows)
    return {u["n"]: ((float(u["x"]) - mx) * fx + (float(u["z"]) - mz) * fz,
                     (float(u["x"]) - mx) * fz - (float(u["z"]) - mz) * fx) for u in rows}


def front_cluster(inf, heading):
    """(names of the front cluster, the frame) (module doc)."""
    fl = frame(inf, heading)
    order = sorted(fl, key=lambda n: -fl[n][0])
    groups, cur = [], [order[0]]
    for a, b in zip(order, order[1:]):
        if fl[a][0] - fl[b][0] > RANK_M:
            groups.append(cur)
            cur = []
        cur.append(b)
    groups.append(cur)
    k = next((i for i, g in enumerate(groups) if len(g) >= 2), len(groups) - 1)
    front = [n for g in groups[:k + 1] for n in g]     # (the first group of two and the strays ahead of it)
    return front, fl


def middle_pair(front, fl):
    lat = sorted(front, key=lambda n: (fl[n][1], slot_no(n)))
    n = len(lat)
    if n <= 2:
        return lat
    if n % 2 == 0:
        return lat[n // 2 - 1: n // 2 + 1]
    m = n // 2
    other = min((lat[m - 1], lat[m + 1]), key=slot_no)
    return sorted([lat[m], other], key=lambda x: fl[x][1])


def centre_pair(inf, heading):
    """[names] of the centre pair of inf (standing infantry rows) along heading, no memory (module doc)."""
    if not inf or heading is None:
        return []
    front, fl = front_cluster(inf, heading)
    return middle_pair(front, fl)


def witnessed(e, inf):
    """(own infantry rows in witnessed melee with enemy row e, ambiguous?) (module doc)."""
    ok, odd = [], False
    for u in inf:
        link = u.get("t") == e["n"] or e.get("t") == u["n"]
        # (no link, but a target link naming another opponent: two separate fights side by side - not each other's)
        other = u.get("t") not in (None, "", e["n"]) or e.get("t") not in (None, "", u["n"])
        d = dist(u, e)
        both = u.get("m") is True and e.get("m") is True
        if both and ((link and d < LINK_M) or (not link and not other and d < NEAR_M)):
            ok.append(u)
        elif (link and d < LINK_M) or (both and d < LINK_M):
            odd = True
    return ok, odd and not ok


class LordScript:
    """One battle's lord script (module doc). names, keys, sides: the state's units in order; own: our side."""

    def __init__(self, names, keys, sides, own=1, arm="T"):
        self.arm, self.own = arm_of(arm), own
        self.names, self.keys = list(names), list(keys)
        lords = [i for i, n in enumerate(self.names) if int(sides[i]) == own and n.endswith("_lord")]
        if len(lords) != 1:
            raise ValueError(f"lord script: {len(lords)} own lords named *_lord in {self.names} - exactly one wanted")
        i = lords[0]
        self.lord, key = self.names[i], self.keys[i]
        if key not in LOADOUT:
            raise ValueError(f"lord script: own lord {self.lord} is {key!r}, not a known loadout {sorted(LOADOUT)}")
        units_db, self.passports = passport.load(), {**model_abilities.load_spells(), **model_abilities.load()}
        self.press = sim_abilities.press_map(key, units_db, self.passports)     # fails closed on an unmapped active
        if set(self.press.values()) != LOADOUT[key]:
            raise ValueError(f"lord script: {key} presses {self.press}, the known loadout {sorted(LOADOUT[key])}")
        self.slot_of = {k: s for s, k in self.press.items()}
        self.btn = {k: {"st": "ready", "tries": 0, "cd_end": None, "req": None, "misses": [], "watch": []}
                    for k in self.press.values()}
        self.last_post, self.first_req = None, None
        self.heading, self.pair, self.contact_t, self.touched_first = None, [], None, None
        self.kind = self.attacking = self.last_point = self.move_t = None
        self.target, self.lost_t = None, None
        self.step, self.out, self.valour_due = None, False, False
        self.pair_prev, self.pair_orders, self.advance, self.nudge = None, {}, False, 1
        self.fast, self.led = 0, False             # (seconds in a row the pair went > ADV_MPS; the last point led)
        self.failed = []                  # the run fails (the reader and the series stop): causes
        self.note, self._saved = None, None
        # BAI_LORD_PRESS=0 (the lord line's arm N, tmp/lordline/t2): no button ever pressed; unset / anything else - on
        self.buttons = os.environ.get("BAI_LORD_PRESS", "") != "0"

    # -- delivery
    def commit(self, delivered):
        """After the reply is built: delivered False (the lord marked out) undoes this decision's order and press."""
        if not delivered and self._saved is not None:
            for k, v in self._saved.items():
                setattr(self, k, v)
            if self.note is not None:
                self.note["undelivered"] = True
        self._saved = None

    def fail(self, why, note):
        if why not in self.failed:
            self.failed.append(why)
            print(f"LORD SCRIPT FAILED: {why}", flush=True)
        note["FAILED"] = list(self.failed)

    # -- the buttons (requested -> called -> took)
    def _cards(self, rows, key, names):
        """{name: key's phase on that unit's card now (bool), None when its card is not read}."""
        ph = set(self.passports[key].get("phases") or (key,))
        out = {}
        for n in names:
            fx = phases(rows.get(n))
            out[n] = None if fx is None else bool(fx & ph)
        return out

    def _took(self, key, b, r, t, new, note, late):
        """A new phase seen for request r: the cooldown from its call; misses inside that real cooldown (requests the
        game refused because it had in fact taken) are not failures."""
        p = self.passports[key]
        c = r["call"]
        b["st"], b["cd_end"] = "cooldown", c + float(p["active_s"]) + float(p["recharge_s"]) + READY_PAD_S
        b["misses"] = [m for m in b["misses"] if not (r["t"] - 1e-6 <= m < b["cd_end"])]
        b["tries"], b["watch"] = len(b["misses"]), []
        if b.get("req") is not None and b["req"] is not r:      # (a request made inside the real cooldown: dropped)
            note.setdefault("in_cooldown", []).append({"key": key, "req": b["req"]["t"]})
        b["req"] = None
        note.setdefault("took", []).append({"key": key, "req": r["t"], "move": r["move"], "call": c, "seen": round(t, 1),
                                            "delay": round(t - c, 1), "late": late, "new": new})
        if key == VALOUR:
            self.valour_due = False

    def _follow(self, doc, rows, t, note):
        used = (doc.get("abilities_used") or {}).get(self.lord) or {}
        if self.first_req is not None:                  # (the battle's first press: its cards at +1, +2, +3 s)
            fk, ft = self.first_req
            c = used.get(fk)
            c0 = float(c) / 1000.0 if fin(c) and ft * 1000 <= float(c) < ft * 1000 + 1000 else None
            dt = t - (c0 if c0 is not None else ft)
            if 0 < dt <= READS_S + 1e-6:
                note["first_reads"] = {"key": fk, "req": ft, "call": c0, "dt": round(dt, 1),
                                       "cards": self._cards(rows, fk, self.first_names)}
        for key, b in self.btn.items():
            if b["st"] == "cooldown" and t >= b["cd_end"]:
                b["st"], b["tries"], b["misses"] = "ready", 0, []
            if b["st"] == "unknown" and t >= b.get("retry", 0) and not b["watch"]:
                b["st"] = "ready"                               # (no repeat while an earlier call can still show)
            active = float(self.passports[key]["active_s"])
            if b["st"] == "pending":
                r = b["req"]
                if (r.get("call") is None and fin(used.get(key))
                        and r["t"] * 1000 <= float(used[key]) < r["t"] * 1000 + 1000):
                    r["call"] = float(used[key]) / 1000.0
                t0 = r["call"] if r.get("call") is not None else r["t"]
                if t - t0 > CONFIRM_S + 1e-6:                   # (the deadline first)
                    if r.get("call") is None:
                        why = "not_called"
                    elif r.get("reads", 0) == 0:
                        why = "card_not_read"                   # (no readable card after the call: proves nothing)
                    else:
                        why = "no_phase"
                    if why != "card_not_read":
                        b["misses"].append(r["t"])
                    b["tries"] = len(b["misses"])
                    # (the TRIES-th miss: no new request; the wall only once no earlier call can still show late)
                    b["st"] = "wall_wait" if b["tries"] >= TRIES else "unknown"
                    b["retry"] = t + RETRY_S
                    b["req"] = None
                    if r.get("call") is not None:
                        b["watch"].append(r)                    # (a late phase still starts the real cooldown)
                    note.setdefault("unconfirmed", []).append({"key": key, "req": r["t"], "call": r.get("call"),
                                                               "why": why, "tries": b["tries"], "st": b["st"]})
                elif r.get("call") is not None and t > r["call"]:
                    now = self._cards(rows, key, list(r["base"]))
                    readable = [n for n, was in r["base"].items() if was is False and now.get(n) is not None]
                    r["reads"] = r.get("reads", 0) + bool(readable)
                    new = sorted(n for n in readable if now[n] is True)
                    if new:
                        self._took(key, b, r, t, new, note, late=False)
                        continue
            for r in list(b["watch"]):                          # late phases (after CONFIRM_S, within active_s)
                if t - r["call"] > active + 1e-6:
                    b["watch"].remove(r)
                    continue
                now = self._cards(rows, key, list(r["base"]))
                new = sorted(n for n, was in r["base"].items() if was is False and now.get(n) is True)
                if new:
                    self._took(key, b, r, t, new, note, late=True)
                    break
            if b["st"] == "wall_wait" and not b["watch"]:
                b["st"] = "wall"
                self.fail(f"stop rule: {key} not took after {TRIES} attempts", note)
                if key == VALOUR:
                    self.valour_due = False

    def request(self, rows, t, key, why, note=None, move=None, recipients=(), **extra):
        """THE press route (the rules here and the parity test's forced choice alike): key requested now if it is the
        lord's, ready and not on his card -> [{unit, key}] (the bridge's `ability` line; the simulator's Orders.ability
        = self.slot_of[key] on its landing step), else []. recipients: the other units whose cards may show it."""
        note = {} if note is None else note
        b = self.btn.get(key)
        if b is None or b["st"] != "ready" or b["watch"] or not standing(rows.get(self.lord)):
            return []                                           # (one call of a key in its window at a time)
        aura = float(self.passports[key].get("range_m") or 0.0) > 0      # (a self buff shows on the lord's card only)
        base = self._cards(rows, key, [self.lord] + ([n for n in recipients if n != self.lord] if aura else []))
        if all(v is None for v in base.values()):
            self.fail(f"no card (fx) readable for {key} at {t:.0f} s (the lord's"
                      f"{' nor a recipient' if aura else ''}): no press can be proved", note)
            note["card_unreadable"] = key
            return []
        if any(v is True for v in base.values()):
            return []                                           # (its phase already on: not pressable, not provable)
        if base[self.lord] is None:
            note["lord_card_unread"] = key                      # (proved on the recipients' readable cards)
        b["st"] = "pending"
        b["req"] = {"t": t, "move": move, "base": base}
        if self.first_req is None:
            self.first_req, self.first_names = (key, t), list(base)
        note["press"] = {"key": key, "slot": self.slot_of[key], "why": why, "try": b["tries"] + 1, "move": move,
                         "unread": sorted(n for n, v in base.items() if v is None), **extra}
        return [{"unit": self.lord, "key": key}]

    def follow(self, doc):
        """The buttons' bookkeeping alone on one state document (no decision): -> its note."""
        note = {"t": round(doc.get("t", 0) / 1000.0, 1)}
        self._follow(doc, {u["n"]: u for u in doc["units"]}, doc.get("t", 0) / 1000.0, note)
        return note

    # -- one decision
    def decide(self, doc, orders=()):
        """-> (the lord's order dict (exchange.orders_list's form; kind 'keep': none new), [at most one ability use])
        for one state document; orders: this reply's lines for our units (the chain's, before the lord's is replaced:
        the run-in's onset). self.note: the decision's record (the companion's log row 'lord'). Then commit()."""
        for o in orders or ():
            if o.get("unit") != self.lord and o.get("kind") not in (None, "keep"):
                self.pair_orders[o["unit"]] = o["kind"]        # (the order in force: a 'keep' line keeps it)
        t = doc.get("t", 0) / 1000.0
        rows = {u["n"]: u for u in doc["units"]}
        lord = rows.get(self.lord)
        note = {"t": round(t, 1), "move": doc.get("move")}
        self.note = note
        keep = {"unit": self.lord, "kind": "keep"}
        self._follow(doc, rows, t, note)                 # (observations: kept whatever the delivery)
        self._saved = {k: copy.deepcopy(getattr(self, k)) for k in DECISIONS}
        note["btn"] = {k[k.rfind("abilities_") + 10:]: b["st"] for k, b in self.btn.items()}
        if self.failed:
            note["FAILED"] = list(self.failed)
        if not standing(lord):
            note["ph"] = "down"
            return keep, []
        inf = [u for u in doc["units"] if int(u.get("side", 0)) == self.own and u["n"] != self.lord and standing(u)]
        foes = [u for u in doc["units"] if int(u.get("side", 0)) != self.own and u.get("v") is True and standing(u)]
        hp = float(lord["hp"]) if fin(lord.get("hp")) else None
        melee = lord.get("m")
        if self.contact_t is None and any(u.get("m") is True for u in inf):
            self.contact_t = t
        if melee is True and self.contact_t is None and self.touched_first is None:
            self.touched_first = t
            note["touched_first"] = t
        self.heading = line_heading(inf, foes, self.heading)
        post = self._post(lord, inf, note, t)
        if post is not None:
            self.last_post = post
        near = [u for u in inf if dist(u, lord) < AURA_M]
        n_melee = sum(1 for u in near if u.get("m") is True)
        n_waver = sum(1 for u in near if u.get("w") is True)
        note.update(hp=hp, m=melee, contact=self.contact_t, n35=len(near), melee35=n_melee, waver35=n_waver)
        order = keep
        if self.out:
            note["ph"] = "out"
        elif self.step is not None:
            order = self._stepping(lord, inf, t, note)
        elif hp is not None and hp < STEP_OUT_HP:
            order = self._step_out(lord, inf, t, note)
        elif melee is not False:
            note["ph"] = "melee" if melee is True else "melee_unread"      # hygiene: no new order in melee
            self._keep_target(lord, rows, inf, t, note)
        else:
            order = self._move_or_fight(lord, rows, inf, foes, post, t, note)
        note["tg"] = self.target
        if order["kind"] != "keep":
            note["o"] = order["kind"]
        uses = self._press(rows, t, note, melee is True, n_melee, n_waver, [u["n"] for u in near], doc.get("move"))
        return order, uses

    def _post(self, lord, inf, note, t=None):
        if not inf or self.heading is None:
            self.pair, self.pair_prev, self.advance, self.fast = [], None, False, 0
            return None
        front, fl = front_cluster(inf, self.heading)
        mean_f = sum(fl[n][0] for n in front) / len(front)
        keep = (self.pair and all(n in fl for n in self.pair)
                and all(abs(fl[n][0] - mean_f) <= KEEP_M for n in self.pair)
                and not (len(self.pair) == 1 and len(front) >= 2))
        if not keep:
            self.pair = middle_pair(front, fl)
        rows = {u["n"]: u for u in inf}
        pr = [rows[n] for n in self.pair]
        px = sum(float(u["x"]) for u in pr) / len(pr)
        pz = sum(float(u["z"]) for u in pr) / len(pr)
        fx, fz = fwd(self.heading)
        back = POST_BACK_M
        # never in front of the line: at least POST_MIN_M behind the rearmost centre of the front cluster
        pf = sum(fl[n][0] for n in self.pair) / len(self.pair)
        rear = min(fl[n][0] for n in front)
        # the run-in (module doc): the pair's forward speed since the last decision (the same pair) or its ATTACK
        v = None
        pp = self.pair_prev
        if t is not None and pp is not None and pp[0] == list(self.pair) and t > pp[1]:
            v = ((px - pp[2]) * fx + (pz - pp[3]) * fz) / (t - pp[1])
        if t is not None:
            self.pair_prev = (list(self.pair), t, px, pz)
        ordered = any(self.pair_orders.get(n) == "attack" for n in self.pair)
        self.fast = self.fast + 1 if (v is not None and v > ADV_MPS) else 0
        self.advance = ordered or self.fast >= ADV_TICKS
        lead = LEAD_M if self.advance else 0.0
        back = max(back - lead, pf - rear + POST_MIN_M)
        post = (px - back * fx, pz - back * fz)
        dx, dz = float(lord["x"]) - px, float(lord["z"]) - pz
        if self.advance:
            note["run_in"] = {"v": None if v is None else round(v, 2), "ordered": ordered, "lead": lead}
        note.update(pair=self.pair, heading=round(self.heading, 1), post=[round(post[0], 1), round(post[1], 1)],
                    behind=round(-(dx * fx + dz * fz), 1), lateral=round(dx * fz - dz * fx, 1),
                    d_pair=[round(dist(lord, u), 1) for u in pr],
                    d_post=round(math.hypot(float(lord["x"]) - post[0], float(lord["z"]) - post[1]), 1))
        return post

    def _move(self, lord, x, z, face, t, kind="move", run_in=False):
        run = kind == "withdraw" or run_in or math.hypot(float(lord["x"]) - x, float(lord["z"]) - z) > RUN_FAR_M
        lp = self.last_point
        if ((run_in or self.led) and self.kind == "move" and lp is not None
                and math.hypot(x - lp[0], z - lp[1]) <= REORDER_M):
            face, self.nudge = face + FACE_NUDGE_DEG * self.nudge, -self.nudge   # (passes the bridge's filter)
        self.kind, self.last_point, self.move_t, self.attacking = kind, (x, z), t, None
        self.led = run_in
        return {"unit": self.lord, "kind": kind, "x": round(x, 1), "z": round(z, 1), "run": bool(run),
                "face": round(face % 360, 2)}

    def _keep_target(self, lord, rows, inf, t, note):
        """The latched target's checks (module doc 'fight'); True while it stays."""
        if self.target is None:
            return False
        e = rows.get(self.target)
        if e is None or e.get("v") is not True or not standing(e):
            note["dropped"] = [self.target, "unseen" if (e is not None and e.get("v") is not True) else "down"]
            self.target, self.lost_t = None, None
            return False
        held, _ = witnessed(e, inf)
        lord_on = lord.get("m") is True and (lord.get("t") == e["n"] or e.get("t") == self.lord)
        if held or lord_on:
            self.lost_t = None
            return True
        self.lost_t = t if self.lost_t is None else self.lost_t
        if t - self.lost_t >= GRACE_S:
            note["dropped"] = [self.target, "disengaged"]
            self.target, self.lost_t = None, None
            return False
        return True

    def _move_or_fight(self, lord, rows, inf, foes, post, t, note):
        keep = {"unit": self.lord, "kind": "keep"}
        has = self._keep_target(lord, rows, inf, t, note)
        if self.contact_t is not None and not has:
            held, odd = [], []
            for e in foes:
                ok, amb = witnessed(e, inf)
                if ok:
                    held.append(e)
                elif amb:
                    odd.append(e["n"])
            if odd:
                note["ambiguous"] = odd
            if held:
                self.target, self.lost_t = min(held, key=lambda e: dist(e, lord))["n"], None
                has = True
        if has:
            note["ph"] = "fight"
            if self.kind != "attack" or self.attacking != self.target:
                self.kind, self.attacking, self.last_point = "attack", self.target, None
                return {"unit": self.lord, "kind": "attack", "target": self.target, "run": True}
            return keep
        if post is None:
            note["ph"] = "alone"
            if self.kind == "attack":                  # (the target lost, no infantry left: no old ATTACK kept)
                lp = self.last_post
                if lp is not None:
                    note["o_why"] = "alone_back_to_last_post"
                    return self._move(lord, lp[0], lp[1], self.heading if self.heading is not None else 0.0, t)
                note["o_why"] = "alone_hold"
                self.kind, self.attacking, self.last_point = "hold", None, None
                return {"unit": self.lord, "kind": "hold"}
            return keep
        note["ph"] = "place"
        d_lord = math.hypot(float(lord["x"]) - post[0], float(lord["z"]) - post[1])
        lp = self.last_point
        if self.kind == "move" and lp is not None:
            moved = math.hypot(post[0] - lp[0], post[1] - lp[1])
            if moved > REORDER_M or ((self.advance or self.led) and moved > ADV_REORDER_M):
                return self._move(lord, post[0], post[1], self.heading, t, run_in=self.advance)
            if (self.move_t is not None and t - self.move_t >= STUCK_S and lord.get("mv") is not True
                    and math.hypot(float(lord["x"]) - lp[0], float(lord["z"]) - lp[1]) > ARRIVED_M):
                note["stuck"] = round(t - self.move_t, 1)
            return keep
        if self.kind == "attack" or d_lord > REORDER_M:
            return self._move(lord, post[0], post[1], self.heading, t, run_in=self.advance)
        return keep

    def _out_point(self, lord, inf, avoid=()):
        cand = [u for u in inf if u["n"] not in avoid] or inf
        h = self.heading if self.heading is not None else (float(lord["b"]) if fin(lord.get("b")) else 0.0)
        fx, fz = fwd(h)
        if cand:
            u = min(cand, key=lambda v: dist(v, lord))
            return float(u["x"]) - OUT_BACK_M * fx, float(u["z"]) - OUT_BACK_M * fz, h, u["n"]
        return float(lord["x"]) - 2 * OUT_BACK_M * fx, float(lord["z"]) - 2 * OUT_BACK_M * fz, h, None

    def _step_out(self, lord, inf, t, note):
        note["ph"] = "step_out"
        x, z, h, ally = self._out_point(lord, inf)
        d = math.hypot(float(lord["x"]) - x, float(lord["z"]) - z)
        self.step = {"x": x, "z": z, "face": h, "ally": ally, "t": t, "d": d, "resends": 0, "tried": [ally]}
        self.valour_due = VALOUR in self.btn
        note["step"] = {"x": round(x, 1), "z": round(z, 1), "d": round(d, 1), "ally": ally}
        return self._move(lord, x, z, h, t, kind="withdraw")

    def _stepping(self, lord, inf, t, note):
        s = self.step
        if s.get("failed"):
            note["ph"] = "step_failed"
            return {"unit": self.lord, "kind": "keep"}
        d = math.hypot(float(lord["x"]) - s["x"], float(lord["z"]) - s["z"])
        note.update(ph="stepping", step_d=round(d, 1))
        if d <= ARRIVED_M:
            self.out = True
            note["ph"] = "out_seen"
            return {"unit": self.lord, "kind": "keep"}
        names = {u["n"] for u in inf}
        gone = s["ally"] is not None and s["ally"] not in names
        stalled = t - s["t"] >= RESEND_S and s["d"] - d < PROGRESS_M
        if t - s["t"] >= RESEND_S and not stalled:
            s["t"], s["d"] = t, d                                 # (progress: the clock restarts)
        if not (gone or stalled):
            return {"unit": self.lord, "kind": "keep"}
        if s["resends"] >= RESENDS:
            s["failed"] = True
            note["ph"] = "step_failed"
            return {"unit": self.lord, "kind": "keep"}
        x, z, h, ally = self._out_point(lord, inf, avoid=s["tried"] if stalled else ())
        if math.hypot(x - s["x"], z - s["z"]) <= REORDER_M:          # (the bridge would filter the same point)
            fx, fz = fwd(h)
            x, z = x - 2 * REORDER_M * fz * (1 if s["resends"] % 2 == 0 else -1), z + 2 * REORDER_M * fx * (
                1 if s["resends"] % 2 == 0 else -1)
        s.update(x=x, z=z, face=h, ally=ally, t=t, d=math.hypot(float(lord["x"]) - x, float(lord["z"]) - z),
                 resends=s["resends"] + 1)
        s["tried"].append(ally)
        note["rechosen"] = {"why": "ally_gone" if gone else "no_progress", "n": s["resends"], "ally": ally}
        return self._move(lord, x, z, h, t, kind="withdraw")

    def _press(self, rows, t, note, melee, n_melee, n_waver, near, move):
        """At most one button (module doc's priority) -> [] or [{unit, key}]."""
        if not self.buttons:
            note["buttons"] = "off"
            return []
        want = []
        if self.valour_due and not self.out:
            want.append((VALOUR, "step_out"))
        if self.step is None:
            if n_melee >= SYG_MIN:
                want.append((SYG, "melee35"))
            if n_waver >= 1:
                want.append((RALLY, "waver35"))
            if melee:
                want += [(FOE_SEEKER, "self_melee"), (ONSLAUGHT, "self_melee")]
        for key, why in want:
            uses = self.request(rows, t, key, why, note, move=move, recipients=near, melee35=n_melee,
                                waver35=n_waver)
            if uses or note.get("card_unreadable"):
                return uses
        return []
