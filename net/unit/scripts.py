"""The unit level - a script, no network (METHOD.md 'Уровень', «Личный уровень — без сети»): the tactician's order to a
unit (a skill, its frame, its main target, the pace) is carried out by the script below, every second, in the unit
level's place (operative/lower.Units: the same latency, the order gate net/common/reissue.py, the leash). Whether to
start a skill, for whom and where is not its business; inside a skill it goes from phase to phase by facts only - in
the frame, in melee, the target within X m, the target routing or unseen. It reads only the input's columns of the
contract (net/unit/view.RAW: what the bridge gives as unit_input.game_raw), never the simulator's own state.

The skills of stage 0 (SKILLS; design/unit, METHOD.md table):
    hold      to the frame's centre (a run beyond RUN_M, the order's pace first) facing the frame's course; there
              (within the gate's MOVE_M, facing within its FACE_DEG) settled: a HOLD where it stands with no facing,
              kept while it is in the frame
    counter   'meet': hold; out of melee, the main target or else the nearest seen standing enemy coming at the unit
              (Facts.coming) within the counter's reach (CHARGE_M + its run x LEAD_S, edge to edge: ~35 m) - attack it
              at a run (the charge on the last metres), kept on that target while it stands, is seen and is within
              KEEP_M (no switch to a nearer one, no drop at the reach's edge); in melee no attack of its own (the
              reflexes strike / melee_still); its fight over out of the frame - a HOLD where it stands (not back to the
              frame: the same health, 0.10 fewer orders a unit-minute, tmp/tactic_slice/v2/stage1c). Not memorised:
              a frame moved while the unit holds out of it is not followed (the tactician's 'hold' brings it back)
    attack    attack the main target (a run unless the pace is slow) while it stands and is seen; then stand where it is.
              With the order's approach the target's left / right (drill T0-2, tmp/drill2/design_merged.md; order
              'approach', 'age' - s since the order; config tactic.json orders.side) first WALK: a MOVE at a run to P =
              the target's centre + side x (its front / 2 + off_m + our depth / 2) across its facing, from its raw
              columns each second (sent again only when P moved reissue.MOVE_M from the point sent and side.reissue_s
              passed since that MOVE), facing the target; the order's first second out of melee always walks; then
              the attack above (CHARGE) once within near_m of P or walk_max_s after the order - or at once, for good
              (FALLBACK), when no other standing unit of our side is in melee within STICK_M of the target, the target
              turned its front sector over the side P was on at the order's birth (the level's face0), or (side.cancel_comer, off: the game's probe) another standing seen
              enemy's centre is within cancel_clear_m of the way left (seen always: Unit.comer); an attack on the target already given is kept (one way: the order
              in force 'last' is the only memory). Unit.trace [B, N] - the PHASES code of each unit this second
              (game/probe/unit_probe.py side2r measured this walk: P 35 m off, 8 m, 30 s)
    withdraw  out of the frame: break off and go to its centre (WITHDRAW: a run), facing its course, no strike on the
              way; in the frame: hold (the game's price - 24 s with no blow, struck in the back - is paid once: the
              WITHDRAW is one order to the game, never flipped to a strike and back; a push out of the frame up to
              WD_SLACK_M after it came does not start it again)
The reflexes (REFLEXES, a short list with their mechanics):
    strike    in melee, strike the one touching (the nearest standing seen enemy in melee whose formation is within
              TOUCH_M of ours; the one it already attacks first while within STICK_M - its own order in force): ATTACK
              it. Not over 'attack' while the main target stands and is seen, not over
              'withdraw' out of the frame. The mechanic: a unit fights back under any order, but its order decides
              whom it strikes (sim.json contact.fresh_why: an order given in melee - another kind or target - and it
              strikes the others 0 HP for 24 s).
    melee_still  under hold / counter, in melee (its m column) with nobody touching it - a HOLD where it stands, no
              facing: a MOVE to a point 10 m off or a facing given in melee is a leave (sim.json contact.fresh_why,
              move_melee_leave, face_breakoff: 24 s with no blow, struck in the back).
    taken     under hold / counter, out of its frame and out of melee: a seen standing enemy's centre within TAKEN_M
              of the frame (the enemy holds it) - a HOLD where it stands facing that enemy: it does not walk into the
              enemy alone (a wider rule - an enemy within 60 m of a frame no own unit covers - stopped U2's unit short
              of its frame: in the frame at contact 0.33, met head-on 0.53) (game/manual.md §3: the game's AI sends its units into the line one by one - the punished
              stupidity). The cause of 'straight' (the script without it) trading worse than 'stand' in the general
              battles (tmp/u2/razbor2.md: battles 84, 10, 233 seen - a unit routed, rallied 100-200 m back and walked
              alone 500-1300 m back into its frame, into the enemy that broke it, and broke again).
    turn      (on; Unit(turn=False) without it) out of melee, standing in its frame under hold (counter's charge faces
              the one coming itself): the
              nearest seen standing enemy coming at it (its velocity towards us over APPROACH_MS; one in melee, whose
              velocity column is 0 by the bridge's rule: moving with us in its front - Facts.coming) within the turn's
              reach (CHARGE_M + its run x TURN_S) and off our front by more than FRONT_DEG - face it (a HOLD facing
              it): the brace and the stand-first strike work only for the front (game/manual.md 'Упор'; sim.json
              contact.front_deg).

A player: intend(raw, order, own) -> Orders [B, N] (KEEP for the slots not own); raw [B, N, len(view.RAW)] the input's
columns now; order {skill [B, N] long (SKILLS), area {x, z, b, front, depth} [B, N], target [B, N] long (-1 none),
pace [B, N] long (net.PACE_*)}.
    Unit()        the script (any skill)
    Stand()       the default ('hold'): stand where the unit is, no facing given; the strike reflex
    Random()      the bad one (run2's flaw in the game: a new course every second): hold with a facing at random
"""
import math

import torch

from sim import geometry
from sim import orders as O
from net.common import reissue
from net.unit import decode as Dc
from net.unit import net as Nt
from net.unit import view as V

SKILLS = ("hold", "counter", "attack", "withdraw")
HOLD, COUNTER, ATTACK, WITHDRAW = range(len(SKILLS))
REFLEXES = ("strike", "melee_still", "taken", "turn")

RUN_M = 40.0          # m to the frame's centre beyond which a hold runs (pace 'any')
# the edge gap of two formations in melee: the fronts part 1.1-1.4 m at contact, 3.0 m later (sim.json melee.wave_why,
# the wave probe), + 1 m of the sim's push (from the simulator, not checked in the game)
TOUCH_M = 4.0
# the counter's reach: the enemy's charge distance (the database's charge_distance_commence_run, 30 m for infantry) +
# its run x LEAD_S. The game (game/probe/evidence/counter_game.json, 13.11: swordsmen against running clanrats, the
# trade of 30 s): the attack ordered at a front-to-front gap of 17 / 26 / 37 m +0.038 / +0.047 / +0.032, at 46 m
# +0.026, at 68 m +0.017, at contact -0.014, never -0.067 - the plateau is 17-37 m, so the order goes at ~35 m and
# the unit stays in its line till then (spearmen gain nothing from it over an attack at contact: whom to give the
# skill is the tactician's business)
CHARGE_M = 30.0
LEAD_S = 1.0
# the turn's reach: CHARGE_M + its run x TURN_S - the time a standing unit needs to face about (the game, the reissue
# probe game/probe/evidence/reissue_game.json: a turn of 11-22.5 deg costs 7-8 s with no brace; the simulator's
# turn.face_delay 3.9 s) is ~35 m of an infantry run, + the charge distance: the turn is over before the contact
TURN_S = 4.0
# a counter's attack in force is kept while its target is within this edge gap: the game's counter probe still gains at
# its farthest lane (counter_game.json c60, the order at a measured 64-67 m: swordsmen +0.016 over 30 s against -0.069
# standing, sim/evidence/counter/spec.md) - the plateau's far end as the plan laid it; beyond, nothing is measured
KEEP_M = 60.0
STICK_M = 8.0         # the strike keeps its target while within this edge gap (TOUCH_M + a pushed rank's depth)
FRONT_DEG = 45.0      # our front sector (sim.json contact.front_deg: the sector a contact is labelled front by)
APPROACH_MS = 0.5     # 'coming at us': its velocity towards us over this (m/s; a walk is 1-1.5)
WD_SLACK_M = 20.0     # (doc: withdraw) a push this far out of the frame after the withdraw keeps it over
TAKEN_M = 10.0        # (doc: taken) an enemy centre this far out of the frame still holds it (the melee's push)


def _c(raw, k):
    return raw[..., V.I[k]]


class Facts:
    """The facts the script goes by, from the input's columns only (module doc)."""

    def __init__(self, raw):
        self.raw = raw
        side, alive = _c(raw, "side"), _c(raw, "alive") > 0
        self.alive = alive
        self.standing = alive & (_c(raw, "r") <= 0)
        self.melee = _c(raw, "m") > 0
        self.enemy = (side[:, :, None] != side[:, None, :]) & (side[:, None, :] > 0)          # [B, i, j]
        self.seen = (_c(raw, "vis") > 0)[:, None, :] | ~self.enemy
        self.gap = geometry.rect_gap(_c(raw, "x"), _c(raw, "z"), _c(raw, "b"), _c(raw, "front"), _c(raw, "depth"))
        self.foe = self.enemy & self.standing[:, None, :] & self.seen                         # standing, seen enemies

    def nearest(self, ok):
        """[B, N] the nearest j (by the edge gap) where ok [B, i, j], -1 none."""
        g = torch.where(ok, self.gap, torch.full_like(self.gap, float("inf")))
        j = g.argmin(2)
        return torch.where(ok.any(2), j, torch.full_like(j, -1))

    def touching(self):
        """[B, N] the one touching each unit (module doc: strike), -1 none."""
        ok = self.foe & self.melee[:, :, None] & self.melee[:, None, :] & (self.gap <= TOUCH_M)
        return self.nearest(ok)

    def strike(self, last=None):
        """[B, N] whom the strike reflex attacks (-1 none): the one it already attacks (last: its order in force) while
        that one stands, is seen, fights and is within STICK_M - the same attack again is no new order, another target
        in melee costs 24 s of blows on the others (sim.json contact.fresh_why); else the one touching."""
        j = self.touching()
        if last is None:
            return j
        lt = torch.where(last.kind == O.ATTACK, last.target, torch.full_like(last.target, -1))
        fights = self.melee & self.melee.gather(1, lt.clamp(min=0))
        stick = self.valid(lt) & fights & (self.gap_to(lt) <= STICK_M)
        return torch.where(stick, lt, j)

    def valid(self, t):
        """[B, N] the slot t stands and is seen (an enemy)."""
        tc = t.clamp(min=0)
        ok = self.foe.gather(2, tc[:, :, None])[:, :, 0]
        return (t >= 0) & ok

    def gap_to(self, t):
        return self.gap.gather(2, t.clamp(min=0)[:, :, None])[:, :, 0]

    def reach(self, t, lead_s=LEAD_S):
        """[B, N] the reach of enemy t: its charge distance + its run x lead_s (module doc: counter, turn)."""
        return CHARGE_M + _c(self.raw, "run").gather(1, t.clamp(min=0)) * lead_s

    def bearing(self, t):
        """[B, N] the world bearing from each unit to slot t, degrees 0..360."""
        x, z = _c(self.raw, "x"), _c(self.raw, "z")
        tc = t.clamp(min=0)
        return torch.remainder(torch.rad2deg(torch.atan2(x.gather(1, tc) - x, z.gather(1, tc) - z)), 360.0)

    def off_front(self, t):
        return (torch.remainder(self.bearing(t) - _c(self.raw, "b") + 180.0, 360.0) - 180.0).abs()

    def coming(self, t):
        """[B, N] enemy t comes at the unit: its velocity towards us over APPROACH_MS; or, where its velocity column says
        nothing - 0 by the bridge's rule for a unit in melee in both snapshots (view.shift_velocity: the centre of a
        formation in melee jitters) - it moves (its mv, the game's is_moving) in melee and has us in its front sector
        (FRONT_DEG of its facing: a unit walks along its facing). Mostly a pursuer or a unit sliding off a fight at a run
        (tmp/unit_vel: 92 % of the turns the shift column lost; the game's records: in melee with mv in both seconds the
        centre moves 2.2 m/s by the median, without mv 0.45)."""
        raw = self.raw
        tc = t.clamp(min=0)
        x, z = _c(raw, "x"), _c(raw, "z")
        wx, wz = x - x.gather(1, tc), z - z.gather(1, tc)
        n = torch.sqrt(wx * wx + wz * wz).clamp(min=1e-6)
        vx, vz = _c(raw, "vx").gather(1, tc), _c(raw, "vz").gather(1, tc)
        v = (vx * wx + vz * wz) / n
        g = lambda k: _c(raw, k).gather(1, tc)                                       # noqa: E731
        unread = (vx == 0) & (vz == 0) & (g("m") > 0) & (g("mv") > 0)
        to_us = torch.rad2deg(torch.atan2(wx, wz))
        faces = (torch.remainder(to_us - g("b") + 180.0, 360.0) - 180.0).abs() <= FRONT_DEG
        return (v > APPROACH_MS) | (unread & faces)

    def comes(self):
        """[B, i, j] coming(j) of each unit i for every slot j (the same rule; mask the enemies yourself)."""
        raw = self.raw
        x, z = _c(raw, "x"), _c(raw, "z")
        wx, wz = x[:, :, None] - x[:, None, :], z[:, :, None] - z[:, None, :]                # from j to i
        n = torch.sqrt(wx * wx + wz * wz).clamp(min=1e-6)
        vx, vz = _c(raw, "vx")[:, None, :], _c(raw, "vz")[:, None, :]
        v = (vx * wx + vz * wz) / n
        unread = (vx == 0) & (vz == 0) & (_c(raw, "m") > 0)[:, None, :] & (_c(raw, "mv") > 0)[:, None, :]
        to_us = torch.rad2deg(torch.atan2(wx, wz))
        faces = (torch.remainder(to_us - _c(raw, "b")[:, None, :] + 180.0, 360.0) - 180.0).abs() <= FRONT_DEG
        return (v > APPROACH_MS) | (unread & faces)


def _keep(raw):
    B, N = raw.shape[:2]
    o = O.hold(B, N, raw.device)
    o.kind[:] = O.KEEP
    return o


def _set(o, m, kind, x=None, z=None, target=None, run=None, face=None):
    """o's fields where m [B, N] set to these (None: leave; scalars broadcast)."""
    full = lambda v, ref: v if torch.is_tensor(v) else torch.full_like(ref, v)        # noqa: E731
    o.kind = torch.where(m, torch.full_like(o.kind, kind), o.kind)
    for f, v in (("x", x), ("z", z), ("target", target), ("run", run), ("face", face)):
        if v is not None:
            ref = getattr(o, f)
            setattr(o, f, torch.where(m, full(v, ref).to(ref.dtype), ref))
    return o


class Unit:
    """The unit's script (module doc). turn - the reflex 'turn to the one coming at our flank'; taken - 'the frame held by
    the enemy: stop' (both on)."""
    name = "unit"

    def __init__(self, turn=True, taken=True):
        self.turn = turn
        self.taken = taken
        self.trace = None
        self.comer = None

    def intend(self, raw, order, own):
        f = Facts(raw)
        skill, ar, tgt, pace = order["skill"], order["area"], order["target"], order["pace"]
        x, z = _c(raw, "x"), _c(raw, "z")
        o = _keep(raw)
        dist = torch.sqrt((ar["x"] - x) ** 2 + (ar["z"] - z) ** 2)
        run_to = torch.where(pace == Nt.PACE_FAST, torch.ones_like(own),
                             torch.where(pace == Nt.PACE_SLOW, torch.zeros_like(own), dist > RUN_M))
        course = torch.remainder(ar["b"], 360.0)
        sdf = Dc.signed_dist(x, z, ar)
        in_frame = sdf >= 0
        last = order.get("last")
        j = f.strike(last)
        # withdraw is over once the unit came to its frame: its order in force no longer the WITHDRAW; a push out of
        # the frame up to WD_SLACK_M does not start it again (another WITHDRAW in melee: 24 s with no blow again)
        came = in_frame
        if last is not None:
            came = came | ((sdf >= -WD_SLACK_M) & (last.kind != O.WITHDRAW) & (last.kind != O.KEEP))
        # hold (and the base of counter, and of withdraw once in its frame): to the centre facing the course
        base = own & ((skill == HOLD) | (skill == COUNTER) | ((skill == WITHDRAW) & came))
        o = _set(o, base, O.MOVE, x=ar["x"], z=ar["z"], run=run_to, face=course)
        # settled: at the centre facing the course (or settled before - its order in force a hold - and still in the
        # frame) - a HOLD where it stands, no facing: a MOVE or a facing left in force keeps the formation from turning
        # to the enemy at its side (sim.json contact.push_keep_facing, turn.face_melee_why), the default does not
        off = (torch.remainder(_c(raw, "b") - course + 180.0, 360.0) - 180.0).abs()
        arrived = (dist <= reissue.MOVE_M) & (off <= reissue.FACE_DEG)
        if last is not None:
            arrived = arrived | (last.kind == O.HOLD)
        o = _set(o, base & in_frame & arrived, O.HOLD, x=x, z=z, run=False, face=-1.0)
        # melee_still: in melee, nobody touching - stand as it is
        o = _set(o, base & f.melee & (j < 0), O.HOLD, x=x, z=z, run=False, face=-1.0)
        if self.taken:
            # taken: out of the frame and of melee, an enemy holds the frame - stop facing it
            a1 = {k: v[:, :, None] for k, v in ar.items() if k in ("x", "z", "b", "front", "depth")}
            sd = Dc.signed_dist(_c(raw, "x")[:, None, :], _c(raw, "z")[:, None, :], a1)        # [B, i frame, j]
            held = f.foe & (sd >= -TAKEN_M)
            k = f.nearest(held)
            stop = base & ~in_frame & ~f.melee & (k >= 0)
            o = _set(o, stop, O.HOLD, x=x, z=z, run=False, face=f.bearing(k))
        # counter (module doc): out of melee - the attack in force kept on its target while that one stands, is seen and
        # is within KEEP_M; else the main target within its reach, else the nearest enemy coming at the unit within its
        # reach (CHARGE_M + its run x LEAD_S); none, out of the frame and its fight over - stand where it is
        main_ok = f.valid(tgt)
        cnt = own & (skill == COUNTER) & ~f.melee
        pool = f.foe & f.comes() & (f.gap <= CHARGE_M + _c(raw, "run")[:, None, :] * LEAD_S)
        c_t = torch.where(main_ok & (f.gap_to(tgt) <= f.reach(tgt)), tgt, f.nearest(pool))
        if last is not None:
            lt = torch.where(last.kind == O.ATTACK, last.target, torch.full_like(last.target, -1))
            c_t = torch.where(f.valid(lt) & (f.gap_to(lt) <= KEEP_M), lt, c_t)
            after = cnt & ~in_frame & (c_t < 0) & ((last.kind == O.ATTACK) | (last.kind == O.HOLD))
            o = _set(o, after, O.HOLD, x=x, z=z, run=False, face=-1.0)
        charge = cnt & (c_t >= 0)
        o = _set(o, charge, O.ATTACK, target=c_t, run=True, face=-1.0)
        # attack: the main target while it stands and is seen; else stand where it is
        att = own & (skill == ATTACK)
        o = _set(o, att & main_ok, O.ATTACK, target=tgt, run=pace != Nt.PACE_SLOW, face=-1.0)
        o = _set(o, att & ~main_ok, O.HOLD, x=x, z=z, run=False, face=-1.0)
        # attack round the target's flank (drill T0-2; module doc 'attack'): WALK to P, then the attack above
        ap = order.get("approach")
        self.trace = None
        self.comer = None
        if ap is not None and bool((att & (ap > 0)).any()):
            walk, code, px, pz, pb, hold = self.flank(f, raw, att & (ap > 0), tgt, ap, order.get("age"), last, main_ok,
                                                      order.get("walk_s"), order.get("face0"), order.get("move_s"))
            o = _set(o, walk, O.MOVE, x=px, z=pz, run=pace != Nt.PACE_SLOW, face=pb)
            if hold is not None:
                # (the walk's MOVE in force given again as it was: no new order to the game this second)
                o = _set(o, hold, O.MOVE, x=last.x, z=last.z, run=last.run, face=last.face)
            self.trace = code
        # withdraw: out of its frame - break off and go to the centre (one WITHDRAW, kept: no strike on the way)
        wd = own & (skill == WITHDRAW) & ~came
        o = _set(o, wd, O.WITHDRAW, x=ar["x"], z=ar["z"], run=True, face=course)
        # the reflexes
        if self.turn:
            n = f.nearest(f.foe)
            standing_in = base & (skill != COUNTER) & in_frame & ~f.melee & (n >= 0)
            side_on = standing_in & f.coming(n) & (f.gap_to(n) <= f.reach(n, TURN_S)) & (f.off_front(n) > FRONT_DEG)
            o = _set(o, side_on, O.HOLD, x=x, z=z, run=False, face=f.bearing(n))
        busy = (att & main_ok) | wd
        hit = own & (j >= 0) & ~busy
        o = _set(o, hit, O.ATTACK, target=j, run=True, face=-1.0)
        return o

    def flank(self, f, raw, on, tgt, ap, age, last, main_ok, walk_s=None, face0=None, move_s=None):
        """(walk [B, N], code [B, N] PHASES index, px, pz, pb [B, N], hold [B, N] or None) of the units on [B, N] under
        an attack whose approach ap is the target's left (1) or right (2) (module doc 'attack'): P from the target's raw
        columns now, the facing at P towards the target; walk - the unit goes on to P (MOVE), else the attack in force
        stands: arrived, late, or a fact of the fallback; an attack on the target already given (last) is kept - one
        way. face0 - the target's facing at the order's birth (birth(); the side P was on then), move_s - seconds since
        the unit's last MOVE sent to the game (sent_move()); hold - walking, the MOVE in force given again as it was
        (P moved less than reissue.MOVE_M from the point sent, or less than side.reissue_s since that MOVE)."""
        c = side_cfg()
        B, N = tgt.shape
        dev = raw.device
        tc = tgt.clamp(min=0)
        x, z = _c(raw, "x"), _c(raw, "z")
        tg = lambda k: _c(raw, k).gather(1, tc)                                         # noqa: E731
        tb = tg("b")
        px, pz, pb = flank_point(raw, tgt, ap)
        z0 = torch.zeros((B, N), dtype=torch.bool, device=dev)
        charged = z0 if last is None else (last.kind == O.ATTACK) & (last.target == tgt)
        if last is not None and age is not None:
            # a fresh flank order (its first second) starts with the walk whatever the unit did before - out of melee
            # an attack in force on the same target (a counter's charge) is no strike begun; in melee it is kept
            charged = charged & ((age >= 0.5) | f.melee)
            # one way: past the order's first second the walk goes on only from its own MOVE (a hold sent while the
            # target was unseen, an attack - the walk is over)
            charged = charged | ((age >= 0.5) & (last.kind != O.MOVE))
        # the anvil: another standing unit of my side in melee within STICK_M of the target (edge to edge)
        side = _c(raw, "side")
        eye = torch.eye(N, dtype=torch.bool, device=dev)[None]
        ours = (side[:, :, None] == side[:, None, :]) & f.standing[:, None, :] & ~eye
        gt = f.gap.gather(1, tc[:, :, None].expand(B, N, N))                            # [b, i, j] gap(tgt_i, j)
        no_anvil = ~(ours & f.melee[:, None, :] & (gt <= STICK_M)).any(2)
        # the target turned its front to P's side: the side P was on at the order's birth (the target's facing then
        # + 90 deg to its right, - 90 to its left - flank_point's geometry) now in its front sector; measured from the
        # birth, not from the last MOVE (P turns with the target: the point last sent never enters its front)
        turned = z0
        if face0 is not None:
            sg = torch.where(ap == 1, torch.full_like(tb, -1.0), torch.ones_like(tb))
            w0 = face0 + sg * 90.0
            turned = (torch.isfinite(face0)
                      & ((torch.remainder(w0 - tb + 180.0, 360.0) - 180.0).abs() < FRONT_DEG))
        # a comer: another standing seen enemy's centre within cancel_clear_m of the way left - always seen (self.comer,
        # for the exam), a fallback only with the switch (config side.cancel_comer; CANCEL_COMER overrides it)
        other = f.foe & (torch.arange(N, device=dev)[None, None, :] != tc[:, :, None])
        d = seg_dist(x[:, None, :], z[:, None, :], x[:, :, None], z[:, :, None], px[:, :, None], pz[:, :, None])
        seen = (other & (d < float(c["cancel_clear_m"]))).any(2)
        switch = bool(c.get("cancel_comer", False)) if CANCEL_COMER is None else bool(CANCEL_COMER)
        comer = seen if switch else z0
        arrived = torch.sqrt((px - x) ** 2 + (pz - z) ** 2) <= float(c["near_m"])
        # the walk's time (a guard against a stuck unit, not the drill's rule): walk_s of the order - walk_limit at the
        # order (the level's Units); without it the flat walk_max_s
        lim = walk_s if walk_s is not None else torch.full_like(x, float(c["walk_max_s"]))
        late = z0 if age is None else age >= lim
        code = torch.full((B, N), PHASES.index("walk"), dtype=torch.long, device=dev)
        for name, m in (("timeout", late), ("arrived", arrived), ("fb_comer", comer), ("fb_turned", turned),
                        ("fb_anvil", no_anvil), ("attacking", charged), ("fb_invalid", ~main_ok)):
            code = torch.where(m, torch.full_like(code, PHASES.index(name)), code)
        code = torch.where(on, code, torch.zeros_like(code))
        walk = on & (code == PHASES.index("walk"))
        self.comer = on & seen & ((code == PHASES.index("walk")) | (code == PHASES.index("fb_comer")))
        hold = None
        if last is not None and age is not None:
            # the walk's MOVE again only when P moved MOVE_M from the point sent AND reissue_s passed since that MOVE
            # (arrival above is tested against P now): a turning target moves P every second
            moved = (px - last.x) ** 2 + (pz - last.z) ** 2 >= reissue.MOVE_M ** 2
            if move_s is not None:
                moved = moved & (move_s >= float(c["reissue_s"]))
            hold = walk & (age >= 0.5) & (last.kind == O.MOVE) & ~moved
        return walk, code, px, pz, pb, hold


def flank_point(raw, tgt, ap):
    """(px, pz, pb [B, N]) P of each unit at its target tgt's side ap (1 left, else right; orders.side off_m): the
    target's centre + side x (front / 2 + off_m + own depth / 2) across its facing; pb faces the target."""
    c = side_cfg()
    tc = tgt.clamp(min=0)
    x = _c(raw, "x")
    tg = lambda k: _c(raw, k).gather(1, tc)                                             # noqa: E731
    tx, tz, tb = tg("x"), tg("z"), tg("b")
    sg = torch.where(ap == 1, torch.full_like(x, -1.0), torch.ones_like(x))
    half = tg("front") / 2 + float(c["off_m"]) + _c(raw, "depth") / 2
    th = torch.deg2rad(tb)
    px, pz = tx + sg * half * torch.cos(th), tz - sg * half * torch.sin(th)
    pb = torch.remainder(torch.rad2deg(torch.atan2(tx - px, tz - pz)), 360.0)
    return px, pz, pb


def walk_limit(raw, tgt, ap):
    """(walk_s, dist [B, N]) at the order: walk_base_s + walk_k x |unit -> P| / the unit's run speed (raw 'run', m/s;
    orders.side) - the walk's time guard; dist the way's length, m."""
    c = side_cfg()
    px, pz, _ = flank_point(raw, tgt, ap)
    d = torch.sqrt((px - _c(raw, "x")) ** 2 + (pz - _c(raw, "z")) ** 2)
    run = _c(raw, "run").clamp(min=0.5)
    return float(c["walk_base_s"]) + float(c["walk_k"]) * d / run, d


def birth(raw, tgt):
    """[B, N] the target's facing (raw 'b', deg) at a flank order's birth - the side its P was on then (Unit.flank's
    'turned'); NaN with no target."""
    b = _c(raw, "b").gather(1, tgt.clamp(min=0))
    return torch.where(tgt >= 0, b, torch.full_like(b, float("nan")))


def sent_move(intended, gate):
    """[B, N] the intended orders that go to the game as a new MOVE (reissue.Gate's own test against the last sent;
    no gate: every MOVE) - before gate.apply; the level keeps the second of each for Unit.flank's move_s."""
    mv = intended.kind == O.MOVE
    if gate is None:
        return mv
    return mv & reissue.differs(intended, gate.last, gate.move_m, gate.face_deg, gate.width_m)


CANCEL_COMER = None     # the exam's override of config side.cancel_comer (None: the config; exam_side TEACH2_ON)

PHASES =("none", "walk", "arrived", "timeout", "fb_anvil", "fb_turned", "fb_comer", "fb_invalid", "attacking")


def side_cfg():
    """The flank approach's settings (drill T0-2: net/config/tactic.json orders.side)."""
    from net.tactic import cfg
    return cfg()["orders"]["side"]


def seg_dist(ps, pz, ax, az, bx, bz):
    """Distance of points (ps, pz) to the segments a -> b (broadcast)."""
    dx, dz = bx - ax, bz - az
    L2 = (dx * dx + dz * dz).clamp(min=1e-9)
    t = (((ps - ax) * dx + (pz - az) * dz) / L2).clamp(0.0, 1.0)
    return torch.sqrt((ps - ax - t * dx) ** 2 + (pz - az - t * dz) ** 2)


class Stand:
    """The default ('hold' as the game's units stand): a HOLD where the unit is, no facing; the strike reflex."""
    name = "stand"

    def intend(self, raw, order, own):
        f = Facts(raw)
        o = _set(_keep(raw), own, O.HOLD, x=_c(raw, "x"), z=_c(raw, "z"), run=False)
        j = f.strike(order.get("last"))
        return _set(o, own & (j >= 0), O.ATTACK, target=j, run=True, face=-1.0)


class Straight(Unit):
    """The script without the reflex 'taken' (tmp/u2/razbor.md 'прямо' with the input's columns): hold, walking back to
    its frame's centre whatever holds it."""
    name = "straight"

    def __init__(self):
        super().__init__(taken=False)


class Random(Unit):
    """The bad one: hold with a new facing at random every second (the formation never re-forms)."""
    name = "random"

    def __init__(self, seed=0):
        super().__init__()
        self.gen, self.seed = None, seed

    def intend(self, raw, order, own):
        if self.gen is None or self.gen.device != raw.device:
            self.gen = torch.Generator(device=raw.device).manual_seed(self.seed)
        b = torch.rand(order["area"]["b"].shape, generator=self.gen, device=raw.device) * 360.0
        return super().intend(raw, {**order, "area": {**order["area"], "b": b}}, own)


PLAYERS = {"unit": Unit, "stand": Stand, "random": Random, "straight": Straight,
           "unit_noturn": lambda: Unit(turn=False)}


def player(name):
    return PLAYERS[name]()
