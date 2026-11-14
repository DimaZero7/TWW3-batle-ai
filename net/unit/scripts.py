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
    counter   hold; the target (the main one, else the nearest seen standing enemy) within the counter's reach
              (CHARGE_M + its run x LEAD_S, edge to edge: ~35 m) - attack it at a run (the charge on the last metres);
              back to hold once it routs, dies or is unseen
    attack    attack the main target (a run unless the pace is slow) while it stands and is seen; then stand where it is
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
    turn      (on; Unit(turn=False) without it) out of melee, standing in its frame under hold / counter: the
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
        # counter: the target within its reach - attack it at a run
        main_ok = f.valid(tgt)
        c_t = torch.where(main_ok, tgt, f.nearest(f.foe))
        charge = own & (skill == COUNTER) & (c_t >= 0) & (f.gap_to(c_t) <= f.reach(c_t))
        o = _set(o, charge, O.ATTACK, target=c_t, run=True, face=-1.0)
        # attack: the main target while it stands and is seen; else stand where it is
        att = own & (skill == ATTACK)
        o = _set(o, att & main_ok, O.ATTACK, target=tgt, run=pace != Nt.PACE_SLOW, face=-1.0)
        o = _set(o, att & ~main_ok, O.HOLD, x=x, z=z, run=False, face=-1.0)
        # withdraw: out of its frame - break off and go to the centre (one WITHDRAW, kept: no strike on the way)
        wd = own & (skill == WITHDRAW) & ~came
        o = _set(o, wd, O.WITHDRAW, x=ar["x"], z=ar["z"], run=True, face=course)
        # the reflexes
        if self.turn:
            n = f.nearest(f.foe)
            standing_in = (base & ~charge) & in_frame & ~f.melee & (n >= 0)
            side_on = standing_in & f.coming(n) & (f.gap_to(n) <= f.reach(n, TURN_S)) & (f.off_front(n) > FRONT_DEG)
            o = _set(o, side_on, O.HOLD, x=x, z=z, run=False, face=f.bearing(n))
        busy = (att & main_ok) | (charge & f.melee) | wd
        hit = own & (j >= 0) & ~busy
        o = _set(o, hit, O.ATTACK, target=j, run=True, face=-1.0)
        return o


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
