"""Layer 1: the bonus bricks (docs/en/training/ability_price.md, the agreed formulas build/abilities_bonus/formulas.md).

A brick is one kind of bonus (melee damage, charge bonus, melee defence, morale, speed, charge speed, vigour). It counts
the bonus's USEFUL work on a trajectory - only the seconds (or impacts, or morale points) where the bonus could change
something - as a number >= 0, and declares what it does (Description): the condition under which it works, whether the
counter grows in proportion to the bonus's size, buff or debuff, its unit. The rate k (gold per counter unit) belongs to
the KIND of bonus, not to an ability: +50 % damage gives twice the units of +25 %.

The trajectory (TRACE) of one press: samples a second apart (dt_s), sample 0 the moment of the press (before it acts),
sample s the state after second s; U units - the presser's side, the presser `owner` among them. mask [S+1, U]: the
press's bonus is on that unit during second s (layer 2, ability.py; mask[0] is False). Every count is over s = 1..S.

    count(tr, mask, value) -> float >= 0      the counter
    switch_off(tr, mask, value) -> trace      the same trajectory with the brick's condition false everywhere (the
                                              validator's test: its counter must be 0)

Only numpy.
"""
from dataclasses import dataclass

import numpy as np

# The trajectory's fields: name -> (kind, meaning); "su" [S+1, U], "s" [S+1], "u" [U], "1" a number
TRACE = {
    "x": ("su", "position east, m"),
    "z": ("su", "position north, m"),
    "alive": ("su", "present, men left, not departed"),
    "r": ("su", "routing"),
    "m": ("su", "in melee (the simulator's flag `m`)"),
    "mv": ("su", "moving (walking or running: `mv`)"),
    "sprint": ("su", "the charge sprint: an attack order, not in melee, moving, its target's edge within charge_dist"),
    "charge": ("su", "the charge left, 0-1 (1 at a charge's first blow, then down over 13 s)"),
    "morale": ("su", "morale points"),
    "fatigue": ("su", "fatigue points (0 fresh .. 30000)"),
    "mmax": ("u", "its morale's normal top: leadership + its start bonus (morale points)"),
    "on": ("s", "the owner's ability: active seconds left"),
    "owner": ("1", "the presser's index among the U units"),
    "dt_s": ("1", "seconds between samples"),
}
FLAGS = ("alive", "r", "m", "mv", "sprint")

# The game's rules a brick reads (config/nn/game_rules.json; defaults = the database's numbers)
MORALE_PCT = 0.15          # morale.percent_update_per_tick: the points move this share of the gap a tick
MORALE_MIN = 1.0           # morale.minimium_increment_update_per_tick: at least this many points a tick
TICK_S = 0.5               # a morale tick, s
FATIGUE_THRESHOLDS = (0.0, 2800.0, 6600.0, 12600.0, 18000.0, 27000.0)    # fresh .. exhausted
FATIGUE_MAX = 30000.0
# fatigue points a tick by activity (the database's _kv_fatigue_tables) and ticks a second (sim.json fatigue.calibration):
# only to rebuild a second where the recorded fatigue sat at 0 or the maximum (its change there is cut off)
FATIGUE_RATES = {"combat": 19.0, "charging": 34.0, "running": 4.0, "ready": -7.0}
FATIGUE_TICKS_S = 10.0
RISE = 0.05                # a charge counts as a new impact when the charge left rises by more than this in a second


@dataclass(frozen=True)
class Rules:
    """The game's numbers the bricks need (rules() reads them from game_rules.json)."""
    morale_pct: float = MORALE_PCT
    morale_min: float = MORALE_MIN
    tick_s: float = TICK_S
    fatigue_thresholds: tuple = FATIGUE_THRESHOLDS
    fatigue_max: float = FATIGUE_MAX
    fatigue_rates: tuple = tuple(FATIGUE_RATES.items())
    fatigue_ticks_s: float = FATIGUE_TICKS_S


def rules(game_rules=None, sim=None):
    """Rules from config/nn/game_rules.json's dict and sim.json's (None: the defaults, the same numbers)."""
    if game_rules is None:
        return Rules()
    m, f = game_rules["morale"], game_rules["fatigue"]
    th = tuple(float(f[k]) for k in ("threshold_fresh", "threshold_active", "threshold_winded", "threshold_tired",
                                     "threshold_very_tired", "threshold_exhausted"))
    rates = tuple((k, float(f[k])) for k in FATIGUE_RATES)
    ticks = float(((sim or {}).get("fatigue") or {}).get("calibration", {}).get("per_second", FATIGUE_TICKS_S))
    return Rules(float(m["percent_update_per_tick"]), float(m["minimium_increment_update_per_tick"]), TICK_S, th,
                 float(f["threshold_max"]), rates, ticks)


@dataclass(frozen=True)
class Description:
    """What a brick declares (the validator builds its tests from it, knowing nothing else about the bonus)."""
    kind: str               # the bonus kind (its rate k is per kind)
    condition: str          # in words: when the bonus does useful work
    proportional: bool      # the counter grows in proportion to the bonus's size (mult: value - 1; add / rate: value)
    effect: str             # "buff" (adds to the price) or "debuff" (subtracts)
    how: str                # "mult", "add" or "rate": how the bonus's size scales (scale())
    unit: str               # the counter's unit
    stats: tuple            # the database's (stat, how) the brick prices (ability.py maps a passport's effects by it)
    example: float = 1.0    # a typical bonus value (the validator's tests)


def scale(how, value, f):
    """The bonus `value` made f times as big (mult: 1 + (value - 1) f; add, rate: value f)."""
    return 1.0 + (value - 1.0) * f if how == "mult" else value * f


def size(how, value):
    """The bonus's size: mult (value - 1) x 100 (per cent), add / rate the value."""
    return (value - 1.0) * 100.0 if how == "mult" else value


class Brick:
    desc: Description

    def count(self, tr, mask, value):
        raise NotImplementedError

    def switch_off(self, tr, mask, value):
        raise NotImplementedError


def _on(tr, mask):
    """[S+1, U] the bonus on a living unit, sample 0 excluded."""
    on = mask & tr["alive"]
    on = on.copy()
    on[0] = False
    return on


def _with(tr, **kw):
    out = dict(tr)
    out.update(kw)
    return out


class _Seconds(Brick):
    """size x seconds of the affected units while the condition (_cond) holds."""
    def _cond(self, tr):
        raise NotImplementedError

    def count(self, tr, mask, value):
        on = _on(tr, mask) & self._cond(tr)
        return float(max(0.0, size(self.desc.how, value)) * on.sum() * tr["dt_s"])


class MeleeDamage(_Seconds):
    desc = Description("melee_damage", "the unit is in melee", True, "buff", "mult", "% x s",
                       (("stat_melee_damage_base", "mult"), ("stat_melee_damage_ap", "mult")), 1.25)

    def _cond(self, tr):
        return tr["m"]

    def switch_off(self, tr, mask, value):
        return _with(tr, m=np.zeros_like(tr["m"]))


class MeleeDefence(_Seconds):
    desc = Description("melee_defence", "the unit is in melee", True, "buff", "add", "points x s",
                       (("stat_melee_defence", "add"),), 24.0)

    def _cond(self, tr):
        return tr["m"]

    def switch_off(self, tr, mask, value):
        return _with(tr, m=np.zeros_like(tr["m"]))


class Speed(_Seconds):
    desc = Description("speed", "the unit walks or runs and is not in melee", True, "buff", "mult", "% x s",
                       (("scalar_speed", "mult"),), 1.25)

    def _cond(self, tr):
        return tr["mv"] & ~tr["m"]

    def switch_off(self, tr, mask, value):
        return _with(tr, mv=np.zeros_like(tr["mv"]))


class ChargeSpeed(_Seconds):
    desc = Description("charge_speed", "the unit runs its charge sprint at an enemy", True, "buff", "mult", "% x s",
                       (("scalar_charge_speed", "mult"),), 1.25)

    def _cond(self, tr):
        return tr["sprint"]

    def switch_off(self, tr, mask, value):
        return _with(tr, sprint=np.zeros_like(tr["sprint"]))


class Charge(Brick):
    """size x the charge impacts of the affected units: seconds where the charge left rises (a new charge's first
    blow sets it to 1)."""
    desc = Description("charge", "a charge's impact (the first blow of a charge) on an enemy", True, "buff", "mult",
                       "% x impact", (("stat_charge_bonus", "mult"),), 1.6)

    def count(self, tr, mask, value):
        c = tr["charge"].astype(np.float64)
        rise = np.zeros_like(tr["alive"])
        rise[1:] = c[1:] > c[:-1] + RISE
        return float(max(0.0, size("mult", value)) * (_on(tr, mask) & rise).sum())

    def switch_off(self, tr, mask, value):
        return _with(tr, charge=np.zeros_like(tr["charge"]))


def morale_lift(mask, value, dt_s, rules=Rules()):
    """[S+1, U] how far the bonus has lifted each unit's morale points by each sample: the morale's own pace (each
    tick the points move max(morale_min, morale_pct x gap) towards the target), towards +value while it is on, back
    towards 0 after."""
    S1, U = mask.shape
    e = np.zeros(U)
    out = np.zeros((S1, U))
    ticks = max(1, int(round(dt_s / rules.tick_s)))
    for s in range(1, S1):
        goal = np.where(mask[s], value, 0.0)
        for _ in range(ticks):
            gap = goal - e
            move = np.maximum(rules.morale_min, rules.morale_pct * np.abs(gap))
            e = e + np.clip(gap, -move, move)
        out[s] = e
    return out


class Morale(Brick):
    """Per affected unit: min(bonus, the most its morale fell short of its normal top while the bonus was on, the
    bonus's own lift taken out) - the deficit at the press plus its growth during the action, at most the bonus; a unit
    at its top all along gets 0."""
    desc = Description("morale", "the unit's morale is below its normal top (leadership + start bonus)", False, "buff",
                       "add", "morale points", (("stat_morale", "add"),), 16.0)

    def __init__(self, rules=Rules()):
        self.rules = rules

    def _deficit(self, tr, mask, value):
        lift = morale_lift(mask & tr["alive"], value, tr["dt_s"], self.rules)
        return tr["mmax"][None, :] - (tr["morale"] - lift)

    def count(self, tr, mask, value):
        if value <= 0:
            return 0.0
        on = _on(tr, mask)
        d = self._deficit(tr, mask, value)
        # the seconds it is on, and the sample just before each start (the deficit when the bonus reaches the unit:
        # at the press for the units in range then)
        hit = on.copy()
        hit[:-1] |= on[1:] & ~on[:-1]
        best = np.where(hit, np.clip(d, 0.0, None), 0.0).max(0)
        return float(np.minimum(value, best).sum())

    def switch_off(self, tr, mask, value):
        lift = morale_lift(mask & tr["alive"], value, tr["dt_s"], self.rules)
        return _with(tr, morale=tr["mmax"][None, :] + lift)


def tier(f, thresholds=FATIGUE_THRESHOLDS):
    return np.searchsorted(np.asarray(thresholds), f, side="right") - 1


class _Vigour(Brick):
    """Seconds the unit fights or moves one fatigue tier apart from where it would be without the ability: its fatigue
    without it rebuilt from the recorded one - each second's recorded change less the ability's (value x the maximum a
    second while the bonus is on), kept within 0..max; a second the recorded fatigue sat at 0 or the maximum (its
    change cut off) takes the activity's own rate instead (the database's points a tick: melee - combat, the sprint -
    charging, moving - running, else ready). From the press to the horizon (the gap lasts after the ability ends until
    the fatigue reaches 0 on both)."""
    sign = 0                # +1: the actual unit is in the WORSE tier (debuff), -1: in the better (buff)

    def __init__(self, rules=Rules()):
        self.rules = rules

    def counterfactual(self, tr, mask, value):
        F = tr["fatigue"].astype(np.float64)
        on = (mask & tr["alive"]).astype(np.float64)
        on[0] = 0.0
        cf = F.copy()
        R = dict(self.rules.fatigue_rates)
        tick = self.rules.fatigue_ticks_s * tr["dt_s"]
        own = np.where(tr["m"], R["combat"], np.where(tr["sprint"], R["charging"],
                                                      np.where(tr["mv"], R["running"], R["ready"]))) * tick
        top = self.rules.fatigue_max
        for s in range(1, len(F)):
            ab = value * top * tr["dt_s"] * on[s]
            cut = (F[s] <= 0.0) | (F[s] >= top)
            act = np.where(cut, own[s], (F[s] - F[s - 1]) - ab)
            cf[s] = np.clip(cf[s - 1] + act, 0.0, top)
        return cf

    def count(self, tr, mask, value):
        if value * self.sign <= 0:
            return 0.0
        cf = self.counterfactual(tr, mask, value)
        th = self.rules.fatigue_thresholds
        apart = (tier(tr["fatigue"], th) - tier(cf, th)) * self.sign > 0
        busy = (tr["m"] | tr["mv"]) & tr["alive"]
        hit = mask.any(0)[None, :] & apart & busy
        hit[0] = False
        return float(hit.sum() * tr["dt_s"])

    def switch_off(self, tr, mask, value):
        return _with(tr, m=np.zeros_like(tr["m"]), mv=np.zeros_like(tr["mv"]))


class VigourGain(_Vigour):
    desc = Description("vigour", "the ability left the unit a fatigue tier fresher while it fights or moves", False,
                       "buff", "rate", "s", (("vigour_per_s", "recover"),), -0.01)
    sign = -1


class FatigueCost(_Vigour):
    desc = Description("fatigue_cost", "the ability left the unit a fatigue tier more tired while it fights or moves",
                       False, "debuff", "rate", "s", (("vigour_per_s", "drain"),), 0.01)
    sign = +1


def bricks(rules=Rules()):
    """{kind: brick} of every brick."""
    out = [MeleeDamage(), Charge(), MeleeDefence(), Morale(rules), Speed(), ChargeSpeed(), VigourGain(rules),
           FatigueCost(rules)]
    return {b.desc.kind: b for b in out}
