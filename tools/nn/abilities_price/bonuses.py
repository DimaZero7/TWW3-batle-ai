"""Layer 1: the bonus bricks (docs/en/training/ability_price.md, the agreed formulas build/abilities_bonus/formulas.md).

A brick is one kind of bonus (melee damage, charge bonus, melee defence, morale, speed, charge speed, vigour; and what a
spell does - its bombardment's damage to enemies, its damage to friends, the enemy men it knocks down, the winds it
spends: SpellDamage, SpellFriendly, SpellKnock, WindsCost, read from the simulator's own bookkeeping). It counts
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
    # spells (16.10.2026; the simulator's bookkeeping of the presser, state bm_gold_e / bm_gold_f / bm_knock, wm_cur;
    # 0 for a presser without spells)
    "sp_gold_e": ("s", "the gold of enemy health the presser's bombardments have taken so far"),
    "sp_gold_f": ("s", "the gold of his own side's health they have taken so far (friendly fire)"),
    "sp_knock": ("s", "enemy men they have knocked down so far (knock events)"),
    "winds": ("s", "his side's current winds of magic"),
}
SPELL_FIELDS = ("sp_gold_e", "sp_gold_f", "sp_knock", "winds")
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
# the morale states' edges in points (game_rules.json ums_*): wavering below ums_wavering_threshold_upper (16), routing
# at / below ums_wavering_threshold_lower (0), shaken below ums_steady_threshold_lower (30)
MORALE_ROUT = 0.0
MORALE_WAVER = 16.0
MORALE_STEADY = 30.0
# the fatigue states' stat multipliers (the database's unit_fatigue_effects_tables, config/nn/sim.json fatigue.effects;
# missing x1), tier 0 fresh .. 5 exhausted
FATIGUE_EFFECTS = (
    (),
    (("stat_melee_attack", 0.95),),
    (("scalar_speed", 0.95), ("stat_melee_attack", 0.95), ("stat_melee_damage_ap", 0.9)),
    (("scalar_speed", 0.9), ("stat_melee_attack", 0.85), ("stat_charge_bonus", 0.9), ("stat_melee_damage_ap", 0.9)),
    (("scalar_speed", 0.85), ("stat_melee_attack", 0.75), ("stat_armour", 0.9), ("stat_charge_bonus", 0.75),
     ("stat_melee_damage_ap", 0.9)),
    (("scalar_speed", 0.85), ("stat_melee_attack", 0.7), ("stat_melee_defence", 0.9), ("stat_armour", 0.75),
     ("stat_charge_bonus", 0.7), ("stat_melee_damage_ap", 0.9), ("stat_reloading", 0.9)),
)
FATIGUE_LEVELS = ("threshold_fresh", "threshold_active", "threshold_winded", "threshold_tired", "threshold_very_tired",
                  "threshold_exhausted")
# which of a tier's stats work when (a fixed choice, not fitted): in melee the melee stats, on the move its speed; a
# charge's impact its charge bonus (a second the charge left rises, as Charge)
MELEE_STATS = ("stat_melee_attack", "stat_melee_defence", "stat_melee_damage_ap", "stat_armour")
MOVE_STATS = ("scalar_speed",)
IMPACT_STATS = ("stat_charge_bonus",)
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
    morale_rout: float = MORALE_ROUT
    morale_waver: float = MORALE_WAVER
    morale_steady: float = MORALE_STEADY
    fatigue_effects: tuple = FATIGUE_EFFECTS


def rules(game_rules=None, sim=None):
    """Rules from config/nn/game_rules.json's dict and sim.json's (None: the defaults, the same numbers)."""
    if game_rules is None:
        return Rules()
    m, f = game_rules["morale"], game_rules["fatigue"]
    th = tuple(float(f[k]) for k in ("threshold_fresh", "threshold_active", "threshold_winded", "threshold_tired",
                                     "threshold_very_tired", "threshold_exhausted"))
    rates = tuple((k, float(f[k])) for k in FATIGUE_RATES)
    ticks = float(((sim or {}).get("fatigue") or {}).get("calibration", {}).get("per_second", FATIGUE_TICKS_S))
    table = ((sim or {}).get("fatigue") or {}).get("effects")
    eff = (tuple(tuple(sorted((st, float(v)) for st, v in (table.get(lv) or {}).items())) for lv in FATIGUE_LEVELS)
           if table else FATIGUE_EFFECTS)
    return Rules(float(m["percent_update_per_tick"]), float(m["minimium_increment_update_per_tick"]), TICK_S, th,
                 float(f["threshold_max"]), rates, ticks, float(m["ums_wavering_threshold_lower"]),
                 float(m["ums_wavering_threshold_upper"]), float(m["ums_steady_threshold_lower"]),
                 eff)


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

    def undo(self, tr, mask, value):
        """The trajectory with this bonus's own mark on the recorded fields taken out (the same press not made): only a
        brick whose counter reads a field the bonus itself moves (morale, fatigue) changes it; the flags stay."""
        return tr

    def apply(self, tr, mask, value):
        """The inverse of undo: the bonus's mark put on a trajectory without it (a press at another second)."""
        return tr


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


def morale_weight(m, rules=Rules()):
    """How much a morale point at m points matters (0..1), by the game's morale states: 1 below the wavering edge
    (16: in the wavering band a unit is one shock from the rout at 0, and below 0 it routs or cannot rally), falling
    linearly through the shaken band to 0 at the steady edge (30), 0 above (a point there changes no state)."""
    lo, hi = rules.morale_waver, rules.morale_steady
    return np.clip((hi - np.asarray(m, dtype=float)) / max(hi - lo, 1e-9), 0.0, 1.0)


def morale_area(m, rules=Rules()):
    """The integral of morale_weight from m up to the steady edge (its weight's area above m): the points between a and
    b > a are worth morale_area(a) - morale_area(b)."""
    lo, hi = rules.morale_waver, rules.morale_steady
    m = np.asarray(m, dtype=float)
    band = (hi - lo) / 2.0
    return np.where(m >= hi, 0.0, np.where(m >= lo, (hi - m) ** 2 / (2.0 * max(hi - lo, 1e-9)), band + (lo - m)))


class Morale(Brick):
    """Per affected unit: the bonus's lift weighted by how near the unit is to the wavering / rout edges - the area of
    morale_weight between its morale without the bonus (the recorded morale less the bonus's own lift, morale_lift) and
    with it - at the second this is largest while the bonus is on (the most it held the unit up at once), summed over
    the units. At most the bonus a unit; a unit steady (30+) without the bonus gets 0, however far below its top."""
    desc = Description("morale", "the unit's morale without the bonus is near or below the wavering edge (< 30 points)",
                       False, "buff", "add", "morale points", (("stat_morale", "add"),), 16.0)

    def __init__(self, rules=Rules()):
        self.rules = rules

    def count(self, tr, mask, value):
        if value <= 0:
            return 0.0
        on = _on(tr, mask)
        lift = morale_lift(mask & tr["alive"], value, tr["dt_s"], self.rules)
        M = tr["morale"].astype(np.float64)
        held = morale_area(M - lift, self.rules) - morale_area(M, self.rules)
        best = np.where(on, np.clip(held, 0.0, None), 0.0).max(0)
        return float(np.minimum(value, best).sum())

    def switch_off(self, tr, mask, value):
        # everyone steady without the bonus: its morale less the lift at the steady edge or above
        lift = morale_lift(mask & tr["alive"], value, tr["dt_s"], self.rules)
        return _with(tr, morale=np.maximum(tr["morale"], self.rules.morale_steady + lift))

    def undo(self, tr, mask, value):
        return _with(tr, morale=tr["morale"] - morale_lift(mask & tr["alive"], value, tr["dt_s"], self.rules))

    def apply(self, tr, mask, value):
        return _with(tr, morale=tr["morale"] + morale_lift(mask & tr["alive"], value, tr["dt_s"], self.rules))


def tier(f, thresholds=FATIGUE_THRESHOLDS):
    return np.searchsorted(np.asarray(thresholds), f, side="right") - 1


def tier_penalty(rules=Rules()):
    """{activity: [tiers] the tier's penalty in per cent of the stats that work then} - melee: the sum over MELEE_STATS
    of (1 - multiplier) x 100, move: MOVE_STATS, impact: IMPACT_STATS (the database's fatigue effects)."""
    out = {}
    for name, stats in (("melee", MELEE_STATS), ("move", MOVE_STATS), ("impact", IMPACT_STATS)):
        out[name] = np.array([sum((1.0 - dict(row).get(st, 1.0)) * 100.0 for st in stats)
                              for row in rules.fatigue_effects])
    return out


class _Vigour(Brick):
    """The fatigue tier's penalties the ability spared (or cost) the unit while it fights or moves, in per cent x s:
    each second, the penalty of the tier it would be in without the ability less the penalty of its actual tier, of the
    stats that work then (tier_penalty: in melee the melee stats, moving out of melee the speed, at a charge's impact
    the charge bonus too) - one tier fresher from 'fresh' to 'active' in melee is 5 (attack x0.95), from 'tired' to
    'winded' 10 + 5 ... The fatigue without the ability is rebuilt from the recorded one - each second's recorded
    change less the ability's (value x the maximum a second while the bonus is on), kept within 0..max; a second the
    recorded fatigue sat at 0 or the maximum (its change cut off) takes the activity's own rate instead (the database's
    points a tick: melee - combat, the sprint - charging, moving - running, else ready). From the press to the horizon
    (the gap lasts after the ability ends until the fatigue reaches 0 on both)."""
    sign = 0                # +1: the actual unit is in the WORSE tier (debuff), -1: in the better (buff)

    def __init__(self, rules=Rules()):
        self.rules = rules
        self.penalty = tier_penalty(rules)

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
        actual, without = tier(tr["fatigue"], th), tier(cf, th)
        ok = mask.any(0)[None, :] & tr["alive"]
        ok[0] = False
        c = tr["charge"].astype(np.float64)
        impact = np.zeros_like(ok)
        impact[1:] = c[1:] > c[:-1] + RISE
        gain = np.zeros(ok.shape)
        for name, when in (("melee", tr["m"]), ("move", tr["mv"] & ~tr["m"]), ("impact", impact)):
            pen = self.penalty[name]
            gain += np.where(when, pen[without] - pen[actual], 0.0)
        gain = np.clip(-self.sign * gain, 0.0, None)
        return float(np.where(ok, gain, 0.0).sum() * tr["dt_s"])

    def switch_off(self, tr, mask, value):
        return _with(tr, m=np.zeros_like(tr["m"]), mv=np.zeros_like(tr["mv"]), charge=np.zeros_like(tr["charge"]))

    def undo(self, tr, mask, value):
        return _with(tr, fatigue=self.counterfactual(tr, mask, value))

    def apply(self, tr, mask, value):
        return _with(tr, fatigue=self.counterfactual(tr, mask, -value))


class VigourGain(_Vigour):
    desc = Description("vigour", "the ability left the unit a fatigue tier fresher while it fights or moves", False,
                       "buff", "rate", "% x s", (("vigour_per_s", "recover"),), -0.01)
    sign = -1


class FatigueCost(_Vigour):
    desc = Description("fatigue_cost", "the ability left the unit a fatigue tier more tired while it fights or moves",
                       False, "debuff", "rate", "% x s", (("vigour_per_s", "drain"),), 0.01)
    sign = +1


def _window(tr, mask):
    """[S+1] bool: the seconds the press's mask holds for its owner (a spell's: from its launch to its effect's end)."""
    w = np.asarray(mask)[:, int(tr["owner"])].copy()
    w[0] = False
    return w


class _SpellGain(Brick):
    """value x the rise of a spell bookkeeping field (FIELD: a running total of the presser) over the press's window -
    what the press's own bombardment did (a later cast of the same caster falls outside it: the window ends with its
    effect, the next cast waits its recharge)."""
    FIELD = ""

    def count(self, tr, mask, value):
        f = np.asarray(tr.get(self.FIELD, np.zeros(len(tr["alive"]))), dtype=np.float64)
        rise = np.zeros_like(f)
        rise[1:] = np.clip(f[1:] - f[:-1], 0.0, None)
        return float(max(0.0, value) * (rise * _window(tr, mask)).sum())

    def switch_off(self, tr, mask, value):
        return _with(tr, **{self.FIELD: np.zeros(len(tr["alive"]))})


class SpellDamage(_SpellGain):
    desc = Description("spell_damage", "the spell's bombardment takes enemy health", True, "buff", "add", "gold",
                       (("bombardment", "enemies"),), 1.0)
    FIELD = "sp_gold_e"


class SpellFriendly(_SpellGain):
    desc = Description("spell_friendly", "the spell's bombardment takes the caster's own side's health", True,
                       "debuff", "add", "gold", (("bombardment", "allies"),), 1.0)
    FIELD = "sp_gold_f"


class SpellKnock(_SpellGain):
    desc = Description("spell_knock", "the spell's bombardment knocks enemy men down (they are out of the fight "
                       "until up: magic.knock)", True, "buff", "add", "men knocked", (("bombardment", "knock"),), 1.0)
    FIELD = "sp_knock"


class WindsCost(Brick):
    """The winds a cast spends, when they are short: value (its cost) if the side's current winds after the payment, at
    some second of the press's window, are below the cost - the cast took winds the next cast would have needed then;
    0 when the side could pay again at once, or the cast never launched (no window). A debuff: winds are the resource
    of future casts (the rate: what a wind point is worth in the casts it would pay for)."""
    desc = Description("winds_cost", "the cast leaves its side's current winds below another cast's cost", False,
                       "debuff", "add", "winds", (("mana", "cost"),), 6.0)

    def count(self, tr, mask, value):
        w = _window(tr, mask)
        if value <= 0 or not w.any():
            return 0.0
        winds = np.asarray(tr.get("winds", np.full(len(w), np.inf)), dtype=np.float64)
        return float(value) if float(winds[w].min()) < value else 0.0

    def switch_off(self, tr, mask, value):
        return _with(tr, winds=np.full(len(tr["alive"]), 1e9))


SPELL_KINDS = ("spell_damage", "spell_friendly", "spell_knock", "winds_cost")


def bricks(rules=Rules()):
    """{kind: brick} of every brick."""
    out = [MeleeDamage(), Charge(), MeleeDefence(), Morale(rules), Speed(), ChargeSpeed(), VigourGain(rules),
           FatigueCost(rules), SpellDamage(), SpellFriendly(), SpellKnock(), WindsCost()]
    return {b.desc.kind: b for b in out}
