"""Layer 2: an ability as data (docs/en/training/ability_price.md).

An ability is a group of bonus bricks with their numbers (bonuses.py), its duration, recharge, radius and targets - all
read from its passport (config/nn/abilities.json); no code per ability. A new ability of known bonus kinds is data only;
a passport stat no brick declares is listed in Ability.unpriced (a new kind needs a new brick).

Who gets the bonus (mask), as the simulator lays it (tools/nn/sim/abilities.py):
    self      the owner while the ability is active (its timer `on` > 0) and he is alive;
    friends   update_targets true - an aura that follows him: his side's units within range_m of him each second while
              it is active (Rally); false - laid once at the cast on the units within range_m then, kept active_s
              seconds wherever they go (Stand Your Ground).

The press's price (gold):
    benefit  = sum over buff parts of k[kind] x counter
    debuffs  = sum over debuff parts of k[kind] x counter
    price    = benefit - debuffs - recharge cost
The recharge cost: static (Recharge) - the mean net benefit of this ability's useful presses x recharge_s / 60 (a 90 s
recharge costs more than a 60 s one); dynamic (DynamicRecharge) - per press, what the ability ready over the next
recharge_s seconds is worth in this situation: the net benefit of the same ability pressed at a later second of the
window instead (Ability.later: the press's own trajectory, its own mark taken out, the press moved) - at the first
second its bonus would have work, less the credit of getting the ability back earlier (next_useful; best_later, the
best second of the window, is kept for comparison: biased up), regressed on the situation at the press (validate.py,
train_price.py: no look at the future).

Only numpy.
"""
from dataclasses import dataclass, field

import numpy as np

from tools.nn.abilities_price import bonuses as bz

GROUPS = ("self", "friends", "enemies")


def stat_kinds(bricks):
    """{(stat, how): kind} of the bricks' declared stats."""
    return {s: k for k, b in bricks.items() for s in b.desc.stats}


@dataclass(frozen=True)
class Part:
    kind: str
    value: float
    groups: tuple            # who gets it: "self", "friends"


@dataclass(frozen=True)
class Ability:
    key: str
    active_s: float
    recharge_s: float
    range_m: float
    update_targets: bool
    parts: tuple
    unpriced: tuple = field(default=())     # the passport's stats no brick prices (or on targets not traced)

    @classmethod
    def from_passport(cls, key, p, bricks):
        """The ability of passport p (an entry of abilities.json `abilities`)."""
        kinds = stat_kinds(bricks)
        by = {}                                  # (kind, groups) -> {(stat, how): value}
        unpriced = []
        for e in p["effects"]:
            s = (e["stat"], e["how"])
            groups = tuple(g for g in GROUPS if g in e["on"])
            if s not in kinds or "enemies" in groups:
                unpriced.append(f"{e['stat']} {e['how']} {e['value']:g} on {'+'.join(groups)}")
                continue
            d = by.setdefault((kinds[s], groups), {})
            d[s] = d.get(s, 1.0) * float(e["value"]) if e["how"] == "mult" else d.get(s, 0.0) + float(e["value"])
        parts = []
        for (kind, groups), vals in sorted(by.items()):
            desc = bricks[kind].desc
            if desc.how == "mult":
                # a kind of several stats (damage: base and armour-piercing): the mean of their (x - 1), a missing one 1
                value = 1.0 + float(np.mean([vals.get(st, 1.0) - 1.0 for st in desc.stats]))
            else:
                value = float(sum(vals.values()))
            parts.append(Part(kind, value, groups))
        v = float(p.get("vigour_per_s") or 0.0)
        if v:
            kind = "vigour" if v < 0 else "fatigue_cost"          # negative: the owner recovers (the database's sign)
            if kind in bricks:
                parts.append(Part(kind, v, ("self",)))
            else:
                unpriced.append(f"vigour_per_s {v:g}")
        return cls(key, float(p["active_s"]), float(p["recharge_s"]), float(p["range_m"]),
                   bool(p.get("update_targets", True)), tuple(parts), tuple(unpriced))

    def masks(self, tr):
        """{group: [S+1, U] bool} who gets the bonus each second (module doc)."""
        S1, U = tr["alive"].shape
        o = int(tr["owner"])
        alive = tr["alive"]
        active = (np.asarray(tr["on"]) > 0) & alive[:, o]
        active[0] = False
        me = np.zeros((S1, U), dtype=bool)
        me[:, o] = active
        others = np.ones(U, dtype=bool)
        others[o] = False
        d = np.hypot(tr["x"] - tr["x"][:, o:o + 1], tr["z"] - tr["z"][:, o:o + 1])
        if self.update_targets:
            fr = active[:, None] & (d <= self.range_m) & others[None, :] & alive
        else:
            fr = np.zeros((S1, U), dtype=bool)
            if active.any() and self.range_m > 0:
                laid = (d[0] <= self.range_m) & others & alive[0]
                secs = np.arange(S1) * tr["dt_s"]
                fr = (secs[:, None] <= self.active_s + 1e-6) & (secs[:, None] > 0) & laid[None, :] & alive
        return {"self": me, "friends": fr}

    def counters(self, tr, bricks):
        """{kind: counter} of the press's trajectory (a kind twice: summed)."""
        m = self.masks(tr)
        out = {}
        for part in self.parts:
            mask = np.zeros_like(tr["alive"])
            for g in part.groups:
                mask = mask | m[g]
            out[part.kind] = out.get(part.kind, 0.0) + bricks[part.kind].count(tr, mask, part.value)
        return out

    def kinds(self):
        return tuple(dict.fromkeys(p.kind for p in self.parts))

    def _part_masks(self, tr):
        m = self.masks(tr)
        out = []
        for part in self.parts:
            mask = np.zeros_like(tr["alive"])
            for g in part.groups:
                mask = mask | m[g]
            out.append(mask)
        return out

    def later(self, tr, bricks, starts, horizon):
        """[{kind: counter}] of the same ability pressed at each second in `starts` (samples) instead: the press's own
        mark on the recorded morale / fatigue taken out (Brick.undo), the trajectory cut at the start, the ability's timer
        laid anew (active_s seconds from there, the owner alive), its mark put back at the new second (Brick.apply), and
        the counters over `horizon` samples from there (fewer where the trajectory ends). A start the owner is not
        standing at (dead, routing) gets nothing."""
        base = tr
        for part, mask in zip(self.parts, self._part_masks(tr)):
            base = bricks[part.kind].undo(base, mask, part.value)
        S1 = len(tr["alive"])
        o = int(tr["owner"])
        out = []
        for s0 in starts:
            s0 = int(s0)
            if s0 >= S1 - 1 or not tr["alive"][s0, o] or tr["r"][s0, o]:
                out.append({})
                continue
            end = min(S1, s0 + int(horizon) + 1)
            cut = {k: (v[s0:end] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == S1 and k != "mmax" else v)
                   for k, v in base.items()}
            secs = np.arange(end - s0) * tr["dt_s"]
            on = np.clip(self.active_s + tr["dt_s"] - secs, 0.0, None)
            on[0] = 0.0
            cut["on"] = on * cut["alive"][:, o]
            c = {}
            for part, mask in zip(self.parts, self._part_masks(cut)):
                b = bricks[part.kind]
                c[part.kind] = c.get(part.kind, 0.0) + b.count(b.apply(cut, mask, part.value), mask, part.value)
            out.append(c)
        return out


def abilities(passports, bricks, pressable=True):
    """{key: Ability} of the passports (pressable: only the ones a player or the network fires - active, self-cast,
    not fired by the game itself)."""
    out = {}
    for k, p in passports.items():
        if pressable and (p["passive"] or p.get("auto") or not p.get("self_cast")):
            continue
        out[k] = Ability.from_passport(k, p, bricks)
    return out


def split(counters, rates, bricks):
    """(benefit, debuffs) in gold of one press's counters."""
    ben = deb = 0.0
    for kind, c in counters.items():
        g = float(rates.get(kind, 0.0)) * c
        if bricks[kind].desc.effect == "debuff":
            deb += g
        else:
            ben += g
    return ben, deb


class Recharge:
    """The recharge cost of a press (gold): static per ability (the start) - the mean net benefit (benefit - debuffs)
    of its useful presses (> 0) x recharge_s / 60. Subclass and override cost() for the dynamic version (the ability
    critic's value of having the ability ready over the next recharge_s seconds in the press's situation)."""

    def __init__(self, per_ability=None):
        self.per_ability = dict(per_ability or {})

    @classmethod
    def fit(cls, net_by_ability, abilities_):
        """net_by_ability {key: array of presses' net benefit}."""
        out = {}
        for k, net in net_by_ability.items():
            net = np.asarray(net, dtype=float)
            useful = net[net > 0]
            out[k] = float(useful.mean() if len(useful) else 0.0) * abilities_[k].recharge_s / 60.0
        return cls(out)

    def cost(self, key, press=None):
        return self.per_ability.get(key, 0.0)


class DynamicRecharge(Recharge):
    """Per-press recharge costs (gold): costs[p] of press p (module doc)."""

    def __init__(self, costs):
        super().__init__()
        self.costs = np.asarray(costs, dtype=float)

    def cost(self, key, press=None):
        return float(self.costs[press])


def best_later(ability, tr, bricks, rates, horizon, step=5, credit_s=0.0):
    """The most a press of the same ability at a later second of its recharge window (step, 2 step, .. recharge_s)
    would have netted (benefit - debuffs - credit_s x the second, gold; >= 0: not pressing at all is always there) - the
    dynamic recharge cost seen in hindsight on this trajectory, the max kind: biased up (the maximum of ~18 noisy
    seconds; a policy does not know the best second in advance) - kept for comparison, the cost is next_useful."""
    starts = np.arange(step, ability.recharge_s + 1e-6, step) / tr["dt_s"]
    best = 0.0
    for s, c in zip(starts, ability.later(tr, bricks, starts, horizon)):
        ben, deb = split(c, rates, bricks)
        best = max(best, ben - deb - (credit_s * float(s) * tr["dt_s"] if c else 0.0))
    return best


def next_useful(ability, tr, bricks, rates, horizon, step=5, credit_s=0.0):
    """The net benefit (gold, >= 0) of the same ability pressed at the FIRST later second of its recharge window (step,
    2 step, .. recharge_s) where its bonus would do useful work (some counter > 0) - the press a sensible player makes
    if he waits now (press when the bonus has work, as the game's AI does) - less credit_s x that second (pressing now
    brings the ability back that many seconds earlier: credit_s = the ability's mean press benefit / (active_s +
    recharge_s)); 0 when no second of the window has work. The dynamic recharge cost seen in hindsight on this
    trajectory, without best_later's maximum (the best of ~18 noisy seconds: an upper bound no causal policy reaches)."""
    starts = np.arange(step, ability.recharge_s + 1e-6, step) / tr["dt_s"]
    for s, c in zip(starts, ability.later(tr, bricks, starts, horizon)):
        if c and sum(c.values()) > 0:
            ben, deb = split(c, rates, bricks)
            return max(0.0, ben - deb - credit_s * float(s) * tr["dt_s"])
    return 0.0


TARGETS = {"next": next_useful, "best": best_later}     # the hindsight recharge cost's kinds (config: recharge_target)


def price(ability, counters, rates, bricks, recharge, press=None):
    """{"benefit", "debuffs", "recharge", "net", "price"} of one press, gold."""
    ben, deb = split(counters, rates, bricks)
    r = recharge.cost(ability.key, press)
    return {"benefit": ben, "debuffs": deb, "recharge": r, "net": ben - deb, "price": ben - deb - r}
