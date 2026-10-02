"""Random battles with an equal budget (times the factions' budget factors): two armies of a lord
and 0-19 units each, deployed.

    arena = battle(seed)                       # deterministic; seeds of TRAIN_SEEDS or EVAL_SEEDS
    arena = generate(np.random.default_rng(1), factions=["wh_main_emp_empire"], budget_range=(1000, 3000))

An arena is the dict tools/nn/scenario.py takes (map, zones, gap_m, "sides": {"own", "enemy"}),
plus "budget" (B) and, per side, "budget" (what it may spend), "cost" and "army" (the template it
follows, or "random").

A battle:
1. factions - each side one of `factions`, independently (mirrors included);
2. budget B - log-uniform between what both can field (the dearer lord and the cheapest unit)
   and what both can reach with max_units units; redrawn until both sides can spend it;
   a side's budget is B x its faction's budget_factor / the larger factor of the two sides
   (config/nn/pools.json; Skaven 0.8: against the Empire they get 0.8 B, in a mirror both B);
3. each side spends between (1 - TOLERANCE) and 1 of its budget (lord included), so with equal
   factors the two sides differ by at most 5%:
   * a template army (share mix["template"]) follows one of its faction's army templates
     (pools.Pool.templates, the game's army generator), buying unit by unit with the template's
     shares until the budget is spent; categories under caps (config/nn/pools.json);
   * a random army (share mix["random"]) picks the number of units uniformly among those that
     can spend the budget, then its units with random preferences (Dirichlet): cheap swarms,
     few elite units, missile stacks. No caps.
   Every pick is taken only if the army can still end inside the budget window (numpy tables
   of what the pool can buy), so a side never fails and never overspends.
4. deployment - tools/nn/armies/place.py.
"""
import math

import numpy as np

from tools.nn import scenario as arena_scenario
from tools.nn.armies import place as placement
from tools.nn.armies import pools as P

MAX_UNITS = 19                 # besides the lord
TOLERANCE = 0.05               # a side spends (1 - TOLERANCE) B .. B
STYLE_ALPHA = 0.7              # Dirichlet concentration of a random army's preferences
TRAIN_SEEDS = range(0, 1_000_000_000)
EVAL_SEEDS = range(1_000_000_000, 1_001_000_000)
BUDGET_TRIES = 1000


def _shift(a, offs, back=False):
    """a moved by offs along its axes (forward: index + off; back: index - off), zeros shifted in."""
    out = np.zeros_like(a)
    near = tuple(slice(0, n - o) for n, o in zip(a.shape, offs))
    far = tuple(slice(o, None) for o in offs)
    if back:
        out[near] = a[far]
    else:
        out[far] = a[near]
    return out


class Market:
    """What a pool can buy with up to max_units units.

    Tables over (units bought k, count per capped category g..., cost s in steps of q):
    A - reachable from nothing; capok - the category counts allowed for an army of k units."""

    def __init__(self, pool, max_units=MAX_UNITS):
        self.pool, self.max_units = pool, max_units
        self.q = math.gcd(*(u.cost for u in pool.units))
        self.c = [u.cost // self.q for u in pool.units]
        self.groups = tuple(sorted(pool.caps))
        self.g = [tuple(int(u.category == grp) for grp in self.groups) for u in pool.units]
        self.S = max_units * max(self.c) + 1
        dims = (max_units + 1,) * len(self.groups)
        A = np.zeros((max_units + 1,) + dims + (self.S,), bool)
        A[(0,) * (1 + len(dims)) + (0,)] = True
        for k in range(max_units):
            for g, c in zip(self.g, self.c):
                A[k + 1] |= _shift(A[k], g + (c,))
        self.A = A
        capok = np.ones((max_units + 1,) + dims, bool)
        for n in range(max_units + 1):
            for i, grp in enumerate(self.groups):
                idx = np.arange(max_units + 1).reshape([-1 if j == i else 1 for j in range(len(dims))])
                capok[n] &= idx <= pool.cap(grp, n)
        self.capok = capok
        valid = (A & capok[..., None]).reshape(-1, self.S).any(0)
        self.totals = pool.lord.cost + self.q * np.nonzero(valid)[0]   # what a capped army can cost

    def window(self, budget):
        """(lo, hi) of the units' cost in steps of q for a budget (lord excluded)."""
        lo = max(0, math.ceil((budget * (1 - TOLERANCE) - self.pool.lord.cost) / self.q - 1e-9))
        hi = math.floor((budget - self.pool.lord.cost) / self.q + 1e-9)
        return lo, min(hi, self.S - 1)

    def can_spend(self, budget):
        """A capped army can cost between (1 - TOLERANCE) budget and budget."""
        t = self.totals
        return bool(np.any((t >= budget * (1 - TOLERANCE) - 1e-9) & (t <= budget + 1e-9)))

    def _in_window(self, budget):
        lo, hi = self.window(budget)
        m = np.zeros(self.S, bool)
        if hi >= lo:
            m[lo:hi + 1] = True
        return m

    def _reach(self, final):
        """R[k]: from k units, counts g, cost s the army can end in `final` [k, g..., s]."""
        R = np.zeros_like(final)
        R[self.max_units] = final[self.max_units]
        for k in range(self.max_units - 1, -1, -1):
            R[k] = final[k]
            for g, c in zip(self.g, self.c):
                R[k] |= _shift(R[k + 1], g + (c,), back=True)
        return R

    def _walk(self, rng, R, weights, stop_at=None):
        """Buy unit by unit, keeping R reachable: [unit index]."""
        k, pos, bought = 0, (0,) * len(self.groups) + (0,), []
        assert R[(0,) + pos], "the budget cannot be spent"
        while k < self.max_units and k != stop_at:
            ok = []
            for u, (g, c) in enumerate(zip(self.g, self.c)):
                nxt = tuple(p + d for p, d in zip(pos, g + (c,)))
                ok.append(nxt[-1] < self.S and max(nxt[:-1], default=0) <= self.max_units and bool(R[(k + 1,) + nxt]))
            ok = np.array(ok)
            if not ok.any():
                break
            w = np.where(ok, weights, 0.0)
            w = w / w.sum() if w.sum() > 0 else ok / ok.sum()
            u = int(rng.choice(len(w), p=w))
            bought.append(u)
            pos = tuple(p + d for p, d in zip(pos, self.g[u] + (self.c[u],)))
            k += 1
        return bought

    def template_army(self, rng, budget, shares):
        """Units (indices) following a template's shares until the budget is spent, under the caps."""
        final = self.capok[..., None] & self._in_window(budget)
        return self._walk(rng, self._reach(final), np.asarray(shares, float))

    def counts(self, budget):
        """Numbers of units an uncapped army can spend the budget with."""
        inw = self._in_window(budget)
        return [n for n in range(self.max_units + 1) if (self.A[n] & inw).any()]

    def random_army(self, rng, budget, alpha=STYLE_ALPHA):
        """Units (indices): a uniform number of units among those that fit, random preferences."""
        n = int(rng.choice(self.counts(budget)))
        final = np.zeros_like(self.A)
        final[n] = self._in_window(budget)
        R = self._reach(final)
        style = rng.dirichlet([alpha] * len(self.c))
        bought = self._walk(rng, R, style, stop_at=n)
        assert len(bought) == n
        return bought


class Generator:
    """Pools and their markets, loaded once."""

    def __init__(self, pools=None, mix=None, max_units=MAX_UNITS, base=None):
        self.pools = pools or P.load()
        self.mix = mix or P.mix()
        self.max_units = max_units
        self.base = base or arena_scenario.load_arena()
        self._markets = {}

    def market(self, faction, max_units=None):
        key = (faction, max_units or self.max_units)
        if key not in self._markets:
            self._markets[key] = Market(self.pools[faction], key[1])
        return self._markets[key]

    def shares(self, factions):
        """{faction: share of B it may spend}: its budget_factor / the largest factor of `factions`."""
        top = max(self.pools[f].budget_factor for f in factions)
        return {f: self.pools[f].budget_factor / top for f in factions}

    def budget_bounds(self, factions, max_units=None):
        """(least, most) budget B every faction of `factions` can field (its share of B)."""
        share = self.shares(factions)
        lo = max((self.pools[f].lord.cost + min(u.cost for u in self.pools[f].units)) / share[f] for f in factions)
        hi = min(float(self.market(f, max_units).totals.max()) / share[f] for f in factions)
        return lo, hi

    def generate(self, rng, factions=None, budget_range=None, max_units=None, sides=None, name="random"):
        """A random battle (an arena dict, see the module's doc)."""
        factions = list(factions or self.pools)
        own, enemy = sides or (factions[rng.integers(len(factions))], factions[rng.integers(len(factions))])
        lo, hi = self.budget_bounds((own, enemy), max_units)
        if budget_range:
            lo, hi = max(lo, budget_range[0]), min(hi, budget_range[1])
        assert lo <= hi, f"no budget both {own} and {enemy} can field in {budget_range}"
        markets = {"own": self.market(own, max_units), "enemy": self.market(enemy, max_units)}
        share = self.shares((own, enemy))
        share = {"own": share[own], "enemy": share[enemy]}
        for _ in range(BUDGET_TRIES):
            budget = float(np.exp(rng.uniform(math.log(lo), math.log(hi))))
            if all(m.can_spend(budget * share[s]) for s, m in markets.items()):
                break
        else:
            raise ValueError(f"no budget in {lo}..{hi} both {own} and {enemy} can spend")
        arena = {k: v for k, v in self.base.items() if k not in ("faction", "units", "description")}
        arena.update(name=name, budget=round(budget), sides={},
                     note=f"random battle: {own} against {enemy}, budget {round(budget)}")
        for side, faction in (("own", own), ("enemy", enemy)):
            m, pool, side_budget = markets[side], self.pools[faction], budget * share[side]
            if pool.templates and rng.random() < self.mix.get("template", 0.0):
                names, weights, shares = zip(*pool.templates)
                t = int(rng.choice(len(names), p=np.asarray(weights) / sum(weights)))
                bought, army = m.template_army(rng, side_budget, shares[t]), names[t]
            else:
                bought, army = m.random_army(rng, side_budget), "random"
            units = [pool.units[i] for i in bought]
            arena["sides"][side] = {"faction": faction, "army": army, "budget": round(side_budget),
                                    "cost": pool.lord.cost + sum(u.cost for u in units),
                                    "units": placement.place(pool.lord, units, arena["deployment_m"])}
        return arena


_DEFAULT = None


def default():
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Generator()
    return _DEFAULT


def generate(rng, factions=None, budget_range=None, max_units=None, sides=None, name="random"):
    """A random battle from the default pools (config/nn/pools.json)."""
    return default().generate(rng, factions, budget_range, max_units, sides, name)


def battle(seed, **kw):
    """The battle of a seed: the same seed, the same battle. Train on TRAIN_SEEDS, evaluate on EVAL_SEEDS."""
    return generate(np.random.default_rng(seed), name=f"random_{seed}", **kw)


def split(seed):
    """'train' or 'eval' for a seed."""
    if seed in TRAIN_SEEDS:
        return "train"
    if seed in EVAL_SEEDS:
        return "eval"
    raise ValueError(f"seed {seed} is in neither range")
