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
   and what both can reach with max_units units, at most budget_max (config/nn/pools.json, and a faction's own
   budget_max); redrawn until both sides can spend it. A rich battle (pools.json "budget_rare": share, max; share 0
   by default - then no draw is made and every battle is as before): with that probability B is drawn
   log-uniform between the usual top and `max` (at most what both can reach with max_units units);
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

Two schemes (armies=): "v1", the default, is the one above: the evaluation seeds, the gate, the twin and
tools.build --army-seed stay on it, the same seed the same battle as always. "v2" (user 11.10.2026; training with
run.py --armies v2 and AI-vs-AI battles) draws the numbers of units first (Generator.generate_v2):
1. factions as in v1;
2. each side's number of units besides the lord: a band of config/nn/pools.json "armies_v2" counts by its share
   (15-19 40 %, 10-14 30 %, 5-9 20 %, 1-4 10 %), then evenly inside it (at most max_units); the budget ceiling
   evenly in "ceiling" (2000-15000); each side template or random as in v1;
3. the side with fewer units (a tie: either, at random) is built first: its price evenly between its cheapest army
   of that many units and min(ceiling x its share, its dearest one), so a small army is not always elite (when the
   ceiling is below its cheapest army the price is the cheapest: the number of units wins over the ceiling);
4. the other side buys for that army's cost x its share / the first's share, +-TOLERANCE, with its drawn number of
   units, or the nearest number that can (fewer when even its cheapest units are too dear);
   the pools' and the factions' budget_max do not apply (the market of 19 units does).
"v3" (training since 23.10.2026, run.py --armies v3) is v2's algorithm with its own numbers (config/nn/pools.json
"armies_v3"): a side's number of units besides the lord evenly 1-19 (the lord with them: 2-20, the game's full army 20),
a share "full" (5 %) of battles both sides full (19 units and the lord each: a uniform draw alone gives 19 against 19
in 1/361 of battles), the ceiling evenly 2000-20000. v2 stays for the game's exam and its proxy (exam_sim: the same
seeds, the same battles), the AI-vs-AI battles and the abilities' price.

Casters (both schemes, config/nn/pools.json "mages", share 0 by default - then no draw is made): with that probability
a battle gives each side its faction's caster (the Gold Wizard, the Warlock Engineer) beside the lord, on top of its
army, with exactly his spells (Generator.add_mages). A side a script plays loses its caster when the battle starts
(tools/nn/sim/magic.py leave_scripted: no script casts).
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
ARMIES = ("v1", "v2", "v3")    # the army schemes (module doc); v1 the default
COUNT_SCHEMES = ("v2", "v3")   # the schemes that draw the numbers of units first (generate_v2); v3's pools add units_v3


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
        self._of_count = {}

    def totals_of(self, n, capped):
        """What an army of the lord and exactly n units can cost (sorted; capped: under the caps)."""
        key = (n, capped)
        if key not in self._of_count:
            a = self.A[n] & self.capok[n][..., None] if capped else self.A[n]
            self._of_count[key] = self.pool.lord.cost + self.q * np.nonzero(a.reshape(-1, self.S).any(0))[0]
        return self._of_count[key]

    def fits(self, n, lo, hi, capped):
        """An army of the lord and exactly n units can cost between lo and hi."""
        t = self.totals_of(n, capped)
        return bool(np.any((t >= lo - 1e-9) & (t <= hi + 1e-9)))

    def _span(self, lo, hi):
        """[S] the units' costs (steps of q) of an army costing lo..hi with the lord."""
        a = max(0, math.ceil((lo - self.pool.lord.cost) / self.q - 1e-9))
        b = min(math.floor((hi - self.pool.lord.cost) / self.q + 1e-9), self.S - 1)
        m = np.zeros(self.S, bool)
        if b >= a:
            m[a:b + 1] = True
        return m

    def army_of(self, rng, n, lo, hi, template=None, alpha=STYLE_ALPHA):
        """Units (indices): exactly n of them, the army costing lo..hi with the lord; a template under the caps
        (as template_army), or (template None) random preferences without caps (as random_army)."""
        final = np.zeros_like(self.A)
        if template is not None:
            final[n] = self.capok[n][..., None] & self._span(lo, hi)
            weights = template.weights if hasattr(template, "weights") else np.asarray(template, float)
        else:
            final[n] = self._span(lo, hi)
            weights = rng.dirichlet([alpha] * len(self.c))
        bought = self._walk(rng, self._reach(final), weights, stop_at=n)
        assert len(bought) == n
        return bought

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
            w = weights(ok) if callable(weights) else np.where(ok, weights, 0.0)
            w = w / w.sum() if w.sum() > 0 else ok / ok.sum()
            u = int(rng.choice(len(w), p=w))
            bought.append(u)
            pos = tuple(p + d for p, d in zip(pos, self.g[u] + (self.c[u],)))
            k += 1
        return bought

    def template_army(self, rng, budget, template):
        """Units (indices) following a template until the budget is spent, under the caps. template: a
        pools.Template (each pick: its weights of the units that can still be bought, Template.weights) or plain
        shares per unit."""
        final = self.capok[..., None] & self._in_window(budget)
        weights = template.weights if hasattr(template, "weights") else np.asarray(template, float)
        return self._walk(rng, self._reach(final), weights)

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
        # the v2 scheme's pools: the factions' units_v2 join (config/nn/pools.json; given pools: those)
        self.pools_v2 = pools or P.load(scheme="v2")
        # the v3 scheme's pools: units_v2 and units_v3 join (25.10.2026, line g43: the Huntsmen, Ironsides, Long Rifles,
        # Gutter Runners, Jezzails); v2 stays bit for bit
        self.pools_v3 = pools or P.load(scheme="v3")
        self.mix = mix or P.mix()
        # the config's budget cap goes with the config's pools (pools given: no cap)
        self.budget_max = P.budget_max() if pools is None else None
        self.rare = P.budget_rare() if pools is None else (0.0, None)
        self.v2 = P.armies_v2()
        self.v3 = P.armies_v2(scheme="v3")
        # the casters (pools.json "mages"; given pools: none): that share of battles gives each side its faction's
        # caster (add_mages)
        self.mage_share, self.mages = P.mages() if pools is None else (0.0, {})
        self.max_units = max_units
        self.base = base or arena_scenario.load_arena()
        self._markets = {}

    def market(self, faction, max_units=None, scheme="v1"):
        key = (faction, max_units or self.max_units, scheme)
        if key not in self._markets:
            pools = {"v2": self.pools_v2, "v3": self.pools_v3}.get(scheme, self.pools)
            self._markets[key] = Market(pools[faction], key[1])
        return self._markets[key]

    def shares(self, factions):
        """{faction: share of B it may spend}: its budget_factor / the largest factor of `factions`."""
        top = max(self.pools[f].budget_factor for f in factions)
        return {f: self.pools[f].budget_factor / top for f in factions}

    def budget_bounds(self, factions, max_units=None, capped=True):
        """(least, most) budget B every faction of `factions` can field (its share of B); capped=False: without
        budget_max and the factions' budget_max (a rich battle's top, budget_rare)."""
        share = self.shares(factions)
        lo = max((self.pools[f].lord.cost + min(u.cost for u in self.pools[f].units)) / share[f] for f in factions)
        hi = min(float(self.market(f, max_units).totals.max()) / share[f] for f in factions)
        if not capped:
            return lo, hi
        if self.budget_max is not None:
            hi = min(hi, self.budget_max)
        # a faction's own cap on what its side spends (B x its share), when a dear unit joins its pool
        hi = min([hi] + [self.pools[f].budget_max / share[f] for f in factions if self.pools[f].budget_max])
        return lo, hi

    def generate(self, rng, factions=None, budget_range=None, max_units=None, sides=None, name="random", armies="v1"):
        """A random battle (an arena dict, see the module's doc); armies: the scheme (ARMIES)."""
        if armies in COUNT_SCHEMES:
            assert budget_range is None, f"armies {armies} draws no budget range"
            return self.generate_v2(rng, factions, max_units, sides, name, scheme=armies)
        assert armies == "v1", f"armies {armies!r}: one of {ARMIES}"
        factions = list(factions or self.pools)
        own, enemy = sides or (factions[rng.integers(len(factions))], factions[rng.integers(len(factions))])
        lo, hi = self.budget_bounds((own, enemy), max_units)
        share_rare, most = self.rare
        if share_rare > 0 and rng.random() < share_rare:        # a rich battle: above the usual top
            top = min(most, self.budget_bounds((own, enemy), max_units, capped=False)[1])
            if top > hi:
                lo, hi = hi, top
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
                weights = np.asarray([t.weight for t in pool.templates])
                t = int(rng.choice(len(weights), p=weights / weights.sum()))
                bought, army = m.template_army(rng, side_budget, pool.templates[t]), pool.templates[t].name
            else:
                bought, army = m.random_army(rng, side_budget), "random"
            units = [pool.units[i] for i in bought]
            arena["sides"][side] = {"faction": faction, "army": army, "budget": round(side_budget),
                                    "cost": pool.lord.cost + sum(u.cost for u in units),
                                    "units": placement.place(pool.lord, units, arena["deployment_m"])}
        return self.add_mages(rng, arena)

    def add_mages(self, rng, arena):
        """The casters (pools.json "mages"): with probability mage_share each side gets its faction's caster beside
        the lord (placement.hero), with exactly his spells in the battle file (special_abilities: tools/nn/scenario.py
        capabilities) and his cost on top of the side's (both sides at once: the gold stays as even as the two
        casters' costs). Share 0 (the default): no draw, the arena as before. In place; returns the arena."""
        if self.mage_share <= 0 or rng.random() >= self.mage_share:
            return arena
        for side in arena["sides"].values():
            mage = self.mages.get(side["faction"])
            if mage is None:
                continue
            side["units"].append(placement.hero(mage.unit, side["units"][0], spells=mage.spells))
            side["cost"] += mage.unit.cost
            side["mage"] = mage.unit.key
        return arena

    def count(self, rng, max_units=None, scheme="v2"):
        """A side's number of units besides the lord in the v2 (v3) scheme: a band by its share, then evenly inside
        it."""
        bands = getattr(self, scheme)["counts"]
        p = np.asarray([s for _, _, s in bands])
        first, last, _ = bands[int(rng.choice(len(bands), p=p / p.sum()))]
        return min(int(rng.integers(first, last + 1)), max_units or self.max_units)

    def _template(self, rng, pool):
        """A side's template (as v1 draws it), None: a random army."""
        if pool.templates and rng.random() < self.mix.get("template", 0.0):
            weights = np.asarray([t.weight for t in pool.templates])
            return pool.templates[int(rng.choice(len(weights), p=weights / weights.sum()))]
        return None

    def generate_v2(self, rng, factions=None, max_units=None, sides=None, name="random", scheme="v2"):
        """A random battle of the v2 (v3) scheme (the module's doc): the numbers of units first, the smaller army's
        price between its cheapest and the ceiling, the other army for the same gold. A "full" battle (its share of
        the scheme's numbers, drawn once before the tries; share 0, v2: no draw): both sides top units."""
        assert scheme in COUNT_SCHEMES, f"armies {scheme!r}: one of {COUNT_SCHEMES}"
        cfg = getattr(self, scheme)
        factions = list(factions or self.pools)
        own, enemy = sides or (factions[rng.integers(len(factions))], factions[rng.integers(len(factions))])
        top = max_units or self.max_units
        fac = {"own": own, "enemy": enemy}
        pools = self.pools_v3 if scheme == "v3" else self.pools_v2      # the scheme's pools (units_v2, units_v3)
        markets = {s: self.market(f, max_units, scheme) for s, f in fac.items()}
        full = cfg.get("full", 0.0) > 0 and rng.random() < cfg["full"]
        share = self.shares((own, enemy))
        share = {s: share[f] for s, f in fac.items()}
        for _ in range(BUDGET_TRIES):
            n = {s: top if full else self.count(rng, top, scheme) for s in fac}
            ceiling = float(rng.uniform(*cfg["ceiling"]))
            tpl = {s: self._template(rng, pools[f]) for s, f in fac.items()}
            if n["own"] != n["enemy"]:
                a = "own" if n["own"] < n["enemy"] else "enemy"
            else:
                a = ("own", "enemy")[int(rng.integers(2))]
            b = "enemy" if a == "own" else "own"
            # the first (smaller) army: a price between its cheapest and min(ceiling, its dearest); from what the
            # second's cheapest army of its number of units costs when the first can reach that (the numbers kept)
            ma, mb = markets[a], markets[b]
            capped = tpl[b] is not None
            tot = ma.totals_of(n[a], tpl[a] is not None)
            need = mb.totals_of(n[b], capped)[0] * share[a] / share[b] / ((1 + TOLERANCE) * (1 - TOLERANCE))
            low = max(tot[0], need) if need <= tot[-1] else tot[0]
            price = float(rng.uniform(low, max(low, min(ceiling * share[a], tot[-1]))))
            fit = float(tot[tot <= price + 1e-9][-1])
            bought_a = ma.army_of(rng, n[a], min(fit, price * (1 - TOLERANCE)), price, tpl[a])
            cost_a = pools[fac[a]].lord.cost + sum(pools[fac[a]].units[i].cost for i in bought_a)
            # the second army: the same gold (its share), its number of units or the nearest that can spend it
            target = cost_a * share[b] / share[a]
            lo, hi = target * (1 - TOLERANCE), target * (1 + TOLERANCE)
            can = [m for m in range(1, top + 1) if mb.fits(m, lo, hi, capped)]
            if not can:
                continue
            m = n[b]
            if m not in can:
                tot_b = mb.totals_of(m, capped)
                lower, higher = [c for c in can if c < m], [c for c in can if c > m]
                if len(tot_b) == 0 or tot_b[0] > hi:          # too dear: fewer units
                    m = max(lower) if lower else min(higher)
                else:                                          # too cheap: more units
                    m = min(higher) if higher else max(lower)
            bought = {a: bought_a, b: mb.army_of(rng, m, lo, hi, tpl[b])}
            budget = {a: price, b: target}
            break
        else:
            raise ValueError(f"armies {scheme}: no battle of {own} and {enemy} in {BUDGET_TRIES} draws")
        arena = {k: v for k, v in self.base.items() if k not in ("faction", "units", "description")}
        arena.update(name=name, budget=round(price / share[a]), sides={}, armies=scheme, ceiling=round(ceiling),
                     note=f"random battle (armies {scheme}): {own} against {enemy}, budget {round(price / share[a])}")
        for side, faction in fac.items():
            pool = pools[faction]
            units = [pool.units[i] for i in bought[side]]
            arena["sides"][side] = {"faction": faction, "army": tpl[side].name if tpl[side] else "random",
                                    "budget": round(budget[side]),
                                    "cost": pool.lord.cost + sum(u.cost for u in units),
                                    "units": placement.place(pool.lord, units, arena["deployment_m"])}
        return self.add_mages(rng, arena)


_DEFAULT = None


def default():
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Generator()
    return _DEFAULT


def generate(rng, factions=None, budget_range=None, max_units=None, sides=None, name="random", armies="v1"):
    """A random battle from the default pools (config/nn/pools.json); armies: the scheme (ARMIES)."""
    return default().generate(rng, factions, budget_range, max_units, sides, name, armies)


def battle(seed, **kw):
    """The battle of a seed: the same seed, the same battle. Train on TRAIN_SEEDS, evaluate on EVAL_SEEDS.
    armies="v2" / "v3": that scheme's battle of the seed (another battle than v1's)."""
    return generate(np.random.default_rng(seed), name=f"random_{seed}", **kw)


def split(seed):
    """'train' or 'eval' for a seed."""
    if seed in TRAIN_SEEDS:
        return "train"
    if seed in EVAL_SEEDS:
        return "eval"
    raise ValueError(f"seed {seed} is in neither range")
