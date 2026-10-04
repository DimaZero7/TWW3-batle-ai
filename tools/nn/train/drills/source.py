"""Where drill battles come from in training and evaluation (docs/en/training/training.md "Drills").

Mixed: one bank (scenes.Bank) of the generated battles (as scenes.Generated) and, after them, every
drill's generated battles (half with our side 1, half side 2), all under the standard battle limit. A
battle row whose opponent is a drill
(league "drill_<name>") restarts as a random battle of that drill with OUR side = the side the
learner plays in that row; every other row as scenes.Generated would.
"""
import numpy as np
import torch

from tools.nn.sim import scenario
from tools.nn.train import drills as D
from tools.nn.train import league, scenes


def generated_armies(seeds, max_units, small=None):
    """[(army description, name)] of the generated battles of seeds (as scenes.Generated)."""
    from tools.nn.armies import generate
    seeds = [int(s) for s in seeds]
    n_small = int(round(small[0] * len(seeds))) if small else 0
    arenas = [generate.battle(s, max_units=min(max_units, small[1]) if i < n_small else max_units)
              for i, s in enumerate(seeds)]
    roles = ["attack" if i % 2 == 0 else "defend" for i in range(len(seeds))]
    return [(scenario.from_arena(a, r), f"{a['name']}-{r}") for a, r in zip(arenas, roles)]


def drill_codes(layout):
    """{drill name: league code} of the drills the layout's rows play."""
    have = set(np.unique(layout.opponent).tolist())
    return {n: league.CODE[D.opponent(n)] for n in D.NAMES if league.CODE[D.opponent(n)] in have}


class Mixed:
    """Generated battles and drill battles in one bank. layout: the batch's league.Layout (its rows'
    opponents and learner sides; the drill rows take their drill's battles with our side = the
    learner's). per_drill: battles of each drill in the bank; drill_seeds: their seed range."""

    def __init__(self, seeds, max_units, params, device, layout, seed=0, small=None, per_drill=256,
                 drill_seeds=D.DRILL_SEEDS, sequential=False, drills=None):
        rng = np.random.default_rng(seed)
        self.max_units, self.sequential = max_units, sequential
        gen = generated_armies(seeds, max_units, small) if len(seeds) else []
        codes = drill_codes(layout)
        drills = drills or D.load(list(codes))
        missing = [n for n in codes if n not in drills]
        assert not missing, f"no drill modules for {missing}"
        descs, names, ours, group, wide, kinds = [], [], [], [], [], []
        for n, code in codes.items():
            pick = rng.integers(drill_seeds.start, drill_seeds.stop, per_drill) if not sequential \
                else np.arange(drill_seeds.start, drill_seeds.start + per_drill)
            for i, (desc, side) in enumerate(D.battles(drills[n], pick)):
                descs.append(fill_factions(desc))
                names.append(f"{n}_{int(pick[i])}-side{side}")
                ours.append(side)
                group.append(code)
                wide.append(bool(desc.get("broad")))
                kinds.append(desc.get("frame", "broad" if desc.get("broad") else "clean"))
        armies = [a for a, _ in gen] + descs
        H = max([max_units + 1 if gen else 1] + [len(d["sides"][s]["units"]) for d in descs for s in (1, 2)])
        self.bank = scenes.Bank(armies, params, device, H, names=[n for _, n in gen] + names)
        dev = self.bank.state.device
        self.M_gen = len(gen)
        self.group = torch.tensor([0] * len(gen) + group, device=dev)          # [M] 0 generated, else the drill's code
        self.ours = torch.tensor([0] * len(gen) + ours, device=dev)            # [M] our side in a drill battle
        self.broad = np.array([False] * len(gen) + wide, dtype=bool)           # [M] a drill battle of the broad frame
        self.frame = np.array([""] * len(gen) + kinds, dtype=object)          # [M] its frame (drills.FRAMES), "" generated
        self.gen = torch.Generator(device=dev).manual_seed(int(rng.integers(1 << 30)))
        att = self.bank.attacker[:len(gen)]
        self.by_attacker = {s: torch.nonzero(att == s).flatten() for s in (1, 2)}
        self.pools = {(c, s): torch.nonzero((self.group == c) & (self.ours == s)).flatten()
                      for c in codes.values() for s in (1, 2)}
        self.row_code = torch.as_tensor(layout.opponent, device=dev)
        self.row_side = torch.as_tensor(layout.learner, device=dev)

    def pick(self, want=None):
        """[B] bank rows for every battle (the caller takes only those that restart)."""
        B = len(self.row_code)
        dev = self.bank.state.device
        if self.sequential:
            return torch.arange(B, device=dev)
        idx = torch.zeros(B, dtype=torch.long, device=dev)
        if self.M_gen:
            idx = torch.randint(self.M_gen, (B,), device=dev, generator=self.gen)
            if want is not None:
                for s in (1, 2):
                    pool = self.by_attacker[s]
                    if len(pool):
                        p = pool[torch.randint(len(pool), (B,), device=dev, generator=self.gen)]
                        idx = torch.where(want == s, p, idx)
        for (c, s), pool in self.pools.items():
            if len(pool):
                p = pool[torch.randint(len(pool), (B,), device=dev, generator=self.gen)]
                idx = torch.where((self.row_code == c) & (self.row_side == s), p, idx)
        return idx


_FACTION = {}


def faction_of(key):
    """The faction of a unit key (config/nn/pools.json; None if not in a pool)."""
    if not _FACTION:
        from tools.nn.armies import pools as P
        for f, pool in P.load().items():
            _FACTION[pool.lord.key] = f
            for u in pool.units:
                _FACTION[u.key] = f
    return _FACTION.get(key)


def fill_factions(desc):
    """A description whose sides have no faction get the faction of their first unit (in place)."""
    for s in (1, 2):
        side = desc["sides"][s]
        if not side.get("faction") and side["units"]:
            side["faction"] = faction_of(side["units"][0]["key"])
    return desc


def evaluation(names, n, params, device, drills=None):
    """(Mixed source, league.Layout) of n battles of every drill in names on DRILL_EVAL_SEEDS, battle b =
    bank row b (our side alternating), for evaluate.play_drills."""
    names = [x for x in D.NAMES if x in names]                # the bank's order
    codes = [league.CODE[D.opponent(x)] for x in names]
    lay = league.Layout(np.zeros(n * len(names), dtype=int), np.tile(1 + np.arange(n) % 2, len(names)),
                        np.repeat(codes, n))
    src = Mixed([], 0, params, device, lay, per_drill=n, drill_seeds=D.DRILL_EVAL_SEEDS, sequential=True,
                drills=drills)
    # the bank's drill battles alternate our side 1, 2 in seed order (D.battles), as the layout
    assert torch.equal(src.ours.cpu(), torch.as_tensor(lay.learner)), "drill bank and layout disagree on our side"
    return src, lay
