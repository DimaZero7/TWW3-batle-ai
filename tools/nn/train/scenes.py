"""The battles training plays (docs/en/training/training.md).

Two sources of battles, both as a Bank: ready start states that a battle takes when it (re)starts.
* Fixed: the named arenas (config/nn/arenas.json) in both roles. A scene is (arena, role of side
  1): "attack" - side 1 attacks, "defend" - side 2 attacks. Battle b always plays its scene.
* Generated: random armies of tools/nn/armies (equal budget, 1 lord + 0..max_units units a side),
  half with side 1 attacking; a battle that ends takes a random battle of the bank. Train seeds
  for training, eval seeds (never trained on) for evaluation.
The learner plays side 1 in some battles and side 2 in others (tools/nn/train/league.py), so it plays
every army in both roles.
"""
import numpy as np
import torch

from tools.nn.model import observation, sources
from tools.nn.sim import scenario

SCENES = (
    ("arena", "attack"), ("arena", "defend"),                        # the Empire mirror arena
    ("whole_emp_v_skv", "attack"), ("whole_emp_v_skv", "defend"),    # Empire (side 1) against Skaven
    ("whole_skv_v_emp", "attack"), ("whole_skv_v_emp", "defend"),    # Skaven (side 1) against Empire
)


def army(scene):
    name, role = scene
    return scenario.from_arena(name, role)


def attackers(scenes=SCENES):
    """[len(scenes)] the side that attacks in each scene (1 or 2)."""
    return [1 if role == "attack" else 2 for _, role in scenes]


def factions(armies):
    """[(faction of side 1, faction of side 2)] per battle."""
    return [(a["sides"][1]["faction"], a["sides"][2]["faction"]) for a in armies]


def per_side(scenes=SCENES):
    """Slots per side that fit every scene."""
    return max(len(army(s)["sides"][k]["units"]) for s in scenes for k in (1, 2))


def build(scene_of, scenes=SCENES, params=None, device="cpu", H=None):
    """(State, Setup) of a batch: battle b plays scenes[scene_of[b]] from its start."""
    armies = [army(scenes[int(i)]) for i in scene_of]
    st = scenario.build(armies, params, device=device, per_side=H or per_side(scenes))
    setup, _ = sources.from_sim(st, factions(armies))
    # The network sees the map as the game shows it (the minimap frame the companion uses), not
    # the simulator's square (the edge where routing units leave): its orders stay inside it.
    setup = observation.Setup(keys=setup.keys, side=setup.side, bounds=np.tile(np.array(sources.CROSSROADS,
                              np.float32), (st.B, 1)), factions=setup.factions, attacker=setup.attacker)
    return st, setup


def reset_rows(st, template, rows):
    """Battles where rows [B] (bool tensor) is true start again from the template (in place)."""
    keep = ~rows
    for k, v in st.u.items():
        st.u[k] = torch.where(rows[:, None], template.u[k], v)
    st.t = torch.where(keep, st.t, template.t)
    st.done = torch.where(keep, st.done, template.done)
    st.winner = torch.where(keep, st.winner, template.winner)
    st.attacker = torch.where(keep, st.attacker, template.attacker)
    st.lord_dead_s = torch.where(keep[:, None], st.lord_dead_s, template.lord_dead_s)
    return st


def spread_evenly(B, n, seed=0):
    """[B] indices 0..n-1, each about B / n times, in a shuffled order."""
    idx = np.arange(B) % n
    return np.random.default_rng(seed).permutation(idx)


class LiveSetup:
    """What observation.observe needs of a Setup, as tensors on the device that change per battle:
    a battle that ends takes a new army, and its row is copied in place (the compiled observe keeps
    working on the same tensors)."""

    FIELDS = ("side", "bounds", "rank", "lord_level", "passport", "men0", "ammo0", "present", "attacker", "lord",
              "abil", "abil_owned", "abil_use")      # abil: the ability slots' passports [B, N, SLOTS, STATIC]

    def __init__(self, arrays, char, adapt, factions):
        self.arrays, self.char, self.adapt, self.factions = arrays, char, adapt, factions

    @classmethod
    def of(cls, setup, device):
        like = setup.like(torch.zeros(1, device=device))
        arrays = observation._Arrays(**{k: getattr(like, k).clone() for k in cls.FIELDS})
        char = {s: torch.as_tensor(setup.character(s), device=device).float() for s in (1, 2)}
        adapt = {s: torch.as_tensor(setup.adapter(s), device=device).long() for s in (1, 2)}
        return cls(arrays, char, adapt, list(setup.factions))

    def like(self, x):
        return self.arrays

    def character(self, side):
        return self.char[side]

    def adapter(self, side):
        return self.adapt[side]

    @property
    def bounds(self):
        return self.arrays.bounds

    def take(self, other, rows, idx):
        """Rows [B] (bool) take row idx[b] of another LiveSetup (the bank's), in place."""
        def put(dst, src):
            m = rows.reshape(-1, *([1] * (dst.dim() - 1)))
            dst.copy_(torch.where(m, src[idx], dst))
        for k in self.FIELDS:
            put(getattr(self.arrays, k), getattr(other.arrays, k))
        for s in (1, 2):
            put(self.char[s], other.char[s])
            put(self.adapt[s], other.adapt[s])
        for b in torch.nonzero(rows).flatten().tolist():
            self.factions[b] = other.factions[int(idx[b])]


class Bank:
    """M ready battles: their start states and what each side knows before battle."""

    def __init__(self, armies, params=None, device="cpu", per_side_=None, names=None):
        self.armies = armies
        self.names = names or [f"battle_{i}" for i in range(len(armies))]
        H = per_side_ or max(len(a["sides"][k]["units"]) for a in armies for k in (1, 2))
        self.state = scenario.build(armies, params, device=device, per_side=H)
        setup, _ = sources.from_sim(self.state, factions(armies))
        setup = observation.Setup(keys=setup.keys, side=setup.side, bounds=np.tile(
            np.array(sources.CROSSROADS, np.float32), (self.state.B, 1)), factions=setup.factions,
            attacker=setup.attacker)
        self.setup = LiveSetup.of(setup, device)
        self.attacker = self.state.attacker

    @property
    def M(self):
        return self.state.B

    @property
    def N(self):
        return self.state.N


def take_rows(st, bank_state, rows, idx):
    """Battles where rows [B] is true start again as bank battle idx[b] (in place)."""
    keep = ~rows
    for k, v in st.u.items():
        st.u[k] = torch.where(rows[:, None], bank_state.u[k][idx], v)
    st.t = torch.where(keep, st.t, bank_state.t[idx])
    st.done = torch.where(keep, st.done, bank_state.done[idx])
    st.winner = torch.where(keep, st.winner, bank_state.winner[idx])
    st.attacker = torch.where(keep, st.attacker, bank_state.attacker[idx])
    st.lord_dead_s = torch.where(keep[:, None], st.lord_dead_s, bank_state.lord_dead_s[idx])
    for b in torch.nonzero(rows).flatten().tolist():
        st.keys[b] = list(bank_state.keys[int(idx[b])])
    return st


class Fixed:
    """The named arenas: battle b always restarts as scenes[scene_of[b]]."""

    def __init__(self, scene_of, scene_list=SCENES, params=None, device="cpu"):
        armies = [army(scene_list[int(i)]) for i in scene_of]
        self.scene_list = scene_list
        self.bank = Bank(armies, params, device, per_side(scene_list),
                         names=[f"{scene_list[int(i)][0]}-{scene_list[int(i)][1]}" for i in scene_of])
        self.index = torch.arange(len(armies), device=self.bank.state.device)

    def pick(self, want=None):
        return self.index


class Generated:
    """Random armies of tools/nn/armies: `M` battles from `seeds`, alternately side 1 and side 2
    attacking; a battle that ends takes a random one (`want`: the side that must attack, per battle,
    0 for any). sequential: battle b is bank battle b (evaluation: every battle once)."""

    def __init__(self, seeds, max_units=19, params=None, device="cpu", seed=0, sequential=False, small=None):
        """small: (share, units): that share of the bank with at most `units` units a side (the
        generator's budget rarely gives such armies when up to 19 are allowed)."""
        from tools.nn.armies import generate
        seeds = [int(s) for s in seeds]
        n_small = int(round(small[0] * len(seeds))) if small else 0
        arenas = [generate.battle(s, max_units=min(max_units, small[1]) if i < n_small else max_units)
                  for i, s in enumerate(seeds)]
        roles = ["attack" if i % 2 == 0 else "defend" for i in range(len(seeds))]
        armies = [scenario.from_arena(a, r) for a, r in zip(arenas, roles)]
        self.seeds, self.max_units, self.sequential = seeds, max_units, sequential
        self.bank = Bank(armies, params, device, max_units + 1,
                         names=[f"{a['name']}-{r}" for a, r in zip(arenas, roles)])
        dev = self.bank.state.device
        self.gen = torch.Generator(device=dev).manual_seed(seed)
        att = self.bank.attacker
        self.by_attacker = {s: torch.nonzero(att == s).flatten() for s in (1, 2)}

    def pick(self, want=None):
        """[B] bank rows for every battle (the caller takes only those that restart)."""
        B = len(want) if want is not None else self.bank.M
        dev = self.bank.state.device
        if self.sequential:
            return torch.arange(B, device=dev)
        idx = torch.randint(self.bank.M, (B,), device=dev, generator=self.gen)
        if want is not None:
            for s in (1, 2):
                pool = self.by_attacker[s]
                pick = pool[torch.randint(len(pool), (B,), device=dev, generator=self.gen)]
                idx = torch.where(want == s, pick, idx)
        return idx
