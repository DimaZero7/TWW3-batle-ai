"""The attacking enemy of drill O0-1's set G_def (the orchestrator 22.11: our mission HOLD, the enemy attacks from the
start with the stage's scripts; kept beside sally.py, which is not touched - its hash is in the key of exam2's arms).

    KINDS             the stage's attacking scripts: 'human' (sim/scripts.human with the enemy as the attacker - the
                      player by the manual's defence-in-depth rules), 'nearest' (sim/scripts.nearest)
    general_def(n, seed)   n stage-0 general battles (stage0/specs.drill_generals, the stage's armies, maps and sizes)
                      with our mission hold (o1.att 0), the enemy 'attack' with its script drawn per battle by the
                      stage's weights over KINDS (stage0.json general.enemies, renormalised), the timer o1.limit_s
    Attacker(specs, device, per_side)   the enemy side of such a batch (duel.Script's interface: start, orders(st, j));
                      its lord holds as in sally.Enemy; .sally.fired_s - nan (exam_o1.play reads it: nothing sallies)
"""
import random

import torch

from sim import orders as O
from sim import scripts as SS

KINDS = ("human", "nearest")
SEED = 74000              # (G_def's draw: SP.seeded(SEED), the script's draw Random(SEED + 1); clear of G 73000, G_hold
#                           73500 and every registry range - to register: tmp/engine/b/NEEDS_A.md)
N = 256                   # (the orchestrator 22.11: 256 battles)


def _weights():
    from net.stage0 import cfg as scfg
    e = scfg()["general"]["enemies"]
    w = {k: float(e[k]) for k in KINDS}
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def general_def(n=None, seed=SEED):
    from net.operative import lines as LN
    from net.stage0 import specs as SP
    n = int(N if n is None else n)
    c = LN.conf()
    sp = SP.drill_generals(n, SP.seeded(seed), limit=int(c["limit_s"]))
    rnd = random.Random(seed + 1)
    w = _weights()
    for x in sp:
        r, acc, k = rnd.random(), 0.0, KINDS[-1]
        for kk in KINDS:
            acc += w[kk]
            if r < acc:
                k = kk
                break
        x["enemy"] = "o1"
        x["o1"] = {"att": 0, "enemy": "attack", "script": k, "sally_m": None, "comer": None,
                   "t_dec": float(c["start_s"]), "cause": False, "twin": None, "held": None}
    return sp


def generals(n, seed, mission):
    """n stage-0 general battles for the training mixture (the orchestrator 22.11 F4: METHOD step 5's share of the
    stage's general battles with the operative on top): mission 'attack' - G's recipe (scenes_o1.general_set: att 1
    against the sally holder, its gap drawn from the training axis), 'hold' - G_def's recipe (general_def); the battles
    from SP.seeded(seed), the per-battle draws from Random(seed + 1); spec['seed'] = seed + i (the registry's
    identity of a general battle)."""
    from net.operative import lines as LN
    from net.operative import scenes_o1 as SO
    from net.stage0 import specs as SP
    assert mission in ("attack", "hold"), mission
    if mission == "hold":
        sp = general_def(n, seed)
    else:
        c = LN.conf()
        sp = SP.drill_generals(int(n), SP.seeded(seed), limit=int(c["limit_s"]))
        rnd = random.Random(seed + 1)
        for x in sp:
            x["enemy"] = "o1"
            x["o1"] = {"att": 1, "enemy": "sally", "sally_m": SO.draw(rnd, "sally_m", None), "comer": None,
                       "t_dec": float(c["start_s"]), "cause": True, "twin": None, "held": None}
    for i, x in enumerate(sp):
        x["seed"] = int(seed) + i
        x["o1"]["general"] = mission
    return sp


class _NoSally:
    def __init__(self, B, device):
        self.fired_s = torch.full((B,), float("nan"), device=device)


class Attacker:
    side = 2

    def __init__(self, specs, device, per_side):
        assert all(s["o1"]["enemy"] == "attack" for s in specs), "Attacker: a batch of G_def battles only"
        self.m_human = torch.tensor([s["o1"]["script"] == "human" for s in specs], device=device)
        self.sally = _NoSally(len(specs), device)

    def start(self, *a):
        pass

    def orders(self, st, j):
        B, N_ = st.B, st.N
        o = SS.nearest(st)
        if bool(self.m_human.any()):
            att = st.attacker
            st.attacker = torch.full_like(att, self.side)        # (Human.attack: the enemy is the side that attacks)
            try:
                h = SS.human(st)
            finally:
                st.attacker = att
            o = O.merge(o, h, self.m_human[:, None].expand(B, N_))
        lord = st.u["lord"] & (st.u["side"] == self.side)
        return O.merge(o, O.hold(B, N_, st.device), lord)
