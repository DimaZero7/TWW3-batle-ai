"""The operative's drills out of the user's battles (sim/evidence/diag_m1/drills.md; tmp/old-build/diag_label/human_rules.md episodes):
a piece of a matchup's armies on their places, a script per enemy role, a measure that tells the teacher from a bad
operator. Each drill guards against one stupidity.

    threat   (O1, M1 80-155 s) my whole army; the enemy's fast units (wolf rats) alone, 220-260 m from my shooters, at
             them at a run (nearest). Right: the shooters and the nearest pair of halberds on one fast unit at a time.
             Measure: the share of my shooters' men kept by 120 s (and the fast routed by then)
    guns     (O4 M2 17-160 s, O5 M3 6-120 s) my army; the enemy's artillery and snipers standing where they stood (hold:
             they shoot), the rest of the enemy out. Right: my artillery on theirs, my snipers on their snipers.
             Measure: the enemy's artillery and snipers' share of men lost (a routed unit: all) by 240 s
    skirm    (O2, M1 155-300 s) my army; the enemy's shooters (slingers, skirmishers) 140-180 m from my snipers, standing
             and shooting (hold), the rest out. Right: a pair of halberds goes for the nearest, the line stays.
             Measure: the share of the enemy's shooters routed or dead by 180 s
    still    (Г14, «такт — это взгляд») both armies of M1 on their places, the enemy holds 90 s (nobody comes within
             range). Right: no change of plan. Measure: changes of plan (duel.Chain plan_changes) in 45 - 90 s, a battle
Bad operators: the old formula (v1: the clumps' rule, no target to the units) and the teacher without the target to the
units; the network on the same starts.

    play(drill, battles, op, ...) -> {measure [B], ...}; exam(...) the table of an operator
"""
import random

import numpy as np
import torch

from net.matchup import armies as AR
from net.matchup import duel as DU
from net.cards import passport as PP
from net.operative import army as AM
from sim import orders as O
from sim import scripts as opponents

DRILLS = {"threat": {"matchup": "m1", "limit": 150, "enemy_roles": ("fast",), "gap": (220.0, 260.0), "script": "run"},
          "guns": {"matchup": "m2", "limit": 240, "enemy_roles": ("art", "sniper"), "gap": None, "script": "hold"},
          "skirm": {"matchup": "m1", "limit": 180, "enemy_roles": ("shooter",), "gap": (140.0, 180.0), "script": "hold"},
          "still": {"matchup": "m1", "limit": 91, "enemy_roles": None, "gap": None, "script": "hold"}}


def roles_of(keys):
    return AM.roles(np.asarray(PP.table(list(keys))))


def start(drill, rnd):
    """One battle's units {1: [...], 2: [...]} of the drill (module doc)."""
    d = DRILLS[drill]
    b = AR.augment(AR.base(d["matchup"]), rnd, aug=dict(AR.cfg()["augment"], drop=[0, 0], mirror=0.5))
    rl2 = roles_of([u["key"] for u in b[2]])
    if d["enemy_roles"] is None:                 # (the whole army, on its places)
        return b
    keep = [u for u, r in zip(b[2], rl2) if r in d["enemy_roles"]]
    if not keep:
        keep = b[2][:1]
    lord2 = [u for u in b[2] if u["general"]]
    # the enemy's general stays (the scenario wants one; far behind, holding) unless he is of the drill
    if lord2 and lord2[0] not in keep:
        lg = dict(lord2[0])
        lg["x"] = 480.0
        keep = [lg] + keep
    if d["gap"] is not None:
        rl1 = roles_of([u["key"] for u in b[1]])
        sh = [u for u, r in zip(b[1], rl1) if r in ("shooter", "sniper")] or b[1]
        tx = max(u["x"] for u in sh)
        gap = rnd.uniform(*d["gap"])
        for u in keep:
            if not u["general"]:
                u["x"] = tx + gap + rnd.uniform(-10, 10)
    b[2] = keep
    return b


class MixScript(DU.Script):
    """The enemy of a drill: its drill units by the drill's script (run: nearest at a run; hold), the general holds."""

    def __init__(self, mode):
        self.side, self.mode = 2, mode

    def orders(self, st, j):
        if self.mode == "run":
            o = opponents.nearest(st)
            gen = st.u["lord"]
            return O.merge(o, O.hold(st.B, st.N, st.device), gen)
        return O.hold(st.B, st.N, st.device)


def play(drill, B, op, unit_net, P, device, seed=0, targets=True):
    """(readings + the drill's measure [B], stats). op: 'v1', 'v2' (the teacher), or an OperativeNet."""
    from net.operative import cfg as ocfg
    d = DRILLS[drill]
    rnd = random.Random(seed)
    battles = [start(drill, rnd) for _ in range(B)]
    am = ocfg()["teacher"]["army"]["min"]
    if op == "v1":
        ocfg()["teacher"]["army"]["min"] = 10 ** 6
        chain = DU.Chain(1, "formula", command="hold", targets=False, still=False, aim=False)
    elif op == "v2":
        chain = DU.Chain(1, "formula", command="hold", targets=targets)
    else:
        chain = DU.Chain(1, op, command="hold", targets=targets)
    st0 = DU.build(d["matchup"], battles, device, P)[0]
    u0 = st0.u
    try:
        rd, stats = DU.play(d["matchup"], battles, {1: chain, 2: MixScript(d["script"])}, unit_net, P, device,
                            limit_s=d["limit"])
    finally:
        ocfg()["teacher"]["army"]["min"] = am
    st = chain.setup.st
    u = st.u
    side = u["side"]
    keys = st.keys
    roles = [roles_of(k) for k in keys]
    rmask = lambda names, s: torch.tensor([[r in names and int(side[b, i]) == s for i, r in enumerate(rr)]   # noqa
                                           for b, rr in enumerate(roles)], device=device)
    men0 = u["men0"].float().clamp(min=1)
    kept = (u["men"].float() / men0) * (~u["r"]).float()
    if drill == "threat":
        sh = rmask(("shooter", "sniper"), 1)
        m = (kept * sh).sum(1) / sh.float().sum(1).clamp(min=1)
    elif drill == "guns":
        tg = rmask(("art", "sniper"), 2)
        lost = 1.0 - kept                                     # (a gun keeps a few crew long after it is silenced)
        m = (lost * tg).sum(1) / tg.float().sum(1).clamp(min=1)
    elif drill == "still":
        # changes of plan in the windows 45 - 90 s (60 s: nobody moves, the situation is the same) a battle
        m = torch.full((B,), sum(chain.by_window[2:]) / B, device=device)
    else:
        # (the slingers out-range nobody at 140 - 180 m: what tells the teacher is whether they are driven off)
        tg = rmask(("shooter",), 2)
        out = (u["r"] | (u["men"] <= 0) | u["gone"]) & tg
        m = out.float().sum(1) / tg.float().sum(1).clamp(min=1)
    rd["measure"] = m
    return rd, stats


def exam(drill, op, unit_net, P, device, B=32, targets=True):
    rd, st = play(drill, B, op, unit_net, P, device, targets=targets)
    s = DU.summary(rd)
    s["measure"] = round(float(rd["measure"].mean()), 3)
    s["drill"], s["op"] = drill, op if isinstance(op, str) else "network"
    return s, st
