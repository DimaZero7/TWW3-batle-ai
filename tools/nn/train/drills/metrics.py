"""Direct measures of a drill's skill (docs/en/training/training.md "Drills"): what our units do, step by
step, in a drill's battles - whoever plays them (the network in evaluate.play_drills, a script in verify).

A drill may give roles(st) -> (correct [B, N, N], bad [B, N, N]): for unit i, the enemy j it should attack
(correct) and the enemies it must not (bad: e.g. its hard counter). Without roles only the order kinds and
the engagement are counted.

Per battle, over our standing units' seconds (the battle still running):
    correct / bad / other   share attacking (ATTACK order) the correct target / a bad one / another enemy
    hold / move / withdraw   share of the other order kinds
    melee                    share in melee
    first_correct_s          per unit, the battle time its attack order first goes to its correct target
                             (median over the units that ever did; "ever" = their share)
the same over the battle's last `tail_s` seconds before the battle limit (the battles still running then).
"""
import numpy as np
import torch

from tools.nn.sim import orders as O

KEYS = ("correct", "bad", "other", "hold", "move", "withdraw", "melee")


class Tracker:
    def __init__(self, st, ours, roles=None, tail_s=100.0, limit_s=None):
        """ours [B]: our side per battle; roles: the drill's roles function (or None); limit_s: the battle
        limit (default the simulator's)."""
        self.ours, self.roles, self.tail_s = ours.to(st.device), roles, tail_s
        B, N = st.B, st.N
        z = lambda: torch.zeros(B, device=st.device)
        self.all = {k: z() for k in KEYS + ("n",)}
        self.tail = {k: z() for k in KEYS + ("n",)}
        self.first = torch.full((B, N), -1.0, device=st.device)
        if limit_s is None:
            from tools.nn.sim.params import load
            limit_s = load().limit_s
        self.limit = torch.full((B,), float(limit_s), device=st.device)

    @torch.no_grad()
    def update(self, st, live=None, rows=None):
        """One simulator step of st; rows: st's battles are these places of the tracker's batch (a batch
        narrowed to its running battles, drills/verify.run), None: the whole batch."""
        u = st.u
        live = ~st.done if live is None else live
        ours = self.ours if rows is None else self.ours[rows]
        mine = (u["side"] == ours[:, None]) & (u["men"] > 0) & ~u["gone"] & ~u["r"] & live[:, None]
        kind, tgt = u["order_kind"], u["order_target"]
        att = mine & (kind == O.ATTACK) & (tgt >= 0)
        parts = {"hold": mine & (kind == O.HOLD), "move": mine & (kind == O.MOVE),
                 "withdraw": mine & (kind == O.WITHDRAW), "melee": mine & u["m"]}
        if self.roles is not None:
            correct, bad = self.roles(st)
            t = tgt.clamp(min=0)[:, :, None]
            on_correct = att & correct.gather(2, t).squeeze(2)
            on_bad = att & bad.gather(2, t).squeeze(2) & ~on_correct
            parts.update(correct=on_correct, bad=on_bad, other=att & ~on_correct & ~on_bad)
            first = self.first if rows is None else self.first[rows]
            first = torch.where((first < 0) & on_correct, st.t[:, None].expand_as(first), first)
            if rows is None:
                self.first = first
            else:
                self.first[rows] = first
        else:
            parts.update(correct=torch.zeros_like(att), bad=torch.zeros_like(att), other=att)
        limit = self.limit if rows is None else self.limit[rows]
        late = (st.t >= limit - self.tail_s)[:, None]
        for acc, m in ((self.all, mine), (self.tail, mine & late)):
            add = {"n": m.float().sum(1), **{k: (parts[k] & m).float().sum(1) for k in KEYS}}
            for k, v in add.items():
                if rows is None:
                    acc[k] += v
                else:
                    acc[k][rows] += v

    def summary(self, sel=None, st=None):
        """{"all": {key: share}, "tail": {...}, "first_correct_s": median, "ever_correct": share of our units}
        over the battles sel ([B] bool numpy, default all)."""
        out = {}
        for name, acc in (("all", self.all), ("tail", self.tail)):
            a = {k: v.cpu().numpy() for k, v in acc.items()}
            s = np.ones_like(a["n"], dtype=bool) if sel is None else np.asarray(sel)
            n = a["n"][s].sum()
            out[name] = {k: (round(float(a[k][s].sum() / n), 3) if n > 0 else None) for k in KEYS}
            out[name]["unit_s"] = float(n)
        if self.roles is not None and st is not None:
            mine = (st.u["side"] == self.ours[:, None]).cpu().numpy() & ~st.u["lord"].cpu().numpy()
            f = self.first.cpu().numpy()
            s = np.ones(f.shape[0], dtype=bool) if sel is None else np.asarray(sel)
            m = mine & s[:, None]
            got = f[m]
            out["ever_correct"] = round(float((got >= 0).mean()), 3) if got.size else None
            out["first_correct_s"] = round(float(np.median(got[got >= 0])), 1) if (got >= 0).any() else None
        return out
