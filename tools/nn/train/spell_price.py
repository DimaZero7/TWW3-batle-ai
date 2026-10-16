"""A damage spell's cast priced by direct accounting (the user, 16.10.2026; docs/en/training/ability_price.md "Spells"):
no "pressed / not pressed" replay - the simulator knows which damage came from the spell. ability_branch.Brancher with
Config.spell_price "direct" (the default; "branch": the replays, as the S25 abilities) hands every cast of the learner
to the ledger and gets its price back for the ability head's own signal:

    price (gold) = the gold of enemy health the cast's bombardment took (the caster's bm_gold_e over its window)
                   - the gold of own health it took (bm_gold_f)
                   - k_winds x the winds it spent (its cost, when it launched)

k_winds (gold a wind point): what else the winds would have bought - the mean direct gold of a good cast (a dense
standing enemy in range: the spell lab's "dense" pairs, lab_spell.py) a wind point; config/ability_price.json
"spell_direct". The caster's walk-up and his own death are the battle's reward (gold lost), not this price.

The window: from the decision that orders the cast to the end of its effect - the caster's spell timer (sp{k}_on) seen
on and then off at two decisions in a row (the last landings: Warp Lightning's last bolt 0.2 s after its 5 s) - or
horizon_s at most (a cast that never launched: out of winds, the target gone - its winds are not spent), or the
battle's end (its values frozen at the last decision of that battle). A cast whose window is still open at the chunk's
end is not priced (its row gets no ability advantage). The caster's next cast waits its recharge (38 s), so nothing
of it is in the window.

Torch; the ledger's tensors stay on the device (no read back but the chunk's end).
"""
import json
from pathlib import Path

import torch

from tools.nn.model import abilities as mab
from tools.nn.model import heads as hd
from tools.nn.train import reward

ROOT = Path(__file__).resolve().parents[3]
FILE = ROOT / "config" / "ability_price.json"
HORIZON_S = 30.0
FIELDS = ("t", "r", "n", "b", "j", "e0", "f0", "e", "f", "tb", "cd0", "launched", "seen_on", "zeros", "done", "lp",
          "kind", "mana", "budget", "age")


def load(path=None):
    """(k_winds, horizon_s) of config/ability_price.json "spell_direct" (0, HORIZON_S without it)."""
    p = Path(path or FILE)
    spec = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    d = spec.get("spell_direct") or {}
    return float(d.get("k_winds", 0.0)), float(d.get("horizon_s", HORIZON_S))


class Ledger:
    """The learner's casts of a chunk and their running accounts (module doc)."""

    def __init__(self, k_winds=0.0, horizon_s=HORIZON_S):
        self.k, self.horizon_s = float(k_winds), float(horizon_s)
        self.rows = None                       # {field: [P] tensor} of the chunk's casts
        self.stats = {}

    def reset(self):
        self.rows = None

    def add(self, env, info, t):
        """The casts ordered at decision t (info: Brancher.capture's - obs, action, logits of the learner rows)."""
        obs, action = info["obs"], info["action"]
        if action.ability is None or obs.get("abil_idx") is None:
            return
        rows = env.rows_learn
        live = obs["ctrl"] & ~env.st.done[rows % env.B][:, None]
        cast = live & hd.is_spell(action.ability)
        nz = cast.nonzero()                                                 # (reads the GPU: a few a chunk)
        if not len(nz):
            return
        r, n = nz[:, 0], nz[:, 1]
        slot = action.ability[r, n]
        b = rows[r] % env.B
        u = env.st.u
        j = (slot - mab.SPELL_AT).clamp(min=0, max=mab.SPELLS - 1)
        sp_cd = torch.stack([u[f"sp{k}_cd"] for k in range(mab.SPELLS)], -1)
        la = torch.log_softmax(torch.nan_to_num(info["logits"]["ability"].float(), nan=hd.NEG), -1)
        idx = obs["abil_idx"][r, n, slot].long()
        new = {"t": torch.full_like(r, t), "r": r, "n": n, "b": b, "j": j,
               "e0": u["bm_gold_e"][b, n].float(), "f0": u["bm_gold_f"][b, n].float(),
               "e": u["bm_gold_e"][b, n].float(), "f": u["bm_gold_f"][b, n].float(), "tb": env.st.t[b].float(),
               "cd0": sp_cd[b, n, j].float(), "launched": torch.zeros_like(r, dtype=torch.bool),
               "seen_on": torch.zeros_like(r, dtype=torch.bool), "zeros": torch.zeros_like(r),
               "done": torch.zeros_like(r, dtype=torch.bool), "lp": la[r, n, slot + 1],
               "kind": torch.where(idx >= 0, idx, torch.zeros_like(idx)),
               "mana": mab.column("mana", idx).float() * mab.WINDS_SCALE,
               "budget": reward.budget(u)[b].float(), "age": torch.zeros_like(r, dtype=torch.float32)}
        self.rows = new if self.rows is None else {k: torch.cat([self.rows[k], new[k]]) for k in FIELDS}

    def update(self, env, dt_s):
        """The open casts' accounts from the state now (dt_s: seconds since the last update)."""
        R = self.rows
        if R is None:
            return
        u = env.st.u
        b, n, j = R["b"], R["n"], R["j"]
        tb = env.st.t[b].float()
        same = (tb >= R["tb"]) & ~R["done"] & ~env.st.done[b]          # the same battle, still running
        ended = ~R["done"] & ~same                                     # it ended or restarted: frozen as last seen
        sp_on = torch.stack([u[f"sp{k}_on"] for k in range(mab.SPELLS)], -1)[b, n, j]
        sp_cd = torch.stack([u[f"sp{k}_cd"] for k in range(mab.SPELLS)], -1)[b, n, j]
        on = same & (sp_on > 0)
        launched = R["launched"] | on | (same & (sp_cd > R["cd0"] + 1.0))
        e = torch.where(same, u["bm_gold_e"][b, n].float(), R["e"])
        f = torch.where(same, u["bm_gold_f"][b, n].float(), R["f"])
        zeros = torch.where(same & R["seen_on"] & ~on, R["zeros"] + 1, R["zeros"])
        age = R["age"] + dt_s
        over = same & ((zeros >= 2) | (age >= self.horizon_s))
        R.update(e=e, f=f, tb=torch.where(same, tb, R["tb"]), launched=launched, seen_on=R["seen_on"] | on,
                 zeros=zeros, age=age, done=R["done"] | over | ended)

    def result(self, env, T):
        """(t, r, n, share, lp, kind) of the chunk's casts whose window closed (price / the battle's budget), and the log's
        stats; the ledger is emptied."""
        R = self.rows
        self.rows = None
        self.stats = {}
        if R is None:
            return None
        gold = (R["e"] - R["e0"]) - (R["f"] - R["f0"]) - self.k * R["mana"] * R["launched"].float()
        ok = R["done"] & (R["t"] < T)
        n = int(ok.sum())
        self.stats = {"spell_casts": int(len(gold)), "spell_priced": n}
        if n:
            g, ln = gold[ok], R["launched"][ok]
            self.stats.update(spell_gold=round(float(g.mean()), 2), spell_launched=round(float(ln.float().mean()), 3),
                              spell_good=round(float((g > 0).float().mean()), 3),
                              spell_damage_gold=round(float((R["e"] - R["e0"])[ok].mean()), 2),
                              spell_friendly_gold=round(float((R["f"] - R["f0"])[ok].mean()), 2))
        share = gold / R["budget"].clamp(min=1.0)
        return {k: v[ok] for k, v in (("t", R["t"]), ("r", R["r"]), ("n", R["n"]), ("share", share), ("lp", R["lp"]),
                                      ("kind", R["kind"]))}
