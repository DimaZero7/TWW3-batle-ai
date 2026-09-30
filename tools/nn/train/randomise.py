"""Domain randomisation: every battle gets its numbers moved a little, so the network does not
learn the simulator's exact numbers (the simulator is off from the game by up to ~20 %,
docs/en/training/simulator.md).

Per battle, at its start:
* damage (melee and missile, base and armour-piercing), speed (walk, run, charge) and morale
  (leadership) are each multiplied by one factor for the battle, uniform in 1 +- `common`,
  times a factor per side, 1 +- `side` (a side may be a little stronger than its passport says);
* every unit's place moves by up to `jitter_m` metres.
The network does not see these factors: its input takes the passports from config/nn/units.json.
"""
from dataclasses import dataclass

import torch

GROUPS = {
    "damage": ("damage", "ap_damage", "m_damage", "m_ap"),
    "speed": ("walk", "run", "charge_speed"),
    "morale": ("leadership",),
}


@dataclass(frozen=True)
class Spread:
    common: float = 0.15     # +- share, the same for both sides
    side: float = 0.05       # +- share, per side on top
    jitter_m: float = 5.0    # start places

    @property
    def off(self):
        return self.common == 0 and self.side == 0 and self.jitter_m == 0


NONE = Spread(0.0, 0.0, 0.0)


def factors(B, spread, device="cpu", generator=None):
    """{group: [B, 2]} the factor of each side of each battle."""
    out = {}
    for g in GROUPS:
        c = 1 + (torch.rand(B, 1, device=device, generator=generator) * 2 - 1) * spread.common
        s = 1 + (torch.rand(B, 2, device=device, generator=generator) * 2 - 1) * spread.side
        out[g] = c * s
    return out


def apply(st, rows, spread, generator=None):
    """Randomise the battles where rows [B] is true, in place; call right after they (re)start.
    Returns the factors ({group: [B, 2]}, 1 where rows is false)."""
    u = st.u
    B, N = u["side"].shape
    f = factors(B, spread, st.device, generator)
    side = (u["side"] - 1).clamp(min=0)
    for g, names in GROUPS.items():
        f[g] = torch.where(rows[:, None], f[g], torch.ones_like(f[g]))
        per_unit = f[g].gather(1, side)                                     # [B, N]
        for name in names:
            u[name] = u[name] * per_unit
    u["morale"] = torch.where(rows[:, None], u["leadership"].clone(), u["morale"])
    if spread.jitter_m:
        for k in ("x", "z"):
            noise = (torch.rand(B, N, device=st.device, generator=generator) * 2 - 1) * spread.jitter_m
            u[k] = torch.where(rows[:, None], u[k] + noise, u[k])
        u["ox"] = torch.where(rows[:, None], u["x"], u["ox"])
        u["oz"] = torch.where(rows[:, None], u["z"], u["oz"])
    return f
