"""What every unit does in a batch of battles, per decision (docs/en/training/training.md).

facts() reads the simulator's state after a step and says, per unit [B, N]:

    melee          standing and in melee
    flanked        ... and struck from the flank or rear now (the simulator's flank_hit >= 1;
                   the game: a flank attack costs the defender ~x1.74 losses)
    missile_melee  a missile unit (range > 0, not a lord) standing in melee
    flank_attack   standing, in melee, its target a standing enemy it reaches from that enemy's
                   flank or rear (the angle off the target's facing, contact.front_deg / rear_deg)
    crowded        one of more than `crowd` own units on the same enemy (its attack order, or the
                   enemy it fights) while another standing enemy strikes an own unit in flank or rear;
                   as the share of the excess, (n - crowd) / n, so the whole pile weighs n - crowd
    idle_near      a standing melee unit (not missile, not a lord) out of melee, with no attack order,
                   while an own unit within IDLE_M fights in melee: it lets its fellows fight alone
    aim            the enemy slot it attacks or fights (-1 none)

The test protocol's metrics (Tracker, tools/nn/train/test5.py) and the per-unit reward terms
(tools/nn/train/reward.py unit_step) both read it.
"""
import math

import torch

from tools.nn.sim import geometry
from tools.nn.sim import orders as O

CROWD = 2          # more own units than this on one enemy is a pile
IDLE_M = 60.0      # a fight this near an idle own melee unit is its business
ABILITY_SLOTS = 3


def standing_mask(u):
    return (u["side"] > 0) & (u["men"] > 0) & ~u["gone"] & ~u["r"]


def _gather(x, idx):
    return x.gather(1, idx.clamp(min=0))


def facts(st, params, crowd=CROWD):
    u = st.u
    side = u["side"]
    stand = standing_mask(u)
    melee = stand & u["m"]
    cc = params.sim["contact"]

    # Whom each unit fights, and from which side of that enemy.
    tgt = u["target"]
    t_ok = (tgt >= 0) & (_gather(side, tgt) != side) & (_gather(side, tgt) > 0) & _gather(stand, tgt)
    dx = _gather(u["x"], tgt) - u["x"]
    dz = _gather(u["z"], tgt) - u["z"]
    rel = geometry.wrap(torch.atan2(dx, dz) + math.pi - _gather(u["b"], tgt) * geometry.DEG)
    sector = geometry.sector(rel, cc["front_deg"], cc["rear_deg"])
    flank_attack = melee & t_ok & (sector >= 1)

    # Piles: own units on one enemy (an attack order counts before contact), while another flanks us.
    attack = (u["order_kind"] == O.ATTACK) & (u["order_target"] >= 0)
    aim = torch.where(attack, u["order_target"], torch.where(melee & (tgt >= 0), tgt, torch.full_like(tgt, -1)))
    aim = torch.where(stand & (aim >= 0) & (_gather(side, aim) != side), aim, torch.full_like(aim, -1))
    B, N = aim.shape
    count = torch.zeros(B, N + 1, device=aim.device).scatter_add_(1, aim + 1, torch.ones(B, N, device=aim.device))
    on_mine = _gather(count[:, 1:], aim) * (aim >= 0).float()              # [B, N] units on my aim, me included
    flankers = [(flank_attack & (side == s)).sum(1, keepdim=True) for s in (1, 2)]   # side s's units in our flank
    enemy_flankers = torch.where(side == 1, flankers[1], flankers[0])
    other = enemy_flankers - (_gather(flank_attack, aim) & (aim >= 0)).long()   # not the one we pile on
    crowded = torch.where((aim >= 0) & (on_mine > crowd) & (other > 0) & stand,
                          (on_mine - crowd) / on_mine.clamp(min=1), torch.zeros_like(on_mine))

    missile = (u["range"] > 0) & (u["ammo0"] > 0) & ~u["lord"]
    dx = u["x"][:, :, None] - u["x"][:, None, :]
    dz = u["z"][:, :, None] - u["z"][:, None, :]
    fight_near = ((dx * dx + dz * dz <= IDLE_M ** 2) & (side[:, :, None] == side[:, None, :])
                  & melee[:, None, :]).any(2)
    idle_near = stand & ~melee & ~missile & ~u["lord"] & ~attack & fight_near
    return {"idle_near": idle_near, "melee": melee, "flanked": melee & (u["flank_hit"] >= 1), "missile_melee": melee & missile,
            "missile": stand & missile, "flank_attack": flank_attack, "crowded": crowded, "aim": aim}


def ability_marks(u):
    """[B, N, SLOTS] the abilities' cooldowns: a use sets its cooldown up (tools/nn/sim/abilities.py)."""
    return torch.stack([u.get(f"ab{k}_cd", torch.zeros_like(u["men"])) for k in range(ABILITY_SLOTS)], -1)


def track(st, params, mine, live, sums, marks):
    """Tracker.update: adds this step to sums (in place) -> the abilities' new marks."""
    f = facts(st, params)
    dt = params.dt
    m = mine & live[:, None]
    add = {"steps": live.float(),
           "melee_s": (f["melee"] & m).float().sum(1) * dt,
           "flanked_s": (f["flanked"] & m).float().sum(1) * dt,
           "missile_melee_s": (f["missile_melee"] & m).float().sum(1) * dt,
           "missile_s": (f["missile"] & m).float().sum(1) * dt,
           "flank_attack_s": (f["flank_attack"] & m).float().sum(1) * dt,
           "crowd_steps": ((f["crowded"] > 0) & m).any(1).float(),
           "crowded_units": (f["crowded"] * m.float()).sum(1)}
    new = ability_marks(st.u)
    add["abilities"] = ((new > marks + 1e-3) & m[..., None]).float().sum((1, 2))
    for k, v in add.items():
        sums[k] += v
    return new


class Tracker:
    """Per battle [B] sums over the learner's units (mine [B, N]) while the battle runs: seconds in
    melee, flanked, missile units in melee, flank attacks; decisions with a pile; ability uses."""
    FIELDS = ("steps", "melee_s", "flanked_s", "missile_melee_s", "missile_s", "flank_attack_s", "crowd_steps",
              "crowded_units", "abilities")

    def __init__(self, st, params, mine, wrap=None):
        """wrap: e.g. tools/nn/train/rollout.py fast (compiles the update on CUDA)."""
        self.params = params
        self.mine = mine
        self.dt = params.dt
        self.sums = {k: torch.zeros(st.B, device=st.device) for k in self.FIELDS}
        self.marks = ability_marks(st.u)
        self._track = wrap(track) if wrap else track

    def update(self, st, live):
        """After a step; live [B]: the battle was running before it."""
        self.marks = self._track(st, self.params, self.mine, live, self.sums, self.marks)

    def summary(self, sel):
        """Metrics over the battles sel [B] (numpy bool)."""
        s = {k: v.cpu().numpy()[sel] for k, v in self.sums.items()}
        n = max(1, int(sel.sum()))
        melee = max(1e-9, float(s["melee_s"].sum()))
        steps = max(1e-9, float(s["steps"].sum()))
        minutes = steps * self.dt / 60
        return {"missile_melee_s": float(s["missile_melee_s"].sum() / n),
                "missile_melee_share": float(s["missile_melee_s"].sum() / max(1e-9, s["missile_s"].sum())),
                "flanked_share": float(s["flanked_s"].sum() / melee),
                "flank_attack_share": float(s["flank_attack_s"].sum() / melee),
                "crowding_share": float(s["crowd_steps"].sum() / steps),
                "crowded_units_per_min": float(s["crowded_units"].sum() / minutes),
                "abilities_per_battle": float(s["abilities"].sum() / n)}
