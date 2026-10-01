"""The reward of a side (docs/en/training/training.md).

* The end of battle: +win for the winner, -win for the loser. At the time limit (both sides still
  have a unit standing) the defender gets +win and the attacker -timeout: worse than losing a
  fight, so an attacker that only stands loses more than one that attacks and fails.
* Every decision, zero-sum shaping (side 2 gets minus side 1's), both terms potential differences
  (they add a steady signal and do not change which outcome is best):
    hp        x (the share of the enemy's health lost - the share of own health lost),
    standing  x (the share of the enemy's army, by cost, that stopped standing (routed, dead, gone)
                 - the same of own army). Routs, not health, decide battles; a rally gives it back.
    lord      x (the enemy's lord died - own lord died): the game's morale rule makes a lord's death
                 decisive (-16, then -10 points to every unit of the side); the cost share of the lord
                 alone (in `standing`) does not show it.
  A share is of the side at the start of battle.
* The attacker pays `idle` every decision in which none of its units fights in melee or shoots:
  standing still has a price from the first second, not only at the limit an hour later. The price
  grows with battle time (x (1 + t / idle_ramp_s), at most x idle_cap), and past `tempo_after_s` the
  attacker pays `tempo` every decision whatever it does: the time limit's -timeout an hour away is
  discounted by gamma^7200 ~ 0.11 at 0.9997 and hardly seen; these costs are seen every step.
* Every real order change costs `order_change`, divided by the side's number of units: a new kind,
  a new attack target, or a move / withdraw point more than `order_move_m` from the one in force.
  KEEP and re-issuing the same order cost nothing. A unit that changes its order every decision
  (2 a second) costs its side ~1.2 over a 10-minute battle, more than a win; one change every 5 s
  ~0.12.
* Every switch of an attack target to another while the old one still stands costs `retarget` more
  (divided the same way): hysteresis against a unit dithering between two enemies (seen in the
  gate battles: a new target every second or two).
No style terms yet (the faction characters are placeholders).

Per unit (unit_step, for per-unit credit in PPO: tools/nn/train/ppo.py), what happens to the unit
itself, beside the side's reward; none of it enters the side's reward:
    unit_hp       x n_own x (HP it dealt / the enemy's starting HP - HP it lost / own starting HP):
                  its share of the side's health trade, scaled to one unit (sums to unit_hp x n_own x
                  the side's trade, as `hp` does for the side);
    flanked       per decision struck in the flank or rear in melee (the game: ~x1.74 losses);
    missile_melee per decision a missile unit spends in melee;
    crowd         per decision of a pile, by the excess share (tools/nn/train/behaviour.py);
    idle_near     per decision a melee unit with no attack order stands out of melee while a fellow
                  within 60 m fights (a unit's own credit otherwise pays it to let others fight);
    flank_attack  per decision striking an enemy's flank or rear (a bonus, as large as `flanked`: a flank
                  exchange is zero-sum between the two units);
then `neighbour` x the mean of the same of own units within neighbour_m (what happens next to it).
Each shaped term at most 1200 decisions x weight per unit in a 10-minute battle: at 2e-4, 0.24, a
quarter of a win.
"""
import math
from dataclasses import dataclass

import torch

from tools.nn.sim import orders as O


@dataclass(frozen=True)
class Weights:
    win: float = 1.0
    timeout: float = 1.5          # the attacker's loss at the time limit
    hp: float = 0.5
    standing: float = 0.5
    idle: float = 2e-4            # the attacker, per decision with no unit fighting or shooting ...
    idle_ramp_s: float = 0.0      # ... x (1 + t / idle_ramp_s) (0: flat)
    idle_cap: float = 4.0         # ... at most this many times
    tempo: float = 0.0            # the attacker, per decision once the battle is older than tempo_after_s
    tempo_after_s: float = 600.0
    order_change: float = 0.001
    order_move_m: float = 10.0
    lord: float = 0.3             # the enemy lord's death - own lord's death
    retarget: float = 0.003       # an attack switched to another target while the old one stands
    # per unit (unit_step), not in the side's reward
    unit_hp: float = 0.05         # the unit's own health trade, scaled to one unit
    flanked: float = 2e-4         # per decision struck in flank / rear
    missile_melee: float = 2e-4   # per decision a missile unit is in melee
    crowd: float = 2e-4           # per decision of a pile (excess share)
    idle_near: float = 0.0        # per decision a melee unit stands by while a fellow within 60 m fights
    flank_attack: float = 2e-4    # per decision striking an enemy's flank / rear (bonus; = flanked: zero-sum)
    neighbour: float = 0.5        # + this x the mean of own units' terms within neighbour_m
    neighbour_m: float = 40.0


def standing_mask(u):
    return (u["side"] > 0) & (u["men"] > 0) & ~u["gone"] & ~u["r"]


def measure(st):
    """[B, 2, 3]: per side (1, 2) the share of starting health left, the share of the army's cost
    still standing, and 1 while its lord lives (or it has none), 0 once the lord is dead."""
    u = st.u
    stand = standing_mask(u).float()
    alive = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"]
    out = []
    for s in (1, 2):
        mine = (u["side"] == s).float()
        hp = (u["hp_abs"] * mine).sum(1) / (u["hp0"] * mine).sum(1).clamp(min=1e-6)
        cost = u["cost"].clamp(min=1.0) * mine
        lords = u["lord"] & (u["side"] == s)
        lord = (~lords.any(1) | (lords & alive).any(1)).float()
        out.append(torch.stack([hp, (cost * stand).sum(1) / cost.sum(1).clamp(min=1e-6), lord], 1))
    return torch.stack(out, 1)


def health(st):
    """[B, 2] the share of each side's starting health still left (0-1)."""
    return measure(st)[:, :, 0]


def step(before, after, finished, winner, attacker, weights=Weights()):
    """[B, 2] reward of side 1 and side 2 for one step, without the idle and order costs.

    before, after: measure() [B, 2, 3] around the step (the lord column optional); finished [B]: the battle ended in this step;
    winner [B] 1 or 2 (read where finished); attacker [B] 1 or 2."""
    lost = (before - after).clamp(min=0)                                  # [B, side, (hp, standing, lord)]
    r1 = weights.hp * (lost[:, 1, 0] - lost[:, 0, 0]) + weights.standing * (lost[:, 1, 1] - lost[:, 0, 1])
    if lost.shape[-1] > 2:
        r1 = r1 + weights.lord * (lost[:, 1, 2] - lost[:, 0, 2])
    timeout = finished & (after[:, 0, 1] > 0) & (after[:, 1, 1] > 0)
    won1 = (winner == 1).float() - (winner == 2).float()
    end = torch.where(finished, weights.win * won1, torch.zeros_like(r1))
    r = torch.stack([r1 + end, -r1 - end], 1)
    # At the limit the attacker loses more than a fight: -timeout instead of -win.
    extra = torch.where(timeout, torch.full_like(r1, weights.win - weights.timeout), torch.zeros_like(r1))
    side = torch.stack([attacker == 1, attacker == 2], 1).float()
    return r + side * extra[:, None]


def idle_cost(st, weights=Weights()):
    """[B, 2]: the attacker pays `idle` when none of its units is in melee or shooting."""
    u = st.u
    busy = (u["m"] | u["fire"]) & standing_mask(u)
    out = []
    scale = torch.ones_like(st.t)
    if weights.idle_ramp_s > 0:
        scale = (1 + st.t / weights.idle_ramp_s).clamp(max=weights.idle_cap)
    late = (st.t > weights.tempo_after_s).float() * weights.tempo
    for s in (1, 2):
        mine = (st.attacker == s) & ~st.done
        idle = ~(busy & (u["side"] == s)).any(1) & mine
        out.append(weights.idle * scale * idle.float() + late * mine.float())
    return torch.stack(out, 1)


def order_changes(u, orders, move_m=Weights.order_move_m):
    """[B, N] bool: standing units whose order in force really changes with `orders` (before the
    simulator's step takes them): a new kind, a new attack target, or a move / withdraw point
    more than move_m metres from the one in force. KEEP is never a change."""
    k = orders.kind
    new_kind = k != u["order_kind"]
    new_target = (k == O.ATTACK) & (orders.target != u["order_target"])
    point = (k == O.MOVE) | (k == O.WITHDRAW)
    moved = point & ((orders.x - u["ox"]) ** 2 + (orders.z - u["oz"]) ** 2 > move_m ** 2)
    return standing_mask(u) & (k != O.KEEP) & (new_kind | new_target | moved)


def retargets(u, orders):
    """[B, N] bool: standing units told to attack another enemy while their attack target in force
    still stands."""
    old = u["order_target"].clamp(min=0)
    old_stands = (u["order_kind"] == O.ATTACK) & (u["order_target"] >= 0) & standing_mask(u).gather(1, old)
    return standing_mask(u) & (orders.kind == O.ATTACK) & old_stands & (orders.target != u["order_target"])


def order_cost(changes, side, weights=Weights(), switched=None):
    """[B, 2] what each side pays for its order changes [B, N] and target switches [B, N] (side
    [B, N]: 1, 2 or 0)."""
    out = []
    for s in (1, 2):
        mine = side == s
        n = mine.float().sum(1).clamp(min=1)
        c = weights.order_change * (changes & mine).float().sum(1)
        if switched is not None:
            c = c + weights.retarget * (switched & mine).float().sum(1)
        out.append(c / n)
    return torch.stack(out, 1)


UNIT_FIELDS = ("hp_abs", "k", "dealt")


def unit_before(u):
    """What unit_step needs from before the step."""
    return {k: u[k].clone() for k in UNIT_FIELDS}


def unit_step(before, st, facts, params, weights=Weights()):
    """[B, N] each unit's own reward for the step (0 for empty slots): its health trade and the
    shaped terms of behaviour.facts (after the step), then its neighbours' mean (see the module)."""
    u = st.u
    side = u["side"]
    present = side > 0
    lost = (before["hp_abs"] - u["hp_abs"]).clamp(min=0)
    fade = math.exp(-params.dt / params.sim["morale"]["recent_s"])
    melee = (u["dealt"] - fade * before["dealt"]).clamp(min=0)            # melee HP dealt this step
    tgt = u["target"].clamp(min=0)
    shot = torch.where(u["fire"] & (u["target"] >= 0), (u["k"] - before["k"]).clamp(min=0) * u["hp_man"].gather(1, tgt),
                       torch.zeros_like(melee))
    dealt = torch.where(u["m"], melee, shot)
    hp0 = torch.stack([(u["hp0"] * (side == s)).sum(1) for s in (1, 2)], 1).clamp(min=1e-6)   # [B, 2]
    n = torch.stack([(side == s).sum(1) for s in (1, 2)], 1).float()
    own_i = (side - 1).clamp(min=0)
    own_hp0, enemy_hp0 = hp0.gather(1, own_i), hp0.gather(1, 1 - own_i)
    trade = n.gather(1, own_i) * (dealt / enemy_hp0 - lost / own_hp0)
    r = (weights.unit_hp * trade - weights.flanked * facts["flanked"].float()
         - weights.missile_melee * facts["missile_melee"].float() - weights.crowd * facts["crowded"]
         - weights.idle_near * facts["idle_near"].float()
         + weights.flank_attack * facts["flank_attack"].float())
    r = torch.where(present, r, torch.zeros_like(r))
    if weights.neighbour:
        dx = u["x"][:, :, None] - u["x"][:, None, :]
        dz = u["z"][:, :, None] - u["z"][:, None, :]
        alive = present & (u["men"] > 0) & ~u["gone"]
        eye = torch.eye(side.shape[1], dtype=torch.bool, device=side.device)[None]
        near = ((dx * dx + dz * dz) <= weights.neighbour_m ** 2) & (side[:, :, None] == side[:, None, :])             & alive[:, :, None] & alive[:, None, :] & ~eye
        nf = near.float()
        mean = (nf * r[:, None, :]).sum(2) / nf.sum(2).clamp(min=1)
        r = r + weights.neighbour * mean
    return r
