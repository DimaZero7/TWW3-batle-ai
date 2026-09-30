"""The reward of a side (docs/en/training/training.md).

* The end of battle: +win for the winner, -win for the loser. At the time limit (both sides still
  have a unit standing) the defender gets +win and the attacker -timeout: worse than losing a
  fight, so an attacker that only stands loses more than one that attacks and fails.
* Every decision, zero-sum shaping (side 2 gets minus side 1's), both terms potential differences
  (they add a steady signal and do not change which outcome is best):
    hp        x (the share of the enemy's health lost - the share of own health lost),
    standing  x (the share of the enemy's army, by cost, that stopped standing (routed, dead, gone)
                 - the same of own army). Routs, not health, decide battles; a rally gives it back.
  A share is of the side at the start of battle.
* The attacker pays `idle` every decision in which none of its units fights in melee or shoots:
  standing still has a price from the first second, not only at the limit an hour later.
* Every real order change costs `order_change`, divided by the side's number of units: a new kind,
  a new attack target, or a move / withdraw point more than `order_move_m` from the one in force.
  KEEP and re-issuing the same order cost nothing. A unit that changes its order every decision
  (2 a second) costs its side ~1.2 over a 10-minute battle, more than a win; one change every 5 s
  ~0.12.
No style terms yet (the faction characters are placeholders).
"""
from dataclasses import dataclass

import torch

from tools.nn.sim import orders as O


@dataclass(frozen=True)
class Weights:
    win: float = 1.0
    timeout: float = 1.5          # the attacker's loss at the time limit
    hp: float = 0.5
    standing: float = 0.5
    idle: float = 2e-4            # the attacker, per decision with no unit fighting or shooting
    order_change: float = 0.001
    order_move_m: float = 10.0


def standing_mask(u):
    return (u["side"] > 0) & (u["men"] > 0) & ~u["gone"] & ~u["r"]


def measure(st):
    """[B, 2, 2]: per side (1, 2) the share of starting health left and the share of the army's cost
    still standing."""
    u = st.u
    stand = standing_mask(u).float()
    out = []
    for s in (1, 2):
        mine = (u["side"] == s).float()
        hp = (u["hp_abs"] * mine).sum(1) / (u["hp0"] * mine).sum(1).clamp(min=1e-6)
        cost = u["cost"].clamp(min=1.0) * mine
        out.append(torch.stack([hp, (cost * stand).sum(1) / cost.sum(1).clamp(min=1e-6)], 1))
    return torch.stack(out, 1)


def health(st):
    """[B, 2] the share of each side's starting health still left (0-1)."""
    return measure(st)[:, :, 0]


def step(before, after, finished, winner, attacker, weights=Weights()):
    """[B, 2] reward of side 1 and side 2 for one step, without the idle and order costs.

    before, after: measure() [B, 2, 2] around the step; finished [B]: the battle ended in this step;
    winner [B] 1 or 2 (read where finished); attacker [B] 1 or 2."""
    lost = (before - after).clamp(min=0)                                  # [B, side, (hp, standing)]
    r1 = weights.hp * (lost[:, 1, 0] - lost[:, 0, 0]) + weights.standing * (lost[:, 1, 1] - lost[:, 0, 1])
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
    for s in (1, 2):
        idle = ~(busy & (u["side"] == s)).any(1) & (st.attacker == s) & ~st.done
        out.append(weights.idle * idle.float())
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


def order_cost(changes, side, weights=Weights()):
    """[B, 2] what each side pays for its order changes [B, N] (side [B, N]: 1, 2 or 0)."""
    out = []
    for s in (1, 2):
        mine = side == s
        n = mine.float().sum(1).clamp(min=1)
        out.append(weights.order_change * (changes & mine).float().sum(1) / n)
    return torch.stack(out, 1)
