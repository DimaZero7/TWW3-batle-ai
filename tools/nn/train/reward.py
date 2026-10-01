"""The reward of a side (docs/en/training/training.md).

* The end of battle: +win for the winner, -win for the loser. At the time limit (both sides still
  have a unit standing) the defender gets +win and the attacker -timeout: worse than losing a
  fight, so an attacker that only stands loses more than one that attacks and fails.
* Every decision, zero-sum shaping (side 2 gets minus side 1's), all terms potential differences
  (they add a steady signal and do not change which outcome is best):
    gold      x (enemy gold destroyed - own gold lost) / budget (gold_lost): the gold a unit has lost
                 is its multiplayer cost x the share of it lost - its share of health lost (for a unit
                 of many men the same as its men lost; a single entity counts before it dies); a unit
                 dead, gone off the map or shattered is lost whole; a routing unit that may still
                 rally loses `rout_share` of what it has left besides (out of the fight now, but it
                 may come back), given back when it rallies. budget: the mean of the two armies'
                 starting cost. It replaces `hp` and `standing` (weights 0 by default: with gold
                 they would count the same losses twice).
    hp        x (the share of the enemy's health lost - the share of own health lost) (0: in gold),
    standing  x (the share of the enemy's army, by cost, that stopped standing (routed, dead, gone)
                 - the same of own army) (0: in gold).
    lord      x (the enemy's lord died - own lord died): the game's morale rule makes a lord's death
                 decisive (-16, then -10 points to every unit of the side); the lord's gold alone
                 does not show it.
  A share is of the side at the start of battle.
* The attacker pays `idle` x m every decision in which none of its units fights in melee or shoots
  (marching is idling too): the time limit's -timeout an hour away is discounted by gamma^7200 ~
  0.11 at 0.9997 and hardly seen; this cost is seen every step. The defender never pays it. m
  depends on the attacker's damage (any HP the defender has lost, in melee or to missiles;
  idle_cost's last_hit):
    before its first damage  m = exp(t / idle_tau_s) - 1: almost nothing for the first minutes
                             (time to deploy and march), then sharply more;
    after it                 0 while it deals damage; once it has dealt none for idle_pause_s, k =
                             floor(seconds since its last damage / idle_pause_s) and m = exp(k x
                             idle_step) - 1: a step up every idle_pause_s, faster than the first
                             curve and without its grace; new damage sets it back to 0;
  m at most idle_cap. The defaults (2e-4, 150 s, 30 s, 0.5, 20) cost a battle with no damage 0.006
  by minute 1, 0.07 by 3, 0.26 by 5, 2.2 by 10 (more than the -timeout), then 0.48 a minute (the cap
  from minute 7.6); a pause after damage 0 by 30 s, 0.03 by 90 s, 0.28 by 3 minutes
  (docs/en/training/training.md).
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
    unit_gold     x n_own x (gold it destroyed - gold it lost) / budget: its share of the side's gold
                  trade, scaled to one unit; destroyed = the HP it dealt x the target's cost / the
                  target's starting HP, lost = the change of its gold_lost (routs and rallies too);
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
    gold: float = 1.0             # (enemy gold destroyed - own gold lost) / budget (= the old hp + standing)
    rout_share: float = 0.5       # a routing unit that may rally: this share of what it has left is lost
    hp: float = 0.0               # the old health trade (in gold now)
    standing: float = 0.0         # the old cost share that stopped standing (in gold now)
    idle: float = 2e-4            # the attacker, per decision with no unit fighting or shooting, x m:
    idle_tau_s: float = 150.0     # ... before its first damage m = exp(t / idle_tau_s) - 1
    idle_pause_s: float = 30.0    # ... after it, k = floor(s since its last damage / idle_pause_s),
    idle_step: float = 0.5        # ... m = exp(k x idle_step) - 1
    idle_cap: float = 20.0        # ... m at most this
    order_change: float = 0.001
    order_move_m: float = 10.0
    lord: float = 0.3             # the enemy lord's death - own lord's death
    retarget: float = 0.003       # an attack switched to another target while the old one stands
    # per unit (unit_step), not in the side's reward
    unit_gold: float = 0.05       # the unit's own gold trade, scaled to one unit
    flanked: float = 2e-4         # per decision struck in flank / rear
    missile_melee: float = 2e-4   # per decision a missile unit is in melee
    crowd: float = 2e-4           # per decision of a pile (excess share)
    idle_near: float = 0.0        # per decision a melee unit stands by while a fellow within 60 m fights
    flank_attack: float = 2e-4    # per decision striking an enemy's flank / rear (bonus; = flanked: zero-sum)
    neighbour: float = 0.5        # + this x the mean of own units' terms within neighbour_m
    neighbour_m: float = 40.0


def standing_mask(u):
    return (u["side"] > 0) & (u["men"] > 0) & ~u["gone"] & ~u["r"]


def gold_lost(u, rout_share=Weights.rout_share):
    """[B, N] the gold each unit has lost: its cost x the share of it lost. The share is the health
    lost; a unit dead, gone or shattered is lost whole; a routing one (it may rally) loses
    rout_share of what it has left besides. 0 for empty slots."""
    hp_lost = (1 - u["hp_abs"] / u["hp0"].clamp(min=1e-6)).clamp(0, 1)
    out = (u["men"] <= 0) | u["gone"] | u["s"]
    share = torch.where(out, torch.ones_like(hp_lost), hp_lost + (1 - hp_lost) * rout_share * u["r"].float())
    return torch.where(u["side"] > 0, u["cost"] * share, torch.zeros_like(share))


def budget(u):
    """[B] the mean of the two armies' starting cost (gold)."""
    return torch.stack([(u["cost"] * (u["side"] == s)).sum(1) for s in (1, 2)], 1).mean(1).clamp(min=1.0)


def gold_sides(u, rout_share=Weights.rout_share):
    """[B, 2] the gold each side (1, 2) has lost."""
    g = gold_lost(u, rout_share)
    return torch.stack([(g * (u["side"] == s)).sum(1) for s in (1, 2)], 1)


def measure(st, rout_share=Weights.rout_share):
    """[B, 2, 4]: per side (1, 2) the share of starting health left, the share of the army's cost
    still standing, 1 while its lord lives (or it has none), 0 once the lord is dead, and
    1 - its gold lost / budget."""
    u = st.u
    gold = 1 - gold_sides(u, rout_share) / budget(u)[:, None]
    stand = standing_mask(u).float()
    alive = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"]
    out = []
    for s in (1, 2):
        mine = (u["side"] == s).float()
        hp = (u["hp_abs"] * mine).sum(1) / (u["hp0"] * mine).sum(1).clamp(min=1e-6)
        cost = u["cost"].clamp(min=1.0) * mine
        lords = u["lord"] & (u["side"] == s)
        lord = (~lords.any(1) | (lords & alive).any(1)).float()
        out.append(torch.stack([hp, (cost * stand).sum(1) / cost.sum(1).clamp(min=1e-6), lord, gold[:, s - 1]], 1))
    return torch.stack(out, 1)


def health(st):
    """[B, 2] the share of each side's starting health still left (0-1)."""
    return measure(st)[:, :, 0]


def step(before, after, finished, winner, attacker, weights=Weights()):
    """[B, 2] reward of side 1 and side 2 for one step, without the idle and order costs.

    before, after: measure() [B, 2, 4] around the step (the lord and gold columns optional); finished [B]:
    the battle ended in this step; winner [B] 1 or 2 (read where finished); attacker [B] 1 or 2."""
    lost = (before - after).clamp(min=0)                                  # [B, side, (hp, standing, lord, gold)]
    r1 = weights.hp * (lost[:, 1, 0] - lost[:, 0, 0]) + weights.standing * (lost[:, 1, 1] - lost[:, 0, 1])
    if lost.shape[-1] > 2:
        r1 = r1 + weights.lord * (lost[:, 1, 2] - lost[:, 0, 2])
    if lost.shape[-1] > 3:
        g = before[:, :, 3] - after[:, :, 3]                              # not clamped: a rally gives its gold back
        r1 = r1 + weights.gold * (g[:, 1] - g[:, 0])
    timeout = finished & (after[:, 0, 1] > 0) & (after[:, 1, 1] > 0)
    won1 = (winner == 1).float() - (winner == 2).float()
    end = torch.where(finished, weights.win * won1, torch.zeros_like(r1))
    r = torch.stack([r1 + end, -r1 - end], 1)
    # At the limit the attacker loses more than a fight: -timeout instead of -win.
    extra = torch.where(timeout, torch.full_like(r1, weights.win - weights.timeout), torch.zeros_like(r1))
    side = torch.stack([attacker == 1, attacker == 2], 1).float()
    return r + side * extra[:, None]


def idle_scale(t, last_hit, weights=Weights()):
    """[B] the attacker's idle multiplier m (see the module): t [B] the battle time, last_hit [B] the
    battle time of its last damage, negative before the first."""
    first = (t / weights.idle_tau_s).clamp(max=60.0)
    k = torch.floor((t - last_hit).clamp(min=0) / weights.idle_pause_s)
    later = (k * weights.idle_step).clamp(max=60.0)
    return (torch.where(last_hit < 0, first, later).exp() - 1).clamp(max=weights.idle_cap)


def struck(before, after, attacker):
    """[B] bool: the attacker dealt damage in the step - the defender's share of health left
    (measure() column 0) fell between before and after."""
    d = (2 - attacker).long()                                           # the defender's index (0, 1)
    hp_b = before[:, :, 0].gather(1, d[:, None])[:, 0]
    hp_a = after[:, :, 0].gather(1, d[:, None])[:, 0]
    return hp_a < hp_b


def idle_cost(st, weights=Weights(), last_hit=None):
    """[B, 2]: the attacker pays `idle` x idle_scale when none of its units is in melee or shooting
    (last_hit [B]: the battle time of its last damage, negative before the first; None: no damage
    yet in any battle); the defender nothing."""
    u = st.u
    busy = (u["m"] | u["fire"]) & standing_mask(u)
    if last_hit is None:
        last_hit = torch.full_like(st.t, -1.0)
    scale = idle_scale(st.t, last_hit, weights)
    out = []
    for s in (1, 2):
        mine = (st.attacker == s) & ~st.done
        idle = ~(busy & (u["side"] == s)).any(1) & mine
        out.append(weights.idle * scale * idle.float())
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


def unit_before(u, rout_share=Weights.rout_share):
    """What unit_step needs from before the step."""
    return dict({k: u[k].clone() for k in UNIT_FIELDS}, gold=gold_lost(u, rout_share))


def unit_step(before, st, facts, params, weights=Weights()):
    """[B, N] each unit's own reward for the step (0 for empty slots): its gold trade and the
    shaped terms of behaviour.facts (after the step), then its neighbours' mean (see the module)."""
    u = st.u
    side = u["side"]
    present = side > 0
    lost = gold_lost(u, weights.rout_share) - before["gold"]                # gold; a rally gives it back
    fade = math.exp(-params.dt / params.sim["morale"]["recent_s"])
    melee = (u["dealt"] - fade * before["dealt"]).clamp(min=0)            # melee HP dealt this step
    tgt = u["target"].clamp(min=0)
    shot = torch.where(u["fire"] & (u["target"] >= 0), (u["k"] - before["k"]).clamp(min=0) * u["hp_man"].gather(1, tgt),
                       torch.zeros_like(melee))
    dealt = torch.where(u["m"], melee, shot)                              # HP
    worth = (u["cost"] / u["hp0"].clamp(min=1e-6)).gather(1, tgt)          # the target's gold a HP
    dealt = torch.where(u["target"] >= 0, dealt * worth, torch.zeros_like(dealt))
    n = torch.stack([(side == s).sum(1) for s in (1, 2)], 1).float()
    own_i = (side - 1).clamp(min=0)
    trade = n.gather(1, own_i) * (dealt - lost) / budget(u)[:, None]
    r = (weights.unit_gold * trade - weights.flanked * facts["flanked"].float()
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
