"""The reward of a side (docs/en/training/training.md).

* The end of battle: +win for the winner, -win for the loser. At the time limit (both sides still
  have a unit standing) the defender wins (the simulator's rule): the attacker gets the same -win as
  for a lost fight (until 03.10 -1.5, "worse than losing"; audit R4: timeouts are ~0 in practice and
  the idle cost below already presses the attacker).
* Every decision, zero-sum shaping (side 2 gets minus side 1's), all terms potential differences
  (they add a steady signal and do not change which outcome is best):
    gold      x (enemy gold destroyed - own gold lost) / budget (gold_lost): the gold a unit has lost
                 is its multiplayer cost x the share of it lost - its share of health lost (for a unit
                 of many men the same as its men lost; a single entity counts before it dies); a unit
                 dead, gone off the map or shattered is lost whole; a routing unit that may still
                 rally loses `rout_share` of what it has left besides (out of the fight now, but it
                 may come back). A loss counts once: per unit the worst share lost so far (the
                 state's `lost_worst`, track()) - a rally gives nothing back, a rout after a rally
                 adds nothing until the unit loses more than at its worst (audit B3, 03.10); the
                 progress clock and the network's progress inputs count the same. budget: the mean
                 of the two armies' starting cost.
    lord      x (the enemy's lord died - own lord died): the game's morale rule makes a lord's death
                 decisive (-16, then -10 points to every unit of the side); the lord's gold alone
                 does not show it. With lord_rout > 0 a shattered lord counts as dead and a routing
                 one as lord_rout of a death, given back when it rallies (measure() column 4): in
                 the game a shattered lord breaks the army as a dead one does (gate 02.10, fix45b/m45:
                 3 of 4 battles ended within 3 s of a lord shattering; docs/en/training/training.md).
  A share is of the side at the start of battle.
* The attacker pays `idle` x m every decision while it makes no progress, whoever of its units is
  busy (progress only: units waiting - a reserve, a second line, the lord kept back - cost nothing
  while the army as a whole makes progress; audit R1, 03.10): the time limit's loss an hour away is
  discounted by gamma^7200 ~ 0.11 at 0.9997 and hardly seen; this cost is seen every step. The
  defender never pays it. m depends on the attacker's damage (idle_cost's last_hit):
    before its first damage  m = exp(t / idle_tau_s) - 1: almost nothing for the first minutes
                             (time to deploy and march), then sharply more;
    after it                 0 while it deals damage; once it has dealt none for idle_pause_s, k =
                             floor(seconds since its last damage / idle_pause_s) and m = exp(k x
                             idle_step) - 1: a step up every idle_pause_s, faster than the first
                             curve and without its grace; new damage sets it back to 0;
  m at most idle_cap. The defaults (2e-4, 150 s, 30 s, 0.5, 20) cost a battle with no damage 0.006
  by minute 1, 0.07 by 3, 0.26 by 5, 2.2 by 10, then 0.48 a minute (the cap from minute 7.6); a pause
  after damage 0 by 30 s, 0.03 by 90 s, 0.28 by 3 minutes (docs/en/training/training.md).
  Damage: with idle_rate > 0 only a damage rate (hit_rate: the defender's gold lost, share of the
  budget a minute, exponential mean over idle_window_s) of at least idle_rate counts (a scratch now
  and then no longer resets m: the skirmisher's loophole, 01.10 -> 02.10); idle_rate 0: any damage
  (struck). The network sees this clock (the rate, the threshold reached, seconds since):
  tools/nn/model/observation.py PROGRESS.
* Every real order change costs `order_change`, divided by the side's number of units: a new kind,
  a new attack target, or a move / withdraw point more than `order_move_m` from the one in force.
  KEEP and re-issuing the same order cost nothing. A unit that changes its order every decision
  (2 a second) costs its side ~1.2 over a 10-minute battle, more than a win; one change every 5 s
  ~0.12.
* Every switch of an attack target to another while the old one still stands costs `retarget` more
  (divided the same way): hysteresis against a unit dithering between two enemies (seen in the
  gate battles: a new target every second or two).
No style terms yet (the faction characters are placeholders).
"""
import math
from dataclasses import dataclass

import torch

from tools.nn.sim import orders as O


@dataclass(frozen=True)
class Weights:
    win: float = 1.0              # the end: +win the winner, -win the loser (the attacker at the time limit too)
    gold: float = 1.0             # (enemy gold destroyed - own gold lost) / budget
    rout_share: float = 0.5       # a routing unit that may rally: this share of what it has left is lost
    idle: float = 2e-4            # the attacker, per decision without progress, x m:
    idle_tau_s: float = 150.0     # ... before its first damage m = exp(t / idle_tau_s) - 1
    idle_pause_s: float = 30.0    # ... after it, k = floor(s since its last damage / idle_pause_s),
    idle_step: float = 0.5        # ... m = exp(k x idle_step) - 1
    idle_cap: float = 20.0        # ... m at most this
    idle_rate: float = 0.0        # 0: any damage resets the timer; > 0: only a damage rate (defender gold
    #                               lost, share of the budget a minute, over ~idle_window_s) of at least this
    idle_window_s: float = 30.0   # ... the time constant of that rate
    order_change: float = 0.001
    order_move_m: float = 10.0
    lord: float = 0.3             # the enemy lord's death - own lord's death
    lord_rout: float = 0.0        # > 0: in the lord term a shattered lord is dead, a routing one this share of it
    retarget: float = 0.003       # an attack switched to another target while the old one stands


def standing_mask(u):
    return (u["side"] > 0) & (u["men"] > 0) & ~u["gone"] & ~u["r"]


def lost_now(u, rout_share=Weights.rout_share):
    """[B, N] the share of each unit lost by its state now: the health lost; a unit dead, gone or
    shattered is lost whole; a routing one (it may rally) loses rout_share of what it has left
    besides. 0 for empty slots. No memory: a rally takes the rout share off (see gold_lost)."""
    hp_lost = (1 - u["hp_abs"] / u["hp0"].clamp(min=1e-6)).clamp(0, 1)
    out = (u["men"] <= 0) | u["gone"] | u["s"]
    share = torch.where(out, torch.ones_like(hp_lost), hp_lost + (1 - hp_lost) * rout_share * u["r"].float())
    return torch.where(u["side"] > 0, share, torch.zeros_like(share))


def track(u, rout_share=Weights.rout_share):
    """[B, N] the unit's worst share lost so far (state field `lost_worst`, 0 at the start of a battle):
    the larger of the one kept and lost_now. The caller stores it after every simulator step
    (rollout.Battles); the simulator itself does not know rout_share."""
    return torch.maximum(u["lost_worst"], lost_now(u, rout_share))


def gold_lost(u, rout_share=Weights.rout_share):
    """[B, N] the gold each unit has lost: its cost x the worst share of it lost so far (lost_now, and
    the state's `lost_worst` kept by track()). A loss counts once: a rally gives nothing back and a
    rout after a rally adds nothing until the unit loses more than at its worst (audit B3, 03.10: a
    defender unit routing, rallying and routing again was new damage each time - the attacker's
    progress clock reset and the trade paid with no real progress). 0 for empty slots."""
    share = lost_now(u, rout_share)
    if "lost_worst" in u:
        share = torch.maximum(share, u["lost_worst"])
    return u["cost"] * share


def budget(u):
    """[B] the mean of the two armies' starting cost (gold)."""
    return torch.stack([(u["cost"] * (u["side"] == s)).sum(1) for s in (1, 2)], 1).mean(1).clamp(min=1.0)


def gold_sides(u, rout_share=Weights.rout_share):
    """[B, 2] the gold each side (1, 2) has lost."""
    g = gold_lost(u, rout_share)
    return torch.stack([(g * (u["side"] == s)).sum(1) for s in (1, 2)], 1)


def measure(st, rout_share=Weights.rout_share, lord_rout=0.0):
    """[B, 2, 4]: per side (1, 2) the share of starting health left, the share of the army's cost
    still standing, 1 while its lord lives (or it has none), 0 once the lord is dead, and
    1 - its gold lost / budget. With lord_rout > 0, [B, 2, 5]: column 4 is 1 while the lord stands
    (or it has none), 1 - lord_rout while it routes, 0 once it is shattered or dead."""
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
        cols = [hp, (cost * stand).sum(1) / cost.sum(1).clamp(min=1e-6), lord, gold[:, s - 1]]
        if lord_rout:
            up = lords & alive & ~u["s"]
            each = torch.where(up, 1 - lord_rout * u["r"].float(), torch.zeros_like(mine))
            cols.append(torch.where(lords.any(1), (each * lords.float()).amax(1), torch.ones_like(hp)))
        out.append(torch.stack(cols, 1))
    return torch.stack(out, 1)


def health(st):
    """[B, 2] the share of each side's starting health still left (0-1)."""
    return measure(st)[:, :, 0]


PARTS = ("trade", "lord", "end", "idle", "orders")   # the side's reward by term (rollout logs them per role)


def parts(before, after, finished, winner, weights=Weights()):
    """{"trade", "lord", "end"}: [B, 2] the terms of step() for side 1 and side 2 (trade: gold; end:
    win / loss, the time limit's too: the defender wins it). Arguments as step()."""
    t1 = torch.zeros_like(before[:, 0, 0])
    l1 = torch.zeros_like(t1)
    if before.shape[-1] > 4:
        g = before[:, :, 4] - after[:, :, 4]                              # not clamped: a rally gives it back
        l1 = weights.lord * (g[:, 1] - g[:, 0])
    elif before.shape[-1] > 2:
        lost = (before[:, :, 2] - after[:, :, 2]).clamp(min=0)
        l1 = weights.lord * (lost[:, 1] - lost[:, 0])
    if before.shape[-1] > 3:
        g = before[:, :, 3] - after[:, :, 3]                              # the worst so far: never negative
        t1 = weights.gold * (g[:, 1] - g[:, 0])
    won1 = (winner == 1).float() - (winner == 2).float()
    end1 = torch.where(finished, weights.win * won1, torch.zeros_like(t1))
    return {"trade": torch.stack([t1, -t1], 1), "lord": torch.stack([l1, -l1], 1), "end": torch.stack([end1, -end1], 1)}


def step(before, after, finished, winner, weights=Weights()):
    """[B, 2] reward of side 1 and side 2 for one step, without the idle and order costs.

    before, after: measure() [B, 2, 4] around the step (the lord and gold columns optional); finished [B]:
    the battle ended in this step; winner [B] 1 or 2 (read where finished)."""
    p = parts(before, after, finished, winner, weights)
    return p["trade"] + p["lord"] + p["end"]


def idle_scale(t, last_hit, weights=Weights()):
    """[B] the attacker's idle multiplier m (see the module): t [B] the battle time, last_hit [B] the
    battle time of its last damage, negative before the first."""
    first = (t / weights.idle_tau_s).clamp(max=60.0)
    k = torch.floor((t - last_hit).clamp(min=0) / weights.idle_pause_s)
    later = (k * weights.idle_step).clamp(max=60.0)
    return (torch.where(last_hit < 0, first, later).exp() - 1).clamp(max=weights.idle_cap)


def damage_share(before, after, attacker):
    """[B] the share of the budget in gold the defender lost in the step (measure() column 3: each unit's
    worst so far, so a rally is not damage and a rout after a rally only beyond the unit's worst)."""
    d = (2 - attacker).long()
    g_b = before[:, :, 3].gather(1, d[:, None])[:, 0]
    g_a = after[:, :, 3].gather(1, d[:, None])[:, 0]
    return (g_b - g_a).clamp(min=0)


def hit_rate(rate, before, after, attacker, dt, window_s):
    """[B] the attacker's damage rate (the defender's gold lost, share of the budget a minute), an
    exponential mean over ~window_s: rate [B] the one before the step."""
    a = math.exp(-dt / window_s)
    return a * rate + (1 - a) * damage_share(before, after, attacker) * (60.0 / dt)


def struck(before, after, attacker):
    """[B] bool: the attacker dealt damage in the step - the defender's share of health left
    (measure() column 0) fell between before and after."""
    d = (2 - attacker).long()                                           # the defender's index (0, 1)
    hp_b = before[:, :, 0].gather(1, d[:, None])[:, 0]
    hp_a = after[:, :, 0].gather(1, d[:, None])[:, 0]
    return hp_a < hp_b


def idle_cost(st, weights=Weights(), last_hit=None):
    """[B, 2]: the attacker pays `idle` x idle_scale every decision of a running battle, whoever of its
    units is busy (progress only: the clock - last_hit [B], the battle time of its last damage, negative
    before the first; None: no damage yet in any battle - alone decides); the defender nothing."""
    if last_hit is None:
        last_hit = torch.full_like(st.t, -1.0)
    scale = idle_scale(st.t, last_hit, weights)
    return torch.stack([weights.idle * scale * ((st.attacker == s) & ~st.done).float() for s in (1, 2)], 1)


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
