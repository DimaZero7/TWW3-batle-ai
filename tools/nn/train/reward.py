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
                 does not show it. With lord_rout > 0 a shattered lord counts as dead and a routing
                 one as lord_rout of a death, given back when it rallies (measure() column 4): in
                 the game a shattered lord breaks the army as a dead one does (gate 02.10, fix45b/m45:
                 3 of 4 battles ended within 3 s of a lord shattering; docs/en/training/training.md).
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
  Two options close the skirmisher's loophole (01.10 -> 02.10: one unit shooting or fighting kept the
  whole army free of the cost and every scratch reset m, so attackers stood; docs/en/training/training.md):
  idle_share 1 pays x the share of its standing army, by cost, that neither fights nor shoots (instead
  of "no unit busy"), and idle_rate > 0 resets m only while the attacker's damage rate (hit_rate: the
  defender's gold lost, share of the budget a minute, exponential mean over idle_window_s) is at least
  idle_rate, instead of on any damage. Both 0: the old rule.
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
    unit_idle     x the attacker's idle multiplier m (idle_scale, as the side's idle cost) per decision a
                  unit of the attacking side stands without fighting or shooting: the side's idle cost
                  laid on the units that cause it (02.10: a side-wide cost could not tell the holding units
                  from the fighting ones, and the policy stayed in "hold");
    lord_exposed  per decision a lord fights in melee with less than lord_exposed_hp of its health
                  (gate 02.10: the network's lord led the line and went back into melee at 19 % health
                  until it shattered);
    lord_lead     x 0-1 per decision a lord stands more than lord_lead_m ahead of the centre of its
                  side's other standing units (towards the enemy's centre) while an enemy is within
                  lord_lead_near (lord_lead()): full at twice lord_lead_m. Gates 02.10 (fix45b, it1): the
                  game's AI keeps its lord at or behind its line's centre (-3 to -12 m), ours 10-45 m
                  ahead, and its shattering broke the army; in the simulator ai_like's lord leads as far
                  as ours (34 m), so the simulator alone does not teach it;
    lord_fall     x the lord's fall in the step (lord_up: 1 standing, 1 - lord_rout (0.5 when lord_rout
                  is 0) routing, 0 shattered, dead or gone; a rally gives it back): the army-wide price of
                  its rout laid on the lord's own credit. Gate 02.10 (it2/m15, 8 battles): our lord routed
                  or shattered in 4 of the 5 losses and in none of the 3 wins, after 38-109 s in melee
                  below half health; its own credit saw only its gold (~0.06 for a whole lord at 19 units)
                  against ~0.2 for the gold it kills, so fighting on paid;
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
    idle_share: float = 0.0       # 0: idle while no unit fights or shoots (one busy unit stops the cost);
    #                               1: x the share of its standing army, by cost, not fighting or shooting
    idle_rate: float = 0.0        # 0: any damage resets the timer; > 0: only a damage rate (defender gold
    #                               lost, share of the budget a minute, over ~idle_window_s) of at least this
    idle_window_s: float = 30.0   # ... the time constant of that rate
    order_change: float = 0.001
    order_move_m: float = 10.0
    lord: float = 0.3             # the enemy lord's death - own lord's death
    lord_rout: float = 0.0        # > 0: in the lord term a shattered lord is dead, a routing one this share of it
    retarget: float = 0.003       # an attack switched to another target while the old one stands
    # per unit (unit_step), not in the side's reward
    unit_gold: float = 0.05       # the unit's own gold trade, scaled to one unit
    flanked: float = 2e-4         # per decision struck in flank / rear
    missile_melee: float = 2e-4   # per decision a missile unit is in melee
    crowd: float = 2e-4           # per decision of a pile (excess share)
    idle_near: float = 0.0        # per decision a melee unit stands by while a fellow within 60 m fights
    flank_attack: float = 2e-4    # per decision striking an enemy's flank / rear (bonus; = flanked: zero-sum)
    unit_idle: float = 0.0        # x the attacker's idle m, per decision an attacking unit neither fights nor shoots
    lord_exposed: float = 0.0     # per decision a lord fights in melee below lord_exposed_hp of its health
    lord_exposed_hp: float = 0.5
    lord_lead: float = 0.0        # per decision a lord stands ahead of its own line near the enemy (x 0-1, lord_lead())
    lord_lead_m: float = 10.0     # ... the lead it may have free; full weight at twice this
    lord_lead_near: float = 100.0  # ... only while a standing enemy is within this of it
    lord_fall: float = 0.0        # per unit: the lord pays this x its fall (lord_up() before - after; a rally gives back)
    neighbour: float = 0.5       # + this x the mean of own units' terms within neighbour_m
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


def parts(before, after, finished, winner, attacker, weights=Weights()):
    """{"trade", "lord", "end"}: [B, 2] the terms of step() for side 1 and side 2 (trade: gold + hp +
    standing; end: win / loss / the attacker's timeout). Arguments as step()."""
    lost = (before - after).clamp(min=0)                                  # [B, side, (hp, standing, lord, gold)]
    t1 = weights.hp * (lost[:, 1, 0] - lost[:, 0, 0]) + weights.standing * (lost[:, 1, 1] - lost[:, 0, 1])
    l1 = torch.zeros_like(t1)
    if lost.shape[-1] > 4:
        g = before[:, :, 4] - after[:, :, 4]                              # not clamped: a rally gives it back
        l1 = weights.lord * (g[:, 1] - g[:, 0])
    elif lost.shape[-1] > 2:
        l1 = weights.lord * (lost[:, 1, 2] - lost[:, 0, 2])
    if lost.shape[-1] > 3:
        g = before[:, :, 3] - after[:, :, 3]                              # not clamped: a rally gives its gold back
        t1 = t1 + weights.gold * (g[:, 1] - g[:, 0])
    timeout = finished & (after[:, 0, 1] > 0) & (after[:, 1, 1] > 0)
    won1 = (winner == 1).float() - (winner == 2).float()
    end1 = torch.where(finished, weights.win * won1, torch.zeros_like(t1))
    # At the limit the attacker loses more than a fight: -timeout instead of -win.
    extra = torch.where(timeout, torch.full_like(t1, weights.win - weights.timeout), torch.zeros_like(t1))
    side = torch.stack([attacker == 1, attacker == 2], 1).float()
    end = torch.stack([end1, -end1], 1) + side * extra[:, None]
    return {"trade": torch.stack([t1, -t1], 1), "lord": torch.stack([l1, -l1], 1), "end": end}


def step(before, after, finished, winner, attacker, weights=Weights()):
    """[B, 2] reward of side 1 and side 2 for one step, without the idle and order costs.

    before, after: measure() [B, 2, 4] around the step (the lord and gold columns optional); finished [B]:
    the battle ended in this step; winner [B] 1 or 2 (read where finished); attacker [B] 1 or 2."""
    p = parts(before, after, finished, winner, attacker, weights)
    return p["trade"] + p["lord"] + p["end"]


def idle_scale(t, last_hit, weights=Weights()):
    """[B] the attacker's idle multiplier m (see the module): t [B] the battle time, last_hit [B] the
    battle time of its last damage, negative before the first."""
    first = (t / weights.idle_tau_s).clamp(max=60.0)
    k = torch.floor((t - last_hit).clamp(min=0) / weights.idle_pause_s)
    later = (k * weights.idle_step).clamp(max=60.0)
    return (torch.where(last_hit < 0, first, later).exp() - 1).clamp(max=weights.idle_cap)


def damage_share(before, after, attacker):
    """[B] the share of the budget in gold the defender lost in the step (measure() column 3; a rally
    is not damage: 0)."""
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
    """[B, 2]: the attacker pays `idle` x idle_scale when none of its units is in melee or shooting
    (with idle_share: x the share of its standing army, by cost, that is not) (last_hit [B]: the
    battle time of its last damage, negative before the first; None: no damage yet in any battle);
    the defender nothing."""
    u = st.u
    stand = standing_mask(u)
    busy = (u["m"] | u["fire"]) & stand
    if last_hit is None:
        last_hit = torch.full_like(st.t, -1.0)
    scale = idle_scale(st.t, last_hit, weights)
    cost = u["cost"].clamp(min=1.0)
    out = []
    for s in (1, 2):
        side = u["side"] == s
        mine = (st.attacker == s) & ~st.done
        idle = (~(busy & side).any(1)).float()
        if weights.idle_share:
            # the share of its standing army (by cost) that neither fights nor shoots: one unit
            # skirmishing while the rest stand no longer stops the cost
            standing = (cost * (stand & side)).sum(1)
            share = 1 - (cost * (busy & side)).sum(1) / standing.clamp(min=1e-6)
            share = torch.where(standing > 0, share, torch.ones_like(share))
            idle = (1 - weights.idle_share) * idle + weights.idle_share * share
        out.append(weights.idle * scale * idle * mine.float())
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


def lord_lead(u, margin_m=10.0, near_m=100.0):
    """[B, N] 0-1 per standing lord: how far it stands ahead of its own line, (lead - margin_m) /
    margin_m clipped to 0-1, while a standing enemy is within near_m of it; 0 for every other unit.
    lead: the lord's distance ahead of the cost-weighted centre of its side's other standing units,
    along the line from that centre to the centre of the standing enemies. A lord with no other
    standing unit has no line (0)."""
    stand = standing_mask(u)
    side = u["side"]
    cost = u["cost"].clamp(min=1.0)
    out = torch.zeros_like(u["x"])
    for s in (1, 2):
        own = stand & (side == s)
        rest = (own & ~u["lord"]).float() * cost
        foe = (stand & (side > 0) & (side != s)).float() * cost
        n_rest, n_foe = rest.sum(1, keepdim=True), foe.sum(1, keepdim=True)
        cx = (u["x"] * rest).sum(1, keepdim=True) / n_rest.clamp(min=1e-6)
        cz = (u["z"] * rest).sum(1, keepdim=True) / n_rest.clamp(min=1e-6)
        ex = (u["x"] * foe).sum(1, keepdim=True) / n_foe.clamp(min=1e-6)
        ez = (u["z"] * foe).sum(1, keepdim=True) / n_foe.clamp(min=1e-6)
        norm = torch.sqrt((ex - cx) ** 2 + (ez - cz) ** 2).clamp(min=1e-6)
        lead = ((u["x"] - cx) * (ex - cx) + (u["z"] - cz) * (ez - cz)) / norm
        dx = u["x"][:, :, None] - u["x"][:, None, :]
        dz = u["z"][:, :, None] - u["z"][:, None, :]
        near = ((dx * dx + dz * dz <= near_m ** 2) & (stand & (side > 0) & (side != s))[:, None, :]).any(2)
        share = ((lead - margin_m) / max(margin_m, 1e-6)).clamp(0, 1)
        lords = own & u["lord"] & near & (n_rest > 0) & (n_foe > 0)
        out = torch.where(lords, share, out)
    return out


UNIT_FIELDS = ("hp_abs", "k", "dealt")


def lord_up(r, out, share):
    """[B, N] 1 for a standing unit, 1 - share for a routing one, 0 for one shattered, dead or gone
    (r: routing, out: men <= 0 | gone | shattered)."""
    return torch.where(out, torch.zeros_like(r, dtype=torch.float32), 1 - share * r.float())


def unit_out(u):
    return (u["men"] <= 0) | u["gone"] | u["s"]


def unit_before(u, rout_share=Weights.rout_share):
    """What unit_step needs from before the step."""
    return dict({k: u[k].clone() for k in UNIT_FIELDS}, gold=gold_lost(u, rout_share), r=u["r"].clone(),
                out=unit_out(u))


def unit_step(before, st, facts, params, weights=Weights(), idle_m=None):
    """[B, N] each unit's own reward for the step (0 for empty slots): its gold trade and the
    shaped terms of behaviour.facts (after the step), then its neighbours' mean (see the module).
    idle_m [B]: the attacker's idle multiplier (idle_scale) for unit_idle (None: no such term)."""
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
    if weights.unit_idle and idle_m is not None:
        stand = standing_mask(u)
        idle = stand & ~(u["m"] | u["fire"]) & (side == st.attacker[:, None]) & ~st.done[:, None]
        r = r - weights.unit_idle * idle_m[:, None] * idle.float()
    if weights.lord_exposed:
        low = u["hp_abs"] < weights.lord_exposed_hp * u["hp0"]
        r = r - weights.lord_exposed * (u["lord"] & standing_mask(u) & u["m"] & low).float()
    if weights.lord_lead:
        r = r - weights.lord_lead * lord_lead(u, weights.lord_lead_m, weights.lord_lead_near)
    if weights.lord_fall and "out" in before:
        share = weights.lord_rout or 0.5
        fall = lord_up(before["r"], before["out"], share) - lord_up(u["r"], unit_out(u), share)
        r = r - weights.lord_fall * u["lord"].float() * fall
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
