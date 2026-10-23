"""A lone man (a lord) and routers in melee: every number and condition of these rules in one place
(docs/en/training/simulator.md "Lords"; config/nn/sim.json contact.lone_rout, the reasons in contact.lone_rout_why).

By the game's rules: the flank / rear penalty is decided for every blow by where the striking man stands against the
struck man's facing, for lone men as for everyone; a router runs with his back to his chasers and does not turn to
them. A unit chasing its routing attack target strikes with every blow at the whole charge bonus (the database's
pursuit_charge_bonus_modifier, no decay), at the usual interval. For formations the simulator keeps the measured pursuit
(contact.pursuit_rate, no charge: the recordings) - a known gap between the rule and the measure.

The block (absent or empty: the old rules - contact.lord_direction and contact.pursuit_rate for a routing lone man too,
no charge in a lone man's pursuit, a lone chaser striking across the pursuit's box gap):
  direction  the share of the database's flank / rear rule a ROUTING lone target takes (1: in full). A standing one
             keeps contact.lord_direction (the lord swarm probe measured no extra loss from his back or flank)
  share      the share of the rule a lone striker strikes a router at, in place of contact.pursuit_rate (1: in full)
  charge     1: a lone striker whose attack target routs strikes it with his whole charge bonus x the database's
             pursuit_charge_bonus_modifier (0: no charge)
  own_reach  1: a lone chaser strikes a lone router only within his own reach (the regular contact), not across the
             pursuit's box gap (contact.pursue_gap_m)
A lone man = a unit of one man (men0 <= 1); routing = the rout flag u["r"]. Called by tools/nn/sim/melee.py,
tools/nn/sim/battle.py and the fighters lab (tools/nn/fighters/melee.py). Needs torch."""
import torch


def rules(cc):
    """The contact section's lone_rout block, or None (the old rules)."""
    b = cc.get("lone_rout")
    return b if b else None


def _num(cc, key):
    b = rules(cc)
    v = None if b is None else b.get(key)
    return None if v is None else float(v)


def direction(cc, routing, dtype=torch.float32):
    """[shape of routing] the share of the database's flank / rear rule a lone target takes: standing,
    contact.lord_direction; routing, the block's direction (absent: lord_direction). routing: bool."""
    w = torch.full(routing.shape, float(cc.get("lord_direction", 1.0)), dtype=dtype, device=routing.device)
    d = _num(cc, "direction")
    if d is not None:
        w = torch.where(routing, torch.full_like(w, d), w)
    return w


def share(cc, pursuit_share, single_i):
    """The pursuit's share of the rule [B, N, N] (pursuit_share: the formations' - contact.pursuit_rate, the rout's
    exit) with a lone striker (single_i [B, N, 1]) at the block's share; the caller applies it to routers only."""
    s = _num(cc, "share")
    if s is None:
        return pursuit_share
    return torch.where(single_i.expand_as(pursuit_share), torch.full_like(pursuit_share, s), pursuit_share)


def charge(cc, rules_db):
    """The charge left a lone striker's blows on his routing attack target carry: the database's
    pursuit_charge_bonus_modifier with the block's charge on, else None (no charge)."""
    c = _num(cc, "charge")
    if not c:
        return None
    return float(rules_db.get("pursuit_charge_bonus_modifier", 1.0))


def chasing(single_i, order_kind_i, order_target_i, target, routing_j, attack_kind):
    """[broadcast] a lone striker whose attack order is on the struck unit (target: its slot), and that unit routs."""
    return single_i & (order_kind_i == attack_kind) & (order_target_i == target) & routing_j


def strike_cut(cc, men0, chase_touch, touch_geo):
    """[B, N, N] the pairs that do NOT strike: a lone man on a lone router he touches only by the pursuit's box gap
    (chase_touch and not the regular contact touch_geo), when the block's own_reach is on; else none."""
    b = rules(cc)
    if b is None or not b.get("own_reach"):
        return torch.zeros_like(chase_touch)
    lone = men0 <= 1
    return lone[:, :, None] & lone[:, None, :] & chase_touch & ~touch_geo
