"""Morale (docs/en/training/simulator.md, docs/en/game/units/morale.md).

Morale is points: leadership + a start bonus + the effects of the moment (the game's database,
_kv_morale_tables). MoralePercent = points / leadership. Each 0.5 s tick the points move towards
that sum by max(1, 15 % of the gap) (minimium_increment_update_per_tick, percent_update_per_tick).

Effects (points): lord within 70 m +4; lord died recently -16, dead -10; neighbour within 120 m
(flanks secure) +5; casualties over the battle (share of HP) -2 ... -74; recent casualties
-6 ... -80; winning / losing the melee +3/+6/+8, -3/-8; attacked in the flank / rear -6 / -14;
routing friends within 100 m -3 each (at most 4; expendable units scare nobody); routing
enemies within 100 m +2.5 each (at most 5); under fire -5; very tired -2, exhausted -6;
a stronger enemy within 70 m -3.

States: wavering below 16 points; routing at 0 or below (not within 10 s of a rally); the third
rout shatters. A routing unit regains rally_rate points a second while no standing enemy is
within rally_free_m and rallies at MoralePercent rally_mp; then its morale follows the effects
again (a unit that lost much soon routs again, as in the game).
"""
import torch

def table(rules, prefix, shares):
    """[(share, points)] from _kv_morale_tables keys like total_casualties_penalty_10."""
    return ((0.0, 0.0),) + tuple((s / 100, rules[f"{prefix}{s}"]) for s in shares)


def steps(x, points):
    """The penalty of the highest threshold reached: (share, points) pairs, 0 below the first."""
    out = torch.zeros_like(x)
    for share, pts in points[1:]:
        out = torch.where(x >= share, torch.full_like(x, pts), out)
    return out


def combat_points(dealt, taken, in_melee, cal, rules):
    """Winning or losing the melee, by HP dealt / taken recently."""
    ratio = (dealt + 1.0) / (taken + 1.0)
    c = cal["combat_ratio"]
    pts = torch.zeros_like(ratio)
    pts = torch.where(ratio >= c["slightly"], rules["winning_combat_slightly"], pts)
    pts = torch.where(ratio >= c["yes"], rules["winning_combat"], pts)
    pts = torch.where(ratio >= c["significantly"], rules["winning_combat_significantly"], pts)
    pts = torch.where(ratio <= 1 / c["slightly"], rules["losing_combat"], pts)
    pts = torch.where(ratio <= 1 / c["significantly"], rules["losing_combat_significantly"], pts)
    return torch.where(in_melee, pts, torch.zeros_like(pts))


def target_points(u, ctx, params):
    """The morale points the unit is heading to [B, N] (leadership + bonus + effects)."""
    R = params.morale
    cal = params.sim["morale"]
    L = u["leadership"]
    pts = L + u["morale_bonus"]
    pts = pts + torch.where(ctx["aura"], R["general_inspire_effect_amount_max"], 0.0)
    pts = pts + ctx["lord_dead_points"]
    pts = pts + torch.where(ctx["neighbour"], R["ume_encouraged_flanks_secure"], 0.0)
    lost = 1 - u["hp_abs"] / u["hp0"].clamp(min=1e-6)
    pts = pts + steps(lost, table(R, "total_casualties_penalty_", (10, 20, 30, 40, 50, 60, 70, 80, 90)))
    base = (u["hp_abs"] + u["recent"]) if cal["recent_of_current"] else u["hp0"]
    recent = u["recent"] / base.clamp(min=1e-6)
    pts = pts + steps(recent, table(R, "recent_casualties_penalty_", (6, 10, 15, 33, 50)))
    pts = pts + combat_points(u["dealt"], u["taken"], ctx["in_melee"], cal, R)
    flank = torch.where(u["flank_hit"] >= 2, R["was_attacked_in_rear"],
                        torch.where(u["flank_hit"] >= 1, R["was_attacked_in_flank"], 0.0))
    pts = pts + flank
    pts = pts - R["routing_friends_effect_weighting"] * ctx["routing_friends"].clamp(max=R["max_routing_friends_to_consider"])
    pts = pts + R["routing_enemies_effect_weighting"] * ctx["routing_enemies"].clamp(max=R["max_routing_enemies_to_consider"])
    pts = pts + torch.where(ctx["under_fire"], R["ume_concerned_attacked_by_projectile"], 0.0)
    pts = pts + torch.where(u["fat"] >= 5, R["ume_concerned_exhausted"],
                            torch.where(u["fat"] >= 4, R["ume_concerned_very_tired"], 0.0))
    pts = pts + torch.where(ctx["strong_enemy"], R["enemy_morale_penalty_value_min"], 0.0)
    return pts


def step(u, ctx, params, dt):
    """Move morale points, set wavering / routing / shattered and rally. Changes u in place.
    Returns the units that began to rout this step."""
    R = params.morale
    cal = params.sim["morale"]
    L = u["leadership"].clamp(min=1)
    alive = (u["men"] > 0) & ~u["gone"] & (u["side"] > 0)
    target = target_points(u, ctx, params)
    M = u["morale"]
    ticks = dt / 0.5
    gap = target - M
    move = torch.maximum(torch.full_like(gap, R["minimium_increment_update_per_tick"]),
                         R["percent_update_per_tick"] * gap.abs()) * ticks
    stepped = M + torch.clamp(gap, -move, move)
    routing = u["r"]
    # Routing: morale does not follow the fight; it recovers once no standing enemy is near.
    free = ~ctx["enemy_near"]
    routed_M = torch.where(free, M + cal["rally_rate"] * dt, torch.minimum(M, stepped))
    M = torch.where(routing, routed_M, stepped)
    u["rout_s"] = torch.where(routing, u["rout_s"] + dt, torch.zeros_like(u["rout_s"]))
    u["rally_s"] = u["rally_s"] + dt

    # Rally.
    rally = routing & ~u["s"] & free & (M >= cal["rally_mp"] * L) & alive
    u["r"] = u["r"] & ~rally
    u["rally_s"] = torch.where(rally, torch.zeros_like(u["rally_s"]), u["rally_s"])

    # Rout.
    no_rout = u["rally_s"] < R["post_rally_no_rout_timer"]
    start = alive & ~u["r"] & (M <= 0) & ~(no_rout & (u["rout_count"] > 0))
    u["r"] = u["r"] | start
    u["rout_count"] = u["rout_count"] + start.float()
    u["s"] = u["s"] | (start & (u["rout_count"] >= R["shatter_after_rout_count"]))
    M = torch.where(start, torch.minimum(M, cal["rout_floor_mp"] * L), M)
    u["morale"] = torch.where(alive, M, u["morale"])
    u["w"] = alive & ~u["r"] & (M < R["ums_wavering_threshold_upper"])
    mp = M / L
    u["mp"] = mp
    ms = torch.where(mp >= 1.0, 1, torch.where(mp >= R["ums_confident_threshold_lower"], 2, 3))
    ms = torch.where(M < R["ums_steady_threshold_lower"], 4, ms)
    ms = torch.where(u["w"], 5, ms)
    ms = torch.where(u["r"], 6, ms)
    ms = torch.where(u["s"], 7, ms)
    present = u["side"] > 0
    u["mp"] = torch.where(present, mp, torch.zeros_like(mp))
    u["ms"] = torch.where(present, ms.float(), torch.zeros_like(mp))
    return start
