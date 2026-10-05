"""Morale (docs/en/training/simulator.md, docs/en/game/units/morale.md).

Morale is points: leadership + a start bonus + the effects of the moment (the game's database,
_kv_morale_tables). MoralePercent = points / leadership. Each 0.5 s tick the points move towards
that sum by max(1, 15 % of the gap) (minimium_increment_update_per_tick, percent_update_per_tick).

Effects (points): lord within 70 m +4 (fading to 0 at 105 m); the lord killed -16 for 45 s, then -10; routed
off the map -16 for 120 s (routed on the field: his aura only; battle.py, sim.json morale.lord_fall); neighbour within 120 m
(flanks secure) +5; casualties over the battle (share of HP) -2 ... -74; recent casualties
-6 ... -80 (HP lost in the last 4 s); extended casualties -4 ... -60 (the last 60 s); winning / losing the melee +3/+6/+8,
-3/-8; attacked in the flank / rear -6 / -14 for the tick of the first contact from that side; flanks exposed (an enemy threatens the left, right or rear:
lf / rf / bf) -3, several -6;
routing friends within 100 m -3 each (at most 4; a routing expendable unit scares only expendable units); routing
enemies within 100 m +2.5 each (at most 5); under fire -5 until 15 s after the last projectile hit; very tired -2,
exhausted -6; a stronger enemy within 70 m: the database's -3, 0 here (not seen in the recordings,
morale.strong_enemy_points); the army beaten as a whole -120 (army destruction).

States: wavering below 16 points; routing at 0 or below (not within 10 s of a rally); the third
rout shatters, as does falling below the database's broken band (-50 points) during army destruction. A routing unit
regains rally_rate points a second while no standing enemy is within rally_free_m and its army
is not collapsing, and rallies at MoralePercent rally_mp; then its morale follows the effects
again (a unit that lost much soon routs again, as in the game).
"""
import math

import torch


def army_collapse(u, params):
    """Army destruction [B, N], from current combat strength and the database thresholds.

    Strategic strength approximates recorded unit:strategic_value(): HP times unit value,
    with the missile part falling sigmoidally as ammunition runs out. Balance of power counts
    routers at half strength; shattered, dead and departed units contribute nothing.
    No recording clock or winner is consulted.
    """
    cal = params.sim["morale"].get("collapse") or {}
    present = u["side"] > 0
    if not cal.get("on"):
        return torch.zeros_like(present)
    alive = present & (u["men"] > 0) & ~u["gone"] & ~u["s"]
    count = alive & ~u["r"] if cal.get("count") == "standing" else alive
    value = u["cost"]
    weight = torch.ones_like(value)
    if cal.get("strength") == "strategic":
        value = value * torch.where(u["lord"], float(cal["lord_value_scale"]), 1.0)
        ammo = (u["a"] / u["ammo0"].clamp(min=1)).clamp(0, 1)
        slope, midpoint = float(cal["ammo_slope"]), float(cal["ammo_midpoint"])
        lo = 1 / (1 + math.exp(slope * midpoint))
        hi = 1 / (1 + math.exp(-slope * (1 - midpoint)))
        available = (torch.sigmoid(slope * (ammo - midpoint)) - lo) / (hi - lo)
        floor = torch.where(u["direct"], float(cal["ammo_empty_direct"]), float(cal["ammo_empty"]))
        weight = torch.where(u["ammo0"] > 0, floor + (1 - floor) * available, weight)
        weight = weight * torch.where(u["r"], float(cal["routing_weight"]), 1.0)
    power = value * u["hp"].clamp(0, 1).pow(float(cal.get("hp_power", 1.0))) * weight * count
    strength = torch.stack([(power * (u["side"] == s)).sum(1) for s in (1, 2)], 1)
    initial = torch.stack([(value * (u["side"] == s)).sum(1) for s in (1, 2)], 1)
    rules = params.morale
    beaten = ((initial > 0) & (strength.flip(1) > 0)
              & (strength.flip(1) >= rules["army_destruction_enemy_strength_ratio"] * strength)
              & (strength <= rules["army_destruction_alliance_strength_ratio"] * initial))
    return beaten.gather(1, (u["side"] - 1).clamp(min=0)) & present


def window_steps(seconds, dt):
    """Steps in a sliding window of `seconds` (at least one)."""
    return max(1, int(round(float(seconds) / dt)))


def history_steps(params, dt):
    """Steps of HP lost the state keeps (u["lost_hist"]) for the sliding casualty windows: the longer of
    morale.casualties_s and extended_s; 0 when morale.casualties_window is not "sliding" (decaying sums)."""
    cm = params.sim["morale"]
    if cm.get("casualties_window") != "sliding":
        return 0
    return max(window_steps(cm["casualties_s"], dt), window_steps(cm.get("extended_s") or 0.0, dt))


def casualty_windows(u, taken, params, dt):
    """HP lost recently (u["recent"]: recent casualties, morale.casualties_s) and over the extended window
    (u["extended"]: extended casualties, morale.extended_s; 0: off), with this step's losses `taken` [B, N]
    (in place). casualties_window "sliding": exactly the HP lost in the last casualties_s / extended_s seconds
    (u["lost_hist"]: the HP lost each step, newest first; the database's 'last 4 s' / 'last 60 s', measured: a
    volley holds the game's PercentHpLostRecently 3 one-second samples, then 0); otherwise decaying sums with
    those time constants (the old fit)."""
    cm = params.sim["morale"]
    ext_s = float(cm.get("extended_s") or 0.0)
    K = history_steps(params, dt)
    if K:
        B, N = taken.shape
        hist = torch.cat([taken[:, None, :], u["lost_hist"].reshape(B, K, N)[:, :K - 1]], 1)
        u["lost_hist"] = hist.reshape(B, K * N)
        u["recent"] = hist[:, :window_steps(cm["casualties_s"], dt)].sum(1)
        if ext_s > 0:
            u["extended"] = hist[:, :window_steps(ext_s, dt)].sum(1)
        return
    u["recent"] = u["recent"] * math.exp(-dt / float(cm.get("casualties_s", cm["recent_s"]))) + taken
    if ext_s > 0:
        u["extended"] = u["extended"] * math.exp(-dt / ext_s) + taken


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
    # The lord's aura: full within general_aura_radius, fading to 0 at x inspiration_radius_max_effect_range_modifier
    # (battle.py gives the share 0-1).
    pts = pts + ctx["aura"].float() * R["general_inspire_effect_amount_max"]
    pts = pts + ctx["lord_dead_points"]
    pts = pts + torch.where(ctx["neighbour"], R["ume_encouraged_flanks_secure"], 0.0)
    lost = 1 - u["hp_abs"] / u["hp0"].clamp(min=1e-6)
    pts = pts + steps(lost, table(R, "total_casualties_penalty_", (10, 20, 30, 40, 50, 60, 70, 80, 90)))
    base = (u["hp_abs"] + u["recent"]) if cal["recent_of_current"] else u["hp0"]
    recent = u["recent"] / base.clamp(min=1e-6)
    pts = pts + steps(recent, table(R, "recent_casualties_penalty_", (6, 10, 15, 33, 50)))
    if cal.get("extended_s"):
        ext = u["extended"] / u["hp0"].clamp(min=1e-6)
        pts = pts + steps(ext, table(R, "extended_casualties_penalty_", (10, 15, 33, 50, 80)))
    pts = pts + combat_points(u["dealt"], u["taken"], ctx["in_melee"], cal, R)
    # Attacked in the flank / rear: measured points (the database's -6 / -14 is not what a unit
    # fighting on its flank or rear shows in the recordings).
    if cal.get("attacked_event"):
        # The database's was_attacked_in_flank / _rear (-6 / -14) at the first contact from that side:
        # battle.py gives the points for the step a worse side is first struck (measured: a unit
        # first struck in the flank / rear drops 1.5 / 1.9 points more in 1-2 s than one struck in
        # front, which is one 0.5 s tick of -6 / -14).
        pts = pts + ctx.get("flank_event", 0.0)
    else:
        flank = torch.where(u["flank_hit"] >= 2, cal.get("attacked_rear", R["was_attacked_in_rear"]),
                            torch.where(u["flank_hit"] >= 1, cal.get("attacked_flank", R["was_attacked_in_flank"]), 0.0))
        pts = pts + flank
    # Flanks exposed: the same lf/rf/bf observed by the network, one (-3) or several (-6).
    # These proximity threats are separate from flank_hit / flank_event above: an
    # actual incoming blow keeps its contact sector even if threat geometry is calibrated.
    exposed = u["lf"].float() + u["rf"].float() + u["bf"].float()
    pts = pts + torch.where(exposed >= 2, R["ume_concerned_flanks_exposed_multiple"],
                            torch.where(exposed >= 1, R["ume_concerned_flanks_exposed_single"], 0.0))
    pts = pts - R["routing_friends_effect_weighting"] * ctx["routing_friends"].clamp(max=R["max_routing_friends_to_consider"])
    pts = pts + R["routing_enemies_effect_weighting"] * ctx["routing_enemies"].clamp(max=R["max_routing_enemies_to_consider"])
    pts = pts + torch.where(ctx["under_fire"], R["ume_concerned_attacked_by_projectile"], 0.0)
    pts = pts + torch.where(u["fat"] >= 5, R["ume_concerned_exhausted"],
                            torch.where(u["fat"] >= 4, R["ume_concerned_very_tired"], 0.0))
    # A stronger enemy within 70 m: the database's enemy_morale_penalty_value_min (-3) unless
    # morale.strong_enemy_points says otherwise (measured 0: config/nn/sim.json morale.strong_enemy_why).
    pts = pts + torch.where(ctx["strong_enemy"], float(cal.get("strong_enemy_points", R["enemy_morale_penalty_value_min"])), 0.0)
    # The army is beaten as a whole (army_collapse: the database's 2.6x / 0.22 thresholds).
    pts = pts + torch.where(ctx.get("collapse", torch.zeros_like(pts, dtype=torch.bool)),
                            R["ume_concerned_army_destruction"], 0.0)
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
    # Army losses continue even far from enemies: a terminal router cannot recover out of it.
    free = ~ctx["enemy_near"] & ~ctx.get("collapse", torch.zeros_like(alive))
    routed_M = torch.where(free, M + cal["rally_rate"] * dt, torch.minimum(M, stepped))
    M = torch.where(routing, routed_M, stepped)
    u["rout_s"] = torch.where(routing, u["rout_s"] + dt, torch.zeros_like(u["rout_s"]))
    u["rally_s"] = u["rally_s"] + dt
    # Unbreakable units (attribute unbreakable: flagellants) never lose leadership: their points stay
    # at leadership or above, so they never waver or rout (docs/en/game/mechanics/abilities.md).
    M = torch.where(u["unbreakable"], torch.maximum(M, L), M)

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
    if (cal.get("collapse") or {}).get("shatter_below_broken"):
        # The lower edge of the DB's broken band is -50 points. In the CCO recordings,
        # all 274 army-loss shatters before a third rout have crossed this edge.
        shattered = (alive & ctx.get("collapse", torch.zeros_like(alive))
                     & ~u["unbreakable"] & (M < R["ums_broken_threshold_lower"]))
        u["s"] = u["s"] | shattered
        u["r"] = u["r"] | shattered
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
