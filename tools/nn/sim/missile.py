"""Shooting (docs/en/training/simulator.md).

A missile unit shoots when it stands still, is not in melee, has projectiles and an enemy within
range (from its formation's edge to the target's: the game's AI shoots from 118-135 m between
centres with a 120-130 m range). It first aims for aim_s seconds after halting (measured 3.3 s arrows, 4.3 s
sling); then every man shoots once per reload (measured 11.0 / 11.5 s, longer than the passport).
Volleys: the men reload all the time (moving too) and every loaded man shoots as soon as the unit
can, so the first shot after halting or after a pause is a volley of the whole unit (measured:
archers on the range, build/archer-range/runs 28.09.2026, ~80 of 90 arrows within 1 s of the first,
then ~10 s nothing); a unit shooting on keeps the steady rate men / reload.

    hits   = shots x hit_rate x distance factor (x single_entity_factor at a lone man); aimed
             at a unit in melee, a measured share lands on the shooter's own units in contact
             with it (friendly fire: 0.26 arrows, 0.56 sling); and a share by distance
             lands on the target's neighbours out of melee (spill: 0.115 within 30 m,
             0.034 at 30-60 m, 0.015 at 60-90 m)
    per hit = ap + base x (1 - 0.75 armour / 100); a shield blocks its chance from the front
              (within shield_defence_angle_missile, 60 deg); x (1 - missile resistance)

Target: the ATTACK order's target if in range, else the nearest standing enemy in range, else
the nearest routing one (fire at will; into melee too: allow_fire_at_will_into_melee 1).

Line of fire (direct fire only: projectiles with trajectory `low`, the passport's missile.direct;
arrows and slings arc over friends): the shooter's men aim at the target's centre; a friendly unit
between them (nearer than the target's edge) blocks the lines that pass through its width across
the line, widened by the friendly man radius coefficient (projectile_friendly_fire_man_radius_
coefficient 2.2: friends count wider, so a formation is a wall). Blocked men do not shoot; at
unit_firing_line_of_sight_considered_obstructed_ratio (0.75) blocked the unit holds fire at that
target and takes the next one in range (an ordered target too: the bridge releases a blocked
shooter to fire at will), or holds fire. Enemies in the way do not block (the game's clear-shot
test is about friends). Flat map: no fire over friends from higher ground.
"""
import torch

from tools.nn.sim import geometry, melee

FAR = 1e9


def distance_factor(dist, table):
    """Piecewise-linear factor over centre distance (config/nn/sim.json missile.distance_factor)."""
    d = torch.tensor([p[0] for p in table], dtype=dist.dtype, device=dist.device)
    v = torch.tensor([p[1] for p in table], dtype=dist.dtype, device=dist.device)
    x = dist.clamp(min=float(table[0][0]), max=float(table[-1][0]))
    idx = torch.searchsorted(d, x.contiguous()).clamp(1, len(table) - 1)
    d0, d1 = d[idx - 1], d[idx]
    v0, v1 = v[idx - 1], v[idx]
    w = (x - d0) / (d1 - d0)
    return v0 + w * (v1 - v0)


def choose_target(u, pw, can_shoot, order_target, order_attack, exclude=None):
    """[B, N] slot each shooter aims at (-1 none). exclude [B, N, N]: targets i may not take."""
    alive = (u["men"] > 0) & ~u["gone"]
    in_range = pw["enemy"] & alive[:, None, :] & (pw["gap"] <= u["range"][:, :, None])
    if exclude is not None:
        in_range = in_range & ~exclude
    standing = in_range & ~u["r"][:, None, :]
    score = torch.where(standing, pw["dist"], torch.where(in_range, pw["dist"] + 1e5, torch.full_like(pw["dist"], FAR)))
    nearest = score.argmin(dim=2)
    has = score.min(dim=2).values < FAR
    ordered = order_attack & (order_target >= 0)
    ot = order_target.clamp(min=0)
    ordered_ok = ordered & in_range.gather(2, ot[:, :, None]).squeeze(2)
    target = torch.where(ordered_ok, ot, torch.where(has, nearest, torch.full_like(nearest, -1)))
    return torch.where(can_shoot, target, torch.full_like(target, -1))


def blocked_share(u, pw, target, params):
    """[B, N] share of shooter i's men whose line to its target (slot, -1 none: 0) passes through a
    friendly unit (module docstring). Rectangles across the line: front x |cos| + depth x |sin|."""
    t = target.clamp(min=0)[:, :, None]
    theta = pw["theta"].gather(2, t)                              # [B, i, 1] bearing to the target
    dist = pw["dist"].gather(2, t).clamp(min=1e-3)
    near_edge = pw["reach_j"].gather(2, t)                         # to the target's edge
    rel = pw["theta"] - theta                                      # [B, i, k] k seen off the line
    along = pw["dist"] * torch.cos(rel)
    across = pw["dist"] * torch.sin(rel)
    b = u["b"] * geometry.DEG
    front, depth = pw["front"], pw["depth"]
    phi = b[:, None, :] - theta                                    # k's facing against the line
    coef = params.battle.get("projectile_friendly_fire_man_radius_coefficient", 2.2)
    half = (front[:, None, :] / 2 * torch.cos(phi).abs() + depth[:, None, :] / 2 * torch.sin(phi).abs()
            + (coef - 1) * u["radius"][:, None, :])
    phi_i = b[:, :, None] - theta
    width = (front[:, :, None] * torch.cos(phi_i).abs() + depth[:, :, None] * torch.sin(phi_i).abs()).clamp(min=1e-3)
    # The lines from the shooter's front converge on the target's centre: at `along` the line from
    # offset s across passes at s x (1 - along / dist); k covers the offsets lo..hi.
    shrink = (1 - along / dist).clamp(min=1e-3)
    lo, hi = (across - half) / shrink, (across + half) / shrink
    cover = (torch.minimum(hi, width / 2) - torch.maximum(lo, -width / 2)).clamp(min=0) / width
    N = u["men"].shape[1]
    eye = torch.eye(N, dtype=torch.bool, device=u["men"].device)[None]
    friend = ((u["side"][:, :, None] == u["side"][:, None, :]) & ~eye & (u["side"][:, None, :] > 0)
              & ((u["men"] > 0) & ~u["gone"])[:, None, :])
    between = friend & (along > 0) & (along < near_edge)
    blocked = torch.where(between, cover, torch.zeros_like(cover)).sum(2).clamp(max=1)
    return torch.where(target >= 0, blocked, torch.zeros_like(blocked))


def clear_shot(u, pw, target, can_shoot, order_target, order_attack, params):
    """Direct fire's line of fire (module docstring): (target [B, N], clear [B, N] share of the men
    that shoot). A direct-fire unit blocked at its target beyond the obstructed ratio takes the next
    target in range; blocked there too, it holds fire (-1). Other shooters: as given, clear 1."""
    ratio = params.battle.get("unit_firing_line_of_sight_considered_obstructed_ratio", 0.75)
    direct = u["direct"]
    blocked = blocked_share(u, pw, target, params)
    bad = direct & (target >= 0) & (blocked >= ratio)
    exclude = torch.zeros_like(pw["enemy"]).scatter_(2, target.clamp(min=0)[:, :, None], True) & bad[:, :, None]
    again = choose_target(u, pw, can_shoot & bad, order_target, order_attack, exclude=exclude)
    target = torch.where(bad, again, target)
    blocked = torch.where(bad, blocked_share(u, pw, target, params), blocked)
    held = direct & (target >= 0) & (blocked >= ratio)
    target = torch.where(held, torch.full_like(target, -1), target)
    clear = torch.where(direct, 1 - blocked, torch.ones_like(blocked))
    return target, torch.where(target >= 0, clear, torch.zeros_like(clear))


def volley(u, pw, target, dt, params, contact=None, clear=None, loaded=None):
    """Shots, and HP taken per pair [B, N, N] (i shoots, f is hit) this step; per-hit damage
    [B, N, N]. clear [B, N]: share of the shooter's men with a clear line (clear_shot; None: all).
    loaded [B, N]: share of the men loaded, who all shoot now (None: men x dt / reload, the steady rate).
    contact [B, N, N]: which units touch in melee. Of the hits aimed at a unit in
    melee, the shooter's friendly_fire share lands on its own units in contact with the target
    (measured, docs/en/training/simulator.md)."""
    ms = params.sim["missile"]
    B = params.battle
    shooting = target >= 0
    per_man = dt / u["reload"].clamp(min=1e-6) if loaded is None else loaded
    shots = torch.where(shooting, u["men"] * per_man, torch.zeros_like(u["men"]))
    if clear is not None:
        shots = shots * clear
    shots = torch.minimum(shots, u["a"])
    t = target.clamp(min=0)
    onehot = torch.zeros_like(pw["dist"]).scatter_(2, t[:, :, None], 1.0) * shooting[:, :, None]
    dist = pw["dist"]
    factor = distance_factor(dist, ms["distance_factor"])
    rate = (u["hit_rate"][:, :, None] * factor).clamp(max=1.0)
    aimed = onehot * shots[:, :, None] * rate                 # [B, i, j] hits aimed at j
    men = u["men"].clamp(min=0)
    lone = torch.where(u["men0"] <= 1, torch.full_like(men, ms["single_entity_factor"]), torch.ones_like(men))
    if contact is None:
        landed = aimed * lone[:, None, :]
    else:
        # Hits aimed at a unit in melee: the shooter's `friendly_fire` share lands on its own
        # units in contact with the target (split by their men), the rest on the target.
        near = contact.transpose(1, 2).float() * men[:, None, :]  # [B, j, f] men of f in contact with j
        crowd = near.sum(2)
        engaged = (crowd > 0).float()
        friends = near / crowd.clamp(min=1e-6)[:, :, None]
        ff = u["friendly_fire"][:, :, None] * engaged[:, None, :]  # [B, i, j]
        # A lone man among them (a lord) is hit as a lone target is: single_entity_factor of his
        # share (without it a lord in melee with shot enemies took the whole friendly fire).
        landed = torch.bmm(aimed * ff, friends * lone[:, None, :]) + aimed * (1 - ff) * lone[:, None, :]
    # Spill: hits aimed at j also land on j's neighbours of its own side that are out of melee,
    # a share by distance between centres (measured).
    if ms.get("spill"):
        same = (u["side"][:, :, None] == u["side"][:, None, :]) & (u["side"][:, :, None] > 0)
        eye = torch.eye(men.shape[1], dtype=torch.bool, device=men.device)[None]
        free = (men > 0) if contact is None else (men > 0) & ~contact.any(2)
        spill = distance_factor(dist, ms["spill"]) * (same & ~eye & free[:, None, :]).float()
        if contact is not None and ms.get("spill_melee"):
            # Hits aimed at a unit in melee also land on its own side's units in melee near it
            # (measured: the spearmen around a General the slingers shoot at).
            fighting = (men > 0) & contact.any(2)
            near = distance_factor(dist, ms["spill_melee"]) * (same & ~eye & fighting[:, None, :]).float()
            spill = spill + near * fighting[:, :, None].float()
        landed = landed + torch.bmm(aimed, spill * lone[:, None, :])
    front = pw["rel_j"].abs() <= B["shield_defence_angle_missile"] * geometry.DEG
    shield = torch.where(front, u["shield"][:, None, :], torch.zeros_like(dist))
    hit = melee.per_hit(u["m_damage"][:, :, None], u["m_ap"][:, :, None], u["armour"][:, None, :],
                        u["hp_man"][:, None, :], u["resist_missile"][:, None, :])
    hp = landed * hit * (1 - shield)
    return shots, hp, hit


def friendly(hp, u):
    """Friendly fire in gold this step from the HP per pair [B, N, N] (i shoots, f is hit; as dealt):
    (dealt [B, N]: what each shooter took from its own side, taken [B, N]: what each unit lost to its
    own side's shots). Gold = HP x the victim's cost / its starting HP (reward.py's worth). The side's
    reward pays it anyway (its gold lost); reward.unit_step lays it on the shooter instead of the
    victim (gate 02.10.2026: four slinger units shot a Warlord in melee with our spearmen for 119 s,
    ~1.8-2.3 HP of our own per shot in the game and in the simulator alike)."""
    same = (u["side"][:, :, None] == u["side"][:, None, :]) & (u["side"][:, :, None] > 0)
    worth = u["cost"] / u["hp0"].clamp(min=1e-6)
    gold = torch.where(same, hp, torch.zeros_like(hp)) * worth[:, None, :]
    return gold.sum(2), gold.sum(1)
