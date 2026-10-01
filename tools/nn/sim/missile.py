"""Shooting (docs/en/training/simulator.md).

A missile unit shoots when it stands still, is not in melee, has projectiles and an enemy within
range (from its formation's edge to the target's: the game's AI shoots from 118-135 m between
centres with a 120-130 m range). It first aims for aim_s seconds after halting (measured 3.3 s arrows, 4.3 s
sling); then every man shoots once per reload (measured 11.0 / 11.5 s, longer than the passport).

    hits   = shots x hit_rate x distance factor (x single_entity_factor at a lone man); aimed
             at a unit in melee, a measured share lands on the shooter's own units in contact
             with it (friendly fire: 0.26 arrows, 0.56 sling); and a share by distance
             lands on the target's neighbours out of melee (spill: 0.115 within 30 m,
             0.034 at 30-60 m, 0.015 at 60-90 m)
    per hit = ap + base x (1 - 0.75 armour / 100); a shield blocks its chance from the front
              (within shield_defence_angle_missile, 60 deg); x (1 - missile resistance)

Target: the ATTACK order's target if in range, else the nearest standing enemy in range, else
the nearest routing one (fire at will; into melee too: allow_fire_at_will_into_melee 1).
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


def choose_target(u, pw, can_shoot, order_target, order_attack):
    """[B, N] slot each shooter aims at (-1 none) and whether it is in range."""
    alive = (u["men"] > 0) & ~u["gone"]
    in_range = pw["enemy"] & alive[:, None, :] & (pw["gap"] <= u["range"][:, :, None])
    standing = in_range & ~u["r"][:, None, :]
    score = torch.where(standing, pw["dist"], torch.where(in_range, pw["dist"] + 1e5, torch.full_like(pw["dist"], FAR)))
    nearest = score.argmin(dim=2)
    has = score.min(dim=2).values < FAR
    ordered = order_attack & (order_target >= 0)
    ot = order_target.clamp(min=0)
    ordered_ok = ordered & in_range.gather(2, ot[:, :, None]).squeeze(2)
    target = torch.where(ordered_ok, ot, torch.where(has, nearest, torch.full_like(nearest, -1)))
    return torch.where(can_shoot, target, torch.full_like(target, -1))


def volley(u, pw, target, dt, params, contact=None):
    """Shots, and HP taken per pair [B, N, N] (i shoots, f is hit) this step; per-hit damage
    [B, N, N]. contact [B, N, N]: which units touch in melee. Of the hits aimed at a unit in
    melee, the shooter's friendly_fire share lands on its own units in contact with the target
    (measured, docs/en/training/simulator.md)."""
    ms = params.sim["missile"]
    B = params.battle
    shooting = target >= 0
    shots = torch.where(shooting, u["men"] * dt / u["reload"].clamp(min=1e-6), torch.zeros_like(u["men"]))
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
