"""Shooting (docs/en/training/simulator.md).

A missile unit shoots when it stands still, is not in melee, has projectiles and an enemy within
range (from its formation's edge to the target's: the game's AI shoots from 118-135 m between
centres with a 120-130 m range). It first aims for aim_s seconds after halting (measured 3.3 s arrows, 4.3 s
sling); then every man shoots once per reload (measured 11.0 / 11.5 s, longer than the passport).

    hits   = shots x hit_rate x distance factor (x single_entity_factor at a lone man; a lone
             man in melee takes 1 / (1 + men fighting him) of them)
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


def volley(u, pw, target, dt, params, crowd=None):
    """Shots, and HP taken per pair [B, N, N] (i shoots j) this step; per-hit damage [B, N, N].
    crowd [B, N]: men fighting each unit in melee; a lone man among them takes his share of the
    hits (1 / (1 + crowd)), the rest fall on the crowd and are lost (no friendly fire yet)."""
    ms = params.sim["missile"]
    B = params.battle
    shooting = target >= 0
    shots = torch.where(shooting, u["men"] * dt / u["reload"].clamp(min=1e-6), torch.zeros_like(u["men"]))
    shots = torch.minimum(shots, u["a"])
    t = target.clamp(min=0)
    onehot = torch.zeros_like(pw["dist"]).scatter_(2, t[:, :, None], 1.0) * shooting[:, :, None]
    dist = pw["dist"]
    factor = distance_factor(dist, ms["distance_factor"])
    single = (u["men0"] <= 1)[:, None, :]
    rate = (u["hit_rate"][:, :, None] * factor).clamp(max=1.0)
    rate = torch.where(single, rate * ms["single_entity_factor"], rate)
    if crowd is not None:
        rate = torch.where(single, rate / (1 + crowd[:, None, :]), rate)
    front = pw["rel_j"].abs() <= B["shield_defence_angle_missile"] * geometry.DEG
    shield = torch.where(front, u["shield"][:, None, :], torch.zeros_like(dist))
    hit = melee.per_hit(u["m_damage"][:, :, None], u["m_ap"][:, :, None], u["armour"][:, None, :],
                        u["hp_man"][:, None, :], u["resist_missile"][:, None, :])
    hp = onehot * shots[:, :, None] * rate * hit * (1 - shield)
    return shots, hp, hit
