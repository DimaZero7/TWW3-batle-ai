"""Movement (docs/en/training/simulator.md).

Each unit heads for a point at walk or run speed (passport), speeding up and slowing down with
the passport's acceleration and deceleration, and stops on the point. A unit locked in melee
stands, and its formation turns slowly (melee_turn_deg_s, measured); out of melee a standing unit turns
in place at turn.formation_deg_s (a lord: turn.single_deg_s), a walking one faces the way it walks. A routing unit runs (at
rout_speed of its run, measured) away from the enemies near it (or towards its own edge of the map)
and leaves the battle when it crosses the map's edge; standing units stay inside. Formations do
not pass through each other: overlapping units are pushed apart along the line of centres.
"""
import torch


def ramp(t, table):
    """Linear in t between the points [(t, value), ...] of table (sorted by t), the end values beyond them."""
    out = torch.full_like(t, float(table[0][1]))
    for (t0, v0), (t1, v1) in zip(table[:-1], table[1:]):
        w = ((t - float(t0)) / max(float(t1) - float(t0), 1e-9)).clamp(0, 1)
        out = torch.where(t > float(t0), float(v0) + w * (float(v1) - float(v0)), out)
    return out


def velocity(u, goal_x, goal_z, speed_goal, moving, dt):
    """New velocity [B, N] towards the goal (per-unit acceleration / deceleration, stop on the point)."""
    dx, dz = goal_x - u["x"], goal_z - u["z"]
    dist = torch.sqrt(dx * dx + dz * dz + 1e-9)
    speed = torch.sqrt(u["vx"] ** 2 + u["vz"] ** 2)
    # Slow down so as to stop on the point.
    brake = torch.sqrt(2 * u["decel"] * (dist - 0.5).clamp(min=0))
    want = torch.where(moving, torch.minimum(speed_goal, brake), torch.zeros_like(speed))
    new = torch.where(want > speed, torch.minimum(want, speed + u["accel"] * dt),
                      torch.maximum(want, speed - u["decel"] * dt))
    new = torch.minimum(new, dist / dt)
    ux, uz = dx / dist, dz / dist
    return new * ux, new * uz


def flee_goal(u, pw, alive, bounds, near_m=0.0):
    """Where a routing unit runs: away from the standing enemies, weighted by nearness (1/d) - every standing enemy
    (near_m 0, morale.flee_near_m; the recordings: far from every enemy a router keeps running from the enemy army,
    not to its own edge) or those within near_m; with none, towards its own side's edge (side 1 west, side 2 east)."""
    enemy = pw["enemy"] & alive[:, None, :] & ~u["r"][:, None, :]
    near = enemy & (pw["dist"] < near_m) if near_m > 0 else enemy
    w = torch.where(near, 1 / pw["dist"].clamp(min=1), torch.zeros_like(pw["dist"]))
    ax = -(w * torch.sin(pw["theta"])).sum(2)
    az = -(w * torch.cos(pw["theta"])).sum(2)
    home = torch.where(u["side"] == 1, -1.0, 1.0)
    none = (ax.abs() + az.abs()) < 1e-9
    ax = torch.where(none, home, ax)
    az = torch.where(none, torch.zeros_like(az), az)
    n = torch.sqrt(ax * ax + az * az).clamp(min=1e-9)
    far = 4 * bounds
    return u["x"] + ax / n * far, u["z"] + az / n * far


def dodge_goal(u, pw, alive, fx, fz, within_m, deg, bounds):
    """The flee goal (fx, fz) of a routing formation with a standing enemy formation within within_m: the line from
    the nearest such enemy to the unit turned by deg, to a side fixed per (battle row, unit) (an integer hash);
    others keep (fx, fz)."""
    enemy = pw["enemy"] & alive[:, None, :] & ~u["r"][:, None, :] & (u["men0"][:, None, :] > 1)
    d = torch.where(enemy, pw["dist"], torch.full_like(pw["dist"], 1e9))
    dn, j = d.min(2)
    chased = alive & u["r"] & (u["men0"] > 1) & (dn <= within_m)
    ex, ez = u["x"].gather(1, j), u["z"].gather(1, j)
    ax, az = u["x"] - ex, u["z"] - ez                                 # from the chaser to the unit
    n = torch.sqrt(ax * ax + az * az).clamp(min=1e-6)
    ax, az = ax / n, az / n
    B, N = u["x"].shape
    b = torch.arange(B, device=u["x"].device)[:, None]
    i = torch.arange(N, device=u["x"].device)[None, :]
    h = (b * 1000003 + i * 7919 + 55511) & 0x7FFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0x7FFFFFFF
    side = torch.where((h & 1) == 0, 1.0, -1.0)
    a = torch.deg2rad(torch.full_like(ax, deg)) * side
    c, s = torch.cos(a), torch.sin(a)
    rx, rz = ax * c - az * s, ax * s + az * c
    far = 4 * bounds
    return (torch.where(chased, u["x"] + rx * far, fx), torch.where(chased, u["z"] + rz * far, fz))


def separate(pw, pairs, strength=0.5):
    """Push overlapping formations apart: each unit of a pair (pairs [B, N, N] true) moves
    `strength` of the overlap away from the other. Returns the shift (px, pz) [B, N]."""
    overlap = (-pw["gap"]).clamp(min=0)
    push = torch.where(pairs, overlap * strength, torch.zeros_like(overlap))
    px = -(push * torch.sin(pw["theta"])).sum(2)
    pz = -(push * torch.cos(pw["theta"])).sum(2)
    return px, pz


def face(u, dir_x, dir_z, turn):
    """Set the bearing towards (dir_x, dir_z) where turn [B, N] is true."""
    b = torch.rad2deg(torch.atan2(dir_x, dir_z)) % 360
    u["b"] = torch.where(turn, b, u["b"])


def limit_turn(u, before, mask, max_deg):
    """Where mask [B, N] is true, the bearing moves at most max_deg (a number or [B, N]) from `before`."""
    turn = torch.remainder(u["b"] - before + 180, 360) - 180
    lim = torch.as_tensor(max_deg, dtype=turn.dtype, device=turn.device)
    b = torch.remainder(before + torch.maximum(torch.minimum(turn, lim), -lim), 360)
    u["b"] = torch.where(mask, b, u["b"])


def clamp_to_map(u, bounds):
    """Standing units stay on the map; routing units that cross the edge leave the battle."""
    out = (u["x"].abs() > bounds) | (u["z"].abs() > bounds)
    leave = out & u["r"]
    u["gone"] = u["gone"] | leave
    keep = ~u["r"]
    u["x"] = torch.where(keep, u["x"].clamp(-bounds, bounds), u["x"])
    u["z"] = torch.where(keep, u["z"].clamp(-bounds, bounds), u["z"])
    return leave

