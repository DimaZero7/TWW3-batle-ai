"""Movement (docs/en/training/simulator.md).

Each unit heads for a point at walk or run speed (passport), speeding up and slowing down with
the passport's acceleration and deceleration, and stops on the point. A unit locked in melee
stands, and its formation turns slowly (melee_turn_deg_s, measured). A routing unit runs (at
rout_speed of its run, measured) away from the enemies near it (or towards its own edge of the map)
and leaves the battle when it crosses the map's edge; standing units stay inside. Formations do
not pass through each other: overlapping units are pushed apart along the line of centres.
"""
import torch


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


def flee_goal(u, pw, alive, bounds):
    """Where a routing unit runs: away from the standing enemies within 150 m, weighted by
    nearness; with none near, towards its own side's edge (side 1 west, side 2 east)."""
    enemy = pw["enemy"] & alive[:, None, :] & ~u["r"][:, None, :]
    near = enemy & (pw["dist"] < 150)
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
    """Where mask [B, N] is true, the bearing moves at most max_deg from `before`."""
    turn = torch.remainder(u["b"] - before + 180, 360) - 180
    b = torch.remainder(before + turn.clamp(-max_deg, max_deg), 360)
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

