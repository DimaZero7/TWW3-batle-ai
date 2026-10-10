"""The game's skirmish mode (docs/en/training/simulator.md, config/nn/sim.json skirmish).

A shooter whose mode is on (STATIC `sk`; the database's can_skirmish gives it the gaps `sk_m`, `sk_stop_m`) runs away,
over any order it has, from an enemy that comes at it: a standing enemy that moves (faster than moving_mps) with its
way within cone_deg of the line from it to the shooter. Such an enemy whose nearest man comes sk_m or nearer to the
shooter's nearest man (soldier_gap: the rectangles of the men's places, the probes' measure) starts the run; the shooter stops shooting and runs at its run speed
straight away from it; the run goes on while such an enemy is within sk_stop_m, and ends as soon as none is (the enemy
went past, turned to another unit, stopped, or fell sk_stop_m behind). Then its order goes on (a held shooter turns
back to its target - `skt`, battle.py's standing turn - aims and shoots) and the next approach starts another run.
A new order (another kind or target, or a point new_point_m away; cal order_cuts) ends the run at once: the unit carries
out its order until the trigger starts a run again (the game's AI cuts its shooters' runs with its own orders).
Caught in melee it goes on running (melee_run: a leave, battle.py - it strikes nobody, the enemies in contact strike
it); with melee_run off a caught unit stands and fights. Routing: no mode. On the run it does not shoot, but a unit
that fires on the move (fire_move: the Night Runners' throwing stars) does, as on any move (battle.py).

The skirmish probes (tools/nn/charge_probe.py plans skirmish and skirmish2; build/charge-probe/runs 20261010-033006 /
-033102, -053344 / -053448; build/skirm, build/skirmsim/angle2.py): a halted shooter and enemy infantry that attacks it,
runs past it or attacks its neighbour.
"""
import math

import torch

from tools.nn.sim import geometry


def mode_on(u, everyone=False):
    """[B, N] the mode is on: a unit that can skirmish (sk_m > 0) whose toggle is on (or every such unit: everyone)."""
    can = u["sk_m"] > 0
    return can & (u["sk"] | bool(everyone))


def soldier_gap(u, spacing):
    """[B, i, j] the distance between the two units' outermost soldiers (nearest man to nearest man, as the game's
    probe measured the start): each formation the rectangle of its men's places - (files - 1) h x (ranks - 1) v, a lord
    a point - and the distance between the rectangles, the least over each one's corners to the other (exact for
    rectangles apart; 0 when a corner lies inside the other)."""
    front, depth = geometry.dims(u, spacing)
    single = u["men0"] <= 1
    h = torch.where(u["sp_h"] > 0, u["sp_h"], torch.full_like(u["sp_h"], float(spacing))) if "sp_h" in u else spacing
    v = torch.where(u["sp_v"] > 0, u["sp_v"], torch.full_like(u["sp_v"], float(spacing))) if "sp_v" in u else spacing
    hw = torch.where(single, torch.zeros_like(front), ((front - h) / 2).clamp(min=0))
    hd = torch.where(single, torch.zeros_like(depth), ((depth - v) / 2).clamp(min=0))
    b = u["b"] * geometry.DEG
    fx, fz = torch.sin(b), torch.cos(b)                 # facing
    rx, rz = fz, -fx                                    # right
    x, z = u["x"], u["z"]

    def to_rect(px, pz):
        """[B, i, j] distance from points (px, pz) [B, i, j] to rectangle j."""
        dx, dz = px - x[:, None, :], pz - z[:, None, :]
        lx = (dx * rx[:, None, :] + dz * rz[:, None, :]).abs() - hw[:, None, :]
        lz = (dx * fx[:, None, :] + dz * fz[:, None, :]).abs() - hd[:, None, :]
        return torch.sqrt(lx.clamp(min=0) ** 2 + lz.clamp(min=0) ** 2)

    best = None
    for sr in (-1.0, 1.0):
        for sf in (-1.0, 1.0):
            cx = x + sr * hw * rx + sf * hd * fx          # unit i's corner
            cz = z + sr * hw * rz + sf * hd * fz
            d = to_rect(cx[:, :, None].expand(-1, -1, x.shape[1]), cz[:, :, None].expand(-1, -1, x.shape[1]))
            best = d if best is None else torch.minimum(best, d)
    return torch.minimum(best, best.transpose(1, 2))


def coming(u, pw, standing, cone_deg, moving_mps):
    """[B, i, j] enemy j is standing and moving at unit i: its speed over moving_mps, its velocity within cone_deg of
    the line from j's centre to i's."""
    vx, vz = u["vx"][:, None, :], u["vz"][:, None, :]                  # j's velocity
    v = torch.sqrt(vx * vx + vz * vz)
    # pw theta [b, i, j]: the bearing from i to j; from j to i is its opposite: (-sin, -cos)
    toward = -(vx * torch.sin(pw["theta"]) + vz * torch.cos(pw["theta"]))
    cos_min = math.cos(math.radians(float(cone_deg)))
    return pw["enemy"] & standing[:, None, :] & (v > float(moving_mps)) & (toward >= cos_min * v)


def update(u, pw, standing, in_touch, bounds, cal, spacing=1.5, t=None, dt=0.5, new_order=None):
    """Who runs this step (and keeps u["skr"], u["skt"], u["sk_in"]): (run [B, N] bool, goal x, goal z [B, N]) - the
    goal far away from the nearest enemy coming at the unit, straight from its centre through the unit's. in_touch: in
    contact with a standing enemy (no new run; with cal melee_run off no run at all). cal: config/nn/sim.json skirmish;
    t [B]: the battle time (the hazard's draw). An enemy coming that attacks the unit (its attack order on it) starts a
    run within sk_m and keeps it to sk_stop_m (ending with cal attacker_end_per_s a second; 0: never - an attacker
    within sk_m starts it again at once); one coming that attacks another starts one only as it comes within sk_m
    (not again while it stays there) and its run ends with cal other_end_per_s a second (a fixed hash draw per battle
    id, unit and step, as morale's rally_hazard). new_order [B, N] (None: off): the unit got another order this step
    (battle.py sk_new: another kind or target, or a point moved cal new_point_m or more) - it ends a run under way and
    starts none this step: the order is carried out until the mode's trigger starts a new run (the game's AI ends its
    shooters' runs by its own orders: 91 % of the runs' ends in the 72 replay battles come with a new ordered point
    12 m from the unit, median; the far point is back within 1 s in 44 % of them, median 3 s - an order 'erased shortly
    after it is given' while the enemy is near)."""
    from tools.nn.sim import morale
    from tools.nn.sim.orders import ATTACK
    foe = coming(u, pw, standing, cal["cone_deg"], cal["moving_mps"])
    sg = soldier_gap(u, spacing)
    big = torch.full_like(sg, 1e9)
    me = torch.arange(u["x"].shape[1], device=u["x"].device)[None, :, None]
    att = foe & (u["order_kind"][:, None, :] == ATTACK) & (u["order_target"][:, None, :] == me)
    g, j = torch.where(foe, sg, big).min(2)
    ga, ja = torch.where(att, sg, big).min(2)
    go, jo = torch.where(foe & ~att, sg, big).min(2)
    melee_run = bool(cal.get("melee_run"))
    can = mode_on(u, cal.get("network")) & standing & (u["range"] > 0)
    free = ~in_touch | melee_run
    in_other = go <= u["sk_m"]
    # (sk_in: any standing enemy not attacking the unit within sk_m, whatever its way: a new start from such an enemy
    # needs it to come in from beyond the gap - not a flicker of its way in and out of the cone)
    near_other = (pw["enemy"] & standing[:, None, :] & ~att & (sg <= u["sk_m"][:, :, None])).any(2)
    start = can & ~in_touch & ((ga <= u["sk_m"]) | (in_other & ~u["sk_in"]))
    cut = new_order if new_order is not None else torch.zeros_like(start)
    start = start & ~cut
    k = torch.round((t if t is not None else torch.zeros(u["x"].shape[0], device=u["x"].device)) / dt).long()
    draw = morale._hash_uniform(u["x"].shape[0], u["x"].shape[1], k, u["x"].device, salt=52361, ids=morale.battle_ids(u))

    def stays(p_end):
        return draw >= 1.0 - (1.0 - float(p_end)) ** dt
    chased = ga < u["sk_stop_m"]
    keep_att = can & u["skr"] & free & chased & stays(cal.get("attacker_end_per_s", 0.0))
    keep_other = can & u["skr"] & free & ~chased & (go < u["sk_stop_m"]) & stays(cal.get("other_end_per_s", 0.0))
    run = start | ((keep_att | keep_other) & ~cut)
    u["sk_in"] = near_other & can
    # a run that ends out of melee leaves the unit facing away: it turns back to its target once it stands (battle.py)
    u["skt"] = (u["skt"] | (u["skr"] & ~run & standing & ~in_touch)) & ~run & standing
    u["skr"] = run
    # away from the nearest attacker coming, else the nearest other coming
    jj = torch.where(ga < 1e9, ja, torch.where(go < 1e9, jo, j))
    ex, ez = u["x"].gather(1, jj), u["z"].gather(1, jj)
    ax, az = u["x"] - ex, u["z"] - ez
    n = torch.sqrt(ax * ax + az * az).clamp(min=1e-6)
    far = 4 * bounds
    return run, u["x"] + ax / n * far, u["z"] + az / n * far
