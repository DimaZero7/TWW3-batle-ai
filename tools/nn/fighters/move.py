"""The men's own moves after the simulator moved their units (lab.json move; docs/en/training/fighters_lab.md).

A man is carried by his unit's centre (his place is an offset from it). Then:
  his place    his file and rank in the formation (state.slots: the man behind steps up as men fall), turned with the
               unit; a formation that turned about (more than 90 deg in one step) swaps its ranks and files (the game's
               about-face in place) instead of the men walking through each other
  in a fight   a man with an enemy man within reach + the tether (melee.resolve's F.eng) stands at the standoff from him
               (lab.json standoff: measured, closer at the contact, ~2.3 m later), on the line between them, at most the
               database's formed tether distance (2.0 m) from his place
  pace         he walks there at most his unit's run speed (relative to the unit's centre) a step
  bodies       men closer than the sum of their radii (the database's battle_entities radius) push apart, half each,
               friends and enemies alike
A lone man (a lord) is his unit's centre."""
import torch

from tools.nn.fighters import cells, melee as FM, state as S


def step(F, u, params, lab, dt, near):
    B, M = F.B, F.M
    live = S.alive(F)
    lying = live & (F.down > 0)            # knocked down: he lies where he fell (knock.py)
    # an about-face in place
    db = S.wrap_deg(u["b"] - F.b_last)
    F.flip = torch.where(db.abs() > 90.0, ~F.flip, F.flip)
    F.sh = torch.where(db.abs() > 90.0, torch.zeros_like(F.sh), F.sh)
    F.b_last = u["b"].clone()
    sa, sf = S.slots(F, u)
    ux, uz = S.ugather(u, F, "x"), S.ugather(u, F, "z")
    px, pz = ux + F.dx, uz + F.dz
    N = u["men"].shape[1]
    # the front rank: the men whose place is the formation's foremost (an about-face: the old back rank)
    v = torch.where(u["sp_v"] > 0, u["sp_v"], torch.full_like(u["sp_v"], 1.5)).gather(1, F.uid)
    clean = sf - torch.where(F.flip.gather(1, F.uid), -F.jit_f, F.jit_f)
    top = torch.full((B, N), -1e9, device=sf.device).scatter_reduce(
        1, F.uid, torch.where(live, clean, torch.full_like(clean, -1e9)), reduce="amax")
    front = live & (clean >= top.gather(1, F.uid) - 0.5 * v)
    fighting = live & (F.tgt >= 0) & ~S.ugather(u, F, "r")
    if lab["move"].get("front_only", True):
        fighting = fighting & front
    # the formation keeps its ranks behind its fighting front (lab.json move.front_anchor): its places shift along its
    # facing by the mean way its fighting front men stand off their own places
    if lab["move"].get("front_anchor", True):
        b = S.ugather(u, F, "b") * S.geometry.DEG
        along = F.dx * torch.sin(b) + F.dz * torch.cos(b)
        n_f = S.unit_count(F, fighting, N)
        dev = S.unit_sum(F, torch.where(fighting, along - sf, torch.zeros_like(sf)), N) / n_f.clamp(min=1)
        F.sh = torch.where(n_f > 0, dev, F.sh * 0.5).clamp(-3.0, 3.0)
        sf = sf + F.sh.gather(1, F.uid)
    gx, gz = S.to_world(u, F, sa, sf)                       # the place, as an offset from the unit's centre
    # in a fight: the standoff from the man he fights (the front rank's men only: the men behind keep their places)
    eng = torch.where(fighting, F.tgt, torch.full_like(F.tgt, -1))
    if True:
        ec = eng.clamp(min=0)
        ex, ez = px.gather(1, ec), pz.gather(1, ec)
        vx, vz = px - ex, pz - ez
        n = torch.sqrt(vx * vx + vz * vz).clamp(min=1e-3)
        t_i = u["melee_s"].gather(1, F.uid) if "melee_s" in u else torch.zeros_like(px)
        t_j = u["melee_s"].gather(1, F.uid.gather(1, ec)) if "melee_s" in u else torch.zeros_like(px)
        s = FM.standoff(lab, torch.minimum(t_i, t_j))
        wx, wz = ex + vx / n * s - ux, ez + vz / n * s - uz          # where he would stand, from the unit's centre
        tether = float(params.battle.get("entity_action_attack_formed_tether_distance", 2.0))
        ox, oz = wx - gx, wz - gz
        on = torch.sqrt(ox * ox + oz * oz).clamp(min=1e-6)
        k = (tether / on).clamp(max=1.0)
        fight = eng >= 0
        gx = torch.where(fight, gx + ox * k, gx)
        gz = torch.where(fight, gz + oz * k, gz)
    # the pace
    mx, mz = gx - F.dx, gz - F.dz
    dist = torch.sqrt(mx * mx + mz * mz).clamp(min=1e-6)
    cap = (S.ugather(u, F, "run") * dt / dist).clamp(max=1.0)
    nx, nz = F.dx + mx * cap, F.dz + mz * cap
    # bodies
    cand = near.cand
    ok = (cand >= 0) & cells.gather2(live & ~lying, cand) & (live & ~lying)[:, :, None]
    ax, az = ux + nx, uz + nz
    cx = cells.gather2(ax, cand)
    cz = cells.gather2(az, cand)
    rx, rz = ax[:, :, None] - cx, az[:, :, None] - cz
    d = torch.sqrt(rx * rx + rz * rz).clamp(min=1e-3)
    rad = S.ugather(u, F, "radius")
    need = rad[:, :, None] + cells.gather2(rad, cand)
    over = torch.where(ok & (d < need), (need - d) / 2, torch.zeros_like(d))
    nx = nx + (over * rx / d).sum(2)
    nz = nz + (over * rz / d).sum(2)
    lone = (u["men0"] <= 1).gather(1, F.uid)
    moves = live & ~lying
    F.dx = torch.where(lone, torch.zeros_like(nx), torch.where(moves, nx, F.dx))
    F.dz = torch.where(lone, torch.zeros_like(nz), torch.where(moves, nz, F.dz))
