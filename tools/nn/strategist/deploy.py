"""The deployment by code (plan.md "Расстановка — кодом", "Перенос в игру"): the groups' zones -> every unit's place.

A group's zone at the deployment is its units' slots (operative/groups.grid of n and the form: cols x rows) centred on
the chosen cell, turned by the chosen sector. A slot is as wide as the group's widest formation + GAP_M and as deep as
its deepest + ROW_GAP_M (the formations at their own frontage: the raw state's front / depth - slot()), so the group
stands as tight as the person stands his (build/diag_label/human_rules.md: ~23 m a unit along the line; the game's
zone 416 x 176 m holds 20 units); was the tactician's tile, 80 x 80 m a unit - 20 units need 128 000 m2, the game's zone
has 73 000 (07.11, build/diag_chain/zone_fix). The melee units fill the rows of slots from the front, the shooters after
them (each kind by slot); a row's units are centred in it; a unit stands on its place's centre facing the zone's turn.
The result is per unit exactly the game's teleport_to_location(position, bearing, width_m): x, z, the bearing - the
middle of its sector of the engine's 64 (docs/ru/game/units/commands.md; round_bearing), the width - the unit's own
frontage (a slot is wider by GAP_M: the neighbours never overlap).

    slot(members, front, depth)          (across, deep) [B, G] m: a group's slot (above); members [B, G, K]
    offsets(members, shooter, order, line, across, deep)   (s, f) each member's place in its group's axes
    apart_units(px, pz, heading, s, f, front, depth, member, placed, gap)   [B, M] bool: the group's units at the cells
                                      clear of the units placed already (the walk's ban: by formations, not zones)
    footprint(n, line, across, deep)     (front, depth) [B, G] m: the group's zone at the deployment (cols x rows slots)
    places(members, shooter, cx, cz, heading, line, width0, slot)   {x, z, b, width, on} [B, N]; slot (across, deep)
                                      [B, G] the walk's (net.choose plan slot_w, slot_d: slot_of(plan)); None - the
                                      tactician's tile_m (the old layout)
    errors(st, own, zone, spacing)    [B] long: the deployment's errors on the built state - an own unit in no group or
                                      in two, a unit's formation (geometry.dims) not wholly inside my deployment zone,
                                      two own formations overlapping (separating axes)
"""
import torch

from tools.nn.operative.groups import grid
from tools.nn.tactic import cfg as tcfg
from tools.nn.unit import zmap as Z

SECTORS = 64              # the engine keeps a unit's bearing in 64 sectors
GAP_M = 6.0               # tools/nn/armies/place.py LATERAL_GAP_M: between neighbours in a row
ROW_GAP_M = 6.0           # between a row of slots and the next (as across)


def round_bearing(b):
    """The facing the engine stands at for bearing b: the MIDDLE of its sector of 64 (measured 28.09.2026, the game's
    rule in src/apps/orders/facing.lua engine(): 90 -> 92.81). The place sent to the game is that facing, so the
    simulator's unit faces where the game's does (was the sector's edge: 2.8 deg off the game's)."""
    step = 360.0 / SECTORS
    return torch.remainder((torch.floor(torch.remainder(b, 360.0) / step) + 0.5) * step, 360.0)


def slot(members, front, depth):
    """(across, deep) [B, G] (module doc): members [B, G, K] bool, front, depth [B, K] the formations' (m)."""
    m = members.float()
    across = (m * front[:, None, :]).amax(2) + GAP_M
    deep = (m * depth[:, None, :]).amax(2) + ROW_GAP_M
    return across, deep


def footprint(n, line, across, deep):
    rows, cols = grid(n, line)
    return cols.float() * across, rows.float() * deep


def offsets(members, shooter, order, line, across, deep):
    """(s, f) [..., N] each member's place in its group's axes (right, forward): members [..., N] bool, shooter [..., N],
    order [..., N] the units' slot numbers; line, across, deep [...]. The melee units fill the rows of slots from the
    front, the shooters after them (each kind by slot); a row's units are centred in it."""
    n = members.sum(-1)
    rows, cols = grid(n, line)
    key = torch.where(members, shooter.float() * 1000.0 + order.float(), torch.full_like(order.float(), 1e9))
    rank = key.argsort(-1).argsort(-1)                                   # the place in the group's order
    r = torch.div(rank, cols[..., None], rounding_mode="floor")
    i = rank % cols[..., None]
    k = torch.minimum(cols[..., None], n[..., None] - r * cols[..., None])
    s = (i.float() + 0.5 - k.float() / 2) * across[..., None]
    f = (rows[..., None].float() / 2 - r.float() - 0.5) * deep[..., None]
    return s, f


def apart_units(px, pz, heading, s, f, front, depth, member, placed, gap):
    """[B, M] bool: a group's units at the cells clear of the units placed already by gap m or more (their formations,
    not the groups' zones: two groups' zones may overlap where their units do not - as the person stands his; the
    simulator's geometry.rect_gap). px, pz [B, M] the group's centre at each cell; heading [B]; s, f [B, K] its
    members' places in its axes (offsets), front, depth [B, K] every row's formation, member [B, K] bool the group's;
    placed {x, z, b} [B, K] the units placed, on: placed["on"] [B, K]."""
    from tools.nn.sim import geometry
    B, M = px.shape
    out = torch.ones((B, M), dtype=torch.bool, device=px.device)
    on = placed["on"]
    if not bool(on.any()):
        return out
    K = member.shape[1]
    b = round_bearing(heading)
    for c0 in range(0, M, 32):
        c1 = min(M, c0 + 32)
        m = c1 - c0
        ux, uz = Z.world(s[:, None, :], f[:, None, :], px[:, c0:c1, None], pz[:, c0:c1, None], heading[:, None, None])
        ex = lambda v: v[:, None, :].expand(B, m, K)                       # noqa: E731
        x = torch.cat([ux, ex(placed["x"])], 2).reshape(B * m, 2 * K)
        z = torch.cat([uz, ex(placed["z"])], 2).reshape(B * m, 2 * K)
        bb = torch.cat([b[:, None, None].expand(B, m, K), ex(placed["b"])], 2).reshape(B * m, 2 * K)
        fr = torch.cat([ex(front), ex(front)], 2).reshape(B * m, 2 * K)
        dp = torch.cat([ex(depth), ex(depth)], 2).reshape(B * m, 2 * K)
        g = geometry.rect_gap(x, z, bb, fr, dp)[:, :K, K:].reshape(B, m, K, K)
        hit = (g < gap) & member[:, None, :, None] & on[:, None, None, :]
        out[:, c0:c1] = ~hit.flatten(2).any(2)
    return out


def slot_of(plan):
    """(across, deep, nudge) [B, G]: the walk's slots and each group's units' shift along its facing (m, - back;
    the formula's direct-fire snipers stand SNIPER_BACK_M behind the front row: strategist/army.py)."""
    return plan["slot_w"], plan["slot_d"], plan.get("nudge", torch.zeros_like(plan["slot_w"]))


def places(members, shooter, cx, cz, heading, line, width0, slot=None):
    """members [B, G, N] bool; shooter [B, N]; cx, cz, heading [B, G]; line [B, G] bool; width0 [B, N] the units' own
    frontage (m); slot (module doc)."""
    B, G, N = members.shape
    dev = members.device
    tile = float(tcfg()["tile_m"])
    if slot is None:
        sw = sd = torch.full((B, G), tile, device=dev)
        wmax = torch.full((B, N), tile - GAP_M, device=dev)
    else:
        sw, sd = slot[0], slot[1]
        wmax = torch.full((B, N), 1e9, device=dev)
    nudge = slot[2] if slot is not None and len(slot) > 2 else torch.zeros((B, G), device=dev)
    slot = torch.arange(N, device=dev, dtype=torch.float32).expand(B, G, N)
    s, f = offsets(members, shooter[:, None, :].expand(B, G, N), slot, line, sw, sd)
    f = f + nudge[..., None]
    x, z = Z.world(s, f, cx[..., None], cz[..., None], heading[..., None])
    on = members.any(1)
    g = members.float().argmax(1)                                        # [B, N] each unit's group
    pick = lambda v: v.gather(1, g[:, None, :])[:, 0]                    # noqa: E731
    b = round_bearing(heading.gather(1, g))
    return {"x": torch.where(on, pick(x), torch.zeros(B, N, device=dev)),
            "z": torch.where(on, pick(z), torch.zeros(B, N, device=dev)), "b": torch.where(on, b, torch.zeros_like(b)),
            "width": torch.where(on, torch.minimum(width0, wmax), width0), "on": on, "group": torch.where(on, g, -1)}


def corners(x, z, b, front, depth):
    """[..., 4, 2] a turned rectangle's corners."""
    out = []
    for ss, ff in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
        cx, cz = Z.world(ss * front / 2, ff * depth / 2, x, z, b)
        out.append(torch.stack([cx, cz], -1))
    return torch.stack(out, -2)


def _overlap(ca, cb):
    """[...] bool: two convex quads [..., 4, 2] overlap (separating axes, a touch is no overlap)."""
    sep = torch.zeros(ca.shape[:-2], dtype=torch.bool, device=ca.device)
    for c in (ca, cb):
        for e in range(2):
            d = c[..., e + 1, :] - c[..., e, :]
            nrm = torch.stack([-d[..., 1], d[..., 0]], -1)
            pa = (ca * nrm[..., None, :]).sum(-1)
            pb = (cb * nrm[..., None, :]).sum(-1)
            sep = sep | (pa.amax(-1) <= pb.amin(-1) + 1e-3) | (pb.amax(-1) <= pa.amin(-1) + 1e-3)
    return ~sep


def errors(st, own, member, zone, spacing):
    """[B] long (module doc). member [B, N] long each slot's group (-1 none)."""
    from tools.nn.sim import geometry
    u = st.u
    front, depth = geometry.dims(u, spacing)
    B, N = own.shape
    err = (own & (member < 0)).sum(1)
    cs = corners(u["x"], u["z"], u["b"], front, depth)                  # [B, N, 4, 2]
    s, f = Z.local(cs[..., 0], cs[..., 1], zone["x"][:, None, None], zone["z"][:, None, None], zone["b"][:, None, None])
    out = (s.abs() > zone["front"][:, None, None] / 2 + 1e-3) | (f.abs() > zone["depth"][:, None, None] / 2 + 1e-3)
    err = err + (out.any(2) & own).sum(1)
    for i in range(N):
        for j in range(i + 1, N):
            both = own[:, i] & own[:, j]
            if bool(both.any()):
                err = err + (_overlap(cs[:, i], cs[:, j]) & both).long()
    return err
