"""Own formations standing in each other in melee (docs/ru/training/simulator.md, melee; config/nn/sim.json
melee.stack_block, melee.stack_side, melee.stack_no_wrap, melee.stack_why).

The game's rules: a man starts a blow only when his body touches an enemy's body; own bodies do not let him through -
a man of a second unit that runs into a man of his own ahead stops there and does not walk round; a unit gives
targets only to its front rank, the rest go to their places in the formation; there is no rule "go round to another
side" - a second unit comes at the same nearest side of the enemy as the first. So a formation standing in another of
ours that fights already brings to the enemy only the part of its front the other's box leaves free (its edges), it
belongs to the side of the target the other one fights, and it does not wrap round the target's flank (its way is
our own men).

Who stands ahead: of two own formations whose boxes overlap, the one longer in its current melee (contact_s; the same:
the lower slot) - it came first, the later one stopped on its back. Only formations (men0 > 1) that are in contact
with an enemy cover others.

  cover [B, N, 4]  the share of each side of formation i (0 front, 1 / 2 the flanks towards its front line's + / -
                   end, 3 back - melee.strikes' side codes) its own formations ahead stand on: their boxes seen along
                   that side's line (across i's facing for the front and back, along it for a flank), their hull (the
                   span from the first to the last of them; a gap between two of them counts covered), within the side,
                   over its length. A friend counts for a side when its centre is on that side of i's: for a flank, on
                   that flank's half; for the front (back), not behind (ahead of) i's centre by more than half i's depth
                   (the simulator's boxes walk into each other, so the later unit may stand a little deeper than the one
                   it stopped behind in the game)
  ahead [B, N]     the own formation ahead covering most of i's front, -1 none
  behind [B, N, N] (i, j): i stands behind `ahead` on the side of target j that `ahead` fights - both touch j, and
                   i's centre is not nearer to j than `ahead`'s near edge along that side's outward normal (a unit
                   beside the target's flank whose corner only touches the front unit's box is not behind it)
  side_k [B, N, N] (i, j): the side of j (0 front, 1 / 2 flanks, 3 back - seen from j) `ahead` stands on
Needs torch."""
import torch


def side_of_target(pw):
    """[B, N, N] (i, j): the side of j that i stands on, 0 front, 1 / 2 the flanks, 3 back - by the line between the
    centres against j's box (the target_face_cap rule)."""
    s_j, c_j = torch.sin(pw["rel_j"]), torch.cos(pw["rel_j"])
    front_j, depth_j = pw["front"][:, None, :], pw["depth"][:, None, :]
    thru_j = s_j.abs() * depth_j <= c_j.abs() * front_j
    return torch.where(thru_j, torch.where(c_j >= 0, 0, 3), torch.where(s_j >= 0, 1, 2))


def stacked(u, pw, contact, side_j, dt=0.5, side_i=None):
    """(cover [B, N, 4], ahead [B, N], behind [B, N, N], side_k [B, N, N]) - the module's doc. contact [B, N, N]: i
    strikes j (melee.strikes' contact); side_j: side_of_target(pw). side_i [B, N, N] (the side of i that meets j,
    melee.strikes' codes): cover is then [B, N, N] - per pair, the share of the part of i's side that faces j (j's box
    seen along that side's line, within the side) the ones ahead cover: a target opposite the free part of the front
    is struck by the men there in full (a target that faces none of the side: the whole side's share)."""
    Bn, N = u["men"].shape
    dev = u["men"].device
    form = (u["men0"] > 1) & (u["men"] > 0)
    eng = contact.any(2) | contact.any(1)                                  # in contact with an enemy
    same = (u["side"][:, :, None] == u["side"][:, None, :]) & (u["side"][:, :, None] > 0)
    eye = torch.eye(N, dtype=torch.bool, device=dev)[None]
    clock = u["contact_s"] if "contact_s" in u else torch.zeros_like(u["men"])
    slot = torch.arange(N, device=dev)
    ck, ci = clock[:, None, :], clock[:, :, None]
    first = (ck > ci + 0.5 * dt) | (((ck - ci).abs() <= 0.5 * dt) & (slot[None, None, :] < slot[None, :, None]))
    over = pw["gap"] < 0                                                   # the boxes overlap (friends too)
    A = same & ~eye & over & first & form[:, :, None] & form[:, None, :] & eng[:, None, :]   # k (column) ahead of i
    b = u["b"] * (torch.pi / 180)
    sb, cb = torch.sin(b), torch.cos(b)
    rx = u["x"][:, None, :] - u["x"][:, :, None]
    rz = u["z"][:, None, :] - u["z"][:, :, None]

    def span(lx, lz, half):
        """k's box along i's line (lx, lz) within i's half extent `half` [B, N, 1]: (lo, hi) [B, N, N]."""
        off = rx * lx + rz * lz
        ext = (pw["front"][:, None, :] / 2 * (cb[:, None, :] * lx - sb[:, None, :] * lz).abs()
               + pw["depth"][:, None, :] / 2 * (sb[:, None, :] * lx + cb[:, None, :] * lz).abs())
        return torch.maximum(off - ext, -half), torch.minimum(off + ext, half)

    def hull(lo, hi, half, A):
        """(the ones ahead's span along the line [B, N, 2] - empty: lo > hi -, its share of the side [B, N])"""
        big = torch.full_like(lo, 1e9)
        lo_all = torch.where(A, lo, big).min(2).values
        hi_all = torch.where(A, hi, -big).max(2).values
        share = torch.where(A.any(2), ((hi_all - lo_all).clamp(min=0) / (2 * half[:, :, 0])).clamp(max=1.0),
                            torch.zeros_like(lo_all))
        return torch.stack((lo_all, hi_all), dim=2), share

    # i's front line (cos b, -sin b) - as geometry.blocked_share, along i's own front - and its facing (its flanks)
    half_f = (pw["front"] / 2).clamp(min=1e-6)[:, :, None]
    half_s = (pw["depth"] / 2).clamp(min=1e-6)[:, :, None]
    lo, hi = span(cb[:, :, None], -sb[:, :, None], half_f)
    lo_s, hi_s = span(sb[:, :, None], cb[:, :, None], half_s)
    own = (hi - lo).clamp(min=0)                                           # each friend's cover of i's front, m
    A = A & ((own > 0) | (hi_s > lo_s))
    off_f = rx * sb[:, :, None] + rz * cb[:, :, None]                      # k's centre along i's facing
    off_l = rx * cb[:, :, None] - rz * sb[:, :, None]                      # ... along i's front line
    sides = (hull(lo, hi, half_f, A & (off_f >= -half_s)), hull(lo_s, hi_s, half_s, A & (off_l >= 0)),
             hull(lo_s, hi_s, half_s, A & (off_l <= 0)), hull(lo, hi, half_f, A & (off_f <= half_s)))
    cover = torch.stack([sh for _, sh in sides], dim=2)
    if side_i is not None:
        # per pair: the part of i's side facing j (j's span along that side's line, every unit's span is lo / hi)
        g = side_i.clamp(0, 3)
        bound = torch.stack([bd for bd, _ in sides], dim=2)                 # [B, N, 4, 2]
        h_lo = bound[..., 0].gather(2, g)
        h_hi = bound[..., 1].gather(2, g)
        thru = (g == 0) | (g == 3)
        s_lo, s_hi = torch.where(thru, lo, lo_s), torch.where(thru, hi, hi_s)
        seg = (s_hi - s_lo).clamp(min=0)
        got = (torch.minimum(h_hi, s_hi) - torch.maximum(h_lo, s_lo)).clamp(min=0)
        whole = cover.gather(2, g)
        cover = torch.where(seg > 1e-3, (got / seg.clamp(min=1e-3)).clamp(max=1.0), whole)
    k = torch.where(A, own, torch.full_like(own, -1.0)).argmax(2)
    ahead = torch.where(A.any(2), k, torch.full_like(k, -1))
    # behind: ahead touches j too, and i is not nearer to j than ahead's near edge on ahead's side of j
    ka = ahead.clamp(min=0)
    side_k = side_j.gather(1, ka[:, :, None].expand(-1, -1, N))            # (i, j): ahead's side of j
    k_on_j = (contact | contact.transpose(1, 2)).gather(1, ka[:, :, None].expand(-1, -1, N))
    bj = b[:, None, :]
    fjx, fjz = torch.sin(bj), torch.cos(bj)                                # j's facing
    ljx, ljz = torch.cos(bj), -torch.sin(bj)                               # j's front line
    thru = (side_k == 0) | (side_k == 3)
    ax_x = torch.where(thru, fjx, ljx).expand(-1, N, -1)
    ax_z = torch.where(thru, fjz, ljz).expand(-1, N, -1)
    xk, zk = u["x"].gather(1, ka)[:, :, None], u["z"].gather(1, ka)[:, :, None]
    xj, zj = u["x"][:, None, :], u["z"][:, None, :]
    pk = (xk - xj) * ax_x + (zk - zj) * ax_z
    sgn = torch.where(pk >= 0, torch.ones_like(pk), -torch.ones_like(pk))  # the side's outward normal: towards ahead
    pk = pk * sgn
    pi = ((u["x"][:, :, None] - xj) * ax_x + (u["z"][:, :, None] - zj) * ax_z) * sgn
    # ahead's half extent along that normal (its box: front across its facing, depth along it)
    bk = b.gather(1, ka)[:, :, None]
    fkx, fkz = torch.sin(bk), torch.cos(bk)
    frk, dpk = pw["front"].gather(1, ka)[:, :, None], pw["depth"].gather(1, ka)[:, :, None]
    ek = frk / 2 * (fkz * ax_x - fkx * ax_z).abs() + dpk / 2 * (fkx * ax_x + fkz * ax_z).abs()
    behind = (ahead >= 0)[:, :, None] & k_on_j & contact & (pi >= pk - ek)
    return cover, ahead, behind, side_k
