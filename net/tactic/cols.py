"""The tactic network's extra row columns (review_attack_v23 M; tmp/tactic_slice/design_attack_v23.md section 6) - one
formula for the label probe's buffer (tactic/learn.features) and the network (tactic/heads.net_inputs): plain input,
never the teacher's verdicts (the X control's three columns come from the caller).

    COLS         the row columns the probe reads (INPUTS of learn.py pick among them); PAIR_EXTRA the pair columns
    D2           the network's input: THE INPUT IS D2 (review M, the second mend read) - the row's right, forward, v_right,
                 v_forward, mt + D's coming columns + D2's; E2's pair columns not taken
    ingredients(fx)   D's and D2's row columns, the E pair columns (its doc)
    row_cols(fx, x=None)   [R, K, len(COLS)]
"""
import torch

from net.tactic import orders as OR
from net.tactic import rows as RW

COLS = ("right", "forward", "v_right", "v_forward", "mt",
        "d_coming_n", "d_coming_dist", "d_comes_at", "d_comes_at_melee", "d_gap_melee_still",
        "pinned", "waited", "free_nc",
        "d2_free_coming_n", "d2_free_coming_gap", "d2_gap_melee_nr")
PAIR_EXTRA = ("near", "e_comes_at_g", "e_comes_at_other", "e_other_melee", "e_gap_my_melee",
              "e2_free_comes_at_g", "e2_free_comes_gap")
_D2 = (0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 13, 14)          # (C, D's coming columns, the gap 'in melee, not routing', D2's two)


def ingredients(fx):
    """(review_attack_v23 M: the ingredients, never the teacher's verdicts - no threshold of the tactic teacher; the
    only rule is orders.comes, the unit script's own fact of the lower level's input) ([R, K, 5] row columns, [R, G, K,
    4] pair columns):
      rows (mine and enemies alike; 'opposing' = the other side, routers not counted as coming): the count of opposing
      units coming at the row (cap 3, / 3); the centre distance of the nearest of them (/ 240, 1 with none, capped at
      1); the row comes at an opposing unit; one of those is in melee; the smallest edge gap to an opposing unit in
      melee and not moving (the row's mv flag off) (/ 40, 1 beyond)
      pairs (g, k): k comes at g; k comes at another unit of my group; one of those is in melee; the edge gap of k to my
      nearest unit standing in melee (mine, in melee, mv off, not routing) (/ 40, 1 beyond)
      D2 / E2 (review M, the first mend read): rows - the comers NOT in melee (count, smallest edge gap / 240) and the
      edge gap to an opposing unit in melee and not routing; pairs - k, out of melee, comes at g, and its edge gap
    -> (D's rows [R, K, 5], D2's rows [R, K, 3], pairs [R, G, K, 6])"""
    R, G, K = fx.R, fx.G, fx.K
    cm = OR.comes(fx)                                                              # [R, i, j] j comes at i
    x = fx.rr["rows"]
    mv = (x[..., RW.RI["mv"]] > 0.5) & fx.mask
    own = fx.mask & ~fx.enemy
    opp = (own[:, :, None] & fx.enemy[:, None, :]) | (fx.enemy[:, :, None] & own[:, None, :])
    at = cm & opp & ~fx.r[:, None, :]                                              # [R, i, j] opposing j comes at i
    n = at.float().sum(2).clamp(max=3.0) / 3.0
    dn = torch.where(at, fx.dist, torch.full_like(fx.dist, OR.INF)).amin(2)
    dn = torch.where(torch.isfinite(dn), (dn / 240.0).clamp(max=1.0), torch.ones_like(dn))
    goes = (cm & opp).transpose(1, 2)                                              # [R, i, j] i comes at j
    c3 = goes.any(2).float()
    c4 = (goes & fx.m[:, None, :]).any(2).float()
    still = fx.m & ~mv
    gm = torch.where(opp & still[:, None, :], fx.gap, torch.full_like(fx.gap, OR.INF)).amin(2)
    c5 = torch.where(torch.isfinite(gm), (gm / 40.0).clamp(0.0, 1.0), torch.ones_like(gm))
    rows = torch.stack([n, dn, c3, c4, c5], -1) * fx.mask[..., None]
    # D2 (review M, the first mend read): the comers NOT in melee - count (cap 3, / 3) and their smallest EDGE gap (/ 240,
    # 1 with none); the smallest edge gap to an opposing unit in melee and not routing (/ 40, 1 beyond)
    af = at & ~fx.m[:, None, :]
    nf = af.float().sum(2).clamp(max=3.0) / 3.0
    gf = torch.where(af, fx.gap, torch.full_like(fx.gap, OR.INF)).amin(2)
    gf = torch.where(torch.isfinite(gf), (gf / 240.0).clamp(0.0, 1.0), torch.ones_like(gf))
    gr = torch.where(opp & (fx.m & ~fx.r)[:, None, :], fx.gap, torch.full_like(fx.gap, OR.INF)).amin(2)
    gr = torch.where(torch.isfinite(gr), (gr / 40.0).clamp(0.0, 1.0), torch.ones_like(gr))
    rows2 = torch.stack([nf, gf, gr], -1) * fx.mask[..., None]
    # pairs (g, k)
    gi = fx.g_row[:, :, None].expand(R, G, K)
    p1 = cm.gather(1, gi)                                                          # [R, G, K] k comes at g
    mine = fx.mine & ~fx.r
    other = mine[:, :, None].expand(R, K, G).clone()                               # [R, a, G] a is mine and not g
    other.scatter_(1, fx.g_row[:, None, :], False)
    oa = other.transpose(1, 2)                                                     # [R, G, a]
    ca = cm & fx.enemy[:, None, :]                                                 # [R, a, k] enemy k comes at a
    p2 = (oa[..., None] & ca[:, None]).any(2)                                      # [R, G, K]
    p3 = (oa[..., None] & ca[:, None] & fx.m[:, None, :, None]).any(2)
    stand = mine & fx.m & ~mv
    gk = torch.where(stand[:, :, None], fx.gap, torch.full_like(fx.gap, OR.INF)).amin(1)          # [R, K]
    gk = torch.where(torch.isfinite(gk), (gk / 40.0).clamp(0.0, 1.0), torch.ones_like(gk))
    p5 = p1 & fx.enemy[:, None, :] & ~fx.r[:, None, :] & ~fx.m[:, None, :]          # k, out of melee, comes at g
    g5 = torch.where(p5, (fx.g_gap / 240.0).clamp(0.0, 1.0), torch.ones_like(fx.g_gap))
    pair = torch.stack([p1.float(), p2.float(), p3.float(), gk[:, None, :].expand(R, G, K), p5.float(), g5], -1)
    return rows, rows2, pair


D2 = tuple(COLS[i] for i in _D2)        # the network's extra row columns (review_attack_v23 M: THE INPUT IS D2)


def row_cols(fx, x=None):
    """[R, K, len(COLS)] the row columns COLS of a tick's facts (features' and the network's one formula): x [R, K, 3]
    the X control's teacher columns (pinned, waited, free_nc; None: zeros - not an input of the network)."""
    rows, mask = fx.rr["rows"], fx.rr["mask"]
    RI = RW.RI
    cap = float(OR.c_orders()["net"]["mt_cap"])
    ing, ing2, _ = ingredients(fx)
    if x is None:
        x = torch.zeros(mask.shape + (3,), device=rows.device)
    return torch.cat([torch.stack([rows[..., RI["right"]], rows[..., RI["forward"]], rows[..., RI["v_right"]],
                                   rows[..., RI["v_forward"]], fx.mt.clamp(max=int(cap)).float() / cap], -1), ing,
                      x, ing2], -1) * mask[..., None]
