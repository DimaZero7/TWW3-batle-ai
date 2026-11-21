"""The tactic network's extra row columns (review_attack_v23 M; tmp/tactic_slice/design_attack_v23.md section 6) - one
formula for the label probe's buffer (tactic/learn.features) and the network (tactic/heads.net_inputs): plain input,
never the teacher's verdicts (the X control's three columns come from the caller).

    COLS         the row columns the probe reads (INPUTS of learn.py pick among them); PAIR_EXTRA the pair columns
    D2           the network's input: THE INPUT IS D2 (review M, the second mend read) - the row's right, forward, v_right,
                 v_forward, mt + D's coming columns + D2's; E2's pair columns not taken
    ingredients(fx)   D's and D2's row columns, the E pair columns (its doc)
    row_cols(fx, x=None)   [R, K, len(COLS)]
    S2 (drill T0-2 'side of the strike', tmp/drill2/design_merged.md, the rule's numbers config orders.side): the raw
    ingredients of 'open' and 'way clear' - quantities, never the teacher's verdicts. Rows (appended to COLS as D2 was:
    D2's indices stand) SIDE_ROW - the lateral clearance left / right (lateral(): the nearest other non-routing unit of
    the row's own side in its band, / band_lat_m, 1 none) and the least off-facing angle of a standing unit of mine in
    melee within the teacher's touch (anvil_off: frontal()'s fact; / 180, 1 none); pairs SIDE_PAIR (side_pairs(fx) [R, G, K, 8]) - per side of k (its left, its right) the way g -> P's
    clearance to other enemies (/ clear_enemy_m) and to my other units (/ clear_own_m), both capped at 3 (3 none), the
    extra path (|gP| + |Pk| - |gk|) / pos_m and |gP| / pos_m. ways(fx) the same in metres (the teacher's one formula)
"""
import torch

from net.tactic import cfg
from net.tactic import orders as OR
from net.tactic import rows as RW

SIDE_ROW = ("s2_lat_left", "s2_lat_right", "s2_anvil_off")
SIDE_PAIR = tuple(f"s2_{sd}_{n}" for sd in ("l", "r") for n in ("seg_enemy", "seg_own", "extra", "dist_p"))
SEG_CAP = 3.0
COLS = ("right", "forward", "v_right", "v_forward", "mt",
        "d_coming_n", "d_coming_dist", "d_comes_at", "d_comes_at_melee", "d_gap_melee_still",
        "pinned", "waited", "free_nc",
        "d2_free_coming_n", "d2_free_coming_gap", "d2_gap_melee_nr") + SIDE_ROW
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
                      x, ing2, side_rows(fx)], -1) * mask[..., None]


# ------------------------------------------------------------------ S2: drill T0-2's ingredients (module doc)
def c_side():
    return OR.c_orders()["side"]


def _sincos(b):
    th = torch.deg2rad(b)
    return torch.sin(th), torch.cos(th)


def lateral(fx):
    """[R, K, 2] metres (inf none) - each row's lateral clearance (LEFT, RIGHT of its own facing): the nearest other
    non-routing unit of the row's own side whose centre is within band_along_m along the row's facing and 0 < lateral
    offset <= band_lat_m (tmp/drill2/flank_feas.py: for an enemy row its other enemies)."""
    c = c_side()
    sn, cs = _sincos(fx.b)
    ds = fx.s[:, None, :] - fx.s[:, :, None]                                       # [R, k, j] j from k
    df = fx.f[:, None, :] - fx.f[:, :, None]
    along = ds * sn[:, :, None] + df * cs[:, :, None]
    lat = ds * cs[:, :, None] - df * sn[:, :, None]
    same = (fx.enemy[:, :, None] & fx.enemy[:, None, :]) | (fx.mine[:, :, None] & fx.mine[:, None, :])
    eye = torch.eye(fx.K, dtype=torch.bool, device=fx.s.device)[None]
    ok = same & ~eye & (fx.mask & ~fx.r)[:, None, :] & (along.abs() <= float(c["band_along_m"]))
    out = []
    for sg in (-1.0, 1.0):
        q = ok & (sg * lat > 0) & (sg * lat <= float(c["band_lat_m"]))
        out.append(torch.where(q, sg * lat, torch.full_like(lat, OR.INF)).amin(2))
    return torch.stack(out, -1)


def anvil_off(fx, touch_m=None):
    """[R, K] degrees (NaN none): the least off-facing angle (|bearing from row k to it - k's facing|) of a standing
    unit of mine in melee within touch_m of k, edge to edge (None: scripts.TOUCH_M) - frontal()'s own fact, the same
    formula (drill T0-2 stage B: the label probe showed the nearest-opposing-unit form missed the teacher's test,
    tmp/t02b/diag.py)."""
    from net.unit import scripts as SC
    hold = fx.mine & ~fx.r & fx.m
    t = hold[:, :, None] & (fx.gap <= float(SC.TOUCH_M if touch_m is None else touch_m))   # [R, a, k]
    ds = fx.s[:, :, None] - fx.s[:, None, :]                                       # a from k
    df = fx.f[:, :, None] - fx.f[:, None, :]
    brg = torch.rad2deg(torch.atan2(ds, df))
    off = (torch.remainder(brg - fx.b[:, None, :] + 180.0, 360.0) - 180.0).abs()
    least = torch.where(t, off, torch.full_like(off, OR.INF)).amin(1)
    return torch.where(t.any(1), least, torch.full_like(least, float("nan")))


def frontal(fx, touch_m):
    """[R, K] a standing unit of mine in melee within touch_m of row k (edge to edge) has its centre within
    frontal_deg of k's facing (the anvil in k's front sector: 'pinned frontally', design_opus 1)."""
    hold = fx.mine & ~fx.r & fx.m
    t = hold[:, :, None] & (fx.gap <= float(touch_m))                              # [R, a, k]
    ds = fx.s[:, :, None] - fx.s[:, None, :]                                       # a from k
    df = fx.f[:, :, None] - fx.f[:, None, :]
    brg = torch.rad2deg(torch.atan2(ds, df))
    off = (torch.remainder(brg - fx.b[:, None, :] + 180.0, 360.0) - 180.0).abs()
    return (t & (off <= float(c_side()["frontal_deg"]))).any(1)


def _seg_dist(ps, pf, as_, af, bs, bf):
    """Distance of points (ps, pf) to the segments a -> b (broadcast)."""
    ds, df = bs - as_, bf - af
    L2 = (ds * ds + df * df).clamp(min=1e-9)
    t = (((ps - as_) * ds + (pf - af) * df) / L2).clamp(0.0, 1.0)
    return torch.sqrt((ps - as_ - t * ds) ** 2 + (pf - af - t * df) ** 2)


def ways(fx):
    """{ps, pf [R, G, K, 2] P of my unit g at row k's (LEFT, RIGHT): k's centre + side x (front_k / 2 + off_m +
    depth_g / 2) across k's facing; seg_enemy, seg_own [R, G, K, 2] the least centre distance of the way g -> P to
    another non-routing enemy (not k) / to another non-routing unit of mine (not g, not k), inf none; extra (|gP| +
    |Pk| - |gk|), dist_p |gP|} metres - the teacher's and the columns' one formula (module doc)."""
    c = c_side()
    R, G, K = fx.R, fx.G, fx.K
    dev = fx.s.device
    sn, cs = _sincos(fx.b)                                                         # [R, K]
    gs, gf = fx.s.gather(1, fx.g_row), fx.f.gather(1, fx.g_row)                    # [R, G]
    dg = fx.depth.gather(1, fx.g_row)
    half = fx.front[:, None, :] / 2 + float(c["off_m"]) + dg[:, :, None] / 2       # [R, G, K]
    sg = torch.tensor([-1.0, 1.0], device=dev)                                     # (LEFT, RIGHT of k)
    ps = fx.s[:, None, :, None] + sg * half[..., None] * cs[:, None, :, None]      # [R, G, K, 2]
    pf = fx.f[:, None, :, None] - sg * half[..., None] * sn[:, None, :, None]
    A = lambda t: t[:, :, None, None, None]                                        # noqa: E731 [R, G, 1, 1, 1]
    J = lambda t: t[:, None, None, None, :]                                        # noqa: E731 [R, 1, 1, 1, J]
    d = _seg_dist(J(fx.s), J(fx.f), A(gs), A(gf), ps[..., None], pf[..., None])   # [R, G, K, 2, J]
    eye = torch.eye(K, dtype=torch.bool, device=dev)
    not_k = ~eye[None, None, :, None, :]                                           # j != k
    gone = torch.zeros((R, G, K), dtype=torch.bool, device=dev)
    gone.scatter_(2, fx.g_row[..., None], True)                                    # [R, G, j] j is g
    stand = fx.mask & ~fx.r
    en = J(fx.enemy & stand) & not_k
    ow = J(fx.mine & stand) & not_k & ~gone[:, :, None, None, :]
    inf = torch.full_like(d, OR.INF)
    seg_e = torch.where(en, d, inf).amin(-1)
    seg_o = torch.where(ow, d, inf).amin(-1)
    gp = torch.sqrt((ps - gs[..., None, None]) ** 2 + (pf - gf[..., None, None]) ** 2)
    pk = torch.sqrt((ps - fx.s[:, None, :, None]) ** 2 + (pf - fx.f[:, None, :, None]) ** 2)
    extra = gp + pk - fx.g_dist[..., None]
    return {"ps": ps, "pf": pf, "seg_enemy": seg_e, "seg_own": seg_o, "extra": extra, "dist_p": gp}


def side_rows(fx):
    """[R, K, len(SIDE_ROW)] (module doc)."""
    c = c_side()
    lat = (lateral(fx) / float(c["band_lat_m"])).clamp(max=1.0)
    from net.tactic import pin as PN
    off = anvil_off(fx, PN.teach_opts()["touch"])                                  # (the teacher's touch distance)
    off = torch.where(torch.isfinite(off), off / 180.0, torch.ones_like(off))
    return torch.cat([lat, off[..., None]], -1) * fx.mask[..., None]


def side_pairs(fx, w=None):
    """[R, G, K, len(SIDE_PAIR)] (module doc); w - ways(fx) if already built."""
    c = c_side()
    w = w or ways(fx)
    pos = float(cfg()["pos_m"])
    se = (w["seg_enemy"] / float(c["clear_enemy_m"])).clamp(max=SEG_CAP)
    so = (w["seg_own"] / float(c["clear_own_m"])).clamp(max=SEG_CAP)
    per = [torch.stack([se[..., i], so[..., i], w["extra"][..., i] / pos, w["dist_p"][..., i] / pos], -1)
           for i in range(2)]
    return torch.cat(per, -1) * fx.mask[:, None, :, None]
