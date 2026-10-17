"""The neighbour scans of the step, each a GPU kernel (Triton) with the plain torch code beside it - the same numbers:

  contact_scan  hook.py: each man's lowest and highest enemy unit within his reach (the units' contact)
  melee_scan    melee.py: each man's target - the one he fights kept while in reach, else the nearest (his unit's
                attack target first) - and the man he closes on (within reach + the tether)
  push_scan     move.py: the bodies' push (men closer than the sum of their radii push apart, half each)

A kernel reads each man's 3 x 3 cells straight from the cell table (hook.Near.table: cells.Table) - the men of a slot
side by side in the sorted order - and skips the empty slots; torch builds the neighbour lists [B, M, 9 x cap]
(hook.Near.cand) and computes every slot. Each function takes `use_triton` (None: on CUDA when Triton is there;
kernels.py)."""
import torch

from tools.nn.fighters import cells
from tools.nn.fighters import kernels as K
from tools.nn.fighters.kernels import _triton_ok, triton

if triton is not None:
    import triton.language as tl

    @triton.jit
    def _rows(M, MB: tl.constexpr):
        """The program's men: flat index b x M + m [MB], their battle b and place m."""
        rows = tl.program_id(0) * MB + tl.arange(0, MB)
        b = rows // M
        return rows, b, rows - b * M

    @triton.jit
    def _lanes(start_ptr, count_ptr, man_ptr, b, mx, mz, live, H, c, P1: tl.constexpr, P2: tl.constexpr,
               CAP: tl.constexpr, BLOCK: tl.constexpr):
        """[MB, BLOCK] the 3 x 3 cells' men round each (mx, mz) [MB] where live [MB] (lane l = cell x CAP + k:
        cells.Table.query's order): at - their index in the sorted order, ok - a man there, man - his place."""
        lanes = tl.arange(0, BLOCK)[None, :]
        cell = lanes // CAP
        k = lanes - cell * CAP
        ix = tl.floor(mx / c).to(tl.int64)[:, None]
        iz = tl.floor(mz / c).to(tl.int64)[:, None]
        slot = b[:, None] * H + (((ix + (cell // 3 - 1)) * P1 + (iz + (cell % 3 - 1)) * P2) & (H - 1))
        on = live[:, None] & (lanes < 9 * CAP)
        n = tl.load(count_ptr + slot, mask=on, other=0)
        ok = on & (k < n)
        at = tl.load(start_ptr + slot, mask=ok, other=0) + k
        man = tl.load(man_ptr + at, mask=ok, other=-1)
        return at, ok, man, lanes

    @triton.jit
    def _contact_kernel(start_ptr, count_ptr, man_ptr, sx_ptr, sz_ptr, slive_ptr, suid_ptr,
                        px_ptr, pz_ptr, live_ptr, uid_ptr, side_ptr, rad_ptr, lo_ptr, hi_ptr,
                        BM, M, N, H, c, combat, P1: tl.constexpr, P2: tl.constexpr, CAP: tl.constexpr,
                        BLOCK: tl.constexpr, MB: tl.constexpr):
        rows, b, m = _rows(M, MB)
        ex = rows < BM
        me = ex & (tl.load(live_ptr + rows, mask=ex, other=0) != 0)
        mx = tl.load(px_ptr + rows, mask=me, other=0.0)
        mz = tl.load(pz_ptr + rows, mask=me, other=0.0)
        sm = tl.load(side_ptr + b * N + tl.load(uid_ptr + rows, mask=me, other=0), mask=me, other=0)
        at, ok, man, lanes = _lanes(start_ptr, count_ptr, man_ptr, b, mx, mz, me, H, c, P1, P2, CAP, BLOCK)
        ok = ok & (man != m[:, None]) & (tl.load(slive_ptr + at, mask=ok, other=0) != 0)
        cu = tl.load(suid_ptr + at, mask=ok, other=0)
        bn = b[:, None] * N
        ok = ok & (tl.load(side_ptr + bn + cu, mask=ok, other=0) != sm[:, None])
        dx = tl.load(sx_ptr + at, mask=ok, other=0.0) - mx[:, None]
        dz = tl.load(sz_ptr + at, mask=ok, other=0.0) - mz[:, None]
        d = tl.sqrt_rn(dx * dx + dz * dz + 1e-9)
        touch = ok & (d <= combat + tl.load(rad_ptr + bn + cu, mask=ok, other=0.0))
        tl.store(lo_ptr + rows, tl.min(tl.where(touch, cu, N), axis=1), mask=ex)
        tl.store(hi_ptr + rows, tl.max(tl.where(touch, cu, -1), axis=1), mask=ex)

    @triton.jit
    def _melee_kernel(start_ptr, count_ptr, man_ptr, sx_ptr, sz_ptr, scl_ptr, suid_ptr,
                      px_ptr, pz_ptr, up_ptr, uid_ptr, side_ptr, rad_ptr, permit_ptr, tgt_ptr,
                      pref_ptr, keep_ptr, has_ptr, best_ptr, hase_ptr, eng_ptr,
                      BM, M, N, H, c, combat, tether, P1: tl.constexpr, P2: tl.constexpr, CAP: tl.constexpr,
                      BLOCK: tl.constexpr, MB: tl.constexpr):
        rows, b, m = _rows(M, MB)
        ex = rows < BM
        me = ex & (tl.load(up_ptr + rows, mask=ex, other=0) != 0)
        mx = tl.load(px_ptr + rows, mask=me, other=0.0)
        mz = tl.load(pz_ptr + rows, mask=me, other=0.0)
        um = tl.load(uid_ptr + rows, mask=me, other=0)
        sm = tl.load(side_ptr + b * N + um, mask=me, other=0)
        tg = tl.load(tgt_ptr + rows, mask=me, other=-1)
        pu = tl.load(pref_ptr + rows, mask=me, other=-1)
        at, ok, man, lanes = _lanes(start_ptr, count_ptr, man_ptr, b, mx, mz, me, H, c, P1, P2, CAP, BLOCK)
        ok = ok & (man != m[:, None]) & (tl.load(scl_ptr + at, mask=ok, other=0) != 0)
        cu = tl.load(suid_ptr + at, mask=ok, other=0)
        bn = b[:, None] * N
        ok = ok & (tl.load(permit_ptr + (bn + um[:, None]) * N + cu, mask=ok, other=0) != 0)
        ok = ok & (tl.load(side_ptr + bn + cu, mask=ok, other=0) != sm[:, None])
        dx = tl.load(sx_ptr + at, mask=ok, other=0.0) - mx[:, None]
        dz = tl.load(sz_ptr + at, mask=ok, other=0.0) - mz[:, None]
        d = tl.sqrt_rn(dx * dx + dz * dz + 1e-9)
        reach = combat + tl.load(rad_ptr + bn + cu, mask=ok, other=0.0)
        inr = ok & (d <= reach)
        eng = ok & (d <= reach + tether)
        pref = (cu == pu[:, None]).to(tl.float32)
        bi = tl.argmin(tl.where(inr, d - 2.0 * pref, 1e9), axis=1)
        ei = tl.argmin(tl.where(eng, d - 2.0 * pref, 1e9), axis=1)
        tl.store(keep_ptr + rows, tl.max((inr & (man == tg[:, None])).to(tl.int8), axis=1), mask=ex)
        tl.store(has_ptr + rows, tl.max(inr.to(tl.int8), axis=1), mask=ex)
        tl.store(best_ptr + rows, tl.sum(tl.where(lanes == bi[:, None], man, 0), axis=1), mask=ex)
        tl.store(hase_ptr + rows, tl.max(eng.to(tl.int8), axis=1), mask=ex)
        tl.store(eng_ptr + rows, tl.sum(tl.where(lanes == ei[:, None], man, 0), axis=1), mask=ex)

    @triton.jit
    def _push_kernel(start_ptr, count_ptr, man_ptr, sax_ptr, saz_ptr, smov_ptr, srad_ptr,
                     px_ptr, pz_ptr, ax_ptr, az_ptr, mov_ptr, rad_ptr, sx_ptr, sz_ptr,
                     BM, M, H, c, P1: tl.constexpr, P2: tl.constexpr, CAP: tl.constexpr, BLOCK: tl.constexpr,
                     MB: tl.constexpr):
        rows, b, m = _rows(M, MB)
        ex = rows < BM
        me = ex & (tl.load(mov_ptr + rows, mask=ex, other=0) != 0)
        at, ok, man, lanes = _lanes(start_ptr, count_ptr, man_ptr, b, tl.load(px_ptr + rows, mask=me, other=0.0),
                                    tl.load(pz_ptr + rows, mask=me, other=0.0), me, H, c, P1, P2, CAP, BLOCK)
        ok = ok & (man != m[:, None]) & (tl.load(smov_ptr + at, mask=ok, other=0) != 0)
        rx = tl.load(ax_ptr + rows, mask=me, other=0.0)[:, None] - tl.load(sax_ptr + at, mask=ok, other=0.0)
        rz = tl.load(az_ptr + rows, mask=me, other=0.0)[:, None] - tl.load(saz_ptr + at, mask=ok, other=0.0)
        d = tl.maximum(tl.sqrt_rn(rx * rx + rz * rz), 1e-3)
        need = tl.load(rad_ptr + rows, mask=me, other=0.0)[:, None] + tl.load(srad_ptr + at, mask=ok, other=0.0)
        over = tl.where(ok & (d < need), (need - d) / 2, 0.0)
        tl.store(sx_ptr + rows, tl.sum(over * rx / d, axis=1), mask=ex)
        tl.store(sz_ptr + rows, tl.sum(over * rz / d, axis=1), mask=ex)


    _C = ("P1", "P2", "CAP", "BLOCK")
    K.register("contact", _contact_kernel, (torch.int64, torch.int64), _C, (4, 2))
    K.register("melee", _melee_kernel, (torch.int8, torch.int8, torch.int64, torch.int8, torch.int64), _C, (2, 1))
    K.register("push", _push_kernel, (torch.float32, torch.float32), _C, (2, 2))


def _table(near):
    t = near.table
    return t.start, t.count, t.man


def _consts(near):
    return (cells._P1, cells._P2, near.cap, 1 << (9 * near.cap - 1).bit_length())


def _i8(t):
    return t.to(torch.int8).contiguous()


def _by_unit(t_u, c_uid):
    """[B, N] by unit at c_uid [B, Q, K]."""
    B = t_u.shape[0]
    return t_u.gather(1, c_uid.reshape(B, -1)).reshape(c_uid.shape)


def contact_scan(near, live, uid, side_u, rad_u, combat, use_triton=None):
    """(lo, hi) [B, M]: the lowest / highest enemy unit with a man of it within combat + his radius of each living man
    (live [B, M]: alive, not lying; both men live), N / -1 none."""
    B, M = live.shape
    N = side_u.shape[1]
    if _triton_ok(live, use_triton):
        t = near.table
        live8 = _i8(live)
        ins = [*_table(near), t.sorted_of(near.px), t.sorted_of(near.pz), t.sorted_of(live8),
               t.sorted_of(uid.to(torch.int32)), near.px, near.pz, live8, uid, side_u, rad_u.float()]
        lo, hi = K.launch("contact", B * M, ins, (B * M, M, N, t.H), (t.c, combat), _consts(near))
        return lo.reshape(B, M), hi.reshape(B, M)
    cand, c_uid = near.cand, near.c_uid
    side_m = side_u.gather(1, uid)
    touch = (live[:, :, None] & (cand >= 0) & cells.gather2(live, cand)
             & (_by_unit(side_u, c_uid) != side_m[:, :, None]) & (near.d <= combat + _by_unit(rad_u, c_uid)))
    lo = torch.where(touch, c_uid, torch.full_like(c_uid, N)).amin(2)
    hi = torch.where(touch, c_uid, torch.full_like(c_uid, -1)).amax(2)
    return lo, hi


def melee_rows(near, rows, up, c_live, uid, side_u, rad_u, permit, combat, tether):
    """torch: for the men of rows [B, L] (None: all) their neighbours cand [B, L, K], distances d, units c_uid, the
    enemies in reach inreach and within reach + the tether engage (melee.resolve's rules)."""
    B, M = up.shape
    N = side_u.shape[1]
    if rows is None:
        cand, d, c_uid = near.cand, near.d, near.c_uid
        me_up, me_uid = up, uid
    else:
        qx, qz = near.px.gather(1, rows), near.pz.gather(1, rows)
        cand = near.table.query(qx, qz, near.cap)
        cand = torch.where(cand == rows[:, :, None], torch.full_like(cand, -1), cand)
        d = torch.sqrt((cells.gather2(near.px, cand) - qx[:, :, None]) ** 2
                       + (cells.gather2(near.pz, cand) - qz[:, :, None]) ** 2 + 1e-9)
        c_uid = cells.gather2(uid, cand)
        me_up, me_uid = up.gather(1, rows), uid.gather(1, rows)
    ok = (cand >= 0) & cells.gather2(c_live, cand)
    pair = (me_uid[:, :, None] * N + c_uid).reshape(B, -1)
    ok = ok & permit.reshape(B, N * N).gather(1, pair).reshape(cand.shape)
    ok = ok & (_by_unit(side_u, c_uid) != side_u.gather(1, me_uid)[:, :, None])          # enemies only
    reach = combat + _by_unit(rad_u, c_uid)
    me = me_up[:, :, None]
    return cand, d, c_uid, me & ok & (d <= reach), me & ok & (d <= reach + tether)


def melee_scan(near, up, c_live, uid, side_u, rad_u, permit, tgt, pref_unit, combat, tether, use_triton=None):
    """Each man's (keep, has, best, has_e, eng) [B, M]: keep - the man he fights (tgt) is still in his reach; has - an
    enemy man in reach (of a unit pair permit [B, N, N] allows); best - the nearest of them (a man of pref_unit, his
    unit's attack target, counts 2 m nearer); eng / has_e - the same within reach + the tether. up: he can strike
    (alive, not lying); c_live: he can be struck."""
    B, M = up.shape
    N = side_u.shape[1]
    if _triton_ok(up, use_triton):
        t = near.table
        ins = [*_table(near), t.sorted_of(near.px), t.sorted_of(near.pz), t.sorted_of(_i8(c_live)),
               t.sorted_of(uid.to(torch.int32)), near.px, near.pz, _i8(up), uid, side_u, rad_u.float(), _i8(permit),
               tgt, pref_unit]
        out = K.launch("melee", B * M, ins, (B * M, M, N, t.H), (t.c, combat, tether), _consts(near))
        keep, has, best, hase, eng = (v.reshape(B, M) for v in out)
        return keep.bool(), has.bool(), best, hase.bool(), eng
    cand, d, c_uid, inreach, engage = melee_rows(near, None, up, c_live, uid, side_u, rad_u, permit, combat, tether)
    keep = (inreach & (cand == tgt[:, :, None])).any(2)
    pref = (c_uid == pref_unit[:, :, None]).float()
    big = torch.full_like(d, 1e9)
    best = cand.gather(2, torch.where(inreach, d - 2.0 * pref, big).argmin(2)[:, :, None]).squeeze(2)
    eng = cand.gather(2, torch.where(engage, d - 2.0 * pref, big).argmin(2)[:, :, None]).squeeze(2)
    return keep, inreach.any(2), best, engage.any(2), eng


def push_scan(near, ax, az, mov, rad, use_triton=None):
    """(sx, sz) [B, M]: the bodies' push on each man who moves (mov) at (ax, az): from every moving neighbour (of the
    step's cell table) closer than the sum of their radii (rad [B, M]), half the overlap each, apart."""
    B, M = mov.shape
    if _triton_ok(mov, use_triton):
        t = near.table
        ax, az, rad, mov8 = ax.float(), az.float(), rad.float(), _i8(mov)
        ins = [*_table(near), t.sorted_of(ax), t.sorted_of(az), t.sorted_of(mov8), t.sorted_of(rad), near.px, near.pz,
               ax, az, mov8, rad]
        sx, sz = K.launch("push", B * M, ins, (B * M, M, t.H), (t.c,), _consts(near))
        return sx.reshape(B, M), sz.reshape(B, M)
    cand = near.cand
    ok = (cand >= 0) & cells.gather2(mov, cand) & mov[:, :, None]
    rx, rz = ax[:, :, None] - cells.gather2(ax, cand), az[:, :, None] - cells.gather2(az, cand)
    d = torch.sqrt(rx * rx + rz * rz).clamp(min=1e-3)
    need = rad[:, :, None] + cells.gather2(rad, cand)
    over = torch.where(ok & (d < need), (need - d) / 2, torch.zeros_like(d))
    return (over * rx / d).sum(2), (over * rz / d).sum(2)
