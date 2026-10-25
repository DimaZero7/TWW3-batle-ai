"""The lab's hand-written GPU kernels (Triton): the same numbers as the plain torch code beside each, faster where most
of the fixed-size work is empty - a kernel skips it (a projectile not loosed, an empty cell slot), torch computes it.

  path_first  missile.py: the first man along each projectile's low path (the cells round points from the landing
              point back towards the shooter; module missile's docstring); with bmax (25.10.2026, line g43: a projectile
              through several men) the first beyond a point of the path - kernel path_after, the same and one test more

Each function takes `use_triton` (None: on CUDA when Triton is there) and falls back to torch (CPU, tests). A kernel
runs through one torch op, fighters::launch (launch()): torch.compile calls it as it is - compiled into the step's
graph a user kernel ran 3-5x slower (its launch settings lost)."""
import torch

from tools.nn.fighters import cells

try:
    import triton
    import triton.language as tl
except Exception:                                   # noqa: BLE001 - no Triton (the host, CPU): torch only
    triton = None


def _triton_ok(t, use_triton):
    if use_triton is not None:
        return bool(use_triton) and triton is not None
    return triton is not None and t.is_cuda


# the kernels: name -> (jit function, output dtypes, constexpr names); TUNE: name -> (rows a program or 0 - one, warps)
# measured on the bench's battles (fighters.bench, RTX 5070 Ti, 1024 battles)
KERNELS = {}
TUNE = {"path": (0, 1), "path_after": (0, 1)}


def register(name, fn, dtypes, consts, tune):
    KERNELS[name] = (fn, dtypes, consts)
    TUNE.setdefault(name, tune)


@torch.library.custom_op("fighters::launch", mutates_args=())
def _launch(name: str, rows: int, ins: list[torch.Tensor], ints: list[int], floats: list[float],
            consts: list[int]) -> list[torch.Tensor]:
    fn, dtypes, names = KERNELS[name]
    mb, warps = TUNE[name]
    outs = [torch.empty(rows, dtype=dt, device=ins[0].device) for dt in dtypes]
    kw = dict(zip(names, consts))
    if mb:
        kw["MB"] = mb
    fn[(triton.cdiv(rows, mb or 1),)](*ins, *outs, *ints, *floats, num_warps=warps, **kw)
    return outs


@_launch.register_fake
def _launch_fake(name, rows, ins, ints, floats, consts):
    return [ins[0].new_empty(rows, dtype=dt) for dt in KERNELS[name][1]]


def launch(name, rows, ins, ints, floats, consts):
    """Kernel `name` over `rows` rows: its inputs ins (contiguous tensors), then its outputs ([rows] each, KERNELS'
    dtypes), ints, floats and the constexprs consts, in its signature's order."""
    return _launch(name, int(rows), [t.contiguous() for t in ins], [int(v) for v in ints],
                   [float(v) for v in floats], [int(v) for v in consts])


if triton is not None:
    @triton.jit
    def _path_first_kernel(start_ptr, count_ptr, man_ptr, px_ptr, pz_ptr, hit_ptr, uid_ptr, rad_ptr, hgt_ptr,
                           lx_ptr, lz_ptr, ex_ptr, ez_ptr, tan_ptr, sh_ptr, ok_ptr, np_ptr, out_ptr,
                           P, M, N, H, OWN, c,
                           P1: tl.constexpr, P2: tl.constexpr, NPTS: tl.constexpr, CAP: tl.constexpr,
                           BLOCK: tl.constexpr):
        q = tl.program_id(0)
        b = q // P
        ok = tl.load(ok_ptr + q)
        if ok == 0:
            tl.store(out_ptr + q, -1)
            return
        lx = tl.load(lx_ptr + q)
        lz = tl.load(lz_ptr + q)
        ex = tl.load(ex_ptr + q)
        ez = tl.load(ez_ptr + q)
        tan = tl.load(tan_ptr + q)
        sh = tl.load(sh_ptr + q)
        su = tl.load(uid_ptr + b * M + tl.maximum(sh, 0))     # the shooter's unit (OWN: its men are passed)
        npq = tl.minimum(tl.load(np_ptr + q), NPTS)          # this projectile's points (its low path's length)
        lanes = tl.arange(0, BLOCK)
        lane_ok = lanes < CAP
        best = -1e9
        who = -1
        for k in range(0, npq):
            qx = lx - ex * k * c
            qz = lz - ez * k * c
            ix = tl.floor(qx / c).to(tl.int64)
            iz = tl.floor(qz / c).to(tl.int64)
            for cc in tl.static_range(9):
                h = ((ix + (cc // 3 - 1)) * P1 + (iz + (cc % 3 - 1)) * P2) & (H - 1)    # (H a power of 2)
                slot = b * H + h
                n = tl.load(count_ptr + slot)
                live = lane_ok & (lanes < n)
                man = tl.load(man_ptr + tl.load(start_ptr + slot) + lanes, mask=live, other=-1)
                mi = b * M + man
                okm = live & (tl.load(hit_ptr + mi, mask=live, other=0) != 0) & (man != sh)
                cx = tl.load(px_ptr + mi, mask=okm, other=0.0)
                cz = tl.load(pz_ptr + mi, mask=okm, other=0.0)
                cu = tl.load(uid_ptr + mi, mask=okm, other=0)
                okm = okm & ((OWN == 0) | (cu != su))
                r = tl.load(rad_ptr + b * N + cu, mask=okm, other=0.0)
                hg = tl.load(hgt_ptr + b * N + cu, mask=okm, other=0.0)
                wx = cx - lx
                wz = cz - lz
                back = -(wx * ex + wz * ez)
                side = tl.abs(wx * ez - wz * ex)
                low = hg / tan
                hit = okm & (side <= r) & (back >= -r) & (back <= low + r)
                bm = tl.where(hit, back, -1e9)
                mx = tl.max(bm, axis=0)
                am = tl.argmax(bm, axis=0)
                cand = tl.sum(tl.where(lanes == am, man, 0), axis=0)
                better = mx > best
                who = tl.where(better, cand, who)
                best = tl.where(better, mx, best)
        tl.store(out_ptr + q, tl.where(best > -1e8, who, -1))

    register("path", _path_first_kernel, (torch.int64,), ("P1", "P2", "NPTS", "CAP", "BLOCK"), (0, 1))

    @triton.jit
    def _path_after_kernel(start_ptr, count_ptr, man_ptr, px_ptr, pz_ptr, hit_ptr, uid_ptr, rad_ptr, hgt_ptr,
                           lx_ptr, lz_ptr, ex_ptr, ez_ptr, tan_ptr, sh_ptr, ok_ptr, np_ptr, bmax_ptr, out_ptr,
                           P, M, N, H, OWN, c,
                           P1: tl.constexpr, P2: tl.constexpr, NPTS: tl.constexpr, CAP: tl.constexpr,
                           BLOCK: tl.constexpr):
        # _path_first_kernel's, only the men nearer the landing point than bmax (metres back from it: beyond the man the
        # projectile passed, path_first's bmax)
        q = tl.program_id(0)
        b = q // P
        ok = tl.load(ok_ptr + q)
        if ok == 0:
            tl.store(out_ptr + q, -1)
            return
        lx = tl.load(lx_ptr + q)
        lz = tl.load(lz_ptr + q)
        ex = tl.load(ex_ptr + q)
        ez = tl.load(ez_ptr + q)
        tan = tl.load(tan_ptr + q)
        sh = tl.load(sh_ptr + q)
        bmax = tl.load(bmax_ptr + q)
        su = tl.load(uid_ptr + b * M + tl.maximum(sh, 0))
        npq = tl.minimum(tl.load(np_ptr + q), NPTS)
        lanes = tl.arange(0, BLOCK)
        lane_ok = lanes < CAP
        best = -1e9
        who = -1
        for k in range(0, npq):
            qx = lx - ex * k * c
            qz = lz - ez * k * c
            ix = tl.floor(qx / c).to(tl.int64)
            iz = tl.floor(qz / c).to(tl.int64)
            for cc in tl.static_range(9):
                h = ((ix + (cc // 3 - 1)) * P1 + (iz + (cc % 3 - 1)) * P2) & (H - 1)
                slot = b * H + h
                n = tl.load(count_ptr + slot)
                live = lane_ok & (lanes < n)
                man = tl.load(man_ptr + tl.load(start_ptr + slot) + lanes, mask=live, other=-1)
                mi = b * M + man
                okm = live & (tl.load(hit_ptr + mi, mask=live, other=0) != 0) & (man != sh)
                cx = tl.load(px_ptr + mi, mask=okm, other=0.0)
                cz = tl.load(pz_ptr + mi, mask=okm, other=0.0)
                cu = tl.load(uid_ptr + mi, mask=okm, other=0)
                okm = okm & ((OWN == 0) | (cu != su))
                r = tl.load(rad_ptr + b * N + cu, mask=okm, other=0.0)
                hg = tl.load(hgt_ptr + b * N + cu, mask=okm, other=0.0)
                wx = cx - lx
                wz = cz - lz
                back = -(wx * ex + wz * ez)
                side = tl.abs(wx * ez - wz * ex)
                low = hg / tan
                hit = okm & (side <= r) & (back >= -r) & (back <= low + r) & (back < bmax)
                bm = tl.where(hit, back, -1e9)
                mx = tl.max(bm, axis=0)
                am = tl.argmax(bm, axis=0)
                cand = tl.sum(tl.where(lanes == am, man, 0), axis=0)
                better = mx > best
                who = tl.where(better, cand, who)
                best = tl.where(better, mx, best)
        tl.store(out_ptr + q, tl.where(best > -1e8, who, -1))

    register("path_after", _path_after_kernel, (torch.int64,), ("P1", "P2", "NPTS", "CAP", "BLOCK"), (0, 1))


def path_first(table, cap, n_pts, px, pz, hittable, uid, radius, height, lx, lz, ex, ez, tan, sh, ok,
               use_triton=None, npts=None, own_pass=False, bmax=None):
    """[B, P] the man (index into M) each projectile strikes first along its low path, -1 none. table: cells.Table of
    the men (their places px, pz [B, M]); hittable [B, M]; uid [B, M]; radius, height [B, N] by unit; the projectiles
    [B, P]: landing (lx, lz), unit direction from the shooter (ex, ez), tan of the launch angle, shooter sh, ok. A man
    is struck when the path passes within his radius over its last height / tan metres (+ radius); the first is the
    farthest back from the landing point. npts [B, P] (None: n_pts each): the points each projectile searches, at most
    n_pts (lab.json missile.path_full: as many as its low path is long). own_pass: the men of the shooter's own unit
    are passed (lab.json missile.own_pass); else only the shooter himself. bmax [B, P] (None: none): only the men
    nearer the landing point than bmax metres back from it - beyond the man a penetrating projectile passed (missile.py)."""
    B, P = lx.shape
    M = px.shape[1]
    N = radius.shape[1]
    if npts is None:
        npts = torch.full_like(sh, int(n_pts))
    if _triton_ok(px, use_triton):
        ins = [table.start, table.count, table.man, px, pz, hittable.to(torch.int8), uid, radius.float(),
               height.float(), lx, lz, ex, ez, tan, sh, ok.to(torch.int8), npts.to(torch.int32).contiguous()]
        if bmax is not None:
            ins.append(bmax.float())
        out, = launch("path" if bmax is None else "path_after", B * P, ins, (P, M, N, table.H, int(bool(own_pass))),
                      (table.c,),
                      (cells._P1, cells._P2, n_pts, cap, max(16, 1 << (int(cap) - 1).bit_length())))
        return out.reshape(B, P)
    c = table.c
    cand = torch.cat([torch.where((k < npts)[:, :, None], table.query(lx - ex * k * c, lz - ez * k * c, cap), -1)
                      for k in range(n_pts)], 2)                                                     # [B, P, K]
    c_ok = (cand >= 0) & cells.gather2(hittable, cand) & (cand != sh[:, :, None])
    if own_pass:
        c_ok = c_ok & (cells.gather2(uid, cand) != uid.gather(1, sh.clamp(min=0))[:, :, None])
    cx, cz = cells.gather2(px, cand), cells.gather2(pz, cand)
    wx, wz = cx - lx[:, :, None], cz - lz[:, :, None]
    back = -(wx * ex[:, :, None] + wz * ez[:, :, None])                     # metres back from the landing point
    side = (wx * ez[:, :, None] - wz * ex[:, :, None]).abs()
    c_uid = cells.gather2(uid, cand)
    r = radius.gather(1, c_uid.reshape(B, -1)).reshape(cand.shape)
    hgt = height.gather(1, c_uid.reshape(B, -1)).reshape(cand.shape)
    low = hgt / tan[:, :, None]
    hit_c = c_ok & (side <= r) & (back >= -r) & (back <= low + r) & ok[:, :, None]
    if bmax is not None:
        hit_c = hit_c & (back < bmax[:, :, None])
    first = torch.where(hit_c, back, torch.full_like(back, -1e9)).argmax(2)
    return torch.where(hit_c.any(2), cand.gather(2, first[:, :, None]).squeeze(2), torch.full_like(sh, -1))
