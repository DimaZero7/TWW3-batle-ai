"""The melee core (melee.core; sim/config/sim.json melee.core_why, melee.core_share_why; design
tmp/melee_core/design_v2.md with its binding consolidation tmp/melee_core/v2/review.md; evidence
sim/evidence/melee_core/spec.md). Off (every switch 0) nothing here runs and the state has none of its keys.

S1 (melee.core.share): the enemy's blows on a shared line. Per striker row i in scope (i and every enemy it touches are
formations, not large, without a missile weapon, not routing, alive):
  pair clock  core_ps [B, N * N]: seconds the pair (i, j) has touched (the formations' contact of battle.step, before
              the strike masks), kept through gaps up to contact.reset_s (core_gs: the gap's seconds), then 0
  senior a    R2: an enemy e that i touches, i's own cover towards e >= cover_c and i's `ahead` touching e (the oldest
              such pair); else the touched enemy of the oldest pair, ties within two steps: i's front side, then the
              lower slot. Latched (afw) pairs are in the ledger (R3 a)
  classes     of every other touched enemy j, the first that holds (R1): C same side of i as a, ahead[j] == a, cover
              [j, i] >= cover_c; K adjacent side; M same side, the boxes of a and j overlap, both covers towards i
              under cover_c; S same side, spans along i's side disjoint; X same side, spans overlap, none of the above;
              O the opposite side. R5 (ii): kept from the pair's first contact, set again when i's senior changes
              (core_cls [B, N * N] the kept code, core_sen [B, N] the step's senior)
  the share   (tmp/melee_core/harness/lines_s1.py - the arithmetic the lines were predicted with - on the step's own
              F): F_s = F[i, a], or where a C / M junior shares a's side the line after i's own cover F_geo[i, a]
              (1 - cover[i, a]) if larger (the per-side split and answer_struck's 'none' bypassed for the senior);
              with C / K juniors the senior keeps F_s (1 - x(tau_o) m), tau_o the oldest C / K junior's pair clock, m =
              min(1, men_jo / men_a) (R4, R5 i); the C juniors - and the K juniors of a row covered towards a (cover
              [i, a] >= cover_c; every K junior with share_k, R6) - split y(tau_o) m L by their men, L = F_geo[i, a],
              or for a covered row the larger of it and its `ahead`'s F_geo[ahead, a] (ahead_line: variant (b) of
              tmp/melee_core/v2/review.md decision 1); a K junior without that take and a latched S / X / O one get
              HEAD's F without answer_struck's 'none on the held side' (its cap 'at most the newcomer's steady men'
              kept: max(F, min(F before answer_struck, F_pre[j, i]))); an M junior and the senior share one line by men,
              max(F_geo[i, a], F_geo[i, j]) x men / (men_a + men_j). X is taken as S (HEAD's).
Codes: 0 no touch, 1 senior, 2 C, 3 K, 4 M, 5 S, 6 X, 7 O.
S2 phase 1 (tmp/melee_core/design_v21.md A.1): classes() also returns the junior's geometry (geometry(): spans, shared /
free / near-free metres, w, q, rho, sigma, g) and hands it to the read-out hook; no rule reads it yet."""
import torch

C, K, M, S, X, O = 2, 3, 4, 5, 6, 7


def cfg(params):
    return params.sim["melee"].get("core") or {}


def on(params, step="share"):
    return bool(cfg(params).get(step))


def ensure(u):
    """The S1 state, allocated the first time (zeros; the senior -1)."""
    B, N = u["men"].shape
    if "core_ps" not in u:
        z = torch.zeros((B, N * N), device=u["men"].device)
        u["core_ps"], u["core_gs"], u["core_cls"] = z.clone(), z.clone(), z.clone()
        u["core_sen"] = torch.full((B, N), -1.0, device=u["men"].device)
    if "core_held" not in u:
        u["core_held"] = torch.zeros((B, N), device=u["men"].device)          # (v2.1a G: the order state's last value)


def held_of(u, params):
    """v2.1a G: the units melee.strikes cuts by contact.hold_rate (sim/melee.py: order HOLD, or a MOVE still in contact
    under contact.hold_move; formations; missile units only with contact.hold_missile) - the same tensor, hoisted."""
    from sim import orders as Od
    cc = params.sim["contact"]
    hk = u["order_kind"] == Od.HOLD
    if cc.get("hold_move"):
        mv = u["order_kind"] == Od.MOVE
        if cc.get("move_far_full") and cc.get("leave_m"):
            far = torch.sqrt((u["ox"] - u["x"]) ** 2 + (u["oz"] - u["z"]) ** 2) >= float(cc["leave_m"])
            mv = mv & ~far
        hk = hk | mv
    return hk & ((u["range"] <= 0) | bool(cc.get("hold_missile"))) & (u["men0"] > 1)


def clock(u, touch, params, dt):
    """The pair clock (battle.step, before melee.strikes): touch [B, N, N] the enemy formations' contact."""
    B, N = u["men"].shape
    reset_s = float(params.sim["contact"].get("reset_s", 10.0))
    ps = u["core_ps"].reshape(B, N, N)
    gs = u["core_gs"].reshape(B, N, N)
    ps = torch.where(touch, ps + dt, ps)
    gs = torch.where(touch, torch.zeros_like(gs), gs + dt)
    ps = torch.where(gs > reset_s, torch.zeros_like(ps), ps)
    u["core_ps"], u["core_gs"] = ps.reshape(B, N * N), gs.reshape(B, N * N)


def _adj(p, q):
    return (p >= 0) & (q >= 0) & (((p == 0) | (p == 3)) != ((q == 0) | (q == 3)))


def classes(u, pw, touch, side, cov, ahead, params, dt):
    """(code kept [B, N, N] long, senior a [B, N] long, scope [B, N]); updates core_cls / core_sen. side: melee.strikes'
    side_of; cov / ahead: sim/stack.py stacked per pair (None: no cover)."""
    B, N = u["men"].shape
    dev = touch.device
    cc = float((params.sim["melee"].get("core_share") or {}).get("cover_c", 0.5))
    if cov is None:
        cov = torch.zeros((B, N, N), device=dev)
        ahead = torch.full((B, N), -1, dtype=torch.long, device=dev)
    large = u["large"] if "large" in u else torch.zeros_like(u["men"], dtype=torch.bool)
    okE = (u["men0"] > 1) & ~large & (u["range"] <= 0) & (u["men"] > 0)
    ok = okE & ~u["r"]
    # (v2.1 consolidated A.5) a routing enemy leaves the row's ledger, the row stays in scope: its pair is HEAD's
    touch = touch & ~u["r"][:, None, :]
    has = touch.any(2)
    scope = ok & has & ~(touch & ~okE[:, None, :]).any(2)
    hld = held_of(u, params)
    pair_s = u["core_ps"].reshape(B, N, N)
    ps = torch.where(touch, pair_s, torch.full_like(pair_s, -1.0))
    j = torch.arange(N, device=dev)
    # R2: covered towards e (own cover >= cover_c) and its `ahead` touching e
    ah = ahead.clamp(min=0)
    ah_touch = touch.gather(1, ah[:, :, None].expand(B, N, N)) & (ahead >= 0)[:, :, None]
    r2 = touch & (cov >= cc) & ah_touch
    ps2 = torch.where(r2, ps, torch.full_like(ps, -1.0))
    top = ps.max(2, keepdim=True).values
    tie = touch & (ps >= top - 2 * dt - 1e-6)
    front = side == 0
    score = torch.where(tie, 1000.0 + 100.0 * front.float() - j[None, None, :].float(), torch.full_like(ps, -1e9))
    a = torch.where(r2.any(2), ps2.argmax(2), score.argmax(2))
    a = torch.where(has, a, torch.full_like(a, -1))
    ac = a.clamp(min=0)

    def g(Xt):                                                     # X[b, i, a_i] broadcast over j
        return Xt.gather(2, ac[:, :, None].expand(B, N, N))
    s_ij, s_ia = side, g(side)
    same = (s_ij == s_ia) & (s_ij >= 0)
    adj = _adj(s_ij, s_ia)
    covT = cov.transpose(1, 2)                                     # [b, i, j] = cover[j, i]
    cov_ai = g(covT)                                               # cover[a, i]
    gapJA = pw["gap"].gather(2, ac[:, None, :].expand(B, N, N)).transpose(1, 2)   # [b, i, j] = gap[j, a_i]
    isC = same & (ahead[:, None, :] == ac[:, :, None]) & (covT >= cc)
    isM = same & (gapJA < 0) & (covT < cc) & (cov_ai < cc)
    # j's and a's spans along i's side line (across i's front for the front / back, along its facing for a flank)
    b = u["b"] * (torch.pi / 180)
    sb, cb = torch.sin(b), torch.cos(b)
    rx = u["x"][:, None, :] - u["x"][:, :, None]
    rz = u["z"][:, None, :] - u["z"][:, :, None]
    thru = (s_ij == 0) | (s_ij == 3)
    lx = torch.where(thru, cb[:, :, None], sb[:, :, None])
    lz = torch.where(thru, -sb[:, :, None], cb[:, :, None])
    off = rx * lx + rz * lz
    ext = (pw["front"][:, None, :] / 2 * (cb[:, None, :] * lx - sb[:, None, :] * lz).abs()
           + pw["depth"][:, None, :] / 2 * (sb[:, None, :] * lx + cb[:, None, :] * lz).abs())
    offa = rx.gather(2, ac[:, :, None]) * lx + rz.gather(2, ac[:, :, None]) * lz
    fa, da, ba = pw["front"].gather(1, ac)[:, :, None], pw["depth"].gather(1, ac)[:, :, None], b.gather(1, ac)[:, :, None]
    exta = (fa / 2 * (torch.cos(ba) * lx - torch.sin(ba) * lz).abs()
            + da / 2 * (torch.sin(ba) * lx + torch.cos(ba) * lz).abs())
    ov = torch.minimum(off + ext, offa + exta) - torch.maximum(off - ext, offa - exta)
    code = torch.full((B, N, N), O, dtype=torch.long, device=dev)
    code = torch.where(same & (ov > 0), torch.full_like(code, X), code)
    code = torch.where(same & (ov <= 0), torch.full_like(code, S), code)
    code = torch.where(isM, torch.full_like(code, M), code)
    code = torch.where(adj, torch.full_like(code, K), code)
    code = torch.where(isC, torch.full_like(code, C), code)
    # (v2.1a G) a held enemy is no junior: class O (HEAD's rules); it may still be the senior
    code = torch.where(hld[:, None, :], torch.full_like(code, O), code)
    code = torch.where(j[None, None, :] == a[:, :, None], torch.ones_like(code), code)
    code = torch.where(touch, code, torch.zeros_like(code))
    # R5 (ii): kept from the pair's first contact (its clock one step old), set again when the senior changes
    keep = u["core_cls"].reshape(B, N, N).long()
    new = touch & (pair_s <= 1.5 * dt)
    sen_chg = (a != u["core_sen"].long())[:, :, None]
    h_chg = (hld != (u["core_held"] > 0))[:, None, :]                  # (v2.1a G: R5 ii's third trigger)
    # (R5 ii) a pair out of touch when the senior changes is set again at its return (else it keeps the old senior's 1)
    keep = torch.where(~touch & (sen_chg | h_chg), torch.zeros_like(keep), keep)
    keep = torch.where(touch & (new | sen_chg | h_chg | (keep == 0)), code, keep)
    u["core_held"] = hld.float()
    keep = torch.where(pair_s <= 0, torch.zeros_like(keep), keep)
    u["core_cls"] = keep.reshape(B, N * N).float()
    u["core_sen"] = a.float()
    keep = torch.where(touch, keep, torch.zeros_like(keep))
    # (S2 read-out, design v2.1 A.1) the junior's geometry - returned and traced only, no rule reads it yet
    geo = geometry(u, pw, side, a, keep, pair_s, params)
    from sim import core_trace
    if core_trace.HOOK is not None:
        for k_, v_ in geo.items():
            core_trace.HOOK.note("geo_" + k_, v_)
    return {"keep": keep, "a": a, "scope": scope, "geo": geo, "touch": touch, "held": hld}


GEO_KEYS = ("len", "lena", "s", "f", "fn", "w", "q", "rho", "sigma")


def geometry(u, pw, side, a, keep, pair_s, params):
    """The junior's geometry of tmp/melee_core/design_v21.md A.1 for every pair (i, j) [B, N, N] (and g [B, N]), along
    i's side line towards j (the line of side[i, j]: across i's front for the front / back, along its facing for a
    flank), h = the half length of that side of i (front / 2 across, depth / 2 along):
      len / lena   j's / the senior a's span on that line, each clipped to [-h, h] (metres)
      s, f         the two spans' overlap (shared metres); len - s (free metres)
      fn           the part of j's clipped span within melee.wrap.max_m beyond either end of a's clipped span
      w, q         s / lena (at most 1); s / len
      rho          nearness of j's contact to the corner it shares with a's side: t = h - sgn(a's offset) x mid (mid the
                   middle of j's clipped span), rho = clamp(2 (1 - t / 2h), 0, 1)
      sigma        by the kept class: w for C, X, S; rho for K; 0 for O, M, the senior and no contact
      g            per row: (1 - prod_j (1 - x(tau_j) sigma_j)) / sum_j x(tau_j) sigma_j over the juniors of class C, X, S,
                   K (1 when the sum is 0); m_j = 1 here (rule 8's m_j needs rule 5 / 6's men: phase 2)
    Every number follows the boxes each step (the class code is the kept one). keep: the kept code [B, N, N] (0 no
    touch); pair_s: the pair clock [B, N, N]; a: the senior [B, N] (-1 none)."""
    B, N = u["men"].shape
    dev = keep.device
    ac = a.clamp(min=0)
    wrap_m = float((params.sim["melee"].get("wrap") or {}).get("max_m", 9.0))
    sc = params.sim["melee"].get("core_share") or {}
    b = u["b"] * (torch.pi / 180)
    sb, cb = torch.sin(b), torch.cos(b)
    rx = u["x"][:, None, :] - u["x"][:, :, None]
    rz = u["z"][:, None, :] - u["z"][:, :, None]
    sd = side if side is not None else torch.zeros((B, N, N), dtype=torch.long, device=dev)
    thru = (sd == 0) | (sd == 3)
    lx = torch.where(thru, cb[:, :, None], sb[:, :, None])
    lz = torch.where(thru, -sb[:, :, None], cb[:, :, None])
    off = rx * lx + rz * lz
    ext = (pw["front"][:, None, :] / 2 * (cb[:, None, :] * lx - sb[:, None, :] * lz).abs()
           + pw["depth"][:, None, :] / 2 * (sb[:, None, :] * lx + cb[:, None, :] * lz).abs())
    offa = rx.gather(2, ac[:, :, None]) * lx + rz.gather(2, ac[:, :, None]) * lz
    fa, da, ba = pw["front"].gather(1, ac)[:, :, None], pw["depth"].gather(1, ac)[:, :, None], b.gather(1, ac)[:, :, None]
    exta = (fa / 2 * (torch.cos(ba) * lx - torch.sin(ba) * lz).abs()
            + da / 2 * (torch.sin(ba) * lx + torch.cos(ba) * lz).abs())
    h = torch.where(thru, pw["front"][:, :, None] / 2, pw["depth"][:, :, None] / 2).clamp(min=1e-3)
    jlo, jhi = torch.maximum(off - ext, -h), torch.minimum(off + ext, h)
    alo, ahi = torch.maximum(offa - exta, -h), torch.minimum(offa + exta, h)
    ln_j = (jhi - jlo).clamp(min=0.0)
    ln_a = (ahi - alo).clamp(min=0.0)
    # (an empty clipped span: a point at its clamped centre)
    pj = torch.minimum(torch.maximum(off, -h), h)
    pa = torch.minimum(torch.maximum(offa, -h), h)
    jlo, jhi = torch.where(ln_j > 0, jlo, pj), torch.where(ln_j > 0, jhi, pj)
    alo, ahi = torch.where(ln_a > 0, alo, pa), torch.where(ln_a > 0, ahi, pa)
    s = (torch.minimum(jhi, ahi) - torch.maximum(jlo, alo)).clamp(min=0.0)
    f = (ln_j - s).clamp(min=0.0)
    fn = ((torch.minimum(jhi, alo) - torch.maximum(jlo, alo - wrap_m)).clamp(min=0.0)
          + (torch.minimum(jhi, ahi + wrap_m) - torch.maximum(jlo, ahi)).clamp(min=0.0))
    w = torch.where(ln_a > 0, s / ln_a.clamp(min=1e-6), torch.zeros_like(s)).clamp(max=1.0)
    q = torch.where(ln_j > 0, s / ln_j.clamp(min=1e-6), torch.zeros_like(s)).clamp(max=1.0)
    mid = (jlo + jhi) / 2
    t = h - torch.sign(offa) * mid
    rho = (2 * (1 - t / (2 * h))).clamp(0.0, 1.0)
    j = torch.arange(N, device=dev)
    is_a = j[None, None, :] == a[:, :, None]
    sig = torch.where((keep == C) | (keep == X) | (keep == S), w, torch.zeros_like(w))
    sig = torch.where(keep == K, rho, sig)
    sig = torch.where(is_a | (keep <= 1), torch.zeros_like(sig), sig)
    rel = ((keep == C) | (keep == X) | (keep == S) | (keep == K)) & ~is_a & (a >= 0)[:, :, None]
    if sc.get("x_tau") is not None:
        z = torch.where(rel, _table(pair_s, sc["x_tau"], sc["x"]) * sig, torch.zeros_like(sig))
    else:
        z = torch.zeros_like(sig)
    tot = z.sum(2)
    prod = torch.where(rel, 1 - z, torch.ones_like(z)).prod(2)
    g = torch.where(tot > 1e-9, (1 - prod) / tot.clamp(min=1e-9), torch.ones_like(tot))
    return {"len": ln_j, "lena": ln_a, "s": s, "f": f, "fn": fn, "w": w, "q": q, "rho": rho, "sigma": sig, "g": g}


def _table(t, xs, ys):
    """Piecewise linear y(t) through (xs, ys), flat after the last point (xs[0] = 0)."""
    out = torch.full_like(t, float(ys[0]))
    for k in range(len(xs) - 1):
        slope = (float(ys[k + 1]) - float(ys[k])) / (float(xs[k + 1]) - float(xs[k]))
        out = out + slope * (t - float(xs[k])).clamp(0.0, float(xs[k + 1]) - float(xs[k]))
    return out


def share_s1(u, F, F_geo, F_ans, F_pre, cov, ahead, contact, cls, params):
    """S1's share as accepted (16.11 00:40; frozen outputs tmp/melee_core/out/S1_*): kept for reference, unused."""
    """F [B, i, j] men striking after S1's rules (module doc) for the rows in scope; contact: the strike mask (no F
    outside it). F: HEAD's after answer_struck / answer_free; F_geo: from the line (before every side, stack and answer
    cap); F_ans: before answer_struck; F_pre: before the wave."""
    keep, a, scope = cls["keep"], cls["a"], cls["scope"]
    B, N = u["men"].shape
    dev = F.device
    sc = params.sim["melee"].get("core_share") or {}
    cc = float(sc.get("cover_c", 0.5))
    r6 = bool(cfg(params).get("share_k"))
    if cov is None:
        cov = torch.zeros_like(F)
        ahead = torch.full((B, N), -1, dtype=torch.long, device=dev)
    j = torch.arange(N, device=dev)
    ac = a.clamp(min=0)
    valid = scope & (a >= 0)
    is_a = (j[None, None, :] == a[:, :, None]) & valid[:, :, None]
    jun = (keep >= C) & (keep <= O) & ~is_a & valid[:, :, None]

    def g(Xt):                                                     # [b, i]: X[b, i, a_i]
        return Xt.gather(2, ac[:, :, None])[:, :, 0]
    men = u["men"]
    men_j = men[:, None, :].expand(B, N, N)
    men_a = men.gather(1, ac).clamp(min=1.0)                       # [b, i]
    F_ia, Fg_ia, cov_ia = g(F), g(F_geo), g(cov)
    covd = cov_ia >= cc
    hasCM = (jun & ((keep == C) | (keep == M))).any(2)
    F_s = torch.where(hasCM, torch.maximum(F_ia, Fg_ia * (1 - cov_ia)), F_ia)
    rel = jun & ((keep == C) | (keep == K))
    pair_s = u["core_ps"].reshape(B, N, N)
    jo = torch.where(rel, pair_s, torch.full_like(pair_s, -1.0)).argmax(2)          # the oldest C / K junior
    tau_o = pair_s.gather(2, jo[:, :, None])[:, :, 0]
    m_o = (men.gather(1, jo) / men_a).clamp(max=1.0)
    hasrel = rel.any(2)
    Fn_ia = torch.where(hasrel, F_s * (1 - _table(tau_o, sc["x_tau"], sc["x"]) * m_o), F_s)
    # the take of the C juniors (and the K ones of a covered row; every K one with share_k)
    take = rel & ((keep == C) | ((keep == K) & (covd[:, :, None] | r6)))
    L = Fg_ia
    if sc.get("ahead_line"):
        ah = ahead.clamp(min=0)
        Fg_ah = F_geo.gather(1, ah[:, :, None].expand(B, N, N)).gather(2, ac[:, :, None])[:, :, 0]   # F_geo[ahead_i, a_i]
        L = torch.where(covd & (ahead >= 0) & (ahead != j[None, :]), torch.maximum(L, Fg_ah), L)
    tot = torch.where(take, men_j, torch.zeros_like(men_j)).sum(2).clamp(min=1.0)
    y = _table(tau_o, sc["x_tau"], sc["y"]) * m_o * L               # [b, i]
    F_take = y[:, :, None] * men_j / tot[:, :, None]
    # a K junior without the take, a latched S / X / O one: HEAD's without 'none on the held side'
    lat = None
    if "afw" in u:
        afw = u["afw"].reshape(B, N, N) > 0
        lat = afw | afw.transpose(1, 2)
    unnone = jun & (keep == K) & ~covd[:, :, None]
    if r6:
        unnone = torch.zeros_like(unnone)
    if lat is not None:
        unnone = unnone | (jun & (keep >= S) & lat)
    F_un = torch.maximum(F, torch.minimum(F_ans, F_pre.transpose(1, 2)))
    # M: the senior and the junior share one line by men
    isM = jun & (keep == M)
    Fm = torch.maximum(Fg_ia[:, :, None], F_geo)
    s_m = (men_a[:, :, None] + men_j).clamp(min=1.0)
    F_M = Fm * men_j / s_m
    cap_a = torch.where(isM, Fm * men_a[:, :, None] / s_m, torch.full_like(Fm, float("inf"))).min(2).values
    Fn_ia = torch.minimum(Fn_ia, cap_a)
    out = torch.where(is_a, Fn_ia[:, :, None].expand(B, N, N), F)
    out = torch.where(take, F_take, out)
    out = torch.where(unnone, F_un, out)
    out = torch.where(isM, F_M, out)
    return torch.where(contact, out, torch.zeros_like(out))


# ---------------------------------------------------------------- S2 (design v2.1 A.2 / A.3 under v2/review.md "v2.1
# CONSOLIDATED" A and "v2.1a ADOPTED"; tmp/melee_core/design_v21a.md G, W, F, R)
def prepare(u, pw, cls, side, F_geo, length, q_wrap, ff, along, params):
    """Everything S2's two rules read, once a step (melee.strikes, after core.classes): per row i (the enemy whose blows
    are shared) and junior j [B, N, N] - on the SENIOR's side line of i (a kept same-side class keeps its line;
    question 5: a span empty there -> HEAD's F, `empty`):
      len, lena   j's / a's span on that line clipped to i's side [-h, h]; j's bounded by HEAD's contact length[j, i]
                  (v2.1 cons. A.4: s + f at most it)
      s, f, fn    shared metres; free metres once (against the union of a's span and the OLDER same-side juniors' spans,
                  cons. A.1; the older juniors' spans taken as their union); the free metres within melee.wrap.max_m
                  beyond a's ends (the same)
      Lj          j's whole extent along the line (2 ext, unclipped); w = s / min(Lj, lena) (v2.1a W), q = s / len
      dens_i      [B, N] i's men a metre towards a: F_geo x (1 - q_wrap) / (length x (1 - q_wrap)) of the pair (i, a)
                  (cons. A.8; HEAD's tensors); dens_j [B, N, N] j's men a metre of a FULL line on i: ff / along[j, i]
      m           min(1, men_j / (dens_j Lj)) (v2.1a F)
      x, y, v, d  by the pair clock tau_j: the attacking senior's rows (v2.1a R) for a C / X junior of a senior that is
                  not held, else the held rows (y the stacked lanes' own row, cons. A.2; d anchored, A.8)
      sig         w for C / X / S, 1 for K (cons. A.6: rho is not built), 0 for M, O, held, the senior, empty
      z, g        z = x sig m; g = (1 - prod (1 - z)) / sum z over the row's juniors (1 when the sum is 0)
    and the masks: jun (a junior of a row in scope), ss (same side C / X / S, not empty), cx (C / X), kk (K), mm (M)."""
    keep, a, scope, hld = cls["keep"], cls["a"], cls["scope"], cls["held"]
    B, N = u["men"].shape
    dev = keep.device
    sc = params.sim["melee"].get("core_share") or {}
    wrap_m = float((params.sim["melee"].get("wrap") or {}).get("max_m", 9.0))
    ac = a.clamp(min=0)
    valid = scope & (a >= 0)
    j = torch.arange(N, device=dev)
    is_a = (j[None, None, :] == a[:, :, None]) & valid[:, :, None]
    jun = (keep >= C) & (keep <= O) & ~is_a & valid[:, :, None]
    s_ia = side.gather(2, ac[:, :, None])                            # [b, i, 1] the senior's side of i
    b = u["b"] * (torch.pi / 180)
    sb, cb = torch.sin(b), torch.cos(b)
    rx = u["x"][:, None, :] - u["x"][:, :, None]
    rz = u["z"][:, None, :] - u["z"][:, :, None]
    thru = (s_ia == 0) | (s_ia == 3)
    lx = torch.where(thru, cb[:, :, None], sb[:, :, None])           # [b, i, 1]
    lz = torch.where(thru, -sb[:, :, None], cb[:, :, None])
    off = rx * lx + rz * lz
    ext = (pw["front"][:, None, :] / 2 * (cb[:, None, :] * lx - sb[:, None, :] * lz).abs()
           + pw["depth"][:, None, :] / 2 * (sb[:, None, :] * lx + cb[:, None, :] * lz).abs())
    h = torch.where(thru, pw["front"][:, :, None] / 2, pw["depth"][:, :, None] / 2).clamp(min=1e-3)
    offa, exta = off.gather(2, ac[:, :, None]), ext.gather(2, ac[:, :, None])
    jlo, jhi = torch.maximum(off - ext, -h), torch.minimum(off + ext, h)
    alo, ahi = torch.maximum(offa - exta, -h), torch.minimum(offa + exta, h)
    lena = (ahi - alo).clamp(min=0.0)
    pa = torch.minimum(torch.maximum(offa, -h), h)
    alo, ahi = torch.where(lena > 0, alo, pa), torch.where(lena > 0, ahi, pa)
    lenj = (jhi - jlo).clamp(min=0.0)
    jhi = torch.where(lenj > 0, jhi, jlo)
    Lj = 2 * ext
    lenT = length.transpose(1, 2)                                     # [b, i, j] = length[j, i]
    bnd = torch.where(lenj > 0, (lenT / lenj.clamp(min=1e-6)).clamp(max=1.0), torch.zeros_like(lenj))
    s_raw = (torch.minimum(jhi, ahi) - torch.maximum(jlo, alo)).clamp(min=0.0)
    pair_s = u["core_ps"].reshape(B, N, N)
    same_cls = jun & ((keep == C) | (keep == X) | (keep == S)) & ~hld[:, None, :]
    # the older same-side juniors k of row i (a larger pair clock, ties by the lower slot)   [b, i, j, k]
    pj, pk = pair_s[:, :, :, None], pair_s[:, :, None, :]
    older = same_cls[:, :, None, :] & ((pk > pj) | ((pk == pj) & (j[None, None, None, :] < j[None, None, :, None])))
    older = older & (j[None, None, None, :] != j[None, None, :, None])
    klo, khi = jlo[:, :, None, :], jhi[:, :, None, :]

    def free(lo, hi):
        ln = (hi - lo).clamp(min=0.0)
        # (cons. A.1) the older juniors' spans within [lo, hi] as a union: by the left end, each past the rightmost before
        cl = torch.where(older, torch.maximum(lo[..., None], klo), lo[..., None])
        ch = torch.where(older, torch.minimum(hi[..., None], khi), lo[..., None])
        cl, ix = cl.sort(3)
        ch = ch.gather(3, ix)
        prev = torch.cat([lo[..., None], ch[..., :-1]], 3).cummax(3).values
        taken = (ch - torch.maximum(cl, prev)).clamp(min=0.0).sum(3)
        return (ln - taken).clamp(min=0.0)
    f = (free(jlo, torch.minimum(jhi, alo)) + free(torch.maximum(jlo, ahi), jhi)) * bnd
    fn = (free(torch.maximum(jlo, alo - wrap_m), torch.minimum(jhi, alo))
          + free(torch.maximum(jlo, ahi), torch.minimum(jhi, ahi + wrap_m))) * bnd
    # ([I], phase 2: the contact bound of cons. A.4 on a COVERED junior's shared metres would cut them to the part of
    # its front HEAD's reach_cap sees past the senior - 6.4 of 21 m on the staggered lane: those men reach through the
    # senior's line, so a C junior's shared metres keep its span; its free metres and every X / S junior's are bounded)
    bnd_s = torch.where(keep == C, torch.ones_like(bnd), bnd)
    s = s_raw * bnd_s
    lenb = s + f
    w = torch.where(s > 0, s / torch.minimum(Lj, lena).clamp(min=1e-6), torch.zeros_like(s)).clamp(max=1.0)
    q = torch.where(lenb > 0, s / lenb.clamp(min=1e-6), torch.zeros_like(s)).clamp(max=1.0)
    empty = same_cls & ((s + f) <= 1e-6)
    ss = same_cls & ~empty
    cx = ss & ((keep == C) | (keep == X))
    kk = jun & (keep == K) & ~hld[:, None, :]
    mm = jun & (keep == M)

    def gA(Xt):
        return Xt.gather(2, ac[:, :, None])[:, :, 0]
    qw = q_wrap if q_wrap is not None else torch.zeros_like(F_geo)
    dens_i = gA(F_geo * (1 - qw)) / gA(length * (1 - qw)).clamp(min=1e-6)            # [b, i]
    dens_j = float(ff) / along.transpose(1, 2).clamp(min=1e-6)                       # [b, i, j]: j's full line on i
    men_j = u["men"][:, None, :].expand(B, N, N)
    # (v2.1a F; the merge's fix) m's line is the junior's ORDERED fighting line - the front its width order gives
    # (files0 = floor(width / h), not cut to the men it has; ranks ceil(men0 / files0)), seen along the line as Lj: 36.5
    # men on 30 m for a 120-man unit in the default formation - not its living box (geometry.dims cuts the files to the
    # men: a 30-man joiner's box is 30 files, m stayed 1; the q30 twin's v 0.467 against the game's 0.181)
    sp = float(params.sim["formation"]["spacing_m"])
    if "sp_h" in u:
        hh = torch.where(u["sp_h"] > 0, u["sp_h"], torch.full_like(u["sp_h"], sp))
        vv = torch.where(u["sp_v"] > 0, u["sp_v"], torch.full_like(u["sp_v"], sp))
    else:
        hh = vv = torch.full_like(u["men"], sp)
    men0 = u["men0"].clamp(min=1)
    files_o = torch.clamp(torch.floor(u["width"] / hh + 1e-4), min=1)
    front_o, depth_o = files_o * hh, torch.ceil(men0 / files_o) * vv
    Lo = (front_o[:, None, :] * (cb[:, None, :] * lx - sb[:, None, :] * lz).abs()
          + depth_o[:, None, :] * (sb[:, None, :] * lx + cb[:, None, :] * lz).abs())
    m = (men_j / (dens_j * Lo).clamp(min=1e-6)).clamp(max=1.0)
    att = cx & ~hld.gather(1, ac)[:, :, None]
    xt = sc["x_tau"]

    def T(key, alt):
        return _table(pair_s, xt, sc.get(key) or alt)
    z7 = [0.0] * len(xt)
    x = torch.where(att, T("x_att", sc["x"]), T("x", z7))
    y = torch.where(att, T("y_att", sc["y"]), T("y_stack", sc["y"]))
    v = torch.where(att, T("v_att", z7), T("v", z7))
    d = torch.where(att, T("d_att", z7), T("d", z7))
    sig = torch.where(ss, w, torch.zeros_like(w))
    sig = torch.where(kk, torch.ones_like(sig), sig)
    rel = ss | kk
    z = torch.where(rel, x * sig * m, torch.zeros_like(w))
    tot = z.sum(2)
    prod = torch.where(rel, 1 - z, torch.ones_like(z)).prod(2)
    g = torch.where(tot > 1e-9, (1 - prod) / tot.clamp(min=1e-9), torch.ones_like(tot))
    out = dict(cls, valid=valid, is_a=is_a, jun=jun, ss=ss, cx=cx, kk=kk, mm=mm, empty=empty, s=s, f=f, fn=fn, w=w, q=q,
               len=lenb, lena=lena, Lj=Lj, m=m, dens_i=dens_i, dens_j=dens_j, x=x, y=y, v=v, d=d, att=att, sig=sig, z=z,
               g=g, prod=prod, ac=ac, q_wrap_ia=gA(qw))
    from sim import core_trace
    if core_trace.HOOK is not None:
        for k_ in ("len", "lena", "s", "f", "fn", "w", "q", "m", "sig", "g"):
            core_trace.HOOK.note("geo_" + ("sigma" if k_ == "sig" else k_), out[k_])
        core_trace.HOOK.note("core_keep", keep)
        core_trace.HOOK.note("core_empty", empty)
        core_trace.HOOK.note("core_gated", jun & hld[:, None, :])
        core_trace.HOOK.note("core_att", att)
    return out


def stack(u, F, F_pre, F0, F_rank, P, cov, params):
    """S2 (design v2.1 A.3 with v2.1a W, F, R): the own men on a shared line, after the wave block, before pursuit
    (melee.strikes). P: prepare()'s dict. Per row i (an enemy) and same-side junior j (C / X):
      F[j, i] = min(men_j, F_H + blend x q max(0, dens_j f - F_H) + g v w m dens_j lena)   (blend: core_share.free_blend)
    the added men into F and F_pre (not F0 / F_rank: no wave, no first strike on them); S: HEAD's; the senior a of
    same-side juniors: F[a, i] x prod_j (1 - d w m) (F, F_pre, F0, F_rank alike); a K junior x g (1 alone).
    stack_k (sub-switch): a K junior after its own first interval brings HEAD's flank line with the cover cut lifted,
    F_H / (1 - cov[j, i]), its senior x (1 - dK(tau)); the mask of those pairs [b, j, i] is returned for the frontal
    sector (melee.strikes). NOT built of stack_k: the two leaving each other's target_face_cap. Returns
    (F, F_pre, F0, F_rank, k_turn)."""
    sc = params.sim["melee"].get("core_share") or {}
    B, N = u["men"].shape
    blend = float(sc.get("free_blend", 0))

    def tr(X):
        return X.transpose(1, 2)
    FH = tr(F)                                                                       # [b, i, j] = F[j, i]
    men_j = u["men"][:, None, :].expand(B, N, N)
    add = (blend * P["q"] * (P["dens_j"] * P["f"] - FH).clamp(min=0.0)
           + P["g"][:, :, None] * P["v"] * P["w"] * P["m"] * P["dens_j"] * P["lena"])
    newF = torch.minimum(men_j, FH + add)
    added = torch.where(P["cx"], (newF - FH).clamp(min=0.0), torch.zeros_like(FH))
    fac = torch.where(P["cx"], 1 - P["d"] * P["w"] * P["m"], torch.ones_like(FH)).prod(2)          # [b, i]
    kfac = torch.where(P["kk"], P["g"][:, :, None].expand(B, N, N), torch.ones_like(FH))
    k_turn = None
    if on(params, "stack_k") and "contact_s" in u:
        dk = _table(u["core_ps"].reshape(B, N, N), sc["x_tau"], sc.get("dk") or [0.0] * len(sc["x_tau"]))
        done = (u["contact_s"] >= u["interval"])[:, None, :].expand(B, N, N)           # j's own first interval
        kq = P["kk"] & done
        covT = tr(cov) if cov is not None else torch.zeros_like(FH)
        lift = torch.minimum(men_j, FH / (1 - covT).clamp(min=0.05))
        added = torch.where(kq, (lift - FH).clamp(min=0.0), added)
        kfac = torch.where(kq, torch.ones_like(kfac), kfac)
        fac = fac * torch.where(kq, 1 - dk, torch.ones_like(dk)).prod(2)
        k_turn = tr(kq)
    sen = torch.where(P["is_a"], fac[:, :, None].expand(B, N, N), torch.ones_like(FH))              # [b, i, a]
    mult = tr(sen * kfac)                                                            # [b, j, i]
    addT = tr(added)
    F = F * mult + addT
    F_pre = F_pre * mult + addT
    if F0 is not None:
        F0 = F0 * mult
    if F_rank is not None:
        F_rank = F_rank * mult
    from sim import core_trace
    if core_trace.HOOK is not None:
        core_trace.HOOK.note("core_added", addT)
    return F, F_pre, F0, F_rank, k_turn


def share(u, F, F_geo, F_ans, F_pre, cov, ahead, contact, P, params):
    """S2's enemy's blows (design v2.1 A.2 with the consolidated changes and v2.1a W, F, R): F [b, i, j] men of i on j
    for the rows in scope (P: prepare()'s dict; F_pre already holds core.stack's added men):
      1  F_s = max(F[i, a], F_geo[i, a] (1 - cov[i, a])) when a same-side junior (C / X / S) or an M one is present
      2  the wrap yields: F_s0 = F_s - min(F_s q_wrap[i, a], dens_i sum_ss fn)
      4  senior: F_s0 prod (1 - x sig m) (and S1's M cap)
      5  same-side junior: m [g y w dens_i lena + min(dens_i f, F_pre[j, i])]  ('none on the held side' dropped)
      6  K junior: S1's F_un x g; a K junior of a covered row: S1's take (y x the line, ahead_line, split by men) x m
      7  O and a held junior: HEAD's (latched O: F_un); M: S1's one line by men; an empty same-side span: HEAD's"""
    keep, valid, jun = P["keep"], P["valid"], P["jun"]
    B, N = u["men"].shape
    dev = F.device
    sc = params.sim["melee"].get("core_share") or {}
    cc = float(sc.get("cover_c", 0.5))
    if cov is None:
        cov = torch.zeros_like(F)
        ahead = torch.full((B, N), -1, dtype=torch.long, device=dev)
    j = torch.arange(N, device=dev)
    ac, is_a = P["ac"], P["is_a"]

    def g_(Xt):
        return Xt.gather(2, ac[:, :, None])[:, :, 0]
    men = u["men"]
    men_j = men[:, None, :].expand(B, N, N)
    men_a = men.gather(1, ac).clamp(min=1.0)
    F_ia, Fg_ia, cov_ia = g_(F), g_(F_geo), g_(cov)
    covd = cov_ia >= cc
    ss, kk, mm = P["ss"], P["kk"], P["mm"]
    F_s = torch.where((ss | mm).any(2), torch.maximum(F_ia, Fg_ia * (1 - cov_ia)), F_ia)
    F_s0 = F_s - torch.minimum(F_s * P["q_wrap_ia"], P["dens_i"] * torch.where(ss, P["fn"], torch.zeros_like(F)).sum(2))
    Fn_ia = F_s0 * P["prod"]
    g = P["g"][:, :, None]
    take = P["m"] * (g * P["y"] * P["w"] * P["dens_i"][:, :, None] * P["lena"]
                     + torch.minimum(P["dens_i"][:, :, None] * P["f"], F_pre.transpose(1, 2)))
    F_un = torch.maximum(F, torch.minimum(F_ans, F_pre.transpose(1, 2)))
    kcov = kk & covd[:, :, None]
    L = Fg_ia
    if sc.get("ahead_line"):
        ah = ahead.clamp(min=0)
        Fg_ah = F_geo.gather(1, ah[:, :, None].expand(B, N, N)).gather(2, ac[:, :, None])[:, :, 0]
        L = torch.where(covd & (ahead >= 0) & (ahead != j[None, :]), torch.maximum(L, Fg_ah), L)
    totk = torch.where(kcov, men_j, torch.zeros_like(men_j)).sum(2).clamp(min=1.0)
    F_kc = (_table(u["core_ps"].reshape(B, N, N), sc["x_tau"], sc.get("y_stack") or sc["y"]) * P["m"]
            * L[:, :, None] * men_j / totk[:, :, None])
    F_k = F_un * g
    oo = jun & (keep == O) & ~P["held"][:, None, :]
    Fm = torch.maximum(Fg_ia[:, :, None], F_geo)
    s_m = (men_a[:, :, None] + men_j).clamp(min=1.0)
    F_M = Fm * men_j / s_m
    cap_a = torch.where(mm, Fm * men_a[:, :, None] / s_m, torch.full_like(Fm, float("inf"))).min(2).values
    Fn_ia = torch.minimum(Fn_ia, cap_a)
    out = torch.where(is_a, Fn_ia[:, :, None].expand(B, N, N), F)
    out = torch.where(ss, take, out)
    out = torch.where(kk & ~kcov, F_k, out)
    out = torch.where(kcov, F_kc, out)
    if "afw" in u:
        afw = u["afw"].reshape(B, N, N) > 0
        out = torch.where(oo & (afw | afw.transpose(1, 2)), F_un, out)
    out = torch.where(mm, F_M, out)
    return torch.where(contact, out, torch.zeros_like(out))
