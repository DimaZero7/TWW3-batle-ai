"""The knock-down man by man: a blow of mass or force throws a man off his feet (docs/en/training/fighters_lab.md).

One mechanism for every source (a spell's projectile and its shockwave now; a charge, a monster's blow later):
  knocked      the speed the impact gives a man - a body that strikes him (the projectile's mass x impact speed / his
               mass) or a blast (its force / his mass x (1 - d / R) at d from its centre) - against the database's
               ladder: knocked back from collision_ / shockwave_speed_delta_threshold_knocked_back 4 / 2 m/s (thrown,
               stays up), knocked down from _knocked_down 7 / 3.5 m/s (thrown and lying) - levels()
  on the ground  he lies a time uniform in the database's knockdown_getting_up_delay_min..max (1.0-2.7 s); lying he
               strikes nobody (his attack clock stops), nobody strikes him and no projectile or blast takes his health
               (lab.json knock.invulnerable: the simulator's magic.knock reading - 'a knocked-down man takes nothing more
               until he is up', the community: knocked-down infantry survive what would kill them; checked on the
               knock probe, magic battle 6), he does not move with his unit (he lies where he fell)
  thrown       away from the impact's centre by lab.json knock.throw: measured on the knock probe (magic battle 6,
               knockdata.py; no database number), a median 3.0 m
  up           then he walks back to his place in the formation (move.py)
Needs torch."""
import math

import torch

from tools.nn.fighters import melee as FM, state as S


def down(F):
    """[B, M] the men lying (knocked down)."""
    return F.down > 0


def levels(params, dv_body, dv_wave):
    """(thrown, down) [B, M] from the impact's speed a man gets from a body (dv_body: mass x speed / his mass) and from a
    blast (dv_wave: force / his mass x (1 - d / R)) against the database's ladder: knocked back from
    collision_speed_delta_threshold_knocked_back 4 / shockwave_ 2 m/s (thrown, stays up), knocked down from 7 / 3.5 m/s
    (thrown and lying; knocked_flying 10.5 / 5.25 is a knock-down too)."""
    R = params.battle
    thrown = ((dv_body >= float(R["collision_speed_delta_threshold_knocked_back"]))
              | (dv_wave >= float(R["shockwave_speed_delta_threshold_knocked_back"])))
    down_ = ((dv_body >= float(R["collision_speed_delta_threshold_knocked_down"]))
             | (dv_wave >= float(R["shockwave_speed_delta_threshold_knocked_down"])))
    return thrown | down_, down_


def knock(F, u, params, lab, thrown, lying, ox, oz):
    """The men of thrown [B, M] (living, standing) are thrown away from (ox, oz) [B, M] by lab.json knock.throw (a
    measured distance: the median x exp(N(0, sigma))); those of lying [B, M] also lie their getting-up time (the
    database's knockdown_getting_up_delay_min..max). Returns the men thrown."""
    R = params.battle
    lo, hi = float(R["knockdown_getting_up_delay_min"]), float(R["knockdown_getting_up_delay_max"])
    m = thrown & S.alive(F) & ~down(F)
    ml = m & lying
    g = F.gen
    t = lo + (hi - lo) * torch.rand(F.hp.shape, generator=g, device=F.hp.device)
    F.down = torch.where(ml, t, F.down)
    F.tgt = torch.where(ml, torch.full_like(F.tgt, -1), F.tgt)
    th = lab.get("knock", {}).get("throw") or {}
    dist = float(th.get("median_m", 0.0)) * torch.exp(float(th.get("sigma_log", 0.0))
                                                       * torch.randn(F.hp.shape, generator=g, device=F.hp.device))
    dist = torch.where(m, dist.clamp(max=float(th.get("max_m", 10.0))), torch.zeros_like(dist))
    px, pz = S.positions(F, u)
    vx, vz = px - ox, pz - oz
    n = torch.sqrt(vx * vx + vz * vz).clamp(min=1e-3)
    F.dx = F.dx + vx / n * dist
    F.dz = F.dz + vz / n * dist
    return m


def tick(F, dt):
    F.down = (F.down - dt).clamp(min=0)


def bombard(F, u, params, lab, landings, dt):
    """The step's spell projectiles man by man (tools/nn/sim/magic.py: the same projectiles, landing times and strike
    points; here each lands at a point - its distance from the strike point uniform in [0, spread] at any angle,
    magic.landing 'radius' - and strikes the men there): the contact (collision_m + the man's radius, at most its
    penetration's nearest men) and the blast (detonation radius less magic.explosion_radius_cut_m, damage x (1 - (d /
    R)^p)) of the passport, each man's armour roll, overkill lost; the impact knocks down (knock()). Returns hp [B, i, j]
    (caster i's strike on unit j) and the hits [B, i, j]."""
    from tools.nn.sim import magic
    B, N = u["men"].shape
    M = F.M
    dev = u["men"].device
    hp_pair = torch.zeros((B, N * N), device=dev)
    hit_pair = torch.zeros((B, N * N), device=dev)
    if not landings:
        return hp_pair.reshape(B, N, N), hit_pair.reshape(B, N, N)
    c = magic.cal(params)
    T = magic.table(params, dev)
    col = lambda name: T[:, magic.COL[name]]
    p_fall = float(c.get("explosion_falloff_power", 1.0))
    mode = str(c.get("landing", "radius"))
    R = params.battle
    gen = F.gen
    # the projectiles of the step: rows (battle, caster, spell, strike point) x their count (a fraction rounded at random)
    bs, cs, ss, xs, zs, ns = [], [], [], [], [], []
    for k, rows, n in landings:
        b_, i_ = rows[:, 0], rows[:, 1]
        cnt = torch.floor(n + torch.rand(n.shape, generator=gen, device=dev)).long()
        bs.append(b_)
        cs.append(i_)
        ss.append(u[f"sp{k}"][b_, i_])
        xs.append(u[f"bm{k}_x"][b_, i_])
        zs.append(u[f"bm{k}_z"][b_, i_])
        ns.append(cnt)
    bb, ii, sp, sx, sz, cnt = (torch.cat(v) for v in (bs, cs, ss, xs, zs, ns))
    side_m = S.ugather(u, F, "side")
    mass = S.ugather(u, F, "mass").clamp(min=1.0)
    rad = S.ugather(u, F, "radius")
    arm = S.ugather(u, F, "armour")
    res_magic = S.ugather(u, F, "resist_magic")
    res_phys = (S.ugather(u, F, "resist_missile") + S.ugather(u, F, "resist_physical")).clamp(max=0.9)
    inv = bool(lab.get("knock", {}).get("invulnerable", True))
    for j in range(int(cnt.max()) if len(cnt) else 0):
        on = cnt > j                                                        # [P] this projectile lands now
        if not bool(on.any()):
            break
        P = bb.shape[0]
        r_l = torch.rand(P, generator=gen, device=dev)
        spread = col("spread_m")[sp]
        rho = spread * (r_l if mode == "radius" else torch.sqrt(r_l))
        ang = 2 * math.pi * torch.rand(P, generator=gen, device=dev)
        lx, lz = sx + rho * torch.cos(ang), sz + rho * torch.sin(ang)
        px, pz = S.positions(F, u)
        d = torch.sqrt((px[bb] - lx[:, None]) ** 2 + (pz[bb] - lz[:, None]) ** 2)          # [P, M]
        caster_side = u["side"][bb, ii]
        allies = col("allies")[sp] > 0
        able = S.alive(F)[bb] & (~down(F)[bb] if inv else torch.ones_like(d, dtype=torch.bool))
        able = able & ((side_m[bb] != caster_side[:, None]) | allies[:, None]) & on[:, None]
        magical = col("magical")[sp][:, None] > 0
        res = torch.where(magical, res_magic[bb], res_phys[bb])
        # contact: the nearest `penetration` men within collision_m + their radius
        reach_c = col("c_radius_m")[sp][:, None] + rad[bb]
        in_c = able & (d <= reach_c) & ((col("c_base") + col("c_ap"))[sp][:, None] > 0)
        dc = torch.where(in_c, d, torch.full_like(d, 1e9))
        rank = torch.argsort(torch.argsort(dc, dim=1), dim=1)
        in_c = in_c & (rank < col("c_pen")[sp][:, None])
        dmg_c = FM.armour_roll(col("c_base")[sp][:, None].expand_as(d), col("c_ap")[sp][:, None].expand_as(d),
                               arm[bb], res, gen, d.shape)
        # the blast
        R_e = col("e_radius_m")[sp][:, None]
        in_e = able & (d < R_e) & (R_e > 0)
        fall = (1 - (d / R_e.clamp(min=1e-6)).clamp(max=1) ** p_fall)
        dmg_e = FM.armour_roll(col("e_base")[sp][:, None] * fall, col("e_ap")[sp][:, None] * fall, arm[bb], res, gen,
                               d.shape)
        amt = torch.where(in_c, dmg_c, 0.0) + torch.where(in_e & ((col("e_base") + col("e_ap"))[sp][:, None] > 0),
                                                          dmg_e, 0.0)
        # onto the men (overkill lost), the pairs
        flat = (bb[:, None] * M + torch.arange(M, device=dev)[None, :]).reshape(-1)
        raw = torch.zeros(B * M, device=dev).index_add_(0, flat, amt.reshape(-1)).reshape(B, M)
        room = (F.hp - F.pend).clamp(min=0)
        kf = torch.where(raw > 0, torch.minimum(raw, room) / raw.clamp(min=1e-9), torch.zeros_like(raw))
        eff = amt * kf[bb]
        F.pend = F.pend + torch.zeros(B * M, device=dev).index_add_(0, flat, eff.reshape(-1)).reshape(B, M)
        pair = (bb[:, None] * (N * N) + ii[:, None] * N + F.uid[bb]).reshape(-1)
        hp_pair = hp_pair.reshape(-1).index_add_(0, pair, eff.reshape(-1)).reshape(B, N * N)
        hit_pair = hit_pair.reshape(-1).index_add_(0, pair, ((amt > 0).float()).reshape(-1)).reshape(B, N * N)
        # the knock: the impact's speed a man gets - the projectile's (contact) or the blast's (shockwave)
        dv_c = torch.where(in_c, col("c_mom")[sp][:, None] / mass[bb], 0.0)
        dv_e = torch.where(in_e, col("e_force")[sp][:, None] / mass[bb] * (1 - d / R_e.clamp(min=1e-6)), 0.0)
        hit_k, lie_k = levels(params, dv_c, dv_e)
        mk = torch.zeros(B * M, device=dev).index_add_(0, flat, hit_k.float().reshape(-1)).reshape(B, M) > 0
        ml = torch.zeros(B * M, device=dev).index_add_(0, flat, lie_k.float().reshape(-1)).reshape(B, M) > 0
        # the impact's centre: the nearest landing that knocked him (one projectile a step per man in practice)
        ox = torch.zeros(B * M, device=dev).index_reduce_(0, flat, torch.where(hit_k, lx[:, None].expand_as(d), -1e9)
                                                          .reshape(-1), "amax", include_self=False).reshape(B, M)
        oz = torch.zeros(B * M, device=dev).index_reduce_(0, flat, torch.where(hit_k, lz[:, None].expand_as(d), -1e9)
                                                          .reshape(-1), "amax", include_self=False).reshape(B, M)
        knock(F, u, params, lab, mk & (F.hp - F.pend > 0), ml, ox, oz)
    return hp_pair.reshape(B, N, N), hit_pair.reshape(B, N, N)
