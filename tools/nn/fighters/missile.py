"""Projectiles man by man (docs/en/training/fighters_lab.md).

The simulator decides when a unit fires and how many projectiles (missile.volley's shots: its volleys, reload, aim,
clear lines, ammunition). Here those shots are loosed by its living men (each with the chance shots / men) and each
flies to one man:
  aim          his own target man while that one lives in the unit he shoots (the simulator's missile.reaim), else a
               random living man of it, at his middle: h / (2 tan theta) beyond his feet (the
               simulator's spread model, missile.hit_chance_plane: the database's calibration area and distance,
               accuracy + marksmanship; sigma = sqrt(area) x d / calibration distance x (1 - accuracy / 100), or
               x sqrt(1 - accuracy / 100) with missile.accuracy.shrink "area")
  landing      the simulator's spread (missile.accuracy.spread): "square" - uniform +-side / 2 across the line of fire,
               +-side / (2 sin theta) along it on the ground (sim missile.square_side); "gauss" - sigma across, sigma /
               sin(theta) along (theta: the launch angle that carries the muzzle velocity to the distance, the lower arc)
  low path     over its last h / tan(theta) metres the projectile is below the head of a man h tall: the first man
               along the path (of any unit, friend or foe, the shooter's own excepted) whose body - a circle of his
               radius - it passes through is hit; none: a miss (lab.json missile.path_full: the whole low path searched)
  shield       the struck man's shield (its block chance) stops a small-arms projectile coming within the database's
               shield_defence_angle_missile of his facing (to the man he fights, else his unit's)
  damage       the armour-piercing part whole, the base part less his own armour roll, resistance (missile + physical,
               at most 90 %), at least 1: his health, beyond it lost
The hits are the men's own: an arrow that falls into the middle of a formation takes that man's health; he dies at 0."""
import math

import torch

from tools.nn.fighters import kernels as K, state as S
from tools.nn.fighters.melee import armour_roll
from tools.nn.sim import geometry, missile as sim_missile

G = 9.81


def _random_alive(F, live, unit, gen, u01=None):
    """[B, P] a random living man (index into M) of unit [B, P] (-1: none alive). u01: the uniform draws [B, P]
    (state.rnd; else from the generator gen)."""
    ca = torch.cumsum(live.long(), 1)                                   # [B, M]
    start = F.ustart.gather(1, unit.clamp(min=0))                       # first man of the unit
    end = (start + F.umen0.gather(1, unit.clamp(min=0)) - 1).clamp(max=F.M - 1)
    before = torch.where(start > 0, ca.gather(1, (start - 1).clamp(min=0)), torch.zeros_like(start))
    n = (ca.gather(1, end.clamp(min=0)) - before).clamp(min=0)
    if u01 is None:
        u01 = torch.rand(unit.shape, generator=gen, device=unit.device)
    r = torch.minimum(torch.floor(u01 * n.float()).long(), (n - 1).clamp(min=0))
    idx = torch.searchsorted(ca, (before + r + 1).contiguous())
    return torch.where((n > 0) & (unit >= 0), idx.clamp(max=F.M - 1), torch.full_like(idx, -1))


def fire(F, u, params, lab, shots, target, near, dt):
    """The step's projectiles (module docstring): F.pend + the HP each man takes; returns hp [B, i, k] (unit i's
    projectiles on unit k, friends too) and the projectiles traced / wanted (0-dim tensors: no sync)."""
    B, M = F.B, F.M
    N = u["men"].shape[1]
    cfg = lab["missile"]
    gen = None                     # (the draws: state.rnd)
    live = S.alive(F)
    n_live = S.unit_count(F, live, N)
    q = torch.where(target >= 0, shots / n_live.clamp(min=1), torch.zeros_like(shots))
    q_m = q.gather(1, F.uid)
    reps = int(cfg.get("per_man_max", 1))
    P = int(cfg["proj_max"])
    hp_pair = torch.zeros((B, N * N), device=shots.device)
    traced = wanted = torch.zeros((), device=shots.device)
    for rep in range(reps):
        # (every rep runs, masked: no data-dependent branch, the step compiles)
        want = live & (S.rnd(F, M, 200 + 20 * rep) < (q_m - rep))
        score = want.float() * (1 + S.rnd(F, M, 201 + 20 * rep))
        Pk = min(P, M)
        # the shooters [B, P]: the Pk best scores (a sort: torch.topk with a large k is slow on the GPU)
        val, sh = torch.sort(score, dim=1, descending=True)
        val, sh = val[:, :Pk], sh[:, :Pk]
        ok = val > 0
        wanted = wanted + want.sum()
        traced = traced + ok.sum()
        hp_pair = hp_pair + _trace(F, u, params, lab, sh, ok, target, near, live, 210 + 20 * rep)
    return hp_pair.reshape(B, N, N), traced, wanted


def _trace(F, u, params, lab, sh, ok, target, near, live, salt=210):
    B, M = F.B, F.M
    N = u["men"].shape[1]
    gen = None                     # (the draws: state.rnd)
    dev = sh.device
    px, pz = near.px, near.pz
    i = F.uid.gather(1, sh)                                                  # shooter unit
    tu = target.gather(1, i)
    # a man keeps his target man while that one lives in the unit he shoots (the simulator's missile.reaim: 'a man keeps
    # his target man; when that man has died ... he aims again'); else a random living man of it
    kept = F.aim.gather(1, sh)
    kc = kept.clamp(min=0)
    still = (kept >= 0) & live.gather(1, kc) & (F.uid.gather(1, kc) == tu)
    Pk = sh.shape[1]
    aim = torch.where(still, kept, _random_alive(F, live, tu, gen, u01=S.rnd(F, Pk, salt)))
    F.aim = F.aim.scatter(1, sh, torch.where(ok, aim, kept))
    ok = ok & (aim >= 0)
    ac = aim.clamp(min=0)
    sx, sz = px.gather(1, sh), pz.gather(1, sh)
    ax, az = px.gather(1, ac), pz.gather(1, ac)
    ex, ez = ax - sx, az - sz
    d = torch.sqrt(ex * ex + ez * ez).clamp(min=1e-3)
    ex, ez = ex / d, ez / d
    d = d.clamp(min=5.0)
    gu = lambda name: u[name].gather(1, i)
    # the accuracy's cut as the simulator's (missile.acc_shrink: the radius or the area)
    a = (1 - gu("acc") / 100).clamp(min=0)
    shrink = a if params.sim["missile"]["accuracy"].get("shrink") == "radius" else torch.sqrt(a)
    sig = (torch.sqrt(gu("cal_area").clamp(min=0)) * d / gu("cal_dist").clamp(min=1.0) * shrink).clamp(min=1e-3)
    v = gu("muzzle_v").clamp(min=1.0)
    theta = (0.5 * torch.asin((d * G / (v * v)).clamp(max=1.0))).clamp(min=1e-3)
    tan = torch.tan(theta)
    h_aim = u["height"].gather(1, F.uid.gather(1, ac))
    # the simulator's spread (missile.accuracy.spread, aim_frac; lab.json missile.spread / aim_frac over them for a test):
    # the aim point's height on the man (0.5: his middle) and a Gaussian or the square spread (sim missile.square_side)
    cfg = lab["missile"]
    acc_cfg = params.sim["missile"]["accuracy"]
    aim_h = float(cfg.get("aim_frac", acc_cfg.get("aim_frac", 0.5))) * h_aim / tan
    if cfg.get("spread", acc_cfg.get("spread", "gauss")) == "square":
        # a uniform square in the target's plane, side sqrt(calibration area / p),
        # p = accuracy / 100 x (calibration distance / d)^2 x (1 - launch angle / 180) within [0.01, 1]
        pq = (gu("acc") / 100 * (gu("cal_dist") / d) ** 2 * (1 - theta / math.pi)).clamp(0.01, 1.0)
        side = torch.sqrt(gu("cal_area").clamp(min=0) / pq)
        along = aim_h + side * (S.rnd(F, Pk, salt + 1) - 0.5) / torch.sin(theta)
        across = side * (S.rnd(F, Pk, salt + 2) - 0.5)
    else:
        along = aim_h + sig / torch.sin(theta) * S.rnorm(F, Pk, salt + 1)
        across = sig * S.rnorm(F, Pk, salt + 2)
    lx = ax + ex * along + ez * across
    lz = az + ez * along - ex * across
    # the bodies near the low path: cells round points from the landing point back towards the shooter (all
    # path_points: the points past the longest low path add only men beyond it, whom the path test drops).
    # lab.json missile.path_full: each projectile searches its whole low path - the tallest body's height / tan theta
    # + the widest radius, no farther than its shooter - up to path_points_max points (a flat bullet's low path is
    # 37 m at 100 m and 61 m at 60 m: path_points' 6 x 3.6 = 21.6 m lost the bodies beyond, a counting error)
    npts = None
    if lab["missile"].get("path_full"):
        n_pts = max(1, int(lab["missile"]["path_points_max"]))
        c = float(near.table.c)
        hmax = u["height"].amax(1, keepdim=True)
        rmax = u["radius"].amax(1, keepdim=True)
        reach = torch.minimum(hmax / tan + rmax, (d + along).clamp(min=0) + rmax)    # (the landing point from the shooter)
        npts = (torch.ceil(reach / c) + 1).clamp(1, n_pts).to(torch.int32)
    else:
        n_pts = max(1, int(lab["missile"]["path_points"]))
    # (a man lying - knocked down - is passed over: lab.json knock.invulnerable)
    hittable = live & ~(F.down > 0) if lab.get("knock", {}).get("invulnerable", True) else live
    # lab.json missile.proj_radius_m (the H5 test, 0: off): the projectile's own radius added to the bodies'
    body_r = u["radius"] + float(cfg.get("proj_radius_m", 0.0))
    who = K.path_first(near.table, int(lab["cells"]["cap"]), n_pts, px, pz, hittable, F.uid, body_r, u["height"],
                       lx, lz, ex, ez, tan, sh, ok, npts=npts)
    wc = who.clamp(min=0)
    ku = F.uid.gather(1, wc)                                                 # the struck man's unit
    # the shield: his facing (to the man he fights, else his unit's) within the shield angle of the shooter
    hx, hz = px.gather(1, wc), pz.gather(1, wc)
    h_t = F.tgt.gather(1, wc)
    fx = torch.where(h_t >= 0, px.gather(1, h_t.clamp(min=0)) - hx, torch.sin(u["b"].gather(1, ku) * geometry.DEG))
    fz = torch.where(h_t >= 0, pz.gather(1, h_t.clamp(min=0)) - hz, torch.cos(u["b"].gather(1, ku) * geometry.DEG))
    tx, tz = sx - hx, sz - hz
    cosang = (fx * tx + fz * tz) / (torch.sqrt(fx * fx + fz * fz) * torch.sqrt(tx * tx + tz * tz)).clamp(min=1e-6)
    arc = math.cos(float(params.battle["shield_defence_angle_missile"]) * geometry.DEG)
    front = (cosang >= arc) & gu("small_arms")
    # (the simulator's missile.shield_ap_pierce: an armour-piercing projectile leaves the shield part of its chance)
    block = front & (S.rnd(F, Pk, salt + 3) < u["shield"].gather(1, ku) * sim_missile.shield_pierce(u, params, i))
    resist = (u["resist_missile"].gather(1, ku) + u["resist_physical"].gather(1, ku)).clamp(max=0.9)
    amt = armour_roll(gu("m_damage"), gu("m_ap"), u["armour"].gather(1, ku), resist, gen, who.shape,
                      u01=S.rnd(F, Pk, salt + 4))
    amt = torch.where((who >= 0) & ~block, amt, torch.zeros_like(amt))
    raw = torch.zeros((B, M), device=dev).scatter_add(1, wc, amt)
    room = (F.hp - F.pend).clamp(min=0)
    k = torch.where(raw > 0, torch.minimum(raw, room) / raw.clamp(min=1e-9), torch.zeros_like(raw))
    eff = amt * k.gather(1, wc)
    F.pend = F.pend.scatter_add(1, wc, eff)
    return torch.zeros((B, N * N), device=dev).scatter_add(1, i * N + ku, eff)
