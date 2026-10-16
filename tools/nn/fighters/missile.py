"""Projectiles man by man (docs/en/training/fighters_lab.md).

The simulator decides when a unit fires and how many projectiles (missile.volley's shots: its volleys, reload, aim,
clear lines, ammunition). Here those shots are loosed by its living men (each with the chance shots / men) and each
flies to one man:
  aim          his own target man while that one lives in the unit he shoots (the simulator's missile.reaim), else a
               random living man of it, at his middle: h / (2 tan theta) beyond his feet (the
               simulator's spread model, missile.hit_chance_plane: the database's calibration area and distance,
               accuracy + marksmanship; sigma^2 = area x (d / calibration distance)^2 x (1 - accuracy / 100))
  landing      Gaussian: sigma across the line of fire, sigma / sin(theta) along it on the ground (theta: the launch
               angle that carries the muzzle velocity to the distance, the lower arc)
  low path     over its last h / tan(theta) metres the projectile is below the head of a man h tall: the first man
               along the path (of any unit, friend or foe, the shooter's own excepted) whose body - a circle of his
               radius - it passes through is hit; none: a miss
  shield       the struck man's shield (its block chance) stops a small-arms projectile coming within the database's
               shield_defence_angle_missile of his facing (to the man he fights, else his unit's)
  damage       the armour-piercing part whole, the base part less his own armour roll, resistance (missile + physical,
               at most 90 %), at least 1: his health, beyond it lost
The hits are the men's own: an arrow that falls into the middle of a formation takes that man's health; he dies at 0."""
import math

import torch

from tools.nn.fighters import cells, state as S
from tools.nn.fighters.melee import armour_roll
from tools.nn.sim import geometry

G = 9.81


def _random_alive(F, live, unit, gen):
    """[B, P] a random living man (index into M) of unit [B, P] (-1: none alive)."""
    ca = torch.cumsum(live.long(), 1)                                   # [B, M]
    start = F.ustart.gather(1, unit.clamp(min=0))                       # first man of the unit
    end = (start + F.umen0.gather(1, unit.clamp(min=0)) - 1).clamp(max=F.M - 1)
    before = torch.where(start > 0, ca.gather(1, (start - 1).clamp(min=0)), torch.zeros_like(start))
    n = (ca.gather(1, end.clamp(min=0)) - before).clamp(min=0)
    r = torch.floor(torch.rand(unit.shape, generator=gen, device=unit.device) * n.float()).long()
    idx = torch.searchsorted(ca, (before + r + 1).contiguous())
    return torch.where((n > 0) & (unit >= 0), idx.clamp(max=F.M - 1), torch.full_like(idx, -1))


def fire(F, u, params, lab, shots, target, near, dt):
    """The step's projectiles (module docstring): F.pend + the HP each man takes; returns hp [B, i, k] (unit i's
    projectiles on unit k, friends too) and the projectiles traced / wanted."""
    B, M = F.B, F.M
    N = u["men"].shape[1]
    cfg = lab["missile"]
    gen = F.gen
    live = S.alive(F)
    n_live = S.unit_count(F, live, N)
    q = torch.where(target >= 0, shots / n_live.clamp(min=1), torch.zeros_like(shots))
    q_m = q.gather(1, F.uid)
    reps = int(cfg.get("per_man_max", 1))
    P = int(cfg["proj_max"])
    hp_pair = torch.zeros((B, N * N), device=shots.device)
    traced = wanted = 0.0
    for rep in range(reps):
        want = live & (torch.rand((B, M), generator=gen, device=q.device) < (q_m - rep))
        if not bool(want.any()):
            break
        score = want.float() * (1 + torch.rand((B, M), generator=gen, device=q.device))
        Pk = min(P, M)
        val, sh = torch.topk(score, Pk, dim=1)                              # the shooters [B, P]
        ok = val > 0
        wanted += float(want.sum())
        traced += float(ok.sum())
        hp_pair = hp_pair + _trace(F, u, params, lab, sh, ok, target, near, live)
    return hp_pair.reshape(B, N, N), traced, wanted


def _trace(F, u, params, lab, sh, ok, target, near, live):
    B, M = F.B, F.M
    N = u["men"].shape[1]
    gen = F.gen
    dev = sh.device
    px, pz = near.px, near.pz
    i = F.uid.gather(1, sh)                                                  # shooter unit
    tu = target.gather(1, i)
    # a man keeps his target man while that one lives in the unit he shoots (the simulator's missile.reaim: 'a man keeps
    # his target man; when that man has died ... he aims again'); else a random living man of it
    kept = F.aim.gather(1, sh)
    kc = kept.clamp(min=0)
    still = (kept >= 0) & live.gather(1, kc) & (F.uid.gather(1, kc) == tu)
    aim = torch.where(still, kept, _random_alive(F, live, tu, gen))
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
    sig = (torch.sqrt(gu("cal_area").clamp(min=0)) * d / gu("cal_dist").clamp(min=1.0)
           * torch.sqrt((1 - gu("acc") / 100).clamp(min=0))).clamp(min=1e-3)
    v = gu("muzzle_v").clamp(min=1.0)
    theta = (0.5 * torch.asin((d * G / (v * v)).clamp(max=1.0))).clamp(min=1e-3)
    tan = torch.tan(theta)
    h_aim = u["height"].gather(1, F.uid.gather(1, ac))
    along = 0.5 * h_aim / tan + sig / torch.sin(theta) * torch.randn((B, sh.shape[1]), generator=gen, device=dev)
    across = sig * torch.randn((B, sh.shape[1]), generator=gen, device=dev)
    lx = ax + ex * along + ez * across
    lz = az + ez * along - ex * across
    # the bodies near the low path: cells round points from the landing point back towards the shooter
    c = near.table.c
    L_max = float((u["height"].max() / tan.clamp(min=1e-3)).clamp(max=1e4).max()) if B else 0.0
    n_pts = max(1, min(int(lab["missile"]["path_points"]), int(math.ceil(L_max / c)) + 1))
    cand = []
    for k in range(n_pts):
        cand.append(near.table.query(lx - ex * k * c, lz - ez * k * c, int(lab["cells"]["cap"])))
    cand = torch.cat(cand, 2)                                                # [B, P, K]
    # (a man lying - knocked down - is passed over: lab.json knock.invulnerable)
    hittable = live & ~(F.down > 0) if lab.get("knock", {}).get("invulnerable", True) else live
    c_ok = (cand >= 0) & cells.gather2(hittable, cand) & (cand != sh[:, :, None])
    cx, cz = cells.gather2(px, cand), cells.gather2(pz, cand)
    wx, wz = cx - lx[:, :, None], cz - lz[:, :, None]
    back = -(wx * ex[:, :, None] + wz * ez[:, :, None])                     # metres back from the landing point
    side = (wx * ez[:, :, None] - wz * ex[:, :, None]).abs()
    c_uid = cells.gather2(F.uid, cand)
    r = u["radius"].reshape(B, N).gather(1, c_uid.reshape(B, -1)).reshape(cand.shape)
    hgt = u["height"].gather(1, c_uid.reshape(B, -1)).reshape(cand.shape)
    low = hgt / tan[:, :, None]
    hit_c = c_ok & (side <= r) & (back >= -r) & (back <= low + r) & ok[:, :, None]
    first = torch.where(hit_c, back, torch.full_like(back, -1e9)).argmax(2)
    any_hit = hit_c.any(2)
    who = torch.where(any_hit, cand.gather(2, first[:, :, None]).squeeze(2), torch.full_like(sh, -1))
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
    block = front & (torch.rand(who.shape, generator=gen, device=dev) < u["shield"].gather(1, ku))
    resist = (u["resist_missile"].gather(1, ku) + u["resist_physical"].gather(1, ku)).clamp(max=0.9)
    amt = armour_roll(gu("m_damage"), gu("m_ap"), u["armour"].gather(1, ku), resist, gen, who.shape)
    amt = torch.where((who >= 0) & ~block, amt, torch.zeros_like(amt))
    raw = torch.zeros((B, M), device=dev).scatter_add(1, wc, amt)
    room = (F.hp - F.pend).clamp(min=0)
    k = torch.where(raw > 0, torch.minimum(raw, room) / raw.clamp(min=1e-9), torch.zeros_like(raw))
    eff = amt * k.gather(1, wc)
    F.pend = F.pend.scatter_add(1, wc, eff)
    return torch.zeros((B, N * N), device=dev).scatter_add(1, i * N + ku, eff)
