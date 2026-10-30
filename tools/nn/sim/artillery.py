"""Artillery: guns with their crew, their volleys and their shells (docs/ru/training/simulator.md "Артиллерия";
config/nn/sim.json artillery with its sources; the analyses build/diag_arty/empire.md, skaven.md and the game's rules
read from its program, R-work notes/artillery_rules.md). Every number is a passport's (config/nn/units.json `engines`,
`missile`, `missile.blast`: the game's database) or a rule of the game's (config/nn/game_rules.json). Only a unit with
guns (STATIC eng0 > 0) is touched: a battle without artillery steps as before, bit for bit.

The unit (start, guns, shooters, split):
  crew        the unit's men (the database's man_entity: 44 x 56 HP the Empire's), they fight in melee with their short
              swords and die as infantry; the box: one rank of the guns, eng0 files h apart (geometry.dims)
  guns        eng0 guns of eng_hp each (4 x 473): their health eng_w is part of the unit's hp_abs (one number, as the
              game's card: 4356); a gun is lost when the pool falls a gun's health (eng = ceil(eng_w / eng_hp))
  firing      min(guns left, floor(crew / crew_k)) guns fire (artillery.crew "pool": 3 men a gun, the crew re-man)
  shots       a volley is the firing guns x shots_per_volley projectiles (the Helblaster 4 x 9; shooters()), the
              ammunition per gun (missile.ammo_per engine)
  damage      the guns take guns / (crew + guns) of every HP the unit takes (artillery.gun_share "count": a random
              entity of the unit is struck; split()); the crew the rest; all crew dead - the unit is dead
The shells (launch, bursts, land): a projectile with a blast (projectiles_explosions shockwave) or a 'fixed' trajectory
(the Mortar) is a shell. A volley's shells land together after the flight T - the fixed arc's sqrt(2 d tan(el) / g),
the low arc's d / (v cos theta). Where (build/diag_arty/mechanic.md, R-work shooting_module.md 2.2-2.3: the game's
program): every gun aims at one man - the target's most central with prefer_central_targets (STATIC central), else
the man nearest it (the near edge, under each gun) - at 0.75 of his height, led fully (+ his velocity x T); the miss
is the barrel turned by a uniform yaw and pitch within +-atan(r / 2d), r = sqrt(calibration area x zeroing-in / p):
across d tan(yaw), along the arc's range at the turned angle (the Mortar's 50 deg: a few metres; the rockets' 11 deg:
+-55 m); a shell falling through its target's box bursts on its first man. The expectation over the miss by a
quadrature (bursts) against each unit's men in cells of its box (cells, exposure).
  blast       a man whose body (radius r) reaches the ring (d^2 - r^2 < r_e^2) takes detonation_damage x (1 - clamp((d -
              r) / r_e, 0, 1)) (linear from his body's edge, no radius cut: the code's growing ring; artillery_rules.md
              1.1-1.2), armour on the base part, AP whole, no shield, spell resistance on a magical one; friends too
              (affects_allies)
  own hit     a fixed shell's own projectile (a direct shot's is missile.volley's): the men within its collision radius +
              a man's radius, at most missile.men_struck of them
  knock       a blast knocks down the men at closeness >= 0.5 when (1 + e) F S / (F + m) > 3.5 m/s (the game's
              shockwave thresholds, artillery_rules.md 1.4), a fixed shell's own hit at momentum / m >= 7; a man down
              takes no more of that landing (magic.open_share, its shells at once)
  effect      the blast's contact effect (the Plagueclaw's -10 morale) on the enemies it strikes (effects.lay)
Morale (morale_mark; artillery.morale, R-work artillery_rules.md 3): a unit an artillery projectile is fired at (the
mark 2 s after the shot), one a shell lands within 12 m of (artillery_near_miss_distance_squared 144, its box's edge;
not the shooter's side) and one whose man a projectile hits (its own hit: a ball, a fixed shell's - not a blast) is
'attacked by artillery' for 30 s (-8); one whose man is hit, 'damaged by artillery' for 30 s (-10, instead of -8).
A magical projectile's own hit (p_magic: the Warp Lightning Cannon, the Plagueclaw) meets spell resistance, not the
physical (missile.volley, land; as the blast's ex_magic).
The cycle: a gun's reload + its aim and fire animations (artillery.cycle_add_s, measured; STATIC rl_add).
Not modelled (config/nn/sim.json artillery.shell_why): the cannonball's bounce and roll, each gun's own clock (the
battery fires as one), the rockets' extra scatter, the knocked men's pause in melee.
Needs torch.
"""
import math

import torch

from tools.nn.sim import magic
from tools.nn.sim.magic import SMOOTH_M

G = 9.81


def cal(params):
    return (params.sim.get("artillery") or {}) if params.sim else {}


def on(params):
    """Artillery is modelled (artillery.on) and some passport has guns (params.has_art)."""
    return bool(getattr(params, "has_art", False))


def start(u):
    """At the battle's start (scenario.build): an artillery unit's guns (eng = eng0) and their health eng_w = eng0 x
    eng_hp; its hp_abs the crew's (men x hp_man: a unit given fewer men starts with fewer crew) + eng_w. In place;
    nothing for a unit without guns."""
    if "eng0" not in u:
        return
    art = u["eng0"] > 0
    zero = torch.zeros_like(u["eng0"])
    u["eng"] = torch.where(art, u["eng0"], zero)
    u["eng_w"] = torch.where(art, u["eng0"] * u["eng_hp"], zero)
    u["hp_abs"] = torch.where(art, u["men"] * u["hp_man"] + u["eng_w"], u["hp_abs"])


def guns(u):
    """[B, N] the guns that fire (artillery.crew "pool"): min(the guns left, floor(the crew left / crew_k)); 0 for a
    unit without guns."""
    g = torch.minimum(u["eng"], torch.floor(u["men"].clamp(min=0) / u["crew_k"].clamp(min=1.0) + 1e-6))
    return torch.where(u["eng0"] > 0, g, torch.zeros_like(g))


def shooters(u):
    """[B, N] the projectiles a unit's volley holds at a whole load: its men (each shoots once a reload) or, for an
    artillery unit, its firing guns x shots_per_volley (spv)."""
    if "eng0" not in u:
        return u["men"]
    return torch.where(u["eng0"] > 0, guns(u) * u["spv"], u["men"])


def can_fire(u):
    """[B, N] a unit without guns, or one with a gun that fires."""
    return (u["eng0"] <= 0) | (guns(u) >= 1)


def gun_share(u):
    """[B, N] the share of the HP an artillery unit takes that lands on its guns (artillery.gun_share "count": guns /
    (crew + guns)); 0 for the others and for a unit without guns left."""
    g = u["eng"]
    s = g / (u["men"].clamp(min=0) + g).clamp(min=1e-6)
    return torch.where((u["eng0"] > 0) & (g > 0), s, torch.zeros_like(s))


def guns_after(u, gun_hp):
    """In place, after the step's damage: the guns' health less gun_hp [B, N] (their share, split by the caller), the
    guns left (the pool over a gun's health, rounded up) - none when the crew is all dead (the unit is dead)."""
    art = u["eng0"] > 0
    w = (u["eng_w"] - gun_hp).clamp(min=0)
    w = torch.where(u["men"] > 0, w, torch.zeros_like(w))
    n = torch.ceil(w / u["eng_hp"].clamp(min=1e-6) - 1e-6).clamp(min=0)
    u["eng_w"] = torch.where(art, w, u["eng_w"])
    u["eng"] = torch.where(art, torch.minimum(n, u["eng0"]), u["eng"])


def theta_low(v, d):
    """The low arc's launch angle (rad) to d m at speed v (flat ground; 45 deg out of reach)."""
    return 0.5 * torch.asin((d * G / (v * v).clamp(min=1e-6)).clamp(max=1.0))


def flight(u, d):
    """[B, N] seconds a shooter's projectile flies to d m [B, N]: the fixed arc's (fix_el > 0) sqrt(2 d tan(el) / g),
    else the low arc's d / (v cos theta)."""
    v = u["muzzle_v"].clamp(min=1.0)
    el = u["fix_el"] * (math.pi / 180)
    fixed = torch.sqrt(2 * d.clamp(min=0) * torch.tan(el) / G)
    low = d / (v * torch.cos(theta_low(v, d))).clamp(min=1e-3)
    return torch.where(u["fix_el"] > 0, fixed, low)


def shell(u):
    """[B, N] its projectiles are shells (artillery.py launch): a gun's with a blast or a fixed trajectory."""
    return (u["eng0"] > 0) & ((u["ex_r"] > 0) | (u["fix_el"] > 0))


def spread_side(u, d, el, n):
    """[B, N] the side r, m, of a gun's spread square at its aim d m off (R-work shooting_module.md 2.2,
    build/diag_arty/mechanic.md 2): p = accuracy / 100 x (calibration distance / d)^2 x (1 - launch angle / 180 deg)
    within [0.01, 1]; the zeroing-in A' = calibration area x max(1 - ZERO_STEP x n, ZERO_FLOOR) (n: the gun's shots so
    far at this target); r = sqrt(A' / p). The launch is turned by a uniform yaw and pitch within +-atan(r / 2d)."""
    p = (u["acc"] / 100 * (u["cal_dist"] / d.clamp(min=1.0)) ** 2 * (1 - el / math.pi)).clamp(0.01, 1.0)
    z = (1 - ZERO_STEP * n).clamp(min=ZERO_FLOOR)
    return torch.sqrt(u["cal_area"].clamp(min=0) * z / p)


ZERO_STEP, ZERO_FLOOR = 0.1, 0.5       # the zeroing-in: the area -10 % a shot, at most halved (shooting_module.md 2.2)
AIM_H = 0.75                           # the aim point: the target man's feet + 0.75 of his height (shooting_module 2.3)


def launch(u, pw, params, shots, target, keep=None):
    """In place: the shells of this step's volleys (shots [B, N] projectiles at target [B, N]; keep [B, N]: the share no
    trunk stopped, ground.trunk_keep) take flight - the slot sh_* of each shooter of shells that fired (land() has landed
    a volley still in flight first). The aim (build/diag_arty/mechanic.md 1; STATIC central): every gun takes the
    target's man of the least cost - with prefer_central_targets its most central man (the box's centre), else the man
    nearest the gun (the box's near edge along the line; across, the man under each gun: uniform over the guns' line,
    at most the target's span); the aim point is AIM_H of his height up, so the arc lands AIM_H h / tan(angle) beyond
    him; it leads the target fully (its velocity x the flight). Kept for land(): the line of fire (sh_wx, sh_wz), the
    aim's spread across (sh_aa), the spread's side (sh_r; its zeroing-in by sh_k shots at sh_kt), the distance to the
    aim (sh_d), the launch angle (sh_el) and the target (sh_t)."""
    go = shell(u) & (shots > 0) & (target >= 0)
    t = target.clamp(min=0)
    d0 = pw["dist"].gather(2, t[:, :, None]).squeeze(2)
    th = pw["theta"].gather(2, t[:, :, None]).squeeze(2)                    # bearing shooter -> target
    wx, wz = torch.sin(th), torch.cos(th)
    phi = u["b"].gather(1, t) * (math.pi / 180) - th
    f, dp = pw["front"].gather(1, t), pw["depth"].gather(1, t)
    c, s = torch.cos(phi).abs(), torch.sin(phi).abs()
    across, along = f * c + dp * s, f * s + dp * c                           # the target's span across / along the line
    central = u["central"]
    near = torch.where(central, torch.zeros_like(d0), along / 2)
    d = (d0 - near).clamp(min=1.0)
    v = u["muzzle_v"].clamp(min=1.0)
    el = torch.where(u["fix_el"] > 0, u["fix_el"] * (math.pi / 180), theta_low(v, d)).clamp(min=1e-3)
    T = flight(u, d)
    beyond = AIM_H * u["height"].gather(1, t) / torch.tan(el)
    # the guns' line seen across the fire (eng0 guns at their files' centres: (g - 1) / g of the box's front)
    g = u["eng0"].clamp(min=1)
    span = u["width"] * (g - 1) / g * torch.cos(u["b"] * (math.pi / 180) - th).abs()
    aa = torch.where(central, torch.zeros_like(d0), torch.minimum(span, across) / 2)
    x = u["x"].gather(1, t) + u["vx"].gather(1, t) * T + wx * (beyond - near)
    z = u["z"].gather(1, t) + u["vz"].gather(1, t) * T + wz * (beyond - near)
    # the zeroing-in: the gun's shots at this target since it was taken (a new target starts it again)
    tf = target.to(u["sh_kt"].dtype)
    k = torch.where(u["sh_kt"] == tf, u["sh_k"], torch.zeros_like(u["sh_k"]))
    n = shots * (keep if keep is not None else 1.0)
    for name, val in (("sh_s", T), ("sh_x", x), ("sh_z", z), ("sh_n", n), ("sh_wx", wx), ("sh_wz", wz), ("sh_aa", aa),
                      ("sh_r", spread_side(u, d, el, k)), ("sh_d", d), ("sh_el", el), ("sh_t", tf), ("sh_k", k + 1),
                      ("sh_kt", tf)):
        u[name] = torch.where(go, val, u[name])


Q_AIM, Q_YAW, Q_PITCH = 3, 10, 24      # the landing's quadrature: the aim across, the yaw, the pitch (midpoints)
CELLS = (6, 3)                         # a unit's box in cells across x along its facing (its men's places)
CHUNK = 16                             # landing rows computed at once (memory: rows x units x cells x points)


def _mid(n, device):
    """n midpoints of [-1, 1]."""
    return (torch.arange(n, dtype=torch.float32, device=device) + 0.5) / n * 2 - 1


def bursts(u, front, depth, b_, i_):
    """[M, Q, 2] the burst points (x, z) of the landing rows' (b_, i_) shells, Q equal-weight points of the game's miss
    (build/diag_arty/mechanic.md 2): across - the aim (uniform +-sh_aa) + d tan(yaw); along - the arc turned by the
    pitch at its speed (a fixed trajectory the speed that reaches d at its angle, a low one its muzzle speed): the range
    v^2 sin 2(el + pitch) / g less the aimed one; the yaw and the pitch uniform within +-atan(sh_r / 2d). A shell coming
    down through its target's box bursts on the first man (percussive, mechanic.md 3): landing past the box's near edge
    within the shadow h / tan(el), at max(near edge, landing - shadow) (the box now, along the line; within its span
    across)."""
    dev = u["men"].device
    M = len(b_)
    d, el, r = u["sh_d"][b_, i_], u["sh_el"][b_, i_], u["sh_r"][b_, i_]
    half = torch.atan(r / (2 * d))
    qa = _mid(Q_AIM, dev)[None, :] * u["sh_aa"][b_, i_][:, None]                       # [M, Qa]
    yaw = _mid(Q_YAW, dev)[None, :] * half[:, None]                                       # [M, Qy]
    pit = _mid(Q_PITCH, dev)[None, :] * half[:, None]                                     # [M, Qp]
    fixed = u["fix_el"][b_, i_] > 0
    v2 = torch.where(fixed, d * G / torch.sin(2 * el).clamp(min=1e-3), u["muzzle_v"][b_, i_].clamp(min=1.0) ** 2)
    dl = v2[:, None] * (torch.sin(2 * (el[:, None] + pit).clamp(min=0.0)) - torch.sin(2 * el)[:, None]) / G
    across = (qa[:, :, None] + d[:, None, None] * torch.tan(yaw)[:, None, :]).reshape(M, -1)     # [M, Qa Qy]
    # the target's box along / across the line now (its first man stops a falling shell)
    t = u["sh_t"][b_, i_].long().clamp(min=0)
    wx, wz = u["sh_wx"][b_, i_], u["sh_wz"][b_, i_]
    ox = u["x"][b_, t] - u["sh_x"][b_, i_]
    oz = u["z"][b_, t] - u["sh_z"][b_, i_]
    o_l, o_a = ox * wx + oz * wz, ox * wz - oz * wx
    phi = u["b"][b_, t] * (math.pi / 180) - torch.atan2(wx, wz)
    F, D = front[b_, t], depth[b_, t]
    c, s = torch.cos(phi).abs(), torch.sin(phi).abs()
    span_a, span_l = F * c + D * s, F * s + D * c
    alive = (u["men"][b_, t] > 0) & (u["sh_t"][b_, i_] >= 0) & ~u["gone"][b_, t]
    near, far = (o_l - span_l / 2)[:, None, None], (o_l + span_l / 2)[:, None, None]
    shadow = (u["height"][b_, t] / torch.tan(el).clamp(min=1e-3))[:, None, None]
    L = dl[:, None, :].expand(M, across.shape[1], Q_PITCH)
    A = across[:, :, None].expand_as(L)
    lat = ((A - o_a[:, None, None]).abs() <= span_a[:, None, None] / 2) & alive[:, None, None]
    stop = lat & (L >= near) & (L - shadow <= far)
    L = torch.where(stop, torch.maximum(near, L - shadow), L).reshape(M, -1)
    A = A.reshape(M, -1)
    # the line of fire's frame: along (wx, wz), across (wz, -wx)
    X = u["sh_x"][b_, i_][:, None] + L * wx[:, None] + A * wz[:, None]
    Z = u["sh_z"][b_, i_][:, None] + L * wz[:, None] - A * wx[:, None]
    return torch.stack([X, Z], -1)


def cells(u, front, depth, b_):
    """([M, N, C, 2], [M, N]) the places (x, z) of every unit's men in the landing rows' battles - its box (front across,
    depth along its facing) in CELLS equal cells, a cell's men at its centre - and a cell's area, m^2."""
    dev = u["men"].device
    ca, cl = CELLS
    F, D = front[b_], depth[b_]                                                          # [M, N]
    a = (F[..., None, None] * (_mid(ca, dev) / 2)[:, None]).expand(*F.shape, ca, cl).reshape(*F.shape, -1)
    l = (D[..., None, None] * (_mid(cl, dev) / 2)[None, :]).expand(*F.shape, ca, cl).reshape(*F.shape, -1)
    rad = torch.deg2rad(u["b"][b_])
    sb, cb = torch.sin(rad)[..., None], torch.cos(rad)[..., None]
    x = u["x"][b_][..., None] + a * cb + l * sb
    z = u["z"][b_][..., None] - a * sb + l * cb
    return torch.stack([x, z], -1), F * D / (ca * cl)


def exposure(P, Cx, r_b, r_e, smooth):
    """For each cell's men (Cx [M, N, C, 2]) against the bursts P [M, Q, 2] of one shell each (equal weights): (m1, m2)
    the mean and mean square of the blast's closeness (R-work artillery_rules.md 1.1-1.2: a man whose body r_b reaches
    the ring, d^2 - r_b^2 < r_e^2, takes 1 - clamp((d - r_b) / r_e, 0, 1)), pk the chance a burst knocks him (closeness
    >= 0.5: the shockwave's 'knocked down' where its speed is over the threshold), pc the bursts a m^2 near him (within smooth m: a fixed
    shell's own hit) - each per shell, [M, N, C]."""
    d = torch.cdist(Cx.reshape(Cx.shape[0], -1, 2), P).reshape(*Cx.shape[:3], P.shape[1])           # [M, N, C, Q]
    rb = r_b[..., None, None]
    re = r_e[:, None, None, None]
    reach = d * d - rb * rb < re * re
    k = torch.where(reach & (re > 0), 1 - ((d - rb) / re.clamp(min=1e-6)).clamp(0, 1), torch.zeros_like(d))
    m1, m2 = k.mean(-1), (k * k).mean(-1)
    pk = ((k >= 0.5) & reach).float().mean(-1)
    pc = (d < smooth).float().mean(-1) / (math.pi * smooth * smooth)
    return m1, m2, pk, pc


def flung(P, Cx, r_b, r_e, ux, uz, b_deg, F, D, delta):
    """[M, N, C, C + 1]: where the men a burst knocks down in each cell land - the share of them in each cell of their
    unit's box, the last 'out of the box' (artillery.holes): a knocked man (closeness >= 0.5, exposure's) flies delta m
    on from the burst, straight away from it. P [M, Q, 2] the bursts, Cx [M, N, C, 2] the cells' centres, r_b [M, N],
    r_e [M]; ux, uz, b_deg, F, D [M, N]: the units' centres, bearings and boxes (cells' layout)."""
    ca, cl = CELLS
    v = Cx[:, :, :, None, :] - P[:, None, None, :, :]                                           # [M, N, C, Q, 2]
    d = torch.sqrt((v * v).sum(-1)).clamp(min=1e-6)
    rb = r_b[..., None, None]
    re = r_e[:, None, None, None]
    kn = ((d * d - rb * rb < re * re) & (re > 0) & (1 - ((d - rb) / re.clamp(min=1e-6)).clamp(0, 1) >= 0.5)).float()
    to = Cx[:, :, :, None, :] + v / d[..., None] * delta
    rad = torch.deg2rad(b_deg)[:, :, None, None]
    dx, dz = to[..., 0] - ux[:, :, None, None], to[..., 1] - uz[:, :, None, None]
    a = dx * torch.cos(rad) - dz * torch.sin(rad)                                               # across
    l = dx * torch.sin(rad) + dz * torch.cos(rad)                                               # along
    col = torch.floor((a / F[:, :, None, None].clamp(min=1e-3) + 0.5) * ca)
    row = torch.floor((l / D[:, :, None, None].clamp(min=1e-3) + 0.5) * cl)
    inside = (col >= 0) & (col < ca) & (row >= 0) & (row < cl)
    idx = torch.where(inside, col * cl + row, torch.full_like(col, float(ca * cl))).long()
    out = torch.zeros((*kn.shape[:3], ca * cl + 1), device=P.device)
    out.scatter_add_(3, idx, kn)
    # (a cell no burst knocks: its men stay - the knocks of a fixed shell's own hit fly nowhere here)
    own = torch.eye(ca * cl + 1, device=P.device)[: ca * cl]                                    # [C, C + 1]
    out = torch.where(out.sum(-1, keepdim=True) > 0, out, own.expand_as(out))
    return out / out.sum(-1, keepdim=True)


def _eager(fn):
    return torch.compiler.disable(fn)


def holes_cal(params):
    """artillery.holes (config/nn/sim.json): {move_v, knock_min} or {} (off)."""
    h = cal(params).get("holes") or {}
    return h if h.get("on") else {}


def holes_reform(u, params):
    """In place, each step (land()): a unit holding its men unformed (hole_m > 0) forms again - its men fill its box
    evenly, the box closes up to its men (geometry.dims) - once it moves (its speed over holes.move_v), routs, re-forms
    to a new width (reform.py), is gone or dead (artillery.holes; the probe: walking targets close up, standing ones
    under the Mortar and the rockets do not)."""
    hc = holes_cal(params)
    if not hc or "cellm" not in u:
        return
    speed = torch.sqrt(u["vx"] ** 2 + u["vz"] ** 2)
    go = (u["hole_m"] > 0) & ((speed > float(hc["move_v"])) | u["r"] | (u["men"] <= 0) | u["gone"] | (u["wnew"] >= 0))
    u["hole_m"] = torch.where(go, torch.zeros_like(u["hole_m"]), u["hole_m"])
    B, N = go.shape
    cm = u["cellm"].view(B, N, -1)
    u["cellm"] = torch.where(go[..., None], torch.zeros_like(cm), cm).reshape(B, -1)


def holes_box(u, params, front, depth):
    """(front, depth) [B, N]: the box a unit's men stand in for the shells - a unit holding them unformed (hole_m > 0)
    keeps the box of its men then (geometry.dims at hole_m: the probe's standing targets under the Mortar kept 29-31 x
    16-23 m from 160 to 100 men), the others the one given (geometry.dims at their men)."""
    from tools.nn.sim import geometry
    holed = u["hole_m"] > 0
    f, d = geometry.dims(dict(u, men=torch.where(holed, u["hole_m"], u["men"])), params.sim["formation"]["spacing_m"])
    return torch.where(holed, f, front), torch.where(holed, d, depth)


def morale_cal(params):
    """artillery.morale (config/nn/sim.json): {window_s, delay_s, near_m} or {} (off)."""
    m = cal(params).get("morale") or {}
    return m if m.get("on") else {}


def near_m(params):
    """The near miss's reach, m: sqrt(artillery_near_miss_distance_squared) (the database's 144: 12 m)."""
    for table in (params.morale, params.battle):
        if "artillery_near_miss_distance_squared" in table:
            return math.sqrt(float(table["artillery_near_miss_distance_squared"]))
    return 12.0


def morale_mark(u, params, dt, fired_at, near, hit):
    """In place, each step: the units' artillery clocks (R-work artillery_rules.md 3) - art_s (attacked) and artd_s
    (damaged) run on; a unit an artillery projectile was fired at now (fired_at [B, N]) is marked attacked delay_s
    later (art_s -delay_s; a unit marked already keeps it, its window renewed), one a shell landed near (near) or whose
    man was hit (hit) at once; a hit marks it damaged too. 'attacked' while 0 <= art_s < window_s, 'damaged' while
    artd_s < window_s (morale.terms: -8 / -10)."""
    m = morale_cal(params)
    if not m or "art_s" not in u:
        return
    win, delay = float(m["window_s"]), float(m.get("delay_s", 0.0))
    # (the clocks stop at the battle's start value, 1e6: a unit never shot at keeps it - scenario.build)
    a = (u["art_s"] + dt).clamp(max=1e6)
    on = (a >= 0) & (a < win)
    a = torch.where(fired_at & ~on & (a >= win), torch.full_like(a, -delay), a)
    a = torch.where(fired_at & on, torch.zeros_like(a), a)
    u["art_s"] = torch.where(near | hit, torch.zeros_like(a), a)
    u["artd_s"] = torch.where(hit, torch.zeros_like(u["artd_s"]), (u["artd_s"] + dt).clamp(max=1e6))


def morale_points(u, params):
    """[B, N] the artillery's morale term (morale.terms): ume_concerned_damaged_by_artillery (-10) while damaged,
    else ume_concerned_attacked_by_artillery (-8) while attacked, else 0; None with artillery.morale off."""
    m = morale_cal(params)
    if not m or "art_s" not in u:
        return None
    win = float(m["window_s"])
    R = params.morale
    attacked = (u["art_s"] >= 0) & (u["art_s"] < win)
    damaged = u["artd_s"] < win
    zero = torch.zeros_like(u["art_s"])
    return torch.where(damaged, zero + float(R["ume_concerned_damaged_by_artillery"]),
                       torch.where(attacked, zero + float(R["ume_concerned_attacked_by_artillery"]), zero))


@_eager
def land(u, params, dt, front, depth, firing, noise=False, info=None):
    """The shells landing this step: (hp, kills, hits, knocks) [B, i, j] - the HP shooter i's shells take from unit j,
    the men they kill, the hits they land (own hits and blasts), the men knocked down; zeros when none lands. A shell in
    flight lands when its time is up within this step or its shooter fires again now (firing [B, N]). front, depth
    [B, N]: the boxes (geometry.dims). In place: the slots' clocks (sh_s -1 once landed). info (a dict): filled with
    "near" [B, N] (> 0: a shell of the other side landed within near_m of its box's edge - the aim point: morale_mark)
    and "own" [B, N] (its men's expected hits from the shells' own projectiles).
    Where they burst: bursts() - the game's miss (a uniform yaw and pitch, the arc's range) round the aim launch() kept;
    each unit's men in CELLS cells of its box (cells()); a cell's men meet each shell's burst by exposure() - the
    blast's closeness moments, the knock's chance, the bursts near them (a fixed shell's own hit) - the shells of a
    volley alike and independent."""
    from tools.nn.sim import melee
    B, N = u["men"].shape
    dev = u["men"].device
    hp = torch.zeros((B, N, N), device=dev)
    kills, hits, knocks = torch.zeros_like(hp), torch.zeros_like(hp), torch.zeros_like(hp)
    hc = holes_cal(params) if "cellm" in u else {}
    if hc:
        # (artillery.holes) a unit that moved forms again; the holed keep their box of the men they had then
        holes_reform(u, params)
        front, depth = holes_box(u, params, front, depth)
    fly = u["sh_s"] >= 0
    left = u["sh_s"] - dt
    now = fly & ((left <= 1e-6) | firing)
    u["sh_s"] = torch.where(now, torch.full_like(left, -1.0), torch.where(fly, left, u["sh_s"]))
    if info is not None:
        info["near"] = torch.zeros((B, N), device=dev)
        info["own"] = torch.zeros((B, N), device=dev)
    if not bool(now.any()):
        return hp, kills, hits, knocks
    rules = params.battle
    thr_c = float(rules["collision_speed_delta_threshold_knocked_down"])
    thr_s = float(rules["shockwave_speed_delta_threshold_knocked_down"])
    e_rest = float(rules["collision_restitution_coefficient"])
    t_down = (float(rules["knockdown_getting_up_delay_min"]) + float(rules["knockdown_getting_up_delay_max"])) / 2
    limit = int(getattr(params, "pen_men", 1))
    present = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"]
    men = u["men"].clamp(min=0)
    rho = torch.where(present, men / (front * depth).clamp(min=1e-3), torch.zeros_like(men))
    hm = u["hp_man"].clamp(min=1e-6)
    crew_hp = u["hp_abs"] - u["eng_w"]
    W = (men * hm - crew_hp).clamp(min=0)
    many = u["men0"] > 1
    rad = torch.deg2rad(u["b"])
    res_m = (u["resist_missile"] + u["resist_physical"]).clamp(max=0.9)
    C = CELLS[0] * CELLS[1]
    if hc:
        # (artillery.holes) a holed unit's men by cell: its cells' shares of its men now (other losses thin all alike)
        holed = u["hole_m"] > 0
        cm = u["cellm"].view(B, N, C + 1)                                  # (the last: the men flung out of the box)
        cshare = cm / cm.sum(-1, keepdim=True).clamp(min=1e-9)
        kcell = torch.zeros((B, N, C + 1), device=dev)
        delta = float(hc.get("flight_m", 0.0))
    rows = torch.nonzero(now)
    for lo in range(0, len(rows), CHUNK):
        bb, ii = rows[lo:lo + CHUNK, 0], rows[lo:lo + CHUNK, 1]
        nn = u["sh_n"][bb, ii]                                               # [M] shells
        r_e = u["ex_r"][bb, ii]
        if info is not None:
            # (morale_mark) a near miss: the aim point within near_m of a box's edge, a unit of the other side
            dx = u["sh_x"][bb, ii][:, None] - u["x"][bb]                    # [M, N]
            dz = u["sh_z"][bb, ii][:, None] - u["z"][bb]
            sb, cb = torch.sin(rad[bb]), torch.cos(rad[bb])
            along, across = dx * sb + dz * cb, dx * cb - dz * sb
            gap = torch.sqrt((across.abs() - front[bb] / 2).clamp(min=0) ** 2
                             + (along.abs() - depth[bb] / 2).clamp(min=0) ** 2)
            friend0 = u["side"][bb] == u["side"][bb, ii][:, None]
            near = present[bb] & ~friend0 & (gap <= near_m(params))
            info["near"].index_put_((bb,), near.float(), accumulate=True)
        P = bursts(u, front, depth, bb, ii)                                  # [M, Q, 2]
        Cx, area = cells(u, front, depth, bb)                                # [M, N, C, 2], [M, N]
        m1, m2, pk, pc = exposure(P, Cx, u["radius"][bb], r_e, SMOOTH_M)     # [M, N, C]
        friend = u["side"][bb] == u["side"][bb, ii][:, None]
        whom = present[bb] & (~friend | u["ex_allies"][bb, ii][:, None])
        band_men = (men[bb] / C * whom.float())[..., None].expand(-1, -1, C)  # [M, N, C]: a cell's men
        if hc:
            band_men = torch.where(holed[bb][..., None], men[bb][..., None] * cshare[bb][..., :C] * whom.float()[..., None],
                                   band_men)
        # a fixed shell's own hit: the men within its collision radius + a man's radius, at most the men it strikes
        fixed = (u["fix_el"][bb, ii] > 0)[:, None]                           # [M, 1]
        r_c = u["c_rad"][bb, ii][:, None] + u["radius"][bb]                  # [M, N]
        passes = u["size_k"][bb] < u["pen_cap"][bb, ii][:, None]
        mx = u["pen_max"][bb, ii][:, None]
        more = torch.where(mx < 0, torch.full_like(mx, float(limit)),
                           torch.floor(mx / u["pen_res"][bb].clamp(min=1e-3)))
        struck = 1 + torch.where(passes, more.clamp(0, float(max(limit - 1, 0))), torch.zeros_like(more))
        per_landing = rho[bb] * math.pi * r_c ** 2
        cap = torch.where(per_landing > struck, struck / per_landing.clamp(min=1e-9), torch.ones_like(per_landing))
        lam_c = nn[:, None, None] * pc * (math.pi * r_c ** 2 * cap)[..., None] * fixed[..., None].float()
        # the blast: equal hits of the cell's mean closeness size (its mean and mean square)
        ok = m1 > 0
        size = torch.where(ok, m2 / m1.clamp(min=1e-12), torch.zeros_like(m1))
        lam_e = torch.where(ok, nn[:, None, None] * (m1 * m1 / m2.clamp(min=1e-12)), torch.zeros_like(m1))
        res_e = torch.where(u["ex_magic"][bb, ii][:, None], u["resist_magic"][bb], res_m[bb])[..., None]
        arm, hmj = u["armour"][bb][..., None], hm[bb][..., None]
        single = ~many[bb][..., None]
        base_c = (u["m_damage"][bb, ii][:, None] + u["m_bvl"][bb, ii][:, None] * u["large"][bb].float())[..., None]
        # (a magical projectile's own hit: spell resistance, not the physical - p_magic)
        res_c = torch.where(u["p_magic"][bb, ii][:, None], (u["resist_missile"] + u["resist_magic"]).clamp(-1.0, 0.9)[bb],
                            res_m[bb]) if "p_magic" in u else res_m[bb]
        ph_c = melee.per_hit(base_c, u["m_ap"][bb, ii][:, None, None].expand_as(base_c), arm, hmj,
                             res_c[..., None], single=single)                                    # [M, N, 1]
        ph_e = melee.per_hit(u["ex_d"][bb, ii][:, None, None] * size, u["ex_ap"][bb, ii][:, None, None] * size, arm,
                             hmj, res_e, single=single)                                          # [M, N, C]
        # the knock: the blast's closeness >= 0.5 at dv over the threshold; a fixed shell's own hit by momentum
        mass = u["mass"][bb].clamp(min=1.0)                                                       # [M, N]
        F, S = u["ex_f"][bb, ii][:, None], u["ex_v"][bb, ii][:, None]
        dv = (1 + e_rest) * F * S / (F + mass)
        k_s = (dv > thr_s).float()[..., None]
        k_c = (u["c_mom"][bb, ii][:, None] / mass >= thr_c).float() * fixed.float()
        lam_k = nn[:, None, None] * (pc * (math.pi * r_c ** 2 * cap * k_c)[..., None] + pk * k_s)
        op = magic.open_share(lam_k, 0.0, t_down)
        knocks.index_put_((bb, ii), (lam_k * op * band_men).sum(-1), accumulate=True)
        lam_c = lam_c * torch.where(ph_c >= hmj - 1e-6, torch.ones_like(op), op)
        lam_e = lam_e * torch.where(ph_e >= hmj - 1e-6, torch.ones_like(op), op)
        Hc, He = lam_c * band_men, lam_e * band_men
        if noise:
            Hc, He = torch.poisson(Hc), torch.poisson(He)
        if info is not None:
            info["own"].index_put_((bb,), Hc.sum(-1) * (~friend).float(), accumulate=True)
        # the cell's men and their health: the unit's wounds by the cell's share of its men
        A = torch.minimum(band_men, men[bb][..., None])
        Wk = W[bb][..., None] * A / men[bb][..., None].clamp(min=1e-6)
        hpA = (A * hmj - Wk).clamp(min=0)
        m_ = many[bb][..., None]
        rem = hpA / A.clamp(min=1e-6)
        kc, hpc = magic._kills(Hc, A, hmj, hpA, ph_c.expand_as(Hc), rem, m_)
        A2, hpA2 = (A - kc).clamp(min=0), (hpA - hpc).clamp(min=0)
        rem2 = hpA2 / A2.clamp(min=1e-6)
        ke, hpe = magic._kills(He, A2, hmj, hpA2, ph_e, rem2, m_)
        hp.index_put_((bb, ii), (hpc + hpe).sum(-1), accumulate=True)
        kills.index_put_((bb, ii), (kc + ke).sum(-1), accumulate=True)
        hits.index_put_((bb, ii), (Hc + He).sum(-1), accumulate=True)
        if hc:
            # the dead leave their cells; the men knocked down fly delta m on from the burst (flung: the cell they land
            # in, or out of the box)
            moved = kc + ke
            if delta > 0:
                down = lam_k * op * band_men
                down = torch.minimum(down, (band_men - kc - ke).clamp(min=0))
                dest = flung(P, Cx, u["radius"][bb], r_e, u["x"][bb], u["z"][bb], u["b"][bb], front[bb], depth[bb], delta)
                moved = moved + down
                kcell.index_put_((bb,), -(down[..., None] * dest).sum(2), accumulate=True)
            kcell.index_put_((bb,), torch.cat([moved, torch.zeros_like(moved[..., :1])], -1), accumulate=True)
    if hc:
        # (artillery.holes) a standing unit whose men the shells knock down (at least knock_min of them this step) holds
        # its men where they stand from now on: its cells keep the men they had less this step's dead there
        kn = knocks.sum(1)
        speed = torch.sqrt(u["vx"] ** 2 + u["vz"] ** 2)
        start = (~holed & (kn >= float(hc["knock_min"])) & (speed <= float(hc["move_v"])) & many & ~u["r"]
                 & (u["wnew"] < 0) & present)
        keep = holed | start
        even = torch.cat([(men / C)[..., None].expand(-1, -1, C), torch.zeros_like(men)[..., None]], -1)
        base = torch.where(holed[..., None], men[..., None] * cshare, even)
        new_cm = (base - kcell).clamp(min=0)
        u["cellm"] = torch.where(keep[..., None], new_cm, cm).reshape(B, -1)
        u["hole_m"] = torch.where(start, men, u["hole_m"])
    return hp, kills, hits, knocks
