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
The shells (launch, land): a projectile with a blast (projectiles_explosions shockwave) or a 'fixed' trajectory (the
Mortar) is a shell. A volley's shells land together after the flight T - the fixed arc's sqrt(2 d tan(el) / g), the
low arc's d / (v cos theta) - at the target's centre + its velocity x T (the game leads linearly at the shot),
uniformly over a disc of radius R: its per-axis variance R^2 / 4 the mean of the target's box (seen along the line of
fire) and the projectile's square spread (missile.square_side; along the ground / sin of the fall angle) as uniforms.
  blast       each man within the detonation radius r_e of a shell takes detonation_damage x (1 - d / r_e) (linear, no
              radius cut: the code's growing ring; artillery_rules.md 1.2), armour on the base part, AP whole, no shield,
              spell resistance on a magical one; friends too (affects_allies); the men by the bands round the landing
              point (magic.profile, landing 'area')
  own hit     a fixed shell's own projectile (a direct shot's is missile.volley's): the men within its collision radius +
              a man's radius, at most missile.men_struck of them
  knock       a blast knocks down the men within half its radius when (1 + e) F S / (F + m) > 3.5 m/s (the game's
              shockwave thresholds, artillery_rules.md 1.4), a fixed shell's own hit at momentum / m >= 7; a man down
              takes no more of that landing (magic.open_share, its shells at once)
  effect      the blast's contact effect (the Plagueclaw's -10 morale) on the enemies it strikes (effects.lay)
Not modelled (config/nn/sim.json artillery.shell_why): the cannonball's bounce and roll, the volley's rhythm, the
rockets' extra scatter, morale from artillery, the knocked men's pause in melee.
Needs torch.
"""
import math

import torch

from tools.nn.sim import magic

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


def landing_radius(u, pw, target, d, params):
    """[B, N] the radius of the disc a shooter's shells land over at its target (slot) d m off (module doc): the target's
    box seen along the line and the square spread at the launch angle, stretched along the ground by 1 / sin(the fall
    angle); the disc's per-axis variance the mean of the two (uniforms: w^2 / 12)."""
    t = target.clamp(min=0)
    v = u["muzzle_v"].clamp(min=1.0)
    th_low = theta_low(v, d.clamp(min=1.0))
    el = torch.where(u["fix_el"] > 0, u["fix_el"] * (math.pi / 180), th_low).clamp(min=1e-3)
    p = (u["acc"] / 100 * (u["cal_dist"] / d.clamp(min=1.0)) ** 2 * (1 - el / math.pi)).clamp(0.01, 1.0)
    side = torch.sqrt(u["cal_area"].clamp(min=0) / p)
    th = pw["theta"].gather(2, t[:, :, None]).squeeze(2)                 # bearing shooter -> target
    phi = u["b"].gather(1, t) * (math.pi / 180) - th
    f, dp = pw["front"].gather(1, t), pw["depth"].gather(1, t)
    c, s = torch.cos(phi).abs(), torch.sin(phi).abs()
    across, along = f * c + dp * s, f * s + dp * c
    var = (across ** 2 + side ** 2 + along ** 2 + (side / torch.sin(el)) ** 2) / 12 / 2
    step = float(cal(params).get("landing_step_m", 0.5))
    R = torch.sqrt(4 * var)
    return (torch.round(R / step) * step).clamp(min=step)


def launch(u, pw, params, shots, target, keep=None):
    """In place: the shells of this step's volleys (shots [B, N] projectiles at target [B, N]; keep [B, N]: the share no
    trunk stopped, ground.trunk_keep) take flight - the slot sh_* of each shooter of shells that fired (land() has landed
    a volley still in flight first)."""
    go = shell(u) & (shots > 0) & (target >= 0)
    t = target.clamp(min=0)
    d = pw["dist"].gather(2, t[:, :, None]).squeeze(2)
    T = flight(u, d)
    x = u["x"].gather(1, t) + u["vx"].gather(1, t) * T
    z = u["z"].gather(1, t) + u["vz"].gather(1, t) * T
    n = shots * (keep if keep is not None else 1.0)
    u["sh_s"] = torch.where(go, T, u["sh_s"])
    u["sh_x"] = torch.where(go, x, u["sh_x"])
    u["sh_z"] = torch.where(go, z, u["sh_z"])
    u["sh_n"] = torch.where(go, n, u["sh_n"])
    u["sh_R"] = torch.where(go, landing_radius(u, pw, target, d, params), u["sh_R"])


def _eager(fn):
    return torch.compiler.disable(fn)


@_eager
def land(u, params, dt, front, depth, firing, noise=False):
    """The shells landing this step: (hp, kills, hits, knocks) [B, i, j] - the HP shooter i's shells take from unit j,
    the men they kill, the hits they land (own hits and blasts), the men knocked down; zeros when none lands. A shell in
    flight lands when its time is up within this step or its shooter fires again now (firing [B, N]). front, depth
    [B, N]: the boxes (geometry.dims). In place: the slots' clocks (sh_s -1 once landed)."""
    from tools.nn.sim import melee
    B, N = u["men"].shape
    dev = u["men"].device
    hp = torch.zeros((B, N, N), device=dev)
    kills, hits, knocks = torch.zeros_like(hp), torch.zeros_like(hp), torch.zeros_like(hp)
    fly = u["sh_s"] >= 0
    left = u["sh_s"] - dt
    now = fly & ((left <= 1e-6) | firing)
    u["sh_s"] = torch.where(now, torch.full_like(left, -1.0), torch.where(fly, left, u["sh_s"]))
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
    rows = torch.nonzero(now)
    b_, i_ = rows[:, 0], rows[:, 1]
    R_ = u["sh_R"][b_, i_]
    re_ = u["ex_r"][b_, i_]
    groups = torch.unique(torch.stack([R_, re_], 1), dim=0).tolist()
    for R, r_e in groups:
        sel = (R_ == R) & (re_ == r_e)
        bb, ii = b_[sel], i_[sel]
        nn = u["sh_n"][bb, ii]                                               # [M] shells
        edges, f, m1, m2 = magic.profile(float(R), float(r_e), 1.0, "area", dev)
        # each unit's box seen from the landing point
        dx = u["sh_x"][bb, ii][:, None] - u["x"][bb]                        # [M, N]
        dz = u["sh_z"][bb, ii][:, None] - u["z"][bb]
        sb, cb = torch.sin(rad[bb]), torch.cos(rad[bb])
        along, across = dx * sb + dz * cb, dx * cb - dz * sb
        hf, hd = front[bb] / 2, depth[bb] / 2
        half = math.sqrt(math.pi) / 2 * edges
        inside = (magic._overlap(across[..., None], hf[..., None], half)
                  * magic._overlap(along[..., None], hd[..., None], half))
        band_men = (rho[bb][..., None] * (inside[..., 1:] - inside[..., :-1])).clamp(min=0)    # [M, N, K]
        friend = u["side"][bb] == u["side"][bb, ii][:, None]
        whom = present[bb] & (~friend | u["ex_allies"][bb, ii][:, None])
        band_men = band_men * whom[..., None]
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
        lam_c = nn[:, None, None] * f * (math.pi * r_c ** 2 * cap)[..., None] * fixed[..., None].float()
        # the blast: equal hits of the band's size with the falloff's mean and mean square
        ok = (m1 > 0) & (r_e > 0)
        size = torch.where(ok, m2 / m1.clamp(min=1e-12), torch.zeros_like(m1))                # [K]
        lam_e = torch.where(ok, nn[:, None, None] * (m1 * m1 / m2.clamp(min=1e-12)), torch.zeros_like(m1))
        lam_e = lam_e.expand_as(lam_c)
        res_e = torch.where(u["ex_magic"][bb, ii][:, None], u["resist_magic"][bb], res_m[bb])[..., None]
        arm, hmj = u["armour"][bb][..., None], hm[bb][..., None]
        single = ~many[bb][..., None]
        base_c = (u["m_damage"][bb, ii][:, None] + u["m_bvl"][bb, ii][:, None] * u["large"][bb].float())[..., None]
        ph_c = melee.per_hit(base_c, u["m_ap"][bb, ii][:, None, None].expand_as(base_c), arm, hmj,
                             res_m[bb][..., None], single=single)                                # [M, N, 1]
        ph_e = melee.per_hit(u["ex_d"][bb, ii][:, None, None] * size, u["ex_ap"][bb, ii][:, None, None] * size, arm,
                             hmj, res_e, single=single)                                          # [M, N, K]
        # the knock: the blast within half its radius at dv over the threshold; a fixed shell's own hit by momentum
        mass = u["mass"][bb].clamp(min=1.0)                                                       # [M, N]
        F, S = u["ex_f"][bb, ii][:, None], u["ex_v"][bb, ii][:, None]
        dv = (1 + e_rest) * F * S / (F + mass)
        r_k = torch.where(dv > thr_s, torch.full_like(dv, 0.5 * float(r_e)), torch.zeros_like(dv))
        k_c = (u["c_mom"][bb, ii][:, None] / mass >= thr_c).float() * fixed.float()
        lam_k = nn[:, None, None] * f * (math.pi * (r_c ** 2 * cap * k_c + r_k ** 2))[..., None]
        op = magic.open_share(lam_k, 0.0, t_down)
        knocks.index_put_((bb, ii), (lam_k * op * band_men).sum(-1), accumulate=True)
        lam_c = lam_c * torch.where(ph_c >= hmj - 1e-6, torch.ones_like(op), op)
        lam_e = lam_e * torch.where(ph_e >= hmj - 1e-6, torch.ones_like(op), op)
        Hc, He = lam_c * band_men, lam_e * band_men
        if noise:
            Hc, He = torch.poisson(Hc), torch.poisson(He)
        # the band's men and their health: the unit's wounds by the band's share of its men
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
    return hp, kills, hits, knocks
