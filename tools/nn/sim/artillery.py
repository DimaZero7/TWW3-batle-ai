"""Artillery: guns with their crew, their volleys and their shells (docs/ru/training/simulator.md "Артиллерия";
config/nn/sim.json artillery with its sources; the analyses build/diag_arty/empire.md, skaven.md and the game's rules
read from its program, R-work notes/artillery_rules.md). Every number is a passport's (config/nn/units.json `engines`,
`missile`, `missile.blast`: the game's database) or a rule of the game's (config/nn/game_rules.json). Only a unit with
guns (STATIC eng0 > 0) or a shell without guns (STATIC rd, below) is touched: a battle without either steps as before, bit
for bit.

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
program): every gun aims at one man - the target's most central living one with prefer_central_targets (STATIC central;
a holed target's empty centre: live_aim, artillery.aim_live), else the man nearest it (the near edge, under each gun) - at 0.75 of his height, led fully (+ his velocity x T); the miss
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
Shells without guns (artillery.rounds, STATIC rd - by the passport's data: a shockwave blast and no engine; the Warpfire
Throwers, the Globadiers next; build/diag_roster/wh2_main_skv_inf_warpfire_thrower/mechanic.md): the same land() and ring,
with these differences:
  shots       the men's bursts (spv = shots_per_volley x burst_size x projectile_number at the measured rate; missile.burst_share)
  flight      round_delay: the whole stream of a burst arrives its flight later (ammunition too; the reload's clock from the
              start); launch() then lands it the next step at the target's place now: the lead is taken as true
  aim         each crew at its nearest man: uniform over the shooters' front across the fire, at most the target's span
  holes       they start none (a choice of approximation: 6 x 3 cells are coarser than ranks, the box keeps its edge where
              the men are dead)
  box         round_box: the men thin in the box of their full strength, it grows with the men down (fling); line_in_box: a shell
              stops at the box's real edge on its own line
  gravity     the projectile's own (m_g) in every arc; exp_m ends the path
  zeroing-in  a burst counts one shot (the program: each shell as at the burst's start)
  morale      the artillery's only for cat_art: the Warpfire Throwers' category musket has none, its blast sets the -5
  resistance  round_resist (the game's table, M20), fire too; the fire clock of a regeneration (regen.burn)
  the men     a man knocked down is no immune (memory: a hit a man-second 0.95 standing, 1.76 in the first 2 s down); the dead go by
              melee.front_kills, its front pool the third of the men the row most hit holds (kills.missile_front's rule: the dead
              of the shell's first men; memory: dead by depth 59 / 52 / 35 / 14 / 0 %); no fitted number
  lying       the men knocked lie artillery.rounds.lie_s (memory: 5.8 s), their unit's way and blows x the standing share (lie)
  the ring    it ends at its last whole tick (ring_ticks: the Globadiers' 2 m at 15 m/s only 1.5 m) and is a sphere round the burst, the vector to a man's feet: a shell that bursts at the
              height of the heads strikes nobody (exposure_round; bursts: where a rank stops it, how high). The memory's parts of a volley of 48 on spearmen: own hit 15-23 men, ring 37, poison 22 strikes
  the lob     dual_low_fixed (lob_el: the Globadiers' 30 deg): the line closed by friends - the lob's angle at the speed that reaches the aim, else the low
              arc at the muzzle speed (missile.clear_shot's lob; launch, flight, bursts)
  the window  a volley of single shells lands over artillery.volley_land_s (land_w; the men's throws are not together), its aim as it left; a burst's
              stream does not need it
  the poison  a blast's contact effect with direct damage (the catalogue's direct: the Globadiers' poisoned wind): direct.from_contact, each 0.1 s tick with a
              shell's ring on the unit one strike, then its own ticks; info["touch"] the shells whose ring reaches a unit's box
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


def theta_low(v, d, g=G):
    """The low arc's launch angle (rad) to d m at speed v and gravity g (flat ground; 45 deg out of reach)."""
    return 0.5 * torch.asin((d * g / (v * v).clamp(min=1e-6)).clamp(max=1.0))


def flight(u, d, lob=None):
    """[B, N] seconds a shooter's projectile flies to d m [B, N]: the fixed arc's (fix_el > 0, or lob [B, N] at lob_el) sqrt(2 d tan(el) / g),
    else the low arc's d / (v cos theta)."""
    v = u["muzzle_v"].clamp(min=1.0)
    deg = u["fix_el"] if lob is None else torch.where(lob, u["lob_el"], u["fix_el"])
    el = deg * (math.pi / 180)
    g = u["m_g"]
    fixed = torch.sqrt(2 * d.clamp(min=0) * torch.tan(el) / g)
    low = d / (v * torch.cos(theta_low(v, d, g))).clamp(min=1e-3)
    return torch.where(deg > 0, fixed, low)


def shell(u):
    """[B, N] its projectiles are shells (artillery.py launch): a gun's with a blast or a fixed trajectory, or a unit's
    with a blast and no gun (rd)."""
    return ((u["eng0"] > 0) & ((u["ex_r"] > 0) | (u["fix_el"] > 0))) | u["rd"]


def stream(u):
    """[B, N] a shell without guns fired as a burst (bst_rate > 0): its shells arrive as a stream (round_delay), each step's
    land the next, not its own flight later."""
    return u["rd"] & (u["bst_rate"] > 0) & (u["spv"] > 1)


def round_delay(u, pw, target):
    """[B, N] seconds a burst of a shell without guns waits before its stream of hits starts (0 for every other unit;
    missile.burst_share): the flight to the target's near edge, launch()'s. So its own hits and its contact effect come at
    the game's moments too (the first hit 1.1 s after the first shell at 36 m: the record); the ammunition falls that much
    later than the game's (bridge_parity.md)."""
    t = target.clamp(min=0)
    d0 = pw["dist"].gather(2, t[:, :, None]).squeeze(2)
    phi = u["b"].gather(1, t) * (math.pi / 180) - pw["theta"].gather(2, t[:, :, None]).squeeze(2)
    along = pw["front"].gather(1, t) * torch.sin(phi).abs() + pw["depth"].gather(1, t) * torch.cos(phi).abs()
    T = flight(u, (d0 - along / 2).clamp(min=1.0))
    return torch.where(stream(u) & (target >= 0), T, torch.zeros_like(T))


def round_box(u, params):
    """(front, depth) [B, N]: the box shells without guns meet - each unit's at its full strength (geometry.dims at men0): its men thin in it, they do not close up
    under this fire. The memory (build/diag_warpfire/mem_men.py, mem_haz.py): the box does not shrink within a burst - front x depth 25.7 x 8.5 m for 120 men, 28.9 x 10 m for 79
    (the knocked flung out) - and a man's hits a second stay 0.9-1.2 from burst to burst while the men fall 120 -> 79 -> 28: a shell's hits go with the men left. (The game's box does
    re-form between bursts, 25.4 x 5.9 m for 79, and the hits still fall with the men: why, not found; this box is that law, without a number.)"""
    from tools.nn.sim import geometry
    f, d = geometry.dims(dict(u, men=u["men0"]), params.sim["formation"]["spacing_m"])
    # (the knocked men are flung out of their places: the box grows with the share lying, artillery.rounds.fling, measured; 0 lying: as it stood)
    fl = (cal(params).get("rounds") or {}).get("fling") or {}
    return f * (1 + float(fl.get("front", 0.0)) * u["lie_p"]), d * (1 + float(fl.get("depth", 0.0)) * u["lie_p"])


def lie(u, params, dt, knocks):
    """In place: lie_p, lie_s - the share of a unit's men lying from the knocks of shells without guns, with the mean seconds they still
    lie (artillery.rounds.lie_s from each knock; the memory's spell 4.8-5.8 s, a man hit again keeps his): knocks [B, i, j] the men shooter i's
    shells knocked this step; only rows of the class count. Those men take their unit's way and blows out (missile.hit_slow_factor,
    battle.py); nothing lies without such a shell (0: the same bits)."""
    rc = cal(params).get("rounds") or {}
    T = float(rc.get("lie_s") or 0.0)
    if "lie_p" not in u or T <= 0:
        return
    k = torch.where(u["rd"][:, :, None], knocks, torch.zeros_like(knocks)).sum(1)
    p = u["lie_p"]
    add = (k / u["men"].clamp(min=1.0)).clamp(0, 1) * (1 - p)
    p2 = p + add
    s = (u["lie_s"] - dt).clamp(min=0)
    s2 = torch.where(p2 > 0, (p * s + add * T) / p2.clamp(min=1e-9), torch.zeros_like(p2))
    up = (s2 <= 0) | (u["men"] <= 0)
    u["lie_p"] = torch.where(up, torch.zeros_like(p2), p2)
    u["lie_s"] = torch.where(up, torch.zeros_like(s2), s2)


def round_resist(u, magical, spell, fire):
    """[B, i, j] the resistance a shell without guns meets (the game's table, build/diag_skv_mon/mechanic.md M20; the same as
    vortex.hit_hp): missile resistance always, for a shell and its blast alike; the physical unless it is magical, else the
    ward alone; spell resistance only for a spell (is_spell), the fire's (resist_flame: Regeneration's -20 %) for fire; the
    sum within -100 ... 90 %. magical, spell, fire: [B, i] of the shooters. A gun's p_magic / ex_magic path is kept as it was
    (spell resistance for every magical projectile; artillery.rounds.resist_why: not the program's, only the Knights of the
    Blazing Sun would change)."""
    rm = u["resist_missile"][:, None, :]
    base = torch.where(magical[:, :, None], u["resist_all"][:, None, :], u["resist_physical"][:, None, :])
    base = torch.where(spell[:, :, None], u["resist_magic"][:, None, :], base)
    flame = torch.where(fire[:, :, None], u["resist_flame"][:, None, :], torch.zeros_like(base))
    return (rm + base + flame).clamp(-1.0, 0.9)


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


def launch(u, pw, params, shots, target, keep=None, lob=None):
    """In place: the shells of this step's volleys (shots [B, N] projectiles at target [B, N]; keep [B, N]: the share no
    trunk stopped, ground.trunk_keep) take flight - the slot sh_* of each shooter of shells that fired (land() has landed
    a volley still in flight first). The aim (build/diag_arty/mechanic.md 1; STATIC central): every gun takes the
    target's man of the least cost - with prefer_central_targets its most central man (the box's centre), else the man
    nearest the gun (the box's near edge along the line; across, the man under each gun: uniform over the guns' line,
    at most the target's span); the aim point is AIM_H of his height up, so the arc lands AIM_H h / tan(angle) beyond
    him; it leads the target fully (its velocity x the flight). Kept for land(): the line of fire (sh_wx, sh_wz), the
    aim's spread across (sh_aa), the spread's side (sh_r; its zeroing-in by sh_k shots at sh_kt), the distance to the
    aim (sh_d), the launch angle (sh_el) and the target (sh_t). lob [B, N]: the forced arc of missile.clear_shot - a shell with lob_el flies it
    (sh_lb); None: nobody."""
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
    lobbed = (u["lob_el"] > 0) & (lob if lob is not None else torch.zeros_like(u["rd"]))
    deg = torch.where(lobbed, u["lob_el"], u["fix_el"])
    el = torch.where(deg > 0, deg * (math.pi / 180), theta_low(v, d, u["m_g"])).clamp(min=1e-3)
    T = flight(u, d, lobbed)
    flow = stream(u)
    # (a stream of shells: the burst's flight waited already, burst_share; what lands next step is where the target is then)
    T = torch.where(flow, torch.zeros_like(T), T)
    beyond = AIM_H * u["height"].gather(1, t) / torch.tan(el)
    # the guns' line seen across the fire (eng0 guns at their files' centres: (g - 1) / g of the box's front)
    g = u["eng0"].clamp(min=1)
    span = u["width"] * (g - 1) / g * torch.cos(u["b"] * (math.pi / 180) - th).abs()
    # (a shell without guns: each crew at its nearest man, over the whole front of the shooters - no point of its own)
    span = torch.where(u["rd"], pw["front"] * torch.cos(u["b"] * (math.pi / 180) - th).abs(), span)
    aa = torch.where(central, torch.zeros_like(d0), torch.minimum(span, across) / 2)
    x = u["x"].gather(1, t) + u["vx"].gather(1, t) * T + wx * (beyond - near)
    z = u["z"].gather(1, t) + u["vz"].gather(1, t) * T + wz * (beyond - near)
    if aim_live_on(params) and "cellm" in u:
        # (artillery.aim_live) the most central LIVING man: a holed target's centre may be empty - the aim moves to the
        # cell most likely to hold the man nearest the centre (live_aim); a formed target keeps its centre
        x, z = aim_holed(u, params, pw["front"], pw["depth"], t, central, x, z)
    # the zeroing-in: the gun's shots at this target since it was taken (a new target starts it again)
    tf = target.to(u["sh_kt"].dtype)
    k = torch.where(u["sh_kt"] == tf, u["sh_k"], torch.zeros_like(u["sh_k"]))
    n = shots * (keep if keep is not None else 1.0)
    # (the zeroing-in: a gun's shot counts one; a stream's shells the share of their burst - a burst counts one, each shell
    # as at its start: the program)
    k_add = torch.where(flow, shots / (u["men"].clamp(min=1.0) * u["spv"].clamp(min=1.0)), torch.ones_like(k))
    for name, val in (("sh_s", T), ("sh_x", x), ("sh_z", z), ("sh_n", n), ("sh_wx", wx), ("sh_wz", wz), ("sh_aa", aa),
                      ("sh_r", spread_side(u, d, el, k)), ("sh_d", d), ("sh_el", el), ("sh_t", tf), ("sh_k", k + k_add),
                      ("sh_kt", tf), ("sh_lb", lobbed), ("sh_w", u["land_w"]), ("sh_left", n)):
        u[name] = torch.where(go, val, u[name])


def aim_live_on(params):
    """artillery.aim_live (config/nn/sim.json) with artillery.holes on: the central aim takes the living man nearest the
    centre (a formed unit's centre always holds one: only a holed unit's aim moves)."""
    return bool(cal(params).get("aim_live")) and bool(holes_cal(params))


Q_AIM, Q_YAW, Q_PITCH = 3, 10, 24      # the landing's quadrature: the aim across, the yaw, the pitch (midpoints)
CELLS = (6, 3)                         # a unit's box in cells across x along its facing (its men's places)
CHUNK = 16                             # landing rows computed at once (memory: rows x units x cells x points)


def _mid(n, device):
    """n midpoints of [-1, 1]."""
    return (torch.arange(n, dtype=torch.float32, device=device) + 0.5) / n * 2 - 1


def line_in_box(a, phi, F, D):
    """(meets, enter, leave): the line of fire at the offset a across it from a box's centre, the box F across its facing x D along it, its facing phi off
    the line: whether it meets the box, the seconds of path - metres along the line from the centre's foot - where it enters and leaves."""
    c, s = torch.cos(phi), torch.sin(phi)
    # along the facing: |x c + a s| <= D / 2; across it: |-x s + a c| <= F / 2
    cc = c.abs().clamp(min=1e-3)
    ss = s.abs().clamp(min=1e-3)
    sign = lambda x: torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    mid_l, half_l = -a * s * sign(c) / cc, D / 2 / cc
    mid_c, half_c = a * c * sign(s) / ss, F / 2 / ss
    enter = torch.maximum(mid_l - half_l, mid_c - half_c)
    leave = torch.minimum(mid_l + half_l, mid_c + half_c)
    return enter <= leave, enter, leave


def bursts(u, front, depth, b_, i_, heights=None, rank_stop=1.0):
    """[M, Q, 2] the burst points (x, z) of the landing rows' (b_, i_) shells, Q equal-weight points of the game's miss
    (build/diag_arty/mechanic.md 2): across - the aim (uniform +-sh_aa) + d tan(yaw); along - the arc turned by the
    pitch at its speed (a fixed trajectory the speed that reaches d at its angle, a low one its muzzle speed): the range
    v^2 sin 2(el + pitch) / g less the aimed one; the yaw and the pitch uniform within +-atan(sh_r / 2d). A shell coming
    down through its target's box bursts on the first man (percussive, mechanic.md 3): landing past the box's near edge
    within the shadow h / tan(el), at max(near edge, landing - shadow) (the box now, along the line; within its span
    across). heights (a list; only rows of shells without guns): [M, Q] how high each burst is over the men's feet is appended - the shell comes below the heads of any unit on its way, the
    next rank from that box's edge stops it with the chance rank_stop (artillery.rounds.rank_stop; a low-discrepancy draw per point of the miss) or lets it pass; past the last rank, or down
    on the ground first, it bursts at its landing, 0 high."""
    dev = u["men"].device
    M = len(b_)
    d, el, r = u["sh_d"][b_, i_], u["sh_el"][b_, i_], u["sh_r"][b_, i_]
    half = torch.atan(r / (2 * d))
    qa = _mid(Q_AIM, dev)[None, :] * u["sh_aa"][b_, i_][:, None]                       # [M, Qa]
    yaw = _mid(Q_YAW, dev)[None, :] * half[:, None]                                       # [M, Qy]
    pit = _mid(Q_PITCH, dev)[None, :] * half[:, None]                                     # [M, Qp]
    fixed = (u["fix_el"][b_, i_] > 0) | u["sh_lb"][b_, i_]
    g = u["m_g"][b_, i_]
    v2 = torch.where(fixed, d * g / torch.sin(2 * el).clamp(min=1e-3), u["muzzle_v"][b_, i_].clamp(min=1.0) ** 2)
    dl = v2[:, None] * (torch.sin(2 * (el[:, None] + pit).clamp(min=0.0)) - torch.sin(2 * el)[:, None]) / g[:, None]
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
    rd_row = u["rd"][b_, i_]
    if bool(rd_row.any()):
        # (a shell without guns, its ring of 3 m: the box's real near edge on the shell's own line - a corner seen 3 deg off stands 0.8 m before the
        # men of the front, the span's rectangle along the line 12 m at 30 deg; a gun's ring of 5-6 m keeps the span's)
        lat_r, near_r, far_r = line_in_box(A - o_a[:, None, None], phi[:, None, None], F[:, None, None], D[:, None, None])
        rm = rd_row[:, None, None]
        lat = torch.where(rm, lat_r & alive[:, None, None], lat)
        near = torch.where(rm, o_l[:, None, None] + near_r, near)
        far = torch.where(rm, o_l[:, None, None] + far_r, far)
    stop = lat & (L >= near) & (L - shadow <= far)
    Ls = torch.where(stop, torch.maximum(near, L - shadow), L)
    if heights is not None:
        # (all the rows here are shells without guns: land's chunks)
        Qn = L.shape[1] * L.shape[2]

        def first_rank(j, nr, fr, sh, st, salt):
            """[M, Qa Qy, Qp] where a rank of unit j (slots [M]) stops each shell, inf where none does: below the heads from sh before its landing."""
            draw = ((torch.arange(Qn, device=dev, dtype=torch.float32) + 0.5 + salt) * 0.61803399) % 1.0
            passed = torch.floor(torch.log(draw.clamp(min=1e-6)) / math.log(max(1 - rank_stop, 1e-6))).reshape(1, L.shape[1], L.shape[2])
            v_rank = u["sp_v"][b_, j][:, None, None].clamp(min=0.1)
            at = nr + (torch.ceil(((L - sh - nr) / v_rank).clamp(min=0) - 1e-6) + passed) * v_rank
            return torch.where(st & (at <= torch.minimum(L, fr)), at, torch.full_like(L, float("inf")))
        best = first_rank(t, near, far, shadow, stop, 0.0)
        # (any other unit on the shell's way, friends too: its men stop it first - the probe's friends in melee on the line take more than the target)
        for j_ in range(u["men"].shape[1]):
            jj = torch.full_like(t, j_)
            on = (u["side"][b_, jj] > 0) & (u["men"][b_, jj] > 0) & ~u["gone"][b_, jj] & (jj != t) & (jj != i_)
            if not bool(on.any()):
                continue
            ox_j = u["x"][b_, jj] - u["sh_x"][b_, i_]
            oz_j = u["z"][b_, jj] - u["sh_z"][b_, i_]
            ol_j, oa_j = ox_j * wx + oz_j * wz, ox_j * wz - oz_j * wx
            phi_j = u["b"][b_, jj] * (math.pi / 180) - torch.atan2(wx, wz)
            lat_j, nr_j, fr_j = line_in_box(A - oa_j[:, None, None], phi_j[:, None, None], front[b_, jj][:, None, None], depth[b_, jj][:, None, None])
            nr_j, fr_j = ol_j[:, None, None] + nr_j, ol_j[:, None, None] + fr_j
            sh_j = (u["height"][b_, jj] / torch.tan(el).clamp(min=1e-3))[:, None, None]
            st_j = lat_j & on[:, None, None] & (L >= nr_j) & (L - sh_j <= fr_j)
            best = torch.minimum(best, first_rank(jj, nr_j, fr_j, sh_j, st_j, float(j_ + 1)))
        Ls = torch.where(torch.isfinite(best), best, L)
        # (a shell without guns) how high it is where it bursts: the way left to the ground x the arc's slope - the ring is a sphere round that point (ring_ticks, exposure)
        heights.append(((L - Ls).clamp(min=0) * torch.tan(el)[:, None, None]).reshape(M, -1))
    L = Ls.reshape(M, -1)
    A = A.reshape(M, -1)
    exp = u["exp_m"][b_, i_]
    if bool((exp > 0).any()):
        # (artillery.rounds) the path's end: a shell flying past expiry_m bursts there (the launch at d m off its aim)
        L = torch.where((exp > 0)[:, None], torch.minimum(L, (exp - d)[:, None]), L)
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


def ring_ticks(r_e, v):
    """The ticks of 0.1 s a blast's ring grows (shells without guns): it lives max(duration 0, radius / speed) and takes a radius of speed x 0.1 a tick, so it ends at the last
    whole tick - at least one (the program, artillery_rules.md 1.1; the memory, build/diag_globadiers/: the Warpfire Throwers' 3 m at 14 m/s reach 2.8 m, their least hit 6 HP
    = 0.26 of the blast, the Globadiers' 2 m at 15 m/s only 1.5 m, their least hit 13 HP = 0.51 of it)."""
    return torch.floor(r_e / (v.clamp(min=1e-6) * 0.1) + 1e-6).clamp(min=1.0)


def ring_reach(r_e, v):
    """The radius a blast's ring really reaches, m (ring_ticks): r_e where the speed fits it, else its last tick's; the damage still falls over r_e."""
    return torch.minimum(ring_ticks(r_e, v) * v * 0.1, r_e)


def exposure_round(P, h_b, Cx, F, D, rad, r_b, r_e, r_t, smooth):
    """exposure() for a shell without guns, whose ring may be far smaller than a cell: (m1, m2, pk, pc), per shell, [M, N, C], with a cell's men spread evenly over the cell. The ring is a
    sphere round the burst, h_b [M, Q] high, the vector to a man's feet (the game's program; the memory, build/diag_globadiers/mem_rule3d.py: half the men a shell strikes take no ring, it burst
    at their heads; the rule on the real men and the real shells' heights 43 men a volley of 48, HP 760, the game's 37 and 750). On the ground it takes the men within sqrt(r_t^2 + r_b^2 - h^2);
    the disc's overlap with a cell by the square of its area; the closeness at the mean of the disc (2 / 3 of its radius). P [M, Q, 2], Cx [M, N, C, 2], F, D, rad, r_b [M, N], r_e, r_t [M]."""
    ca, cl = CELLS
    dx, dz = P[:, None, None, :, 0] - Cx[..., 0, None], P[:, None, None, :, 1] - Cx[..., 1, None]      # [M, N, C, Q]
    sb, cb = torch.sin(rad)[:, :, None, None], torch.cos(rad)[:, :, None, None]
    a, l = dx * cb - dz * sb, dx * sb + dz * cb
    hx, hl = (F / (2 * ca))[:, :, None, None].clamp(min=1e-3), (D / (2 * cl))[:, :, None, None].clamp(min=1e-3)
    rb = r_b[:, :, None, None]
    re, rt = r_e[:, None, None, None], r_t[:, None, None, None]
    h2 = (h_b ** 2)[:, None, None, :]
    SQ = math.sqrt(math.pi) / 2

    def share(r):
        """[M, N, C, Q] the share of a cell's men within r m of the burst on the ground."""
        w = SQ * r
        ov_a = (torch.minimum(a + w, hx) - torch.maximum(a - w, -hx)).clamp(min=0)
        ov_l = (torch.minimum(l + w, hl) - torch.maximum(l - w, -hl)).clamp(min=0)
        return ov_a * ov_l / (4 * hx * hl)
    r_h = torch.sqrt((rt * rt + rb * rb - h2).clamp(min=0))
    s_h = share(r_h)
    # (the closeness: 1 at the body's skin, 0 at r_e)
    d3 = torch.sqrt((r_h * 2 / 3) ** 2 + h2)
    k = 1 - ((d3 - rb) / re.clamp(min=1e-6)).clamp(0, 1)
    m1, m2 = (s_h * k).mean(-1), (s_h * k * k).mean(-1)
    # (the knock: closeness >= 0.5)
    r_k = torch.sqrt(((rb + re / 2) ** 2 - h2).clamp(min=0))
    pk = share(torch.minimum(r_h, r_k)).mean(-1)
    d = torch.sqrt(a * a + l * l)
    pc = (d < smooth).float().mean(-1) / (math.pi * smooth * smooth)
    return m1, m2, pk, pc


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


def ready_cal(hc):
    """artillery.holes.ready (config/nn/sim.json): the game's rule of a standing unit's re-forming, or {} (the measured
    'knocked >= knock_min, then held until it moves')."""
    r = hc.get("ready") or {}
    return r if r.get("on") else {}


def holes_ready(u, hc, rd, dt, speed):
    """The game's stand order (artillery.holes.ready; R-work notes/p4/formation.md): with an enemy unit within near_m of
    its box, edge to edge (keep_m once in), it is READY and does not re-form at all; else IDLE: it re-forms at the first
    moment min_s after its last re-forming (the mean distance of its men from their places over 2.5 of a man's radius:
    any unit with holes here), every idle_s anyway. A re-forming within delay_s of a man's death (a shell's victim
    flying or down) lays no new places: the men go back to the old ones and the next, min_s on, closes the box.
    -> close [B, N] bool: the units whose box closes now. In place: hole_rd, hole_t, hole_a."""
    from tools.nn.sim import geometry
    holed = u["hole_m"] > 0
    f, d = geometry.dims(dict(u, men=torch.where(holed, u["hole_m"], u["men"])), rd["spacing"])
    gap = geometry.rect_gap(u["x"], u["z"], u["b"], f, d)
    alive = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"]
    foe = (u["side"][:, :, None] != u["side"][:, None, :]) & alive[:, :, None] & alive[:, None, :]
    near = torch.where(foe, gap, torch.full_like(gap, float("inf"))).min(-1).values
    ready = (near < float(rd["near_m"])) | ((u["hole_rd"] > 0) & (near < float(rd["keep_m"])))
    u["hole_rd"] = ready.to(u["hole_rd"].dtype)
    t = u["hole_t"] + dt
    a = torch.where(holed, u["hole_a"] + dt, torch.zeros_like(u["hole_a"]))
    due = holed & ~ready & (t >= float(rd["min_s"]))
    idle = ~holed & ~ready & (t >= float(rd["idle_s"]))
    close = due & (a >= float(rd["delay_s"]))
    t = torch.where(due | idle | (speed > float(hc["move_v"])), torch.zeros_like(t), t)
    u["hole_t"], u["hole_a"] = t, a
    return close


def holes_reform(u, params, dt=0.0):
    """In place, each step (land()): a unit holding its men unformed (hole_m > 0) forms again - its men fill its box
    evenly, the box closes up to its men (geometry.dims) - once it moves (its speed over holes.move_v), routs, re-forms
    to a new width (reform.py), is gone or dead, or (holes.ready) standing by the game's rule, holes_ready (artillery.holes;
    the probes: walking targets close up, standing ones with an enemy near under the Mortar and the rockets do not)."""
    hc = holes_cal(params)
    if not hc or "cellm" not in u:
        return
    speed = torch.sqrt(u["vx"] ** 2 + u["vz"] ** 2)
    go = (u["hole_m"] > 0) & ((speed > float(hc["move_v"])) | u["r"] | (u["men"] <= 0) | u["gone"] | (u["wnew"] >= 0))
    rd = ready_cal(hc)
    if rd:
        rd = dict(rd, spacing=params.sim["formation"]["spacing_m"])
        go = go | holes_ready(u, hc, rd, dt, speed)
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


def live_aim(u, params, front, depth):
    """(dx, dz, moved) [B, N]: where the man nearest a unit's box centre stands, as an offset from the centre, for a unit
    holding its men unformed (hole_m > 0, artillery.holes); moved False (offset 0) for the others and where that man is
    in one of the cells at the centre. The game's prefer_central_targets takes the living man of the least distance to
    the unit's centre (build/diag_arty/mechanic.md 1; R-work shooting_module.md 2.3): with the men by cell (cellm: k
    expected men a cell, a cell holding one with 1 - exp(-k)), the cell holding the nearest man is the one of the
    largest chance 'it holds a man and every cell nearer the centre is empty' (cells by their centres' distance); the
    aim goes to that cell's centre (its men's place in the cells' model, cells())."""
    B, N = u["men"].shape
    ca, cl = CELLS
    C = ca * cl
    dev = u["men"].device
    f, d = holes_box(u, params, front, depth)
    cm = u["cellm"].view(B, N, C + 1)
    k = u["men"].clamp(min=0)[..., None] * cm[..., :C] / cm.sum(-1, keepdim=True).clamp(min=1e-9)     # [B, N, C]
    a = (f[..., None, None] * (_mid(ca, dev) / 2)[:, None]).expand(B, N, ca, cl).reshape(B, N, C)
    l = (d[..., None, None] * (_mid(cl, dev) / 2)[None, :]).expand(B, N, ca, cl).reshape(B, N, C)
    r2 = a * a + l * l
    # (cells by distance, ties by index: stable)
    order = torch.argsort(r2 + torch.arange(C, device=dev) * 1e-6, dim=-1)
    p = 1 - torch.exp(-k.gather(-1, order))
    empty_before = torch.cumprod(torch.cat([torch.ones_like(p[..., :1]), 1 - p[..., :-1]], -1), -1)
    best = order.gather(-1, (p * empty_before).argmax(-1, keepdim=True))                          # [B, N, 1]
    r2b = r2.gather(-1, best).squeeze(-1)
    moved = (u["hole_m"] > 0) & (r2b > r2.min(-1).values * (1 + 1e-4) + 1e-6)
    da, dl = a.gather(-1, best).squeeze(-1), l.gather(-1, best).squeeze(-1)
    rad = torch.deg2rad(u["b"])
    dx = da * torch.cos(rad) + dl * torch.sin(rad)
    dz = -da * torch.sin(rad) + dl * torch.cos(rad)
    zero = torch.zeros_like(dx)
    return torch.where(moved, dx, zero), torch.where(moved, dz, zero), moved


@_eager
def aim_holed(u, params, front, depth, t, central, x, z):
    """(x, z) the aim points of the shooters with a central aim at targets t [B, N], moved to the living man nearest the
    centre where the target is holed (live_aim); as given when no unit is (out of the compiled step: a bool of a tensor)."""
    if not bool((u["hole_m"] > 0).any()):
        return x, z
    lx, lz, moved = live_aim(u, params, front, depth)
    mv = central & moved.gather(1, t)
    return torch.where(mv, x + lx.gather(1, t), x), torch.where(mv, z + lz.gather(1, t), z)


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
    "near" [B, N] (> 0: a shell of the other side landed within near_m of its box's edge - the aim point: morale_mark),
    "own" [B, N] (its men's expected hits from the shells' own projectiles) and "touch" [B, i, j] (the shells of i whose ring reaches j's box:
    a blast's contact effect with direct damage, direct.from_contact; zeros without such a shell).
    A volley with sh_w > 0 lands over that many seconds: dt / sh_w of its shells a step at its aim (sh_left: not landed yet).
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
        holes_reform(u, params, dt)
        front, depth = holes_box(u, params, front, depth)
    fly = u["sh_s"] >= 0
    left = u["sh_s"] - dt
    now = fly & ((left <= 1e-6) | firing)
    # (land_w: the window) this step's share of the volley; the rest waits at sh_s 0 for the next step (none of them with a new volley: all now)
    windowed = u["sh_w"] > 0
    part = torch.where(firing, u["sh_left"], torch.minimum(u["sh_left"], u["sh_n"] * dt / u["sh_w"].clamp(min=1e-6)))
    land_n = torch.where(windowed, part, u["sh_n"])
    rest = windowed & now & (u["sh_left"] - part > 1e-6)
    u["sh_left"] = torch.where(windowed & now, (u["sh_left"] - part).clamp(min=0), u["sh_left"])
    u["sh_s"] = torch.where(rest, torch.zeros_like(left), torch.where(now, torch.full_like(left, -1.0), torch.where(fly, left, u["sh_s"])))
    if info is not None:
        info["near"] = torch.zeros((B, N), device=dev)
        info["own"] = torch.zeros((B, N), device=dev)
        info["touch"] = torch.zeros((B, N, N), device=dev)
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
    # (shells without guns start no holes and their knocks fly nowhere: flung only where a hole may start or is held)
    rounds = bool(u["rd"][now].any())
    if rounds:
        rr_e = round_resist(u, u["ex_magic"], u["ex_spell"], u["ex_fire"])
        rr_c = round_resist(u, u["p_magic"], u["p_spell"], u["p_fire"])
        rd_h, rd_p, rd_f = torch.zeros_like(hp), torch.zeros_like(hp), torch.zeros_like(hp)
    fling = bool(hc) and (bool((u["eng0"][now] > 0).any()) or bool(holed.any()))
    if rounds:
        # (a shell without guns: the men thin in their box as it stood at full strength - round_box; a gun's rows first, then these)
        rows = rows[torch.argsort(u["rd"][rows[:, 0], rows[:, 1]].long(), stable=True)]
        n_gun = int((~u["rd"][rows[:, 0], rows[:, 1]]).sum())
        F_rd, D_rd = round_box(u, params)
    starts = list(range(0, n_gun, CHUNK)) + list(range(n_gun, len(rows), CHUNK)) if rounds else list(range(0, len(rows), CHUNK))
    for lo in starts:
        hi = min(lo + CHUNK, n_gun) if rounds and lo < n_gun else lo + CHUNK
        bb, ii = rows[lo:hi, 0], rows[lo:hi, 1]
        if rounds and lo >= n_gun:
            front, depth = F_rd, D_rd
        nn = land_n[bb, ii]                                                  # [M] shells
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
            near = present[bb] & ~friend0 & (gap <= near_m(params)) & u["cat_art"][bb, ii][:, None]
            info["near"].index_put_((bb,), near.float(), accumulate=True)
        b_h = [] if rounds and lo >= n_gun else None
        P = bursts(u, front, depth, bb, ii, b_h, float((cal(params).get("rounds") or {}).get("rank_stop") or 1.0))    # [M, Q, 2]
        if info is not None and rounds and bool((u["ex_fx"][bb, ii] >= 0).any()):
            # (a blast's contact effect, direct.from_contact) the bursts whose ring + a man's body reaches a box: its first men
            sb, cb = torch.sin(rad[bb])[..., None], torch.cos(rad[bb])[..., None]
            dx, dz = P[:, None, :, 0] - u["x"][bb][..., None], P[:, None, :, 1] - u["z"][bb][..., None]
            reach = (ring_reach(r_e, u["ex_v"][bb, ii])[:, None] + u["radius"][bb])[..., None]
            ins = ((dx * cb - dz * sb).abs() <= front[bb][..., None] / 2 + reach) & ((dx * sb + dz * cb).abs() <= depth[bb][..., None] / 2 + reach)
            tgt = present[bb] & (~(u["side"][bb] == u["side"][bb, ii][:, None]) | u["ex_allies"][bb, ii][:, None])
            info["touch"].index_put_((bb, ii), ins.float().mean(-1) * nn[:, None] * tgt.float() * (u["ex_fx"][bb, ii] >= 0)[:, None].float(), accumulate=True)
        Cx, area = cells(u, front, depth, bb)                                # [M, N, C, 2], [M, N]
        # (a shell without guns: the ring's last whole tick, ring_ticks; a gun's as it was)
        r_t = ring_reach(r_e, u["ex_v"][bb, ii]) if rounds and lo >= n_gun else None
        if rounds and lo >= n_gun:
            m1, m2, pk, pc = exposure_round(P, b_h[0], Cx, front[bb], depth[bb], rad[bb], u["radius"][bb], r_e, r_t, SMOOTH_M)     # [M, N, C]
        else:
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
        res_e = torch.where(u["ex_magic"][bb, ii][:, None], u["resist_magic"][bb], res_m[bb])
        if rounds:
            # (a shell without guns: the game's table, round_resist; a gun's as before)
            res_e = torch.where(u["rd"][bb, ii][:, None], rr_e[bb, ii], res_e)
        res_e = res_e[..., None]
        arm, hmj = u["armour"][bb][..., None], hm[bb][..., None]
        single = ~many[bb][..., None]
        base_c = (u["m_damage"][bb, ii][:, None] + u["m_bvl"][bb, ii][:, None] * u["large"][bb].float())[..., None]
        # (a magical projectile's own hit: spell resistance, not the physical - p_magic)
        res_c = torch.where(u["p_magic"][bb, ii][:, None], (u["resist_missile"] + u["resist_magic"]).clamp(-1.0, 0.9)[bb],
                            res_m[bb]) if "p_magic" in u else res_m[bb]
        if rounds:
            res_c = torch.where(u["rd"][bb, ii][:, None], rr_c[bb, ii], res_c)
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
        # (a shell without guns: no immunity lying - the memory's hits on the men down, artillery module doc)
        op_h = torch.where(u["rd"][bb, ii][:, None, None], torch.ones_like(op), op)
        lam_c = lam_c * torch.where(ph_c >= hmj - 1e-6, torch.ones_like(op), op_h)
        lam_e = lam_e * torch.where(ph_e >= hmj - 1e-6, torch.ones_like(op), op_h)
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
        if rounds:
            # (a shell without guns: its dead by the pools of melee.front_kills, below: the row most hit is the front)
            is_rd = u["rd"][bb, ii][:, None]                                   # [M, 1]
            by_row = He.view(*He.shape[:2], CELLS[0], CELLS[1]).sum(2)         # [M, N, rows]
            rd_h.index_put_((bb, ii), torch.where(is_rd, He.sum(-1), torch.zeros_like(He[..., 0])), accumulate=True)
            rd_p.index_put_((bb, ii), torch.where(is_rd, (He * ph_e).sum(-1), torch.zeros_like(He[..., 0])), accumulate=True)
            rd_f.index_put_((bb, ii), torch.where(is_rd, by_row.max(-1).values, torch.zeros_like(He[..., 0])), accumulate=True)
            ke = torch.where(is_rd[..., None], torch.zeros_like(ke), ke)
            hpe = torch.where(is_rd[..., None], torch.zeros_like(hpe), hpe)
        hp.index_put_((bb, ii), (hpc + hpe).sum(-1), accumulate=True)
        kills.index_put_((bb, ii), (kc + ke).sum(-1), accumulate=True)
        hits.index_put_((bb, ii), (Hc + He).sum(-1), accumulate=True)
        if hc:
            # the dead leave their cells; the men knocked down fly delta m on from the burst (flung: the cell they land
            # in, or out of the box)
            moved = kc + ke
            if delta > 0 and fling:
                down = lam_k * op * band_men
                down = torch.minimum(down, (band_men - kc - ke).clamp(min=0))
                dest = flung(P, Cx, u["radius"][bb], r_e, u["x"][bb], u["z"][bb], u["b"][bb], front[bb], depth[bb], delta)
                moved = moved + down
                kcell.index_put_((bb,), -(down[..., None] * dest).sum(2), accumulate=True)
            kcell.index_put_((bb,), torch.cat([moved, torch.zeros_like(moved[..., :1])], -1), accumulate=True)
    if rounds:
        # the pools (melee.front_kills; module doc): the hits a target took from these shells, their row most hit its front third
        H = rd_h.sum(1)
        ph = rd_p.sum(1) / H.clamp(min=1e-9)
        f = rd_f.sum(1)
        k_t, lost, fw_new = melee.front_kills(torch.minimum(f, H), (H - f).clamp(min=0), men, men / CELLS[1], hm, crew_hp, ph.clamp(min=1e-6), u["fw"])
        taken = torch.where(H > 0, torch.where(many, (H - lost).clamp(min=0) * ph, H * ph), torch.zeros_like(H))
        sh = rd_h / H.clamp(min=1e-9)[:, None, :]
        hp = hp + sh * taken[:, None, :]
        kills = kills + sh * torch.where(many & (H > 0), k_t, torch.zeros_like(k_t))[:, None, :]
        u["fw"] = torch.where(many & (H > 0), fw_new, u["fw"])
    if hc:
        # (artillery.holes) a standing unit whose men the shells knock down (at least knock_min of them this step) holds
        # its men where they stand from now on: its cells keep the men they had less this step's dead there
        # (only a gun's shells: shells without guns start none - the box's cells are coarser than their ranks, module doc)
        by_gun = (u["eng0"] > 0)[:, :, None]
        kn = torch.where(by_gun, knocks, torch.zeros_like(knocks)).sum(1)
        speed = torch.sqrt(u["vx"] ** 2 + u["vz"] ** 2)
        rd = ready_cal(hc)
        struck = (kn + torch.where(by_gun, kills, torch.zeros_like(kills)).sum(1)) > 0 if rd else kn >= float(hc["knock_min"])
        start = (~holed & struck & (speed <= float(hc["move_v"])) & many & ~u["r"] & (u["wnew"] < 0) & present)
        keep = holed | start
        if rd:
            # (holes.ready) a man who dies now holds the re-laying of places off: its clock starts again
            u["hole_a"] = torch.where(struck, torch.zeros_like(u["hole_a"]), u["hole_a"])
        even = torch.cat([(men / C)[..., None].expand(-1, -1, C), torch.zeros_like(men)[..., None]], -1)
        base = torch.where(holed[..., None], men[..., None] * cshare, even)
        new_cm = (base - kcell).clamp(min=0)
        u["cellm"] = torch.where(keep[..., None], new_cm, cm).reshape(B, -1)
        u["hole_m"] = torch.where(start, men, u["hole_m"])
    return hp, kills, hits, knocks
