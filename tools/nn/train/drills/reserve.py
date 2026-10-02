"""Drill "reserve": the enemy has more units; its main body attacks our line head-on and, once the lines
are engaged, one or two of its units go round our line to our missile units. We keep one or two reserve
units behind our line: thrown into the main melee they leave our missile units (and then the line's back)
to the flankers; kept back to intercept the flankers they win the battle (the user's idea, 02.10.2026).

Frame (one battle):
* ours (the defender: we win at the drill's time limit): a line of L melee infantry units facing the
  enemy, M missile units (indirect fire) BACK_M behind it, R reserve melee units RES_M behind the line;
* the enemy (the attacker): a main body of L .. L + 1 stronger melee units (MAIN) GAP_M away, and F
  flankers behind the ends of its line (FLANK_BACK_M behind it); its script below;
* random: the factions, the unit types, L, M, R, F, the gap, the spacing, the place and bearing on the
  map, our side (1 or 2).
Scripts:
* enemy: the main body attacks the nearest of ours at a run; a flanker (an enemy unit that starts
  behind its line; it keeps a MOVE order or an attack on a missile unit, so the stateless script knows
  it again) walks to a point beside the outer end of our line until its side's units are in melee,
  then attacks our nearest missile unit at a run (our nearest unit when none is left);
* naive (ours): the line and the missile units hold (shoot at will, fight back when attacked); a melee
  unit out of melee joins the fight: it attacks the nearest enemy that is in melee (else holds);
* skilled (ours): the same, but a melee unit out of melee attacks the nearest enemy that is NOT in melee
  within INTERCEPT_M of it or of our missile units (the flankers), and only joins the melee once no
  such enemy is left.
"""
import torch

from tools.nn.sim import orders as O
from tools.nn.train import drills as D

EMPIRE, SKAVEN = "wh_main_emp_empire", "wh2_main_skv_skaven"
LINE = {EMPIRE: ("wh_main_emp_inf_spearmen_0", "wh_main_emp_inf_spearmen_1", "wh_main_emp_inf_swordsmen"),
        SKAVEN: ("wh2_main_skv_inf_clanrat_spearmen_0", "wh2_main_skv_inf_clanrats_1")}
# the enemy's main body: stronger infantry (our line holds it only with the missile units' help)
MAIN = {EMPIRE: ("wh_main_emp_inf_swordsmen", "wh_main_emp_inf_greatswords"),
        SKAVEN: ("wh2_main_skv_inf_clanrats_1",)}
FLANK = {EMPIRE: ("wh_main_emp_inf_swordsmen", "wh_dlc04_emp_inf_flagellants_0"),
         SKAVEN: ("wh2_main_skv_inf_clanrats_1",)}
MAIN_EXTRA = (0, 1)      # the main body's units beyond our line's count (inclusive)
MISSILE = {EMPIRE: ("wh2_dlc13_emp_inf_archers_0",),
           SKAVEN: ("wh2_main_skv_inf_skavenslave_slingers_0", "wh2_main_skv_inf_night_runners_1")}
L_N = (2, 3)             # our line = the enemy's main body (inclusive)
M_N = (2, 3)             # our missile units
R_N = (1, 2)             # our reserve
F_N = (1, 2)             # the enemy's flankers (at most R + ... : drawn up to our reserve's count)
GAP_M = (200.0, 300.0)
BACK_M = (25.0, 35.0)    # our missile units behind the line
RES_M = (60.0, 80.0)     # our reserve behind the line
FLANK_BACK_M = 40.0      # the enemy's flankers behind its line
WIDTH_M = 30.0
MISSILE_WIDTH_M = 35.0
WE_ATTACK = False        # True: we count as the attacker (the enemy wins at the time limit: we must break it in time)
STAGE_OUT_M = 60.0       # the flankers' staging point: this far beyond our line's end ...
STAGE_AHEAD_M = 20.0     # ... and this far in front of it
INTERCEPT_M = 120.0      # skilled: a free enemy this near a reserve unit or our missile units is a flanker


def frame(rng):
    fo = str(rng.choice([EMPIRE, SKAVEN]))
    fe = str(rng.choice([EMPIRE, SKAVEN]))
    pick = lambda pool: str(pool[int(rng.integers(len(pool)))])
    n_line = int(rng.integers(L_N[0], L_N[1] + 1))
    n_mis = int(rng.integers(M_N[0], M_N[1] + 1))
    n_res = int(rng.integers(R_N[0], R_N[1] + 1))
    n_fl = int(rng.integers(F_N[0], min(F_N[1], n_res) + 1))
    gap = float(rng.uniform(*GAP_M))
    lat_gap = float(rng.uniform(4.0, 14.0))
    ours, enemy = [], []
    step = WIDTH_M + lat_gap
    for i in range(n_line):
        ours.append(D.unit(pick(LINE[fo]), *D.local(-gap / 2, (i - (n_line - 1) / 2) * step), 0.0, WIDTH_M))
    back = float(rng.uniform(*BACK_M))
    for i in range(n_mis):
        lat = (i - (n_mis - 1) / 2) * (MISSILE_WIDTH_M + lat_gap)
        ours.append(D.unit(pick(MISSILE[fo]), *D.local(-gap / 2 - back, lat), 0.0, MISSILE_WIDTH_M))
    res = float(rng.uniform(*RES_M))
    for i in range(n_res):
        ours.append(D.unit(pick(LINE[fo]), *D.local(-gap / 2 - res, (i - (n_res - 1) / 2) * step), 0.0, WIDTH_M))
    n_main = n_line + int(rng.integers(MAIN_EXTRA[0], MAIN_EXTRA[1] + 1))
    for i in range(n_main):
        enemy.append(D.unit(pick(MAIN[fe]), *D.local(gap / 2, (i - (n_main - 1) / 2) * step), 180.0, WIDTH_M))
    ends = [1.0, -1.0] if n_fl == 2 else [float(rng.choice([1.0, -1.0]))]
    half = (n_main - 1) / 2 * step
    for s in ends:
        enemy.append(D.unit(pick(FLANK[fe]), *D.local(gap / 2 + FLANK_BACK_M, s * half), 180.0, WIDTH_M))
    return D.army(ours, enemy, attacker=1 if WE_ATTACK else 2, ours_faction=fo, enemy_faction=fe)


def _flankers(st, v):
    """[B, N] the flankers of each side: at the start the units behind their side's melee centre (along the
    side's forward), later the units with a MOVE order or an attack on a missile unit."""
    u = st.u
    (cx, cz), (ex, ez) = v.side_centroid(v.melee)
    fx, fz = ex - cx, ez - cz
    n = torch.sqrt(fx * fx + fz * fz).clamp(min=1e-3)
    ahead = ((u["x"] - cx) * fx + (u["z"] - cz) * fz) / n
    start = (st.t < 0.25)[:, None] & v.melee & (ahead < -0.5 * FLANK_BACK_M)
    tgt = u["order_target"].clamp(min=0)
    on_missile = (u["order_kind"] == O.ATTACK) & (u["order_target"] >= 0) & (u["range"].gather(1, tgt) > 0)
    later = (st.t >= 0.25)[:, None] & v.melee & ((u["order_kind"] == O.MOVE) | on_missile)
    return start | later


def enemy(st):
    v = D.View(st)
    u = st.u
    o = D.nearest(st)                                   # the main body: the nearest, at a run
    fl = _flankers(st, v)
    fighting = torch.zeros_like(v.standing)
    for s in (1, 2):
        mine = u["side"] == s
        fighting = torch.where(mine, (u["m"] & mine & v.standing).any(1, keepdim=True), fighting)
    # the staging point: beside the outer end of the enemy's (our) line, on the flanker's side
    foe_melee = v.foe & v.melee[:, None, :]
    (cx, cz), (ex, ez) = v.side_centroid(v.melee)       # own melee centre, the enemy's melee centre
    fx, fz = cx - ex, cz - ez                           # from the enemy's line towards us
    n = torch.sqrt(fx * fx + fz * fz).clamp(min=1e-3)
    fx, fz = fx / n, fz / n
    rx, rz = fz, -fx                                    # a right of that direction
    lat_e = (u["x"][:, None, :] - ex[:, :, None]) * rx[:, :, None] + (u["z"][:, None, :] - ez[:, :, None]) * rz[:, :, None]
    span = torch.where(foe_melee, lat_e.abs(), torch.zeros_like(lat_e)).amax(2)   # half the enemy line's width
    my_lat = (u["x"] - ex) * rx + (u["z"] - ez) * rz
    sgn = torch.where(my_lat >= 0, torch.ones_like(my_lat), -torch.ones_like(my_lat))
    off = span + WIDTH_M / 2 + STAGE_OUT_M
    px, pz = ex + sgn * off * rx + STAGE_AHEAD_M * fx, ez + sgn * off * rz + STAGE_AHEAD_M * fz
    D.move(o, fl & ~fighting & ~u["m"], px, pz, run=False)
    # once its side fights: our nearest missile unit (the nearest of ours when none is left)
    mis = v.standing & v.missile
    i_m, d_m = v.nearest(mis)
    i_a, _ = v.nearest()
    t = torch.where(d_m < 1e9, i_m, i_a)
    D.attack(o, fl & (fighting | u["m"]) & (v.nearest()[1] < 1e9), t, run=True)
    o.kind = torch.where(v.missile & v.standing, torch.full_like(o.kind, O.HOLD), o.kind)
    o.target = torch.where(o.kind == O.ATTACK, o.target, torch.full_like(o.target, -1))
    return o


def _ours(st, intercept):
    v = D.View(st)
    u = st.u
    o = O.hold(st.B, st.N, st.device)                   # the line and the missile units hold
    idle = v.melee & ~u["m"]
    in_melee = v.standing & u["m"]
    i_j, d_j = v.nearest(in_melee)                      # the nearest enemy in melee: join the fight
    join = idle & (d_j < 1e9)
    if not intercept:
        return D.attack(o, join, i_j, run=True)
    # the flankers: free enemies outside our line - beyond its ends or behind its front - near our missile
    # units or our reserve; the reserve: our idle melee units behind the front line
    (cx, cz), (ex, ez) = v.side_centroid(v.standing)
    fx, fz = ex - cx, ez - cz
    n = torch.sqrt(fx * fx + fz * fz).clamp(min=1e-3)
    fx, fz = fx / n, fz / n                              # each unit's side's forward
    rx, rz = fz, -fx
    ahead = (u["x"][:, None, :] - cx[:, :, None]) * fx[:, :, None] + (u["z"][:, None, :] - cz[:, :, None]) * fz[:, :, None]
    lat = (u["x"][:, None, :] - cx[:, :, None]) * rx[:, :, None] + (u["z"][:, None, :] - cz[:, :, None]) * rz[:, :, None]
    own_melee = v.friend & v.melee[:, None, :]           # [b, i, j]: j a melee unit of i's side
    front = torch.where(own_melee, ahead, torch.full_like(ahead, -1e9)).amax(2)            # [b, i]
    span = torch.where(own_melee & (ahead >= front[:, :, None] - 15.0), lat.abs(), torch.zeros_like(lat)).amax(2)
    my_ahead = torch.diagonal(ahead, dim1=1, dim2=2)
    reserve = idle & (my_ahead < front - 25.0)
    outside = (lat.abs() > span[:, :, None] + WIDTH_M / 2) | (ahead < front[:, :, None] - 10.0)
    mis = v.standing & v.missile
    near_mis = torch.zeros_like(v.d, dtype=torch.bool)  # [b, i, j]: j within INTERCEPT_M of a missile unit of i's side
    for s in (1, 2):
        mine = (u["side"] == s) & mis
        dk = torch.where(mine[:, :, None], v.d, torch.full_like(v.d, 1e9)).min(1).values    # [b, j]
        near_mis = near_mis | ((u["side"] == s)[:, :, None] & (dk <= INTERCEPT_M)[:, None, :])
    free = v.standing & ~u["m"]
    threat = v.foe & free[:, None, :] & outside & ((v.d <= INTERCEPT_M) | near_mis)
    d_t = torch.where(threat, v.d, torch.full_like(v.d, 1e9))
    dt_v, i_t = d_t.min(2)
    cur = u["order_target"].clamp(min=0)
    keep = (u["order_kind"] == O.ATTACK) & (u["order_target"] >= 0) & ~u["m"]         & (v.foe & free[:, None, :]).gather(2, cur[:, :, None]).squeeze(2) & v.melee & (u["range"].gather(1, cur) >= 0)
    keep = keep & reserve
    has_t = reserve & (dt_v < 1e9)
    i_t = torch.where(keep, cur, i_t)
    # the reserve waits while the enemy still has flankers; then joins the melee
    flank_left = _flankers(st, v)
    waiting = torch.zeros_like(v.standing)
    for s in (1, 2):
        mine = u["side"] == s
        waiting = torch.where(mine, (flank_left & ~mine).any(1, keepdim=True), waiting)
    D.attack(o, join & ~(reserve & waiting) & ~has_t & ~keep, i_j, run=True)
    D.attack(o, has_t | keep, i_t, run=True)
    return o


def naive(st):
    return _ours(st, intercept=False)


def skilled(st):
    return _ours(st, intercept=True)


def diag_key(desc, ours):
    m, e = desc["sides"][ours]["units"], desc["sides"][3 - ours]["units"]
    n_mis = sum(u["key"] in sum(MISSILE.values(), ()) for u in m)
    n_res = sum(1 for u in m[1:] if u["key"] in sum(LINE.values(), ())) - 0
    return (desc["sides"][ours]["faction"][-6:], desc["sides"][3 - ours]["faction"][-6:], len(m) - n_mis, len(e))


DRILL = D.Drill("reserve", frame, enemy, naive, skilled,
                "keep the reserve back for the enemy's flankers on our missile units, not in the main melee")
