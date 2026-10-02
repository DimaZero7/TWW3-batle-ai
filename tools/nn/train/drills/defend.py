"""Drill "defend": we defend with missile units and a formed line against an attacker of stronger
melee; marching out to meet it loses, holding the line and shooting wins (the network marched out
150-250 m as the defender in the game: the habit this drill breaks).

The simulator is flat: the defender's advantage is its missile units' fire on the attacker's walk
in, a formed line, and being fresh (the attacker comes at a run, tools/nn/sim/fatigue.py).

Frame (one battle):
* ours (the defender: wins at the drill's time limit), Skaven: a line of L = 2-3 melee infantry units
  (clanrat spearmen, slave spearmen) facing the enemy, and 4 missile units (Night Runners, slave
  slingers; indirect fire) in a second row BACK_M behind it;
* the enemy (the attacker): L + 2 melee infantry units (Empire swordsmen or flagellants, or Skaven
  clanrats: stronger one for one), GAP_M (300-450 m) away facing us; it plays `ai_like` (a line that
  advances at a run and charges from ~80 m);
* random: the enemy's faction, the units, L, the gap, the spacing, the place and bearing on the map,
  our side (1 or 2).
Scripts (ours):
* naive: every unit attacks the nearest enemy, running (the line marches out and meets the enemy
  half-way, 125-225 m out; missile units run up to range and shoot);
* skilled: hold the position: missile units shoot at will; a melee unit counter-charges only an
  enemy within REACT_M (or fights back when attacked) - opponents.hold_shoot as the defender.
"""
import torch

from tools.nn.train import drills as D
from tools.nn.train import opponents

EMPIRE, SKAVEN = "wh_main_emp_empire", "wh2_main_skv_skaven"
# our line: ordinary melee infantry
OURS_LINE = {EMPIRE: ("wh_main_emp_inf_spearmen_0", "wh_main_emp_inf_spearmen_1"),
             SKAVEN: ("wh2_main_skv_inf_clanrat_spearmen_0", "wh2_main_skv_inf_skavenslave_spearmen_0")}
# our missile units (indirect fire: they shoot over the line)
OURS_MISSILE = {EMPIRE: ("wh2_dlc13_emp_inf_archers_0",),
                SKAVEN: ("wh2_main_skv_inf_night_runners_1", "wh2_main_skv_inf_skavenslave_slingers_0")}
# the attacker's line: stronger melee infantry
ENEMY_LINE = {EMPIRE: ("wh_main_emp_inf_swordsmen", "wh_dlc04_emp_inf_flagellants_0"),
              SKAVEN: ("wh2_main_skv_inf_clanrats_1",)}
ENEMY_MISSILE = {EMPIRE: ("wh2_dlc13_emp_inf_archers_0",), SKAVEN: ("wh2_main_skv_inf_skavenslave_slingers_0",)}
# we are the Skaven: with Empire archers and spearmen holding wins and marching out wins too (128 battles:
# naive 0.5-1.0 by roster), so the Empire defender is left out
OURS_FACTIONS = (SKAVEN,)
ENEMY_FACTIONS = (EMPIRE, SKAVEN)
LINE_N = (2, 3)          # our melee units (inclusive)
MISSILE_N = (4, 4)       # our missile units (inclusive)
ENEMY_EXTRA = (2, 2)     # the enemy's melee units beyond our line's count (inclusive)
ENEMY_MISSILE_P = 0.0    # the enemy has one missile unit this often
GAP_M = (300.0, 450.0)
BACK_M = (20.0, 35.0)    # our missile row behind the line
WIDTH_M = 30.0
MISSILE_WIDTH_M = 35.0
REACT_M = 40.0           # skilled: a melee unit counter-charges an enemy this close
OUT_S = 60.0             # naive: the army marches out this long (at a run: ~190 m) ...
OUT_STOP_M = 80.0        # ... or until an enemy is this close, then attacks the nearest


def frame(rng):
    fo = str(rng.choice(OURS_FACTIONS))
    fe = str(rng.choice(ENEMY_FACTIONS))
    pick = lambda pool: str(pool[int(rng.integers(len(pool)))])
    n_line = int(rng.integers(LINE_N[0], LINE_N[1] + 1))
    n_mis = int(rng.integers(MISSILE_N[0], MISSILE_N[1] + 1))
    n_enemy = n_line + int(rng.integers(ENEMY_EXTRA[0], ENEMY_EXTRA[1] + 1))
    gap = float(rng.uniform(*GAP_M))
    back = float(rng.uniform(*BACK_M))
    lat_gap = float(rng.uniform(4.0, 16.0))
    ours, enemy = [], []
    for i in range(n_line):
        lat = (i - (n_line - 1) / 2) * (WIDTH_M + lat_gap)
        x, z = D.local(-gap / 2, lat)
        ours.append(D.unit(pick(OURS_LINE[fo]), x, z, 0.0, WIDTH_M))
    for i in range(n_mis):
        lat = (i - (n_mis - 1) / 2) * (MISSILE_WIDTH_M + lat_gap)
        x, z = D.local(-gap / 2 - back, lat)
        ours.append(D.unit(pick(OURS_MISSILE[fo]), x, z, 0.0, MISSILE_WIDTH_M))
    for i in range(n_enemy):
        lat = (i - (n_enemy - 1) / 2) * (WIDTH_M + lat_gap)
        x, z = D.local(gap / 2, lat)
        enemy.append(D.unit(pick(ENEMY_LINE[fe]), x, z, 180.0, WIDTH_M))
    if rng.random() < ENEMY_MISSILE_P:
        x, z = D.local(gap / 2 + back, 0.0)
        enemy.append(D.unit(pick(ENEMY_MISSILE[fe]), x, z, 180.0, MISSILE_WIDTH_M))
    return D.army(ours, enemy, attacker=2, ours_faction=fo, enemy_faction=fe)


def enemy(st):
    return opponents.ai_like(st)


def naive(st):
    v = D.View(st)
    o = D.nearest(st)
    _, d = v.nearest()
    (_, _), (ex, ez) = v.side_centroid(v.standing)
    x, z = st.u["x"], st.u["z"]
    dx, dz = ex - x, ez - z
    n = torch.sqrt(dx * dx + dz * dz).clamp(min=1.0)
    out = v.standing & (st.t[:, None] < OUT_S) & (d > OUT_STOP_M) & (d < 1e9)
    return D.move(o, out, x + dx / n * 40.0, z + dz / n * 40.0, run=True)


def skilled(st):
    return opponents.hold_shoot(st, react_m=REACT_M)


def diag_key(desc, ours):
    """(our faction, the enemy's line units, our line and missile counts) for diagnosis."""
    m, e = desc["sides"][ours]["units"], desc["sides"][3 - ours]["units"]
    n_mis = sum(u["key"] in sum(OURS_MISSILE.values(), ()) for u in m)
    return (desc["sides"][ours]["faction"][-6:], e[0]["key"].split("_inf_")[-1][:10], len(m) - n_mis, n_mis)


DRILL = D.Drill("defend", frame, enemy, naive, skilled,
                "defend with missiles and a line vs stronger melee: hold and shoot, do not march out")
