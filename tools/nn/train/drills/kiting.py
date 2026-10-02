"""Drill "kiting": our missile units are faster than the melee infantry chasing them; standing and
shooting gets them caught in melee and beaten, shoot - run back - shoot breaks the chasers with the
ammunition.

Frame (one battle):
* ours: M = 1-2 Skaven Night Runners (run 5.4 m/s, sling range 140 m, 22 shots a man) side by side,
  facing the enemy; we attack - running away until the time limit loses (the defender wins at the
  drill's limit), so the chasers must be broken or killed;
* the enemy: K Empire melee infantry (run 3.0 m/s: 2.4 slower than ours), GAP_M away in a line
  facing us: M = 1 -> one swordsmen unit; M = 2 -> two swordsmen (PAIR_P) or three of spearmen,
  shielded spearmen, swordsmen. These are the rosters where standing loses the melee (the measured
  win rates are in the drill's notes, docs/en/training/training.md "Drills"). The `enemy` script:
  every unit runs at our nearest missile unit (CHASE);
* random: M, K, the chasers' types, the gap, the lines' spacing, the place and bearing on the map,
  our side (1 or 2).
Not in the frame: skavenslave slingers (run 4.2; their 7-damage stones do not break Empire infantry
within the ammunition even kited), greatswords (armour 95: neither script breaks them), Skaven
infantry chasing Night Runners (4.2 against 5.4: standing already wins against the slaves).
Scripts:
* naive: hold - stand and shoot at will (the chasers reach us and win the melee);
* skilled: stand and shoot while the nearest chaser is farther than RUN_M (centre to centre); then
  run back (away from the chasers near, bent sideways and towards the map's centre near its edge)
  until it is STOP_M or farther, then halt and shoot again (short stops: every halt gives a volley
  of the men reloaded meanwhile, tools/nn/sim/missile.py).
"""
import torch

from tools.nn.sim import orders as O
from tools.nn.train import drills as D

EMPIRE, SKAVEN = "wh_main_emp_empire", "wh2_main_skv_skaven"
# (key, run m/s, width m) - our missile unit (config/nn/units.json; width: config/nn/pools.json)
OURS = ("wh2_main_skv_inf_night_runners_1", 5.4, 30.0)
# (key, run m/s) - the Empire melee infantry that chases (not unbreakable: the skill is breaking it)
SWORDS = ("wh_main_emp_inf_swordsmen", 3.0)
CHASERS = (("wh_main_emp_inf_spearmen_0", 3.0), ("wh_main_emp_inf_spearmen_1", 3.0), SWORDS)
MIN_GAP_MS = 1.0         # every chaser runs at least this much slower than every unit of ours
PAIR_P = 1 / 3           # M = 2: two swordsmen with this chance, else three chasers of CHASERS
GAP_M = (150.0, 300.0)   # centre to centre, our line to the enemy's
WIDTH_M = 30.0
RUN_M = 45.0             # a chaser this near (centres): run back
STOP_M = 65.0            # ... until the nearest is this far: halt and shoot
AWAY_M = 60.0            # the retreat point this far ahead
NEAR_M = 250.0           # chasers within this push the retreat
EDGE_M = 450.0           # from here to EDGE_M + BEND_M from the map's centre the retreat bends inward
BEND_M = 350.0
INWARD = 0.5             # at the edge: sideways plus this much towards the centre


def frame(rng):
    m = int(rng.integers(1, 3))
    if m == 1:
        chasers = [SWORDS]
    elif rng.random() < PAIR_P:
        chasers = [SWORDS, SWORDS]
    else:
        chasers = [CHASERS[int(rng.integers(len(CHASERS)))] for _ in range(3)]
    assert all(r <= OURS[1] - MIN_GAP_MS for _, r in chasers)
    k = len(chasers)
    gap = float(rng.uniform(*GAP_M))
    lat_gap = float(rng.uniform(10.0, 30.0))
    ours, enemy = [], []
    for i in range(m):
        lat = (i - (m - 1) / 2) * (OURS[2] + lat_gap)
        x, z = D.local(-gap / 2, lat)
        ours.append(D.unit(OURS[0], x, z, 0.0, OURS[2]))
    for i, (key, _) in enumerate(chasers):
        lat = (i - (k - 1) / 2) * (WIDTH_M + lat_gap)
        x, z = D.local(gap / 2, lat)
        enemy.append(D.unit(key, x, z, 180.0, WIDTH_M))
    return D.army(ours, enemy, attacker=1, ours_faction=SKAVEN, enemy_faction=EMPIRE)


def enemy(st):
    """CHASE: every standing unit attacks (running) the nearest standing enemy missile unit, else
    the nearest standing enemy."""
    v = D.View(st)
    i_m, d_m = v.nearest(v.missile)
    i_a, d_a = v.nearest()
    tgt = torch.where(d_m < 1e9, i_m, i_a)
    o = O.hold(st.B, st.N, st.device)
    return D.attack(o, v.standing & (d_a < 1e9), tgt, run=True)


def naive(st):
    return D.hold(st)


def skilled(st):
    v = D.View(st)
    u = st.u
    x, z = u["x"], u["z"]
    _, dist = v.nearest()
    # away from the enemies near, weighted by nearness (dx, dz: from i to j)
    near = v.foe & (v.d < NEAR_M)
    w = torch.where(near, 1.0 / v.d.clamp(min=5.0) ** 2, torch.zeros_like(v.d))
    ax = -(w * v.dx / v.d.clamp(min=1e-3)).sum(2)
    az = -(w * v.dz / v.d.clamp(min=1e-3)).sum(2)
    n = torch.sqrt(ax * ax + az * az).clamp(min=1e-9)
    ax, az = ax / n, az / n
    # near the map's edge: bend sideways (the side nearer the centre) and inward
    r = torch.sqrt(x * x + z * z).clamp(min=1e-3)
    cx, cz = -x / r, -z / r
    tx, tz = -az, ax                                   # a side of the retreat line
    sgn = torch.where(tx * cx + tz * cz >= 0, torch.ones_like(tx), -torch.ones_like(tx))
    tx, tz = tx * sgn, tz * sgn
    bend = ((r - EDGE_M) / BEND_M).clamp(0, 1)
    dx = ax * (1 - bend) + (tx + INWARD * cx) * bend
    dz = az * (1 - bend) + (tz + INWARD * cz) * bend
    n = torch.sqrt(dx * dx + dz * dz).clamp(min=1e-9)
    px, pz = x + dx / n * AWAY_M, z + dz / n * AWAY_M
    running = u["order_kind"] == O.MOVE
    go = v.standing & v.missile & (dist < 1e9) & ((dist < RUN_M) | (running & (dist < STOP_M)))
    o = O.hold(st.B, st.N, st.device)
    return D.move(o, go, px, pz, run=True)


DRILL = D.Drill("kiting", frame, enemy, naive, skilled,
                "faster missile units against slower chasers: shoot, run back, shoot")
