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
Broad frame (`broad`, drills.BROAD of the battles; the same situation in a messier battle):
* our kiters: M = 2-3 Night Runners, with BROAD_SLING_P one slave slingers unit beside them (run 4.2;
  SLING_CHASERS more swordsmen among the chasers);
* the chasers: as the clean frame's for M = 2; M = 3: two swordsmen and two of CHASERS; with our Warlord
  the Empire general (run 3.4) 40 m behind them, unless the slingers are there (he is not MIN_GAP_MS
  slower than them); no flagellants (unbreakable: kiting cannot break them);
* our background: 0-2 units of Skaven slaves or slave spearmen (SKV_MELEE) and with BROAD_LORD_P the
  Warlord, BEHIND_M behind our kiters; each background unit beyond BG_FREE brings one more chaser
  (BG_CHASER) into the enemy's line; they guard: hold until an enemy comes within GUARD_M, then attack
  the nearest (both check scripts; the network plays them as it likes).
Verified (256 battles, the broad frame only): naive 0.125, skilled 0.871; by part naive / skilled - no
background 0.13 / 0.95, with background 0.12-0.13 / 0.82-0.84, with the Warlord 0.10 / 0.87, M = 2 0.17 /
0.81, M = 3 0.09 / 0.92 (build/xfer, 04.10.2026).
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


# --- the broad frame ---
SLING = ("wh2_main_skv_inf_skavenslave_slingers_0", 4.2, 35.0)
GENERAL = ("wh_main_emp_cha_general_0", 3.4)
SKV_MELEE = ("wh2_main_skv_inf_skavenslaves_0", "wh2_main_skv_inf_skavenslave_spearmen_0")
WARLORD = "wh2_main_skv_cha_warlord_0"
BROAD_SLING_P = 0.35
BG_FREE = 0              # one more chaser (BG_CHASER) in the enemy's line for each background unit of ours beyond
#                          this many
BG_CHASER = CHASERS[0]   # spearmen
SLING_CHASERS = 2        # ... for the slingers
BROAD_LORD_P = 0.5
BEHIND_M = (70.0, 150.0)
GUARD_M = 60.0


def broad(rng):
    m = int(rng.integers(2, 4))
    # the kiters' chasers as in the clean frame (M = 2), M = 3: two swordsmen + two of CHASERS
    if m == 2:
        chasers = [SWORDS, SWORDS] if rng.random() < PAIR_P else [CHASERS[int(rng.integers(len(CHASERS)))] for _ in range(3)]
    else:
        chasers = [SWORDS, SWORDS] + [CHASERS[int(rng.integers(len(CHASERS)))] for _ in range(2)]
    sling = rng.random() < BROAD_SLING_P
    kiters = [OURS] * m + ([SLING] if sling else [])
    if sling:
        chasers += [SWORDS] * SLING_CHASERS
    slowest = min(r for _, r, _ in kiters)
    n_bg = int(rng.integers(0, 3))
    lord = rng.random() < BROAD_LORD_P
    bg = [str(SKV_MELEE[int(rng.integers(len(SKV_MELEE)))]) for _ in range(n_bg)]
    chasers += [BG_CHASER] * max(0, n_bg - BG_FREE)
    general = lord and GENERAL[1] <= slowest - MIN_GAP_MS
    assert all(r <= slowest - MIN_GAP_MS for _, r in chasers)
    gap = float(rng.uniform(*GAP_M))
    lat_gap = float(rng.uniform(10.0, 30.0))
    ours, enemy = [], []
    for i, (key, _, w) in enumerate(kiters):
        lat = (i - (len(kiters) - 1) / 2) * (w + lat_gap)
        ours.append(D.unit(key, *D.local(-gap / 2, lat), 0.0, w))
    back = -gap / 2 - float(rng.uniform(*BEHIND_M))
    rear = bg + ([WARLORD] if lord else [])
    for i, key in enumerate(rear):
        lat = (i - (len(rear) - 1) / 2) * (WIDTH_M + lat_gap) + float(rng.uniform(-20.0, 20.0))
        gen = key == WARLORD
        ours.append(D.unit(key, *D.local(back - (20.0 if gen else 0.0), lat), 0.0, None if gen else WIDTH_M, general=gen))
    line = [key for key, _ in chasers]
    for i, key in enumerate(line):
        lat = (i - (len(line) - 1) / 2) * (WIDTH_M + lat_gap)
        enemy.append(D.unit(key, *D.local(gap / 2, lat), 180.0, WIDTH_M))
    if general:
        enemy.append(D.unit(GENERAL[0], *D.local(gap / 2 + 40.0, float(rng.uniform(-30.0, 30.0))), 180.0, general=True))
    return D.army(ours, enemy, attacker=1, ours_faction=SKAVEN, enemy_faction=EMPIRE,
                  meta={"m": m, "sling": sling, "bg": n_bg, "lord": lord})


def guard(st, o):
    """Our melee units (the broad frame's background) hold until an enemy is within GUARD_M, then attack
    the nearest (in place on the orders o of every unit; returns o)."""
    v = D.View(st)
    i, d = v.nearest()
    return D.attack(o, v.standing & (st.u["range"] <= 0) & (d < GUARD_M), i)


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
    return guard(st, D.hold(st))


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
    o = guard(st, O.hold(st.B, st.N, st.device))
    return D.move(o, go, px, pz, run=True)


# --- transfer: the situation in an ordinary battle (drills/transfer.py) ---
T_NEAR_M = 60.0          # a slower melee enemy this near (centres) ...
T_CLOSING_MS = 0.5       # ... coming at the unit at least this fast (its velocity towards it), or the unit in melee
T_AWAY_MS = 1.0          # applied: the unit moves away from that enemy at least this fast, out of melee


def transfer(st):
    """(situation, applied, mistake) [B, N]: situation = a standing missile unit (with ammunition, not a
    lord) with a standing melee enemy (no missile) at least MIN_GAP_MS slower than it (run speeds) within
    T_NEAR_M that comes at it (T_CLOSING_MS) or that it is already in melee with; applied = it runs back:
    out of melee, moving away from the nearest such enemy at T_AWAY_MS or more; mistake = it is in melee."""
    v = D.View(st)
    u = st.u
    me = v.standing & v.missile & ~u["lord"]
    slower = u["run"][:, None, :] <= u["run"][:, :, None] - MIN_GAP_MS                        # [B, i, j]
    melee_j = (u["range"] <= 0)[:, None, :]
    d = v.d.clamp(min=1e-3)
    # (dx, dz: from i to j) j's speed towards i, i's speed away from j
    towards = -(u["vx"][:, None, :] * v.dx + u["vz"][:, None, :] * v.dz) / d
    away = -(u["vx"][:, :, None] * v.dx + u["vz"][:, :, None] * v.dz) / d
    caught = u["m"][:, :, None]
    threat = v.foe & melee_j & slower & (v.d < T_NEAR_M) & ((towards > T_CLOSING_MS) | caught)
    sit = me & threat.any(2)
    near = torch.where(threat, v.d, torch.full_like(v.d, 1e9)).argmin(2, keepdim=True)
    applied = sit & ~u["m"] & (away.gather(2, near).squeeze(2) >= T_AWAY_MS)
    return sit, applied, sit & u["m"]


DRILL = D.Drill("kiting", frame, enemy, naive, skilled,
                "faster missile units against slower chasers: shoot, run back, shoot", broad=broad, transfer=transfer)
