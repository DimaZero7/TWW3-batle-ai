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
Embedded frame (`embedded`, drills.EMBED of the battles, the default: the situation inside a normal battle,
so nothing but the situation tells it from one): a normal generated battle (tools/nn/armies, our side the
Skaven, the enemy the Empire - the only pair with a missile unit MIN_GAP_MS faster than the other side's
infantry; both lords, normal sizes, the arena's deployment, attacking or defending) with M = 1-2 Night Runners
(TAG_OURS) EMB_GAP_M beyond the end of our line on one flank, EMB_BACK_M behind our front, and the clean
frame's chasers for M (TAG_ENEMY) at the end of the enemy's front line on the same flank, marching in its line
until one of our kiters comes within EMB_CHASE_M (then they CHASE: the situation starts as in a normal battle,
when our shooters come near slower infantry); each side's
formation shifted sideways by half its inserted block (centred as a normal deployment); for every inserted
unit its side gave up the unit of nearest cost and was deployed again without them (drills.generated: a
normal battle's unit count and gold, no hole in the lines), the gold evened (drills.balance). The enemy plays `ai_like`, its tagged chasers
CHASE our tagged kiters once near; the check scripts play `ai_like` for our army, the kiters as the clean scripts
(naive: hold; skilled: kite). The rest of the battle is a normal one, so the separation is in the gold
trade (paired, same battles), not the win rate.
Not in the frame: skavenslave slingers (run 4.2; their 7-damage stones do not break Empire infantry
within the ammunition even kited), greatswords (armour 95: neither script breaks them), Skaven
infantry chasing Night Runners (4.2 against 5.4: standing already wins against the slaves).
Scripts:
* naive: hold - stand and shoot at will (the chasers reach us and win the melee);
* skilled: stand and shoot while the nearest chaser is farther than RUN_M (centre to centre); then
  run back (away from the chasers near, bent sideways and towards the map's centre near its edge)
  until it is STOP_M or farther, then halt and shoot again (short stops: every halt gives a volley
  of the men reloaded meanwhile, tools/nn/sim/missile.py).
In an embedded battle both check scripts are `ai_like` for every unit of ours but the tagged kiters.
The teacher (drills/teach.py) labels with `kite` (the skilled rule for every unit, tag-blind) at the
situation's `moments` (the transfer detector's, and the run-back it keeps going) in embedded and normal
battles; on every unit in the clean and broad frames.
"""
import torch

from tools.nn.sim import orders as O
from tools.nn.train import drills as D
from tools.nn.train import opponents

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


# --- the embedded frame ---
EMB_GAP_M = (6.0, 20.0)      # from the end of our line to our first kiter's edge (and of the enemy's to its chasers;
#                              the deployment's own gap between units is 6 m, tools/nn/armies/place.py)
EMB_LAT_GAP_M = 6.0          # between the inserted units, as in a deployed line
EMB_BACK_M = (0.0, 25.0)     # our kiters this far behind our front line (the missile line stands ~25 m back)
EMB_AHEAD_M = (0.0, 0.0)     # the chasers this far ahead of the enemy's front line (in it)
EMB_CHASE_M = 180.0          # the chasers march in ai_like's line until a tagged kiter is this near (sling 140 m)


def embedded(rng):
    best = None
    for _ in range(D.EMBED_TRIES):
        m = int(rng.integers(1, 3))
        if m == 1:
            chasers = [SWORDS]
        elif rng.random() < PAIR_P:
            chasers = [SWORDS, SWORDS]
        else:
            chasers = [CHASERS[int(rng.integers(len(CHASERS)))] for _ in range(3)]
        desc = D.generated(rng, SKAVEN, EMPIRE, swap={1: [OURS[0]] * m, 2: [k for k, _ in chasers]})
        ax = D.axes(desc, 1)
        ours, enemy = desc["sides"][1]["units"], desc["sides"][2]["units"]
        b_ours, b_enemy = ours[0]["b"], enemy[0]["b"]
        s = float(rng.choice([-1.0, 1.0]))
        lat_gap = EMB_LAT_GAP_M
        # our kiters beyond our line's end on flank s, a little behind our front
        start = D.extent(ours, ax, s) + float(rng.uniform(*EMB_GAP_M))
        f_k = D.front(ours, ax) - float(rng.uniform(*EMB_BACK_M))
        w = OURS[2]
        kiters = [D.unit(OURS[0], *ax.world(f_k, s * (start + i * (w + lat_gap) + w / 2)), b_ours, w, tag=D.TAG_OURS)
                  for i in range(m)]
        ours += kiters
        D.shift(ours, ax, -s * (m * w + (m - 1) * lat_gap) / 2)
        # the chasers beyond the same flank of the enemy's line, a little ahead of its front
        start = D.extent(enemy, ax, s) + float(rng.uniform(*EMB_GAP_M))
        f_c = D.front(enemy, ax, enemy=True) - float(rng.uniform(*EMB_AHEAD_M))
        k = len(chasers)
        enemy += [D.unit(key, *ax.world(f_c, s * (start + i * (WIDTH_M + lat_gap) + WIDTH_M / 2)), b_enemy, WIDTH_M,
                         tag=D.TAG_ENEMY) for i, (key, _) in enumerate(chasers)]
        D.shift(enemy, ax, -s * (k * WIDTH_M + (k - 1) * lat_gap) / 2)
        desc["meta"] = {"m": m, "chasers": k}
        if D.balance(desc):
            return desc
        if best is None or D.imbalance(desc) < D.imbalance(best):
            best = desc
    return best


def guard(st, o):
    """Our melee units (the broad frame's background) hold until an enemy is within GUARD_M, then attack
    the nearest (in place on the orders o of every unit; returns o)."""
    v = D.View(st)
    i, d = v.nearest()
    return D.attack(o, v.standing & (st.u["range"] <= 0) & (d < GUARD_M), i)


def chase(st):
    """CHASE: every standing unit attacks (running) the nearest standing tagged unit of the enemy (an
    embedded battle's kiters), else the nearest standing enemy missile unit, else the nearest standing enemy."""
    v = D.View(st)
    i_t, d_t = v.nearest(D.tagged(st, D.TAG_OURS) | D.tagged(st, D.TAG_ENEMY))
    i_m, d_m = v.nearest(v.missile)
    i_a, d_a = v.nearest()
    tgt = torch.where(d_t < 1e9, i_t, torch.where(d_m < 1e9, i_m, i_a))
    o = O.hold(st.B, st.N, st.device)
    return D.attack(o, v.standing & (d_a < 1e9), tgt, run=True)


def _embed(st, clean, tag, inner):
    """Orders: `clean` in the clean / broad battles; in the embedded ones `ai_like`, with `inner`'s orders for
    the units tagged `tag`."""
    emb = D.embedded_rows(st)[:, None]
    mixed = O.merge(opponents.ai_like(st), inner, D.tagged(st, tag))
    return O.merge(clean, mixed, emb.expand_as(st.u["tag"]))


def enemy(st):
    """CHASE (the clean / broad frames: every unit); in an embedded battle `ai_like`, its tagged chasers too
    (they march in its line) until one of our tagged kiters is within EMB_CHASE_M of them, then CHASE (and
    keep chasing while their attack order is on a tagged unit of ours)."""
    o = chase(st)
    v = D.View(st)
    _, d_t = v.nearest(D.tagged(st, D.TAG_OURS))
    u = st.u
    on = (u["order_kind"] == O.ATTACK) & (u["order_target"] >= 0)
    chasing = on & (u["tag"].gather(1, u["order_target"].clamp(min=0)) == D.TAG_OURS)
    go = D.tagged(st, D.TAG_ENEMY) & ((d_t < EMB_CHASE_M) | chasing)
    emb = D.embedded_rows(st)[:, None].expand_as(go)
    return O.merge(o, O.merge(opponents.ai_like(st), o, go), emb)


def naive(st):
    """Hold, the broad frame's background guards; in an embedded battle `ai_like`, the tagged kiters hold."""
    return _embed(st, guard(st, D.hold(st)), D.TAG_OURS, D.hold(st))


def skilled(st):
    """kite(); in an embedded battle `ai_like`, the tagged kiters kite."""
    o = kite(st)
    return _embed(st, o, D.TAG_OURS, o)


def kite(st):
    """The skill, tag-blind (the clean frame's skilled script; the teacher's labels): every standing missile
    unit stands and shoots, runs back from an enemy within RUN_M until it is STOP_M away; melee units guard."""
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


def moments(st, o):
    """[B, N] where the teacher's orders `o` (kite) label a unit outside the clean frames: the situation
    (transfer) and, after it, while the unit runs back on the script's word (its move order on, the script
    says move, a slower melee enemy within STOP_M) - the run-back kept to its end."""
    v = D.View(st)
    u = st.u
    sit = transfer(st)[0]
    slower = u["run"][:, None, :] <= u["run"][:, :, None] - MIN_GAP_MS
    chaser = v.foe & (u["range"] <= 0)[:, None, :] & slower & (v.d < STOP_M)
    running = v.standing & v.missile & ~u["lord"] & (o.kind == O.MOVE) & (u["order_kind"] == O.MOVE) & chaser.any(2)
    return sit | running


DRILL = D.Drill("kiting", frame, enemy, naive, skilled,
                "faster missile units against slower chasers: shoot, run back, shoot", broad=broad, transfer=transfer,
                embedded=embedded, teacher=kite, moments=moments)
