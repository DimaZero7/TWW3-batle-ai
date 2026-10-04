"""Drill "hold_fire": our infantry is locked in melee with the enemy lord; our missile
units shooting into that melee kill more of ours than of him (friendly fire); they must shoot the free
enemies instead and, when none are left, hold fire on the melee.

The simulator's rule (tools/nn/sim/missile.py volley, config/nn/sim.json missile): of the hits aimed at a
unit in melee, friendly_fire (0.26 arrows, 0.56 sling) lands on the shooter's own units in contact with it,
at their armour; the rest on the target - a lord takes single_entity_factor 0.43 of it through 85-90 armour
and a shield, an armoured unit through its armour. Measured here (2 units of shooters, 200 s, see the
drill's test and docs/en/training/training.md "Drills"): into a lord the shooters take ~4-10 HP of ours
per HP of his, into Greatswords ~1-3; into an ordinary unit fire pays.

Frame (one battle, our side = 1; we attack, the enemy wins at the battle limit):
* the melee: our infantry unit (OURS_MELEE of our faction) in contact with the enemy's lord (fire into an
  ordinary or armoured unit in melee pays in the simulator: measure() below);
* our shooters: K = 2-3 missile units of one kind (SHOOT: Empire archers, Skaven Night Runners or slave
  slingers; arcing fire: they can shoot into the melee) in a line DIST_M behind our infantry, facing it;
* free enemies: F missile units of the enemy's faction (FREE; F by our shooters, SHOOT) standing off one end (or both ends)
  of our shooters' line, LAT_M beyond it: nearer to our shooters than to the melee, farther from them
  than the melee is;
* random: factions, units, K, F, the distances, the place and bearing on the map, our side (1 or 2).
Broad frame (`broad`, drills.BROAD of the battles; the same situation in a messier battle): the clean
frame, and with MIX_P one of our shooters of another kind of our faction (Empire: Free Company militia,
which shoots on the move; Skaven: the other of Night Runners / slave slingers); with OUR_LORD_P our lord
OUR_LORD_M behind our shooters (holds: uninvolved); with SECOND_P a second melee SECOND_M to the side of
the first: one of our infantry units in contact with one of the enemy's (an ordinary unit: fire into it
pays), and with FAR_P an enemy melee unit FAR_M behind the enemy's lord (it comes on, as every enemy melee
unit). Verified (256 battles): the broad frame only naive 0.164 / skilled 0.855; the 50 % mix 0.117 /
0.848 (04.10.2026).
Embedded frame (`embedded`, drills.EMBED of the battles, the default: the situation inside a normal battle):
a normal generated battle (tools/nn/armies: factions at random, both lords, normal sizes, the arena's
deployment, attacking or defending) with the clean frame inserted beyond one flank of our line: our infantry
unit (TAG_OURS) EMB_GAP_M beyond our line's end, EMB_AHEAD_M ahead of our front, in contact with the ENEMY'S
OWN LORD (taken out of his army, TAG_ENEMY); our K shooters (TAG_OURS) DIST_M behind it; the free enemy
missile units (TAG_ENEMY) LAT_M beyond the outer end of our shooters' line; the armies bought for the drawn
for every inserted unit its side gave up the unit of nearest cost and was deployed again without them
(drills.generated), the gold evened (drills.balance). The enemy plays `ai_like`, its tagged units as the clean enemy (the lord keeps fighting,
the free missile units hold and shoot); the check scripts `ai_like` for our army, the tagged units as the
clean scripts.
Scripts:
* enemy: melee units attack the nearest enemy (the engaged one keeps fighting), missile units hold
  (shoot at will at the nearest);
* naive: our shooters attack the nearest enemy single entity in melee (focus fire on the lord), else the
  nearest enemy in melee; melee units (not our lord: he holds) the nearest enemy;
* skilled: our shooters attack the nearest free (not in melee) enemy, else the nearest ordinary (not a
  single entity) enemy in melee; with none left they step out of range of the melee (HOLD_OUT_M beyond
  their range) and hold; melee units (not our lord) attack the nearest enemy.
"""
import torch

from tools.nn.sim import orders as O
from tools.nn.sim.replay import half_depth
from tools.nn.train import drills as D
from tools.nn.train import opponents

EMPIRE, SKAVEN = "wh_main_emp_empire", "wh2_main_skv_skaven"
LORD = {EMPIRE: "wh_main_emp_cha_general_0", SKAVEN: "wh2_main_skv_cha_warlord_0"}
OURS_MELEE = {EMPIRE: ("wh_main_emp_inf_spearmen_0", "wh_main_emp_inf_spearmen_1", "wh_main_emp_inf_swordsmen"),
              SKAVEN: ("wh2_main_skv_inf_clanrat_spearmen_0", "wh2_main_skv_inf_clanrats_1")}
# our shooters (their faction is ours) and how many free enemies they face, inclusive: archers (friendly
# fire 0.26) only lose by shooting into the melee with two free enemies to shoot instead; the slave
# slingers win a duel with one free enemy only (128 battles each at a 900 s limit, build/drills/hf_diag.py:
# naive / skilled - archers 0.45 / 0.84 with one, 0.00 / 0.73 with two; Night Runners 0.22 / 1.00, 0.00 /
# 0.88; slave slingers 0.00 / 0.94, 0.00 / 0.24)
SHOOT = ((EMPIRE, "wh2_dlc13_emp_inf_archers_0", (2, 2)),
         (SKAVEN, "wh2_main_skv_inf_night_runners_1", (1, 2)),
         (SKAVEN, "wh2_main_skv_inf_skavenslave_slingers_0", (1, 1)))
FREE = {EMPIRE: ("wh2_dlc13_emp_inf_archers_0", "wh_dlc04_emp_inf_free_company_militia_0"),
        SKAVEN: ("wh2_main_skv_inf_skavenslave_slingers_0", "wh2_main_skv_inf_night_runners_1")}
WIDTH = {"wh2_dlc13_emp_inf_archers_0": 40.0, "wh2_main_skv_inf_skavenslave_slingers_0": 35.0}
MEN = {"wh_main_emp_cha_general_0": 1, "wh2_main_skv_cha_warlord_0": 1, "wh_main_emp_inf_greatswords": 120,
       "wh_main_emp_inf_spearmen_0": 120, "wh_main_emp_inf_spearmen_1": 120, "wh_main_emp_inf_swordsmen": 120,
       "wh2_main_skv_inf_clanrat_spearmen_0": 160, "wh2_main_skv_inf_clanrats_1": 160}
K_SHOOT = (2, 3)         # our shooters, inclusive
DIST_M = (70.0, 95.0)    # our shooters' line behind our infantry's centre
LAT_M = (65.0, 95.0)     # a free enemy beyond the end of our shooters' line
LAT_GAP_M = (8.0, 18.0)  # between neighbours in a line
HOLD_OUT_M = 25.0        # skilled: with no free enemy, stand this far beyond range from the melee


def _w(key):
    return WIDTH.get(key, 30.0)


def frame(rng):
    fo, s_key, f_free = SHOOT[int(rng.integers(len(SHOOT)))]
    fe = str(rng.choice([EMPIRE, SKAVEN]))
    pick = lambda pool: str(pool[int(rng.integers(len(pool)))])
    # the melee: ours facing +z at the origin, the engaged enemy in contact in front
    m_key = pick(OURS_MELEE[fo])
    lord = True                         # the engaged enemy is his lord (into an armoured unit fire still pays)
    e_key = LORD[fe]
    hm = float(half_depth(MEN[m_key], 30.0))
    he = 1.0 if lord else float(half_depth(MEN[e_key], 30.0))
    ours = [D.unit(m_key, 0.0, 0.0, 0.0, 30.0)]
    enemy = [D.unit(e_key, 0.0, hm + he + 0.5, 180.0, None if lord else 30.0, general=lord)]
    # our shooters: a line behind
    k = int(rng.integers(K_SHOOT[0], K_SHOOT[1] + 1))
    dist = float(rng.uniform(*DIST_M))
    gap = float(rng.uniform(*LAT_GAP_M))
    step = _w(s_key) + gap
    for i in range(k):
        ours.append(D.unit(s_key, (i - (k - 1) / 2) * step, -dist, 0.0, _w(s_key)))
    half = (k - 1) / 2 * step + _w(s_key) / 2
    # free enemies: off one end (or both ends) of our line, facing it
    f = int(rng.integers(f_free[0], f_free[1] + 1))
    sides = [1.0, -1.0] if f == 2 and rng.random() < 0.5 else [float(rng.choice([1.0, -1.0]))] * f
    placed = {1.0: 0, -1.0: 0}
    for s in sides:
        key = pick(FREE[fe])
        lat = half + float(rng.uniform(*LAT_M)) + placed[s] * (_w(key) + gap)
        placed[s] += 1
        z = -dist + float(rng.uniform(-20.0, 20.0))
        enemy.append(D.unit(key, s * lat, z, 270.0 if s > 0 else 90.0, _w(key)))
    desc = D.army(ours, enemy, attacker=1, ours_faction=fo, enemy_faction=fe,
                  engaged="lord" if lord else "unit")
    desc["sides"][2]["ai"] = True       # the enemy's lord uses his abilities on either side (mirror)
    desc["sides"][1]["ai"] = False
    return desc


# --- the broad frame ---
MILITIA = "wh_dlc04_emp_inf_free_company_militia_0"
OTHER = {"wh2_dlc13_emp_inf_archers_0": MILITIA, "wh2_main_skv_inf_night_runners_1": "wh2_main_skv_inf_skavenslave_slingers_0",
         "wh2_main_skv_inf_skavenslave_slingers_0": "wh2_main_skv_inf_night_runners_1"}
MIX_P, OUR_LORD_P, SECOND_P, FAR_P = 0.5, 0.5, 0.5, 0.3
OUR_LORD_M = (30.0, 60.0)
SECOND_M = (150.0, 250.0)
FAR_M = (150.0, 250.0)
LINE = {EMPIRE: ("wh_main_emp_inf_spearmen_0", "wh_main_emp_inf_spearmen_1", "wh_main_emp_inf_swordsmen"),
        SKAVEN: ("wh2_main_skv_inf_clanrat_spearmen_0", "wh2_main_skv_inf_clanrats_1", "wh2_main_skv_inf_skavenslaves_0")}
MEN.update({"wh2_main_skv_inf_skavenslaves_0": 180})


def broad(rng):
    desc = frame(rng)
    fo, fe = desc["sides"][1]["faction"], desc["sides"][2]["faction"]
    ours, enemy = desc["sides"][1]["units"], desc["sides"][2]["units"]
    pick = lambda pool: str(pool[int(rng.integers(len(pool)))])
    shooters = [u for u in ours[1:]]
    if rng.random() < MIX_P:
        u = shooters[int(rng.integers(len(shooters)))]
        u["key"] = OTHER[u["key"]]
        u["width"] = _w(u["key"])
    back = min(u["z"] for u in shooters)
    if rng.random() < OUR_LORD_P:
        ours.append(D.unit(LORD[fo], float(rng.uniform(-30.0, 30.0)), back - float(rng.uniform(*OUR_LORD_M)), 0.0,
                           general=True))
    if rng.random() < SECOND_P:
        m_key, e_key = pick(LINE[fo]), pick(LINE[fe])
        x = float(rng.choice([1.0, -1.0])) * float(rng.uniform(*SECOND_M))
        hm, he = float(half_depth(MEN[m_key], 30.0)), float(half_depth(MEN[e_key], 30.0))
        ours.append(D.unit(m_key, x, 0.0, 0.0, 30.0))
        enemy.append(D.unit(e_key, x, hm + he + 0.5, 180.0, 30.0))
    if rng.random() < FAR_P:
        enemy.append(D.unit(pick(LINE[fe]), float(rng.uniform(-60.0, 60.0)), float(rng.uniform(*FAR_M)), 180.0, 30.0))
    return desc


# --- the embedded frame ---
EMB_GAP_M = (20.0, 50.0)     # from the end of our line to our infantry unit's edge
EMB_AHEAD_M = (10.0, 30.0)   # our infantry unit this far ahead of our front line


def embedded(rng):
    best = None
    for _ in range(D.EMBED_TRIES):
        fo, s_key, f_free = SHOOT[int(rng.integers(len(SHOOT)))]
        fe = str(rng.choice([EMPIRE, SKAVEN]))
        pick = lambda pool: str(pool[int(rng.integers(len(pool)))])
        k = int(rng.integers(K_SHOOT[0], K_SHOOT[1] + 1))
        m_key = pick(OURS_MELEE[fo])
        f = int(rng.integers(f_free[0], f_free[1] + 1))
        free = [pick(FREE[fe]) for _ in range(f)]
        desc = D.generated(rng, fo, fe, swap={1: [m_key] + [s_key] * k, 2: free})
        ax = D.axes(desc, 1)
        ours, enemy = desc["sides"][1]["units"], desc["sides"][2]["units"]
        b_ours, b_enemy = ours[0]["b"], enemy[0]["b"]
        s = float(rng.choice([-1.0, 1.0]))
        # our infantry beyond our line's end, ahead of our front; the enemy's lord in contact in front of it
        lat0 = s * (D.extent(ours, ax, s) + float(rng.uniform(*EMB_GAP_M)) + 15.0)
        f0 = D.front(ours, ax) + float(rng.uniform(*EMB_AHEAD_M))
        hm = float(half_depth(MEN[m_key], 30.0))
        lord = next(u for u in enemy if u.get("general"))
        lord["x"], lord["z"] = ax.world(f0 + hm + 1.0 + 0.5, lat0)
        lord["b"], lord["tag"] = b_enemy, D.TAG_ENEMY
        ours.append(D.unit(m_key, *ax.world(f0, lat0), b_ours, 30.0, tag=D.TAG_OURS))
        # our shooters: a line behind it
        dist = float(rng.uniform(*DIST_M))
        gap = float(rng.uniform(*LAT_GAP_M))
        step = _w(s_key) + gap
        for i in range(k):
            ours.append(D.unit(s_key, *ax.world(f0 - dist, lat0 + (i - (k - 1) / 2) * step), b_ours, _w(s_key),
                               tag=D.TAG_OURS))
        half = (k - 1) / 2 * step + _w(s_key) / 2
        # the free enemies: beyond the outer end of our shooters' line, facing it
        for i, key in enumerate(free):
            lat = lat0 + s * (half + float(rng.uniform(*LAT_M)) + i * (_w(key) + gap))
            fz = f0 - dist + float(rng.uniform(-20.0, 20.0))
            enemy.append(D.unit(key, *ax.world(fz, lat), (b_ours + (-90.0 if s > 0 else 90.0)) % 360.0, _w(key),
                                tag=D.TAG_ENEMY))
        desc["meta"] = {"k": k, "free": f, "shoot": s_key}
        if D.balance(desc):
            return desc
        if best is None or D.imbalance(desc) < D.imbalance(best):
            best = desc
    return best


def _embed(st, clean, tag, inner):
    """Orders: `clean` in the clean / broad battles; in the embedded ones `ai_like`, with `inner`'s orders for
    the units tagged `tag`."""
    emb = D.embedded_rows(st)[:, None]
    mixed = O.merge(opponents.ai_like(st), inner, D.tagged(st, tag))
    return O.merge(clean, mixed, emb.expand_as(st.u["tag"]))


def _engaged(st):
    """[B, N] standing units in melee."""
    v = D.View(st)
    return v, v.standing & st.u["m"]


def enemy(st):
    """Melee units attack the nearest, missile units hold; in an embedded battle `ai_like`, the tagged units
    (his lord, the free missile units) so."""
    v = D.View(st)
    i, d = v.nearest()
    o = O.hold(st.B, st.N, st.device)
    o = D.attack(o, v.melee & (d < 1e9), i)
    return _embed(st, o, D.TAG_ENEMY, o)


def _melee_orders(st, v):
    """Every standing unit attacks the nearest enemy, our lord holds (he is not in the clean frame)."""
    o = D.nearest(st)
    o.kind = torch.where(st.u["lord"], torch.full_like(o.kind, O.HOLD), o.kind)
    return o


def naive(st):
    """focus(); in an embedded battle `ai_like`, the tagged units of ours so."""
    o = focus(st)
    return _embed(st, o, D.TAG_OURS, o)


def skilled(st):
    """spare(); in an embedded battle `ai_like`, the tagged units of ours so."""
    o = spare(st)
    return _embed(st, o, D.TAG_OURS, o)


def focus(st):
    """The mistake, tag-blind: shooters on the nearest enemy single entity in melee."""
    v, eng = _engaged(st)
    o = _melee_orders(st, v)
    shooter = v.standing & v.missile
    single = st.u["men0"] <= 1
    i1, d1 = v.nearest(eng & single)
    i2, d2 = v.nearest(eng)
    i = torch.where(d1 < 1e9, i1, i2)
    return D.attack(o, shooter & (d2 < 1e9), i)


def spare(st):
    """The skill, tag-blind (the clean frame's skilled script; the teacher's labels): shooters on the free
    enemies, else on an ordinary unit in melee, else out of range of the melee."""
    v, eng = _engaged(st)
    u = st.u
    o = _melee_orders(st, v)
    shooter = v.standing & v.missile
    i1, d1 = v.nearest(v.standing & ~eng)
    i2, d2 = v.nearest(eng & (u["men0"] > 1))                  # an ordinary unit in melee: fire pays
    i = torch.where(d1 < 1e9, i1, i2)
    d = torch.minimum(d1, d2)
    D.attack(o, shooter & (d < 1e9), i)
    # no free enemy: out of range of the melee, then hold
    j, dj = v.nearest()
    none = shooter & (d >= 1e9)
    keep = u["range"] + HOLD_OUT_M
    near = none & (dj < 1e9) & (dj < keep)
    ex, ez = u["x"].gather(1, j), u["z"].gather(1, j)
    ax, az = u["x"] - ex, u["z"] - ez
    n = torch.sqrt(ax * ax + az * az).clamp(min=1e-3)
    D.move(o, near, ex + ax / n * (keep + 15.0), ez + az / n * (keep + 15.0), run=True)
    o.kind = torch.where(none & ~near, torch.full_like(o.kind, O.HOLD), o.kind)
    return o


# --- transfer: the situation in an ordinary battle (drills/transfer.py) ---
T_CONTACT_M = 30.0       # one of ours in melee this near (centres) to an enemy single entity in melee: engaged with it
T_RANGE_EXTRA_M = 10.0   # the single entity within the shooter's range + this


def transfer(st):
    """(situation, applied, mistake) [B, N]: situation = a standing missile unit (with ammunition, not a
    lord) with an enemy single entity (a lord or hero: one man at the start) in melee with one of its own
    side's units (in melee, within T_CONTACT_M) within its range; mistake = it fires at that single entity
    (its current target, firing); applied = it does not (shoots another enemy, holds, moves)."""
    v = D.View(st)
    u = st.u
    me = v.standing & v.missile & ~u["lord"]
    single = v.standing & (u["men0"] <= 1) & u["m"]                                         # [B, j]
    fighting = v.standing & u["m"]                                                          # [B, k]
    # j engaged by a unit k of i's side: k in melee within T_CONTACT_M of j
    eng = v.foe & single[:, None, :]                                                        # [B, i, j]: j an enemy of i
    near_k = (v.d < T_CONTACT_M) & fighting[:, :, None]                                    # [B, k, j]
    mates = v.friend.float() @ near_k.float()                                               # [B, i, j]: i's side's k near j
    eng = eng & (mates > 0)
    in_range = v.d <= (u["range"] + T_RANGE_EXTRA_M)[:, :, None]
    lords = eng & in_range
    sit = me & lords.any(2)
    t = u["target"].clamp(min=0)[:, :, None]
    mistake = sit & u["fire"] & (u["target"] >= 0) & lords.gather(2, t).squeeze(2)
    return sit, sit & ~mistake, mistake


def moments(st, o):
    """[B, N] where the teacher's orders label a unit outside the clean frames: the situation (transfer)."""
    return transfer(st)[0]


DRILL = D.Drill("hold_fire", frame, enemy, naive, skilled,
                "our infantry in melee with the enemy lord: shoot the free enemies, not into the melee",
                broad=broad, transfer=transfer, embedded=embedded, teacher=spare, moments=moments)


def measure(seconds=200.0, shooters=2):
    """Friendly fire into a melee in the simulator: our infantry unit attacks an enemy in contact, two
    units of our shooters 85 m behind fire at will into it (or are absent) -> one line per pair: HP our
    shooters took from ours and from him, and how the melee ended (python -m tools.nn.train.drills.hold_fire)."""
    from tools.nn.sim import battle, missile, scenario
    from tools.nn.sim.params import load
    params = load()
    mel = ("wh_main_emp_inf_spearmen_0", "wh_main_emp_inf_swordsmen", "wh2_main_skv_inf_clanrat_spearmen_0",
           "wh2_main_skv_inf_clanrats_1")
    eng = (LORD[EMPIRE], LORD[SKAVEN], "wh_main_emp_inf_greatswords", "wh_main_emp_inf_swordsmen")
    sho = (None, "wh2_dlc13_emp_inf_archers_0", "wh2_main_skv_inf_skavenslave_slingers_0")
    fac = lambda k: EMPIRE if "_emp_" in k else SKAVEN
    rows, descs = [], []
    for m in mel:
        for e in eng:
            for s in sho:
                lord = e in LORD.values()
                hm, he = float(half_depth(MEN[m], 30.0)), 1.0 if lord else float(half_depth(MEN.get(e, 120), 30.0))
                ours = [D.unit(m, 0.0, 0.0, 0.0, 30.0)]
                ours += [D.unit(s, (i - (shooters - 1) / 2) * 50.0, -85.0, 0.0, _w(s)) for i in range(shooters if s else 0)]
                en = [D.unit(e, 0.0, hm + he + 0.5, 180.0, None if lord else 30.0, general=lord)]
                descs.append(D.army(ours, en, 1, fac(m), fac(e)))
                rows.append((m, e, s))
    st = scenario.build(descs)
    B, N = st.B, st.N
    H = N // 2
    acc = torch.zeros(B, N, N)
    volley = missile.volley

    def rec(*a, **k):
        out = volley(*a, **k)
        acc.add_(out[1])
        return out
    missile.volley = rec
    try:
        for _ in range(int(seconds / params.dt)):
            o = O.hold(B, N)
            o.kind[:, 0], o.target[:, 0] = O.ATTACK, H
            o.kind[:, H], o.target[:, H] = O.ATTACK, 0
            battle.step(st, o, params, params.dt)
    finally:
        missile.volley = volley
    u = st.u
    for b, (m, e, s) in enumerate(rows):
        ff, on = float(acc[b, 1:H, 0].sum()), float(acc[b, 1:H, H].sum())
        print(f"{m.split('_inf_')[-1]:22s} v {e.split('_', 2)[-1]:18s} {(s or '-').split('_inf_')[-1]:22s} "
              f"ours hp {float(u['hp'][b, 0]):.2f} rout {bool(u['r'][b, 0] | (u['men'][b, 0] <= 0))!s:5s} | "
              f"his hp {float(u['hp'][b, H]):.2f} rout {bool(u['r'][b, H] | (u['men'][b, H] <= 0))!s:5s} | "
              f"fire: ours {ff:5.0f} HP, his {on:5.0f} HP", flush=True)


if __name__ == "__main__":
    measure()
