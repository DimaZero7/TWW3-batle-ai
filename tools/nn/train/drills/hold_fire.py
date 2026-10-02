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
Scripts:
* enemy: melee units attack the nearest enemy (the engaged one keeps fighting), missile units hold
  (shoot at will at the nearest);
* naive: our shooters attack the enemy fighting our infantry (focus fire on the lord), melee units the
  nearest enemy;
* skilled: our shooters attack the nearest free (not in melee) enemy; with none left they step out of
  range of the melee (HOLD_OUT_M beyond their range) and hold; melee units attack the nearest enemy.
"""
import torch

from tools.nn.sim import orders as O
from tools.nn.sim.replay import half_depth
from tools.nn.train import drills as D

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


def _engaged(st):
    """[B, N] standing units in melee."""
    v = D.View(st)
    return v, v.standing & st.u["m"]


def enemy(st):
    v = D.View(st)
    i, d = v.nearest()
    o = O.hold(st.B, st.N, st.device)
    return D.attack(o, v.melee & (d < 1e9), i)


def naive(st):
    v, eng = _engaged(st)
    o = D.nearest(st)
    shooter = v.standing & v.missile
    i, d = v.nearest(eng)
    return D.attack(o, shooter & (d < 1e9), i)


def skilled(st):
    v, eng = _engaged(st)
    u = st.u
    o = D.nearest(st)
    shooter = v.standing & v.missile
    i, d = v.nearest(v.standing & ~eng)
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


DRILL = D.Drill("hold_fire", frame, enemy, naive, skilled,
                "our infantry in melee with the enemy lord: shoot the free enemies, not into the melee")


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
