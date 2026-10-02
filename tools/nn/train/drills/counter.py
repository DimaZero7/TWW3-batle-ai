"""Drill "counter": the enemy unit nearest to one of ours is its hard counter; sending each unit at
the enemy it beats (and our counter unit at the enemy's) wins the fight, attacking the nearest loses
it - under the standard battle limit, decided by the fight, not the clock.

The counters in the simulator (1 v 1, the attacker charging a holding unit, 16 battles each, the
training's randomised numbers, build/drills/m1v1.py): greatswords (armour 95, AP 25 of 35 damage) beat
every infantry unit of the pools; flagellants (unbreakable) beat every line unit and lose only to
greatswords; spearmen lose to every other Empire unit. The 2 v 2 pairings below come from running all
same-faction pairings where X loses to C and Z beats C (build/drills/m2v2.py, 16 battles each).

Frame (one battle; Empire against Empire):
* two pairs facing each other GAP_M apart, side by side (WIDTH_M fronts, LAT_GAP_M between them), in
  random order along the line: our X opposite the enemy's C (its counter), our Z opposite the enemy's
  Y; (X, Z, C, Y) from COMBOS: spearmen + flagellants v flagellants + shield spearmen / swordsmen;
  shield spearmen or swordsmen + greatswords v greatswords + flagellants; spearmen + greatswords v
  greatswords + swordsmen;
* the enemy (`enemy`): holds and fights back; a unit free of its opponent (it broke or died) and
  hurt attacks the nearest of ours - it presses where it has won; it is the defender;
* random: the pairing, the order of the pairs, the gap, the lateral spacing, the place and bearing
  on the map, our side (1 or 2).
Scripts:
* naive: every unit attacks the nearest enemy, running: X breaks on its counter, which then joins
  the other fight against our Z;
* skilled: a matchup from the passports (melee damage rates and health, unbreakable) says which
  enemies each unit beats (the attacker's charge at CHARGE_W); each unit attacks, among the
  enemies it beats, the one the fewest of its side beat, then the most dangerous to its side, then
  the nearest; a unit that beats none attacks the enemy it does best against; a unit in melee keeps
  its target. roles(): its choice is the "correct" target, an enemy that beats the unit a "bad" one
  (drills/metrics.py).
"""
import torch

from tools.nn.sim import orders as O
from tools.nn.train import drills as D

EMPIRE = "wh_main_emp_empire"
GS = "wh_main_emp_inf_greatswords"
FLAG = "wh_dlc04_emp_inf_flagellants_0"
SPEAR, SHIELD, SWORD = "wh_main_emp_inf_spearmen_0", "wh_main_emp_inf_spearmen_1", "wh_main_emp_inf_swordsmen"
# (our X, our Z, the enemy's C opposite X, the enemy's Y opposite Z): C beats X, Z beats C. Picked from the
# simulator's 2 v 2 runs (build/drills/m2v2.py, 16 battles each, the standard limit, the enemy as
# `enemy` below): attacking the nearest - crossing over (X at Y, Z at C) won
COMBOS = ((SPEAR, FLAG, FLAG, SHIELD),     # 0.00 - 0.75
          (SHIELD, GS, GS, FLAG),          # 0.06 - 0.75
          (SPEAR, GS, GS, SWORD),          # 0.38 - 1.00
          (SWORD, GS, GS, FLAG),           # 0.38 - 1.00
          (SPEAR, FLAG, FLAG, SWORD))      # 0.25 - 0.81
GAP_M = (110.0, 200.0)
WIDTH_M = 30.0
LAT_GAP_M = (20.0, 60.0)

# The skilled script's matchup (tools/nn/sim/melee.py strikes(), per man in contact, front on).
HIT_BASE, HIT_SLOPE = 35.0, 0.1
BREAK_SHARE = 0.45       # a breakable unit routs at about this share of its health lost (1 v 1 runs)
CHARGE_W = 0.5           # the attacker's charge bonus counts at this weight (it fades over the fight)
BEATS = 1.15             # i beats j: j's health goes this many times faster than i's


def frame(rng):
    x, z, c, y = COMBOS[int(rng.integers(len(COMBOS)))]
    pairs = [(x, c), (z, y)]                                         # (ours, the enemy opposite)
    order = rng.permutation(len(pairs))
    gap = float(rng.uniform(*GAP_M))
    lat_gap = float(rng.uniform(*LAT_GAP_M))
    n = len(pairs)
    ours, enemy = [], []
    for slot, k in enumerate(order):
        lat = (slot - (n - 1) / 2) * (WIDTH_M + lat_gap)
        o, e = pairs[int(k)]
        ours.append(D.unit(o, *D.local(-gap / 2, lat), 0.0, WIDTH_M))
        enemy.append(D.unit(e, *D.local(gap / 2, lat), 180.0, WIDTH_M))
    return D.army(ours, enemy, attacker=1, ours_faction=EMPIRE, enemy_faction=EMPIRE)


def enemy(st):
    """Hold (stand, fight back); a unit free of its opponent (it broke or died) after taking damage
    attacks the nearest of ours: the enemy presses where it has won."""
    v = D.View(st)
    i, d = v.nearest()
    o = D.hold(st)
    free = v.melee & ~st.u["m"] & (st.u["hp"] < 0.97)
    return D.attack(o, free & (d < 1e9), i, run=True)


def naive(st):
    return D.nearest(st)


def rates(u, charge_w=0.0):
    """[B, i, j]: share of j's breaking health i takes per second (per man in contact, front on;
    i's charge bonus at charge_w)."""
    bonus = torch.where(u["large"][:, None, :], u["bonus_v_large"][:, :, None], u["bonus_v_inf"][:, :, None])
    bonus = bonus + charge_w * u["charge_bonus"][:, :, None]
    p = torch.clamp(HIT_BASE + HIT_SLOPE * (u["attack"][:, :, None] + bonus - u["defence"][:, None, :]), 8.0, 90.0) / 100
    dmg, ap = u["damage"][:, :, None], u["ap_damage"][:, :, None]
    share = dmg / (dmg + ap).clamp(min=1e-6)
    base, pierce = dmg + bonus * share, ap + bonus * (1 - share)
    hit = pierce + base * torch.clamp(1 - 0.75 * u["armour"][:, None, :] / 100, min=0)
    hit = torch.minimum(hit * (1 - u["resist_physical"][:, None, :]), u["hp_man"][:, None, :])
    keep = torch.where(u["unbreakable"], torch.ones_like(u["men"]), torch.full_like(u["men"], BREAK_SHARE))
    hp = u["hp_man"] * u["men"].clamp(min=1) * keep
    return p * hit / u["interval"][:, :, None].clamp(min=1e-6) / hp[:, None, :].clamp(min=1e-6)


def _choice(st, v):
    """(the target the matchup picks for each unit [B, N], foe [B, N, N], adv [B, N, N]), before
    keeping a target in melee."""
    u = st.u
    adv = rates(u, CHARGE_W) / rates(u).transpose(1, 2).clamp(min=1e-9)     # [B, i, j]: i attacking j
    foe = v.foe & v.melee[:, :, None]
    beats = foe & (adv >= BEATS)
    # scarcity of j: how many of j's enemies (standing melee units) beat it
    beaten_by = (beats & v.standing[:, :, None]).sum(1)              # [B, j]
    # danger of j: how hard it beats the unit of i's side it beats most (adv[j, k], capped)
    mine = v.friend & v.melee[:, None, :] & v.standing[:, None, :]                  # [B, i, k]
    adv_t = adv.transpose(1, 2)                                                     # [B, k, j] = adv[j, k]
    danger = torch.where(mine[:, :, :, None], adv_t[:, None, :, :], torch.zeros_like(adv_t)[:, None]).amax(2)  # [B, i, j]
    big = torch.full_like(v.d, 1e9)
    # among the enemies it beats: the scarcest (fewest of its side beat it; 1000 m a unit), then the most
    # dangerous to its side (up to 500 m), then the nearest
    score = torch.where(beats, beaten_by[:, None, :].float() * 1000.0 - 10.0 * danger.clamp(max=50.0) + v.d, big)
    s_best, best = score.min(2)
    # none it beats: the one it does best against
    fallback = torch.where(foe, -adv, big).argmin(2)
    return torch.where(s_best < 1e9, best, fallback), foe, adv


def roles(st):
    """(correct, bad) [B, N, N]: the enemy the matchup sends unit i at; the enemies that beat i (its counters)."""
    v = D.View(st)
    tgt, foe, adv = _choice(st, v)
    slot = torch.arange(st.N, device=st.device)
    correct = foe & (slot[None, None, :] == tgt[:, :, None])
    bad = foe & (adv.transpose(1, 2) >= BEATS)
    return correct, bad


def skilled(st):
    v = D.View(st)
    u = st.u
    tgt, foe, adv = _choice(st, v)
    # a unit in melee keeps its target
    cur = u["order_target"].clamp(min=0)
    committed = u["m"] & (u["order_kind"] == O.ATTACK) & (u["order_target"] >= 0) & v.foe.gather(2, cur[:, :, None]).squeeze(2)
    tgt = torch.where(committed, cur, tgt)
    o = O.hold(st.B, st.N, st.device)
    has = v.standing & foe.any(2)
    D.attack(o, has, tgt, run=True)
    # missile units (not in this drill's frame): as `nearest`
    i, d = v.nearest()
    D.attack(o, v.standing & v.missile & (d < 1e9), i)
    return o


def diag_key(desc, ours):
    m, e = desc["sides"][ours]["units"], desc["sides"][3 - ours]["units"]
    sh = lambda u: u["key"].split("_inf_")[-1][:10]
    return tuple(sorted(sh(u) for u in m)) + ("v",) + tuple(sorted(sh(u) for u in e))


DRILL = D.Drill("counter", frame, enemy, naive, skilled,
                "the nearest enemy is the unit's hard counter: send each unit at the enemy it beats", roles=roles)
