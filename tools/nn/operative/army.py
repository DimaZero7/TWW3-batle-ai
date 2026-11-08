"""The operative's formula for a whole army (build/diag_m1/night.md; build/diag_label/human_rules.md - the user's three
won battles on the Kislev plains): groups by role and a target order, not the clumps' rule. Plain numpy over a battle's
rows; teacher.formula takes it for a battle with army_min own units or more (config teacher.army).

Roles (the row's passport: caste, missile range, bonus against large):
    lord        caste lord
    art         caste warmachine (or a missile range of art_range_m and more)
    sniper      a missile range of sniper_range_m and more (jezzails, long rifles)
    shooter     the other missile units (and missile cavalry)
    fast        cavalry, chariots, monsters, beasts (and missile-less units of run run_fast_ms and more)
    anti        melee infantry with a bonus against large (halberds, spears): pairs, the reserve against the fast
    line        the other melee infantry: pairs, the line
Groups (human_rules 1, 5: pairs and threes of one role, the shooters' bonds): line and anti in pairs from left to right
(across the way to the enemy), shooters in threes, the snipers, the artillery and the lord each one group, the fast in
pairs. The groups in force are kept while they still cover the units (no re-forming by places).

Targets (human_rules 2: the order of the first attacks), a seed enemy row per group:
    art      the enemy artillery, then snipers, shooters, the nearest
    sniper   the enemy snipers, then shooters, a threat, the nearest
    shooter  a threat (an enemy fast unit within threat_m of my shooters, snipers, artillery), then the enemy shooters
             within range + 50 m, the nearest
    anti     a threat (the one nearest my shooters), then an enemy shooter within chase_m of my snipers or shooters
             (skirmishers, human_rules 2.4), else none
    fast     a threat, then the enemy shooters or artillery within reach_m, else none
    lord     the enemy lord when the front shakes (a share of the enemy's units routing of shake or more, or their lord
             within lord_m of mine), else none
    line     the nearest enemy
and the enemy lord for the fast too when the front shakes (human_rules 2.5).
Zones: a group that holds (the line, the shooters, the snipers, the artillery, an idle reserve, the lord) keeps the zone
it got at its forming (its centre then); a group on a threat, a skirmisher or the lord goes for it (a point approach_m
short of the seed). Under the command 'destroy' the line goes for its target too.

plan(...) -> per battle [(rows, seed row | None, goal (x, z), holds)].
"""
import math

import numpy as np

from tools.nn.model import passport as PP

CASTE0 = 36                          # passport.features: the caste's one-hot from this column (CASTES + 'other')
RANGE_COL, RUN_COL, VLARGE_COL = 29, 5, 12


def roles(cards):
    """[K] role names of the rows' passports cards [K, 69] (numpy)."""
    caste = cards[:, CASTE0:CASTE0 + len(PP.CASTES) + 1].argmax(1)
    rng = cards[:, RANGE_COL] * 500.0
    run = cards[:, RUN_COL] * 10.0
    vl = cards[:, VLARGE_COL] * 50.0
    out = []
    for c, r, v, sp in zip(caste, rng, vl, run):
        name = PP.CASTES[c] if c < len(PP.CASTES) else "other"
        if name == "lord":
            out.append("lord")
        elif name == "warmachine" or r >= 400:
            out.append("art")
        elif r >= 250:
            out.append("sniper")
        elif r > 0:
            out.append("shooter")
        elif name in ("melee_cavalry", "monstrous_cavalry", "chariot", "monster", "war_beast", "monstrous_infantry"):
            out.append("fast")
        elif name in ("melee_infantry", "hero", "other") and v > 0:
            out.append("anti")
        else:
            out.append("line" if sp < 7.0 else "fast")
    return out


def _nearest(xy, cands, pos):
    if not cands:
        return None
    return min(cands, key=lambda e: math.hypot(*(pos[e] - xy)))


def plan(b, own_rows, enemy_rows, pos, cards, routing, have, destroy, tc, ranges):
    """[(rows, seed, goal, holds)] of battle b (module doc). own_rows / enemy_rows: row indices; pos [K, 2]; cards
    [K, 69]; routing [K] bool; have {g: set(rows)} the groups in force; destroy: the command; tc: config teacher.army;
    ranges [K] the missile range, m."""
    if not own_rows:
        return []
    rl = roles(cards)
    me = np.mean([pos[j] for j in own_rows], 0)
    foes = [e for e in enemy_rows if not routing[e]]
    fc = np.mean([pos[e] for e in foes], 0) if foes else me + np.array([1.0, 0.0])
    f = fc - me
    f = f / max(float(np.hypot(*f)), 1e-6)
    right = np.array([f[1], -f[0]])
    side = lambda j: float((pos[j] - me) @ right)                                       # noqa: E731
    # the groups: those in force while they cover my alive units by one role each, else formed anew
    alive = set(own_rows)
    now = [sorted(m & alive) for m in have.values() if m & alive]
    size = {"line": 2, "anti": 2, "shooter": 3, "fast": 2, "sniper": 10, "art": 10, "lord": 1}
    if now and set().union(*map(set, now)) == alive and all(len({rl[j] for j in g}) == 1 for g in now):
        groups = now
    else:
        # only what is wrong is formed anew (08.11, build/diag_m1/night.md 20): a group in force of one role and not
        # over its size stays; the rest of my units (mixed groups, no group) go by role into groups of its size
        groups = [g for g in now if len({rl[j] for j in g}) == 1 and len(g) <= size[rl[g[0]]]]
        kept = set().union(*map(set, groups)) if groups else set()
        by = {}
        for j in own_rows:
            if j not in kept:
                by.setdefault(rl[j], []).append(j)
        for r in ("line", "anti", "shooter", "sniper", "art", "fast", "lord"):
            rows = sorted(by.get(r, []), key=side)
            k = size[r]
            groups += [rows[i:i + k] for i in range(0, len(rows), k)]
    # the enemy by role
    er = {}
    for e in foes:
        er.setdefault(rl[e], []).append(e)
    mine_shooting = [j for j in own_rows if rl[j] in ("shooter", "sniper", "art")]
    threat = [e for e in er.get("fast", [])
              if mine_shooting and min(math.hypot(*(pos[e] - pos[j])) for j in mine_shooting) <= tc["threat_m"]]
    near_mine = lambda e: min(math.hypot(*(pos[e] - pos[j])) for j in (mine_shooting or own_rows))  # noqa: E731
    threat.sort(key=near_mine)
    lords = [e for e in enemy_rows if rl[e] == "lord" and not routing[e]]
    n_enemy = max(len(enemy_rows), 1)
    my_lord = [j for j in own_rows if rl[j] == "lord"]
    shake = (sum(bool(routing[e]) for e in enemy_rows) / n_enemy >= tc["shake"]) or bool(
        lords and my_lord and math.hypot(*(pos[lords[0]] - pos[my_lord[0]])) <= tc["lord_m"])
    out = []
    for g in groups:
        r = rl[g[0]]
        c = np.mean([pos[j] for j in g], 0)
        rng = max(float(ranges[j]) for j in g)
        seed, holds = None, True
        # (within its range and a step: art and snipers first - the nearest of them may stand across the field)
        reach0 = min(float(ranges[j]) for j in g) + float(tc.get("range_step_m", 0.0))
        within = lambda rr: [e for e in er.get(rr, []) if math.hypot(*(pos[e] - c)) <= reach0]   # noqa: E731
        if r == "art":
            seed = (_nearest(c, within("art"), pos) or _nearest(c, within("sniper"), pos)
                    or _nearest(c, er.get("art", []), pos) or _nearest(c, er.get("sniper", []), pos)
                    or _nearest(c, er.get("shooter", []), pos) or _nearest(c, foes, pos))
        elif r == "sniper":
            seed = (_nearest(c, within("sniper"), pos) or _nearest(c, within("art"), pos)
                    or _nearest(c, er.get("sniper", []), pos) or _nearest(c, er.get("shooter", []), pos)
                    or (threat[0] if threat else None) or _nearest(c, foes, pos))
        elif r == "shooter":
            inr = [e for e in er.get("shooter", []) + er.get("sniper", []) if math.hypot(*(pos[e] - c)) <= rng + 50]
            seed = (threat[0] if threat else None) or _nearest(c, inr, pos) or _nearest(c, foes, pos)
        elif r == "anti":
            if threat:
                seed, holds = _nearest(c, threat, pos), False
            else:
                sk = [e for e in er.get("shooter", []) if mine_shooting and
                      min(math.hypot(*(pos[e] - pos[j])) for j in mine_shooting) <= tc["chase_m"]]
                if sk:
                    seed, holds = _nearest(c, sk, pos), False
        elif r == "fast":
            if shake and lords:
                seed, holds = lords[0], False
            elif threat:
                seed, holds = _nearest(c, threat, pos), False
            else:
                soft = [e for e in er.get("shooter", []) + er.get("art", []) + er.get("sniper", [])
                        if math.hypot(*(pos[e] - c)) <= tc["reach_m"]]
                if soft:
                    seed, holds = _nearest(c, soft, pos), False
        elif r == "lord":
            # the general stays behind the line (the bridge agent's base: our lord in melee at 210 / 234 s and lost; the
            # user's lord in M2 stood back to 275 s): his target the nearest enemy, his zone the one he got
            seed = _nearest(c, foes, pos)
        else:                                                    # line
            seed = _nearest(c, foes, pos)
            holds = not destroy
        # snipers and guns step up to their range (the user's M2: snipers on the Jezzails from 75 s; build/diag_m1/night.md
        # 18): a target beyond the range of all the group, by at most range_step_m -> a point range_margin_m inside it
        reach = min(float(ranges[j]) for j in g)
        if r in ("art", "sniper") and seed is not None and holds and not threat and reach > 0:
            d = pos[seed] - c
            n = max(float(np.hypot(*d)), 1e-6)
            if reach < n <= reach + float(tc.get("range_step_m", 0.0)):
                out.append((g, seed, pos[seed] - d / n * (reach - float(tc.get("range_margin_m", 10.0))), False))
                continue
        if seed is not None and not holds:
            d = pos[seed] - c
            n = max(float(np.hypot(*d)), 1e-6)
            goal = pos[seed] - d / n * min(float(tc["approach_m"]), n)
        else:
            goal = None                                          # (holds: its zone in force, else its centre)
        out.append((g, seed, goal, holds))
    return out
