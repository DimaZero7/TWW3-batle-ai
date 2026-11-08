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


DIRECT = 66                     # the passport's column 'direct' (flat, trajectory low: friends block its line)


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
    wavers = sum(bool(routing[e]) for e in enemy_rows) / n_enemy >= tc["shake"]
    shake = wavers or bool(
        lords and my_lord and math.hypot(*(pos[lords[0]] - pos[my_lord[0]])) <= tc["lord_m"])
    # (08.11, the bridge's game battles: the snipers' zone stepped 60-110 m ahead onto the enemy lord at 0 s, in melee
    # from 53-56 s; the user's rules: snipers 33-44 m from their infantry, on the enemy's shooters and snipers)
    no_lord = [e for e in foes if rl[e] != "lord"] or foes                      # (the lord only when the front wavers)
    mel = [j for j in own_rows if rl[j] in ("line", "anti") and not routing[j]]
    front = max(float(pos[j] @ f) for j in mel) if mel else None              # my infantry's front edge, along f
    back = min(float(pos[j] @ f) for j in mel) if mel else None
    enemy_melee = [e for e in foes if rl[e] not in ("shooter", "sniper", "art")]
    # the halberd pairs on the threats and skirmishers: at most per_target pairs a target, the nearest pairs first, and
    # only targets within chase_ahead_m of my infantry's front (08.11, the whole-battle exam: every pair went for every
    # fast enemy and slinger, the pairs routed first in 27 of 32 M1 battles and kept 0.17 of their men; the user:
    # "по одной паре на цель")
    anti_t = {}
    if tc.get("per_target"):
        sk = [e for e in er.get("shooter", []) if mine_shooting and
              min(math.hypot(*(pos[e] - pos[j])) for j in mine_shooting) <= tc["chase_m"]]
        lim = None if front is None else front + float(tc.get("chase_ahead_m", 1e9))
        tg = [e for e in (threat or sk) if lim is None or float(pos[e] @ f) <= lim]
        pairs = [k for k, g in enumerate(groups) if rl[g[0]] == "anti"]
        cen = {k: np.mean([pos[j] for j in groups[k]], 0) for k in pairs}
        cand = sorted((float(np.hypot(*(pos[e] - cen[k]))), k, e) for k in pairs for e in tg)
        used = {}
        for dd, k, e in cand:
            if k in anti_t or used.get(e, 0) >= int(tc["per_target"]):
                continue
            anti_t[k] = e
            used[e] = used.get(e, 0) + 1
    out = []
    for gk, g in enumerate(groups):
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
            # the user's order: a threat to my shooters, the enemy's snipers and shooters, artillery, the rest (no lord);
            # a threat only within sniper_threat_m of the snipers (08.11, the whole-battle exam M1: ~500 shots a unit
            # for 44-72 HP, spent on the running wolf rats; the enemy's Jezzails kept all their men)
            near_t = [e for e in threat if math.hypot(*(pos[e] - c)) <= float(tc.get("sniper_threat_m", 1e9))]
            seed = ((_nearest(c, near_t, pos) if near_t else None)
                    or _nearest(c, within("sniper") + within("shooter"), pos) or _nearest(c, within("art"), pos)
                    or _nearest(c, er.get("sniper", []) + er.get("shooter", []), pos)
                    or _nearest(c, er.get("art", []), pos) or _nearest(c, foes if wavers else no_lord, pos))
        elif r == "shooter":
            inr = [e for e in er.get("shooter", []) + er.get("sniper", []) if math.hypot(*(pos[e] - c)) <= rng + 50]
            seed = ((threat[0] if threat else None) or _nearest(c, inr, pos)
                    or _nearest(c, foes if wavers else no_lord, pos))
        elif r == "anti" and tc.get("per_target"):
            if gk in anti_t:
                seed, holds = anti_t[gk], False
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
            # user's lord in M2 stood back to 275 s): no target, his zone the one he got - until the enemy's front wavers
            # (wavers: shake's share of routing units): then the nearest enemy, to finish (the user: the lord is let go
            # when the enemy runs; 08.11, the bridge's teacher battles: the lord in melee in M2, M3 - his target was the
            # nearest enemy)
            if tc.get("lord_target", "wavers") == "nearest":
                seed = _nearest(c, foes, pos)                    # (the rule before 08.11, for the checks)
            elif wavers:
                seed, holds = _nearest(c, foes, pos), False
            else:
                # his place: lord_back_m behind my melee line's centre (the user: -19 / -15 m in M1 / M2); the zone
                # moves there once it is move_m off (teacher phase B), else it stays
                mel = [j for j in own_rows if rl[j] in ("line", "anti") and not routing[j]]
                if mel and tc.get("lord_back_m") is not None:
                    lc = np.mean([pos[j] for j in mel], 0)
                    out.append((g, None, lc - f * float(tc["lord_back_m"]), False))
                    continue
        else:                                                    # line
            seed = _nearest(c, foes, pos)
            holds = not destroy
        # snipers and guns step up to their range (the user's M2: snipers on the Jezzails from 75 s; build/diag_m1/night.md
        # 18): a target beyond the range of all the group, by at most range_step_m -> a point range_margin_m inside it
        reach = min(float(ranges[j]) for j in g)
        # shooters and snipers step back behind the infantry from an enemy melee unit within back_m (the user's rule:
        # 25-40 m behind his infantry when an enemy comes within ~70 m)
        if r in ("shooter", "sniper") and back is not None and tc.get("back_m"):
            near_e = [e for e in enemy_melee if math.hypot(*(pos[e] - c)) <= float(tc["back_m"])]
            behind_m = float(tc.get("behind_m", 30.0))
            if near_e and float(c @ f) > back - behind_m + 5.0:
                out.append((g, seed, c + f * ((back - behind_m) - float(c @ f)), False))
                continue
        # direct-fire snipers on the flank of the line (the user: 33-44 m beside the outermost infantry unit, an open
        # line of fire; 08.11, the bridge's games: behind the line their fire was blocked by our own and the attack
        # order walked them through it): a goal there when the group stands farther than move_m from it
        direct = all(float(cards[j][DIRECT]) > 0.5 for j in g)
        # (only while my infantry closes its line to every target worth it: first the user's order among the targets
        # with an open line from where the group stands - the walk to the flank cost the simulator's snipers 60-80 %
        # of their fire on the user's own places)
        def closed(e_):
            v = pos[e_] - c
            L = max(float(np.hypot(*v)), 1e-6)
            ev = v / L
            for j in mel:
                w = pos[j] - c
                a_, x_ = float(w @ ev), abs(float(w[0] * ev[1] - w[1] * ev[0]))
                if 10.0 < a_ < L - 10.0 and x_ < float(tc.get("fire_band_m", 30.0)):
                    return True
            return False
        blocked = False
        # (not with a threat at my shooters: then they stay and shoot what they can, or step back - the walk along
        # the flank under the wolf rats' charge took them 100-150 m in 240 s)
        if (r == "sniper" and direct and mel and seed is not None and not threat
                and math.hypot(*(pos[seed] - c)) <= reach and closed(seed)):
            tiers = [near_t, within("sniper") + within("shooter"), within("art"),
                     [e_ for e_ in (foes if wavers else no_lord) if math.hypot(*(pos[e_] - c)) <= reach]]
            alt = None
            for tier in tiers:
                alt = _nearest(c, [e_ for e_ in tier if not closed(e_)], pos)
                if alt is not None:
                    break
            if alt is not None:
                seed = alt
            else:
                blocked = True
        if r == "sniper" and direct and mel and tc.get("flank_m") and blocked:
            lats = [float(pos[j] @ right) for j in mel]
            mid = 0.5 * (max(lats) + min(lats))
            sgn = 1.0 if float(c @ right) >= mid else -1.0
            edge = max(lats) if sgn > 0 else min(lats)
            ej = mel[int(np.argmax(lats))] if sgn > 0 else mel[int(np.argmin(lats))]
            gl = right * (edge + sgn * float(tc["flank_m"])) + f * float(pos[ej] @ f)
            if float(np.hypot(*(gl - c))) > float(tc.get("flank_move_m", 40.0)):
                out.append((g, seed, gl, False))
                continue
        # guns: no step up (08.11, the bridge's games: the step to the infantry's front drove M2's mortars 270-360 m,
        # 8 and 1 shots, into melee; behind, 30 and 20) - their zone moves only by art_step (off by default)
        if r == "art" and not tc.get("art_step"):
            out.append((g, seed, None, True))
            continue
        if r in ("art", "sniper") and seed is not None and holds and not threat and reach > 0:
            d = pos[seed] - c
            n = max(float(np.hypot(*d)), 1e-6)
            if reach < n <= reach + float(tc.get("range_step_m", 0.0)):
                gl = pos[seed] - d / n * (reach - float(tc.get("range_margin_m", 10.0)))
                # (not past my infantry's front edge by more than ahead_m: the step stops there)
                lim = None if front is None else front + float(tc.get("ahead_m", 10.0))
                # the snipers' duel (08.11, the whole-battle exam M1: the Jezzails, range 275 against our 260, kept every
                # man): on an enemy sniper or shooter that outranges them, with no enemy melee unit within duel_safe_m
                # of the snipers or of the point, they go just far enough to have it in range (range - duel_in_m), at
                # most duel_ahead_m past my infantry's front; an enemy melee unit within back_m sends them back (above)
                if (r == "sniper" and tc.get("duel_ahead_m") and rl[seed] in ("sniper", "shooter")
                        and float(ranges[seed]) > reach and front is not None):
                    gd = pos[seed] - d / n * (reach - float(tc.get("duel_in_m", 5.0)))
                    safe = float(tc.get("duel_safe_m", 75.0))
                    if all(min(math.hypot(*(pos[e] - c)), math.hypot(*(pos[e] - gd))) > safe for e in enemy_melee):
                        gl, lim = gd, front + float(tc["duel_ahead_m"])
                if lim is not None and float(gl @ f) > lim:
                    gl = gl - f * (float(gl @ f) - lim)
                if lim is None or float(gl @ f) > float(c @ f) + 1.0:
                    out.append((g, seed, gl, False))
                    continue
        if seed is not None and not holds:
            d = pos[seed] - c
            n = max(float(np.hypot(*d)), 1e-6)
            goal = pos[seed] - d / n * min(float(tc["approach_m"]), n)
        else:
            goal = None                                          # (holds: its zone in force, else its centre)
        out.append((g, seed, goal, holds))
    return out
