"""The engine's O0-1 scenes made fair (Astra's o1 training review 2; the orchestrator 22.11 F5): scenes_o1's layouts
leave metres that tell a holder from a come twin besides the cause (the hold itself - motion and braced). A NEW
wrapper over scenes_o1.scene for the training rounds, V and the probe only (scenes_o1.py and sally.py stay: their
hashes key exam2's cached arms).

Measured on probe_held's 64 pairs at the first decision (tmp/engine/b/geom.log, the 22.11 scenes):
    the come twin's walkers lag their pose by 14 m (median; 0.7-31 m) - E's centroid, the stand-off line and both
        candidate distances differ by the same metres (cause - come: median -12.9 m, 10 % -18 m)
    the holders stand on exact 40 m ranks (to 1 cm) - 54 of 64; the walkers 10 of 64 (median irregularity 1.7 m)
    the holder's lord exactly lord_back_m (60 m) from its infantry's centroid (to 1 cm) in 57 of 64; no walker
fair(sp)   the same scene, changed alike in a family (every draw from the scene's seed_used - the cause scene and its
           twins get the same numbers):
    JITTER_M     every enemy unit's pose moved by a uniform +-JITTER_ALONG_M along our facing and +-JITTER_M across
                 (both twins: the ranks are no longer exact for a holder either), its goal and start with it
    LORD_M       the enemy lord moved by a uniform +-LORD_M along and across (both twins)
    WALK-IN      the come twin's units start walk x t_dec behind the pose (each unit its own walk speed): sally.WalkIn
                 walks while it is ahead of its run schedule, so every unit WALKS the whole way in formation and arrives
                 at its pose at the decision tick - the same formation and rank spacing a holder shows (Astra's second
                 o1 review 2: with a 0.8 run-in the walkers broke formation - their depth spacing off the 40 m ranks
                 told them from holders, 68.75 % balanced with the motion erased)
           a point the map's 'cannot stand' ground would take is kept unmoved
"""
import math
import random

JITTER_M = 6.0
JITTER_ALONG_M = 10.0
LORD_M = 6.0
LEAD = None              # (replaced by the walk-in: start = pose + walk x t_dec, see WALK-IN)
_WHY = ("ASSUMPTION JITTER_ALONG_M 10 m / JITTER_M 6 m (Astra's second o1 review 2, tmp/engine/b/geom_fair_try2.log): "
        "at +-3 m two holders' units differ by at most 6 m along the facing - her rule (every pair within 6.01 m of a "
        "multiple of 40 m) took 16 of 16 holders and 11 of 16 walk-in comers (0.656); the walk-in's own error along "
        "the facing reaches 10-17 m (rank_err q90 9.7, max 16.6 m), so the holder's depth spread is drawn as wide; "
        "small beside the sally gap 60-100 m and the stand-off 45 m. Before: JITTER_M 3 m: above the walkers' median rank irregularity (1.7 m) so an exact rank tells nothing, small "
        "beside the 40 m ranks; LORD_M 6 m: at least the walkers' 10 % lord deviation (0.75 m) and well past the 1 cm "
        "tell; the walk-in: walkers in formation at their pose at the decision (tmp/engine/b/geom_*.log)")


def _stand(mp, x, z, b):
    from net.tactic import scenes as TS
    return TS.can_stand(mp, x, z, b)


def fair(sp):
    o = sp["o1"]
    rnd = random.Random(int(sp.get("seed_used", sp["seed"])) * 7 + 3)
    mp = sp["map"]
    pose, goals = list(o["pose"]), list(o["goals"])
    pt = list(sp["places_theirs"])
    starts = list(o["starts"])
    hb = pt[0][2] if pt else 0.0
    fx, fz = math.sin(math.radians((hb + 180.0) % 360.0)), math.cos(math.radians((hb + 180.0) % 360.0))  # (our fwd)
    rx, rz = fz, -fx
    new_pose, new_goals, new_starts = [], [], []
    from net.tactic import scenes as TS
    t_dec = float(o["t_dec"])
    for k, ((x, z), (gx, gz), (sx, sz)) in enumerate(zip(pose, goals, starts)):
        da, dc = rnd.uniform(-JITTER_ALONG_M, JITTER_ALONG_M), rnd.uniform(-JITTER_M, JITTER_M)
        dx, dz = da * fx + dc * rx, da * fz + dc * rz
        if not (_stand(mp, x + dx, z + dz, hb) and _stand(mp, sx + dx, sz + dz, hb)):
            dx = dz = 0.0
        px, pz = x + dx, z + dz
        new_pose.append((px, pz))
        new_goals.append((gx + dx, gz + dz))
        bx, bz = sx - x, sz - z                                         # (start minus pose: back along our fwd)
        bn = math.hypot(bx, bz)
        walk = float(TS._passports()[sp["theirs"][k]]["speed"]["walk"]) * t_dec
        lx, lz = (px + bx / bn * walk, pz + bz / bn * walk) if bn > 1e-6 else (px, pz)
        new_starts.append((lx, lz) if _stand(mp, lx, lz, hb) else (sx + dx, sz + dz))
    o["pose"], o["goals"], o["starts"] = new_pose, new_goals, new_starts
    if o.get("twin") == "come":
        sp["places_theirs"] = [(x, z, b) for (x, z), (_, _, b) in zip(new_starts, pt)]
    else:
        sp["places_theirs"] = [(x, z, b) for (x, z), (_, _, b) in zip(new_pose, pt)]
    lx, lz, lb = sp["lords"][2]
    da, dc = rnd.uniform(-LORD_M, LORD_M), rnd.uniform(-LORD_M, LORD_M)
    nx, nz = lx + da * fx + dc * rx, lz + da * fz + dc * rz
    if _stand(mp, nx, nz, lb):
        sp["lords"] = {**sp["lords"], 2: (nx, nz, lb)}
    sp["fair"] = {"jitter_m": JITTER_M, "jitter_along_m": JITTER_ALONG_M, "lord_m": LORD_M, "walk_in": True}
    return sp


def scene(seed, held=None, twin=None):
    from net.operative import scenes_o1 as SO
    return fair(SO.scene(seed, held=held, twin=twin))
