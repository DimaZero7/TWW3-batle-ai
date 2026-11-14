"""The new units in the game before the simulator knows them (build/newunits/plan.md 4): the Ratling Gun, the Plague
Monks, the Plague Censer Bearers, the Wolf Rats (plain and poisoned) and the Halberdiers. The lanes run on the melee probe
(game/probe/charge_probe.py, bridge/lua/entries/charge_probe.lua: its plans by these names); one mechanic a plan:

  ratling   the gun's fire (R1): it shoots held fearless swordsmen 60 / 100 / 135 m off (front to front), shield spearmen
            100 m off, the swordsmen 100 m off with half its crew killed before the go, and the lone General 100 m off (few
            hits: the whole ammunition's cadence); fire at will, one ranged attack at a walk, RATLING_S - 2 battles of 6
            lanes, rotated. Measures: crew and ammunition at the go, the first shot, bullets a second and a man, the bursts
            (ammo drops between gaps of 1+ s), the time to empty, HP the target lost per bullet, the target's phases (fx:
            suppressing fire?) and card speed;
  ratlord   the gun's bullets on a lone man (the itS26_15 exam: our General out under the guns' fire, the simulator's
            lone-man hits half the game's): the General 135 m off (front to front) facing the gun (his shield, 55 %, in its
            60 deg) or his back to it, 100 m off his back to it (the ratling plan has him 100 m off facing), RATLORD_S -
            3 battles of one lane (the army has one General). Measures: as ratling (HP per bullet);
  ratlone   flat and arcing fire at a lone man (the gun's lone-man hits 0.7-0.8 of the game, arrows' right: the gun's or
            flat fire's?): the gun at a Gold Wizard 60 / 100 / 135 m off, handguns at a Warlock Engineer 60 / 100 / 135 m
            off, crossbows at him 100 / 135 m off (RATLONE_LANES) - 1 battle of 8 lanes, RATLONE_S. Measures: as ratling;
  ratform   the gun on formations (sim 1.43 of the game where lone men are ~1.0): the gun 100 m off at swordsmen
            (shield 35 %) facing / their backs to it and at flagellants (no shield, armour 0), at flagellants 60 m; handguns
            the same at clanrats (shield) and skavenslaves (none) (RATFORM_LANES) - 1 battle of 7 lanes, RATFORM_S. Measures: as ratling;
  ratshape  the shape of the gun's spread: the gun 100 m off at swordsmen 6 / 12 / 30 m wide and a lone Gold Wizard
            (RATSHAPE_LANES) - 1 battle of 6 lanes, RATSHAPE_S. Measures: as ratling;
  suppress  suppressing fire (R2): swordsmen attack the gun from SUPPRESS_GAP m at a walk / at a run, the gun aimed at them
            (fire at will, its skirmish mode off) or silent (never ordered: the control); the swordsmen's speed under fire and
            without, their card speed - 2 battles of 4 lanes (the gun fires on in melee: the phase never ends there);
  suppress2 the phase's life (R2): the same walk / run with the gun falling silent (fire at will off, a halt) 25 / 15 s
            after the go; the phase after its last shot, the speed back - 1 battle of 4 lanes;
  censer    the censer's contagion, the monks' frenzy, the halberds (C1, H1): censer bearers -> swordsmen, monks ->
            swordsmen (no contact effect: the control), censers -> shield spearmen, swordsmen -> held censers, censers
            leaving 20 s after contact with the swordsmen halting (the effect's life after the last blow), the two pairs again
            with the swordsmen keeping their morale (the rout), halberdiers -> stormvermin with shields; from 30 m at a run,
            60 s - 2 battles of 8 lanes on a grid, rotated;
  wolfrats  the beasts' pace and charge (W1, W2): a run 150 m aside and back (speed, acceleration, the turn), a charge from
            120 m on swordsmen (the charge speed), from 40 m on braced spearmen, on swordsmen facing away (the rear), on the
            General, a recharge (30 m out, 3 s, in again), the chase of routed swordsmen (half their men killed at
            placement, routed 15 s after contact) and the same chase by clanrats (the control) - 2 battles of 8 lanes;
  losfriend flat fire past friends (08.11, the bridge's chain battles: long rifles / jezzails behind their infantry
            walked through it into melee or did not fire; the simulator holds or takes another target): the shooter
            (long rifles, jezzails) ordered to attack ('shoot') an enemy LOSFRIEND_GAP m off (front to front, within
            range), its own infantry unit standing LOSFRIEND_COVER m ahead of it (between it and the target, as wide),
            and the control without it - 1 battle of 4 lanes, LOSFRIEND_S. Measures: the first shot, bullets, how far the
            shooter moved forward / aside and when it stopped, its offset from the cover at the first shot;
  losrange  why flat fire did not fire in the chain's battles (08.11; losfriend: friends ahead are not it): the shooters
            alone, ordered to attack, at 200 / 230 / 250 m front to front facing the target, and at 230 m with their
            backs to it (the battles' jezzails stood at bearing 265 with the target east) - battle 1, flat; the M3
            battle's jezzails on ksl_plains_01 at their place and bearing of 14 s (the flagellants 244 m off between
            the centres, the ground between) - battle 2, and the same place facing the target - battle 3 (LOSRANGE_*).
            Measures: losfriend's, and the distance to the target (centres) at the go, at the first shot and at the end,
            the way walked;
  losmove   a target coming at the shooter (08.11, the chain's battles: long rifles under an attack on advancing
            swordsmen walked 45 m ahead without a shot; jezzails held 40 s without one): the target walks / runs at the
            shooter from LOSMOVE_GAP m (front to front) through it; the shooter (long rifles, jezzails) ordered to attack
            it ('shoot') or with no order and fire at will on (a_free: our 'hold' of a shooter) - 1 battle of 8 lanes,
            LOSMOVE_S. Measures: the first shot and the distance then, shots before the contact, how far the shooter
            walked forward;
  snipe     snipers in the woods (08.11, build/diag_snipe: the attribute snipe - firing does not reveal a hidden unit -
            and gunpowder fired from the forest's depth): on Kislev (ksl_plains_01), the shooter's front at SNIPE_SITES' P,
            facing b - 'edge' 15-19 m inside the forest's edge, 'deep' 55-68 m inside, 'open' on grass (the control): a
            ranged attack ('shoot') SNIPE_GO_S after the go (it hides first: 5 s in the woods) on a held target SNIPE_GAP m
            off (front to front) on open ground; 'approach': 15 m inside the edge, fire at will and no order (the bridge's
            'stand' of a shooter), an enemy walks at it from SNIPE_APPROACH_GAP m and stops SNIPE_STOP_M m off its front
            (push_to). Battle 1 the Hochland Long Rifles (hide_forest + snipe) on clanrats, battle 2 the Jezzails
            (hide_forest) on swordsmen - 2 battles of 4 lanes, SNIPE_S / SNIPE_APPROACH_S. Both rows carry hid
            (is_hidden) and v (visible to the other alliance; stealth_rows 'both'). Measures: ratling's shots and HP per
            bullet, the shooter hidden / seen at the order, while firing and after it, the gap (edge to edge) when the
            approaching enemy first saw it and its state at 100 / 60 / 40 m;
  wolfpoison the poison (W3): poisoned wolf rats -> swordsmen, plain ones (the control), the poisoned leaving 15 s after
            contact with the swordsmen halting or chasing them, the plain leaving chased (the control) - 2 battles of 5 lanes.
  flank2v1  two on one (stage 0's drill 'pin and strike the flank'; F2_* below): our spearmen pin clanrats frontally, 20 s
            later our swordsmen strike from the flank / the rear / the front beside the anvil / straight from their place in
            the line, or stand (the control); a deep and a shallow target - 8 battles. Table and twin:
                python -m game.probe.unit_probe --flank2v1 game [runs] [--sim]   # the game (+ the twin on its starts)
                python -m game.probe.unit_probe --flank2v1 layout                # the twin alone (no game run)
  counterfar the meetfar probe's part 1 (CF_* below): the counter plan from 200 m - clanrats at a run / a walk, ours hold or
            attack once the gap is 35-150 m; clanrats standing, ours attack from 45 / 100 / 160 m - 6 battles;
  intercept the meetfar probe's part 2 (IC_* below): two clanrat units sent at our A, our free neighbour B holds / takes
            the second one far out / near / strikes either once pinned - 8 battles. Tables and twins:
                python -m game.probe.unit_probe --counterfar game [runs] [--sim] | --counterfar layout
                python -m game.probe.unit_probe --intercept game [runs] [--sim]  | --intercept layout

All the units are fearless (but the t_morale targets); every lane's rows carry the morale fields with the active phases (fx)
and the unit card's values (cs: leadership, speed, melee attack / defence, weapon and missile damage).

    python -m game.probe.charge_probe plan --plan ratling            # the battles
    python -m game.probe.charge_probe run --plan ratling --speed 10   # build + launch each, in turn
    python -m game.probe.unit_probe [runs...]                         # the five plans' tables (the game's numbers)

Units by short name (game/probe/charge_probe.UNITS) or any Empire / Skaven key with a passport (sim/config/units.json,
sim/data/probe-units.json); the plans' units are this module's constants (RATLING, CENSER, ...).
"""
import argparse
import math
import json
import sys
from pathlib import Path

import numpy as np

from game.probe import charge_probe as cp

RATLING, MONKS, CENSER, WOLF, WOLFP, HALB = "ratling", "monks", "censer", "wolfrat", "wolfratp", "halb"
GRID = cp.RETARGET2_PLACES          # 10 places 290+ m apart in x, 400 m in z
FX = dict(morale_rows=True, card_rows=True)
RATLING_S = 200          # the game: a burst of 18 a man every ~12 s (7 s of fire, 5 of reload), 15 bursts: ~180 s
# (kind, target, gap front to front m, share of the crew killed at placement)
RATLING_LANES = (("d60", "swords", 60, None), ("d100", "swords", 100, None), ("d135", "swords", 135, None),
                 ("shield100", "spearsh", 100, None), ("half100", "swords", 100, 0.5), ("lord100", "general", 100, None))
# one lord lane a battle (the army has one General); the ratling plan's lord100 lane (facing) is the 100 m front
RATLORD_LANES = (("lord135_front", 135, "front"), ("lord100_back", 100, "back"), ("lord135_back", 135, "back"))
RATLORD_S = 60           # the General (4068 HP) lives ~2 bursts at 100 m
# the ratcover plan: a lone Gold Wizard (no shield, 3420 HP; the army has one General, the plan needs one lone man a lane)
# facing the gun RATCOVER_GAP m off, an Empire swordsmen unit of his side (30 m wide, 9.6 m deep) gap m off him -
# (kind, where, gap): ahead (towards the gun), behind, beside; none (the control); 2 battles, the places rotated
RATCOVER_TARGET, RATCOVER_UNIT, RATCOVER_GAP, RATCOVER_R = "goldwiz", "swords", 130, 0.6
RATCOVER_LANES = (("alone", None, 0), ("front5", "front", 5), ("front10", "front", 10), ("front20", "front", 20),
                  ("side5", "side", 5), ("back1", "back", 1), ("back5", "back", 5))
# the ratlone plan: shots at a lone man by flat and arcing fire (is the gun's lone-man deficit the gun's or flat fire's?):
# the gun at a Gold Wizard, handguns (flat) and crossbows (arcing) at a Warlock Engineer (no shields, both heroes: one a
# lane), facing the shooter, (kind, shooter, target, gap front to front m); 1 battle
RATLONE_LANES = (("gun60", RATLING, "goldwiz", 60), ("gun100", RATLING, "goldwiz", 100),
                 ("gun135", RATLING, "goldwiz", 135), ("hgun60", "hgun", "warlock", 60),
                 ("hgun100", "hgun", "warlock", 100), ("hgun135", "hgun", "warlock", 135),
                 ("xbow100", "xbow", "warlock", 100), ("xbow135", "xbow", "warlock", 135))
RATLONE_S = 60
# the ratform plan: the gun's excess on formations (sim 1.43 of the game, lone men ~1.0): the shields or the hits? The gun
# at 100 m at swordsmen (shield 35 %) facing and their backs to it, and at flagellants (no shield, armour 0: HP per bullet
# / 8 is the hits), at flagellants 60 m too; handguns the same at clanrats (shield 35 %) and skavenslaves (none, armour 0)
# - an enemy of the Empire; (kind, shooter, target, gap, target mode); 1 battle
RATFORM_LANES = (("f_swords", RATLING, "swords", 100, "hold"), ("b_swords", RATLING, "swords", 100, "rear"),
                 ("f_flag", RATLING, "flag", 100, "hold"), ("f_flag60", RATLING, "flag", 60, "hold"),
                 ("hg_f_clanrat", "hgun", "clanrat", 100, "hold"), ("hg_b_clanrat", "hgun", "clanrat", 100, "rear"),
                 ("hg_f_slave", "hgun", "slave", 100, "hold"))
RATFORM_S = 200
# the ratshape plan: the shape of the gun's spread (its formation / lone ratio is ~1.3x the game's in the model: wider in
# height than across?): the gun 100 m off at swordsmen 6 / 12 / 30 m wide (4 / 8 / 20 files, the same men deeper) and a
# lone Gold Wizard - a narrow column catches the lateral spread's middle only, a wide front all of it, both the same
# height; (kind, target, width m); 1 battle
RATSHAPE_LANES = (("w6", "swords", 6), ("w6b", "swords", 6), ("w12", "swords", 12), ("w12b", "swords", 12),
                  ("w30", "swords", 30), ("lone", "goldwiz", 5))
RATSHAPE_S = 60
# the losfriend plan: (kind, shooter, its cover (None: the control), the target); the cover LOSFRIEND_COVER m ahead of the
# shooter's front edge (its back edge), the target LOSFRIEND_GAP m off the shooter (front to front)
LOSFRIEND_LANES = (("lr_cover", "wh3_dlc25_emp_inf_hochland_long_rifles", "swords", "clanrat"),
                   ("lr_alone", "wh3_dlc25_emp_inf_hochland_long_rifles", None, "clanrat"),
                   ("jz_cover", "wh2_dlc12_skv_inf_warplock_jezzails_0", "clanrat", "swords"),
                   ("jz_alone", "wh2_dlc12_skv_inf_warplock_jezzails_0", None, "swords"))
LOSFRIEND_GAP, LOSFRIEND_COVER, LOSFRIEND_S = 180, 20, 120
# the losrange plan: (kind, shooter, target, gap front to front, facing 'front' / 'back'); the map lanes (the M3 battle
# 20261108-070927 at 14 s, ksl_plains_01's world metres): (kind, shooter, (x, z, b), target, (x, z, b))
LR, JZ = "wh3_dlc25_emp_inf_hochland_long_rifles", "wh2_dlc12_skv_inf_warplock_jezzails_0"
LOSRANGE_LANES = (("lr200", LR, "clanrat", 200, "front"), ("lr230", LR, "clanrat", 230, "front"),
                  ("lr250", LR, "clanrat", 250, "front"), ("jz200", JZ, "swords", 200, "front"),
                  ("jz230", JZ, "swords", 230, "front"), ("jz250", JZ, "swords", 250, "front"),
                  ("lr230_back", LR, "clanrat", 230, "back"), ("jz230_back", JZ, "swords", 230, "back"))
LOSRANGE_MAP = (("m3_as_game", JZ, (-104.4, 160.6, 265.0), "flag", (137.5, 132.0, 277.0)),
                ("m3_facing", JZ, (-104.4, 160.6, 97.0), "flag", (137.5, 132.0, 277.0)))
LOSRANGE_S = 90
# the losmove plan: (kind, shooter, target, order 'shoot' / 'free', the target's pace 'walk' / 'run')
LOSMOVE_LANES = tuple((f"{s}_{o}_{p}", key, tg, o, p) for s, key, tg in (("lr", LR, "clanrat"), ("jz", JZ, "swords"))
                      for p in ("walk", "run") for o in ("shoot", "free"))
LOSMOVE_GAP, LOSMOVE_S = 320, 300
# the snipe plan (build/diag_snipe/find_sites.py, sites.png): kind -> (P x, z: the shooter's front centre, its bearing,
# the forest it fires through m); the target SNIPE_GAP m off (front to front) on open level ground, the line of sight
# over the ground clear; the approach: the enemy from SNIPE_APPROACH_GAP m to SNIPE_STOP_M m off the shooter's front
SNIPE_SITES = {"edge": (416.0, 376.0, 165.0, 19), "deep": (-40.0, 376.0, 120.0, 62), "open": (-328.0, 160.0, 195.0, 0),
               "approach": (-40.0, -312.0, 75.0, 16)}
SNIPE_SHOOTERS = ((LR, "clanrat"), (JZ, "swords"))       # battle 1, battle 2
SNIPE_GAP, SNIPE_APPROACH_GAP, SNIPE_STOP_M, SNIPE_GO_S, SNIPE_S, SNIPE_APPROACH_S = 150, 240, 25, 8, 100, 240
SUPPRESS_GAP = 120
SUPPRESS2_STOP = {"walk_stop25": (True, 25), "run_stop15": (False, 15)}     # kind -> (walk, the gun silent after s)
HITSLOW_SHOOTERS = ("hgun", "xbow")      # the hitslow plan's shooters (no contact effect): handguns flat, bolts arcing
HITSLOW_RUNNER = "wh2_main_skv_inf_clanrats_0"     # ... and the runners at them: clanrats without shields
PACE_M, PACE_BACK_S = 150, 20
CHASE = dict(t_morale=True, at_rout="none", after_rout_s=45, max_s=200, rout_at_s=15,
             damage={"t": {"method": "kill", "share": 0.5}})


# ---------------------------------------------------------------- stage 0-1 mechanics (build/diag_stage0_probe)
# the hill plan (build/diag_stage0_probe/find_hill.py; ksl_plains_01): kind -> (the target's centre x, z, its bearing
# towards the enemy's approach, the drop target - ground HILL_GAP m out); the target (tested spearmen, its morale kept)
# holds, the enemy (fearless clanrats) walks up from HILL_GAP m (front to front) to HILL_STOP m off its front
HILL_SITES = {"gentle": (272.0, 96.0, 150.0), "steep": (-200.0, -136.0, 60.0), "level": (64.0, -200.0, 0.0)}
HILL_T, HILL_A = "spear", "clanrat"
HILL_GAP, HILL_STOP, HILL_S = 200, 15, 200
# the flankslot plan (flat): the morale's three neighbour slots (memory: nb1 left, nb2 right, nb3 rear; 0 / 1 / 2) and
# flanks exposed / secure against the script's lf / rf / bf. The target (tested spearmen, its morale kept) holds at its
# place facing +z (b 0; its right +x); the enemy (fearless clanrats, 'script' moves from its start) and the target's
# friends (fearless spearmen, side 't') by kind - kind -> (the enemy's start dx, dz from the target's centre, its
# bearing, [steps], [friends' dx]): side_stand from 160 m on the right in at a run to 120 / 90 / 65 / 45 / 30 m
# (centre to centre), standing between; rear_walk from 200 m behind at a walk to 35 m; past_left at a run 50 m off its
# left from 150 m ahead to 150 m behind (facing its way, not the target); friends - spearmen 40 m left (10 m gap) and
# 80 m right (50 m gap), the enemy from 200 m ahead at a walk to 60 m; friend_screen - a friend 45 m right (15 m gap),
# the enemy from 200 m right to 100 m, then to 75 m (7 m off the friend's edge); side_away - the enemy standing 45 m
# right with its back to the target, at 40 s it turns to face it, at 80 s away again
FLANK_T, FLANK_A, FLANK_F = "spear", "clanrat", "spear"
FLANK_PLACES = ((-400.0, -260.0), (0.0, -260.0), (400.0, -260.0), (-400.0, 240.0), (0.0, 240.0), (400.0, 240.0))
FLANK_LANES = {
    "side_stand": ((160.0, 0.0, 270.0), [(15, -40, 0, True), (35, -70, 0, True), (55, -95, 0, True),
                                          (75, -115, 0, True), (95, -130, 0, True)], []),
    "rear_walk": ((0.0, -200.0, 0.0), [(2, 0, 165, False)], []),
    "past_left": ((-50.0, 150.0, 180.0), [(2, 0, -300, True)], []),
    "friends": ((0.0, 200.0, 180.0), [(20, 0, -140, False)], [-40.0, 80.0]),
    "friend_screen": ((200.0, 0.0, 270.0), [(2, -100, 0, False), (70, -125, 0, False)], [45.0]),
    "side_away": ((45.0, 0.0, 90.0), [(40, "face", 270.0), (80, "face", 90.0)], []),
}
FLANK_S = 130
# the flankslot2 plan (battle 2, the bounds the first left open): front_walk - from 150 m ahead at a walk to 30 m;
# diag_front / diag_rear - from 150 m at 45 deg front-right / 135 deg rear-right at a walk to 25 m; lateral_out - 40 m
# right and 25 m ahead, out to 55 / 65 / 75 m, then 40 m right at 45 / 60 m ahead (at a run, standing between);
# friends2 - friends 55 m left and 70 m right, the enemy from 200 m behind to 100 m; friends3 - friends 50 m left and
# 60 m right, the enemy from 200 m left to 110 m (beyond the left friend)
FLANK2_LANES = {
    "front_walk": ((0.0, 150.0, 180.0), [(2, 0, -120, False)], []),
    "diag_front": ((106.0, 106.0, 225.0), [(2, -88, -88, False)], []),
    "lateral_out": ((40.0, 25.0, 270.0), [(15, 15, 0, True), (30, 25, 0, True), (45, 35, 0, True), (60, 0, 20, True),
                                         (80, 0, 35, True)], []),
    "diag_rear": ((106.0, -106.0, 315.0), [(2, -81, 81, False)], []),
    "friends2": ((0.0, -200.0, 0.0), [(20, 0, 100, False)], [-55.0, 70.0]),
    "friends3": ((-200.0, 0.0, 90.0), [(20, 90, 0, False)], [-50.0, 60.0]),
}
# the lobfriend plan (flat): arcing fire over own infantry (stage 1's 'fire over the head'): (kind, shooter, its width,
# the target, gap front to front m, the cover's short name or None, the cover's back edge this far ahead of the
# shooter's front m); a held target, LOB_S of fire
LOB_LANES = (("arch_c20", "archer", 30, "wh2_main_skv_inf_clanrats_0", 100, "spear", 20),
             ("arch_c5", "archer", 30, "wh2_main_skv_inf_clanrats_0", 100, "spear", 5),
             ("arch_alone", "archer", 30, "wh2_main_skv_inf_clanrats_0", 100, None, 0),
             ("xbow_c20", "xbow", 30, "wh2_main_skv_inf_clanrats_0", 110, "spear", 20),
             ("sling_c20", "slinger", 45, "spear", 80, "wh2_main_skv_inf_clanrats_0", 20),
             ("sling_alone", "slinger", 45, "spear", 80, None, 0))
LOB_S = 70
# the forestwalk plan (ksl_plains_01; build/diag_g49/ground/sites.json: 60 m level legs fully in forest / on grass):
# spearmen to and fro between their place and LEG_M ahead, kind -> (x, z, bearing, run); the game's legs at a run 0.87-
# 0.93 of grass (the ground probe): a walk too, and the men from memory (each man's step speed: a ground rule or the
# way round the trunks?)
FOREST_SITES = {"forest1_walk": (-48.0, -360.0, 0.0, False), "forest2_walk": (-80.0, -200.0, 150.0, False),
                "forest3_run": (120.0, -360.0, 150.0, True), "forest4_run": (408.0, 456.0, 180.0, True),
                "grass1_walk": (416.0, 232.0, 90.0, False), "grass2_walk": (-176.0, 64.0, 300.0, False),
                "grass3_run": (272.0, -24.0, 210.0, True), "grass4_run": (112.0, -40.0, 0.0, True)}
FOREST_S = 150


# the width plan (09.11; the stage 0 unit exam: in the simulator a line filling its area loses the trade -0.156 against
# the network's narrow units -0.033; the wrap probe: the wide side's trade 1.20-1.66 in the game, 1.00-1.03 in the twin):
# fearless clanrats (no shields, 30 m) charge from WIDTH_GAP m at our spearmen of WIDTH_T m front, who hold (braced,
# never ordered) or charge them back ('both': the counter-charge); two neighbours of ours of the same front stand on its
# line, the centres 40 m apart (a line of three: the narrow line's gaps, the wide one edge to edge); kind -> (front m,
# the neighbours' gap m, target mode); WIDTH_S of fight; WIDTH_ROT: the battles, the places rotated by k
WIDTH_A, WIDTH_T = "wh2_main_skv_inf_clanrats_0", "spear"
WIDTH_LANES = (("w38_hold", 38, 2, "hold"), ("w30_hold", 30, 10, "hold"), ("w20_hold", 20, 20, "hold"),
               ("w38_both", 38, 2, "both"), ("w30_both", 30, 10, "both"), ("w20_both", 20, 20, "both"))
WIDTH_GAP, WIDTH_S, WIDTH_ROT = 60, 60, (0, 3, 1, 4)


def width_battles():
    """The width plan's battles (WIDTH_LANES on the grid, rotated by WIDTH_ROT)."""
    base = []
    for kind, w, gap, tmode in WIDTH_LANES:
        nb = [{"short": WIDTH_T, "dx": sgn * (w + gap), "dz": 0.0, "b": 0.0, "width": float(w), "side": "t"}
              for sgn in (-1, 1)]
        base.append(cp.lane(WIDTH_A, WIDTH_T, "attack_run", tmode, gap_m=WIDTH_GAP, fight_s=WIDTH_S,
                            max_s=WIDTH_GAP // 2 + WIDTH_S + 10, answer=False, kind=kind, t_w=float(w), extras=nb, **FX))
    return [_placed(base, k) for k in WIDTH_ROT]


# the reissue plan (13.11; METHOD.md 'Уровень': the code re-gives an order only when its point moved past a threshold -
# net/common/reissue.py MOVE_M 8 / FACE_DEG 22.5, not from a probe; sim.json turn.face_why 'after a move ... OPEN'):
# Empire spearmen (30 m front) alone, the enemy REISSUE_GAP m off (held, never ordered). Battle 'stand': at 1 s a move
# with width and facing (move_formation) REISSUE_STAND_DZ m ahead at a run, settled long before REISSUE_STAND_AT s; then
# the same order again, its point shifted aside by dx m and / or its facing turned by deg ('same': both 0; 'f90': a facing
# order where it stands, never moved before - the face_why control). Battle 'move': at 1 s the same order REISSUE_MOVE_DZ
# m ahead, REISSUE_MOVE_AT s (mid-run) again shifted / turned ('none': no second order; 'mface90': a facing order where
# it is - stop and turn). The facings sit on the game's 64 sectors (5.625 deg): base REISSUE_B, turns 2 / 4 / 8 / 16
# sectors. (kind, dx m, turn deg, how: 'move' | 'face' | None)
REISSUE_GAP, REISSUE_B, REISSUE_W = 200, 182.8125, 30.0
REISSUE_STAND_DZ, REISSUE_STAND_AT, REISSUE_STAND_S = 30, 20, 50
REISSUE_MOVE_DZ, REISSUE_MOVE_AT, REISSUE_MOVE_S = 100, 10, 60
REISSUE_STAND = (("same", 0, 0, "move"), ("d2", 2, 0, "move"), ("d5", 5, 0, "move"), ("d10", 10, 0, "move"),
                 ("d20", 20, 0, "move"), ("a11", 0, 11.25, "move"), ("a22", 0, 22.5, "move"), ("a45", 0, 45, "move"),
                 ("a90", 0, 90, "move"), ("f90", 0, 90, "face"))
REISSUE_MOVE = (("none", 0, 0, None), ("same", 0, 0, "move"), ("d2", 2, 0, "move"), ("d5", 5, 0, "move"),
                ("d10", 10, 0, "move"), ("d20", 20, 0, "move"), ("a11", 0, 11.25, "move"), ("a22", 0, 22.5, "move"),
                ("a45", 0, 45, "move"), ("mface90", 0, 90, "face"))
REISSUE_PLACES = tuple((float(x), z) for z in (-450.0, 100.0) for x in (-480, -240, 0, 240, 480))


def reissue_battles():
    """The reissue plan's 4 battles: stand, move, stand and move with the places rotated by 5."""
    def ln(kind, dx, deg, how, battle):
        dz, at, max_s = ((REISSUE_STAND_DZ, REISSUE_STAND_AT, REISSUE_STAND_S) if battle == "stand"
                         else (REISSUE_MOVE_DZ, REISSUE_MOVE_AT, REISSUE_MOVE_S))
        first = {"at_s": 1, "kind": "move", "dx": 0, "dz": -dz, "bearing": REISSUE_B, "width": REISSUE_W, "run": True}
        b2 = round((REISSUE_B + deg) % 360, 4)
        steps = [] if (battle == "stand" and how == "face") else [first]
        if how == "move":
            steps.append({"at_s": at, "kind": "move", "dx": dx, "dz": -dz, "bearing": b2, "width": REISSUE_W,
                          "run": True})
        elif how == "face":
            steps.append({"at_s": at, "kind": "face", "bearing": b2, "width": REISSUE_W})
        return cp.lane("spear", "clanrat", "script", "hold", gap_m=REISSUE_GAP, fight_s=0, max_s=max_s, answer=False,
                       kind=f"{battle}_{kind}", steps=steps, men_all_s=max_s, reissue_at=at, sim_face=True,
                       turn_deg=deg, shift_m=dx, **FX)
    stand = [ln(*x, "stand") for x in REISSUE_STAND]
    move = [ln(*x, "move") for x in REISSUE_MOVE]
    out = []
    for k in (0, 5):
        for base in (stand, move):
            out.append([dict(s, place=p) for s, p in zip(cp.rotate(list(base), k), REISSUE_PLACES)])
    return out


# the counter plan (13.11; the counter-charge: net/unit/branch.py's moment - the enemy within its charge distance 30 m
# + 4 s of its run - from the simulator): fearless clanrats (no shields, 30 m) charge at a run from COUNTER_GAP m (front
# to front, 'both') our Empire spearmen / swordsmen (30 m), who hold (braced, never ordered) or counter-charge (an attack
# at a run) when the gap is X m: the order's time from the clanrats' run on a held unit (the width plan's runs,
# build/charge-probe/runs/20261109-123515: off at ~1.8 s, then ~3.85 m/s front to front) - the gap at the order is
# measured; COUNTER_S of fight after contact. 4 battles of 10 of the 12 lanes, rotated on the grid
COUNTER_T = "wh2_main_skv_inf_clanrats_0"
COUNTER_DEF = ("spear", "swords")
COUNTER_AT = (None, 60, 40, 30, 20, 10)
COUNTER_GAP, COUNTER_S, COUNTER_T0, COUNTER_V = 90, 35, 1.8, 3.85
COUNTER_ROT = (0, 6, 3, 9)
COUNTER_LATE_S = 24.4


def counter_battles():
    base = []
    for d in COUNTER_DEF:
        for x in COUNTER_AT:
            extra = {} if x is None else {"a_go_s": round(COUNTER_T0 + (COUNTER_GAP - x) / COUNTER_V, 1), "counter_at": x}
            base.append(cp.lane(d, COUNTER_T, "hold" if x is None else "attack_run", "both", gap_m=COUNTER_GAP,
                                fight_s=COUNTER_S, max_s=round(COUNTER_T0 + COUNTER_GAP / COUNTER_V) + COUNTER_S + 5,
                                answer=False, kind=f"{d}_{'hold' if x is None else f'c{x}'}",
                                **FX, **extra))
    out = [_placed(base, k) for k in COUNTER_ROT]
    # battles 5, 6: the counter at contact ('ans': the clanrats attack our unit standing - target mode 'stand' with
    # the answer, our attack order at its first melee flag; ours is the lane's target, ours 'tg') and a late counter
    # (the order COUNTER_LATE_S after the go: the gap ~10 m by battles 1-4), the holds again
    more = []
    for d in COUNTER_DEF:
        more.append(cp.lane(COUNTER_T, d, "attack_run", "stand", gap_m=COUNTER_GAP, fight_s=COUNTER_S,
                            max_s=round(COUNTER_T0 + COUNTER_GAP / COUNTER_V) + COUNTER_S + 5, answer=True,
                            kind=f"{d}_ans", ours="tg", **FX))
        more.append(cp.lane(d, COUNTER_T, "attack_run", "both", gap_m=COUNTER_GAP, fight_s=COUNTER_S,
                            max_s=round(COUNTER_T0 + COUNTER_GAP / COUNTER_V) + COUNTER_S + 5, answer=False,
                            kind=f"{d}_late", a_go_s=COUNTER_LATE_S, counter_at=5, **FX))
    holds = [b for b in base if b["kind"].endswith("_hold")]
    two = more + holds + more
    out += [_placed(two, 0), _placed(two, 5)]
    return out


# the flank2v1 plan (13.11; tmp/tactic_slice/review_flank.md 'Порядок' 2: the drill 'pin and strike the flank' - how much
# better is the second unit joining from the flank / rear than from the front or not at all, in the game, and does the
# simulator agree in magnitude). A lane: the target (clanrats with shields, its morale kept) faces +z at its place; our
# anvil (Empire spearmen, 30 m, an extra) stands F2_ANVIL_GAP m off its front and both attack each other at a walk at the
# go; our hammer (Empire swordsmen, 30 m) stands on the anvil's line F2_LINE_DX m to its right (centre to centre),
# facing the enemy. Before the strike the hammer goes (at a run, from the go) to its approach point - 'flank': F2_OFF m
# (edge to edge) off the target's right flank facing it, 'rear': round the flank (a corner point F2_CORNER m clear of
# it) to F2_OFF m behind its back facing it, 'front': beside the anvil (F2_BESIDE m between their edges) facing the enemy
# - and attacks the target at a run F2_AFTER_S s after the lane's first melee flag (the anvil's; lane.after
# attack_same); 'place' attacks from its place in the line, 'stand' never moves (the control). Everyone of ours is
# fearless (the probe's rule); the target routs and rallies by the game (t_morale, the chase on: at_rout 'none').
# Targets: 'deep' 30 m wide (160 men, ~15 m deep by the database's spacing) and 'shallow' - the same unit 45 m wide
# (~10 m deep): the same stats, only the rectangle (its diagonals from the facing: ~63 and ~77 deg). One battle a
# target and repeat (5 lanes, 400+ m apart; the places rotated by 2 a repeat): 8 battles (4 repeats). Each battle has a
# 'clock' lane too - the two Generals 100 m apart far off the others (F2_CLOCK), never ordered: its men are polled from
# memory every 0.1 s (men_near_m), which keeps the entry's memory poll going at every step, so the targets' morale rows
# (mem_morale) come every 0.5 s (with no lane polled densely they never come: entries/charge_probe.lua mem_poll); it
# is no measure (battles 1-4 of 13.11 17:41-17:50 ran without it: no morale rows)
F2_T, F2_ANVIL, F2_HAMMER = "clanrat", "spear", "swords"
F2_TARGETS = (("deep", 30.0), ("shallow", 45.0))
F2_KINDS = ("flank", "rear", "front", "place", "stand")
F2_LINE_DX, F2_OFF, F2_CORNER, F2_BESIDE, F2_ANVIL_GAP = 48.0, 20.0, 15.0, 2.0, 1.5
F2_AFTER_S, F2_REAR_TURN_S, F2_FIGHT_S, F2_MAX_S = 20, 8, 160, 175
F2_PLACES = ((-400.0, -250.0), (0.0, -250.0), (400.0, -250.0), (-200.0, 250.0), (200.0, 250.0))
F2_WINDOWS = ((0, 5), (5, 20), (20, 60), (60, 120))
F2_CLOCK = ((650.0, 650.0), (650.0, 550.0))      # the clock lane's General and Warlord (centres)
F2_REPEATS = 4
F2_STAND_REF_S = 25.0     # the 'stand' lane's windows start this long after the anvil's contact (the others' hammer contact)


def flank2v1_battles():
    """The flank2v1 plan's 4 battles (deep, shallow, deep, shallow; the repeat's places rotated by 2)."""
    W = cp.WIDTH_M
    tk, tm, _ = cp.UNITS[F2_T]
    ak, am, _ = cp.UNITS[F2_ANVIL]
    hk, hm, _ = cp.UNITS[F2_HAMMER]
    ad, hd = cp.depth(ak, am, W), cp.depth(hk, hm, W)
    out = []
    (gx_, gz_), (wx_, wz_) = F2_CLOCK
    clock = cp.lane("general", "warlord", "hold", "hold", gap_m=round(gz_ - wz_), fight_s=F2_FIGHT_S, max_s=F2_MAX_S,
                    answer=False, kind="clock", a_at={"x": gx_, "z": gz_, "b": 180.0}, t_at={"x": wx_, "z": wz_, "b": 0.0},
                    place=(wx_, wz_))
    for rep in range(F2_REPEATS):
        for tname, tw in F2_TARGETS:
            td = cp.depth(tk, tm, tw)
            anvil_dz = round(td / 2 + F2_ANVIL_GAP + ad / 2, 1)
            lanes = []
            for kind, (px, pz) in zip(cp.rotate(list(F2_KINDS), 2 * rep), F2_PLACES):
                hx, hz = px + F2_LINE_DX, pz + anvil_dz          # the hammer's place on the anvil's line
                steps, mode = [], "hold"
                if kind == "flank":
                    gx, gz = px + tw / 2 + F2_OFF + hd / 2, pz
                    steps = [{"at_s": 0.5, "kind": "move", "dx": round(gx - hx, 1), "dz": round(gz - hz, 1),
                              "bearing": 270.0, "width": W, "run": True}]
                elif kind == "rear":
                    cx, cz = px + tw / 2 + F2_CORNER + W / 2, pz - td / 2 - F2_OFF
                    gx, gz = px, pz - td / 2 - F2_OFF - hd / 2
                    steps = [{"at_s": 0.5, "kind": "move", "dx": round(cx - hx, 1), "dz": round(cz - hz, 1),
                              "run": True},
                             {"at_s": F2_REAR_TURN_S, "kind": "move", "dx": round(gx - hx, 1), "dz": round(gz - hz, 1),
                              "bearing": 0.0, "width": W, "run": True}]
                elif kind == "front":
                    gx = px + W / 2 + F2_BESIDE + W / 2
                    steps = [{"at_s": 0.5, "kind": "move", "dx": round(gx - hx, 1), "dz": 0.0, "bearing": 180.0,
                              "width": W, "run": True}]
                if steps:
                    mode = "script"
                extra = {} if kind == "stand" else {"after": {"at_s": F2_AFTER_S, "kind": "attack_same"}}
                anvil = {"short": F2_ANVIL, "dx": 0.0, "dz": anvil_dz, "b": 180.0, "width": W, "attack": True}
                lanes.append(cp.lane(F2_HAMMER, F2_T, mode, "hold", gap_m=round(anvil_dz + ad / 2), fight_s=F2_FIGHT_S,
                                     max_s=F2_MAX_S, answer=False, kind=f"{tname}_{kind}", t_w=tw, a_w=W,
                                     a_at={"x": round(hx, 1), "z": round(hz, 1), "b": 180.0},
                                     t_at={"x": px, "z": pz, "b": 0.0}, place=(px, pz), steps=steps, sim_face=True,
                                     extras=[anvil], t_attack_extra=True, t_morale=True, at_rout="none",
                                     after_rout_s=F2_MAX_S, **FX, **extra))
            out.append(lanes + [dict(clock)])
    return out


# the meetfar probe (14.11; tmp/tactic_slice/stage1c_results.md decision 4: in the simulator's general battles 'attack an
# enemy nobody holds within 150 m' beats 'meet at ~35 m and strike pinned enemies' by 0.030 of plain health; its one-on-one
# lanes see no gain from attacking earlier, so the lead is battle-level - does the game pay for going out far?). Two plans.
# Part 1 'counterfar' (the counter plan farther out): fearless clanrats_0 (no shields, 30 m) start CF_GAP m off (front to
# front) our fearless Empire spearmen / swordsmen (30 m) and attack them at a run ('run') or at a walk ('walk'); ours hold
# (never ordered) or attack at a run once the centres are within X + the half depths (a_go_m: the front-to-front gap X of
# CF_AT; the gap at the order is measured); and lanes where the clanrats stand (never ordered) and ours attack at the go
# from CF_STAND_AT m ('stand'). One defender a battle (15 lanes on CF_PLACES), each lane 3 times (the places rotated by 5).
CF_T = COUNTER_T
CF_DEF = ("spear", "swords")
CF_GAP, CF_FIGHT_S = 200.0, 65
CF_AT = (35, 60, 90, 120, 150)
CF_STAND_AT = (45, 100, 160)
CF_PACE = (("run", "both", 4.2), ("walk", "both_walk", 1.5))     # the clanrats' order and speed (units.json passport)
CF_PLACES = tuple((float(x), z) for z in (-450.0, -50.0, 350.0) for x in (-480, -240, 0, 240, 480))
CF_REPEATS = 3
CF_WINDOWS = (3, 10, 30, 60)


def counterfar_battles():
    """The counterfar plan's 6 battles: spearmen, swordsmen, three times (the places rotated by 5 a repeat)."""
    W = cp.WIDTH_M
    tk, tm, _ = cp.UNITS[CF_T]
    td = cp.depth(tk, tm, W)
    per = {}
    for d in CF_DEF:
        dk, dm, _ = cp.UNITS[d]
        half = (cp.depth(dk, dm, W) + td) / 2
        lanes = []
        for pace, tmode, v in CF_PACE:
            for x in (None,) + CF_AT:
                extra = {} if x is None else {"a_go_m": round(x + half, 1), "counter_at": x}
                lanes.append(cp.lane(d, CF_T, "hold" if x is None else "attack_run", tmode, gap_m=CF_GAP,
                                     fight_s=CF_FIGHT_S, max_s=round(3 + CF_GAP / v) + CF_FIGHT_S + 10, answer=False,
                                     kind=f"{d}_{pace}_{'hold' if x is None else f'c{x}'}", **extra))
        for x in CF_STAND_AT:
            lanes.append(cp.lane(d, CF_T, "attack_run", "hold", gap_m=float(x), fight_s=CF_FIGHT_S,
                                 max_s=round(3 + x / 2.5) + CF_FIGHT_S + 10, answer=False, kind=f"{d}_stand_c{x}",
                                 counter_at=x))
        per[d] = lanes
    return [[dict(s, place=p) for s, p in zip(cp.rotate(per[d], 5 * rep), CF_PLACES)]
            for rep in range(CF_REPEATS) for d in CF_DEF]


# Part 2 'intercept' (two enemies on one of mine, a free neighbour): our anvil A (Empire spearmen, 30 m, an extra) and B
# (Empire swordsmen, 30 m, the lane's attacker) side by side, IC_DX m centre to centre, facing the enemy; fearless
# clanrats_0 E1 (the lane's target) IC_GAP m (front to front) in front of A and E2 (the second target) IC_E2 from E1's
# centre (toward B's side, behind it), both attack A at a run at the go; A never ordered. B: 'hold' never ordered; 'far'
# attacks E2 at a run once E2 is IC_FAR m off (front to front: the centres less the half depths); 'near' at IC_NEAR m, or
# at E2's closest approach (its distance grew 2 m past its least, or E2 in melee first: a_go 'closest'); 'pinE1' attacks
# E1 IC_PIN_S after E1's first melee flag; 'pinE2' attacks E2 IC_PIN_S after E2's first melee flag. A and B keep their
# morale (they can rout), the clanrats are fearless. One lane of each kind a battle (5 places, rotated) and the clock
# lane (flank2v1's: the memory polls keep going); B on the right (+x) in odd battles, on the left in even ones.
IC_A, IC_B, IC_E = "spear", "swords", COUNTER_T
IC_DX, IC_GAP = 48.0, 200.0
IC_E2 = (24.0, -60.0)
IC_FAR, IC_NEAR, IC_PIN_S = 120, 35, 10
IC_FIGHT_S, IC_MAX_S = 125, 260
IC_KINDS = ("hold", "far", "near", "pinE1", "pinE2")
IC_PLACES = F2_PLACES
IC_BATTLES = 8
IC_WINDOWS = (30, 60, 120)


def intercept_battles(e2=None, n=IC_BATTLES):
    """The intercept plan's battles (module comment above IC_*): 5 lanes and the clock a battle."""
    e2 = e2 or IC_E2
    W = cp.WIDTH_M
    ak, am, _ = cp.UNITS[IC_A]
    bk, bm, _ = cp.UNITS[IC_B]
    ek, em, _ = cp.UNITS[IC_E]
    ad, bd, ed = cp.depth(ak, am, W), cp.depth(bk, bm, W), cp.depth(ek, em, W)
    half = (bd + ed) / 2
    (gx_, gz_), (wx_, wz_) = F2_CLOCK
    clock = cp.lane("general", "warlord", "hold", "hold", gap_m=round(gz_ - wz_), fight_s=IC_FIGHT_S, max_s=IC_MAX_S,
                    answer=False, kind="clock", a_at={"x": gx_, "z": gz_, "b": 180.0}, t_at={"x": wx_, "z": wz_, "b": 0.0},
                    place=(wx_, wz_))
    out = []
    for i in range(n):
        side = 1 if i % 2 == 0 else -1
        lanes = []
        for kind, place in zip(cp.rotate(list(IC_KINDS), i), IC_PLACES):
            mode, extra = {"hold": ("hold", {}),
                           "far": ("attack_t2", {"a_go_m": round(IC_FAR + half, 1)}),
                           "near": ("attack_t2", {"a_go_m": round(IC_NEAR + half, 1), "a_go_closest": True}),
                           "pinE1": ("attack_run", {"a_go_melee_s": IC_PIN_S}),
                           "pinE2": ("attack_t2", {"a_go_melee_s": IC_PIN_S})}[kind]
            anvil = {"short": IC_A, "dx": 0.0, "dz": round(ed / 2 + IC_GAP + ad / 2, 1), "b": 180.0, "width": W,
                     "morale": True}
            lanes.append(cp.lane(IC_B, IC_E, mode, "hold", gap_m=IC_GAP, fight_s=IC_FIGHT_S, max_s=IC_MAX_S, answer=False,
                                 kind=kind, b_side="right" if side > 0 else "left", a_dx=side * IC_DX,
                                 target2=IC_E, t2_place=(side * e2[0], e2[1]), t2_mode="attack_extra",
                                 extras=[anvil], t_attack_extra="run", a_morale=True, mem_morale_ours=True,
                                 morale_rows=True, place=place, **extra))
        out.append(lanes + [dict(clock)])
    return out


def _fwd(b):
    r = math.radians(b)
    return math.sin(r), math.cos(r)


def hill_battles():
    """The hill plan's battle (module doc): the HILL_SITES lanes on Kislev."""
    g = cp.TERRAIN
    tk, tm, _ = cp.UNITS[HILL_T]
    ak, am, _ = cp.UNITS[HILL_A]
    td, ad = cp.depth(tk, tm), cp.depth(ak, am)
    lanes = []
    for kind, (px, pz, b) in HILL_SITES.items():
        fx, fz = _fwd(b)
        far, near = HILL_GAP + (td + ad) / 2, HILL_STOP + (td + ad) / 2
        a_at = {"x": round(px + fx * far, 1), "z": round(pz + fz * far, 1), "b": (b + 180) % 360}
        t_at = {"x": px, "z": pz, "b": b}
        steps = [{"at_s": 2, "kind": "move", "dx": round(-fx * (far - near), 1), "dz": round(-fz * (far - near), 1),
                  "run": False}]
        lanes.append(cp.lane(HILL_A, HILL_T, "script", "hold", gap_m=HILL_GAP, fight_s=0, max_s=HILL_S, answer=False,
                             kind=kind, map=g["map"], catchment=g["catchment"], a_at=a_at, t_at=t_at, place=(px, pz),
                             steps=steps, t_morale=True, flank_rows=True, **FX))
    return [lanes]


def flankslot_battles(table=None):
    """The flankslot plan's battle (module doc): FLANK_LANES (flankslot2: FLANK2_LANES) on FLANK_PLACES, flat."""
    lanes = []
    for (kind, ((dx, dz, ab), steps, friends)), (px, pz) in zip((table or FLANK_LANES).items(), FLANK_PLACES):
        a_at = {"x": px + dx, "z": pz + dz, "b": ab}
        t_at = {"x": px, "z": pz, "b": 0.0}
        st = []
        for s in steps:
            if s[1] == "face":
                st.append({"at_s": s[0], "kind": "face", "bearing": s[2]})
            else:
                st.append({"at_s": s[0], "kind": "move", "dx": float(s[1]), "dz": float(s[2]), "run": bool(s[3])})
        extra = {}
        if friends:
            extra["extras"] = [{"short": FLANK_F, "dx": fx, "dz": 0.0, "b": 0.0, "width": cp.WIDTH_M, "side": "t"}
                               for fx in friends]
        lanes.append(cp.lane(FLANK_A, FLANK_T, "script", "hold", gap_m=round(math.hypot(dx, dz)), fight_s=0,
                             max_s=FLANK_S, answer=False, kind=kind, a_at=a_at, t_at=t_at, place=(px, pz), steps=st,
                             t_morale=True, flank_rows=True, **FX, **extra))
    return [lanes]


def lobfriend_battles():
    """The lobfriend plan's battle (module doc): LOB_LANES on the grid, flat."""
    lanes = []
    for kind, shooter, aw, target, gap, cover, ahead in LOB_LANES:
        extra = {}
        if cover:
            tk, tm, _ = cp.UNITS[target]
            ck, cm, _ = cp.UNITS[cover]
            td, cd = cp.depth(tk, tm), cp.depth(ck, cm)
            dz = td / 2 + gap - ahead - cd / 2           # from the target's centre, towards the shooter
            extra["extras"] = [{"short": cover, "dx": 0.0, "dz": round(dz, 1), "width": cp.WIDTH_M}]
        lanes.append(cp.lane(shooter, target, "shoot", "hold", gap_m=gap, fight_s=LOB_S, max_s=LOB_S, answer=False,
                             kind=kind, a_w=aw, **FX, **extra))
    return [_placed(lanes, 0)]


def forestwalk_battles():
    """The forestwalk plan's battle (module doc): FOREST_SITES shuttles of spearmen on Kislev, the targets parked."""
    g, gr = cp.TERRAIN, cp.GROUND
    lanes = []
    for k, (kind, (x0, z0, b, run)) in enumerate(FOREST_SITES.items()):
        fx, fz = _fwd(b)
        lanes.append(cp.lane("spear", "clanrat", "shuttle", "hold", gap_m=250, fight_s=FOREST_S, max_s=FOREST_S,
                             answer=False, kind=kind, map=g["map"], catchment=g["catchment"], shuttle_walk=not run,
                             a_at={"x": x0, "z": z0, "b": b},
                             p2={"x": round(x0 + gr["leg_m"] * fx, 1), "z": round(z0 + gr["leg_m"] * fz, 1)},
                             t_at={"x": -470.0 + 40 * k, "z": g["park_z"], "b": 180.0}, place=(x0, z0), **FX))
    return [lanes]


def _placed(base, k):
    """The lanes rotated by k on the grid's places; a pace lane runs towards the middle of the field."""
    out = []
    for spec, place in zip(cp.rotate(list(base), k), GRID):
        spec = dict(spec, place=place)
        if spec.get("kind") == "pace":
            dx = -PACE_M if place[0] > 0 else PACE_M
            spec["steps"] = [{"at_s": 1, "kind": "move", "dx": dx, "dz": 0, "run": True},
                             {"at_s": PACE_BACK_S, "kind": "move", "dx": 0, "dz": 0, "run": True}]
        out.append(spec)
    return out


def battles(plan):
    """[[lane spec]] of a new units' plan (game/probe/charge_probe.battles hands these over)."""
    if plan == "ratling":
        def r(kind, target, gap, cut):
            extra = {"damage": {"a": {"method": "kill", "share": cut}}} if cut else {}
            return cp.lane(RATLING, target, "shoot", "hold", gap_m=gap, fight_s=RATLING_S, max_s=RATLING_S, answer=False,
                           kind=kind, **FX, **extra)
        base = [r(*x) for x in RATLING_LANES]
        return [_placed(base, k) for k in (0, 3)]
    if plan == "ratlord":
        def rl(kind, gap, face):
            return cp.lane(RATLING, "general", "shoot", "hold" if face == "front" else "rear", gap_m=gap,
                           fight_s=RATLORD_S, max_s=RATLORD_S, answer=False, kind=kind, **FX)
        return [_placed([rl(*x)], 0) for x in RATLORD_LANES]
    if plan == "ratcover":
        # a lone man of ours under the gun with a unit of his side near him (the itS26_15 exam: our units near the
        # General took about half of the guns' bullets in the game, nearly all in the simulator); RATCOVER_LANES
        def rc(kind, where, gap):
            extra = {}
            if where:
                d = cp.depth(cp.UNITS[RATCOVER_UNIT][0], cp.UNITS[RATCOVER_UNIT][1])
                r = RATCOVER_R
                dx, dz = {"front": (0.0, gap + r + d / 2), "back": (0.0, -(gap + r + d / 2)),
                          "side": (gap + r + cp.WIDTH_M / 2, 0.0)}[where]
                extra["extras"] = [{"short": RATCOVER_UNIT, "dx": round(dx, 1), "dz": round(dz, 1), "b": 0,
                                    "width": cp.WIDTH_M, "side": "t"}]
            return cp.lane(RATLING, RATCOVER_TARGET, "shoot", "hold", gap_m=RATCOVER_GAP, fight_s=RATLORD_S,
                           max_s=RATLORD_S, answer=False, kind=kind, **FX, **extra)
        base = [rc(*x) for x in RATCOVER_LANES]
        return [_placed(base, k) for k in (0, 4)]
    if plan == "losfriend":
        def lf(kind, shooter, cover, target):
            extra = {}
            if cover:
                tk, tm, _ = cp.UNITS[target]
                ck, cm, _ = cp.UNITS[cover]
                td, cd = cp.depth(tk, tm), cp.depth(ck, cm)
                dz = td / 2 + LOSFRIEND_GAP - LOSFRIEND_COVER - cd / 2      # from the target's centre, towards the shooter
                extra["extras"] = [{"short": cover, "dx": 0.0, "dz": round(dz, 1), "width": cp.WIDTH_M}]
            return cp.lane(shooter, target, "shoot", "hold", gap_m=LOSFRIEND_GAP, fight_s=LOSFRIEND_S,
                           max_s=LOSFRIEND_S, answer=False, kind=kind, **FX, **extra)
        return [_placed([lf(*x) for x in LOSFRIEND_LANES], 0)]
    if plan == "losmove":
        lanes = []
        for kind, shooter, target, order, pace in LOSMOVE_LANES:
            ak, am, _ = cp.UNITS[shooter]
            extra = {"a_free": True} if order == "free" else {}
            lanes.append(cp.lane(shooter, target, "shoot" if order == "shoot" else "hold", "push", gap_m=LOSMOVE_GAP,
                                 fight_s=10, max_s=LOSMOVE_S, answer=False, kind=kind,
                                 push_m=round(LOSMOVE_GAP + cp.depth(ak, am) + 20, 1), push_run=pace == "run",
                                 **FX, **extra))
        return [_placed(lanes, 0)]
    if plan == "snipe":
        return snipe_battles()
    if plan == "hill":
        return hill_battles()
    if plan == "flankslot":
        return flankslot_battles()
    if plan == "flankslot2":
        return flankslot_battles(FLANK2_LANES)
    if plan == "lobfriend":
        return lobfriend_battles()
    if plan == "forestwalk":
        return forestwalk_battles()
    if plan == "width":
        return width_battles()
    if plan == "reissue":
        return reissue_battles()
    if plan == "counter":
        return counter_battles()
    if plan == "flank2v1":
        return flank2v1_battles()
    if plan == "counterfar":
        return counterfar_battles()
    if plan == "intercept":
        return intercept_battles()
    if plan == "losrange":
        flat = []
        for kind, shooter, target, gap, face in LOSRANGE_LANES:
            flat.append(cp.lane(shooter, target, "shoot", "hold", gap_m=gap, fight_s=LOSRANGE_S, max_s=LOSRANGE_S,
                                answer=False, kind=kind, back=face == "back", **FX))
        out = []
        for spec in _placed(flat, 0):
            if spec.pop("back"):
                # its back to the target: the layout's places (charge_probe M.layout), the shooter turned 180
                px, pz = spec["place"]
                ak, am, _ = cp.UNITS[spec["attacker"]]
                tk, tm, _ = cp.UNITS[spec["target"]]
                ad, td = cp.depth(ak, am), cp.depth(tk, tm)
                spec["a_at"] = {"x": float(px), "z": round(pz + spec["gap_m"] + ad / 2, 1), "b": 0.0}
                spec["t_at"] = {"x": float(px), "z": round(pz - td / 2, 1), "b": 0.0}
            out.append(spec)
        g = cp.TERRAIN
        maps = []
        for kind, shooter, (ax, az, ab), target, (tx, tz, tb) in LOSRANGE_MAP:
            maps.append([cp.lane(shooter, target, "shoot", "hold", gap_m=round(math.hypot(tx - ax, tz - az)),
                                 fight_s=LOSRANGE_S, max_s=LOSRANGE_S, answer=False, kind=kind, map=g["map"],
                                 catchment=g["catchment"], a_at={"x": ax, "z": az, "b": ab},
                                 t_at={"x": tx, "z": tz, "b": tb}, place=(ax, az), **FX)])
        return [out] + maps
    if plan == "ratlone":
        return [_placed([cp.lane(a, t, "shoot", "hold", gap_m=gap, fight_s=RATLONE_S, max_s=RATLONE_S, answer=False,
                                 kind=kind, **FX) for kind, a, t, gap in RATLONE_LANES], 0)]
    if plan == "ratform":
        return [_placed([cp.lane(a, t, "shoot", mode, gap_m=gap, fight_s=RATFORM_S, max_s=RATFORM_S, answer=False,
                                 kind=kind, **FX) for kind, a, t, gap, mode in RATFORM_LANES], 0)]
    if plan == "ratshape":
        return [_placed([cp.lane(RATLING, t, "shoot", "hold", gap_m=100, fight_s=RATSHAPE_S, max_s=RATSHAPE_S,
                                 answer=False, kind=kind, t_w=w, **FX) for kind, t, w in RATSHAPE_LANES], 0)]
    if plan == "suppress":
        def s(kind, walk, fire):
            return cp.lane("swords", RATLING, "attack_walk" if walk else "attack_run", "skirmish" if fire else "hold",
                           gap_m=SUPPRESS_GAP, fight_s=15, max_s=200, answer=False, kind=kind, t_skirmish=False, **FX)
        base = [s("walk_fire", True, True), s("walk_quiet", True, False), s("run_fire", False, True),
                s("run_quiet", False, False)]
        return [_placed(base, k) for k in (0, 2)]
    if plan == "hitslow":
        # (12.10.2026) the suppress plan's walk / run with clanrats without shields walking / running at handgunners
        # (flat fire, no contact effect) and crossbowmen (arcing): do hits alone slow a unit as the gun's do beyond its
        # x0.7 (the gun's walk 0.91 m/s, run 1.29 against 1.05 / 2.10 with x0.7)? quiet: the handgunners never fire
        def hs(kind, shooter, walk, fire=True):
            return cp.lane(HITSLOW_RUNNER, shooter, "attack_walk" if walk else "attack_run",
                           "skirmish" if fire else "hold", gap_m=SUPPRESS_GAP, fight_s=15, max_s=200, answer=False,
                           kind=kind, t_skirmish=False, **FX)
        base = [hs(f"{gait}_{s}", s, gait == "walk") for s in HITSLOW_SHOOTERS for gait in ("walk", "run")]
        base += [hs(f"{gait}_quiet", "hgun", gait == "walk", False) for gait in ("walk", "run")]
        return [_placed(base, k) for k in (0, 3)]
    if plan == "suppress2":
        def s2(kind):
            walk, stop = SUPPRESS2_STOP[kind]
            return cp.lane("swords", RATLING, "attack_walk" if walk else "attack_run", "skirmish", gap_m=SUPPRESS_GAP,
                           fight_s=5, max_s=150, answer=False, kind=kind, t_skirmish=False, t_stop_fire_s=stop, **FX)
        return [_placed([s2(k) for k in SUPPRESS2_STOP] * 2, 0)]
    if plan == "censer":
        def c(kind, a, t, mode="attack_run", tmode="stand", **kw):
            return cp.lane(a, t, mode, tmode, **dict(dict(gap_m=30, fight_s=60, kind=kind, **FX), **kw))
        leave = dict(answer=False, recharge_after_s=20, back_m=60, recharge_max_s=10 ** 6, t_at_leave="halt")
        rout = dict(t_morale=True, at_rout="halt", after_rout_s=10)
        base = [c("censer", CENSER, "swords"), c("monks", MONKS, "swords"), c("censer_shield", CENSER, "spearsh"),
                c("held_censer", "swords", CENSER, tmode="hold", answer=False),
                c("censer_leave", CENSER, "swords", "withdraw", "both", **leave),
                c("censer_morale", CENSER, "swords", **rout), c("monks_morale", MONKS, "swords", **rout),
                c("halb", HALB, "svsh")]
        return [_placed(base, k) for k in (0, 4)]
    if plan == "wolfrats":
        def w(kind, a, t, mode="attack_run", tmode="stand", **kw):
            return cp.lane(a, t, mode, tmode, **dict(dict(gap_m=40, fight_s=40, kind=kind, **FX), **kw))
        base = [w("pace", WOLF, "swords", "script", "hold", gap_m=60, answer=False, men_all_s=45, max_s=45),
                w("charge120", WOLF, "swords", gap_m=120),
                w("braced", WOLF, "spear", tmode="hold", answer=False),
                w("rear", WOLF, "swords", tmode="rear", answer=False),
                w("general", WOLF, "general"),
                w("recharge", WOLF, "swords", "recharge", "hold", answer=False, recharge_after_s=15, back_m=30,
                  pause_s=3, recharge_max_s=30, after2_s=20, max_s=100),
                w("chase", WOLF, "swords", fight_s=200, **CHASE),
                w("chase_clanrat", "clanrat", "swords", fight_s=200, **CHASE)]
        return [_placed(base, k) for k in (0, 4)]
    if plan == "wolfpoison":
        def p(kind, a, leave=None):
            if leave is None:
                return cp.lane(a, "swords", "attack_run", "stand", gap_m=30, fight_s=40, kind=kind, **FX)
            extra = dict(t_at_leave="halt") if leave == "halt" else {}
            return cp.lane(a, "swords", "withdraw", "both", gap_m=30, fight_s=55, answer=False, recharge_after_s=15,
                           back_m=80, recharge_max_s=10 ** 6, kind=kind, **FX, **extra)
        base = [p("poison", WOLFP), p("plain", WOLF), p("poison_leave_halt", WOLFP, "halt"),
                p("poison_leave_chase", WOLFP, "chase"), p("plain_leave_chase", WOLF, "chase")]
        return [_placed(base, k) for k in (0, 2)]
    raise ValueError(plan)


def snipe_battles():
    """The snipe plan's two battles (module doc): the four SNIPE_SITES lanes with the long rifles, then the jezzails."""
    g = cp.TERRAIN
    out = []
    for shooter, target in SNIPE_SHOOTERS:
        ak, am, _ = cp.UNITS[shooter]
        tk, tm, _ = cp.UNITS[target]
        ad, td = cp.depth(ak, am), cp.depth(tk, tm)
        lanes = []
        for kind, (px, pz, b, _) in SNIPE_SITES.items():
            r = math.radians(b)
            fx, fz = math.sin(r), math.cos(r)
            gap = SNIPE_APPROACH_GAP if kind == "approach" else SNIPE_GAP
            a_at = {"x": round(px - fx * ad / 2, 1), "z": round(pz - fz * ad / 2, 1), "b": b}
            t_at = {"x": round(px + fx * (gap + td / 2), 1), "z": round(pz + fz * (gap + td / 2), 1),
                    "b": (b + 180) % 360}
            common = dict(answer=False, kind=kind, map=g["map"], catchment=g["catchment"], a_at=a_at, t_at=t_at,
                          place=(a_at["x"], a_at["z"]), stealth_rows="both", **FX)
            if kind == "approach":
                # the move's goal is the formation's front centre (the cavcheck plan's recharge): SNIPE_STOP_M off
                push_to = {"x": round(px + fx * SNIPE_STOP_M, 1), "z": round(pz + fz * SNIPE_STOP_M, 1)}
                lanes.append(cp.lane(shooter, target, "hold", "push", gap_m=gap, fight_s=10, max_s=SNIPE_APPROACH_S,
                                     a_free=True, push_to=push_to, push_run=False, **common))
            else:
                lanes.append(cp.lane(shooter, target, "shoot", "hold", gap_m=gap, fight_s=SNIPE_S, max_s=SNIPE_S,
                                     a_go_s=SNIPE_GO_S, **common))
        out.append(lanes)
    return out


# ---------------------------------------------------------------- the recordings

def _t(lane):
    return np.array([x[0] for x in lane["samples"]])


def _row(sample, who):
    return sample[{"a": 1, "tg": 2}[who]] or {}


def _fx(lane, who):
    """[set of phase keys] a sample."""
    return [set(filter(None, (_row(x, who).get("fx") or "").split(","))) for x in lane["samples"]]


def _card(lane, who, key):
    return np.array([np.nan if (_row(x, who).get("cs") or {}).get(key) is None else float(_row(x, who)["cs"][key])
                     for x in lane["samples"]])


def _speed(lane, who, window_s=2.0):
    """The unit's speed (m/s) over the window_s before each sample (from its centre's places: the engine moves a unit in
    steps, a 0.5 s difference is noise)."""
    t = _t(lane)
    x, z = cp.series(lane["samples"], who, "x"), cp.series(lane["samples"], who, "z")
    v = np.full(len(t), np.nan)
    ok = np.isfinite(x) & np.isfinite(z)
    if ok.sum() < 2:
        return v
    for i, ti in enumerate(t):
        if ti - window_s >= t[ok][0] and ok[i]:
            x0, z0 = np.interp(ti - window_s, t[ok], x[ok]), np.interp(ti - window_s, t[ok], z[ok])
            v[i] = float(np.hypot(x[i] - x0, z[i] - z0) / window_s)
    return v


def _rate(t, v, lo, hi):
    """v lost a second from lo to hi (NaN outside the recording)."""
    a, b = cp.at(t, v, lo), cp.at(t, v, hi)
    return (a - b) / (hi - lo) if np.isfinite(a) and np.isfinite(b) and hi > lo else np.nan


def _phase_t(lane, name):
    return next((p["t"] / 1000 for p in lane.get("phases", []) if p["phase"] == name), None)


def _med(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else None


def effects(lane, who, since=None):
    """{phase key: (first s, last s)} the unit carried (from the lane's go; since: only keys not on before it)."""
    t, sets = _t(lane), _fx(lane, who)
    before = set().union(*[s for ti, s in zip(t, sets) if since is not None and ti < since - 1e-6]) if since else set()
    out = {}
    for ti, s in zip(t, sets):
        for k in s - before:
            out[k] = (out.get(k, (float(ti),))[0], float(ti))
    return out


def _new_effect(lane, who, since, part):
    """(key, first s, last s) of the first phase key containing part (or any new one, part None) the unit got after
    `since`, else None."""
    fx = effects(lane, who, since)
    keys = sorted((k for k in fx if part is None or part in k), key=lambda k: fx[k][0])
    return (keys[0],) + fx[keys[0]] if keys else None


def _card_change(lane, who, key, c, lo=0.0, hi=30.0):
    """(card value before c, the extreme after it within lo..hi s: the min, or the max when it went up) or None."""
    t, v = _t(lane), _card(lane, who, key)
    pre, post = v[(t < c) & np.isfinite(v)], v[(t >= c + lo) & (t <= c + hi) & np.isfinite(v)]
    if not len(pre) or not len(post):
        return None
    base = float(pre[-1])
    lo_v, hi_v = float(post.min()), float(post.max())
    return base, (hi_v if base - lo_v < hi_v - base else lo_v)


def _flight_s(spec, d):
    """A projectile's flight, s: the low arc's d / (v cos theta), theta = asin(g d / v^2) / 2 (the shooter's passport
    muzzle velocity); 0 without one."""
    p = cp.passport(spec.get("a_key") or "") or {}
    v = float((p.get("missile") or {}).get("muzzle_velocity") or 0)
    if not v or not d:
        return 0.0
    th = 0.5 * math.asin(min(1.0, d * 9.81 / (v * v)))
    return d / (v * math.cos(th))


def ratling_measure(lane):
    spec, s = lane["spec"], lane["samples"]
    t = _t(lane)
    out = {"kind": spec.get("kind"), "run": lane["run"]}
    ammo, men = cp.series(s, "a", "ammo"), cp.series(s, "a", "men")
    hp, tmen = cp.series(s, "tg", "hp"), cp.series(s, "tg", "men")
    ok = np.isfinite(ammo)
    if not ok.any():
        return out
    out["ammo0"], out["men0"] = float(ammo[ok][0]), float(men[0])
    out["ammo_per_man"] = out["ammo0"] / out["men0"] if out["men0"] else None
    a0, t0 = _row(s[0], "a"), _row(s[0], "tg")
    out["d0_m"] = float(np.hypot(a0["x"] - t0["x"], a0["z"] - t0["z"])) if None not in (a0.get("x"), t0.get("x")) else None
    drop = -np.diff(np.where(ok, ammo, np.nan))           # bullets fired in each half second
    fired = np.nan_to_num(drop) > 0
    if not fired.any():
        return out
    ft = t[1:][fired]
    first, last = float(ft[0]), float(ft[-1])
    bullets = float(np.nansum(drop[fired]))
    out.update(first_shot_s=first, last_shot_s=last, bullets=bullets, fire_s=last - first + 0.5)
    out["bullets_per_s"] = bullets / out["fire_s"]
    out["bullets_per_man_s"] = out["bullets_per_s"] / out["men0"] if out["men0"] else None
    empty = t[ok & (ammo <= 0)]
    out["empty_s"] = float(empty[0]) if len(empty) else None
    # bursts: fired samples with gaps of 1+ s between them
    starts, sizes, ends = [ft[0]], [0.0], [ft[0]]
    for ti, d in zip(ft, drop[fired]):
        if ti - ends[-1] > 1.0 + 1e-6:
            starts.append(ti), sizes.append(0.0), ends.append(ti)
        sizes[-1] += float(d)
        ends[-1] = ti
    out["bursts"] = len(starts)
    out["burst_bullets"] = float(np.median(sizes))
    out["burst_s"] = float(np.median(np.array(ends) - np.array(starts) + 0.5))
    out["burst_every_s"] = float(np.median(np.diff(starts))) if len(starts) > 1 else None
    # HP per bullet counts only the bullets whose hits the recording holds: a projectile lands its flight time after it
    # leaves (the low arc's d / (v cos theta): a bolt 2.5 s at 108 m, a bullet 1.1 s), so the ones loosed in the lane's
    # last flight time + the 0.5 s readout land after it ends (the ratlone crossbows' last volley: 90 of 447 bolts, their
    # HP lost - 1.46 of the game was this). The simulator's hits are at once (run 'sim'): its HP to the same bullets.
    flight = _flight_s(spec, out["d0_m"])
    cut = float(t[-1]) - flight - 0.5
    keep = fired & (t[1:] <= cut + 1e-6)
    if keep.any():
        last_k = float(t[1:][keep][-1])
        n_keep = float(np.nansum(drop[keep]))
    else:
        last_k, n_keep = last, bullets
    # (a bullet loosed in the sample ending at last_k lands by last_k + flight: the health then holds its hit and
    # none of the later ones')
    stop = min(float(t[-1]), last_k + (0.0 if lane.get("run") == "sim" else flight))
    lost = cp.at(t, hp, first - 0.5) - cp.at(t, hp, stop)
    out["tg_hp_lost"] = lost
    out["tg_men_lost"] = cp.at(t, tmen, first - 0.5) - cp.at(t, tmen, stop)
    out["hp_per_bullet"] = lost / n_keep if n_keep else None
    out["tg_hp_per_s"] = lost / (stop - first + 0.5)
    out["dd_end"] = _row(s[-1], "a").get("dd")
    out["tg_dead"] = bool(np.nanmin(tmen) == 0) if np.isfinite(tmen).any() else None
    sup = _new_effect(lane, "tg", first, None)
    if sup:
        out["tg_fx"], out["tg_fx_first_s"] = sup[0], sup[1] - first
    spd = _card_change(lane, "tg", "scalar_speed", first, 0, RATLING_S)
    if spd:
        out["tg_card_speed"], out["tg_card_speed_fx"] = spd
    return out


def suppress_measure(lane):
    spec, s = lane["spec"], lane["samples"]
    t = _t(lane)
    out = {"kind": spec.get("kind"), "run": lane["run"]}
    c = lane["contacts"].get(1)
    out["contact_s"] = c
    end = c if c is not None else float(t[-1])
    d = np.hypot(cp.series(s, "a", "x") - cp.series(s, "tg", "x"), cp.series(s, "a", "z") - cp.series(s, "tg", "z"))
    v = _speed(lane, "a")
    on = np.array([any("suppress" in k for k in fx) for fx in _fx(lane, "a")])
    way = (t >= 4) & (t < end) & (d > 50)            # on the way, before the last 30 m sprint (the charge)
    out["speed_way"] = _med(v[way])
    out["speed_way_fx"] = _med(v[way & on])
    out["speed_way_nofx"] = _med(v[way & ~on])
    out["fx_share_way"] = float(on[way].mean()) if way.any() else None
    keys = set().union(*_fx(lane, "a"))
    out["a_fx"] = ",".join(sorted(keys)) or None
    spd = _card(lane, "a", "scalar_speed")
    if np.isfinite(spd).any():
        out["card_speed0"] = float(spd[np.isfinite(spd)][0])
        out["card_speed_fx"] = _med(spd[on])
        out["card_speed_min"] = float(np.nanmin(spd))
    ammo = cp.series(s, "tg", "ammo")
    drop = np.nan_to_num(np.diff(ammo)) < 0
    if drop.any():
        last = float(t[1:][drop][-1])
        out["shots"] = float(np.nansum(-np.diff(ammo)[drop]))
        out["last_shot_s"] = last
        if on.any():
            # the phase's last sample after the gun's last shot (negative: gone before it); still on at the end -> no end
            last_on = float(t[on].max())
            out["fx_after_last_shot_s"] = last_on - last
            out["fx_ended"] = last_on < float(t[-1]) - 1e-6
    hp = cp.series(s, "a", "hp")
    out["a_hp_lost_way"] = (cp.at(t, hp, 0.0) - cp.at(t, hp, end)) / cp.at(t, hp, 0.0) if np.isfinite(hp).any() else None
    return out


WIN = ((0, 5), (5, 15), (15, 30), (30, 60))


def _fight(lane, out, c, stop=None):
    """HP lost a second by both sides in WIN after the contact c (to stop)."""
    t = _t(lane)
    stop = float(t[-1]) if stop is None else stop
    for who in ("tg", "a"):
        hp = cp.series(lane["samples"], who, "hp")
        for lo, hi in WIN:
            if c + hi <= stop + 1e-6:
                out[f"{who}_hp_{lo}_{hi}"] = _rate(t, hp, c + lo, c + hi)


def censer_measure(lane):
    spec = lane["spec"]
    out = {"kind": spec.get("kind"), "run": lane["run"]}
    c = lane["contacts"].get(1)
    out["contact_s"] = c
    if c is None:
        return out
    t = _t(lane)
    leave = _phase_t(lane, "out")
    _fight(lane, out, c, leave)
    for who in ("tg", "a"):
        fx = effects(lane, who, c)
        out[f"{who}_fx_new"] = ",".join(sorted(fx)) or None
    hit = _new_effect(lane, "tg", c, "censer") or _new_effect(lane, "tg", c, None)
    if hit:
        out["tg_fx"], out["tg_fx_first_s"] = hit[0], hit[1] - c
    for who, key in (("tg", "stat_morale"), ("a", "stat_melee_attack"), ("a", "stat_weapon_damage"),
                     ("tg", "stat_melee_attack")):
        ch = _card_change(lane, who, key, c, 0, 30)
        if ch:
            out[f"{who}_{key}"], out[f"{who}_{key}_fight"] = ch
    mp = cp.series(lane["samples"], "tg", "mp")
    sel = (t >= c + 5) & (t <= c + 30)
    out["tg_mp_5_30"] = _med(mp[sel])
    for dt in (5, 10):
        out[f"tg_mp_{dt}"] = cp.at(t, mp, c + dt)
    rout = _phase_t(lane, "rout")
    out["rout_s"] = rout - c if rout is not None else None
    if leave is not None:
        m = cp.series(lane["samples"], "tg", "m")
        inside = (t >= leave) & (m > 0.5)
        clear = float(t[inside].max()) if inside.any() else leave
        out["leave_clear_s"] = clear - leave
        if hit and hit[2] > clear:
            # the phase's last sample after the last melee sample; still on at the lane's end -> no end seen
            out["tg_fx_after_clear_s"] = hit[2] - clear
            out["tg_fx_still_on"] = hit[2] >= float(t[-1]) - 1e-6
    return out


def wolfrats_measure(lane):
    spec = lane["spec"]
    kind = spec.get("kind")
    out = {"kind": kind, "run": lane["run"]}
    t, v = _t(lane), _speed(lane, "a")
    if kind == "pace":
        # per leg (the order at lo): the cruise speed (the median of the samples above half the leg's top: the centre
        # moves in steps, a 2 s peak is noise), the seconds to 90 % of it, the leg's time to a stop and its metres
        x, z = cp.series(lane["samples"], "a", "x"), cp.series(lane["samples"], "a", "z")
        for leg, (lo, hi) in enumerate(((1, PACE_BACK_S), (PACE_BACK_S, float(t[-1]) + 1)), 1):
            sel = (t >= lo) & (t < hi) & np.isfinite(v)
            if sel.sum() < 3:
                continue
            tv, vv = t[sel], v[sel]
            cruise = float(np.median(vv[vv > 0.5 * vv.max()]))
            reach = tv[vv >= 0.9 * cruise]
            out[f"leg{leg}_cruise"] = cruise
            out[f"leg{leg}_to90_s"] = float(reach[0]) - lo if len(reach) else None
            v05 = _speed(lane, "a", 0.5)[sel]          # a stop: no step in the half second
            moving = tv[v05 > 0.5]
            if len(moving):
                end = float(moving[-1])
                out[f"leg{leg}_s"] = end - lo
                out[f"leg{leg}_m"] = float(np.hypot(cp.at(t, x, end) - cp.at(t, x, lo), cp.at(t, z, end) - cp.at(t, z, lo)))
        return out
    if kind == "recharge":
        r = cp.recontact_measure(lane)
        for k in ("first_tg_0_3", "first_tg_10_20", "re_tg_0_3", "re_tg_10_20", "first_a_0_3", "re_a_0_3", "out_s"):
            out[k] = r.get(k)
        return out
    m = cp.measure(lane)
    for k in ("contact_s", "speed_peak30", "speed_last30", "speed_30_60", "approach_s", "tg_hp_0_1", "tg_hp_0_2",
              "tg_hp_0_5", "tg_hp_5_15", "tg_hp_steady", "a_hp_0_5", "a_hp_steady", "a_near_0_5", "a_near_5_15"):
        out[k] = m.get(k)
    c = m.get("contact_s")
    if c is not None:
        before = (t < c) & np.isfinite(v)
        out["speed_peak_way"] = float(v[before].max()) if before.any() else None
    if kind and kind.startswith("chase"):
        rout = _phase_t(lane, "rout")
        out["rout_s"] = rout - c if rout is not None and c is not None else None
        if rout is not None:
            hp, men = cp.series(lane["samples"], "tg", "hp"), cp.series(lane["samples"], "tg", "men")
            stop = min(float(t[-1]), rout + 45)
            out["chase_s"] = stop - rout
            out["chase_tg_hp_per_s"] = _rate(t, hp, rout, stop)
            out["chase_tg_men_lost"] = cp.at(t, men, rout) - cp.at(t, men, stop)
            out["chase_tg_men_left"] = cp.at(t, men, stop)
    return out


def wolfpoison_measure(lane):
    spec = lane["spec"]
    out = {"kind": spec.get("kind"), "run": lane["run"]}
    c = lane["contacts"].get(1)
    out["contact_s"] = c
    if c is None:
        return out
    t = _t(lane)
    leave = _phase_t(lane, "out")
    _fight(lane, out, c, leave)
    fx = effects(lane, "tg", c)
    out["tg_fx_new"] = ",".join(sorted(fx)) or None
    hit = _new_effect(lane, "tg", c, "poison")
    if hit:
        out["tg_fx"], out["tg_fx_first_s"] = hit[0], hit[1] - c
    for key in ("scalar_speed", "stat_weapon_damage", "stat_melee_attack", "stat_missile_damage_over_time"):
        ch = _card_change(lane, "tg", key, c, 0, 15)
        if ch:
            out[f"tg_{key}"], out[f"tg_{key}_fight"] = ch
    if leave is not None:
        m = cp.series(lane["samples"], "tg", "m")
        inside = (t >= leave) & (m > 0.5)
        clear = float(t[inside].max()) if inside.any() else leave
        out["leave_clear_s"] = clear - leave
        v = _speed(lane, "tg")
        a_m = cp.series(lane["samples"], "a", "m")
        chase = (t >= clear + 2) & (t <= clear + 12) & ~(m > 0.5) & ~(a_m > 0.5)
        on = np.array([any("poison" in k for k in s) for s in _fx(lane, "tg")])
        out["tg_speed_chase"] = _med(v[chase])
        out["tg_speed_chase_fx"] = _med(v[chase & on])
        out["tg_speed_chase_nofx"] = _med(v[chase & ~on])
        if hit and hit[2] > clear:
            out["tg_fx_after_clear_s"] = hit[2] - clear
            out["tg_fx_still_on"] = hit[2] >= float(t[-1]) - 1e-6
    return out


def losfriend_measure(lane):
    """ratling_measure's shots, and the shooter's way: its place along the lane (z towards the target: -) and across at
    the go, at the first shot and at the end; when it last moved (> 0.5 m in a half second); its offset across from the
    cover's centre at the first shot (the cover's half width: is the line clear?)."""
    out = ratling_measure(lane)
    s = lane["samples"]
    t = _t(lane)
    x, z = cp.series(s, "a", "x"), cp.series(s, "a", "z")
    ok = np.isfinite(x) & np.isfinite(z)
    if not ok.any():
        return out
    x0, z0 = float(x[ok][0]), float(z[ok][0])
    fwd = z0 - z                                          # metres towards the target (the shooter faces -z)
    side = x - x0
    out["fwd_end_m"], out["side_end_m"] = float(fwd[ok][-1]), float(side[ok][-1])
    out["fwd_max_m"] = float(np.nanmax(fwd))
    step = np.hypot(np.diff(x), np.diff(z))
    moving = np.nan_to_num(step) > 0.5
    out["moved_until_s"] = float(t[1:][moving][-1]) if moving.any() else 0.0
    out["started_s"] = float(t[1:][moving][0]) if moving.any() else None
    if out.get("first_shot_s") is not None:
        k = int(np.searchsorted(t, out["first_shot_s"]))
        k = min(k, len(t) - 1)
        out["fwd_at_shot_m"], out["side_at_shot_m"] = float(fwd[k]), float(side[k])
    ex = lane.get("extras") or []
    if ex and out.get("first_shot_s") is not None:
        # the cover's HP lost while the shooter fired (its own bullets through / into it?)
        te = np.array([tt for tt, _ in ex])
        he = np.array([np.nan if not rows or rows[0].get("hp") is None else float(rows[0]["hp"]) for _, rows in ex])
        out["cover_hp_lost"] = cp.at(te, he, out["first_shot_s"] - 0.5) - cp.at(te, he, float(te[-1]))
    return out


def losmove_measure(lane):
    """losrange's, and the shots before the contact (the shooter's first melee sample), the distance at the first shot,
    the contact's time."""
    out = losrange_measure(lane)
    s = lane["samples"]
    t = _t(lane)
    m = cp.series(s, "a", "m")
    ammo = cp.series(s, "a", "ammo")
    hit = np.nan_to_num(m) > 0
    out["contact_s"] = float(t[hit][0]) if hit.any() else None
    ok = np.isfinite(ammo)
    if ok.any():
        end = out["contact_s"] if out["contact_s"] is not None else float(t[-1])
        before = ok & (t <= end)
        out["shots_before_contact"] = float(ammo[ok][0] - ammo[before][-1]) if before.any() else 0.0
    if out.get("first_shot_s") is None:
        out["first_shot_s"] = None
    return out


def losrange_measure(lane):
    """losfriend's, and the distance between the centres (go / first shot / end / least), the way walked."""
    out = losfriend_measure(lane)
    s = lane["samples"]
    t = _t(lane)
    ax, az = cp.series(s, "a", "x"), cp.series(s, "a", "z")
    tx, tz = cp.series(s, "tg", "x"), cp.series(s, "tg", "z")
    d = np.hypot(ax - tx, az - tz)
    ok = np.isfinite(d)
    if not ok.any():
        return out
    out["dist0_m"], out["dist_end_m"], out["dist_min_m"] = float(d[ok][0]), float(d[ok][-1]), float(np.nanmin(d))
    out["walked_m"] = float(np.nansum(np.hypot(np.diff(ax), np.diff(az))))
    if out.get("first_shot_s") is not None:
        k = min(int(np.searchsorted(t, out["first_shot_s"])), len(t) - 1)
        out["dist_at_shot_m"] = float(d[k])
    b = cp.series(s, "a", "b")
    if np.isfinite(b).any():
        out["b0"], out["b_end"] = float(b[np.isfinite(b)][0]), float(b[np.isfinite(b)][-1])
    return out


def _gap_edges(lane):
    """[samples] the gap between the shooter's and the target's blocks along the line between their centres, m (the
    centres' distance less the half depths: the approach lane's head-on walk)."""
    s, spec = lane["samples"], lane["spec"]
    d = np.hypot(cp.series(s, "a", "x") - cp.series(s, "tg", "x"), cp.series(s, "a", "z") - cp.series(s, "tg", "z"))
    return d - (float(spec.get("a_depth") or 0) + float(spec.get("t_depth") or 0)) / 2


def snipe_measure(lane):
    """ratling_measure's shots (bullets, HP per bullet, the target's HP a second) and the shooter's hiding: hidden at
    the go and at the order (a_go_s), the share of the samples hidden / seen by the enemy from the first shot to the
    lane's end, its first reveal after the first shot; the approach lane: the gap (edge to edge) when the enemy first saw
    it (v) and when it lost its hiding, its hid / v at 100 / 60 / 40 m, the shots before the enemy saw it."""
    out = ratling_measure(lane)
    s, spec = lane["samples"], lane["spec"]
    t = _t(lane)
    hid, v = cp.series(s, "a", "hid"), cp.series(s, "a", "v")
    if not np.isfinite(hid).any():
        return out
    hid_b, v_b = np.nan_to_num(hid) > 0, np.nan_to_num(v) > 0
    out["hid_go"] = float(hid_b[0])
    go = float(spec.get("a_go_s") or 0)
    k_go = min(int(np.searchsorted(t, go)), len(t) - 1)
    out["hid_order"] = float(hid_b[k_go])
    first = out.get("first_shot_s")
    if first is not None:
        after = t >= first - 1e-6
        out["hid_firing_share"] = float(hid_b[after].mean())
        out["seen_firing_share"] = float(v_b[after].mean())
        off = after & ~hid_b
        out["revealed_s"] = float(t[off][0] - first) if off.any() else None
    else:
        out["hid_share"] = float(hid_b.mean())
    if spec.get("target_mode") == "push":
        gap = _gap_edges(lane)
        # seen by the enemy after it first hid (before: not yet hidden - its 5 s in the woods after the placement)
        k0 = int(np.argmax(hid_b)) if hid_b.any() else 0
        seen = v_b & np.isfinite(gap) & (np.arange(len(t)) >= k0)
        out["gap_seen_m"] = float(gap[seen][0]) if seen.any() else None
        lost = ~hid_b & np.isfinite(gap) & (np.arange(len(t)) >= k0)
        out["gap_unhid_m"] = float(gap[lost][0]) if lost.any() else None
        out["gap_min_m"] = float(np.nanmin(gap))
        for m in (100, 60, 40):
            k = np.flatnonzero(np.nan_to_num(gap, nan=1e9) <= m)
            out[f"hid_{m}"] = float(hid_b[k[0]]) if len(k) else None
            out[f"v_{m}"] = float(v_b[k[0]]) if len(k) else None
        ammo = cp.series(s, "a", "ammo")
        ok = np.isfinite(ammo)
        if ok.any():
            end = float(t[seen][0]) if seen.any() else float(t[-1])
            before = ok & (t <= end)
            out["shots_unseen"] = float(ammo[ok][0] - ammo[before][-1]) if before.any() else 0.0
    return out


def ratcover_measure(lane):
    """ratling_measure's numbers for the lone man, and the HP his cover (the lane's extra unit) lost per bullet over the
    same span (e_hp_per_bullet)."""
    out = ratling_measure(lane)
    ex = lane.get("extras") or []
    if not ex or not out.get("bullets") or out.get("first_shot_s") is None:
        return out
    te = np.array([t for t, _ in ex])
    he = np.array([np.nan if not rows or rows[0].get("hp") is None else float(rows[0]["hp"]) for _, rows in ex])
    first, last = out["first_shot_s"], out["last_shot_s"]
    stop = min(float(te[-1]), last + 2.0)
    lost = cp.at(te, he, first - 0.5) - cp.at(te, he, stop)
    out["e_hp_lost"] = lost
    out["e_hp_per_bullet"] = lost / out["bullets"]
    return out


def forestwalk_measure(lane):
    """The shuttle's cruise: the median of the 2 s speeds over each leg from 4 s after its start to 1 s before the next
    (the turn and the start left out; legs from the entry's 'leg' phases, the first from the go at 0), their mean over
    the legs, the legs done and the mean time of a whole leg (start to the next start)."""
    t = _t(lane)
    v = _speed(lane, "a")
    starts = [0.0] + [p["t"] / 1000 for p in lane.get("phases", []) if p["phase"] == "leg"]
    ends = starts[1:] + [float(t[-1])]
    cruise, legs = [], []
    for a, b in zip(starts, ends):
        m = (t >= a + 4) & (t <= b - 1) & np.isfinite(v)
        if m.sum() >= 3:
            cruise.append(float(np.median(v[m])))
        if b > a and b < float(t[-1]):
            legs.append(b - a)
    return {"kind": lane["spec"]["kind"], "run": lane.get("run"), "cruise": float(np.mean(cruise)) if cruise else None,
            "cruise_first": cruise[0] if cruise else None, "legs": float(len(starts) - 1),
            "leg_s": float(np.mean(legs)) if legs else None}


WIDTH_WIN = ((0, 5), (5, 15), (15, 30), (30, 60))


def width_measure(lane):
    """The width plan: the first contact (either unit's melee flag), each side's share of its starting health lost in
    WIDTH_WIN windows after it and over 0-60 s, the trade (the clanrats' share lost less ours, 0-60 s), and the
    neighbours' share lost (are they drawn in). No contact (a battle whose orders never started): no numbers."""
    t = _t(lane)
    s = lane["samples"]
    m = np.nan_to_num(cp.series(s, "a", "m")) + np.nan_to_num(cp.series(s, "tg", "m"))
    out = {"kind": lane["spec"]["kind"], "run": lane.get("run")}
    if not (m > 0).any():
        return out
    c = float(t[m > 0][0])
    out["contact_s"] = c
    for who in ("a", "tg"):
        hp = cp.series(s, who, "hp")
        h0 = float(hp[np.isfinite(hp)][0])
        for lo, hi in WIDTH_WIN:
            out[f"{who}_{lo}_{hi}"] = (cp.at(t, hp, c + lo) - cp.at(t, hp, c + hi)) / h0
        out[f"{who}_lost"] = (h0 - cp.at(t, hp, c + 60)) / h0
    out["trade"] = out["a_lost"] - out["tg_lost"]
    ex = lane.get("extras") or []
    if ex:
        te = np.array([x for x, _ in ex])
        hs = np.array([sum(float(r.get("hp") or 0) for r in rows) for _, rows in ex])
        out["nb_lost"] = (hs[0] - cp.at(te, hs, c + 60)) / hs[0] if hs[0] else None
    return out


def _still_after(t, v, t0, lim=0.3, mv=None):
    """s after t0 to the unit's last sample moving (the game's is_moving flag mv when there is one: its centre jumps
    1-2 m between two readings at rest; else faster than lim m/s - the simulator's), 0: it never moved after t0."""
    if mv is not None and np.isfinite(mv).any():
        sel = (t > t0 + 1e-6) & (np.nan_to_num(mv) > 0)
    else:
        sel = (t > t0 + 1e-6) & np.isfinite(v) & (v > lim)
    return float(t[sel][-1] - t0) if sel.any() else 0.0


def reissue_measure(lane):
    """The reissue plan: from the second order (the step's phase; 'none': its nominal time) - the centre's way and
    speed, s to the last sample moving (> 0.3 m/s over 0.5 s), s to the bearing staying within 10 deg of the facing
    ordered, the bearing's widest swing from the one at the order within 15 s, the 'braced' flag (the game's) lost and
    s to its return, the men's median way and s to the last 0.5 s in which they moved > 0.25 m/s on average, the
    arrival (from the go: the last sample moving)."""
    spec, s = lane["spec"], lane["samples"]
    t = _t(lane)
    x, z, b = cp.series(s, "a", "x"), cp.series(s, "a", "z"), cp.series(s, "a", "b")
    v = _speed(lane, "a", 0.5)
    steps = spec.get("steps") or []
    tr = _phase_t(lane, f"step{len(steps)}:{steps[-1]['kind']}") if steps else None
    if tr is None or (len(steps) < 2 and steps and steps[0]["kind"] == "move"):
        tr = float(spec.get("reissue_at") or 0)            # ('none': no second order)
    goal = float(steps[-1].get("bearing", np.nan)) if steps else np.nan
    out = {"kind": spec["kind"], "run": lane.get("run")}
    ok = np.isfinite(x) & np.isfinite(b)
    if ok.sum() < 4 or tr > t[ok][-1]:
        return out
    after = (t >= tr) & ok
    xr, zr = cp.at(t, x, tr), cp.at(t, z, tr)
    out["v_at_order"] = cp.at(t, v, tr)
    out["moved_m"] = float(np.hypot(x[ok][-1] - xr, z[ok][-1] - zr))
    mv = cp.series(s, "a", "mv")
    out["still_s"] = _still_after(t, v, tr, mv=mv)
    out["arrive_s"] = _still_after(t, v, 0.0, mv=mv)
    off = np.abs(cp._ang(b - goal))
    bad = after & (off > 10)
    out["b10_s"] = float(t[bad][-1] - tr + (t[1] - t[0])) if bad.any() else 0.0
    br = cp.at(t, b, tr)
    win = after & (t <= tr + 15)
    out["b_swing_deg"] = float(np.max(np.abs(cp._ang(b[win] - br)))) if win.any() else None
    out["b_end_off_deg"] = float(off[ok][-1])
    out["ready_s"] = max(out["still_s"], out["b10_s"])     # still and on its facing (the simulator has no brace flag)
    st = np.array(["braced" in str(_row(r, "a").get("st") or "") for r in s])
    if any(_row(r, "a").get("st") is not None for r in s):
        i0 = int(np.argmin(np.abs(t - tr)))
        lost = after & ~st
        out["braced_at_order"] = bool(st[i0])
        out["braced_lost"] = bool(lost.any())
        out["braced_back_s"] = float(t[lost][-1] - tr + (t[1] - t[0])) if lost.any() else 0.0
    men = [(tm, np.asarray(a, float).reshape(-1, 2) / 10) for tm, a, _ in lane.get("men", []) if len(a) >= 8]
    men = [(tm, p) for tm, p in men if tm >= tr - 0.6]
    if len(men) >= 3:
        n = min(len(p) for _, p in men)
        p0 = men[0][1][:n]
        out["men_moved_m"] = float(np.median(np.hypot(*(men[-1][1][:n] - p0).T)))
        last = 0.0
        for (t1, p1), (t2, p2) in zip(men, men[1:]):
            if t2 > t1 and np.mean(np.hypot(*(p2[:n] - p1[:n]).T)) / (t2 - t1) > 0.25:
                last = t2 - tr
        out["men_settle_s"] = float(last)
    return out


def counter_measure(lane):
    """The counter plan: the first contact (either unit's melee flag), the front-to-front gap at our counter-charge's
    order ('a_go': the centres less the half depths), our way from the start and speed over the 2 s before contact, the
    clanrats' too, each side's share of its starting health lost 0-3 / 0-10 / 0-30 s after contact, the trades (theirs
    lost less ours) over 10 and 30 s."""
    spec, s = lane["spec"], lane["samples"]
    if spec.get("ours") == "tg":
        # (the 'ans' lanes: ours is the lane's target) the rows swapped, 'a' ours as in the other lanes
        s = [(x[0], x[2], x[1]) + tuple(x[3:]) for x in s]
        lane = dict(lane, samples=s)
    t = _t(lane)
    m = np.nan_to_num(cp.series(s, "a", "m")) + np.nan_to_num(cp.series(s, "tg", "m"))
    out = {"kind": spec["kind"], "run": lane.get("run")}
    if not (m > 0).any():
        return out
    c = float(t[m > 0][0])
    out["contact_s"] = c
    ax, az = cp.series(s, "a", "x"), cp.series(s, "a", "z")
    tx, tz = cp.series(s, "tg", "x"), cp.series(s, "tg", "z")
    half = (spec.get("a_depth", 0) + spec.get("t_depth", 0)) / 2
    go = _phase_t(lane, "a_go") if spec.get("ours") != "tg" else c
    if go is not None:
        out["order_s"] = go
        out["order_gap_m"] = float(np.hypot(cp.at(t, ax, go) - cp.at(t, tx, go), cp.at(t, az, go) - cp.at(t, tz, go))
                                   - half)
    ok = np.isfinite(ax)
    out["a_way_m"] = float(np.hypot(cp.at(t, ax, c) - ax[ok][0], cp.at(t, az, c) - az[ok][0]))
    # the health from the sample before the first flag: the simulator's first strike lands in the flag's own sample,
    # the game's 0.3-0.6 s after it (sim/evidence/counter)
    i_c = int(np.searchsorted(t, c - 1e-6))
    c0 = float(t[max(i_c - 1, 0)])
    for who in ("a", "tg"):
        v = _speed(lane, who, 2.0)
        out[f"{who}_v_contact"] = cp.at(t, v, c)
        hp = cp.series(s, who, "hp")
        h0 = float(hp[np.isfinite(hp)][0])
        for hi in (3, 10, 30):
            out[f"{who}_0_{hi}"] = (cp.at(t, hp, c0) - cp.at(t, hp, c + hi)) / h0
    for hi in (10, 30):
        out[f"trade_{hi}"] = out[f"tg_0_{hi}"] - out[f"a_0_{hi}"]
    return out


def _first(t, on, lo=-np.inf):
    """The first time from lo on where `on` holds, else None."""
    k = np.flatnonzero(np.asarray(on, bool) & (t >= lo - 1e-6))
    return float(t[k[0]]) if len(k) else None


def _episodes(t, r, sh):
    """(first rout s, routs, rallies, shattered s) from the routing / shattered flags."""
    r = np.nan_to_num(r) > 0
    sh = np.nan_to_num(sh) > 0
    starts = np.flatnonzero(r[1:] & ~r[:-1]) + 1
    if r[0]:
        starts = np.r_[0, starts]
    ends = np.flatnonzero(~r[1:] & r[:-1]) + 1
    rallies = int(sum(1 for e in ends if not sh[e]))
    return (float(t[starts[0]]) if len(starts) else None, int(len(starts)), rallies,
            float(t[np.flatnonzero(sh)[0]]) if sh.any() else None)


def flank2v1_measure(lane):
    """The flank2v1 plan's lane (module doc; F2_*): c0 the anvil's contact (the target's first melee flag), the
    hammer's order (after:attack_same) and first melee flag h; from ref (h; the 'stand' lane: c0 + F2_STAND_REF_S) the
    target's, the hammer's and the anvil's shares of their starting health lost in F2_WINDOWS (from the sample before
    ref, as counter_measure); the target's first waver and rout (s from ref), its routs, rallies and shattering; the
    simulator's rows also carry its charge (chg: 0-1), the target's side struck from (hs_a / hs_e1: 1 front, 2 flank,
    3 rear) and its fight-balance points (cmb). The game's memory measures: flank2v1_mem."""
    spec, s = lane["spec"], lane["samples"]
    t = _t(lane)
    out = {"kind": spec["kind"], "run": lane.get("run"), "target": spec["kind"].split("_")[0],
           "how": spec["kind"].split("_", 1)[1]}
    m_t = np.nan_to_num(cp.series(s, "tg", "m")) > 0
    m_a = np.nan_to_num(cp.series(s, "a", "m")) > 0
    c0 = _first(t, m_t)
    if c0 is None:
        return out
    out["c0"] = c0
    order = _phase_t(lane, "after:attack_same")
    out["order_s"] = None if order is None else order - c0
    h = _first(t, m_a, order) if order is not None else None
    out["h_s"] = None if h is None else h - c0
    ref = h if h is not None else (c0 + F2_STAND_REF_S if spec["kind"].endswith("stand") else None)
    if ref is None:
        return out
    out["ref"] = ref
    i_r = int(np.searchsorted(t, ref - 1e-6))
    r0 = float(t[max(i_r - 1, 0)])
    ex = lane.get("extras") or []
    e_t = np.array([x[0] for x in ex]) if ex else np.array([])
    e_hp = np.array([np.nan if (x[1][0] or {}).get("hp") is None else float(x[1][0]["hp"]) for x in ex]) if ex else None
    series = {"tg": (t, cp.series(s, "tg", "hp")), "a": (t, cp.series(s, "a", "hp"))}
    if e_hp is not None and len(e_hp):
        series["e"] = (e_t, e_hp)
    h0 = {}
    for who, (tt, hp) in series.items():
        ok = np.isfinite(hp)
        if not ok.any():
            continue
        h0[who] = float(hp[ok][0])
        for lo, hi in F2_WINDOWS:
            a_, b_ = cp.at(tt, hp, r0 if lo == 0 else ref + lo), cp.at(tt, hp, ref + hi)
            out[f"{who}_{lo}_{hi}"] = (a_ - b_) / h0[who] if np.isfinite(a_) and np.isfinite(b_) else None
        a_, b_ = cp.at(tt, hp, r0), cp.at(tt, hp, ref)
        out[f"{who}_before"] = (cp.at(tt, hp, c0) - b_) / h0[who] if np.isfinite(b_) else None
    for lo, hi in ((0, 20), (0, 60), (0, 120)):
        parts = [out.get(f"tg_{a}_{b}") for a, b in F2_WINDOWS if a >= lo and b <= hi]
        out[f"tg_0_{hi}"] = sum(parts) if parts and None not in parts else None
        mine = []
        for who in ("a", "e"):
            p_ = [out.get(f"{who}_{a}_{b}") for a, b in F2_WINDOWS if a >= lo and b <= hi]
            mine.append(sum(p_) * h0[who] if who in h0 and p_ and None not in p_ else None)
        own = (sum(mine) / (h0.get("a", 0) + h0.get("e", 0))) if None not in mine and h0.get("e") else None
        out[f"own_0_{hi}"] = own
        out[f"trade_0_{hi}"] = (out[f"tg_0_{hi}"] - own) if out[f"tg_0_{hi}"] is not None and own is not None else None
    w = cp.series(s, "tg", "w")
    if np.isfinite(w).any():
        wv = _first(t, np.nan_to_num(w) > 0, c0)
        out["waver_s"] = None if wv is None else wv - ref
    r_, routs, rallies, shat = _episodes(t, cp.series(s, "tg", "r"),
                                         cp.series(s, "tg", "sh" if lane.get("run") != "sim" else "s"))
    out["rout_s"] = None if r_ is None else r_ - ref
    out["routs"], out["rallies"] = routs, rallies
    out["shattered_s"] = None if shat is None else shat - ref
    out["end_s"] = float(t[-1]) - ref
    if lane.get("run") == "sim":
        chg = cp.series(s, "a", "chg")
        if h is not None and np.isfinite(chg).any():
            win = (t >= h - 0.6) & (t <= h + 2.0)
            out["charge"] = float(np.nanmax(chg[win])) if win.any() else None
        hs = cp.series(s, "tg", "hs_a")
        if h is not None and np.isfinite(hs).any():
            k = np.flatnonzero((t >= h - 0.6) & (np.nan_to_num(hs) > 0))
            out["side"] = ("front", "flank", "rear")[int(hs[k[0]]) - 1] if len(k) else None
        he = cp.series(s, "tg", "hs_e1")
        if np.isfinite(he).any():
            k = np.flatnonzero(np.nan_to_num(he) > 0)
            out["side_anvil"] = ("front", "flank", "rear")[int(he[k[0]]) - 1] if len(k) else None
        cmb = cp.series(s, "tg", "cmb")
        if np.isfinite(cmb).any():
            for dt in (-1, 5, 20, 60):
                out[f"cmb_{dt}"] = cp.at(t, cmb, ref + dt)
    return out


# the game's fight outcome state (unit + 0x3640: reverse/notes/mem/morale.md) -> the morale points it gives
F2_OUTCOME_PTS = {0: 0, 1: 8, 2: 6, 3: 3, 4: 0, 5: 0, 6: -3, 7: -8}


def flank2v1_mem(run_dir, rows):
    """The memory measures of a flank2v1 game run (built with --mem), added to its measure rows in place: the target's
    morale effects (eff_h2 / eff_h10: name -> points at the poll nearest ref + 2 / + 10 s), the sides its enemies struck
    from (the morale object's contact list: 0 front / 1 flank / 2 rear - the anvil the enemy in the list before ref, the
    hammer the one new after it), its fight outcome state (unit + 0x3640) and points at ref - 1 / + 5 / + 20 / + 60 s;
    the hammer's men within 4 m of a target man (the means over the windows) and its charge at the first contact (the
    share of its men with the charge multiplier clamp((deadline - now) / 13 s) > 0.5, and the mean multiplier)."""
    from game.probe import mem_probe as mp
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["config"]
    names = {l["name"]: (l["attacker"], l["target"]) for l in cfg["lanes"]}
    mor = mp.load_morale(run_dir)
    men = mp.load_men(run_dir)
    g = (mor["go_ms"] or 0) / 1000
    for p in men["polls"]:
        p["t"] = p["t"] - g
    col = mp.COL
    for row in rows:
        if row.get("ref") is None or row.get("lane") not in names:
            continue
        a_name, t_name = names[row["lane"]]
        ref = row["ref"]
        ser = mp.morale_series(mor, t_name)
        tt = np.array(ser["t"]) if ser["t"] else np.array([])
        if len(tt):
            for key, dt in (("eff_h2", 2), ("eff_h10", 10)):
                k = int(np.argmin(np.abs(tt - (ref + dt))))
                row[key] = {n: v[k] for n, v in ser["fx"].items() if v[k]}
            comb = ser["uf"].get("combat")
            if comb:
                for dt in (-1, 5, 20, 60):
                    k = int(np.argmin(np.abs(tt - (ref + dt))))
                    st_ = comb[k]
                    row[f"outcome_{dt}"] = st_
                    row[f"cmb_{dt}"] = F2_OUTCOME_PTS.get(st_) if st_ is not None else None
            # the contact list: [enemy, side, ...] a poll
            cts = []
            for p in mor["polls"]:
                u = p["units"].get(t_name)
                if u is not None:
                    cts.append((p["t"], mp._pairs(u.get("ct"))))
            # the anvil is the first enemy in the list (its contact at c0); the hammer any other. The list takes the
            # hammer up to ~1 s before the script's melee flag; a hammer standing beside the anvil (the 'front' lanes) may
            # touch the target before its strike: side_pre is its first side then, side the one it has from 3 s
            # before ref on
            ids = [e for _, ps in cts for e, _ in ps]
            anvil_id = ids[0] if ids else None
            anvil = [sd for tp, ps in cts for e, sd in ps if e == anvil_id]
            row["side_anvil"] = ("front", "flank", "rear")[int(anvil[0])] if anvil else None
            ham = [(tp, e, sd) for tp, ps in cts for e, sd in ps if e != anvil_id]
            new = [x for x in ham if x[0] >= ref - 3]
            pre = [x for x in ham if x[0] < ref - 3]
            row["side"] = ("front", "flank", "rear")[int(new[0][2])] if new else None
            row["side_t"] = (new[0][0] - ref) if new else None
            row["side_pre"] = ("front", "flank", "rear")[int(pre[0][2])] if pre else None
            row["side_pre_t"] = (pre[0][0] - ref) if pre else None
            row["sides_seen"] = sorted({("front", "flank", "rear")[int(sd)] for _, _, sd in new})
            for eff in ("flank_attacked", "rear_attacked", "flanks_exposed_single", "flanks_exposed_multiple"):
                v = ser["fx"].get(eff)
                if v:
                    on = [tp - ref for tp, x in zip(tt, v) if x]
                    row[f"{eff}_first"] = on[0] if on else None
        # the hammer's men near the target's, and its charge at the first contact (the sparse polls: 0.5 s)
        near, chg = [], None
        for p in men["polls"]:
            ua, ut = p["units"].get(a_name), p["units"].get(t_name)
            if ua is None or ut is None or not len(ua["m"]) or not len(ut["m"]):
                continue
            ma, mt = ua["m"], ut["m"]
            la, lt = ma[ma[:, col["live"]] == 1], mt[mt[:, col["live"]] == 1]
            if not len(la) or not len(lt):
                continue
            d = np.hypot(la[:, None, col["x"]] - lt[None, :, col["x"]], la[:, None, col["z"]] - lt[None, :, col["z"]])
            near.append((p["t"] - ref, int((d.min(1) <= 4.0).sum())))
            if chg is None and row.get("h_s") is not None and p["t"] >= ref - 0.25 and p.get("tick") is not None:
                mult = np.clip((la[:, col["chg"]] - p["tick"] / 10.0) / 13.0, 0, 1)
                chg = {"charge_share": float((mult > 0.5).mean()), "charge_mean": float(mult.mean()),
                       "charge_poll_s": p["t"] - ref}
        if near:
            nt, nn = np.array(near).T
            for lo, hi in F2_WINDOWS:
                k = (nt >= lo) & (nt < hi)
                row[f"near4_{lo}_{hi}"] = float(nn[k].mean()) if k.any() else None
        if chg:
            row.update(chg)
            row["charge"] = chg["charge_mean"]
    return rows


# The numbers of the flank2v1 table (game | twin), in order
F2_KEY = ("h_s", "tg_before", "tg_0_5", "tg_5_20", "tg_20_60", "tg_60_120", "tg_0_60", "tg_0_120", "a_0_5", "a_5_20",
          "e_0_5", "e_5_20", "own_0_60", "own_0_120", "trade_0_60", "trade_0_120", "waver_s", "rout_s", "routs",
          "rallies", "shattered_s", "charge", "side", "side_anvil", "cmb_-1", "cmb_5", "cmb_20", "cmb_60",
          "near4_0_5", "near4_5_20", "near4_20_60")


def flank2v1_lanes(source, runs=(), copies=8, device="cpu", params=None):
    """The flank2v1 lanes measured: source 'game' (the runs, with the memory measures), 'twin' (the simulator on the
    runs' recorded starts, `copies` a lane) or 'layout' (the simulator on the plan's own places - before any game run)."""
    if source == "game":
        rows = []
        for d in runs:
            lanes = [ln for ln in cp.load_run(d) if ln.get("plan") == "flank2v1" and ln["spec"]["kind"] != "clock"]
            rs = []
            for ln in lanes:
                r = flank2v1_measure(ln)
                r.update(plan="flank2v1", battle=ln.get("battle"), lane=ln["spec"]["name"])
                rs.append(r)
            rows += flank2v1_mem(d, rs)
        return rows
    if source == "twin":
        lanes = [ln for d in runs for ln in cp.load_run(d)
                 if ln.get("plan") == "flank2v1" and ln["spec"]["kind"] != "clock"]
    else:
        lanes = []
        for i in range(1, 3):                    # (the places do not matter to the twin: one battle a target)
            config, _, _ = cp.run_config("flank2v1", i)
            for l in config["lanes"]:
                if l["kind"] == "clock":
                    continue
                a = dict(l["a_at"], men=cp.UNITS[F2_HAMMER][1])
                tg = dict(l["t_at"], men=cp.UNITS[F2_T][1])
                lanes.append({"run": "layout", "plan": "flank2v1", "battle": i, "spec": l, "samples": [(0.0, a, tg, None)],
                              "men": [], "contacts": {}, "end": {"t": l["max_s"] * 1000}, "abilities": [], "phases": []})
    rows = []
    for ln, s in zip([ln for ln in lanes for _ in range(copies)],
                     cp.sim_lanes(lanes, params=params, device=device, copies=copies, seed=0)):
        r = flank2v1_measure(dict(s, run="sim"))
        r.update(plan="flank2v1", battle=ln.get("battle"), lane=ln["spec"]["name"])
        rows.append(r)
    return rows


def flank2v1_summary(rows):
    """{kind: {field: [mean, sd, n] for numbers; {value: count} for texts}}."""
    out = {}
    for kind in dict.fromkeys(r["kind"] for r in rows):
        rs = [r for r in rows if r["kind"] == kind]
        row = {"n": len(rs)}
        for f in F2_KEY + ("rout_any",):
            if f == "rout_any":
                row[f] = [float(np.mean([r.get("routs", 0) > 0 for r in rs])), None, len(rs)]
                continue
            vals = [r.get(f) for r in rs]
            nums = [float(v) for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None
                    and np.isfinite(float(v))]
            txt = [str(v) for v in vals if isinstance(v, str)]
            if nums:
                row[f] = [float(np.mean(nums)), float(np.std(nums, ddof=1)) if len(nums) > 1 else None, len(nums)]
            elif txt:
                row[f] = {v: txt.count(v) for v in sorted(set(txt))}
        out[kind] = row
    return out


def _f2(v):
    if v is None:
        return "-"
    if isinstance(v, dict):
        return "/".join(f"{k}:{n}" for k, n in v.items())
    m, sd, n = v
    p = 3 if abs(m) < 1 else 1
    return f"{m:.{p}f}" + (f"±{sd:.{p}f}" if sd is not None else "")


def flank2v1_report(runs=(), source="game", sim=False, copies=8, device="cpu", out=None):
    """Prints the flank2v1 table (per target and kind; with sim the twin on the same recorded starts beside it) and
    writes build/charge-probe/flank2v1_<source>.json (not in Git)."""
    rows = flank2v1_lanes(source, runs, copies=copies, device=device)
    table = flank2v1_summary(rows)
    result = {"source": source, "runs": [str(r) for r in runs], "lanes": rows, "summary": table}
    twin_table = None
    if sim and source == "game":
        trows = flank2v1_lanes("twin", runs, copies=copies, device=device)
        twin_table = flank2v1_summary(trows)
        result["twin"] = {"lanes": trows, "summary": twin_table}
    print(f"== flank2v1 ({source}{' | twin' if twin_table else ''}): mean±sd over lanes")
    for kind, r in table.items():
        cells = []
        for f in F2_KEY + ("rout_any",):
            if f not in r and not (twin_table and f in twin_table.get(kind, {})):
                continue
            c = _f2(r.get(f))
            if twin_table:
                c += " | " + _f2(twin_table.get(kind, {}).get(f))
            cells.append(f"{f} {c}")
        print(f"  {kind} (n {r['n']}): " + "; ".join(cells))
    out = Path(out) if out else cp.ROOT / f"flank2v1_{source}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(out)
    return result


# ---------------------------------------------------------------- the meetfar probe (counterfar, intercept)

def _fat_level(v):
    """A fatigue row value (the game's state key, the simulator's level) as the level 0 fresh .. 5 exhausted."""
    if v is None:
        return None
    if isinstance(v, str):
        return cp.FAT_LEVELS.index(v) if v in cp.FAT_LEVELS else None
    return float(v)


def _path_m(t, x, z, lo, hi):
    """The way walked (m) from lo to hi: the centre's path at 1 s steps (the engine moves a unit's centre in jerks: the
    0.25 s samples' path is ~1.5x the way)."""
    ok = np.isfinite(x) & np.isfinite(z)
    if ok.sum() < 2 or hi - lo < 1:
        return None
    ts = np.append(np.arange(lo, hi, 1.0), hi)
    xs, zs = np.interp(ts, t[ok], x[ok]), np.interp(ts, t[ok], z[ok])
    return float(np.hypot(np.diff(xs), np.diff(zs)).sum())


def counterfar_measure(lane):
    """The counterfar plan's lane: counter_measure's numbers (the contact, the front-to-front gap at our order - the
    'stand' lanes: the start's gap -, our way and speed, the shares of health lost from the sample before the first flag)
    over CF_WINDOWS, the trades over 10 / 30 / 60 s; at the contact: our way since the order (path), the fatigue level of
    each side (0 fresh .. 5 exhausted), and (the simulator's rows) each side's charge left (0-1, the most in -0.6..+2 s)."""
    spec = lane["spec"]
    out = counter_measure(lane)
    out.update(pace=spec["kind"].split("_")[1], unit=spec["kind"].split("_")[0], at=spec.get("counter_at"))
    c = out.get("contact_s")
    if c is None:
        return out
    s, t = lane["samples"], _t(lane)
    if spec["kind"].split("_")[1] == "stand":
        out["order_s"], out["order_gap_m"] = float(t[0]), float(spec["gap_m"])
    i_c = int(np.searchsorted(t, c - 1e-6))
    c0 = float(t[max(i_c - 1, 0)])
    for who in ("a", "tg"):
        hp = cp.series(s, who, "hp")
        h0 = float(hp[np.isfinite(hp)][0])
        out[f"{who}_0_60"] = (cp.at(t, hp, c0) - cp.at(t, hp, c + 60)) / h0
        k = min(max(i_c, 0), len(s) - 1)
        out[f"{who}_fat"] = _fat_level(_row(s[k], who).get("fat"))
        chg = cp.series(s, who, "chg")
        if np.isfinite(chg).any():
            win = (t >= c - 0.6) & (t <= c + 2.0)
            out[f"{who}_charge"] = float(np.nanmax(chg[win])) if win.any() else None
    out["trade_60"] = out["tg_0_60"] - out["a_0_60"]
    ax, az = cp.series(s, "a", "x"), cp.series(s, "a", "z")
    if out.get("order_s") is not None:
        out["a_path_m"] = _path_m(t, ax, az, out["order_s"], c)
    return out


def _charge_at(p, name, col):
    """(share of the unit's living men with the charge multiplier clamp((deadline - now) / 13 s) > 0.5, the mean) in one
    memory poll (flank2v1_mem), or None."""
    u = p["units"].get(name)
    if u is None or not len(u["m"]) or p.get("tick") is None:
        return None
    m = u["m"][u["m"][:, col["live"]] == 1]
    if not len(m):
        return None
    mult = np.clip((m[:, col["chg"]] - p["tick"] / 10.0) / 13.0, 0, 1)
    return float((mult > 0.5).mean()), float(mult.mean())


def _depth_m(p, name, col, bearing):
    """The unit's formation depth (m) in one memory poll: its living men along its facing, 2-98 percentiles."""
    u = p["units"].get(name)
    if u is None or not len(u["m"]) or bearing is None:
        return None
    m = u["m"][u["m"][:, col["live"]] == 1]
    if len(m) < 4:
        return None
    fx, fz = _fwd(float(bearing))
    proj = m[:, col["x"]] * fx + m[:, col["z"]] * fz
    return float(np.percentile(proj, 98) - np.percentile(proj, 2))


def counterfar_mem(run_dir, lanes, rows):
    """The memory numbers of a counterfar game run, added to its rows in place: each side's charge at the contact (the
    first poll from 0.25 s before it: the share of men in their charge, the mean multiplier) and our formation's depth
    then (m)."""
    from game.probe import mem_probe as mp
    men = mp.load_men(run_dir)
    if not men["polls"]:
        return rows
    mor = mp.load_morale(run_dir)
    g = (mor["go_ms"] or 0) / 1000
    polls = [dict(p, t=p["t"] - g) for p in men["polls"]]
    col = mp.COL
    by = {ln["spec"]["name"]: ln for ln in lanes}
    for row in rows:
        ln = by.get(row.get("lane"))
        c = row.get("contact_s")
        if ln is None or c is None:
            continue
        spec = ln["spec"]
        p = next((p for p in polls if p["t"] >= c - 0.25 and spec["attacker"] in p["units"]), None)
        if p is None:
            continue
        row["mem_poll_s"] = p["t"] - c
        for who, name in (("a", spec["attacker"]), ("tg", spec["target"])):
            ch = _charge_at(p, name, col)
            if ch:
                row[f"{who}_charge_share"], row[f"{who}_charge"] = ch
        t = _t(ln)
        b = cp.series(ln["samples"], "a", "b")
        row["a_depth_m"] = _depth_m(p, spec["attacker"], col, cp.at(t, b, c))
    return rows


def _roles(lane):
    """{role: {key: np.array over the lane's samples}} of an intercept lane: B the attacker, E1 the target, A the first
    extra, E2 the second target (ct: the engine's target's script name, the game's rows only)."""
    t = _t(lane)
    s = lane["samples"]
    rows = {"B": [x[1] for x in s], "E1": [x[2] for x in s]}
    ex = {round(tt, 3): (r or [None])[0] for tt, r in lane.get("extras") or []}
    t2 = {round(tt, 3): r for tt, r in lane.get("t2") or []}
    rows["A"] = [ex.get(round(tt, 3)) for tt in t]
    rows["E2"] = [t2.get(round(tt, 3)) for tt in t]
    out = {}
    for role, rs in rows.items():
        d = {}
        for k in ("hp", "m", "x", "z", "b", "r", "w", "chg", "fh", "cmb", "mv", "fast"):
            d[k] = np.array([np.nan if (r or {}).get(k) is None else float(r[k]) for r in rs])
        d["ct"] = [(r or {}).get("ct") for r in rs]
        d["fat"] = [(r or {}).get("fat") for r in rs]
        out[role] = d
    return t, out


def _corners(x, z, b, w, d):
    fx, fz = _fwd(b)
    rx, rz = fz, -fx
    return np.array([(x + sx * rx * w / 2 + sz * fx * d / 2, z + sx * rz * w / 2 + sz * fz * d / 2)
                     for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def _seg_d(p, a, b):
    ab = b - a
    k = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1))
    return float(np.hypot(*(p - (a + k * ab))))


def rect_gap(u1, u2):
    """The edge-to-edge gap (m) of two units' rectangles (x, z, bearing, width, depth); 0 when they overlap."""
    P, Q = _corners(*u1), _corners(*u2)
    for poly in (P, Q):
        sep = False
        for i in range(2):
            ax_ = poly[(i + 1) % 4] - poly[i]
            n = np.array([-ax_[1], ax_[0]])
            pp, qq = P @ n, Q @ n
            if pp.max() < qq.min() or qq.max() < pp.min():
                sep = True
        if sep:
            break
    else:
        return 0.0
    return min(min(_seg_d(p, Q[i], Q[(i + 1) % 4]) for p in P for i in range(4)),
               min(_seg_d(q, P[i], P[(i + 1) % 4]) for q in Q for i in range(4)))


def _gap_series(t, R, r1, r2, dims):
    out = np.full(len(t), np.nan)
    for i in range(len(t)):
        a, b = R[r1], R[r2]
        v = (a["x"][i], a["z"][i], a["b"][i], b["x"][i], b["z"][i], b["b"][i])
        if all(np.isfinite(v)):
            out[i] = rect_gap((v[0], v[1], v[2]) + dims[r1], (v[3], v[4], v[5]) + dims[r2])
    return out


def _ang_d(a):
    return float((a + 180) % 360 - 180)


def intercept_measure(lane):
    """The intercept plan's lane (IC_*): c0 the lane's first melee flag (any of the four units); every unit's first melee
    flag, B's order (a_go: s after the go, why, the gap then - the centres less the half depths) and the rectangles' gaps
    (edge to edge, at the ordered widths) B-E2, E2-A over time; each unit's share of its starting health lost over
    0-IC_WINDOWS s from the sample before c0, ours (A + B) and theirs (E1 + E2) as shares of the side's starting health,
    the trade (theirs - ours) at 30 / 60 / 120 s; A's and B's first waver and rout (s from c0), routs; E2: the first time
    its gap to A is within 3 m (it reached A), B's first melee with E2 (gap within 3 m), intercepted (B on E2 before E2
    reached A), E2's way to A after B met it, and the per-second track of E2 from 5 s before to 20 s after B met it (its
    speed, its heading off the way to A and to B, its melee flag, the gaps, its engine target in the game); the simulator's
    rows: each unit's charge at its first flag, A's worst side struck from (fh) and fight balance (cmb)."""
    spec = lane["spec"]
    t, R = _roles(lane)
    W = cp.WIDTH_M
    dims = {"A": (W, cp.depth(*cp.UNITS[IC_A][:2], W)), "B": (spec["a_width"], spec["a_depth"]),
            "E1": (spec["t_width"], spec["t_depth"]), "E2": (spec["t_width"], spec["t_depth"])}
    out = {"kind": spec["kind"], "side": spec.get("b_side"), "run": lane.get("run")}
    first = {r: _first(t, np.nan_to_num(R[r]["m"]) > 0) for r in R}
    for r, v in first.items():
        out[f"m_{r}"] = v
    cs = [v for v in first.values() if v is not None]
    if not cs:
        return out
    c0 = min(cs)
    out["c0"] = c0
    for r in R:
        out[f"m_{r}"] = None if first[r] is None else first[r] - c0
    go = next((p for p in lane.get("phases", []) if p["phase"] == "a_go"), None)
    if go is not None:
        tg_ = "E2" if spec["mode"] == "attack_t2" else "E1"
        out["order_s"] = go["t"] / 1000 - c0
        out["order_why"] = go.get("why")
        if go.get("d") is not None:
            out["order_gap_m"] = float(go["d"]) - (dims["B"][1] + dims[tg_][1]) / 2
    i_c = int(np.searchsorted(t, c0 - 1e-6))
    r0 = float(t[max(i_c - 1, 0)])
    h0, lost = {}, {}
    for r in R:
        hp = R[r]["hp"]
        ok = np.isfinite(hp)
        if not ok.any():
            continue
        h0[r] = float(hp[ok][0])
        for hi in IC_WINDOWS:
            v = cp.at(t, hp, r0) - cp.at(t, hp, c0 + hi)
            lost[(r, hi)] = v if np.isfinite(v) else None
            out[f"{r}_0_{hi}"] = None if lost[(r, hi)] is None else v / h0[r]
    for hi in IC_WINDOWS:
        sides = {}
        for side, rs in (("own", ("A", "B")), ("en", ("E1", "E2"))):
            vals = [lost.get((r, hi)) for r in rs]
            sides[side] = (sum(vals) / sum(h0[r] for r in rs)) if None not in vals and all(r in h0 for r in rs) else None
            out[f"{side}_0_{hi}"] = sides[side]
        out[f"trade_{hi}"] = (sides["en"] - sides["own"]) if None not in sides.values() else None
    for r in ("A", "B"):
        w = R[r]["w"]
        wv = _first(t, np.nan_to_num(w) > 0) if np.isfinite(w).any() else None
        out[f"{r}_waver_s"] = None if wv is None else wv - c0
        rt, routs, rallies, _ = _episodes(t, R[r]["r"], np.zeros(len(t)))
        out[f"{r}_rout_s"] = None if rt is None else rt - c0
        out[f"{r}_routs"] = routs
    gBE2 = _gap_series(t, R, "B", "E2", dims)
    gE2A = _gap_series(t, R, "E2", "A", dims)
    gE1A = _gap_series(t, R, "E1", "A", dims)
    reachA = _first(t, np.nan_to_num(gE2A, nan=99) <= 3.0)
    out["e2_reach_A_s"] = None if reachA is None else reachA - c0
    lo = (go["t"] / 1000) if go is not None and spec["mode"] == "attack_t2" else -np.inf
    metB = _first(t, (np.nan_to_num(gBE2, nan=99) <= 3.0) & (np.nan_to_num(R["B"]["m"]) > 0), lo)
    out["b_e2_s"] = None if metB is None else metB - c0
    out["e1_reach_A_s"] = (lambda v: None if v is None else v - c0)(_first(t, np.nan_to_num(gE1A, nan=99) <= 3.0))
    if spec["mode"] == "attack_t2" and go is not None:
        k = int(np.searchsorted(t, go["t"] / 1000 - 1e-6))
        out["order_gap_rect_m"] = float(gBE2[min(k, len(t) - 1)])
        out["order_e2_gap_A_m"] = float(gE2A[min(k, len(t) - 1)])
    if metB is not None:
        out["intercepted"] = bool(reachA is None or metB < reachA)
        out["e2_to_A_after_B_s"] = None if reachA is None or reachA < metB else reachA - metB
        win = (t >= metB) & (t <= metB + 60)
        heldB = win & (np.nan_to_num(gBE2, nan=99) <= 3.0) & (np.nan_to_num(gE2A, nan=99) > 3.0)
        out["e2_held_by_B_s_60"] = float(heldB.sum() * np.median(np.diff(t))) if win.any() else None
        out["e2_gapA_at_B"] = float(cp.at(t, gE2A, metB))
        track = []
        x, z, b = R["E2"]["x"], R["E2"]["z"], R["E2"]["b"]
        for dt in range(-5, 21):
            tt = metB + dt
            if tt < t[0] + 1 or tt > t[-1]:
                continue
            vx = (cp.at(t, x, tt) - cp.at(t, x, tt - 1.0))
            vz = (cp.at(t, z, tt) - cp.at(t, z, tt - 1.0))
            ax_, az_ = cp.at(t, R["A"]["x"], tt) - cp.at(t, x, tt), cp.at(t, R["A"]["z"], tt) - cp.at(t, z, tt)
            bx_, bz_ = cp.at(t, R["B"]["x"], tt) - cp.at(t, x, tt), cp.at(t, R["B"]["z"], tt) - cp.at(t, z, tt)
            v = math.hypot(vx, vz)
            head = math.degrees(math.atan2(vx, vz)) if v > 0.3 else None
            to_a, to_b = math.degrees(math.atan2(ax_, az_)), math.degrees(math.atan2(bx_, bz_))
            i = int(np.argmin(np.abs(t - tt)))
            ct = R["E2"]["ct"][i]
            track.append({"dt": dt, "v": round(v, 2),
                          "head_off_A": None if head is None else round(abs(_ang_d(head - to_a)), 0),
                          "face_off_A": round(abs(_ang_d(cp.at(t, b, tt) - to_a)), 0),
                          "face_off_B": round(abs(_ang_d(cp.at(t, b, tt) - to_b)), 0),
                          "m": bool(R["E2"]["m"][i] > 0), "gapA": round(float(cp.at(t, gE2A, tt)), 1),
                          "gapB": round(float(cp.at(t, gBE2, tt)), 1),
                          "ct": None if ct is None else ("A" if ct == spec["extras"][0]["name"] else
                                                         "B" if ct == spec["attacker"] else ct or "")})
        out["e2_track"] = track
        after = [x_ for x_ in track if 1 <= x_["dt"] <= 20]
        if after:
            out["e2_v_1_20"] = float(np.mean([x_["v"] for x_ in after]))
            out["e2_faceB_1_20"] = float(np.mean([x_["face_off_B"] < 45 for x_ in after]))
            cts = [x_["ct"] for x_ in after if x_["ct"] is not None]
            out["e2_ctB_1_20"] = float(np.mean([c_ == "B" for c_ in cts])) if cts else None
            out["e2_ctA_1_20"] = float(np.mean([c_ == "A" for c_ in cts])) if cts else None
    for r in R:
        chg = R[r]["chg"]
        if first[r] is not None and np.isfinite(chg).any():
            win = (t >= first[r] - 0.6) & (t <= first[r] + 2.0)
            out[f"{r}_charge"] = float(np.nanmax(chg[win])) if win.any() else None
    if np.isfinite(R["A"]["fh"]).any():
        k = (t >= c0) & (t <= c0 + 60)
        out["A_fh_max_60"] = float(np.nanmax(R["A"]["fh"][k])) if k.any() else None
        for dt in (5, 20, 60):
            out[f"A_cmb_{dt}"] = cp.at(t, R["A"]["cmb"], c0 + dt)
    return out


def intercept_mem(run_dir, lanes, rows):
    """The memory numbers of an intercept game run, added to its rows in place: A's (and B's) morale effects ({name:
    points}) at c0 + 2 / + 10 s and at E2's first flag + 2 / + 10 s, the first time A held flank_attacked /
    rear_attacked / enemy_numbers (s from c0), A's fight outcome state (unit + 0x3640) at c0 - 1 / + 5 / + 20 / + 60 s and
    at E2's flag + 5; each unit's charge at its first melee flag (the first poll from 0.25 s before it)."""
    from game.probe import mem_probe as mp
    run_dir = Path(run_dir)
    mor = mp.load_morale(run_dir)
    men = mp.load_men(run_dir)
    g = (mor["go_ms"] or 0) / 1000
    polls = [dict(p, t=p["t"] - g) for p in men["polls"]]
    col = mp.COL
    by = {ln["spec"]["name"]: ln for ln in lanes}
    for row in rows:
        ln = by.get(row.get("lane"))
        c0 = row.get("c0")
        if ln is None or c0 is None:
            continue
        spec = ln["spec"]
        names = {"B": spec["attacker"], "E1": spec["target"], "A": spec["extras"][0]["name"], "E2": spec["target2"]}
        e2 = row.get("m_E2")
        for r in ("A", "B"):
            ser = mp.morale_series(mor, names[r])
            tt = np.array(ser["t"]) if ser["t"] else np.array([])
            if not len(tt):
                continue
            marks = [("c2", c0 + 2), ("c10", c0 + 10)]
            if e2 is not None:
                marks += [("e2_2", c0 + e2 + 2), ("e2_10", c0 + e2 + 10)]
            for key, at_ in marks:
                k = int(np.argmin(np.abs(tt - at_)))
                row[f"{r}_eff_{key}"] = {n: v[k] for n, v in ser["fx"].items() if v[k]}
            if r == "A":
                for eff in ("flank_attacked", "rear_attacked", "enemy_numbers", "losing"):
                    v = ser["fx"].get(eff)
                    on = [tp - c0 for tp, x_ in zip(tt, v or []) if x_]
                    row[f"A_{eff}_first"] = on[0] if on else None
                comb = ser["uf"].get("combat")
                if comb:
                    marks = [(-1, c0 - 1), (5, c0 + 5), (20, c0 + 20), (60, c0 + 60)]
                    if e2 is not None:
                        marks.append(("e2_5", c0 + e2 + 5))
                    for dt, at_ in marks:
                        k = int(np.argmin(np.abs(tt - at_)))
                        row[f"A_outcome_{dt}"] = comb[k]
        for r, name in names.items():
            f = row.get(f"m_{r}")
            if f is None:
                continue
            p = next((p for p in polls if p["t"] >= c0 + f - 0.25 and name in p["units"]), None)
            ch = _charge_at(p, name, col) if p is not None else None
            if ch:
                row[f"{r}_charge_share"], row[f"{r}_charge"] = ch
    return rows


def _mf_lanes(plan, source, runs=(), copies=8, device="cpu", params=None):
    """The counterfar / intercept lanes measured: source 'game' (the runs, with the memory numbers), 'twin' (the
    simulator on the runs' recorded starts, `copies` a lane) or 'layout' (the simulator on the plan's own places, before
    any game run)."""
    measure = counterfar_measure if plan == "counterfar" else intercept_measure
    mem = counterfar_mem if plan == "counterfar" else intercept_mem
    if source == "game":
        rows = []
        for d in runs:
            lanes = [ln for ln in cp.load_run(d) if ln.get("plan") == plan and ln["spec"]["kind"] != "clock"]
            rs = []
            for ln in lanes:
                r = measure(ln)
                r.update(plan=plan, battle=ln.get("battle"), lane=ln["spec"]["name"], run=Path(d).name)
                rs.append(r)
            try:
                rows += mem(d, lanes, rs)
            except (OSError, ValueError, KeyError) as e:          # a run without memory rows
                print(f"{d}: no memory numbers ({e})")
                rows += rs
        return rows
    if source == "twin":
        lanes = [ln for d in runs for ln in cp.load_run(d) if ln.get("plan") == plan and ln["spec"]["kind"] != "clock"]
    else:
        lanes = []
        which = range(1, 3) if plan == "counterfar" else range(1, 3)      # (both defenders; both sides of B)
        for i in which:
            config, _, _ = cp.run_config(plan, i)
            for l in config["lanes"]:
                if l["kind"] == "clock":
                    continue
                a = {"x": l["x"] + l.get("a_dx", 0), "z": l["z"] + l["gap_m"] + l["a_depth"] / 2, "b": 180.0,
                     "men": cp.UNITS[cp.SHORT[l["a_key"]]][1]}
                tg = {"x": l["x"], "z": l["z"] - l["t_depth"] / 2, "b": 0.0, "men": cp.UNITS[cp.SHORT[l["t_key"]]][1]}
                lanes.append({"run": "layout", "plan": plan, "battle": i, "spec": l, "samples": [(0.0, a, tg, None)],
                              "men": [], "contacts": {}, "end": {"t": l["max_s"] * 1000}, "abilities": [], "phases": []})
    rows = []
    for ln, s in zip([ln for ln in lanes for _ in range(copies)],
                     cp.sim_lanes(lanes, params=params, device=device, copies=copies, seed=0)):
        r = measure(dict(s, run="sim"))
        r.update(plan=plan, battle=ln.get("battle"), lane=ln["spec"]["name"], run="sim")
        rows.append(r)
    return rows


CF_KEY = ("order_gap_m", "contact_s", "a_way_m", "a_path_m", "a_v_contact", "tg_v_contact", "a_fat", "tg_fat",
          "a_charge", "tg_charge", "a_charge_share", "tg_charge_share", "a_depth_m", "a_0_3", "a_0_10", "a_0_30", "a_0_60",
          "tg_0_3", "tg_0_10", "tg_0_30", "tg_0_60", "trade_10", "trade_30", "trade_60")
IC_KEY = ("order_s", "order_gap_m", "order_gap_rect_m", "order_e2_gap_A_m", "m_A", "m_B", "m_E1", "m_E2",
          "e1_reach_A_s", "e2_reach_A_s", "b_e2_s", "intercepted", "e2_to_A_after_B_s", "e2_held_by_B_s_60",
          "e2_gapA_at_B", "e2_v_1_20", "e2_faceB_1_20", "e2_ctB_1_20", "e2_ctA_1_20",
          "A_0_30", "A_0_60", "A_0_120", "B_0_30", "B_0_60", "B_0_120", "E1_0_30", "E1_0_60", "E1_0_120",
          "E2_0_30", "E2_0_60", "E2_0_120", "own_0_60", "own_0_120", "en_0_60", "en_0_120", "trade_30", "trade_60",
          "trade_120", "A_waver_s", "A_rout_s", "A_routs", "B_waver_s", "B_rout_s", "B_routs", "A_charge", "B_charge",
          "E1_charge", "E2_charge", "E2_charge_share", "B_charge_share", "A_fh_max_60", "A_cmb_5", "A_cmb_20",
          "A_flank_attacked_first", "A_rear_attacked_first", "A_enemy_numbers_first", "A_outcome_-1", "A_outcome_5",
          "A_outcome_20", "A_outcome_e2_5")


def mf_summary(rows, keys):
    """{kind: {field: [mean, standard error of the mean, n] for numbers (booleans as shares); {value: count} for texts}}."""
    out = {}
    for kind in dict.fromkeys(r["kind"] for r in rows):
        rs = [r for r in rows if r["kind"] == kind]
        row = {"n": len(rs)}
        for f in keys:
            vals = [r.get(f) for r in rs if r.get(f) is not None]
            nums = [float(v) for v in vals if isinstance(v, (int, float, bool)) and np.isfinite(float(v))]
            txt = [str(v) for v in vals if isinstance(v, str)]
            if nums:
                row[f] = [float(np.mean(nums)), float(np.std(nums, ddof=1) / math.sqrt(len(nums))) if len(nums) > 1
                          else None, len(nums)]
            elif txt:
                row[f] = {v: txt.count(v) for v in sorted(set(txt))}
        out[kind] = row
    return out


def mf_paired(rows, base, keys, by=("run", "battle")):
    """{kind: {field: [mean, standard error, n]}}: each lane minus the `base` kind's lane of the same battle (a game run),
    or minus the base kind's mean (the simulator: its copies are not paired)."""
    out = {}
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r.get(k) for k in by), {}).setdefault(r["kind"], []).append(r)
    sim = all(r.get("run") == "sim" for r in rows)
    base_mean = {f: np.nanmean([float(r[f]) for r in rows if r["kind"] == base and r.get(f) is not None] or [np.nan])
                 for f in keys}
    for kind in dict.fromkeys(r["kind"] for r in rows):
        if kind == base:
            continue
        row = {}
        for f in keys:
            d = []
            for g in groups.values():
                for r in g.get(kind, []):
                    b_ = base_mean[f] if sim else (np.mean([float(x[f]) for x in g.get(base, []) if x.get(f) is not None])
                                                   if any(x.get(f) is not None for x in g.get(base, [])) else np.nan)
                    if r.get(f) is not None and np.isfinite(b_):
                        d.append(float(r[f]) - float(b_))
            if d:
                row[f] = [float(np.mean(d)), float(np.std(d, ddof=1) / math.sqrt(len(d))) if len(d) > 1 else None, len(d)]
        out[kind] = row
    return out


def _mf_cell(v):
    if v is None:
        return "-"
    if isinstance(v, dict):
        return "/".join(f"{k}:{n}" for k, n in v.items())
    m, se, n = v
    p = 3 if abs(m) < 1 else 1
    return f"{m:.{p}f}" + (f"±{se:.{p}f}" if se is not None else "") + f"({n})"


def mf_report(plan, runs=(), source="game", sim=False, copies=8, device="cpu", out=None):
    """Prints the counterfar / intercept table per kind (game | twin on the same recorded starts with sim) and the pairs
    against the hold lanes; writes build/charge-probe/<plan>_<source>.json (not in Git)."""
    keys = CF_KEY if plan == "counterfar" else IC_KEY
    rows = _mf_lanes(plan, source, runs, copies=copies, device=device)
    table = mf_summary(rows, keys)
    pair_keys = ("trade_30", "trade_60", "trade_120", "own_0_60", "own_0_120", "en_0_60", "en_0_120") \
        if plan == "intercept" else ("trade_10", "trade_30", "trade_60", "a_0_60", "tg_0_60")
    result = {"plan": plan, "source": source, "runs": [str(r) for r in runs], "lanes": rows, "summary": table}
    if plan == "intercept":
        result["paired"] = mf_paired(rows, "hold", pair_keys)
    twin_table = None
    if sim and source == "game":
        trows = _mf_lanes(plan, "twin", runs, copies=copies, device=device)
        twin_table = mf_summary(trows, keys)
        result["twin"] = {"lanes": trows, "summary": twin_table}
        if plan == "intercept":
            result["twin"]["paired"] = mf_paired(trows, "hold", pair_keys)
    print(f"== {plan} ({source}{' | twin' if twin_table else ''}): mean±se(n) over lanes")
    for kind, r in table.items():
        cells = []
        for f in keys:
            if f not in r and not (twin_table and f in twin_table.get(kind, {})):
                continue
            c = _mf_cell(r.get(f))
            if twin_table:
                c += " | " + _mf_cell(twin_table.get(kind, {}).get(f))
            cells.append(f"{f} {c}")
        print(f"  {kind} (n {r['n']}): " + "; ".join(cells))
    for name, pt in (("paired", result.get("paired")), ("twin paired", (result.get("twin") or {}).get("paired"))):
        if pt:
            print(f"== {name} (minus hold)")
            for kind, r in pt.items():
                print(f"  {kind}: " + "; ".join(f"{f} {_mf_cell(v)}" for f, v in r.items()))
    out = Path(out) if out else cp.ROOT / f"{plan}_{source}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(out)
    return result


MEASURE = {"reissue": reissue_measure, "counter": counter_measure, "width": width_measure,"lobfriend": losfriend_measure, "forestwalk": forestwalk_measure, "snipe": snipe_measure, "losmove": losmove_measure, "losrange": losrange_measure, "losfriend": losfriend_measure, "ratling": ratling_measure, "ratlord": ratling_measure, "ratcover": ratcover_measure, "ratlone": ratling_measure, "ratform": ratling_measure, "ratshape": ratling_measure, "suppress": suppress_measure, "suppress2": suppress_measure,
           "hitslow": suppress_measure, "censer": censer_measure, "flank2v1": flank2v1_measure,
           "wolfrats": wolfrats_measure, "wolfpoison": wolfpoison_measure}


def summary(rows):
    """{(plan, kind): {field: (mean, min, max, n) for numbers, the distinct values for texts}}."""
    groups = {}
    for r in rows:
        groups.setdefault((r["plan"], r["kind"]), []).append(r)
    out = {}
    for key, rs in groups.items():
        row = {"n": len(rs)}
        for f in dict.fromkeys(k for r in rs for k in r):
            if f in ("plan", "kind", "run", "battle", "lane"):
                continue
            vals = [r[f] for r in rs if r.get(f) is not None]
            nums = [float(x) for x in vals if isinstance(x, (int, float)) and not isinstance(x, bool)
                    and np.isfinite(float(x))]
            if nums and len(nums) == len(vals):
                row[f] = (float(np.mean(nums)), min(nums), max(nums), len(nums))
            elif vals:
                row[f] = sorted({str(x) for x in vals})
        out[key] = row
    return out


def _fmt(v):
    if isinstance(v, tuple):
        mean, lo, hi, n = v
        p = 0 if abs(mean) >= 100 else 2 if abs(mean) < 10 else 1
        return f"{mean:.{p}f}" if lo == hi else f"{mean:.{p}f} ({lo:.{p}f}-{hi:.{p}f})"
    return "/".join(v)


# The numbers the simulator's twin is checked on (report --sim: game | sim side by side), per plan
KEY = {"reissue": ("v_at_order", "moved_m", "still_s", "b10_s", "ready_s", "b_swing_deg", "b_end_off_deg", "arrive_s",
                   "braced_lost", "braced_back_s", "men_moved_m", "men_settle_s"),
       "counter": ("order_gap_m", "contact_s", "a_way_m", "a_v_contact", "tg_v_contact", "a_0_3", "a_0_10", "a_0_30",
                   "tg_0_3", "tg_0_10", "tg_0_30", "trade_10", "trade_30"),
       "width": ("contact_s", "tg_0_5", "tg_5_15", "tg_15_30", "tg_30_60", "a_0_5", "a_5_15", "a_15_30", "a_30_60",
                 "tg_lost", "a_lost", "trade", "nb_lost"),
       "lobfriend": ("first_shot_s", "bullets", "hp_per_bullet", "tg_hp_per_s", "cover_hp_lost", "fwd_max_m"),
       "forestwalk": ("cruise", "cruise_first", "legs", "leg_s"),
       "snipe": ("first_shot_s", "bullets", "hp_per_bullet", "tg_hp_per_s", "hid_go", "hid_order", "hid_firing_share",
                 "seen_firing_share", "revealed_s", "hid_share", "gap_seen_m", "gap_unhid_m", "hid_100", "hid_60",
                 "hid_40", "v_100", "v_60", "v_40", "shots_unseen", "gap_min_m"),
       "losmove": ("first_shot_s", "dist_at_shot_m", "shots_before_contact", "contact_s", "fwd_max_m", "walked_m"),
       "losrange": ("first_shot_s", "bullets", "dist0_m", "dist_at_shot_m", "dist_end_m", "walked_m", "moved_until_s",
                    "b0", "b_end"),
       "losfriend": ("first_shot_s", "bullets", "hp_per_bullet", "cover_hp_lost", "moved_until_s", "fwd_max_m",
                     "fwd_end_m", "side_end_m"),
       "ratling": ("first_shot_s", "bullets_per_s", "bullets_per_man_s", "bursts", "burst_bullets", "burst_s",
                   "burst_every_s", "hp_per_bullet", "tg_hp_per_s", "tg_men_lost", "tg_fx_first_s", "tg_card_speed_fx"),
       "ratlord": ("first_shot_s", "bullets_per_s", "bursts", "burst_bullets", "hp_per_bullet", "tg_hp_per_s"),
       "ratcover": ("first_shot_s", "bullets", "hp_per_bullet", "e_hp_per_bullet", "tg_hp_per_s"),
       "ratlone": ("first_shot_s", "bullets", "hp_per_bullet", "tg_hp_per_s", "tg_dead"),
       "ratform": ("first_shot_s", "bullets", "hp_per_bullet", "tg_hp_per_s", "tg_dead"),
       "ratshape": ("first_shot_s", "bullets", "hp_per_bullet", "tg_hp_per_s", "tg_dead"),
       "suppress": ("contact_s", "speed_way", "speed_way_fx", "fx_share_way", "card_speed_fx", "shots",
                    "fx_after_last_shot_s"),
       "suppress2": ("contact_s", "speed_way", "speed_way_fx", "speed_way_nofx", "fx_share_way", "shots",
                     "fx_after_last_shot_s"),
       "hitslow": ("contact_s", "speed_way", "a_hp_lost_way", "shots"),
       "censer": ("contact_s", "tg_hp_0_5", "tg_hp_5_15", "tg_hp_15_30", "a_hp_0_5", "a_hp_5_15", "tg_fx_first_s",
                  "a_stat_melee_attack", "a_stat_weapon_damage", "tg_mp_5", "tg_mp_10", "tg_mp_5_30", "rout_s",
                  "tg_fx_after_clear_s", "leave_clear_s"),
       "wolfrats": ("leg1_cruise", "leg1_to90_s", "leg2_cruise", "contact_s", "speed_peak30", "speed_last30",
                    "speed_peak_way", "tg_hp_0_1", "tg_hp_0_2", "tg_hp_0_5", "tg_hp_5_15", "tg_hp_steady", "a_hp_0_5",
                    "a_hp_steady", "first_tg_0_3", "re_tg_0_3", "chase_tg_hp_per_s", "chase_tg_men_lost"),
       "wolfpoison": ("contact_s", "tg_hp_0_5", "tg_hp_5_15", "tg_hp_15_30", "a_hp_0_5", "a_hp_5_15", "a_hp_15_30",
                      "tg_fx_first_s", "tg_scalar_speed_fight", "tg_stat_weapon_damage_fight", "tg_fx_after_clear_s",
                      "leave_clear_s", "tg_speed_chase")}


def measure_lanes(lanes):
    rows = []
    for lane in lanes:
        if lane.get("plan") in MEASURE:
            r = MEASURE[lane["plan"]](lane)
            r.update(plan=lane["plan"], battle=lane.get("battle"), lane=lane["spec"]["name"])
            rows.append(r)
    return rows


# the plans whose lanes stand on a map (their twin: the map's heights, ground and trunks)
from sim.maps import MAPS  # noqa: E402
MAP_TWIN = {"snipe": MAPS / "ksl_plains_01" / "heights.npz",
            "forestwalk": MAPS / "ksl_plains_01" / "heights.npz"}


def twin(lanes, device="cpu", copies=4, params=None):
    """The simulator on the recorded lanes (game/probe/charge_probe.sim_lanes: each lane its own battle, its units at
    their places at the go, the same scripted orders), `copies` a lane, in the game's shape with each lane's plan."""
    out = []
    for plan in MEASURE:
        mine = [ln for ln in lanes if ln.get("plan") == plan]
        if not mine:
            continue
        mid = None
        if plan in MAP_TWIN:
            # (the snipe plan) the lanes stand on Kislev: its heights, ground and trunks (sim/terrain.py)
            from sim import terrain as sim_terrain
            mid = sim_terrain.load_npz(MAP_TWIN[plan])
        for ln, s in zip([ln for ln in mine for _ in range(copies)],
                         cp.sim_lanes(mine, params=params, device=device, copies=copies, terrain_map=mid)):
            s.update(plan=plan, battle=ln.get("battle"))
            out.append(s)
    return out


def _side_by_side(game, sim):
    """Prints game | sim of the KEY numbers per plan and kind (means; the game's min-max)."""
    for plan, keys in KEY.items():
        kinds = sorted({k for p, k in game if p == plan}, key=str)
        if not kinds:
            continue
        print(f"== {plan} (game | sim)")
        for kind in kinds:
            g, s = game.get((plan, kind), {}), sim.get((plan, kind), {})
            cells = []
            for f in keys:
                if f not in g and f not in s:
                    continue
                gs = _fmt(g[f]) if f in g else "-"
                ss = _fmt((s[f][0], s[f][0], s[f][0], 1)) if isinstance(s.get(f), tuple) else (
                    "/".join(s[f]) if f in s else "-")
                cells.append(f"{f} {gs} | {ss}")
            print(f"  {kind}: " + "; ".join(cells))


def report(run_dirs, out=None, sim=False, device="cpu", copies=4):
    """The five plans' lanes in the runs: one row per lane, the means (min-max) per plan and kind; prints the tables,
    writes build/charge-probe/newunits.json (not in Git). sim: the simulator's twin on the same lanes beside them
    (newunits_sim.json)."""
    lanes = [ln for d in run_dirs for ln in cp.load_run(d) if ln.get("plan") in MEASURE]
    rows = measure_lanes(lanes)
    table = summary(rows)
    for plan in MEASURE:
        keys = [k for k in table if k[0] == plan]
        if not keys:
            continue
        print(f"== {plan}")
        for key in sorted(keys, key=lambda k: str(k[1])):
            r = table[key]
            print(f"  {key[1]} (n {r['n']}): " + "; ".join(f"{f} {_fmt(v)}" for f, v in r.items() if f != "n"))
    result = {"lanes": rows, "summary": [dict({k: v for k, v in r.items()}, plan=p, kind=k) for (p, k), r in table.items()]}
    if sim:
        srows = measure_lanes(twin(lanes, device=device, copies=copies))
        stable = summary(srows)
        _side_by_side(table, stable)
        result["sim"] = {"lanes": srows, "summary": [dict(r, plan=p, kind=k) for (p, k), r in stable.items()]}
    out = out or cp.ROOT / ("newunits_sim.json" if sim else "newunits.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(out)
    return result


def _plan_of(run_dir):
    try:
        return json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))["config"].get("plan")
    except (OSError, ValueError, KeyError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path, help="run folders (default: build/charge-probe/runs/*)")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sim", action="store_true", help="the simulator's twin on the same lanes beside the game's "
                                                           "(torch; game | sim per plan and kind)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--copies", type=int, default=4)
    parser.add_argument("--flank2v1", choices=("game", "layout"),
                        help="the flank2v1 plan's table: 'game' the runs (+ the twin on their starts with --sim), "
                             "'layout' the twin on the plan's own places (no game run needed)")
    for plan in ("counterfar", "intercept"):
        parser.add_argument(f"--{plan}", choices=("game", "layout"),
                            help=f"the meetfar probe's {plan} table: 'game' the runs (+ the twin on their starts with "
                                 "--sim), 'layout' the twin on the plan's own places (the predictions, no game run)")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    for plan in ("counterfar", "intercept"):
        if getattr(args, plan):
            runs = args.runs or [d for d in cp.runs() if _plan_of(d) == plan]
            mf_report(plan, runs, source=getattr(args, plan), sim=args.sim, copies=args.copies, device=args.device,
                      out=args.out)
            return 0
    if args.flank2v1:
        runs = args.runs or [d for d in cp.runs() if _plan_of(d) == "flank2v1"]
        flank2v1_report(runs, source=args.flank2v1, sim=args.sim, copies=args.copies, device=args.device, out=args.out)
        return 0
    report(args.runs or cp.runs(), out=args.out, sim=args.sim, device=args.device, copies=args.copies)
    return 0


if __name__ == "__main__":
    sys.exit(main())
