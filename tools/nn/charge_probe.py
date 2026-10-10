"""The melee probe (src/entries/charge_probe.lua): one melee indicator, one scenario, in the game and in
the simulator alike. Lanes far apart, in each an attacker and a target gap_m apart (front to front),
everyone held by script and fearless; the attacker gets one kind of order (attack at a run or at a
walk, a move order into the target, a recharge), the target stands (answering with an attack order at
contact), holds (braced, never ordered), attacks too, or stands facing away.

Plans (each a few battles of 2-5 lanes; lanes swap places between battles):
  charge  A: swordsmen -> clanrats, the four orders (4 battles); B: clanrats -> braced spearmen
          (charge reflection: run / walk / spears facing away / swordsmen without reflection; 2);
          C: the General and the Warlord charging, and the charge-speed rush: spearmen attacking
          skavenslave spearmen from 150 m and from 20 m (2 battles) - 8 battles;
  hit     formation pairs over a span of attack - defence (swordsmen, greatswords, spearmen with
          shields on unarmoured skavenslaves, flagellants on clanrats) and clanrat spearmen on Empire
          spearmen with the General behind them using Stand Your Ground at contact (battle 1) or not
          (battle 2, the control) - 2 battles;
  move    turning and leaving melee (build/movelords): battle 1 turns - spearmen turn in place 90 deg and
          back, then 180 deg (a facing order); spearmen run to a point 150 m behind them and 120 m to their
          side; the General and the Warlord turn in place 180 and 90 deg, then run to a point behind them
          (soldier places all the way); battle 2 melee exit - swordsmen / spearmen withdraw 10 s after contact
          and never change the order, chased by clanrats ordered to attack them or left standing; the General
          attacks clanrats and uses Foe-Seeker 50 s after contact (vigour) - 2 battles;
  vv      T-E3 of the effects (build/effects/spec.md 5): the Warlord attacks swordsmen and uses Verminous Valour
          20 s after contact (its 25 m blast: are the men around him thrown back, do they stop striking?); the
          soldiers' places all the fight; the control is the charge plan's Warlord lane - 1 battle;
  syg2    melee P1 + P2 (build/open_melee/spec.md): clanrat spearmen -> Empire spearmen with the General 8 m
          behind casting Stand Your Ground when the centres are 25 m + the half depths apart, the same pair without
          him (control); flagellants -> clanrats, swordsmen -> clanrats, clanrats -> held flagellants (who deals
          what with flagellants) - 2 battles, lanes rotated;
  pair    melee P3: clanrat spearmen attack Empire spearmen from 80 m for 240 s; the spearmen hold / walk to a far
          point through the attacker (push: the game's planner) / answer with an attack order / hold - 1 battle;
  fatleave  fatigue and the leaver: swordsmen <-> clanrats both attacking from 3 m (no run-up, 150 s), the same
          with the swordsmen 60 m wide (2 ranks) on clanrats 15 m wide (deep); swordsmen, spearmen, greatswords
          leaving melee 10 s after contact from held clanrats (no chase) - 1 battle;
  fresh   a new order in melee (OPEN, build/v2gap: the network's fresh orders on a near target killed 0.20/s in the
          game against 0.38 in the simulator): swordsmen and clanrats attacking each other from 3 m, a second clanrat
          unit standing 4 m beside the clanrats (never ordered; in every lane alike); 10 s after contact the
          swordsmen get (a) an attack on the second unit, (b) the same attack again, (c) a halt (the bridge's hold),
          (d) a move 5 m back at a walk, (e) nothing (the control) - 2 battles, lanes rotated;
  meleeorders  hold and move in melee (OPEN after fresh: the network's units in melee killed 0.08/s under hold and
          0.03/s under move in the game, 0.21 / 0.14 in the simulator): swordsmen <-> clanrats attacking each other
          from 3 m; 10 s after contact the swordsmen get a halt (the bridge's hold), a move at a run to a point 5 / 15
          / 40 m ahead (through the clanrats) or 5 / 15 m aside, or nothing (the control) - 2 battles of 5 lanes
          (battle 1: halt, ahead 5 / 15 / 40, control; battle 2: aside 5 / 15, ahead 15, halt, control);
  damaged  damaged units in melee and the under-fire flag (build/routgap: the twin's battered units lose health
          1.5-2.7x faster than the game's, fresh ones slower; at 30 % health the twin keeps 85-92 men where the game
          keeps 63-74): swordsmen <-> clanrats from 3 m with the clanrats at 100 % or brought to 30 % before the go -
          the engine's unit:reduce_hitpoints_unary(0.7) ('reduce') or unit:kill_number_of_men(70 %) ('kill');
          battle 1: 1 v 1 at 100 / 30 reduce / 30 kill, and crossbowmen shooting held clanrats with a second clanrat
          unit 15 / 40 m (edge to edge) beside them (is_under_missile_attack of the bystander); battle 2: two clanrat
          units on the swordsmen (the second attacks from the side) at 100 / 30 reduce / 30 kill, 1 v 1 30 reduce and
          100 again;
  reform  how a formation re-forms as it loses men (build/v2gap: the twin's units fight with the same front at any
          strength, the game's thin out): the soldiers' places every second for the whole fight (men_after_s 260),
          240 s fights from 3 m (both attack): swordsmen v clanrats 1 v 1 (twice), v clanrats brought to 30 % before the
          go (kill_number_of_men), Empire spearmen v clanrats, and two clanrat units on one swordsmen unit - 1 battle;
  reform2 what makes a unit's blows fall as the fight goes on (reform: the clanrats' fell with their men, the
          swordsmen's did not): swordsmen brought to 30 % (kill_number_of_men) v full clanrats (the striker's own losses
          against the target's), greatswords v clanrats (another blow), clanrats 15 m wide (deep) and 50 m wide
          (shallow) v swordsmen (the depth), and the swordsmen v clanrats control - 240 s from 3 m, the soldiers' places
          all fight - 1 battle;
  defender  the order of a unit attacked in melee (build/shotgap/hold_pair.py, the replay of the it1-it9 gate battles: in
          clean 1 v 1 contacts the network's unit under hold / an attack on another unit / a move / a withdraw lost 0.62 /
          0.57 / 0.53 / 0.51 % of its health a second in the game against 0.24 / 0.28 / 0.28 / 0.30 in the simulator,
          under an attack on the enemy it touched 0.41 against 0.28 - while its enemy lost about the same): swordsmen and
          clanrats attacking each other from 3 m, the clanrats never re-ordered (they keep attacking the swordsmen);
          1 s after contact the swordsmen get (a) the same attack again, (b) a halt (the bridge's hold), (c) an attack on
          a second clanrat unit 150 m aside (edge to edge, towards the middle of the field), (d) a move at a run 30 m
          aside; 90 s fights, the soldiers' places all fight; lanes 320 m apart - 2 battles of 4 lanes, rotated;
  routmob the routing mob's shape (build/fable/reach.py: in the game a router loses 0.10-0.17 % of its health a second
          with the nearest enemy's centre 12-35 m off and nobody targeting it, the chaser's melee flag on 13-19 %: the
          strung-out tail is struck while the centre is far; the simulator's router is its formation box, 6 m along the
          flight): the target keeps its morale (t_morale) and is attacked by fearless spearmen / swordsmen until it
          routs; at the rout the attacker goes on attacking (at_rout 'none': the chase) or halts (at_rout 'halt': the
          mob's own stretch); the soldiers' places every second to 40 s after the rout (men_after_rout_s), wherever the
          enemy is: skavenslaves v spearmen (chase / halt), clanrats v swordsmen (chase / halt) - 2 battles, lanes
          rotated; the lane ends 45 s after the rout;
  routmob2 what bleeds a router far from its chaser (build/fable/passby.py: in the game a router with no enemy formation
          within 12 m and nobody targeting it loses 0.08 % of its health a second, 0.035 beyond 30 m; the twin 0.039 /
          0.013): skavenslaves with half their men killed at placement (damage kill 0.5), normal morale, attacked by
          fearless spearmen from 40 m; if they have not routed rout_at_s after the contact they are routed by script
          (morale_behavior_rout, probe_phase 'rout_forced'); lanes: the chase (at_rout none), missiles (the attacker
          halts at the rout; crossbowmen of its side 80 m beyond the target, fire at will), neighbours (halts; two
          spearmen units of its side 25 m to either side and 20 m behind the target, halted, striking only what touches
          them), the control (halts, nobody else); soldiers' places 40 s after the rout - 2 battles, lanes rotated;
  rallysecure how 'flanks secure' comes back after a rally (the game: the rallied unit far from enemies gains +1/+2/+3/+4
          /+7 morale 3/10/20/30/45 s after the rally, the simulator +5 at once): skavenslaves with half their men killed
          at placement (damage kill 0.5), normal morale, held (never ordered), attacked by fearless spearmen from 40 m
          until they rout by themselves (the engine's morale_behavior_rout shatters at once - lord_fall: is_shattered
          true at the call - so it cannot be rallied from); at the rout the spearmen are teleported 400 m away (at_rout
          'away': no chase, no enemy within 200 m); at the rally the target halts and, by lane, nothing more ('alone'),
          two fearless clanrats of its side teleported 35 m (centre to centre) to its left and right ('neighbours',
          rally_friends), its Warlord teleported 20 m behind it ('lord'); 'control': the same damaged slaves never
          attacked, the spearmen standing 300 m off; every 0.5 s the target's MoralePercent, MoraleState,
          MoraleGreatestEffect, ActiveEffectList, routing / wavering / shattered and place, to 60 s after the rally
          (after_rally_s) - 2 battles of 4 lanes 300 m apart, rotated; no simulator twin yet;
  wavemiss the opening wave against shooters (build/routmorale: in the it1-it9 gate battles a shooter unit in melee loses
          0.54-0.69 % of its health a second in the game whatever its time in melee - x1.28 -> x1 from 0-10 s to 60+ s -
          while the replay's falls x1.95 -> x1, the opening wave measured formation on formation: do the fronts part from
          a loose shooter block as they do from swordsmen, or do the attackers stay inside it?): clanrats charge standing
          crossbowmen / archers / handgunners and swordsmen charge standing night runners, from 30 m at a run; the
          shooters halt and are never ordered ('hold', they fight back on their own) or answer with an attack at contact
          ('stand'); 90 s, the soldiers' places all fight; battle 2 swaps hold / stand and rotates the lanes - 2 battles
          of 4 lanes;
  wavemiss2 the shooters' orders in melee (wavemiss: a shooter that answers with an attack at contact keeps losing
          0.7-0.8 % a second with the fronts 2.1-2.9 m apart and 25-28 of its men within reach, one never ordered falls
          from 1.8 to 0.55 as the fronts part to 3.3 m; one lane a shooter and order): 'stand' lanes as wavemiss's
          (clanrats / swordsmen charge the shooters from 30 m at a run, the shooters answer with an attack at contact);
          'move' lanes - the shooters stand unordered, the melee unit charges them from 30 m at a run, and 3 s after the
          contact the shooters get a move at a run 40 m straight back (the leaver: is it held, struck as it goes?);
          90 s, the soldiers' places all fight; battle 2 swaps stand / move and rotates the lanes - 2 battles of 4 lanes;
  dmgmelee  damaged units one on one (the twin's units below half health, most below 0.3, lose health 1.2-1.8x faster
          in 1 v 1 melee than the game's - the skaven more, the Empire less - while fresh ones match; does the game let
          fewer enemy men strike a small unit, those with a target within reach, where the twin strikes with the whole
          front?): both attack from 3 m, 240 s, fearless (no rout, the lane runs to the end): (1) greatswords v
          stormvermin with shields, both whole; (2) the same, the stormvermin brought to 30 % of their men before the go
          (kill_number_of_men, as the damaged plan); (3) the same, the greatswords at 30 %; (4) spearmen with shields v
          clanrats with shields; (5) the Empire General v the stormvermin. Every 0.5 s both units' health, men, morale
          (morale_rows: MoralePercent, MoraleState, wavering), fatigue, melee flag, kills; every 1 s both units' soldier
          places, all fight (men_after_s 260). Lanes on a 3 + 2 grid (DMGMELEE_PLACES: 300 m apart in a row, rows 400 m
          apart: 5 lanes 250+ m apart do not fit in one row inside the map's |x| <= 480) - 2 battles, lanes rotated;
  reengage  the opening wave of a second contact (sim.json melee.wave was measured on fresh pairs only; in the gate battles
          tired units back in melee after 10+ s out of it lose little more in their first 10 s than later: x1.46 above half
          health, x1.12 at or below it, fresh x2.44): both attack each other at a run from 30 m, fearless; (1) swordsmen v
          clanrats and (2) greatswords v stormvermin with shields (mode 'reengage'): they fight 150 s (part_after_s), then
          both get a move 30 m straight back at a run (part_m, probe_phase 'out'; 'clear' once neither is in melee), 15 s
          later (part_s) both attack each other at a run again ('back'; contact 2 = the first melee flag after 'clear'),
          the lane ends 60 s after contact 2 (after2_s); (3) the control: a fresh pair swordsmen v clanrats from 30 m, 60 s;
          (4) the same pair tired by running, no fight (mode 'tire'): both shuttle at a run straight back tire_leg_m 100 m
          and to their start until both are very tired by the game's own state (tire_until; the database's running +4 a
          tick reaches threshold_very_tired 18000 in ~450 s; tire_max_s 540 caps it, probe_phase 'tire_end'), return to
          their places facing each other ('tired' when there or after ready_max_s) and attack from 30 m, 60 s. Every 0.5 s
          both units' health, men, fatigue state, melee flag, kills; every 1 s the soldier places, all fight (men_after_s
          260; not while tiring). 4 lanes 300 m apart in one row - 2 battles, lanes rotated;
  retarget a shooter's new target (build/fable2 k_reaim, g_volley: in the gate battles a standing shooter given an attack on
          another unit is silent ~5 s (median), then fires a few men at a time, and those first shots hit about half as well;
          whole volleys are few; the simulator's aim reset on an order, missile.aim_reset_on_order 3.3 s, hides in the reload
          and its first volley on the new target is whole and accurate; the network re-orders its shooters every ~11 s):
          a shooter standing between two held fearless skavenslave units ~98 m off (centre to centre; 80 m front to
          front; the two 30 m apart edge to edge, a_dx puts it between them); 'A' one ranged attack on target 1 at the go for 120 s, 'B10' / 'C5' a ranged
          attack on the other target every 10 / 5 s (probe_phase 'aim'); crossbowmen A / B10 / C5 and handgunners A / B10,
          fire at will on (as the bridge). Every 0.5 s the shooter's ammo (its drop: the shots), fire flag, damage dealt,
          place, bearing, and both targets' health and men. Lanes on the dmgmelee grid (RETARGET_PLACES) - 2 battles,
          lanes rotated; 'retarget' prints the table (--sim: + the simulator on the same lanes);
  retarget2 the same lanes for more shooters and switches (RETARGET2_LANES): crossbowmen every 3 / 7 s and one order
          (the control), archers one order / every 5 s, handgunners every 5 s, the Skaven's slingers and Night Runners
          (throwing stars) one order / every 5 s on two held fearless Empire spearmen ~70 % of their range off (slings
          ~84 m, stars ~49 m centre to centre); 10 lanes a battle (RETARGET2_PLACES), 2 battles, lanes rotated; the
          'retarget' table reads its runs too;
  leave   the exit from melee (build/nightdip: the simulator's leaving unit is out of contact in ~2 s, the game's in
          4-4.5 s): a pair fights from 3 m (both attack) 30 s after the first contact, then the attacker withdraws -
          a move 60 m straight back at a run, as the bridge gives the network's withdraw - and the lane ends 40 s
          later; the target keeps its attack (chase) or halts at the withdraw (stand): swordsmen from clanrats chase /
          stand, shield spearmen from stormvermin chase / stand, the Empire General from clanrats chase, and a
          control pair that never leaves; every 0.5 s both units' health, men, melee flag, place, bearing (speed from
          the places), the soldiers' places every 1 s to 95 s after the contact within 150 m; 6 lanes, 2 battles,
          lanes rotated; --sim: the simulator's withdraw (orders.WITHDRAW) on the same lanes;
  retarget3 is the switch cost the formation's turn? (RETARGET3_LANES): crossbowmen C5 / B7 and archers C5 with the
          second target right behind the first on the line of fire ('ray', ~98 / ~122 m) or 60 m across ('side'),
          and a crossbow A control; the shooter's bearing every 0.5 s; 7 lanes, 2 battles, lanes rotated;
  wave    the opening wave (build/shotgap/wave_geom.py: men within 2.5 m of an enemy 46-51 in the first 5 s, 12 from 20 s,
          the fronts 1.3 m apart then 2.9 m; is it the collision of the approach or the start inside reach?): swordsmen
          <-> clanrats, both attacking, 90 s, the soldiers' places all fight - (1) placed 1 m apart (front to front), both
          at a walk; (2) 2.5 m apart (the database's formed combat distance), both at a walk; (3) 3 m, both at a walk;
          (4) 3 m, the clanrats at a run (the earlier probes' set-up, the control); (5) 30 m, the clanrats at a run (a
          real charge), the swordsmen at a walk - 2 battles, lanes rotated.
  skirmish the game's skirmish mode (a shooter steps back from an approaching enemy by itself, over its order; the
          simulator has none; build/bench2/zo_chase: the game AI's shooters under the bridge get away from a chase even
          under 'hold'): a shooter stands - halted, fire at will on and aimed at the chaser (as the bridge aims a held
          shooter) - and melee infantry attacks it at a run from 60 m (front to front), 90 s; the mode on or off by
          script (change_behaviour_active) for the Skaven's slave slingers and Night Runners (chased by swordsmen),
          the Empire's archers, crossbowmen and handgunners (by clanrats): 10 lanes, 2 battles, lanes rotated. Every
          0.5 s both units' place, bearing, melee flag, health, men, and the shooter's ammo, fire flag, damage dealt and
          the mode's flag (sk: unit:is_behaviour_active); the soldiers' places every 1 s within 200 m;
          probe_skirmish: the mode before and after it was set. 'skirmish' prints the table: when the shooter starts
          moving away (the distance), its direction and speed, its fire and shots on the way, caught or not.
  skirmish2 what sets the skirmish mode off (build/skirm: it starts at ~40 m for the Empire's shooters, ~33 m for the
          Skaven's; the shooter runs straight away, silent, and shoots again 70-105 m off; the Empire's speed up to x1.6
          on a long run): slave slingers, archers and crossbowmen with the mode on, standing as in 'skirmish'; four lane
          kinds - 'pass': enemy infantry (12 m front) runs past the shooter, centre 30 m beside it (edge ~9 m), to a
          point 100 m behind its front, never attacking it; 'neighbour': enemy infantry attacks the shooter's
          neighbour (a unit of its side, 20 m front, centre 30 m beside it); 'control': the infantry attacks the
          shooter (as 'skirmish'); 60 s each; 'long': fast infantry (clanrats 4.2 m/s; flagellants 3.6 for the
          slingers) chases the shooter for 120 s from 60 m in a long lane (~1080 m of runway): the speed-up and its
          cause (the samples' fatigue and run flags). 12 lanes a battle (the short ones on two rows of 5 columns 150 m
          apart, the long ones in three columns of their own), 2 battles, lanes rotated. The 'skirmish' table reads
          them per shooter and kind (+ speed per 30 s and fatigue for the long ones).
  shootcontact does a shooter that moves in contact lose as one standing? (the 72 gate battles' replay: a moving
          shooter in melee loses 0.47-0.58 %/s in the game, about half that in the replay): archers and crossbowmen,
          each against clanrats one on one, the clanrats attacking at a run from 30 m (front to front), 90 s; four
          lane kinds of the shooter (target_mode 'skirmish': halted, fire at will on, the rows with ammo, fire, sk) -
          'stand' (mode off, a ranged attack on the clanrats at a walk: as the bridge holds a shooter), 'walk' / 'run'
          (mode off, at the go a move 300 m straight away from them at a walk / a run: lane.t_move, never changed) and
          'skirmish' (the mode on, aimed as 'stand'); every 0.5 s both units' place, melee flag, health, men, the
          shooter's ammo; the soldiers' places every 1 s within 200 m; 8 lanes on the skirmish plan's places, 2 battles,
          the rows swapped. 'shootcontact' prints per shooter and kind the shooter's HP lost %/s in melee, moving (the
          game's is_moving flag) and not, the clanrats' HP lost to it and the shots.

    python -m tools.nn.charge_probe plan [--plan charge|hit]          # the battles
    python -m tools.build charge-probe --probe-plan hit --probe-battle 1   # one battle's build
    python -m tools.nn.charge_probe run --plan hit [--battles 1,2]      # build + launch each, in turn
    python -m tools.nn.charge_probe report [runs...]                    # the game's table
    python -m tools.nn.charge_probe report --sim                        # + the simulator on the same lanes
    python -m tools.nn.charge_probe retarget [runs...] [--sim]          # the retarget plan's table
    python -m tools.nn.charge_probe run --plan retarget2                # 10 more shooter lanes (the same table)
    python -m tools.nn.charge_probe run --plan retarget3                # targets in line / across (the turn?)
    python -m tools.nn.charge_probe run --plan leave                    # the exit from melee (report --sim)
    python -m tools.nn.charge_probe run --plan skirmish                 # the shooters' skirmish mode
    python -m tools.nn.charge_probe skirmish [runs...]                  # its table (skirmish2 too)
    python -m tools.nn.charge_probe run --plan skirmish2                # what triggers the mode, the long run
    python -m tools.nn.charge_probe run --plan shootcontact             # a shooter moving in melee
    python -m tools.nn.charge_probe shootcontact [runs...]              # its table

Measures per lane (the game's recording, and the simulator's run of the same lane from the same start):
the attacker's speed on the way in (the last 30 m, the peak), the first contact; HP lost by the target
and by the attacker in 0-1, 0-2, 0-5, 5-15, 15-30 s after contact and per second from 15 s to the end
(steady); men lost per second steady; for a recharge the same after the second contact; from the
soldiers' places (game only) how many men of each side stand within 1.5 / 2.5 / 3.5 m of an enemy
soldier, per second after contact. Writes build/charge-probe/analysis.json (not in Git).
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from tools import config as project
from tools.nn import scenario as nn_scenario

ROOT = project.BUILD / "charge-probe"
RUNS = ROOT / "runs"
OUT = ROOT / "analysis.json"
LAUNCHER = project.ROOT / "tools" / "launcher"
UNITS_JSON = project.ROOT / "config" / "nn" / "units.json"
EMP, SKV = "wh_main_emp_empire", "wh2_main_skv_skaven"
# short name -> (unit key, men, faction)
UNITS = {
    "swords": ("wh_main_emp_inf_swordsmen", 120, EMP),
    "gs": ("wh_main_emp_inf_greatswords", 120, EMP),
    "spear": ("wh_main_emp_inf_spearmen_0", 120, EMP),
    "spearsh": ("wh_main_emp_inf_spearmen_1", 120, EMP),
    "flag": ("wh_dlc04_emp_inf_flagellants_0", 120, EMP),
    "general": ("wh_main_emp_cha_general_0", 1, EMP),
    "xbow": ("wh_main_emp_inf_crossbowmen", 90, EMP),
    "archer": ("wh2_dlc13_emp_inf_archers_0", 90, EMP),
    "hgun": ("wh_main_emp_inf_handgunners", 90, EMP),
    "slinger": ("wh2_main_skv_inf_skavenslave_slingers_0", 140, SKV),
    "nrun": ("wh2_main_skv_inf_night_runners_0", 120, SKV),
    "clanrat": ("wh2_main_skv_inf_clanrats_1", 160, SKV),
    "cspear": ("wh2_main_skv_inf_clanrat_spearmen_0", 160, SKV),
    "slave": ("wh2_main_skv_inf_skavenslaves_0", 180, SKV),
    "sspear": ("wh2_main_skv_inf_skavenslave_spearmen_0", 180, SKV),
    "warlord": ("wh2_main_skv_cha_warlord_0", 1, SKV),
    "svsh": ("wh2_main_skv_inf_stormvermin_1", 160, SKV),     # stormvermin with sword and shield
}
LORDS = {EMP: "general", SKV: "warlord"}
SHORT = {key: short for short, (key, _, _) in UNITS.items()}
FACTION = {key: faction for key, _, faction in UNITS.values()}
SYG = "wh_main_character_abilities_stand_your_ground"
FOE_SEEKER = "wh_main_character_abilities_foe_seeker"
WIDTH_M = 30
LANE_DX = 240          # lanes this far apart across x (the General's auras reach 35-40 m)
GAP_M = 80
SETTLE_MS = 4000
TICK_MS = 500
MEN_MS = 1000
WINDOWS = ((0, 1), (0, 2), (0, 5), (0, 10), (5, 15), (15, 30))
WINDOWS2 = ((0, 1), (0, 3), (0, 5), (0, 10), (5, 15), (10, 20))   # after the second contact
STEADY_FROM_S = 15.0   # the charge bonus fades over 13 s (charge_decay_duration)
RADII = (1.5, 2.5, 3.5)
NEAR_WINDOWS = ((0, 5), (5, 15), (15, 30), (30, 90))
NEAR_KEYS = tuple(f"{who}_near_{lo}_{hi}" for who in ("a", "tg") for lo, hi in NEAR_WINDOWS)
PLANS = ("charge", "hit", "move", "vv", "syg2", "pair", "fatleave", "fresh", "meleeorders", "damaged", "reform",
         "reform2", "defender", "wave", "routmob", "routmob2", "wavemiss", "wavemiss2", "rallysecure",
         "dmgmelee", "reengage", "retarget", "retarget2", "retarget3", "leave", "skirmish", "skirmish2", "shootcontact")
# the wavemiss plan: (attacker, shooter target) of its lanes; battle 1's shooters hold / stand / hold / stand, battle 2's
# the other way round
WAVEMISS = (("clanrat", "xbow"), ("clanrat", "archer"), ("clanrat", "hgun"), ("swords", "nrun"))
ROUTMOB2_EXTRAS = {"fire": [{"short": "xbow", "dx": 0, "dz": 80, "fire": True}],
                   "neighbours": [{"short": "spear", "dx": -25, "dz": -20}, {"short": "spear", "dx": 25, "dz": -20}]}
# the rallysecure plan: units of the target's side teleported beside the rallied target (dx to its right, dz ahead,
# from its centre and facing at the rally; entries/charge_probe.lua rally_friends), parked far until then
RALLYSECURE_FRIENDS = {"neighbours": [{"short": "clanrat", "dx": -35, "dz": 0, "width": 25},
                                      {"short": "clanrat", "dx": 35, "dz": 0, "width": 25}],
                       "lord": [{"short": "warlord", "dx": 0, "dz": -20, "width": 5}]}
RALLYSECURE_LANE_DX = 300   # lanes 300 m apart (x -450 .. 450)
RALLYSECURE_AWAY_DZ = 400   # at the rout the attacker is teleported this far beyond the target's start (+z)
RALLYSECURE_PARK_Z = 600    # the rally friends wait here (+z, beyond the attackers) until the rally
RALLYSECURE_AFTER_S = 60    # the lane ends this long after the rally
# the dmgmelee plan: the lane places (x, z of the target's front), by lane index - a row of 3 300 m apart and a row of 2
# 400 m behind it (the nearest lanes of two rows 427 m apart); the lanes rotate over the places between battles
DMGMELEE_PLACES = ((-300, -200), (0, -200), (300, -200), (-150, 200), (150, 200))
# the reengage plan: the second contact (lane mode 'reengage') and the pair tired by running (mode 'tire');
# entries/charge_probe.lua
REENGAGE = {"part_after_s": 150, "part_m": 30, "part_s": 15, "after2_s": 60}
REENGAGE_TIRE = {"tire_leg_m": 100, "tire_until": "threshold_very_tired", "tire_max_s": 540, "ready_max_s": 40}
REENGAGE_LANE_DX = 300   # 4 lanes 300 m apart (x -450 .. 450); every unit moves along its lane (z) only
# the retarget plan: (shooter, seconds between the switches of its target or None: one order) by lane, the lane
# names, the timing, the places (the dmgmelee grid: lanes 300+ m apart, x and z within 480 m)
RETARGET_LANES = (("xbow", None), ("xbow", 10), ("xbow", 5), ("hgun", None), ("hgun", 10))
RETARGET_KIND = {None: "A", 10: "B10", 5: "C5", 7: "B7", 3: "C3"}
# the retarget2 plan: the crossbows' rule tried at other switches (3 / 7 s), bows, handguns every 5 s, the Skaven's
# slings and throwing stars (one order / every 5 s, on held fearless Empire spearmen without shields), a crossbow A
# control (the spread between runs); 10 lanes a battle on a 4 x 3 grid (places 400 m apart in z, 290 / 300 m in x: every
# target 250+ m from the other lanes' shooters, |x| <= 440 with the second target inward), 2 battles, the lanes rotated
RETARGET2_LANES = (("xbow", 3), ("xbow", 7), ("archer", None), ("archer", 5), ("hgun", 5), ("xbow", None),
                   ("slinger", None), ("slinger", 5), ("nrun", None), ("nrun", 5))
RETARGET2_TARGET = {"slinger": "spear", "nrun": "spear"}        # else the skavenslaves
# centre to centre to each target ~70 % of the range for the Skaven (slings 120 m -> 84, stars 70 m -> 49); the others
# keep the retarget plan's 80 m front to front (~98 m)
RETARGET2_RANGE_SHARE = {"slinger": 0.7, "nrun": 0.7}
RETARGET2_PLACES = tuple((x, z) for z in (-430, -30, 370) for x in (-440, -150, 150, 440))[:10]
# the retarget3 plan: does the switch cost come from turning the formation between the targets? (shooter, switch_s,
# geometry): 'ray' - the second target right behind the first on the line of fire (centres ~98 and ~122 m, 6 m between
# the blocks: no turn needed), 'side' - the two 60 m apart across, the shooter between them (the retarget plan's, ~17 deg
# each side); crossbows C5 / B7 both ways, archers C5 both ways, a crossbow A control; the shooter's bearing (b) is in its
# 0.5 s sample row; 7 lanes on the retarget2 grid, 2 battles, lanes rotated
RETARGET3_LANES = (("xbow", 5, "ray"), ("xbow", 5, "side"), ("xbow", 7, "ray"), ("xbow", 7, "side"),
                   ("archer", 5, "ray"), ("archer", 5, "side"), ("xbow", None, "side"))
RETARGET3_RAY_GAP_M = 6
# the leave plan: a pair fights (both attack from gap_m) fight_s after the first contact, then the attacker withdraws
# (a move back_m straight back at a run - the bridge's withdraw: goto_location(point, run)); the lane ends after_s after
# that. (attacker, target, kind): 'chase' - the target's attack order stays (it follows); 'stand' - the target halts at
# the withdraw; 'control' - nobody leaves. Lanes on the retarget2 grid (300+ m apart), 2 battles, lanes rotated
LEAVE = {"gap_m": 3, "fight_s": 30, "back_m": 60, "after_s": 40}
LEAVE_LANES = (("swords", "clanrat", "chase"), ("swords", "clanrat", "stand"), ("spearsh", "svsh", "chase"),
               ("spearsh", "svsh", "stand"), ("general", "clanrat", "chase"), ("swords", "clanrat", "control"))
# the skirmish plan: (shooter, chaser) - the chaser attacks at a run from gap_m (front to front), the lane ends after max_s;
# each pair with the shooter's skirmish mode on and off. Two rows of 5 lanes 200 m apart, the rows 600 m apart in z and
# shifted 100 m in x (a shooter runs to -z, away from its chaser: the upper row's way passes between the lower row's
# lanes; up to ~360 m in 90 s), all within the map's flat free square (+-600 m). Battle 2: the rows swapped
SKIRMISH = {"gap_m": 60, "max_s": 90, "men_near_m": 200}
SKIRMISH_PAIRS = (("slinger", "swords"), ("archer", "clanrat"), ("xbow", "clanrat"), ("hgun", "clanrat"),
                  ("nrun", "swords"))
SKIRMISH_PLACES = tuple((float(x), 500.0) for x in (-400, -200, 0, 200, 400)) + \
    tuple((float(x), -100.0) for x in (-500, -300, -100, 100, 300))
# the skirmish2 plan: (shooter, infantry of the short lanes, the long lane's fast chaser, the shooter's neighbour); short
# lanes on two rows (z 470 / -80) of five columns 150 m apart (x -525..75), the long ones in the columns x 225 / 375 /
# 525 from z 480 down to the map's edge (-600: ~1080 m); battle 2 rotates the short lanes by 5 places, the long by 1
SKIRMISH2 = {"gap_m": 60, "short_s": 60, "long_s": 120, "side_m": 30, "pass_beyond_m": 100, "pass_w": 12,
             "neighbour_w": 20, "men_near_m": 200}
SKIRMISH2_SHOOTERS = (("slinger", "swords", "flag", "clanrat"), ("archer", "clanrat", "clanrat", "spear"),
                      ("xbow", "clanrat", "clanrat", "spear"))
SKIRMISH2_KINDS = ("pass", "neighbour", "control")
SKIRMISH2_SHORT = tuple((float(x), z) for z in (470.0, -80.0) for x in (-525, -375, -225, -75, 75))
SKIRMISH2_LONG = ((225.0, 480.0), (375.0, 480.0), (525.0, 480.0))
SKIRMISH_WINDOWS = ((0, 30), (30, 60), (60, 90), (90, 120))   # the long lanes' speed and fatigue windows, s
# the shootcontact plan: (shooter, infantry) one on one, the infantry attacking at a run from gap_m (front to front) for
# max_s; the shooter's kinds (target_mode 'skirmish'): 'stand' (mode off, aimed), 'walk' / 'run' (mode off, at the go a
# move back_m straight away, -z), 'skirmish' (mode on, aimed); on SKIRMISH_PLACES (a shooter's way back of up to ~300 m
# passes between the lower row's lanes), battle 2 with the rows swapped
SHOOTCONTACT = {"gap_m": 30, "max_s": 90, "back_m": 300, "men_near_m": 200}
SHOOTCONTACT_PAIRS = (("archer", "clanrat"), ("xbow", "clanrat"))
SHOOTCONTACT_KINDS = ("stand", "walk", "run", "skirmish")
SKIRMISH_MOVE_MPS = 1.0    # the shooter moves away: its speed away from the chaser above this (m/s) ...
SKIRMISH_MOVE_S = 1.0      # ... for this long
RETARGET3_RAY_D_M = 98.0        # as the 'side' lanes' (80 m front to front, 30 m aside)
RETARGET = {"fire_s": 120, "gap_m": 80, "t2_gap_m": 30}   # 80 m front to front: ~98 m centre to centre
RETARGET_PLACES = DMGMELEE_PLACES
RETARGET_LAG_S = 2.0       # a shot's flight (bolts 45 m/s, bullets 100 m/s over ~110 m) and the health readout's lag
ROUTMOB_MEN_S = 40     # the routmob plan: the soldiers' places this long after the target's rout
ROUTMOB_END_S = 45     # ... and the lane ends this long after it
# the reform plan: soldier places this long after contact (the whole fight)
REFORM_MEN_S = 260
# the fresh plan's orders 10 s after contact (entries/charge_probe.lua lane.after.kind)
FRESH = ("attack_t2", "attack_same", "halt", "move_near", "none")
# the defender plan's orders 1 s after contact (lane.after.kind; move_near 30 m aside at a run)
DEFENDER = ("attack_same", "halt", "attack_t2", "move_near")
DEFENDER_LANE_DX = 320  # (its second target stands 180 m aside, inward: 140 m from the next lane; |x| <= 480 as in
#                         the other probes)
VV = "wh2_main_character_abilities_verminous_valour"
TURN_TICK_MS = 250     # the turning battle: 0.25 s samples (a lord turns 180 deg in 1-2 s)


def lane(attacker, target, mode="attack_run", target_mode="stand", gap_m=GAP_M, fight_s=40, **extra):
    """One lane's spec (before it gets a place)."""
    out = {"attacker": attacker, "target": target, "mode": mode, "target_mode": target_mode, "gap_m": gap_m,
           "fight_s": fight_s, "answer": target_mode == "stand"}
    if mode == "recharge":
        out.update(recharge_after_s=10, back_m=40, recharge_max_s=25, fight_s=max(fight_s, 70))
    if mode == "withdraw":
        out.update(recharge_after_s=10, back_m=150, recharge_max_s=10 ** 6)
    out.update(extra)
    return out


def rotate(items, k):
    k %= len(items)
    return items[k:] + items[:k]


def battles(plan):
    """[[lane spec]] of a plan: each inner list one battle."""
    assert plan in PLANS, plan
    out = []
    if plan == "charge":
        a = [lane("swords", "clanrat", m) for m in ("attack_run", "attack_walk", "move_run", "recharge")]
        out += [rotate(a, k) for k in range(4)]
        b = [lane("clanrat", "spear", "attack_run", "hold"), lane("clanrat", "spear", "attack_walk", "hold"),
             lane("clanrat", "spear", "attack_run", "rear"), lane("clanrat", "swords", "attack_run", "hold")]
        out += [rotate(b, k) for k in (0, 2)]
        rush = [lane("spear", "sspear", "attack_run", gap_m=150), lane("spear", "sspear", "attack_run", gap_m=20)]
        out += [[lane("general", "clanrat", "attack_run"), lane("warlord", "swords", "attack_run")] + rush,
                [lane("general", "clanrat", "attack_walk"), lane("warlord", "swords", "recharge")] + rush[::-1]]
    elif plan == "move":
        # bearings: the attacker starts facing 180 (towards -z, its target 250 m away, never ordered)
        far = dict(target_mode="hold", gap_m=250, answer=False)
        spin = [{"at_s": 1, "kind": "face", "bearing": 270}, {"at_s": 16, "kind": "face", "bearing": 180},
                {"at_s": 31, "kind": "face", "bearing": 0}]
        lord_spin = [{"at_s": 1, "kind": "face", "bearing": 0}, {"at_s": 11, "kind": "face", "bearing": 270},
                     {"at_s": 21, "kind": "move", "dx": 100, "dz": 0, "run": True}]
        out.append([
            lane("spear", "clanrat", "script", steps=spin, men_all_s=50, max_s=50, **far),
            lane("spear", "slave", "script", steps=[{"at_s": 1, "kind": "move", "dx": 0, "dz": 150, "run": True}],
                 men_all_s=25, max_s=60, **far),
            lane("spear", "sspear", "script", steps=[{"at_s": 1, "kind": "move", "dx": 120, "dz": 0, "run": True}],
                 men_all_s=25, max_s=55, **far),
            lane("general", "clanrat", "script", steps=lord_spin, max_s=50, **far),
            lane("warlord", "swords", "script", steps=lord_spin, max_s=50, **far)])
        out.append([
            lane("swords", "clanrat", "withdraw", gap_m=40, fight_s=75),
            lane("swords", "clanrat", "withdraw", target_mode="hold", gap_m=40, fight_s=50),
            lane("spear", "clanrat", "withdraw", gap_m=40, fight_s=75),
            lane("general", "clanrat", "attack_run", fight_s=100, a_ability=FOE_SEEKER, a_ability_after_s=50)])
    elif plan == "syg2":
        syg = lane("cspear", "spear", gap_m=40, fight_s=60, lord={"name": "general", "ability": SYG, "dz": 8, "at_m": 25})
        base = [syg, lane("cspear", "spear", gap_m=40, fight_s=60), lane("flag", "clanrat", gap_m=40, fight_s=60),
                lane("swords", "clanrat", gap_m=40, fight_s=60), lane("clanrat", "flag", "attack_run", "hold", gap_m=40,
                                                                      fight_s=60)]
        out += [base, rotate(base, 2)]
    elif plan == "pair":
        p = dict(gap_m=80, fight_s=240)
        out.append([lane("cspear", "spear", target_mode="hold", **p), lane("cspear", "spear", target_mode="push", push_m=60, **p),
                    lane("cspear", "spear", target_mode="stand", **p), lane("cspear", "spear", target_mode="hold", **p)])
    elif plan == "fatleave":
        out.append([lane("swords", "clanrat", "attack_walk", "both", gap_m=3, fight_s=150),
                    lane("swords", "clanrat", "attack_walk", "both", gap_m=3, fight_s=150, a_w=60, t_w=15),
                    lane("swords", "clanrat", "withdraw", target_mode="hold", gap_m=40, fight_s=40),
                    lane("spear", "clanrat", "withdraw", target_mode="hold", gap_m=40, fight_s=40),
                    lane("gs", "clanrat", "withdraw", target_mode="hold", gap_m=40, fight_s=40)])
    elif plan == "fresh":
        base = [lane("swords", "clanrat", "attack_walk", "both", gap_m=3, fight_s=45, target2="clanrat", t2_gap_m=4,
                     after={"at_s": 10, "kind": kind, "dx": 0, "dz": 5, "walk": kind == "move_near"})
                for kind in FRESH]
        out += [base, rotate(base, 2)]
    elif plan == "meleeorders":
        # the attacker faces -z (its target): ahead = dz < 0 (through the clanrats), aside = dx > 0; at a run (the
        # network's move orders: 951 of 1054 at a run in the it1 battles)
        def mo(kind, dx=0, dz=0):
            return lane("swords", "clanrat", "attack_walk", "both", gap_m=3, fight_s=45,
                        after={"at_s": 10, "kind": kind, "dx": dx, "dz": dz, "walk": False})
        out.append([mo("halt"), mo("move_near", dz=-5), mo("move_near", dz=-15), mo("move_near", dz=-40), mo("none")])
        out.append([mo("move_near", dx=5), mo("move_near", dx=15), mo("move_near", dz=-15), mo("halt"), mo("none")])
    elif plan == "damaged":
        def dm(method):
            return None if method is None else {"method": method, "share": 0.7}

        def one(method):
            d = dm(method)
            return lane("swords", "clanrat", "attack_walk", "both", gap_m=3, fight_s=60, **({"damage": {"t": d}} if d else {}))

        def two(method):
            d = dm(method)
            return lane("swords", "clanrat", "attack_walk", "both", gap_m=3, fight_s=60, target2="clanrat", t2_gap_m=4,
                        t2_mode="attack", **({"damage": {"t": d, "t2": d}} if d else {}))

        def fire(gap):
            return lane("xbow", "clanrat", "shoot", "hold", gap_m=100, fight_s=30, answer=False, target2="clanrat",
                        t2_gap_m=gap, max_s=120)
        out.append([one(None), one("reduce"), one("kill"), fire(15), fire(40)])
        out.append([two(None), two("reduce"), two("kill"), one("reduce"), one(None)])
    elif plan == "reform":
        f = dict(gap_m=3, fight_s=240)
        out.append([lane("swords", "clanrat", "attack_walk", "both", **f),
                    lane("swords", "clanrat", "attack_walk", "both", damage={"t": {"method": "kill", "share": 0.7}}, **f),
                    lane("spear", "clanrat", "attack_walk", "both", **f),
                    lane("swords", "clanrat", "attack_walk", "both", target2="clanrat", t2_gap_m=4, t2_mode="attack", **f),
                    lane("swords", "clanrat", "attack_walk", "both", **f)])
    elif plan == "reform2":
        f = dict(gap_m=3, fight_s=240)
        out.append([lane("swords", "clanrat", "attack_walk", "both", damage={"a": {"method": "kill", "share": 0.7}}, **f),
                    lane("gs", "clanrat", "attack_walk", "both", **f),
                    lane("swords", "clanrat", "attack_walk", "both", t_w=15, **f),
                    lane("swords", "clanrat", "attack_walk", "both", t_w=50, **f),
                    lane("swords", "clanrat", "attack_walk", "both", **f)])
    elif plan == "defender":
        def dl(kind):
            extra = dict(target2="clanrat", t2_gap_m=150, t2_inward=True) if kind == "attack_t2" else {}
            return lane("swords", "clanrat", "attack_walk", "both", gap_m=3, fight_s=90, lane_dx=DEFENDER_LANE_DX,
                        after={"at_s": 1, "kind": kind, "dx": 30 if kind == "move_near" else 0, "dz": 0, "walk": False},
                        **extra)
        base = [dl(k) for k in DEFENDER]
        out += [base, rotate(base, 2)]
    elif plan == "wave":
        w = dict(fight_s=90)
        base = [lane("swords", "clanrat", "attack_walk", "both_walk", gap_m=1, **w),
                lane("swords", "clanrat", "attack_walk", "both_walk", gap_m=2.5, **w),
                lane("swords", "clanrat", "attack_walk", "both_walk", gap_m=3, **w),
                lane("swords", "clanrat", "attack_walk", "both", gap_m=3, **w),
                lane("swords", "clanrat", "attack_walk", "both", gap_m=30, **w)]
        out += [base, rotate(base, 2)]
    elif plan == "routmob":
        def rm(attacker, target, at_rout):
            return lane(attacker, target, "attack_run", "stand", gap_m=40, fight_s=200, t_morale=True, at_rout=at_rout,
                        after_rout_s=ROUTMOB_END_S, max_s=260)
        base = [rm("spear", "slave", "none"), rm("spear", "slave", "halt"), rm("swords", "clanrat", "none"),
                rm("swords", "clanrat", "halt")]
        out += [base, rotate(base, 2)]
    elif plan == "routmob2":
        def rm2(kind):
            extra = dict(extras=ROUTMOB2_EXTRAS[kind]) if kind in ROUTMOB2_EXTRAS else {}
            return lane("spear", "slave", "attack_run", "stand", gap_m=40, fight_s=200, t_morale=True,
                        at_rout="none" if kind == "chase" else "halt", after_rout_s=ROUTMOB_END_S, max_s=260,
                        damage={"t": {"method": "kill", "share": 0.5}}, rout_at_s=30, kind=kind, **extra)
        base = [rm2(k) for k in ("chase", "fire", "neighbours", "control")]
        out += [base, rotate(base, 2)]
    elif plan == "rallysecure":
        def rs(kind):
            common = dict(t_morale=True, damage={"t": {"method": "kill", "share": 0.5}}, kind=kind,
                          lane_dx=RALLYSECURE_LANE_DX)
            if kind == "control":
                # never attacked: the spearmen stand 300 m off, both held; the lane runs as long as a rallying one
                return lane("spear", "slave", "hold", "hold", gap_m=300, fight_s=240, answer=False, max_s=180, **common)
            extra = dict(rally_friends=RALLYSECURE_FRIENDS[kind]) if kind in RALLYSECURE_FRIENDS else {}
            return lane("spear", "slave", "attack_run", "hold", gap_m=40, fight_s=240, answer=False, at_rout="away",
                        away_dz=RALLYSECURE_AWAY_DZ, after_rout_s=200, after_rally_s=RALLYSECURE_AFTER_S, max_s=240,
                        **common, **extra)
        base = [rs(k) for k in ("alone", "neighbours", "control", "lord")]
        out += [base, rotate(base, 2)]
    elif plan == "wavemiss":
        def wm(pair, mode):
            return lane(pair[0], pair[1], "attack_run", mode, gap_m=30, fight_s=90)
        out.append([wm(p, ("hold", "stand")[k % 2]) for k, p in enumerate(WAVEMISS)])
        out.append(rotate([wm(p, ("stand", "hold")[k % 2]) for k, p in enumerate(WAVEMISS)], 2))
    elif plan == "wavemiss2":
        def wm2(pair, how):
            if how == "stand":
                return lane(pair[0], pair[1], "attack_run", "stand", gap_m=30, fight_s=90)
            # the shooter is the lane's attacker (the unit lane.after orders), unordered; the melee unit charges it
            return lane(pair[1], pair[0], "hold", "both", gap_m=30, fight_s=90, answer=False,
                        after={"at_s": 3, "kind": "move_near", "dx": 0, "dz": 40, "walk": False})
        out.append([wm2(p, ("stand", "move")[k % 2]) for k, p in enumerate(WAVEMISS)])
        out.append(rotate([wm2(p, ("move", "stand")[k % 2]) for k, p in enumerate(WAVEMISS)], 2))
    elif plan == "dmgmelee":
        def dmg(attacker, target, damage=None):
            # kind: which unit starts at 30 % ('t30' the target, 'a30' the attacker) or 'full' - the cell's name
            kind = "full" if not damage else "a30" if "a" in damage else "t30"
            return lane(attacker, target, "attack_walk", "both", gap_m=3, fight_s=240, morale_rows=True, kind=kind,
                        **({"damage": damage} if damage else {}))
        cut = {"method": "kill", "share": 0.7}
        base = [dmg("gs", "svsh"), dmg("gs", "svsh", {"t": cut}), dmg("gs", "svsh", {"a": cut}),
                dmg("spearsh", "clanrat"), dmg("general", "svsh")]
        for b in (base, rotate(base, 2)):
            out.append([dict(l, place=p) for l, p in zip(b, DMGMELEE_PLACES)])
    elif plan == "reengage":
        def re(attacker, target, kind):
            common = dict(gap_m=30, kind=kind, lane_dx=REENGAGE_LANE_DX)
            if kind == "reengage":
                # 30 m at a run, 150 s, 15 s apart, back in, 60 s
                return lane(attacker, target, "reengage", "both", fight_s=REENGAGE["part_after_s"], max_s=310,
                            **REENGAGE, **common)
            if kind == "tired":
                # tire_max_s + the way back and the wait + 30 m + 60 s
                return lane(attacker, target, "tire", "both", fight_s=60,
                            max_s=REENGAGE_TIRE["tire_max_s"] + REENGAGE_TIRE["ready_max_s"] + 90, **REENGAGE_TIRE,
                            **common)
            return lane(attacker, target, "attack_run", "both", fight_s=60, **common)
        base = [re("swords", "clanrat", "reengage"), re("gs", "svsh", "reengage"), re("swords", "clanrat", "control"),
                re("swords", "clanrat", "tired")]
        out += [base, rotate(base, 2)]
    elif plan == "retarget":
        def rt(shooter, switch_s):
            # the shooter between the two targets (a_mid), fire at will on, never in melee: the lane ends at max_s
            return lane(shooter, "slave", "retarget", "hold", gap_m=RETARGET["gap_m"], fight_s=RETARGET["fire_s"],
                        answer=False, target2="slave", t2_gap_m=RETARGET["t2_gap_m"], a_mid=True, max_s=RETARGET["fire_s"],
                        kind=RETARGET_KIND[switch_s], **({"switch_s": switch_s} if switch_s else {}))
        base = [rt(s, sw) for s, sw in RETARGET_LANES]
        for b in (base, rotate(base, 2)):
            out.append([dict(l, place=p) for l, p in zip(b, RETARGET_PLACES)])
    elif plan == "retarget2":
        def rt2(shooter, switch_s):
            target = RETARGET2_TARGET.get(shooter, "slave")
            gap = RETARGET["gap_m"]
            if shooter in RETARGET2_RANGE_SHARE:
                # front to front for the centre distance: the shooter a_dx (half the targets' centre spacing) aside
                a_key, a_men, _ = UNITS[shooter]
                t_key, t_men, _ = UNITS[target]
                with open(UNITS_JSON, encoding="utf-8") as f:
                    rng = json.load(f)["units"][a_key]["missile"]["range_m"]
                d, a_dx = RETARGET2_RANGE_SHARE[shooter] * rng, (WIDTH_M + RETARGET["t2_gap_m"]) / 2
                gap = round(math.sqrt(d * d - a_dx * a_dx) - (depth(a_key, a_men, WIDTH_M)
                                                              + depth(t_key, t_men, WIDTH_M)) / 2, 1)
            return lane(shooter, target, "retarget", "hold", gap_m=gap, fight_s=RETARGET["fire_s"], answer=False,
                        target2=target, t2_gap_m=RETARGET["t2_gap_m"], t2_inward=True, a_mid=True,
                        max_s=RETARGET["fire_s"], kind=RETARGET_KIND[switch_s],
                        **({"switch_s": switch_s} if switch_s else {}))
        base = [rt2(s, sw) for s, sw in RETARGET2_LANES]
        for b in (base, rotate(base, 5)):
            out.append([dict(l, place=p) for l, p in zip(b, RETARGET2_PLACES)])
    elif plan == "retarget3":
        def rt3(shooter, switch_s, geom):
            where = (dict(t2_ray=True, t2_gap_m=RETARGET3_RAY_GAP_M) if geom == "ray"
                     else dict(t2_gap_m=RETARGET["t2_gap_m"], t2_inward=True, a_mid=True))
            gap = RETARGET["gap_m"]
            if geom == "ray":                        # straight ahead: the first target's centre RETARGET3_RAY_D_M off
                a_key, a_men, _ = UNITS[shooter]
                t_key, t_men, _ = UNITS["slave"]
                gap = round(RETARGET3_RAY_D_M - (depth(a_key, a_men) + depth(t_key, t_men)) / 2, 1)
            return lane(shooter, "slave", "retarget", "hold", gap_m=gap, fight_s=RETARGET["fire_s"],
                        answer=False, target2="slave", max_s=RETARGET["fire_s"],
                        kind=f"{RETARGET_KIND[switch_s]}-{geom}", **where, **({"switch_s": switch_s} if switch_s else {}))
        base = [rt3(*x) for x in RETARGET3_LANES]
        for b in (base, rotate(base, 3)):
            out.append([dict(l, place=p) for l, p in zip(b, RETARGET2_PLACES)])
    elif plan == "leave":
        def lv(attacker, target, kind):
            leave = kind != "control"
            extra = dict(recharge_after_s=LEAVE["fight_s"], back_m=LEAVE["back_m"], recharge_max_s=10 ** 6,
                         as_withdraw=True) if leave else {}
            if kind == "stand":
                extra["t_at_leave"] = "halt"
            return lane(attacker, target, "withdraw" if leave else "attack_run", "both", gap_m=LEAVE["gap_m"],
                        fight_s=LEAVE["fight_s"] + LEAVE["after_s"], answer=False, kind=kind, **extra)
        base = [lv(*x) for x in LEAVE_LANES]
        for b in (base, rotate(base, 3)):
            out.append([dict(l, place=p) for l, p in zip(b, RETARGET2_PLACES)])
    elif plan == "skirmish":
        def sk(shooter, chaser, on):
            return lane(chaser, shooter, "attack_run", "skirmish", gap_m=SKIRMISH["gap_m"], fight_s=SKIRMISH["max_s"],
                        max_s=SKIRMISH["max_s"], answer=False, kind="on" if on else "off", t_skirmish=on)
        base = [sk(s_, c, on) for s_, c in SKIRMISH_PAIRS for on in (True, False)]
        for b in (base, rotate(base, 5)):
            out.append([dict(l, place=p) for l, p in zip(b, SKIRMISH_PLACES)])
    elif plan == "skirmish2":
        g = SKIRMISH2

        def sk2(shooter, infantry, fast, neighbour, kind):
            common = dict(gap_m=g["gap_m"], answer=False, kind=kind, t_skirmish=True)
            if kind == "long":
                return lane(fast, shooter, "attack_run", "skirmish", fight_s=g["long_s"], max_s=g["long_s"], **common)
            short = dict(fight_s=g["short_s"], max_s=g["short_s"], **common)
            if kind == "pass":
                return lane(infantry, shooter, "pass", "skirmish", a_dx=g["side_m"], a_w=g["pass_w"],
                            pass_beyond_m=g["pass_beyond_m"], **short)
            if kind == "neighbour":
                # the neighbour's centre side_m beside the shooter's (t2_dx = t_w + t2_gap_m), the infantry ahead of it
                return lane(infantry, shooter, "attack_t2", "skirmish", target2=neighbour,
                            t2_gap_m=g["side_m"] - WIDTH_M, t2_width=g["neighbour_w"], a_dx=g["side_m"],
                            a_w=g["neighbour_w"], **short)
            return lane(infantry, shooter, "attack_run", "skirmish", **short)
        short = [sk2(*x, kind) for x in SKIRMISH2_SHOOTERS for kind in SKIRMISH2_KINDS]
        long_ = [sk2(*x, "long") for x in SKIRMISH2_SHOOTERS]
        for k in range(2):
            places = rotate(list(SKIRMISH2_SHORT), 5 * k)
            out.append([dict(l, place=p) for l, p in zip(short, places)]
                       + [dict(l, place=p) for l, p in zip(long_, rotate(list(SKIRMISH2_LONG), k))])
    elif plan == "shootcontact":
        g = SHOOTCONTACT

        def sc(shooter, infantry, kind):
            move = {"t_move": {"back_m": g["back_m"], "run": kind == "run"}} if kind in ("walk", "run") else {}
            return lane(infantry, shooter, "attack_run", "skirmish", gap_m=g["gap_m"], fight_s=g["max_s"],
                        max_s=g["max_s"], answer=False, kind=kind, t_skirmish=kind == "skirmish", **move)
        base = [sc(s_, i_, kind) for s_, i_ in SHOOTCONTACT_PAIRS for kind in SHOOTCONTACT_KINDS]
        for k in range(2):
            out.append([dict(l, place=p) for l, p in zip(base, rotate(list(SKIRMISH_PLACES), 5 * k))])
    elif plan == "vv":
        # (a second lane: one Warlord a battle; the swordsmen on clanrats only fill the plan's two-lane frame)
        out.append([lane("warlord", "swords", "attack_run", fight_s=45, a_ability=VV, a_ability_after_s=20),
                    lane("swords", "clanrat", "attack_run", fight_s=45)])
    else:
        pairs = [lane("swords", "slave", gap_m=40, fight_s=90), lane("gs", "slave", gap_m=40, fight_s=90),
                 lane("spearsh", "slave", gap_m=40, fight_s=90), lane("flag", "clanrat", gap_m=40, fight_s=90)]
        for k, ability in enumerate((SYG, "")):
            syg = lane("cspear", "spear", gap_m=40, fight_s=60, lord={"name": "general", "ability": ability, "dz": 8})
            out.append(rotate(pairs, 2 * k) + [syg])
    return out


def spacing(key):
    sp = json.loads(UNITS_JSON.read_text(encoding="utf-8"))["units"][key]["spacing"]
    return float(sp["h"]), float(sp["v"])


def depth(key, men, width=WIDTH_M):
    """The close block's depth, m (the database's spacing); 0 for a lone man."""
    if men <= 1:
        return 0.0
    h, v = spacing(key)
    files = max(1, int(width // h))
    return math.ceil(men / files) * v


def layout(specs):
    """A battle's lanes with places and script names, and the arena (tools/nn/scenario.py shape).
    Every lane's units get their own slots; each side's lord is always there (parked far unless a
    lane uses him)."""
    sides = {EMP: [], SKV: []}
    side_of = {EMP: "own", SKV: "enemy"}
    lanes, used_lords = [], set()

    def add(short, k, tag=""):
        key, men, faction = UNITS[short]
        if short in LORDS.values():
            used_lords.add(short)
            return f"{side_of[faction]}_lord"
        slot = f"{short}{tag}_{k}"
        sides[faction].append({"slot": slot, "key": key, "men": men, "forward": -20 - 40 * (len(sides[faction]) // 4),
                               "lateral": 40 * (len(sides[faction]) % 4 - 1.5), "width": WIDTH_M})
        return f"{side_of[faction]}_{slot}"
    n = len(specs)
    for k, spec in enumerate(specs, 1):
        a_key, a_men, a_fac = UNITS[spec["attacker"]]
        t_key, t_men, t_fac = UNITS[spec["target"]]
        assert a_fac != t_fac, spec
        x = max(s.get("lane_dx", LANE_DX) for s in specs) * (k - 1 - (n - 1) / 2)
        z = 0.0
        if spec.get("place"):                 # an explicit place (the dmgmelee plan's grid)
            x, z = spec["place"]
        aw, tw = spec.get("a_w", WIDTH_M), spec.get("t_w", WIDTH_M)
        row = dict(spec, name=f"L{k}", x=round(float(x), 1), z=float(z), attacker=add(spec["attacker"], k),
                   target=add(spec["target"], k), a_key=a_key, t_key=t_key,
                   a_depth=round(depth(a_key, a_men, aw), 2), t_depth=round(depth(t_key, t_men, tw), 2),
                   a_width=5 if a_men == 1 else aw, t_width=5 if t_men == 1 else tw,
                   move_beyond_m=5.0, max_s=spec.get("max_s") or round(spec["gap_m"] / 1.4 + spec["fight_s"] + 40))
        if spec.get("target2"):
            # a second target beside the first, t2_gap_m from its edge (+x), facing the same way, never ordered
            assert UNITS[spec["target2"]][2] == t_fac, spec
            row["target2"] = add(spec["target2"], k, "b")
            row["t2_dx"] = round(tw + spec.get("t2_gap_m", 4), 1)
            if spec.get("t2_ray"):                   # right behind the target on the line of fire (retarget3)
                row["t2_dx"], row["t2_dz"] = 0.0, -round(row["t_depth"] + spec.get("t2_gap_m", 4), 1)
            if spec.get("t2_inward") and x > 0:      # towards the middle of the field (the lanes' outer ones)
                row["t2_dx"] = -row["t2_dx"]
            if spec.get("a_mid"):                    # the attacker between the two targets (the retarget plan)
                row["a_dx"] = round(row["t2_dx"] / 2, 1)
        if spec.get("extras"):
            # units of the attacker's side placed (dx, dz) from the target's centre, halted, fearless, never ordered
            # (fire: fire at will on); entries/charge_probe.lua start
            assert all(UNITS[e["short"]][2] == a_fac for e in spec["extras"]), spec
            row["extras"] = [dict(e, name=add(e["short"], k, f"e{j}")) for j, e in enumerate(spec["extras"], 1)]
        if spec.get("rally_friends"):
            # units of the target's side parked far (+z, px / pz) until the target rallies, then teleported beside it;
            # entries/charge_probe.lua rally_friends
            assert all(UNITS[f["short"]][2] == t_fac for f in spec["rally_friends"]), spec
            n_f = len(spec["rally_friends"])
            row["rally_friends"] = [dict(f, name=add(f["short"], k, f"f{j}"), px=round(x + 40 * (j - (n_f + 1) / 2), 1),
                                         pz=float(RALLYSECURE_PARK_Z))
                                    for j, f in enumerate(spec["rally_friends"], 1)]
        if spec.get("lord"):
            row["lord"] = dict(spec["lord"], name=add(spec["lord"]["name"], k))
            if row["lord"].get("at_m"):       # front to front -> centre to centre
                row["lord"]["at_m"] = round(row["lord"]["at_m"] + (row["a_depth"] + row["t_depth"]) / 2, 1)
        lanes.append(row)
    base = nn_scenario.load_arena()
    armies = {}
    for faction, units in sides.items():
        lord = LORDS[faction]
        key, _, _ = UNITS[lord]
        armies[side_of[faction]] = {"faction": faction, "units": [
            {"slot": "lord", "key": key, "men": 1, "general": True, "forward": -80, "lateral": 0, "width": 5}] + units}
    park = [{"name": f"{side_of[f]}_lord", "x": (-700 if f == EMP else 700), "z": -400, "bearing": 0, "width": 5}
            for f, lord in LORDS.items() if lord not in used_lords]
    arena = {k: v for k, v in base.items() if k not in ("faction", "units", "description")}
    arena.update(name="charge_probe", gap_m=400, defend_radius_m=150, sides=armies)
    return lanes, park, arena


def run_config(plan, index):
    """(entry config, model seconds, arena) of battle `index` (1-based) of a plan."""
    specs = battles(plan)[index - 1]
    lanes, park, arena = layout(specs)
    turn = plan == "move" and any(l["mode"] == "script" for l in lanes)
    config = {"plan": plan, "battle": index, "settle_ms": SETTLE_MS, "tick_ms": TURN_TICK_MS if turn else TICK_MS,
              "men_ms": 500 if turn else MEN_MS,
              # (reengage: the pair 30 + 30 m apart between its two fights, centre to centre ~75 m)
              "men_near_m": (100 if plan == "reengage" else 150 if plan == "leave"
                             else SKIRMISH["men_near_m"] if plan == "skirmish"
                             else SKIRMISH2["men_near_m"] if plan == "skirmish2"
                             else SHOOTCONTACT["men_near_m"] if plan == "shootcontact" else 60),
              # the soldiers' places: the first 30 s (the charge plan) or the whole fight (hit: men in contact)
              "men_after_s": (REFORM_MEN_S if plan in ("reform", "reform2", "dmgmelee", "reengage") else 95 if plan in ("defender", "wave", "wavemiss", "wavemiss2", "leave", "skirmish", "skirmish2",
                                                                                                  "shootcontact")
                              else 90 if plan in ("hit", "move", "vv") else 30),
              "lanes": lanes, "park": park}
    if plan in ("routmob", "routmob2"):
        config["men_after_rout_s"] = ROUTMOB_MEN_S
    model_s = max(l["max_s"] for l in lanes) + SETTLE_MS / 1000 + 20
    return config, model_s, arena


def write_scenario(plan, index, path=None):
    _, _, arena = run_config(plan, index)
    path = path or ROOT / f"charge_probe_{plan}_{index}.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(nn_scenario.scenario_xml(arena, "enemy", 3600), encoding="utf-8", newline="\n")
    return path


def run(plan, which=None, python=sys.executable, dry=False, deadline=900):
    """Builds and launches each battle in turn (tools/launcher/launch.ps1: Normal difficulty set and
    restored by the launcher)."""
    done = []
    for i in which or range(1, len(battles(plan)) + 1):
        build = [python, "-m", "tools.build", "charge-probe", "--probe-plan", plan, "--probe-battle", str(i),
                 "--deadline", str(deadline)]
        launch = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(LAUNCHER / "launch.ps1"),
                  "-Target", "charge-probe"]
        print(f"--- {plan} {i}", flush=True)
        if dry:
            print(" ".join(build), "&&", " ".join(launch))
            continue
        code = subprocess.run(build, cwd=project.ROOT, stdout=subprocess.DEVNULL).returncode
        if code == 0:
            code = subprocess.run(launch, cwd=project.ROOT).returncode
        done.append((plan, i, code))
    return done


# ---------------------------------------------------------------- the recordings

def load_run(run_dir):
    """{lane name: {spec, samples [(t s, a row, tg row, lord row)], men [(t, a xz, tg xz)], contacts, end,
    abilities, phases}} of one run."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["config"]
    lanes = {l["name"]: {"run": run_dir.name, "plan": cfg.get("plan"), "battle": cfg.get("battle"), "spec": l,
                         "samples": [], "men": [], "contacts": {}, "end": None, "abilities": [], "phases": []}
             for l in cfg.get("lanes", [])}
    for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if '"probe_' not in line:
            continue
        r = json.loads(line)
        ev = r["event"]
        if ev == "probe_sample":
            for x in r["lanes"]:
                if x["lane"] in lanes:
                    lanes[x["lane"]]["samples"].append((x["t"] / 1000, x["a"], x["tg"], x.get("l")))
                    if x.get("e"):
                        lanes[x["lane"]].setdefault("extras", []).append((x["t"] / 1000, x["e"]))
                    if x.get("f"):
                        lanes[x["lane"]].setdefault("friends", []).append((x["t"] / 1000, x["f"]))
                    if x.get("t2"):
                        lanes[x["lane"]].setdefault("t2", []).append((x["t"] / 1000, x["t2"]))
        elif ev == "probe_men":
            for x in r["lanes"]:
                if x["lane"] in lanes:
                    lanes[x["lane"]]["men"].append((x["t"] / 1000, x.get("a") or [], x.get("tg") or []))
                    if x.get("e"):
                        lanes[x["lane"]].setdefault("extra_men", []).append((x["t"] / 1000, x["e"]))
        elif ev == "probe_contact":
            lanes[r["lane"]]["contacts"][r["n"]] = r["t"] / 1000
        elif ev == "probe_lane_end":
            lanes[r["lane"]]["end"] = r
        elif ev == "probe_ability":
            lanes[r["lane"]]["abilities"].append(r)
        elif ev == "probe_phase":
            lanes[r["lane"]]["phases"].append(r)
        elif ev == "probe_skirmish":
            lanes[r["lane"]].setdefault("skirmish", []).append(r)
    return [x for x in lanes.values() if x["samples"]]


def series(samples, who, key):
    i = {"a": 1, "tg": 2, "l": 3}[who]
    return np.array([np.nan if (s[i] or {}).get(key) is None else float(s[i][key]) for s in samples])


def at(t, v, when):
    """v interpolated at time `when` (NaN outside the recording)."""
    ok = np.isfinite(v)
    if ok.sum() < 2 or when < t[ok][0] or when > t[ok][-1]:
        return np.nan
    return float(np.interp(when, t[ok], v[ok]))


def near_counts(a, b, radii=RADII):
    """Soldiers of a within each radius of their nearest soldier of b (flat decimetre lists)."""
    if len(a) < 2 or len(b) < 2:
        return [0] * len(radii)
    pa = np.asarray(a, float).reshape(-1, 2) / 10
    pb = np.asarray(b, float).reshape(-1, 2) / 10
    d = np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return [int((d <= r).sum()) for r in radii]


def measure(lane):
    """One lane's numbers (game or simulator: the same row format)."""
    spec, s = lane["spec"], lane["samples"]
    t = np.array([x[0] for x in s])
    out = {"run": lane["run"], "lane": spec["name"], "attacker": SHORT[spec["a_key"]],
           "target": SHORT[spec["t_key"]], "a_key": spec["a_key"], "t_key": spec["t_key"],
           "mode": spec["mode"], "target_mode": spec["target_mode"], "gap_m": spec["gap_m"],
           "ability": (spec.get("lord") or {}).get("ability"), "cell": cell(spec)}
    ax, az, tx, tz = (series(s, w, k) for w, k in (("a", "x"), ("a", "z"), ("tg", "x"), ("tg", "z")))
    c = lane["contacts"].get(1)
    out["contact_s"] = c
    if c is None:
        return out
    # the way in: the attacker's mean speed over distance bands before contact (the engine moves a unit in
    # steps, so speeds come from the times the distance crossed each band, not from 0.5 s differences); a tired lane's
    # way in starts at its 'tired' phase (before it the pair shuttles)
    phase_t = {}
    for p in lane.get("phases", []):
        phase_t.setdefault(p["phase"], p["t"] / 1000)
    t_in = phase_t.get("tired", float(t[0]))
    if "tire_end" in phase_t:
        out["tire_s"] = phase_t["tire_end"] - float(t[0])
    dist = np.hypot(ax - tx, az - tz)
    before = (t >= t_in - 1e-6) & (t <= c) & np.isfinite(dist)
    tb, db = t[before], dist[before]
    if len(tb) > 3:
        reach = db[-1]
        left = db - reach                      # metres still to go
        def crossed(m):
            k = np.nonzero(left <= m)[0]
            if not len(k) or k[0] == 0:
                return None
            i = k[0]
            return float(np.interp(m, [left[i], left[i - 1]], [tb[i], tb[i - 1]]))
        t30, t60, t10 = crossed(30), crossed(60), crossed(10)
        out["speed_last30"] = 30 / (tb[-1] - t30) if t30 is not None and tb[-1] > t30 else None
        out["speed_30_60"] = 30 / (t30 - t60) if None not in (t30, t60) and t30 > t60 else None
        out["speed_last10"] = 10 / (tb[-1] - t10) if t10 is not None and tb[-1] > t10 else None
        # the peak: the fastest 2 s on the way in
        w = int(round(2.0 / max(1e-6, float(np.median(np.diff(tb))))))
        if len(tb) > w:
            out["speed_peak30"] = float(np.max((db[:-w] - db[w:]) / (tb[w:] - tb[:-w])))
    out["approach_s"] = float(c - t_in)
    end_t = t[-1]
    # a window from the contact starts at the sample before it (the game's first contact sample already holds the
    # blows struck since that sample: a lord's first blow; the simulator's contact time is that sample already)
    before_c = t[t < c - 1e-6]
    c0 = float(before_c[-1]) if lane["run"] != "sim" and len(before_c) else c
    for who in ("tg", "a"):
        hp, men = series(s, who, "hp"), series(s, who, "men")
        for lo, hi in WINDOWS:
            if c + hi <= end_t + 1e-6:
                start = c0 if lo == 0 else c + lo
                out[f"{who}_hp_{lo}_{hi}"] = at(t, hp, start) - at(t, hp, c + hi)
                out[f"{who}_men_{lo}_{hi}"] = at(t, men, start) - at(t, men, c + hi)
        stop = min(end_t, c + spec["fight_s"])
        if lane["contacts"].get(2) is not None:
            stop = min(stop, lane["contacts"][2] - 1)
            phases = [p["t"] / 1000 for p in lane.get("phases", []) if p["phase"] == "out"]
            if phases:
                stop = min(stop, phases[0])
        if stop - (c + STEADY_FROM_S) >= 5:
            out[f"{who}_hp_steady"] = (at(t, hp, c + STEADY_FROM_S) - at(t, hp, stop)) / (stop - c - STEADY_FROM_S)
            out[f"{who}_men_steady"] = (at(t, men, c + STEADY_FROM_S) - at(t, men, stop)) / (stop - c - STEADY_FROM_S)
        c2 = lane["contacts"].get(2)
        if c2 is not None:
            for lo, hi in WINDOWS2:
                if c2 + hi <= end_t + 1e-6:
                    out[f"{who}_hp2_{lo}_{hi}"] = at(t, hp, c2 + lo) - at(t, hp, c2 + hi)
            # the second fight's steady rate (15 s on, to the lane's end) and both waves: the first 10 s over 10 s of it
            stop2 = min(end_t, c2 + (spec.get("after2_s") or spec["fight_s"]))
            if stop2 - (c2 + STEADY_FROM_S) >= 5:
                out[f"{who}_hp2_steady"] = (at(t, hp, c2 + STEADY_FROM_S) - at(t, hp, stop2)) / (stop2 - c2 - STEADY_FROM_S)
        for k, steady, first in (("wave_10", f"{who}_hp_steady", f"{who}_hp_0_10"),
                                 ("wave2_10", f"{who}_hp2_steady", f"{who}_hp2_0_10")):
            if out.get(steady) and out[steady] > 0 and out.get(first) is not None and np.isfinite(out[first]):
                out[f"{who}_{k}"] = out[first] / (10 * out[steady])
        if who == "tg" and lane.get("abilities"):
            out["ability_status"] = lane["abilities"][0].get("status")
            if c + 18 <= end_t:
                out["tg_hp_0_18"] = at(t, hp, c) - at(t, hp, c + 18)
                out["a_hp_0_18"] = at(t, series(s, "a", "hp"), c) - at(t, series(s, "a", "hp"), c + 18)
    out["contact2_s"] = lane["contacts"].get(2)
    # the fatigue state (0 fresh .. 5 exhausted) at each contact: the last sample at or before it
    for n, ct in lane["contacts"].items():
        k = np.nonzero(t <= ct + 1e-6)[0]
        if len(k) and n in (1, 2):
            for who, i in (("a", 1), ("tg", 2)):
                v = _level((s[k[-1]][i] or {}).get("fat"))
                if np.isfinite(v):
                    out[f"{who}_fat_c{n}"] = v
    # men in contact (soldiers' places): per second after contact, the mean over 0-5, 5-15, 15-30 s
    if lane.get("men"):
        rows = [(tm - c, near_counts(a, b), near_counts(b, a)) for tm, a, b in lane["men"] if tm >= c - 0.01]
        for lo, hi in NEAR_WINDOWS:
            sel = [r for r in rows if lo <= r[0] < hi]
            if sel:
                out[f"a_near_{lo}_{hi}"] = [round(float(np.mean([r[1][k] for r in sel])), 1) for k in range(len(RADII))]
                out[f"tg_near_{lo}_{hi}"] = [round(float(np.mean([r[2][k] for r in sel])), 1) for k in range(len(RADII))]
    return out


def cell(spec):
    """The lane's experimental cell: who on whom, how."""
    a = SHORT[spec["a_key"]] if "a_key" in spec else spec["attacker"]
    t = SHORT[spec["t_key"]] if "t_key" in spec else spec["target"]
    name = f"{a}>{t} {spec['mode']}/{spec['target_mode']}"
    if spec["gap_m"] not in (GAP_M, 40):
        name += f" gap{spec['gap_m']}"
    ab = (spec.get("lord") or {}).get("ability")
    if ab is not None:
        name += " SYG" if ab else " noSYG"
    if spec.get("kind"):                  # routmob2 / rallysecure / dmgmelee: the lane's kind tells same-looking lanes apart
        name += f" {spec['kind']}"
    return name


def runs(root=RUNS):
    if not root.exists():
        return []
    return sorted(d for d in root.iterdir() if (d / "events.jsonl").exists() and (d / "manifest.json").exists())


# ---------------------------------------------------------------- the simulator on the same lanes

def sim_lanes(lanes, params=None, device="cpu", copies=8, jitter_m=1.0, seed=0):
    """The simulator on recorded lanes: each lane its own battle (its units at their places at the
    probe's go), the same scripted orders as the entry's modes, everyone fearless. Returns lanes in
    the game's shape (samples every 0.5 s, contacts, phases), `copies` per recorded lane."""
    import torch
    from tools.nn.sim import abilities as sim_abilities, battle, orders as O, scenario as sim_scenario
    from tools.nn.sim.params import load
    params = params or load()
    if any(ln["spec"].get("at_rout") == "away" or ln["spec"].get("rally_friends") for ln in lanes):
        raise NotImplementedError("the rallysecure lanes (teleports at the rout and the rally) have no simulator twin")
    armies, meta = [], []
    for ln in lanes:
        spec = ln["spec"]
        s0 = ln["samples"][0]
        sides = {1: {"faction": EMP, "ai": False, "units": []}, 2: {"faction": SKV, "ai": False, "units": []}}
        roles = {}
        for role, row, key, width in (("a", s0[1], spec["a_key"], spec["a_width"]),
                                      ("tg", s0[2], spec["t_key"], spec["t_width"])):
            fac = 1 if FACTION[key] == EMP else 2
            lord = key in (UNITS["general"][0], UNITS["warlord"][0])
            sides[fac]["units"].append({"key": key, "x": row["x"], "z": row["z"], "b": row["b"], "men": row["men"],
                                        "width": None if lord else width, "general": lord, "name": role})
            roles[role] = fac
        if spec.get("target2"):
            # the second target: tx + t2_dx, beside the target, facing its way (entries/charge_probe.lua place)
            t2 = spec["target2"]                     # the layout's script name: <side>_<short>b_<lane>
            t2 = t2 if t2 in UNITS else t2.split("_", 1)[1].rsplit("_", 1)[0][:-1]
            t2_key = UNITS[t2][0]
            r2 = ln["t2"][0][1] if ln.get("t2") else None      # its recorded place (the retarget plan records it)
            sides[roles["tg"]]["units"].append({"key": t2_key, "x": r2["x"] if r2 else s0[2]["x"] + spec.get("t2_dx", 0),
                                                "z": r2["z"] if r2 else s0[2]["z"] + spec.get("t2_dz", 0),
                                                "b": r2["b"] if r2 else s0[2]["b"],
                                                "men": r2["men"] if r2 else UNITS[t2][1], "width": spec["t_width"],
                                                "general": False, "name": "t2"})
        if spec.get("lord") and s0[3]:
            r = s0[3]
            sides[1]["units"].append({"key": UNITS["general"][0], "x": r["x"], "z": r["z"], "b": r["b"],
                                      "general": True, "name": "l"})
        for j, e in enumerate(spec.get("extras") or [], 1):
            # the attacker's side's extra units (dx, dz) from the target's centre, halted (a shooter fires at will)
            sides[roles["a"]]["units"].append({"key": UNITS[e["short"]][0], "x": s0[2]["x"] + e["dx"],
                                               "z": s0[2]["z"] + e["dz"], "b": s0[1]["b"], "men": UNITS[e["short"]][1],
                                               "width": spec["a_width"], "general": False, "name": f"e{j}"})
        if spec.get("t_morale"):
            # the target can rout (the routmob plan): a far-off standing unit of its side keeps its battle going
            sides[roles["tg"]]["units"].append({"key": spec["t_key"], "x": s0[2]["x"], "z": s0[2]["z"] - 400.0,
                                                "b": s0[2]["b"], "men": s0[2]["men"], "width": spec["t_width"],
                                                "general": False, "name": "keep"})
        army = {"attacker": roles["a"], "sides": sides}
        for _ in range(copies):
            armies.append(army)
            meta.append(ln)
    per_side = max(len(a["sides"][s]["units"]) for a in armies for s in (1, 2))
    st = sim_scenario.build(armies, params, device=device, per_side=per_side)
    gen = torch.Generator().manual_seed(seed)
    for k in ("x", "z"):
        st.u[k] = st.u[k] + ((torch.rand(st.u[k].shape, generator=gen) * 2 - 1) * jitter_m).to(device)
    B, N = st.B, st.N
    slot = {r: torch.tensor([sim_scenario.slots(a, per_side).get(r, -1) for a in armies], device=device)
            for r in ("a", "tg", "l", "t2")}
    # everyone fearless but a target that keeps its morale (t_morale: the routmob plan)
    keep = torch.zeros((B, N), dtype=torch.bool, device=device)
    for b, ln in enumerate(meta):
        if ln["spec"].get("t_morale") and int(slot["tg"][b]) >= 0:
            keep[b, int(slot["tg"][b])] = True
    st.u["leadership"] = torch.where(keep, st.u["leadership"], st.u["leadership"] + 1e4)
    st.u["morale"] = torch.where(keep, st.u["morale"], st.u["morale"] + 1e4)
    specs = [ln["spec"] for ln in meta]
    mode = [sp["mode"] for sp in specs]
    tmode = [sp["target_mode"] for sp in specs]
    syg_slot = sim_abilities.slot_keys(UNITS["general"][0], params.units, params.abilities).index(SYG) \
        if SYG in sim_abilities.slot_keys(UNITS["general"][0], params.units, params.abilities) else -1
    st_ = {"contact": [None] * B, "contact2": [None] * B, "phase": ["in"] * B, "out_t": [0.0] * B, "after": [None] * B,
           "out_from": [None] * B, "fired": [False] * B, "last": {}, "phases": [[] for _ in range(B)],
           "a_ability": [False] * B, "start": [None] * B, "faced": set(), "rout": [None] * B,
           "re": [{"phase": "in", "clear": False} for _ in range(B)], "tire": [None] * B, "aim": [None] * B,
           "t_halt": {}}
    # the retarget plan's switches run on the game's clock from the go: the twin starts at the first sample
    t_first = [float(ln["samples"][0][0]) for ln in meta]
    rec = {"t": [], "rows": []}
    fields = ("x", "z", "b", "men", "hp_abs", "m", "fatigue", "fat", "k", "r", "s", "a")
    a_slot = {}
    for sp in specs:
        if sp.get("a_ability") and sp["a_key"] not in a_slot:
            keys = sim_abilities.slot_keys(sp["a_key"], params.units, params.abilities)
            a_slot[sp["a_key"]] = keys.index(sp["a_ability"]) if sp["a_ability"] in keys else -1

    def policy(st):
        u = st.u
        t = float(st.t.max())
        b_idx = torch.arange(B, device=device)
        m_a = u["m"][b_idx, slot["a"]].bool().cpu().numpy()
        m_t = u["m"][b_idx, slot["tg"]].bool().cpu().numpy()
        r_t = u["r"][b_idx, slot["tg"]].bool().cpu().numpy()
        ax, az = u["x"][b_idx, slot["a"]].cpu().numpy(), u["z"][b_idx, slot["a"]].cpu().numpy()
        tx, tz = u["x"][b_idx, slot["tg"]].cpu().numpy(), u["z"][b_idx, slot["tg"]].cpu().numpy()
        fat_a, fat_t = u["fat"][b_idx, slot["a"]].cpu().numpy(), u["fat"][b_idx, slot["tg"]].cpu().numpy()
        o = O.hold(B, N, device)
        kind, target, run_, x, z, ab = (o.kind.clone(), o.target.clone(), o.run.clone(), o.x.clone(), o.z.clone(),
                                        o.ability.clone())
        for b in range(B):
            A, T, L = int(slot["a"][b]), int(slot["tg"][b]), int(slot["l"][b])
            sp = specs[b]
            # the contact began in the step that ended now (its blows are in this sample already): its start
            # (a tired lane's only once its fight is on: before it the pair shuttles)
            fighting = mode[b] != "tire" or (st_["tire"][b] or {}).get("phase") == "fight"
            if st_["contact"][b] is None and fighting and (m_a[b] or m_t[b]):
                st_["contact"][b] = t - params.dt
                if L >= 0 and sp["lord"].get("ability") and syg_slot >= 0 and not sp["lord"].get("at_m"):
                    ab[b, L] = syg_slot
            if (L >= 0 and sp["lord"].get("at_m") and sp["lord"].get("ability") and syg_slot >= 0
                    and st_["contact"][b] is None and b not in st_["faced"]):
                tx_, tz_ = float(u["x"][b, T]), float(u["z"][b, T])
                if math.hypot(ax[b] - tx_, az[b] - tz_) <= sp["lord"]["at_m"]:
                    st_["faced"].add(b)
                    ab[b, L] = syg_slot
            c = st_["contact"][b]
            if st_["start"][b] is None:
                st_["start"][b] = (float(ax[b]), float(az[b]))
            # the attacker's own ability (Foe-Seeker) a_ability_after_s after the contact, once
            if sp.get("a_ability") and c is not None and not st_["a_ability"][b]                     and t - c >= sp.get("a_ability_after_s", 0) and a_slot.get(sp["a_key"], -1) >= 0:
                ab[b, A] = a_slot[sp["a_key"]]
                st_["a_ability"][b] = True
            # the attacker
            if mode[b] == "script":
                # a 'move' step: MOVE to its start + (dx, dz); a 'face' step has no order in the simulator (it has no
                # facing order): the unit holds and the twin turns it to the step's bearing at once (so that a later
                # move starts from the game's facing)
                due = [x for x in sp.get("steps", []) if t >= x["at_s"]]
                for j, x_ in enumerate(due):
                    if x_["kind"] == "face" and (b, j) not in st_["faced"]:
                        st_["faced"].add((b, j))
                        st.u["b"][b, A] = float(x_["bearing"])
                if due and due[-1]["kind"] == "move":
                    sx, sz = st_["start"][b]
                    kind[b, A], x[b, A], z[b, A] = O.MOVE, sx + due[-1].get("dx", 0), sz + due[-1].get("dz", 0)
                    run_[b, A] = bool(due[-1].get("run"))
            elif mode[b] in ("recharge", "withdraw"):
                ph = st_["phase"][b]
                if ph == "in" and c is not None and t - c >= sp["recharge_after_s"]:
                    st_["phase"][b], st_["out_t"][b], st_["out_from"][b] = "out", t, (float(ax[b]), float(az[b]))
                    st_["phases"][b].append({"phase": "out", "t": t * 1000})
                elif ph == "out":
                    fx, fz = st_["out_from"][b]
                    moved = math.hypot(ax[b] - fx, az[b] - fz)
                    if mode[b] == "recharge" and ((not m_a[b] and moved >= sp["back_m"] - 5)
                                                  or t - st_["out_t"][b] >= sp["recharge_max_s"]):
                        st_["phase"][b] = "back"
                        st_["phases"][b].append({"phase": "back", "t": t * 1000})
                if st_["phase"][b] == "back" and st_["contact2"][b] is None and m_a[b]:
                    st_["contact2"][b] = t - params.dt
                if st_["phase"][b] == "out":
                    fx, fz = st_["out_from"][b]
                    # (as_withdraw: the network's own withdraw order, as the bridge gives it in the game - a run)
                    kind[b, A], x[b, A], z[b, A], run_[b, A] = (O.WITHDRAW if sp.get("as_withdraw") else O.MOVE,
                                                                fx, fz + sp["back_m"], True)
                else:
                    kind[b, A], target[b, A], run_[b, A] = O.ATTACK, T, True
            elif mode[b] in ("attack_run", "attack_walk"):
                kind[b, A], target[b, A], run_[b, A] = O.ATTACK, T, mode[b] == "attack_run"
            elif mode[b] == "retarget":
                # a ranged attack on the target, with switch_s on the other one in turns (entries/charge_probe.lua
                # retarget_aim); KEEP below makes each switch one new order
                aim = retarget_aim(t + t_first[b], sp.get("switch_s"))
                T2 = int(slot["t2"][b])
                kind[b, A], target[b, A], run_[b, A] = O.ATTACK, (T2 if aim == 2 and T2 >= 0 else T), False
                if st_["aim"][b] != aim:
                    n = sum(1 for p in st_["phases"][b] if p["phase"] == "aim")
                    st_["phases"][b].append({"phase": "aim", "target": aim, "t": t * 1000, "n": n})
                    st_["aim"][b] = aim
            # the target's rout (t_morale lanes): at_rout 'halt' halts the attacker there (entries/charge_probe.lua);
            # rout_at_s: not routed that long after the contact, it is routed by script (morale_behavior_rout)
            if (sp.get("t_morale") and sp.get("rout_at_s") and st_["rout"][b] is None and c is not None
                    and t - c >= sp["rout_at_s"] and not r_t[b]):
                st.u["morale"][b, T] = -10.0                    # below 0: a rout (the broken floor -50 would shatter)
                st_["phases"][b].append({"phase": "rout_forced", "t": t * 1000})
            if sp.get("t_morale") and st_["rout"][b] is None and r_t[b]:
                st_["rout"][b] = t - params.dt
                st_["phases"][b].append({"phase": "rout", "t": (t - params.dt) * 1000})
                if sp.get("at_rout") == "halt":
                    st_["after"][b] = (float(ax[b]), float(az[b]))
            if st_["rout"][b] is not None and sp.get("at_rout") == "halt":
                kind[b, A], target[b, A], run_[b, A] = O.HOLD, -1, False
                x[b, A], z[b, A] = st_["after"][b]
            elif mode[b] == "move_run":
                kind[b, A], x[b, A], z[b, A], run_[b, A] = O.MOVE, sp["x"], sp["z"] - sp["move_beyond_m"], True
            # lane.after (entries/charge_probe.lua): at_s after the first contact the attacker gets one more order,
            # kept from then on (attack_t2 / attack_same / halt / move_near to its place then + (dx, dz) / none)
            af = sp.get("after")
            if af and c is not None and t - c >= af["at_s"] - 1e-6 and af["kind"] != "none":
                if st_["after"][b] is None:
                    st_["after"][b] = (float(ax[b]), float(az[b]))
                    st_["phases"][b].append({"phase": "after:" + af["kind"], "t": t * 1000})
                T2 = int(slot["t2"][b])
                walk = bool(af.get("walk"))
                if af["kind"] == "attack_t2" and T2 >= 0:
                    kind[b, A], target[b, A], run_[b, A] = O.ATTACK, T2, not walk
                elif af["kind"] == "attack_same":
                    kind[b, A], target[b, A], run_[b, A] = O.ATTACK, T, not walk
                elif af["kind"] == "halt":
                    kind[b, A], target[b, A], run_[b, A] = O.HOLD, -1, False
                    x[b, A], z[b, A] = st_["after"][b]
                elif af["kind"] == "move_near":
                    sx, sz = st_["after"][b]
                    kind[b, A], target[b, A], run_[b, A] = O.MOVE, -1, not walk
                    x[b, A], z[b, A] = sx + af.get("dx", 0), sz + af.get("dz", 0)
            # the second target: halted unless it attacks (t2_mode)
            if sp.get("t2_mode") == "attack" and int(slot["t2"][b]) >= 0:
                kind[b, int(slot["t2"][b])], target[b, int(slot["t2"][b])], run_[b, int(slot["t2"][b])] = O.ATTACK, A, False
            # the target
            if sp.get("t_at_leave") == "halt" and st_["phase"][b] == "out":
                # the leave plan's 'stand': the target halts where it is at the withdraw (entries/charge_probe.lua)
                if st_["t_halt"].get(b) is None:
                    st_["t_halt"][b] = (float(u["x"][b, T]), float(u["z"][b, T]))
                kind[b, T], target[b, T], run_[b, T] = O.HOLD, -1, False
                x[b, T], z[b, T] = st_["t_halt"][b]
            elif tmode[b] in ("both", "both_walk") or (sp.get("answer") and c is not None):
                kind[b, T], target[b, T], run_[b, T] = O.ATTACK, A, tmode[b] == "both"
            elif tmode[b] == "push":
                kind[b, T], x[b, T], z[b, T], run_[b, T] = O.MOVE, sp["x"], sp["z"] + sp.get("push_m", 60), False
            # the reengage plan (entries/charge_probe.lua): the attacker stands at +z facing the target at -z, so straight
            # back is +z for it and -z for the target
            if mode[b] == "reengage":
                re_ = st_["re"][b]
                if re_["phase"] == "in" and c is not None and t - c >= sp["part_after_s"] - 1e-6:
                    re_.update(phase="out", t=t, a=(float(ax[b]), float(az[b]) + sp["part_m"]),
                               tg=(float(tx[b]), float(tz[b]) - sp["part_m"]))
                    st_["phases"][b].append({"phase": "out", "t": t * 1000})
                elif re_["phase"] == "out":
                    if not re_["clear"] and not m_a[b] and not m_t[b]:
                        re_["clear"] = True
                        st_["phases"][b].append({"phase": "clear", "t": t * 1000})
                    if t - re_["t"] >= sp["part_s"] - 1e-6:
                        re_["phase"] = "back"
                        st_["phases"][b].append({"phase": "back", "t": t * 1000, "cleared": re_["clear"]})
                if re_["phase"] == "back" and re_["clear"] and st_["contact2"][b] is None and (m_a[b] or m_t[b]):
                    st_["contact2"][b] = t - params.dt
                if re_["phase"] == "out":
                    for who, i in (("a", A), ("tg", T)):
                        kind[b, i], target[b, i], run_[b, i] = O.MOVE, -1, True
                        x[b, i], z[b, i] = re_[who]
                else:
                    kind[b, A], target[b, A], run_[b, A] = O.ATTACK, T, True
            elif mode[b] == "tire":
                ti = st_["tire"][b]
                if ti is None:
                    ti = st_["tire"][b] = {"phase": "tire", "t0": t, "leg": {"a": 1, "tg": 1},
                                           "start": {"a": (float(ax[b]), float(az[b])), "tg": (float(tx[b]), float(tz[b]))}}
                pos = {"a": (float(ax[b]), float(az[b])), "tg": (float(tx[b]), float(tz[b]))}

                def point(who, leg):
                    sx, sz = ti["start"][who]
                    return (sx, sz + (1 if who == "a" else -1) * sp["tire_leg_m"]) if leg % 2 else (sx, sz)
                if ti["phase"] == "tire":
                    for who in ("a", "tg"):
                        if math.dist(pos[who], point(who, ti["leg"][who])) <= 5:
                            ti["leg"][who] += 1
                    lvl = FAT_LEVELS.index(sp["tire_until"])
                    if (fat_a[b] >= lvl and fat_t[b] >= lvl) or t - ti["t0"] >= sp["tire_max_s"]:
                        ti.update(phase="return", t=t)
                        st_["phases"][b].append({"phase": "tire_end", "t": t * 1000})
                elif ti["phase"] == "return":
                    home = all(math.dist(pos[w], ti["start"][w]) <= 4 for w in ("a", "tg"))
                    if home or t - ti["t"] >= sp["ready_max_s"]:
                        ti["phase"] = "fight"
                        st_["phases"][b].append({"phase": "tired", "t": t * 1000})
                for who, i in (("a", A), ("tg", T)):
                    if ti["phase"] == "fight":
                        kind[b, i], target[b, i], run_[b, i] = O.ATTACK, (T if who == "a" else A), True
                    else:
                        kind[b, i], target[b, i], run_[b, i] = O.MOVE, -1, True
                        x[b, i], z[b, i] = point(who, ti["leg"][who]) if ti["phase"] == "tire" else ti["start"][who]
        # KEEP the orders in force (re-issuing every step would be a new order each time)
        key = torch.stack([kind.float(), target.float(), run_.float(), x, z], -1)
        last = st_["last"].get("key")
        if last is not None:
            same = (key == last).all(-1)
            kind = torch.where(same, torch.full_like(kind, O.KEEP), kind)
        st_["last"]["key"] = key
        return O.Orders(kind=kind, x=x, z=z, target=target, run=run_, ability=ab)

    next_t = [0.0]

    def record(st):
        t = float(st.t.max())
        if t + 1e-6 < next_t[0]:
            return
        next_t[0] = t + TICK_MS / 1000
        rec["t"].append(t)
        rec["rows"].append({k: st.u[k].detach().cpu().numpy().copy() for k in fields if k in st.u})
    until = max((ln["end"] or {}).get("t", 0) / 1000 for ln in lanes) + 5 if any(ln["end"] for ln in lanes) else 200
    if any(ln["spec"].get("t_morale") for ln in lanes):
        until = max(until, max(ln["spec"].get("max_s", 200) for ln in lanes))
    battle.run(st, policy, params, until_s=until, record=record)
    out = []
    for b, ln in enumerate(meta):
        A, T, L = int(slot["a"][b]), int(slot["tg"][b]), int(slot["l"][b])
        samples, t2_rows = [], []
        for t, r in zip(rec["t"], rec["rows"]):
            def row(i):
                if i < 0:
                    return None
                out = {"x": float(r["x"][b, i]), "z": float(r["z"][b, i]), "b": float(r["b"][b, i]),
                       "men": float(r["men"][b, i]), "hp": float(r["hp_abs"][b, i]), "m": bool(r["m"][b, i]),
                       "r": bool(r["r"][b, i]) if "r" in r else None, "s": bool(r["s"][b, i]) if "s" in r else None}
                for k in ("fatigue", "fat", "k"):
                    if k in r:
                        out[k] = float(r[k][b, i])
                if "a" in r:
                    out["ammo"] = float(r["a"][b, i])
                return out
            samples.append((t, row(A), row(T), row(L)))
            if int(slot["t2"][b]) >= 0:
                t2_rows.append((t, row(int(slot["t2"][b]))))
        contacts = {}
        if st_["contact"][b] is not None:
            contacts[1] = st_["contact"][b]
        if st_["contact2"][b] is not None:
            contacts[2] = st_["contact2"][b]
        out.append({"run": "sim", "spec": ln["spec"], "samples": samples, "men": [], "contacts": contacts,
                    "end": None, "abilities": ln.get("abilities") and [{"status": "sim"}], "phases": st_["phases"][b],
                    **({"t2": t2_rows} if t2_rows else {})})
    return out


# ---------------------------------------------------------------- turning (plan move, battle 1)

def _ang(a):
    """Degrees wrapped to -180..180."""
    return (np.asarray(a, float) + 180) % 360 - 180


def turn_measure(lane, run_speed=None):
    """A script lane's turns and starts, per step: the bearing's change over time (s to within 10 deg of the
    facing ordered, or of the way to the point), the centre's speed 1..6 s after the order (share of the run) and
    when it first reaches 0.9 of the run; from the soldiers' places (game): how far the men moved in the first
    10 s and whether the formation turned as a rigid block (each man keeps his rank: +1) or about-faced in place
    (the front rank becomes the back: -1)."""
    spec, smp = lane["spec"], lane["samples"]
    t = np.array([x[0] for x in smp])
    b, x, z = series(smp, "a", "b"), series(smp, "a", "x"), series(smp, "a", "z")
    starts = {}
    for ph in lane.get("phases", []):
        if str(ph.get("phase", "")).startswith("step"):
            starts[int(ph["phase"][4:].split(":")[0])] = ph["t"] / 1000
    steps = spec.get("steps") or []
    out = []
    for i, step in enumerate(steps, 1):
        t0 = starts.get(i, step["at_s"])
        t1 = starts.get(i + 1, steps[i]["at_s"] if i < len(steps) else t[-1])
        sel = (t >= t0 - 1e-6) & (t <= t1 + 1e-6) & np.isfinite(b)
        if sel.sum() < 2:
            continue
        tt, bb, xx, zz = t[sel] - t0, b[sel], x[sel], z[sel]
        if step["kind"] == "face":
            goal = float(step["bearing"])
        else:
            goal = float(np.degrees(np.arctan2(step.get("dx", 0), step.get("dz", 0))) % 360)
        off = np.abs(_ang(bb - goal))
        done = np.nonzero(off <= 10)[0]
        row = {"step": i, "kind": step["kind"], "goal_deg": round(goal), "turn_deg": round(float(abs(_ang(bb[0] - goal)))),
               "s_to_10deg": round(float(tt[done[0]]), 2) if len(done) else None,
               "bearing": [(round(float(a), 2), round(float(v))) for a, v in zip(tt, bb)][:40]}
        if step["kind"] == "move":
            d = np.hypot(xx - xx[0], zz - zz[0])
            moved = np.nonzero(d > 1.0)[0]
            row["s_to_move_1m"] = round(float(tt[moved[0]]), 2) if len(moved) else None
            sp = []
            for k in range(1, 9):
                lo, hi = at(tt, d, k - 1.0), at(tt, d, k + 0.0)
                sp.append(None if not np.isfinite(lo) or not np.isfinite(hi) else round(hi - lo, 2))
            row["speed_by_s"] = sp
            if run_speed:
                row["run_share_by_s"] = [None if v is None else round(v / run_speed, 2) for v in sp]
                fast = [k + 1 for k, v in enumerate(sp) if v is not None and v >= 0.9 * run_speed]
                row["s_to_0.9_run"] = fast[0] if fast else None
        men = [(tm - t0, a) for tm, a, _ in lane.get("men", []) if t0 - 0.6 <= tm <= t0 + 12 and len(a) >= 4]
        if len(men) >= 3:
            p0 = np.asarray(men[0][1], float).reshape(-1, 2) / 10
            p_end = np.asarray(men[-1][1], float).reshape(-1, 2) / 10
            n = min(len(p0), len(p_end))
            p0, p_end = p0[:n], p_end[:n]
            row["men_moved_m"] = round(float(np.median(np.hypot(*(p_end - p0).T))), 1)
            # each man's place along the facing, before (old facing) and after (new facing, from the centre)
            fb, fe = np.radians(bb[0]), np.radians(bb[-1])
            c0, ce = p0.mean(0), p_end.mean(0)
            along0 = (p0 - c0) @ np.array([np.sin(fb), np.cos(fb)])
            along1 = (p_end - ce) @ np.array([np.sin(fe), np.cos(fe)])
            if np.std(along0) > 0.1 and np.std(along1) > 0.1:
                row["rank_kept_corr"] = round(float(np.corrcoef(along0, along1)[0, 1]), 2)
            row["men_window_s"] = round(float(men[-1][0]), 1)
        out.append(row)
    return out


def turn_report(run_dirs, sim=False, device="cpu", copies=4, out=None, params=None):
    """The turning lanes (mode script) of the runs: the game's table and, with sim, the simulator's on the same
    lanes. Writes build/charge-probe/turns.json (not in Git)."""
    units = json.loads(UNITS_JSON.read_text(encoding="utf-8"))["units"]
    lanes = [ln for d in run_dirs for ln in load_run(d) if ln["spec"]["mode"] == "script"]
    result = {"game": [], "sim": []}
    for ln in lanes:
        run = units[ln["spec"]["a_key"]]["speed"]["run"]
        result["game"].append({"cell": cell(ln["spec"]), "run": ln["run"], "steps": turn_measure(ln, run)})
    if sim and lanes:
        for ln in sim_lanes(lanes, params=params, device=device, copies=copies):
            run = units[ln["spec"]["a_key"]]["speed"]["run"]
            result["sim"].append({"cell": cell(ln["spec"]), "run": "sim", "steps": turn_measure(ln, run)})
    keys = ("turn_deg", "s_to_10deg", "s_to_move_1m", "s_to_0.9_run", "run_share_by_s", "men_moved_m", "rank_kept_corr")
    for who in ("game", "sim"):
        for r in result[who]:
            for st in r["steps"]:
                print(who, r["cell"], st["step"], st["kind"], {k: st.get(k) for k in keys if st.get(k) is not None})
    out = out or ROOT / "turns.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(out)
    return result


# ---------------------------------------------------------------- melee exit and vigour (plan move, battle 2)

FAT_LEVELS = ("threshold_fresh", "threshold_active", "threshold_winded", "threshold_tired", "threshold_very_tired",
              "threshold_exhausted")


def _level(v):
    """A fatigue level index from the game's name or the simulator's index."""
    if v is None:
        return np.nan
    if isinstance(v, str):
        return float(FAT_LEVELS.index(v)) if v in FAT_LEVELS else np.nan
    return float(v)


def exit_measure(lane):
    """A withdraw lane after its order ('out'): seconds the leaver stays in melee (the last melee sample), its kills
    and the HP its enemy lost in 0-24 s and 26-50 s after the order, its own HP lost in 0-24 s; an attack lane with
    a_ability: the attacker's fatigue level times (first sample at each level after contact) and the ability's time."""
    spec, smp = lane["spec"], lane["samples"]
    t = np.array([x[0] for x in smp])
    out = {"cell": cell(spec), "run": lane["run"]}
    outs = [p["t"] / 1000 for p in lane.get("phases", []) if p["phase"] == "out"]
    if spec["mode"] == "withdraw" and outs:
        o = outs[0]
        m = np.array([bool((x[1] or {}).get("m")) for x in smp])
        after = (t >= o) & m
        out["in_melee_s"] = round(float(t[after][-1] - o), 1) if after.any() else 0.0
        k, ehp, hp = series(smp, "a", "k"), series(smp, "tg", "hp"), series(smp, "a", "hp")
        def r(v, p=0):
            return None if not np.isfinite(v) else round(float(v), p)
        for lo, hi in ((0, 24), (26, 50)):
            out[f"kills_{lo}_{hi}"] = r(at(t, k, o + hi) - at(t, k, o + lo), 1)
            out[f"enemy_hp_{lo}_{hi}"] = r(at(t, ehp, o + lo) - at(t, ehp, o + hi))
        out["own_hp_0_24"] = r(at(t, hp, o) - at(t, hp, o + 24))
    if spec.get("a_ability"):
        c = lane["contacts"].get(1) or 0.0
        lv = np.array([_level((x[1] or {}).get("fat")) for x in smp])
        out["ability_at_s"] = spec.get("a_ability_after_s")
        out["level_at_s"] = {FAT_LEVELS[j][10:]: round(float(t[np.nonzero(lv >= j)[0][0]] - c), 1)
                             for j in range(1, 6) if (lv >= j).any()}
        pts = series(smp, "a", "fatigue")
        if np.isfinite(pts).any():
            out["points_by_10s"] = [None if not np.isfinite(at(t, pts, c + s)) else round(at(t, pts, c + s))
                                    for s in range(0, 101, 10)]
    return out


def exit_report(run_dirs, sim=False, device="cpu", copies=4, params=None):
    """The melee-exit and vigour lanes (plan move battle 2) of the runs: the game's numbers and, with sim, the
    simulator's on the same lanes."""
    lanes = [ln for d in run_dirs for ln in load_run(d)
             if ln["spec"]["mode"] == "withdraw" or ln["spec"].get("a_ability")]
    rows = [exit_measure(ln) for ln in lanes]
    if sim and lanes:
        rows += [exit_measure(ln) for ln in sim_lanes(lanes, params=params, device=device, copies=copies)]
    for r in rows:
        print(r)
    return rows


# ---------------------------------------------------------------- tables

KEYS = ("contact_s", "speed_30_60", "speed_last30", "speed_last10", "speed_peak30",
        "tg_hp_0_1", "tg_hp_0_2", "tg_hp_0_5", "tg_hp_5_15", "tg_hp_15_30", "tg_hp_steady", "tg_men_steady",
        "a_hp_0_1", "a_hp_0_2", "a_hp_0_5", "a_hp_5_15", "a_hp_15_30", "a_hp_steady", "a_men_steady",
        "tg_hp2_0_5", "a_hp2_0_3", "tg_hp_0_18", "a_hp_0_18",
        # the reengage plan: the first 10 s and the steady rate of each fight, their ratio (the wave), the fatigue
        "tg_hp_0_10", "a_hp_0_10", "tg_wave_10", "a_wave_10", "tire_s", "a_fat_c1", "tg_fat_c1",
        "tg_hp2_0_10", "a_hp2_0_10", "tg_hp2_steady", "a_hp2_steady", "tg_wave2_10", "a_wave2_10", "a_fat_c2", "tg_fat_c2")


def summary(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["cell"], []).append(r)
    out = []
    for name, rs in sorted(groups.items()):
        row = {"cell": name, "n": len(rs)}
        for k in KEYS:
            v = [r[k] for r in rs if r.get(k) is not None and np.isfinite(r[k])]
            if v:
                row[k] = (float(np.mean(v)), float(np.min(v)), float(np.max(v)))
        for k in NEAR_KEYS:
            v = [r[k] for r in rs if r.get(k)]
            if v:
                row[k] = [round(float(x), 1) for x in np.mean(v, axis=0)]
        out.append(row)
    return out


def report(run_dirs, sim=False, out=OUT, device="cpu", copies=8):
    lanes = [ln for d in run_dirs for ln in load_run(d)]
    game = [measure(ln) for ln in lanes]
    table = summary(game)
    result = {"lanes": game, "summary": table}
    sim_table = None
    if sim:
        sims = sim_lanes([ln for ln in lanes if ln["contacts"].get(1) is not None], device=device, copies=copies)
        sim_rows = [measure(ln) for ln in sims]
        sim_table = {r["cell"]: r for r in summary(sim_rows)}
        result["sim_summary"] = list(sim_table.values())
    f = lambda v, p=0: "-" if v is None else f"{v:.{p}f}"
    print("HP lost after the first contact (target tg / attacker a), mean of the cell's lanes; sim = the simulator "
          "on the same lanes")
    for r in table:
        print(f"\n{r['cell']}  (n={r['n']})")
        s = (sim_table or {}).get(r["cell"], {})
        for k in KEYS:
            if k in r:
                g = r[k]
                sv = s.get(k)
                p = 2 if any(w in k for w in ("speed", "men", "wave", "fat")) else 0
                extra = "" if not sim_table else f"   sim {f(sv and sv[0], p)}"
                print(f"  {k:14} {f(g[0], p):>8} ({f(g[1], p)}-{f(g[2], p)}){extra}")
        for k in NEAR_KEYS:
            if k in r:
                print(f"  {k:14} men within {RADII} m of an enemy: {r[k]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(out)
    return result


# ---------------------------------------------------------------- a shooter's new target (plan retarget)

def retarget_aim(t_s, switch_s):
    """The retarget lane's target t_s after the go: 1 without switch_s, else 1 and 2 in turns of switch_s seconds
    (entries/charge_probe.lua retarget_aim)."""
    if not switch_s:
        return 1
    return 1 if int(math.floor(t_s / switch_s + 1e-9)) % 2 == 0 else 2


def _drop(t, v, lo, hi):
    """How much v fell over (lo, hi] (NaN outside the recording)."""
    return at(t, v, lo) - at(t, v, hi)


def _fin(v):
    return v is not None and np.isfinite(v)


def retarget_measure(lane, lag_s=RETARGET_LAG_S):
    """A retarget lane (game or simulator): per order (probe_phase 'aim') the seconds to its first shot, its shots per
    man in 0-5 and 5-10 s after it, the first second of fire's share of the men (the first burst), the aimed target's
    health lost per shot in the first 5 s of fire and later (the hits counted to lag_s after each span: the game's
    projectiles fly and its health readout follows; the simulator lands them on the shot's step, lag 0), the other
    target's (spill);
    per lane the shots, whole volleys (>= 50 % of the men in one second) and partial ones (10-50 %), the fire flag's
    share of the samples (game) and the men each target lost. Shots = the drop of the shooter's ammo (one a man)."""
    spec, s = lane["spec"], lane["samples"]
    lag_s = 0.0 if lane["run"] == "sim" else lag_s
    t = np.array([x[0] for x in s])
    ammo, men_a = series(s, "a", "ammo"), series(s, "a", "men")
    t2 = lane.get("t2") or []
    tt = {1: t, 2: np.array([x[0] for x in t2]) if t2 else t}
    hp, men = {1: series(s, "tg", "hp")}, {1: series(s, "tg", "men")}
    for key, d in (("hp", hp), ("men", men)):
        d[2] = (np.array([np.nan if r.get(key) is None else float(r[key]) for _, r in t2]) if t2
                else np.full(len(t), np.nan))
    out = {"run": lane["run"], "lane": spec["name"], "cell": cell(spec), "shooter": SHORT[spec["a_key"]],
           "kind": spec.get("kind"), "switch_s": spec.get("switch_s")}
    ok = np.isfinite(ammo)
    if ok.sum() < 3:
        return out
    shot_t = t[1:]
    shots = np.where(ok[1:] & ok[:-1], np.maximum(0.0, ammo[:-1] - ammo[1:]), 0.0)
    end = float(t[-1])
    men0 = float(np.nanmax(men_a)) if np.isfinite(men_a).any() else np.nan

    def n_shots(lo, hi):
        return float(shots[(shot_t > lo + 1e-6) & (shot_t <= hi + 1e-6)].sum())
    aims = sorted((p["t"] / 1000, int(p.get("target", 1))) for p in lane.get("phases", []) if p.get("phase") == "aim")
    rows = []
    for i, (o, who) in enumerate(aims):
        e = aims[i + 1][0] if i + 1 < len(aims) else end
        if e - o < 1:
            continue
        m = at(t, men_a, o)
        m = m if np.isfinite(m) and m > 0 else men0
        other = 2 if who == 1 else 1
        r = {"t": round(o, 1), "target": who, "span_s": round(e - o, 1), "men": m, "shots": n_shots(o, e)}
        fired = shot_t[(shot_t > o + 1e-6) & (shot_t <= e + 1e-6) & (shots > 0)]
        r["first_s"] = round(float(fired[0] - o), 2) if len(fired) else None
        for lo, hi in ((0, 5), (5, 10)):
            if o + hi <= e + 1e-6:
                r[f"per_man_{lo}_{hi}"] = n_shots(o + lo, o + hi) / m
        if len(fired):
            f0 = float(fired[0]) - 0.5                 # the drop shows at the sample after the shots
            first_hi = min(f0 + 5.0, e)
            r["burst_share"] = n_shots(f0, f0 + 1.0) / m
            r["shots_first"] = n_shots(f0, first_hi)
            r["hp_first"] = _drop(tt[who], hp[who], f0, first_hi + lag_s)
            r["spill_first"] = _drop(tt[other], hp[other], f0, first_hi + lag_s)
            if e > first_hi + 1:
                r["shots_later"] = n_shots(first_hi, e)
                r["hp_later"] = _drop(tt[who], hp[who], first_hi + lag_s, e + lag_s)
                r["spill_later"] = _drop(tt[other], hp[other], first_hi + lag_s, e + lag_s)
        rows.append(r)
    out["orders"] = rows
    out["shots"] = float(shots.sum())
    out["men"] = men0
    # volleys: the shots in each whole second of the lane over the men
    bins = {}
    for ts, n in zip(shot_t, shots):
        if n > 0:
            k = int(math.floor(ts - 1e-6))
            bins[k] = bins.get(k, 0.0) + n
    shares = [n / men0 for n in bins.values()] if np.isfinite(men0) and men0 > 0 else []
    out["whole_volleys"] = sum(1 for x in shares if x >= 0.5)
    out["partial_volleys"] = sum(1 for x in shares if 0.1 <= x < 0.5)
    flags = [(x[1] or {}).get("fire") for x in s]
    if any(f is not None for f in flags):
        on = np.array([bool(f) for f in flags])
        out["fire_share"] = float(on.mean())
        # the fire flag on with no shot showing in that half second or the next
        sh = np.concatenate([[0.0], shots])
        nxt = np.concatenate([sh[1:], [0.0]])
        out["flag_no_shot"] = float(np.mean((sh[on] + nxt[on]) == 0)) if on.any() else None
    for who in (1, 2):
        span = len(tt[who]) > 1
        out[f"t{who}_men_lost"] = _drop(tt[who], men[who], float(tt[who][0]), float(tt[who][-1])) if span else None
        out[f"t{who}_hp_lost"] = _drop(tt[who], hp[who], float(tt[who][0]), float(tt[who][-1])) if span else None
    hp_all = sum(v for v in (out["t1_hp_lost"], out["t2_hp_lost"]) if _fin(v))
    out["hp_per_shot"] = hp_all / out["shots"] if out["shots"] else None
    return out


RETARGET_KEYS = ("first_s_median", "per_man_0_5", "per_man_5_10", "burst_share", "hp_per_shot_first",
                 "hp_per_shot_later", "spill_share", "hp_per_shot", "shots", "whole_volleys", "partial_volleys",
                 "fire_share", "flag_no_shot", "t1_men_lost", "t2_men_lost")


def retarget_summary(rows):
    """{cell: {key: value}} over the lanes of each cell: per-order means; health per shot as sums over sums."""
    groups = {}
    for r in rows:
        groups.setdefault(r["cell"], []).append(r)
    out = {}
    for name, rs in sorted(groups.items()):
        orders = [o for r in rs for o in r.get("orders", [])]
        row = {"cell": name, "n": len(rs), "orders_n": len(orders),
               "no_shot_orders": sum(1 for o in orders if o.get("first_s") is None)}
        first = [o["first_s"] for o in orders if o.get("first_s") is not None]
        row["first_s_median"] = float(np.median(first)) if first else None
        for k in ("per_man_0_5", "per_man_5_10", "burst_share"):
            v = [o[k] for o in orders if _fin(o.get(k))]
            row[k] = float(np.mean(v)) if v else None

        def ratio(num, den):
            pairs = [(o[num], o[den]) for o in orders if _fin(o.get(num)) and o.get(den)]
            sd = sum(d for _, d in pairs)
            return sum(n for n, _ in pairs) / sd if sd else None
        row["hp_per_shot_first"] = ratio("hp_first", "shots_first")
        row["hp_per_shot_later"] = ratio("hp_later", "shots_later")
        aimed = sum(o[k] for o in orders for k in ("hp_first", "hp_later") if _fin(o.get(k)))
        spill = sum(o[k] for o in orders for k in ("spill_first", "spill_later") if _fin(o.get(k)))
        row["spill_share"] = spill / (aimed + spill) if aimed + spill > 0 else None
        for k in ("hp_per_shot", "shots", "whole_volleys", "partial_volleys", "fire_share", "flag_no_shot",
                  "t1_men_lost", "t2_men_lost"):
            v = [r[k] for r in rs if _fin(r.get(k))]
            row[k] = float(np.mean(v)) if v else None
        out[name] = row
    return out


def retarget_report(run_dirs, sim=False, device="cpu", copies=8, out=None, params=None):
    """The retarget lanes of the runs: the game's table and, with sim, the simulator's on the same lanes. Writes
    build/charge-probe/retarget.json (not in Git)."""
    lanes = [ln for d in run_dirs for ln in load_run(d) if ln["spec"]["mode"] == "retarget"]
    game = [retarget_measure(ln) for ln in lanes]
    result = {"lanes": game, "summary": retarget_summary(game)}
    if sim and lanes:
        sims = [retarget_measure(ln) for ln in sim_lanes(lanes, params=params, device=device, copies=copies)]
        result["sim_lanes"] = sims
        result["sim_summary"] = retarget_summary(sims)
    f = lambda v: "-" if v is None else f"{v:.2f}"
    print("A shooter's new target, per cell (A one order; B10 / C5 the other target every 10 / 5 s); sim = the "
          "simulator on the same lanes")
    for name, r in result["summary"].items():
        s_ = (result.get("sim_summary") or {}).get(name, {})
        print(f"\n{name}  (n={r['n']}, orders {r['orders_n']}, without a shot {r['no_shot_orders']})")
        for k in RETARGET_KEYS:
            print(f"  {k:18} {f(r.get(k)):>8}" + ("" if not sim else f"   sim {f(s_.get(k))}"))
    out = out or ROOT / "retarget.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(out)
    return result


# ---------------------------------------------------------------- the shooters' skirmish mode (plan skirmish)

def skirmish_measure(lane):
    """A skirmish lane (game): the shooter (the lane's target) and its chaser. Speeds from the places (differences
    over 1 s). A move away: the shooter's speed away from the chaser (along the line chaser -> shooter) above
    SKIRMISH_MOVE_MPS for SKIRMISH_MOVE_S, before the contact. Returns the lane's row: moves (how many); the first
    one's start (move_s, s after the go; move_d_m, the centres' distance then), its direction against the line away
    from the chaser (move_dir_deg, 0 = straight away, to the contact or the end) and the mean / peak speed away while
    moving (move_speed, move_peak); away_share, the share of the time before contact moving away; the fire flag's
    share and the shots (the ammo's drop, one a man) standing and moving away; the chaser's median speed; the contact
    (contact_s, caught) and contact_after_move_s; end_d_m (the distance at the end); sk_share (the mode's flag on) and
    sk_set (probe_skirmish: can, before, after)."""
    spec, s = lane["spec"], lane["samples"]
    t = np.array([x[0] for x in s])
    ax, az, tx, tz = (series(s, w, k) for w, k in (("a", "x"), ("a", "z"), ("tg", "x"), ("tg", "z")))
    ammo, fire, sk = series(s, "tg", "ammo"), series(s, "tg", "fire"), series(s, "tg", "sk")
    # the catch: the shooter's own melee flag (in skirmish2 'neighbour' the lane's contact is the neighbour's fight)
    t_m = series(s, "tg", "m")
    c = float(t[np.nan_to_num(t_m, nan=0.0) > 0][0]) if (np.nan_to_num(t_m, nan=0.0) > 0).any() else None
    if c is None and not np.isfinite(t_m).any():
        c = lane["contacts"].get(1)
    out = {"run": lane["run"], "lane": spec["name"], "shooter": SHORT[spec["t_key"]], "chaser": SHORT[spec["a_key"]],
           "mode": "on" if spec.get("t_skirmish") else "off", "kind": spec.get("kind"), "contact_s": c,
           "caught": c is not None, "min_d_m": None,
           "sk_set": next(({k: p.get(k) for k in ("can", "before", "after")} for p in lane.get("skirmish", [])), None)}
    if len(t) < 3:
        return out
    dist = np.hypot(tx - ax, tz - az)
    ux, uz = (tx - ax) / np.maximum(dist, 1e-6), (tz - az) / np.maximum(dist, 1e-6)    # away from the chaser
    lag = max(1, int(round(1.0 / max(float(np.median(np.diff(t))), 1e-3))))           # samples a second
    vx, vz, cvx, cvz = (np.full(len(t), np.nan) for _ in range(4))
    dt = t[lag:] - t[:-lag]
    vx[lag:], vz[lag:] = (tx[lag:] - tx[:-lag]) / dt, (tz[lag:] - tz[:-lag]) / dt
    cvx[lag:], cvz[lag:] = (ax[lag:] - ax[:-lag]) / dt, (az[lag:] - az[:-lag]) / dt
    away = vx * ux + vz * uz
    before = t < (c if c is not None else np.inf)
    moving = np.nan_to_num(away, nan=0.0) > SKIRMISH_MOVE_MPS
    # a move: SKIRMISH_MOVE_S of moving samples in a row; it starts at the last place before the first step away
    # within the speed's second before the first of them (a step: over 0.1 m away between two samples)
    step = np.concatenate([[0.0], (tx[1:] - tx[:-1]) * ux[:-1] + (tz[1:] - tz[:-1]) * uz[:-1]])
    need = max(1, int(round(SKIRMISH_MOVE_S * lag)))
    moves, k = [], 0
    while k < len(t):
        if moving[k] and before[k]:
            j = k
            while j + 1 < len(t) and moving[j + 1] and before[j + 1]:
                j += 1
            if j - k + 1 >= need:
                first = next((i for i in range(max(1, k - lag + 1), k + 1) if step[i] > 0.1), k)
                moves.append((first - 1, j))
            k = j + 1
        else:
            k += 1
    out["moves"] = len(moves)
    pre = np.isfinite(away) & before
    out["away_share"] = round(float(np.mean(moving[pre])), 2) if pre.any() else None
    in_move = np.zeros(len(t), bool)
    for a, b in moves:
        in_move[a + 1:b + 1] = True
    if moves:
        k0 = moves[0][0]
        out["move_s"], out["move_d_m"] = round(float(t[k0]), 1), round(float(dist[k0]), 1)
        end = len(t) - 1 if c is None else max(k0 + 1, int(np.searchsorted(t, c)) - 1)
        dx, dz = tx[end] - tx[k0], tz[end] - tz[k0]
        out["move_dir_deg"] = round(math.degrees(math.atan2(dx * uz[k0] - dz * ux[k0], dx * ux[k0] + dz * uz[k0])))
        out["move_speed"] = round(float(np.nanmean(away[in_move])), 2)
        out["move_peak"] = round(float(np.nanmax(away[in_move])), 2)
    drop = np.concatenate([[0.0], -np.diff(ammo)])
    shots = np.where(np.isfinite(drop), np.maximum(0.0, drop), 0.0)
    for name, sel in (("stand", before & ~in_move), ("moving", in_move)):
        f = fire[sel]
        f = f[np.isfinite(f)]
        out[f"fire_{name}"] = round(float(np.mean(f)), 2) if len(f) else None
        out[f"shots_{name}"] = int(shots[sel].sum())
        out[f"secs_{name}"] = round(float(sel.sum()) / lag, 1)
    out["shots_total"] = int(shots.sum())
    cs = np.hypot(cvx, cvz)[before & np.isfinite(cvx)]
    out["chaser_speed"] = round(float(np.median(cs)), 2) if len(cs) else None
    if c is not None and "move_s" in out:
        out["contact_after_move_s"] = round(c - out["move_s"], 1)
    out["end_d_m"] = round(float(dist[-1]), 1)
    pre_d = dist[before & np.isfinite(dist)]
    out["min_d_m"] = round(float(pre_d.min()), 1) if len(pre_d) else None
    if spec.get("kind") == "long":
        # speed (m/s, from the places) and the last fatigue state per window; the run flags' share (is_moving_fast)
        fat = [(x[0], (x[2] or {}).get("fat"), (x[1] or {}).get("fat")) for x in s]
        fast = series(s, "tg", "fast")
        sp, csp = np.hypot(vx, vz), np.hypot(cvx, cvz)
        for lo, hi in SKIRMISH_WINDOWS:
            w = (t > lo) & (t <= hi) & np.isfinite(sp)
            out[f"speed_{lo}_{hi}"] = round(float(np.mean(sp[w])), 2) if w.any() else None
            out[f"chaser_speed_{lo}_{hi}"] = round(float(np.nanmean(csp[w])), 2) if w.any() else None
            f_ = fast[w]
            out[f"fast_{lo}_{hi}"] = round(float(np.mean(f_[np.isfinite(f_)])), 2) if np.isfinite(f_).any() else None
            last = [f for f in fat if lo < f[0] <= hi]
            out[f"fat_{lo}_{hi}"] = last[-1][1] if last else None
            out[f"chaser_fat_{lo}_{hi}"] = last[-1][2] if last else None
    f_sk = sk[np.isfinite(sk)]
    out["sk_share"] = round(float(np.mean(f_sk)), 2) if len(f_sk) else None
    return out


SKIRMISH_KEYS = ("caught", "contact_s", "moves", "move_d_m", "move_s", "move_dir_deg", "move_speed", "move_peak",
                 "away_share", "fire_stand", "fire_moving", "shots_stand", "shots_moving", "secs_moving",
                 "chaser_speed", "end_d_m", "min_d_m", "sk_share") + tuple(
                     f"{k}_{lo}_{hi}" for lo, hi in SKIRMISH_WINDOWS for k in ("speed", "chaser_speed", "fast"))


def skirmish_summary(rows):
    """{'<shooter> <mode>': {n, the mean of each SKIRMISH_KEYS over its lanes}}."""
    cells = {}
    for r in rows:
        kind = r.get("kind")
        cells.setdefault(f"{r['shooter']} {r['mode']}" + (f" {kind}" if kind not in (None, "on", "off") else ""),
                         []).append(r)
    out = {}
    for name, rs in sorted(cells.items()):
        row = {"n": len(rs)}
        for k in SKIRMISH_KEYS:
            v = [float(r[k]) for r in rs if r.get(k) is not None]
            row[k] = round(float(np.mean(v)), 2) if v else None
        for k in (f"{w}fat_{lo}_{hi}" for lo, hi in SKIRMISH_WINDOWS for w in ("", "chaser_")):
            v = [r[k] for r in rs if r.get(k)]
            if v:
                row[k] = ",".join(v)
        out[name] = row
    return out


def skirmish_report(run_dirs, out=None):
    """The skirmish lanes of the runs: a row per lane and the mean per shooter and mode. Writes
    build/charge-probe/skirmish.json (not in Git)."""
    lanes = [ln for d in run_dirs for ln in load_run(d) if ln["spec"].get("target_mode") == "skirmish"]
    rows = [skirmish_measure(ln) for ln in lanes]
    table = skirmish_summary(rows)
    f = lambda v: "-" if v is None else f"{v:g}"
    print("The shooters' skirmish mode: per shooter and mode (on / off), the mean over its lanes")
    for name, r in table.items():
        print(f"\n{name}  (n={r['n']})")
        for k in SKIRMISH_KEYS:
            if r[k] is not None:
                print(f"  {k:20} {f(r[k]):>8}")
        for k, v in r.items():
            if "fat_" in k:
                print(f"  {k:20} {v}")
    result = {"lanes": rows, "summary": table}
    out = out or ROOT / "skirmish.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(out)
    return result


# ---------------------------------------------------------------- a shooter moving in melee (plan shootcontact)

SHOOTCONTACT_KEYS = ("contact_s", "melee_s", "loss", "moving_share", "melee_s_moving", "loss_moving", "melee_s_still",
                     "loss_still", "speed_melee", "inf_loss", "shots", "shots_melee", "hp_end")


def shootcontact_measure(lane):
    """A shootcontact lane (game): the shooter (the lane's target) in melee - its own melee flag on at both ends of a
    sample interval. loss: its HP lost (unary_hitpoints, % of the full unit) a second in melee; _moving / _still by its
    is_moving flag at the interval's end (melee_s_* the seconds); moving_share; speed_melee (m/s, from the places);
    inf_loss: the infantry's HP lost %/s in the same intervals; shots: the ammo's drop (one a man), all and in melee;
    hp_end: the shooter's health left at the end; contact_s: its first melee flag (s after the go)."""
    spec, s = lane["spec"], lane["samples"]
    t = np.array([x[0] for x in s])
    out = {"run": lane["run"], "lane": spec["name"], "shooter": SHORT[spec["t_key"]], "infantry": SHORT[spec["a_key"]],
           "kind": spec.get("kind")}
    m = np.nan_to_num(series(s, "tg", "m"), nan=0.0) > 0
    mv = np.nan_to_num(series(s, "tg", "mv"), nan=0.0) > 0
    hp, a_hp = series(s, "tg", "hpu"), series(s, "a", "hpu")
    x, z, ammo = series(s, "tg", "x"), series(s, "tg", "z"), series(s, "tg", "ammo")
    out["contact_s"] = round(float(t[m][0]), 1) if m.any() else None
    ok_hp = np.isfinite(hp)
    out["hp_end"] = round(float(hp[ok_hp][-1]), 3) if ok_hp.any() else None
    drop = np.concatenate([[0.0], -np.diff(ammo)])
    shots = np.where(np.isfinite(drop), np.maximum(0.0, drop), 0.0)
    out["shots"] = int(shots.sum())
    if len(t) < 2:
        return out
    dt = np.diff(t)
    both = m[1:] & m[:-1]
    out["shots_melee"] = int(shots[1:][both].sum())
    lost = 100 * np.maximum(0.0, hp[:-1] - hp[1:])
    a_lost = 100 * np.maximum(0.0, a_hp[:-1] - a_hp[1:])
    speed = np.hypot(np.diff(x), np.diff(z)) / np.maximum(dt, 1e-6)
    moving = mv[1:]

    def rate(sel, v):
        sel = sel & np.isfinite(v)
        return (round(float(v[sel].sum() / dt[sel].sum()), 3) if dt[sel].sum() > 0 else None), round(float(dt[sel].sum()), 1)
    out["loss"], out["melee_s"] = rate(both, lost)
    out["loss_moving"], out["melee_s_moving"] = rate(both & moving, lost)
    out["loss_still"], out["melee_s_still"] = rate(both & ~moving, lost)
    out["inf_loss"], _ = rate(both, a_lost)
    out["moving_share"] = round(float(dt[both & moving].sum() / dt[both].sum()), 2) if dt[both].sum() > 0 else None
    sp = both & np.isfinite(speed)
    out["speed_melee"] = round(float(np.mean(speed[sp])), 2) if sp.any() else None
    return out


def shootcontact_summary(rows):
    """{'<shooter> <kind>': {n, the mean of each SHOOTCONTACT_KEYS over its lanes}}."""
    cells = {}
    for r in rows:
        cells.setdefault(f"{r['shooter']} {r['kind']}", []).append(r)
    out = {}
    for name, rs in sorted(cells.items()):
        row = {"n": len(rs)}
        for k in SHOOTCONTACT_KEYS:
            v = [float(r[k]) for r in rs if r.get(k) is not None]
            row[k] = round(float(np.mean(v)), 3) if v else None
        out[name] = row
    return out


def shootcontact_report(run_dirs, out=None):
    """The shootcontact lanes of the runs: a row per lane and the mean per shooter and kind. Writes
    build/charge-probe/shootcontact.json (not in Git)."""
    lanes = [ln for d in run_dirs for ln in load_run(d) if ln.get("plan") == "shootcontact"]
    rows = [shootcontact_measure(ln) for ln in lanes]
    table = shootcontact_summary(rows)
    f = lambda v: "-" if v is None else f"{v:g}"
    print("A shooter in melee (clanrats attacking from 30 m): per shooter and kind, the mean over its lanes; loss = "
          "the shooter's HP lost %/s in melee (moving / still: its is_moving flag); inf_loss = the clanrats'")
    print(f"  {'cell':18} {'n':>2} " + " ".join(f"{k:>13}" for k in SHOOTCONTACT_KEYS))
    for name, r in table.items():
        print(f"  {name:18} {r['n']:>2} " + " ".join(f"{f(r[k]):>13}" for k in SHOOTCONTACT_KEYS))
    result = {"lanes": rows, "summary": table}
    out = out or ROOT / "shootcontact.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(out)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=("report", "plan", "run", "turns", "exits", "retarget", "skirmish",
                                                     "shootcontact"),
                        default="report")
    parser.add_argument("runs", nargs="*", type=Path, help="report: run folders (default: build/charge-probe/runs/*)")
    parser.add_argument("--plan", choices=PLANS, default="charge")
    parser.add_argument("--battles", help="run: battle numbers, comma separated (default: all of the plan)")
    parser.add_argument("--sim", action="store_true", help="report: also the simulator on the same lanes (torch)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--copies", type=int, default=8)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.command == "plan":
        for i, b in enumerate(battles(args.plan), 1):
            print(i, [cell(l) for l in b])
        return 0
    if args.command == "run":
        which = [int(x) for x in args.battles.split(",")] if args.battles else None
        done = run(args.plan, which, dry=args.dry)
        print("done:", done)
        return 0 if all(d[-1] == 0 for d in done) else 1
    if args.command == "exits":
        exit_report(args.runs or runs(), sim=args.sim, device=args.device, copies=min(args.copies, 4))
        return 0
    if args.command == "retarget":
        retarget_report(args.runs or runs(), sim=args.sim, device=args.device, copies=args.copies,
                        out=None if args.out == OUT else args.out)
        return 0
    if args.command == "skirmish":
        skirmish_report(args.runs or runs(), out=None if args.out == OUT else args.out)
        return 0
    if args.command == "shootcontact":
        shootcontact_report(args.runs or runs(), out=None if args.out == OUT else args.out)
        return 0
    if args.command == "turns":
        turn_report(args.runs or runs(), sim=args.sim, device=args.device, copies=min(args.copies, 4),
                    out=None if args.out == OUT else args.out)
        return 0
    report(args.runs or runs(), sim=args.sim, out=args.out, device=args.device, copies=args.copies)
    return 0


if __name__ == "__main__":
    sys.exit(main())
