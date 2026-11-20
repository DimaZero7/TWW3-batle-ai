# side2r — the intercepted hammer

**Reject the patch.** Prioritising the ordered target closes **0.016% narrow and 0.029% wide** of the cforced−cfront gap against a 50% requirement. The narrow twin misses target damage in both minutes. On wide, the largest error is the hammer's own 120 s loss, 0.767 versus 0.544 in game.

## Where the loss occurs

**Game → original twin**, fractions of starting HP lost after the hammer order. “Ours” averages two 8,280-HP units. The night JSON stores the combined loss. Per-unit figures come from game event streams and twin tensors using the probe's interpolation (`game/probe/unit_probe.py:2532,2593`). All 160 game rows have complete windows.

| Width | Lane | Seconds | Target | Anvil | Hammer | Ours |
|---|---|---|---:|---:|---:|---:|
| narrow | cforced | 0–60 | 0.214→0.142 | 0.209→0.192 | 0.278→0.305 | 0.244→0.248 |
| narrow | cforced | 60–120 | 0.223→0.148 | 0.137→0.128 | 0.260→0.317 | 0.199→0.223 |
| narrow | flank | 0–60 | 0.228→0.179 | 0.202→0.183 | 0.045→0.029 | 0.124→0.106 |
| narrow | flank | 60–120 | 0.211→0.166 | 0.140→0.122 | 0.065→0.048 | 0.102→0.085 |
| wide | cforced | 0–60 | 0.234→0.211 | 0.270→0.324 | 0.308→0.410 | 0.289→0.367 |
| wide | cforced | 60–120 | 0.206→0.194 | 0.177→0.206 | 0.236→0.358 | 0.206→0.282 |
| wide | flank | 0–60 | 0.241→0.252 | 0.271→0.310 | 0.040→0.033 | 0.156→0.171 |
| wide | flank | 60–120 | 0.229→0.221 | 0.179→0.196 | 0.054→0.055 | 0.117→0.126 |

Paired against closing-front, the game's forced maneuver recovers trade in minute two (+0.039 narrow, +0.033 wide). The twin continues losing ground (−0.066, −0.074). On the unobstructed narrow flank the target deficit builds in both minutes. On wide flank the twin reproduces total target loss (0.473 versus 0.470), but wide **front** target loss is already 0.395 versus 0.351. The control accounts for much of the missing wide flank advantage.

Without the comer, all flank orders switch to attack at P. Arrival averages 22.9/23.9 s in game and 21.7/22.9 s in the twin (narrow/wide). First hammer melee is 32.3/33.8 versus 30.8/31.8 s. The recorded side is flank in 31/32 game and 497/512 twin lanes.

## Contact and mechanism

READ = code, CONTEXT = observations, GUESS = hypothesis. Simulator lines refer to `project/sim`. Game addresses refer to the local executable used by the [turn](../project/reverse/notes/melee_turn.md), [wrap](../project/reverse/notes/melee_wrap.md), [direction](../project/reverse/notes/melee_flee_module.md) and [charge](../project/reverse/notes/charge_module.md) notes.

- **KILLED — timeout means the target is unreachable.** [READ] P is an 8 m waypoint tolerance. At 30 s the script attacks the target anyway (`charge_probe.py:2876–2893`). [CONTEXT] Forced timeouts, narrow/wide, are 4/16 and 16/16 in game, 256/256 and 247/256 in the twin. Every hammer registers on the target. The recorded contact side is flank in 15/16 and 11/16 game lanes, 256/256 and 181/256 twin lanes. The label search begins 3 s before the hammer melee flag (`unit_probe.py:2384–2392`). First observed hammer swing at the target **after** the attack command averages 36.0/36.5 s from the original order. The twin first damages it at 48.7/46.8 s. Fourteen game lanes also record an earlier brush.
- **KILLED — a helpless hammer taking free rear hits throughout the move.** [CONTEXT] Forced hammer melee flags rise at 20.0/20.2 s in game, 19.7/20.5 s in the twin. The comer's own game flag rises at 20.6/21.4 s. All 32 hammers show `moving_fast` then, under MOVE, with zero charge at their first melee flag. Across those approaches, 2,035 hammer swing starts select the comer. The twin’s first actual comer contact is 19.7/20.8 s. All 512 are on MOVE. On that step 503 push and answer, nine are pinned. The first blow is frontal in 503, flank in nine. [READ] The simulation can push and fight on an existing MOVE (`battle.py:360,1613`). The game swing calls the struck soldier's conditional reply before the hit roll (`143080dfc → 143074dcc`, issue at `143074e9c`). This is an individual response. An immediate whole-formation turn is unsupported.
- **KILLED — the 24 s silence is spurious here.** [CONTEXT] Between 2 and 24 s after the forced attack order, sampled hammer attack-start markers select the comer 3 times and the target 1,423 times. Later both receive swings in all 32 battles. [READ] `143016902–143016929` clears the ordered enemy's timer and starts breakoff for other touching enemies. `14305b300` checks eligibility. The twin deals zero hammer damage to the comer during those first 24 s and drops zero attack orders. Its fresh incidental rate is zero (`melee.py:1541–1555`). Its separate facing-order timer requires a new facing **in contact** (`battle.py:399`).
- **UNDECIDED — sustained flank/rear bonus inflates hammer losses.** [READ] Game defence uses each defender's facing and the attacker's forward half-plane (`1430a6380/14309b698/142e53208`). The twin usually uses the unit-centre sector (`melee.py:1087–1109`). [CONTEXT] At the first poll with identified comer swing starts, 0/73 starts came from behind. After breakoff, 58% of identified starts came from the front, 6% from the rear. In one twin copy per start, 7%/23% of damaging comer ticks were frontal/rear. Poll angles approximate impact angles, and ticks weight frequency differently from swings.
- **SUPPORTED — charge can occur after interception.** [CONTEXT] In the first 13 s after the forced attack order, 461/464 identified hammer starts on the target had a live individual charge deadline. In the separate 32-lane baseline trace, 0/32 hammers had charge on first damage to the target. [READ] Game charge gait refreshes each soldier's deadline (`14307e1a0/14308cee8/14305e9c0`). The simulator clears unit run-up during sustained melee (`battle.py:534,551–553`). The size of the lost charge contribution is unresolved.

## One switch and its test

**SUPPORTED — both enemies draw hammer attacks after breakoff.** [CONTEXT] After the 24 s window, 46.7%/47.2% of identified game hammer swing starts select the target. In the baseline twin it receives 15.7%/17.9% of the hammer's damage to the two clanrat units. Starts and HP are distinct measures. [READ] The game assigns attack-order targets per soldier (`1432781b8`). Local search adds 3 for the ordered unit and multiplies the current target's score by 1.25 (`14308d400,14308d40e`). The twin selects the older contact when its cover rule finds no anchor (`core.py:106–115`), then allocates fighters by that choice (`core.py:551–583`).

[GUESS] `melee.core.ordered_contact=1` gives an in-contact attack target priority over the age fallback. This translates a soldier preference into a formation rule, with no new gameplay constants. In a one-copy-per-battle trace it changed the ordered target's share of simultaneous-contact ticks after breakoff from 1,723/4,159 to 4,159/4,159. **KILLED as the main cause.** In the full twin, mean late hammer damage to the target rose by less than 1 HP per lane at either width. The individual contributions of contact geometry, charge and facing remain unresolved.

| Probe and trade contrast at 120 s | Narrow game | Before | After | Wide game | Before | After |
|---|---:|---:|---:|---:|---:|---:|
| side2r flank − front | +.09603 | +.03920 | +.03920 | +.08852 | +.06109 | +.06109 |
| side2r cforced − cfront | +.02246 | −.17903 | −.17899 | −.01597 | −.23855 | −.23849 |
| side2r cteach − cforced | −.04676 | +.19225 | +.19222 | −.04089 | +.21208 | +.21201 |
| flank2v1 flank − front | +.1097 | +.0911 | +.0911 | +.1010 | +.0800 | +.0800 |
| flank2v1 rear − front | +.1121 | +.1132 | +.1132 | +.1805 | +.0370 | +.0370 |

CPU, recorded starts, seed 0. All 2,560 baseline trades reproduce the night file, which contains **16** copies per lane. Regression uses eight copies (320 rows) and its original hammer-contact clock. All regression rows match before/after. With the switch at its default 0, every tick's tensor state hashed identically to the original on two recorded battles of each plan (`7a145a08…4a2729`). [Patch](change.patch) · [Replay and measurements](work/)
