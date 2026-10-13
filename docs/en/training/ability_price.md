# The price of an ability press

[← Back](README.md) · [Documentation](../README.md) › [Training data](README.md) › The price of an ability press · [Русский](../../ru/training/ability_price.md)

An ability press priced by what its bonuses actually did in the battle (seconds of melee under a damage bonus, morale
points that held a unit up near the wavering threshold, and so on), and one generic check of that price against the
simulator's "pressed / not pressed" replay. Why: the replay is exact but expensive and noisy; the price by the bonuses
is cheap and every press has one. Code: `tools/nn/abilities_price/`; the network and the simulator do not change.

## In short

- **Three layers.** (1) A brick per bonus kind: a counter of the bonus's useful work (>= 0) and its description.
  (2) An ability is data: a set of bricks with the passport's numbers (`config/nn/abilities.json`), duration,
  recharge, radius, targets; no code per ability. (3) One generic validator that knows nothing about any bonus.
- **The rate k** (gold per counter unit) is one per bonus kind, not per ability; it is measured by regressing the
  replay prices on the counters.
- **The recharge cost depends on the situation**: what the same press would have given at the first moment of the
  recharge window when its bonus gets work, less the gain of the ability coming back earlier when pressed now;
  predicted from what is seen at the press. Not "the best press later": the maximum is biased up, with it the price was
  > 0 in 6 % of Stand Your Ground's presses.
- **The result on 4 563 presses of the network itS25_12** (2 560 of its own presses + 2 003 moments of a ready
  ability, the price at 60 s): Stand Your Ground, Deadly Onslaught and Foe-Seeker pass (rank correlation with the
  replay 0.74, 0.63, 0.44; Deadly Onslaught's sign agreement 0.65 - below the 0.7 threshold because of the recharge
  cost's level, see Result); Rally has no signal, Verminous Valour fails (section [Result](#result)).

## Layer 1: the bonus bricks

File `bonuses.py`. A press's trajectory is sampled once a second: sample 0 is the moment of the press (before it acts),
sample s the state after second s; the units are the presser's side. The mask "the bonus is on that unit in second s"
comes from layer 2. Each brick: `count(trajectory, mask, value) -> number >= 0`, `switch_off(...)` - the same
trajectory with its condition false (for the generated tests), `undo` / `apply` - take the bonus's own mark out of the
recorded trajectory and put it back (morale, fatigue; needed for a press moved later, layer 2). The description
(`Description`): kind, condition in words, whether the counter is proportional to the bonus's size, buff or debuff,
how the size scales (`mult`: (x - 1) x 100 %, `add`: the number), a typical value for the tests.

| Kind | Counter (unit) | Useful in second s when | Prop. |
|---|---|---|---|
| `melee_damage` (base + armour-piercing) | sum (x - 1) x 100 x [in melee] (% x s) | the unit is in melee | yes |
| `charge` | sum (x - 1) x 100 over charge impacts (% x impact) | the charge left rose: the first blow of a new charge | yes |
| `melee_defence` | sum over the affected bonus x [in melee] (points x s) | an affected unit is in melee | yes |
| `morale` (leadership) | sum over the affected: the lift's points weighted by how near the wavering and rout thresholds they are, at the worst second (points) | the morale without the bonus below 30 points | no |
| `speed` | sum (x - 1) x 100 x [moving, not in melee] (% x s) | the unit walks or runs out of melee | yes |
| `charge_speed` | sum (x - 1) x 100 x [charge sprint] (% x s) | an attack order, not in melee, moving, the target's edge within charge_dist | yes |
| `vigour` (buff) | sum (the fatigue tier's penalty without the ability - with it) over the stats at work that second (% x s) | the ability left the unit a tier fresher while it fights or moves | no |
| `fatigue_cost` (debuff) | the same with the opposite sign (% x s) | the ability left the unit a tier more tired | no |

- **Morale: a point weighted by the game's thresholds.** A morale point changes a battle only near the thresholds:
  below 16 points a unit wavers (`ums_wavering_threshold_upper`), below 0 it routs (`ums_wavering_threshold_lower`),
  below 30 it is shaken (`ums_steady_threshold_lower`, `game_rules.json`). A point's weight at m: 1 below 16, falling
  linearly to 0 at 30, 0 above 30. A unit's counter is the weight's area between its morale without the bonus (the
  recorded morale less the bonus's own lift) and with it, at the second it is largest while the bonus is on; per unit
  at most the bonus. The lift follows the game's morale pace (each 0.5 s tick the points move max(1, 15 % of the gap)
  towards the target). A unit at 30 points or more without the bonus gives 0, however far below its normal top - so
  the counter no longer grows with the battle time (the shortfall to the top used to pile up).
- **Vigour: a tier weighted by its penalties.** The fatigue "without the ability" is rebuilt from the recorded one:
  each second's change less the ability's (value x 30 000 a second while it is on), within 0..30 000; where the
  recorded fatigue sat at 0 or the maximum, the activity's own rate from the database is taken (`_kv_fatigue_tables`:
  melee +19, the sprint +34, moving +4, ready -7 a tick, 10 ticks a second). Each second: the difference of the tier's
  penalties without the ability and with it (the database's `unit_fatigue_effects_tables`, `sim.json
  fatigue.effects`) over the stats at work then: in melee attack, defence, armour-piercing damage, armour (fresh 0,
  active 5, winded 15, tired 25, very tired 45, exhausted 75 %); moving out of melee the speed (0, 0, 5, 10, 15, 15 %);
  at a charge's impact the charge bonus too (0, 0, 0, 10, 25, 30 %). Standing or resting - 0. From the press to the
  end of the window (the gap lasts after the ability ends). The morale of fatigue (-2 / -6 points very tired /
  exhausted) is not counted.
- **Foe-Seeker's sign.** The database's `fatigue_change_ratio` is -0.01, in the simulator -300 fatigue points a
  second, and the in-game probe (`build/movelords`) agrees: the General with it tires later (91.5 s against ~51). It is
  a **buff** (vigour). Layer 2 picks the brick by the number's sign: -0.01 -> `vigour`, a positive number ->
  `fatigue_cost`.

## Layer 2: an ability is data

File `ability.py`. `Ability.from_passport` maps the passport's effects to bricks by the (stat, how) pairs the
descriptions declare; several stats of one kind (damage: base and armour-piercing) give the mean (x - 1). A stat no
brick declares, or one on enemies, goes to `unpriced` - a new bonus kind is a new brick. The five pressable abilities
of S25 map completely (`unpriced` empty).

Who gets the bonus, as the simulator lays it (`tools/nn/sim/abilities.py`): **the lord himself** while the ability's
timer is > 0 and he is alive; **friends** with `update_targets` 1 - an aura that follows him (the friends within the
radius each second, Rally); with 0 - laid at the cast on the friends within the radius and kept `active_s` seconds
wherever they go (Stand Your Ground). A container test checks the masks against the simulator's own record
(`abilities.RECEIVED`): >= 97 % of the cells agree.

The press's price (gold):

    benefit  = sum over buffs of k_kind x counter
    debuffs  = sum over debuffs of k_kind x counter
    price    = benefit - debuffs - recharge cost

**The recharge cost is dynamic.** Pressed now, the ability is gone for recharge_s seconds; the cost is what it would
have given pressed later. `Ability.later` moves the press to second s of the same trajectory: the real press's mark is
taken out of the morale and fatigue (`undo`), the trajectory is cut at s, the ability's timer laid anew (`active_s`
seconds), the new press's mark put back (`apply`), the counters over the horizon from s. `next_useful` - the benefit
of the press at the **first** of the seconds 5, 10, ... recharge_s where its bonus has work (some counter > 0; the
game's AI presses so too - when the bonus has something to do), less the gain of the earlier return: pressed now
rather than tau seconds later, the ability is back tau seconds earlier - tau x the ability's mean press benefit /
(active_s + recharge_s) (`credit_gold_s` in `config/ability_price.json`: Stand Your Ground 0.11, Deadly Onslaught
0.055, Foe-Seeker 0.11 gold a second); >= 0, no work in the whole window - 0. This is the cost in hindsight: it sees
the future. The price takes its **prediction from the situation at the press**
(`situation`, `validate.situation_recharge`): a ridge regression on what is seen then (battle time, the lord in melee
/ moving, friends within 35 m and those of them in melee, own units standing, own units below 30 morale points,
wavering and routing, the lord's fatigue tier), one per ability, cross-fitted (each press by the other folds), >= 0.
No hand numbers per ability. The trajectory must cover the window: collected with a 120 s horizon (`--horizon 120`),
the price taken at the 60 s mark. Beside it in the card for comparison: the formulas' start (the mean benefit of the
useful presses x recharge_s / 60, `Recharge`) and the former cost `best_later` - the best press of the window
(`--target best`).

Why not the best press of the window: it is the maximum over the window's 18 seconds on one random trajectory - only
someone who knows the future picks so. On itS25_12's presses a Stand Your Ground press in melee gives 25.5 gold, the
same press at any given later second less (25.2 after 5 s ... 18.9 after 85 s), and the window's maximum 29.4: the
price was > 0 in 11 % of the presses in melee and in 14 % of those where the lord fought >= 15 s of the 18 under the
bonus. The copies' noise is not the cause: the maximum of the mean over 4 copies gives the same 10 %.

## Layer 3: the validator

File `validate.py`, one for every bonus: it knows only the brick interface, the trajectory's schema
(`bonuses.TRACE`), the abilities as data and the replay prices.

- **(a) Tests from the descriptions** on random trajectories: condition off -> 0; a bonus of size 0 -> 0; bonus x2 ->
  counter x2 (proportional) or not smaller; a longer action -> not smaller; >= 0; finite; not always 0. A broken
  brick (counting outside the mask, declared proportional but capped) is caught - a test.
- **(b) Distributions** over real presses: the share of prices > 0 (alarm > 95 % or < 5 %), tails beyond 3 sd
  (alarm > 1 %), a brick always 0, scale dependence - the price's rank correlation with the army's budget and the
  battle time against the replay's (alarm when they differ by more than 0.3).
- **(c) Agreement with the replay** of the same presses (the rates fitted on the other parts of the data, 5-fold
  cross-fitting): sign agreement of the price net of the recharge cost (the same cost taken off both), Spearman's rank
  correlation (the replay holds no recharge, so the benefit is compared with it directly); the replay's mean price by
  thirds of the price (lowest / middle / highest third).
- **(d) The rates k**: the replay price (gold) on the counters, one rate per kind, k >= 0 (non-negative least squares;
  a debuff enters with a minus), 95 % by bootstrap over the presses; plain least squares beside.
- **(e) A validity coefficient** V = (2 x sign agreement - 1) / 2 + Spearman / 2 and a verdict: **pass** with sign
  agreement >= 0.7 and Spearman >= 0.4; **no signal** when the replay's reliability (the share of its spread that is
  not the copies' noise) is below 0.25 - its ceiling of rank correlation is sqrt(0.25) = 0.5, nothing to compare
  with; else **fail**. Why these thresholds: a random sign gives 0.5, 0.7 is clearly better; Spearman 0.4 is about half
  the ceiling the replay's own noise allows (sqrt of the reliability, 0.7-0.9 for the abilities with a signal).
- **The recharge cost's variants** - a table of their own in the card: the formulas' start, the best press of the
  window (`hindsight`), the first useful press (`next`), by the situation (mean cost, share of price > 0, share of
  replay less the cost > 0, sign agreement).
- **The sign agreement depends on the recharge cost's level**, not only on its quality: the same cost is taken off
  both sides, and the more prices it makes negative, the higher the agreement. A constant cost with the same share of
  prices > 0 agrees as well as any of the dynamic ones (on itS25_12's presses: Deadly Onslaught 0.67 against 0.68,
  Stand Your Ground 0.75 against 0.82; with the best press of the window 0.74 against 0.73 and 0.80 against 0.88).
  So the recharge cost is not chosen by this number; the rank correlation does not depend on it.

## Data

`collect.py` (container, GPU): the network's battles in the simulator (v2 armies, 128 battles, 30 % self-play, 70 %
against `ai_like`), the moments captured by the training's machinery (`Brancher.capture`), each pair of branches open
loop ("not" plays "pressed"'s orders, without the pressed slot), 4 copies, a 120 s horizon with the price also at 30,
60 and 90 s (`--marks`, field `diff_at`), and the presser's side recorded once a second in the pressed branch (the
`each` hook). Two sets:

- **the network's own presses** - every press of itS25_12 (`--share 1`): 2 560, 35 min;
- **moments of a ready ability** (`--hold`): the network does not press in the main battles, the lords' abilities are
  ready at every stage of a battle, in melee too; a share of the moments is replayed (`--missed 0.02`), the slot a
  random ready one (`--random-slot`): 2 003, 62 min (the GPU was shared with another container for the first 30 min).
  Without this set nearly every press is at a battle's start (the network presses whatever is ready at once).

**"Not pressed" drops only the pressed slot** - in training (`ability_branch.play_pair`, open loop) and here. The
training's "not" used to set "no ability" for the unit at every decision, dropping also its other abilities the
network presses later in "pressed"; a lord has 2-3 of them, and in 54-92 % of S25's presses "pressed" orders another
slot of the same lord within 60 s - its effect landed in the pressed one's price. Measured on 747 of itS25_12's own
presses (the battles' first 2 minutes): the old way changed 29 % of the prices, the mean price 4.3 -> 1.9 gold; Rally
2.9 -> 0.0 (the two ways' rank correlation 0.04, the sign flips in 38 %), Foe-Seeker 6.8 -> 0.2 (Stand Your Ground
5.8 -> 5.7). Tests: another slot ordered in "pressed" fires in "not" too, the pressed one does not;
`collect.play_pair` gives `ability_branch.play_pair`'s prices to the bit. `--all-slots` is the old way (for
comparison, `--compare-slots`).

## Result

Data: 4 563 presses (`natural120.npz` + `hold120.npz`), the replay price at 60 s, the counters over 60 s, the recharge
cost by the situation. The rates k (gold per counter unit):

| Kind | k | 95 % |
|---|---|---|
| melee_damage | 0.020 per % x s | 0.014..0.030 |
| melee_defence | 0.017 per point x s | 0.015..0.020 |
| morale | 0.33 per point near the thresholds | 0.12..0.58 |
| vigour | 0.015 per % x s | 0.011..0.020 |
| charge_speed | 0.059 per % x s | 0.024..0.109 |
| charge | 0.056 per % x impact | 0..0.30 |
| speed | 0.003 per % x s | 0..0.007 |

| Ability | presses (own) | sign | Spearman (ceiling) | reliability | V | verdict | recharge cost, gold | replay by price third | game-AI style: n, price > 0, replay - recharge > 0 |
|---|---|---|---|---|---|---|---|---|---|
| Verminous Valour | 908 (580) | 0.55 | 0.00 (0.51) | 0.26 | 0.05 | fail | 0.7 | 1.7 / 2.5 / -0.4 | 440; 0.25; 0.34 |
| Deadly Onslaught | 747 (402) | 0.65 | 0.63 (0.93) | 0.87 | 0.47 | passes by the rank correlation, the sign below the threshold (see below) | 8.2 | 0.0 / 14.4 / 11.2 | - (the AI never presses) |
| Foe-Seeker | 1 013 (547) | 0.71 | 0.44 (0.53) | 0.28 | 0.42 | pass | 7.1 | 1.1 / 8.1 / 14.9 | 458; 0.62; 0.46 |
| Rally | 973 (587) | 0.82 | 0.08 (0.16) | 0.03 | 0.36 | no signal | 1.5 | 0.0 / 0.1 / 2.8 | 251; 0.40; 0.14 |
| Stand Your Ground | 922 (444) | 0.76 | 0.74 (0.73) | 0.53 | 0.63 | pass | 10.0 | 0.0 / 11.4 / 27.0 | 308; 0.54; 0.43 |

The tests (a) pass for all 8 bricks.

**What each change gave** (the same data, the previous code -> the current; sign / Spearman / verdict):

| Ability | previous code | current |
|---|---|---|
| Verminous Valour | 0.55 / 0.00 / fail | 0.63 / 0.00 / fail |
| Deadly Onslaught | 0.92 / 0.63 / pass (price > 0 in 2 %) | 0.72 / 0.63 / pass (price > 0 in 30 %) |
| Foe-Seeker | 0.66 / 0.34 / fail | 0.77 / 0.44 / pass |
| Rally | 0.73 / 0.08 / no signal | 0.90 / 0.08 / no signal |
| Stand Your Ground | 0.77 / 0.72 / pass | 0.80 / 0.74 / pass |

- **Vigour weighted by the tier** lifted Foe-Seeker: Spearman 0.34 -> 0.44.
- **Morale at the thresholds** no longer grows with the battle time (Rally's price against the time +0.59 -> -0.27,
  the replay's +0.07) and tells almost faultlessly the presses that change anything (Rally's counter 0 -> the replay
  price exactly 0 nearly always; it tells the presses with a non-zero price with an area under the curve of 0.94). But
  it cannot rank: where morale changed something, Rally's replay is plus and minus equally often (27 % and 25 % on the
  previous data) - a unit held from routing now saves the trade, now loses more in melee; nothing in the recorded
  trajectory predicts the sign. The counter catches the mean: the top 10 % by the counter - the replay 35 gold on
  average, the bottom half - 0.
- **The recharge cost "the first useful press later"** instead of "the best press of the window": the price > 0 in
  45 / 45 / 31 % of the presses (Deadly Onslaught / Foe-Seeker / Stand Your Ground; before 24 / 16 / 9 %), in the
  game-AI-style presses (in melee, friends near) 62 % and 54 % for Foe-Seeker and Stand Your Ground (before 22 % and
  12 %). The sign agreement fell (0.74 -> 0.65, 0.74 -> 0.71, 0.78 -> 0.76): Deadly Onslaught's below the 0.7
  threshold, though its rank correlation is the same (0.63) - the sign agreement measures the recharge cost's level
  (section "Layer 3"). Deadly Onslaught's sign gap comes from the replay's skew: 12 % of the presses with >= 20 s of the
  lord's melee under the bonus replay to exactly 0, the median a third of the mean (5.9 against 15.5 gold), while the
  counter's price gives the mean; the replay less the recharge cost is > 0 in 18 % of the presses against 45 % for the
  price. In training the ability stays priced by its bonuses (the valid list is `train_price.VALID`).
- **The horizon.** The replay's mean price at 30 / 60 / 90 / 120 s: Foe-Seeker 2.8 / 8.0 / 14.4 / 22.4, Verminous
  Valour 0.9 / 1.3 / 4.5 / 5.7, Stand Your Ground 12.2 / 12.8 / 14.0 / 21.6 - vigour's and speed's worth shows after
  60 s; at 90 s the speed rate is already significant (0.006, 95 % 0.002..0.010). Rally is most reliable at 30 s
  (reliability 0.60, Spearman 0.19), at 60 s it is noise (0.03).

**What fails and why.** Verminous Valour: at 30 s its replay has a signal (reliability 0.48), yet Spearman 0.00 - its
worth is not what the counters count (+25 % speed and +8 morale of the lord): arriving sooner leads now into a good
fight, now into a bad one. Rally - see above; it needs the sign of the held unit's outcome (whether it wins its fight),
and the trajectory has no units' health.

### Tried and rejected

- **Morale as the shortfall to the normal top** (min(bonus, the largest shortfall) over the affected): the counter
  grew with the battle time (+0.53..0.59), the replay did not; Spearman Rally 0.08, Verminous Valour 0.04 (5 091
  presses, 60 s).
- **Vigour as seconds a tier fresher** (no tier weight): Foe-Seeker Spearman 0.26-0.34; the lord's melee seconds alone
  relate to the price more (0.37).
- **The recharge cost as the best press of the window** (`best_later`, the maximum over the seconds 5 ... recharge_s):
  biased up - the maximum of ~18 noisy values, only someone who knows the future picks so. The price > 0 in 7 / 30 /
  26 % of the presses of Stand Your Ground / Deadly Onslaught / Foe-Seeker, in Stand Your Ground's presses in melee
  11 %, the mean price there -5.7 gold (the head learns not to press even in melee). Tried to soften it on the same
  4 563 presses (the recharge cost by the situation, the lab's rates; the share of price > 0 in Stand Your Ground's
  presses in melee / out of it): the maximum of the mean over 4 copies - 18 % (the copies' noise is not the cause);
  the maximum of the values predicted from the situation at each second - 18 %; the best given second predicted at
  the press (the lower bound of what waiting without knowing the future gives) - 50 % / 6 %, the mean price in melee
  -0.2 (pressing in melee no matter); the mean useful press benefit x the chance of work in the window - 51 % / 5 % (in
  46 % of the presses where the lord fought >= 15 s under the bonus the price < 0); the first second where the press
  would give >= half the mean useful benefit - 77 % / 11 %, but unstable for Foe-Seeker (at a 0.75 threshold the price
  > 0 in 74 % of the presses, out of melee 88 %). Chosen: the first useful press (no threshold to tune): Stand Your
  Ground in melee 71 %, out of it 18 %; where the lord fought >= 15 s under the bonus 79 % (Deadly Onslaught, >= 20 s,
  83 %).
- **The recharge cost as the mean benefit x recharge_s / 60**: for a 90 s recharge 1.5 means, the price > 0 in 1-2 %
  of Deadly Onslaught's presses and in 14-20 % of Stand Your Ground's (the replay > 0 in 54-55 % and 49-50 %); the
  sign agreement was inflated by it (0.92).
- **Another morale threshold** (the previous data, 60 s): weight 1 below 0 and 0 above (the rout only) - Stand Your
  Ground Spearman 0.23 against 0.41, Rally tells the non-zero presses with an area of 0.64 against 0.94; the band
  0..16 - 0.30 and 0.90. The band 16..30 is kept (the wavering threshold -> steady).

## The bonus lab

The rate k of each bonus kind measured in the simulator **without the network's presses**: an experiment where the
lab, not the network, switches the bonus on. Why: the regression over the network's presses sees only how the network
presses (it hardly ever presses the charge and speed bonuses at the right moment - their rates there are ~0 with a huge
spread), and several kinds in one press get mixed up. Code: `tools/nn/abilities_price/lab.py` (numpy: the
experiments, counters, rates, curves, the layer-3 check) and `lab_sim.py` (torch: the simulator runs); layers 1-3 do
not change.

**An experiment is a pair of battles.** Two copies of one battle with one battle id (the simulator's randomness is a
hash of it, so the copies are the same to the bit) and the same orders: the orders are made on the copy "with the
bonus" and replayed in the copy "without" (open loop, as the replay prices). In the copy "with the bonus" the lord
presses a **lab ability** at the chosen second: a real ability's passport with only one bonus kind left (the same
duration, radius, targets), its number x a dose 0.5 / 1 / 2. The lord's own real abilities are taken out, the game's
AI fires nobody's. The measure: the gold over the 60 s after the press - (enemy destroyed - own lost) "with" minus the
same "without" (the trade of the reward and of the replay prices, in gold). The counter is **the same** as for the
network's presses: the layer-1 brick through the layer-2 ability (the lab passport) on the "with" copy's trajectory.
The rate k = sum of gold / sum of counter over the pairs at the real number (dose 1), 95 % by bootstrap over pairs.

**Three families of experiments** (the conditions from the real ones: the lords that carry the bonus - the Empire's
General and the Skaven Warlord; units from our armies' pools `config/nn/pools.json`, both factions):

- **small fights** (8 192 pairs a kind, the lab's script): "line" - the lord and 0-4 friends against 1-4 enemies,
  lines 10-80 m apart, some enemies on a flank, the press at second 0-45 (before, at or deep in the fight); "run-up" -
  the lord 20-60 m from an enemy unit's edge (speed and vigour 20-150 m), at any angle from its front to its rear, the
  enemy often already fighting a unit of ours, the press as the run starts. The lord's fatigue from the network's
  presses' distribution (vigour: every tier). Each side keeps a far reserve of 4-10 units (700 m, holding), so a local
  fight does not end the battle and army destruction counts as in a real army;
- **whole battles** (3 072 pairs a kind): v2 armies as in training, both sides played by the `ai_like` script; the lord
  of a side that carries the bonus presses it at a battle time drawn from the network's presses' times (up to 600 s);
  the battle is played once to the press, then doubled;
- **whole battles on the condition** (8 192 pairs a kind): the same, but the press at the first second the bonus has
  work (damage - the lord in melee; defence - a unit of his within 35 m in melee; charge and charge speed - the lord
  runs at his attack target closer than 70 m; speed - the lord moves out of melee; morale - a unit within 35 m 8 points
  or more below its normal top; vigour - the lord fights or moves), within 120 s.

The rate in the table is the whole battles' (both families together): they are closer to the network's battles. Run
time on the GPU: small fights ~4.7 min a kind, whole battles ~1.5 min, on the condition ~5.5 min (~1.5 h for 7 kinds).

| Kind | Rate k, whole battles | 95 % | small fights | regression (layer 3) |
|---|---|---|---|---|
| melee_damage (per % x s) | 0.016 | 0.014...0.018 | 0.0098 (0.0087...0.011) | 0.021 (0.014...0.028) |
| charge (per % x impact) | 0.056 | -0.02...0.13 | 0.029 (0.019...0.043) | 0.003 (0...0.11) |
| melee_defence (per point x s) | 0.0225 | 0.021...0.024 | 0.020 (0.019...0.021) | 0.014 (0.010...0.017) |
| morale (per point) | 0.008 | -0.06...0.07 | 0.081 (0.037...0.13) | 0.15 (0.09...0.24) |
| speed (per % x s) | 0.009 | 0.002...0.017 | 0.041 (0.036...0.046) | 0 (0...0.002) |
| charge_speed (per % x s) | 0.016 | 0.005...0.027 | 0.014 (0.010...0.019) | 0.004 (0...0.033) |
| vigour (per s) | 0.37 | 0.33...0.41 | 0.20 (0.18...0.21) | 0.24 (0.14...0.31) |

**Note: this table's vigour and morale rates were counted with the earlier bricks** (vigour - seconds a tier fresher,
morale - the shortfall to the norm). With today's bricks the same pairs give vigour 0.022 a % x s (0.020...0.025; by
the lord's fatigue tier 0.008 fresh ... 0.024 exhausted) and morale 0.037 (-0.24...0.34); the other rates are the
same, and with them all three abilities pass: Stand Your Ground 0.79 / 0.73, Deadly Onslaught 0.89 / 0.69, Foe-Seeker
0.73 / 0.41 (`python -m tools.nn.abilities_price.train_price --relab build/abilities_price/lab`; the training takes
them so - [the ability head's own signal](training.md#the-ability-heads-own-signal)).

**Curves** (the rate by conditions; the full table is `build/abilities_price/lab/lab.md`):

- **vigour** depends strongly on the lord's fatigue tier at the press: fresh 0.05, active 0.11, winded 0.32, tired
  0.41, very tired 0.54, exhausted 0.29 (whole battles; small fights the same shape);
- **defence** in small fights grows with the enemy's attack: attack < 25 - 0.012, 25-32 - 0.024, 32-40 - 0.028 (the
  hit formula 35 + attack - defence within 8...90: against a weak attack the defence runs into the lower bound); in
  whole battles flatter (0.020-0.024);
- **morale**: only units whose morale at the press is 32-48 points gain (small fights 0.25 a point; above 48 ~0.01);
  in whole battles the rate is ~0 at any morale (16-32: 0.08, a wide interval);
- **charge** (small fights): from the rear 0.012 against 0.026-0.034 from the front and flank;
- **dose** (proportionality): damage and defence at x2 give less a unit (damage 0.015 / 0.010 / 0.0067 at
  x0.5 / x1 / x2 in small fights; defence 0.021 / 0.020 / 0.013), vigour at x2 more (0.37 -> 0.64). The damage and
  defence bricks declare themselves proportional; in the simulator the return diminishes.

**The layer-3 check** (the prices of itS25_12's 5 090 presses by the lab's rates against the replay prices; the
regression is fitted on these very presses, the lab never saw them; "with curves" - morale by the unit's morale at the
press, vigour by the lord's fatigue tier):

| Ability | regression: sign / Spearman | lab | lab with curves | mean price / replay, gold (lab with curves) |
|---|---|---|---|---|
| Deadly Onslaught | 0.92 / 0.66 | 0.89 / 0.69 | 0.89 / 0.69 | 6.6 / 8.5 |
| Stand Your Ground | 0.75 / 0.71 | 0.76 / 0.72 | 0.80 / 0.73 | 13.5 / 13.8 |
| Foe-Seeker | 0.64 / 0.26 | 0.58 / 0.18 | 0.69 / 0.34 | 10.9 / 7.8 |
| Rally | 0.73 / 0.08 | 0.75 / 0.08 | 0.87 / 0.03 | 0.6 / 1.4 |
| Verminous Valour | 0.52 / 0.04 | 0.52 / 0.00 | 0.57 / 0.02 | no signal |

The lab's rates with curves reproduce the replay at least as well as the regression for every ability with a signal,
except Deadly Onslaught's sign (0.89 against 0.92: the lab's damage rate is lower, mean price 6.6 against 8.5). The
small fights' rates alone do worse (Deadly Onslaught's sign 0.79, Foe-Seeker's Spearman -0.21): a small fight has few
targets, damage is worth less there and morale and speed more than in a whole battle.

**Found on the way** (the decisions are not here):

- **A run-up into a busy target.** 61-66 % of the `sprint` seconds in the recorded trajectory (`collect.fields`) are
  the lord's run at an enemy already fighting a unit of ours. The simulator gives neither the sprint nor the charge
  then (`charge.free_target_only`) - from any side, the rear too. So the charge-speed brick counts seconds where the
  bonus does nothing (the record needs a "target free" flag), and a hammer-and-anvil charge into the rear of a busy
  enemy gets no charge in the simulator - worth checking against the game (rule 17).
- **Speed counts idle seconds.** The condition "moves out of melee" catches much useless running in the network's
  presses: with the small fights' rate (0.041) Foe-Seeker's price goes against the replay (Spearman -0.21). The
  condition "moves out of melee and reaches melee later in the window" (a probe outside the code) keeps the same rate in
  small fights (0.042) and lifts Foe-Seeker's Spearman -0.21 -> +0.22 (with the whole battles' rates 0.18 -> 0.25).
- **Morale at the line.** The counter "shortfall to the normal top" hardly relates to the gold (Spearman with a pair's
  gold 0.14 in small fights, 0.09 in whole battles); the counter "seconds the bonus keeps the unit above the waver (16)
  or rout (0) line it would be below without it" relates better in small fights (0.31), not in whole battles (0.10),
  and does not fix Rally (0.06).

**What stays uncertain:** the charge in whole battles (an impact comes in only 17 % of the condition presses - interval
-0.02...0.13; the small fights' 0.029 is more reliable), morale (~0 +- 0.07 in whole battles, 0.08 in small fights -
the rate depends on the situation, the counter is not the right one), speed (0.009 in whole battles against 0.041 in
small fights). The `ai_like` script plays both sides: the AI's lord keeps back and rarely runs in to attack, the
network's lord plays otherwise.

    DOCK_NAME=agent-bonuslab DOCK_BUILD=<main>/build bash tools/nn/dock.sh tools.nn.abilities_price.lab_sim \
        --pairs 8192 --battle-pairs 3072 --out build/abilities_price/lab          # small fights and whole battles
    ... --pairs 0 --battle-pairs 8192 --battle-trigger 120 ...                     # whole battles on the condition
    python -m tools.nn.abilities_price.lab --dir build/abilities_price/lab       # rates, curves, layer 3 -> lab.md

Tests: `tests/tools/test_nn_abilities_price_lab.py` (host: a lab ability is the same layer-2 part of the real ability
with the same counter, a x2 dose doubles a proportional counter, the experiments' layout, the rate and its interval,
the curves), `tests/tools/test_nn_abilities_price_lab_torch.py` (container: with a bonus of size 0 the copies are the
same to the bit - in small fights and whole battles; with the bonus they part only after the press; layer 2's masks
match the simulator's record >= 97 %).

## How to run

    DOCK_NAME=agent-abprice DOCK_BUILD=<main>/build bash tools/nn/dock.sh tools.nn.abilities_price.collect \
        --ckpt build/v2/itS25_12.pt --share 1 --missed 0 --no-compile --horizon 120 --marks 30,60,90 \
        --presses 2500 --out build/abilities_price/natural120.npz
    ... --hold --share 0 --missed 0.02 --random-slot ... --out build/abilities_price/hold120.npz
    python -m tools.nn.abilities_price.validate --data build/abilities_price/natural120.npz \
        build/abilities_price/hold120.npz --horizon 60 [--recharge situation|next|hindsight|static] [--target next|best]

The card is `build/abilities_price/card.md` (+ `.json`). Tests: `tests/tools/test_nn_abilities_price.py` (host),
`tests/tools/test_nn_abilities_price_torch.py` and `tests/tools/test_nn_ability_branch.py` (container).
