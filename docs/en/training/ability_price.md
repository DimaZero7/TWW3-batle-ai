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
- **The recharge cost depends on the situation**: what the same press would have been worth later in the recharge
  window (the best of those), predicted from what is seen at the press.
- **The result on 4 563 presses of the network itS25_12** (2 560 of its own presses + 2 003 moments of a ready
  ability, the price at 60 s): Stand Your Ground, Deadly Onslaught and Foe-Seeker pass (rank correlation with the
  replay 0.74, 0.63, 0.44); Rally has no signal, Verminous Valour fails (section [Result](#result)).

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
seconds), the new press's mark put back (`apply`), the counters over the horizon from s. `best_later` - the best
benefit of such presses 5, 10, ... recharge_s seconds later (>= 0: never pressing is always there) - is the cost in
hindsight (`hindsight`): it sees the future. The price takes its **prediction from the situation at the press**
(`situation`, `validate.situation_recharge`): a ridge regression on what is seen then (battle time, the lord in melee
/ moving, friends within 35 m and those of them in melee, own units standing, own units below 30 morale points,
wavering and routing, the lord's fatigue tier), one per ability, cross-fitted (each press by the other folds), >= 0.
No hand numbers per ability. The trajectory must cover the window: collected with a 120 s horizon (`--horizon 120`),
the price taken at the 60 s mark. The formulas' start (the mean benefit of the useful presses x recharge_s / 60,
`Recharge`) is kept for comparison - beside it in the card.

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
- **The recharge cost's variants** - a table of their own in the card: the formulas' start, hindsight, by the
  situation (mean cost, share of price > 0, share of replay less the cost > 0, sign agreement).

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
| Verminous Valour | 908 (580) | 0.63 | 0.00 (0.51) | 0.26 | 0.13 | fail | 1.1 | 1.7 / 2.5 / -0.4 | 440; 0.12; 0.31 |
| Deadly Onslaught | 747 (402) | 0.72 | 0.63 (0.93) | 0.87 | 0.54 | pass | 12.5 | 0.0 / 14.4 / 11.2 | - (the AI never presses) |
| Foe-Seeker | 1 013 (547) | 0.77 | 0.44 (0.53) | 0.28 | 0.49 | pass | 12.8 | 1.1 / 8.1 / 14.9 | 458; 0.22; 0.32 |
| Rally | 973 (587) | 0.90 | 0.08 (0.16) | 0.03 | 0.43 | no signal | 3.3 | 0.0 / 0.1 / 2.8 | 251; 0.16; 0.10 |
| Stand Your Ground | 922 (444) | 0.80 | 0.74 (0.73) | 0.53 | 0.67 | pass | 17.9 | 0.0 / 11.4 / 27.0 | 308; 0.12; 0.30 |

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
- **The dynamic recharge cost** removed Deadly Onslaught's inflated sign agreement (0.92 came from a price almost always
  < 0): now the price is > 0 in 30 % of the presses, the replay less the recharge > 0 in 12 %, agreement 0.72. The gap
  comes from the replay's skew: 12 % of the presses with >= 20 s of the lord's melee under the bonus replay to exactly
  0, the median a third of the mean (5.9 against 15.5 gold), while the counter's price gives the mean.
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
- **The recharge cost as the mean benefit x recharge_s / 60**: for a 90 s recharge 1.5 means, the price > 0 in 1-2 %
  of Deadly Onslaught's presses and in 14-20 % of Stand Your Ground's (the replay > 0 in 54-55 % and 49-50 %); the
  sign agreement was inflated by it (0.92).
- **Another morale threshold** (the previous data, 60 s): weight 1 below 0 and 0 above (the rout only) - Stand Your
  Ground Spearman 0.23 against 0.41, Rally tells the non-zero presses with an area of 0.64 against 0.94; the band
  0..16 - 0.30 and 0.90. The band 16..30 is kept (the wavering threshold -> steady).

## How to run

    DOCK_NAME=agent-abprice DOCK_BUILD=<main>/build bash tools/nn/dock.sh tools.nn.abilities_price.collect \
        --ckpt build/v2/itS25_12.pt --share 1 --missed 0 --no-compile --horizon 120 --marks 30,60,90 \
        --presses 2500 --out build/abilities_price/natural120.npz
    ... --hold --share 0 --missed 0.02 --random-slot ... --out build/abilities_price/hold120.npz
    python -m tools.nn.abilities_price.validate --data build/abilities_price/natural120.npz \
        build/abilities_price/hold120.npz --horizon 60 [--recharge situation|hindsight|static]

The card is `build/abilities_price/card.md` (+ `.json`). Tests: `tests/tools/test_nn_abilities_price.py` (host),
`tests/tools/test_nn_abilities_price_torch.py` and `tests/tools/test_nn_ability_branch.py` (container).
