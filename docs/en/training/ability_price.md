# The price of an ability press

[← Back](README.md) · [Documentation](../README.md) › [Training data](README.md) › The price of an ability press · [Русский](../../ru/training/ability_price.md)

An ability press priced by what its bonuses actually did in the battle (seconds of melee under a damage bonus, the
morale shortfall a bonus closed, and so on), and one generic check of that price against the simulator's "pressed /
not pressed" replay. Why: the replay is exact but expensive and noisy; the price by the bonuses is cheap and every
press has one. Code: `tools/nn/abilities_price/`; training, the network and the simulator do not change.

## In short

- **Three layers.** (1) A brick per bonus kind: a counter of the bonus's useful work (>= 0) and its description.
  (2) An ability is data: a set of bricks with the passport's numbers (`config/nn/abilities.json`), duration,
  recharge, radius, targets; no code per ability. (3) One generic validator that knows nothing about any bonus.
- **The rate k** (gold per counter unit) is one per bonus kind, not per ability; it is measured by regressing the
  replay prices on the counters.
- **The result on 5 091 presses of the network itS25_12** (2 560 of its own presses + 2 531 moments of a ready
  ability): Stand Your Ground and Deadly Onslaught pass (rank correlation with the replay 0.71 and 0.66), Foe-Seeker
  and Rally fail, Verminous Valour's replay is noise (no signal).
- Found on the way: a bug in the training's branches - "not pressed" drops **all** of the lord's abilities for 60 s,
  not only the pressed one (section [Data](#data)).

## Layer 1: the bonus bricks

File `bonuses.py`. A press's trajectory is sampled once a second: sample 0 is the moment of the press (before it acts),
sample s the state after second s; the units are the presser's side. The mask "the bonus is on that unit in second s"
comes from layer 2. Each brick: `count(trajectory, mask, value) -> number >= 0` and `switch_off(...)` - the same
trajectory with its condition false (for the generated tests). The description (`Description`): kind, condition in
words, whether the counter is proportional to the bonus's size, buff or debuff, how the size scales (`mult`:
(x - 1) x 100 %, `add`: the number), a typical value for the tests.

| Kind | Counter (unit) | Useful in second s when | Prop. |
|---|---|---|---|
| `melee_damage` (base + armour-piercing) | sum (x - 1) x 100 x [in melee] (% x s) | the unit is in melee | yes |
| `charge` | sum (x - 1) x 100 over charge impacts (% x impact) | the charge left rose: the first blow of a new charge | yes |
| `melee_defence` | sum over the affected bonus x [in melee] (points x s) | an affected unit is in melee | yes |
| `morale` (leadership) | sum over the affected min(bonus, the largest shortfall) (points) | morale below its normal top (leadership + start bonus) | no |
| `speed` | sum (x - 1) x 100 x [moving, not in melee] (% x s) | the unit walks or runs out of melee | yes |
| `charge_speed` | sum (x - 1) x 100 x [charge sprint] (% x s) | an attack order, not in melee, moving, the target's edge within charge_dist | yes |
| `vigour` (buff) | seconds a fatigue tier fresher while fighting or moving (s) | the ability removed fatigue | no |
| `fatigue_cost` (debuff) | seconds a fatigue tier more tired while fighting or moving (s) | the ability added fatigue | no |

- **Morale.** Shortfall = top - (morale - the bonus's own lift). The lift follows the game's morale pace (each 0.5 s
  tick the points move max(1, 15 % of the gap) towards the target, `game_rules.json`): a unit at its top has the whole
  bonus above the top as the lift, shortfall 0. The shortfall at the press (for a unit an aura reaches later: just
  before) and its growth while the ability acts; per unit at most the bonus.
- **Fatigue.** The fatigue "without the ability" is rebuilt from the recorded one: each second's change less the
  ability's (value x 30 000 a second while it is on), within 0..30 000. Where the recorded fatigue sat at 0 or the
  maximum (the change cut off) the activity's own rate from the game's database is taken (`_kv_fatigue_tables`: melee
  +19, the sprint +34, moving +4, ready -7 a tick, 10 ticks a second). Counted from the press to the end of the window
  (the gap lasts after the ability ends).
- **Foe-Seeker's sign.** The database's `fatigue_change_ratio` is -0.01, in the simulator -300 fatigue points a
  second, and the in-game probe (`build/movelords`) agrees: the General with it tires later (91.5 s against ~51). It is
  a **buff** (vigour), not "extra fatigue" as the agreed formulas say (`build/abilities_bonus/formulas.md`). Layer 2
  picks the brick by the number's sign: -0.01 -> `vigour`, a positive number -> `fatigue_cost`.

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

The recharge cost (`Recharge`), the formulas' start: the mean benefit of this ability's useful presses x recharge_s /
60. The dynamic version (the critic's value of having the ability ready over the next recharge_s seconds in this
situation) plugs in as a subclass with a cost per press.

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
  correlation (the replay holds no recharge, so the benefit is compared with it directly).
- **(d) The rates k**: the replay price (gold) on the counters, one rate per kind, k >= 0 (non-negative least squares;
  a debuff enters with a minus), 95 % by bootstrap over the presses; plain least squares beside.
- **(e) A validity coefficient** V = (2 x sign agreement - 1) / 2 + Spearman / 2 and a verdict: **pass** with sign
  agreement >= 0.7 and Spearman >= 0.4; **no signal** when the replay's reliability (the share of its spread that is
  not the copies' noise) is below 0.25 - its ceiling of rank correlation is sqrt(0.25) = 0.5, nothing to compare
  with; else **fail**. Why these thresholds: a random sign gives 0.5, 0.7 is clearly better; Spearman 0.4 is about half
  the ceiling the replay's own noise allows (sqrt of the reliability, 0.7-0.9 for the abilities with a signal).

## Data

`collect.py` (container, GPU): the network's battles in the simulator (v2 armies, 128 battles, 30 % self-play, 70 %
against `ai_like`), the moments captured by the training's machinery (`Brancher.capture`), each pair of branches open
loop (`ability_branch`: "not" plays "pressed"'s orders), 4 copies, 60 s, and the presser's side recorded once a second
in the pressed branch (the `each` hook). Two sets:

- **the network's own presses** - every press of itS25_12 (`--share 1`): 2 560, 18 min;
- **moments of a ready ability** (`--hold`): the network does not press in the main battles, the lords' abilities are
  ready at every stage of a battle, in melee too; a share of the moments is replayed (`--missed 0.02`), the slot a
  random ready one (`--random-slot`): 2 531, 24 min. Without this set nearly every press is at a battle's start (the
  network presses whatever is ready at once).

**A bug in the training's branches.** `ability_branch.play_pair` sets "no ability" for the unit in "not pressed" at
**every** decision, so it also drops the unit's other abilities that the network presses later in "pressed". A lord
has 2-3 of them and the network presses what is ready at once: in 54-92 % of S25's presses "pressed" orders another
slot of the same lord within 60 s, and its effect lands in the pressed one's price. `collect.play_pair` drops only the
pressed slot by default (`--all-slots` - as in training; a test: then the prices equal `ability_branch.play_pair`'s to
the bit). `--compare-slots` plays each batch both ways. Measured on 747 of itS25_12's own presses (the battles' first 2
minutes): the training's way changes 29 % of the prices, the mean price 1.9 -> 4.3 gold; Rally 0.0 -> 2.9 (the two
ways' rank correlation 0.04, the sign flips in 38 %), Foe-Seeker 0.2 -> 6.8 - in training they are credited with the
same lord's other abilities (Stand Your Ground 5.7 -> 5.8, unchanged).

## Result (itS25_12, 5 091 presses)

The rates k (gold per counter unit):

| Kind | k | 95 % |
|---|---|---|
| melee_damage | 0.021 per % x s | 0.014..0.028 |
| melee_defence | 0.014 per point x s | 0.010..0.017 |
| morale | 0.15 per point | 0.09..0.23 |
| vigour | 0.24 per s | 0.15..0.31 |
| charge | 0.003 per % x impact | 0..0.11 |
| charge_speed | 0.004 per % x s | 0..0.033 |
| speed | 0 | 0..0.002 |

| Ability | presses (own) | sign | Spearman (ceiling) | reliability | V | verdict | game-AI style: n, price > 0, replay > 0 | alarms |
|---|---|---|---|---|---|---|---|---|
| Verminous Valour | 1 042 (580) | 0.52 | 0.04 (0.35) | 0.12 | 0.04 | no signal | 511; 0.81; 0.40 | the price grows with the battle time (+0.58), the replay does not |
| Deadly Onslaught | 907 (402) | 0.92 | 0.66 (0.87) | 0.76 | 0.75 | pass | - (the AI never presses) | price > 0 in only 1 % |
| Foe-Seeker | 1 092 (547) | 0.64 | 0.26 (0.74) | 0.55 | 0.27 | fail | 485; 0.70; 0.74 | - |
| Rally | 1 024 (587) | 0.73 | 0.08 (0.66) | 0.43 | 0.27 | fail | 292; 0.76; 0.27 | tails 1.5 %; the price grows with the time (+0.53) |
| Stand Your Ground | 1 025 (444) | 0.75 | 0.71 (0.84) | 0.71 | 0.61 | pass | 379; 0.52; 0.92 | - |

The tests (a) pass for all 8 bricks.

**Why fail, and what to propose** (the formulas are unchanged - they are the user's decisions):

- **Morale counts the shortfall, not the outcome.** The morale counter grows with the battle time (the shortfall
  piles up), the replay does not; for Rally and Verminous Valour it does not predict the price (Spearman 0.08 and
  0.04). A morale bonus changes a battle only near the wavering and rout thresholds - the non-linearity the formulas
  postponed ("morale near the rout threshold").
- **Foe-Seeker: the tier, not its weight.** Vigour carries the price (k 0.24 a second, significant), but "a tier
  fresher" counts the same for fresh/active (attack x0.95) and tired/very tired; the lord's melee seconds alone relate
  to the price more (0.37) than the counter (0.26). Proposal: weigh a tier by its penalties (`fatigue.effects`). Speed
  is worth 0 (k = 0; the speed counter relates to the price negatively: while the lord runs he does not fight).
- **The recharge cost's start is too dear for a 90 s recharge.** The mean benefit x 90/60 = 1.5 means: Deadly
  Onslaught's price > 0 in only 1 % of presses, Stand Your Ground's in 20 % (the replay > 0 in 54 % and 50 %); in the
  game AI's style Stand Your Ground 0.52 against 0.92. The benefit without the recharge in those moments is > 0 in
  92-100 %. An argument for the dynamic recharge cost (the critic).

## How to run

    DOCK_NAME=agent-abprice DOCK_BUILD=<main>/build bash tools/nn/dock.sh tools.nn.abilities_price.collect \
        --ckpt build/v2/itS25_12.pt --share 1 --missed 0 --no-compile --out build/abilities_price/natural.npz
    ... --hold --share 0 --missed 0.02 --random-slot --out build/abilities_price/hold.npz
    python -m tools.nn.abilities_price.validate --data build/abilities_price/natural.npz build/abilities_price/hold.npz

The card is `build/abilities_price/card.md` (+ `.json`). Tests: `tests/tools/test_nn_abilities_price.py` (host),
`tests/tools/test_nn_abilities_price_torch.py` (container).
