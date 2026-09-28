# States and passive effects

[← Back](README.md) · [Units](README.md) · [Русский](../../../ru/game/units/states.md) · [Evidence](evidence.md)

## What each unit actually exposed

| Runtime property | Shieldless Spearmen | Archers | Foot General |
|---|---|---|---|
| Unit key | `wh_main_emp_inf_spearmen_0` | `wh2_dlc13_emp_inf_archers_0` | `wh_main_emp_cha_general_0` |
| Initial entities | 120 | 90 | 1 |
| Guard supported | yes | yes | yes |
| Free fire / skirmish supported | no / no | yes / yes | no / no |
| Formation-spacing behavior supported | no | no | no |
| `hide_forest` | true | true | true |
| `charge_defense_vs_large` | true | false | false |
| `charge_reflection` | true | false | false |
| `encourages` | false | false | true |
| Active abilities in these XML fixtures | none | none | Stand Your Ground, Foe Seeker |
| Passive abilities returned | none | none | Single Entity, Hold the Line |

Attributes above were read with `unit:has_attribute(key)`, not inferred from appearance. Behaviors were queried with `can_use_behaviour` and `is_behaviour_active`; abilities with `owned_non_passive_special_abilities()` and `owned_passive_special_abilities()`.

The General's active abilities are a property of these XML fixtures, including the explicitly rank-1 fixture. They must not be copied into a universal campaign rank-1 loadout. No ability was unlocked with a Lua cheat. The XML default assignment itself is the important limitation.

## Read the actual state

Native reads verified during the runs include position, bearing, ordered position/bearing/width, movement/fast movement, melee, ammunition, alive count, unary HP, fatigue, routing, leaving, hidden and script-controlled flags. `current_target()` may be nil: guard clearing its target is an observed example.

CCO access also worked:

```lua
local id = tostring(unit:unique_ui_id())
local count = common.get_context_value('CcoBattleUnit', id, 'StatusList.Size')
local braced = nil -- unknown if the list is unavailable
if type(count) == 'number' then
    braced = false
    for i = 0, count - 1 do
        local key = common.get_context_value(
            'CcoBattleUnit', id, 'StatusList.At(' .. i .. ').Key')
        if key == 'braced' then braced = true end
    end
end
```

Observed status keys include `braced`, `moving`, `moving_fast`, `firing`, `melee`, `hidden`, `withdraw`, `routing`, `shaken` and `wavering`. They can coexist; the initial list can temporarily be empty. Do not treat them as mutually exclusive states or complete descriptions of every soldier.

Also verified: CCO `IsFiringMissiles`, `IsWithdrawing`, `MoralePercent`, `MoraleState`, `CharacterRank`, `HasCharacterRank`, `ExperienceLevel` and `ActiveEffectList.At(i).PhaseRecordContext.Key`. `MoralePercent` can exceed 1; it is not a probability and is not a raw leadership score. The attempted field `Morale` returned nil and is excluded from the interface.

## Bracing and charge reception

All three types acquired `braced` when settled, even with guard off. After halting a moving formation, this appeared after a settling interval; no universal preparation timer was established. Some samples contained both `braced` and movement/melee. An aggregate unit flag is not proof that every model is perfectly facing an attacker.

Two additional battles put Empire Knights against each type, once stationary and once walking forward. The last sample before initial damage reported `braced` in the stationary trial and `moving` in the walking trial. The stationary Spearmen lost **15.91%** of initial HP during the 45-second stage; walking Spearmen lost **34.41%**. Archers: **51.93% / 89.16%**; General: **14.90% / 15.73%**. Walking Archers routed.

These are observations from **one trial per condition**, not measured universal protection coefficients. Contact occurred at different times; movement changed collision and melee geometry. This does not isolate charge-bonus negation, charge reflection, ordinary impact damage or shield blocking. Exact angles, enemy exceptions, timers and separate numerical effects remain unverified. In particular, generic `braced` is not the same thing as possessing Spearmen's two charge attributes.

## Fatigue and hiding

After repeated running, Spearmen and General reached tired states and Archers reached very tired in the baseline trials. During **240 simulated seconds** of rest all returned to `threshold_fresh`. Damage did not recover with fatigue. This is a measured case, not a guaranteed recovery time in every battle.

On the exact Kislev `catchment_04` variant, forest fixtures used Spearmen at (−45.5, −165.5), Archers at (−45.5, −345.5), General at (134.5, 389.5), with 15/15/5 m ordered widths. Ground readings confirmed forest and all three became hidden. Moving enemy formations to points 30 m east made all three visible. This is a reveal observation, not a measured universal 30 m detection radius: formations have area and multiple entities.

Ground type at the unit reference position alone does not establish concealment for the whole formation. Read `unit:is_hidden()` as well. No invisibility override was used.

## General support and abilities

In the support fixture, Spearmen remained near (180,0), Archers near (180,−80). Moving the General from (350,0) to (195,0), then back, increased and then restored their reported morale fractions. Hold the Line appeared on the nearby Spearmen and disappeared after removal; the more distant Archers did not receive that effect in the baseline trial. This establishes position-dependent support, not separate radii or additive coefficients for encouragement, the general aura and Hold the Line.

Self-targeted Stand Your Ground created its active phase on the General and nearby Spearmen. Foe Seeker created its own phase on the General. Check actual ownership and `can_perform_special_ability(key)` before using either. Exact buff strength and their availability to a fresh campaign lord were not established by this experiment.

The installed General record has a missile-blocking shield. Turning is tested; the shield's angular blocking chance was not separately measured. There is no verified “raise shield” command in this guide.

General readout contracts and new numeric HP/morale evidence: [common state sensors](state-sensors.md). This page retains earlier passive-effect experiments.
