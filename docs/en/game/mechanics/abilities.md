# Abilities, attributes, army mechanics

[← Back](README.md) · [Documentation](../../README.md) › [Game knowledge](../README.md) › [Battle mechanics](README.md) › Abilities · [Русский](../../../ru/game/mechanics/abilities.md)

Common unit attributes and passives with their numbers, contact effects, barriers, army-wide
mechanics (Waaagh!, battle currency) and reinforcements. Conventions: [index](README.md). Ours:
abilities of our lords are read from the database (`config/nn/abilities.json`, `tools/nn/sim/abilities.py`).

## General rules

- Abilities from several sources don't stack; passives show their recharge and condition in the
  UI; some start on cooldown (Gate of Khorne 60 s, counting only in melee). Whether a recharge
  starts at the cast or at the end of the active time: not found. · [fandom Abilities][fw-ab] · medium.
- **The recharge context** (`special_ability_to_recharge_contexts`): the recharge runs only in this context, the
  initial one always. Strength of the Penitent fires by itself as soon as it is ready, in melee; its 3 s ("while
  losing", CA hotfix 6.2.2) stand only while the unit wins its melee and run while it loses, is even, and out of
  melee (probe T-E1: winning flagellants - one fire at the contact and none for 70 s more, losing and even ones
  exactly 3 s; recordings of 54 units - gaps of exactly 3 s in 92 %, after leaving melee the next fire 3 s later,
  out of melee too). Wounds - 5 s below 25 % health. · database,
  CA, recordings, probe · high.
- **How an ability picks its targets** (`unit_special_abilities.update_targets`, the community's schemas:
  `update_targets_every_frame`): 1 - every frame, an aura following its owner (Rally, Hold the Line): one coming in
  gets it, one leaving loses it; 0 - once at the cast: the friends in range get it and keep it for the whole active
  time wherever they go, later arrivals get nothing (Stand Your Ground). · database, 142 recorded battles, probe T-E2 ·
  high.
- **Auras and routing**: Hold the Line, Rally and Stand Your Ground act from a routing lord and on routing friends
  (the recordings: 1312 of 1312 s and 2017 of 2017 s); a routing lord cannot cast. The lord's morale aura (+4) is
  another rule: a routing lord loses it. · recordings · high.
- **The range's edge**: the distance from the unit's centre to its owner's centre ≤ the range (35 m), not to the
  nearest man: in probe T-E2 a unit at 34.8 m gets it, at 35.5 / 35.9 m none, a side-on unit at 40.1 m (its nearest men
  ~25 m away) none. In the battle recordings the edge is softer (half at 36 m, none by 41 m) - the cause is not found
  (likely movement and the 1 s sampling). · probe · medium.
- **Healing cap** 75 % of maximum HP (`healing_percentage_cap`); life-steal ratio
  `damage_to_heal_ratio` 0.5. · [twwstats][tws] · high (values).
- **Barrier** regenerates 30 s after the last hit at 20 per second (`barrier_replenishment_delay`,
  `_rate`). Mark of Tzeentch barriers at Ultra: Chaos Warriors 800, Chosen 1000, Marauders 600. ·
  twwstats, fandom · high/medium.

## Attributes (selection)

| Attribute | Effect | Confidence |
|---|---|---|
| Strength in Numbers (Skaven) | +6 leadership, +8 melee defence, −10 % speed while HP > 50 % | high |
| Scurry Away! (Skaven) | +10 % speed at wavering or worse | high |
| Hold the Line! (Empire lord) | +5 melee defence, +4 leadership to allies within 35 m (WH3); from a routing lord too, and on routers | medium |
| Encourage | +4 leadership to nearby allies; doesn't stack with the lord's aura (5.3) | high |
| Causes Fear / Terror | −8 within 20 m / rout for 14 s on a hit if morale ≤ 13 within 5 m | high |
| Unbreakable | never loses leadership, never routs | high |
| Immune to Psychology | immune to fear and terror | high |
| Frenzy | while leadership > 50 % of base: +10 % damage and charge, +10 melee attack, Immune to Psychology | medium |
| Regeneration | 0.10 % HP a second, +20 % fire weakness, within the 75 % cap | medium |
| Perfect Vigour | never tires | high |
| Strider | ignores terrain/slope speed and combat penalties, passes trees | medium |
| Devastating Flanker | ×2 charge bonus vs flank/rear | medium |
| Executioner | kills targets below 20 % HP (`execute_threshold` 0.2) | high |
| Expendable | its rout scares only other expendables | high |
| Vanguard Deployment | may deploy outside the deployment zone, not in the enemy's | medium |
| Hide (forest) / Stalk / Unspottable / Snipe | hidden in forests / hidden moving anywhere / revealed only at 20 m / stays hidden while firing | medium/high |
| Daemonic | 20 % physical resistance; no rout (instability) | medium |
| Glorious Charge (4.2.0) | double charge duration; target takes the flank morale penalty | high |

Sources: [fandom Attributes][fw-attr], individual fandom pages, [twwstats][tws]. Ours: every
attribute and passive of our units is an innate effect (`config/nn/effects.json`, the database's
numbers and conditions; [unit passports](../../training/units.md#innate-effects)); Strength in
Numbers and Scurry Away! are modelled, with their traces measured in the recordings
([simulator](../../training/simulator.md#innate-effects)).

## Contact effects

Only one at a time on a unit, refreshed by hits, not reduced by resistances, not applied by
friendly missile fire (since 2020): Poison −15 % speed and damage, 10 s; Frostbite −30 % speed,
10 s; Disorganised −6 leadership and no charge defence, 20 s; Sundered armour −30, 10 s;
Disrupted −10 % physical resistance + silenced, 10 s. · [fandom Contact Effect][fw-ce] · medium.

## Army-wide mechanics

- **Waaagh! (Greenskins):** points from entities in melee; threshold 250 × an army-size factor
  (0.6 / 0.7 / 0.85 / 1 small→ultra), ×2 after each use; effect 18 s: +25 % damage, +24 melee
  attack, Immune to Psychology. · twwstats, fandom · high/medium.
- **Battle currency** (Slaanesh etc.): income every 1 s; charging units give 6; enemies under
  negative ability effects pay ×3.5, contact effects ×1.75; next to a shattered enemy within 40 m;
  unit-size factor small 4 … ultra 1. · twwstats · high (values).
- **Grand Cathay Harmony:** Yin and Yang units within `harmony_proximity_radius` 60 m keep their
  buffs. · twwstats, fandom · high/medium.
- **Army abilities** are mostly campaign-only (not in custom battles). · fandom · medium.

## Reinforcements and withdrawal

- Reinforcements arrive on a timer: initial wait 45 s, estimate 50–300 s, +60 s base factor per
  general level; moving the entry point costs 0.1 s per metre or 90 s for another side (no range
  limit since 4.0); up to 5 units per batch, 5 s queue waits. · [twwstats][tws] · high (values).
- Withdrawing by teleport: 10 s countdown, unit HP ×0.5, range check 30. · twwstats · high (values), low (meaning).
- About 40 units per side can be on the field; extra units wait for others to rout/die. · Steam · medium.

[fw-ab]: https://totalwarwarhammer.fandom.com/wiki/Abilities
[fw-attr]: https://totalwarwarhammer.fandom.com/wiki/Attributes
[fw-ce]: https://totalwarwarhammer.fandom.com/wiki/Contact_Effect
[tws]: https://twwstats.com/kv/rules
