# Abilities, attributes, army mechanics

[← Back](README.md) · [Documentation](../../README.md) › [Game knowledge](../README.md) › [Battle mechanics](README.md) › Abilities · [Русский](../../../ru/game/mechanics/abilities.md)

Common unit attributes and passives with their numbers, contact effects, barriers, army-wide
mechanics (Waaagh!, battle currency) and reinforcements. Conventions: [index](README.md). Ours:
abilities of our lords are read from the database (`config/nn/abilities.json`, `tools/nn/sim/abilities.py`).

## General rules

- Abilities from several sources don't stack; passives show their recharge and condition in the
  UI; some start on cooldown (Gate of Khorne 60 s, counting only in melee). Whether a recharge
  starts at the cast or at the end of the active time: not found. · [fandom Abilities][fw-ab] · medium.
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
| Hold the Line! (Empire lord) | +5 melee defence, +4 leadership to allies within 35 m (WH3) | medium |
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

Sources: [fandom Attributes][fw-attr], individual fandom pages, [twwstats][tws]. Ours: Strength in
Numbers and Scurry Away! are shown to the network but have no effect in the simulator (`sim.json`
abilities); their measured traces are in [morale](morale.md). **Gap.**

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
