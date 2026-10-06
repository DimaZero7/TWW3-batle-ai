# Units

[← Back](../README.md) · [Documentation](../../README.md) › [Game knowledge](../README.md) › Units · [Русский](../../../ru/game/units/README.md)

What is verified about units in the game: what can be read, which orders work
and how the battle mechanics work. Command and readout experiments — 26
September 2026 on WH3 **v9.0.0, build 50218** and 27–28 September on **v9.0.1,
build 50381**; mechanics from the game's database and arena battles — 30
September 2026, **v9.0.1, build 50381**.

<!-- generated:docs:index -->
- **[Battle unit catalogue](catalog/README.md)** — Short unit cards grouped by faction
- [Commands tested in battle](commands.md) — Which unit orders from Lua we checked in battle and how the engine carries them out
- [Starting deployment](deployment.md) — Verified on 26 September 2026 in five controlled diagnostic launches on [The Moorlands Route](../maps/moorlands-route.md), using 15 Empire units per side (Gener…
- [Live experiments and reproducibility](evidence.md) — Ten completed automatic runs, **55,356 samples**, with raw orders/state transitions and verified cleanup records
- [Indicator registry of our units](indicators.md) — Every indicator of our 17 units that the simulator must count the way the game does: card numbers, the mechanics that depend on them, attributes, innate effects…
- [A lord surrounded](lord-swarm.md) — How much a lord loses when one to four infantry units attack him from different sides, and what an armour-piercing unit or the enemy lord adds
- [Melee](melee.md) — How the game counts blows in melee and how many men really fight
- [Missile damage](missile-damage.md) — How much health an Empire archers' arrow takes, how often they shoot and at whom
- [Missile attack range](missile-range.md) — **Verified 2026-09-26:** Lua can read missile range for own units, a separate same-alliance army, and currently visible enemies
- [Morale](morale.md) — Morale decides when a unit runs
- [Pace: running, walking and fatigue](pace.md) — How fast units walk and run in battle and how fatigue builds up
- [Unit roster](roster.md) — The roster is data collected in advance for each unit type: the card as the player sees it and the measured formation at each width
- [Verified state fields](state-fields.md) — Every unit state field we read in three tests, with its type and the values we saw
- [Common unit state sensors](state-sensors.md) — This is shared infrastructure knowledge: how to read a unit's changing condition
- [States and passive effects](states.md) — Which modes, abilities and passive effects the game returned for Empire spearmen, archers and the general, and what of it was checked in battle
- [Visibility and hiding](visibility.md) — When the enemy sees a hidden unit, how an ambush hides again and how the ground hides the enemy from us
<!-- /generated -->

**Battle mechanics:** [morale](morale.md) · [melee](melee.md) · [missile damage](missile-damage.md) · [pace and fatigue](pace.md) · [a lord surrounded](lord-swarm.md) — rules from the game's database and arena battle recordings.

**[Indicator registry of our units](indicators.md)** — every indicator of our 17 units: the game's rule, status in the simulator, the in-game check; a new unit adds its own there.

## Test conditions

Test units: **120 shieldless Spearmen**, **90 Archers**, **one foot General of the Empire**. Unit experience is zero, Ultra unit size, requested battle speed ×20, minimum graphics. Only the diagnostic script pack is loaded; unit DB records are unchanged. The final general setup is explicitly commanding, character rank 1, unit experience 0; these are separate runtime fields.

**An XML battle's default abilities are not a campaign skill build.** The rank-1 XML general exposes two active abilities and Hold the Line. This proves their availability in this scenario, not their availability to a newly recruited campaign lord. Always read the actual unit's abilities before ordering one.

The guide contains measured behavior and explicit limits. Unsupported hypotheses remain in ignored `tmp/unit-actions/`. Games started for these experiments are closed after capture.
