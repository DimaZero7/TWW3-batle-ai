# Readout catalogue

[Game knowledge](README.md) · [Русский](../../ru/game/readouts.md)

Everything readable about units and the battle: what we already collect and what the game offers but we have not tested yet. Each run picks a profile — the list of what to collect ([config/readouts](../../../config/readouts/)). Generated from [data/readouts/catalog.json](../../../data/readouts/catalog.json) by `python -m tools.readouts docs`; edit the JSON, not this page.

**Status:** ✅ tracked and verified in game · ⬜ in the game's API, not tested. **AI access:** own only · own and visible enemies · public · full summary only (never an AI input).

Total 139: tracked 85, not yet tested 54.

## What the unit is

| ID | What it is | Status | AI access |
|---|---|---|---|
| `unit.type` | Unit type (game DB key) | ✅ tracked | own and visible enemies |
| `unit.name` | Script name from the scenario | ✅ tracked | own and visible enemies |
| `unit.id` | Unique unit id in the battle | ✅ tracked | full summary only |
| `unit.commanding` | Is the army general | ✅ tracked | own and visible enemies |
| `unit.class` | Unit class (infantry, cavalry…) | ⬜ in the game | own only |
| `unit.kind_flags` | Kind: infantry, cavalry, anti-cavalry, chariot, beasts, elephants, artillery, war machine | ⬜ in the game | own only |
| `unit.is_character` | Character (hero or general) | ⬜ in the game | own only |
| `unit.renown` | Regiment of Renown | ⬜ in the game | own only |
| `unit.spawned` | Spawned during the battle (by an ability) | ⬜ in the game | own only |
| `unit.attributes` | Attributes: hide in forest, stalk, charge defence, charge reflection, encourages | ✅ tracked | own only |
| `unit.behaviours_available` | Available behaviours: defend, skirmish, fire at will, spacing | ✅ tracked | own only |
| `unit.card_stats` | All card stats: attack, defence, armour, charge, damage… | ⬜ in the game | own only |
| `unit.card_morale` | Leadership from the card | ✅ tracked | own only |
| `unit.mass` | Unit mass (charge impact) | ⬜ in the game | own only |
| `unit.experience_level` | Experience level | ✅ tracked | own only |
| `unit.experience_progress` | Progress to the next experience level | ⬜ in the game | own only |
| `unit.character_rank` | Character rank | ✅ tracked | own only |

## Numbers and strength

| ID | What it is | Status | AI access |
|---|---|---|---|
| `native.number_of_men_alive` | Men alive | ✅ tracked | own only |
| `native.initial_number_of_men` | Men at start | ✅ tracked | own only |
| `native.unary_of_men_alive` | Share of men alive | ✅ tracked | own only |
| `cco.NumEntities` | Entities now (UI) | ✅ tracked | own only |
| `cco.NumEntitiesInitial` | Entities at start (UI) | ✅ tracked | own only |
| `unit.strategic_value` | Game's strength estimate (one number) | ⬜ in the game | own only |
| `cco.PercentCasualtiesRecently` | Share of men lost in the last 4 s | ⬜ in the game | own only |

## Health and shields

| ID | What it is | Status | AI access |
|---|---|---|---|
| `native.unary_hitpoints` | Health as share of start | ✅ tracked | own only |
| `cco.HealthValue` | Health (raw) | ✅ tracked | own only |
| `cco.HealthMax` | Max health | ✅ tracked | own only |
| `cco.HealthPercent` | Health percent (UI) | ✅ tracked | own only |
| `cco.PercentHpLostRecently` | Health lost in the last 4 s | ✅ tracked | own only |
| `cco.IsTakingDamage` | Taking damage right now | ✅ tracked | own only |
| `health.delayed` | Delayed health (UI bar) | ⬜ in the game | own only |
| `health.barrier` | Barrier: remaining, recharging, time to recharge | ⬜ in the game | own only |
| `health.healing` | Healing: power and how much can be restored | ⬜ in the game | own only |
| `health.wounded` | Wounded state | ⬜ in the game | own only |
| `health.invulnerable` | Invulnerable | ⬜ in the game | own only |

## Morale

| ID | What it is | Status | AI access |
|---|---|---|---|
| `native.is_wavering` | Wavering | ✅ tracked | own only |
| `native.is_routing` | Routing | ✅ tracked | own only |
| `native.is_shattered` | Shattered | ✅ tracked | own only |
| `cco.MoralePercent` | Morale percent | ✅ tracked | own only |
| `cco.MoraleState` | Morale state (number) | ✅ tracked | own only |
| `cco.MoraleName` | Morale state (text) | ✅ tracked | own only |
| `cco.MoraleGreatestEffect` | Biggest morale factor | ✅ tracked | own only |
| `cco.IsRouting` | Routing (UI) | ✅ tracked | own only |
| `cco.IsShattered` | Shattered (UI) | ✅ tracked | own only |
| `cco.IsWavering` | Wavering (UI) | ✅ tracked | own only |
| `cco.IsInLastStand` | Last stand | ✅ tracked | own only |
| `cco.IsOutOfControl` | Out of control | ✅ tracked | own only |
| `cco.IsAwaitingOrderAfterRally` | Rallied, awaiting orders | ✅ tracked | own only |
| `morale.terrified` | Terrified | ⬜ in the game | own only |
| `morale.rampaging` | Rampaging | ⬜ in the game | own only |
| `morale.undead` | Undead/daemons: crumbling, unstable | ⬜ in the game | own only |

## Fatigue

| ID | What it is | Status | AI access |
|---|---|---|---|
| `native.fatigue_state` | Fatigue state | ✅ tracked | own only |
| `cco.FatigueState` | Fatigue (number) | ✅ tracked | own only |
| `cco.FatigueName` | Fatigue (text) | ✅ tracked | own only |

## Movement and position

| ID | What it is | Status | AI access |
|---|---|---|---|
| `native.position` | Unit position | ✅ tracked | own and visible enemies |
| `native.ordered_position` | Ordered position | ✅ tracked | own only |
| `native.bearing` | Facing | ✅ tracked | own only |
| `native.ordered_bearing` | Ordered facing | ✅ tracked | own only |
| `native.ordered_width` | Ordered width | ✅ tracked | own only |
| `native.is_moving` | Moving | ✅ tracked | own only |
| `native.is_moving_fast` | Moving fast | ✅ tracked | own only |
| `native.is_idle` | Idle | ✅ tracked | own only |
| `native.is_leaving_battle` | Leaving the battle | ✅ tracked | own only |
| `native.slow_speed` | Walking speed | ✅ tracked | own only |
| `native.fast_speed` | Running speed | ✅ tracked | own only |
| `cco.IsWithdrawing` | Withdrawing | ✅ tracked | own only |
| `move.walk_run` | Walking or running (UI) | ⬜ in the game | own only |
| `move.flying` | Can fly, flying now, can toggle | ⬜ in the game | own only |
| `move.officer_position` | Officer position | ⬜ in the game | own only |
| `move.soldier_positions` | Every soldier's position (real formation) | ⬜ in the game | own only |
| `move.buildings_walls` | In a building, on a wall, climbing a ladder | ⬜ in the game | own only |
| `move.radar_position` | Minimap position | ⬜ in the game | own only |
| `nav.can_reach` | Can the unit reach a point | ✅ tracked | own only |
| `nav.deploy_state` | Deploying / deployed | ✅ tracked | own only |

## Orders and control

| ID | What it is | Status | AI access |
|---|---|---|---|
| `native.is_script_controlled` | Script controlled | ✅ tracked | own only |
| `native.is_controllable` | Controllable | ✅ tracked | own only |
| `native.behaviours_active` | Active behaviours: defend, skirmish, fire at will, spacing | ✅ tracked | own only |
| `orders.controller` | Who controls the unit: AI or player | ⬜ in the game | own only |
| `army.player_controlled` | Army is player controlled | ✅ tracked | public |

## Combat: who hits whom

| ID | What it is | Status | AI access |
|---|---|---|---|
| `native.is_in_melee` | In melee | ✅ tracked | own only |
| `native.current_target` | Current target (only if visible) | ✅ tracked | own only |
| `native.flank_threats` | Left/right/rear flank threat and by whom | ✅ tracked | own only |
| `cco.DamageInflictedRecently` | Damage dealt in the last 4 s | ✅ tracked | own only |
| `cco.NumKills` | Kills | ✅ tracked | own only |
| `combat.kills_native` | Kills (engine) | ⬜ in the game | own only |
| `combat.threat_of_unit` | How dangerous a given enemy is for the unit | ⬜ in the game | own only |
| `combat.execution` | Can an enemy hero execute the unit | ⬜ in the game | own only |
| `combat.capturing` | Capturing a point, capture power | ⬜ in the game | own only |
| `combat.pair_distance` | Distance between two units (with footprints) | ✅ tracked | own only |

## Shooting and ammo

| ID | What it is | Status | AI access |
|---|---|---|---|
| `native.ammo_left` | Ammo left | ✅ tracked | own only |
| `native.starting_ammo` | Starting ammo | ✅ tracked | own only |
| `cco.PrimaryAmmoPercent` | Ammo percent | ✅ tracked | own only |
| `native.is_under_missile_attack` | Under missile fire | ✅ tracked | own only |
| `cco.IsFiringMissiles` | Firing now | ✅ tracked | own only |
| `range.missile_range` | Missile range (engine and card) | ✅ tracked | own only |
| `range.in_range` | Target in range | ✅ tracked | own only |
| `ranged.reload` | Reload: time to next shot | ⬜ in the game | own only |
| `ranged.projectile` | Current projectile type | ⬜ in the game | own only |
| `ranged.fire_at_will` | Fire at will on (UI) | ⬜ in the game | own only |
| `ranged.secondary_ammo` | Secondary ammo, infinite ammo | ⬜ in the game | own only |

## Abilities and magic

| ID | What it is | Status | AI access |
|---|---|---|---|
| `abilities.owned` | Abilities the unit owns | ✅ tracked | own only |
| `abilities.can_perform` | Can perform an ability now | ✅ tracked | own only |
| `abilities.state` | Ability state: recharge, uses left, active, range, mana, miscast chance | ⬜ in the game | own only |
| `abilities.effects_count` | Number of active effects | ✅ tracked | own only |
| `abilities.effects` | Which effects are active and time left | ⬜ in the game | own only |
| `cco.StatusList` | Status icons: braced, hidden, firing, melee… | ✅ tracked | own only |
| `magic.winds` | Army winds of magic and recharge | ⬜ in the game | own only |
| `magic.can_use` | Unit can use magic | ⬜ in the game | own only |

## Visibility and hiding

| ID | What it is | Status | AI access |
|---|---|---|---|
| `vis.visible_to_side` | Is the unit visible to a side | ✅ tracked | own and visible enemies |
| `vis.last_seen` | Where and when an enemy was last seen | ✅ tracked | own and visible enemies |
| `native.is_hidden` | Unit is hidden | ✅ tracked | own only |
| `vis.unspottable` | Unspottable by nature | ⬜ in the game | own only |
| `vis.guerrilla` | Can deploy outside its zone | ⬜ in the game | own only |

## Army and sides

| ID | What it is | Status | AI access |
|---|---|---|---|
| `army.role` | Attacker or defender | ✅ tracked | public |
| `army.counts` | Army: units and men left, men died, reinforcements | ⬜ in the game | own only |
| `army.general_alive` | Is the army general alive | ⬜ in the game | own only |
| `army.score` | Army score | ⬜ in the game | own only |
| `battle.balance_of_power` | Balance of power (top bar) | ⬜ in the game | public |
| `alliance.timeout_winner` | Who wins on timeout | ⬜ in the game | own only |

## Battlefield

| ID | What it is | Status | AI access |
|---|---|---|---|
| `battle.phase` | Battle phase | ✅ tracked | public |
| `battle.time` | Battle time | ✅ tracked | public |
| `battle.time_left` | Time left until the limit | ⬜ in the game | public |
| `battle.outcome` | Outcome decided, winner | ✅ tracked | public |
| `battle.speed` | Game speed | ✅ tracked | public |
| `battle.kind` | Battle kind: campaign, multiplayer, siege, ambush… | ✅ tracked | public |
| `battle.boundaries` | Playable area boundary | ⬜ in the game | public |
| `battle.weather` | Weather, climate, season | ⬜ in the game | public |
| `battle.capture_points` | Capture points: owner, progress, contested by | ⬜ in the game | public |
| `battle.reinforcements` | Reinforcements and their currency | ⬜ in the game | own only |

## Battle events

| ID | What it is | Status | AI access |
|---|---|---|---|
| `event.entity_hit` | Projectile hit: which unit, artillery or not | ⬜ in the game | full summary only |
| `event.orders` | Issued orders: attack whom, move, ability, shot type, withdraw, halt | ⬜ in the game | own only |
| `event.routs` | Unit routed, general routed | ⬜ in the game | own only |
| `event.left_battlefield` | Unit left the battlefield | ⬜ in the game | own only |
| `event.armies_engaging` | Armies engaged | ⬜ in the game | own only |

## Map

| ID | What it is | Status | AI access |
|---|---|---|---|
| `map.radar_frame` | Minimap frame (field size) | ✅ tracked | public |
| `map.height` | Terrain height | ✅ tracked | public |
| `map.ground` | Ground type: grass, forest, water… | ✅ tracked | public |
| `map.clear` | Is an area clear | ✅ tracked | public |
| `map.buildings` | Buildings: where, type, health, bridge | ✅ tracked | public |

<details><summary>Technical calls</summary>

| ID | API | Module |
|---|---|---|
| `unit.type` | `unit:type()` | entries.unit_readout (profile) |
| `unit.name` | `unit:name()` | apps.battle.adapter |
| `unit.id` | `unit:unique_ui_id()` | apps.telemetry.sampler_adapter |
| `unit.commanding` | `unit:is_commanding_unit()` | entries.unit_readout (profile) |
| `unit.class` | `unit:unit_class()` | — |
| `unit.kind_flags` | `unit:is_infantry/is_cavalry/is_pikemen/is_anti_cavalry_infantry/is_lancers/is_chariot/is_war_beasts/is_elephants/is_artillery/is_war_machine()` | — |
| `unit.is_character` | `CCO IsCharacter / IsGeneral` | — |
| `unit.renown` | `CCO IsRenown` | — |
| `unit.spawned` | `CCO IsSpawnedUnit` | — |
| `unit.attributes` | `unit:has_attribute(key)` | entries.unit_readout (profile) |
| `unit.behaviours_available` | `unit:can_use_behaviour(key)` | entries.unit_readout (profile) |
| `unit.card_stats` | `CCO UnitDetailsContext.StatList / BaseStatValueFromKey` | — |
| `unit.card_morale` | `CCO UnitDetailsContext.StatList stat_morale` | apps.units.state_adapter |
| `unit.mass` | `CCO UnitDetailsContext.Mass` | — |
| `unit.experience_level` | `CCO ExperienceLevel` | entries.unit_readout (profile) |
| `unit.experience_progress` | `CCO ExperiencePercent` | — |
| `unit.character_rank` | `CCO CharacterRank / HasCharacterRank` | entries.unit_readout (profile) |
| `native.number_of_men_alive` | `unit:number_of_men_alive()` | apps.units.state_adapter |
| `native.initial_number_of_men` | `unit:initial_number_of_men()` | apps.units.state_adapter |
| `native.unary_of_men_alive` | `unit:unary_of_men_alive()` | apps.units.state_adapter |
| `cco.NumEntities` | `CCO NumEntities` | apps.units.state_adapter |
| `cco.NumEntitiesInitial` | `CCO NumEntitiesInitial` | apps.units.state_adapter |
| `unit.strategic_value` | `unit:strategic_value()` | — |
| `cco.PercentCasualtiesRecently` | `CCO PercentCasualtiesRecently` | — |
| `native.unary_hitpoints` | `unit:unary_hitpoints()` | apps.units.state_adapter |
| `cco.HealthValue` | `CCO HealthValue` | apps.units.state_adapter |
| `cco.HealthMax` | `CCO HealthMax` | apps.units.state_adapter |
| `cco.HealthPercent` | `CCO HealthPercent` | apps.units.state_adapter |
| `cco.PercentHpLostRecently` | `CCO PercentHpLostRecently` | apps.units.state_adapter |
| `cco.IsTakingDamage` | `CCO IsTakingDamage` | apps.units.state_adapter |
| `health.delayed` | `CCO DelayedHealthPercent` | — |
| `health.barrier` | `CCO BarrierHp / BarrierMaxHp / BarrierHpPercent / IsBarrierCharging / BarrierSecsUntilRecharge` | — |
| `health.healing` | `CCO HealingPower / HealthReplenishRateBase / MaxHealthPercentCanReplenish` | — |
| `health.wounded` | `CCO IsWounded` | — |
| `health.invulnerable` | `unit:is_invulnerable() / CCO IsInvincible` | — |
| `native.is_wavering` | `unit:is_wavering()` | apps.units.state_adapter |
| `native.is_routing` | `unit:is_routing()` | apps.units.state_adapter |
| `native.is_shattered` | `unit:is_shattered()` | apps.units.state_adapter |
| `cco.MoralePercent` | `CCO MoralePercent` | apps.units.state_adapter |
| `cco.MoraleState` | `CCO MoraleState` | apps.units.state_adapter |
| `cco.MoraleName` | `CCO MoraleName` | apps.units.state_adapter |
| `cco.MoraleGreatestEffect` | `CCO MoraleGreatestEffect` | apps.units.state_adapter |
| `cco.IsRouting` | `CCO IsRouting` | apps.units.state_adapter |
| `cco.IsShattered` | `CCO IsShattered` | apps.units.state_adapter |
| `cco.IsWavering` | `CCO IsWavering` | apps.units.state_adapter |
| `cco.IsInLastStand` | `CCO IsInLastStand` | apps.units.state_adapter |
| `cco.IsOutOfControl` | `CCO IsOutOfControl` | apps.units.state_adapter |
| `cco.IsAwaitingOrderAfterRally` | `CCO IsAwaitingOrderAfterRally` | apps.units.state_adapter |
| `morale.terrified` | `CCO IsTerrified` | — |
| `morale.rampaging` | `unit:is_rampaging() / CCO IsForceRampage` | — |
| `morale.undead` | `unit:is_crumbling() / is_unstable() / CCO IsUndead` | — |
| `native.fatigue_state` | `unit:fatigue_state()` | apps.units.state_adapter |
| `cco.FatigueState` | `CCO FatigueState` | apps.units.state_adapter |
| `cco.FatigueName` | `CCO FatigueName` | apps.units.state_adapter |
| `native.position` | `unit:position()` | apps.units.state_adapter |
| `native.ordered_position` | `unit:ordered_position()` | apps.units.state_adapter |
| `native.bearing` | `unit:bearing()` | apps.units.state_adapter |
| `native.ordered_bearing` | `unit:ordered_bearing()` | apps.units.state_adapter |
| `native.ordered_width` | `unit:ordered_width()` | apps.units.state_adapter |
| `native.is_moving` | `unit:is_moving()` | apps.units.state_adapter |
| `native.is_moving_fast` | `unit:is_moving_fast()` | apps.units.state_adapter |
| `native.is_idle` | `unit:is_idle()` | apps.units.state_adapter |
| `native.is_leaving_battle` | `unit:is_leaving_battle()` | apps.units.state_adapter |
| `native.slow_speed` | `unit:slow_speed()` | apps.units.state_adapter |
| `native.fast_speed` | `unit:fast_speed()` | apps.units.state_adapter |
| `cco.IsWithdrawing` | `CCO IsWithdrawing` | apps.units.state_adapter |
| `move.walk_run` | `CCO IsWalking / IsRunning` | — |
| `move.flying` | `unit:can_fly() / is_currently_flying() / CCO CanToggleFlying` | — |
| `move.officer_position` | `unit:position_of_officer()` | — |
| `move.soldier_positions` | `CCO ManList / EntityList -> Position` | — |
| `move.buildings_walls` | `unit:is_currently_garrisoned() / is_on_top_of_wall() / is_climbing_ladder()` | — |
| `move.radar_position` | `CCO RadarPosition` | — |
| `nav.can_reach` | `unit:can_reach_position(p)` | apps.navigation.adapter |
| `nav.deploy_state` | `unit:is_deploying() / is_deployed()` | apps.navigation.diagnostics_adapter |
| `native.is_script_controlled` | `unit:is_script_controlled()` | apps.units.state_adapter |
| `native.is_controllable` | `unit:is_controllable()` | apps.units.state_adapter |
| `native.behaviours_active` | `unit:is_behaviour_active(key)` | apps.units.state_adapter |
| `orders.controller` | `unit:is_ai_controlled() / is_player_controlled()` | — |
| `army.player_controlled` | `army:is_player_controlled()` | entries.ai_vs_ai |
| `native.is_in_melee` | `unit:is_in_melee()` | apps.units.state_adapter |
| `native.current_target` | `unit:current_target()` | apps.units.state_adapter |
| `native.flank_threats` | `unit:is_*_flank_threatened() / left_flank_threat() / right_flank_threat() / rear_threat()` | apps.units.state_adapter |
| `cco.DamageInflictedRecently` | `CCO DamageInflictedRecently` | apps.units.state_adapter |
| `cco.NumKills` | `CCO NumKills` | apps.units.state_adapter |
| `combat.kills_native` | `unit:number_of_enemies_killed()` | — |
| `combat.threat_of_unit` | `CCO ThreatPercentOfUnit(unit)` | — |
| `combat.execution` | `CCO CanBeExecutedByNearbyEnemy / CanNearlyBeExecutedByNearbyEnemy` | — |
| `combat.capturing` | `CCO IsCapturing / CapturePower` | — |
| `combat.pair_distance` | `unit:unit_distance(other)` | apps.units.range_adapter |
| `native.ammo_left` | `unit:ammo_left()` | apps.units.state_adapter |
| `native.starting_ammo` | `unit:starting_ammo()` | apps.units.state_adapter |
| `cco.PrimaryAmmoPercent` | `CCO PrimaryAmmoPercent` | apps.units.state_adapter |
| `native.is_under_missile_attack` | `unit:is_under_missile_attack() / CCO IsUnderMissileAttack` | apps.units.state_adapter |
| `cco.IsFiringMissiles` | `CCO IsFiringMissiles` | apps.units.state_adapter |
| `range.missile_range` | `unit:missile_range() / CCO scalar_missile_range` | apps.units.range_adapter |
| `range.in_range` | `unit:unit_in_range(target)` | apps.units.range_adapter |
| `ranged.reload` | `CCO ReloadPercentMax; per soldier IsReloading / ReloadPercent / ReloadRemainingTime` | — |
| `ranged.projectile` | `CCO ActiveProjectileContext` | — |
| `ranged.fire_at_will` | `CCO IsFiringAtWill` | — |
| `ranged.secondary_ammo` | `CCO SecondaryAmmoPercent / HasInfinitePrimaryAmmo / HasInfiniteSecondaryAmmo` | — |
| `abilities.owned` | `unit:owned_passive_special_abilities() / owned_non_passive_special_abilities()` | entries.unit_readout (profile) |
| `abilities.can_perform` | `unit:can_perform_special_ability(key)` | research (unit-state) |
| `abilities.state` | `CCO BattleAbilityList` | — |
| `abilities.effects_count` | `CCO ActiveEffectList.Size` | entries.unit_readout (profile) |
| `abilities.effects` | `CCO ActiveEffectList -> Name / TimeRemainingSeconds` | — |
| `cco.StatusList` | `CCO StatusList` | apps.units.state_adapter |
| `magic.winds` | `CCO WindsOfMagicPoolContext / army:winds_of_magic_current()` | — |
| `magic.can_use` | `unit:can_use_magic()` | — |
| `vis.visible_to_side` | `unit:is_visible_to_alliance(a)` | apps.intel.adapter |
| `vis.last_seen` | `apps.intel.services memory` | apps.intel.services |
| `native.is_hidden` | `unit:is_hidden()` | apps.units.state_adapter |
| `vis.unspottable` | `CCO IsUnspottable` | — |
| `vis.guerrilla` | `CCO CanGuerrillaDeploy / CanGuerrillaDeployCurrently` | — |
| `army.role` | `alliance:is_attacker()` | apps.battle.adapter |
| `army.counts` | `CCO CcoBattleArmy NumUnits / NumEntities / NumMenDied / NumReinforcements` | — |
| `army.general_alive` | `army:is_commander_alive()` | — |
| `army.score` | `CCO CcoBattleArmy Score` | — |
| `battle.balance_of_power` | `CCO BattleRoot.BalanceOfPowerPercent` | — |
| `alliance.timeout_winner` | `CCO CcoBattleAlliance WillWinOnTimeout` | — |
| `battle.phase` | `bm:get_current_phase_name()` | apps.battle / entries |
| `battle.time` | `bm:time_elapsed_ms()` | entries |
| `battle.time_left` | `bm:remaining_conflict_time()` | — |
| `battle.outcome` | `bm:battle_outcome_decided() / victorious_alliance()` | entries |
| `battle.speed` | `bm:current_battle_speed()` | apps.battle.adapter |
| `battle.kind` | `bm:is_from_campaign() … / battle_type()` | apps.battle.adapter |
| `battle.boundaries` | `bm:battle_boundaries()` | — |
| `battle.weather` | `CCO BattleRoot WeatherRecordContext / ClimateRecordContext / SeasonRecordContext` | — |
| `battle.capture_points` | `CCO BattleRoot.CapturePointList` | — |
| `battle.reinforcements` | `CCO BattleRoot reinforcement pool / currency` | — |
| `map.radar_frame` | `BattleRadarPosition` | apps.map.adapter |
| `map.height` | `bm:get_terrain_height(x, z)` | apps.map.adapter |
| `map.ground` | `bm:ground_type(p)` | apps.map.adapter |
| `map.clear` | `bm:is_area_clear(p)` | apps.map.adapter |
| `map.buildings` | `bm:buildings() / CCO BuildingsList` | apps.map.adapter |
| `event.entity_hit` | `command handler 'Entity Hit'` | apps.telemetry.sampler_adapter (parser only) |
| `event.orders` | `command handler 'Attack Unit' / 'Move' / 'Special Ability' / …` | — |
| `event.routs` | `BattleUnitRouts / BattleCommandingUnitRouts` | — |
| `event.left_battlefield` | `command handler 'Unit Left Battlefield'` | — |
| `event.armies_engaging` | `ScriptEventBattleArmiesEngaging` | — |

</details>
