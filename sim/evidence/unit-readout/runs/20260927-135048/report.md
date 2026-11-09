# Unit readout test — 20260927-135048

**121 / 121 checks passed.** Map: The Moorlands Route (catchment_03). Samples: 906 own-unit, 2250 range, 906 intel, 30 sampler frames.

| Group | Check | Expected | Actual | Result |
|---|---|---|---|---|
| run | no Lua errors | `0` | `0` | ✅ |
| run | run completed | `completed` | `completed` | ✅ |
| run | all stages reached | `['idle', 'march', 'ranged', 'cease_fire', 'melee', 'halt']` | `['idle', 'march', 'ranged', 'cease_fire', 'melee', 'halt']` | ✅ |
| profile | a_general.type | `wh_main_emp_cha_general_0` | `wh_main_emp_cha_general_0` | ✅ |
| profile | a_general.initial_men | `1` | `1` | ✅ |
| profile | a_general.attribute_hide_forest | `True` | `True` | ✅ |
| profile | a_general.attribute_charge_defense_vs_large | `False` | `False` | ✅ |
| profile | a_general.attribute_charge_reflection | `False` | `False` | ✅ |
| profile | a_general.attribute_encourages | `True` | `True` | ✅ |
| profile | a_general.can_defend | `True` | `True` | ✅ |
| profile | a_general.can_fire_at_will | `False` | `False` | ✅ |
| profile | a_general.can_skirmish | `False` | `False` | ✅ |
| profile | a_general.can_change_formation_spacing | `False` | `False` | ✅ |
| profile | a_general active abilities (doc: Stand Your Ground, Foe Seeker) | `2` | `2` | ✅ |
| profile | a_general passive abilities (doc: Single Entity, Hold the Line) | `2` | `2` | ✅ |
| profile | a_general.cco_CharacterRank readable | `number/boolean` | `1` | ✅ |
| profile | a_general.cco_HasCharacterRank readable | `number/boolean` | `True` | ✅ |
| profile | a_general.cco_ExperienceLevel readable | `number/boolean` | `0` | ✅ |
| profile | a_spears.type | `wh_main_emp_inf_spearmen_0` | `wh_main_emp_inf_spearmen_0` | ✅ |
| profile | a_spears.initial_men | `120` | `120` | ✅ |
| profile | a_spears.missile_range | `0` | `0` | ✅ |
| profile | a_spears.attribute_hide_forest | `True` | `True` | ✅ |
| profile | a_spears.attribute_charge_defense_vs_large | `True` | `True` | ✅ |
| profile | a_spears.attribute_charge_reflection | `True` | `True` | ✅ |
| profile | a_spears.attribute_encourages | `False` | `False` | ✅ |
| profile | a_spears.can_defend | `True` | `True` | ✅ |
| profile | a_spears.can_fire_at_will | `False` | `False` | ✅ |
| profile | a_spears.can_skirmish | `False` | `False` | ✅ |
| profile | a_spears.can_change_formation_spacing | `False` | `False` | ✅ |
| profile | a_spears.cco_CharacterRank readable | `number/boolean` | `0` | ✅ |
| profile | a_spears.cco_HasCharacterRank readable | `number/boolean` | `False` | ✅ |
| profile | a_spears.cco_ExperienceLevel readable | `number/boolean` | `0` | ✅ |
| profile | a_archers.type | `wh2_dlc13_emp_inf_archers_0` | `wh2_dlc13_emp_inf_archers_0` | ✅ |
| profile | a_archers.initial_men | `90` | `90` | ✅ |
| profile | a_archers.missile_range | `130` | `130` | ✅ |
| profile | a_archers.attribute_hide_forest | `True` | `True` | ✅ |
| profile | a_archers.attribute_charge_defense_vs_large | `False` | `False` | ✅ |
| profile | a_archers.attribute_charge_reflection | `False` | `False` | ✅ |
| profile | a_archers.attribute_encourages | `False` | `False` | ✅ |
| profile | a_archers.can_defend | `True` | `True` | ✅ |
| profile | a_archers.can_fire_at_will | `True` | `True` | ✅ |
| profile | a_archers.can_skirmish | `True` | `True` | ✅ |
| profile | a_archers.can_change_formation_spacing | `False` | `False` | ✅ |
| profile | a_archers.cco_CharacterRank readable | `number/boolean` | `0` | ✅ |
| profile | a_archers.cco_HasCharacterRank readable | `number/boolean` | `False` | ✅ |
| profile | a_archers.cco_ExperienceLevel readable | `number/boolean` | `0` | ✅ |
| profile | b_general.type | `wh_main_emp_cha_general_0` | `wh_main_emp_cha_general_0` | ✅ |
| profile | b_general.initial_men | `1` | `1` | ✅ |
| profile | b_general.attribute_hide_forest | `True` | `True` | ✅ |
| profile | b_general.attribute_charge_defense_vs_large | `False` | `False` | ✅ |
| profile | b_general.attribute_charge_reflection | `False` | `False` | ✅ |
| profile | b_general.attribute_encourages | `True` | `True` | ✅ |
| profile | b_general.can_defend | `True` | `True` | ✅ |
| profile | b_general.can_fire_at_will | `False` | `False` | ✅ |
| profile | b_general.can_skirmish | `False` | `False` | ✅ |
| profile | b_general.can_change_formation_spacing | `False` | `False` | ✅ |
| profile | b_general active abilities (doc: Stand Your Ground, Foe Seeker) | `2` | `2` | ✅ |
| profile | b_general passive abilities (doc: Single Entity, Hold the Line) | `2` | `2` | ✅ |
| profile | b_general.cco_CharacterRank readable | `number/boolean` | `1` | ✅ |
| profile | b_general.cco_HasCharacterRank readable | `number/boolean` | `True` | ✅ |
| profile | b_general.cco_ExperienceLevel readable | `number/boolean` | `0` | ✅ |
| profile | b_spears.type | `wh_main_emp_inf_spearmen_0` | `wh_main_emp_inf_spearmen_0` | ✅ |
| profile | b_spears.initial_men | `120` | `120` | ✅ |
| profile | b_spears.missile_range | `0` | `0` | ✅ |
| profile | b_spears.attribute_hide_forest | `True` | `True` | ✅ |
| profile | b_spears.attribute_charge_defense_vs_large | `True` | `True` | ✅ |
| profile | b_spears.attribute_charge_reflection | `True` | `True` | ✅ |
| profile | b_spears.attribute_encourages | `False` | `False` | ✅ |
| profile | b_spears.can_defend | `True` | `True` | ✅ |
| profile | b_spears.can_fire_at_will | `False` | `False` | ✅ |
| profile | b_spears.can_skirmish | `False` | `False` | ✅ |
| profile | b_spears.can_change_formation_spacing | `False` | `False` | ✅ |
| profile | b_spears.cco_CharacterRank readable | `number/boolean` | `0` | ✅ |
| profile | b_spears.cco_HasCharacterRank readable | `number/boolean` | `False` | ✅ |
| profile | b_spears.cco_ExperienceLevel readable | `number/boolean` | `0` | ✅ |
| profile | b_archers.type | `wh2_dlc13_emp_inf_archers_0` | `wh2_dlc13_emp_inf_archers_0` | ✅ |
| profile | b_archers.initial_men | `90` | `90` | ✅ |
| profile | b_archers.missile_range | `130` | `130` | ✅ |
| profile | b_archers.attribute_hide_forest | `True` | `True` | ✅ |
| profile | b_archers.attribute_charge_defense_vs_large | `False` | `False` | ✅ |
| profile | b_archers.attribute_charge_reflection | `False` | `False` | ✅ |
| profile | b_archers.attribute_encourages | `False` | `False` | ✅ |
| profile | b_archers.can_defend | `True` | `True` | ✅ |
| profile | b_archers.can_fire_at_will | `True` | `True` | ✅ |
| profile | b_archers.can_skirmish | `True` | `True` | ✅ |
| profile | b_archers.can_change_formation_spacing | `False` | `False` | ✅ |
| profile | b_archers.cco_CharacterRank readable | `number/boolean` | `0` | ✅ |
| profile | b_archers.cco_HasCharacterRank readable | `number/boolean` | `False` | ✅ |
| profile | b_archers.cco_ExperienceLevel readable | `number/boolean` | `0` | ✅ |
| sensors | own-unit sensors per sample | `68` | `68` | ✅ |
| sensors | sensors known for at least one unit | `68` | `68` | ✅ |
| dynamics | march: a_spears is_moving | `True` | `True` | ✅ |
| dynamics | march: a_general is_moving_fast (run order) | `True` | `True` | ✅ |
| dynamics | march: ordered_position known | `True` | `True` | ✅ |
| dynamics | ranged: a_archers IsFiringMissiles | `True` | `True` | ✅ |
| dynamics | ranged: a_archers ammo_left decreases | `True` | `True` | ✅ |
| dynamics | ranged: b_spears under missile attack (native or CCO) | `True` | `True` | ✅ |
| dynamics | ranged: b_spears HealthValue decreases | `True` | `True` | ✅ |
| dynamics | melee: a_spears is_in_melee | `True` | `True` | ✅ |
| dynamics | melee: b_spears is_in_melee | `True` | `True` | ✅ |
| dynamics | melee: a_spears number_of_men_alive decreases | `True` | `True` | ✅ |
| dynamics | melee: a_general defend behaviour on | `True` | `True` | ✅ |
| dynamics | melee: current_target is a visible unit id | `True` | `True` | ✅ |
| dynamics | any: card stat_morale.Value known | `True` | `True` | ✅ |
| dynamics | any: MoralePercent known | `True` | `True` | ✅ |
| dynamics | any: StatusList keys read | `True` | `True` | ✅ |
| dynamics | halt: a_spears stops moving | `True` | `True` | ✅ |
| gate | enemy view withholds all sensors | `0` | `0` | ✅ |
| gate | enemy gate samples | `>0` | `906` | ✅ |
| range | a_archers missile_range_m = 130 | `130` | `[130]` | ✅ |
| range | a_archers card scalar_missile_range = 130 | `[130]` | `[130]` | ✅ |
| range | a_archers unit_in_range true in ranged stage (100 m) | `True` | `True` | ✅ |
| range | a_spears missile_range_m = 0 | `[0]` | `[0]` | ✅ |
| range | engine unit_distance < centre distance (footprints) | `True` | `True` | ✅ |
| range | withheld pairs expose no range | `True` | `True` | ✅ |
| intel | visibility samples | `>0` | `{'not_visible': 6, 'visible': 900}` | ✅ |
| intel | no current position for hidden units | `0` | `0` | ✅ |
| intel | last_seen present after a sighting | `True` | `True` | ✅ |
| telemetry | sampler frames | `>0` | `30` | ✅ |
| navigation | nav diagnostics per stage | `>0` | `42` | ✅ |
| navigation | can_reach_position own origin answered | `True` | `True` | ✅ |

## All own-unit sensors (value range or seen values; unknown reasons)

| Sensor | a_general | a_spears | a_archers | b_general | b_spears | b_archers |
|---|---|---|---|---|---|---|
| `cco.DamageInflictedRecently` | 0 | 0..218 | 0..923 | 0 | 0..343 | 0..974 |
| `cco.FatigueName` | [[col:green]]Свежие[[/col]] | [[col:green]]Активные[[/col]], [[col:green]]Свежие[[/col]] | [[col:green]]Активные[[/col]], [[col:green]]Свежие[[/col]] | [[col:green]]Свежие[[/col]] | [[col:green]]Активные[[/col]], [[col:green]]Свежие[[/col]] | [[col:green]]Свежие[[/col]] |
| `cco.FatigueState` | 0 | 0..1 | 0..1 | 0 | 0..1 | 0 |
| `cco.HealthMax` | 4068 | 8280 | 6210 | 4068 | 8280 | 6210 |
| `cco.HealthPercent` | 1 | 0.7581..1 | 1 | 1 | 0.4494..1 | 1 |
| `cco.HealthValue` | 4068 | 6277..8280 | 6210 | 4068 | 3721..8280 | 6210 |
| `cco.IsAlive` | True | True | True | True | True | True |
| `cco.IsAwaitingOrderAfterRally` | False | False | False | False | False | False |
| `cco.IsFiringMissiles` | False | False | False, True | False | False | False, True |
| `cco.IsInLastStand` | False | False | False | False | False | False |
| `cco.IsOutOfControl` | False | False | False | False | False | False |
| `cco.IsRouting` | False | False | False | False | False | False |
| `cco.IsShattered` | False | False | False | False | False | False |
| `cco.IsTakingDamage` | False | False, True | False | False | False, True | False |
| `cco.IsUnderMissileAttack` | False | False, True | False | False | False, True | False |
| `cco.IsWavering` | False | False | False | False | False | False |
| `cco.IsWithdrawing` | False | False | False | False | False | False |
| `cco.MoraleGreatestEffect` | , [[col:green]]Вдохновлены с помощью способности[[/col]], [[col:green]]Фланги прикрыты[[/col]] | , [[col:green]]Лорд неподалеку[[/col]], [[col:green]]Натиск[[/col]], [[col:green]]Одерживают верх[[/col]]… | , [[col:green]]Лорд неподалеку[[/col]], [[col:green]]Одерживают верх[[/col]], [[col:red]]Фланг не прикрыт[[/col]] | , [[col:green]]Вдохновлены с помощью способности[[/col]], [[col:green]]Фланги прикрыты[[/col]] | , [[col:green]]Натиск[[/col]], [[col:green]]Фланги прикрыты[[/col]], [[col:red]]Недавний урон[[/col]]… | , [[col:green]]Лорд неподалеку[[/col]], [[col:green]]Фланги прикрыты[[/col]], [[col:red]]Фланг не прикрыт[[/col]] |
| `cco.MoraleName` | [[col:green]]{{tr:confident}}[[/col]], [[col:green]]Жаждут боя[[/col]] | [[col:green]]{{tr:confident}}[[/col]], [[col:green]]Жаждут боя[[/col]] | [[col:green]]{{tr:confident}}[[/col]], [[col:green]]Жаждут боя[[/col]] | [[col:green]]{{tr:confident}}[[/col]], [[col:green]]Жаждут боя[[/col]] | [[col:green]]{{tr:confident}}[[/col]], [[col:green]]{{tr:steady}}[[/col]], [[col:green]]Жаждут боя[[/col]], [[col:red]]Колеблются[[/col]] | [[col:green]]{{tr:confident}}[[/col]], [[col:green]]Жаждут боя[[/col]] |
| `cco.MoralePercent` | 0..1.129 | 0..1.133 | 0..1.24 | 0..1.129 | 0..1.217 | 0..1.26 |
| `cco.MoraleState` | 1..2 | 1..2 | 1..2 | 1..2 | 1..4 | 1..2 |
| `cco.NumEntities` | 1 | 116..120 | 90 | 1 | 83..120 | 90 |
| `cco.NumEntitiesInitial` | 1 | 120 | 90 | 1 | 120 | 90 |
| `cco.NumKills` | 0 | 0..18 | 0..19 | 0 | 0..4 | 0 |
| `cco.PercentHpLostRecently` | 0 | 0..0.1176 | 0 | 0 | 0..0.1115 | 0 |
| `cco.PrimaryAmmoPercent` | 0 | 0 | 0.8389..1 | 0 | 0 | 0.9506..1 |
| `cco.StatusList.Keys` | braced, moving_fast | braced, melee, moving, moving_fast… | braced, firing | braced, moving | braced, melee, moving, moving_fast… | braced, firing, moving |
| `cco.StatusList.Size` | 0..2 | 0..2 | 0..2 | 0..1 | 0..3 | 0..2 |
| `cco.card.stat_morale.DisplayedValue` | 0..79 | 0..70 | 0..62 | 0..79 | 0..73 | 0..63 |
| `cco.card.stat_morale.Value` | 0..79 | 0..70 | 0..62 | 0..79 | 0..73 | 0..63 |
| `cco.card.stat_morale.ValueBase` | 0..79 | 0..70 | 0..62 | 0..79 | 0..73 | 0..63 |
| `native.ammo_left` | 0 | 0 | 1510..1800 | 0 | 0 | 1711..1800 |
| `native.bearing` | 270 | 1.818..357.4 | 251.4..317.8 | 90..237.6 | 92.81..188.8 | 18.96..183.9 |
| `native.behaviour.change_formation_spacing` | False | False | False | False | False | False |
| `native.behaviour.defend` | False, True | False | False | False, True | False | False |
| `native.behaviour.fire_at_will` | False | False | False, True | False | False | False, True |
| `native.behaviour.skirmish` | False | False | False | False | False | False |
| `native.current_target` |  | , b_spears | , b_spears |  | , a_spears | , a_spears |
| `native.fast_speed` | 3.4 | 3 | 3.3 | 3.4 | 3 | 3.3 |
| `native.fatigue_state` | threshold_fresh | threshold_active, threshold_fresh | threshold_active, threshold_fresh | threshold_fresh | threshold_active, threshold_fresh | threshold_fresh |
| `native.initial_number_of_men` | 1 | 120 | 90 | 1 | 120 | 90 |
| `native.is_controllable` | True | True | True | True | True | True |
| `native.is_hidden` | False | False | False | False | False | False |
| `native.is_idle` | False, True | False, True | False, True | False, True | False, True | False, True |
| `native.is_in_melee` | False | False, True | False | False | False, True | False |
| `native.is_leaving_battle` | False | False | False | False | False | False |
| `native.is_left_flank_threatened` | False | False | False, True | False | False, True | False, True |
| `native.is_moving` | False, True | False, True | False | False, True | False, True | False, True |
| `native.is_moving_fast` | False, True | False, True | False | False | False, True | False |
| `native.is_rear_flank_threatened` | False | False | False | False | False | False |
| `native.is_right_flank_threatened` | False | False | False, True | False | False | False |
| `native.is_routing` | False | False | False | False | False | False |
| `native.is_script_controlled` | False, True | False, True | False, True | False, True | False, True | False, True |
| `native.is_shattered` | False | False | False | False | False | False |
| `native.is_under_missile_attack` | False | False, True | False | False | False, True | False |
| `native.is_wavering` | False | False | False | False | False | False |
| `native.left_flank_threat` |  |  | , b_spears |  | , a_archers | , a_archers |
| `native.number_of_men_alive` | 1 | 116..120 | 90 | 1 | 83..120 | 90 |
| `native.ordered_bearing` | 270 | 269.6..341.5 | 270..272.8 | 90..213.6 | 90..190.6 | 90..182.8 |
| `native.ordered_position` |  |  |  |  |  |  |
| `native.ordered_width` | 1.4 | 29.42 | 29.2 | 1.4 | 29.42 | 29.2 |
| `native.position` |  |  |  |  |  |  |
| `native.rear_threat` |  |  |  |  |  |  |
| `native.right_flank_threat` |  |  | , b_archers |  |  |  |
| `native.slow_speed` | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 |
| `native.starting_ammo` | 0 | 0 | 1800 | 0 | 0 | 1800 |
| `native.unary_hitpoints` | 1 | 0.7581..1 | 1 | 1 | 0.4494..1 | 1 |
| `native.unary_of_men_alive` | 1 | 0.9667..1 | 1 | 1 | 0.6917..1 | 1 |
