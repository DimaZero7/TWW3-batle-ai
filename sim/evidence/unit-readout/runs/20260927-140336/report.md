# Unit readout test — 20260927-140336

**167 / 168 checks passed.** Map: The Moorlands Route (catchment_03). Samples: 1860 own-unit, 8460 range, 1860 intel, 186 full-summary frames, 372 side views.

| Group | Check | Expected | Actual | Result |
|---|---|---|---|---|
| run | no Lua errors | `0` | `0` | ✅ |
| run | run completed | `completed` | `completed` | ✅ |
| run | all stages reached | `['idle', 'march', 'ranged', 'cease_fire', 'melee', 'scout', 'halt']` | `['idle', 'march', 'ranged', 'cease_fire', 'melee', 'scout', 'halt']` | ✅ |
| profile | a_general.attribute_stalk | `False` | `False` | ✅ |
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
| profile | a_spears.attribute_stalk | `False` | `False` | ✅ |
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
| profile | a_archers.attribute_stalk | `False` | `False` | ✅ |
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
| profile | a_forest.attribute_stalk | `False` | `False` | ✅ |
| profile | a_forest.type | `wh_main_emp_inf_spearmen_0` | `wh_main_emp_inf_spearmen_0` | ✅ |
| profile | a_forest.initial_men | `120` | `120` | ✅ |
| profile | a_forest.missile_range | `0` | `0` | ✅ |
| profile | a_forest.attribute_hide_forest | `True` | `True` | ✅ |
| profile | a_forest.attribute_charge_defense_vs_large | `True` | `True` | ✅ |
| profile | a_forest.attribute_charge_reflection | `True` | `True` | ✅ |
| profile | a_forest.attribute_encourages | `False` | `False` | ✅ |
| profile | a_forest.can_defend | `True` | `True` | ✅ |
| profile | a_forest.can_fire_at_will | `False` | `False` | ✅ |
| profile | a_forest.can_skirmish | `False` | `False` | ✅ |
| profile | a_forest.can_change_formation_spacing | `False` | `False` | ✅ |
| profile | a_forest.cco_CharacterRank readable | `number/boolean` | `0` | ✅ |
| profile | a_forest.cco_HasCharacterRank readable | `number/boolean` | `False` | ✅ |
| profile | a_forest.cco_ExperienceLevel readable | `number/boolean` | `0` | ✅ |
| profile | a_stalkers.type | `wh2_dlc13_emp_inf_huntsmen_0` | `wh2_dlc13_emp_inf_huntsmen_0` | ✅ |
| profile | a_stalkers.attribute_stalk | `True` | `True` | ✅ |
| profile | a_stalkers.cco_CharacterRank readable | `number/boolean` | `0` | ✅ |
| profile | a_stalkers.cco_HasCharacterRank readable | `number/boolean` | `False` | ✅ |
| profile | a_stalkers.cco_ExperienceLevel readable | `number/boolean` | `0` | ✅ |
| profile | b_general.attribute_stalk | `False` | `False` | ✅ |
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
| profile | b_spears.attribute_stalk | `False` | `False` | ✅ |
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
| profile | b_archers.attribute_stalk | `False` | `False` | ✅ |
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
| profile | b_forest.attribute_stalk | `False` | `False` | ✅ |
| profile | b_forest.type | `wh_main_emp_inf_spearmen_0` | `wh_main_emp_inf_spearmen_0` | ✅ |
| profile | b_forest.initial_men | `120` | `120` | ✅ |
| profile | b_forest.missile_range | `0` | `0` | ✅ |
| profile | b_forest.attribute_hide_forest | `True` | `True` | ✅ |
| profile | b_forest.attribute_charge_defense_vs_large | `True` | `True` | ✅ |
| profile | b_forest.attribute_charge_reflection | `True` | `True` | ✅ |
| profile | b_forest.attribute_encourages | `False` | `False` | ✅ |
| profile | b_forest.can_defend | `True` | `True` | ✅ |
| profile | b_forest.can_fire_at_will | `False` | `False` | ✅ |
| profile | b_forest.can_skirmish | `False` | `False` | ✅ |
| profile | b_forest.can_change_formation_spacing | `False` | `False` | ✅ |
| profile | b_forest.cco_CharacterRank readable | `number/boolean` | `0` | ✅ |
| profile | b_forest.cco_HasCharacterRank readable | `number/boolean` | `False` | ✅ |
| profile | b_forest.cco_ExperienceLevel readable | `number/boolean` | `0` | ✅ |
| profile | b_stalkers.type | `wh2_dlc13_emp_inf_huntsmen_0` | `wh2_dlc13_emp_inf_huntsmen_0` | ✅ |
| profile | b_stalkers.attribute_stalk | `True` | `True` | ✅ |
| profile | b_stalkers.cco_CharacterRank readable | `number/boolean` | `0` | ✅ |
| profile | b_stalkers.cco_HasCharacterRank readable | `number/boolean` | `False` | ✅ |
| profile | b_stalkers.cco_ExperienceLevel readable | `number/boolean` | `0` | ✅ |
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
| dynamics | melee: a_spears number_of_men_alive decreases | `True` | `False` | ❌ |
| dynamics | melee: a_general defend behaviour on | `True` | `True` | ✅ |
| dynamics | melee: current_target is a visible unit id | `True` | `True` | ✅ |
| dynamics | any: card stat_morale.Value known | `True` | `True` | ✅ |
| dynamics | any: MoralePercent known | `True` | `True` | ✅ |
| dynamics | any: StatusList keys read | `True` | `True` | ✅ |
| dynamics | halt: a_spears stops moving | `True` | `True` | ✅ |
| gate | enemy view withholds all sensors | `0` | `0` | ✅ |
| gate | enemy gate samples | `>0` | `1860` | ✅ |
| range | a_archers missile_range_m = 130 | `130` | `[130]` | ✅ |
| range | a_archers card scalar_missile_range = 130 | `[130]` | `[130]` | ✅ |
| range | a_archers unit_in_range true in ranged stage (100 m) | `True` | `True` | ✅ |
| range | a_spears missile_range_m = 0 | `[0]` | `[0]` | ✅ |
| range | engine unit_distance < centre distance (footprints) | `True` | `True` | ✅ |
| range | withheld pairs expose no range | `True` | `True` | ✅ |
| intel | visibility samples | `>0` | `{'not_visible': 554, 'visible': 1306}` | ✅ |
| intel | no current position for hidden units | `0` | `0` | ✅ |
| intel | last_seen present after a sighting | `True` | `True` | ✅ |
| telemetry | full summary frames | `>0` | `186` | ✅ |
| telemetry | side views (both sides) | `>0` | `372` | ✅ |
| navigation | nav diagnostics per stage | `>0` | `80` | ✅ |
| navigation | can_reach_position own origin answered | `True` | `True` | ✅ |

## All own-unit sensors (value range or seen values; unknown reasons)

| Sensor | a_general | a_spears | a_archers | a_forest | a_stalkers | b_general | b_spears | b_archers | b_forest | b_stalkers |
|---|---|---|---|---|---|---|---|---|---|---|
| `cco.DamageInflictedRecently` | 0 | 0..166 | 0..860 | 0 | 0 | 0 | 0..255 | 0 | 0 | 0 |
| `cco.FatigueName` | [[col:green]]Свежие[[/col]] | [[col:fatigue_winded]]Измотанные[[/col]], [[col:green]]Активные[[/col]], [[col:green]]Свежие[[/col]] | [[col:green]]Активные[[/col]], [[col:green]]Свежие[[/col]] | [[col:green]]Свежие[[/col]] | [[col:green]]Свежие[[/col]] | [[col:green]]Свежие[[/col]] | [[col:fatigue_winded]]Измотанные[[/col]], [[col:green]]Активные[[/col]], [[col:green]]Свежие[[/col]] | [[col:green]]Свежие[[/col]] | [[col:green]]Свежие[[/col]] | [[col:green]]Свежие[[/col]] |
| `cco.FatigueState` | 0 | 0..2 | 0..1 | 0 | 0 | 0 | 0..2 | 0 | 0 | 0 |
| `cco.HealthMax` | 4068 | 8280 | 6210 | 8280 | 4140 | 4068 | 8280 | 6210 | 8280 | 4140 |
| `cco.HealthPercent` | 1 | 0.8077..1 | 0.9097..1 | 1 | 1 | 1 | 0.3441..1 | 1 | 1 | 1 |
| `cco.HealthValue` | 4068 | 6688..8280 | 5649..6210 | 8280 | 4140 | 4068 | 2849..8280 | 6210 | 8280 | 4140 |
| `cco.IsAlive` | True | True | True | True | True | True | True | True | True | True |
| `cco.IsAwaitingOrderAfterRally` | False | False | False | False | False | False | False | False | False | False |
| `cco.IsFiringMissiles` | False | False | False, True | False | False | False | False | False | False | False |
| `cco.IsInLastStand` | False | False | False | False | False | False | False | False | False | False |
| `cco.IsOutOfControl` | False | False | False | False | False | False | False | False | False | False |
| `cco.IsRouting` | False | False | False | False | False | False | False | False | False | False |
| `cco.IsShattered` | False | False | False | False | False | False | False | False | False | False |
| `cco.IsTakingDamage` | False | False, True | False, True | False | False | False | False, True | False | False | False |
| `cco.IsUnderMissileAttack` | False | False | False | False | False | False | False, True | False | False | False |
| `cco.IsWavering` | False | False | False | False | False | False | False | False | False | False |
| `cco.IsWithdrawing` | False | False | False | False | False | False | False | False | False | False |
| `cco.MoraleGreatestEffect` | , [[col:green]]Вдохновлены с помощью способности[[/col]], [[col:green]]Фланги прикрыты[[/col]], [[col:red]]Фланги не прикрыты[[/col]] | , [[col:green]]Лорд неподалеку[[/col]], [[col:green]]Натиск[[/col]], [[col:green]]Одерживают верх[[/col]]… | , [[col:green]]Лорд неподалеку[[/col]], [[col:green]]Одерживают верх[[/col]], [[col:red]]Фланги не прикрыты[[/col]] | , [[col:green]]Фланги прикрыты[[/col]] | , [[col:green]]Фланги прикрыты[[/col]] | , [[col:green]]Вдохновлены с помощью способности[[/col]], [[col:green]]Фланги прикрыты[[/col]] | , [[col:green]]Лорд неподалеку[[/col]], [[col:green]]Натиск[[/col]], [[col:green]]Фланги прикрыты[[/col]]… | , [[col:green]]Лорд неподалеку[[/col]], [[col:green]]На возвышенности[[/col]], [[col:green]]Фланги прикрыты[[/col]]… | , [[col:green]]Фланги прикрыты[[/col]] | , [[col:green]]Лорд неподалеку[[/col]] |
| `cco.MoraleName` | [[col:green]]{{tr:confident}}[[/col]], [[col:green]]Жаждут боя[[/col]] | [[col:green]]{{tr:confident}}[[/col]], [[col:green]]Жаждут боя[[/col]] | [[col:green]]{{tr:confident}}[[/col]], [[col:green]]Жаждут боя[[/col]] | [[col:green]]{{tr:confident}}[[/col]], [[col:green]]Жаждут боя[[/col]] | [[col:green]]{{tr:confident}}[[/col]], [[col:green]]Жаждут боя[[/col]] | [[col:green]]{{tr:confident}}[[/col]], [[col:green]]Жаждут боя[[/col]] | [[col:green]]{{tr:confident}}[[/col]], [[col:green]]{{tr:steady}}[[/col]], [[col:green]]Жаждут боя[[/col]], [[col:red]]Колеблются[[/col]] | [[col:green]]{{tr:confident}}[[/col]], [[col:green]]Жаждут боя[[/col]] | [[col:green]]{{tr:confident}}[[/col]], [[col:green]]Жаждут боя[[/col]] | [[col:green]]{{tr:confident}}[[/col]], [[col:green]]Жаждут боя[[/col]] |
| `cco.MoralePercent` | 0..1.129 | 0..1.267 | 0..1.24 | 0..1.083 | 0..1.091 | 0..1.129 | 0..1.217 | 0..1.46 | 0..1.217 | 0..1.145 |
| `cco.MoraleState` | 1..2 | 1..2 | 1..2 | 1..2 | 1..2 | 1..2 | 1..4 | 1..2 | 1..2 | 1..2 |
| `cco.NumEntities` | 1 | 109..120 | 90 | 120 | 60 | 1 | 70..120 | 90 | 120 | 60 |
| `cco.NumEntitiesInitial` | 1 | 120 | 90 | 120 | 60 | 1 | 120 | 90 | 120 | 60 |
| `cco.NumKills` | 0 | 0..36 | 0..14 | 0 | 0 | 0 | 0..11 | 0 | 0 | 0 |
| `cco.PercentHpLostRecently` | 0 | 0..0.02041 | 0..0.02029 | 0 | 0 | 0 | 0..0.1039 | 0 | 0 | 0 |
| `cco.PrimaryAmmoPercent` | 0 | 0 | 0.8339..1 | 0 | 1 | 0 | 0 | 1 | 0 | 1 |
| `cco.StatusList.Keys` | braced, moving_fast | braced, melee, moving, moving_fast | braced, firing, melee, moving_fast | braced, hidden | braced, hidden | braced, moving, moving_fast | braced, melee, moving, moving_fast… | braced, moving, moving_fast | braced | braced, hidden |
| `cco.StatusList.Size` | 0..2 | 0..2 | 0..2 | 0..2 | 0..2 | 0..2 | 0..4 | 0..1 | 0..1 | 0..2 |
| `cco.card.stat_morale.DisplayedValue` | 0..79 | 0..76 | 0..61 | 0..65 | 0..60 | 0..79 | 0..73 | 0..73 | 0..73 | 0..63 |
| `cco.card.stat_morale.Value` | 0..79 | 0..76 | 0..61 | 0..65 | 0..60 | 0..79 | 0..73 | 0..73 | 0..73 | 0..63 |
| `cco.card.stat_morale.ValueBase` | 0..79 | 0..76 | 0..61 | 0..65 | 0..60 | 0..79 | 0..73 | 0..73 | 0..73 | 0..63 |
| `native.ammo_left` | 0 | 0 | 1501..1800 | 0 | 1320 | 0 | 0 | 1800 | 0 | 1320 |
| `native.bearing` | 270..316.2 | 0.4779..357.1 | 187.5..317.8 | 272.5..272.8 | 263.5..274.5 | 24.34..254.2 | 92.81..188.6 | 18.87..182.8 | 92.81..183.4 | 92.81..182.8 |
| `native.behaviour.change_formation_spacing` | False | False | False | False | False | False | False | False | False | False |
| `native.behaviour.defend` | False, True | False | False | False | False | False, True | False | False | False | False |
| `native.behaviour.fire_at_will` | False | False | False, True | False | False, True | False | False | False, True | False | False, True |
| `native.behaviour.skirmish` | False | False | False | False | False | False | False | False | False | False |
| `native.current_target` |  | , b_spears | , b_spears |  |  |  | , a_spears | , a_spears |  |  |
| `native.fast_speed` | 3.4 | 3 | 3.3 | 3 | 3.6 | 3.4 | 3 | 3.3 | 3 | 3.6 |
| `native.fatigue_state` | threshold_fresh | threshold_active, threshold_fresh, threshold_winded | threshold_active, threshold_fresh | threshold_fresh | threshold_fresh | threshold_fresh | threshold_active, threshold_fresh, threshold_winded | threshold_fresh | threshold_fresh | threshold_fresh |
| `native.initial_number_of_men` | 1 | 120 | 90 | 120 | 60 | 1 | 120 | 90 | 120 | 60 |
| `native.is_controllable` | True | True | True | True | True | True | True | True | True | True |
| `native.is_hidden` | False | False | False | False, True | False, True | False | False | False | False | False, True |
| `native.is_idle` | False, True | False, True | False, True | True | False, True | False, True | False, True | False, True | True | True |
| `native.is_in_melee` | False | False, True | False, True | False | False | False | False, True | False | False | False |
| `native.is_leaving_battle` | False | False | False | False | False | False | False | False | False | False |
| `native.is_left_flank_threatened` | False, True | False | False, True | False | False | False | False, True | False, True | False | False |
| `native.is_moving` | False, True | False, True | False, True | False | False | False, True | False, True | False, True | False | False |
| `native.is_moving_fast` | False, True | False, True | False, True | False | False | False, True | False, True | False, True | False | False |
| `native.is_rear_flank_threatened` | False, True | False, True | False, True | False | False | False | False, True | False, True | False | False |
| `native.is_right_flank_threatened` | False | False, True | False, True | False | False | False | False | False, True | False | False |
| `native.is_routing` | False | False | False | False | False | False | False | False | False | False |
| `native.is_script_controlled` | False, True | False, True | False, True | False, True | False, True | False, True | False, True | False, True | False, True | False, True |
| `native.is_shattered` | False | False | False | False | False | False | False | False | False | False |
| `native.is_under_missile_attack` | False | False | False | False | False | False | False, True | False | False | False |
| `native.is_wavering` | False | False | False | False | False | False | False | False | False | False |
| `native.left_flank_threat` | , b_archers |  | , b_archers, b_spears |  |  |  | , a_archers | , a_archers, a_general |  |  |
| `native.number_of_men_alive` | 1 | 109..120 | 90 | 120 | 60 | 1 | 70..120 | 90 | 120 | 60 |
| `native.ordered_bearing` | 270..316.2 | 269.6..340.1 | 220.5..272.7 | 270..272.8 | 270..272.8 | 44.21..213.1 | 90..190.6 | 90..182.8 | 90..182.8 | 90..182.8 |
| `native.ordered_position` |  |  |  |  |  |  |  |  |  |  |
| `native.ordered_width` | 1.4 | 29.42 | 29.2 | 29.42 | 28.8 | 1.4 | 29.42 | 29.2 | 29.42 | 28.8 |
| `native.position` |  |  |  |  |  |  |  |  |  |  |
| `native.rear_threat` | , b_archers | , b_archers | , b_archers |  |  |  | , a_archers | , a_archers, a_general |  |  |
| `native.right_flank_threat` |  | , b_archers | , b_archers |  |  |  |  | , a_archers, a_spears |  |  |
| `native.slow_speed` | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 |
| `native.starting_ammo` | 0 | 0 | 1800 | 0 | 1320 | 0 | 0 | 1800 | 0 | 1320 |
| `native.unary_hitpoints` | 1 | 0.8077..1 | 0.9097..1 | 1 | 1 | 1 | 0.3441..1 | 1 | 1 | 1 |
| `native.unary_of_men_alive` | 1 | 0.9083..1 | 1 | 1 | 1 | 1 | 0.5833..1 | 1 | 1 | 1 |
