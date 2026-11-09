## Skavenslaves, Clanrats with shields, Night Runners

The Skaven wave, added like the Empire's: the cheapest chaff, a shielded line and fast long-ranged
slingers. Passports from the database; no cards from battle yet (list
`config/roster/capture_skv_wave.json`, see "Recordings wanted" below). We already had the clanrat
**spearmen without shields** (`wh2_main_skv_inf_clanrat_spearmen_0`); the variant added is the
**Clanrats with sword and shield** (`wh2_main_skv_inf_clanrats_1`, 350): a shield against the Empire's
arrows and pistols. The plain sword clanrats (`_0`, 300, defence 18, no shield) differ from it only by
the shield and 4 defence. The Night Runners added are the **slings** (`_1`); the `_0` throw stars
(70 m, direct fire, fire whilst moving).

| | Skavenslaves | Clanrats with shields | Night Runners (slings) |
|---|---:|---:|---:|
| Key | `wh2_main_skv_inf_skavenslaves_0` | `wh2_main_skv_inf_clanrats_1` | `wh2_main_skv_inf_night_runners_1` |
| Men; health of a man / unit | 180; 50 / 9000 | 160; 60 / 9600 | 120; 56 / 6720 |
| Mass; run / charge, m/s | 90; 4.2 / 4.8 | 100; 4.2 / 4.8 | 90; **5.4 / 6.0** (acceleration 5) |
| Attack / defence | 12 / 10 | 24 / 22 | 24 / 14 |
| Charge | 5 | 10 | 10 |
| Damage: normal + piercing | 14 + 4 (sword) | 20 + 6 (sword) | 20 + 6 (sword) |
| Armour; shield | 0; none | 25; **35 %** (wood) | 10; none |
| Leadership | 35 | 45 | 48 |
| Physical resistance | — | — | **20 %** |
| Attributes | **expendable**, hide_forest | hide_forest | **guerrilla_deploy** (vanguard), hide_forest; can skirmish |
| Innate abilities | Strength in Numbers, Scurry Away! | the same | the same |
| Shooting | — | — | sling, arcing (`dual_low_fixed`), **140 m**, 22 shots, 10 + 1, reload 8 s, calibration 4.0 m at 85 m |
| Cost (multiplayer) | **125** | 350 | 450 |

### What each feature does, and where it comes from

| Unit | Feature | What it does | Source | Simulator | Network |
|---|---|---|---|---|---|
| all three | Strength in Numbers | +6 leadership, +8 melee defence, speed ×0.9 while health is above 50 % of the start | DB: the ability passport (`stat_morale` +6, `stat_melee_defence` +8, `scalar_speed` ×0.9; off `health_below_50%_base`); knowledge base | innate effect (below); the Skaven's fitted +6 start reserve (`morale.faction_bonus`) was this, so it is 0 now | effect pair (owned, on); ability slot |
| all three | Scurry Away! | speed ×1.1 while wavering or routing | DB (`scalar_speed` ×1.1; off `morale_is_higher_than_wavering`); knowledge base | innate effect: a Skaven rout runs ×1.1 (measured ×1.09, below) | effect pair; ability slot |
| Skavenslaves | Expendable | their rout scares only other expendable units | DB attribute `expendable`; knowledge base | innate effect (rule `expendable`) | effect pair; passport attribute |
| Skavenslaves | Cheapest chaff (125), defence 10, no armour | soak and hold for the cost | passport | from the numbers | passport |
| Clanrats with shields | Shield 35 % | blocks 35 % of arrows, slings and pistol shots from within 60° of the front | DB `unit_shield_types` | the shield rule ([shields](#shields)) | passport `shield` |
| Night Runners | Fast (run 5.4, acceleration 5) | outpace every infantry in both pools | passport | from the passport | passport |
| Night Runners | Long-ranged sling (140 m) with a tighter spread (4.0 m at 85 m against the slave slingers' 4.8 m) | outranges the archers (130 m) | DB `projectiles` | the measured sling hit rate 0.47 at the edge of range (the spread at 140 m about the slaves' at 120 m); reload 11.5 × 8 / 9 = 10.2 s (`sim.json` missile.reload_ref_s: an estimate, not measured) | passport (`spread`, range, reload) |
| Night Runners | Physical resistance 20 % | every melee hit and every arrow / sling takes 20 % less | DB `land_units.damage_mod_physical` | the per-hit rule (melee; missiles: resistances are those of the passport) | passport |
| Night Runners | Vanguard deployment | may deploy ahead of the deployment zone | DB attribute `guerrilla_deploy` | schema only: placements come from the army generator | effect pair; passport attribute |
| Night Runners | Skirmish | can keep away from melee by itself | DB `land_units.can_skirmish` | not modelled (no skirmish mode) | passport `can_skirmish` |
| Night Runners | No Stalk | the WH3 database gives them no `stalk` (the sling Night Runners of older patches had it) | DB | — | — |

The bridge needs nothing for them: nothing is orderable (their abilities are innate).

### Recordings wanted

1. **Cards**: `python -m tools.build roster-capture --capture config/roster/capture_skv_wave.json`,
   launch, `python -m tools.roster update …/events.jsonl`, then `py -3.14 -m tools.nn.units` (the card check).
2. **Night Runners' reload and hit rate**: the archer range (`--range-mode damage`) with the Night Runners in
   the archers' place against standing Empire spearmen at 100, 120 and 140 m, 3 runs each.
3. **Strength in Numbers and Scurry Away! on the bridge**: any Skaven battle with the bridge's
   `ActiveEffectList` every second: when each switches (health 50 %, wavering).

## Innate effects

Every attribute and every passive or game-fired ability of a unit is an **innate effect**, one
catalogue for all: `config/nn/effects.json`, written by `python -m tools.nn.effects` from the unit
and ability passports (both from the database). Which unit owns which effect comes from the database
rows (the passport's `attributes`, and its `abilities` that are passive or `auto`), never from a hand
list. The simulator lays them on by one mechanism (`tools/nn/sim/effects.py`,
[simulator](simulator.md#innate-effects)); the network sees each as a pair of inputs, owned and on
now ([model](model.md#innate-effects)). An attribute's rule is the engine's (the database gives only
its key): `ATTRIBUTES` in `tools/nn/effects.py` says what it does, with its numbers and source.

For each effect the file holds: `kind` (attribute, passive, timed), `stats` (stat, how, value, on
self or friends, the simulator's stat), `rules` (flags such as `unbreakable`), `range_m` (an aura),
`needs` and `off_when` (the conditions, as the simulator's predicates), `timed` (active and recharge
time, when the game fires it), `when` in words, `source` (tables and row), `kb` (knowledge base),
`modelled` and `why` (what the simulator lacks). `order` is append-only: an effect keeps its place
when the file is rebuilt, so the network's inputs keep their meaning.

The game's condition flags and what the simulator checks for them (`_fields.predicates` in the file):

| Database flag | Predicate | The simulator |
|---|---|---|
| `health_below_50%_base` | hp_below_half | health < 50 % of the start |
| `health_below_25%` | hp_below_quarter | health < 25 % |
| `morale_is_higher_than_wavering` | not_wavering | not wavering and not routing |
| `morale_is_lower_than_half_of_base_morale` | morale_below_half | morale points < half of leadership |
| `out_of_melee`, `engaged_in_melee` | out_of_melee, in_melee | in melee with a standing enemy |
| `losing_melee_combat` | losing_melee | in melee and HP taken ≥ 1.5 × dealt recently (the morale rule's ratio; ours) |

The 14 effects of our 17 units:

| Effect | Kind | What it does | When | Owners | Simulator |
|---|---|---|---|---|---|
| Unbreakable (`unbreakable`) | attribute | never loses leadership, never routs | always | Flagellants | modelled: morale points never below leadership |
| Expendable (`expendable`) | attribute | its rout scares only other expendables | always | slave spearmen, slingers, skavenslaves | modelled: not counted among routing friends |
| Encourage (`encourages`) | attribute | +4 leadership to friends within 70 m, to 0 at 105 m | always | the two lords | modelled: the lord's aura |
| Charge Reflection (`charge_reflection`) | attribute | braced, meets a frontal charge as a charge | always | spearmen, halberdiers, clanrat spearmen, stormvermin | modelled (bracing) |
| Fire Whilst Moving (`mounted_fire_move`) | attribute | shoots on the move | always | militia | modelled |
| Charge Defence vs. Large (`charge_defense_vs_large`) | attribute | braced, cancels a large charger's bonus | always | the spear units | schema only: no large units in our pools |
| Vanguard (`guerrilla_deploy`) | attribute | deploys ahead | always | militia, Night Runners | schema only: placements are the generator's |
| Hide (forest) (`hide_forest`) | attribute | hidden in woods | always | all 17 units | schema only: no woods, no hiding |
| Strength in Numbers | passive | +6 leadership, +8 melee defence, speed ×0.9 | health ≥ 50 % of the start | Skaven infantry (7) | modelled |
| Scurry Away! | passive | speed ×1.1 | wavering or routing | Skaven infantry and the Warlord (8) | modelled |
| Single Entity | passive | speed ×0.9, melee damage and AP ×0.8 | health < 25 % (its recharge context `health_below_25%`) | the two lords | modelled; the reading of the context is ours: the cards (full health) show the full damage |
| Hold the Line! | passive aura | +5 melee defence, +4 leadership to self and friends within 35 m | always (a standing lord) | the General | modelled |
| Frenzy | passive | +10 melee attack, ×1.1 damage, AP and charge; Immune to Psychology | morale ≥ half of leadership | Flagellants | modelled (Immune to Psychology: nothing to act on) |
| Strength of the Penitent | timed | 20 s of +14 melee defence, +15 % physical resistance; ready 3 s after | fired by the game when losing the melee; ends out of melee | Flagellants | modelled |

Stubborn and Hatred are not in the WH3 database as unit attributes or passives (WH2 had them);
Perfect Vigour is the attribute `fatigue_immune` (in `ATTRIBUTES`, modelled: no fatigue) for when a
unit with it comes. Effect bundles (`effect_bundles`, `effect_bonus_value_*`) are campaign effects
(skills, technologies, `battle_context_unit_attribute_junctions`): custom battles have none.

**Measured** (02.10.2026, all fair recordings with unit keys; speed over 1 s steps): routing
Empire units run at 0.865 of their run, routing Skaven below half health at 0.945 (×1.09: Scurry
Away!) and above half health at 0.866 (×1.1 × 0.9); running in order, steady, above half health:
Empire 0.97, Skaven 0.88 (×0.91: Strength in Numbers). Crossing 50 % health, Skaven units drop
2.3 points more morale in the next 3 s than Empire units (7.7 against 5.5; at 40 % and 60 % both
the same): Strength in Numbers' +6 switching off. So `morale.rout_speed` is 0.86 (the Empire's) and
the Skaven's ×1.1 comes from Scurry Away!.

