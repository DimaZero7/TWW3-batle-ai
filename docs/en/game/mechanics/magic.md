# Magic and the Winds of Magic

[← Back](README.md) · [Documentation](../../README.md) › [Game knowledge](../README.md) › [Battle mechanics](README.md) › Magic · [Русский](../../../ru/game/mechanics/magic.md)

The battle side of magic: the Winds pool, casting, overcast and miscast, spell types, resistances.
Conventions: [index](README.md). Ours: no casters in our armies yet; not modelled.

## The Winds pool

- **Reserve cap 100** (`pool_max_amount`); custom battles start with a reserve of 80, of which
  0.15 (~12) is usable at once; a pre-battle gamble moves it by −10…+10. ·
  [twwstats Winds params][tws-w] (`_kv_winds_of_magic_params`, 9.0.0) · high (values), medium (meaning).
- **Flow.** Casting spends the usable pool, which refills from the reserve; refilling is faster with
  a fuller reserve and stops when the reserve is empty; the pool is shared by all casters. Base
  `restored_points_base` 0.1 (likely per second), estimate multiplier 1.33; players saw ~+1 every
  2 s. In WH3 the rate stays constant through the battle and is faster than WH2. · twwstats,
  [fandom Winds of Magic][fw-wom], [Steam][glean] · high (values), low/medium (rates).
- **Campaign reserve** by wind strength: calm 40, blowing 50, strong 50 (+5 a turn); Channelling
  stance +15 a turn. Arcane Conduit (WH3 passive): +40 % recharge. Glean Magic: 8 Winds, +80 %
  recharge and +0.1 reserve/s for 23 s. · fandom, Steam · medium/low.
- **Storm of Magic** (Chaos Realms): infinite reserve, only the draw rate limits. · [PCGamesN][storm] · high.

## Casting

- **Overcast** costs more for a stronger effect; **miscast** can happen only on an overcast (spell
  tables show 50 %); some enemies (Empire Arch Lectors, Warrior Priests) impose a miscast chance on
  all your casters. Miscast explosion: 288 base + 300 AP magic damage to the caster. · fandom,
  Steam · medium/low.
- **Spell Mastery** sets intensity 100–200 % (damage and buff/debuff values, not duration, range,
  cost, cooldown); every caster has it since 5.0. · fandom · medium/high.
- **Bound spells** cost no Winds, can't miscast or overcast, have limited uses. Summons cost Winds
  and the summoned units usually lose HP over time. · fandom · medium.
- **Casters reveal themselves** when casting (since 4.1.0) unless they have Snipe. · fandom 4.1.0 · high.
- **AI mages** switch to melee when their mana is more than 300 s of recharge from full. · twwstats · high.

## Spell types

- **Magic missiles** can miss, are blocked by terrain, can hit flyers; **direct damage** hits the
  target unit (flyers too); **winds** move in a line, **breaths** a cone; **bombardments** fall on an
  area; **vortexes** stay for a time and usually wander. Most damaging spells hit friends too. ·
  fandom · medium.
- **Examples (WH3):** Purple Sun 18 Winds, vortex, 15 dmg/s per entity (73 % AP), 12 m, 2 m/s, 11 s,
  range 150, cooldown 54. Comet of Casandora 13 Winds, 1728 damage (50 % AP) + explosive, range 200,
  7 s delay. Wind of Death 15 Winds, 24 dmg/s (100 % AP), 6 m wide, 20 m/s, grounded units only. ·
  fandom lore pages · medium. Per-spell data: `battle_vortexs`, `special_ability_phases`.
- **Direct-damage spells scale with unit size** (×0.25 / 0.5 / 0.75 / 1, minimum 0.5 — `direct_damage_*`
  in `_kv_unit_ability_scaling_rules`); 8.0 fixed small-size scaling. · [twwstats scaling][tws-s] · high.

## Resistances

- WH3 turned magic resistance into **spell resistance**: magical attacks bypass physical resistance
  and only spell resistance reduces spell damage. Ward save applies to everything; all resistances
  add up to a 90 % cap (`ward_save_max_value` 90). Fire adds to another channel and acts only via fire
  resistance/weakness. · [CA Discord Q&A via fandom][fw-dc], twwstats · high/medium.

## Faction magic mechanics

- Daemon armies get bonuses at a high reserve (Tzeentch faster barrier recharge, Khorne spell
  resistance, Nurgle melee defence, Slaanesh speed) and penalties at a low one. · CA blog via PCGamesN · high.

[tws-w]: https://twwstats.com/kv/winds_of_magic_params
[tws-s]: https://twwstats.com/kv/unit_ability_scaling_rules
[fw-wom]: https://totalwarwarhammer.fandom.com/wiki/Winds_of_Magic
[fw-dc]: https://totalwarwarhammer.fandom.com/wiki/Damage_channel
[glean]: https://steamcommunity.com/app/1142710/discussions/0/3826415019288347326
[storm]: https://www.pcgamesn.com/total-war-warhammer-3/storm-of-magic-reserve
