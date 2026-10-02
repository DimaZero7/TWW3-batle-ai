# Terrain, weather, visibility

[← Back](README.md) · [Documentation](../../README.md) › [Game knowledge](../README.md) › [Battle mechanics](README.md) › Terrain · [Русский](../../../ru/game/mechanics/terrain.md)

Height, slopes, forests, water, line of sight, hiding, weather and time of day. Conventions:
[index](README.md). Ours: the simulator's map is flat and empty; terrain is not modelled. What the
game exposes about the map: [map](../map/README.md).

## Height

- **Per entity.** Height changes damage per attacker–target pair. · [CA elevation blog][elev] · high.
- **Missiles:** damage changes linearly with the height difference, up to ±30 % at 40 m
  (`missile_height_damage_modifier_max_coefficient` 0.3, `_max_difference` 40). · CA · high.
- **Melee:** the same ±30 %, reached at a 1 m difference (`melee_height_damage_modifier_*` 0.3 / 1 m). · CA · high.
  Nothing else (hit chance, range) is said to change with height.
- **Speed:** downhill up to +50 %; uphill slower (−22 % in CA's example). · CA · high.
- **Fatigue:** uphill movement costs +50 / +100 / +150 % at gradients > 0.05 / 0.1 / 0.2. · CA · high.
- **Morale:** +10 for being uphill of all enemies (`ume_encouraged_on_the_hill`). · [twwstats morale][tws-m] (WH2 description) · high (value).
- **Charges** downhill hit harder (speed → collision damage). Strider ignores the height penalty
  when attacking, not when attacked. · CA · high.
- The AI ignores hills more than 80 m away as defensive features. · twwstats · high.

## Ground

- **Forests:** almost all units can hide in them while walking or standing (running reveals);
  trees physically block missiles (players: most fire absorbed, a few metres of grace at the edge;
  Wood Elf missiles reportedly ignore trees for ~40 m); large units without Strider ~80 % melee
  attack and speed. Woodsman lets a unit pass trees. · fandom, [Steam][trees], [Steam][forest] · medium/low.
- **Shallow water:** small units without Aquatic −20 % speed and −20 % melee attack; Aquatic units
  +20 % melee attack and defence. · fandom Aquatic · medium.
- **Steep slopes are impassable**; 9.0 removed a hidden speed penalty on "sharp stones". · fandom, 9.0 notes · medium/high.
- **Cover:** `missile_target_in_cover_penalty` 0.2 off the hit chance in cover. · twwstats · high (value).

## Visibility and hiding

- A unit is visible only if some enemy has line of sight to it, within a maximum range; hills and
  terrain block it; flyers help spotting; very large units and flyers are seen above trees. ·
  fandom Visibility · medium.
- Unspottable units are revealed at 20 m; a stalking unit is spotted at ~170 m (a wiki note). In a
  night battle the attacker's hiding is ×0.75. Spotting values are said to be in
  `spotting_and_hiding_values_tables`. · twwstats, fandom, twcenter · high/low.
- Casters reveal themselves when casting (4.1.0); fire at will can give hidden units away. · high/low.
  - Ours: `vis` is always true in the simulator; our game measurements: [visibility](../units/visibility.md).

## Weather and time

- **Rain and snow** slow fatigue recovery while idle (−10 instead of −18). No documented effect on
  accuracy or range in WH3. · DB, fandom Fatigue · high / low.
- **Night battles:** keys exist (`ume_concerned_night_battle_unprepared` −5, attacker hiding ×0.75),
  but WH3 seems to have no day/night mechanic. · DB, Steam · low.
- Region presets (ash, blood rain, spores) are visual. · Workshop · medium.

## Battle types (for reference)

Land, Ambush, Chokepoint, Subterranean, Island, Minor Settlement, Siege, Survival, Domination,
Conquest. In an ambush the ambushed side can't deploy, starts in column and gets "Discombobulated!"
for 120 s (−15 armour, −5 melee defence). · fandom Battle map, Ambush battle · high/medium.

[elev]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/11-feature-focus-1-elevation
[tws-m]: https://twwstats.com/kv/morale
[trees]: https://steamcommunity.com/app/1142710/discussions/0/591767669138558048/
[forest]: https://steamcommunity.com/app/594570/discussions/0/1744479063986685346/
