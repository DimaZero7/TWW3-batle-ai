# Flying units

[← Back](README.md) · [Documentation](../../README.md) › [Game knowledge](../README.md) › [Battle mechanics](README.md) › Flying units · [Русский](../../../ru/game/mechanics/flying.md)

Flight, landing, fighting from the air, flyers against missiles and the AI. Conventions:
[index](README.md). Ours: not modelled.

- **Flyers ignore troops and terrain while airborne.** Most must land to melee ground units
  (Pegasus Knights, Vargheists, Griffons); a few attack from the air (Terradons, Gyrobombers,
  dragon breath). Two flying units can melee each other in the air. · [fandom Units][fw-units] · high.
- **Ground melee can't hit an airborne flyer.** 4.1.0 reduced AI melee units idling underneath
  flying enemies. · fandom 4.1.0 · high.
- **Airborne flyers are exposed to missiles** (no screening by terrain or units); some ranged
  units can't target flyers at all. Magic missiles and direct-damage spells hit flyers; some wind
  spells hit only grounded units. · fandom · medium.
- **Toggle flight (WH3).** "Can Fly" units can land and walk; "Always Flying" units (some Dwarf and
  Cathay units) never land and, since 5.0, can't melee gates/walls/barricades. · [fandom Can Fly][fw-cf],
  [fandom Always Flying][fw-af] · high.
- **Speed.** Fly speed is `battle_entities.fly_speed` (card ×10). Dragons: 7.0 raised mass 4000 →
  5000 and fly speed 9 → 9.5 (Sun Dragon 9.5 → 10, Star Dragon 8 → 8.5, mass 5900), turn speed
  100–120 → 120–130. · [7.0 blog][p70] · high.
- **Charging from the air.** 5.3 fixed flyers slowing down on an air-to-ground charge (charge speed
  raised above flight speed). Fliers sit about 16 m up, which gives them ≈ +12 % missile damage
  from height when shooting down. · fandom 5.3, [CA elevation blog][elev] · high.
- **Take-off/landing** is slow ("airborne units are slow to respond when moving to the ground",
  1.1.0); no times or fly heights are published. · fandom 1.1.0 · medium.
- **Limits.** Flyers can't capture points, can't hide in forests (seen above the trees). An army of
  only flyers gets a leadership penalty over time (WH2 made it milder). · fandom · high/medium.
- **Bug (4.2.3):** WH3-added flyers don't slow to the group's slowest unit. · CA bug tracker · medium.
- **AI.** 7.2: the AI no longer splits its army to chase groups of only flyers. Players use flyers to
  hunt artillery/missiles (vanguard from the back corners), as decoys (the AI targets the closest
  unit), to spoil charges from behind. · [7.2][p72], [Steam guide][fguide] · high/low.
- Script API: `unit:can_fly()`, `unit:is_currently_flying()`. · [CA script docs][chv] · high.

[fw-units]: https://totalwarwarhammer.fandom.com/wiki/Units
[fw-cf]: https://totalwarwarhammer.fandom.com/wiki/Can_Fly
[fw-af]: https://totalwarwarhammer.fandom.com/wiki/Always_Flying
[p70]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/90-total-war-warhammer-iii-update-7-0-0
[p72]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/95-total-war-warhammer-iii-patch-7-2-release-notes
[elev]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/11-feature-focus-1-elevation
[fguide]: https://steamcommunity.com/sharedfiles/filedetails/?id=2850811697
[chv]: https://chadvandy.github.io/tw_modding_resources/WH3/battle/battle_unit.html
