# Battle rules: size, time, victory

[← Back](README.md) · [Documentation](../../README.md) › [Game knowledge](../README.md) › [Battle mechanics](README.md) › Battle rules · [Русский](../../../ru/game/mechanics/battle-rules.md)

Unit size, army caps, the time limit, how a battle is won, the balance of power and difficulty.
Conventions: [index](README.md).

## Unit size

- **Models per unit** scale with the setting: Small 25 %, Medium 50 %, Large 75 %, Ultra 100 %;
  total HP shrinks with it. · [fandom Units][fw-units] · high.
- **Single entities** (lords, heroes, monsters) scale health and damage the same way, and so do
  direct-damage spells and tower damage (CA's example: a 500-damage tower must not one-shot a
  500-HP unit on Small). WH3 is balanced around Ultra (WH2: Large). · [GameWatcher][gw],
  [WH3 kv guide][g3] · high/medium. Unit-size factors also appear in the Waaagh! threshold and
  battle currency (`_kv_rules`).
  - Ours: our recordings use Ultra numbers (spearmen 120, clanrats 160, slaves 180).

## Army caps and time

- **20 units per army**; with reinforcements about 40 per side on the field; extra units wait. ·
  fandom, Steam · high/medium.
- **Time limit:** 20, 40 or 60 minutes or unlimited (settings). **When time runs out, or on a tie,
  the defender wins.** · [fandom Battles][fw-battles], CA bug report 4789, Steam · medium-high.
  - Ours: the simulator's limit is 60 minutes and the attacker loses on time (`battle_limit_s`
    3600). Agreement; our arena recordings use their own 900 s cut.
- **Playable area** `scaled_playable_area_size_min/max` 1024 (WH2 key). · twwstats · medium.
  - Ours: routers leave the map at |x|, |z| = 1020 m (`map_half_m`). Agreement.

## Victory

- **Win by killing or routing every enemy unit**; in sieges also by holding the victory point. ·
  fandom Battles · medium.
- **Army losses:** when an army is beaten as a whole (enemy current strength ≥ 2.6 × own and own ≤
  0.22 of the start), every unit gets −120 morale and routs, except unbreakable ones. · [twwstats
  morale][tws-m] (WH2 descriptions), [fandom Leadership][fw-lead] · high.
  - Ours: not modelled — a simulated side fights to its last standing unit. **Gap**: the game ends
    battles earlier; this may be part of why our simulated battles end later than the recordings
    (72 % still running when the recording ends, `simulator.md`).
- **Balance of power** compares the sides' combat potential (HP, ammunition, leadership, Winds
  reserve count; spells/abilities count as a fixed amount). · Steam, CA forum · low-medium.
- **Battle result points:** decisive win needs ≥ 4950, loser ≤ 3000 (`battle_result_*`). · twwstats · high (values).

## Difficulty

See [battle AI](battle-ai.md#difficulty-what-changes-in-battle). At Normal the player gets 0
morale and the AI dodges only direct-damage spells and shoots the first enemy it meets.

[fw-units]: https://totalwarwarhammer.fandom.com/wiki/Units
[gw]: https://www.gamewatcher.com/news/total-war-warhammer-3-lords-monsters-debuff-small-unit-size-scaling
[g3]: https://steamcommunity.com/sharedfiles/filedetails/?id=2776861563
[fw-battles]: https://totalwarwarhammer.fandom.com/wiki/Battles
[tws-m]: https://twwstats.com/kv/morale
[fw-lead]: https://totalwarwarhammer.fandom.com/wiki/Leadership
