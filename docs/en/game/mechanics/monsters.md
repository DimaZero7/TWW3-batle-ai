# Monsters and large units

[← Back](README.md) · [Documentation](../../README.md) › [Game knowledge](../README.md) › [Battle mechanics](README.md) › Monsters · [Русский](../../../ru/game/mechanics/monsters.md)

Single-entity and small-entity monsters: mass, knockdowns, splash, fear and terror. Lords on foot
are in [lords and heroes](single-entities.md). Conventions: [index](README.md). Ours: not modelled.

## Mass and knockdowns

- **Mass decides who moves.** A charger loses momentum with every entity it hits, so heavy units
  push further in. Knock chances per hit: knockback 0.1 × mass ratio, knockdown 0.05 × ratio (×2
  scaled by the charge factor when charging); none below ratios 0.35 / 0.5; impact interrupts at
  speed changes of 4 (back), 7 (down), 10.5 (flying) m/s. · [twwstats][tws] (WH2 descriptions,
  WH3 values) · high.
- **Masses:** monstrous infantry ~1700–1900; Doomwheel 2000; Tamurkhan 2500 (5.1); dragons
  4000 → 5000, Star Dragon 5900 (7.0); Dread Saurian ~12000. · [5.1.0][p510], [5.3.0][p530],
  [7.0][p70] · high.
- **Knock-ignore chance** per entity (`battle_entities`): 20–80 %; 6.2.2 raised Arachnarok 15 → 85 %
  and set ≥ 40 % for every foot character. · [6.2.2 hotfix][h622] · high.
- **Monsters stall** against braced or dense infantry when they can't reach the speed needed to
  knock it down; a single stray soldier can pin one ("unit glue"). · [Steam][stuck] · medium.
- **Size class drives knockback**: War Sleds moved from very large to large (5.3) to bring their
  knockback in line. Ogres get knocked down like other monstrous infantry again (5.1). · high.

## Splash

- Damage of one attack is divided among the entities hit (each rolls to hit), up to
  `max_splash_targets`, only on targets no larger than the splash target size (small, medium,
  large, very large). Before 6.0: monsters 10–12; 5.1/5.3 cut many (Great Unclean One 12 → 5,
  Rogue Idol 12 → 10 → 8, Dread Saurian 12 → 8, Mutant Rat Ogre 8 → 4). 6.0: targets = weapon
  strength / 100 for single entities. · [splash thread][splash], [5.1.0][p510], [5.3.0][p530] · high/medium.

## Who can hit a monster

- No number is published; only models touching its large hitbox can strike; surrounding adds
  attackers and the flank/rear defence multipliers (×0.6 / ×0.3) per attacker. · [Steam][surr] · low.
- Missiles: bigger hitboxes catch more shots; there is no hit-chance bonus vs large. Many large
  single entities have +15 % missile resistance. · medium/low.
- Anti-large units add bonus vs large to attack and damage. · [cavalry](cavalry.md) · medium.

## Fear and terror

- **Fear:** −8 leadership to enemies within 20 m (`fear_effect_range`; the wiki says 30 m —
  trust the database); does not stack; fear-causers are immune. · [twwstats morale][tws-m],
  [fandom Causes Fear][fw-fear] · high (values).
- **Terror:** on a melee hit, an enemy within 5 m whose morale is ≤ 13 points routs for 14 s;
  immune to the same enemy's terror for 85 s; 4 terror routs shatter. Every terror-causer also
  causes fear (WH3). Unbreakable, undead, daemons are immune. · [twwstats morale][tws-m],
  [fandom Causes Terror][fw-ter], [Steam][terror] · high/medium.
- **Health and size.** Single-entity monster health and damage scale with unit size (25–100 %);
  "wounded" state at low HP. · [GameWatcher][gw] · high.

[tws]: https://twwstats.com/kv/rules
[tws-m]: https://twwstats.com/kv/morale
[p510]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/23-total-war-warhammer-iii-patch-5-1-0
[p530]: https://community.creative-assembly.com/total-war/total-war-warhammer/forums/7-total-war-warhammer/threads/7586-total-war-warhammer-iii-patch-5-3-0-battle-balance-details
[p70]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/90-total-war-warhammer-iii-update-7-0-0
[h622]: https://community.creative-assembly.com/total-war/total-war-warhammer/forums/7-patch-notes-amp-announcements/threads/10367-total-war-warhammer-iii-hotfix-6-2-2
[stuck]: https://steamcommunity.com/app/1142710/discussions/0/664957733316912540/
[splash]: https://community.creative-assembly.com/total-war/total-war-warhammer/forums/10-battles/threads/6818-dividing-damage-among-models-during-splash-attacks-is-a-bad-design-decision
[surr]: https://steamcommunity.com/app/594570/discussions/0/6980058308319576309/
[fw-fear]: https://totalwarwarhammer.fandom.com/wiki/Causes_Fear
[fw-ter]: https://totalwarwarhammer.fandom.com/wiki/Causes_Terror
[terror]: https://steamcommunity.com/app/1142710/discussions/0/688619343242660842/
[gw]: https://www.gamewatcher.com/news/total-war-warhammer-3-lords-monsters-debuff-small-unit-size-scaling
