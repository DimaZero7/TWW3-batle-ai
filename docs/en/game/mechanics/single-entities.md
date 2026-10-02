# Lords and heroes (single entities)

[← Back](README.md) · [Documentation](../../README.md) › [Game knowledge](../README.md) › [Battle mechanics](README.md) › Lords and heroes · [Русский](../../../ru/game/mechanics/single-entities.md)

A lord or hero on foot is one entity with a lot of health. How infantry fights him, how he fights
infantry, missiles at him, abilities. Monsters are in [monsters](monsters.md). Conventions:
[index](README.md). Our measurements: [lord swarm](../units/lord-swarm.md).

## Infantry against a lord

- **How many can strike him.** No CA number. A WH1-era guide says ~9 small entities can hit a
  surrounded small entity, ~12 horse-sized, more for bigger ones; only models touching his hitbox
  attack. · [WH1 kv guide][g1] · low; [Steam thread][surr] · low.
  - Ours: `contact.lord_max_attackers` 9 was fitted on the lord swarm probe (4–5 soldiers within
    2.5 m, 7–10 within 3 m; HP/s the same for 1–4 units). Agreement with the old guide's 9.
- **Flank/rear rule per entity.** The flank/rear defence multiplier is decided per attacking model
  and ends when the struck entity faces its attacker. A lone entity turns at once. · [CA damage blog][dmg] · high.
  - Ours: the probe found no extra loss from back or flank (`contact.lord_direction` 0). Agreement.
- **Knockdowns on a lord.** Since 4.1.0, when a unit knocks a single entity down outside its
  formation it moves over him and waits; inside, it keeps the surround. 6.2.2 gave every foot
  character at least 40 % chance to ignore knockback. Knock chance = 0.1 (back) / 0.05 (down) ×
  attacker/defender mass, none below a ratio of 0.35 / 0.5. · fandom 4.1.0, [6.2.2 hotfix][h622],
  [twwstats][tws] · high.
  - Ours: infantry mass 100 vs General 600 → ratio 0.17: infantry can't knock him down by the
    rules. Not modelled, and by this rule not needed for our units.
- **The general can't die early.** `general_auto_survive_threshold` 0.5: while more than 50 % of the
  entities of the general's unit are alive, the general can't be killed (matters for lords with a
  bodyguard/crew). · [twwstats][tws] (WH2 description) · high (value), medium (WH3).
- **"Wounded" state.** WH3 single entities get stat penalties at low health; 2.0 made it harder to
  heal back from very low HP. Thresholds not found. · [GameWatcher][gw] · high (exists).
  - Ours: the database's passive Single Entity (both our lords) has speed ×0.9 and melee damage and
    AP ×0.8 with the recharge context `health_below_25%` (`config/nn/effects.json`); the recordings
    show no speed drop below 25 % (lords running out of melee: 0.84–0.85 of their run in every
    health band), so the simulator leaves it out (`sim.json` effects.off); the network sees it.

## A lord against infantry

- **Splash.** A lord's blow hits up to its `max_splash_targets` (since 6.0.0 = weapon strength /
  100, at least 1, single entities only); the damage is divided among the targets and each rolls
  its own hit. Lords/heroes had ~6 before 6.0. · [melee](melee.md#splash) · medium-high.
  - Ours: General's passport splash 4 (430 / 100) matches the measured kills: 1–4 men per event,
    0.25 events/s. The simulator gives each target the full hit (capped at a man's HP); same kills
    here.
- **Executioner** kills any target below 20 % HP (`execute_threshold` 0.2). · [twwstats][tws], fandom · high.

## Missiles at a lord

- **No "Look out, sir!"** in Warhammer. Players put foot heroes inside infantry blocks so the
  bodies stop shots physically. · [Steam][los] · low.
- **Shields** count only from the front 120°. The Empire General's shield is 55 % since 5.0.0
  (was 35 %). · [5.0.0 blog][p500] · high.
  - Ours: a lord loses 0.43 of the unit rule per projectile aimed at him out of melee
    (`missile.single_entity_factor`), measured on ~33k shots. Consistent with physical projectiles
    that miss a single small body more often; no public number to compare.
- **Big single entities have bigger hitboxes** and get hit more; there is no hit-chance bonus
  against "large". · Steam, fandom · medium.

## Health and unit size

- Single-entity health and damage scale with unit size: Small 25 %, Medium 50 %, Large 75 %,
  Ultra 100 %. · [GameWatcher][gw] · high. See [battle rules](battle-rules.md).
- Healing in battle is capped at 75 % of maximum HP (`healing_percentage_cap` 0.75, in
  `_kv_unit_ability_scaling_rules`). · [twwstats][tws-s] · high.

## Abilities

- Every ability has an active time and a recharge; abilities from several sources do not stack. ·
  [fandom Abilities][fw-ab] · medium. **Whether the recharge starts at the cast or when the active
  time ends was not found.** Ours: the simulator starts it when the active time ends.
- Some abilities start the battle already on cooldown (e.g. Gate of Khorne 60 s, counting only in
  melee). · fandom (1.3.0) · high.
- **AI and abilities.** No public description of when the battle AI uses them. A modder notes the
  AI wastes summons. · [Flogis mod][flogis] · low. Ours: the simulator's AI triggers are assumptions
  (`sim.json` abilities.triggers).
- Lord abilities in our armies: **Rally!** +16 leadership, 35 m, 14 s, recharge 60 s; **Hold the
  Line!** (passive) +5 melee defence, +4 leadership within 35 m. · fandom · high/medium.

## Leadership aura

- +4 within 70 m, full; fading to 0 at 105 m; auras don't stack (highest applies); the lord's own
  leadership doesn't matter. Inspiring Presence-type skills raise it. · [twwstats morale][tws-m] · high.
  See [morale](morale.md).

[g1]: https://steamcommunity.com/sharedfiles/filedetails/?id=719168880
[surr]: https://steamcommunity.com/app/594570/discussions/0/6980058308319576309/
[dmg]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/6-feature-focus-2-damage-part-1
[h622]: https://community.creative-assembly.com/total-war/total-war-warhammer/forums/7-patch-notes-amp-announcements/threads/10367-total-war-warhammer-iii-hotfix-6-2-2
[tws]: https://twwstats.com/kv/rules
[tws-s]: https://twwstats.com/kv/unit_ability_scaling_rules
[tws-m]: https://twwstats.com/kv/morale
[gw]: https://www.gamewatcher.com/news/total-war-warhammer-3-lords-monsters-debuff-small-unit-size-scaling
[los]: https://steamcommunity.com/app/1142710/discussions/0/3775743049795461974/
[p500]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/17-total-war-warhammer-iii-update-5-0-0
[fw-ab]: https://totalwarwarhammer.fandom.com/wiki/Abilities
[flogis]: https://catalogue.smods.ru/archives/238069
