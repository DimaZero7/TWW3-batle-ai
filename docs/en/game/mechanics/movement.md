# Movement and formations

[← Back](README.md) · [Documentation](../../README.md) › [Game knowledge](../README.md) › [Battle mechanics](README.md) › Movement · [Русский](../../../ru/game/mechanics/movement.md)

Speeds, charges, formations, collisions, breaking off and pursuit. Conventions: [index](README.md).
Ours: [pace and fatigue](../units/pace.md), [simulator](../../training/simulator.md).

## Speeds

- **The card's speed is metres per second × 10.** `battle_entities.run_speed` / `walk_speed` /
  `fly_speed` are m/s; the card shows them ×10 (card 40 = 4 m/s). Fatigue scales speed
  (`scalar_speed`, see [fatigue](fatigue.md)). · twwstats front-end code · high.
  - Ours: agreement — card 3.0 m/s for spearmen, 3.0 m/s recorded.
- **Hidden fields per entity** (`battle_entities`): `charge_speed`, `flying_charge_speed`,
  `acceleration`, `deceleration`, `turn_speed`, `combat_reaction_radius`, and the charge distances
  `charge_distance_commence_run`, `_adopt_charge_pose`, `_pick_target`. · [twwstats][tws] · high.
  - Ours: the passports carry walk, run, charge speed, acceleration and deceleration; the simulator
    has no turn rate out of melee ("Not modelled" in `simulator.md`). **Gap**: `turn_speed` exists.
- **Charge speed** is ~20–40 % above run speed (a Cold One rider measured at ~10 m/s against card
  6.8). Units switch to it 20–60 m from the target (Cold Ones 40 m; Chaos Knights speed at 40 m,
  pose at 35 m); typical ~35 m. · [charge speed][chspeed], [charge distance][chdist] · medium.
  - Ours: the Warlord was recorded running 5–6.5 m/s against run 4.0 / charge 4.7 — the
    simulator attributes it to Verminous Valour.
- **Slopes.** Downhill up to +50 % speed, uphill slower (a Skink Chief, 4.6 m/s base: 5.8 m/s down,
  −22 % up). Uphill costs extra fatigue. · [CA elevation blog][elev] · high. See [terrain](terrain.md).
- **Ground types.** 9.0 removed a hidden speed penalty on "sharp stones" ground; shallow water
  −20 % speed for small units without Aquatic. · [9.0 notes][p90], fandom Aquatic · high/medium.
- **Contact effects:** Poison −15 % speed (10 s), Frostbite −30 % (10 s). · fandom · medium.

## Formations

- **Spacing.** `entity_formation_spacing` 1.8 m between entity slots (WH2 description notes "set to
  1.8 when EF work is completed"). · [twwstats][tws] · high (value).
  - Ours: measured 1.5 m (120 spearmen, 30 m wide, 6 ranks → 20 files 1.5 m apart;
    `formation.spacing_m`). **Conflict**: the global slot spacing is 1.8 m; units may override it
    (formation templates, tight/loose behaviour `change_formation_spacing` in the script API).
- **Maximum drag width** 42 m (`unit_max_drag_width`, WH2); defence-line pivot 0.8 of the depth
  from the front. · [twwstats][tws] · high (WH2 value).
- **Spawn.** Units spawn 22 m from the ordered position (`unit_spawn_offset` −22, WH2), radius 2 m. · twwstats · medium.
- **Pikes** lower only within 45° of the movement direction; turning delays it. · twwstats · high.

## Collisions and pathing

- **Pushing.** Entities have soft collision; braced mass over 600 can't be pushed by small entities.
  Knock-backs and knock-downs come from mass ratios and speed changes (see [melee](melee.md#mass-collisions-knockdowns)).
- **Pathing costs** (WH2): default 2, climb/jump/wall door 10. · twwstats · high.
- **Patch fixes:** 6.1.0 made infantry "stickier" (a player report); 7.2 fixed spaghetti lines into
  minor settlements; 8.0 fixed an army clumping when every unit got the same order; 8.1 fixed
  armies stacking on the map edge while re-forming. · [7.2][p72], [8.0][p80], [8.1][p81] · high
  (except the 6.1 report: low).

## Breaking off and pursuit

- **Melee break-off.** WH3-only keys `melee_breakoff_secs` 24 and `melee_breakoff_total_immunity_secs`
  10, without description; WH2 keys `break_off_ignore_collision_time` 0 and
  `break_off_speed_and_direction_checks` 0 ("enabling makes breaking off more sticky"). Players
  advise repeated move orders to get cavalry out ("unit glue"). · [twwstats][tws], Steam · high
  (values), low (meaning).
  - Ours: in the game a fight breaks off at 0.013/s (infantry), 0.030/s (lords), 0.033/s (missile
    units), cause unknown; a break-off with the 10 s immunity was tried in the simulator and
    dropped. The public sources don't explain the trigger either.
- **Pursuit.** All pursuer attacks on routers count as charges; 5.0.0 removed the cap of 35 % of a
  pursuing unit attacking. WH3 keys: `pursue_max_charge_time` 5 s, `pursue_charge_max_unit_separation`
  8 m, `melee_seconds_in_close_proximity_to_stop_pursue` 1.5 s (no descriptions). · twwstats,
  fandom · high (values), low (meaning).
- **Routing.** No public number for rout speed; Skaven Scurry Away +10 % at wavering or worse.
  Ours: 0.80–0.96 of run measured, units leave the map at |x|, |z| = 1020 m. The WH2 key
  `scaled_playable_area_size_min/max` is 1024 — consistent with our 1020 m edge (medium).

[tws]: https://twwstats.com/kv/rules
[chspeed]: https://steamcommunity.com/app/1142710/discussions/0/3589960830790650436/
[chdist]: https://steamcommunity.com/app/1142710/discussions/0/4630357120384001354/
[elev]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/11-feature-focus-1-elevation
[p90]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/110-total-war-warhammer-iii-update-9-0-release-notes
[p72]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/95-total-war-warhammer-iii-patch-7-2-release-notes
[p80]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/98-total-war-warhammer-iii-update-8-0-patch-notes
[p81]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/101
