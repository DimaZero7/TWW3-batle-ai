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
  - Ours: the passports carry walk, run, charge speed, acceleration, deceleration and a model's turn rate
    (`speed.turn_deg_s` from `battle_entities` turn_rate: infantry and the General 120 °/s, archers and the
    Warlord 180, militia 240). A unit on the move turns to its point at it and runs only the way it faces
    (speed × cos of the angle, standing beyond 90°); a standing lord turns at it too
    ([turning](#turning-on-the-move-probe)).

<a id="turning-on-the-move-probe"></a>
- **Turning on the move and in place (the turning probe, `build/movelords`).** No game rule for a formation
  was found online (CA bugs 8606, 6374: a formation pivots on a point near its edge, men circle while
  re-forming). The probe (1 battle, 0.25 s samples, soldier places): spearmen sent at a run to a point 150 m
  behind them **about-face where they stand** — every man turns, the front rank becomes the back (each man's
  place along the front before and after: correlation −0.98), the unit's bearing turns 180° in ~1 s; the men's
  centre reaches the run after 2 s (0.45 / 1.1 / 2.3 / 3.0 m/s at 0.5 / 1 / 1.5 / 2 s). To a point 120 m to the
  side — 2.3 m/s at 1.3 s, the run at 2 s. Lords on a move order turn at the model's rate (the General ~120 °/s,
  the Warlord ~200 °/s) and move off after 1.2 s. A facing order in place (`goto_location_angle_width`):
  spearmen 90° in 5–6 s (the formation re-forms, men walk ~8 m), 180° in 4.8 s (an about-face); the General
  ~110 °/s after ~0.7 s, the Warlord ~75 °/s — slower than on a move order and than the database's 180
  (**open**). · probe · high.
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
  - **Own formations in each other.** In the recordings (all fair battles, 5.3 M pair-seconds of own standing formations)
    own centres come within 2 / 2–4 / 4–6 / 6–8 m in 0.2–0.3 / 0.5–0.6 / 0.6 / 0.6–0.7 % of pair-seconds; two still
    formations out of melee within 4 m keep their distance over 5 s (median 0.00 m), in melee they drift 0.25–0.44 m.
    Own formations are not pushed apart. Simulator: `contact.friend_push` false ([simulator](../../training/simulator.md));
    before it pushed them apart, and the network that piles its units up in the game (13–34 % of seconds with an own
    unit within 8 m) never got a pile in the twin (0 %).
  - **A router in a crowd is slowed.** In the recordings a routing formation with one / two or more own standing
    formations within 15 m runs at 0.68 / 0.43 of its card run (one / two or more enemy: 0.79 / 0.51), a free one at 0.93.
    Simulator: `morale.rout_crowd` ([simulator](../../training/simulator.md)).
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
    dropped. The public sources don't explain the trigger either. Ordered away, a unit without a
    missile weapon stays in melee in the game only if it is chased: the melee-exit probe — unchased out of
    contact after ~4 s, chased 24–26 s (until the `melee_breakoff_secs` 24 window); 46 episodes of the
    network's battles — chased in melee for all of the first 6 s, unchased 0.58 at 6 s; the leaver takes ×1.23
    damage. The simulator: a unit with an attack order on the leaver follows it and strikes it
    (`contact.chase`), missile units are held 5 s.
- **Pursuit.** All pursuer attacks on routers count as charges; 5.0.0 removed the cap of 35 % of a
  pursuing unit attacking. WH3 keys: `pursue_max_charge_time` 5 s, `pursue_charge_max_unit_separation`
  8 m, `melee_seconds_in_close_proximity_to_stop_pursue` 1.5 s (no descriptions). · twwstats,
  fandom · high (values), low (meaning).
  - Ours: in the recordings a router chased by one pursuer loses 0.64 of what a standing target in
    melee loses, without a charge burst in the first 5 s; the simulator strikes a router at 0.43 of
    the rule (`contact.pursuit_rate`), from the rear, without the charge.
  - **A router's exit.** In the recordings (216 battles, 3,987 routs from melee) a unit that routs from melee
    stays in melee 7 s more (median; p25 5, p75 12), without a chase too (median 6 s), gathers speed over ~7 s
    (0.41 of its rout speed in the first second, 0.81 in the fourth) and all that time loses 0.77 of what a
    standing unit in melee loses. Simulator: `contact.rout_pin_s` 7 s in contact, struck at 0.83, the speed by
    `contact.rout_pin_speed` ([simulator](../../training/simulator.md#a-routers-exit-from-the-fight)).
- **Routing.** No public number for rout speed; Skaven Scurry Away +10 % at wavering or worse.
  Ours: 0.80–0.96 of run measured (fatigued; without it 0.98–1.0), units leave the map at |x|, |z| =
  1020 m. The WH2 key
  `scaled_playable_area_size_min/max` is 1024 — consistent with our 1020 m edge (medium).
  - **Where it runs.** No public formula. The recordings (328 fair battles, ~400k routing seconds,
    `build/fable/routdir2.py`; the heading = the move over the next 6 s, "explained" = within 30°): with an enemy
    within 150 m the heading is best explained by "away from every standing enemy weighted 1/distance" - 48-69 % of
    the seconds (cos 0.65-0.81); "away from the chaser" 37-68 % (defined in only 10-49 % of the seconds), "away from
    the nearest" 38-58 %, "to the own map edge" 33-37 %, "to the deployment point" 3-27 % (they run from it). With no
    standing enemy within 150 m (24 % of the seconds from the 15th s) a router still runs from the enemy army: "1/d
    over all" 85 % (cos 0.92), "away from the standing enemies' centre" 77 %, "to the own edge" 35 %, "to the nearest
    edge" 58 %. An enemy formation 40 m ahead: through it 12-34 %, around 46-54 %, back 16-33 %. Simulator:
    `movement.flee_goal`, 1/d over every standing enemy (`morale.flee_near_m` 0), the own edge only with none standing
    ([simulator](../../training/simulator.md#where-a-router-runs)).

## Tried and rejected

- **The rout's direction as a mix of directions** (`morale.rout_direction`): a router's way the sum of "away from the
  enemies within 150 m" (weighted 1/d), "away from the standing enemies' centre" and "to the nearest map edge" with
  weights 0.5 / 0.5 / 0 to 20 s, 0.5 / 0.4 / 0.1 from 20 s, 0.2 / 0.6 / 0.2 from 45 s (estimated from 4.8k / 22k / 54k /
  67k routing seconds of the recordings, not a formula of the game; `build/routgap/rout_dir.py`). The twin of 24 it3-it5
  battles, the network's Skaven, before -> after (game): running away from the near enemies 0.96-1.00 -> 0.42-0.95
  (0.60-0.72), away from the enemies' centre 0.67-0.87 -> 0.92-0.99 (0.47-0.83 - past the game); rally after 33 -> 32 s
  (54); an enemy within 95 m after 18 s of a rout 0.55 -> 0.69 (0.79); morale <= 0 0.54 -> 0.37 (0.33); shattered for good
  0.27 -> 0.24 (0.41); our losses while routing 0.025 -> 0.025 (0.070); the it6 card: 0.022 -> 0.019 (0.067). Rejected:
  the weights are fitted and the main gap (losses while routing) did not move.

[tws]: https://twwstats.com/kv/rules
[chspeed]: https://steamcommunity.com/app/1142710/discussions/0/3589960830790650436/
[chdist]: https://steamcommunity.com/app/1142710/discussions/0/4630357120384001354/
[elev]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/11-feature-focus-1-elevation
[p90]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/110-total-war-warhammer-iii-update-9-0-release-notes
[p72]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/95-total-war-warhammer-iii-patch-7-2-release-notes
[p80]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/98-total-war-warhammer-iii-update-8-0-patch-notes
[p81]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/101
