# Morale (leadership)

[← Back](README.md) · [Documentation](../../README.md) › [Game knowledge](../README.md) › [Battle mechanics](README.md) › Morale · [Русский](../../../ru/game/mechanics/morale.md)

Every morale modifier the game's tables and the community name, with values, plus the state
machine (waver, rout, rally, shatter). Conventions: [index](README.md). Our measurements:
[morale](../units/morale.md); the simulator's rules: `tools/nn/sim/morale.py`.

The best source here is the game's own `_kv_morale_tables`: the WH3 9.0.0 values on
[twwstats][tws-m] equal ours (`config/nn/game_rules.json`), and the WH2 copy still carries
CA's description column, which explains most keys. "WH2 description" below means that column.

## How morale works

- **Points against thresholds, no dice.** Morale is a sum of points: leadership plus every
  active modifier. A unit routs when the sum falls below 0. · [1d6chan][1d6], [fandom Leadership][fw-lead] · medium.
  - Ours: agreement (rout at `MoralePercent` 0, median 0.00 in the recordings).
- **Smooth change.** Each tick morale moves towards its target by 15 % of the gap
  (`percent_update_per_tick` 0.15, "the ideal changed amount"), at least 1 point
  (`minimium_increment_update_per_tick` 1). · [twwstats morale][tws-m] · high.
  - Ours: agreement; the simulator's 0.5 s tick comes from +2 points/s measured in every
    recording. A player claims the battle tick is 0.1 s (fatigue, see [fatigue](fatigue.md)); if
    morale moved 1 point per 0.1 s tick it would rise 10 points/s, which we never see — so morale
    updates on a slower tick (our measurement stands). The recordings of 92 battles (10 s windows where the game's
    morale moved by 12–25 points and settled): the share of the gap closed in 1–5 s is 0.26 / 0.46 / 0.62 / 0.75 /
    0.87, the rule gives 0.28 / 0.48 / 0.62 / 0.73 / 0.80. The battle casualties table agrees too: where the game's
    greatest effect is "Casualties taken", its morale needs exactly the table's −2 … −74 (medians).
- **State thresholds — the unit is now known.** The descriptions mark some thresholds "% basis"
  (a share of leadership) and leave the others in points: impetuous ≥ 1.1×, eager 0.9–1.1×,
  confident 0.65–1.0×, steady from 30 points up to 0.8×, shaken 12–32 points, wavering 0–16
  points, broken −50…0 points. · [twwstats morale][tws-m] · high.
  - Ours: [our morale page](../units/morale.md) lists these as "unit not checked"; the simulator
    already uses wavering < 16 points and confident ≥ 0.65×. Resolved.
- **Thresholds with a margin.** The overlapping bands are a margin: a unit starts to waver only below 12 points (the
  bottom of "shaken") and stops only above 16 (the top of "wavering"); "shaken" below 30, back to "steady" above 32.
  MoralePercent is points over the card's leadership (Strength in Numbers' +6 is not in the divisor). · the recordings
  of 92 battles and the morale probe · high.
  - Measured: the recordings of 92 battles (1 s): of 33,791 seconds at 12–16 points 5.3 % waver when the unit came from
    above (over 16) and 98.2 % when it came from below (under 12); wavering starts at 9.9–11 points (p25–p75), 3.2 % at
    12 or more. The morale probe's lanes (0.25–0.5 s): all 32 wavering starts go 12 → 11 points, all 9 ends 16 → 17;
    shaken → steady at 33.
  - Ours: the rule `morale.waver_band` (wavering and shaken with the margin).
- **Leadership above 100** shows as 100 on the card but still absorbs penalties. WH2 · Steam · medium.

## Modifiers (points)

All values are WH3 9.0.0 = our `game_rules.json`; descriptions from WH2.

| Modifier | Points | What the table says | Ours |
|---|---:|---|---|
| Total casualties 10…90 % | −2, −4, −7, −11, −16, −22, −32, −47, −74 | share of HP lost in the battle (`use_hitpoints_instead_of_casualties_prop…` 1) | same |
| **Recent casualties** 6/10/15/33/50 % | −6, −12, −20, −44, −80 | lost **in the last 4 s** | same: a sliding 4 s window (`morale.casualties_s`; probes: 'HP lost recently' holds ~4 s after a volley) |
| **Extended casualties** 10/15/33/50/80 % | −4, −6, −14, −32, −60 | lost **in the last 60 s** | same: a sliding 60 s window (`morale.extended_s`) |
| Morale shock | — | 25 % lost in 4 s triggers a "morale shock" (`recent_casualties_shock_threshold` 25) | not modelled: the WH3 recordings show none — of 30,938 rout starts (1080 battles, `build/rally/shock.py`) not one at morale above 0.3 after losing 25 % in 4 s; the `morale_shock_*` timers (14 s, 85 s, threshold 13) are terror (below) |
| Lord's aura | +4 | full within 70 m, then fading to 0 at 70 × 1.5 = 105 m (`inspiration_radius_max_effect_range_modifier` 1.5); scaled by command stars between min and max (both 4) | same (70 m, fading to 105 m); the aura does not reach the lord himself — measured: a lord at full health with a unit of his near stands at (leadership + 4 Hold the Line + 5) / leadership, without his own +4 (General 1.129 in 106 battles, Warlord 1.083 in 54) |
| Encourage (unit) | +4 | flat, "adjusted for distance"; since 5.3 all Encourage is +4 and does not stack with the lord's aura | — |
| Lord died recently / dead / fled | −16 / −10 / −16 | | death: −16 for 45 s, then −10 to the end; rout on the field: the aura only; "fled" −16 when the lord leaves the map, ~120 s ([measured](../../apps/entries.md#lord_fall)) |
| Winning slightly / winning / significantly | +3 / +6 / +8 | the ratios are not in the tables | ratios 1.5 / 2.5 / 4 calibrated; never for a single entity (a lord). The rule `morale.combat_shooter` (off): a shooter out of melee weighs only its exchange with the enemy it shoots at (not other enemies' fire on it) — recordings: shooters in a fight need +8 at any balance of all HP from 0.67 and +5 below 0.4, where the old rule gave −3 / −8 |
| Losing / significantly | −3 / −8 | | an even or slightly lost fight gives nothing (the database has no "losing slightly" key): −3 only once the enemies it fights have lost less than 0.6 of its share of health (`morale.combat_even`; the probe hyp, two even fights of 80 s: clanrats behind spearmen at a share balance of 0.62–0.71 — 35 s without −3, −3 from 0.58; spearmen behind clanrats at 0.59 — −3; the HP balance does not part them: none at 0.53–0.61, −3 at 0.68); without the rule −3 once the unit takes more damage than it deals (Goumin's WH3 kv_rules guide), −8 once 2.5 times as much (the morale probe: a balance of 0.32 showed −8 in the game, 0.58–0.63 −3); a single entity in melee always −3 (measured: the game lords' morale in melee −3.8 points, 95 % CI −4.7…−3.0, whatever the balance) |
| Attacked in flank / rear | −6 / −14 | "first contact from flank / rear" | −6 / −14 while an enemy strikes from that side (the rear instead of the flank, not both: the probe hyp — spearmen struck in the rear, a second enemy joins on the flank 3 s later, the morale stays at 40 points for 19 s more; summed −20 would give 34). The rule `morale.attacked_first` (off): the side is where this pair's contact **began**, held while the pair stays in contact (as the database's text says; recordings: "attacked in the rear" in 18 % of the seconds where the contact began in the rear and the attacker is now in front — the unit turned — and in 8.5 % where it began in front and the attacker is now behind). See [flanking](flanking.md) |
| Flanks exposed one / both | −3 / −6 | lost within `open_flanks_effect_range` 120 m | same points, 60 m |
| Flanks secure | +5 | | +5 (neighbour within 120 m) |
| Routing units near | −3 per routing friend (≤ 4), +2.5 per routing enemy (≤ 5) | weighting × rout balance; within 100 m front/flank | routing friends and enemies cancel: only the surplus counts — (enemies − friends) × 2.5 when more enemies rout, × −3 when more friends (`morale.routing_net`; the probe hyp: 2 friends + 2 enemies — 60 points and no label at all, summed would be 59 and "Allies routing"; 2 friends + 4 enemies — 65, summed 64; apart, 2 enemies +5, 2 friends −6, as the database); without the rule summed, 100 m centre to centre. The rule `morale.routing_enemies_reach` (off): routing **enemies** counted to 200 m — measured on the recordings: where the game's greatest effect is "Enemy routing", its morale needs 2.5 × the routing enemies within 200 m (median error +4.25 at 100 m, 0.0 at 200 m); one router at 175–200 m and none nearer still needs +3.5. Friends fit 100 m better. Why the enemies' reach is longer is not known |
| Strong enemy near | −3 … −24 | by enemy combat power 4 … 32, within `enemy_effect_range` 70 m | by default −3 only (an enemy worth 3 times as much within 70 m). A rule by the database's scale exists, off (`morale.strong_enemy` `balance`): combat power = 1.975 × the standing enemies' value / (its own + its standing friends'), each weighted 1 − d / 180 m — measured (the probe strong2: error ≤ 0.4 point; 72 recorded battles: without friends the rule contradicts the game). No formula online. Greatswords give nothing in the probe: they are slower — an enemy slower than 0.75 of the unit's run speed (effects counted: skavenslaves with Strength in Numbers 4.2 × 0.9 = 3.78 m/s) is not counted (`strong_enemy.speed_ratio`; greatswords 2.8 m/s — not counted, swordsmen and spearmen 3.0, flagellants 3.6, the General 3.4 — counted; any threshold 0.741–0.794 parts them so; the game's label is "superior in strength and speed"). Off: it does not reduce the rout deficit, same winner 60 → 55.5 of 72 ([simulator](../../training/simulator.md#expendable-units-resistance-a-stronger-enemy)) |
| Under missile fire | −5 | "attacked by projectile" | same; holds 15 s after the last hit (probes: the game's flag goes off exactly 15 s later in 9 barrages of 9) |
| Attacked / damaged by artillery | −8 / −10 | near miss = within 12 m (`artillery_near_miss_distance_squared` 144) | — |
| Fear | −8 | enemy "is frightening"; range 20 m (fandom says 30 m) | — |
| Very tired / exhausted | −2 / −6 | tired 0 | same |
| On the hill | +10 | "up-hill of all enemies" | — (flat map) |
| Charging | +15 | morale bonus of a charge, timeout 60 s (guides: lasts ~10 s) | the same: +15 in blocks of 6 s from the charge pose, a second block while the unit still runs at its target or its charge has landed (`morale.charge`; recordings: the label 'Charging' lasts exactly 6 or 12 s; the morale probe: it starts 4.5–6.5 s before contact). `charge_timeout` 60 is, it seems, the block's length in tenths of a second (6 s, as `waver_base_timeout` 25 = 2.5 s). The rule `morale.charge.moving` (off): another block while the unit moves under an attack order, with no cap on blocks; in melee a unit moving under an attack order starts a block again. The recordings of 92 battles: after a block's end another followed in 54–64 % of the cases with the game's moving flag on and 0–2 % with it off (1st, 2nd and 3rd block alike; labels of 18 and 24 s 499 and 155 times — the old cap of 2 blocks cut them); deep in melee 93 % of the label's starts came with a move (12 % of such seconds) |
| Surprised / panic | −30 / −50 | (ambush; panic not described) | — |
| Inspired | +30 | "a friendly is having an inspiring effect" (abilities) | — |
| Army on the brink | −120 | when enemy/own current strength ≥ 2.6 **and** own strength ≤ 0.22 of the start | modelled (strength: the DB's combat potential × health) |
| Night battle unprepared | −5 | WH3 seems to have no night battles | — |
| Difficulty | player: Easy +4, Normal 0, Hard −2, Very Hard −4 | AI: WH2 −4 / 0 / +4 / +10; WH3 replaced it with a range (`difficulty_modifier_ai_extra_multiplier_low/high` 0.4 / 0.8) | Normal: 0 |

Sources: [twwstats morale][tws-m] (high), [fandom Leadership][fw-lead], [fandom Flanking][fw-flank],
[fandom Encourage][fw-enc] and the 5.3 notes (Encourage +4), [WH3 kv guide][g3] (medium).

- **Starting reserve.** Ours: units start above leadership (Skaven +6 points, the fitted
  `morale.faction_bonus`). This matches **Strength in Numbers** (Skaven passive: +6 leadership,
  +8 melee defence, −10 % speed while the unit has more than 50 % HP). · [fandom][fw-sin] · high.
  - Ours: modelled as the database says (an innate effect, `config/nn/effects.json`); the fitted
    `morale.faction_bonus` +6 was this and is 0 now. Crossing 50 % health, Skaven units drop 2.3
    points more morale in the next 3 s than Empire units (at 40 % and 60 % both the same): the +6
    switching off.
- **Hold the Line!** (Empire lord passive, WH3): +5 melee defence and +4 leadership to allies
  within 35 m. · [fandom][fw-htl] · medium. Ours: modelled from the ability passport.
- **Rally!** (lord ability): +16 leadership, 35 m, 14 s, 60 s recharge (WH3). · [fandom Rally!][fw-rally] · high.

## Waver, rout, rally, shatter

- **Wavering → rout.** `waver_base_timeout` 25 is in tenths of a second: 2.5 s. It is the least wavering before a
  rout, not a delay after 0: a unit routs when its morale is below 0 (at exactly 0 it still wavers) and it has wavered
  without a break at least 2.5 s; one that has wavered longer routs as soon as its morale is below 0. Without losses a
  wavering unit recovers (Empire's "casualties while wavering -> rout at the timer's end" is not in WH3). A wavering
  unit drops its current order. · [fandom Leadership][fw-lead] ("below 0 for 2.5 s"); the probe timer
  (`python -m tools.nn.morale_probe run --plan timer --kv waver_base_timeout=x10`, 4 battles x 6 lanes, normal
  difficulty; the key changed in the probe's pack, `tools/build.py --kv`): with 25 a unit whose morale went through
  wavering to below 0 in ~1 s routed 3.0 s after the wavering began; with 250 after 25.5 s (three lanes of three),
  standing up to 24 s below 0; with 0 after 1.2 s, as soon as below 0 · high (unit and meaning proved by the probe).
  While the timer runs the points do not go below 0: the game holds them at 0 (at most one 0.5 s step below), however
  low the target; once the timer is up they go below 0 and the unit routs. The timer grows with the rank: 2.5 s + 0.5 s a
  rank.
  - The floor at 0: the same lanes with 250 — for 8–24 s the samples alternate 0 / −1 … −6 points (one step below 0,
    never deeper) with the target far below; the probe hyp (`build/morale-probe/runs/20261020-014414`), rank 9 — 7 s at
    0 / −4. The recordings of 92 battles (2,851 rout starts, 1 s): the points 1 s before a rout median 0, p25 0, p5 −15
    (one step from a far target); with 25 % of the health lost in 4 s (88 starts) the same as without: the "morale
    shock" (`recent_casualties_shock_threshold`) does not lift the floor. Ours: `morale.waver_timer.floor`; without it a
    unit whose target is far below 0 reached the −50 floor within the 2.5 s and shattered before it routed.
  - The rank: the probe hyp (skavenslaves cut by the engine): rank 0 routs 3.0 s after the wavering began, rank 9 7.5 s
    (2.5 + 0.5 × 9 = 7.0). Our units are all rank 0; the simulator has no rank.
- **Broken → rally timer.** `broken_finish_base_timeout` 180 is "the base timeout for going
  from broken to rally", plus 10 × experience level (`broken_finish_timer_experience_bonus`); in the unit of
  `waver_base_timeout` (tenths of a second) that is 18 s - exactly the measured least rout before a rally
  (`morale.rally_after_s` 18, the morale probe: not before 18.5 s). Not checked by changing the key. Experience
  shortens the timer (`broken_finish_timer_experience_bonus`): rank 9 has none — the probe hyp: rank 0 rallied 18.3 s
  after the rout, rank 9 0.6 s after, as soon as its morale was above 0 (no enemy near: the engine killed them at the
  rout); the ranks between not checked (the key's 10 in tenths, 1 s a rank, would leave rank 9 another 9 s - it does
  not) ·
  [twwstats morale][tws-m] · high (value), medium (unit: by `waver_base_timeout` and the measurement).
  - Ours: a rally when morale is above 0, the unit has routed at least 18 s (the morale probe: not before
    18.5 s), no living enemy is within 95 m, and all this has held without a break for 7 s
    (`morale.rally_wait_s`: the recordings' median from the first such second to the rally, p25 4, p75 12;
    points, fire on it and the enemy's target do not matter). Measured rallies in the game take 44 s
    (median, 25–83 s). **Unresolved**: the game's own clock — rout length to the rally peaks at 18–19 and
    36–37 s; 180 = 18 s if the morale timers run in 0.1 s ticks (a guess; morale probe T-E).
- **Rally needs distance.** Players report routers within ~10–15 m of enemies or under fire keep
  running. WH2 · Steam · low. Ours: the morale probe — a rally once the enemy is beyond 94–96 m; the
  simulator 95 m (`morale.rally_free_m`), and **any** living enemy blocks the rally, routing ones too
  (`morale.rally_any_enemy`; the recordings: next to a routing enemy within 95 m 0.9 % of 4285 seconds rallied
  within 1 s, with none 12.6 % of 17,319). Our number is better. The rally itself in the simulator is not the
  "beyond 95 m" gate but the measured process (`morale.rally_hazard`): the chance to rally in a second by the nearest
  standing enemy's distance - 0 / 4.3 / 9.5 / 8.5 / 1.3 % within 95 / 95-110 / 110-125 / 125-150 / beyond 150 m (the
  recordings it1-it8, 64 battles, routers 18 s or more into the rout: ours 0.0 / 4.6 / 11.2 / 9.9 / 2.0 % over 30,457 /
  3,597 / 2,054 / 1,512 / 4,201 seconds, the game AI's 0.3 / 3.6 / 6.6 / 6.5 / 0.9 % over 8,279 / 1,455 / 1,184 / 1,147 /
  8,256; `build/fable/rallyhaz.py`). Estimated from the recordings, the mechanism not found: the probe rally2 (one
  pursuer) rallies as soon as the enemy is 92-96 m away - the gap stays open.
- **A weak enemy does not hold the rally.** An enemy weaker than a third of the router (strength: combat potential ×
  health, as `strategic_value`) does not block it. · the probe hyp and the recordings · high (that a weak one does not
  block), medium (the 1/3 edge).
  - The probe hyp (`build/morale-probe/runs/20261020-014751`): night runners cut by the engine to 3 % (4 men) keep
    25–30 m from routing spearmen — the rally at 18.6 s, on the clock; spearmen of 4 men 50–78 m from routing
    skavenslaves — 18.6 s (a full enemy holds it to ~95 m). The recordings of 92 battles (routers 18 s or more into the
    rout, morale above 0, out of melee, `build/morale-hyp/g2_weak_rec.py`): with the strongest enemy within 95 m at
    0–0.17 / 0.17–0.33 / 0.33–0.5 / 0.5–1 / more than 1 of the router's strength, a rally 0.119 / 0.087 / 0.006 / 0.005 /
    0.003 a second (42 / 149 / 1,287 / 3,425 / 26,748 s), no enemy within 95 m 0.136: the edge between 0.17–0.33 and
    0.33–0.5.
  - Ours: `morale.rally_weak` (a weak enemy neither holds the rally nor counts as the nearest).
- **The rally distance is from the enemy's formation edge.** A rally once the router's centre is ~96 m from the
  enemy's formation edge, not from its centre. · the probe hyp · medium (2 lanes).
  - The probe hyp (`build/morale-probe/runs/20261020-014634`): the enemy (spearmen) keeps 40 m from routing
    skavenslaves and halts 20 s later; the router runs off sideways along the enemy's front: with a 12 m front the
    rally at 102 m centre to centre (96 m to the edge), with a 60 m front at 127 m (97 m to the edge); the difference
    25 m = the difference of the half-fronts 24 m. The earlier lanes (a 30 m front, the router leaving along its depth):
    92–96 m centre to centre, ~89–93 m to the edge. "70 m from the edge" (the hypothesis) may be edge to edge: a
    scattered router spans another ~20–25 m, which centres do not show.
  - Ours: no — the simulator's rally goes by the measured chance by the centre distance (`morale.rally_hazard`), its
    table taken by centres.
- **No re-rout for 10 s after a rally** (`post_rally_no_rout_timer` 10). · high. Ours: same.
- **Shattering.** Always after 3 routs (`shatter_after_rout_count`); when the points reach the floor −50
  (`ums_broken_threshold_lower`, the bottom of "broken": the game never goes below) — at once; after the
  1st / 2nd rout — by casualty rules (`shatter_after_first_rout_if_casulties_higher_than` 0.05, `…_second_…`
  0.1; the key's description is garbled: "[(THIS × starting men) < current men]"). The living men count, not the
  health: shattered once fewer than 5 % / 10 % of the starting men live (`use_hitpoints_instead_of_casualties` 1 is not
  about this rule). · [twwstats][tws-m], [fandom][fw-lead], [Goumin's WH3 kv guide][g3], the recordings of 92 battles ·
  high.
  - Measured (the recordings of 92 battles, `build/morale-hyp/g3_rec*.py`): every shatter of a 1st / 2nd rout above the
    floor (41 / 46) came the second the living men fell below 5 % / 10 % (the second before: exactly 5.0 % / 10.0 %);
    routers below 5 % of their health with more men alive ran on — 5,756 such seconds on a 1st rout, 1.9 % shattered
    within 2 s (units at 4–5 % health rallied, ran 150–360 s or left the map).
  - Ours: the same (`morale.shatter_rules`): shattered at the 3rd rout; at the −50 floor always, army
    destruction or not (unbreakable units never); a router on its 1st rout below 0.05 of its starting health,
    on its 2nd below 0.10; with `morale.shatter_by` "men" by the share of living men. The recordings of 221 battles (1st and 2nd routs of infantry and missile units,
    army-destruction waves excluded): all 92 + 61 shatters on the run met one of the conditions (the floor
    0.75 / 0.54, the health 0.46 / 0.84), the rallied 0.000–0.003 of 3205; shattered units sit at −48…−50.
- **Terror.** A terror-causer's melee hit makes an enemy within 5 m with morale ≤ 13 points rout
  for 14 s; the same enemy can't do it again for 85 s; 4 terror routs shatter (3 for normal
  routs). Every terror-causer also causes fear. · [twwstats][tws-m], [fandom Causes Terror][fw-ter],
  [Steam thread][terror] · high (values), medium (the 4-rout rule).
- **Undead and daemons don't rout**: they crumble (WH3 14–28 HP/s) when broken and disintegrate
  when shattered; daemons take "instability" (132–264 HP/s) and "Banished" (798–1596 HP/s).
  Unbreakable units never lose leadership. · [fandom][fw-lead] · high.
- **Army losses.** When the army as a whole is beaten (the −120 rule above), every unit routs
  except unbreakable ones; one source says the trigger moved from ~92 % of the balance-of-power
  bar (WH2) to ~75 % (WH3); 2.6 : 1 is 72 % of the bar. · [fandom][fw-lead], Steam · medium.
  - Ours: modelled (`sim.json` `morale.collapse`: −120 at enemy / own strength 2.6 and own 22 % of
    the start; a unit's strength is its combat potential from the DB, as the recorded `strategic_value`:
    (`melee_cp` + the abilities' potential + `missile_cp` × the ammunition curve) × health — the General 950,
    the Warlord 900; routers at 0.5 in the side's sum, shattered units 0; 89 recorded battles: error
    0.01–0.04 %, the onset in the same second in 0.94 of cases, `build/movelords/cp`). Before: In the
    network's gate battles (14, 02.10.2026) the Skaven armies collapsed all at once (every unit's
    `MoralePercent` −1…−2.7 within 1–2 s) at strength 0.31–0.42 of the start and enemy / own
    1.2–1.9 by that measure (standing units only: 0.12–0.26 and 1.7–5.9): the rule with these numbers
    does not fire for them. **Gap** — record the game's balance of power (CCO
    `BalanceOfPowerPercent`) to find the trigger.
- **Rout speed.** No public number. **Scurry Away!** (Skaven): +10 % speed at wavering or worse.
  · [fandom][fw-scurry] · high.
  - Ours: measured routing speed (all recordings): Empire 0.865 of the run, Skaven 0.945 below half
    health and 0.866 above it — Scurry Away!'s ×1.1 and Strength in Numbers' ×0.9. That is the run with
    fatigue: divided by the database's fatigue multiplier, routers run at 0.98–1.0 of the run in every
    fatigue state. The simulator: `rout_speed` 0.985 of the fatigued run and both passives as innate
    effects. Agreement.
- **Expendable** units don't scare others when they rout (except other expendables); Knights
  ignore routing peasants (3.1.0). · [fandom Attributes][fw-attr] · high. Ours: a routing expendable
  unit scares only expendable units (recordings: an expendable unit −3.0 points in 4 s, 350 cases; a
  normal one 0.0, 178).

[tws-m]: https://twwstats.com/kv/morale
[1d6]: https://1d6chan.miraheze.org/wiki/Total_War_Warhammer/Tactics
[fw-lead]: https://totalwarwarhammer.fandom.com/wiki/Leadership
[fw-flank]: https://totalwarwarhammer.fandom.com/wiki/Flanking
[fw-enc]: https://totalwarwarhammer.fandom.com/wiki/Encourage
[fw-sin]: https://totalwarwarhammer.fandom.com/wiki/Strength_in_Numbers
[fw-scurry]: https://totalwarwarhammer.fandom.com/wiki/Scurry_Away!
[fw-htl]: https://totalwarwarhammer.fandom.com/wiki/Hold_the_Line!
[fw-rally]: https://totalwarwarhammer.fandom.com/wiki/Rally!
[fw-ter]: https://totalwarwarhammer.fandom.com/wiki/Causes_Terror
[fw-attr]: https://totalwarwarhammer.fandom.com/wiki/Attributes
[g3]: https://steamcommunity.com/sharedfiles/filedetails/?id=2776861563
[terror]: https://steamcommunity.com/app/1142710/discussions/0/688619343242660842/
