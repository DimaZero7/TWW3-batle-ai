# Missile probe

[← Back](README.md) · [Units](README.md) · [Русский](../../../ru/game/units/missile-probe.md) · [Shooting](../mechanics/missiles.md)

Short battles in the game, one shooting indicator at a time, and the same battle in the simulator (the "twin").
10 battles, 41 lanes: rate and reload of all four shooters, range, fire arc, hits by distance, by target, on
lords, shields, a moving target, rank 9. Normal difficulty, the True Sight mod, x20.

## How it works

Lanes 240 m apart; in each a shooter (30 m front, except the arc probe) and a target whose centre stands d
metres away at an angle to the shooter's facing; the target faces the shooter (or shows its flank, its back).
Everyone is held by script and fearless; the shooter fires at will (no order) until it runs out of projectiles
or the target is gone. Unarmoured targets (skavenslaves for the Empire's shooters, flagellants for the Skaven's)
so a hit counts as HP lost / the rule's damage a hit (`tools/nn/measure.py` `per_hit`: armour, overkill). The
teleport puts a unit so that the real centre distance is ~10 m more than ordered; the tables give the real one.

    python -m tools.nn.missile_probe run --plan dist|arc|range|targets|shield|moving|rank
    python -m tools.nn.missile_probe report <runs...> --sim     # the game and the twin (torch, in the container)

Lua: `src/entries/missile_probe.lua`, build target `missile-probe`; runs in `build/missile/probe_runs`, tables
`build/missile/table.py`. Below: game / simulator before / simulator after the shooting changes.

## What was measured

**Rate.** Volleys come exactly one interval apart: archers 10.0 s (database 10), militia 11.0 (database 9),
slave slingers 11.5 (database 9), Night Runners 10.5 (database 8): medians over 3-18 lanes. While the target
still has more than 3/4 of its men, a man fires that often too (10.4 / 11.0 / 11.5 / 10.2 s). Rank 9 cuts the
cycle by 2 % a rank, as the database says (`stat_reloading`): archers 8.5 s, slingers 9.5 s. The first volley
is the whole unit (0.92-1.00 of the men).

| Rate, s a man | Archers | Militia | Slingers | Night Runners |
|---|---|---|---|---|
| volley interval | 10.0 / 11.0 / 10.0 | 11.0 / 11.0 / 11.0 | 11.5 / 11.5 / 11.5 | 10.5 / 10.5 / 10.5 |
| over the whole fight (the target thins) | 12.9 / 11.0 / 10.0 | 14.7 / 11.0 / 11.0 | 13.1 / 11.5 / 11.5 | 12.3 / 10.5 / 10.5 |

**Range is centre to centre.** The target steps 2 m closer every 2 s; the first arrow comes at 128.9 / 131.3 /
130.0 / 130.9 m between centres for targets 6 / 15 / 36 / 31 m deep (archers, range 130). The edge rule would
give 139-152 m (the simulator before: 139.5-151.7), the centre rule 128.6-129.7 (after). Archers at 138 m, Night
Runners at 150 m (range 140), slingers at 126 and 133 m (range 120) never fired, though the edges were in range.
Exception: the militia (pistols) fired at 97.5 m with range 90, in partial volleys.

**The fire arc is each man's, to the nearest part of the target; without an order a unit does not turn.**
Archers, 46 m front, the target 60-70 m away:

| Target | Share of the men in the first volley | Turn |
|---|---|---|
| narrow (6 m) at 35 deg | 0.42 / 1.00 / 0.48 | none / none / none |
| wide (60 m) at 35 deg | 0.93 / 1.00 / 1.00 | none |
| narrow at 50 deg | 0.06 / 1.00 / 0.08 | none / 6 deg / none |
| militia (32 m), narrow at 30 deg | 0.97 / 1.00 / 1.00 | 12 deg / none / none |

**Hits a projectile** (the target with 3/4 of its men or more):

| Shooter -> target | Distance, m | Game / before / after |
|---|---|---|
| archers -> skavenslaves | 50 / 89 / 111 / 129 | 0.90 / 0.75 / 0.74 / 0.67 · 0.54 / 0.47 / 0.42 / 0.42 · 0.67 / 0.63 / 0.61 / 0.58 |
| archers -> clanrat spearmen | 102 / 112 | 0.82 / 0.84 · 0.43 / 0.42 · 0.72 / 0.71 |
| slingers -> flagellants | 57 / 96 / 114 | 0.77 / 0.63 / 0.55 · 0.61 / 0.51 / 0.47 · 0.73 / 0.65 / 0.60 |
| Night Runners -> flagellants | 52 / 93 / 135 | 0.92 / 0.75 / 0.66 · 0.61 / 0.51 / 0.47 · 0.76 / 0.69 / 0.60 |
| militia -> skavenslaves | 41 / 60 / 82 / 97 | 0.87 / 0.63 / 0.53 / 0.47 · 0.65 / 0.65 / 0.59 / 0.53 · 0.68 / 0.67 / 0.65 / - |
| archers -> slave slingers (loose order) | 131 | 0.37 / 0.42 / 0.39 |
| archers -> Warlord (his back) | 104 | 0.13 / 0.18 / 0.11 |
| slingers -> General (his back) | 110 | 0.09 / 0.20 / 0.06 |

Mean |log(sim / game)| over 29 lanes of standing targets: before 0.406, after 0.154 (arrows 0.459 -> 0.143,
slings 0.364 -> 0.126, Night Runners 0.378 -> 0.123, pistols 0.227 -> 0.257). Over the whole fight (the target
thins) 0.223 -> 0.182.

**Now** - spread in the plane across the line of fire, the calibration area an area in m², no fitted k
([missiles](../mechanics/missiles.md#accuracy-and-hitting)). Over 27 lanes of standing full-strength targets the
error is 0.096 (arrows 0.101, pistols 0.104, slings 0.050, Night Runners 0.121) against 0.171 for the "after"
model with k 1.1. Pistols at 41 / 60 / 82 / 98 m - 0.71 / 0.58 / 0.49 / 0.45 (game 0.87 / 0.63 / 0.53 / 0.47),
archers at 50 m 0.88 (0.90), the Warlord 0.13 (0.13), the General 0.08 (0.09).

**Shields** (archers, 90-100 m; clanrats with a 35 % shield and clanrat spearmen without one: the same body):
facing the archers 0.54 against 0.82 - lets through 0.66 (database 0.65; simulator 0.47 / 0.72 = 0.65); flank
on 0.86 (the shield does nothing), back on 0.79. A target without a shield turned flank on: 0.86 against 0.82
facing - the formation's shape does not change the hits.

**A moving target.** Skavenslaves running at the archers (145 -> 25 m): 0.86 a shot against 0.67-0.74 for
standing ones at the same 110-130 m - x1.2-1.3; Night Runners on flagellants running at them: 0.80 against
0.66-0.75. Walking across (~110 m): 0.76 against 0.74 standing - no difference; running away (from 47 m): 0.83
against 0.90 standing at 50 m. The first shot at a target running in comes at 120 m (centres): ~3 s after it
came into range.

**Rank 9:** archers 0.83 against 0.74 at rank 0 (x1.12), slingers 0.75 against 0.55 (x1.36).

## Probes P1–P4 (thinned target, pistols, running target, friends in the line of fire)

5 battles, plans `thin` (2), `pistol`, `moving2`, `lof`; recordings and analysis — `build/probes7` (`mband.py`,
`mcum.py`, `mvolley.py`; the twin — the same battle in the simulator, 4 copies).

**P1. A thinned target closes ranks, no holes.** Archers on skavenslaves at ~100 m: a full target of 180 (thinned
by the fire), fresh 90 and 45 (the same 30 m front, fewer ranks), 180 ordered to shift 5 m at 60 men. Hits a
projectile at the same men left, the game (2 battles):

| Target men | 180 thinning | fresh 90 | fresh 45 | re-formed |
|---|---|---|---|---|
| 60–90 | 0.52 / 0.47 | 0.52 / 0.53 | — | 0.41 / 0.46 |
| 40–60 | 0.32 / 0.41 | 0.45 / 0.34 | 0.42 / 0.42 | 0.43 / 0.39 |
| 20–40 | 0.21 / 0.31 | 0.28 / 0.28 | 0.32 / 0.28 | 0.18 / 0.22 |

A thinned target is hit as often as a fresh one with the same men: the formation closes up (its centre also moves
4–6 m towards the shooter, the rear men step up). The "holes where the dead fell" rule is rejected; the simulator
(the formation by the men left now) follows the rule. The mismatch is elsewhere: for the same health lost fewer men
die in the simulator (after 720 arrows the target has 8 % health and 86 men, in the game 15 % and 47–49), so a thinned
target stays "thicker" in it and is hit more late on. The cause is the missile kill rule (`kills.exponent`), not the
hit chance. The target's health steps in the recording are 19 (a wound) and 12 (a killing third hit on a 50-HP
slave), so the game's hits are rebuilt exactly; with them the rule "hits evenly over the living" (a man's hits
Poisson, nothing fitted) gives the game's men: 180 slaves at 25 / 41 / 60 / 80 / 100 / 120 s 167 / 146 / 109 / 70 /
37 / 16 against 157 / 133 / 101 / 68 / 42 / 22 (`build/open2/poisson_test.py`). The simulator now does so
(`kills.missile_uniform`): the twin after 720 arrows has 42–44 men (game 47–49, was 86). OPEN: the twin's health after
720 arrows is 8–9 % (game 15–16 %): on a fresh target the simulator lands 0.77 arrows a shot, the game 0.63–0.70
(down to 150 men), while its HP a hit is the mean 16 instead of a wound's 19 — the errors cancel in the first
volleys' health and part towards the end.

**P2. Pistol range — by ranks.** Militia (range 90, 7 ranks 1.7 m apart) on skavenslaves (10 ranks):

| Centre to centre, m | 85 | 90 | 94.5 | 97.6 | 89 (target 80 m wide) |
|---|---|---|---|---|---|
| game: share of the men in a volley | 0.99 | 0.97 | 0.94–0.99 | a stream at 0.54 of the full rate, firing flag off | 1.00 |
| the rank rule | 1 | 1 | 1 | 4 of 7 = 0.57 | 1 |
| simulator before (centre) | 1 | 1 | 0 | 0 | 1 |

The whole unit fires while the target's centre is within range of its front rank (94.5 − 5.1 = 89.4 m); beyond
that only the ranks within 90 m of the target's nearest men. Now so in the simulator (`missile.per_man_range_direct`,
direct fire only); archers stay centre to centre.

**P3. A running target.** Archers on skavenslaves, each lane's first volley: running at the shooters at 125–126 m —
0.94 and 0.96 a projectile, standing at 122 m — 0.82 (x1.15–1.17; simulator 0.75 and 0.75); running away at 84–85 m —
0.73 / 0.59 against 0.75 standing at 89 m, at 117–122 m — 0.53 / 0.17 against 0.77–0.82. The sign is clear (towards
the shooter more, away less), the size scattered; the simulator's hit chance has no target motion — OPEN.

**P4. Friends in the line of fire (militia, spearmen 40 m in front of them, the target 70 m away).** Friends right in
front: the first volley 0.44 of the men, 0.19 a projectile on the target, 295 HP to the friends; after that the unit
hardly fires (5 % of the rate). Friends shifted by half a front: 0.7–0.9 of the rate all fight, 3313 HP to the friends
(45 killed), 0.76 of the damage on the target without friends. Shifted by 3/4: almost the full rate, 912 HP to the
friends. The simulator: right in front — not a shot, half — 0.54 of the men, 0 to the friends (blocked men do not
fire). In the game blocked men fire and the bullets hit friends — who fires and the friendly damage are not
established — OPEN.

## The third wave's probes

Played 07.10.2026 (`build/steps/newunits_probe.ps1`, Normal difficulty; runs `build/missile-probe/runs/20261007-1247…1254`, analysis `build/step4`). They answer the
questions the simulator's estimates for the new shooters rest on ([passports](../../training/units.md#handgunners-crossbowmen-clanrats-stormvermin-throwing-stars)).

| Plan | Battles | Lanes | Question |
|---|---:|---|---|
| `newdist` | 3 | handgunners → skavenslaves and → stormvermin with halberds (armour 90) at 50 / 100 / 143 m; crossbowmen → skavenslaves at 50 / 110 / 158; stars → flagellants at 25 / 50 / 68; the 3rd battle also handgunners → skavenslaves at 150 m (past the range 145) | the reload of `musket` (13 and 7 s in the database; the pistols' cycle ×1.2) and of the crossbow (13 s); hits of flat fire by distance (open for the pistols); damage through armour 90; the per-rank range rule at 145 m |
| `starsmove` | 1 | stars walking away from flagellants that walk after them; walking and running 100 m across standing flagellants 50 m off; standing with the target 90° and 180° off the front | the 360° arc on the move and standing: do they shoot behind and aside (on the move now ±90°, fitted to the militia) |
| `meleefire` | 1 | Empire spearmen walk into melee with clanrat spearmen; handgunners 80 m off fire at will: from behind our men, at 45°, along the contact (90°); a control - the same melee without fire | the fire-position drill (`build/drill_fire/design.md`): the fire rate, the damage to the target and to our men beyond the control's |
| `hglof` | 1 | handgunners, our spearmen 40 m ahead offset 0 / 15 / 22.5 m, 15 m ahead not offset; no friends; skavenslaves 80 m off | fire past friends (P4 with pistols): who fires, how much our men catch - the data for the rule |

New in the probe: a shooter mode `hold` (no fire - a control), the shooter's own move (`s_move_fwd`, `s_move_lat`,
`s_move_run`), friends walking into melee with the target (`friend_engage`); in the report our men's HP
(`friend_hp_lost`), the target's angle off the front and whether the shooter moved at each volley (`rel_at_volleys`,
`moving_at_volleys`). The simulator's twin gives the same orders (the shooter's move, our men's attack, the control
without range).

### The third wave's results (the game / the simulator before / after the changes)

In the game the target's centre stands ~9–10 m farther than planned (`d` 50 → 60 m centre to centre), so the lanes "at the
edge" (handguns 143 and 150, crossbows 158) ended past the range: hits by distance exist at 60–110 m only, and the range's
edge needs a probe of its own (`rangenew`, below).

| Indicator | Game | Simulator before | After | Source of the change |
|---|---|---|---|---|
| A man's cycle, handgunners (database 13 s) | 14.8 s (14.6–15.1, 8 lanes) | 16.0–16.2 (13 × 1.2) | 15.0 | measured: `missile.reload_projectile_s` 14.8 |
| A man's cycle, throwing stars (database 7 s) | 8.1 s (8.0–8.2, 4 lanes) | 8.5 (7 × 1.2) | 8.5 (0.5 s steps) | measured: 8.1 |
| A man's cycle, crossbowmen (database 13 s) | 12.7–14.1 | 13.0–13.25 | unchanged | the arrows' rule (cycle = database) holds |
| First shot, `musket` | 1.5–2.5 s (the militia too) | 4.0 | 3.0 (the 3 s of a new target) | measured: `missile.aim_s.musket` 2.0 |
| Hits per shot on a full target: handguns → slaves 60 / 90 / 110 m | 0.82 / 0.72 / 0.53 | 0.78 / 0.64 / 0.58 | the same | no change: the spread model matches |
| the same: handguns → stormvermin (armour 90) 62 / 112 m | 0.79 / 0.68 | 0.84 / 0.65 | the same | armour-piercing by the blow rule |
| the same: stars → flagellants 34 / 60 / 79 m | 0.78 / 0.59 / 0.54 | 0.78 / 0.60 / 0.55 | the same | — |
| the same: crossbows → slaves 59 / 119 m | 0.82 / 0.69 | 0.86 / 0.77 | the same | +12 % at 119 m - within the probe |
| Stars walking away, the target behind | 1200 projectiles, first after 2 s, volleys 178° off the front on the move | 600, first after 111 s (after the halt) | 1200, after 3 s | database: the 360° arc (`missile.move_fire_own_arc`) |
| Stars walking / running across the target | 324 / 237 projectiles (82–151°) | 0 / 0 | 600 / 232 | the same; the walking stream across is lower in the game - open |
| Stars standing, the target 90° / 180° off | turn 82° / 180°, then volleys | no turn, the same hits | the same | open: a standing unit's turn to a target off its front |
| Handguns past the range: 153 / 153 / 159 m centre to centre | not one shot in 300 s | the front ranks fire: 612 / 774 projectiles | the same | **open**: the per-rank rule (from the pistols) does not fit the handgun - probe `rangenew` |
| Friends 40 m ahead (offset 0 / 15 / 22.5 m), 15 m ahead; target 90 m | fire at the full rate (1214–1315 projectiles); the friends lose 6054 / 3303 / 1926 / 7512 HP | 0 / 660 / 893 / 11 projectiles, friends 0 | the same | **open**: the militia in P4 (target 71 m) hardly fire - probe `lofab` |
| Fire into a melee: from behind our men / 45° / 90° (in 60 s, beyond the no-fire control) | 1 volley, then stop (friends +6.8 HP a shot, target +7.7) / 0.69 of the rate (friends +0.8, target +10.5) / 0.95 (friends +1.2, target +13.3) | 0 / full (friends +3.9, target +8.4) / full (+3.9 / +8.6) | the same | **open**: the (fitted) share on friends in melee is three times the game's at an angle - a rule after `lofab` |

Cards: all six new passports equal their cards from battle (`check_card`: men, health, mass, speeds, attack, defence,
charge, damage, armour, abilities, ammo, range, missile damage per 10 s).

### Probes `rangenew` and `lofab` (the game / the simulator before / after)

Played 07.10.2026 (runs `20261007-134716`, `-134807`; analysis `build/step4b`).

| Indicator | Game | Before | After | Source |
|---|---|---|---|---|
| Handguns, a target stepping in (30 m wide): the first shot | 146.9 m centre to centre (range 145), then at the full rate | 153.2 m, 0.55 of the men (the per-rank rule) | 142.2 m (the centre rule) | measured: only units that fire whilst moving count their range per rank (`missile.per_man_range_fire_move_only`) |
| Handguns, a target standing at 153 / 153 / 159 m | not one shot in 300 s | 630 / 792 projectiles | 0 / 0 | the same |
| Handguns, a 60 m wide target stepping in | one volley of 0.18 of the men at 150 m, then 30 s without a shot | 149.1 m, 0.40 | 142.1 m, 1.0 | the same; the lone volley at the target's teleport is a transient |
| Crossbows, a target stepping in | 159.6 m (range 160) | 157.0 | 157.0 | the centre rule (arrows) holds |
| Militia / stars, a target stepping in | **112.9 / 91.8 m** (range 90 / 70), then at the full rate | 100.2 / 78.4 | the same | **OPEN**: a standing target at 97.6 / 78.7 m they shoot by a stream / one volley (P2, `newdist`); a stepping one 20 m beyond any rule |
| Friends 40 m straight ahead, the target at 69 m: militia / handguns | 199 / 25 projectiles (a volley, then stop), friends −986 / −399 HP | 0 / 0 | 0 / 0 | matches: fire held |
| the same, the target at 90 m | 2160 / 1279–1315 projectiles (the full rate), friends −6183 / −6268 HP | 0 / 0 | 0 / 0 | **OPEN**: the distance to the target decides, not the weapon |
| Friends 60 m ahead, the target at 88 m: handguns | 1141 projectiles (0.6 of the rate), friends −6432 HP | 0 | 0 | OPEN |
| Friends catch bullets fired past them | 0.22–0.27 hits a shot (5 lanes, both weapons) | 0 | 0 | the number is there, the rule "when it fires" is not |
| Fire into a melee at 45° / 90° (in 60 s, beyond the control) | friends +0.8 / +1.2 HP a shot; the target +10.5 / +13.3 | +3.9 / +3.8; +8.4 / +8.6 | +0.7 / +0.8; +11.5 / +11.7 | measured: `missile.friendly_fire.musket` 0.08 (was the mean of arrows and slings, 0.41) |
| Fire into a melee from behind our men | 1 volley (friends +6.8 HP a shot), then stop | 0 | 0 | close: the simulator does not fire through its men |

Fire past friends, in short: militia and handguns behave alike - the weapon does not decide. A target ~70 m off behind friends
40 m ahead - stop after a volley (and the militia's P4, target at 71 m); a target ~90 m off - fire at the full rate, and the
friends catch ~¼ of the bullets; friends 60 m and 15 m ahead with the target at 90 m - fire at ~0.6 of the rate. It looks
like the bullet's arc over the friends' heads (at 90 m the bullet rises higher than at 70), but the database has neither the
muzzle's height nor the aim point - no rule follows.

Also OPEN:
- crossbows at the range's edge: hits a shot 0.45 at 160 m (the game) against 0.72 (the simulator); at 119 m 0.69 / 0.77;
- the thinned target (P1): the game needs more shots (handguns 1669 against 1170).

Measurements next (3 battles, prepared, `build/steps/newunits_probe.ps1 -Only probes -Plans lofthresh,lofheight,rangestand`).
A web search (`build/lof_research/notes.md`) explained fire past friends: a check per man along the bullet's real arc (`low`:
a fixed speed, the angle rises with the range), friends at their real size (the True Sight mod); two numbers are missing - the
muzzle's height and the aim point's height. The probes give them (the target's centre is `d` + `CENTRE_OFFSET` 9.5 m: the game
places the target with its front at `d`):

| Plan | Lanes | What it gives |
|---|---|---|
| `lofthresh` | handguns, friends 40 m ahead, the target's centre ~70 / 75 / 80 / 85 m; militia the same at ~80 m | the target distance from which a covered unit fires: the muzzle's and aim point's heights together (and the pistol against the handgun) |
| `lofheight` | handguns, the target ~85 m, friends 15 / 25 / 50 / 60 m ahead; stars on a standing target ~90 m | friends near the shooter test the muzzle's height, near the target the aim point's (by the arc the clearance changes sign in these lanes for a muzzle at 1.3–1.7 m and an aim point at 0.9–1.2 m) |
| `rangestand` | militia on a standing target ~100 / 105 / 110 m, stars ~80 / 85 m | do units that fire on the move shoot that far without the target stepping in (`rangenew`: 113 / 92 m at a stepping one) |


## What does not match (open)

- Pistols fire as a stream from the first volley; arrows ~10 % more often than the game by the end of a lane (re-aim).
- On a fresh target the simulator lands 0.77 arrows a shot, the game 0.63–0.70; its HP a hit is the mean 16 instead of a wound's 19 (P1): the errors cancel in the first volleys' health.
- A target running at the shooter x1.15–1.25 a projectile, running away less (P3); no motion in the model.
- Fire through friends: in the game blocked men fire and hit friends (P4); in the simulator they do not fire.
- Rank: the simulator has no ranks (all battles rank 0).
