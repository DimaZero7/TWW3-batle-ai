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

## What does not match (open)

- Over a whole fight a man fires less often than the volley interval (12.3-14.7 s): as the target thins, more
  men skip a volley. No rule in the database or from the community; not in the simulator.
- Pistols: their hits fall fast with distance in the game (0.87 -> 0.47 at 41-97 m), the model's barely; the
  pistols' range is not centre to centre.
- A target running at the shooter takes x1.2-1.3 a shot; the model has no motion.
- Near, dense targets: the model is 10-20 % low (archers on spearmen 0.84 against 0.71).
- The game's hits fall more as the target thins.
- Rank: the simulator has no ranks (every battle is rank 0).
- The first shot at a target that has just run into range comes ~3 s later than in the simulator.
