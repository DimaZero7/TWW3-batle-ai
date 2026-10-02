# Random armies

[← Back](README.md) · [Documentation](../README.md) › [Data for training](README.md) › Random armies · [Русский](../../ru/training/armies.md)

The random battle generator for training: two armies with an equal budget (a faction's
`budget_factor` can change its share, see below), each a lord and 0 to 19 units, already deployed. The same battle can be played both in the
[simulator](simulator.md) and in the game. The rules come from the
[network model](network.md#armies-and-battle-rules).

## How to get them

```bash
.venv/Scripts/python -m tools.nn.armies                       # 5 samples and a summary over 10,000 battles
.venv/Scripts/python -m tools.nn.armies --samples 3 --xml build/nn-armies   # + the game's battle file per sample
bash tools/nn/dock.sh tools.nn.armies --samples 8 --stats 0 --sim         # + the battles in the simulator (torch)
py -3.14 -m tools.nn.armies.templates                         # army templates from the game's database
```

```mermaid
flowchart LR
  pools["config/nn/pools.json"] --> gen["tools/nn/armies<br/>generate + place"]
  units["config/nn/units.json"] --> gen
  db["data/db.pack"] --> tpl["tools/nn/armies/templates.py<br/>→ config/nn/army_templates.json"]
  tpl --> gen
  gen --> xml["tools/nn/scenario.py<br/>the game's battle file"]
  gen --> sim["tools/nn/sim/scenario.py<br/>simulator"]
```

## Files

| File | What |
|---|---|
| `config/nn/pools.json` | each faction's units for training, its budget factor (`budget_factor`), the share of template armies (`mix`), limits (`caps`) |
| `config/nn/army_templates.json` | the factions' army templates from the game's database (written by `templates.py`) |
| `tools/nn/armies/pools.py` | pools: unit passports, families, template shares |
| `tools/nn/armies/generate.py` | budget, buying units, a battle from a seed |
| `tools/nn/armies/place.py` | deployment |
| `tools/nn/armies/export.py` | the game's battle file, the `arenas.json` format, army descriptions for the simulator |
| `tests/tools/test_nn_armies.py` | tests (levels 1–2, numpy only) |

## How a battle is built

1. **Factions.** Each side takes a faction from the list independently, so there are both
   mirror battles and battles of different factions.
2. **Budget B.** Log-uniform between "the dearer lord and the cheapest unit" and what both
   factions can field in 1 lord and 19 units. The budget is drawn again until both sides
   can spend it.
   - **Budget factor.** A side's budget is B × its faction's `budget_factor` / the larger
     factor of the two sides (`config/nn/pools.json`, 1.0 by default). The bounds of
     B take the factor into account, so both sides can always spend their share. Now
     every faction has 1.0. The Skaven had 0.8 from 01.10.2026: at an equal budget they
     won all 10 whole battles in the game
     ([measurements](measurements.md#whole-battles-empire-against-skaven)); at 0.8 they lost
     ~90% in the simulator, so on 02.10.2026 the factor went back to 1.0 and the Empire got
     the spearmen with shields and the swordsmen instead
     ([unit passports](units.md#spearmen-with-shields-and-swordsmen)).
   - **A faction's cap.** `budget_max` in a faction's entry caps what its side spends (B × its
     share). The Skaven's is 6700, their most before their wave (the Warlord and 19 clanrat
     spearmen): the Night Runners (450) and the clanrats with shields (350) would otherwise raise
     every battle with Skaven to richer armies. Their cheapest unit, the skavenslaves (125), lowers
     the least Skaven-mirror budget from 675 to 650.
3. **Buying.** Each side spends 0.95 to 1 of its budget, the lord included, so sides with
   the same factor differ by at most 5%. A unit is taken only if the army can still end
   inside that window. For this
   there are numpy tables of what the pool can buy with k units. So a side never fails
   and never overspends.
   - **A template army** (75%, `mix.template`) follows one of its faction's templates from
     the game's database, the way the AI recruits. It buys by the template's shares while
     the budget lasts. The number of units comes out by itself. `caps` apply.
   - **A random army** (25%, `mix.random`) is what a human might build, stacks included.
     The number of units is drawn uniformly among those that fit the budget, and the
     preferences for units are random (Dirichlet, α = 0.7): a swarm of cheap units, a few
     dear ones, missile only. No limits.
4. **Deployment** — below.

A battle is set by its seed: `battle(seed)`. For training — seeds `TRAIN_SEEDS`
(0 … 10⁹ − 1), for evaluation — `EVAL_SEEDS` (10⁹ … 10⁹ + 10⁶ − 1). The ranges do not
overlap, so the network never sees the evaluation battles in training.

## Army templates from the game's database

The proportions of template armies come from the game's army generator
(`cdir_military_generator_*`, the "campaign director"). The tables are read whole to the
last byte (`tools/nn/dbtables.decode`). The field names are ours, from the values.

| Table | What it holds |
|---|---|
| `cdir_military_generator_template_priorities` | generator config → its templates and weight (`WH_Empire` → `WH_Empire_land` … `_4`) |
| `cdir_military_generator_template_ratios` | template → share per unit group (`..._melee_infantry_main_frontline_spears_high_quality` 20, …) |
| `cdir_military_generator_unit_qualities` | unit → group and quality step |
| `cdir_military_generator_unit_group_overrides` | group → parent group |

- The multiplayer army generator uses the same templates: config `WH_mp_Empire_land`
  leads to `WH_Empire_land` … `_4`.
- A faction's config is named in `pools.json` (`generator`), by name. A "faction →
  config" table in the database was not searched for.
- A group's **family** is the end of its parent chain: all melee infantry ends at
  `melee_infantry_trash`, missile infantry at `ranged_infantry`. A template's shares are
  summed over the families the pool has. The rest (artillery, cavalry, monsters, heroes)
  is dropped, then the shares are normalised.
- Inside a family the share is split **evenly** between the pool's units. Which unit of a
  family the game's AI takes depends on the quality step (`tier_1_step_2` for slaves,
  `tier_1_step_4` for clanrats) and on how rich the army is. How the generator picks the
  step is not in the tables. Here the budget decides cheap against dear.
- Not checked in the game: whether the campaign AI recruits exactly by these templates.
  They are the templates of the game's army generator.

Shares by number of units (lord not counted), per template:

| Faction | Template | Spearmen | Spearmen with shields | Swordsmen | Flagellants | Greatswords | Archers | Militia |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Empire | `WH_Empire_land` | 12% | 12% | 12% | 12% | 12% | 19% | 19% |
| | `WH_Empire_land_2` | 13% | 13% | 13% | 13% | 13% | 17% | 17% |
| | `WH_Empire_land_3` | 12% | 12% | 12% | 12% | 12% | 19% | 19% |
| | `WH_Empire_land_4` | 11% | 11% | 11% | 11% | 11% | 21% | 21% |

Since 02.10.2026 (the second wave, [unit passports](units.md#flagellants-greatswords-free-company-militia)):
the flagellants are in `..._melee_infantry_main_flanking`, the greatswords in
`..._melee_infantry_main_frontline_swords_high_quality`, both of the family `melee_infantry_trash`;
the militia in `..._ranged_infantry_trash`, of the family `ranged_infantry` with the archers. A
family's share is split evenly among its units (five melee, two missile). Before the wave the shares were 21 / 21 / 21 / 38% and so on; the paragraph below describes that pool (three melee units).

The spearmen (both) are in group `..._melee_infantry_main_frontline_spears`, the swordsmen
in `..._melee_infantry_main_frontline_swords` (quality step `tier_1_step_4` all three).
Both groups end at the family `melee_infantry_trash`, like all melee infantry, so the
templates' spear and sword shares (e.g. 20 + 20 in `WH_Empire_land`) are summed into one
melee share, split evenly among the three units. Before 02.10.2026 the pool had the
spearmen without shields only: 62 / 67 / 62 / 57%.

| Faction | Template | Clanrat spearmen | Skavenslave spearmen | Skavenslaves | Clanrats with shields | Skavenslave slingers | Night Runners |
|---|---|---:|---:|---:|---:|---:|---:|
| Skaven | `WH_Skaven_land` | 15% | 15% | 15% | 15% | 19% | 19% |
| | `WH_Skaven_land_2` | 17% | 17% | 17% | 17% | 15% | 15% |
| | `WH_Skaven_land_3` | 17% | 17% | 17% | 17% | 17% | 17% |
| | `WH_Skaven_land_4` | 19% | 19% | 19% | 19% | 12% | 12% |
| | `WH_Skaven_land_5` | 15% | 15% | 15% | 15% | 19% | 19% |

The Skaven wave ([unit passports](units.md#skavenslaves-clanrats-with-shields-night-runners)): the
skavenslaves are in `..._melee_infantry_trash` (`tier_1_step_2`), the clanrats with shields in
`..._melee_infantry_main_frontline_swords` (`tier_1_step_4`), both of the family
`melee_infantry_trash`; the Night Runners in `..._ranged_infantry_main_light_skirmisher`
(`tier_1_step_5`), of the family `ranged_infantry` with the slingers. Four melee units share the
melee share, two the missile share.

## Limits

`caps` in `pools.json` apply to template armies only: missile units (`inf_ranged`) at most
60% of the army's units, the lord counted, rounded down but at least 1. With 20 units — at
most 12, with 5 — at most 3; a lord with a single unit may have it be a missile unit.

Why 60%: the game's templates give missile units 23–43%. A small army can drift from the
shares by chance, and 60% keeps at least 40% of the army in melee to screen the missile
units. Missile stacks stay in the random armies — the network must be able to play
against them too.

## Deployment

`place.py` puts a side's 1–20 units in lines, in the format of `config/nn/arenas.json`:
`forward` is towards the enemy, `lateral` to the right, a unit's point is the centre of
its front.

- Melee lines in front, missile lines behind them, the lord behind all in the centre.
- 6 m between units of a line (as the arena's spearmen: 30 m front, 36 m between
  centres). Between lines — the formation's depth plus 15 m. The dearest units stand in
  the middle of a line.
- A line is no wider than the deployment zone less 5 m at each edge (300 m → 290 m:
  8 units of 30 m).
- Everything is inside the deployment zone of `tools/nn/scenario.py`: `forward` from
  40 − 300 to 40, `lateral` ±150. The gap between the sides is that of `arena.json`
  (350 m). The deepest army (3 melee lines, 2 missile lines) ends with the lord at about
  −140 m.
- A unit's width comes from `pools.json` (as in the arenas: spearmen and slaves 30 m,
  archers 40, slingers 35; skavenslaves, clanrats with shields and Night Runners 30), otherwise
  men / `rank_depth` × 1.5 m.

## Summary over 10,000 battles

`python -m tools.nn.armies`, seeds 5 … 10,004 (every `budget_factor` 1.0, the Empire's pool with
the second wave and the Skaven's with their wave, budgets capped at `budget_max` 7725 and the
Skaven's side at 6700, as before the waves):

| What | Value |
|---|---|
| Mirror battles | 50% |
| Armies: template / random | 75% / 25% |
| Budget B | 651 … 7725, median 2548 |
| Units per side (lord not counted) | mean 7.3; 1–4: 39%, 5–9: 31%, 10–14: 18%, 15–18: 8%, 19: 5% |
| Cost difference between sides of equal budgets | mean 1.5%, at most 4.9% |
| Skaven cost / Empire cost | mean 1.001, 0.951 … 1.052 (4996 battles) |
| Skaven / Skaven, Empire / Empire | mean 1.000 and 0.999, 0.952 … 1.050 |
| One side has x times the other's units | x ≥ 1.5: 49.1%; x ≥ 2: 26.2%; x ≥ 3: 7.2% |
| Missile share of a side | mean 32%; no missile: 22%; ≥ 50%: 28%; missile only: 3.6% |
| Empire template armies | spearmen 17%, with shields 13%, swordsmen 13%, flagellants 11%, greatswords 9%, archers 19%, militia 19% (before: 27 / 19 / 21 / – / – / 34 / –%) |
| Skaven template armies | clanrat spearmen 15%, slave spearmen 19%, skavenslaves 20%, clanrats with shields 15%, slingers 17%, Night Runners 15% |

Within the same melee share the cheapest unit, the plain spearmen (300), is bought most:
27% against 19% (with shields, 350) and 21% (swordsmen, 375); the market fills the budget
window, and how the dearer two split is the luck of that fit. The many-cheap-against-few-elite battles are mostly Skaven
against Empire. With the Skaven at 0.8·B (01.10.2026) they fielded about as many units as
the Empire: x ≥ 1.5 in 6.7% of battles.

An equal budget was not equal strength ([measurements](measurements.md): Skaven fielded
1601 men against the Empire's 661 and won all 10 whole battles). The 0.8 factor
overshot in the simulator; the shielded spearmen are the second try at the balance.

The first eight battles were run in the simulator (`--sim`: every unit attacks the
nearest enemy): all ended within 250–540 s. With the sides swapped, the same army wins in
13 battles of 16.

## For training

```python
from tools.nn.armies import generate, export
from tools.nn.sim import scenario

arenas = [generate.battle(seed) for seed in seeds]          # seeds from generate.TRAIN_SEEDS
armies = export.to_sim(arenas, "attack")                    # "defend": side 2 attacks
st = scenario.build(armies, per_side=generate.MAX_UNITS + 1)
```

- `generate.generate(rng, factions=None, budget_range=None, max_units=None, sides=None)` —
  the same with your own random generator. `max_units=5` and `budget_range` give small
  armies for the start of training. `factions` is a list of factions, `sides` the sides'
  factions explicitly.
- `per_side` must be 20: a side can have that many units.
- `export.to_sim` writes the battles into a temporary `arenas.json` and calls
  `tools/nn/sim/scenario.from_arena`. Once `from_arena` takes an arena dict, the temporary
  file is not needed.
- A battle in the game: `export.scenario_xml(arena)` — the battle file,
  `export.write_arenas` — `arenas.json`. A build of the battle: `python -m tools.build nn-arena --army-seed N` ([build](../launch/build.md#options),
  [in-game check](../launch/gate.md)).

## A new unit or faction

0. The whole procedure (research of its special features, simulator, network inputs, card, tests):
   [how to add a unit](units.md#how-to-add-a-unit).
1. The unit's passport — in `config/nn/units.json` (`py -3.14 -m tools.nn.units --units …`).
2. An entry in `config/nn/pools.json`: `key`, `slot`, `width`; a new faction also needs
   `lord` and `generator`, and `budget_factor` if it is stronger or weaker than its price.
3. `py -3.14 -m tools.nn.armies.templates` — the unit needs a generator group.
4. A new category (cavalry, monsters) — its own limit in `caps` if wanted.
