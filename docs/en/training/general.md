# The general's network (S28)

[← Back](README.md) · [Documentation](../README.md) › [Data for training](README.md) › The general's network · [Русский](../../ru/training/general.md)

The general is a network that makes the battle plan for the whole army: the map's and the armies' inputs, their
analysis and the units' meeting. The agreed architecture (with the "why so and what was rejected" blocks) is the page
`design/general/index.html`; this page says what of it is already in code (`tools/nn/general/`). The general is not
joined to training or to today's unit network yet.

## The map

The "Landscape" input is `map_input.py` (6 layers on a grid of 128 × 128 cells of 16 m, always as its own side sees
it) and `map_batch.py` (the same layers for many battles at once); the "Map analysis" is `map_cnn.py` (a convolutional
network: 1024 regions of 64 m, 32 numbers each, a summary cache per map); checks: `map_probe.py`, tests
`tests/tools/test_general_map_*.py`.

## The "Armies" input

Code: `tools/nn/general/army_input.py`, no torch.

- **80 slots**: 0–39 the own army, 40–79 the enemy's; one unit a slot. Always as its own side sees it: for side 2 the
  armies swap places (as the map's layers do). The slots' order means nothing; the lord may stand anywhere. An empty
  slot is flagged `empty` and holds zeros. More than 40 units a side is an error.
- **A slot is 45 numbers and 3 labels** (groups and order as in the "Unit slot" table of the design page):

| group | numbers | what and how |
|---|---|---|
| side | 1 | own 1 / enemy 0 |
| cost | 1 | multiplayer cost, log (5000 → 1) |
| size and health | 5 | men in the full unit (log 200), men come — share, health of the full unit (log 20000), health come — share, health of a man (log 10000) |
| body and speed | 4 | mass (log 10000), walk / 5, run / 10, charge / 10 (m/s) |
| melee | 10 | attack, defence, charge bonus / 100; damage and its AP part (log 500); bonuses v large and v infantry / 50; time between blows, s / 6; splash targets / 10; magical attacks — flag |
| defence and morale | 8 | armour / 200; shield, % / 100; leadership / 100; resistances to all, physical, missile, magic, fire, % / 100 (fire may be < 0) |
| missile | 12 + label | range, m / 500; ammo (log 100); projectile damage and its AP part (log 500); reload, s / 20; **spread at 100 m**, m / 10; projectiles a shot / 10; shots a volley / 20; projectile speed, m/s / 100; fire arc each side, ° / 180; aim before the first shot and silence after a new target, s / 10 |
| properties | 4 | flags: expendable, charge reflection, unbreakable, fires on the move |
| labels | 3 | trajectory (direct / high arc / direct with an arc over friends; 0 — no missile weapon), caste (12), size (4) |

  The scales are those of the unit network's passport (`tools/nn/model/passport.py`) wherever it has the field (test
  `test_the_scales_are_the_passports`). New ones: the come shares (0…1 already), the spread at 100 m / 10 m (handguns
  0.30, crossbows 0.44, slings 0.60, throwing stars 0.78), shots a volley / 20 and the aim times / 10 — as in the unit
  network's token. On all our units the numbers lie within −1…1.5 (test `test_numbers_span`).
- **There is no "other" variant.** A caste, size or trajectory not in its list is an `UnknownVariant` error when the
  data is built. The game database's trajectories: `low` → direct, `fixed` → high arc (mortars), `dual_low_fixed` →
  direct with an arc over friends; `high` is only spells' in the database — an error for now. A new variant is
  appended at the end of its list.
- **Where the numbers come from** (`final_stats`). The future bridge will read the final numbers from the unit's card
  in the game (rank, items, skills already inside) and pass them as `card` (allowed names: `CARD_KEYS`, another name
  is an error); what the card lacks comes from the passport (`config/nn/units.json`) or is computed. There is no card
  now: in the simulator and our battles the passport's numbers are the final ones (rank 0). Computed:
  - **spread at 100 m** — as the simulator computes it (`tools/nn/sim/missile.py` `square_side`,
    `config/nn/sim.json` `missile.accuracy.spread` = `square`): the side of the uniform square in the target's plane
    √(calibration area / p), p = accuracy / 100 × (calibration distance / 100)² × (1 − launch angle / 180°),
    p within 0.01…1; accuracy = the unit's accuracy + the projectile's marksmanship + 3 a rank
    (`game_rules.json` `experience_bonus.stat_accuracy`). The design page's formula "spread × (1 − accuracy / 100)"
    is the older Gaussian reading; with `spread` = `gauss` the code gives the side of a square of the same variance
    (√12 σ), so the number means the same. The match with the simulator: test `test_spread_agrees_with_the_simulator`;
  - walk and charge = database × (the card's run / the database's run) — a factor of 1 now;
  - come to the battle = now / full (1 in our battles).

## Army analysis

Code: `tools/nn/general/army_reader.py` (torch). One shared network of two dense layers for all 80 slots — the same
weights for own and enemy:

- layer 1: ReLU(W1 · 45 numbers + b1 + the trajectory's vector + the caste's vector + the size's vector), 256 wide;
- layer 2: card = W2 · h + b2, 256 numbers, no activation;
- the effect cards are added to the card (`effects`; the "Effects" input is not designed yet; 0 by default).

A label is a learned vector of the first layer (`nn.Embedding`), the same as the variant's flags with their weights.
"No missile weapon" is a vector fixed at zero. 82 688 weights (45 × 256 + 256 + 20 × 256 + 256 × 256 + 256). Empty
slots never enter the network (they are taken out before it); their card is 0. A new label variant — `add_variant`:
a new vector of zeros, the old cards stay the same to the bit. The layer's and the card's widths are parameters
(`hidden`, `card`).

Speed on the CPU (6 cores): 1024 battles × 80 slots (47 thousand occupied) — 99 ms without gradients, 272 ms forward
and backward (`python -m tools.nn.general.army_reader --bench`).

## Checks

Group 1 of the design page's "Army analysis checks" — tests `tests/tools/test_general_army_input.py` (no torch) and
`tests/tools/test_general_army_reader_torch.py` (in the container): the slots' order means nothing, empty slots do not
interfere, one network for all (the same unit in two slots — one card; own and enemy differ only by the side flag's
part), side 2's mirror, a new variant with zero weights, an unknown caste / size / trajectory is an error, the
numbers' span (inputs within −1…1.5; cards on the inputs' scale; the first layer neither all on nor all off).
Groups 2 and 3 (dead neurons, card likeness, new units) — after training.
