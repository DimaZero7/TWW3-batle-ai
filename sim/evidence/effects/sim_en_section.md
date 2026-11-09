## Innate effects

`tools/nn/sim/effects.py`: one mechanism for every attribute and every passive or game-fired
ability of a unit. The catalogue is `config/nn/effects.json` (`python -m tools.nn.effects`, from the
passports; [unit passports](units.md#innate-effects)): per effect its stats, rules, conditions,
timers and whether the simulator acts on it (`modelled`). Nothing in the simulator knows a unit or
an effect by its key.

- **Owned.** `params.static` gives each unit `fx`, a bitmask of the effects the catalogue links to
  it, `fxt0` / `fxt1` (its timed effects) and the rule flags of its unconditional effects.
- **Conditions** (each step, from the state at its start): `in_melee` / `out_of_melee` (in contact
  with a standing enemy), `losing_melee` (in melee and HP taken ≥ 1.5 × dealt recently: the morale
  rule's ratio), `morale_below_half` (points < half of leadership), `not_wavering`, `hp_below_half`
  (< 50 % of the start), `hp_below_quarter`. An effect is on while all its `needs` hold and none of
  its `off_when` does; an attribute always.
- **Timed** (Strength of the Penitent): fires by itself for a standing unit when its `fires_when`
  holds and it is ready, lasts `active_s`, ends at once while an `off_when` holds, ready again
  `recharge_s` after it ends. The ability bar slot holding its ability (`ab{k}_on`, `ab{k}_cd`) shows
  the same timers, so the observation reads it as before.
- **Effects.** While on: multipliers multiply, additions add (melee attack, defence, damage, AP,
  charge, speed and charge speed, morale points, physical resistance to the 90 % cap) on the owner,
  and for an aura (`range_m` > 0, stats on friends: Hold the Line!) on the friends within range of a
  standing owner. Rule flags (`unbreakable`, `expendable`, `encourages`, `reflect`, `fire_move`,
  `fatigue_immune`) hold for the step, read by morale, the aura, bracing, shooting and fatigue. An
  effect lies on a unit that is alive, routing too (Scurry Away! speeds a rout). All of it is undone
  at the step's end, as for the cast abilities and fatigue.
- **`fx_on`** (the network's input): the effects whose conditions hold now, whether the simulator
  acts on them or not (Hide (forest) counts as on: in the game it is).
- **Schema only** (`modelled` false, with the reason in the file): Charge Defence vs. Large (no
  large units in our pools), Vanguard Deployment (placements are the army generator's), Hide
  (forest) (no woods), Immune to Psychology (no fear or terror). A new effect whose stats, rules
  and conditions the simulator has works without code (a test gives the spearmen Perfect Vigour by
  the catalogue alone).

**Measured** (02.10.2026, all fair recordings with unit keys; speed over 1 s steps, at least 3 s
into a rout): routing Empire units run at 0.865 of their run (115k s), routing Skaven at 0.945 below
half health (186k s) and 0.866 above it (14k s); running in order, steady, above half health:
Empire 0.97, Skaven 0.88. Scurry Away!'s ×1.1 and Strength in Numbers' ×0.9 (above half health)
give exactly these ratios, so `morale.rout_speed` is the Empire's 0.86. Crossing 50 % health, Skaven
units drop 2.3 points more morale in the next 3 s than Empire units (7.7 against 5.5 over 1 039 and
951 crossings; at 40 % and 60 % both drop the same): Strength in Numbers' +6 switches off there.
The Skaven's former start bonus (`morale.faction_bonus` +6, fitted) was Strength in Numbers: it is 0
now.

