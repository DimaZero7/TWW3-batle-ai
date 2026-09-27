# Empire — short unit cards

[Unit catalogue](README.md) · [Русский](../../../../ru/game/units/catalog/empire.md)

Summaries of our verified WH3 **v9.0.0, build 50218.4334952**, 2026-09-26 data. Entity counts refer to full tested units at **Ultra** unit size and zero unit experience. Detailed mechanics and raw measurements remain in the linked canonical sources below.

## General of the Empire — foot, sword and shield

**Russian:** генерал Империи. **Unit key:** `wh_main_emp_cha_general_0`.

- **Type/equipment:** ordinary non-legendary melee lord; one foot character with a sword and missile-blocking shield. This is the lord, not the Captain hero or a mounted variant.
- **Main properties:** melee combat and nearby troop support; no ranged weapon. Support effects and movement were verified in the test fixtures.
- **Selected setup:** character level 1. Character level and unit experience are separate. The XML fixture's active abilities do not prove a fresh campaign lord's skill loadout.
- **Details:** [support, shield and ability limits](../states.md#general-support-and-abilities), [Lua commands](../commands.md), `exact rank-1 fixture` (local archive: `research/evidence/units/empire-20260926/passives-rank1/map_probe.xml`).

## Spearmen — without shields

**Russian:** копейщики без щитов. **Unit key:** `wh_main_emp_inf_spearmen_0`.

- **Type/equipment:** melee infantry, spears, no shields; **120** soldiers in the tested setup.
- **Main properties:** no shooting; the engine reports charge defence against large units and charge reflection. These are distinct from merely standing still or enabling guard mode. Forest concealment was tested.
- **Details:** [attributes and charge reception](../states.md#bracing-and-charge-reception), [movement, melee and guard commands](../commands.md), [live attribute table](../states.md#what-each-unit-actually-exposed).

## Archers — basic inexpensive bow infantry

**Russian:** лучники. **Unit key:** `wh2_dlc13_emp_inf_archers_0`.

- **Type/equipment:** missile infantry, bows and short swords, no shields; **90** soldiers in the tested setup. These are ordinary Archers, not Huntsmen or a Regiment of Renown.
- **Main properties:** targeted/automatic shooting and switching to melee were tested; reported bow range **130 m**. Forest concealment was tested.
- **Cost scope:** base `multiplayer_cost` **350**, the lowest in the `saved Empire ranged-unit DB comparison` (local archive: `research/evidence/units-catalog/empire-ranged-costs-20260926.json`). This identifies the inexpensive unit requested; it is not a claim about all campaign recruitment prices, discounts or availability.
- **Details:** [shooting, range and ammunition](../commands.md#shooting-two-separate-orders), [attributes and abilities](../states.md#what-each-unit-actually-exposed), `source DB records` (local archive: `research/evidence/units/empire-20260926/installed-db-selection.json`).

## Where to find full information

| Need | Source |
|---|---|
| Tested Lua commands, constraints and actual outcomes | [Commands](../commands.md) |
| Runtime states, attributes, passive effects and ability limitations | [States](../states.md) |
| Exact scenarios, game build, raw logs and reproduction | [Evidence guide](../evidence.md) |
| Saved base unit records | `Installed DB selection` (local archive: `research/evidence/units/empire-20260926/installed-db-selection.json`): find the exact unit key in `main_units_tables` / `land_units_tables`; ability and attribute junctions are included |
| Complete base record dependency lookup | Installed `data/db.pack`: start at `db/main_units_tables` → `land_unit` → `db/land_units_tables`; follow weapon, projectile, armour, shield, entity and ability references into their tables. The saved selection is not every referenced table or a computed final unit card. |
| Final characteristics and available abilities of a particular unit in the current battle | Its in-game unit card and actual permitted runtime observations; static base records are not proof of that battle's modified values or character loadout |

Roster knowledge does not reveal hidden enemy positions, health changes or undisclosed equipment/skills; the [information contract](../../../apps/intel.md) still applies. New measurements or game updates need dated sources, not silent edits to old evidence.
