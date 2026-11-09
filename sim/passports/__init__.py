"""Passports: the game's database (<game>/data/db.pack, the required mods' packs) -> sim/config/*.json.

    py -3.14 -m sim.passports.units --units k1,k2     units.json (+ probe-units.json)
    py -3.14 -m sim.passports.abilities               abilities.json
    python -m sim.passports.effects                   effects.json (from units.json, abilities.json)
    py -3.14 -m sim.passports.gamedb                  game_rules.json
    python -m sim.passports.dbschema                  db_schema.json (the RPFM schema in tmp/rpfm/)

The game's folders and the required mods come from the project's common settings (bridge.config: game_dir,
workshop_dir, required_mods); only these generators need them - the simulator itself reads sim/config alone.
"""


def game_settings():
    """{game_dir, workshop_dir, ...} of this machine (the project's common settings)."""
    from bridge import config          # (the bridge's launcher owns them)
    return config.load()


def required_mods():
    """[{pack, workshop_id, sha256}] the mods every battle loads, in load order (True Sight)."""
    from bridge import config
    return config.required_mods()
