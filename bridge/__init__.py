"""The bridge between the network and the game (bridge/README.md, bridge/docs/README.md).

    lua/          the mod: apps/* (modules), entries/* (6 entries: nn_arena, the 3 probes, map_capture, move_probe)
    pack/, build.py   python -m bridge.build <target>: the Lua bundled into one script and a .pack (build/<target>/)
    launcher/     install the pack, start the game (fair Normal difficulty, graphics), wait, clean up
    companion/    the helper in Docker: the chain (operative -> tacticians -> unit network), the strategist's deployment
    config.py     the game's folders, the required mods (bridge/config/*.json); mem_sites.py memreader Plus (probes)
"""
