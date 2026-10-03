"""The network's inputs and model, no training yet (docs/en/training/model.md).

    passport.py     a unit's static features from config/nn/units.json (numpy)
    abilities.py    an ability's static features from config/nn/abilities.json, a unit's slots (numpy)
    factions.py     faction character (config/nn/factions.json) (numpy)
    frame.py        the side's frame: side-symmetric coordinates (numpy or torch)
    observation.py  what one side sees; the critic's full view (numpy or torch)
    sources.py      Setup and states from recorded battles, the simulator, made-up battles
    config.py       sizes: presets small and target
    encoder.py      unit tokens, attention with masks and a distance bias (torch)
    memory.py       a GRU per token (torch)
    heads.py        order kind, move point bins, target pointer, run; sampling (torch)
    policy.py       the actor (torch)
    critic.py       the centralised critic, training only (torch)
    decide.py       one decision: observation -> actor -> the simulator's Orders (torch)
    bench.py        sizes and timings: bash tools/nn/dock.sh tools.nn.model.bench
"""
