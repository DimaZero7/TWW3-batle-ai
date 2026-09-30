"""The companion: the network outside the game commands our side of a real battle
(docs/en/apps/bridge.md, docs/en/launch/watch.md).

    exchange.py  the state and orders files: read, build the observation's input, write (numpy)
    policy.py    the actor from a checkpoint, or a fresh untrained one (torch)
    loop.py      wait for a state, decide, answer; python -m tools.nn.companion --game <folder>
"""
