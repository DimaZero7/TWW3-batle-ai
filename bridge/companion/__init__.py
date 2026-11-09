"""The companion: the chain outside the game commands our side of a real battle (bridge/docs/apps/bridge.md,
bridge/docs/launch/watch.md). Runs in the snake-ai-trainer container (torch).

    exchange.py          the state and orders files: read, the raw arrays of a document, the orders' text, write
    unit_input.py        the unit network's raw state of a document (view.RAW, the simulator's formulas)
    tactic_input.py      the tactician's raw state (+ effects, buttons); operative_input.py the operative's input
    strategist_input.py  the strategist's input of the deployment document; deploy.py its plan (the places)
    chain.py             the operative -> the tacticians -> the unit network on the documents, as the simulator's chain
    loop.py              wait for a state, decide, answer; python -m bridge.companion --game <folder> --chain
"""
