"""Faction character (config/nn/factions.json).

An unknown faction gets the neutral character 0.5.
"""
import json
from functools import lru_cache

import numpy as np

from tools import config as project

FACTIONS = project.CONFIG_DIR / "nn" / "factions.json"


@lru_cache(maxsize=2)
def load(path=None):
    return json.loads((path or FACTIONS).read_text(encoding="utf-8"))


TRAITS = tuple(load()["traits"])
SIZE = len(TRAITS)
KEYS = tuple(load()["factions"])


def character(faction):
    """The faction's character: [SIZE] float32 in the order of TRAITS."""
    c = load()["factions"].get(faction, {}).get("character", {})
    return np.array([c.get(t, 0.5) for t in TRAITS], dtype=np.float32)

