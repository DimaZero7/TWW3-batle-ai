"""Faction character (config/nn/factions.json) and the LoRA adapter index of a pair "faction + role".

An unknown faction gets the neutral character 0.5 and the last adapter (shared "other").
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
ADAPTERS = 2 * (len(KEYS) + 1)   # (faction or "other") x (attack, defend)


def character(faction):
    """The faction's character: [SIZE] float32 in the order of TRAITS."""
    c = load()["factions"].get(faction, {}).get("character", {})
    return np.array([c.get(t, 0.5) for t in TRAITS], dtype=np.float32)


def adapter(faction, attacks):
    """Index of the LoRA adapter of (faction, role): 2 * faction + (0 attack, 1 defend)."""
    f = KEYS.index(faction) if faction in KEYS else len(KEYS)
    return 2 * f + (0 if attacks else 1)
