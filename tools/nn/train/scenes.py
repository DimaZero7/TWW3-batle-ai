"""The battles training plays: named arenas (config/nn/arenas.json) in both roles, and the batch
of battles built from them (docs/en/training/training.md).

A scene is (arena, role of side 1): "attack" - side 1 attacks, "defend" - side 2 attacks. The
learner plays side 1 in some battles and side 2 in others (tools/nn/train/league.py), so it plays
every army in both roles. There is no random army generator yet.
"""
import numpy as np
import torch

from tools.nn.model import sources
from tools.nn.sim import scenario

SCENES = (
    ("arena", "attack"), ("arena", "defend"),                        # the Empire mirror arena
    ("whole_emp_v_skv", "attack"), ("whole_emp_v_skv", "defend"),    # Empire (side 1) against Skaven
    ("whole_skv_v_emp", "attack"), ("whole_skv_v_emp", "defend"),    # Skaven (side 1) against Empire
)


def army(scene):
    name, role = scene
    return scenario.from_arena(name, role)


def factions(armies):
    """[(faction of side 1, faction of side 2)] per battle."""
    return [(a["sides"][1]["faction"], a["sides"][2]["faction"]) for a in armies]


def per_side(scenes=SCENES):
    """Slots per side that fit every scene."""
    return max(len(army(s)["sides"][k]["units"]) for s in scenes for k in (1, 2))


def build(scene_of, scenes=SCENES, params=None, device="cpu", H=None):
    """(State, Setup) of a batch: battle b plays scenes[scene_of[b]] from its start."""
    armies = [army(scenes[int(i)]) for i in scene_of]
    st = scenario.build(armies, params, device=device, per_side=H or per_side(scenes))
    setup, _ = sources.from_sim(st, factions(armies))
    return st, setup


def reset_rows(st, template, rows):
    """Battles where rows [B] (bool tensor) is true start again from the template (in place)."""
    keep = ~rows
    for k, v in st.u.items():
        st.u[k] = torch.where(rows[:, None], template.u[k], v)
    st.t = torch.where(keep, st.t, template.t)
    st.done = torch.where(keep, st.done, template.done)
    st.winner = torch.where(keep, st.winner, template.winner)
    st.attacker = torch.where(keep, st.attacker, template.attacker)
    st.lord_dead_s = torch.where(keep[:, None], st.lord_dead_s, template.lord_dead_s)
    return st


def spread_evenly(B, n, seed=0):
    """[B] indices 0..n-1, each about B / n times, in a shuffled order."""
    idx = np.arange(B) % n
    return np.random.default_rng(seed).permutation(idx)
