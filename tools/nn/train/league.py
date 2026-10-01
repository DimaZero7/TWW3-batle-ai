"""Who plays whom (docs/en/training/training.md).

Every battle of the batch has a fixed layout for the whole run: its scene, the side the learner
plays and the opponent. Opponents:

    self        the learner on both sides (both sides give training data)
    past        a past version of the learner from the pool (one version for all such
                battles, drawn again every update); the untrained network is always in the pool
    nearest, hold_shoot, hold, ai_like   the scripted opponents (tools/nn/train/opponents.py)

The pool: build/nn-train/pool/*.pt in the checkpoint format (tools/nn/train/checkpoint.py).
"""
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools.nn.train import checkpoint

LEARNER = 0
OPPONENTS = ("self", "past", "nearest", "hold_shoot", "hold", "ai_like")
CODE = {name: i + 1 for i, name in enumerate(OPPONENTS)}     # controller codes; 0 = the learner
MIX = {"self": 0.1, "past": 0.15, "nearest": 0.2, "hold_shoot": 0.1, "hold": 0.05, "ai_like": 0.4}


@dataclass
class Layout:
    scene: np.ndarray       # [B] index into the scene list
    learner: np.ndarray     # [B] the learner's side, 1 or 2
    opponent: np.ndarray    # [B] opponent code (CODE)

    @property
    def B(self):
        return len(self.scene)

    def controllers(self):
        """[B, 2] who gives the orders of side 1 and side 2: LEARNER or an opponent code."""
        out = np.zeros((self.B, 2), dtype=np.int64)
        for s in (1, 2):
            mine = self.learner == s
            self_play = self.opponent == CODE["self"]
            out[:, s - 1] = np.where(mine | self_play, LEARNER, self.opponent)
        return out


def counts(B, mix):
    """Battles per opponent, proportional to mix, summing to B."""
    names = [n for n in OPPONENTS if mix.get(n, 0) > 0]
    w = np.array([mix[n] for n in names], float)
    raw = w / w.sum() * B
    n = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - n))[:B - n.sum()]:
        n[i] += 1
    return dict(zip(names, n))


ATTACK_ONLY = ("hold",)   # `hold` never attacks: as the attacker it only waits out the hour


def layout(B, n_scenes, mix=None, opponent=None, scene_attacker=None, attack_only=ATTACK_ONLY):
    """A layout of B battles: within each opponent's share the scenes and the learner's side
    cycle, so every opponent meets every (scene, side) about equally. opponent: one name for all
    battles (evaluation). scene_attacker [n_scenes] (1 or 2): with it, the opponents in attack_only
    play only the defender (the learner takes the attacker's side)."""
    per = {opponent: B} if opponent else counts(B, mix or MIX)
    scene, side, opp = [], [], []
    for name, n in per.items():
        j = np.arange(n)
        sc = j % n_scenes
        sd = 1 + (j // n_scenes) % 2
        if scene_attacker is not None and name in attack_only:
            sd = np.asarray(scene_attacker)[sc]
        scene.append(sc)
        side.append(sd)
        opp.append(np.full(n, CODE[name]))
    return Layout(np.concatenate(scene), np.concatenate(side), np.concatenate(opp))


class Pool:
    """Past versions of the learner, newest last; the untrained network stays first."""

    def __init__(self, root=checkpoint.POOL, size=8, seed=0):
        self.root = Path(root)
        self.size = size
        self.paths = []
        self.rng = random.Random(seed)

    def add(self, path):
        self.paths.append(Path(path))
        while len(self.paths) > self.size:
            self.paths.pop(1)        # keep the untrained one (first)

    def save(self, actor, critic=None, preset="small", meta=None, name=None):
        path = self.root / (name or f"v{int((meta or {}).get('update', 0)):05d}.pt")
        checkpoint.save(path, actor, critic, preset, meta)
        self.add(path)
        return path

    def sample(self):
        """(path, is the untrained one)."""
        i = self.rng.randrange(len(self.paths))
        return self.paths[i], i == 0
