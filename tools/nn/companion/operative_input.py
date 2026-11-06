"""The operative's input in the game: the bridge's state documents -> the same input (tools/nn/operative/inputs.py)
the simulator's training builds (CLAUDE.md "Вход сети один в один"; build/diag_g49/bridge_parity.md, section
"оперативник"). Nothing new from the bridge: the tactician's raw state (tactic_input.TacticSeer - the same columns, the
same formulas), the field's axes from it (operative/frame.py: the captured map's frame, fixed at the battle's first
decision), the groups - the operative's own last output (kept by the caller, not the game's).

OperativeSeer: one call a decision (decide 15 s); the raw snapshots in between go through raw() once a second (the
tactician's clocks of the effects need them).
"""
import torch

from tools.nn.companion import tactic_input as TI
from tools.nn.companion import unit_input as UI
from tools.nn.operative import frame as FR
from tools.nn.operative import inputs as OI


class OperativeSeer:
    def __init__(self, map_root=UI.MAPS, device="cpu", own=1, params=None, allow_flat=False):
        self.tactic = TI.TacticSeer(map_root, device, own, params, allow_flat)
        self.device, self.own = device, own
        self.axes = None
        self.batch = None

    def raw(self, doc):
        if doc["batch"] != self.batch:
            self.batch, self.axes = doc["batch"], None
        return self.tactic.raw(doc)

    def input(self, doc, groups, command, bounds=None, K=None):
        """The operative's input of this document: groups (operative.groups.Groups of the battle, B = 1), command [1, 7]."""
        r = self.raw(doc)
        side = torch.full((1,), self.own, dtype=torch.long, device=self.device)
        if self.axes is None:
            self.axes = FR.Axes.of(r, side, bounds)
        return OI.build(r, self.tactic.card, side, self.axes, groups, command, bounds=bounds, K=K)
