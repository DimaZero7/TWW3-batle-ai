"""The strategist's input in the game: the bridge's deployment document -> the same input (net/strategist/inputs.py)
the simulator builds before its deployment (CLAUDE.md "Вход сети один в один"; contract/parity.md, section
"стратег"). Nothing new from the game but the document (bridge/lua/apps/bridge/services.lua deploy_document):

    my units          the bridge's full rows (bridge/lua/entries/nn_arena.lua net_rows): the live part - men, health, morale,
                      fatigue, ammunition, effects - the game's
    the enemy's       its composition only: name, key, men, health (the pre-battle screen); no place, no state - it is
                      not seen; every enemy counts as seen (the composition is known)
    the zones         both deployment areas {x, z, b, front, depth} as the battle file declares them
                      (sim/arena.py zones)

The rest is the tactician's raw state (tactic_input.TacticSeer: the same columns, the same formulas) of the document laid
out as the simulator's state before the deployment is (strategist/cases.py Pre, army() without places): the input reads
no place (inputs.blind), but the raw state's rule effects (lord near, fear near) are counted from the places and the
formation's front / depth from the width - so the seer puts my units in a row across my zone's centre, 80 m apart by slot,
facing the zone's forward, at their own frontage (the simulator's default width of the key, scenario.build); the enemy's
the same way across its zone's centre (any place far from mine gives the same input). The field: the battle's captured map
(sim/maps), else flat with the simulator's bounds (sim.json map_half_m: where units leave the Crossroads flat).

StrategistSeer.input(doc, command) -> (inp, ctx, laid): once a battle (the strategist decides once).
"""
import copy

import torch

from bridge.companion import tactic_input as TI
from bridge.companion import unit_input as UI
from net.strategist import inputs as IN
from net.strategist import net as NN
from net.strategist import zones as ZS

ROW_M = 80.0       # strategist/cases.army: the units before the deployment, 80 m apart across the zone's centre
KEYS = ("x", "z", "b", "front", "depth")


def zone_of(z, device="cpu"):
    """{x, z, b, front, depth} [1] of a document's zone."""
    return {k: torch.tensor([float(z[k])], device=device) for k in KEYS}


def own_width(keys, factions, params):
    """{key: the simulator's default frontage (m)} - scenario.build's width of a unit given none (params.static)."""
    out = {}
    for key, fac in zip(keys, factions):
        if key not in out:
            out[key] = float(params.static(key, fac)["width"])
    return out


def row_places(zone, n, device="cpu"):
    """[(x, z)] of n units in a row across the zone's centre, ROW_M apart (cases.army's layout)."""
    if n == 0:
        return []
    s = (torch.arange(n, dtype=torch.float32, device=device) - (n - 1) / 2) * ROW_M
    x, z = ZS.world({k: v for k, v in zone.items()}, s[None], torch.zeros_like(s)[None])
    return list(zip(x[0].tolist(), z[0].tolist()))


def laid_out(doc, params, device="cpu"):
    """A copy of the document with every unit where the simulator has it before the deployment (module doc)."""
    out = copy.deepcopy(doc)
    zones = {1: zone_of(doc["zones"]["own"], device), 2: zone_of(doc["zones"]["enemy"], device)}
    fac = doc.get("factions") or {}
    factions = {1: fac.get("own"), 2: fac.get("enemy")}
    units = out["units"]
    width = own_width([u.get("key") or "" for u in units], [factions[int(u["side"])] for u in units], params)
    for side in (1, 2):
        rows = [u for u in units if int(u["side"]) == side]
        for u, (x, z) in zip(rows, row_places(zones[side], len(rows), device)):
            u["x"], u["z"], u["b"] = x, z, float(zones[side]["b"])
            u["ow"] = width[u.get("key") or ""]
            u["v"] = True
            for k in ("mv", "m", "f", "fire", "lf", "rf", "bf"):     # before the battle nobody moves, fights, shoots
                u[k] = False
            u.pop("t", None)
    return out


class StrategistSeer:
    """The strategist's input of a battle in the game (module doc). own: the side whose bridge wrote the document."""

    def __init__(self, map_root=UI.MAPS, device="cpu", params=None, allow_flat=False):
        self.tactic = TI.TacticSeer(map_root, device, own=1, params=params, allow_flat=allow_flat)
        self.P = self.tactic.P
        self.device = device
        self.raw = None          # the last input's raw state [1, N, RAW] (the places' code reads its sides and ranges)

    def input(self, doc, command):
        """(inp, ctx, laid): the strategist's input (inputs.build) of the deployment document, its walk's context
        (net.Ctx), the laid-out document; command [1, 7] the battle's task (tactic.decide.order_vec)."""
        laid = laid_out(doc, self.P, self.device)
        raw = self.raw = self.tactic.raw(laid)
        side = torch.ones(1, dtype=torch.long, device=self.device)
        mine, theirs = zone_of(doc["zones"]["own"], self.device), zone_of(doc["zones"]["enemy"], self.device)
        bounds = self.P.map_half
        axes = ZS.axes(mine, theirs, raw, bounds)
        inp = IN.build(raw, self.tactic.card, side, axes, mine, theirs, command.to(self.device), bounds=bounds)
        return inp, NN.Ctx(axes, mine, theirs), laid
