"""The tactician's whole input of R groups (design/tactic; the network's side: net.py):

    rows [R, K, 185], mask [R, K], slot [R, K], who [R, K]        rows.py
    near [R, 34, 30, 30], far [R, 28, 30, 30], target [R, 17, 30, 30]   windows.py (the cards' layers zeros until
                                                                  net.paint_cards fills them)
    cover {near, far, target} [R, K, 30, 30], paint {...} [R, K]   the cells' shares of the rows, which rows paint
                                                                  which card layers
    numbers [R, 4]                                                the link (3) and 'no far window' (1)

The same function on the simulator's raw state (tactic/raw.py Seer) and on the bridge's
(tools/nn/companion/tactic_input.py): the input is in the zone's axes, so a battle seen in another frame (the game's
map coordinates) gives the same numbers when the parcel is given in that frame.
"""
import torch

from tools.nn.tactic import raw as TR
from tools.nn.tactic import rows as RW
from tools.nn.tactic import windows as W


def field_of(raw, parcel):
    """{x, z, b, front, depth, on} [R]: each group's battle's playable field (terrain.frame_local of its captured map,
    in the battle's own frame; on False on a flat battle)."""
    from tools.nn.sim import terrain
    g = TR.ground_of(raw)
    terr = g["terr"][parcel.bi, 0]
    on = terr >= 0
    if not bool(on.any()):
        return None
    cx, cz, b, front, depth = terrain.frame_local(terr, [g[k][parcel.bi, 0] for k in terrain.PLACE])
    return {"x": cx, "z": cz, "b": b, "front": front, "depth": depth, "on": on}


def build(raw, card, parcel, tiles, areas=None, bounds=None, rows=None, reach_s=None):
    """raw [B, N, len(tactic.raw.RAW)], card [B, N, 69] (passport.table of the battle's keys), parcel (R groups),
    tiles (zone.Tiles of the groups: zone.Keeper keeps them between decisions), areas {x, z, b, front, depth, on}
    [R, M] (the personal areas of the tactician's last output; None: none yet), bounds: a flat battle's square half
    size (State.bounds; None: no edge). Returns the dict of the module doc (+ h0, J, cell)."""
    r = RW.build(raw, card, parcel, tiles, rows, reach_s)
    ground = {k: v[parcel.bi] for k, v in TR.ground_of(raw).items()}
    w = W.build(r, parcel, tiles, ground, areas, field_of(raw, parcel), bounds)
    return {**r, **w}
