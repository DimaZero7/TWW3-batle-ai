"""The general's map input, "Ландшафт" (design/general/index.html): 6 layers on a 128 x 128 grid of 16 m cells.

The grid is 2048 m square; the map keeps its size in cells and stands in the middle of the grid (the grid's
centre is the map's centre), the cells around it are "outside". A map bigger than the grid shows its middle.
Array index [layer, iz, ix]: ix along +x, iz along +z, cell (0, 0) has its low corner at `origin` (as side 1
sees it).

Layers (numbers fixed; new terrain types will be added at the end):
    0  field          share 0..1 of the cell inside the playable area
    1  outside        share 0..1 beyond the map's edge (= 1 - field while there are no other types)
    2  our zone       flag: the cell lies wholly inside our deployment zone
    3  enemy zone     flag: the same for the enemy's zone
    4  our vanguard   flag: the cell lies wholly in the field and wholly clear of the band of band_m around
                      the enemy's zone (where our vanguard units may deploy)
    5  enemy vanguard flag: the same around our zone
"Ours" is the side the network plays. Side 2 sees the map turned half round its centre, as the network's side
frame turns the world (tools/nn/model/frame.py: turning the world by 180 degrees and swapping the sides gives
the same numbers): unpack(side=2) flips both grid axes and swaps 2<->3, 4<->5. The grid is centred on the map,
so the turn is an exact flip of the cells.

Stored once per map (`build/general/maps/<name>.npz`): the type shares one byte each (bytes of a cell sum to
255 exactly), the 4 flags in one byte (FLAG_* bits, by side) -> 3 bytes per cell; unpack() gives [6, 128, 128]
float32, to_tensor() the same as a torch tensor. Battles whose zones differ per battle (drills): map_batch.py
builds the same layers on the fly with torch, many battles at once.

Our arena (MP Crossroads flat, tools/nn/scenario.py) - what the battles really use:
* field: the square of +-map_half_m (config/nn/sim.json, 1020 m: routing units leave the map there, measured;
  the simulator's edge). The grid's +-1024 m leaves a 4 m strip outside (field share 0.75) on every side.
* zones: the battle file's <deployment_area> of each alliance (scenario.scenario_xml): a deployment_m square,
  reaching ZONE_FRONT_M in front of the side's front line (gap_m apart), sharp corners.
* band: BAND_M (the arena's battle file sets none; our battles place every unit by script).

    python -m tools.nn.general.map_input                # build/general/maps/arena.npz
    python -m tools.nn.general.map_input --arena whole_emp_v_skv --out DIR --png
"""
import argparse
import json
import math
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np

from tools import config as project
from tools.nn.general import geometry as geo

CELL_M = 16.0
GRID = 128
SIZE_M = CELL_M * GRID                # 2048 m: the arena's +-1020 m and every field map of the game (<= 1536 m)
VERSION = 2
TYPES = ("field", "outside")          # share layers, one byte each in the file
FLAG_ZONE = {1: 1, 2: 2}              # the cell is wholly inside side s's zone
FLAG_VANGUARD = {1: 4, 2: 8}          # side s's vanguard may deploy in the whole cell
LAYERS = ("field", "outside", "our_zone", "enemy_zone", "our_vanguard", "enemy_vanguard")
# The band around a zone where the other side's vanguard may not deploy. The game's maps give it as a
# "guerrilla exclusion" polygon per zone: 64 m around the zone in 210 of the 257 zones whose band is even on
# all four sides (catchment_*_layer_bmd_data.bin of tile*.pack, DAC_DEFAULT; others 0, 57.6, 96 m). The arena's
# battle file sets none, so the arena takes the game's usual width.
BAND_M = 64.0
OUT_DIR = project.BUILD / "general" / "maps"
SIM = project.CONFIG_DIR / "nn" / "sim.json"


def map_half():
    """The simulator's map half-size, m (config/nn/sim.json map_half_m: where routing units leave)."""
    return float(json.loads(SIM.read_text(encoding="utf-8"))["map_half_m"]["value"])


def arena_field():
    """(xmin, xmax, zmin, zmax) of the arena's playable area: the simulator's square."""
    h = map_half()
    return (-h, h, -h, h)


@dataclass
class MapSpec:
    """A map in world metres. field_box (xmin, xmax, zmin, zmax) or field_poly [M, 2]: the playable area;
    zones {1: poly, 2: poly}: the deployment zones of side 1 and side 2; centre: the map's centre (default:
    the field's middle)."""
    name: str
    zones: dict
    band_m: float = BAND_M
    field_box: tuple = None
    field_poly: np.ndarray = None
    centre: tuple = None
    source: dict = dc_field(default_factory=dict)

    def middle(self):
        if self.centre is not None:
            return tuple(map(float, self.centre))
        if self.field_box is not None:
            x0, x1, z0, z1 = self.field_box
        else:
            p = np.asarray(self.field_poly, float)
            x0, x1, z0, z1 = p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()
        return (float(x0 + x1) / 2, float(z0 + z1) / 2)


def origin(spec):
    """World (x, z) of the grid's low corner: the map's centre minus half the grid."""
    cx, cz = spec.middle()
    return cx - SIZE_M / 2, cz - SIZE_M / 2


def grid(spec):
    x0, z0 = origin(spec)
    return geo.cells(x0, z0, GRID, CELL_M)


def cell_of(org, x, z):
    """(iz, ix) of the cell holding the world point (x, z) for a grid with low corner org."""
    return int(math.floor((z - org[1]) / CELL_M)), int(math.floor((x - org[0]) / CELL_M))


def field_share(spec, g=None):
    """[128, 128] the part of each cell inside the playable area."""
    g = g if g is not None else grid(spec)
    if spec.field_box is not None:
        return geo.share_box(spec.field_box, g)
    return geo.share(spec.field_poly, g)


def quantise(shares):
    """[T, n, n] shares summing to 1 -> uint8 bytes summing to exactly 255 per cell (largest remainder)."""
    s = np.asarray(shares, float)
    s = s / np.clip(s.sum(0, keepdims=True), 1e-12, None)
    raw = s * 255.0
    base = np.floor(raw).astype(np.int64)
    short = 255 - base.sum(0)
    order = np.argsort(-(raw - base), axis=0, kind="stable")
    for k in range(s.shape[0]):           # the k-th largest remainder gets +1 while bytes are short
        add = (short > k)
        idx = order[k]
        rows = np.take_along_axis(base, idx[None], 0)[0] + add
        np.put_along_axis(base, idx[None], rows[None], 0)
    return base.astype(np.uint8)


def flags(spec, g=None, field=None):
    """[128, 128] uint8 of FLAG_* bits."""
    g = g if g is not None else grid(spec)
    full = (field if field is not None else field_share(spec, g)) >= 1.0 - geo.EPS
    out = np.zeros((GRID, GRID), np.uint8)
    for side, other in ((1, 2), (2, 1)):
        zone = geo.wholly_inside(spec.zones[side], g) & full
        free = geo.clear_of(spec.zones[other], spec.band_m, g) & full
        out |= np.where(zone, FLAG_ZONE[side], 0).astype(np.uint8)
        out |= np.where(free, FLAG_VANGUARD[side], 0).astype(np.uint8)
    return out


def build(spec):
    """The map's file content: {"cells": uint8 [len(TYPES) + 1, 128, 128], "meta": dict}."""
    g = grid(spec)
    fs = field_share(spec, g)
    shares = quantise(np.stack([fs, 1.0 - fs]))
    cells = np.concatenate([shares, flags(spec, g, fs)[None]])
    meta = {"version": VERSION, "name": spec.name, "cell_m": CELL_M, "grid": GRID, "origin": list(origin(spec)),
            "types": list(TYPES), "flag_zone": FLAG_ZONE, "flag_vanguard": FLAG_VANGUARD, "band_m": spec.band_m,
            "layers": list(LAYERS), "source": spec.source}
    return {"cells": cells, "meta": meta}


def save(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, cells=data["cells"], meta=np.array(json.dumps(data["meta"], ensure_ascii=False)))
    return path


def load(path):
    with np.load(Path(path)) as z:
        return {"cells": z["cells"].copy(), "meta": json.loads(str(z["meta"]))}


def half_turn(layers):
    """Layers [..., n, n] turned half round the grid's centre (both grid axes flipped)."""
    return layers[..., ::-1, ::-1]


def unpack(data, side=1):
    """[6, 128, 128] float32 for the side the network plays (1 or 2): side 2 sees the map half-turned."""
    assert side in (1, 2)
    cells = data["cells"]
    t = len(data["meta"]["types"]) if "meta" in data else len(TYPES)
    shares = cells[:t].astype(np.float32) / 255.0
    f = cells[t]
    other = 3 - side
    bits = (FLAG_ZONE[side], FLAG_ZONE[other], FLAG_VANGUARD[side], FLAG_VANGUARD[other])
    fl = np.stack([(f & b) != 0 for b in bits]).astype(np.float32)
    out = np.concatenate([shares, fl])
    return np.ascontiguousarray(half_turn(out)) if side == 2 else out


def to_tensor(data, side=1, device=None):
    """unpack() as a torch float tensor [6, 128, 128]."""
    import torch
    return torch.from_numpy(unpack(data, side)).to(device or "cpu")


# --- our arena ------------------------------------------------------------------------------------------

def arena_rects(arena):
    """{1: rect, 2: rect} (geometry.Rect params: centre, forward unit, half depth, half width) - the
    <deployment_area> squares the battle file gives each alliance (tools/nn/scenario.py scenario_xml: centre =
    the side's front anchor moved back by deployment_m / 2 - ZONE_FRONT_M, side deployment_m)."""
    from tools.nn import scenario
    from tools.nn.armies.place import ZONE_FRONT_M
    out = {}
    for side, key in ((1, "own"), (2, "enemy")):
        ax, az, fwd, _, _ = scenario.frame(key, arena["gap_m"])
        back = arena["deployment_m"] / 2 - ZONE_FRONT_M
        half = arena["deployment_m"] / 2
        out[side] = (ax - fwd[0] * back, az - fwd[1] * back, fwd[0], fwd[1], half, half)
    return out


def arena_zones(arena):
    """{1: poly, 2: poly} of arena_rects."""
    return {s: geo.rect_corners(*r) for s, r in arena_rects(arena).items()}


def arena_spec(name=None):
    """MapSpec of an arena of tools/nn/scenario.py (default: config/nn/arena.json)."""
    from tools.nn import scenario
    arena = scenario.load_arena(name)
    return MapSpec(name=arena.get("name", "arena"), zones=arena_zones(arena), field_box=arena_field(),
                   centre=(0.0, 0.0),
                   source={"map": arena["map"], "catchment": arena["catchment"], "gap_m": arena["gap_m"],
                           "deployment_m": arena["deployment_m"],
                           "field": "the simulator's square +-map_half_m, config/nn/sim.json",
                           "zones": "battle file <deployment_area>, tools/nn/scenario.py",
                           "band": "BAND_M, the game's usual guerrilla exclusion width"})


def example_spec():
    """A made-up map for the tests and the check picture: a 1000 x 600 m field whose edges cut cells,
    zones with rounded corners (radius 64 m as the game's), the game's band."""
    return MapSpec(name="example", field_box=(-500.0, 500.0, -300.0, 300.0), centre=(0.0, 0.0),
                   zones={1: geo.rounded_rect(-300.0, 0.0, 250.0, 420.0, 64.0),
                          2: geo.rounded_rect(300.0, 0.0, 250.0, 420.0, 64.0)},
                   source={"made_up": True})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arena", default=None, help="an arena of config/nn/arenas.json (default: arena.json)")
    ap.add_argument("--example", action="store_true", help="the made-up example map instead")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--png", action="store_true", help="also the check picture next to the file")
    a = ap.parse_args(argv)
    spec = example_spec() if a.example else arena_spec(a.arena)
    data = build(spec)
    path = save(data, Path(a.out) / f"{spec.name}.npz")
    print(path, f"{path.stat().st_size} bytes")
    if a.png:
        from tools.nn.general import render
        print(render.layers_png(data, spec, Path(a.out) / f"{spec.name}.png"))


if __name__ == "__main__":
    main()
