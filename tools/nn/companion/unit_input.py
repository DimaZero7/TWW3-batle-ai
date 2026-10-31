"""The unit network's raw state in the game: the bridge's state document -> view.RAW (tools/nn/unit/view.py) by the same
formulas view.raw uses on a simulator state (CLAUDE.md "Вход сети один в один"; build/diag_g49/bridge_parity.md).
Only what the game gives:

    the bridge's row (src/entries/nn_arena.lua net_rows): x, z, b, men, hp, mp, ms, r, s, w, m, mv, fire, a, fat, v,
        side, key; ow = unit:ordered_width() (the formation's front, depth and fill: geometry.dims of it and the men, as
        view.raw's zmap.rects_of of the simulator's width)
    the document: t (battle ms), map (the battle's terrain path: the map captured once, build/maps/<name>/ - the same
        heights the simulator reads; a battle on a map not captured gets no unit network)
    the passport's static numbers of the key (params.static, as scenario.build gives a simulator unit): spacing, the
        men's radius, run and walk, range, fire arc, ammunition

Not the same as the simulator's yet (view.py reads the simulator's state there; changing view.py waits for the user,
bridge_parity.md): vx / vz - the game has no speed: here the centre's shift since the previous document, 0 for a unit
not moving in both or in melee in both (the game's centre drifts 0.5-0.7 m/s in melee), against the simulator's
instantaneous velocity; f - moving and faster than the walk + 0.3 m/s by that shift (exchange.running_by_speed), against
the simulator's by its velocity; front / depth during a re-form - the game's ordered width, the simulator's width in
force.

UnitSeer keeps the previous document's positions: one call a decision (decide_ms 1000). Numpy in, torch out.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from tools import config as project
from tools.nn.companion import exchange

MAPS = project.ROOT / "build" / "maps"
STATIC = ("men0", "sp_h", "sp_v", "radius", "body_ratio", "eng0", "walk", "run", "range", "arc", "fire_move", "ammo0")
RUN_MARGIN = exchange.RUN_MARGIN


def map_folder(terrain_path, root=MAPS):
    """The captured map folder (build/maps/<name>) whose meta.json "map" is the battle's terrain path (the document's
    "map"; case and slashes ignored), or a folder of that name; None: not captured."""
    want = str(terrain_path or "").replace("/", "\\").strip("\\").lower()
    root = Path(root)
    if not want or not root.exists():
        return None
    for meta in sorted(root.glob("*/meta.json")):
        m = json.loads(meta.read_text(encoding="utf-8")).get("map") or ""
        if m.replace("/", "\\").strip("\\").lower() == want:
            return meta.parent
    folder = root / want.split("\\")[-1]
    return folder if (folder / "heights.npz").exists() else None


_LOADED = {}


def load_map(folder):
    """The map's id in the simulator's registry (terrain.load_npz: heights, ground, trunks, 'cannot stand' - passable.npz,
    not in the network's input yet: build/diag_g49/bridge_parity.md - its field), once a process."""
    from tools.nn.sim import terrain
    key = str(Path(folder).resolve())
    if key not in _LOADED:
        _LOADED[key] = terrain.load_npz(Path(folder) / "heights.npz")
    return _LOADED[key]


@lru_cache(maxsize=512)
def _static_of(key):
    from tools.nn.sim.params import load
    P = load()
    if key not in P.units:
        return None
    s = P.static(key)
    return tuple(float(s[k]) for k in STATIC)


def static_game(keys, device="cpu"):
    """({STATIC: [1, N]}, known [N] bool) of the units by key (params.static); a key the simulator does not know: zeros
    and known False (no unit network for it)."""
    N = len(keys)
    out = np.zeros((len(STATIC), N), np.float32)
    known = np.zeros(N, bool)
    for i, k in enumerate(keys):
        s = _static_of(k or "")
        if s is not None:
            out[:, i] = s
            known[i] = True
    st = {k: torch.tensor(out[j][None], device=device) for j, k in enumerate(STATIC)}
    st["fire_move"] = st["fire_move"] > 0.5
    return st, known


def game_raw(doc, battle, static, prev=None, ground=None, leash=None, params=None, device="cpu"):
    """([1, N, len(view.RAW)], the reference for the next call): the raw state of the document (battle: exchange.Battle;
    prev: (x, z, mv, m, t) of the previous call or None; ground: {terr, tpx, tpz, tpr} [1, N] of the captured map or
    None: flat; leash [1, N] bool: the unit's order replaced by the leash - the companion's decoder's)."""
    from tools.nn.sim import geometry, terrain
    from tools.nn.unit import view as V
    if params is None:
        from tools.nn.sim.params import load
        params = load()
    names = battle.names
    rows = {u["n"]: u for u in doc["units"]}
    a = exchange.arrays(doc, names)
    f = lambda v, nan=0.0: torch.tensor(np.nan_to_num(np.asarray(v, np.float32), nan=nan), device=device)  # noqa: E731
    b = lambda v: torch.tensor(np.asarray(v, bool), device=device)                                       # noqa: E731
    x, z = f(a["x"]), f(a["z"])
    men = f(a["men"])
    width = torch.tensor([[float(rows.get(n, {}).get("ow") or 0.0) for n in names]], device=device)
    side = torch.tensor(np.asarray(battle.side, np.float32)[None], device=device)
    mv, m = b(a["mv"]), b(a["m"])
    alive = (side > 0) & (men > 0)
    # the formation: view.raw's zmap.rects_of - geometry.dims of the width and the men (melee_depth 0, as view.raw's)
    g = {"men": men, "width": width, **{k: static[k] for k in ("men0", "sp_h", "sp_v", "radius", "body_ratio", "eng0")}}
    spacing = params.sim["formation"]["spacing_m"]
    front, depth = geometry.dims(g, spacing, 0.0)
    h = torch.where(static["sp_h"] > 0, static["sp_h"], torch.full_like(front, float(spacing)))
    v_ = torch.where(static["sp_v"] > 0, static["sp_v"], torch.full_like(front, float(spacing)))
    places = (front / h).round() * (depth / v_).round()
    fill = (men.clamp(min=0) / places.clamp(min=1)).clamp(max=1)
    fill = torch.where(static["men0"] <= 1, (men > 0).float(), fill)
    # the velocity: the centre's shift since the previous document (module doc)
    t = float(doc.get("t", 0)) / 1000.0
    vx, vz = torch.zeros_like(x), torch.zeros_like(x)
    if prev is not None and t > prev[4]:
        dt = t - prev[4]
        still = (~mv & ~prev[2]) | (m & prev[3]) | ~alive
        vx = torch.where(still, vx, (x - prev[0]) / dt)
        vz = torch.where(still, vz, (z - prev[1]) / dt)
    run = mv & (torch.sqrt(vx * vx + vz * vz) > static["walk"] + RUN_MARGIN)
    ammo0 = static["ammo0"]
    a_ = f(a["a"])
    cols = {"x": x, "z": z, "b": f(a["b"]), "vx": vx, "vz": vz, "front": front, "depth": depth, "fill": fill,
            "men": men / static["men0"].clamp(min=1), "hp": f(a["hp"]), "mp": f(a["mp"], 1.0), "ms": f(a["ms"]),
            "r": b(a["r"]), "s": b(a["s"]), "w": b(a["w"]), "m": m, "mv": mv, "f": run, "fire": b(a["fire"]),
            "fat": f(a["fat"]), "ammo": torch.where(ammo0 > 0, a_ / ammo0.clamp(min=1), torch.zeros_like(a_)),
            "alive": alive, "vis": b(a["vis"]), "side": side, "range": static["range"], "run": static["run"],
            "leash": leash if leash is not None else torch.zeros_like(alive),
            "arc": V.fire_arc({"x": x, "range": static["range"], "arc": static["arc"], "fire_move": static["fire_move"],
                               "mv": mv}, params.sim["missile"])}
    if ground is not None:
        cols["gh"] = terrain.height(ground, x, z)
        cols["terr"] = ground["terr"].float()
        for k in terrain.PLACE:
            cols[k] = ground[k]
    else:
        cols["gh"], cols["terr"] = torch.zeros_like(x), torch.full_like(x, -1.0)
        for k in terrain.PLACE:
            cols[k] = torch.zeros_like(x)
    raw = torch.stack([cols[k].float() for k in V.RAW], -1)
    return raw, (x, z, mv, m, t)


class UnitSeer:
    """The raw state of every unit of a battle, one call a decision (module doc)."""

    def __init__(self, map_root=MAPS, device="cpu", params=None, allow_flat=False):
        """allow_flat: a battle whose map is not captured runs flat (tests only: in the game the unit network needs its
        map - its heights and its field)."""
        from tools.nn.sim.params import load
        self.map_root, self.device, self.allow_flat = map_root, device, allow_flat
        self.P = params or load()
        self.batch = self.prev = self.battle = self.static = self.known = self.ground = None

    def start(self, doc):
        self.battle = exchange.battle(doc)
        keys = [u.get("key") or "" for u in doc["units"]]
        self.static, self.known = static_game(keys, self.device)
        self.batch, self.prev, self.ground = doc["batch"], None, None
        folder = map_folder(doc.get("map"), self.map_root)
        if folder is None:
            if self.allow_flat:
                return
            raise ValueError(f"the battle's map {doc.get('map')!r} is not captured in {self.map_root}: the unit network "
                             f"needs it (capture it once: python -m tools.nn.map_grid capture --name <name>)")
        mid = load_map(folder)
        N = len(keys)
        zero = torch.zeros((1, N), device=self.device)
        self.ground = {"terr": torch.full((1, N), mid, dtype=torch.long, device=self.device), "tpx": zero,
                       "tpz": zero.clone(), "tpr": zero.clone()}

    def raw(self, doc, leash=None):
        """[1, N, len(view.RAW)] of this document (the battle's units in the document's order)."""
        if doc["batch"] != self.batch:
            self.start(doc)
        raw, self.prev = game_raw(doc, self.battle, self.static, self.prev, self.ground, leash, self.P, self.device)
        return raw
