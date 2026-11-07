"""The page's background of a captured map (numpy + Pillow, no torch): build/maps/<name>/ -> a PNG over the map's
height grid (x0..x0 + (C - 1) step, z0..; the top row the smallest z: the game's view from above - its axes x,
height y, z are right-handed, facing +x the +z is on the right; with the largest z on top the map was a mirror),
cached in build/label/maps/<name>_game.png.

    the ground's height as grey (low dark, high light) with a light hill shade, contour lines every 10 m,
    forest green, shallow water light blue, deep water blue, 'cannot stand' (passable.npz) dark red, the trunks dots
picture(name) -> (png bytes, {min_x, max_x, min_z, max_z}: the picture's extent in world metres)
"""
import io
import json

import numpy as np

from tools import config as project
from tools.label import cfg
from tools.label import matchup as MU
from tools.label import pool as PL

SIZE = 768
VIEW = "_game"         # (the cache's mark: the pictures since 08.11 are in the game's view, z down - before mirrored)
COLORS = {"forest": (60, 110, 50), "shallow_water": (110, 160, 210), "deep_water": (40, 80, 160),
          "mud": (120, 100, 70), "snow": (235, 235, 240), "rock": (140, 130, 120), "sharp_stones": (150, 140, 120)}


def _render(folder):
    from PIL import Image
    d = np.load(folder / "heights.npz")
    h = d["height"].astype(np.float32)
    rows, cols = h.shape
    step = float(d["step"])
    k = max(1, int(np.ceil(max(rows, cols) / SIZE)))
    hs = h[::k, ::k]
    lo, hi = float(np.percentile(hs, 1)), float(np.percentile(hs, 99))
    v = np.clip((hs - lo) / max(hi - lo, 1e-3), 0, 1)
    gz, gx = np.gradient(hs, step * k)
    shade = np.clip(0.5 + (-gx * 0.7 - gz * 0.7) * 0.8, 0, 1)
    base = (60 + 150 * v) * (0.75 + 0.5 * (shade - 0.5))
    rgb = np.stack([base, base, base], -1)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    names = meta.get("ground_names") or []
    if "ground" in d.files:
        g = d["ground"][::k, ::k]
        for code, name in enumerate(names):
            if name in COLORS:
                m = g == code
                rgb[m] = 0.35 * rgb[m] + 0.65 * np.array(COLORS[name], np.float32)
    if (folder / "passable.npz").exists():
        b = np.load(folder / "passable.npz")["blocked"][::k, ::k] > 0
        rgb[b] = 0.55 * rgb[b] + 0.45 * np.array((120, 30, 30), np.float32)
    lev = np.floor(hs / 10.0)
    edge = (np.abs(np.diff(lev, axis=0, prepend=lev[:1])) > 0) | (np.abs(np.diff(lev, axis=1, prepend=lev[:, :1])) > 0)
    rgb[edge] = rgb[edge] * 0.7
    if (folder / "trees.npz").exists():
        t = np.load(folder / "trees.npz")
        ix = ((t["x"] - float(d["x0"])) / (step * k)).astype(int)
        iz = ((t["z"] - float(d["z0"])) / (step * k)).astype(int)
        ok = (ix >= 0) & (ix < rgb.shape[1]) & (iz >= 0) & (iz < rgb.shape[0])
        rgb[iz[ok], ix[ok]] = (25, 60, 25)
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))           # the top row: the smallest z (the game's view)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    ext = {"min_x": float(d["x0"]), "max_x": float(d["x0"]) + (cols - 1) * step, "min_z": float(d["z0"]),
           "max_z": float(d["z0"]) + (rows - 1) * step}
    return buf.getvalue(), ext


def picture(name):
    if MU.safe(name) != name:
        raise ValueError(f"bad map name {name!r}")
    folder = PL.MAPS / name
    cache = project.ROOT / cfg()["out"] / "maps"
    png, js = cache / f"{name}{VIEW}.png", cache / f"{name}{VIEW}.json"
    src = folder / "heights.npz"
    if png.exists() and js.exists() and png.stat().st_mtime >= src.stat().st_mtime:
        return png.read_bytes(), json.loads(js.read_text(encoding="utf-8"))
    data, ext = _render(folder)
    cache.mkdir(parents=True, exist_ok=True)
    png.write_bytes(data)
    js.write_text(json.dumps(ext), encoding="utf-8")
    return data, ext

