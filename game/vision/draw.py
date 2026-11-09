"""The strip: frames of a battle from above in one PNG (numpy + Pillow, no torch).

The view: up - towards the enemy, right - our right flank (record.Battle.view); one crop for all frames (the standing
units of the chosen frames + a margin); 1 px = crop / frame size. Each frame:
  ground   - height grey (light = high), contours every 5 m (dark every 20 m), forest green, 'cannot stand' dark red (sim/maps/<map>/)
  units    - boxes of their real front x depth and facing; ours blue, the enemy's red; fill strength = men left;
             a letter for the role (L lord, S sniper, A artillery, M other missile, F fast, X monster; infantry none), ours with the
             unit's number (S2 = long_rifles_2)
  melee    - black rim; routing - yellow fill; a unit's last 15 s of way - a thin line behind it
  fire     - a thin line from a firing unit to its target (ours light blue, the enemy's pink)
  groups   - (the chain) the group's zone a cross '<n> <verb>', its units the digit <n>, a dashed tether from a unit
             60 m or more off its zone: decided vs done; without groups - our units' order points (dashed) 25 m+ away
  caption  - second, the moment's names, standing units of each side
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim.maps import MAPS

OWN, ENEMY = (35, 85, 230), (215, 35, 35)
MELEE, ROUT = (0, 0, 0), (255, 230, 40)
FIRE = {1: (120, 200, 255), 2: (255, 150, 190)}
GROUP = [(140, 40, 200), (0, 150, 150), (200, 120, 0), (90, 90, 90), (200, 0, 120), (40, 140, 40), (0, 90, 180),
         (150, 90, 40)]
LETTER = {"lord": "L", "sn": "S", "art": "A", "sh": "M", "fast": "F", "mon": "X", "inf": ""}


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                       # Pillow < 10.1
        return ImageFont.load_default()


def crop(fx, secs, margin=30.0):
    """(lat0, lat1, fwd0, fwd1) of the view in metres: the standing units of the chosen frames (2-98 percentiles of
    each side, so a few strays do not shrink the picture)."""
    B = fx.B
    lat, fwd = [], []
    for own in (True, False):
        pts = []
        for s in secs:
            f = B.frames[B.at(s)]
            pts += [B.view(u["x"], u["z"]) for n, u in f.items()
                    if B.alive(u) and not u.get("r") and (B.side[n] == B.own) == own]
        if pts:
            r = np.array([p[0] for p in pts])
            u = np.array([p[1] for p in pts])
            lat += [np.percentile(r, 2), np.percentile(r, 98)]
            fwd += [np.percentile(u, 2), np.percentile(u, 98)]
    return min(lat) - margin, max(lat) + margin, min(fwd) - margin, max(fwd) + margin


class Canvas:
    def __init__(self, fx, box, size):
        self.fx, self.B = fx, fx.B
        l0, l1, f0, f1 = box
        self.s = size / max(l1 - l0, f1 - f0)                    # px a metre
        self.l0, self.f1 = l0, f1
        self.w, self.h = int(round((l1 - l0) * self.s)), int(round((f1 - f0) * self.s))
        self.ground = self._ground()

    def px(self, x, z):
        r, f = self.B.view(x, z)
        return (r - self.l0) * self.s, (self.f1 - f) * self.s

    def _ground(self):
        B = self.B
        rr = self.l0 + (np.arange(self.w) + 0.5) / self.s
        ff = self.f1 - (np.arange(self.h) + 0.5) / self.s
        R, Fw = np.meshgrid(rr, ff)
        X = R * B.rgt[0] + Fw * B.fwd[0]                          # the view's axes are orthonormal: world = transpose
        Z = R * B.rgt[1] + Fw * B.fwd[1]
        rgb = np.full((self.h, self.w, 3), 205.0)
        if B.map:
            d = np.load(MAPS / B.map / "heights.npz")
            hg = d["height"].astype(np.float32)
            x0, z0, st = float(d["x0"]), float(d["z0"]), float(d["step"])
            ix = np.clip(np.round((X - x0) / st).astype(int), 0, hg.shape[1] - 1)
            iz = np.clip(np.round((Z - z0) / st).astype(int), 0, hg.shape[0] - 1)
            h = hg[iz, ix]
            lo, hi = np.percentile(h, 2), np.percentile(h, 98)
            v = np.clip((h - lo) / max(hi - lo, 1.0), 0, 1)
            gy, gx = np.gradient(h)
            shade = np.clip(1.0 - 0.06 * (gx + gy), 0.8, 1.15)
            low, high = np.array((150, 160, 175)), np.array((250, 240, 215))    # low blue-grey, high sand
            rgb = (low + (high - low) * v[..., None]) * shade[..., None]
            for step, k in ((5.0, 0.85), (20.0, 0.6)):
                lev = np.floor(h / step)
                edge = (np.abs(np.diff(lev, axis=0, prepend=lev[:1])) > 0) | (np.abs(np.diff(lev, axis=1, prepend=lev[:, :1])) > 0)
                rgb[edge] *= k
            if "ground" in d.files:
                import json
                names = json.loads((MAPS / B.map / "meta.json").read_text(encoding="utf-8")).get("ground_names") or []
                if "forest" in names:
                    m = d["ground"][iz, ix] == names.index("forest")
                    rgb[m] = 0.3 * rgb[m] + 0.7 * np.array((70, 150, 60))
            if (MAPS / B.map / "passable.npz").exists():
                m = np.load(MAPS / B.map / "passable.npz")["blocked"][iz, ix] > 0
                rgb[m] = 0.6 * rgb[m] + 0.4 * np.array((130, 40, 40))
        return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))

    def box(self, n, u):
        fr, dp = self.B.dims(n, u)
        fr, dp = max(fr, 7 / self.s), max(dp, 4 / self.s)        # not smaller than a few pixels
        b = math.radians(u.get("b") or 0)
        fx, fz = math.sin(b), math.cos(b)
        rx, rz = -math.cos(b), math.sin(b)
        out = []
        for a, c in ((-1, 1), (1, 1), (1, -1), (-1, -1)):
            out.append(self.px(u["x"] + a * fr / 2 * rx + c * dp / 2 * fx, u["z"] + a * fr / 2 * rz + c * dp / 2 * fz))
        return out

    def frame(self, sec, labels, small):
        B, fx = self.B, self.fx
        i = B.at(sec)
        f = B.frames[i]
        img = self.ground.copy().convert("RGBA")
        lay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        fnt = font(10 if small else 12)
        j = B.at(sec - 15)
        for n, u in f.items():                                    # the way of the last 15 s
            p = B.frames[j].get(n)
            if B.alive(u) and p:
                c = OWN if B.side[n] == B.own else ENEMY
                d.line([self.px(p["x"], p["z"]), self.px(u["x"], u["z"])], fill=c + (150,), width=1)
        for n, u in f.items():                                    # fire
            if B.alive(u) and u.get("fire") and u.get("t") in f:
                v = f[u["t"]]
                side = 1 if B.side[n] == B.own else 2
                d.line([self.px(u["x"], u["z"]), self.px(v["x"], v["z"])], fill=FIRE[side] + (170,), width=1)
        gs = [g for t, g in B.groups if t <= sec]
        tag = {}
        if gs:                                                    # the chain's groups: decided (zone +) vs done (units)
            for g in gs[-1]:
                col = GROUP[g["g"] % len(GROUP)]
                zx, zy = self.px(g["cx"], g["cz"])
                zx, zy = min(max(zx, 6), self.w - 6), min(max(zy, 20), self.h - 6)
                for m in g["units"]:
                    u = f.get(m)
                    if not B.alive(u) or u.get("r"):
                        continue
                    tag[m] = (g["g"], col)
                    ux, uy = self.px(u["x"], u["z"])
                    if math.hypot(ux - zx, uy - zy) > 60 * self.s:
                        _dashed(d, (ux, uy), (zx, zy), col + (200,))
                d.line([(zx - 6, zy), (zx + 6, zy)], fill=col + (255,), width=3)
                d.line([(zx, zy - 6), (zx, zy + 6)], fill=col + (255,), width=3)
                d.text((zx + 5, zy - 2), f"{g['g']} {g['verb']}", fill=col + (255,), font=fnt, stroke_width=1,
                       stroke_fill=(255, 255, 255, 255))
        else:                                                     # no groups: our units' order points
            for n, u in f.items():
                if B.side[n] == B.own and B.alive(u) and not u.get("r") and u.get("ox") is not None:
                    a, b = self.px(u["x"], u["z"]), self.px(u["ox"], u["oz"])
                    if math.hypot(a[0] - b[0], a[1] - b[1]) > 25 * self.s:
                        _dashed(d, a, b, OWN + (170,))
                        d.ellipse([b[0] - 2, b[1] - 2, b[0] + 2, b[1] + 2], fill=OWN + (200,))
        for n, u in f.items():                                    # units
            if not B.alive(u):
                continue
            c = OWN if B.side[n] == B.own else ENEMY
            share = max(0.0, min(1.0, (u.get("men") or 0) / max(B.men0.get(n) or 1, 1)))
            poly = self.box(n, u)
            if u.get("r"):
                d.polygon(poly, fill=ROUT + (230,), outline=c + (255,))
            else:
                d.polygon(poly, fill=c + (int(70 + 185 * share),), outline=c + (255,))
            if u.get("m") and not u.get("r"):
                d.line(poly + [poly[0]], fill=MELEE + (255,), width=3 if not small else 2)
            if n in tag:
                x = max(p[0] for p in poly)
                y = min(p[1] for p in poly)
                d.text((x + 1, y - 4), str(tag[n][0]), fill=tag[n][1] + (255,), font=fnt, stroke_width=1,
                       stroke_fill=(255, 255, 255, 255))
            t = LETTER[B.role[n]]
            if t and B.side[n] == B.own and B.role[n] != "lord":
                t += n.rsplit("_", 1)[-1] if n.rsplit("_", 1)[-1].isdigit() else ""
            if t:
                x = sum(p[0] for p in poly) / 4
                y = sum(p[1] for p in poly) / 4
                d.text((x - 3 * len(t), y - 6), t, fill=(255, 255, 255, 255) if not u.get("r") else (0, 0, 0, 255), font=fnt,
                       stroke_width=1, stroke_fill=(0, 0, 0, 255) if not u.get("r") else (255, 255, 255, 255))
        img = Image.alpha_composite(img, lay)
        d = ImageDraw.Draw(img)
        y = img.height - 6
        d.line([(4, y), (4 + 100 * self.s, y)], fill=(0, 0, 0, 255), width=3)
        d.text((8 + 100 * self.s, y - 7), "100 m", fill=(0, 0, 0, 255), font=fnt)
        st = [sum(1 for n, u in f.items() if B.alive(u) and not u.get("r") and (B.side[n] == B.own) == own)
              for own in (True, False)]
        tail = f"up {st[0]}:{st[1]}"
        words, lines = (f"{sec}s " + ", ".join(labels)).split(" "), [""]
        room = img.width - fnt.getlength(tail) - 10
        for w in words:                                           # the caption wraps before the standing count
            if lines[-1] and fnt.getlength(lines[-1] + " " + w) > room:
                lines.append(w)
            else:
                lines[-1] = (lines[-1] + " " + w).strip()
        lh = 13 if small else 15
        d.rectangle([0, 0, img.width, lh * len(lines) + 1], fill=(255, 255, 255, 220))
        for k, ln in enumerate(lines):
            d.text((3, 1 + k * lh), ln, fill=(0, 0, 0, 255), font=fnt)
        d.text((img.width - fnt.getlength(tail) - 3, 1), tail, fill=(0, 0, 0, 255), font=fnt)
        return img.convert("RGB")


def _dashed(d, a, b, col, dash=5):
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    k = 0.0
    while k < L:
        e = min(k + dash, L)
        d.line([(a[0] + (b[0] - a[0]) * k / L, a[1] + (b[1] - a[1]) * k / L),
                (a[0] + (b[0] - a[0]) * e / L, a[1] + (b[1] - a[1]) * e / L)], fill=col, width=1)
        k += 2 * dash


def strip(fx, frames, size=448, cols=4, path=None):
    """frames = [(sec, [labels])] -> one PNG: cols x rows frames of ~size px and a legend line; returns the Image."""
    secs = [s for s, _ in frames]
    cv = Canvas(fx, crop(fx, secs), size)
    small = size < 400
    tiles = [cv.frame(s, lab, small) for s, lab in frames]
    cols = min(cols, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    pad = 3
    fnt = font(10 if small else 12)
    B = fx.B
    W = cols * cv.w + (cols - 1) * pad
    parts = [f"{B.name} {B.kind} {B.map or 'flat'}", "up=enemy, right=our right", "blue ours, red enemy, fill=men",
             "yellow=rout, black rim=melee", "S snip A art M missile F fast X monster L lord (S2 = our ..._2)",
             "<n> hold = group zone, digit = its units", f"100 m={round(100 * cv.s)} px"]
    lines = [""]
    for p_ in parts:                                              # the legend wraps to the strip's width
        if lines[-1] and fnt.getlength(lines[-1] + " | " + p_) > W - 6:
            lines.append(p_)
        else:
            lines[-1] = (lines[-1] + " | " + p_) if lines[-1] else p_
    lh = 13 if small else 15
    head = lh * len(lines) + 3
    H = head + rows * cv.h + (rows - 1) * pad
    out = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(out)
    for k, ln in enumerate(lines):
        d.text((3, 2 + k * lh), ln, fill=(0, 0, 0), font=fnt)
    for k, t in enumerate(tiles):
        out.paste(t, ((k % cols) * (cv.w + pad), head + (k // cols) * (cv.h + pad)))
    if path:
        out.save(path, optimize=True)
    return out


def tokens(w, h):
    """Claude's visual tokens of an image: 28 x 28 px patches (platform.claude.com vision, 11.2026; the high-resolution
    tier: long edge <= 2576 px, <= 4784 tokens, else downscaled)."""
    return math.ceil(w / 28) * math.ceil(h / 28)
