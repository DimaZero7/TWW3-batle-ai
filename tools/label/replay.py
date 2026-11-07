"""Pictures of a live labelling record (Pillow, no torch): the page's drawing (web/app.js drawUnit, view.py's numbers) as
PNG frames - the field from above with a 50 m grid and a scale bar, every unit as its formation (width the ordered one,
depth by the formation's rule from the passport's spacing: geometry.dims) or, when the run has the men's places
(nn_soldiers, within 3 s), the men as dots (a loose unit has no box of its ordered width), my groups by colour (outline and a label at the group's centre), the enemy's groups framed (a
dashed box round their units), my orders as arrows (a move: to the ordered point; an attack: to the target, red), the
caption: the window, what the person decided, the chat's words.

    frame(rec, sec, window=None)   one PIL image
    frames(rec, out, gif=True)     digest_<k>.png a window (k from 1) + battle.gif (a frame every 3 s)
"""
import math
from pathlib import Path

from tools.label import digest as DG
from tools.label import pool as PL

W, H = 1000, 1000
MINE, ENEMY = (37, 99, 235), (220, 38, 38)
BG, GRID, INK = (238, 235, 228), (210, 206, 198), (30, 30, 32)


def _font(size):
    from PIL import ImageFont
    from tools import config as project
    # (the container has no font with Cyrillic: the host venv's matplotlib DejaVu, through the mounted repo)
    for f in ("arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "C:/Windows/Fonts/arial.ttf",
              str(project.ROOT / ".venv" / "Lib" / "site-packages" / "matplotlib" / "mpl-data" / "fonts" / "ttf" /
                  "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _rgb(hexs):
    return tuple(int(hexs[i:i + 2], 16) for i in (1, 3, 5))


class View:
    """World -> picture: the box of every unit's place over the battle (or at the seconds secs only), a margin, y up =
    z."""

    def __init__(self, rec, secs=None, margin=160, names=None):
        xs, zs = [], []
        sts = [rec.state(s) for s in secs] if secs else (list(rec.states.values()) or [rec.state(0)])
        for st in sts:
            for n, r in st.items():
                if names is not None and n not in names:
                    continue
                xs.append(float(r["x"]))
                zs.append(float(r["z"]))
        if not xs:
            xs, zs = [-100, 100], [-100, 100]
        cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
        span = max(max(xs) - min(xs), max(zs) - min(zs), 200) + margin
        self.cx, self.cz, self.s = cx, cz, (W - 40) / span

    def __call__(self, x, z):
        # (z down the picture: the game's view from above - its world is left-handed, facing +x the +z is on the right;
        # with z up the picture would be the battle in a mirror)
        return (W / 2 + (x - self.cx) * self.s, H / 2 + (z - self.cz) * self.s)


def dims(key, men, width):
    p = PL.passports().get(key or "", {})
    sp = p.get("spacing") or {}
    h, v = float(sp.get("h") or 1.5), float(sp.get("v") or 1.5)
    if (men or 0) <= 1:
        r = float(p.get("radius_m") or 1.0)
        return max(2 * r, 3), max(2 * r, 3)
    files = max(1, min(int(men), int(float(width or 30) / h + 1e-4)))
    ranks = 1 if p.get("engine") else math.ceil(int(men) / files)
    return files * h, ranks * v


def _poly(view, x, z, b, w, d):
    a = math.radians(b)
    pts = []
    for s, f in ((-w / 2, d / 2), (w / 2, d / 2), (w / 2, -d / 2), (-w / 2, -d / 2)):
        pts.append(view(x + s * math.cos(a) + f * math.sin(a), z - s * math.sin(a) + f * math.cos(a)))
    return pts


def _dashed(dr, pts, fill, width=2, dash=8):
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        L = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(L / dash))
        for i in range(0, n, 2):
            t0, t1 = i / n, min(1, (i + 1) / n)
            dr.line([(x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0), (x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1)],
                    fill=fill, width=width)


def _arrow(dr, a, b, fill, width=2):
    dr.line([a, b], fill=fill, width=width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    for s in (-0.45, 0.45):
        dr.line([b, (b[0] - 11 * math.cos(ang + s), b[1] - 11 * math.sin(ang + s))], fill=fill, width=width)


def window_of(rec, sec):
    return next((w for w in rec.windows if int(w["sec"]) <= sec < int(w.get("until") or w["sec"] + 15)), None)


def frame(rec, sec, window=None, view=None):
    from PIL import Image, ImageDraw
    view = view or View(rec)
    img = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(img)
    f12, f14, f18 = _font(12), _font(14), _font(18)
    # the grid: 50 m, every 100 m darker; the scale bar
    x0, z1 = view.cx - W / 2 / view.s, view.cz + H / 2 / view.s
    for k in range(int(x0 // 50) * 50, int(x0 + W / view.s) + 50, 50):
        px, _ = view(k, 0)
        dr.line([(px, 0), (px, H)], fill=GRID if k % 100 else (190, 186, 178), width=1)
    for k in range(int((z1 - H / view.s) // 50) * 50, int(z1) + 50, 50):
        _, py = view(0, k)
        dr.line([(0, py), (W, py)], fill=GRID if k % 100 else (190, 186, 178), width=1)
    bar = 100 if 100 * view.s < 240 else 50
    dr.rectangle([20, H - 34, 20 + bar * view.s, H - 30], fill=INK)
    dr.text((20, H - 52), f"{bar} м · сетка 50 м", fill=INK, font=f12)
    w = window or window_of(rec, sec) or {}
    g = w.get("groups") or {}
    own = {int(k): v for k, v in (g.get("own") or {}).items()}
    enemy = {int(k): v for k, v in (g.get("enemy") or {}).items()}
    group_of = {n: k for k, names in own.items() for n in names}
    st = rec.state(sec)
    men = rec.men_at(sec)
    # the enemy's groups: a dashed frame round their units
    for k, names in enemy.items():
        rows = [st[n] for n in names if n in st and (st[n].get("men") or 0) > 0]
        if not rows:
            continue
        xs = [view(float(r["x"]), float(r["z"])) for r in rows]
        pad = 22
        box = [(min(p[0] for p in xs) - pad, min(p[1] for p in xs) - pad), (max(p[0] for p in xs) + pad, min(p[1] for p in xs) - pad),
               (max(p[0] for p in xs) + pad, max(p[1] for p in xs) + pad), (min(p[0] for p in xs) - pad, max(p[1] for p in xs) + pad)]
        c = _rgb(DG.COLORS[(k - 1) % len(DG.COLORS)])
        _dashed(dr, box, c, 2)
        dr.text((box[0][0], box[0][1] - 17), f"враг {k}", fill=c, font=f14)
    # the units
    for n, r in st.items():
        if (r.get("men") or 0) <= 0:
            continue
        side = int(r.get("side") or DG_side(rec, n))
        if side == 2 and r.get("v") is False:
            continue
        x, z, b = float(r["x"]), float(r["z"]), float(r.get("b") or 0)
        wd, dp = dims(rec.key_of.get(n), r.get("men"), r.get("ow"))
        col = MINE if side == 1 else ENEMY
        k = group_of.get(n)
        outline = _rgb(DG.COLORS[(k - 1) % len(DG.COLORS)]) if k else col
        fill = tuple(int(c * 0.25 + 255 * 0.75) for c in col) if not r.get("r") else (245, 245, 245)
        pts = men.get(n) if men else None
        if pts:
            # the men's own places: the formation is what they make (a loose unit - wolf rats - has no box of its width)
            xs = [view(pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)]
            pad = 3
            box = [min(p[0] for p in xs) - pad, min(p[1] for p in xs) - pad, max(p[0] for p in xs) + pad,
                   max(p[1] for p in xs) + pad]
            if k:
                dr.rectangle(box, outline=outline, width=3)
        else:
            dr.polygon(_poly(view, x, z, b, wd, dp), fill=fill, outline=outline, width=3 if k else 1)
        if pts:
            rad = max(0.8, min(2.0, 0.4 * view.s))
            for i in range(0, len(pts) - 1, 2):
                px, py = view(pts[i], pts[i + 1])
                dr.rectangle([px - rad, py - rad, px + rad, py + rad], fill=col)
        a = math.radians(b)
        nose = view(x + math.sin(a) * (dp / 2 + 4), z + math.cos(a) * (dp / 2 + 4))
        dr.ellipse([nose[0] - 3, nose[1] - 3, nose[0] + 3, nose[1] + 3], fill=col)
        # my orders
        if side == 1:
            c = outline
            t = r.get("t") or ""
            if t and t in st:
                _arrow(dr, view(x, z), view(float(st[t]["x"]), float(st[t]["z"])), ENEMY, 2)
            elif r.get("ox") is not None and math.hypot(float(r["ox"]) - x, float(r["oz"]) - z) > 8:
                _arrow(dr, view(x, z), view(float(r["ox"]), float(r["oz"])), c, 3 if r.get("f") else 2)
    for k, names in own.items():
        wh = DG.where(rec, names, st)
        if wh:
            px, py = view(wh[0], wh[1])
            dr.text((px + 8, py + 10), f"группа {k}", fill=_rgb(DG.COLORS[(k - 1) % len(DG.COLORS)]), font=f18)
    # the caption
    a, b = int(w.get("sec", sec)), int(w.get("until") or sec + 15)
    deploy = w.get("source") == "deploy"
    lines = [f"перед боем · {DG.SOURCE_RU['deploy']}" if deploy else
             f"t = {sec} с · окно {a}–{b} с · {DG.SOURCE_RU.get(w.get('source'), w.get('source') or '')}"]
    for c in (w.get("chat") or []) + [c for c in rec.chat if a <= float(c.get("sec") or -1) < b and not w.get("chat")]:
        lines.append(f"«{c.get('text')}»" + ("" if deploy else f" ({c.get('sec')} с)"))
    dr.rectangle([0, 0, W, 10 + 22 * len(lines)], fill=(255, 255, 255))
    for i, t in enumerate(lines):
        dr.text((12, 6 + 22 * i), t, fill=INK, font=f18 if i == 0 else f14)
    return img


def DG_side(rec, n):
    return rec.side_of.get(n) or (1 if str(n).startswith("own") else 2)


def frames(rec, out, gif=True, every_s=3, size=640):
    """digest_<k>.png a window and battle.gif; returns the windows' frames written."""
    out = Path(out)
    view = View(rec)
    n = 0
    for d in getattr(rec, "deploy", [])[:1]:
        # the deployment: the places at the battle's start, the groups as the strategist's sample read them
        g = d.get("groups") or {}
        w = {"sec": 0, "until": 0, "source": "deploy", "chat": list(d.get("chat") or []),
             "groups": {"own": {k: v["members"] for k, v in (d.get("read") or {}).items()}, "enemy": g.get("enemy") or {}}}
        frame(rec, 0, w, view).save(out / "digest_00_deploy.png")
        n += 1
    for k, w in enumerate(rec.windows, 1):
        frame(rec, int(w["sec"]), w, view).save(out / f"digest_{k:02d}.png")
        n += 1
    if gif and (rec.states or rec.orders):
        secs = sorted(rec.states or rec.orders)
        pics = [frame(rec, s, None, view).resize((size, size)) for s in secs[::every_s]]
        if pics:
            pics[0].save(out / "battle.gif", save_all=True, append_images=pics[1:], duration=250, loop=0)
    return n
