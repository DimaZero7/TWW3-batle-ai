"""The in-game overlay's icons (src/apps/battle/label_overlay.lua; tools.build nn-arena --label-game packs them): per
group 1..6 its colour (config/label.json group_colors) - 'g<k>.png' a filled disc with the group's number (my unit in
group k), 't<k>.png' a ring with the number (an enemy, the target of group k). Plain PNG (RGBA, zlib), no Pillow.

    pack_files() -> {pack path: bytes}   (ui\\bai\\label\\g1.png ...); the game's image path: IMAGE.format(...)
"""
import struct
import zlib

from tools.label import cfg

SIZE = 64
FOLDER = "ui\\bai\\label"
IMAGE = "ui/bai/label/{kind}{k}.png"
# a 5 x 7 bitmap font of the digits (rows top down, bit 4 the left column)
DIGITS = {
    "1": (0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E), "2": (0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F),
    "3": (0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E), "4": (0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02),
    "5": (0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E), "6": (0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E),
    "7": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08), "8": (0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E),
    "9": (0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C), "0": (0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E),
}
# the enemy groups' number tabs gn1..gn40 (src/apps/battle/enemy_cards.lua: a group of any number, its colour the k-th of
# group_colors round the circle - neighbours never alike)
TABS = 40


def _rgb(hex_colour):
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def png(pixels, w, h):
    """RGBA rows [h][w][4] -> PNG bytes."""
    raw = b"".join(b"\x00" + bytes(v for px in row for v in px) for row in pixels)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def icon(k, colour, ring=False, size=SIZE):
    """The icon of group k (1..6): a disc (my unit) or a ring (a target) of colour with the digit k."""
    c = _rgb(colour)
    r0 = size / 2 - 1.5
    px = [[(0, 0, 0, 0)] * size for _ in range(size)]
    m = size / 2 - 0.5
    for y in range(size):
        for x in range(size):
            d = ((x - m) ** 2 + (y - m) ** 2) ** 0.5
            if d > r0:
                continue
            if d > r0 - 4:
                px[y][x] = (255, 255, 255, 255)                     # the white rim
            elif ring and d < r0 - 13:
                px[y][x] = (0, 0, 0, 110)                          # the ring's dark inside
            else:
                px[y][x] = c + (255,)
    rows = DIGITS[str(k)]
    s = 4 if ring else 5
    w, h = 5 * s, 7 * s
    x0, y0 = int(m - w / 2 + 1), int(m - h / 2 + 1)
    for j, bits in enumerate(rows):
        for i in range(5):
            if bits >> (4 - i) & 1:
                for yy in range(y0 + j * s, y0 + (j + 1) * s):
                    for xx in range(x0 + i * s, x0 + (i + 1) * s):
                        px[yy][xx] = (255, 255, 255, 255) if not ring else (255, 255, 255, 255)
    return png(px, size, size)


def separator(w=10, h=120):
    """The enemy cards' separator: a gold bar with a dark edge."""
    px = [[(40, 30, 20, 255) if x in (0, w - 1) else (214, 170, 70, 255) for x in range(w)] for _ in range(h)]
    return png(px, w, h)


UNITS_FOLDER = "ui\\bai\\units"


def unit_files(keys):
    """{pack path: bytes} of the units' card pictures (build/label_icons/<key>.png, tools/label/gameart.py) - the enemy
    cards' picture (src/apps/battle/enemy_cards.lua UNIT_IMAGE); a key without a picture is left out."""
    from tools import config as project
    out = {}
    for k in keys:
        f = project.ROOT / "build" / "label_icons" / f"{k}.png"
        if f.exists():
            out[f"{UNITS_FOLDER}\\{k}.png"] = f.read_bytes()
    return out


def frame(w=60, h=130, t=3, colour=(255, 214, 90, 255)):
    """A card's pick frame (src/apps/battle/enemy_cards.lua PICK_IMAGE): a border of t px, the inside clear."""
    px = [[colour if (x < t or x >= w - t or y < t or y >= h - t) else (0, 0, 0, 0) for x in range(w)] for y in range(h)]
    return png(px, w, h)


def number_tab(k, colour, size=SIZE):
    """A group's tab: a filled disc of colour with a white rim and the number k (one or two digits)."""
    c = _rgb(colour)
    r0 = size / 2 - 1.5
    m = size / 2 - 0.5
    px = [[(0, 0, 0, 0)] * size for _ in range(size)]
    for y in range(size):
        for x in range(size):
            d = ((x - m) ** 2 + (y - m) ** 2) ** 0.5
            if d <= r0:
                px[y][x] = (255, 255, 255, 255) if d > r0 - 4 else c + (255,)
    text = str(k)
    s = 5 if len(text) == 1 else 4
    w = len(text) * 5 * s + (len(text) - 1) * s
    x0, y0 = int(m - w / 2 + 1), int(m - 7 * s / 2 + 1)
    for n, ch in enumerate(text):
        for j, bits in enumerate(DIGITS[ch]):
            for i in range(5):
                if bits >> (4 - i) & 1:
                    for yy in range(y0 + j * s, y0 + (j + 1) * s):
                        for xx in range(x0 + n * 6 * s + i * s, x0 + n * 6 * s + (i + 1) * s):
                            px[yy][xx] = (255, 255, 255, 255)
    return png(px, size, size)


def pick_ring(size=SIZE, colour=(255, 214, 90)):
    """The enemy's pick over its unit on the field (src/apps/battle/live_label.lua RING_IMAGE): a yellow ring, as the
    ring the game puts round a selected own unit's banner."""
    px = [[(0, 0, 0, 0)] * size for _ in range(size)]
    m, r0 = size / 2 - 0.5, size / 2 - 1.0
    for y in range(size):
        for x in range(size):
            d = ((x - m) ** 2 + (y - m) ** 2) ** 0.5
            if r0 - 5 <= d <= r0:
                px[y][x] = colour + (255,)
    return png(px, size, size)


def pack_files():
    out = {f"{FOLDER}\\sep.png": separator(), f"{FOLDER}\\pick.png": frame(),
           f"{FOLDER}\\hover.png": frame(colour=(255, 255, 255, 150)),
           f"{FOLDER}\\clear.png": png([[(0, 0, 0, 0)] * 4 for _ in range(4)], 4, 4),
           f"{FOLDER}\\pick_ring.png": pick_ring()}
    colours = cfg()["group_colors"][:6]
    for k in range(1, TABS + 1):
        out[f"{FOLDER}\\gn{k}.png"] = number_tab(k, colours[(k - 1) % len(colours)])
    for k, colour in enumerate(cfg()["group_colors"][:6], start=1):
        out[f"{FOLDER}\\g{k}.png"] = icon(k, colour)
        out[f"{FOLDER}\\gf{k}.png"] = frame(colour=_rgb(colour) + (255,))
        out[f"{FOLDER}\\gb{k}.png"] = png([[_rgb(colour) + (255,)] * 4 for _ in range(4)], 4, 4)   # a frame's edge
        out[f"{FOLDER}\\t{k}.png"] = icon(k, colour, ring=True)
    return out
