"""The units' cards and Russian names from the game's own packs, for the page (host only: the game's packs are
zstd-compressed - Python 3.14+ compression.zstd, as tools/nn/gamedb.py).

    py -3.14 -m tools.label.gameart          -> build/label_icons/<unit key>.png, build/label_icons/names_ru.json

Where they are in the game (found 07.11):
    the card       ui.pack ui\\units\\icons\\<card>.png; the land unit's card name is in db.pack
                   db\\unit_variants_tables (no schema in config/db_schema.json: the row is read as its length-prefixed
                   strings - the unit's key, then the first string that is a card in ui.pack)
    the name       local_ru.pack text\\localisation__.loc, key land_units_onscreen_name_<unit key> (the loc format:
                   FF FE 'LOC' 00, version, count, then key / text as u16 length + UTF-16, and a flag byte)
The units: every unit of the pool (config/nn/pools.json). A unit without a card keeps the page's sign of its kind.
"""
import json
import re
import struct
import sys
from pathlib import Path

from tools import config as project
from tools.label import pool as PL

OUT = project.ROOT / "build" / "label_icons"
# A lord's card the db's unit variant does not name among the unit cards: the character's own card in ui.pack
# ui/portraits/units (60 x 130, the card the game shows for our own general - the person's 07.11: the enemy's lord had no
# picture)
CHARACTER_CARDS = {"wh_main_emp_cha_general_0": "ui/portraits/units/no_culture/emp_generals_campaign_01_0.png",
                   "wh2_main_skv_cha_warlord_0": "ui/portraits/units/no_culture/skv_warlord_campaign_01_0.png"}
BS = "\\"


def _index(pack):
    """[(name, offset, size, compressed)] of a PFH5 pack."""
    out = []
    with open(pack, "rb") as f:
        magic, kind, pc, ps, fc, fs, ts = struct.unpack("<4s6I", f.read(28))
        assert magic == b"PFH5", pack
        ext = 20 if kind & 0x100 else 0
        f.read(ext)
        f.read(ps)
        idx = f.read(fs)
    pos, off = 0, 28 + ext + ps + fs
    for _ in range(fc):
        size = struct.unpack_from("<I", idx, pos)[0]
        pos += 4 + (4 if kind & 0x40 else 0)
        comp = idx[pos]
        pos += 1
        end = idx.index(0, pos)
        out.append((idx[pos:end].decode("latin1"), off, size, comp))
        pos = end + 1
        off += size
    return out


def _read(pack, entries):
    from compression import zstd
    got = {}
    with open(pack, "rb") as f:
        for name, off, size, comp in entries:
            f.seek(off)
            data = f.read(size)
            got[name] = zstd.decompress(data[4:]) if comp else data
    return got


def loc(b):
    """{key: text} of a .loc file."""
    assert b[:2] == b"\xff\xfe" and b[2:5] == b"LOC", b[:8]
    n = struct.unpack_from("<I", b, 10)[0]
    pos, out = 14, {}
    for _ in range(n):
        k = struct.unpack_from("<H", b, pos)[0]
        key = b[pos + 2:pos + 2 + 2 * k].decode("utf-16-le")
        pos += 2 + 2 * k
        k = struct.unpack_from("<H", b, pos)[0]
        text = b[pos + 2:pos + 2 + 2 * k].decode("utf-16-le")
        pos += 2 + 2 * k + 1
        out[key] = text
    return out


def strings_after(blob, key, span=600):
    """The length-prefixed ascii strings in the span after each occurrence of key (as a length-prefixed string)."""
    k = key.encode()
    pat = struct.pack("<H", len(k)) + k
    out = []
    for m in re.finditer(re.escape(pat), blob):
        p, end, row = m.end(), min(len(blob), m.end() + span), []
        while p + 2 < end:
            n = struct.unpack_from("<H", blob, p)[0]
            s = blob[p + 2:p + 2 + n]
            if 2 <= n <= 120 and all(32 < c < 127 for c in s):
                row.append(s.decode())
                p += 2 + n
            else:
                p += 1
        out.append(row)
    return out


def main(argv=None):
    data = Path(project.load()["game_dir"]) / "data"
    OUT.mkdir(parents=True, exist_ok=True)
    keys = sorted({u["key"] for f in PL.factions().values() for u in f["units"]}
                  | set(json.loads((project.CONFIG_DIR / "nn" / "units.json").read_text(encoding="utf-8"))["units"]))
    ui = _index(data / "ui.pack")
    icons = {n.split(BS)[-1][:-4]: (n, o, s, c) for n, o, s, c in ui
             if n.lower().startswith("ui" + BS + "units" + BS + "icons" + BS) and n.lower().endswith(".png")}
    db = _index(data / "db.pack")
    uv = [e for e in db if e[0].startswith("db" + BS + "unit_variants_tables" + BS)]
    blob = b"".join(_read(data / "db.pack", uv).values())
    anyp = {}
    for n, o, sz, c in ui:
        if n.lower().endswith(".png"):
            anyp.setdefault(n.split(BS)[-1][:-4], (n, o, sz, c))
    card = {}
    for key in keys:
        for row in strings_after(blob, key):
            # (the row: the key, then a few names; only its first three - past them the next row begins)
            hit = next((s for s in row[:3] if s in icons), None) or next((s for s in row[:3] if s in anyp), None)
            if hit:
                card[key] = hit
                icons.setdefault(hit, anyp[hit])
                break
    for key, path in CHARACTER_CARDS.items():
        if key in keys and key not in card:
            want = path.replace("/", BS).lower()
            hit = next((e for e in ui if e[0].lower() == want), None)
            if hit:
                name = hit[0].split(BS)[-1][:-4]
                icons[name] = hit
                card[key] = name
    got = _read(data / "ui.pack", [icons[c] for c in set(card.values())])
    for key, c in card.items():
        (OUT / f"{key}.png").write_bytes(got[icons[c][0]])
    lr = [e for e in _index(data / "local_ru.pack") if e[0].endswith(".loc")]
    texts = {}
    for b in _read(data / "local_ru.pack", lr).values():
        texts.update(loc(b))
    names = {k: texts.get(f"land_units_onscreen_name_{k}") for k in keys}
    (OUT / "names_ru.json").write_text(json.dumps({"_source": "local_ru.pack text/localisation__.loc, "
                                                   "land_units_onscreen_name_<key>; cards: ui.pack via db.pack "
                                                   "unit_variants (tools/label/gameart.py)", "cards": card,
                                                   "names": names}, ensure_ascii=False, indent=1), encoding="utf-8")
    miss_c = [k for k in keys if k not in card]
    miss_n = [k for k in keys if not names[k]]
    print(f"units {len(keys)}: cards {len(card)}, names {len(keys) - len(miss_n)}; no card: {miss_c}; no name: {miss_n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
