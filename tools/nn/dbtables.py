"""Rows of the game's database tables (data/db.pack), with the columns' real names.

A DB table is binary with no names inside: a header (its version), the number of rows, then the rows
one after another, each a fixed sequence of fields. Which fields, in which order, with which names and
types, is the modders' RPFM schema for that table's version (tools/nn/dbschema.py, its cut-down copy
config/db_schema.json). `decode` reads a whole table by it, to the last byte: every text field must be
readable text and every flag 0 or 1. A game update that changes a table (a version the copy does not
have, or a layout that does not fit) breaks the read loudly, it never reads shifted numbers.

Field types (the schema's): Boolean (1 byte), I16, I32, I64, F32, F64, ColourRGB (4 bytes), StringU8
(u16 length + UTF-8), StringU16 (u16 length + UTF-16), OptionalStringU8 / OptionalI32 (a flag, then the
value when the flag is 1).
"""
import struct

from tools.nn import dbschema

_NUMBERS = {"I16": ("<h", 2), "I32": ("<i", 4), "I64": ("<q", 8), "F32": ("<f", 4), "F64": ("<d", 8),
            "ColourRGB": ("<I", 4)}


def entry_name(table):
    return rf"db\{table}_tables\data__"


def header(b):
    """(version or None, offset of the row count)."""
    p, version = 0, None
    if b[:4] == b"\xfd\xfe\xfc\xff":                    # GUID marker: u16 length + UTF-16
        p = 6 + 2 * struct.unpack_from("<H", b, 4)[0]
    if b[p:p + 4] == b"\xfc\xfd\xfe\xff":               # version marker
        version = struct.unpack_from("<i", b, p + 4)[0]
        p += 8
    return version, p + 1                               # one more byte before the count


def _text(b, p, wide=False):
    n = struct.unpack_from("<H", b, p)[0] * (2 if wide else 1)
    raw = b[p + 2:p + 2 + n]
    if len(raw) != n:
        raise ValueError(f"text runs past the end at {p}")
    text = raw.decode("utf-16-le" if wide else "utf-8")
    if not text.isprintable():
        raise ValueError(f"not text at {p}: {text[:20]!r}")
    return text, p + 2 + n


def _flag(b, p):
    if b[p] not in (0, 1):
        raise ValueError(f"flag is {b[p]} at {p}")
    return bool(b[p]), p + 1


def read_field(b, p, t, digits=5):
    """(value, next offset) of a field of schema type t at b[p]; floats rounded to digits (None: as read)."""
    if t in _NUMBERS:
        fmt, n = _NUMBERS[t]
        v = struct.unpack_from(fmt, b, p)[0]
        if t[0] == "F" and digits is not None:
            v = round(v, digits)
        return v, p + n
    if t == "Boolean":
        return _flag(b, p)
    if t == "StringU8":
        return _text(b, p)
    if t == "StringU16":
        return _text(b, p, wide=True)
    if t in ("OptionalStringU8", "OptionalI32"):
        present, p = _flag(b, p)
        if not present:
            return None, p
        return _text(b, p) if t == "OptionalStringU8" else read_field(b, p, "I32")
    raise KeyError(t)


def layout(table, version):
    """(version, [(name, type)]) of a table's version from the schema copy (tools/nn/dbschema.fields)."""
    return version, dbschema.fields(table, version)


def decode(b, table, layout_=None, digits=5):
    """Every row of a table as a dict {column name: value}, read by the schema of the table's version (or by
    layout_ = (version, [(name, type)])); raises when the layout does not fit the bytes. Floats are rounded to
    digits (None: as read)."""
    got, p = header(b)
    version, fields = layout_ or layout(table, got)
    if got != version:
        raise ValueError(f"{table}: version {got}, the layout is for {version}")
    rows = struct.unpack_from("<I", b, p)[0]
    p += 4
    out = []
    for r in range(rows):
        row = {}
        for name, t in fields:
            try:
                row[name], p = read_field(b, p, t, digits)
            except (ValueError, KeyError, struct.error, UnicodeDecodeError, IndexError) as e:
                raise ValueError(f"{table}: row {r}, field {name}: {e}") from None
        out.append(row)
    if p != len(b):
        raise ValueError(f"{table}: {len(b) - p} bytes left after {rows} rows")
    return out


def read_tables(pack, tables, digits=5):
    """{table: [row dicts]} from db.pack (needs Python 3.14+: compression.zstd); floats rounded to digits."""
    from tools.nn.gamedb import read_entries
    names = {entry_name(t): t for t in tables}
    raw = read_entries(pack, set(names))
    missing = set(names) - set(raw)
    assert not missing, f"not in {pack}: {sorted(missing)}"
    return {names[n]: decode(b, names[n], digits=digits) for n, b in raw.items()}
