"""The game's database tables by their real column names: the modders' RPFM schema, cut down to our tables.

A table in db.pack is binary with no names inside: a header (its version), the row count, then the rows,
each a fixed sequence of fields. The modding community's RPFM schema (rpfm-schemas, Frodo45127,
schema_wh3.ron) lists, for every table and every version of it, the fields in the order they are written,
with their names and types. config/db_schema.json keeps only the tables we read (TABLES) in the versions
the installed game and the required mods have (`game_versions`), as [name, type] pairs in the written order,
and the source (the RON file's SHA-256).

Why a cut-down copy and not the RON itself: the RON is 11 MB, external and changes with every modding
update; the copy is small, reviewed in git, and the code needs nothing outside the repository. A game
update that changes a table's version breaks the read loudly (`fields` raises): then refresh the copy:

    curl -L -o build/rpfm/schema_wh3.ron https://raw.githubusercontent.com/Frodo45127/rpfm-schemas/master/schema_wh3.ron
    py -3.14 -m tools.nn.dbschema                    # rewrites config/db_schema.json from it

(Python 3.14+ to read the game's versions; --all-versions keeps every version of the schema, any Python.)

Only the schema here; the bytes are read by tools/nn/dbtables.py.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from tools import config as project

OUT = project.CONFIG_DIR / "db_schema.json"
RON = project.BUILD / "rpfm" / "schema_wh3.ron"
URL = "https://raw.githubusercontent.com/Frodo45127/rpfm-schemas/master/schema_wh3.ron"
# The tables the project reads (tools/nn/gamedb.py, units.py, abilities.py, armies/templates.py), without "_tables".
TABLES = (
    "_kv_morale", "_kv_fatigue", "_kv_rules", "unit_experience_bonuses", "unit_experience_thresholds",
    "land_units", "main_units", "battle_entities", "melee_weapons", "missile_weapons", "projectiles",
    "unit_armour_types", "unit_shield_types", "unit_attributes_to_groups_junctions",
    "land_units_to_unit_abilites_junctions", "unit_spacings", "unit_special_abilities", "special_ability_phases",
    "special_ability_to_special_ability_phase_junctions", "special_ability_phase_stat_effects",
    "special_ability_phase_attribute_effects", "special_ability_to_recharge_contexts",
    "special_ability_to_auto_deactivate_flags", "unit_abilities", "projectile_bombardments", "projectiles_explosions",
    "projectile_penetration_junctions",
    "cdir_military_generator_template_priorities", "cdir_military_generator_template_ratios",
    "cdir_military_generator_unit_group_overrides", "cdir_military_generator_unit_qualities",
)
# The schema's field types, each read by tools/nn/dbtables.read_field.
TYPES = ("Boolean", "I16", "I32", "I64", "F32", "F64", "ColourRGB", "StringU8", "StringU16", "OptionalStringU8",
         "OptionalI32")


# --- the RON reader (only to refresh the copy) -------------------------------------------------------------------

_TOKEN = re.compile(r'\s*(?:(")((?:[^"\\]|\\.)*)"|([-+]?\d[\w.+-]*)|([A-Za-z_]\w*)|([()\[\]{}:,]))', re.S)
_ESC = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'", "0": "\0"}


class _Ron:
    """A small reader of the RON the schema is written in: structs (name: value) -> dict, tuples and lists -> list,
    maps -> dict, Some(x) -> x, None -> None, true/false -> bool, other names -> str."""

    def __init__(self, text, pos=0):
        self.text, self.pos = text, pos

    def _next(self):
        m = _TOKEN.match(self.text, self.pos)
        if not m:
            raise ValueError(f"RON: cannot read at {self.pos}: {self.text[self.pos:self.pos + 40]!r}")
        self.pos = m.end()
        if m.group(1):
            return "str", re.sub(r"\\(.)", lambda e: _ESC.get(e.group(1), e.group(1)), m.group(2))
        if m.group(3):
            t = m.group(3)
            return "num", float(t) if any(c in t for c in ".eE") else int(t)
        if m.group(4):
            return "name", m.group(4)
        return "punct", m.group(5)

    def _peek(self):
        pos = self.pos
        tok = self._next()
        self.pos = pos
        return tok

    def value(self):
        kind, v = self._next()
        if kind in ("str", "num"):
            return v
        if kind == "name":
            if self._peek() == ("punct", "("):
                self._next()
                inner = self._seq(")")
                return inner[0] if v == "Some" and len(inner) == 1 else {v: inner}
            return {"None": None, "true": True, "false": False}.get(v, v)
        if v == "(":
            return self._struct()
        if v == "[":
            return self._seq("]")
        if v == "{":
            return self._map()
        raise ValueError(f"RON: unexpected {v!r} at {self.pos}")

    def _seq(self, end):
        out = []
        while self._peek() != ("punct", end):
            out.append(self.value())
            if self._peek() == ("punct", ","):
                self._next()
        self._next()
        return out

    def _struct(self):
        """(name: value, ...) -> dict; (value, ...) -> list."""
        pos = self.pos
        first = self._next()
        if first == ("punct", ")"):
            return []
        if first[0] == "name" and self._next() == ("punct", ":"):
            self.pos = pos
            out = {}
            while self._peek() != ("punct", ")"):
                kind, name = self._next()
                assert kind == "name" and self._next() == ("punct", ":"), f"RON: struct field at {self.pos}"
                out[name] = self.value()
                if self._peek() == ("punct", ","):
                    self._next()
            self._next()
            return out
        self.pos = pos
        return self._seq(")")

    def _map(self):
        out = {}
        while self._peek() != ("punct", "}"):
            key = self.value()
            assert self._next() == ("punct", ":"), f"RON: map entry at {self.pos}"
            out[key] = self.value()
            if self._peek() == ("punct", ","):
                self._next()
        self._next()
        return out


def ron_tables(text, tables=TABLES, versions=None):
    """{table: {version: [[name, type]]}} of the RON schema's definitions (text), only the tables asked for and,
    with versions ({table: {version}}), only those versions (a table missing there: all of its versions)."""
    start, end = text.index("definitions: {"), text.index("\n    patches: {")
    out = {}
    for t in tables:
        at = text.find(f'\n        "{t}_tables": [', start, end)
        if at < 0:
            raise KeyError(f"{t}_tables is not in the schema's definitions")
        reader = _Ron(text, text.index("[", at) + 1)
        found = reader._seq("]")
        out[t] = {}
        for v in sorted(found, key=lambda d: d["version"]):
            if versions is not None and t in versions and v["version"] not in versions[t]:
                continue
            fields = []
            for f in v["fields"]:
                ftype = f["field_type"]
                if not isinstance(ftype, str) or ftype not in TYPES:
                    raise ValueError(f"{t} v{v['version']} {f['name']}: field type {ftype!r} is not read")
                fields.append([f["name"], ftype])
            out[t][str(v["version"])] = fields
        missing = (versions or {}).get(t, set()) - {int(k) for k in out[t]}
        if missing:
            raise ValueError(f"{t}: the game has version {sorted(missing)}, the schema does not: a newer schema needed")
    return out


def game_versions(settings=None):
    """{table: {version}} of TABLES in the game's db.pack and the required mods' packs (no version marker: 0;
    Python 3.14+)."""
    from tools.nn import dbtables, gamedb
    settings = settings or project.load()
    packs = [Path(settings["game_dir"]) / "data" / "db.pack"]
    packs += [Path(settings["workshop_dir"]) / m["workshop_id"] / m["pack"] for m in project.required_mods()]
    out = {}
    for pack in packs:
        for name, b in gamedb.read_prefixed(pack, [f"db\\{t}_tables\\" for t in TABLES]).items():
            table = name.split("\\")[1].removesuffix("_tables")
            out.setdefault(table, set()).add(dbtables.header(b)[0] or 0)
    return out


def extract(ron=RON, versions=None):
    """The cut-down schema document from the RON file (ron); versions: game_versions() (None: all)."""
    ron = Path(ron)
    if not ron.exists():
        raise FileNotFoundError(f"no RPFM schema at {ron}: download it with\n    curl -L -o {ron} {URL}")
    raw = ron.read_bytes()
    text = raw.decode("utf-8")
    m = re.match(r"\(\s*version:\s*(\d+)", text)
    return {"_description": "The RPFM schema of the game's database tables we read (tools/nn/dbschema.py): per table "
                            "(without '_tables') and version, the fields in the order they are written, [name, type]. "
                            "Refresh: python -m tools.nn.dbschema.",
            "_source": {"url": URL, "repository": "rpfm-schemas (Frodo45127), branch master",
                        "format_version": int(m.group(1)) if m else None,
                        "sha256": hashlib.sha256(raw).hexdigest()},
            "tables": ron_tables(text, versions=versions)}


def dumps(doc):
    """The document as text: one field a line."""
    lines = ["{"]
    for k in ("_description", "_source"):
        lines.append(f" {json.dumps(k)}: {json.dumps(doc[k], ensure_ascii=False)},")
    lines.append(' "tables": {')
    tables = list(doc["tables"].items())
    for i, (t, versions) in enumerate(tables):
        lines.append(f"  {json.dumps(t)}: {{")
        for j, (v, fields) in enumerate(versions.items()):
            lines.append(f"   {json.dumps(v)}: [")
            lines += [f"    {json.dumps(f)}" + ("," if n < len(fields) - 1 else "") for n, f in enumerate(fields)]
            lines.append("   ]" + ("," if j < len(versions) - 1 else ""))
        lines.append("  }" + ("," if i < len(tables) - 1 else ""))
    lines += [" }", "}"]
    return "\n".join(lines) + "\n"


# --- the copy, used by the readers --------------------------------------------------------------------------------

_LOADED = {}


def load(path=OUT):
    """The cut-down schema (config/db_schema.json), read once."""
    path = Path(path)
    if path not in _LOADED:
        if not path.exists():
            raise FileNotFoundError(f"no {path}: make it from the RPFM schema with\n    curl -L -o {RON} {URL}\n"
                                    f"    python -m tools.nn.dbschema")
        _LOADED[path] = json.loads(path.read_text(encoding="utf-8"))
    return _LOADED[path]


def fields(table, version, path=OUT):
    """[(name, schema type)] of a table's version (None: a table with no version marker = version 0); raises when
    the schema copy does not have it (refresh: the module's text)."""
    versions = load(path)["tables"].get(table)
    if versions is None:
        raise KeyError(f"{table}: not in {Path(path).name} (add it to tools/nn/dbschema.py TABLES and refresh)")
    got = versions.get(str(version or 0))
    if got is None:
        raise ValueError(f"{table}: version {version} is not in {Path(path).name} (it has {sorted(versions, key=int)}): "
                         f"refresh the schema (python -m tools.nn.dbschema)")
    return [tuple(f) for f in got]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ron", type=Path, default=RON)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--all-versions", action="store_true", help="every version of the schema, not the game's")
    args = parser.parse_args(argv)
    try:
        doc = extract(args.ron, None if args.all_versions else game_versions())
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    args.out.write_text(dumps(doc), encoding="utf-8", newline="\n")
    print(f"written {args.out}: {len(doc['tables'])} tables, "
          f"{sum(len(v) for v in doc['tables'].values())} versions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
