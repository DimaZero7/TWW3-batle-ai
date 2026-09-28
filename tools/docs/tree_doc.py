"""The generated parts of the AI tree's documentation (docs/<lang>/tree/).

    python -m tools.docs.tree_doc           # rewrite the generated blocks
    python -m tools.docs.tree_doc --check   # exit 1 if a block is out of date (tests/docs)

Sources — one truth each (docs/ru/architecture/documentation.md):
* src/apps/tree/services.lua: the nodes, parents, baselines (en), defaults;
* docs/tree.json: titles, what a node does, when it starts, status, modules,
  pages, planned nodes of the battle theory, Russian baselines;
* tools/architecture.py and the requires in src/apps: levels and dependencies.

A generated block sits between <!-- generated:NAME --> and <!-- /generated -->
in a page; NAME is tree:diagram, tree:table, tree:modules, tree:here:<node>
or tree:card:<node>. Only the inside of a block is rewritten.
"""
import argparse
import json
import re
import sys
from pathlib import Path

from tools import architecture
from tools.lua_runtime import load, new_runtime

ROOT = architecture.ROOT
DOCS = ROOT / "docs"
DATA = DOCS / "tree.json"
LANGS = ("ru", "en")
BLOCK = re.compile(r"(<!-- generated:([a-z:_0-9]+) -->\n)(.*?)(<!-- /generated -->)", re.S)

T = {
    "ru": {"base": "Основа", "modules_n": "модулей", "node": "Узел", "kind": "Вид", "level": "Уровень",
           "status": "Статус", "switch": "Переключатель", "without": "Без него", "modules": "Модули",
           "what": "Что делает", "enter": "Когда начинается", "where": "Где в дереве", "on": "вкл", "off": "выкл",
           "root": "корень дерева, не выключается", "core_note": "`core` нужен всем и на схеме не показан",
           "trunk": "ствол уровня", "uses": "использует", "parent": "Растёт из", "none": "—",
           "planned_switch": "появится с узлом"},
    "en": {"base": "Base", "modules_n": "modules", "node": "Node", "kind": "Kind", "level": "Level",
           "status": "Status", "switch": "Switch", "without": "Without it", "modules": "Modules",
           "what": "What it does", "enter": "Starts when", "where": "Place in the tree", "on": "on", "off": "off",
           "root": "the tree's root, cannot be off", "core_note": "`core` is used by all and not drawn",
           "trunk": "the level's trunk", "uses": "uses", "parent": "Grows from", "none": "—",
           "planned_switch": "comes with the node"},
}
FILL = {"done": "fill:#E1F5EE,stroke:#0F6E56,color:#04342C",
        "started": "fill:#FAEEDA,stroke:#854F0B,color:#412402",
        "planned": "fill:#F1EFE8,stroke:#888780,color:#444441"}
# Mermaid gives a node one class with ":::": one class per status, branch (dashed) and "here" (thick).
STYLE = [f"  classDef {st}{b}{h} {fill}{',stroke-dasharray:5 4' if b else ''}{',stroke-width:4px' if h else ''}"
         for st, fill in FILL.items() for b in ("", "_b") for h in ("", "_h")]


def lua_nodes():
    lua = new_runtime()
    tree = load(lua, "apps.tree.services")
    out = {}
    for n in tree.NODES.values():
        out[n.name] = {"kind": n.kind, "parent": n.parent, "baseline": n.baseline,
                       "default": n.default is not False}
    return out


def nodes():
    """The docs' nodes with the code's parent, baseline and default merged in."""
    data = json.loads(DATA.read_text(encoding="utf-8"))
    code = lua_nodes()
    merged = []
    for n in data["nodes"]:
        m = dict(n)
        if n["lua"]:
            c = code[n["id"]]
            m.update(parent=c["parent"], baseline_en=c["baseline"], default=c["default"])
            assert c["kind"] == n["kind"], f'{n["id"]}: kind differs from the code'
        merged.append(m)
    return data, merged


def _label(n, lang):
    title = n["title"][lang]
    if n["kind"] == "phase":
        return f'{n["phase"]} · {title}'
    return f'{title}<br/>{n["id"]}' if n["lua"] else title


def _classes(n, here=None):
    return n["status"] + ("_b" if n["kind"] != "phase" else "") + ("_h" if here == n["id"] else "")


def diagram(lang, only=None, here=None):
    """Mermaid, bottom to top: the base, then the trunk of phases in solid
    arrows; each branch sits beside its phase on a dashed line. The level is
    the second line of a phase (level groups made Mermaid lay the tree out as
    a staircase)."""
    data, ns = nodes()
    t = T[lang]
    shown = [n for n in ns if only is None or n["id"] in only]
    lines = ["```mermaid", "flowchart BT"]
    base_n = sum(1 for v in architecture.LEVELS.values() if v == 0)
    with_base = only is None or "base" in only
    if with_base:
        lines.append(f'  base["{t["base"]}<br/>{base_n} {t["modules_n"]}"]:::done')
    for n in shown:
        label = _label(n, lang)
        if n["kind"] == "phase":
            label += f'<br/><i>{architecture.LEVEL_NAMES[n["level"]][lang]}</i>'
        lines.append(f'  {n["id"]}["{label}"]:::{_classes(n, here)}')
    ids = {n["id"] for n in shown}
    for n in shown:
        parent = n.get("parent")
        if not parent:
            if with_base:
                lines.append(f'  base --> {n["id"]}')
        elif parent in ids:
            # A branch is written first so Mermaid ranks it beside its phase, not above.
            lines.append(f'  {parent} --> {n["id"]}' if n["kind"] == "phase" else f'  {n["id"]} -.- {parent}')
    used = {"done"} | {_classes(n, here) for n in shown}
    lines[2:2] = [line for line in STYLE if line.split()[1] in used]
    lines.append("```")
    return "\n".join(lines)


def _page_link(n, lang, from_tree=True):
    return f'[{n["title"][lang]}]({n["page"]})' if from_tree else f'[{n["title"][lang]}](../tree/{n["page"]})'


def _switch(n, t):
    if not n["lua"]:
        return t["planned_switch"]
    if not n.get("parent"):
        return t["none"]
    return f'`{n["id"]}` ({t["on"] if n["default"] else t["off"]})'


def _without(n, lang, t):
    if not n["lua"]:
        return t["none"]
    if not n.get("parent"):
        return t["root"]
    return n["baseline_ru"] if lang == "ru" else n["baseline_en"]


def _modules(n):
    return ", ".join(f"[{m}](../apps/{m}.md)" for m in n["modules"]) or "—"


def table(lang):
    data, ns = nodes()
    t = T[lang]
    rows = [f'| {t["node"]} | {t["kind"]} | {t["level"]} | {t["status"]} | {t["switch"]} | {t["without"]} | {t["modules"]} |',
            "|---|---|---|---|---|---|---|"]
    for n in ns:
        rows.append(f'| {_page_link(n, lang)} | {data["kinds"][n["kind"]][lang]} | '
                    f'{architecture.LEVEL_NAMES[n["level"]][lang]} | {data["status"][n["status"]][lang]} | '
                    f'{_switch(n, t)} | {_without(n, lang, t)} | {_modules(n)} |')
    return "\n".join(rows)


def card(lang, node_id):
    data, ns = nodes()
    t = T[lang]
    by = {n["id"]: n for n in ns}
    n = by[node_id]
    parent = by.get(n.get("parent"))
    rows = ["| | |", "|---|---|",
            f'| {t["what"]} | {n["what"][lang]} |',
            f'| {t["enter"]} | {n["enter"][lang]} |',
            f'| {t["without"]} | {_without(n, lang, t)} |',
            f'| {t["switch"]} | {_switch(n, t)} |',
            f'| {t["kind"]}, {t["level"].lower()} | {data["kinds"][n["kind"]][lang]}, '
            f'{architecture.LEVEL_NAMES[n["level"]][lang]} |',
            f'| {t["parent"]} | {_page_link(parent, lang) if parent else t["none"]} |',
            f'| {t["modules"]} | {_modules(n)} |',
            f'| {t["status"]} | {data["status"][n["status"]][lang]} |']
    return "\n".join(rows)


def here(lang, node_id):
    """The node in its place: the phases it grows from, itself, what grows from it."""
    _, ns = nodes()
    by = {n["id"]: n for n in ns}
    chain, cur = [], by[node_id]
    while cur:
        chain.append(cur["id"])
        cur = by.get(cur.get("parent"))
    kids = [n["id"] for n in ns if n.get("parent") == node_id]
    return diagram(lang, only=set(chain) | set(kids) | {"base"}, here=node_id)


def modules(lang):
    """The levels as boxes with their modules (the trunk of a level in bold),
    then who uses whom as a table: a picture of every edge would be unreadable."""
    t = T[lang]
    deps = architecture.dependencies()
    used_by = {}
    for app, ds in deps.items():
        for d in ds:
            used_by.setdefault(d, []).append(app)
    lines = ["```mermaid", "flowchart BT",
             "  classDef level fill:#EEEDFE,stroke:#534AB7,color:#26215C,text-align:left"]
    levels = sorted(set(architecture.LEVELS.values()))
    for level in levels:
        apps = sorted(a for a, lv in architecture.LEVELS.items() if lv == level)
        names = [f"<b>{a}</b>" if a in architecture.TRUNKS else a for a in apps]
        rows = ["  ·  ".join(names[i:i + 5]) for i in range(0, len(names), 5)]
        lines.append(f'  M{level}["<b>{architecture.LEVEL_NAMES[level][lang]}</b><br/>' + "<br/>".join(rows) + '"]:::level')
    for lower, upper in zip(levels, levels[1:]):
        lines.append(f"  M{lower} --> M{upper}")
    lines.append("```")
    head = {"ru": ("Модуль", "Уровень", "Пользуется", "Им пользуются"),
            "en": ("Module", "Level", "Uses", "Used by")}[lang]
    lines += ["", f"| {head[0]} | {head[1]} | {head[2]} | {head[3]} |", "|---|---|---|---|"]
    for app in sorted(architecture.LEVELS, key=lambda a: (architecture.LEVELS[a], a)):
        link = f"[{app}](../apps/{app}.md)"
        if app in architecture.TRUNKS:
            link = f"**{link}** ({t['trunk']})"
        uses = ", ".join(d for d in deps.get(app, []) if d != "core") or "—"
        by = ", ".join(sorted(u for u in used_by.get(app, []) if app != "core")) or "—"
        if app == "core":
            by = t["core_note"].split(" ")[0] if False else ("все" if lang == "ru" else "all")
        lines.append(f"| {link} | {architecture.LEVEL_NAMES[architecture.LEVELS[app]][lang]} | {uses} | {by} |")
    return "\n".join(lines)


def _title_and_gist(path):
    """A page's title (its "# " line) and first sentence after the navigation line."""
    lines = path.read_text(encoding="utf-8").splitlines()
    title = next((l[2:].strip() for l in lines if l.startswith("# ")), path.stem)
    gist = ""
    for l in lines[1:]:
        t = l.strip()
        if not t or t.startswith(("[", "#", "|", "```", "<!--", "!", "-", ">")):
            if gist:
                break
            continue
        gist += (" " if gist else "") + t
        if "." in t:
            break
    gist = gist.split(". ")[0].rstrip(".")
    return title, (gist[:160] + "…") if len(gist) > 160 else gist


def index(path):
    """A hub's list: its subfolders (their hubs) and its pages, with a line on each."""
    folder = path.parent
    rows = []
    for sub in sorted(d for d in folder.iterdir() if d.is_dir() and (d / "README.md").exists()):
        title, gist = _title_and_gist(sub / "README.md")
        rows.append(f"- **[{title}]({sub.name}/README.md)**" + (f" — {gist}" if gist else ""))
    pages = sorted(folder.glob("*.md"))
    if folder.name == "tree":
        # In the tree's order (docs/tree.json), then the rest.
        order = []
        for n in json.loads(DATA.read_text(encoding="utf-8"))["nodes"]:
            name = n["page"].split("#")[0]
            if name not in order:
                order.append(name)
        pages.sort(key=lambda q: (order.index(q.name) if q.name in order else len(order), q.name))
    for page in pages:
        if page.name == "README.md":
            continue
        title, gist = _title_and_gist(page)
        rows.append(f"- [{title}]({page.name})" + (f" — {gist}" if gist else ""))
    return "\n".join(rows)


def render(name, lang, path=None):
    if name == "docs:index":
        return index(path)
    if name == "tree:diagram":
        return diagram(lang)
    if name == "tree:trunk":
        return diagram(lang, only={n["id"] for n in nodes()[1] if n["kind"] == "phase"} | {"base"})
    if name == "tree:table":
        return table(lang)
    if name == "tree:modules":
        return modules(lang)
    if name.startswith("tree:here:"):
        return here(lang, name.split(":", 2)[2])
    if name.startswith("tree:card:"):
        return card(lang, name.split(":", 2)[2])
    raise ValueError(f"unknown generated block {name}")


def pages():
    return sorted(p for lang in LANGS for p in (DOCS / lang).rglob("*.md"))


def update(check=False):
    """Rewrite every generated block; returns the pages that were (or would be) changed."""
    changed = []
    for path in pages():
        lang = path.relative_to(DOCS).parts[0]
        text = path.read_text(encoding="utf-8")

        def fill(m):
            return m.group(1) + render(m.group(2), lang, path) + "\n" + m.group(4)
        new = BLOCK.sub(fill, text)
        if new != text:
            changed.append(path)
            if not check:
                path.write_bytes(new.encode("utf-8"))
    return changed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    changed = update(check=args.check)
    for p in changed:
        print(("out of date: " if args.check else "updated: ") + str(p.relative_to(ROOT)))
    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
