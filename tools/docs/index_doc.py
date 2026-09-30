"""The generated page lists of the documentation's hubs (docs/<lang>/**/README.md).

    python -m tools.docs.index_doc           # rewrite the generated blocks
    python -m tools.docs.index_doc --check   # exit 1 if a block is out of date (tests/docs)

A generated block sits between <!-- generated:docs:index --> and
<!-- /generated --> in a hub page: a list of the hub's subfolders (their hubs)
and pages, each with its title and first sentence. Only the inside of a block
is rewritten; the rest of the page is written by hand.
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
LANGS = ("ru", "en")
BLOCK = re.compile(r"(<!-- generated:([a-z:_0-9]+) -->\n)(.*?)(<!-- /generated -->)", re.S)


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
    for page in sorted(folder.glob("*.md")):
        if page.name == "README.md":
            continue
        title, gist = _title_and_gist(page)
        rows.append(f"- [{title}]({page.name})" + (f" — {gist}" if gist else ""))
    return "\n".join(rows)


def render(name, path):
    if name == "docs:index":
        return index(path)
    raise ValueError(f"{path.relative_to(ROOT).as_posix()}: unknown generated block {name} "
                     "(only docs:index is generated)")


def pages():
    return sorted(p for lang in LANGS for p in (DOCS / lang).rglob("*.md"))


def update(check=False, paths=None):
    """Rewrite every generated block; returns the pages that were (or would be) changed."""
    changed = []
    for path in paths or pages():
        text = path.read_text(encoding="utf-8")

        def fill(m):
            return m.group(1) + render(m.group(2), path) + "\n" + m.group(4)
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
