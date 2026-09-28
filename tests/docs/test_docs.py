"""The documentation keeps its rules (docs/ru/architecture/documentation.md)."""
import json
import re
from pathlib import Path

import pytest

from tools import architecture
from tools.docs import tree_doc

ROOT = architecture.ROOT
DOCS = ROOT / "docs"
# Working drafts and research: Russian only.
RU_ONLY = ("architecture/ai-design.md", "architecture/battle-theory.md", "architecture/strategies.md",
           "architecture/tasks/", "research/")
BACK = {"ru": "[← Назад](", "en": "[← Back]("}
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")


def pages(lang):
    return sorted((DOCS / lang).rglob("*.md"))


def rel(path, lang):
    return path.relative_to(DOCS / lang).as_posix()


@pytest.mark.parametrize("lang,other", [("ru", "en"), ("en", "ru")])
def test_every_page_has_its_mirror(lang, other):
    missing = [rel(p, lang) for p in pages(lang)
               if not (DOCS / other / rel(p, lang)).exists() and not (lang == "ru" and rel(p, lang).startswith(RU_ONLY))]
    assert not missing, f"no {other} page for: {missing}"


@pytest.mark.parametrize("lang", ["ru", "en"])
def test_every_page_has_a_back_link_to_an_existing_page(lang):
    bad = []
    for p in pages(lang):
        head = "\n".join(p.read_text(encoding="utf-8").splitlines()[:6])
        at = head.find(BACK[lang])
        if at < 0:
            bad.append(f"{rel(p, lang)}: no back link")
            continue
        target = head[at + len(BACK[lang]):].split(")", 1)[0]
        if not (p.parent / target).resolve().exists():
            bad.append(f"{rel(p, lang)}: back link to a missing page {target}")
    assert not bad, "\n".join(bad)


def test_links_and_pictures_exist():
    bad = []
    for p in sorted(DOCS.rglob("*.md")) + [ROOT / "README.md"]:
        text = re.sub(r"```.*?```", "", p.read_text(encoding="utf-8"), flags=re.S)
        for target in LINK.findall(text):
            if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                continue
            path = target.split("#", 1)[0]
            if path and not (p.parent / path).resolve().exists():
                bad.append(f"{p.relative_to(ROOT).as_posix()}: {target}")
    assert not bad, "missing:\n" + "\n".join(bad)


def test_generated_blocks_match_the_code():
    stale = tree_doc.update(check=True)
    assert not stale, "run: python -m tools.docs.tree_doc\n" + "\n".join(str(p.relative_to(ROOT)) for p in stale)


def test_every_module_has_a_page_in_both_languages():
    apps = sorted(p.name for p in architecture.APPS.iterdir() if p.is_dir())
    missing = [f"{lang}/apps/{a}.md" for a in apps for lang in ("ru", "en") if not (DOCS / lang / "apps" / f"{a}.md").exists()]
    assert not missing, missing


def test_every_tree_node_is_documented():
    data = json.loads(tree_doc.DATA.read_text(encoding="utf-8"))
    documented = {n["id"]: n for n in data["nodes"]}
    code = tree_doc.lua_nodes()
    assert set(code) <= set(documented), f"add to docs/tree.json: {sorted(set(code) - set(documented))}"
    for nid, n in documented.items():
        assert n["lua"] == (nid in code), f"{nid}: 'lua' must say whether the node is in src/apps/tree"
        for field in ("title", "what", "enter"):
            assert n[field].get("ru") and n[field].get("en"), f"{nid}: {field} in both languages"
        if n["lua"] and code[nid]["parent"]:
            assert n.get("baseline_ru"), f"{nid}: baseline_ru"
        page = n["page"].split("#")[0]
        for lang in ("ru", "en"):
            text = (DOCS / lang / "tree" / page).read_text(encoding="utf-8")
            assert f"generated:tree:card:{nid}" in text, f"{lang}/tree/{page}: card of {nid}"
            if n["lua"]:
                assert f"generated:tree:here:{nid}" in text, f"{lang}/tree/{page}: place of {nid}"


def test_a_node_s_module_pages_link_to_the_node():
    data = json.loads(tree_doc.DATA.read_text(encoding="utf-8"))
    bad = []
    for n in data["nodes"]:
        if not n["lua"]:
            continue
        for lang in ("ru", "en"):
            if not any(f"../tree/{n['page']}" in (DOCS / lang / "apps" / f"{m}.md").read_text(encoding="utf-8")
                       for m in n["modules"]):
                bad.append(f"{lang}: no module page of {n['id']} links to tree/{n['page']}")
    assert not bad, "\n".join(bad)
