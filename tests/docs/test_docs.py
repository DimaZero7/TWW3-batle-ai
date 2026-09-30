"""The documentation keeps its rules (docs/ru/architecture/documentation.md)."""
import re

import pytest

from tools import architecture
from tools.docs import index_doc

ROOT = architecture.ROOT
DOCS = ROOT / "docs"
# Research write-ups: Russian only.
RU_ONLY = ("research/",)
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
    stale = index_doc.update(check=True)
    assert not stale, "run: python -m tools.docs.index_doc\n" + "\n".join(str(p.relative_to(ROOT)) for p in stale)


def test_every_module_has_a_page_in_both_languages():
    apps = sorted(p.name for p in architecture.APPS.iterdir() if p.is_dir())
    missing = [f"{lang}/apps/{a}.md" for a in apps for lang in ("ru", "en") if not (DOCS / lang / "apps" / f"{a}.md").exists()]
    assert not missing, missing
