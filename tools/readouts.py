"""Readout catalogue: everything we can read about units and the battle.

data/readouts/catalog.json is the single list of readouts, tracked or not.
config/readouts/<profile>.json chooses what a particular run collects, so we
only track what helps to improve the AI.

  python -m tools.readouts docs            # regenerate docs/{ru,en}/game/readouts.md
  python -m tools.readouts check           # validate every profile against the catalogue
  python -m tools.readouts show combat-focus
"""
import argparse
import json
import sys

from tools import config as project

CATALOG = project.ROOT / "data" / "readouts" / "catalog.json"
PROFILES = project.CONFIG_DIR / "readouts"

STATUS = {"ru": {"tracked": "✅ собираем", "available": "⬜ есть в игре"},
          "en": {"tracked": "✅ tracked", "available": "⬜ in the game"}}
ACCESS = {"ru": {"own": "только свои", "enemy": "свои и видимые враги", "public": "общее для всех",
                 "truth": "только полная сводка"},
          "en": {"own": "own only", "enemy": "own and visible enemies", "public": "public",
                 "truth": "full summary only"}}
TEXT = {
    "ru": {"title": "Каталог показателей", "other": "English",
           "intro": "Всё, что можно прочитать об отрядах и бое: и то, что мы уже собираем, и то, что есть "
                    "в игре, но ещё не проверено. Для каждого прогона выбираем профиль — список того, что "
                    "собирать ([config/readouts](../../../config/readouts/)). Страница создана из "
                    "[data/readouts/catalog.json](../../../data/readouts/catalog.json) командой "
                    "`python -m tools.readouts docs`; правьте JSON, а не эту страницу.",
           "legend": "**Статус:** ✅ собираем и проверено в игре · ⬜ есть в API игры, не проверено. "
                     "**Доступ для ИИ:** только свои · свои и видимые враги · общее для всех · только полная "
                     "сводка (никогда не вход ИИ).",
           "summary": "Всего {n}: собираем {t}, ещё не проверено {a}.",
           "cols": "| ID | Что это | Статус | Доступ для ИИ |"},
    "en": {"title": "Readout catalogue", "other": "Русский",
           "intro": "Everything readable about units and the battle: what we already collect and what the "
                    "game offers but we have not tested yet. Each run picks a profile — the list of what to "
                    "collect ([config/readouts](../../../config/readouts/)). Generated from "
                    "[data/readouts/catalog.json](../../../data/readouts/catalog.json) by "
                    "`python -m tools.readouts docs`; edit the JSON, not this page.",
           "legend": "**Status:** ✅ tracked and verified in game · ⬜ in the game's API, not tested. "
                     "**AI access:** own only · own and visible enemies · public · full summary only (never "
                     "an AI input).",
           "summary": "Total {n}: tracked {t}, not yet tested {a}.",
           "cols": "| ID | What it is | Status | AI access |"},
}


def load_catalog():
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def load_profile(name):
    return json.loads((PROFILES / f"{name}.json").read_text(encoding="utf-8"))


def render(lang, catalog):
    t = TEXT[lang]
    other = "en" if lang == "ru" else "ru"
    items = catalog["readouts"]
    tracked = sum(r["status"] == "tracked" for r in items)
    lines = [f"# {t['title']}", "",
             f"[{'← Назад' if lang == 'ru' else '← Back'}](README.md) · "
             f"[{'Знания об игре' if lang == 'ru' else 'Game knowledge'}](README.md) · "
             f"[{t['other']}](../../{other}/game/readouts.md)", "", t["intro"], "", t["legend"], "",
             t["summary"].format(n=len(items), t=tracked, a=len(items) - tracked), ""]
    for group, names in catalog["groups"].items():
        rows = [r for r in items if r["group"] == group]
        if not rows:
            continue
        lines += [f"## {names[lang]}", "", t["cols"], "|---|---|---|---|"]
        for r in rows:
            lines.append(f"| `{r['id']}` | {r[lang]} | {STATUS[lang][r['status']]} | {ACCESS[lang][r['access']]} |")
        lines.append("")
    lines += ["<details><summary>" + ("Технические вызовы" if lang == "ru" else "Technical calls") + "</summary>", "",
              "| ID | API | " + ("Модуль" if lang == "ru" else "Module") + " |", "|---|---|---|"]
    for r in items:
        lines.append(f"| `{r['id']}` | `{r['api']}` | {r['module'] or '—'} |")
    lines += ["", "</details>", ""]
    return "\n".join(lines)


def check():
    catalog = load_catalog()
    ids = {r["id"] for r in catalog["readouts"]}
    groups = set(catalog["groups"])
    problems = []
    for r in catalog["readouts"]:
        if r["group"] not in groups:
            problems.append(f"catalog: {r['id']} has unknown group {r['group']}")
        if r["status"] not in ("tracked", "available") or r["access"] not in ACCESS["en"]:
            problems.append(f"catalog: {r['id']} has bad status/access")
    for path in sorted(PROFILES.glob("*.json")):
        profile = json.loads(path.read_text(encoding="utf-8"))
        for rid in profile.get("track", []):
            if rid not in ids:
                problems.append(f"{path.name}: unknown readout {rid}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("docs")
    sub.add_parser("check")
    show = sub.add_parser("show")
    show.add_argument("profile")
    args = parser.parse_args(argv)
    if args.command == "docs":
        catalog = load_catalog()
        for lang in ("ru", "en"):
            out = project.ROOT / "docs" / lang / "game" / "readouts.md"
            out.write_text(render(lang, catalog), encoding="utf-8")
            print("written", out.relative_to(project.ROOT))
        return 0
    if args.command == "check":
        problems = check()
        for p in problems:
            print(p)
        print("ok" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    catalog = {r["id"]: r for r in load_catalog()["readouts"]}
    profile = load_profile(args.profile)
    print(f"{args.profile}: {profile.get('description', '')}")
    for rid in profile["track"]:
        r = catalog.get(rid)
        print(f"  {rid:32} {r['ru'] if r else '??? not in catalogue'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
