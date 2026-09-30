"""The rules that let the code grow, checked.

* levels (tools/architecture.py): every app has a level; modules depend only
  downwards;
* pure code (services, data) never touches the engine;
* no unit, race or lord keys in logic: only in data modules (none today; the
  rule stays for future modules);
* module size, with the exceptions and why.
"""
import re
from pathlib import Path

from tools import architecture

SRC = Path(__file__).resolve().parents[2] / "src"
APPS = SRC / "apps"

LEVELS = architecture.LEVELS
MAX_LINES = 1000  # user, 28.09.2026 (was 500)
# Longer modules, and why (to be split when they are next changed).
TOO_LONG = {}
REQUIRE = re.compile(r"require\('apps\.([a-z_]+)\.")
UNIT_KEY = re.compile(r"['\"]wh[0-9]?_[a-z0-9_]+['\"]")
ENGINE = re.compile(r"\bbm:|get_context_value|script_unit|battle_manager|:unique_ui_id\(")


def lua_files():
    return sorted(APPS.rglob("*.lua"))


def app_of(path):
    return path.relative_to(APPS).parts[0]


def test_every_app_has_a_level():
    apps = {p.name for p in APPS.iterdir() if p.is_dir()}
    assert not apps - set(LEVELS), f"give these apps a level: {sorted(apps - set(LEVELS))}"


def test_dependencies_go_only_down():
    bad = []
    for path in lua_files():
        app = app_of(path)
        for dep in set(REQUIRE.findall(path.read_text(encoding="utf-8"))) - {app}:
            if LEVELS[dep] > LEVELS[app]:
                bad.append(f"{path.relative_to(APPS)} -> {dep} (up: {LEVELS[app]} -> {LEVELS[dep]})")
    assert not bad, "\n".join(bad)


def test_pure_code_never_touches_the_engine():
    bad = [str(p.relative_to(APPS)) for p in lua_files()
           if p.stem in ("services", "data") and ENGINE.search(p.read_text(encoding="utf-8"))]
    assert not bad, bad


def test_no_unit_keys_in_logic():
    bad = []
    for p in lua_files():
        if p.stem == "data":
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("--", 1)[0]
            if UNIT_KEY.search(code):
                bad.append(f"{p.relative_to(APPS)}:{n}: {line.strip()}")
    assert not bad, "keys belong in data modules:\n" + "\n".join(bad)


def test_modules_are_not_too_long():
    long = {str(p.relative_to(APPS)).replace("\\", "/"): len(p.read_text(encoding="utf-8").splitlines())
            for p in lua_files()}
    over = {k: v for k, v in long.items() if v > MAX_LINES and k not in TOO_LONG}
    assert not over, f"split these modules (or add them to TOO_LONG with a reason): {over}"
    stale = [k for k in TOO_LONG if long.get(k, 0) <= MAX_LINES]
    assert not stale, f"no longer too long, remove from TOO_LONG: {stale}"
