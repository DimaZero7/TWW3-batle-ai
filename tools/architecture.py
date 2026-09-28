"""The code's architecture as data: levels of the apps and their trunks.

Used by tests/architecture (the rules are checked) and tools/docs (the
documentation's module map is drawn from it), so there is one truth.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPS = ROOT / "src" / "apps"

# 0 — base: engine access, data, rules of the game; 1 — strategic (phase 1);
# 2 — tactical (phase 2); the combat phases will be 3.
LEVELS = {
    "core": 0, "battle": 0, "map": 0, "navigation": 0, "units": 0, "intel": 0, "orders": 0, "telemetry": 0,
    "observation": 0, "sandbox": 0, "ai": 0, "deployment": 0, "tree": 0, "vision": 0, "battlefield": 0,
    "mask": 0, "reach": 0, "missile": 0,
    "assessment": 1, "strategy": 1, "formation": 1, "plan": 1,
    "alignment": 2, "approach": 2, "logistics": 2, "tactics": 2,
}
LEVEL_NAMES = {0: {"ru": "Основа", "en": "Base"}, 1: {"ru": "Стратегический", "en": "Strategic"},
               2: {"ru": "Тактический", "en": "Tactical"}, 3: {"ru": "Боевой", "en": "Combat"}}
# The trunk of a level: the only module there that calls the level's other nodes.
TRUNKS = {"plan", "tactics"}
REQUIRE = re.compile(r"require\('apps\.([a-z_]+)\.")


def dependencies():
    """{app: sorted apps it requires} from the source."""
    deps = {}
    for path in sorted(APPS.rglob("*.lua")):
        app = path.relative_to(APPS).parts[0]
        deps.setdefault(app, set()).update(set(REQUIRE.findall(path.read_text(encoding="utf-8"))) - {app})
    return {k: sorted(v) for k, v in deps.items()}
