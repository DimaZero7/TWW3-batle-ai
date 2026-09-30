"""The code's architecture as data: the level of every app in src/apps.

Used by tests/architecture (the rules are checked) and tests/docs (every app
has a page). Every app today is the base (level 0): engine access, data and the
rules of the game. A future level above it may use the base, never the other way.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPS = ROOT / "src" / "apps"

# 0 - base: engine access, data, rules of the game.
LEVELS = {
    "core": 0, "battle": 0, "map": 0, "navigation": 0, "units": 0, "intel": 0, "orders": 0, "telemetry": 0,
    "observation": 0,
}
