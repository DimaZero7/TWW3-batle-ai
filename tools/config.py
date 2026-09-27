"""Project paths and local settings.

config/default.json is committed; config/local.json (git-ignored) overrides
it per machine, like photo-fixing's config.toml over default_config.toml.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCENARIOS = ROOT / "scenarios"
BUILD = ROOT / "build"
CONFIG_DIR = ROOT / "config"


def load():
    settings = json.loads((CONFIG_DIR / "default.json").read_text(encoding="utf-8"))
    local = CONFIG_DIR / "local.json"
    if local.exists():
        settings.update(json.loads(local.read_text(encoding="utf-8")))
    return settings
