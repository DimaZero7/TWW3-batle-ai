# Setup and configuration

[Documentation](../README.md) · [Русский](../../ru/environment/setup.md)

## Requirements

- Windows with Total War: WARHAMMER III installed (Steam, app id 1142710);
- Python 3.10+;
- PowerShell 5.1 (ships with Windows) to launch the game;
- Git.

The game, its files and third-party mods are not part of the repository.

## Python environment

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt`:

| Package | Why |
|---|---|
| `lupa` | Lua 5.1 inside Python: tests and bundle syntax check at build time |
| `pytest` | Test runner |
| `numpy`, `matplotlib` | `tools/analysis`: height maps, slopes, passages |

The build works without `lupa`, but `manifest.json` then says `"syntax_checked": false`.

## Machine settings: `config/local.json`

As in photo-fixing (`default_config.toml` → `config.toml`): Git holds
`config/default.json`, personal settings go to the git-ignored
`config/local.json`. Values from `local.json` override the defaults.

```bash
cp config/local.example.json config/local.json
```

| Key | Value |
|---|---|
| `game_dir` | Game folder containing `Warhammer3.exe` |
| `workshop_dir` | Workshop folder `...\workshop\content\1142710` |
| `dependencies` | Mods the launcher lists before our pack. Empty by default (vanilla). A True Sight example is in `local.example.json` |

## Non-ASCII paths

The project lives in a folder with Cyrillic letters. Lua 5.1 on Windows
cannot open such paths itself, so the test loader reads sources through
Python (`tests/lua_runtime.py`). In the game this does not matter: scripts
are read from the pack and telemetry files are written to the game folder by
relative path.

## Check

```bash
.venv/Scripts/python -m pytest
```

All tests must pass. Next: [building a pack](../launch/build.md).
