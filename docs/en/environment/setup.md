# Setup and configuration

[← Back](README.md) · [Documentation](../README.md) · [Русский](../../ru/environment/setup.md)

What to install and how to set up the machine to build and run battles.

## Requirements

- Windows with Total War: WARHAMMER III installed (Steam, app id 1142710);
- Python 3.10+;
- PowerShell 5.1 (ships with Windows) to launch the game;
- Git.

The game, its files and third-party mods are not part of the repository.

**Subscribe to the
[True Sight](https://steamcommunity.com/sharedfiles/filedetails/?id=3628832922)
mod in Steam Workshop: battles do not run without it ([details](../launch/run.md#required-mod-true-sight)).**

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
| `numpy`, `scipy`, `matplotlib` | `tools/analysis`: height maps, slopes, passages; `tools/nn/gamedata.py`: arena records as arrays |

The build works without `lupa`, but `manifest.json` then says `"syntax_checked": false`.

## For the training data

- **Python 3.14+** is needed only to read the game's database: the files in
  `db.pack` of the game's `data` folder are zstd-compressed, and `compression.zstd` is in the standard
  library since 3.14. Run: `py -3.14 -m tools.nn.gamedb` → `config/nn/game_rules.json`.
- **Docker with PyTorch and CUDA** — for training later: `bash tools/nn/dock.sh <module>`
  runs a project module in the `snake-ai-trainer` container (the repository mounted at `/repo`).

More: [data for training the network](../training/README.md).

## Machine settings: `config/local.json`

As in photo-fixing, another project of the author (`default_config.toml` → `config.toml`): Git holds
`config/default.json`, personal settings go to the git-ignored
`config/local.json`. Values from `local.json` override the defaults.

```bash
cp config/local.example.json config/local.json
```

| Key | Value |
|---|---|
| `game_dir` | Game folder containing `Warhammer3.exe` |
| `workshop_dir` | Workshop folder `...\workshop\content\1142710` |

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
