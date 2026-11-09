# ops — вокруг работы

Образ Docker `snake-ai-trainer` (локальный, Dockerfile нет; не пересобирать): Python 3.12, torch 2.13 + CUDA 13.2, g++, pytest.

| файл | что |
|---|---|
| `dock.sh` | модуль проекта в контейнере, корень на `/repo`: `DOCK_NAME=agent-<задача> DOCK_CPUS=6 bash ops/dock.sh pytest -q` (`DOCK_GPU=0` — без видеокарты, `DOCK_MEM`, `DOCK_BUILD`) |
| `leftovers.py` | что висит: наши контейнеры, замок видеокарты `build/gpu-train.lock`, циклы ожидания, игра, метки `state/`; `python -m ops.leftovers [--stop ИМЯ \| --unlock \| --kill]` |
| `wait.sh` | ждать условие с пределом: `bash ops/wait.sh -t 3600 <файл> <regex>`, `container <имя>`, `gone <путь>` |
| `guard.py`, `guard.ps1`, `guard.sh` | метки игры: `state/HUMAN_PLAYING` (пользователь играет — наш бой не стартует, кроме его собственного `human`), `state/probe_game.lock` (раннер держит игру; его запуски проходят по `BAI_LOCK_HELD`); то же правило — в `bridge/launcher/launch.ps1` |
| `python.sh` | `$PY` для скриптов: `.venv` проекта, иначе `../TWW3-batle-ai/.venv` |

Хук коммита — `.githooks/pre-commit` (`git config core.hooksPath .githooks`): быстрые тесты на хосте, секунды;
тесты с torch (меры учений, паритет `contract/`) — в контейнере: `bash ops/dock.sh pytest -q`.
