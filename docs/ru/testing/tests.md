# Тесты без игры

[← Назад](README.md) · [Документация](../README.md) · [English](../../en/testing/tests.md)

Тесты запускают **тот же Lua 5.1**, что и WH3, через `lupa` и проверяют код без запуска игры.

```bash
.venv/Scripts/python -m pytest
.venv/Scripts/python -m pytest tests/apps/units -v
```

Они проверяют сервисы, адаптеры на поддельных объектах, связку приложений в
точках входа, инструменты вне игры и правила документации. Механику боя WH3
они не моделируют: поведение движка проверяется только запуском игры
([запуск](../launch/run.md)).

## Уровни проверок

Правило от 27.09.2026: маленькие модули проверяются сами, выше проверяется только
то, что они правильно вызваны. По упавшему уровню сразу видно, где поломка.

| Уровень | Что проверяет | Где |
|---|---|---|
| 1. Функции | Маленькая функция делает своё | `tests/apps/<модуль>/test_*.py` |
| 2. Модуль по замерам | На данных из игры модуль даёт то же, что игра: направления движка, рамка радара, сенсоры из kit | `tests/apps/orders/test_facing.py`, `tests/apps/map/test_services.py`, `tests/apps/units/` |
| 3. Сборка | Точка входа на поддельной игре вызывает модули по порядку и пишет нужные события | `tests/entries/test_entries.py` |
| 4. Проверки движка | Факты об игре, на которые опирается код | Прогоны в игре с вердиктом; гонять при обновлении игры и при новом факте движка |

Кроме уровней:

| Проверка | Что ловит | Где |
|---|---|---|
| Архитектура | у каждого приложения есть уровень (сейчас все — основа), зависимости только вниз, чистый код без движка, нет ключей отрядов в логике, модуль не длиннее 1000 строк | `tests/architecture/` |
| Документация | зеркало языков, «← Назад», живые ссылки и картинки, свежие оглавления, страница у каждого модуля | `tests/docs/` ([правила](../architecture/documentation.md)) |
| Сборка | формат PFH5, бандлер, каждая цель собирается и компилируется, True Sight в каждом pack | `tests/tools/test_build.py` |

**Тесты идут сами:**
- **перед каждым коммитом** (~5 с) — хук `.githooks/pre-commit`. Включить один раз:
  `git config core.hooksPath .githooks`; пропустить — только с причиной, `--no-verify`;
- **на каждый push** — все тесты в GitHub Actions (`.github/workflows/tests.yml`).

## Структура

Как в photo-fixing, другом проекте автора, тесты повторяют структуру кода:

```text
tests/
├── lua_runtime.py          Lua-среда: require('apps.x.y') грузит src/apps/x/y.lua
├── apps/<приложение>/      тесты сервисов и адаптеров
├── entries/                smoke-тесты точек входа на поддельном bm
│   └── fake_battle.lua     минимальная подделка battle manager
├── tools/                  формат pack, бандлер, сборка всех целей, ростер, показатели, разборы
├── architecture/           правила кода
└── docs/                   правила документации
```

## Как писать тест

```python
import pytest
from tests.lua_runtime import load, new_runtime


@pytest.fixture
def lua():
    runtime = new_runtime()
    runtime.globals().intel = load(runtime, "apps.intel.adapter")
    runtime.execute("""
        position_reads = 0
        function enemy(visible)
            local u = {visible = visible}
            function u:is_visible_to_alliance() return self.visible end
            function u:position()
                position_reads = position_reads + 1
                return {get_x=function() return 10 end, get_y=function() return 0 end, get_z=function() return 20 end}
            end
            return u
        end
    """)
    return runtime


def test_hidden_enemy_position_is_never_read(lua):
    lua.execute("""
        intel.observe(enemy(false), {}, 'enemy-2', 0, {})
        assert(position_reads == 0)
    """)
```

Правила:

- **Проверки пишутся на Lua** внутри `lua.execute`: так тест видит те же
  типы, `nil` и `0/0`, что и игра.
- **Адаптер тестируется поддельным объектом**: таблица с нужными методами.
  Для проверки «скрытое не читаем» подделка считает вызовы или бросает
  ошибку при запрещённом чтении (как выше и в `tests/apps/units/test_range_adapter.py`).
- **Точка входа** проверяется через `tests/entries/fake_battle.lua`: колбэки
  копятся в очереди, тест прокручивает их `bm:pump()` и `bm:tick()`.
- Новый метод движка в точке входа → добавить его в `fake_battle.lua`.

## Что покрыто

| Область | Файл |
|---|---|
| Сенсор состояния (перенесён из kit) | `tests/apps/units/test_state_adapter.py` |
| Сенсор дальности (перенесён из kit) | `tests/apps/units/test_range_adapter.py` |
| `core`: чтение, JSON, ошибки, защита колбэков | `tests/apps/core/test_core.py` |
| Подпись здоровья для правила застоя | `tests/apps/battle/test_adapter.py` |
| Рамка радара и сетка | `tests/apps/map/test_services.py` |
| Конец захода: `arrived`, `stopped`, `stuck`, `timeout` | `tests/apps/navigation/test_services.py` |
| Память видимости | `tests/apps/intel/test_intel.py` |
| 64 направления движка | `tests/apps/orders/test_facing.py` |
| Планировщик CA: собравшиеся и стоящие под обстрелом отряды | `tests/apps/orders/test_planner_adapter.py` |
| Точки входа целиком на поддельной игре | `tests/entries/test_entries.py` |
| PFH5, бандлер, все цели сборки | `tests/tools/test_build.py` |
| Ростер, каталог показателей | `tests/tools/test_roster.py`, `tests/tools/test_readouts.py` |
| Разборы: препятствия на карте, группы штатного ИИ | `tests/tools/test_obstacles.py`, `tests/tools/test_enemy_layout.py` |
