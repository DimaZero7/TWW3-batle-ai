# Тесты без игры

[← Назад](README.md) · [Документация](../README.md) · [English](../../en/testing/tests.md)

```bash
.venv/Scripts/python -m pytest
.venv/Scripts/python -m pytest tests/apps/units -v
```

Тесты запускают **тот же Lua 5.1**, что и WH3, через `lupa`. Они проверяют
контракты, сервисы, адаптеры на поддельных объектах и связку приложений в
точках входа. Механику боя WH3 они не моделируют: поведение движка
проверяется только запуском игры ([запуск](../launch/run.md)).

## Уровни проверок

Правило от 27.09.2026 ([архитектура ИИ](../architecture/ai-design.md)): маленькие
модули проверяются сами, выше проверяется только то, что они правильно вызваны.
По упавшему уровню сразу видно, где поломка.

| Уровень | Что проверяет | Где |
|---|---|---|
| 1. Функции | Маленькая функция делает своё | `tests/apps/<модуль>/test_services.py` |
| 2. Модуль по договору | На заданных армиях модуль принимает нужное решение: стратегия выбрана (или отклонена по нужным условиям), строй без пересечений | `tests/cases/start/*.json` + `tests/apps/strategy/test_cases.py`: новый случай — новый файл |
| 3. Сборка цепочки | Модули вызваны по порядку и передали друг другу данные; решения не проверяются — модули подменены шпионами | `tests/apps/plan/`, точки входа на поддельной игре |
| 4. Проверки движка | Факты об игре, на которые опирается ИИ | Прогоны в игре с вердиктом; гонять при обновлении игры и при новом факте движка ([будущие задачи](../architecture/tasks/backlog.md)) |

**Добавлено 28.09.2026** (правила [5.1](../architecture/ai-design.md#51-обязательные-правила-масштабирования-28092026)):

| Проверка | Что ловит | Где |
|---|---|---|
| Эталоны симулятора | правка в одном модуле тихо изменила бой: строй, места, решения сближения и их время, время манёвров, окно, перестрелку (15 армий, допуск 0,05) | `tests/golden/`; сравнить — `python -m tools.sim.golden`, обновить — `--update` с объяснением в коммите |
| Ветки дерева | ветка выключена = её базовое поведение | `tests/tools/test_tree_branches.py` |
| Физика симулятора | отряд прошёл или встал на препятствие; ходок не обходит скалу (задача № 27) | `tests/tools/test_sim_physics.py`, [симулятор](simulator.md) |
| Архитектура | уровни и `require`, чистый код без движка, нет ключей отрядов в логике, размер модулей, дерево, бюджет времени решения | `tests/architecture/` |

**Тесты идут сами**: на каждый push — GitHub Actions (`.github/workflows/tests.yml`);
перед коммитом — хук `.githooks/pre-commit` (включить один раз:
`git config core.hooksPath .githooks`; пропустить — только с причиной, `--no-verify`).

В уровне 2 проверяются **свойства** (прикрыто, без пересечений), а не точные
числа, чтобы настройка параметров не ломала тесты зря.

## Структура

Как в photo-fixing, тесты повторяют структуру кода:

```text
tests/
├── lua_runtime.py          Lua-среда: require('apps.x.y') грузит src/apps/x/y.lua
├── apps/<приложение>/      тесты сервисов, контрактов и адаптеров
├── entries/                smoke-тесты точек входа на поддельном bm
│   └── fake_battle.lua     минимальная подделка battle manager
└── tools/                  формат pack, бандлер, сборка всех целей
```

## Как писать тест

```python
import pytest
from tests.lua_runtime import load, new_runtime


@pytest.fixture
def lua():
    runtime = new_runtime()
    runtime.globals().orders = load(runtime, "apps.orders.contract")
    return runtime


class TestValidate:
    def test_hidden_target_is_rejected(self, lua):
        lua.execute("""
            local own = {spears = {alive = true, position = {x=0, z=0}, kind = 'spearmen'}}
            local enemies = {hidden = {visibility = 'not_visible'}}
            local ok, err = pcall(orders.validate,
                {{unit_id='spears', action='attack', target_id='hidden', mode='melee'}}, own, enemies, 14)
            assert(not ok and err:find('target not visible', 1, true))
        """)
```

Правила:

- **Проверки пишутся на Lua** внутри `lua.execute`: так тест видит те же
  типы, `nil` и `0/0`, что и игра.
- **Адаптер тестируется поддельным объектом**: таблица с нужными методами.
  Для проверки «скрытое не читаем» подделка считает вызовы или бросает
  ошибку при запрещённом чтении (см. `tests/apps/units/test_range_adapter.py`).
- **Точка входа** проверяется через `tests/entries/fake_battle.lua`: колбэки
  копятся в очереди, тест прокручивает их `bm:pump()` и `bm:tick()`.
- Новый метод движка в точке входа → добавить его в `fake_battle.lua`.

## Что покрыто

| Область | Файл |
|---|---|
| Сенсор состояния (перенесён из kit) | `tests/apps/units/test_state_adapter.py` |
| Сенсор дальности (перенесён из kit) | `tests/apps/units/test_range_adapter.py` |
| `core`: чтение, JSON, ошибки | `tests/apps/core/test_core.py` |
| Рамка радара и сетка | `tests/apps/map/test_services.py` |
| Память видимости | `tests/apps/intel/test_intel.py` |
| Политики и контракт ИИ | `tests/apps/ai/test_policies.py` |
| Проверка команд | `tests/apps/orders/test_contract.py` |
| Расстановка v2 | `tests/apps/deployment/test_deployment.py` |
| Песочница политик | `tests/apps/sandbox/test_sandbox.py` |
| Дуэль и захват карты целиком | `tests/entries/test_entries.py` |
| PFH5, бандлер, все цели сборки | `tests/tools/test_build.py` |
| Проигрыватель боя: запись из игры и симуляции, страница | `tests/tools/test_viewer.py` |
