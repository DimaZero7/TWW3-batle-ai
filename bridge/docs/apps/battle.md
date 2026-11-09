# battle

[← Назад](README.md) · [Документация](../README.md) · [Приложения](../architecture/apps.md)

Контекст боя: стороны, армии, отряды и роли. Единственное место, которое
обходит `bm:alliances()`; остальные приложения получают готовые списки.
Код: `bridge/lua/apps/battle/`.

## adapter

| Функция | Что делает |
|---|---|
| `unsupported_reason(bm)` | Первый неподдерживаемый тип боя (`is_from_campaign`, `is_multiplayer`, `is_replay`, `is_quest_battle`, `is_tutorial`, `is_siege_battle`, `is_ambush_battle`) или `nil` |
| `read_sides(bm, expected_units?)` | `{side, alliance, army, units}` для двух сторон; проверяет одну армию на сторону и, если задано, число отрядов |
| `find_by_name(side, script_name)` | Отряд по `script_name` из XML |
| `speed_guard(bm, speed, on_restore?)` | Держит заданную скорость до фазы `Complete`: движок сбрасывает её, когда исход решён. Возвращает `stop()` |
| `read_roles(bm, record?)` | Атакующий/защитник по `alliance:is_attacker()`; никогда не выводится из индекса или места появления |
| `deadline(bm, real_ms, on_expire, name?)` | Страховка по реальному времени: один раз вызывает `on_expire` через `real_ms` мс настоящего времени при любой скорости и фазе боя, чтобы тестовый бой не шёл вечно. Возвращает `cancel()` |
| `health_signature(units)` | Живые бойцы и здоровье отрядов одной строкой `"бойцы:здоровье×10000"`. Строка, а не число: числа в Lua игры одинарной точности, и в большой сумме мелкий урон потерялся бы. Нечитаемый отряд даёт 0 |

## services

| Функция | Что делает |
|---|---|
| `verify_roles(roles, expected)` | Сверяет роли из сценария с ролями движка |
| `new_stall_detector(window_ms)` | Детектор застоя: бой стоит, если `health_signature` всех отрядов не менялась `window_ms` мс модельного времени. `update(now_ms, signature)` возвращает `true`, когда застой наступил; `quiet_ms(now_ms)` — сколько уже без изменений |
| `opponent(side)` | `1 → 2`, `2 → 1` |

## winds — ветра магии армии

`read(army, names?)` — запас, из которого платят маги армии: `cur` (`army:winds_of_magic_current()`), `rate`
(`winds_of_magic_remaining_recharge_rate()`), `res` (резерв); `discover(army)` — какие чтения отвечают числом;
`modify(army, current, reserve)` — добавить ветра скриптом. Мост кладёт свои ветра в состояние (`winds`,
[bridge](bridge.md#файлы)).

## Разметка вживую

Модули боя, который играет человек и одновременно размечает свои группы (цель `human --live-label`,
[запуск](../launch/run.md#бой-который-играет-человек)). Отрядами они не командуют.

| Модуль | Что делает |
|---|---|
| `live_label` | Раз в секунду боя — документ состояния в формате моста (`tww3_bai_nn_state.json`) плюс `groups` (свои — группы игры с панели карточек, чужие — с карточек врага), `selection`, `live = true`; нажатые сочетания клавиш — событие `live_shortcut`; с `--live-test` — команды проверки из `tww3_bai_live_cmd.txt` |
| `enemy_cards` | Карточки врага в полосе карточек после своих: картинка, бойцы, полоса здоровья; клик выбирает отряд, кнопка собирает группу |
| `box_pick`, `screen_proj` | Рамка выделения по врагам на поле: какой отряд под рамкой — по его строю на экране (проекция камерой, без движка — чистый `screen_proj`) |
| `label_chat` | Поле текста в игре: человек пишет, почему делает так; сообщения уходят с документом состояния |
| `label_overlay` | Пометки поверх боя из файла `tww3_bai_nn_overlay.txt` (пишет разметчик `game/label/live.py`): значки над отрядами, углы зон, строка текста, камера |

Как это задумано и что проверено в игре — `sim/evidence/diag_chain/live_label.md`
([разбор](../../../sim/evidence/diag_chain/live_label.md)).

## Пример

```lua
local battle = require('apps.battle.adapter')

local reason = battle.unsupported_reason(bm)
if reason then return end
local sides = battle.read_sides(bm, 1)   -- ровно один отряд на сторону
local enemy = sides[2].units[1]
```

Имена отрядов из XML (`script_name`) переживают границы Lua-окружений,
поэтому точки входа узнают свой сценарий по ним, а не по глобальным
переменным.
