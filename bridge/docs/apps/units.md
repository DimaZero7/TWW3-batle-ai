# units

[← Назад](README.md) · [Документация](../README.md) · [Приложения](../architecture/apps.md)

Карточка отряда, движение и позиции его бойцов. Код: `bridge/lua/apps/units/`.

## card_adapter — карточка и профиль отряда

| Функция | Что делает |
|---|---|
| `stats(cco, unit)` | Вся карточка `UnitDetailsContext.StatList` по порядку: `{key, Value, DisplayedValue, ValueBase}` |
| `details(cco, unit)` | `UnitDetailsContext.Mass`, `Name` и поля отряда `HealthMax`, `NumEntitiesInitial` |
| `profile(unit)` | Тип, командир, класс, вид отряда (пехота, конница…), бойцов, скорости, дальность, стрелы, способности |

Неизвестное пишется как `'unknown:<причина>'`. Модуль грузит наблюдатель боя человека
([observer_adapter](telemetry.md#observer_adapter)).

## formation_adapter — движение и бойцы своего отряда

| Функция | Что делает |
|---|---|
| `motion(unit)` | Позиция, заданная позиция, направление, заданные направление и ширина, число бойцов, флаги движения. Нечитаемое пропускается; без позиции — ошибка |
| `soldiers(cco, unit)` | CCO `ManList.At(i).Position` каждого бойца → `{status = 'ok', count, xz_dm = {x1, z1, ...}}` в целых дециметрах или `{status = 'unavailable', reason}` |

Пользуются: [move_probe](entries.md#move_probe), проба стрельбы ([missile_probe](entries.md#missile_probe)) и
наблюдатель боя человека. Места бойцов врага читаются только в записи для исследования (наблюдатель, пробы), никогда
не во входе сети: они выдали бы скрытых.
