# ai

[← Назад](README.md) · [Документация](../README.md) · [Приложения](../architecture/apps.md) · [English](../../en/apps/ai.md)

Место для **нашего ИИ**: наблюдение на входе, решение на выходе. Никаких
объектов движка — только обычные таблицы, поэтому всё тестируется без игры.
Код: `src/apps/ai/`.

## contract

- `profile(name, speed?)` — профиль армии:

  | Профиль | Отрядов | Команд за шаг | Состав | Скорость |
  |---|---|---|---|---|
  | `empire-7-v1` | 7 | 14 | лорд, 4 копейщика, 2 лучника | ×20 |
  | `empire-15-v1` | 15 | 30 | лорд, 8 копейщиков, 6 лучников | ×7 или ×20 |

  Решение принимается раз в 1000 мс, лимит 1 000 000 инструкций на вызов.
- Контракт дуэли: `policy.decide(observation) → action, reason`,
  `action` — `'attack'` или `'wait'`, `reason` — строка для журнала.

## services

- `duel_observation(unit, target, order, now_ms)` — наблюдение из
  прочитанных значений: `routing`, `shattered`, `men`, `target_valid`,
  `target_visible`, `in_melee`, `idle`, `has_order`, `since_order_ms`.
- `decide_duel(policy, observation)` — вызывает политику и проверяет ответ.

## policies

| Файл | Версия | Поведение |
|---|---|---|
| `forced_melee.lua` | `forced-melee-v2` | Атаковать видимую цель; повторить приказ, если отряд стоит 5 с |
| `delayed_melee.lua` | `forced-melee-v3-delay` | То же, но первые 10 с ждать — видно, что подхватилась новая версия |

## Как добавить свою политику

1. Создать `src/apps/ai/policies/<имя>.lua` с полем `version` и функцией
   `decide`.
2. Подключить её в точке входа (`require('apps.ai.policies.<имя>')`) или
   положить файл как `tww3_bai_policy.lua` в папку игры: дуэль перечитывает
   его перед каждым боем без пересборки pack.
3. Написать тест на `decide` в `tests/apps/ai/`.

Внешний файл политики выполняется с полным доступом к Lua. Чужие политики
загружайте через [sandbox](sandbox.md).
