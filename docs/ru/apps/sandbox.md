# sandbox

[← Назад](README.md) · [Документация](../README.md) · [Приложения](../architecture/apps.md) · [English](../../en/apps/sandbox.md)

Загрузка политик по API v1 в урезанное окружение. Нужна, если запускать
чужой код политики. Код: `src/apps/sandbox/services.lua`.

## API политики v1

```lua
return {
    api_version = 1,
    create = function(context) return state end,
    step = function(state, observation) return commands end,   -- см. orders.contract
    -- по желанию: deployment_version = 2, deployment_contract = 'deployment-placement-v2',
    --             deploy = function(state, context) return plan end
}
```

## Функции

| Функция | Что делает |
|---|---|
| `load(source, context, profile)` | Загружает исходный текст (не байткод), возвращает `{deploy, step}` |
| `bounded(fn, ...)` | Вызов с лимитом 1 000 000 инструкций |
| `copy(data)` | Глубокая копия обычных данных: без метатаблиц и циклов, глубина < 24, ≤ 4096 ключей в таблице |
| `validate(commands, own, enemies, profile, contract?)` | `orders.contract.validate` с лимитом команд профиля |

Политика видит только `pairs`, `ipairs`, `next`, `type`, `tonumber`,
`tostring`, `assert`, `error`, `select` и часть `math`, `string`, `table`.
Данные на входе и выходе копируются.

## Известные ограничения

Это **не защищённая песочница**:

- методы строк доступны через общую метатаблицу строк, например
  `('x'):rep(n)`; вызов C-функции считается за одну инструкцию, поэтому так
  можно занять много памяти;
- `copy` не ограничивает повторное использование одной подтаблицы (граф без
  циклов) и выполняется вне лимита инструкций.

Для своих политик это неважно. Перед запуском чужого кода эти места нужно закрыть.
