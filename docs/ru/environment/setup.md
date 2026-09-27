# Установка и настройка

[Документация](../README.md) · [English](../../en/environment/setup.md)

## Что нужно

- Windows с установленной Total War: WARHAMMER III (Steam, app id 1142710);
- Python 3.10+;
- PowerShell 5.1 (есть в Windows) для запуска игры;
- Git.

Сама игра, её файлы и сторонние моды в репозиторий не входят.

**Обязательно подпишитесь в Steam Workshop на мод
[True Sight](https://steamcommunity.com/sharedfiles/filedetails/?id=3628832922):
без него бои не запускаются ([подробнее](../launch/run.md#обязательный-мод-true-sight)).**

## Python-окружение

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt`:

| Пакет | Зачем |
|---|---|
| `lupa` | Lua 5.1 внутри Python: тесты и проверка синтаксиса бандла при сборке |
| `pytest` | Запуск тестов |
| `numpy`, `matplotlib` | `tools/analysis`: карты высот, склоны, проходы |

Без `lupa` сборка работает, но в `manifest.json` будет `"syntax_checked": false`.

## Настройки машины: `config/local.json`

Как в photo-fixing (`default_config.toml` → `config.toml`): в Git лежит
`config/default.json`, а личные настройки — в `config/local.json`, который Git
игнорирует. Значения из `local.json` заменяют значения по умолчанию.

```bash
cp config/local.example.json config/local.json
```

| Ключ | Значение |
|---|---|
| `game_dir` | Папка игры с `Warhammer3.exe` |
| `workshop_dir` | Папка Workshop-модов `...\workshop\content\1142710` |

## Кириллица в пути

Проект лежит в папке с русскими буквами (`Файлы\Программы`). Lua 5.1 на
Windows не открывает такие пути сам, поэтому тестовый загрузчик читает
исходники через Python (`tests/lua_runtime.py`). В игре это не важно:
скрипты читаются из pack, а файлы телеметрии пишутся в папку игры по
относительному пути.

## Проверка

```bash
.venv/Scripts/python -m pytest
```

Все тесты должны пройти. Дальше: [сборка pack](../launch/build.md).
