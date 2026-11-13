# game — бои в игре, пробы, разметка
Запускает настоящую игру через лаунчер моста (`bridge/launcher`) и разбирает записи. Новый бой — `build/<цель>/runs/<время>/`.
Каждый запуск: «Нормальная», настройки пользователя возвращаются; отказ при `state/HUMAN_PLAYING` и чужом
`state/probe_game.lock` (лаунчер и `ops/guard.*`); раннеры держат замок на всю серию. `DRY=1` / `-DryRun` — только сборка.
## Бой цепочки (М1–М3), ИИ против ИИ, бой глазами
    bash game/battle/chain.sh [-n 2] m1 m2 m3      # сборка nn-arena + watch.ps1 -Chain, x3, BAI_GRAPHICS=low
    python -m game.battle.table <прогон>...        # победитель, время, контакт, золото, «в зоне», снайперы
    python -m game.battle.analyze <прогон> [--json f]   # и compare, units, churn, snip
    bash game/battle/aivai.sh [f1 .. k4]           # ИИ игры против планировщика CA, армии этапа 0; меры и двойник — game.battle.aivai
    python -m game.vision <прогон>...              # лента 6 кадров PNG + текст (бои и пробы), ~2,7 тыс. токенов (vision/README.md)
М1, М2 — прогоны человека (`data/human/runs`, железнобокие → длинноствольные), М3 — seed 1000702112 v3; карта ksl_plains_01.
## Проба механики
    bash game/probe/run.sh <план> [<план>...]       # по бою на план (BATTLE=N — его N-й бой), --mem, BAI_GRAPHICS=low
    python -m game.probe.charge_probe report <прогоны> [--sim]   # игра против двойника симулятора
Планы — `PLANS` в `charge_probe.py` (этап 0 — `unit_probe.py`, отчёт `python -m game.probe.unit_probe`: `reissue`, `counter`,
`flank2v1` — второй отряд к связанному врагу: `--flank2v1 game [--sim]`, двойник до игры `--flank2v1 layout`); ещё
`missile_probe`, `morale_probe`, `move_probe`, `mem_probe` (строки памяти). Выжимки проб этапа 0 — `probe/evidence/`.
## Разметка вживую (играет пользователь)
    powershell -File game/label/live.ps1 [-Match 1]     # Ultra на RTX, мод LOD 3812267837 только на этот бой
    python -m game.label.digest <запись> --export       # лента по 15 с, кадры, gif
    python -m game.label.digest_set <записи>            # набор подражания data/label/human_set (torch)
`keys.ps1` только читает Ctrl, Shift и курсор; кликов и нажатий в систему нет. Запись — `data/label/<дата>_live`.
## Карта местности (один раз на карту)
    python -m game.maps.map_grid capture --name <карта> [--dry]   # снять в игре, свести -> sim/maps/<карта>/ (в git)
## Данные (вне git, `data/`, 220 МБ)
`nn-arena/runs` — база М1–М3 (07.11 22:08, 22:11, 22:15) и after6 (08.11 09:27–09:40); `human/runs` — 5 боёв
человека; `label/` — 3 живых боя 07.11, `human_set`, `maps`, `matchups`, `human_rules.md`; `label_icons/` — портреты.
Тест (без игры): `python -m pytest -q game/tests` — паки боя цепочки и пробы из этого дерева, замок, разбор боя.
Открытый вопрос пользователю: пресет `low` не закрепляет видеокарту (берётся из настроек игры); RTX закреплена только
в `ultrafull`. Системные настройки не трогаем — закрепить RTX в игре для `low` или оставить как есть?
