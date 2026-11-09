# reverse — разбор игры

Статический разбор Warhammer3.exe (Ghidra headless, capstone) и места в памяти для memreader Plus. Отсюда берут
правила механик для симулятора (`_why` ссылаются на `reverse/notes/…`) и файл мест для проб `--mem` (bridge).
Скрипты запускать из корня проекта; Python — `reverse/ghidra/venv/Scripts/python` (сломается — по `tools/requirements.txt`).

| где | что | git |
|---|---|---|
| `notes/*.md`, `notes/<тема>/*.md`, `*.py` | заметки (адреса — exe 15.10) и одноразовые запросы темы | да |
| `notes/<тема>/` прочее | сырьё: `.c` расшифровки, дампы `.txt/.asm/.json`, `.pkl` | нет |
| `tools/*.py` | дизассемблер и поиск (`tdis`, `fn`, `xref`, `idx`, `q`, `vt*`…), `db` — таблицы db.pack | да |
| `tools/ghidra/` | `ghd.sh` (запуск Ghidra), `dec.sh` (расшифровка по адресам), `Decomp.java` | да |
| `mem/` | места и смещения: `sites_exe1103.json` — exe 03.11, по умолчанию; `sites_exe1015.json` — exe 15.10 (есть `dir_table`); `scan.py` — поиск мест на новом exe | да |
| `ghidra/` | Ghidra 12.1.4, JDK 21, проект `gproj_exe1015`, `exe1015/` (exe, pdata/idx_battle/vts.pkl), venv | нет |

Инструменты читают `ghidra/exe1015/Warhammer3.exe` — к нему таблицы и проект; другой exe: `RE_EXE=<путь>`.
Расшифровать: `bash reverse/tools/ghidra/dec.sh out.c 0x1430a74ac […]` (проект только для чтения; параллельно — своя
копия, `GPROJ=<копия>`). Новый exe: `ghd.sh <папка> tww3 -import <exe> -noanalysis`, таблица функций — `tools/funcstart.py`.

После обновления игры: 1) `python reverse/mem/scan.py reverse/mem/sites_exe1103.json "<Steam>/Warhammer3.exe"` —
у каждого места ровно 1 совпадение; 2) новый `mem/sites_exe<ммдд>.json` (новые rva и `exe_sha256`, смещения layout
не трогать), `sites_file` в memreader.json — на него; 3) одна проба с `--mem`: `mem_attach ok`, числа сходятся со скриптом.

Открытые вопросы: адреса в заметках, таблицы функций и проект Ghidra — от exe 15.10, игра обновлена 03.11 (новый
проект — импорт без анализа, минуты; проверить); `dir_table` (векторы взгляда) на exe 03.11 не найден.
