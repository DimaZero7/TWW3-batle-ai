# Флаг «выделен» у отряда в памяти — начато и остановлено 07.11 (решение пользователя: пока вариант (а))

- Родной вид выбранного врага даёт CCO `Select` на `CcoBattleUnit` врага: белое кольцо на значке, ровная красная
  полоса под строем (снимки `s66.png`, `contour_compare.png`); выделение своих снимается; второй `Select` заменяет
  первого (один враг). `IsSelected` врага после него — true. `SetSelected(true)`, `SetIsSelected(true)`, `SetMouseOver`,
  `StartPulseHighlight` — без видимого эффекта; `ToggleHighlight` — красная полоса, гаснет и разгорается.
- План поиска флага (не начат): memreader `ud_topointer(script_unit)` -> +0x8 боевой отряд (цепочка из
  TWW3-malakai-thunderbarge tools/tests/crew_lib.lua, проверка +0x50 = запись land_unit); снимки 0x4400 байт отряда
  «не выделен / выделен» (`select_in_ui` своего, `Select` врага), 2-3 повтора на 2 отрядах, байты, меняющиеся ровно
  с выделением; затем `write` двум врагам.
- Мешает: Warhammer3.exe обновлён (SHA d2a8a582... против 9bd58c63... в build/mem/sites.json) и пак memreader обновлён
  04.11 (SHA 0c751ae0... против закреплённого d034ea46... в config/memreader.json) - сборка с --mem откажет, пока пак и
  места не перепроверены; --mem сейчас только для проб (MEM_TARGETS), для human нужен свой флаг.
- Статический разбор: CCO-регистрация в секции .arch запутана (трамплины push/xchg/ret), строки `IsSelectable`,
  `IsUnderPlayerControl` (0x14380db98, ссылка 0x14dab4a43), `HighlightedUnitList`, `SetMouseOver`.
