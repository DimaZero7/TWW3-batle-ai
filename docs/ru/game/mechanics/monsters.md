# Монстры и большие отряды

[← Назад](README.md) · [Документация](../../README.md) › [Знания об игре](../README.md) › [Механика боя](README.md) › Монстры · [English](../../../en/game/mechanics/monsters.md)

Монстры — одиночные модели и отряды из нескольких моделей: масса, сбивание с ног, удар по площади, страх и ужас. Пешие лорды —
в [лордах и героях](single-entities.md). Условные обозначения: [оглавление](README.md). У нас: не моделируется.

## Масса и сбивание с ног

- **Масса решает, кто сдвинется.** Идущий в натиск теряет инерцию с каждой моделью, в которую врезался, поэтому тяжёлые отряды
  проходят глубже. Шансы на каждый удар: отбросить 0.1 × отношение масс, сбить с ног 0.05 × отношение (×2
  с поправкой на множитель натиска при натиске); ниже отношений 0.35 / 0.5 — нисколько; удар прерывает действие при
  изменении скорости на 4 (отброс), 7 (сбит), 10.5 (летящий) м/с. · [twwstats][tws] (описания WH2,
  значения WH3) · высокая.
- **Массы:** чудовищная пехота ~1700–1900; Doomwheel 2000; Tamurkhan 2500 (5.1); драконы
  4000 → 5000, Star Dragon 5900 (7.0); Dread Saurian ~12000. · [5.1.0][p510], [5.3.0][p530],
  [7.0][p70] · высокая.
- **Шанс не упасть** у каждой модели (`battle_entities`): 20–80 %; 6.2.2 поднял Арахнароку 15 → 85 %
  и поставил ≥ 40 % всем пешим персонажам. · [хотфикс 6.2.2][h622] · высокая.
- **Монстры застревают** в пехоте в упоре или плотной пехоте, когда не могут набрать скорость, нужную,
  чтобы сбить её с ног; один отбившийся солдат может их удержать («склейка отрядов»). · [Steam][stuck] · средняя.
- **Класс размера определяет отброс**: Боевые сани перевели из очень больших в большие (5.3), чтобы выровнять их
  отброс. Огров снова сбивают с ног, как другую чудовищную пехоту (5.1). · высокая.

## Удар по площади

- Урон одной атаки делится между задетыми моделями (каждая проверяет попадание), не более
  `max_splash_targets`, только по целям не больше размера цели удара по площади (малые, средние,
  большие, очень большие). До 6.0: монстры 10–12; 5.1/5.3 урезали многим (Great Unclean One 12 → 5,
  Rogue Idol 12 → 10 → 8, Dread Saurian 12 → 8, Mutant Rat Ogre 8 → 4). 6.0: целей = сила оружия
  / 100 для одиночных моделей. · [тема об ударе по площади][splash], [5.1.0][p510], [5.3.0][p530] · высокая/средняя.

## Кто может ударить монстра

- Число нигде не опубликовано; бить могут только модели, касающиеся его большого хитбокса; окружение добавляет
  атакующих и множители защиты во фланг / в тыл (×0.6 / ×0.3) на каждого атакующего. · [Steam][surr] · низкая.
- Стрельба: большие хитбоксы ловят больше выстрелов; бонуса к шансу попасть против больших нет. Многие большие
  одиночные модели имеют +15 % сопротивления стрельбе. · средняя/низкая.
- Отряды против больших добавляют бонус против больших к атаке и урону. · [кавалерия](cavalry.md) · средняя.

## Страх и ужас

- **Страх:** −8 лидерства врагам в пределах 20 м (`fear_effect_range`; в вики сказано 30 м —
  верить базе); не складывается; сами вызывающие страх к нему невосприимчивы. · [twwstats, мораль][tws-m],
  [fandom Causes Fear][fw-fear] · высокая (значения).
- **Ужас:** при ударе в рукопашной враг в пределах 5 м с моралью ≤ 13 очков бежит 14 с;
  после этого 85 с невосприимчив к ужасу от того же врага; 4 бегства от ужаса ломают отряд. Каждый вызывающий ужас
  вызывает и страх (WH3). Несокрушимые, нежить, демоны невосприимчивы. · [twwstats, мораль][tws-m],
  [fandom Causes Terror][fw-ter], [Steam][terror] · высокая/средняя.
- **Здоровье и размер отряда.** Здоровье и урон монстров-одиночных моделей зависят от размера отрядов (25–100 %);
  состояние «ранен» при малом здоровье. · [GameWatcher][gw] · высокая.

[tws]: https://twwstats.com/kv/rules
[tws-m]: https://twwstats.com/kv/morale
[p510]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/23-total-war-warhammer-iii-patch-5-1-0
[p530]: https://community.creative-assembly.com/total-war/total-war-warhammer/forums/7-total-war-warhammer/threads/7586-total-war-warhammer-iii-patch-5-3-0-battle-balance-details
[p70]: https://community.creative-assembly.com/total-war/total-war-warhammer/blogs/90-total-war-warhammer-iii-update-7-0-0
[h622]: https://community.creative-assembly.com/total-war/total-war-warhammer/forums/7-patch-notes-amp-announcements/threads/10367-total-war-warhammer-iii-hotfix-6-2-2
[stuck]: https://steamcommunity.com/app/1142710/discussions/0/664957733316912540/
[splash]: https://community.creative-assembly.com/total-war/total-war-warhammer/forums/10-battles/threads/6818-dividing-damage-among-models-during-splash-attacks-is-a-bad-design-decision
[surr]: https://steamcommunity.com/app/594570/discussions/0/6980058308319576309/
[fw-fear]: https://totalwarwarhammer.fandom.com/wiki/Causes_Fear
[fw-ter]: https://totalwarwarhammer.fandom.com/wiki/Causes_Terror
[terror]: https://steamcommunity.com/app/1142710/discussions/0/688619343242660842/
[gw]: https://www.gamewatcher.com/news/total-war-warhammer-3-lords-monsters-debuff-small-unit-size-scaling
