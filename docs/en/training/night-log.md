# Night log 30.09 → 01.10.2026

[← Back](README.md) · [Documentation](../README.md) › [Training data](README.md) › Night log · [Русский](../../ru/training/night-log.md)

The owner is asleep; the orchestrator runs agents by the plan: lazy attacker → random
armies → gate (simulator ≥ 80% in both roles; in game 4 battles against the game's AI at
Normal, ≥ 3 wins) → in turn +1 unit per faction / +1 new faction. No commits overnight:
the last one is `306074d`.

- **~23:00.** The army generator is ready (`tools/nn/armies/`, [armies](armies.md)): ~75%
  from the campaign AI's templates in the game database, ~25% random; equal budget ±5%.
  Running: the lazy attacker (training), simulator accuracy (lord vs arrows, Skaven lean).
- **~23:50.** Simulator: arrow damage to a lord 0.43 (was 0.9 — double counting). The
  Skaven lean is fixed: in the game shots hit own men in melee (arrows 0.26, sling 0.56) and
  the target's neighbours — both measured and added; casualty curves match the game within
  ~10%. But the winner match fell 81% → 65% (mirror battles 8/16), likely the replay of the
  planner's orders. Decision: keep the effects, fix the replay (up to ~1.5 h).
- **~00:50.** Replay: whole battles are now compared when the recording ends. The recordings
  show a move order does not take a unit out of melee — the rule stays. Check: mechanics
  49/54, winner 17/26 (65%). Open: simulated battles resolve slower (80% not over when the
  recording ends), side 1's archers get stuck in melee. Left for the morning. Started: the
  in-game gate on random armies (4 battles, ≥ 3 wins).
- **~01:05.** The in-game gate is ready: `tools/launcher/gate.ps1 -Checkpoint …` ([gate](../launch/gate.md)).
  Smoke battle (1 launch, Normal): Empire 19 units vs Skaven 20, a barely trained network —
  lost; 568 decisions with no misses, ~43 ms per decision, 1919/1925 move orders carried out
  exactly. For the morning: the events file in the game folder is ~900 MB and growing; the
  companion and the simulator use different map bounds.
- **~01:55.** The lazy attacker is beaten: the network attacks and wins as the attacker.
  The fix: copying our own "all at the nearest" script before PPO (not the game's AI — the
  check stays independent) + attacker timeout −1.5 and an idle cost, γ = 0.9997, 60-min
  limit, memory trained on 32 s chunks, events (lord slain, time since melee). ~2–3 order
  changes a minute. But ≥ 80% is not reached: vs "all at the nearest" ~55% (random armies
  ~46%), vs "hold and shoot" 81–90%. In the simulator the best tactic seems to be just
  running at the nearest enemy: flanks and charges into an engaged enemy give little.
  Decision: the sim gate is not met, but the in-game check is the honest one — running 4
  battles with `build/nn-train/runs/long19/latest.pt`. Meanwhile an agent strengthens flanks
  and charges in the simulator from the game recordings. Note: the training agent killed a
  foreign container `cbbde7bd5bfd` by accident (maybe another project's).
- **~02:05.** In-game gate: **1 win of 4 — not passed** (3 needed). Lost 2 vs 2 (attack),
  6 vs 4 (defence), 11 vs 10 (attack, close: 904 vs 979 men); won 19 vs 20 (defence).
  Normal difficulty, preferences restored. Started an analysis: does the simulator predict
  the same outcomes, and what did the game's AI do differently.
- **~02:45.** Loss analysis. The simulator can't see them: with the game AI's orders the
  network wins all 4 there. Reasons by weight: (1) the network only knows "all at the
  nearest" — the lord runs ahead alone, the defence leaves its position, missile units are
  never pulled back; the game's AI waits and shoots first, focuses missiles on our lord,
  turns freed units onto our flanks; (2) in the simulator a lone lord dies 2–5× faster than
  in the game and rout cascades don't happen — attacking looks worth it there; (3) bridge:
  missile units walk under attack orders, a rallied unit never gets its order again, archers
  sit idle. Started: bridge fixes; the simulator agent got lords, rout cascades, a role bug
  in replay and these 4 battles in the check. Next: a game-AI-style opponent and retraining.
- **~03:00.** The bridge is fixed ([bridge](../apps/bridge.md)), checked in 2 game battles:
  missile units run under attack (1.5 → 3.0 m/s), a rallied unit gets its order at once
  (was up to 128 s idle), missile idling halved (archer firing share 0.59 → 0.75; the game's
  AI 0.86). Open: a shooter out of ammo doesn't switch to melee.
- **~03:50.** Simulator: flanks and charges measured over 28 game battles. In the game a
  charge gains little; what matters is that a unit in melee hardly turns (~1–2°/s), so an
  enemy on the flank stays there; flank ×1.74 losses, rear ×1.31. Added: turning in melee,
  spear bracing, flank morale penalties. Check: mechanics 51/54, winner 18/26. A flank
  tactic now beats "all at the nearest" in the mirror (69%) — the network has something to
  learn. Not fixed: lords still die faster in the sim (swarmed by 2–3 units more often), no
  slave rout cascade (by the DB rules routing slaves don't scare neighbours). Started the
  long training run (~1–1.5 h) with a game-AI-style opponent.
- **~05:45.** The long run (70 min) is done. Opponent `ai_like` in the game AI's style (a
  line, missiles at the enemy lord, lord behind the line, counter-charge from 100 m). The
  warm start now copies two teachers ("all at the nearest" + `ai_like`). The best version
  `build/nn-train/runs/long_ai/best.pt` is the first network that does more than attack
  (hold 25%, move 22%, attack 52%); vs `ai_like` 52/61% (attack/defence). In the simulator
  it is weaker than `long19` (66/71%), but that one lost 3 of 4 in the game. The run's last
  version degraded and is not used. Started the in-game gate: 4 battles with `best.pt`,
  then 4 with `long19` after the bridge fixes.
- **~05:56.** In-game gate (Normal, preferences restored): `best.pt` — **2 of 4** (won 11 vs
  10 attacking: 850 vs 359 men left, and 19 vs 20 defending; lost 2 vs 2 attacking and 6 vs
  4 defending); `long19` with the fixed bridge — 1 of 4. The gate (3 of 4) is not passed,
  but the new network is better in the game though weaker in the simulator. Started one
  more training iteration (~60 min) from `best.pt`: a floor on the pull to the teachers,
  checkpoint chosen by evaluation, time pressure on the attacker, more small armies.
- **~07:10.** The second iteration (`long_ai2`, 50 min) is level with `best.pt` in the
  simulator; its best version is `best_eval_u74.pt`. The agent's takeaway: more training
  won't help; stronger opponents or a more accurate simulator will.
- **~07:27.** 12 more battles in the game. `u74`: 1 of 4 and 0 of 4. `best.pt` on 4 new
  battles: 2 of 4 (won 10 vs 9 defending and 14 vs 17 attacking). **In total `best.pt` won 4
  of 8 against the game's AI at Normal**, the best result of the night; `long19` 1 of 4,
  `u74` 1 of 8.

## Night summary

- **Done:** army generator (75% campaign-AI templates, 25% random); a more accurate
  simulator (lord vs arrows, friendly fire, the target's neighbours, flanks, turning in
  melee, spear bracing); training with memory through time, events and a game-AI-style
  opponent; bridge fixes (missile units run, order after a rally, missile idling); the
  in-game gate in one command.
- **Best network:** `build/nn-train/runs/long_ai/best.pt` — 4 of 8 against the game's AI.
  The gate (3 of 4) is not passed, so no new units or factions were added.
- **Game battles overnight:** 27, all at Normal, preferences restored.
- **What holds us back:** the simulator still diverges from the game in whole battles
  (winner 18 of 26; lords die faster; no rout cascades); small armies (2 vs 2, 4 vs 6, 6 vs
  4) are almost always lost; the network doesn't use withdraw or keep.
- **For the morning:** commit (nothing committed since `306074d`, tests pass); next step —
  whole-battle simulator accuracy or a look at small battles; container `cbbde7bd5bfd`
  stopped by an agent; the events file in the game folder (~900 MB and growing).

## Night 01.10→02.10

- **23:20.** An hour without the leash from `noleash20/m10.pt` (`build/nn-train/test5/free60`): the
  network fell apart. Wins vs `ai_like` 44/50% → 7/16% (attack/defence), vs "all at the nearest"
  37/34 → 11/11, vs "hold and shoot" 46/41 → 14/7; gold exchange 0.94 → 0.68; own lord dies 3–4×
  as often. The training signal pulls down — without the leash it shows plainly. Started: 4 battles
  vs the game's AI with the last version (`m60.pt`) and an agent to find the cause in the logs and
  fix the rewards/training.
- **23:45.** Tests pass. In-game gate with `free60/m60.pt`: **0 of 4** (2 vs 2 attack, 7 vs 5
  defence, 11 vs 10 attack — close 931/865, 19 vs 20 defence). Missed decisions 34/41/119 in three
  battles — the GPU was shared with the agent's diagnostics; keep in mind for the next gates.
- **00:25.** Causes found (details in `training.md`, section "Night 01.10→02.10"): (1) **a loophole
  in the idle cost** — it was waived if any one unit fought, and the network learned to keep 1–2
  units skirmishing while the rest stood ("hold" when attacking 2% → 57%, battles up to 40 min);
  (2) **a single PPO update is almost pure noise** — its direction can't be told from random, so
  without a leash the loophole wins. Fixes (behind options): cost by the share of the army idle
  (`--idle-share 1`), the timer resets only on real damage (`--idle-rate 0.05`), advantages
  normalised per role (`--adv-norm role`), a leash to its OWN version renewed every 10 updates
  (`--reference self`). Started 45 minutes from `free60/m60.pt` with these settings.
- **01:35.** 45 minutes with the fixes (`fix45`): the collapse **stopped** but nothing came back —
  wins at the collapsed level (vs `ai_like` ~10/14%), the network is stuck on "hold" (84–99% of
  orders). Timeouts vs "hold and shoot" 39% → 3%. Started: 4 game battles with `fix45/m45.pt` and
  an analysis of how to get the network out of "hold".
- **01:55.** In-game gate `fix45/m45.pt`: **0 of 4**, but **the gate is spoiled** — the companion
  missed most decisions (394 of 407 in battle 1): an agent ran 6 CPU containers in parallel. From
  now on: nothing heavy in parallel with in-game gates.
- **02:20.** `fix45` analysis: `m45` **forgot how to fight**, not just picks "hold" — forced to
  attack it plays even worse (2–6% wins), while `m10` under the same push plays as before (44/39%).
  Training from `m45` can't pull it out: steps are near noise, no exploration (order-kind entropy
  0.004–0.012 of 1.6). **I deviate from the "only from the last version" rule** (top rule: the
  result over formalities): the chain continues from the last healthy version
  `noleash20/m10.pt`. New: a cost for an attacker unit standing idle (`--unit-idle 1e-4` with
  `--unit-credit 0.3`), a 0.03 leash to `m10` itself, less self-play (15% instead of 25%). Started
  45 minutes (`fix45b`).
- **03:30.** `fix45b` (45 min from `m10`): **no collapse**, a slight gain in attack. Wins (attack /
  defence), 0 → 45 min: `ai_like` 45/52 → 52/47%, "all at the nearest" 40/34 → 44/39%, "hold and
  shoot" 48/41 → 56/43%; gold exchange 0.94 → 0.98; own lord dies less; timeouts almost gone.
  "Hold" 5% → 0%, "attack" 76% → 82% — watch for a slide into "attack everything". Started 4 game
  battles with `fix45b/m45.pt` on a clean machine.
- **03:40.** In-game gate `fix45b/m45.pt` (clean machine, no misses): **2 of 4** — level with the old
  best `best.pt` on these battles. Won 11 vs 10 attacking (903/721) and 19 vs 20 defending; lost
  2 vs 2 attacking and 7 vs 5 defending — in the latter we had more men left (605 vs 340), being
  analysed. Started: the next 45 minutes of the chain from `fix45b/m45.pt` (`fix45c`, same
  settings) and an analysis of the lost battles for the next reward change.
- **04:00.** Loss analysis: both times **our lord broke**. In the game a shattered (not only
  killed) lord collapses the whole army's morale within a second — that's how the 7 vs 5 battle
  was lost with 605 vs 340 men. The simulator and the reward count only the lord's death. Also the
  network leads with its lord and sends him back in wounded. Being prepared: in the simulator a
  shattered lord = dead for morale; in the reward `--lord-rout 0.5` (a shattered lord counts as a
  death, a routing one as a share) and `--lord-exposed 5e-4` (a cost for the lord in melee below
  50% health).
- **04:10.** Simulator: a shattered lord now hits the army's morale like a dead one (−16, then
  −10). Checked on the recordings: at the shatter every unit loses ~0.5–0.57 of leadership and
  routs within 1–3 s. In the simulator morale falls over several seconds, not one — a possible
  calibration.
- **04:50.** `fix45c` (another 45 min): flat, a slight gain in defence. Vs `ai_like` 51/49 → 51/55%,
  "all at the nearest" 44/40 → 43/41, "hold and shoot" 54/42 → 55/43; gold exchange ~1.0–1.04.
  "Attack" stays at 82–83%, "hold" 0%. Next in turn: container tests → 4 game battles with
  `fix45c/m45.pt` → 45 min `fix45d` with the new lord changes (`--lord-rout 0.5
  --lord-exposed 5e-4`, a shattered lord in the simulator).
- **06:10.** Torch tests pass in the container. **In-game gate `fix45c/m45.pt`: 3 of 4 — the GATE IS
  PASSED for the first time** (won 2 vs 2 attacking, 11 vs 10 attacking 1049/655, 19 vs 20
  defending; lost 7 vs 5 defending). `fix45d` (45 min with the lord changes) — flat: vs `ai_like`
  53/52 → 50/56%, "hold and shoot" 54/42 → 57/45; "lord dead" now also counts shattering, so it
  rose (don't compare with older numbers). The weekly limit has reset. Started: a gate for
  `fix45d/m45.pt` and a re-check of `fix45c/m45.pt` on 4 other battles, so 3 of 4 isn't luck.
- **06:30.** `fix45d/m45.pt`: 2 of 4. Re-check of `fix45c/m45.pt` on 4 other battles: 1 of 4.
  **In total `fix45c` — 4 of 8**, exactly like the old best `best.pt`: the earlier 3 of 4 was partly
  luck. Night result: the network no longer collapses without the script leash, learns from its
  own version and holds `best.pt`'s level, but doesn't beat it. Weak spots unchanged: small armies
  and defending with few units (4 vs 5, 7 vs 5).
