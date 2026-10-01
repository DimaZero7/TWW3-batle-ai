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
