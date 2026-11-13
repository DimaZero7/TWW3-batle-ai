# A standing unit's strike rate: hold against attack (static analysis, 13.11.2026)

Only the file on disk (exe 15.10), open code `.udata`; no game run, no debugger. Raw: `notes/hold_rate/` (`asm.txt`
capstone of the functions below, `kv.txt` and `weapons_emp.txt` from db.pack, `gemini_raw_C2_1.c` the external
model's Ghidra output of `143074eb4`, `143074dcc`, `143056730`, `14308d10c`, `14308b608`, `14308b938`, `14306bc78`).
Builds on `melee_module.md` 2.4–2.6, `melee_cycle.md`, `melee_stacking.md` 2.1–2.4, `charge_module.md` 4.
Marks: **В** read in full, **С** code read, meaning by context, **Н** guess.
Game numbers: `sim/evidence/counter/spec.md`, `game/probe/evidence/counter_game.json`; `contact.hold_rate_why`.

## 1. Function map

| what | function |
|---|---|
| ready to swing: now − mark (+0xa30) > interval | `14305ec7c` |
| interval = weapon +0x54 (`melee_attack_interval`), else kv +0x29e8 (4.0) | `143073320` |
| answering strike of a standing man on an enemy arriving by order (once per approach, flag 0x40) | `143074eb4` ← `143056be8` ← approach `14308f0e0` |
| strike back at the one who swung at me | `143074dcc` (from `143080ca0`) |
| strike the enemy whose body touched me (both sides) | `143056730` ← collision event `14306bc78` |
| attack-order struct init (kind @0 = 0, enemy unit @8, enemy man @0x18, "answer" byte @0x22) | `143053864` |
| free-fight action ctor (copies kind → +0x90, @0x22 → +0xb2) / tick | `143053684` (vtable `143ad24b8`) / `1430877ec` → `14308d10c` |
| kind 2 = "formed" (strike only within combat 2.5 / tether 2.0 × scale of the slot), else free | `14308d5d8`: `14308b608` / `14308b938` |
| new target from an enemy unit, range by the enemy record's state | `14302e704` → `14302c6c4` → `143016670`, `14302e7d8` |
| engagement threshold k: order → kv +0x3750, none → +0x37b8 (missile with ammo +0x36e8) | `14304bbd4` (`14304bc5d..14304bc93`) |
| unit → man melee order with U[0, 0.5) s delay | `143280d08` → vfunc 0x358 `14307072c` → `142f642fc` |
| brace bit read in hit reactions (vfunc 0x548 / 0x550) | `14306f461` in `14306f37c`, `14306f67e` in `14306f594`; window `14305ae80` |

## 2. Rules

### 2.1 The swing timer is the same for both — В
`ready = now − mark > interval` (`14305ecbd..14305ecc8`, strict), mark set at the start of each attack animation
(`melee_module` 2.6). Interval from the weapon: Empire spear 4.4 s, sword 4.3 s, halberd 5.6 s (db `melee_weapons`).
No branch on the unit's order, on hold, or on the brace bit in the timer or in the strike-start paths read
(`143074eb4`, `143074dcc`, `143056730`, `14308b938`, `14308f0e0`). **No constant near 0.5 for "hold" exists in these
paths**; no hold-rate key found in `_kv_rules` by name (idle / ordered / reaction / melee_attack / bracing; `kv.txt`).

### 2.2 Who gets a target — В (code), С (consequence)
- **Attack order**: the unit gives targets to its first rank (`14327c350`, `melee_stacking` 2.1) and keeps feeding
  the front (`1432713ac`, `143288918`, `1432836a0`); every man's order fires after U[0, 0.5) s (`142f642fc`); a man
  whose target dies takes a new one from the target unit with no range limit (`14302c6c4`: target unit → FLT_MAX).
- **Hold / never ordered**: the unit issues no melee orders (the four callers of `143280d08` — `14327c350`,
  `1432713ac`, `143288918`, `1432836a0` — are the attack behaviour's slots, `melee_stacking` 1; С). A man starts a
  swing only when
  1. an enemy body touches him: `14306bc78` → `143056730` (if busy with another target, switches only when the
     toucher is on the line to it — `14315e770`);
  2. an enemy comes at him under an approach order: `143074eb4` — man standing (speed ≤ 0.1 m/s), action state
     +0x324 == 0, not in breakoff (`14305b300`), enemy within ±`bracing_attack_angle` (80°) of his facing; once per
     approach (`143056be8` sets flag 0x40); struct @0x22 = 1;
  3. an enemy swings at him: `143074dcc` (no angle, if not busy: +0x330 vfuncs 0x60/0x70).
  Each opens a free-fight action of kind 0 (`143053864` leaves @0 = 0) that carries the enemy's unit (@8). It lives
  while its target is valid and in reach; when the target dies it re-targets from that unit only on ticks where
  (tick ^ man id) & 7 == 0, within the range of the unit's enemy record state (`14302c6c4` → `143016670`):
  1 → `max_range_for_attack_mode_free_attack` 100 m (→ 2, `…_melee_reaction_attack` 40 m, when the caller's flag
  param_5 is set), 3 → `…_ranged_attack` 15 m, 4 → `…_local_attack` 3 m (0 and 5 = breakoff: skipped). It ends if the target runs off faster than its walk speed or 4 m/s (`14307f80c`, first block of
  `14308b938`), or (missile units, @0x22 answer actions) when `143075e5c` says so. Then the man is idle until the next
  touch / approach / swing.
- The record state (`1430213a0`, `melee_turn` 2.3) uses N = max(1, men × k): k = **0.22 with an order, 0.14 without**
  (db values; `melee_turn.md` 2.1 says 1.0 — wrong for the current data). A held unit reaches "engaged" (1, 100 m)
  with fewer contacts than an attacking one.
- С: so the held unit's men swing only while something keeps re-triggering them; the attacker's are re-fed by the
  unit. The game's 0.49–0.52 swing ratio (probe width) is this difference in the share of contact men who have a
  live target, not a coded factor. **Н**: its size (0.5) — not derivable from code (depends on how often touches and
  re-targets happen).

### 2.3 The pause after the first strike — В (timer), С (consequence)
The answering strike (2.2, case 2) makes every eligible man of the standing unit swing at the arrival together, so
their marks are all set within ~0.1 s. None of them can start another swing before mark + interval (spear 4.4 s,
sword 4.3 s). In 1–4.4 s after contact only men who did not swing at the arrival can strike. A model that adds the
opening wave and then runs the steady rate for everyone from the next second counts those men twice in that window.

### 2.4 Bracing — В (where read), С (effect)
The brace bit (+0x68 & 8) is read in: `1430472a8` (count), hit-reaction handlers vfunc 0x548 `14306f37c`
(`14306f461`) and 0x550 `14306f594` (`14306f67e`), and `143146868`, `143243d7c`, `143248984` (not read). In 0x548 /
0x550 a braced man inside his attack window (`14305ae80`: animation frame within −8..+5 ticks of a mark) ignores the
reaction — his swing is not broken. The melee hit reaction `14306ee40` (vfunc 0x558, `melee_cycle` 2.4) has no brace
check. Which hits go through 0x548 / 0x550 — not found. No brace check in target choice or in the swing timer.

## 3. Rule for the simulator
```
per man in contact with a live target: one swing per interval from the start of his last swing (both orders)
attack:   every contact man has a target (the unit re-feeds)                         → rate = men_contact / interval
hold:     a man gets a target only by touch / an arrival at him (±80°, standing) / being swung at;
          keeps it until it dies (re-target within 100 / 15 / 3 m by engagement) or runs → share of men live < 1
          (hold_rate: measured, no code constant)
opening wave: men who struck at the arrival are out of the steady rate until t0 + interval (spear 4.4, sword 4.3 s)
```
| item | rule | game |
|---|---|---|
| attack, swings/s per contact man | 1 / 4.4 = 0.23 (spear) | 4.1 / 4.8 / 5.4 swings/s at 20 / 30 / 38 m front ⇒ 18 / 21 / 24 men a 4.4 s cycle (consistent if that many touch; not checked) |
| hold / attack swing ratio | < 1, size not in code | 0.49 / 0.50 / 0.52 |
| hold, after the opening wave, 1–4.4 s | opening-wave men silent | swordsmen hold 27 HP/s vs the twin's 43 (0.63 ⇒ ~⅓ of the twin's steady strikers were in the opening wave — not checked) |
| first strike of a standing unit | every standing man with the enemy within ±80° | 106–194 HP in the first second (open_melee R1) |

Against `sim/config/sim.json`: `contact.hold_rate` 0.5 — a fit of an emergent effect, keep it (no code constant to
replace it). `stand_first_strike` / `first_strike_full` — right (code: the answer strike is a full swing of every
standing man in reach and angle, not cut by anything). Missing: the opening-wave men's cooldown
(`melee.cycle`/steady rate should exclude them until t0 + interval) — the likely cause of the 1–10 s overshoot
(`sim/evidence/counter/spec.md`, OPEN). `hold_reflect_full` 0 — the code gives no extra rate to braced men (brace
only shields their own swings from some reactions), so 0 is consistent.

## 4. Not found
- Where kind 2 ("formed" melee, `14308b608`) is set — no caller of `143053864` or `143280d08` writes @0 = 2.
- Which hit events use reaction vfuncs 0x548 / 0x550 (callers `143253864`, `1432539ac`, `14325b75c`, `14325bda4`).
- Meaning of `14301d214` (unit check in the free-fight pursuit branch) and `14306de20`.
- A count of how often a held man is re-triggered — no code number; needs the probe with men's places at contact.
