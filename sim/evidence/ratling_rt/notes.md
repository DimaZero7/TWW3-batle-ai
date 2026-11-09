# Ratling Gun: spin-up after a target change (probe ratlingrt, 13.10)

Plan `ratlingrt` (tools/nn/charge_probe.py): gun between two held fearless Empire spearmen, 70 m front to front,
targets 10 m apart (centres ~40 m: no turn needed). One switch of the ranged attack at T, lane ends before the switch back.
Shots = ammo drop each 0.5 s. Runs: build/charge-probe/runs/20261013-005259, -005417 (2 battles x 9 lanes, Normal, speed 10,
lowigpu). Script: build/ratling_rt/analyze.py (results.json game, results_sim.json twin, 8 copies, CPU).

Normal cycle (control A + pre-switch parts): first bullet 4.0-7.0 s (median 5.5), burst 6-7.5 s of unit fire, next burst 12.0 s
after the first (A lanes 12.0 / 12.0).

| condition | T (s) | game: first bullet of new burst after T | game: from B1 | sim: after T | sim: from B1 |
|---|---|---|---|---|---|
| M1 mid first burst | 6.0 | 4.0, 4.5, 4.5, 4.5 (burst cut 0.5 s after T) | - | 11.0-12.0 (one lane 4.5: its B1 had not started) | 12.0 |
| M2 mid second burst | 18.5 | 4.0, 4.5, 4.5, 4.5; the new burst FULL 465-563 bullets (576 max); next one 12 s after it | 17.0-18.5 | 10.5-11.5 | 24.0 |
| E right after burst 1 | 12.5 | 4.5, 4.5(*), 5.0, 5.0 | 11.0-12.5 | 4.5 | 12.0 |
| L late reload (~2 s before B2) | 14.5 | 4.0, 4.0, 4.5, 5.0 | 13.0-14.0 (normal 12.0) | 4.5 | 12.5-14.0 |
(*) two lanes: 2 stray bullets 0.5 s after T, then the burst at +5.0.
(The twin's switch lands 0.5 s earlier than the game's - its clock starts at the first sample; immaterial here.)

Verdict:
- During the reload (E, L): the game does NOT hide the spin-up in the reload: the new burst comes ~4.5 s after the switch
  (L: 1-2 s later than the normal cycle). The simulator does the same (aim clock reset at the switch, runs in parallel with the
  reload): matches, retarget_projectile_s 4.5 confirmed (game median 4.5, 4.0-5.0, 8 lanes).
- Mid-burst (M1, M2): the game cuts the burst within 0.5-1 s, and 4.0-4.5 s after the switch fires a NEW FULL burst at the
  new target (no reload wait; the cut burst's unfired bullets are not lost to the cycle); the next burst comes 12 s after the
  new one. The simulator cuts the burst too, but waits for the reload from the cut burst's start (12 s) -> 10.5-12 s silent
  (6-7 s too long).

Proposed rule (not applied; tools/nn/sim/missile.py burst_share + battle.py retarget block), e.g. missile.burst_cut_reload 1:
a burst weapon (spv > 1, bst_rate > 0) whose aim is reset by a new target or a new order while its burst is going
(bst_s > 0) ends the burst (bst_s = 0, bst = 0) and its men count as loaded (unready = 0): the next burst starts when the
spin-up is done (aim from 0 to aim_s 4.5) with all 18 shots a man, and the reload cycle counts from that burst's start.
Numbers: ratlingrt probe above (no database or community number found: wikis only say the gun spins before firing).
Not measured: a re-issued order on the SAME target mid-burst; a target change at will (no order) mid-burst.
