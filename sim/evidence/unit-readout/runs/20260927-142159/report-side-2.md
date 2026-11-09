# Side 2 view — 20260927-142159

**12 / 14 checks passed.** Only what side 2 may know about the enemy, checked against the full summary as ground truth.

| Group | Check | Expected | Actual | Result |
|---|---|---|---|---|
| run | side views recorded | `>0` | `206` | ✅ |
| run | one full frame per side view | `206` | `206` | ✅ |
| leaks | enemy records carry only visibility / position / last_seen | `0` | `0` | ✅ |
| leaks | hidden enemy never has a current position | `0` | `0` | ✅ |
| leaks | range to a hidden enemy is withheld | `0` | `0` | ✅ |
| leaks | last_seen = an earlier real sighting of this side | `0` | `0` | ✅ |
| agreement | side visibility = full summary visibility (<=2% at transitions) | `<= 20` | `0` | ✅ |
| own | own units: all 68 fields in the side view | `True` | `True` | ✅ |
| mechanics | a_forest (hide_forest in forest): hidden before scouting | `>0 hidden samples` | `139 hidden / 0 visible` | ✅ |
| mechanics | a_forest (hide_forest in forest): revealed while scouted | `>0 visible samples` | `0` | ❌ |
| mechanics | a_forest: samples hidden with a remembered last position | `info` | `0` | ✅ |
| mechanics | a_stalkers (stalk on open grass): hidden before scouting | `>0 hidden samples` | `139 hidden / 0 visible` | ✅ |
| mechanics | a_stalkers (stalk on open grass): revealed while scouted | `>0 visible samples` | `0` | ❌ |
| mechanics | a_stalkers: samples hidden with a remembered last position | `info` | `0` | ✅ |

## Enemy visibility by stage (visible / not visible samples)

| Enemy | idle | march | ranged | cease_fire | melee | scout | scout2 | halt | done |
|---|---|---|---|---|---|---|---|---|---|
| `a_general` | 7 / 0 | 22 / 0 | 45 / 0 | 10 / 0 | 55 / 0 | 20 / 0 | 35 / 0 | 10 / 0 | 1 / 0 |
| `a_spears` | 7 / 0 | 22 / 0 | 45 / 0 | 10 / 0 | 55 / 0 | 20 / 0 | 35 / 0 | 10 / 0 | 1 / 0 |
| `a_archers` | 7 / 0 | 22 / 0 | 45 / 0 | 10 / 0 | 55 / 0 | 20 / 0 | 35 / 0 | 10 / 0 | 1 / 0 |
| `a_forest` | 0 / 7 | 0 / 22 | 0 / 45 | 0 / 10 | 0 / 55 | 0 / 20 | 0 / 35 | 0 / 10 | 0 / 1 |
| `a_stalkers` | 0 / 7 | 0 / 22 | 0 / 45 | 0 / 10 | 0 / 55 | 0 / 20 | 0 / 35 | 0 / 10 | 0 / 1 |
