# Side 1 view — 20260927-142521

**14 / 14 checks passed.** Only what side 1 may know about the enemy, checked against the full summary as ground truth.

| Group | Check | Expected | Actual | Result |
|---|---|---|---|---|
| run | side views recorded | `>0` | `197` | ✅ |
| run | one full frame per side view | `197` | `197` | ✅ |
| leaks | enemy records carry only visibility / position / last_seen | `0` | `0` | ✅ |
| leaks | hidden enemy never has a current position | `0` | `0` | ✅ |
| leaks | range to a hidden enemy is withheld | `0` | `0` | ✅ |
| leaks | last_seen = an earlier real sighting of this side | `0` | `0` | ✅ |
| agreement | side visibility = full summary visibility (<=2% at transitions) | `<= 19` | `0` | ✅ |
| own | own units: all 68 fields in the side view | `True` | `True` | ✅ |
| mechanics | b_forest (hide_forest in forest): hidden before scouting | `>0 hidden samples` | `139 hidden / 0 visible` | ✅ |
| mechanics | b_forest (hide_forest in forest): revealed while scouted | `>0 visible samples` | `21` | ✅ |
| mechanics | b_forest: samples hidden with a remembered last position | `info` | `27` | ✅ |
| mechanics | b_stalkers (stalk on open grass): hidden before scouting | `>0 hidden samples` | `139 hidden / 0 visible` | ✅ |
| mechanics | b_stalkers (stalk on open grass): revealed while scouted | `>0 visible samples` | `29` | ✅ |
| mechanics | b_stalkers: samples hidden with a remembered last position | `info` | `27` | ✅ |

## Enemy visibility by stage (visible / not visible samples)

| Enemy | idle | march | ranged | cease_fire | melee | scout_a80 | scout_a40 | scout_a15 | scout_b80 | scout_b40 | scout_b15 | halt | done |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `b_general` | 7 / 0 | 22 / 0 | 45 / 0 | 10 / 0 | 55 / 0 | 8 / 0 | 8 / 0 | 8 / 0 | 8 / 0 | 8 / 0 | 8 / 0 | 8 / 0 | 1 / 0 |
| `b_spears` | 7 / 0 | 22 / 0 | 45 / 0 | 10 / 0 | 55 / 0 | 8 / 0 | 8 / 0 | 8 / 0 | 8 / 0 | 8 / 0 | 8 / 0 | 8 / 0 | 1 / 0 |
| `b_archers` | 7 / 0 | 22 / 0 | 45 / 0 | 10 / 0 | 55 / 0 | 8 / 0 | 8 / 0 | 8 / 0 | 8 / 0 | 8 / 0 | 8 / 0 | 8 / 0 | 1 / 0 |
| `b_forest` | 0 / 7 | 0 / 22 | 0 / 45 | 0 / 10 | 0 / 55 | 0 / 8 | 7 / 1 | 8 / 0 | 6 / 2 | 0 / 8 | 0 / 8 | 0 / 8 | 0 / 1 |
| `b_stalkers` | 0 / 7 | 0 / 22 | 0 / 45 | 0 / 10 | 0 / 55 | 7 / 1 | 8 / 0 | 8 / 0 | 6 / 2 | 0 / 8 | 0 / 8 | 0 / 8 | 0 / 1 |
