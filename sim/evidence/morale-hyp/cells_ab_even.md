## build/morale-hyp/ab_even

Windows of 10 s; 46 battles, 42879 unit-windows. HP lost sim/game 0.99 (0.96-1.02); rout starts sim/game 0.88 (0.81-0.95), game 939, sim 822.8.

Melee units, the loss by second of the window (sim/game of the cumulative loss):

| s | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| sim/game | 0.89 | 0.90 | 0.90 | 0.91 | 0.92 | 0.94 | 0.94 | 0.96 | 0.97 | 0.97 |

### HP lost by situation

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| all on the field | 42879 | 46 | 662.9 | 655.2 | 0.99 | 0.96-1.02 | 1.05 | 1.02-1.08 | 1.00 |
| standing, not in melee, no contact in window | 18470 | 46 | 43.5 | 56.2 | 1.29 | 1.20-1.39 | 1.37 | 1.26-1.51 | -1.64 |
| standing, shot at (an enemy shooting it) | 1870 | 46 | 108.2 | 104.1 | 0.96 | 0.92-1.01 | 1.09 | 1.02-1.18 | 0.52 |
| contact begins in window, ran in | 2966 | 46 | 83.2 | 77.3 | 0.93 | 0.88-0.98 | 0.90 | 0.85-0.96 | 0.77 |
| contact begins in window, walked in | 122 | 40 | 4.3 | 3.6 | 0.82 | 0.70-1.00 | 0.75 | 0.60-0.97 | 0.10 |
| melee, first 10 s of the fight | 1851 | 46 | 76.7 | 70.4 | 0.92 | 0.88-0.96 | 0.97 | 0.92-1.03 | 0.81 |
| melee, 10-30 s | 2674 | 46 | 104.8 | 92.7 | 0.88 | 0.86-0.92 | 0.92 | 0.87-0.97 | 1.57 |
| melee, 30 s+ | 8785 | 46 | 244.7 | 250.6 | 1.02 | 0.98-1.07 | 1.08 | 1.04-1.14 | -0.76 |
| melee, sim touches 1 enemy | 4820 | 46 | 162.7 | 146.8 | 0.90 | 0.87-0.94 | 0.96 | 0.91-1.01 | 2.05 |
| melee, sim touches 2+ enemies | 6824 | 46 | 219.8 | 233.2 | 1.06 | 1.02-1.10 | 1.10 | 1.05-1.15 | -1.73 |
| melee, sim touches nobody (game flag on) | 1666 | 46 | 43.6 | 33.6 | 0.77 | 0.70-0.84 | 0.89 | 0.81-0.97 | 1.29 |
| melee, touched in front | 4185 | 46 | 127.3 | 121.2 | 0.95 | 0.91-1.00 | 1.02 | 0.97-1.07 | 0.78 |
| melee, touched in flank | 5533 | 46 | 182.2 | 186.1 | 1.02 | 0.98-1.07 | 1.07 | 1.02-1.13 | -0.51 |
| melee, touched in rear | 1926 | 46 | 73.1 | 72.7 | 0.99 | 0.93-1.06 | 1.00 | 0.92-1.09 | 0.06 |
| melee, health < 30 % | 1705 | 46 | 39.3 | 45.8 | 1.16 | 1.09-1.26 | 1.27 | 1.16-1.41 | -0.83 |
| melee, health >= 30 % | 11605 | 46 | 386.8 | 367.9 | 0.95 | 0.92-0.98 | 1.00 | 0.97-1.04 | 2.45 |
| routing, in melee (pursued) | 1220 | 46 | 22.4 | 20.8 | 0.93 | 0.79-1.09 | 0.90 | 0.75-1.09 | 0.21 |
| routing, free | 5524 | 46 | 9.6 | 11.5 | 1.20 | 1.02-1.42 | 1.32 | 1.05-1.63 | -0.25 |
| standing, health < 30 % | 4844 | 46 | 54.0 | 61.2 | 1.13 | 1.06-1.22 | 1.25 | 1.15-1.37 | -0.93 |

### HP lost by unit kind

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord | 3927 | 46 | 54.6 | 62.1 | 1.14 | 1.03-1.25 | 1.16 | 1.04-1.29 | -0.98 |
| missile | 13719 | 46 | 188.0 | 174.5 | 0.93 | 0.88-0.98 | 1.00 | 0.94-1.07 | 1.75 |
| infantry melee | 25233 | 46 | 420.4 | 418.6 | 1.00 | 0.96-1.03 | 1.06 | 1.02-1.10 | 0.24 |
| network's side | 21084 | 46 | 373.7 | 369.2 | 0.99 | 0.95-1.03 | 1.05 | 1.01-1.10 | 0.58 |
| the other side | 21795 | 46 | 289.2 | 286.0 | 0.99 | 0.94-1.04 | 1.05 | 0.99-1.11 | 0.42 |

### HP lost by unit kind and state

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord, melee | 1398 | 46 | 38.5 | 42.1 | 1.09 | 1.00-1.19 | 1.10 | 0.99-1.23 | -0.46 |
| lord, routing | 319 | 31 | 0.7 | 0.8 | 1.15 | 0.63-1.88 | 1.38 | 0.71-2.07 | -0.01 |
| missile, melee | 1796 | 44 | 89.3 | 79.6 | 0.89 | 0.83-0.95 | 0.95 | 0.89-1.02 | 1.26 |
| missile, routing | 1778 | 44 | 12.1 | 12.0 | 0.99 | 0.77-1.29 | 0.96 | 0.72-1.29 | 0.02 |
| infantry melee, melee | 10116 | 46 | 298.4 | 292.0 | 0.98 | 0.94-1.02 | 1.03 | 0.99-1.08 | 0.82 |
| infantry melee, routing | 4647 | 46 | 19.2 | 19.5 | 1.02 | 0.88-1.17 | 1.06 | 0.88-1.27 | -0.04 |

### Rout starts by the game's morale at the start

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| eager / confident (ms 1-2) | 24797 | 46 | 11 | 16.5 | 1.50 | 0.73-3.50 |
| steady (3) | 4174 | 46 | 20 | 37.8 | 1.89 | 1.00-3.52 |
| shaken (4) | 4759 | 46 | 254 | 270.0 | 1.06 | 0.91-1.24 |
| wavering (5) | 2383 | 46 | 635 | 489.2 | 0.77 | 0.72-0.83 |
| mp < 0.25 | 2359 | 46 | 684 | 524.8 | 0.77 | 0.72-0.82 |
| mp 0.25-0.5 | 4100 | 46 | 206 | 215.0 | 1.04 | 0.87-1.26 |
| mp 0.5-0.75 | 5731 | 46 | 36 | 63.8 | 1.77 | 1.17-2.69 |
| mp >= 0.75 | 23945 | 46 | 13 | 19.2 | 1.48 | 0.74-3.25 |

### Rout starts by situation and unit kind

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all on the field | 36135 | 46 | 939 | 822.8 | 0.88 | 0.81-0.95 |
| standing, not in melee, no contact in window | 18470 | 46 | 133 | 101.5 | 0.76 | 0.54-1.07 |
| standing, shot at (an enemy shooting it) | 1870 | 46 | 120 | 112.0 | 0.93 | 0.81-1.05 |
| contact begins in window, ran in | 2966 | 46 | 63 | 68.2 | 1.08 | 0.88-1.33 |
| contact begins in window, walked in | 122 | 40 | 3 | 2.0 | 0.67 | 0.00-1.42 |
| melee, first 10 s of the fight | 1851 | 46 | 73 | 79.5 | 1.09 | 0.91-1.34 |
| melee, 10-30 s | 2674 | 46 | 147 | 135.0 | 0.92 | 0.79-1.08 |
| melee, 30 s+ | 8785 | 46 | 420 | 349.5 | 0.83 | 0.75-0.92 |
| melee, sim touches 1 enemy | 4820 | 46 | 270 | 236.5 | 0.88 | 0.80-0.96 |
| melee, sim touches 2+ enemies | 6824 | 46 | 294 | 277.8 | 0.94 | 0.83-1.07 |
| melee, sim touches nobody (game flag on) | 1666 | 46 | 76 | 49.8 | 0.65 | 0.52-0.84 |
| melee, touched in front | 4185 | 46 | 165 | 130.2 | 0.79 | 0.67-0.93 |
| melee, touched in flank | 5533 | 46 | 265 | 249.0 | 0.94 | 0.85-1.03 |
| melee, touched in rear | 1926 | 46 | 134 | 135.0 | 1.01 | 0.84-1.19 |
| melee, health < 30 % | 1705 | 46 | 329 | 290.8 | 0.88 | 0.79-0.98 |
| melee, health >= 30 % | 11605 | 46 | 311 | 273.2 | 0.88 | 0.77-1.02 |
| standing, health < 30 % | 4844 | 46 | 543 | 464.0 | 0.85 | 0.77-0.96 |
| lord | 3608 | 46 | 60 | 39.5 | 0.66 | 0.50-0.83 |
| missile | 11941 | 46 | 304 | 278.2 | 0.92 | 0.81-1.03 |
| infantry melee | 20586 | 46 | 575 | 505.0 | 0.88 | 0.80-0.97 |
| network's side | 17102 | 46 | 598 | 535.2 | 0.90 | 0.81-0.99 |
| the other side | 19033 | 46 | 341 | 287.5 | 0.84 | 0.74-0.97 |

### Rallies (routing at the start)

| situation | unit-windows | battles | game rallys | sim rallys | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all routers | 6744 | 46 | 507 | 468.2 | 0.92 | 0.85-0.99 |
| routers in melee | 1220 | 46 | 32 | 48.8 | 1.52 | 1.19-2.02 |
| routers free | 5524 | 46 | 475 | 419.5 | 0.88 | 0.81-0.94 |

### Morale change over the window (standing through it in the game)

| situation | unit-windows | game dMP | sim dMP | sim - game | 95 % CI |
|---|---|---|---|---|---|
| all on the field | 35196 | -0.017 | -0.012 | +0.005 | -0.000 - +0.009 |
| standing, not in melee, no contact in window | 18337 | +0.013 | +0.002 | -0.011 | -0.016 - -0.007 |
| standing, shot at (an enemy shooting it) | 1750 | -0.090 | -0.096 | -0.005 | -0.016 - +0.005 |
| contact begins in window, ran in | 2903 | -0.046 | -0.059 | -0.013 | -0.024 - -0.003 |
| contact begins in window, walked in | 119 | -0.086 | -0.083 | +0.003 | -0.028 - +0.038 |
| melee, first 10 s of the fight | 1778 | -0.118 | -0.126 | -0.007 | -0.020 - +0.004 |
| melee, 10-30 s | 2527 | -0.062 | -0.048 | +0.014 | +0.002 - +0.025 |
| melee, 30 s+ | 8365 | -0.025 | +0.022 | +0.046 | +0.035 - +0.057 |
| melee, sim touches 1 enemy | 4550 | -0.052 | -0.032 | +0.020 | +0.009 - +0.030 |
| melee, sim touches 2+ enemies | 6530 | -0.046 | -0.013 | +0.033 | +0.023 - +0.044 |
| melee, sim touches nobody (game flag on) | 1590 | -0.025 | +0.041 | +0.066 | +0.046 - +0.084 |
| melee, touched in front | 4020 | -0.045 | -0.010 | +0.035 | +0.024 - +0.045 |
| melee, touched in flank | 5268 | -0.048 | -0.022 | +0.026 | +0.015 - +0.036 |
| melee, touched in rear | 1792 | -0.055 | -0.041 | +0.014 | -0.002 - +0.030 |
| melee, health < 30 % | 1376 | -0.037 | -0.025 | +0.011 | -0.007 - +0.028 |
| melee, health >= 30 % | 11294 | -0.046 | -0.011 | +0.035 | +0.026 - +0.044 |
| standing, health < 30 % | 4301 | -0.013 | -0.007 | +0.006 | -0.006 - +0.017 |
| eager / confident (ms 1-2) | 24786 | -0.021 | -0.029 | -0.008 | -0.012 - -0.005 |
| steady (3) | 4154 | -0.022 | +0.017 | +0.039 | +0.028 - +0.049 |
| shaken (4) | 4505 | -0.012 | +0.028 | +0.040 | +0.029 - +0.050 |
| wavering (5) | 1748 | +0.044 | +0.063 | +0.020 | -0.005 - +0.041 |
| mp < 0.25 | 1675 | +0.038 | +0.064 | +0.026 | -0.001 - +0.047 |
| mp 0.25-0.5 | 3894 | -0.003 | +0.041 | +0.044 | +0.034 - +0.053 |
| mp 0.5-0.75 | 5695 | -0.024 | +0.014 | +0.038 | +0.029 - +0.047 |
| mp >= 0.75 | 23932 | -0.021 | -0.032 | -0.011 | -0.015 - -0.007 |

### HP dealt in melee by the dealer's recorded order (the game's: attributed, an estimate)

| dealer | unit-windows | battles | game HP dealt | sim HP dealt | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| hold, formation | 251 | 40 | 46842 | 18535 | 0.40 | 0.32-0.50 |
| hold, missile unit | 606 | 41 | 101222 | 49036 | 0.48 | 0.41-0.57 |
| hold, lord | 270 | 43 | 70981 | 29556 | 0.42 | 0.34-0.52 |
| move, formation | 399 | 42 | 59729 | 20843 | 0.35 | 0.29-0.43 |
| move, missile unit | 338 | 38 | 43388 | 8872 | 0.20 | 0.14-0.31 |
| move, lord | 33 | 20 | 4724 | 1481 | 0.31 | 0.15-0.56 |
| attack, formation | 9466 | 46 | 2562135 | 1701931 | 0.66 | 0.63-0.70 |
| attack, missile unit | 852 | 43 | 169247 | 106611 | 0.63 | 0.56-0.70 |
| attack, lord | 1095 | 46 | 283604 | 182224 | 0.64 | 0.57-0.72 |
