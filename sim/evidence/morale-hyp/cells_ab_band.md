## build/morale-hyp/ab_band

Windows of 10 s; 46 battles, 42879 unit-windows. HP lost sim/game 0.99 (0.96-1.02); rout starts sim/game 0.92 (0.85-1.01), game 939, sim 868.0.

Melee units, the loss by second of the window (sim/game of the cumulative loss):

| s | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| sim/game | 0.89 | 0.90 | 0.90 | 0.91 | 0.92 | 0.94 | 0.94 | 0.96 | 0.96 | 0.97 |

### HP lost by situation

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| all on the field | 42879 | 46 | 662.9 | 655.1 | 0.99 | 0.96-1.02 | 1.05 | 1.02-1.08 | 1.00 |
| standing, not in melee, no contact in window | 18470 | 46 | 43.5 | 56.3 | 1.29 | 1.20-1.39 | 1.37 | 1.26-1.51 | -1.63 |
| standing, shot at (an enemy shooting it) | 1870 | 46 | 108.2 | 104.2 | 0.96 | 0.92-1.01 | 1.09 | 1.02-1.18 | 0.51 |
| contact begins in window, ran in | 2966 | 46 | 83.2 | 77.3 | 0.93 | 0.88-0.98 | 0.90 | 0.84-0.96 | 0.76 |
| contact begins in window, walked in | 122 | 40 | 4.3 | 3.6 | 0.82 | 0.70-1.00 | 0.75 | 0.60-0.97 | 0.10 |
| melee, first 10 s of the fight | 1851 | 46 | 76.7 | 70.4 | 0.92 | 0.88-0.96 | 0.97 | 0.92-1.03 | 0.80 |
| melee, 10-30 s | 2674 | 46 | 104.8 | 92.6 | 0.88 | 0.85-0.92 | 0.92 | 0.87-0.97 | 1.57 |
| melee, 30 s+ | 8785 | 46 | 244.7 | 250.4 | 1.02 | 0.98-1.07 | 1.08 | 1.04-1.14 | -0.73 |
| melee, sim touches 1 enemy | 4820 | 46 | 162.7 | 146.7 | 0.90 | 0.87-0.94 | 0.96 | 0.91-1.01 | 2.05 |
| melee, sim touches 2+ enemies | 6824 | 46 | 219.8 | 233.0 | 1.06 | 1.02-1.10 | 1.10 | 1.05-1.15 | -1.68 |
| melee, sim touches nobody (game flag on) | 1666 | 46 | 43.6 | 33.8 | 0.77 | 0.71-0.84 | 0.90 | 0.82-0.98 | 1.27 |
| melee, touched in front | 4185 | 46 | 127.3 | 121.1 | 0.95 | 0.91-1.00 | 1.02 | 0.96-1.07 | 0.79 |
| melee, touched in flank | 5533 | 46 | 182.2 | 186.0 | 1.02 | 0.97-1.06 | 1.07 | 1.02-1.13 | -0.49 |
| melee, touched in rear | 1926 | 46 | 73.1 | 72.6 | 0.99 | 0.93-1.06 | 1.00 | 0.92-1.08 | 0.07 |
| melee, health < 30 % | 1705 | 46 | 39.3 | 45.7 | 1.16 | 1.09-1.25 | 1.27 | 1.16-1.41 | -0.82 |
| melee, health >= 30 % | 11605 | 46 | 386.8 | 367.7 | 0.95 | 0.92-0.98 | 1.00 | 0.97-1.04 | 2.45 |
| routing, in melee (pursued) | 1220 | 46 | 22.4 | 20.7 | 0.92 | 0.79-1.08 | 0.90 | 0.75-1.09 | 0.22 |
| routing, free | 5524 | 46 | 9.6 | 11.6 | 1.21 | 1.03-1.44 | 1.31 | 1.05-1.62 | -0.26 |
| standing, health < 30 % | 4844 | 46 | 54.0 | 61.1 | 1.13 | 1.06-1.22 | 1.25 | 1.15-1.38 | -0.91 |

### HP lost by unit kind

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord | 3927 | 46 | 54.6 | 62.1 | 1.14 | 1.03-1.25 | 1.16 | 1.04-1.29 | -0.97 |
| missile | 13719 | 46 | 188.0 | 174.6 | 0.93 | 0.88-0.98 | 1.00 | 0.94-1.06 | 1.72 |
| infantry melee | 25233 | 46 | 420.4 | 418.4 | 1.00 | 0.96-1.03 | 1.06 | 1.02-1.10 | 0.25 |
| network's side | 21084 | 46 | 373.7 | 369.2 | 0.99 | 0.95-1.03 | 1.05 | 1.01-1.10 | 0.58 |
| the other side | 21795 | 46 | 289.2 | 285.9 | 0.99 | 0.94-1.04 | 1.05 | 0.99-1.11 | 0.42 |

### HP lost by unit kind and state

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord, melee | 1398 | 46 | 38.5 | 42.0 | 1.09 | 1.00-1.19 | 1.10 | 0.99-1.23 | -0.45 |
| lord, routing | 319 | 31 | 0.7 | 0.8 | 1.15 | 0.63-1.89 | 1.38 | 0.71-2.09 | -0.01 |
| missile, melee | 1796 | 44 | 89.3 | 79.5 | 0.89 | 0.83-0.95 | 0.95 | 0.90-1.02 | 1.26 |
| missile, routing | 1778 | 44 | 12.1 | 12.0 | 0.99 | 0.77-1.29 | 0.96 | 0.72-1.28 | 0.01 |
| infantry melee, melee | 10116 | 46 | 298.4 | 291.9 | 0.98 | 0.94-1.02 | 1.03 | 0.99-1.08 | 0.83 |
| infantry melee, routing | 4647 | 46 | 19.2 | 19.5 | 1.02 | 0.88-1.17 | 1.06 | 0.88-1.27 | -0.04 |

### Rout starts by the game's morale at the start

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| eager / confident (ms 1-2) | 24797 | 46 | 11 | 16.8 | 1.52 | 0.73-3.61 |
| steady (3) | 4174 | 46 | 20 | 38.0 | 1.90 | 1.02-3.50 |
| shaken (4) | 4759 | 46 | 254 | 286.5 | 1.13 | 0.96-1.31 |
| wavering (5) | 2383 | 46 | 635 | 517.5 | 0.81 | 0.76-0.88 |
| mp < 0.25 | 2359 | 46 | 684 | 552.5 | 0.81 | 0.76-0.87 |
| mp 0.25-0.5 | 4100 | 46 | 206 | 230.0 | 1.12 | 0.94-1.34 |
| mp 0.5-0.75 | 5731 | 46 | 36 | 65.8 | 1.83 | 1.22-2.79 |
| mp >= 0.75 | 23945 | 46 | 13 | 19.8 | 1.52 | 0.75-3.42 |

### Rout starts by situation and unit kind

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all on the field | 36135 | 46 | 939 | 868.0 | 0.92 | 0.85-1.01 |
| standing, not in melee, no contact in window | 18470 | 46 | 133 | 100.0 | 0.75 | 0.53-1.05 |
| standing, shot at (an enemy shooting it) | 1870 | 46 | 120 | 111.8 | 0.93 | 0.81-1.05 |
| contact begins in window, ran in | 2966 | 46 | 63 | 69.2 | 1.10 | 0.89-1.37 |
| contact begins in window, walked in | 122 | 40 | 3 | 2.0 | 0.67 | 0.00-1.42 |
| melee, first 10 s of the fight | 1851 | 46 | 73 | 85.8 | 1.17 | 0.97-1.45 |
| melee, 10-30 s | 2674 | 46 | 147 | 139.5 | 0.95 | 0.82-1.10 |
| melee, 30 s+ | 8785 | 46 | 420 | 384.5 | 0.92 | 0.83-1.01 |
| melee, sim touches 1 enemy | 4820 | 46 | 270 | 253.0 | 0.94 | 0.85-1.03 |
| melee, sim touches 2+ enemies | 6824 | 46 | 294 | 300.8 | 1.02 | 0.90-1.16 |
| melee, sim touches nobody (game flag on) | 1666 | 46 | 76 | 56.0 | 0.74 | 0.58-0.95 |
| melee, touched in front | 4185 | 46 | 165 | 147.0 | 0.89 | 0.77-1.04 |
| melee, touched in flank | 5533 | 46 | 265 | 260.5 | 0.98 | 0.89-1.09 |
| melee, touched in rear | 1926 | 46 | 134 | 146.2 | 1.09 | 0.90-1.31 |
| melee, health < 30 % | 1705 | 46 | 329 | 317.2 | 0.96 | 0.88-1.06 |
| melee, health >= 30 % | 11605 | 46 | 311 | 292.5 | 0.94 | 0.82-1.09 |
| standing, health < 30 % | 4844 | 46 | 543 | 489.5 | 0.90 | 0.81-1.00 |
| lord | 3608 | 46 | 60 | 39.5 | 0.66 | 0.50-0.83 |
| missile | 11941 | 46 | 304 | 290.2 | 0.95 | 0.85-1.07 |
| infantry melee | 20586 | 46 | 575 | 538.2 | 0.94 | 0.85-1.03 |
| network's side | 17102 | 46 | 598 | 558.0 | 0.93 | 0.84-1.02 |
| the other side | 19033 | 46 | 341 | 310.0 | 0.91 | 0.80-1.04 |

### Rallies (routing at the start)

| situation | unit-windows | battles | game rallys | sim rallys | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all routers | 6744 | 46 | 507 | 468.5 | 0.92 | 0.85-0.99 |
| routers in melee | 1220 | 46 | 32 | 49.0 | 1.53 | 1.20-2.02 |
| routers free | 5524 | 46 | 475 | 419.5 | 0.88 | 0.81-0.94 |

### Morale change over the window (standing through it in the game)

| situation | unit-windows | game dMP | sim dMP | sim - game | 95 % CI |
|---|---|---|---|---|---|
| all on the field | 35196 | -0.017 | -0.017 | -0.000 | -0.005 - +0.004 |
| standing, not in melee, no contact in window | 18337 | +0.013 | +0.002 | -0.011 | -0.016 - -0.007 |
| standing, shot at (an enemy shooting it) | 1750 | -0.090 | -0.099 | -0.008 | -0.019 - +0.002 |
| contact begins in window, ran in | 2903 | -0.046 | -0.064 | -0.017 | -0.028 - -0.007 |
| contact begins in window, walked in | 119 | -0.086 | -0.086 | -0.000 | -0.031 - +0.034 |
| melee, first 10 s of the fight | 1778 | -0.118 | -0.131 | -0.013 | -0.026 - -0.002 |
| melee, 10-30 s | 2527 | -0.062 | -0.058 | +0.004 | -0.008 - +0.016 |
| melee, 30 s+ | 8365 | -0.025 | +0.007 | +0.032 | +0.021 - +0.043 |
| melee, sim touches 1 enemy | 4550 | -0.052 | -0.043 | +0.009 | -0.002 - +0.019 |
| melee, sim touches 2+ enemies | 6530 | -0.046 | -0.027 | +0.019 | +0.008 - +0.029 |
| melee, sim touches nobody (game flag on) | 1590 | -0.025 | +0.033 | +0.058 | +0.038 - +0.077 |
| melee, touched in front | 4020 | -0.045 | -0.022 | +0.023 | +0.012 - +0.033 |
| melee, touched in flank | 5268 | -0.048 | -0.035 | +0.013 | +0.001 - +0.023 |
| melee, touched in rear | 1792 | -0.055 | -0.053 | +0.002 | -0.014 - +0.018 |
| melee, health < 30 % | 1376 | -0.037 | -0.037 | -0.000 | -0.019 - +0.017 |
| melee, health >= 30 % | 11294 | -0.046 | -0.024 | +0.023 | +0.013 - +0.032 |
| standing, health < 30 % | 4301 | -0.013 | -0.011 | +0.002 | -0.010 - +0.013 |
| eager / confident (ms 1-2) | 24786 | -0.021 | -0.033 | -0.012 | -0.015 - -0.008 |
| steady (3) | 4154 | -0.022 | +0.009 | +0.031 | +0.021 - +0.040 |
| shaken (4) | 4505 | -0.012 | +0.018 | +0.030 | +0.019 - +0.040 |
| wavering (5) | 1748 | +0.044 | +0.057 | +0.013 | -0.011 - +0.035 |
| mp < 0.25 | 1675 | +0.038 | +0.057 | +0.019 | -0.008 - +0.040 |
| mp 0.25-0.5 | 3894 | -0.003 | +0.031 | +0.034 | +0.024 - +0.043 |
| mp 0.5-0.75 | 5695 | -0.024 | +0.005 | +0.029 | +0.021 - +0.037 |
| mp >= 0.75 | 23932 | -0.021 | -0.035 | -0.014 | -0.018 - -0.010 |

### HP dealt in melee by the dealer's recorded order (the game's: attributed, an estimate)

| dealer | unit-windows | battles | game HP dealt | sim HP dealt | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| hold, formation | 251 | 40 | 46842 | 18406 | 0.39 | 0.31-0.50 |
| hold, missile unit | 606 | 41 | 101222 | 48624 | 0.48 | 0.40-0.57 |
| hold, lord | 270 | 43 | 70981 | 29506 | 0.42 | 0.34-0.52 |
| move, formation | 399 | 42 | 59729 | 20877 | 0.35 | 0.29-0.43 |
| move, missile unit | 338 | 38 | 43388 | 8808 | 0.20 | 0.14-0.30 |
| move, lord | 33 | 20 | 4724 | 1481 | 0.31 | 0.15-0.56 |
| attack, formation | 9466 | 46 | 2562135 | 1699921 | 0.66 | 0.63-0.70 |
| attack, missile unit | 852 | 43 | 169247 | 105994 | 0.63 | 0.55-0.70 |
| attack, lord | 1095 | 46 | 283604 | 182227 | 0.64 | 0.57-0.72 |
