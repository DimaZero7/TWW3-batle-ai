## build/morale-hyp/sr_final

Windows of 10 s; 92 battles, 124124 unit-windows. HP lost sim/game 1.01 (0.98-1.05); rout starts sim/game 0.95 (0.87-1.02), game 2691, sim 2543.2.

Melee units, the loss by second of the window (sim/game of the cumulative loss):

| s | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| sim/game | 0.88 | 0.88 | 0.88 | 0.90 | 0.91 | 0.92 | 0.94 | 0.95 | 0.97 | 0.98 |

### HP lost by situation

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| all on the field | 124124 | 92 | 1631.4 | 1649.7 | 1.01 | 0.98-1.05 | 1.08 | 1.04-1.12 | 1.00 |
| standing, not in melee, no contact in window | 51421 | 92 | 79.4 | 114.1 | 1.44 | 1.33-1.57 | 1.54 | 1.42-1.69 | 1.90 |
| standing, shot at (an enemy shooting it) | 3985 | 92 | 196.4 | 193.5 | 0.99 | 0.95-1.02 | 1.14 | 1.08-1.22 | -0.16 |
| contact begins in window, ran in | 9006 | 92 | 204.1 | 197.7 | 0.97 | 0.92-1.02 | 0.95 | 0.90-1.01 | -0.35 |
| contact begins in window, walked in | 327 | 83 | 8.9 | 7.4 | 0.83 | 0.73-0.95 | 0.82 | 0.71-0.95 | -0.08 |
| melee, first 10 s of the fight | 5739 | 92 | 194.2 | 177.1 | 0.91 | 0.88-0.95 | 0.99 | 0.94-1.03 | -0.94 |
| melee, 10-30 s | 8207 | 92 | 268.3 | 245.9 | 0.92 | 0.88-0.95 | 0.98 | 0.94-1.03 | -1.23 |
| melee, 30 s+ | 25884 | 92 | 656.5 | 669.0 | 1.02 | 0.98-1.06 | 1.10 | 1.05-1.14 | 0.69 |
| melee, sim touches 1 enemy | 14716 | 92 | 420.3 | 375.9 | 0.89 | 0.87-0.93 | 0.96 | 0.93-1.00 | -2.43 |
| melee, sim touches 2+ enemies | 19308 | 92 | 580.9 | 625.3 | 1.08 | 1.03-1.12 | 1.14 | 1.09-1.19 | 2.43 |
| melee, sim touches nobody (game flag on) | 5806 | 92 | 117.8 | 90.8 | 0.77 | 0.72-0.83 | 0.93 | 0.87-1.00 | -1.48 |
| melee, touched in front | 12723 | 92 | 319.3 | 305.9 | 0.96 | 0.93-0.99 | 1.03 | 0.99-1.08 | -0.73 |
| melee, touched in flank | 15606 | 92 | 479.3 | 492.2 | 1.03 | 0.99-1.07 | 1.09 | 1.05-1.14 | 0.71 |
| melee, touched in rear | 5695 | 92 | 202.6 | 203.0 | 1.00 | 0.95-1.05 | 1.04 | 0.99-1.10 | 0.02 |
| melee, health < 30 % | 4220 | 92 | 87.0 | 100.4 | 1.15 | 1.10-1.22 | 1.29 | 1.20-1.37 | 0.74 |
| melee, health >= 30 % | 35610 | 92 | 1032.0 | 991.6 | 0.96 | 0.93-0.99 | 1.03 | 0.99-1.07 | -2.21 |
| routing, in melee (pursued) | 3976 | 92 | 63.3 | 75.9 | 1.20 | 1.05-1.35 | 1.09 | 0.95-1.26 | 0.69 |
| routing, free | 16974 | 92 | 28.2 | 36.1 | 1.28 | 1.12-1.48 | 1.27 | 1.08-1.51 | 0.44 |
| standing, health < 30 % | 13319 | 92 | 118.5 | 133.3 | 1.12 | 1.07-1.18 | 1.25 | 1.18-1.33 | 0.81 |

### HP lost by unit kind

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord | 9243 | 92 | 110.7 | 126.1 | 1.14 | 1.07-1.21 | 1.16 | 1.09-1.24 | 0.84 |
| missile | 28313 | 92 | 348.2 | 326.0 | 0.94 | 0.90-0.97 | 1.00 | 0.95-1.05 | -1.22 |
| infantry melee | 86568 | 92 | 1172.5 | 1197.6 | 1.02 | 0.98-1.06 | 1.09 | 1.05-1.14 | 1.37 |
| network's side | 60359 | 92 | 913.7 | 919.0 | 1.01 | 0.97-1.04 | 1.07 | 1.03-1.11 | 0.29 |
| the other side | 63765 | 92 | 717.7 | 730.6 | 1.02 | 0.96-1.08 | 1.09 | 1.03-1.16 | 0.71 |

### HP lost by unit kind and state

| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI | after 5 s: sim/game | 95 % CI | share of the total deficit |
|---|---|---|---|---|---|---|---|---|---|
| lord, melee | 3097 | 92 | 77.8 | 84.5 | 1.09 | 1.02-1.14 | 1.10 | 1.02-1.17 | 0.37 |
| lord, routing | 747 | 69 | 2.5 | 2.8 | 1.11 | 0.83-1.43 | 1.48 | 1.04-2.12 | 0.02 |
| missile, melee | 4468 | 90 | 200.4 | 180.7 | 0.90 | 0.86-0.94 | 0.97 | 0.93-1.01 | -1.08 |
| missile, routing | 3430 | 88 | 20.5 | 26.5 | 1.29 | 1.05-1.60 | 1.23 | 0.94-1.59 | 0.33 |
| infantry melee, melee | 32265 | 92 | 840.8 | 826.9 | 0.98 | 0.95-1.02 | 1.06 | 1.02-1.11 | -0.76 |
| infantry melee, routing | 16773 | 92 | 68.5 | 82.8 | 1.21 | 1.05-1.37 | 1.12 | 0.96-1.30 | 0.78 |

### Rout starts by the game's morale at the start

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| eager / confident (ms 1-2) | 68540 | 92 | 21 | 48.0 | 2.29 | 1.32-4.22 |
| steady (3) | 10339 | 92 | 40 | 67.8 | 1.69 | 1.12-2.47 |
| shaken (4) | 16007 | 92 | 630 | 790.8 | 1.26 | 1.10-1.40 |
| wavering (5) | 8229 | 92 | 1956 | 1613.8 | 0.83 | 0.77-0.89 |
| mp < 0.25 | 7641 | 92 | 1990 | 1617.2 | 0.81 | 0.75-0.88 |
| mp 0.25-0.5 | 12928 | 92 | 581 | 680.2 | 1.17 | 1.04-1.30 |
| mp 0.5-0.75 | 15827 | 92 | 94 | 187.0 | 1.99 | 1.58-2.43 |
| mp >= 0.75 | 66778 | 92 | 26 | 58.8 | 2.26 | 1.32-4.13 |

### Rout starts by situation and unit kind

| situation | unit-windows | battles | game routs | sim routs | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all on the field | 103174 | 92 | 2691 | 2543.2 | 0.95 | 0.87-1.02 |
| standing, not in melee, no contact in window | 51421 | 92 | 381 | 281.0 | 0.74 | 0.58-0.89 |
| standing, shot at (an enemy shooting it) | 3985 | 92 | 238 | 217.2 | 0.91 | 0.83-1.01 |
| contact begins in window, ran in | 9006 | 92 | 166 | 189.5 | 1.14 | 0.96-1.35 |
| contact begins in window, walked in | 327 | 83 | 10 | 5.2 | 0.53 | 0.23-1.00 |
| melee, first 10 s of the fight | 5739 | 92 | 251 | 266.0 | 1.06 | 0.95-1.18 |
| melee, 10-30 s | 8207 | 92 | 403 | 406.0 | 1.01 | 0.92-1.10 |
| melee, 30 s+ | 25884 | 92 | 1280 | 1229.0 | 0.96 | 0.87-1.05 |
| melee, sim touches 1 enemy | 14716 | 92 | 732 | 738.2 | 1.01 | 0.92-1.14 |
| melee, sim touches 2+ enemies | 19308 | 92 | 985 | 1024.5 | 1.04 | 0.94-1.14 |
| melee, sim touches nobody (game flag on) | 5806 | 92 | 217 | 138.2 | 0.64 | 0.53-0.74 |
| melee, touched in front | 12723 | 92 | 467 | 421.0 | 0.90 | 0.81-1.01 |
| melee, touched in flank | 15606 | 92 | 811 | 851.2 | 1.05 | 0.96-1.15 |
| melee, touched in rear | 5695 | 92 | 439 | 490.5 | 1.12 | 0.99-1.25 |
| melee, health < 30 % | 4220 | 92 | 843 | 810.8 | 0.96 | 0.89-1.05 |
| melee, health >= 30 % | 35610 | 92 | 1091 | 1090.2 | 1.00 | 0.90-1.10 |
| standing, health < 30 % | 13319 | 92 | 1409 | 1251.8 | 0.89 | 0.81-0.97 |
| lord | 8496 | 92 | 134 | 96.0 | 0.72 | 0.57-0.87 |
| missile | 24883 | 92 | 591 | 557.5 | 0.94 | 0.86-1.04 |
| infantry melee | 69795 | 92 | 1966 | 1889.8 | 0.96 | 0.87-1.06 |
| network's side | 47765 | 92 | 1745 | 1567.2 | 0.90 | 0.83-0.97 |
| the other side | 55409 | 92 | 946 | 976.0 | 1.03 | 0.88-1.22 |

### Rallies (routing at the start)

| situation | unit-windows | battles | game rallys | sim rallys | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| all routers | 20950 | 92 | 1541 | 1616.5 | 1.05 | 1.00-1.09 |
| routers in melee | 3976 | 92 | 126 | 201.5 | 1.60 | 1.40-1.87 |
| routers free | 16974 | 92 | 1415 | 1415.0 | 1.00 | 0.96-1.04 |

### Morale change over the window (standing through it in the game)

| situation | unit-windows | game dMP | sim dMP | sim - game | 95 % CI |
|---|---|---|---|---|---|
| all on the field | 100483 | -0.016 | -0.009 | +0.006 | +0.003 - +0.010 |
| standing, not in melee, no contact in window | 51040 | +0.012 | +0.004 | -0.009 | -0.012 - -0.006 |
| standing, shot at (an enemy shooting it) | 3747 | -0.080 | -0.083 | -0.002 | -0.010 - +0.006 |
| contact begins in window, ran in | 8840 | -0.046 | -0.050 | -0.004 | -0.012 - +0.004 |
| contact begins in window, walked in | 317 | -0.052 | -0.051 | +0.002 | -0.021 - +0.023 |
| melee, first 10 s of the fight | 5488 | -0.112 | -0.103 | +0.008 | +0.000 - +0.016 |
| melee, 10-30 s | 7804 | -0.056 | -0.041 | +0.015 | +0.008 - +0.021 |
| melee, 30 s+ | 24604 | -0.023 | +0.015 | +0.039 | +0.032 - +0.046 |
| melee, sim touches 1 enemy | 13984 | -0.046 | -0.031 | +0.015 | +0.009 - +0.023 |
| melee, sim touches 2+ enemies | 18323 | -0.046 | -0.020 | +0.026 | +0.019 - +0.033 |
| melee, sim touches nobody (game flag on) | 5589 | -0.024 | +0.051 | +0.075 | +0.065 - +0.085 |
| melee, touched in front | 12256 | -0.040 | -0.009 | +0.030 | +0.023 - +0.037 |
| melee, touched in flank | 14795 | -0.048 | -0.031 | +0.018 | +0.011 - +0.025 |
| melee, touched in rear | 5256 | -0.054 | -0.043 | +0.011 | -0.001 - +0.023 |
| melee, health < 30 % | 3377 | -0.035 | -0.017 | +0.018 | +0.009 - +0.028 |
| melee, health >= 30 % | 34519 | -0.044 | -0.013 | +0.030 | +0.024 - +0.037 |
| standing, health < 30 % | 11910 | -0.012 | -0.003 | +0.008 | +0.001 - +0.017 |
| eager / confident (ms 1-2) | 68519 | -0.021 | -0.028 | -0.007 | -0.009 - -0.004 |
| steady (3) | 10299 | -0.021 | +0.017 | +0.038 | +0.032 - +0.045 |
| shaken (4) | 15377 | -0.011 | +0.021 | +0.031 | +0.021 - +0.042 |
| wavering (5) | 6273 | +0.038 | +0.073 | +0.035 | +0.024 - +0.047 |
| mp < 0.25 | 5651 | +0.035 | +0.073 | +0.037 | +0.023 - +0.054 |
| mp 0.25-0.5 | 12347 | -0.002 | +0.036 | +0.038 | +0.027 - +0.049 |
| mp 0.5-0.75 | 15733 | -0.021 | +0.013 | +0.034 | +0.028 - +0.040 |
| mp >= 0.75 | 66752 | -0.021 | -0.030 | -0.009 | -0.011 - -0.006 |

### HP dealt in melee by the dealer's recorded order (the game's: attributed, an estimate)

| dealer | unit-windows | battles | game HP dealt | sim HP dealt | sim/game | 95 % CI |
|---|---|---|---|---|---|---|
| hold, formation | 889 | 86 | 131214 | 52902 | 0.40 | 0.35-0.46 |
| hold, missile unit | 1445 | 84 | 217281 | 124577 | 0.57 | 0.51-0.64 |
| hold, lord | 596 | 87 | 158713 | 70119 | 0.44 | 0.38-0.51 |
| move, formation | 1485 | 88 | 183842 | 58134 | 0.32 | 0.28-0.36 |
| move, missile unit | 899 | 83 | 84489 | 24486 | 0.29 | 0.22-0.38 |
| move, lord | 90 | 45 | 11857 | 3468 | 0.29 | 0.17-0.45 |
| attack, formation | 29891 | 92 | 7200979 | 4844634 | 0.67 | 0.64-0.70 |
| attack, missile unit | 2124 | 89 | 372731 | 229862 | 0.62 | 0.57-0.67 |
| attack, lord | 2411 | 92 | 555622 | 412719 | 0.74 | 0.68-0.82 |
