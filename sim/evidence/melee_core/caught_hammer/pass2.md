# The intercepted hammer, second pass

**[CONTEXT] Reject A, B and C.** AC is the best combination within the flank2v1 guardrail. It closes 29.7%/28.4% of the narrow/wide forced-trade gap, below the 50% threshold. ABC closes 52.3%/49.8% and worsens flank2v1 narrow rear−front error by 0.0470. AC recovers narrow target loss in minute one (0.218 versus 0.214 in game) but misses minute two (0.143 versus 0.223).

READ = disassembly or simulator code. CONTEXT = measurements. GUESS = experimental rule.

## A. Reach

**[CONTEXT]** H = hammer, C = comer, T = ordered target. Positions are target-relative metres, mirrored to positive x. Traces use one jitter per battle and geometry at tick start. Positive gaps separate rectangles. Counts are effective fighters before incidental-damage scaling.

|Seconds from MOVE|Game H (x,z), narrow / wide|Twin H (x,z), narrow / wide|Twin H–T gap, narrow / wide|
|---:|---|---|---|
|20|(42,11) / (46,10)|(44,9) / (48,10)|16 / 13|
|30|(46,3) / (52,7)|(46,7) / (52,7)|19 / 17|
|36|(35,4) / (42,5)|(46,7) / (50,6)|24 / 21|
|48|(23,3) / (27,2)|(20,3) / (27,4)|0.3 / 0.2|

1. **[READ/CONTEXT]** Around 20 s C intercepts H on MOVE. Blocked frontage slows H's advance to its waypoint. C can stop. At 30 s H allocates 15.7/23.9 fighters to C and zero to T. The game switches to ATTACK at 28.4/30.0 s on average, the twin at 30.0/29.9 s.
2. **[CONTEXT]** In the first narrow trace H turns from 130° at 30 s to alternating 86°/146° half-steps at 31–34.5 s. Each recontact restarts the pin clock. Re-entry precedes target damage in 30/32 traces. At 36 s H still faces 86° at zero speed. The game H faces 274°. All game hammers show `melee,moving_fast` at 30 and 36 s, then `melee` at 48 s. At 36 s, 25/32 comers show only `moving_fast`. Then H fields zero fighters, C fields 12.4/15.7 against H. The twin spends 3.9/3.7 s pinned before damaging T.
3. **[READ/CONTEXT]** H runs once released, then walks the last stretch for 6.5/6.8 s under the retarget rule. C copies the leaving target's velocity while chasing. T attacks the anvil, so `engage_release` applies the −2.5 + 2.0 = −0.5 m gap limit to H–T. At first damage all 32 rectangles overlap by 0.53–1.20 m. H allocates 2.8/3.7 fighters to T and 11.0/16.3 to C. At the game's first target swing the hitter is 13.4/18.1 m from H's mean soldier position. Hammer men within 4 m of T (2.8/1.8) and C (25.1/37.3) form disjoint groups throughout.
4. **[READ]** The attack order assigns targets per soldier at `1432781b8`. **[GUESS]** A gives each free formation slot a straight path from its order-time position to T at order speed, ignoring subsequent obstructions. Slots within reach of C at the order stay behind. Reach uses the unit data's 2.5 m plus enemy radius against outer soldier centres, simplifying the scaled two-radius test at `14308b608`. Arrivals cap both directions after the ordinary reply allocation. All 738 early-contact ticks respect it. **[CONTEXT]** First damage moves from 48.7/46.8 s to 36.7/36.1 s, against game swing starts at 36.0/36.5 s. Narrow first-minute target loss rises from 0.142 to 0.195. Wide 120 s hammer loss rises from 0.767 to 0.784.

## B. Facing of the struck soldier

**[READ]** The swing calls the victim's conditional reply at `143080dfc → 143074dcc`. Occupied actions and breakoff can veto it before the individual order at `143074e9c`. Two paths write a point-derived angle directly to the man's `+0xb0` in one update (`143077298`, flag 0x40, and `14309ee3d`, selected action states). `143161393–1431613d5` derives angular velocity from a supplied angle change. A general reply delay remains unknown. Per blow, `1430a90b7–1430a90e4` constructs its circumstances. `14309b721–14309b776` reads the defender's own facing and both soldiers' positions. `1430a6380` classifies the 45°/135° sectors. `142e53208` tests the attacker's forward half-plane. Eligible side/rear blows multiply defence by 0.6/0.3.

**[CONTEXT/GUESS]** We timed each identified pair from its first off-axis swing start to entry within 45°. Of 633 mutually targeting pairs, 356 reached front before target change, disappearance or window end censored observation. Kaplan–Meier medians were 3.0 s in cforced, 2.0 s in flank2v1 and 2.5 s pooled. This includes relative motion. B uses the pooled median as a half-life starting at pair contact. Reciprocal fighters face each other first. Surplus blows sample the bearings of the defender's other opponents. Rounding and hit-chance limits apply per sector.

**[CONTEXT]** After breakoff, B changes the primary front/rear classification of damaging C→H ticks (before wrap correction) from 7/23% to 98/2%, versus 58/6% of game swing starts. In 625 of 648 game rear starts the victim targeted someone in C. Unit-level reciprocity overcounts exact opponents. B cuts wide hammer loss from 0.767 to 0.638. On the wide rear regression, the original trace already assigns 100% of late H→T damaging ticks to rear, versus 75/1,326 game starts. B gives 18.4%. Wide rear−front target loss is +0.255 in game and +0.062 before. More rear attribution alone cannot supply the missing trade. B worsens wide rear−front by 0.0209 and narrow rear−front absolute error by 0.0470.

## C. Charge after interception

**[READ]** Charge gait states (`14307e1a0`, `142e52cc4`) refresh the soldier's deadline to now plus 13 s (`14308d0a6–14308d0af`). Afterward `14305e9c0` reads his remaining linear bonus. **[CONTEXT]** Of 464 early H→T game starts, 461 have live deadlines with 12.5 s median remaining across all starts. **[GUESS]** C refreshes the approaching reserve inside charge distance on a clear running attack. Each newly admitted fighter retains its own deadline on joining T. The twin applies their mean factor to attack and weapon damage on that pair. **[CONTEXT]** All 512 first target-damage ticks carry full charge with C alone, still at 48.7/46.8 s. Its 13 s bonus has little leverage in minute two.

## Paired outcomes

**[CONTEXT]** CPU, seed 0, recorded starts. Each setting has 2,560 side2r rows from the original order and 320 flank2v1 rows from hammer contact. Trade subtracts combined friendly loss from target loss, each normalized by starting HP. Acceptance requires one-third closure alone or one-half combined at both widths. Ranking uses the smaller closure. AC leads within the 0.01 regression guardrail, ABC overall.

|Probe and 120 s trade difference|Width|Game|Before|A|B|C|AC|ABC|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|side2r flank − front|N|+0.0960|+0.0392|+0.0392|+0.0297|+0.0392|+0.0392|+0.0297|
|side2r flank − front|W|+0.0885|+0.0611|+0.0611|+0.0523|+0.0611|+0.0611|+0.0523|
|side2r cforced − cfront|N|+0.0225|-0.1790|-0.1414|-0.1304|-0.1739|-0.1192|-0.0736|
|side2r cforced − cfront|W|-0.0160|-0.2386|-0.1989|-0.1904|-0.2310|-0.1753|-0.1278|
|side2r cteach − cforced|N|-0.0468|+0.1922|+0.1546|+0.1471|+0.1882|+0.1335|+0.0911|
|side2r cteach − cforced|W|-0.0409|+0.2121|+0.1725|+0.1763|+0.2062|+0.1505|+0.1153|
|flank2v1 flank − front|N|+0.1097|+0.0911|+0.0911|+0.0747|+0.0911|+0.0911|+0.0747|
|flank2v1 flank − front|W|+0.1010|+0.0800|+0.0800|+0.1072|+0.0800|+0.0800|+0.1072|
|flank2v1 rear − front|N|+0.1121|+0.1132|+0.1132|+0.0640|+0.1132|+0.1132|+0.0640|
|flank2v1 rear − front|W|+0.1805|+0.0370|+0.0370|+0.0162|+0.0370|+0.0370|+0.0162|

**cforced HP loss** in target / anvil / hammer order.

|Width|Seconds|Game|Before|A|B|C|AC|ABC|
|---|---|---|---|---|---|---|---|---|
|N|0–60|0.214/0.209/0.278|0.142/0.192/0.305|0.195/0.192/0.333|0.141/0.192/0.278|0.147/0.192/0.305|0.218/0.192/0.332|0.218/0.192/0.307|
|N|60–120|0.223/0.137/0.260|0.148/0.128/0.317|0.145/0.128/0.315|0.146/0.128/0.209|0.148/0.128/0.317|0.143/0.128/0.314|0.142/0.128/0.211|
|W|0–60|0.234/0.270/0.308|0.211/0.324/0.410|0.251/0.313/0.424|0.210/0.324/0.379|0.218/0.324/0.410|0.271/0.313/0.424|0.270/0.313/0.395|
|W|60–120|0.206/0.177/0.236|0.194/0.206/0.358|0.197/0.206/0.360|0.193/0.206/0.259|0.194/0.205/0.357|0.200/0.206/0.360|0.198/0.205/0.261|

**[CONTEXT]** Gap closure N/W is 18.7%/17.8% for A, 24.1%/21.6% for B and 2.5%/3.4% for C. Other combination closures were AB 41.6%/39.3%, BC 26.5%/25.0%. Switches `contact.free_advance`, `melee.individual_facing` and `charge.intercept_deadline`. All default to zero. The copy and original then match every tensor at every tick on 20 recorded lanes (`7a145a08…4a2729`). [A.patch](A.patch) · [B.patch](B.patch) · [C.patch](C.patch) · [Replay and measurements](work/)

**Still not established.** The records sample swing starts. Impact angles, the animation delay for a particular reply, exact soldier paths through the interception and the source of the remaining second-minute trade error are unresolved.
