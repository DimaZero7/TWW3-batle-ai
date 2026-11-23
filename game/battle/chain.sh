#!/usr/bin/env bash
# The chain's battles in the game (the after-N checks): the person's matchups M1-M3 on the Kislev plains, the strategist
# deploys (default, task hold), the chain (operative -> tacticians -> the unit network) commands our side against the
# game's AI on Normal; x3, light graphics on the RTX (BAI_GRAPHICS=low), the game lock (ops/guard.sh).
# usage: bash game/battle/chain.sh [-n BATTLES] m1 [m2 m3]        (default: m1 m2 m3, 2 battles each)
#        CHAIN_ARGS="-ChainOp net -ChainTactic net" bash game/battle/chain.sh m1   (watch.ps1's chain options)
#        DRY=1 bash game/battle/chain.sh m3      (the checks and the builds only: the commands printed, no game)
# Logs: tmp/chain/<m>.build.log, <m>_<i>.log; runs: build/nn-arena/runs/<time>/; the table at the end:
#        python -m game.battle.table <runs>
# The armies: M1, M2 - the person's runs (data/human/runs, the Nuln Ironsides -> the Hochland long rifles: the user 07.11),
# M3 - the v3 generator's seed 1000702112 (game/label/live_matches.json 1); the commands of the after6 battles (08.11).
#        bash game/battle/chain.sh tactic [f1]   the chain 'tactic_v2 only' (bridge/companion/tactic_chain.py): ONE battle
#                 of the stage-0 armies (game/battle/aivai.py, default f1 - flat, the game's AI attacks), the trained tactic
#                 weights/tactic/attack1/final.pt commands our side, the unit script under it; BAI_TICK_MS=100 above 5,000
#                 men; then: bash ops/dock.sh game.battle.tactic_read <run>
#        OP=holder:M|holder:D bash game/battle/chain.sh tactic h01   the game probe plan `holder` (tmp/op1/design_merged.md):
#                 a start h01..h24 of game/battle/holder_read.py (build/holder; our side attacks, the game's AI defends),
#                 the scripted operative net/operative/opside.py on the chain (BAI_OP_SCRIPT); without OP as before
#        bash game/battle/chain.sh tactic r01   the plan's reference arm R: the same start with the roles swapped (the
#                 game's AI attacks, ours defends), the chain without the operative script (OP must be empty)
#        LORDS=park bash game/battle/chain.sh tactic h01   both lords out of the fight (bridge.build --park-lords: two far
#                 corners, held by the script; the manifest's config.park_lords); empty - as before
#        PLACE=<line.json> [ZONES=east] bash game/battle/chain.sh tactic c01   the place switch (drill S0-1's crest8,
#                 starts c01..c08 of game/battle/crest_read.py: the game's AI attacks, ours hold; any start takes it):
#                 our infantry deployed on the file's line, the lord behind (bridge.build --deploy strategist, the
#                 companion's --strategist <file>: bridge/companion/deploy.LinePlace); ZONES=east - bridge.build
#                 --map-zones east (the game's east.xml zones on a map captured without them: kho_wastes_01); empty - as before
cd "$(dirname "$0")/../.." || exit 1
. ops/guard.sh
. ops/python.sh
if [ "$1" = "tactic" ]; then
  b="${2:-f1}"
  MK=game.battle.aivai; MKA=make; AD=build/aivai
  case "$b" in h[0-9]*|r[0-9]*) MK=game.battle.holder_read; MKA=--make; AD=build/holder ;; esac
  case "$b" in c[0-9]*) MK=game.battle.crest_read; MKA=--make; AD=build/crest8 ;; esac
  case "$b" in r[0-9]*) [ -n "$OP" ] && { echo "$b is the reference arm R: no OP" >&2; exit 1; } ;; esac
  if [ -n "$OP" ]; then export BAI_OP_SCRIPT="$OP"; else unset BAI_OP_SCRIPT; fi
  case "${LORDS:-}" in ""|park) ;; *) echo "LORDS must be empty or park" >&2; exit 1 ;; esac
  case "${ZONES:-}" in ""|east) ;; *) echo "ZONES must be empty or east" >&2; exit 1 ;; esac
  [ -n "$PLACE" ] && { [ -f "$PLACE" ] || { echo "no place file $PLACE" >&2; exit 1; }; }
  pt=""; [ -n "$PLACE" ] && pt="$(basename "$PLACE" .json)"
  tag="$b${OP:+_${OP#*:}}${LORDS:+_$LORDS}${pt:+_$pt}"
  W=weights/tactic/attack1/final.pt
  [ -f "$W" ] || { echo "no $W" >&2; exit 1; }
  if [ -n "$DRY" ]; then why="$(guard_busy)" && { echo "not starting: $why" >&2; exit 1; }
  else guard_take "chain tactic $b" || exit 1; trap guard_release EXIT; fi
  export BAI_GRAPHICS="${BAI_GRAPHICS:-low}"
  mkdir -p tmp/chain
  "$PY" -m $MK $MKA > tmp/chain/tactic_make.log 2>&1 || { echo "make failed (tmp/chain/tactic_make.log)"; exit 1; }
  read -r map role men < <("$PY" -c "import json;b=json.load(open('$AD/battles.json'))['$b'];print(b['map'],b['role'],sum(u['men'] for s in b['units'].values() for u in s))")
  men="${men//[!0-9]/}"
  [ "$men" -gt 5000 ] && export BAI_TICK_MS=100
  args=(--own-ai net --arena-json "$AD/$b.json" --speed 3 --timeout 900)
  [ "$map" != "flat" ] && args+=(--map "$map")
  [ "$role" = "defend" ] && args+=(--own-role attack)
  [ "$LORDS" = "park" ] && args+=(--park-lords)
  [ -n "$PLACE" ] && args+=(--deploy strategist)
  [ "$ZONES" = "east" ] && args+=(--map-zones east)
  wargs=(); [ -n "$PLACE" ] && wargs=(-Strategist "$PLACE" -StrategistTask hold)
  echo "=== tactic $b $map ai:$role men $men tick_ms ${BAI_TICK_MS:-engine}${OP:+ op $OP}${LORDS:+ lords $LORDS}${PLACE:+ place $PLACE}${ZONES:+ zones $ZONES} $(date +%T)"
  "$PY" -m bridge.build nn-arena "${args[@]}" > "tmp/chain/tactic_$tag.build.log" 2>&1     || { echo "build failed (tmp/chain/tactic_$tag.build.log)"; exit 1; }
  if [ -n "$DRY" ]; then echo "dry: built; watch.ps1 -NoBuild -Chain -ChainTactic $W ${wargs[*]}"; exit 0; fi
  guard_human && { echo "the user plays: stop"; exit 1; }
  powershell -NoProfile -ExecutionPolicy Bypass -File bridge/launcher/watch.ps1 -NoBuild -Chain -ChainTactic "$W" "${wargs[@]}"     > "tmp/chain/tactic_$tag.log" 2>&1
  code=$?
  run=$(ls -t build/nn-arena/runs 2>/dev/null | head -1)
  echo "--- tactic $tag exit $code $(date +%T) $run; read: bash ops/dock.sh game.battle.tactic_read build/nn-arena/runs/$run"
  exit $code
fi
n=2
if [ "$1" = "-n" ]; then n="$2"; shift 2; fi
[ $# -ge 1 ] || set -- m1 m2 m3
SWAP=wh3_dlc25_emp_inf_nuln_ironsides=wh3_dlc25_emp_inf_hochland_long_rifles
army() {
  case "$1" in
    m1) echo "--army-from-run data/human/runs/20261025-192124 --army-replace $SWAP" ;;
    m2) echo "--army-from-run data/human/runs/20261107-171600 --army-replace $SWAP" ;;
    m3) echo "--army-seed 1000702112 --armies v3" ;;
    *) return 1 ;;
  esac
}
for m in "$@"; do army "$m" > /dev/null || { echo "unknown matchup $m (m1 m2 m3)" >&2; exit 1; }; done
if [ -n "$DRY" ]; then why="$(guard_busy)" && { echo "not starting: $why" >&2; exit 1; }
else guard_take "chain $*" || exit 1; trap guard_release EXIT; fi
export BAI_GRAPHICS="${BAI_GRAPHICS:-low}"
mkdir -p tmp/chain
runs=()
for m in "$@"; do
  if guard_human; then echo "the user plays: stop"; break; fi
  # shellcheck disable=SC2046
  "$PY" -m bridge.build nn-arena --own-ai net --deploy strategist $(army "$m") --map ksl_plains_01 --speed 3 \
    --timeout 1500 > "tmp/chain/$m.build.log" 2>&1 || { echo "build failed $m (tmp/chain/$m.build.log)"; continue; }
  for i in $(seq 1 "$n"); do
    if guard_human; then echo "the user plays: stop"; break 2; fi
    echo "=== $m $i $(date +%T)"
    if [ -n "$DRY" ]; then echo "dry: watch.ps1 -NoBuild -Strategist default -StrategistTask hold -Chain $CHAIN_ARGS"; continue; fi
    # shellcheck disable=SC2086
    powershell -NoProfile -ExecutionPolicy Bypass -File bridge/launcher/watch.ps1 -NoBuild -Strategist default \
      -StrategistTask hold -Chain $CHAIN_ARGS > "tmp/chain/${m}_$i.log" 2>&1
    code=$?
    run=$(ls -t build/nn-arena/runs 2>/dev/null | head -1)
    echo "--- $m $i exit $code $(date +%T) $run"
    [ -n "$run" ] && runs+=("$run")
  done
done
[ ${#runs[@]} -gt 0 ] && "$PY" -m game.battle.table "${runs[@]}"
echo "done $(date +%T)"
