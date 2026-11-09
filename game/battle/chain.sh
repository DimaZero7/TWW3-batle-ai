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
cd "$(dirname "$0")/../.." || exit 1
. ops/guard.sh
. ops/python.sh
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
