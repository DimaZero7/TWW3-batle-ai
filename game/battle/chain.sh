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
cd "$(dirname "$0")/../.." || exit 1
. ops/guard.sh
. ops/python.sh
if [ "$1" = "tactic" ]; then
  b="${2:-f1}"
  W=weights/tactic/attack1/final.pt
  [ -f "$W" ] || { echo "no $W" >&2; exit 1; }
  if [ -n "$DRY" ]; then why="$(guard_busy)" && { echo "not starting: $why" >&2; exit 1; }
  else guard_take "chain tactic $b" || exit 1; trap guard_release EXIT; fi
  export BAI_GRAPHICS="${BAI_GRAPHICS:-low}"
  mkdir -p tmp/chain
  "$PY" -m game.battle.aivai make > tmp/chain/tactic_make.log 2>&1 || { echo "make failed (tmp/chain/tactic_make.log)"; exit 1; }
  read -r map role men < <("$PY" -c "import json;b=json.load(open('build/aivai/battles.json'))['$b'];print(b['map'],b['role'],sum(u['men'] for s in b['units'].values() for u in s))")
  men="${men//[!0-9]/}"
  [ "$men" -gt 5000 ] && export BAI_TICK_MS=100
  args=(--own-ai net --arena-json "build/aivai/$b.json" --speed 3 --timeout 900)
  [ "$map" != "flat" ] && args+=(--map "$map")
  [ "$role" = "defend" ] && args+=(--own-role attack)
  echo "=== tactic $b $map ai:$role men $men tick_ms ${BAI_TICK_MS:-engine} $(date +%T)"
  "$PY" -m bridge.build nn-arena "${args[@]}" > "tmp/chain/tactic_$b.build.log" 2>&1     || { echo "build failed (tmp/chain/tactic_$b.build.log)"; exit 1; }
  if [ -n "$DRY" ]; then echo "dry: built; watch.ps1 -NoBuild -Chain -ChainTactic $W"; exit 0; fi
  guard_human && { echo "the user plays: stop"; exit 1; }
  powershell -NoProfile -ExecutionPolicy Bypass -File bridge/launcher/watch.ps1 -NoBuild -Chain -ChainTactic "$W"     > "tmp/chain/tactic_$b.log" 2>&1
  code=$?
  run=$(ls -t build/nn-arena/runs 2>/dev/null | head -1)
  echo "--- tactic $b exit $code $(date +%T) $run; read: bash ops/dock.sh game.battle.tactic_read build/nn-arena/runs/$run"
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
