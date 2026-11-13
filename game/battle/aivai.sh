#!/usr/bin/env bash
# The game's AI against the game's AI with stage 0 armies (game/battle/aivai.py): one battle a name, Normal, light
# graphics on the RTX (BAI_GRAPHICS=low), the game lock (ops/guard.sh), speed 10, 600 s.
# usage: bash game/battle/aivai.sh [f1 f2 f3 f4 k1 k2 k3 k4]      (default: all eight)
#        DRY=1 bash game/battle/aivai.sh k1                         (the builds only, no game)
# Logs: tmp/aivai/<name>.build.log, <name>.launch.log; runs: build/nn-arena/runs/<time>/; then
#        python -m game.battle.aivai game <runs>   (and sim in the container: bash ops/dock.sh game.battle.aivai sim)
cd "$(dirname "$0")/../.." || exit 1
. ops/guard.sh
. ops/python.sh
[ $# -ge 1 ] || set -- f1 f2 f3 f4 k1 k2 k3 k4
if [ -n "$DRY" ]; then why="$(guard_busy)" && { echo "not starting: $why" >&2; exit 1; }
else guard_take "aivai $*" || exit 1; trap guard_release EXIT; fi
export BAI_GRAPHICS="${BAI_GRAPHICS:-low}"
mkdir -p tmp/aivai
"$PY" -m game.battle.aivai make > tmp/aivai/make.log 2>&1 || { echo "make failed (tmp/aivai/make.log)"; exit 1; }
for name in "$@"; do
  if guard_human; then echo "the user plays: stop"; break; fi
  read -r map role < <("$PY" -c "import json;b=json.load(open('build/aivai/battles.json'))['$name'];print(b['map'],b['role'])")
  args=(--own-ai ai --arena-json "build/aivai/$name.json" --speed 10 --timeout 600)
  [ "$map" != "flat" ] && args+=(--map "$map")
  # the game's AI (side 2) attacks: ours defends (the default); it defends: ours attacks
  [ "$role" = "defend" ] && args+=(--own-role attack)
  echo "=== $name $map ai:$role $(date +%T)"
  "$PY" -m bridge.build nn-arena "${args[@]}" > "tmp/aivai/$name.build.log" 2>&1 \
    || { echo "build failed $name (tmp/aivai/$name.build.log)"; continue; }
  if [ -n "$DRY" ]; then echo "dry: built"; continue; fi
  powershell -NoProfile -ExecutionPolicy Bypass -File bridge/launcher/launch.ps1 -Target nn-arena \
    > "tmp/aivai/$name.launch.log" 2>&1
  echo "--- $name exit $? $(date +%T) $(ls -t build/nn-arena/runs 2>/dev/null | head -1)"
done
echo "done $(date +%T)"
