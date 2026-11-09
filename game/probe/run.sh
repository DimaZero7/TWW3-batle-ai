#!/usr/bin/env bash
# Mechanics probes in the game: the plans in turn (game/probe/charge_probe.py PLANS), one battle each, memreader Plus
# rows (--mem; the sites: bridge/config/memreader.json), light graphics on the RTX (BAI_GRAPHICS=low), the game lock.
# usage: bash game/probe/run.sh <plan> [<plan> ...]          (BATTLE=2 for a plan's second battle; NOMEM=1 without --mem)
#        DRY=1 bash game/probe/run.sh <plan>                 (the checks and the builds only: no game)
# Logs: tmp/probe/<plan>.build.log, <plan>.launch.log; runs: build/charge-probe/runs/<time>/.
# Then: python -m game.probe.charge_probe report <runs> [--sim]   (or unit_probe for the new units' plans)
cd "$(dirname "$0")/../.." || exit 1
. ops/guard.sh
[ $# -ge 1 ] || { echo "usage: bash game/probe/run.sh <plan> [<plan> ...]" >&2; exit 1; }
. ops/python.sh
if [ -n "$DRY" ]; then why="$(guard_busy)" && { echo "not starting: $why" >&2; exit 1; }
else guard_take "probe $*" || exit 1; trap guard_release EXIT; fi
export BAI_GRAPHICS="${BAI_GRAPHICS:-low}"
mkdir -p tmp/probe
mem=(--mem); [ -n "$NOMEM" ] && mem=()
for plan in "$@"; do
  if guard_human; then echo "the user plays: stop"; break; fi
  echo "=== $plan $(date +%T)"
  "$PY" -m bridge.build charge-probe --probe-plan "$plan" --probe-battle "${BATTLE:-1}" "${mem[@]}" --deadline 900 \
    > "tmp/probe/$plan.build.log" 2>&1 || { echo "build failed $plan (tmp/probe/$plan.build.log)"; continue; }
  if [ -n "$DRY" ]; then echo "dry: built (tmp/probe/$plan.build.log); launch.ps1 not run"; continue; fi
  powershell -NoProfile -ExecutionPolicy Bypass -File bridge/launcher/launch.ps1 -Target charge-probe \
    > "tmp/probe/$plan.launch.log" 2>&1
  code=$?
  echo "--- $plan exit $code $(date +%T) $(ls -t build/charge-probe/runs 2>/dev/null | head -1)"
done
echo "done $(date +%T)"
