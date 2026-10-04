#!/bin/bash
# Wait for a condition WITH A TIMEOUT (docs/en/training/workflow.md). Replaces hand-written
# `until grep ...; do sleep 60; done` loops, which ran for hours when the pattern never came
# (pytest prints FAILED in capitals: `grep "passed|failed"` never matched).
#
#   bash tools/ops/wait.sh [-t SECONDS] [-i POLL] FILE REGEX    # a line of FILE matches REGEX (grep -Ei: case-insensitive)
#   bash tools/ops/wait.sh [-t SECONDS] container NAME          # no running docker container NAME
#   bash tools/ops/wait.sh [-t SECONDS] gone PATH               # PATH no longer exists (e.g. build/gpu-train.lock)
#
# Exit 0 when the condition holds (the matching line printed), 2 on timeout (the file's last lines
# printed), 1 on a bad call. Defaults: timeout 3600 s, poll 15 s.
timeout=3600
poll=15
while getopts "t:i:" opt; do
  case "$opt" in
    t) timeout="$OPTARG" ;;
    i) poll="$OPTARG" ;;
    *) echo "usage: wait.sh [-t SECONDS] [-i POLL] FILE REGEX | container NAME | gone PATH" >&2; exit 1 ;;
  esac
done
shift $((OPTIND - 1))
[ $# -eq 2 ] || { echo "usage: wait.sh [-t SECONDS] [-i POLL] FILE REGEX | container NAME | gone PATH" >&2; exit 1; }
kind="$1"; arg="$2"
start=$(date +%s)
while :; do
  case "$kind" in
    container)
      if [ -z "$(docker ps --filter "name=^${arg}$" --format '{{.Names}}' 2>/dev/null)" ]; then
        echo "container $arg is not running"; exit 0; fi ;;
    gone)
      if [ ! -e "$arg" ]; then echo "$arg is gone"; exit 0; fi ;;
    *)
      if [ -f "$kind" ]; then
        line=$(grep -Ei -m1 -- "$arg" "$kind" 2>/dev/null) && { echo "$line"; exit 0; }
      fi ;;
  esac
  now=$(date +%s)
  if [ $((now - start)) -ge "$timeout" ]; then
    echo "wait.sh: timeout after ${timeout} s ($kind $arg)" >&2
    [ -f "$kind" ] && tail -n 5 -- "$kind" >&2
    exit 2
  fi
  sleep "$poll"
done
