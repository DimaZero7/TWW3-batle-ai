#!/bin/bash
# inside agent-adjmelee: exam replays (network / ai_like) + the 72-battle check per variant -> /adj/*_TAG.*
cd /repo || exit 1
run() {
  tag=$1; shift
  sets=(); sets72=()
  for kv in "$@"; do sets+=(--set "$kv"); sets72+=("--set=$kv"); done
  t0=$(date +%s)
  python /work/replay.py build/compare/itS24.json /adj/units_$tag.pkl --workers 8 "${sets[@]}" 2>&1 | tail -1
  python /work/replay.py build/compare/itS24ai.json /adj/ai_$tag.pkl --workers 8 "${sets[@]}" 2>&1 | tail -1
  [ -n "$R72" ] && python /work/r72.py /adj/r72_$tag.json 4 8 "${sets72[@]}" > /adj/r72_$tag.txt 2>&1
  echo "DONE $tag $(( $(date +%s) - t0 )) s"
}
for v in "$@"; do
  case $v in
    base) run base ;;
    full) run full contact.rect_gap=true contact.push_keep_facing=1 contact.square_front=1 melee.frontage_edge_m=0.0 ;;
    norect) run norect contact.push_keep_facing=1 contact.square_front=1 melee.frontage_edge_m=0.0 ;;
    rect) run rect contact.rect_gap=true contact.push_keep_facing=1 ;;
    sq) run sq contact.rect_gap=true contact.push_keep_facing=1 contact.square_front=1 ;;
    edge25) run edge25 contact.rect_gap=true contact.push_keep_facing=1 contact.square_front=1 melee.frontage_edge_m=2.5 ;;
  esac
done
