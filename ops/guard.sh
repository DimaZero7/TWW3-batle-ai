#!/usr/bin/env bash
# Is the game free for us? The marks of state/ (not in git); the same rule as ops/guard.py and ops/guard.ps1.
#   state/HUMAN_PLAYING     the user plays: no game of ours starts
#   state/probe_game.lock   one runner holds the game; its text is the holder's token, exported as BAI_LOCK_HELD to
#                           its children (bridge/launcher/launch.ps1 lets the holder's own launches pass)
# usage (a runner):  . ops/guard.sh; guard_take "probe hill" || exit 1; trap guard_release EXIT
#                    guard_human && break        # between battles: stop when the user starts to play
GUARD_STATE="${GUARD_STATE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/state}"

guard_owner() { [ -e "$GUARD_STATE/probe_game.lock" ] && tr -d '\r' < "$GUARD_STATE/probe_game.lock" ; }

guard_human() { [ -e "$GUARD_STATE/HUMAN_PLAYING" ]; }

guard_busy() {   # prints why the game is busy; status 0 when busy
  if guard_human; then echo "the user plays (state/HUMAN_PLAYING)"; return 0; fi
  if [ -e "$GUARD_STATE/probe_game.lock" ] && [ "$(guard_owner)" != "${BAI_LOCK_HELD:-}" ]; then
    echo "the game is held (state/probe_game.lock: '$(guard_owner)')"; return 0; fi
  return 1
}

guard_take() {   # guard_take <label>: takes the lock (or rides on the caller's), exports BAI_LOCK_HELD
  local why; why="$(guard_busy)" && { echo "not starting: $why" >&2; return 1; }
  if [ -e "$GUARD_STATE/probe_game.lock" ]; then GUARD_TOKEN=""; return 0; fi
  mkdir -p "$GUARD_STATE"
  GUARD_TOKEN="$1 $(date +%Y-%m-%dT%H:%M:%S) pid $$"
  ( set -o noclobber; printf '%s\n' "$GUARD_TOKEN" > "$GUARD_STATE/probe_game.lock" ) 2>/dev/null \
    || { echo "not starting: the game is held (state/probe_game.lock: '$(guard_owner)')" >&2; return 1; }
  export BAI_LOCK_HELD="$GUARD_TOKEN"
}

guard_release() {
  if [ -n "${GUARD_TOKEN:-}" ] && [ "$(guard_owner)" = "$GUARD_TOKEN" ]; then rm -f -- "$GUARD_STATE/probe_game.lock"; fi
}
