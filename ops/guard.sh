#!/usr/bin/env bash
# Is the game free for us? The marks of state/ (not in git); the same rule as ops/guard.py and ops/guard.ps1.
#   state/HUMAN_PLAYING     the user plays: no game of ours starts
#   state/probe_game.lock   one runner holds the game; its text is the holder's token, exported as BAI_LOCK_HELD to
#                           its children (bridge/launcher/launch.ps1 lets the holder's own launches pass); it holds
#                           every launcher slot too (its token is in probe_game.1.lock .. probe_game.3.lock)
#   state/probe_game.N.lock one runner holds launcher slot N = 1..3 (bridge/docs/launch/slots.md), token exported as
#                           BAI_SLOT_LOCK_HELD_N
# The protocol (ops/guard.py's text): exclusive create is the only way in; the whole game takes 1, 2, 3, whole in this
# order; a dead holder (token '... pid <Windows pid> id <8 hex>' with no such process, or an empty file older than 60 s)
# is removed only under '<lock>.break'. Nested takes in one shell ride on the outer one (released by the outermost).
# usage (a runner):  . ops/guard.sh; guard_take "probe hill" || exit 1; trap guard_release EXIT
#                    guard_take_slot 2 "lane a" || exit 1; trap 'guard_release_slot 2' EXIT   # launch.ps1 -Slot 2
#                    guard_human && break        # between battles: stop when the user starts to play
GUARD_STATE="${GUARD_STATE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/state}"
GUARD_WINPID="$(cat "/proc/$$/winpid" 2>/dev/null || echo "$$")"
GUARD_DEPTH="${GUARD_DEPTH:-0}"
declare -a GUARD_SLOT_TOKENS GUARD_SLOT_DEPTHS 2>/dev/null

guard_read() { [ -e "$1" ] || return 1; tr -d '\r' < "$1" 2>/dev/null || printf '<unreadable>'; }   # the token of a lock file

guard_owner() { [ -e "$GUARD_STATE/probe_game.lock" ] && tr -d '\r' < "$GUARD_STATE/probe_game.lock" ; }

guard_human() { [ -e "$GUARD_STATE/HUMAN_PLAYING" ]; }

guard_pid_alive() {   # status 0 when the Windows process <pid> runs; anything unclear counts as running
  if command -v tasklist.exe >/dev/null 2>&1; then
    local out; out="$(MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' tasklist.exe /FI "PID eq $1" /NH /FO CSV 2>/dev/null)" || return 0
    case "$out" in *"\",\"$1\","*) return 0 ;; esac
    [ -n "$out" ] || return 0
    return 1
  fi
  kill -0 "$1" 2>/dev/null
}

guard_dead() {   # guard_dead <file> <token>: the one rule for a dead holder
  local f="$1" t="$2" p
  if [ -z "$t" ]; then
    [ -e "$f" ] && [ $(( $(date +%s) - $(stat -c %Y "$f" 2>/dev/null || date +%s) )) -gt 60 ]; return; fi
  [[ "$t" =~ \ pid\ ([0-9]+)\ id\ [0-9a-f]{8}$ ]] || return 1
  p="${BASH_REMATCH[1]}"
  ! guard_pid_alive "$p"
}

guard_unlink() { local i; for i in $(seq 40); do rm -f -- "$1" 2>/dev/null; [ -e "$1" ] || return 0; sleep 0.05; done; return 1; }

guard_break_dead() {   # removes a dead lock under <file>.break; status 0 when the lock is gone
  local f="$1" t
  t="$(guard_read "$f")" || return 0
  guard_dead "$f" "$t" || return 1
  ( set -o noclobber; : > "$f.break" ) 2>/dev/null || return 1
  [ "$(guard_read "$f")" = "$t" ] && guard_unlink "$f"
  guard_unlink "$f.break"
  [ ! -e "$f" ]
}

guard_create() {   # guard_create <file> <token>: exclusive create, one retry after a dead lock was broken
  local a
  for a in 0 1; do
    if ( set -o noclobber; printf '%s\n' "$2" > "$1" ) 2>/dev/null; then return 0; fi
    [ "$a" = 0 ] && guard_break_dead "$1" && continue
    return 1
  done
  return 1
}

guard_drop() { [ -n "$2" ] && [ "$(guard_read "$1")" = "$2" ] && guard_unlink "$1"; return 0; }

guard_token() {
  local label; label="$(printf '%s' "$1" | LC_ALL=C tr -c ' -~' '?' | sed 's/^ *//; s/ *$//')"
  printf '%s %s pid %s id %04x%04x' "${label:-lock}" "$(date +%Y-%m-%dT%H:%M:%S)" "$GUARD_WINPID" "$RANDOM" "$RANDOM"
}

guard_busy() {   # prints why the game is busy; status 0 when busy
  if guard_human; then echo "the user plays (state/HUMAN_PLAYING)"; return 0; fi
  if [ -e "$GUARD_STATE/probe_game.lock" ] && ! guard_dead "$GUARD_STATE/probe_game.lock" "$(guard_read "$GUARD_STATE/probe_game.lock")"; then   # a dead holder's lock is no one's
    local o; o="$(guard_owner)"
    [ -n "$o" ] && [ "$o" = "${BAI_LOCK_HELD:-}" ] && return 1
    echo "the game is held (state/probe_game.lock: '$o')"; return 0; fi
  local n t; for n in 1 2 3; do
    t="$(guard_read "$GUARD_STATE/probe_game.$n.lock")" || continue
    guard_dead "$GUARD_STATE/probe_game.$n.lock" "$t" && continue        # a dead holder's slot lock is no one's
    echo "launcher slots hold the game (state/probe_game.$n.lock: '$t')"; return 0
  done
  return 1
}

guard_busy_slot() {   # guard_busy_slot N: why slot N is busy; status 0 when busy
  if guard_human; then echo "the user plays (state/HUMAN_PLAYING)"; return 0; fi
  if [ -e "$GUARD_STATE/probe_game.lock" ] && ! guard_dead "$GUARD_STATE/probe_game.lock" "$(guard_read "$GUARD_STATE/probe_game.lock")"; then   # a dead holder's lock is no one's
    local o; o="$(guard_owner)"
    [ -n "$o" ] && [ "$o" = "${BAI_LOCK_HELD:-}" ] && return 1      # the whole-game holder holds every slot
    echo "the game is held (state/probe_game.lock: '$o')"; return 0; fi
  local f="$GUARD_STATE/probe_game.$1.lock" v="BAI_SLOT_LOCK_HELD_$1" t
  if [ -e "$f" ]; then
    t="$(guard_read "$f")"
    [ -n "$t" ] && [ "$t" = "${!v:-}" ] && return 1
    guard_dead "$f" "$t" && return 1
    echo "slot $1 is held ($(basename "$f"): '$t')"; return 0; fi
  return 1
}

guard_busy_live() {   # guard_busy_live [N]: like guard_busy / guard_busy_slot N, but a dead holder's lock is free
  if guard_human; then echo "the user plays (state/HUMAN_PLAYING)"; return 0; fi
  local w="$GUARD_STATE/probe_game.lock" o f t n v
  if o="$(guard_read "$w")"; then
    [ -n "$o" ] && [ "$o" = "${BAI_LOCK_HELD:-}" ] && return 1
    guard_dead "$w" "$o" || { echo "the game is held (state/probe_game.lock: '$o')"; return 0; }
  fi
  for n in ${1:-1 2 3}; do
    f="$GUARD_STATE/probe_game.$n.lock"; v="BAI_SLOT_LOCK_HELD_$n"
    t="$(guard_read "$f")" || continue
    [ -n "${1:-}" ] && [ -n "$t" ] && [ "$t" = "${!v:-}" ] && return 1
    guard_dead "$f" "$t" && continue
    if [ -n "${1:-}" ]; then echo "slot $n is held (probe_game.$n.lock: '$t')"
    else echo "launcher slots hold the game (state/probe_game.$n.lock: '$t')"; fi
    return 0
  done
  return 1
}

guard_take() {   # guard_take <label>: takes the lock (or rides on the caller's), exports BAI_LOCK_HELD
  local o; o="$(guard_owner)"
  if [ -n "$o" ] && [ "$o" = "${BAI_LOCK_HELD:-}" ] && ! guard_human; then GUARD_DEPTH=$((GUARD_DEPTH + 1)); return 0; fi   # the caller's: ride on it
  local why; why="$(guard_busy_live)" && { echo "not starting: $why" >&2; return 1; }
  mkdir -p "$GUARD_STATE"
  local tok f taken=() t; tok="$(guard_token "$1")"
  for f in "$GUARD_STATE/probe_game.1.lock" "$GUARD_STATE/probe_game.2.lock" "$GUARD_STATE/probe_game.3.lock" "$GUARD_STATE/probe_game.lock"; do
    if ! guard_create "$f" "$tok"; then
      for (( t = ${#taken[@]} - 1; t >= 0; t-- )); do guard_drop "${taken[$t]}" "$tok"; done
      if [ "$f" = "$GUARD_STATE/probe_game.lock" ]; then echo "not starting: the game is held (state/probe_game.lock: '$(guard_owner)')" >&2
      else echo "not starting: launcher slots hold the game (state/$(basename "$f"): '$(guard_read "$f")')" >&2; fi
      return 1
    fi
    taken+=("$f")
  done
  if guard_human; then
    for (( t = ${#taken[@]} - 1; t >= 0; t-- )); do guard_drop "${taken[$t]}" "$tok"; done
    echo "not starting: the user plays (state/HUMAN_PLAYING)" >&2; return 1; fi
  GUARD_TOKEN="$tok"; GUARD_DEPTH=0
  export BAI_LOCK_HELD="$GUARD_TOKEN"
}

guard_release() {
  if [ "${GUARD_DEPTH:-0}" -gt 0 ]; then GUARD_DEPTH=$((GUARD_DEPTH - 1)); return 0; fi   # a nested take: the outer one releases
  [ -n "${GUARD_TOKEN:-}" ] || return 0
  local f; for f in probe_game.lock probe_game.3.lock probe_game.2.lock probe_game.1.lock; do guard_drop "$GUARD_STATE/$f" "$GUARD_TOKEN"; done
  [ "${BAI_LOCK_HELD:-}" = "$GUARD_TOKEN" ] && unset BAI_LOCK_HELD
  GUARD_TOKEN=""
  return 0
}

guard_take_slot() {   # guard_take_slot N <label>: takes slot N (or rides on the caller's whole-game lock or own slot lock)
  local n="$1" why; case "$n" in 1|2|3) ;; *) echo "slot must be 1, 2 or 3" >&2; return 1 ;; esac
  why="$(guard_busy_live "$n")" && { echo "not starting: $why" >&2; return 1; }
  GUARD_SLOT="$n"
  local v="BAI_SLOT_LOCK_HELD_$n" o s; o="$(guard_owner)"; s="$(guard_read "$GUARD_STATE/probe_game.$n.lock")"
  if { [ -n "$o" ] && [ "$o" = "${BAI_LOCK_HELD:-}" ]; } || { [ -n "$s" ] && [ "$s" = "${!v:-}" ]; }; then  # the caller's: ride on it
    GUARD_SLOT_DEPTHS[$n]=$(( ${GUARD_SLOT_DEPTHS[$n]:-0} + 1 )); return 0; fi
  mkdir -p "$GUARD_STATE"
  local tok f="$GUARD_STATE/probe_game.$n.lock" w="$GUARD_STATE/probe_game.lock"; tok="$(guard_token "$2")"
  guard_create "$f" "$tok" || { echo "not starting: slot $n is held (state/probe_game.$n.lock: '$(guard_read "$f")')" >&2; return 1; }
  if [ -e "$w" ] && ! guard_break_dead "$w"; then                     # an older runner's whole-game lock
    guard_drop "$f" "$tok"; echo "not starting: the game is held (state/probe_game.lock: '$(guard_owner)')" >&2; return 1; fi
  if guard_human; then guard_drop "$f" "$tok"; echo "not starting: the user plays (state/HUMAN_PLAYING)" >&2; return 1; fi
  GUARD_SLOT_TOKENS[$n]="$tok"; GUARD_SLOT_DEPTHS[$n]=0
  export "BAI_SLOT_LOCK_HELD_$n=$tok"
}

guard_release_slot() {   # guard_release_slot [N] (default: the slot of the last guard_take_slot)
  local n="${1:-${GUARD_SLOT:-x}}"; case "$n" in 1|2|3) ;; *) return 0 ;; esac
  if [ "${GUARD_SLOT_DEPTHS[$n]:-0}" -gt 0 ]; then GUARD_SLOT_DEPTHS[$n]=$(( ${GUARD_SLOT_DEPTHS[$n]} - 1 )); return 0; fi
  local tok="${GUARD_SLOT_TOKENS[$n]:-}" v="BAI_SLOT_LOCK_HELD_$n"
  [ -n "$tok" ] || return 0
  guard_drop "$GUARD_STATE/probe_game.$n.lock" "$tok"
  [ "${!v:-}" = "$tok" ] && unset "$v"
  GUARD_SLOT_TOKENS[$n]=""
  return 0
}
