#!/usr/bin/env bash
# Tier 3 security cadence — the recurring work that has no natural trigger.
#
#   cadence.sh check          list what is overdue; quiet when nothing is
#   cadence.sh list           every task with its interval and days remaining
#   cadence.sh done <id>      record a task as done today
#
# The tasks live in cadence.txt next to this script; the clock lives in the
# guards state directory. This is the difference between a checklist and a
# rule: a checklist in a document is read when you go looking for it, which is
# exactly when you did not need reminding.
#
# Exit: 0 nothing overdue, 1 something is.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS="$HERE/cadence.txt"
STATE="${XDG_CACHE_HOME:-$HOME/.cache}/roofline-guards/cadence"
mkdir -p "$STATE"

ylw() { printf '\033[33m%s\033[0m\n' "$*"; }
dim() { printf '\033[2m%s\033[0m\n' "$*"; }

each_task() { grep -vE '^[[:space:]]*(#|$)' "$TASKS"; }

days_since() { # id -> days since done, or -1 for never
  local f="$STATE/$1"
  [ -f "$f" ] || { echo -1; return; }
  local now mt; now=$(date +%s)
  mt=$(stat -f %m "$f" 2>/dev/null || stat -c %Y "$f" 2>/dev/null || echo 0)
  echo $(( (now - mt) / 86400 ))
}

cmd_check() {
  local out; out=""
  while IFS='|' read -r id days what; do
    id=$(echo "$id" | tr -d ' '); days=$(echo "$days" | tr -d ' ')
    what=$(echo "$what" | sed 's/^ *//; s/ *$//')
    local since; since=$(days_since "$id")
    if [ "$since" -lt 0 ]; then
      out="${out}  $id — never done (every ${days}d): $what\n"
    elif [ "$since" -ge "$days" ]; then
      out="${out}  $id — ${since}d ago, due every ${days}d: $what\n"
    fi
  done < <(each_task)
  [ -z "$out" ] && return 0
  ylw "security cadence overdue:"
  printf "%b" "$out"
  dim "  mark one done: scripts/cadence.sh done <id>"
  return 1
}

cmd_list() {
  while IFS='|' read -r id days what; do
    id=$(echo "$id" | tr -d ' '); days=$(echo "$days" | tr -d ' ')
    what=$(echo "$what" | sed 's/^ *//; s/ *$//')
    local since left
    since=$(days_since "$id")
    if [ "$since" -lt 0 ]; then left="never done"; else left="$(( days - since ))d left"; fi
    printf '%-14s %4sd  %-12s %s\n' "$id" "$days" "$left" "$what"
  done < <(each_task)
}

cmd_done() {
  local id="${1:-}"
  [ -z "$id" ] && { echo "usage: cadence.sh done <id>" >&2; exit 2; }
  each_task | cut -d'|' -f1 | tr -d ' ' | grep -qx "$id" || {
    echo "cadence: no task '$id'. Known:" >&2
    each_task | cut -d'|' -f1 | tr -d ' ' | sed 's/^/  /' >&2; exit 2; }
  : > "$STATE/$id"
  dim "$id: recorded as done $(date +%Y-%m-%d)"
}

case "${1:-check}" in
  check) cmd_check ;;
  list)  cmd_list ;;
  done)  shift; cmd_done "$@" ;;
  *) echo "usage: cadence.sh {check|list|done <id>}" >&2; exit 2 ;;
esac
