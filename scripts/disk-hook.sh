#!/usr/bin/env bash
# SessionStart wrapper around check-disk.sh, emitting hook JSON.
#
# OK    -> context only, nothing shown to the user
# WARN  -> shown to the user and given to the model
# FAIL  -> shown to the user and given to the model
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR/.." || exit 0        # measure the project's filesystem, not the caller's

out=$("$DIR/check-disk.sh" 2>&1); code=$?
clean=$(printf '%s' "$out" | sed $'s/\x1b\\[[0-9;]*m//g')

if [ "$code" -ne 0 ] || printf '%s' "$clean" | grep -q 'DISK WARN'; then
  jq -n --arg m "$clean" \
    '{systemMessage:$m,
      hookSpecificOutput:{hookEventName:"SessionStart",
                          additionalContext:("roofline.c disk guard: " + $m)}}'
else
  jq -n --arg m "$clean" \
    '{suppressOutput:true,
      hookSpecificOutput:{hookEventName:"SessionStart",
                          additionalContext:("roofline.c disk guard: " + $m)}}'
fi
