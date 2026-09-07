#!/usr/bin/env bash
# SessionStart wrapper around the security guards, emitting hook JSON.
#
# It reports two things and nothing else:
#   - a repo in guarded-repos.txt with no commit-time secret hook installed;
#   - a dependency audit that is stale, never run, or last came back dirty.
#
# Both are offline reads of recorded state. No network, no scanning: this runs
# at the start of every session and anything slower would be turned off. The
# scanning itself is `make scan` and `make audit`, run deliberately.
#
# Silence is the design. The disk guard next to it stays quiet when there is
# space; this stays quiet when nothing is owed. A session that opens with two
# green banners teaches you to skim past the red one.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

strip() { sed $'s/\x1b\\[[0-9;]*m//g'; }

owed=""
if ! guards=$("$DIR/install-guards.sh" --check 2>&1 | strip); then
  unguarded=$(printf '%s\n' "$guards" | grep -E 'UNGUARDED|foreign' | sed 's/^ *//')
  [ -n "$unguarded" ] && owed="${owed}secret guard not installed:\n$(printf '%s' "$unguarded" | sed 's/^/  /')\n  fix: make guards\n"
fi
if ! audit=$("$DIR/dep-audit.sh" check 2>&1 | strip); then
  owed="${owed}${audit}\n"
fi

# The cadence list is long by nature and mostly long-horizon. Show two items and
# a count: the point of this block is that you notice it, and a nine-line wall
# at every session start is read exactly as carefully as no block at all.
if ! cad=$("$DIR/cadence.sh" check 2>&1 | strip); then
  body=$(printf '%s\n' "$cad" | sed -n '2,$p' | grep -v 'mark one done')
  n=$(printf '%s\n' "$body" | grep -c '^  ')
  shown=$(printf '%s\n' "$body" | head -2)
  owed="${owed}security cadence overdue:\n${shown}\n"
  [ "$n" -gt 2 ] && owed="${owed}  ...and $((n - 2)) more — scripts/cadence.sh list\n"
fi

if [ -z "$owed" ]; then
  jq -n '{suppressOutput:true,
          hookSpecificOutput:{hookEventName:"SessionStart",
            additionalContext:"roofline.c security guards: secret hooks installed, dependency audit current."}}'
  exit 0
fi

msg=$(printf "roofline.c security guards — action owed:\n%b" "$owed")
jq -n --arg m "$msg" \
  '{systemMessage:$m,
    hookSpecificOutput:{hookEventName:"SessionStart", additionalContext:$m}}'
exit 0
