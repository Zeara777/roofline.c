#!/usr/bin/env bash
# Session-summary rule for roofline.c.
#
# The global CLAUDE.md requires a summary per session at
#   ~/.claude/session-summaries/<workspace>/YYYY-MM-DD-<slug>.md
# deliberately OUTSIDE the repo, so a summary can never land in a commit or PR.
# This script enforces that rule without moving it.
#
#   session-summary.sh check        exit 0 always; prints a reminder if one is due
#   session-summary.sh new <slug>   scaffold today's file with the required headings
#   session-summary.sh path         print this workspace's summary directory
#   session-summary.sh list         list existing summaries, newest first
#
# `check` stays quiet unless a summary is genuinely owed: it says nothing when
# today's file already exists, and nothing when the session has produced no
# commits and no working-tree changes — a read-only session has nothing to
# summarize and should not be nagged.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# `pwd` echoes back whatever case the caller used, and this repo is reachable as
# both CPT and cpt on a case-insensitive filesystem. Taking basename directly
# would write to .../session-summaries/cpt here and .../CPT on the Linux box —
# two folders for one project. Ask the parent directory for the real spelling.
_raw="$(basename "$REPO")"
WORKSPACE="${ROOFLINE_WORKSPACE:-$(cd "$REPO/.." && ls -1 2>/dev/null | grep -ix -- "$_raw" | head -1)}"
WORKSPACE="${WORKSPACE:-$_raw}"
DIR="$HOME/.claude/session-summaries/$WORKSPACE"
TODAY="$(date +%Y-%m-%d)"
THROTTLE_MIN=${ROOFLINE_SUMMARY_THROTTLE_MIN:-30}

today_files() { ls "$DIR/$TODAY"-*.md 2>/dev/null; }

work_done() {
  git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1 || return 1
  [ -n "$(git -C "$REPO" log --since=midnight --oneline 2>/dev/null)" ] && return 0
  [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] && return 0
  return 1
}

cmd_check() {
  # The hook feeds us the event JSON; session_id keys the throttle so a
  # reminder repeats at most once per THROTTLE_MIN per session.
  local payload session stamp
  payload="$(cat 2>/dev/null || true)"
  session="$(printf '%s' "$payload" | jq -r '.session_id // "nosession"' 2>/dev/null || echo nosession)"
  stamp="${TMPDIR:-/tmp}/roofline-summary-$session"

  if [ -n "$(today_files)" ]; then rm -f "$stamp"; exit 0; fi
  work_done || exit 0
  if [ -f "$stamp" ] && [ -z "$(find "$stamp" -mmin "+$THROTTLE_MIN" 2>/dev/null)" ]; then
    exit 0
  fi
  : > "$stamp"

  local latest msg
  latest="$(ls -t "$DIR"/*.md 2>/dev/null | head -1)"
  msg="No session summary for $TODAY yet. The rule: write one to $DIR/$TODAY-<slug>.md"
  msg="$msg with Goal / Done / Found / Decisions / Next / Not verified."
  [ -n "$latest" ] && msg="$msg Newest existing: $(basename "$latest")."
  msg="$msg  (scaffold: make summary SLUG=<slug>)"

  jq -n --arg m "$msg" \
    '{systemMessage:$m,
      hookSpecificOutput:{hookEventName:"Stop", additionalContext:$m}}'
  exit 0
}

cmd_new() {
  local slug="${1:-}"
  [ -z "$slug" ] && { echo "usage: session-summary.sh new <short-slug>" >&2; exit 2; }
  mkdir -p "$DIR"
  local f="$DIR/$TODAY-$slug.md"
  if [ -e "$f" ]; then echo "$f (exists, not overwritten)"; exit 0; fi
  cat > "$f" <<TEMPLATE
# $TODAY — $slug

## Goal
<what the session set out to do, in the user's words where possible>

## Done
<what shipped: issue and PR numbers, branch names, commit SHAs>

## Found
<anything measured or discovered that changes the plan, with the numbers>

## Decisions
<choices the user made, and why>

## Next
<the ordered next steps, so the next session can start from "resume">

## Not verified
<anything asserted but not actually run or measured>
TEMPLATE
  echo "$f"
}

case "${1:-check}" in
  check) cmd_check ;;
  new)   shift; cmd_new "$@" ;;
  path)  echo "$DIR" ;;
  list)  ls -t "$DIR"/*.md 2>/dev/null || echo "(none in $DIR)" ;;
  *)     echo "usage: session-summary.sh {check|new <slug>|path|list}" >&2; exit 2 ;;
esac
