#!/usr/bin/env bash
# Dependency advisory audit across the repos on this machine.
#
# The March 2026 incident came from a package registry one over from these: a
# trojanized VS Code extension. Nothing here would have caught that one — it was
# malice, not a published CVE — so this is not a defence against a repeat. It
# covers the adjacent, duller exposure that is free to watch: dependencies with
# advisories already written about them.
#
#   dep-audit.sh run   [repo...]   do the audit (network); record the result
#   dep-audit.sh check [repo...]   read the record only; quiet unless owed
#   dep-audit.sh status            one line per repo, last result and age
#   dep-audit.sh ack   [repo...]   accept the current findings as known
#
# `check` is the cadence half. It asks whether the audit is STALE, FAILED, or
# last came back dirty — never whether packages changed. That keeps it offline,
# instant, and silent on the ordinary day. A check that talks every time is a
# check that gets muted.
#
# `ack` exists for the same reason the daily security check has --accept. These
# repos carry 11 standing transitive advisories in the Expo build toolchain that
# will clear on an upstream bump and not before. A guard that reports them at
# every session start becomes wallpaper, and the finding that actually matters
# arrives in the same grey block everyone has stopped reading. Acknowledging
# records the exact counts; if they CHANGE, the guard speaks up again.
#
# An audit that could not run is recorded as FAILED, never as clean. Reporting
# "0 vulnerabilities" for a scan that never happened is the one output worse
# than no audit at all, because it is indistinguishable from good news.
#
# Exit: run   -> 0 all audits ran, 2 one or more could not run.
#       check -> 0 nothing owed, 1 owed.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="${XDG_CACHE_HOME:-$HOME/.cache}/roofline-guards"
STALE_DAYS=${ROOFLINE_AUDIT_STALE_DAYS:-30}     # B5 cadence: monthly
mkdir -p "$STATE"

red() { printf '\033[31m%s\033[0m\n' "$*"; }
ylw() { printf '\033[33m%s\033[0m\n' "$*"; }
dim() { printf '\033[2m%s\033[0m\n' "$*"; }

# Which repos. Explicit list beats a filesystem sweep: a guard that discovers
# its own scope silently changes what it covers when a directory appears, and
# the day it stops covering something is the day nobody notices.
default_repos() {
  if [ -n "${ROOFLINE_GUARD_REPOS:-}" ]; then
    printf '%s\n' "$ROOFLINE_GUARD_REPOS" | tr ':' '\n'
  elif [ -f "$HERE/guarded-repos.txt" ]; then
    grep -vE '^[[:space:]]*(#|$)' "$HERE/guarded-repos.txt"
  else
    git -C "$HERE" rev-parse --show-toplevel 2>/dev/null
  fi
}

key_for() { printf '%s' "$1" | sed 's#/#_#g; s#^_##'; }

# --- node -------------------------------------------------------------------
# One lockfile format per tool, and they are not interchangeable: `npm audit`
# returns nothing at all in a pnpm repo, which is how warDrobe first came back
# "clean" while carrying two high advisories.
node_audit() { # repo -> json on stdout, or nothing
  local repo="$1"
  if   [ -f "$repo/pnpm-lock.yaml" ] && command -v pnpm >/dev/null 2>&1; then
    (cd "$repo" && pnpm audit --json 2>/dev/null)
  elif [ -f "$repo/yarn.lock" ] && command -v yarn >/dev/null 2>&1; then
    (cd "$repo" && yarn npm audit --json 2>/dev/null)
  elif [ -f "$repo/package-lock.json" ]; then
    (cd "$repo" && npm audit --json 2>/dev/null)
  else
    (cd "$repo" && npm audit --json 2>/dev/null)
  fi
}

node_tool() { # repo -> the tool name, for the record
  local repo="$1"
  if   [ -f "$repo/pnpm-lock.yaml" ]; then echo pnpm
  elif [ -f "$repo/yarn.lock" ];      then echo yarn
  else                                     echo npm; fi
}

# --- python -----------------------------------------------------------------
# OSV's batch endpoint rather than pip-audit: pip-audit builds a throwaway
# virtualenv to resolve `-r`, and ensurepip aborts under uv's standalone
# Python on this machine. It is not needed here anyway — requirements.txt is
# pinned exactly (that was last session's supply-chain work), and an exact
# version is precisely what OSV wants. The pinning pays for itself twice.
pypi_audit() { # repo -> json on stdout, or nothing
  local repo="$1" q
  [ -f "$repo/requirements.txt" ] || return 1
  q=$(jq -n --rawfile r "$repo/requirements.txt" '
        { queries: ($r | split("\n")
            | map(select(test("^[A-Za-z0-9_.-]+==")))
            | map(sub("\\s*#.*$";"") | split("==")
                  | {package:{name:.[0], ecosystem:"PyPI"}, version:.[1]})) }' 2>/dev/null) || return 1
  [ "$(printf '%s' "$q" | jq '.queries|length')" -gt 0 ] || return 1
  curl -sf --max-time 60 -X POST -H 'Content-Type: application/json' \
       -d "$q" https://api.osv.dev/v1/querybatch 2>/dev/null
}

audit_one() { # repo -> writes state json; echoes "status crit high total"
  local repo="$1" key eco="none" tool="" status="none" crit=0 high=0 total=0
  local sev_known=true note="" ids="[]"
  key="$(key_for "$repo")"

  if [ -f "$repo/package.json" ]; then
    eco="node"; tool="$(node_tool "$repo")"
    local j; j=$(node_audit "$repo")
    if printf '%s' "$j" | jq -e '.metadata.vulnerabilities' >/dev/null 2>&1; then
      status=ok
      crit=$(printf '%s' "$j" | jq -r '.metadata.vulnerabilities.critical // 0')
      high=$(printf '%s' "$j" | jq -r '.metadata.vulnerabilities.high // 0')
      total=$(printf '%s' "$j" | jq -r '[.metadata.vulnerabilities[]] | add // 0')
    else
      status=failed
      note="$tool audit returned no parsable report (offline, or no lockfile)"
    fi

  elif [ -f "$repo/requirements.txt" ]; then
    eco="pypi"; tool="osv"
    local j; j=$(pypi_audit "$repo")
    if printf '%s' "$j" | jq -e '.results' >/dev/null 2>&1; then
      status=ok; sev_known=false
      total=$(printf '%s' "$j" | jq '[.results[].vulns // [] | length] | add // 0')
      ids=$(printf '%s' "$j" | jq -c '[.results[].vulns // [] | .[].id] | unique')
      [ "$total" -gt 0 ] && note="OSV querybatch carries no severity; all $total need reading"
    else
      status=failed
      note="OSV query failed (offline?) or requirements.txt has no exact pins"
    fi

  else
    status=none; note="no package.json or requirements.txt"
  fi

  # An acknowledgement has to survive the next audit, or acknowledging anything
  # is pointless. It is carried forward verbatim; `check` compares it against
  # the FRESH counts, so a re-run that finds something new un-acknowledges
  # itself without anyone having to remember to clear it.
  local prev_ack="" prev_at=""
  if [ -f "$STATE/$key.json" ]; then
    prev_ack=$(jq -r '.acked // ""'    "$STATE/$key.json" 2>/dev/null)
    prev_at=$( jq -r '.acked_at // ""' "$STATE/$key.json" 2>/dev/null)
  fi

  jq -n --arg r "$repo" --arg e "$eco" --arg tl "$tool" --arg st "$status" \
        --arg n "$note" --argjson c "$crit" --argjson h "$high" --argjson t "$total" \
        --argjson sk "$sev_known" --argjson ids "$ids" \
        --arg ack "$prev_ack" --arg ackat "$prev_at" \
        --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{repo:$r, ecosystem:$e, tool:$tl, status:$st, critical:$c, high:$h,
      total:$t, severity_known:$sk, ids:$ids, note:$n, ran:$ts,
      acked:$ack, acked_at:$ackat}' \
    > "$STATE/$key.json.new" && mv "$STATE/$key.json.new" "$STATE/$key.json"
  echo "$status $crit $high $total"
}

# A finding signature, so an acknowledgement covers exactly what was seen and
# nothing more. Any change in the numbers invalidates it.
sig_of() { jq -r '"\(.critical):\(.high):\(.total)"' "$1" 2>/dev/null || echo "?"; }

cmd_ack() {
  local repo key f sig n=0
  for repo in "$@"; do
    key="$(key_for "$repo")"; f="$STATE/$key.json"
    [ -f "$f" ] || { ylw "  $repo: never audited — nothing to acknowledge"; continue; }
    sig="$(sig_of "$f")"
    jq --arg s "$sig" --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
       '.acked = $s | .acked_at = $ts' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
    dim "  $repo: acknowledged at $sig (crit:high:total)"
    n=$((n+1))
  done
  [ "$n" -gt 0 ] && dim "  A change in those numbers un-acknowledges it automatically."
  return 0
}

age_days() {
  local f="$1"; [ -f "$f" ] || { echo 99999; return; }
  local now mt; now=$(date +%s)
  mt=$(stat -f %m "$f" 2>/dev/null || stat -c %Y "$f" 2>/dev/null || echo 0)
  echo $(( (now - mt) / 86400 ))
}

cmd_run() {
  local broke=0 repo st c h t f
  for repo in "$@"; do
    [ -d "$repo" ] || { ylw "audit: $repo does not exist — skipped"; continue; }
    printf '\033[2m>>\033[0m %s\n' "$repo"
    read -r st c h t < <(audit_one "$repo")
    f="$STATE/$(key_for "$repo").json"
    local note tool; note=$(jq -r '.note' "$f"); tool=$(jq -r '.tool' "$f")
    case "$st" in
      ok)
        if [ "$c" -gt 0 ] || [ "$h" -gt 0 ]; then
          red "   $c critical, $h high, $t total   ($tool)"
        elif [ "$t" -gt 0 ]; then
          ylw "   $t advisories, none high or critical   ($tool)"
        else
          dim "   no advisories   ($tool)"
        fi
        [ -n "$note" ] && dim "   $note" ;;
      failed)
        red "   AUDIT DID NOT RUN — $note"; broke=1 ;;
      none)
        dim "   $note — nothing to audit" ;;
    esac
  done
  return $broke
}

cmd_check() {
  local repo key f age c h t st sk
  local msgs; msgs=()
  for repo in "$@"; do
    [ -d "$repo" ] || continue
    key="$(key_for "$repo")"; f="$STATE/$key.json"
    if [ ! -f "$f" ]; then msgs+=("$repo: never audited"); continue; fi
    age=$(age_days "$f")
    st=$(jq -r '.status' "$f")
    case "$st" in
      failed) msgs+=("$repo: last audit FAILED to run — $(jq -r '.note' "$f")"); continue ;;
      none)   continue ;;
    esac
    if [ "$age" -ge "$STALE_DAYS" ]; then
      msgs+=("$repo: last audited ${age}d ago (cadence is ${STALE_DAYS}d)"); continue
    fi
    c=$(jq -r '.critical' "$f"); h=$(jq -r '.high' "$f")
    t=$(jq -r '.total' "$f");    sk=$(jq -r '.severity_known' "$f")
    # Acknowledged and unchanged since: stay quiet. Changed: fall through.
    [ "$(jq -r '.acked // ""' "$f")" = "$(sig_of "$f")" ] && continue
    if [ "$sk" = true ] && { [ "$c" -gt 0 ] || [ "$h" -gt 0 ]; }; then
      msgs+=("$repo: $c critical / $h high open (audited ${age}d ago) — accept with: scripts/dep-audit.sh ack $repo")
    elif [ "$sk" = false ] && [ "$t" -gt 0 ]; then
      msgs+=("$repo: $t advisories, severity unrated (audited ${age}d ago) — accept with: scripts/dep-audit.sh ack $repo")
    fi
  done
  [ ${#msgs[@]} -eq 0 ] && return 0
  ylw "dependency audit owed:"
  printf '  %s\n' "${msgs[@]}"
  printf '  run: make audit\n'
  return 1
}

cmd_status() {
  local repo key f
  for repo in "$@"; do
    key="$(key_for "$repo")"; f="$STATE/$key.json"
    if [ -f "$f" ]; then
      jq -r '"\(.repo)\t\(.tool // "-")\t\(.status)\tcrit=\(.critical) high=\(.high) total=\(.total)\t\(.ran)"' "$f"
    else
      printf '%s\t-\tnever\t-\t-\n' "$repo"
    fi
  done | column -t -s $'\t'
}

sub="${1:-check}"; shift || true
# bash 3.2 (what macOS ships) has neither mapfile nor readarray.
repos=()
if [ $# -eq 0 ]; then
  while IFS= read -r _r; do [ -n "$_r" ] && repos+=("$_r"); done < <(default_repos)
else
  repos=("$@")
fi
[ ${#repos[@]} -eq 0 ] && { echo "dep-audit: no repos configured" >&2; exit 2; }

case "$sub" in
  run)    cmd_run    "${repos[@]}"; exit $(( $? ? 2 : 0 )) ;;
  check)  cmd_check  "${repos[@]}" ;;
  status) cmd_status "${repos[@]}" ;;
  ack)    cmd_ack    "${repos[@]}" ;;
  *) echo "usage: dep-audit.sh {run|check|status|ack} [repo...]" >&2; exit 2 ;;
esac
