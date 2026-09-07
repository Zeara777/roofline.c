#!/usr/bin/env bash
# Secret scanner for roofline.c and the other repos on this machine.
#
# The March 2026 incident arrived through a trojanized editor extension, which
# means the credentials on this machine were readable by an attacker for months.
# The lesson that generalizes is not "scan for that extension" — it is that a
# secret which never enters a repo cannot leak from one. This blocks at commit,
# the last cheap moment; after a push it is a rotation, not a fix.
#
#   secret-scan.sh staged  [repo]   scan what is staged for commit (pre-commit)
#   secret-scan.sh tree    [repo]   scan tracked files in the working tree
#   secret-scan.sh history [repo]   scan the whole history (gitleaks only)
#
# SECRET_SCAN_QUIET=1 silences the clean path entirely and throttles the
# "gitleaks is missing" notice to once a day. The pre-commit hook sets it: a
# guard that says something on every single commit trains you to stop reading
# it, and the one commit that mattered scrolls past in the same grey text.
#
# Exit: 0 clean, 1 findings, 2 usage/environment error.
#
# gitleaks is used when installed. When it is not, a much narrower built-in
# pattern set runs instead and SAYS SO — a fallback that stayed quiet about its
# own coverage would be worse than no scanner, because it would buy confidence
# it has not earned.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-staged}"
REPO="${2:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
REPO="$(cd "$REPO" 2>/dev/null && pwd)" || { echo "secret-scan: no such repo" >&2; exit 2; }

QUIET="${SECRET_SCAN_QUIET:-0}"
red() { printf '\033[31m%s\033[0m\n' "$*"; }
ylw() { printf '\033[33m%s\033[0m\n' "$*"; }
dim() { printf '\033[2m%s\033[0m\n' "$*"; }

git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1 || {
  echo "secret-scan: $REPO is not a git repository" >&2; exit 2; }

case "$MODE" in staged|tree|history) ;;
  *) echo "usage: secret-scan.sh {staged|tree|history} [repo]" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------- gitleaks ---
if command -v gitleaks >/dev/null 2>&1; then
  # gitleaks 8.19 deprecated `detect`/`protect` and 8.30 removed them; these are
  # the current subcommands. Pinned by behaviour, not version: if a future
  # release renames them again the scanner fails loudly rather than reporting
  # a clean tree it never scanned.
  case "$MODE" in
    staged)  args=(git --staged .) ;;
    tree)    args=(dir .) ;;
    history) args=(git .) ;;
  esac
  # A repo's own .gitleaks.toml wins; otherwise use the shared one next to this
  # script, so every repo gets the same exclusions without four copies of them.
  cfg=()
  if [ ! -f "$REPO/.gitleaks.toml" ] && [ -f "$HERE/gitleaks.toml" ]; then
    cfg=(--config "$HERE/gitleaks.toml")
  fi
  # A JSON report rather than the human log: the log line says only "leaks
  # found: 1" and the whole point of the message is to say WHICH line, so that
  # the fingerprint it tells you to copy is actually on screen.
  # BSD mktemp requires the X's to end the template; "gitleaks.XXXXXX.json"
  # fails on macOS and the report is silently never written. gitleaks takes the
  # format from --report-format, not from the extension, so no suffix is needed.
  rpt="$(mktemp "${TMPDIR:-/tmp}/gitleaks-report.XXXXXX")"
  out=$(cd "$REPO" && gitleaks "${args[@]}" ${cfg[@]+"${cfg[@]}"} --redact --no-banner \
          --log-level error --report-format json --report-path "$rpt" 2>&1); code=$?
  if [ $code -ge 2 ]; then
    # 0 = clean, 1 = leaks, anything else = gitleaks itself failed. Reporting
    # that as "clean" is the one outcome worse than not scanning at all.
    red "secret-scan: gitleaks exited $code — the scan DID NOT RUN"
    printf '%s\n' "$out" | sed 's/^/  /'
    rm -f "$rpt"; exit 2
  fi
  if [ $code -eq 0 ]; then
    [ "$QUIET" = 1 ] || dim "secrets: clean (gitleaks, $MODE)"
    rm -f "$rpt"; exit 0
  fi
  n=$(jq 'length' "$rpt" 2>/dev/null || echo '?')
  red "SECRETS FOUND — $n finding(s) (gitleaks, $MODE; values redacted)"
  jq -r '.[] | "  \(.File):\(.StartLine)  [\(.RuleID)]\n      \(.Description)\n      fingerprint: \(.Fingerprint)"' \
     "$rpt" 2>/dev/null || printf '%s\n' "$out" | sed 's/^/  /'
  printf '\n  Rotate first, remove second. A secret that reached a commit is spent.\n'
  printf '  A genuine false positive goes in %s/.gitleaksignore by the\n' "$REPO"
  printf '  fingerprint above — never by removing the hook.\n'
  rm -f "$rpt"
  exit 1
fi

# ---------------------------------------------------------------- fallback ---
# High-precision patterns only. Anything entropy-based is deliberately absent:
# on a repo full of hashes, base64 fixtures and GGUF metadata it would cry wolf
# until the hook was removed, which is how a scanner actually fails.
#
# NOTE: this file necessarily contains the patterns it looks for, so it excludes
# itself below. That is the only exclusion built in.
patterns=(
  'AKIA[0-9A-Z]{16}'                                  # AWS access key id
  'ASIA[0-9A-Z]{16}'                                  # AWS temporary key id
  '-----BEGIN [A-Z ]*PRIVATE KEY'                     # any PEM private key
  'gh[pousr]_[A-Za-z0-9]{36}'                         # GitHub token
  'github_pat_[A-Za-z0-9_]{22,}'                      # GitHub fine-grained PAT
  'xox[baprs]-[A-Za-z0-9-]{10,}'                      # Slack token
  'AIza[0-9A-Za-z_-]{35}'                             # Google API key
  'sk_live_[0-9a-zA-Z]{20,}'                          # Stripe live secret
  'rk_live_[0-9a-zA-Z]{20,}'                          # Stripe live restricted
  'npm_[A-Za-z0-9]{36}'                               # npm publish token
  'sk-ant-[A-Za-z0-9_-]{20,}'                         # Anthropic API key
  'sk-proj-[A-Za-z0-9_-]{20,}'                        # OpenAI project key
  'hf_[A-Za-z0-9]{34,}'                               # Hugging Face token
  'eyJ[A-Za-z0-9_-]{15,}\.eyJ[A-Za-z0-9_-]{15,}\.'    # JWT — Supabase service_role
  'SUPABASE_SERVICE_ROLE[_A-Z]*[[:space:]]*[:=]'      # the name alone is enough
  'postgres(ql)?://[^[:space:]:/]+:[^[:space:]@]+@'   # DSN with an inline password
)
# Files that are secrets by name rather than by content.
name_bad='(^|/)\.env($|\.)|\.(pem|p12|pfx|jks|keystore|mobileprovision)$|(^|/)credentials\.json$|(^|/)service-account[^/]*\.json$'
name_ok='\.env\.(example|sample|template)$|\.env\.d\.ts$'

self="${BASH_SOURCE[0]##*/}"
findings=0
tmp="$(mktemp "${TMPDIR:-/tmp}/secret-scan.XXXXXX")"
trap 'rm -f "$tmp"' EXIT

# Notice throttle: loud the first time each day, silent after.
nag=1
if [ "$QUIET" = 1 ]; then
  _stamp="${XDG_CACHE_HOME:-$HOME/.cache}/roofline-guards/gitleaks-nag-$(date +%Y-%m-%d)"
  mkdir -p "$(dirname "$_stamp")" 2>/dev/null
  [ -f "$_stamp" ] && nag=0 || : > "$_stamp"
fi
if [ "$nag" = 1 ]; then
  ylw "secrets: gitleaks not installed — running the narrow built-in fallback"
  dim "  install the real thing:  brew install gitleaks"
fi

case "$MODE" in
  staged)  files=$(git -C "$REPO" diff --cached --name-only --diff-filter=ACM) ;;
  tree)    files=$(git -C "$REPO" ls-files) ;;
  history) echo "secret-scan: history mode needs gitleaks; install it first" >&2; exit 2 ;;
esac

while IFS= read -r f; do
  [ -z "$f" ] && continue
  [ "${f##*/}" = "$self" ] && continue                      # the pattern table itself

  if ! printf '%s' "$f" | grep -qE "$name_ok" && printf '%s' "$f" | grep -qE "$name_bad"; then
    red "SECRET  $f"
    printf '  a file of this name holds credentials; gitignore it, do not commit it\n'
    findings=$((findings + 1))
    continue
  fi

  if [ "$MODE" = staged ]; then
    git -C "$REPO" show ":$f" > "$tmp" 2>/dev/null || continue
  else
    [ -f "$REPO/$f" ] || continue
    [ "$(wc -c < "$REPO/$f")" -gt 2000000 ] && continue
    cp "$REPO/$f" "$tmp" 2>/dev/null || continue
  fi
  grep -qI . "$tmp" 2>/dev/null || continue                 # skip binaries

  for p in "${patterns[@]}"; do
    while IFS= read -r n; do
      [ -z "$n" ] && continue
      red "SECRET  $f:$n"
      printf '  matches /%s/ — the value is deliberately not printed\n' "$p"
      findings=$((findings + 1))
    done < <(grep -nE "$p" "$tmp" 2>/dev/null | cut -d: -f1 | sort -un)
  done
done <<< "$files"

if [ "$findings" -eq 0 ]; then
  [ "$QUIET" = 1 ] || dim "secrets: clean ($MODE, fallback patterns — narrower than gitleaks)"
  exit 0
fi
echo
red "$findings finding(s). No value was printed; open the file yourself."
printf '  Rotate first, remove second. A secret that reached a commit is spent.\n'
exit 1
