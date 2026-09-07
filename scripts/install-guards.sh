#!/usr/bin/env bash
# Install the commit-time secret guard into repos on this machine.
#
#   install-guards.sh              install into every repo in guarded-repos.txt
#   install-guards.sh <repo>...    install into the named repos
#   install-guards.sh --check      report which repos have the guard, install none
#   install-guards.sh --uninstall  remove the guard (only hooks this wrote)
#
# The hook lives in each repo's .git/hooks/pre-commit and calls back into this
# checkout, so the patterns are maintained in one place. .git/hooks is not
# tracked by git and never travels with a clone — that is a property of git, not
# an oversight here, and it is why this installer exists at all.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCANNER="$HERE/secret-scan.sh"
MARK="# installed by roofline.c scripts/install-guards.sh"

red() { printf '\033[31m%s\033[0m\n' "$*"; }
ylw() { printf '\033[33m%s\033[0m\n' "$*"; }
grn() { printf '\033[32m%s\033[0m\n' "$*"; }
dim() { printf '\033[2m%s\033[0m\n' "$*"; }

repo_list() {
  if [ -n "${ROOFLINE_GUARD_REPOS:-}" ]; then printf '%s\n' "$ROOFLINE_GUARD_REPOS" | tr ':' '\n'
  else grep -vE '^[[:space:]]*(#|$)' "$HERE/guarded-repos.txt"; fi
}

write_hook() { # repo
  local repo="$1" hooks hook
  hooks="$(git -C "$repo" rev-parse --git-path hooks 2>/dev/null)" || return 1
  case "$hooks" in /*) ;; *) hooks="$repo/$hooks" ;; esac
  mkdir -p "$hooks"
  hook="$hooks/pre-commit"

  if [ -e "$hook" ] && ! grep -q "$MARK" "$hook" 2>/dev/null; then
    ylw "  $repo: a pre-commit hook already exists and was NOT written by us — left alone"
    dim "     add this line to it yourself:  \"$SCANNER\" staged || exit 1"
    return 1
  fi

  cat > "$hook" <<HOOK
#!/usr/bin/env bash
$MARK
# Blocks a commit that stages a credential. Bypass with: git commit --no-verify
SCANNER="$SCANNER"
if [ ! -x "\$SCANNER" ]; then
  # Fail OPEN, loudly. A guard that blocks every commit in this repo because an
  # unrelated checkout moved is a guard that gets deleted within the hour; a
  # guard that shouts and steps aside is one that gets repaired. The shout is
  # the load-bearing half — silence here would be the real failure.
  printf '\033[33mpre-commit: secret scanner missing at %s — COMMIT NOT SCANNED\033[0m\n' "\$SCANNER" >&2
  exit 0
fi
exec env SECRET_SCAN_QUIET=1 "\$SCANNER" staged
HOOK
  chmod +x "$hook"
  grn "  $repo: installed"
}

mode=install; targets=()
case "${1:-}" in
  --check)     mode=check;     shift ;;
  --uninstall) mode=uninstall; shift ;;
esac
if [ $# -gt 0 ]; then targets=("$@")
else while IFS= read -r r; do [ -n "$r" ] && targets+=("$r"); done < <(repo_list); fi
[ ${#targets[@]} -eq 0 ] && { echo "install-guards: no repos" >&2; exit 2; }

[ -x "$SCANNER" ] || { red "install-guards: $SCANNER is not executable"; exit 2; }

rc=0
for repo in "${targets[@]}"; do
  if ! git -C "$repo" rev-parse --git-dir >/dev/null 2>&1; then
    ylw "  $repo: not a git repository — skipped"; continue
  fi
  hooks="$(git -C "$repo" rev-parse --git-path hooks 2>/dev/null)"
  case "$hooks" in /*) ;; *) hooks="$repo/$hooks" ;; esac
  hook="$hooks/pre-commit"

  case "$mode" in
    check)
      if [ -e "$hook" ] && grep -q "$MARK" "$hook" 2>/dev/null; then grn "  $repo: guarded"
      elif [ -e "$hook" ]; then ylw "  $repo: has a foreign pre-commit hook"; rc=1
      else red "  $repo: UNGUARDED"; rc=1; fi ;;
    install)
      write_hook "$repo" || rc=1 ;;
    uninstall)
      if [ -e "$hook" ] && grep -q "$MARK" "$hook" 2>/dev/null; then
        rm -f "$hook"; dim "  $repo: removed"
      else dim "  $repo: nothing of ours to remove"; fi ;;
  esac
done

if [ "$mode" = install ]; then
  echo
  dim "The hook is bypassable with --no-verify, by design: a guard you cannot"
  dim "step over gets uninstalled instead of stepped over. It is a seatbelt."
fi
exit $rc
