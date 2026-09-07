# Security guards

Track B4 and B5 of the master plan. Three mechanisms, all of them quiet until
they have something to say:

| What | When it runs | Where |
|---|---|---|
| Secret scan | every commit, in every guarded repo | git `pre-commit` hook |
| Dependency audit | on demand; staleness checked at session start | `make audit` |
| Tier 3 cadence | checked at session start | `scripts/cadence.sh` |

## Why these and not others

The March 2026 incident came in through a trojanized VS Code extension, and
none of this would have caught it — that was malice in a package, not a
published advisory or a committed key. What it does cover is the exposure that
*followed*: a RAT with read access to this filesystem from 2026-03-28 to
2026-09-05, and every credential sitting in a working tree during that window.

The preventive controls that would actually have broken the March chain —
BlockBlock at the LaunchAgent write, LuLu at the outbound call to
`cdn.rraghh.com` — are B2, and they are installs, not scripts.

## Secret scan

`scripts/secret-scan.sh {staged|tree|history} [repo]`

Uses **gitleaks** (8.30.1, installed via Homebrew). Without it, a much narrower
built-in pattern set runs and says so on every invocation — a fallback that
stayed quiet about its own coverage would buy confidence it has not earned.

- `staged` — what the pre-commit hook runs. ~0.2 s.
- `tree` — tracked files. ~1.2 s.
- `history` — every commit. 3 s across divegomobile's 2,409.

Exclusions live in `scripts/gitleaks.toml`, shared by every repo that has no
`.gitleaks.toml` of its own. Without it a full-tree scan walks 2.6 GB of `.gguf`
weights and a 763 MB `.venv` here, and `node_modules` in the React Native repos;
the first attempt ran for minutes and had to be killed. A scan too slow to
finish is a scan that stops being run.

False positives go in `.gitleaksignore` **by fingerprint** — never by disabling
the rule. Disabling `generic-api-key` to silence one line of README prose also
disables it for the line that matters, and nothing would ever say so.

### Installing it in another repo

```
make guards                    # every repo in scripts/guarded-repos.txt
scripts/install-guards.sh --check
scripts/install-guards.sh --uninstall
```

`.git/hooks` is not tracked by git and never travels with a clone. That is why
an installer exists rather than a committed hook.

Two deliberate choices in the hook:

- **It is bypassable** with `git commit --no-verify`. A guard you cannot step
  over gets uninstalled instead of stepped over.
- **It fails open, loudly**, if the scanner is missing. A guard that blocks
  every commit in an unrelated repo because this checkout moved is a guard that
  gets deleted within the hour; one that shouts and stands aside gets repaired.
  The shout is the load-bearing half.

## Dependency audit

`make audit` · `make audit-status` · `scripts/dep-audit.sh ack <repo>`

Per-ecosystem, because the tools are not interchangeable — `npm audit` returns
nothing at all in a pnpm repo, which is how warDrobe first came back "clean"
while carrying two high advisories:

- `pnpm-lock.yaml` → `pnpm audit`, `yarn.lock` → `yarn npm audit`, else `npm audit`
- `requirements.txt` → **OSV's `querybatch` API**, not pip-audit. pip-audit
  builds a throwaway virtualenv to resolve `-r`, and `ensurepip` aborts under
  uv's standalone Python here. It is not needed: requirements.txt is pinned
  exactly, and an exact version is what OSV wants. The supply-chain pinning
  pays for itself a second time.

An audit that could not run is recorded as **FAILED, never as clean**. Reporting
"0 vulnerabilities" for a scan that never happened is the one output worse than
no audit, because it is indistinguishable from good news.

### Evidence that this matters

YC's internal agent harness gates database writes behind a human reviewing a
proposed plan. In production, per their own presenters: *"we've started just
kind of rubber stamping these"* — compared explicitly to how carefully people
read early coding-agent tool calls versus how they read them now. A control that
fires on everything stops being a control, and it degrades silently, because a
rubber stamp and a real review produce the same log line.

That is the whole design brief for the guards here: fire rarely, and mean it.

`ack` exists for the same reason the daily security check has `--accept`.
Standing transitive advisories in a build toolchain do not clear until upstream
bumps. Acknowledging records the exact counts; **if they change, the guard
speaks again** — so an acknowledgement can never hide a new finding.

## Tier 3 cadence

`scripts/cadence.sh {check|list|done <id>}`, tasks in `scripts/cadence.txt`.

The recurring work with no trigger of its own. A dependency advisory announces
itself eventually; a quarterly read of the vulnerability-discovery literature
never does. The difference between this and a checklist in a document is that a
checklist is read when you go looking for it — which is exactly when you did not
need reminding.

## State

`~/.cache/roofline-guards/` — one JSON per repo plus `cadence/` timestamps.
Outside the repo on purpose: it records facts about this machine, not about the
project, and it must never reach a commit.
