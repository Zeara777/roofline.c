# Working in roofline.c

Operational rules. What the project *is* and why it is built this way lives in
[`ARCHITECTURE.md`](ARCHITECTURE.md) and [`README.md`](README.md) — read those
for design, not this. This file is the part that is easy to get wrong.

The master plan is **`~/research/PLAN.md`**, not this repo. It sequences work
across projects and outranks any TODO list here.

---

## Start of session

- **"resume" or "continue" means: read the newest summary in
  `~/.claude/session-summaries/CPT/` before doing anything else.** It carries
  the state that the git log does not.
- Report the daily security check (`~/Library/Logs/daily-security-check.log`)
  if it says FAIL or WARN, before other work.
- Session summaries live **outside the repo on purpose** so one can never land
  in a commit. Scaffold with `make summary SLUG=<slug>`.

## Facts, and how to label them

Borrowed from `PLAN.md`, and it applies in commits, docs and conversation:

- **MEASURED** — a number that came from running something. Say what produced it.
- **DERIVED** — computed from an assumption. Name the assumption.
- **CLAIMED** — asserted, unverified. Say so before it is relied on.

*Every performance number in this repository is reproducible from a command in
it, or it is labelled a target.* The 68 GB/s in the README is **DERIVED from a
spec sheet and has never been measured on this machine** — every roofline figure
inherits that, so do not print it as fact.

**Do not report a task as done when it was not run.** "Tests pass" means the
tests ran in this session. A skipped step gets said out loud.

## Accuracy work

- **Gate against fp32, never f16.** Tolerances then mean "kernel bug", and the
  f16 cost stays a measurement instead of becoming a hidden allowance.
- **Tolerances are set to ~10× the measured error**, not to whatever passes.
  They must hold at `-O2` *and* under ASAN at `-O1`.
- **Probe the checkpoint; never assume architecture.** Activation, pooling,
  LayerScale, eps and tensor order were all determined numerically off the
  weights. The exporter *detects* and records them, and `tc_vit_load` refuses a
  file that lacks the key. Extend that pattern; do not reintroduce assumptions.
- **The scalar fp32 reference kernel for each op is the definition of that op.**
  It is never optimized. Optimized kernels are checked against it, not against
  each other.
- **Fail loudly on the unknown.** An unsupported tensor type reports
  `UNSUPPORTED` with `nbytes == 0`. Guessing a block size turns an unreadable
  file into silently wrong offsets, which is the worst failure a loader has.

## Build and gates

```sh
make            # tools + tests, disk-checked first
make STRICT=1   # -Wconversion -Wsign-conversion -Wdouble-promotion
make ASAN=1     # address + UB sanitizers; tests only, never benchmarks
make scan       # secret scan of the tracked tree
make audit      # dependency advisories across the guarded repos
```

`make STRICT=1 all` is **silent**, and ASAN+UBSan is clean. Keep both that way.
Fix pre-existing warnings rather than working around them: *a gate that always
yells cannot report on new code.* That reasoning applies to every alarm here —
see [`docs/SECURITY_GUARDS.md`](docs/SECURITY_GUARDS.md).

C11, **no dependencies beyond libc**. Do not add one; propose it instead.

## Git

- **Commit to `main`. Do not push.** Standing instruction — ask before any push.
- A pre-commit hook blocks staged credentials. It is bypassable with
  `--no-verify`; do not bypass it to make a commit go through.
- Large artefacts (`*.gguf`, `.venv`, `StdyDocs/`) are gitignored. Check disk
  before an export: a truncated GGUF still parses far enough to look plausible.

## Things that are Rafael's, not mine

Do these **only** when asked, and never on his behalf:

- `bash ~/bin/daily-security-check.sh --accept`
- `scripts/dep-audit.sh ack <repo>`
- Anything that pushes, publishes, or contacts a third party (VirusTotal
  submissions, issue filing, package publication).
- The four decisions in `PLAN.md` §7 — track ordering, whether the divegomobile
  eval is in scope, income framing, and how much of Track C is a deliverable.

## Upkeep that is easy to forget

- **`StdyDocs/RooflineLessons/` is gitignored and must be updated in the same
  session the engine changes.** A lesson describing code that no longer exists
  is worse than no lesson.
- Research goes in **`~/research/`, filed by trigger, never pruned to the
  active task**, marked READ / INDEXED / UNREAD. Do not discard a source for
  being off-topic today.
- `tools/model-pins.json` holds the HF revision + SHA-256 for every checkpoint.
  `fetch_pinned()` verifies *before* torch opens anything. Never widen a pin to
  make a download succeed.

## How to talk to me about this work

- **Blunt criticism is wanted**, on reasoning and judgement, not only on code.
  State the problem once, plainly, with the fix. Do not soften it.
- **Search before disputing a factual claim.** "I have no knowledge of X" is
  weak evidence, including for things inside the training window — that has
  already been wrong here, repeatedly and confidently.
- **Do not stop early.** Finish the whole task, say explicitly what was left
  out and why, and do not hand back a partial result as if it were the scope.
  If part of it is blocked, complete everything else first. Scaling the work
  down is Rafael's call, not mine.
- A rule means **harness enforcement, not a note**: pair a hook with a make
  target, and keep it quiet when nothing is owed.
