# tc — a transformer inference engine in C
# Stage 01: arena, GGUF loader, tokenizer, fp32 forward pass.

CC      ?= cc

# Sanitizer and STRICT builds get their own object trees. Sharing one meant
# `make ASAN=1 test` followed by `make tools` linked ASAN objects without the
# runtime and failed with a wall of undefined __asan_* symbols — a confusing
# error for a mundane cause.
BUILD   ?= build$(if $(ASAN),-asan)$(if $(STRICT),-strict)

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

CSTD    := -std=c11
WARN    := -Wall -Wextra -Wpedantic -Wshadow -Wstrict-prototypes \
           -Wmissing-prototypes -Wpointer-arith -Wcast-align -Wwrite-strings
OPT     ?= -O2 -g -fno-omit-frame-pointer

# Arch flags. The M1 is the primary target; the x86 box is the second ISA for
# stage 02's cross-ISA kernel comparison.
ifeq ($(UNAME_M),arm64)
  ARCH := -mcpu=apple-m1
else ifeq ($(UNAME_M),aarch64)
  ARCH := -march=armv8.2-a+dotprod+fp16
else
  ARCH := -march=native
endif

CFLAGS  := $(CSTD) $(WARN) $(OPT) $(ARCH) -Isrc -MMD -MP
LDFLAGS :=
LDLIBS  := -lm

# make STRICT=1  -> the full conversion-warning firehose. Worth running before commits.
ifdef STRICT
  CFLAGS += -Wconversion -Wsign-conversion -Wdouble-promotion
endif

# make ASAN=1  -> address + UB sanitizers. Slow; use for tests, not benchmarks.
ifdef ASAN
  CFLAGS  += -fsanitize=address,undefined -fno-sanitize-recover=all -O1
  LDFLAGS += -fsanitize=address,undefined
endif

LIB_SRC  := src/arena.c src/gguf.c src/ops.c src/dequant.c src/vit.c
LIB_OBJ  := $(LIB_SRC:%.c=$(BUILD)/%.o)

TOOL_SRC := tools/tc_inspect.c
TOOLS    := $(BUILD)/tc-inspect

TEST_SRC := tests/test_arena.c tests/test_gguf.c tests/test_ops.c tests/test_vit.c
TESTS    := $(TEST_SRC:tests/%.c=$(BUILD)/%)

.PHONY: all tools test clean disk summary scan scan-history audit audit-status guards
.SECONDARY:
all: tools test

tools: $(TOOLS)

$(BUILD)/tc-inspect: $(BUILD)/tools/tc_inspect.o $(LIB_OBJ)
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD)/test_%: $(BUILD)/tests/test_%.o $(LIB_OBJ)
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

# Disk is checked before tests because a test run can write fixtures and
# exports, and a truncated GGUF is worse than a failed write.
disk:
	@scripts/check-disk.sh

test: disk $(TESTS)
	@fail=0; for t in $(TESTS); do \
	  printf '\033[2m>>\033[0m %s\n' "$$t"; \
	  ./$$t || fail=1; \
	done; \
	if [ $$fail -eq 0 ]; then printf '\033[32mall tests passed\033[0m\n'; \
	else printf '\033[31mTESTS FAILED\033[0m\n'; exit 1; fi

# --- security guards ---------------------------------------------------------
# The commit-time gate is git's pre-commit hook, not a make target: it has to
# fire for commits made outside this Makefile too. These targets are the manual
# half — the scan you run deliberately, and the audit that needs the network.

# Secret scan of the tracked tree. The staged equivalent runs on every commit.
scan:
	@scripts/secret-scan.sh tree

# Whole history. Slow, and worth doing once per repo rather than per session.
scan-history:
	@scripts/secret-scan.sh history

# Dependency advisories across every repo in scripts/guarded-repos.txt.
# Needs the network. Records the result so the SessionStart hook can stay quiet.
audit:
	@scripts/dep-audit.sh run

audit-status:
	@scripts/dep-audit.sh status

# Install the pre-commit secret hook into every guarded repo. Idempotent, and
# it refuses to overwrite a pre-commit hook it did not write.
guards:
	@scripts/install-guards.sh

# Scaffold today's session summary. Summaries live OUTSIDE the repo on purpose
# (see the global CLAUDE.md) so one can never end up in a commit or a PR.
summary:
	@scripts/session-summary.sh new $(or $(SLUG),$(error usage: make summary SLUG=<short-slug>))

clean:
	rm -rf $(BUILD)

-include $(shell find $(BUILD) -name '*.d' 2>/dev/null)
