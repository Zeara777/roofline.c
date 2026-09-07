# tc — a transformer inference engine in C
# Stage 01: arena, GGUF loader, tokenizer, fp32 forward pass.

CC      ?= cc
BUILD   ?= build

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

LIB_SRC  := src/arena.c src/gguf.c
LIB_OBJ  := $(LIB_SRC:%.c=$(BUILD)/%.o)

TOOL_SRC := tools/tc_inspect.c
TOOLS    := $(BUILD)/tc-inspect

TEST_SRC := tests/test_arena.c tests/test_gguf.c
TESTS    := $(TEST_SRC:tests/%.c=$(BUILD)/%)

.PHONY: all tools test clean
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

test: $(TESTS)
	@fail=0; for t in $(TESTS); do \
	  printf '\033[2m>>\033[0m %s\n' "$$t"; \
	  ./$$t || fail=1; \
	done; \
	if [ $$fail -eq 0 ]; then printf '\033[32mall tests passed\033[0m\n'; \
	else printf '\033[31mTESTS FAILED\033[0m\n'; exit 1; fi

clean:
	rm -rf $(BUILD)

-include $(shell find $(BUILD) -name '*.d' 2>/dev/null)
