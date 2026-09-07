/* tt.h — a test harness small enough to not need explaining. */
#ifndef TT_H
#define TT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tt_fail = 0;
static int tt_ran  = 0;

#define CHECK(cond) do {                                                   \
    tt_ran++;                                                              \
    if (!(cond)) {                                                         \
        tt_fail++;                                                         \
        fprintf(stderr, "  \033[31mFAIL\033[0m %s:%d  %s\n",               \
                __FILE__, __LINE__, #cond);                                \
    }                                                                      \
} while (0)

#define CHECK_EQ_U(a, b) do {                                              \
    unsigned long long _a = (unsigned long long)(a);                       \
    unsigned long long _b = (unsigned long long)(b);                       \
    tt_ran++;                                                              \
    if (_a != _b) {                                                        \
        tt_fail++;                                                         \
        fprintf(stderr, "  \033[31mFAIL\033[0m %s:%d  %s == %s"            \
                        "  (%llu vs %llu)\n",                              \
                __FILE__, __LINE__, #a, #b, _a, _b);                       \
    }                                                                      \
} while (0)

#define CHECK_STR(s, expect) do {                                          \
    tt_ran++;                                                              \
    if (strcmp((s), (expect)) != 0) {                                      \
        tt_fail++;                                                         \
        fprintf(stderr, "  \033[31mFAIL\033[0m %s:%d  \"%s\" != \"%s\"\n", \
                __FILE__, __LINE__, (s), (expect));                        \
    }                                                                      \
} while (0)

#define TT_DONE() do {                                                     \
    if (tt_fail == 0) {                                                    \
        printf("   %d checks ok\n", tt_ran);                               \
        return 0;                                                          \
    }                                                                      \
    fprintf(stderr, "   %d of %d checks FAILED\n", tt_fail, tt_ran);       \
    return 1;                                                              \
} while (0)

#endif /* TT_H */
