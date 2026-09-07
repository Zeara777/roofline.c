#include "tt.h"
#include "arena.h"

#include <stdint.h>

int main(void)
{
    arena_t a = arena_create(1u << 20, "test");
    CHECK(arena_ok(&a));
    CHECK(a.cap >= (1u << 20));
    CHECK_EQ_U(arena_used(&a), 0);

    /* alignment is honored */
    for (size_t align = 1; align <= 4096; align *= 2) {
        void *p = arena_alloc(&a, 1, align);
        CHECK(p != NULL);
        CHECK_EQ_U((uintptr_t)p % align, 0);
    }

    /* non-power-of-two alignment is rejected rather than silently rounded */
    CHECK(arena_alloc(&a, 8, 3) == NULL);

    /* mark/release is LIFO and exact */
    size_t mark = arena_mark(&a);
    void *tmp = arena_alloc(&a, 4096, 64);
    CHECK(tmp != NULL);
    CHECK(arena_used(&a) > mark);
    arena_release(&a, mark);
    CHECK_EQ_U(arena_used(&a), mark);

    /* releasing to a mark ahead of the bump pointer must not move it forward */
    arena_release(&a, arena_used(&a) + 999999);
    CHECK_EQ_U(arena_used(&a), mark);

    /* peak survives release */
    CHECK(arena_peak(&a) > mark);

    /* memdup round-trips */
    const char msg[] = "grouped-query attention";
    char *copy = (char *)arena_memdup(&a, msg, sizeof msg, 1);
    CHECK(copy != NULL);
    CHECK(copy != msg);
    CHECK_STR(copy, msg);

    /* calloc zeroes even when reusing dirty memory */
    memset(copy, 0xAB, sizeof msg);
    arena_release(&a, mark);
    unsigned char *z = (unsigned char *)arena_calloc(&a, 64, 1, 64);
    CHECK(z != NULL);
    int all_zero = 1;
    for (int i = 0; i < 64; i++) if (z[i] != 0) all_zero = 0;
    CHECK(all_zero);

    /* exhaustion returns NULL instead of wrapping or aborting */
    CHECK(arena_alloc(&a, a.cap + 1, 64) == NULL);
    CHECK(arena_alloc(&a, SIZE_MAX, 64) == NULL);
    CHECK(arena_alloc(&a, SIZE_MAX - 16, 64) == NULL);

    /* the arena is still usable after a refused allocation */
    CHECK(arena_alloc(&a, 128, 64) != NULL);

    /* calloc overflow check */
    CHECK(arena_calloc(&a, SIZE_MAX / 2, 4, 64) == NULL);

    arena_reset(&a);
    CHECK_EQ_U(arena_used(&a), 0);
    CHECK(arena_peak(&a) > 0);

    /* a zero-capacity arena is invalid, not a trap */
    arena_t nil = arena_create(0, "nil");
    CHECK(!arena_ok(&nil));
    CHECK(arena_alloc(&nil, 1, 1) == NULL);
    arena_destroy(&nil);

    char buf[32];
    CHECK_STR(arena_fmt_bytes(buf, sizeof buf, 512), "512 B");
    CHECK_STR(arena_fmt_bytes(buf, sizeof buf, 1024), "1.00 KiB");
    CHECK_STR(arena_fmt_bytes(buf, sizeof buf, 1536), "1.50 KiB");
    CHECK_STR(arena_fmt_bytes(buf, sizeof buf, 1u << 30), "1.00 GiB");

    arena_destroy(&a);
    CHECK(!arena_ok(&a));

    TT_DONE();
}
