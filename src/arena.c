#include "arena.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static size_t page_size(void)
{
    static size_t ps = 0;
    if (ps == 0) {
        long v = sysconf(_SC_PAGESIZE);
        ps = (v > 0) ? (size_t)v : 4096u;
    }
    return ps;
}

static bool is_pow2(size_t x) { return x != 0 && (x & (x - 1)) == 0; }

static size_t align_up(size_t x, size_t a) { return (x + (a - 1)) & ~(a - 1); }

arena_t arena_create(size_t cap, const char *name)
{
    arena_t a;
    memset(&a, 0, sizeof(a));
    a.name = name ? name : "arena";

    if (cap == 0) return a;

    size_t rounded = align_up(cap, page_size());

    /* MAP_NORESERVE where available: we want a large reservation without the
     * kernel accounting for swap we will never touch. Darwin ignores it. */
#ifdef MAP_NORESERVE
    int flags = MAP_PRIVATE | MAP_ANON | MAP_NORESERVE;
#else
    int flags = MAP_PRIVATE | MAP_ANON;
#endif

    void *p = mmap(NULL, rounded, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (p == MAP_FAILED) return a;

    a.base = (unsigned char *)p;
    a.cap  = rounded;
    return a;
}

void arena_destroy(arena_t *a)
{
    if (a == NULL || a->base == NULL) return;
    munmap(a->base, a->cap);
    a->base = NULL;
    a->cap = a->used = a->peak = 0;
}

bool arena_ok(const arena_t *a) { return a != NULL && a->base != NULL; }

void *arena_alloc(arena_t *a, size_t size, size_t align)
{
    if (a == NULL || a->base == NULL) return NULL;
    if (align == 0) align = ARENA_DEFAULT_ALIGN;
    if (!is_pow2(align)) return NULL;

    size_t off = align_up(a->used, align);
    if (off > a->cap) return NULL;
    if (size > a->cap - off) return NULL;   /* overflow-safe capacity check */

    a->used = off + size;
    if (a->used > a->peak) a->peak = a->used;
    return a->base + off;
}

void *arena_alloc_or_die(arena_t *a, size_t size, size_t align)
{
    void *p = arena_alloc(a, size, align);
    if (p == NULL) {
        char want[32], have[32], cap[32];
        arena_fmt_bytes(want, sizeof want, size);
        arena_fmt_bytes(have, sizeof have, arena_avail(a));
        arena_fmt_bytes(cap,  sizeof cap,  a ? a->cap : 0);
        fprintf(stderr,
                "tc: arena '%s' exhausted: wanted %s, %s free of %s\n",
                (a && a->name) ? a->name : "?", want, have, cap);
        abort();
    }
    return p;
}

void *arena_calloc(arena_t *a, size_t count, size_t size, size_t align)
{
    if (count != 0 && size > (size_t)-1 / count) return NULL;  /* mul overflow */
    size_t total = count * size;
    void *p = arena_alloc(a, total, align);
    if (p != NULL && total != 0) memset(p, 0, total);
    return p;
}

void *arena_memdup(arena_t *a, const void *src, size_t n, size_t align)
{
    void *p = arena_alloc(a, n, align);
    if (p != NULL && n != 0) memcpy(p, src, n);
    return p;
}

size_t arena_mark(const arena_t *a) { return a ? a->used : 0; }

void arena_release(arena_t *a, size_t mark)
{
    if (a == NULL) return;
    if (mark <= a->used) a->used = mark;
}

void arena_reset(arena_t *a)
{
    if (a != NULL) a->used = 0;
}

size_t arena_used(const arena_t *a)  { return a ? a->used : 0; }
size_t arena_peak(const arena_t *a)  { return a ? a->peak : 0; }
size_t arena_avail(const arena_t *a) { return a ? a->cap - a->used : 0; }

char *arena_fmt_bytes(char *buf, size_t buflen, size_t bytes)
{
    static const char *unit[] = { "B", "KiB", "MiB", "GiB", "TiB" };
    double v = (double)bytes;
    size_t u = 0;
    while (v >= 1024.0 && u + 1 < sizeof unit / sizeof unit[0]) {
        v /= 1024.0;
        u++;
    }
    if (u == 0) snprintf(buf, buflen, "%zu %s", bytes, unit[u]);
    else        snprintf(buf, buflen, "%.2f %s", v, unit[u]);
    return buf;
}
