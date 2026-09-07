/* arena.h — bump allocator over a lazily-committed mmap reservation.
 *
 * Every allocation in this engine goes through an arena. There is no free();
 * memory comes back in one of two ways:
 *
 *   arena_reset()   throw the whole arena away
 *   arena_mark()/arena_release()  scoped, LIFO — for per-token scratch
 *
 * Why this and not malloc: a forward pass allocates the same shapes in the same
 * order every token. A bump pointer makes that free, keeps activations
 * contiguous (so the prefetcher can see them), and removes any chance of a
 * fragmentation or leak bug in the hot loop.
 *
 * The backing store is an anonymous mmap, so the reservation can be far larger
 * than what is touched — pages only cost RAM once written. Reserve generously.
 */
#ifndef TC_ARENA_H
#define TC_ARENA_H

#include <stddef.h>
#include <stdbool.h>

/* 64 covers every SIMD load we will issue. Apple silicon's cache line is
 * actually 128B; pass that explicitly for anything shared across threads. */
#define ARENA_DEFAULT_ALIGN 64u

typedef struct {
    unsigned char *base;      /* start of the reservation                  */
    size_t         cap;       /* bytes reserved (not necessarily resident) */
    size_t         used;      /* bump offset                               */
    size_t         peak;      /* high-water mark, for reporting            */
    const char    *name;      /* shown in diagnostics                      */
} arena_t;

/* Reserve `cap` bytes (rounded up to a page). Returns an arena with base==NULL
 * on failure; check with arena_ok(). Nothing is resident until you write. */
arena_t arena_create(size_t cap, const char *name);
void    arena_destroy(arena_t *a);
bool    arena_ok(const arena_t *a);

/* Returns NULL when the arena is exhausted. `align` must be a power of two. */
void   *arena_alloc(arena_t *a, size_t size, size_t align);

/* Same, but aborts with a diagnostic instead of returning NULL. Use this for
 * setup-time allocations where running out is a bug, not a condition. */
void   *arena_alloc_or_die(arena_t *a, size_t size, size_t align);

/* Zeroing variant. mmap already hands back zeroed pages, so this only actually
 * memsets memory that a previous arena_reset() left dirty. */
void   *arena_calloc(arena_t *a, size_t count, size_t size, size_t align);

/* Copy `n` bytes into the arena and return the copy. NULL on exhaustion. */
void   *arena_memdup(arena_t *a, const void *src, size_t n, size_t align);

/* Scoped allocation. Release must be called with marks in LIFO order. */
size_t  arena_mark(const arena_t *a);
void    arena_release(arena_t *a, size_t mark);

void    arena_reset(arena_t *a);

size_t  arena_used(const arena_t *a);
size_t  arena_peak(const arena_t *a);
size_t  arena_avail(const arena_t *a);

/* Human-readable "1.94 GiB" into buf. Returns buf. */
char   *arena_fmt_bytes(char *buf, size_t buflen, size_t bytes);

#define ARENA_NEW(a, T)        ((T *)arena_alloc_or_die((a), sizeof(T), _Alignof(T)))
#define ARENA_ARRAY(a, T, n)   ((T *)arena_alloc_or_die((a), sizeof(T) * (size_t)(n), _Alignof(T)))

#endif /* TC_ARENA_H */
