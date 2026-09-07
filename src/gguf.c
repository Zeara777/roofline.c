#include "gguf.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* Both targets (Apple silicon, x86-64) are little-endian and GGUF is defined
 * little-endian, so scalars are read with memcpy straight off the mapping.
 * A byte-swapped magic is detected below and rejected with a clear message
 * rather than silently misparsed. */

/* ---------------------------------------------------------------- cursor -- */

typedef struct {
    const unsigned char *p;
    const unsigned char *end;
    bool ok;
} cur_t;

static bool cur_take(cur_t *c, size_t n, const void **out)
{
    if (!c->ok) return false;
    if (n > (size_t)(c->end - c->p)) { c->ok = false; return false; }
    if (out) *out = c->p;
    c->p += n;
    return true;
}

static bool cur_read(cur_t *c, void *dst, size_t n)
{
    const void *src;
    if (!cur_take(c, n, &src)) return false;
    memcpy(dst, src, n);
    return true;
}

static uint32_t cur_u32(cur_t *c) { uint32_t v = 0; cur_read(c, &v, 4); return v; }
static uint64_t cur_u64(cur_t *c) { uint64_t v = 0; cur_read(c, &v, 8); return v; }

static gguf_str_t cur_str(cur_t *c)
{
    gguf_str_t s = { NULL, 0 };
    uint64_t len = cur_u64(c);
    if (!c->ok) return s;
    /* A length that cannot fit in the remaining file is the usual shape of a
     * corrupt or truncated download; reject before it becomes a huge alloc. */
    if (len > (uint64_t)(c->end - c->p)) { c->ok = false; return s; }
    const void *p;
    if (!cur_take(c, (size_t)len, &p)) return s;
    s.ptr = (const char *)p;
    s.len = len;
    return s;
}

/* ------------------------------------------------------------- type info -- */

typedef struct {
    const char *name;
    uint32_t    blck;
    size_t      size;
    bool        supported;
} type_info_t;

/* Block sizes are the on-disk ggml layouts. The K-quants use a 256-element
 * superblock: Q4_K is 2 (d) + 2 (dmin) + 12 (6-bit scales/mins) + 128 (nibbles)
 * = 144 bytes, i.e. 4.5 bits per weight. */
static const type_info_t g_types[GGML_TYPE_MAX] = {
    [GGML_TYPE_F32]  = { "F32",   1,   4, true },
    [GGML_TYPE_F16]  = { "F16",   1,   2, true },
    [GGML_TYPE_Q4_0] = { "Q4_0", 32,  18, true },
    [GGML_TYPE_Q4_1] = { "Q4_1", 32,  20, true },
    [GGML_TYPE_Q5_0] = { "Q5_0", 32,  22, true },
    [GGML_TYPE_Q5_1] = { "Q5_1", 32,  24, true },
    [GGML_TYPE_Q8_0] = { "Q8_0", 32,  34, true },
    [GGML_TYPE_Q8_1] = { "Q8_1", 32,  36, true },
    [GGML_TYPE_Q2_K] = { "Q2_K", 256, 84, true },
    [GGML_TYPE_Q3_K] = { "Q3_K", 256, 110, true },
    [GGML_TYPE_Q4_K] = { "Q4_K", 256, 144, true },
    [GGML_TYPE_Q5_K] = { "Q5_K", 256, 176, true },
    [GGML_TYPE_Q6_K] = { "Q6_K", 256, 210, true },
    [GGML_TYPE_Q8_K] = { "Q8_K", 256, 292, true },
    [GGML_TYPE_I8]   = { "I8",    1,   1, true },
    [GGML_TYPE_I16]  = { "I16",   1,   2, true },
    [GGML_TYPE_I32]  = { "I32",   1,   4, true },
    [GGML_TYPE_I64]  = { "I64",   1,   8, true },
    [GGML_TYPE_F64]  = { "F64",   1,   8, true },
    [GGML_TYPE_BF16] = { "BF16",  1,   2, true },
    /* The IQ* / TQ* families are deliberately absent. This engine will never
     * read them, and inventing a block size here would turn an unsupported
     * file into silently wrong tensor offsets. */
};

static const type_info_t *type_info(ggml_type_t t)
{
    if ((unsigned)t >= GGML_TYPE_MAX) return NULL;
    const type_info_t *ti = &g_types[t];
    return ti->supported ? ti : NULL;
}

const char *ggml_type_name(ggml_type_t t)
{
    const type_info_t *ti = type_info(t);
    return ti ? ti->name : "UNSUPPORTED";
}
uint32_t ggml_blck_size(ggml_type_t t)
{
    const type_info_t *ti = type_info(t);
    return ti ? ti->blck : 0;
}
size_t ggml_type_size(ggml_type_t t)
{
    const type_info_t *ti = type_info(t);
    return ti ? ti->size : 0;
}
bool ggml_type_supported(ggml_type_t t) { return type_info(t) != NULL; }

static const char *const g_kv_names[GGUF_T_COUNT] = {
    "uint8", "int8", "uint16", "int16", "uint32", "int32",
    "float32", "bool", "string", "array", "uint64", "int64", "float64",
};

const char *gguf_type_name(gguf_type_t t)
{
    return ((unsigned)t < GGUF_T_COUNT) ? g_kv_names[t] : "?";
}

static size_t kv_width(gguf_type_t t)
{
    switch (t) {
    case GGUF_T_UINT8: case GGUF_T_INT8: case GGUF_T_BOOL:    return 1;
    case GGUF_T_UINT16: case GGUF_T_INT16:                    return 2;
    case GGUF_T_UINT32: case GGUF_T_INT32: case GGUF_T_FLOAT32: return 4;
    case GGUF_T_UINT64: case GGUF_T_INT64: case GGUF_T_FLOAT64: return 8;
    default: return 0;   /* string and array are variable-width */
    }
}

/* ---------------------------------------------------------------- strings -- */

char *gguf_str_cstr(char *buf, size_t buflen, gguf_str_t s)
{
    if (buflen == 0) return buf;
    size_t n = s.len < buflen - 1 ? (size_t)s.len : buflen - 1;
    if (n && s.ptr) memcpy(buf, s.ptr, n);
    buf[n] = '\0';
    return buf;
}

bool gguf_str_eq(gguf_str_t s, const char *cstr)
{
    size_t n = strlen(cstr);
    return s.ptr != NULL && s.len == (uint64_t)n && memcmp(s.ptr, cstr, n) == 0;
}

/* ----------------------------------------------------------------- errors -- */

static bool fail(gguf_t *g, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

static bool fail(gguf_t *g, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g->err, sizeof g->err, fmt, ap);
    va_end(ap);
    return false;
}

/* ------------------------------------------------------------------ parse -- */

static bool parse_kv(gguf_t *g, cur_t *c, gguf_kv_t *kv, arena_t *a)
{
    kv->key = cur_str(c);
    if (!c->ok) return fail(g, "truncated while reading a metadata key");

    uint32_t t = cur_u32(c);
    if (!c->ok) return fail(g, "truncated while reading a metadata type");
    if (t >= GGUF_T_COUNT) return fail(g, "unknown metadata value type %u", t);
    kv->type = (gguf_type_t)t;

    if (kv->type == GGUF_T_STRING) {
        kv->v.str = cur_str(c);
        if (!c->ok) return fail(g, "truncated string value");
        return true;
    }

    if (kv->type == GGUF_T_ARRAY) {
        uint32_t et = cur_u32(c);
        uint64_t n  = cur_u64(c);
        if (!c->ok) return fail(g, "truncated array header");
        if (et >= GGUF_T_COUNT) return fail(g, "unknown array element type %u", et);
        if (et == GGUF_T_ARRAY) return fail(g, "nested arrays are not supported");

        kv->v.arr.type = (gguf_type_t)et;
        kv->v.arr.len  = n;
        kv->v.arr.data = NULL;
        kv->v.arr.strs = NULL;

        if (et == GGUF_T_STRING) {
            /* Variable-width elements, so build an O(1) index. The tokenizer
             * vocab arrives this way and is looked up constantly. */
            if (n > (uint64_t)(c->end - c->p) / 8u)
                return fail(g, "array of %llu strings cannot fit in the file",
                            (unsigned long long)n);
            gguf_str_t *v = NULL;
            if (n > 0) {
                v = (gguf_str_t *)arena_alloc(a, (size_t)n * sizeof *v, _Alignof(gguf_str_t));
                if (v == NULL) return fail(g, "arena exhausted indexing %llu strings",
                                           (unsigned long long)n);
            }
            for (uint64_t i = 0; i < n; i++) {
                v[i] = cur_str(c);
                if (!c->ok) return fail(g, "truncated at string %llu of %llu",
                                        (unsigned long long)i, (unsigned long long)n);
            }
            kv->v.arr.strs = v;
            return true;
        }

        size_t w = kv_width((gguf_type_t)et);
        if (w == 0) return fail(g, "array of unsized type %u", et);
        if (n > (uint64_t)(c->end - c->p) / w)
            return fail(g, "array of %llu x %zuB overruns the file",
                        (unsigned long long)n, w);
        const void *p;
        if (!cur_take(c, (size_t)n * w, &p)) return fail(g, "truncated array body");
        kv->v.arr.data = p;
        return true;
    }

    size_t w = kv_width(kv->type);
    if (w == 0) return fail(g, "metadata type %u has no width", t);
    if (!cur_read(c, &kv->v, w)) return fail(g, "truncated scalar value");
    return true;
}

static bool mul_ovf_u64(uint64_t a, uint64_t b, uint64_t *out)
{
    if (a != 0 && b > UINT64_MAX / a) return true;
    *out = a * b;
    return false;
}

static bool parse_tensor(gguf_t *g, cur_t *c, gguf_tensor_t *t)
{
    t->name = cur_str(c);
    if (!c->ok) return fail(g, "truncated tensor name");

    t->n_dims = cur_u32(c);
    if (!c->ok) return fail(g, "truncated tensor rank");
    if (t->n_dims == 0 || t->n_dims > GGUF_MAX_DIMS)
        return fail(g, "tensor rank %u out of range 1..%d", t->n_dims, GGUF_MAX_DIMS);

    t->n_elem = 1;
    for (uint32_t i = 0; i < GGUF_MAX_DIMS; i++) t->dims[i] = 1;
    for (uint32_t i = 0; i < t->n_dims; i++) {
        t->dims[i] = cur_u64(c);
        if (!c->ok) return fail(g, "truncated tensor dimensions");
        if (mul_ovf_u64(t->n_elem, t->dims[i], &t->n_elem))
            return fail(g, "tensor element count overflows");
    }

    uint32_t ty = cur_u32(c);
    if (!c->ok) return fail(g, "truncated tensor type");
    t->type   = (ggml_type_t)ty;
    t->offset = cur_u64(c);
    if (!c->ok) return fail(g, "truncated tensor offset");

    const type_info_t *ti = type_info(t->type);
    if (ti == NULL) {
        /* Not fatal: the inspector should still be able to list the tensor.
         * nbytes stays 0 and any attempt to read it will be rejected. */
        t->nbytes = 0;
        t->data   = NULL;
        return true;
    }
    if (t->n_elem % ti->blck != 0)
        return fail(g, "%.*s: %llu elements is not a multiple of the %s block size %u",
                    (int)t->name.len, t->name.ptr,
                    (unsigned long long)t->n_elem, ti->name, ti->blck);

    t->nbytes = (t->n_elem / ti->blck) * ti->size;
    return true;
}

/* ------------------------------------------------------------------- open -- */

static size_t align_up_sz(size_t x, size_t a) { return (x + (a - 1)) & ~(a - 1); }

bool gguf_open(gguf_t *out, const char *path, arena_t *a)
{
    memset(out, 0, sizeof *out);
    out->fd = -1;

    int fd = open(path, O_RDONLY);
    if (fd < 0) return fail(out, "cannot open %s: %s", path, strerror(errno));

    struct stat st;
    if (fstat(fd, &st) != 0) { int e = errno; close(fd);
        return fail(out, "cannot stat %s: %s", path, strerror(e)); }
    if (st.st_size <= 0 || (uint64_t)st.st_size < 24) {
        close(fd);
        return fail(out, "%s is too small to be a GGUF file (%lld bytes)",
                    path, (long long)st.st_size);
    }

    size_t sz = (size_t)st.st_size;
    void *m = mmap(NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) { int e = errno; close(fd);
        return fail(out, "cannot mmap %s: %s", path, strerror(e)); }

    out->map = (const unsigned char *)m;
    out->map_size = sz;
    out->fd = fd;

    cur_t c = { out->map, out->map + sz, true };

    uint32_t magic = cur_u32(&c);
    if (magic != GGUF_MAGIC) {
        /* "FUGG" is the same four bytes read big-endian. */
        bool swapped = (magic == 0x47475546u);
        gguf_close(out);
        return fail(out, swapped
                    ? "%s is a big-endian GGUF; this reader is little-endian only"
                    : "%s is not a GGUF file (magic 0x%08x)",
                    path, magic);
    }

    out->version = cur_u32(&c);
    if (out->version != 2 && out->version != 3) {
        uint32_t v = out->version;
        gguf_close(out);
        return fail(out, "unsupported GGUF version %u (expected 2 or 3)", v);
    }

    out->n_tensors = cur_u64(&c);
    out->n_kv      = cur_u64(&c);
    if (!c.ok) { gguf_close(out); return fail(out, "truncated GGUF header"); }

    /* Each KV needs >= 12 bytes on disk and each tensor info >= 24; reject
     * absurd counts before allocating for them. */
    if (out->n_kv > sz / 12u || out->n_tensors > sz / 24u) {
        gguf_close(out);
        return fail(out, "header claims %llu KVs and %llu tensors, impossible in %zu bytes",
                    (unsigned long long)out->n_kv,
                    (unsigned long long)out->n_tensors, sz);
    }

    if (out->n_kv > 0) {
        out->kv = (gguf_kv_t *)arena_alloc(a, (size_t)out->n_kv * sizeof *out->kv,
                                           _Alignof(gguf_kv_t));
        if (out->kv == NULL) { gguf_close(out); return fail(out, "arena exhausted (kv)"); }
    }
    for (uint64_t i = 0; i < out->n_kv; i++) {
        if (!parse_kv(out, &c, &out->kv[i], a)) {
            char tmp[sizeof out->err];
            memcpy(tmp, out->err, sizeof tmp);
            gguf_close(out);
            memcpy(out->err, tmp, sizeof out->err);
            return false;
        }
    }

    uint32_t align = GGUF_DEFAULT_ALIGNMENT;
    const gguf_kv_t *kva = gguf_find(out, "general.alignment");
    if (kva != NULL && kva->type == GGUF_T_UINT32) align = kva->v.u32;
    if (align == 0 || (align & (align - 1)) != 0) {
        gguf_close(out);
        return fail(out, "general.alignment %u is not a power of two", align);
    }
    out->alignment = align;

    if (out->n_tensors > 0) {
        out->tensors = (gguf_tensor_t *)arena_alloc(
            a, (size_t)out->n_tensors * sizeof *out->tensors, _Alignof(gguf_tensor_t));
        if (out->tensors == NULL) {
            gguf_close(out); return fail(out, "arena exhausted (tensors)");
        }
    }
    for (uint64_t i = 0; i < out->n_tensors; i++) {
        if (!parse_tensor(out, &c, &out->tensors[i])) {
            char tmp[sizeof out->err];
            memcpy(tmp, out->err, sizeof tmp);
            gguf_close(out);
            memcpy(out->err, tmp, sizeof out->err);
            return false;
        }
    }

    size_t hdr_end   = (size_t)(c.p - out->map);
    size_t data_off  = align_up_sz(hdr_end, align);
    if (data_off > sz) {
        gguf_close(out);
        return fail(out, "tensor data section starts past end of file");
    }
    out->data      = out->map + data_off;
    out->data_size = sz - data_off;

    /* Resolve every tensor payload and check it lies inside the mapping. Doing
     * this once here means the forward pass can dereference without checks. */
    for (uint64_t i = 0; i < out->n_tensors; i++) {
        gguf_tensor_t *t = &out->tensors[i];
        if (t->nbytes == 0) continue;                  /* unsupported type */
        if (t->offset > out->data_size ||
            t->nbytes > out->data_size - t->offset) {
            char nm[128];
            gguf_str_cstr(nm, sizeof nm, t->name);
            uint64_t off = t->offset, nb = t->nbytes;
            size_t ds = out->data_size;
            gguf_close(out);
            return fail(out, "tensor '%s' at +%llu (%llu bytes) overruns the "
                             "%zu-byte data section", nm,
                        (unsigned long long)off, (unsigned long long)nb, ds);
        }
        t->data = out->data + t->offset;
    }

    return true;
}

void gguf_close(gguf_t *g)
{
    if (g == NULL) return;
    if (g->map != NULL) munmap((void *)g->map, g->map_size);
    if (g->fd >= 0) close(g->fd);
    g->map = NULL;
    g->data = NULL;
    g->fd = -1;
    g->map_size = g->data_size = 0;
}

/* ---------------------------------------------------------------- lookups -- */

const gguf_kv_t *gguf_find(const gguf_t *g, const char *key)
{
    /* Linear. A model file has a few hundred KVs and this runs once at load. */
    for (uint64_t i = 0; i < g->n_kv; i++)
        if (gguf_str_eq(g->kv[i].key, key)) return &g->kv[i];
    return NULL;
}

bool gguf_get_u64(const gguf_t *g, const char *key, uint64_t *out)
{
    const gguf_kv_t *kv = gguf_find(g, key);
    if (kv == NULL) return false;
    switch (kv->type) {
    case GGUF_T_UINT8:  *out = kv->v.u8;  return true;
    case GGUF_T_UINT16: *out = kv->v.u16; return true;
    case GGUF_T_UINT32: *out = kv->v.u32; return true;
    case GGUF_T_UINT64: *out = kv->v.u64; return true;
    case GGUF_T_BOOL:   *out = kv->v.b ? 1u : 0u; return true;
    case GGUF_T_INT8:   if (kv->v.i8  < 0) return false; *out = (uint64_t)kv->v.i8;  return true;
    case GGUF_T_INT16:  if (kv->v.i16 < 0) return false; *out = (uint64_t)kv->v.i16; return true;
    case GGUF_T_INT32:  if (kv->v.i32 < 0) return false; *out = (uint64_t)kv->v.i32; return true;
    case GGUF_T_INT64:  if (kv->v.i64 < 0) return false; *out = (uint64_t)kv->v.i64; return true;
    default: return false;
    }
}

bool gguf_get_i64(const gguf_t *g, const char *key, int64_t *out)
{
    const gguf_kv_t *kv = gguf_find(g, key);
    if (kv == NULL) return false;
    switch (kv->type) {
    case GGUF_T_UINT8:  *out = kv->v.u8;  return true;
    case GGUF_T_UINT16: *out = kv->v.u16; return true;
    case GGUF_T_UINT32: *out = kv->v.u32; return true;
    case GGUF_T_UINT64: if (kv->v.u64 > INT64_MAX) return false;
                        *out = (int64_t)kv->v.u64; return true;
    case GGUF_T_INT8:   *out = kv->v.i8;  return true;
    case GGUF_T_INT16:  *out = kv->v.i16; return true;
    case GGUF_T_INT32:  *out = kv->v.i32; return true;
    case GGUF_T_INT64:  *out = kv->v.i64; return true;
    case GGUF_T_BOOL:   *out = kv->v.b ? 1 : 0; return true;
    default: return false;
    }
}

bool gguf_get_f32(const gguf_t *g, const char *key, float *out)
{
    const gguf_kv_t *kv = gguf_find(g, key);
    if (kv == NULL) return false;
    if (kv->type == GGUF_T_FLOAT32) { *out = kv->v.f32; return true; }
    if (kv->type == GGUF_T_FLOAT64) { *out = (float)kv->v.f64; return true; }
    return false;
}

bool gguf_get_str(const gguf_t *g, const char *key, gguf_str_t *out)
{
    const gguf_kv_t *kv = gguf_find(g, key);
    if (kv == NULL || kv->type != GGUF_T_STRING) return false;
    *out = kv->v.str;
    return true;
}

uint64_t gguf_u64_or(const gguf_t *g, const char *key, uint64_t fallback)
{
    uint64_t v;
    return gguf_get_u64(g, key, &v) ? v : fallback;
}

float gguf_f32_or(const gguf_t *g, const char *key, float fallback)
{
    float v;
    return gguf_get_f32(g, key, &v) ? v : fallback;
}

const gguf_tensor_t *gguf_tensor(const gguf_t *g, const char *name)
{
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (gguf_str_eq(g->tensors[i].name, name)) return &g->tensors[i];
    return NULL;
}
