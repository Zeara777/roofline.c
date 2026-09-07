/* Builds valid and deliberately-corrupt GGUF files in memory, writes them to a
 * temp file and parses them back. This keeps stage 01's loader honest without
 * needing a multi-gigabyte download in CI. */
#include "tt.h"
#include "arena.h"
#include "gguf.h"

#include <stdint.h>
#include <unistd.h>

/* ----------------------------------------------------------- byte buffer -- */

typedef struct { unsigned char *p; size_t n, cap; } buf_t;

static void bgrow(buf_t *b, size_t need)
{
    if (b->n + need <= b->cap) return;
    size_t cap = b->cap ? b->cap : 1024;
    while (cap < b->n + need) cap *= 2;
    b->p = (unsigned char *)realloc(b->p, cap);
    if (!b->p) { perror("realloc"); exit(1); }
    b->cap = cap;
}
static void bput(buf_t *b, const void *src, size_t n)
{
    bgrow(b, n);
    memcpy(b->p + b->n, src, n);
    b->n += n;
}
static void bu32(buf_t *b, uint32_t v) { bput(b, &v, 4); }
static void bu64(buf_t *b, uint64_t v) { bput(b, &v, 8); }
static void bf32(buf_t *b, float v)    { bput(b, &v, 4); }
static void bstr(buf_t *b, const char *s)
{
    size_t n = strlen(s);
    bu64(b, n);
    bput(b, s, n);
}
static void bpad(buf_t *b, size_t align)
{
    while (b->n % align) { unsigned char z = 0; bput(b, &z, 1); }
}

/* ---------------------------------------------------- synthetic gguf file -- */

#define N_KV       7
#define N_TENSORS  2
#define T0_ELEMS   32          /* 4 x 8 f32   -> 128 bytes */
#define T1_ELEMS   64          /* 32 x 2 q4_0 -> 2 blocks -> 36 bytes */
#define T0_BYTES   (T0_ELEMS * 4)
#define T1_BYTES   36
#define DATA_BYTES (T0_BYTES + T1_BYTES)

static buf_t build_good(void)
{
    buf_t b = { NULL, 0, 0 };

    bu32(&b, GGUF_MAGIC);
    bu32(&b, 3);                     /* version   */
    bu64(&b, N_TENSORS);
    bu64(&b, N_KV);

    bstr(&b, "general.architecture");
    bu32(&b, GGUF_T_STRING);  bstr(&b, "llama");

    bstr(&b, "general.alignment");
    bu32(&b, GGUF_T_UINT32);  bu32(&b, 32);

    bstr(&b, "llama.block_count");
    bu32(&b, GGUF_T_UINT32);  bu32(&b, 4);

    bstr(&b, "llama.attention.layer_norm_rms_epsilon");
    bu32(&b, GGUF_T_FLOAT32); bf32(&b, 1e-5f);

    bstr(&b, "test.bool");
    bu32(&b, GGUF_T_BOOL);    { unsigned char t = 1; bput(&b, &t, 1); }

    bstr(&b, "test.i32_array");
    bu32(&b, GGUF_T_ARRAY);   bu32(&b, GGUF_T_INT32); bu64(&b, 3);
    bu32(&b, 10u); bu32(&b, 20u); bu32(&b, 30u);

    bstr(&b, "tokenizer.ggml.tokens");
    bu32(&b, GGUF_T_ARRAY);   bu32(&b, GGUF_T_STRING); bu64(&b, 3);
    bstr(&b, "<s>"); bstr(&b, "hello"); bstr(&b, "world");

    /* tensor 0: f32 4 x 8 at data+0 */
    bstr(&b, "token_embd.weight");
    bu32(&b, 2); bu64(&b, 4); bu64(&b, 8);
    bu32(&b, GGML_TYPE_F32); bu64(&b, 0);

    /* tensor 1: q4_0 32 x 2 at data+128 (32-aligned) */
    bstr(&b, "blk.0.attn_q.weight");
    bu32(&b, 2); bu64(&b, 32); bu64(&b, 2);
    bu32(&b, GGML_TYPE_Q4_0); bu64(&b, T0_BYTES);

    bpad(&b, 32);

    for (int i = 0; i < T0_ELEMS; i++) bf32(&b, (float)i * 0.5f);
    for (int i = 0; i < T1_BYTES; i++) { unsigned char v = (unsigned char)i; bput(&b, &v, 1); }

    return b;
}

static char g_path[] = "/tmp/tc_gguf_test_XXXXXX";

static const char *write_temp(const buf_t *b)
{
    static char path[sizeof g_path];
    memcpy(path, g_path, sizeof g_path);
    int fd = mkstemp(path);
    if (fd < 0) { perror("mkstemp"); exit(1); }
    if (write(fd, b->p, b->n) != (ssize_t)b->n) { perror("write"); exit(1); }
    close(fd);
    return path;
}

/* Parse `b`, expecting success or failure. Returns whether open succeeded. */
static bool try_open(const buf_t *b, gguf_t *g, arena_t *a)
{
    const char *path = write_temp(b);
    bool ok = gguf_open(g, path, a);
    unlink(path);
    return ok;
}

int main(void)
{
    arena_t a = arena_create(8u << 20, "test-gguf");
    CHECK(arena_ok(&a));

    /* ------------------------------------------------------- happy path -- */
    buf_t good = build_good();
    gguf_t g;
    bool ok = try_open(&good, &g, &a);
    CHECK(ok);
    if (!ok) { fprintf(stderr, "  open failed: %s\n", g.err); TT_DONE(); }

    CHECK_EQ_U(g.version, 3);
    CHECK_EQ_U(g.alignment, 32);
    CHECK_EQ_U(g.n_kv, N_KV);
    CHECK_EQ_U(g.n_tensors, N_TENSORS);
    CHECK_EQ_U(g.data_size, DATA_BYTES);

    /* strings are views into the mapping, not copies */
    gguf_str_t arch;
    CHECK(gguf_get_str(&g, "general.architecture", &arch));
    CHECK(gguf_str_eq(arch, "llama"));
    CHECK(!gguf_str_eq(arch, "llam"));
    CHECK(!gguf_str_eq(arch, "llamaa"));
    CHECK((const unsigned char *)arch.ptr >= g.map);
    CHECK((const unsigned char *)arch.ptr < g.map + g.map_size);

    char cbuf[8];
    gguf_str_cstr(cbuf, sizeof cbuf, arch);
    CHECK_STR(cbuf, "llama");
    /* truncating conversion must still NUL-terminate */
    char tiny[4];
    gguf_str_cstr(tiny, sizeof tiny, arch);
    CHECK_STR(tiny, "lla");

    /* a u32 on disk reads back through the u64 accessor */
    CHECK_EQ_U(gguf_u64_or(&g, "llama.block_count", 999), 4);
    CHECK_EQ_U(gguf_u64_or(&g, "does.not.exist", 999), 999);

    float eps = gguf_f32_or(&g, "llama.attention.layer_norm_rms_epsilon", 0.0f);
    CHECK(eps > 0.9e-5f && eps < 1.1e-5f);

    const gguf_kv_t *kb = gguf_find(&g, "test.bool");
    CHECK(kb != NULL && kb->type == GGUF_T_BOOL && kb->v.b);

    const gguf_kv_t *ka = gguf_find(&g, "test.i32_array");
    CHECK(ka != NULL && ka->type == GGUF_T_ARRAY);
    CHECK_EQ_U(ka->v.arr.len, 3);
    CHECK_EQ_U(ka->v.arr.type, GGUF_T_INT32);
    /* Array payloads are at file offsets, not aligned offsets — UBSan catches
     * a direct cast here, which is exactly why the reader copies out. */
    int32_t a2;
    memcpy(&a2, (const unsigned char *)ka->v.arr.data + 2 * sizeof a2, sizeof a2);
    CHECK_EQ_U(a2, 30);

    /* string arrays get an O(1) index built at parse time */
    const gguf_kv_t *kt = gguf_find(&g, "tokenizer.ggml.tokens");
    CHECK(kt != NULL && kt->type == GGUF_T_ARRAY);
    CHECK_EQ_U(kt->v.arr.len, 3);
    CHECK(kt->v.arr.strs != NULL);
    CHECK(gguf_str_eq(kt->v.arr.strs[0], "<s>"));
    CHECK(gguf_str_eq(kt->v.arr.strs[1], "hello"));
    CHECK(gguf_str_eq(kt->v.arr.strs[2], "world"));

    /* tensors: shape, size, and payload */
    const gguf_tensor_t *t0 = gguf_tensor(&g, "token_embd.weight");
    CHECK(t0 != NULL);
    CHECK_EQ_U(t0->n_dims, 2);
    CHECK_EQ_U(t0->dims[0], 4);
    CHECK_EQ_U(t0->dims[1], 8);
    CHECK_EQ_U(t0->n_elem, T0_ELEMS);
    CHECK_EQ_U(t0->type, GGML_TYPE_F32);
    CHECK_EQ_U(t0->nbytes, T0_BYTES);
    CHECK(t0->data != NULL);
    CHECK(((const float *)t0->data)[0] == 0.0f);
    CHECK(((const float *)t0->data)[7] == 3.5f);
    CHECK(((const float *)t0->data)[31] == 15.5f);

    const gguf_tensor_t *t1 = gguf_tensor(&g, "blk.0.attn_q.weight");
    CHECK(t1 != NULL);
    CHECK_EQ_U(t1->n_elem, T1_ELEMS);
    CHECK_EQ_U(t1->type, GGML_TYPE_Q4_0);
    CHECK_EQ_U(t1->nbytes, T1_BYTES);          /* 2 blocks x 18 bytes */
    CHECK_EQ_U(t1->offset, T0_BYTES);
    CHECK_EQ_U(((const unsigned char *)t1->data)[0], 0);
    CHECK_EQ_U(((const unsigned char *)t1->data)[35], 35);

    CHECK(gguf_tensor(&g, "nope") == NULL);
    gguf_close(&g);

    /* --------------------------------------------------- block-size table -- */
    CHECK_EQ_U(ggml_blck_size(GGML_TYPE_Q4_K), 256);
    CHECK_EQ_U(ggml_type_size(GGML_TYPE_Q4_K), 144);   /* 4.5 bits/weight */
    CHECK_EQ_U(ggml_type_size(GGML_TYPE_Q6_K), 210);
    CHECK_EQ_U(ggml_type_size(GGML_TYPE_Q8_0), 34);
    CHECK(ggml_type_supported(GGML_TYPE_Q4_K));
    /* the IQ family is deliberately unsupported rather than guessed */
    CHECK(!ggml_type_supported((ggml_type_t)16));
    CHECK_STR(ggml_type_name((ggml_type_t)16), "UNSUPPORTED");
    CHECK(!ggml_type_supported((ggml_type_t)9999));

    /* ------------------------------------------------------ corrupt files -- */
    buf_t bad;

    /* bad magic */
    bad = build_good();
    bad.p[0] = 'X';
    CHECK(!try_open(&bad, &g, &a));
    CHECK(strstr(g.err, "not a GGUF") != NULL);
    free(bad.p);

    /* big-endian magic is named, not misparsed */
    bad = build_good();
    { uint32_t be = 0x47475546u; memcpy(bad.p, &be, 4); }
    CHECK(!try_open(&bad, &g, &a));
    CHECK(strstr(g.err, "big-endian") != NULL);
    free(bad.p);

    /* unsupported version */
    bad = build_good();
    { uint32_t v = 1; memcpy(bad.p + 4, &v, 4); }
    CHECK(!try_open(&bad, &g, &a));
    CHECK(strstr(g.err, "version") != NULL);
    free(bad.p);

    /* truncated mid-metadata must not read past the mapping */
    bad = build_good();
    bad.n = 40;
    CHECK(!try_open(&bad, &g, &a));
    free(bad.p);

    /* truncated so the data section is short: the tensor bounds check fires */
    bad = build_good();
    bad.n -= 8;
    CHECK(!try_open(&bad, &g, &a));
    CHECK(strstr(g.err, "overruns") != NULL);
    free(bad.p);

    /* an absurd tensor count is rejected before allocating for it */
    bad = build_good();
    { uint64_t n = UINT64_MAX; memcpy(bad.p + 8, &n, 8); }
    CHECK(!try_open(&bad, &g, &a));
    CHECK(strstr(g.err, "impossible") != NULL);
    free(bad.p);

    /* a file too small to hold a header */
    bad = build_good();
    bad.n = 16;
    CHECK(!try_open(&bad, &g, &a));
    CHECK(strstr(g.err, "too small") != NULL);
    free(bad.p);

    /* a missing file reports errno, not a crash */
    CHECK(!gguf_open(&g, "/nonexistent/model.gguf", &a));
    CHECK(strstr(g.err, "cannot open") != NULL);

    free(good.p);
    arena_destroy(&a);
    TT_DONE();
}
