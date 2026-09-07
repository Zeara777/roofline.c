/* tc-inspect — dump the structure of a GGUF file.
 *
 * Stage 01's first checkpoint: if this prints a model's architecture,
 * hyperparameters and tensor table correctly, the loader is trustworthy enough
 * to build a forward pass on top of.
 *
 *   tc-inspect model.gguf              header + metadata + type summary
 *   tc-inspect model.gguf --tensors    also list every tensor
 *   tc-inspect model.gguf --arrays=16  show more array elements
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arena.h"
#include "gguf.h"

#define DIM   "\033[2m"
#define BOLD  "\033[1m"
#define RST   "\033[0m"

static bool g_color = true;
static const char *dim(void)  { return g_color ? DIM  : ""; }
static const char *bold(void) { return g_color ? BOLD : ""; }
static const char *rst(void)  { return g_color ? RST  : ""; }

static void print_scalar(const gguf_kv_t *kv)
{
    switch (kv->type) {
    case GGUF_T_UINT8:   printf("%" PRIu8,  kv->v.u8);  break;
    case GGUF_T_INT8:    printf("%" PRId8,  kv->v.i8);  break;
    case GGUF_T_UINT16:  printf("%" PRIu16, kv->v.u16); break;
    case GGUF_T_INT16:   printf("%" PRId16, kv->v.i16); break;
    case GGUF_T_UINT32:  printf("%" PRIu32, kv->v.u32); break;
    case GGUF_T_INT32:   printf("%" PRId32, kv->v.i32); break;
    case GGUF_T_UINT64:  printf("%" PRIu64, kv->v.u64); break;
    case GGUF_T_INT64:   printf("%" PRId64, kv->v.i64); break;
    case GGUF_T_FLOAT32: printf("%g", (double)kv->v.f32); break;
    case GGUF_T_FLOAT64: printf("%g", kv->v.f64); break;
    case GGUF_T_BOOL:    printf("%s", kv->v.b ? "true" : "false"); break;
    default:             printf("?"); break;
    }
}

static void print_arr_elem(const gguf_arr_t *a, uint64_t i)
{
    if (a->type == GGUF_T_STRING) {
        gguf_str_t s = a->strs[i];
        /* Vocab entries contain control bytes and byte-fallback tokens; print
         * them raw but bounded, and let the terminal deal with it. */
        printf("\"%.*s\"", (int)(s.len > 24 ? 24 : s.len), s.ptr);
        if (s.len > 24) printf("…");
        return;
    }
    /* Array elements sit at whatever offset the file put them at, so they are
     * not guaranteed to be naturally aligned. Copy out rather than cast — on
     * arm64 an unaligned load happens to work, but it is still UB and the x86
     * build in stage 02 will not be so forgiving. */
    const unsigned char *p = (const unsigned char *)a->data;
#define ELEM(T) T _v; memcpy(&_v, p + i * sizeof(T), sizeof(T))
    switch (a->type) {
    case GGUF_T_UINT8:   printf("%u", p[i]); break;
    case GGUF_T_INT8:    { ELEM(int8_t);   printf("%d", _v); } break;
    case GGUF_T_UINT16:  { ELEM(uint16_t); printf("%u", _v); } break;
    case GGUF_T_INT16:   { ELEM(int16_t);  printf("%d", _v); } break;
    case GGUF_T_UINT32:  { ELEM(uint32_t); printf("%" PRIu32, _v); } break;
    case GGUF_T_INT32:   { ELEM(int32_t);  printf("%" PRId32, _v); } break;
    case GGUF_T_UINT64:  { ELEM(uint64_t); printf("%" PRIu64, _v); } break;
    case GGUF_T_INT64:   { ELEM(int64_t);  printf("%" PRId64, _v); } break;
    case GGUF_T_FLOAT32: { ELEM(float);    printf("%g", (double)_v); } break;
    case GGUF_T_FLOAT64: { ELEM(double);   printf("%g", _v); } break;
    case GGUF_T_BOOL:    printf("%s", p[i] ? "true" : "false"); break;
    default:             printf("?"); break;
    }
#undef ELEM
}

static void print_meta(const gguf_t *g, uint64_t arr_show)
{
    printf("\n%smetadata%s  %s(%" PRIu64 " keys)%s\n",
           bold(), rst(), dim(), g->n_kv, rst());

    for (uint64_t i = 0; i < g->n_kv; i++) {
        const gguf_kv_t *kv = &g->kv[i];
        printf("  %-40.*s ", (int)kv->key.len, kv->key.ptr);

        if (kv->type == GGUF_T_STRING) {
            gguf_str_t s = kv->v.str;
            /* Chat templates are kilobytes long; summarize rather than flood. */
            if (s.len > 96) {
                printf("%s[string, %" PRIu64 " bytes]%s \"%.60s…\"\n",
                       dim(), s.len, rst(), s.ptr);
            } else {
                printf("\"%.*s\"\n", (int)s.len, s.ptr);
            }
        } else if (kv->type == GGUF_T_ARRAY) {
            const gguf_arr_t *a = &kv->v.arr;
            printf("%s[%s x %" PRIu64 "]%s ",
                   dim(), gguf_type_name(a->type), a->len, rst());
            uint64_t n = a->len < arr_show ? a->len : arr_show;
            printf("{");
            for (uint64_t j = 0; j < n; j++) {
                if (j) printf(", ");
                print_arr_elem(a, j);
            }
            if (a->len > n) printf(", %s…%s", dim(), rst());
            printf("}\n");
        } else {
            print_scalar(kv);
            printf("\n");
        }
    }
}

static void print_tensors(const gguf_t *g)
{
    printf("\n%stensors%s  %s(%" PRIu64 ")%s\n",
           bold(), rst(), dim(), g->n_tensors, rst());
    printf("  %s%-44s %-6s %-26s %14s%s\n",
           dim(), "name", "type", "shape", "bytes", rst());

    char buf[32];
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const gguf_tensor_t *t = &g->tensors[i];
        char shape[64];
        int n = 0;
        for (uint32_t d = 0; d < t->n_dims; d++)
            n += snprintf(shape + n, sizeof shape - (size_t)n, "%s%" PRIu64,
                          d ? " x " : "", t->dims[d]);

        printf("  %-44.*s %-6s %-26s %14s\n",
               (int)t->name.len, t->name.ptr,
               ggml_type_name(t->type), shape,
               t->nbytes ? arena_fmt_bytes(buf, sizeof buf, t->nbytes)
                         : "unsupported");
    }
}

static void print_summary(const gguf_t *g)
{
    /* Per-type totals. On a real Q4_K_M file this is the clearest picture of
     * the quantization mix — mostly Q4_K with Q6_K on the sensitive tensors. */
    uint64_t bytes_by_type[GGML_TYPE_MAX] = { 0 };
    uint64_t elems_by_type[GGML_TYPE_MAX] = { 0 };
    uint64_t count_by_type[GGML_TYPE_MAX] = { 0 };
    uint64_t total_bytes = 0, total_elems = 0, unsupported = 0;

    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const gguf_tensor_t *t = &g->tensors[i];
        if ((unsigned)t->type >= GGML_TYPE_MAX) { unsupported++; continue; }
        if (t->nbytes == 0) { unsupported++; continue; }
        bytes_by_type[t->type] += t->nbytes;
        elems_by_type[t->type] += t->n_elem;
        count_by_type[t->type] += 1;
        total_bytes += t->nbytes;
        total_elems += t->n_elem;
    }

    printf("\n%sweights by type%s\n", bold(), rst());
    printf("  %s%-8s %8s %16s %14s %10s%s\n",
           dim(), "type", "tensors", "parameters", "bytes", "bits/wt", rst());

    char buf[32];
    for (unsigned t = 0; t < GGML_TYPE_MAX; t++) {
        if (count_by_type[t] == 0) continue;
        double bpw = elems_by_type[t]
                   ? 8.0 * (double)bytes_by_type[t] / (double)elems_by_type[t]
                   : 0.0;
        printf("  %-8s %8" PRIu64 " %16" PRIu64 " %14s %10.2f\n",
               ggml_type_name((ggml_type_t)t), count_by_type[t],
               elems_by_type[t],
               arena_fmt_bytes(buf, sizeof buf, bytes_by_type[t]), bpw);
    }

    if (unsupported)
        printf("  %s%" PRIu64 " tensor(s) of unsupported type, not counted%s\n",
               dim(), unsupported, rst());

    double bpw = total_elems ? 8.0 * (double)total_bytes / (double)total_elems : 0.0;
    printf("  %s%-8s %8" PRIu64 " %16" PRIu64 " %14s %10.2f%s\n",
           bold(), "total", g->n_tensors, total_elems,
           arena_fmt_bytes(buf, sizeof buf, total_bytes), bpw, rst());

    /* Coverage check. If any block size in the type table were wrong, tensor
     * payloads would not tile the data section — they would overlap or leave a
     * gap. Reaching the end exactly is strong evidence the table is right. */
    uint64_t high = 0;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const gguf_tensor_t *t = &g->tensors[i];
        uint64_t end = t->offset + t->nbytes;
        if (end > high) high = end;
    }
    printf("\n  %stensor data: %" PRIu64 " bytes used of %" PRIu64 " in the section",
           dim(), high, (uint64_t)g->data_size);
    if (high == g->data_size) printf("  (exact tiling)%s\n", rst());
    else printf("  (%llu unaccounted)%s\n",
                (unsigned long long)(g->data_size - high), rst());


    /* The bandwidth ceiling. Same arithmetic in both cases — weights are read
     * once per unit of work — but the unit differs, and so does what it means.
     * For a causal LM the unit is a token and the ceiling is real: decode is
     * bandwidth-bound and no kernel beats it. For an image encoder the unit is
     * one image at batch 1, and batching amortizes the weight read away, which
     * is exactly why an encoder is compute-bound in practice. */
    if (total_bytes > 0) {
        const double bw_gbs = 68.0;   /* M1 (base) LPDDR4X, spec figure */
        double ceiling = bw_gbs * 1e9 / (double)total_bytes;
        gguf_str_t a;
        bool encoder = gguf_get_str(g, "general.architecture", &a) &&
                       (gguf_str_eq(a, "clip-vit") || gguf_str_eq(a, "clip-vit-fixtures"));
        printf("\n  %sroofline at %.0f GB/s: ", dim(), bw_gbs);
        if (encoder) {
            printf("%.1f img/s at batch 1%s\n", ceiling, rst());
            printf("  %s(batching amortizes the weight read — an encoder is "
                   "compute-bound in practice)%s\n", dim(), rst());
        } else {
            printf("%.1f tok/s decode%s\n", ceiling, rst());
            printf("  %s(weights only — KV cache traffic makes the real ceiling "
                   "lower)%s\n", dim(), rst());
        }
    }
}

static void usage(void)
{
    fprintf(stderr,
        "usage: tc-inspect <model.gguf> [--tensors] [--no-meta] [--arrays=N] [--no-color]\n");
}

int main(int argc, char **argv)
{
    const char *path = NULL;
    bool show_tensors = false, show_meta = true;
    uint64_t arr_show = 6;

    for (int i = 1; i < argc; i++) {
        const char *s = argv[i];
        if (strcmp(s, "--tensors") == 0)       show_tensors = true;
        else if (strcmp(s, "--no-meta") == 0)  show_meta = false;
        else if (strcmp(s, "--no-color") == 0) g_color = false;
        else if (strncmp(s, "--arrays=", 9) == 0) arr_show = strtoull(s + 9, NULL, 10);
        else if (s[0] == '-')                  { usage(); return 2; }
        else if (path == NULL)                 path = s;
        else                                   { usage(); return 2; }
    }
    if (path == NULL) { usage(); return 2; }

    /* 256 MiB is far more than any model's metadata needs; it is a reservation,
     * so the untouched remainder costs nothing. The vocab index dominates. */
    arena_t a = arena_create(256u << 20, "gguf-meta");
    if (!arena_ok(&a)) { fprintf(stderr, "tc: cannot reserve arena\n"); return 1; }

    gguf_t g;
    if (!gguf_open(&g, path, &a)) {
        fprintf(stderr, "tc: %s\n", g.err);
        arena_destroy(&a);
        return 1;
    }

    char buf[32], buf2[32];
    printf("%s%s%s\n", bold(), path, rst());
    printf("  gguf v%" PRIu32 "   alignment %" PRIu32
           "   file %s   tensor data %s\n",
           g.version, g.alignment,
           arena_fmt_bytes(buf, sizeof buf, g.map_size),
           arena_fmt_bytes(buf2, sizeof buf2, g.data_size));

    gguf_str_t arch;
    if (gguf_get_str(&g, "general.architecture", &arch)) {
        char a_[64];
        gguf_str_cstr(a_, sizeof a_, arch);
        printf("  architecture %s%s%s", bold(), a_, rst());

        /* Architecture-scoped keys are named "<arch>.<field>". */
        char k[128];
        uint64_t v;
        snprintf(k, sizeof k, "%s.block_count", a_);
        if (gguf_get_u64(&g, k, &v)) printf("   layers %" PRIu64, v);
        snprintf(k, sizeof k, "%s.embedding_length", a_);
        if (gguf_get_u64(&g, k, &v)) printf("   d_model %" PRIu64, v);
        snprintf(k, sizeof k, "%s.attention.head_count", a_);
        if (gguf_get_u64(&g, k, &v)) printf("   heads %" PRIu64, v);
        snprintf(k, sizeof k, "%s.attention.head_count_kv", a_);
        if (gguf_get_u64(&g, k, &v)) printf("   kv-heads %" PRIu64, v);
        snprintf(k, sizeof k, "%s.context_length", a_);
        if (gguf_get_u64(&g, k, &v)) printf("   ctx %" PRIu64, v);
        printf("\n");
    }

    if (show_meta)    print_meta(&g, arr_show);
    if (show_tensors) print_tensors(&g);
    print_summary(&g);

    printf("\n  %smetadata arena: %s used of %s reserved%s\n",
           dim(), arena_fmt_bytes(buf, sizeof buf, arena_used(&a)),
           arena_fmt_bytes(buf2, sizeof buf2, a.cap), rst());

    gguf_close(&g);
    arena_destroy(&a);
    return 0;
}
