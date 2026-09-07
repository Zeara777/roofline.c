#include "vit.h"

#include "dequant.h"
#include "ops.h"

#include <stdio.h>
#include <string.h>

#define ERR(m, ...) (snprintf((m)->err, sizeof (m)->err, __VA_ARGS__), false)

/* ---------------------------------------------------------------- loading */

/* Resolve one tensor to fp32 and check its element count. The count is checked
 * rather than the dim vector because ggml stores dims reversed and a rank-4
 * conv weight is legitimately read as a rank-2 matrix here; what must not
 * differ is how many numbers there are. */
static const float *want(tc_vit_t *m, arena_t *a, const char *name, size_t n_elem)
{
    const gguf_tensor_t *t = gguf_tensor(&m->gguf, name);
    if (t == NULL) {
        ERR(m, "missing tensor '%s'", name);
        return NULL;
    }
    if (t->n_elem != n_elem) {
        ERR(m, "tensor '%s' has %llu elements, expected %zu",
            name, (unsigned long long)t->n_elem, n_elem);
        return NULL;
    }
    bool converted = false;
    const float *p = tc_dequant_view(t, a, &converted);
    if (p == NULL) {
        ERR(m, "cannot materialize '%s' (%s)", name, ggml_type_name(t->type));
        return NULL;
    }
    if (converted) m->weight_bytes += n_elem * sizeof(float);
    return p;
}

#define WANT(field, name, n) do {                             \
        m->field = want(m, a, (name), (size_t)(n));           \
        if (m->field == NULL) { gguf_close(&m->gguf); return false; } \
    } while (0)

#define WANT_B(blk, field, name, n) do {                      \
        (blk)->field = want(m, a, (name), (size_t)(n));       \
        if ((blk)->field == NULL) { gguf_close(&m->gguf); return false; } \
    } while (0)

bool tc_vit_load(tc_vit_t *out, const char *path, arena_t *a)
{
    tc_vit_t *m = out;
    memset(m, 0, sizeof *m);

    if (!gguf_open(&m->gguf, path, a)) {
        snprintf(m->err, sizeof m->err, "%s", m->gguf.err);
        return false;
    }

    gguf_str_t arch;
    if (!gguf_get_str(&m->gguf, "general.architecture", &arch) ||
        !gguf_str_eq(arch, "clip-vit")) {
        gguf_close(&m->gguf);
        return ERR(m, "not a clip-vit file");
    }

    m->n_layer = (int)gguf_u64_or(&m->gguf, "clip-vit.block_count", 0);
    m->d_model = (int)gguf_u64_or(&m->gguf, "clip-vit.embedding_length", 0);
    m->n_head  = (int)gguf_u64_or(&m->gguf, "clip-vit.attention.head_count", 0);
    m->d_ffn   = (int)gguf_u64_or(&m->gguf, "clip-vit.feed_forward_length", 0);
    m->patch   = (int)gguf_u64_or(&m->gguf, "clip-vit.patch_size", 0);
    m->image   = (int)gguf_u64_or(&m->gguf, "clip-vit.image_size", 0);
    m->d_out   = (int)gguf_u64_or(&m->gguf, "clip-vit.projection_dim", 0);
    m->ln_eps  = gguf_f32_or(&m->gguf, "clip-vit.layer_norm_epsilon", 1e-5f);

    if (m->n_layer <= 0 || m->d_model <= 0 || m->n_head <= 0 || m->d_ffn <= 0 ||
        m->patch <= 0 || m->image <= 0 || m->d_out <= 0) {
        gguf_close(&m->gguf);
        return ERR(m, "incomplete architecture metadata");
    }
    if (m->d_model % m->n_head != 0) {
        gguf_close(&m->gguf);
        return ERR(m, "d_model %d not divisible by %d heads", m->d_model, m->n_head);
    }
    if (m->image % m->patch != 0) {
        gguf_close(&m->gguf);
        return ERR(m, "image %d not divisible by patch %d", m->image, m->patch);
    }
    m->head_dim = m->d_model / m->n_head;
    m->grid     = m->image / m->patch;
    m->n_token  = m->grid * m->grid + 1;

    const uint64_t ctx = gguf_u64_or(&m->gguf, "clip-vit.context_length", 0);
    if (ctx != 0 && ctx != (uint64_t)m->n_token) {
        gguf_close(&m->gguf);
        return ERR(m, "context_length %llu but grid implies %d tokens",
                   (unsigned long long)ctx, m->n_token);
    }

    /* Which GELU. This is not defaulted: open_clip_config.json does not carry
     * it, the two forms differ by ~2e-2 at x=-2, and picking the wrong one
     * yields a plausible embedding that is silently wrong. The exporter probes
     * the checkpoint and records the answer; a file without it is a file whose
     * arithmetic is not determined, so it is refused. */
    gguf_str_t act;
    if (!gguf_get_str(&m->gguf, "clip-vit.activation", &act)) {
        gguf_close(&m->gguf);
        return ERR(m, "no clip-vit.activation — re-export; GELU form must not be guessed");
    }
    if (gguf_str_eq(act, "gelu")) {
        m->quick_gelu = false;
    } else if (gguf_str_eq(act, "gelu_quick")) {
        m->quick_gelu = true;
    } else {
        char b[64];
        gguf_str_cstr(b, sizeof b, act);
        gguf_close(&m->gguf);
        return ERR(m, "unsupported activation '%s'", b);
    }

    /* Pooling. Only CLS is implemented; anything else changes what the final
     * projection consumes, so it fails rather than silently pooling wrongly. */
    gguf_str_t pool;
    if (gguf_get_str(&m->gguf, "clip-vit.pooling", &pool) && !gguf_str_eq(pool, "cls")) {
        char b[64];
        gguf_str_cstr(b, sizeof b, pool);
        gguf_close(&m->gguf);
        return ERR(m, "pooling '%s' not implemented (only cls)", b);
    }

    const gguf_kv_t *kv = gguf_find(&m->gguf, "clip-vit.preprocess.mean");
    if (kv && kv->type == GGUF_T_ARRAY && kv->v.arr.len == 3 &&
        kv->v.arr.type == GGUF_T_FLOAT32) {
        memcpy(m->mean, kv->v.arr.data, sizeof m->mean);
    }
    kv = gguf_find(&m->gguf, "clip-vit.preprocess.std");
    if (kv && kv->type == GGUF_T_ARRAY && kv->v.arr.len == 3 &&
        kv->v.arr.type == GGUF_T_FLOAT32) {
        memcpy(m->std, kv->v.arr.data, sizeof m->std);
    }

    /* size_t, not int: these only ever appear in element-count arithmetic, and
     * an fp32 ViT-L already puts 3*D*D past 3 million per block. */
    const size_t D = (size_t)m->d_model, F = (size_t)m->d_ffn,
                 T = (size_t)m->n_token, P = (size_t)m->patch;

    WANT(patch_embd, "v.patch_embd.weight", D * 3u * P * P);
    WANT(cls_token,  "v.cls_token",         D);
    WANT(pos_embd,   "v.pos_embd",          T * D);
    WANT(ln_pre_g,   "v.ln_pre.weight",     D);
    WANT(ln_pre_b,   "v.ln_pre.bias",       D);
    WANT(ln_post_g,  "v.ln_post.weight",    D);
    WANT(ln_post_b,  "v.ln_post.bias",      D);
    WANT(proj,       "v.proj",              D * (size_t)m->d_out);

    m->blocks = ARENA_ARRAY(a, tc_vit_block_t, m->n_layer);
    for (int i = 0; i < m->n_layer; i++) {
        tc_vit_block_t *b = &m->blocks[i];
        char n[64];
#define NM(suffix) (snprintf(n, sizeof n, "v.blk.%d." suffix, i), n)
        WANT_B(b, ln1_g,  NM("ln1.weight"),       D);
        WANT_B(b, ln1_b,  NM("ln1.bias"),         D);
        WANT_B(b, qkv_w,  NM("attn_qkv.weight"),  3u * D * D);
        WANT_B(b, qkv_b,  NM("attn_qkv.bias"),    3u * D);
        WANT_B(b, out_w,  NM("attn_out.weight"),  D * D);
        WANT_B(b, out_b,  NM("attn_out.bias"),    D);
        WANT_B(b, ln2_g,  NM("ln2.weight"),       D);
        WANT_B(b, ln2_b,  NM("ln2.bias"),         D);
        WANT_B(b, up_w,   NM("ffn_up.weight"),    F * D);
        WANT_B(b, up_b,   NM("ffn_up.bias"),      F);
        WANT_B(b, down_w, NM("ffn_down.weight"),  D * F);
        WANT_B(b, down_b, NM("ffn_down.bias"),    D);
#undef NM
    }
    return true;
}

void tc_vit_free(tc_vit_t *m)
{
    gguf_close(&m->gguf);
}

/* ---------------------------------------------------------------- forward */

size_t tc_vit_scratch_bytes(const tc_vit_t *m)
{
    const size_t T = (size_t)m->n_token, D = (size_t)m->d_model, F = (size_t)m->d_ffn;
    /* x, h, q, k, v, attn  (T*D each)  +  ffn (T*F)  +  softmax scratch (T) */
    return (6 * T * D + T * F + T) * sizeof(float) + 8 * ARENA_DEFAULT_ALIGN;
}

/* The conv is a matmul in disguise: with stride == kernel the patches do not
 * overlap, so each output token is one dot product against a flattened patch.
 * The flattening order is the only thing that can go wrong, and it is fixed by
 * the weight: conv1.weight is [out, C, kh, kw] row-major, so the patch vector
 * must run channel-major with kw fastest. Token order is grid-row-major with
 * gx fastest, matching open_clip's reshape(N, width, -1).permute(0, 2, 1). */
static void im2patch(const tc_vit_t *m, const float *px, float *patches)
{
    const int P = m->patch, G = m->grid, S = m->image;
    const size_t plane = (size_t)S * (size_t)S;
    const size_t pdim  = (size_t)3 * (size_t)P * (size_t)P;

    for (int gy = 0; gy < G; gy++) {
        for (int gx = 0; gx < G; gx++) {
            float *dst = patches + ((size_t)gy * (size_t)G + (size_t)gx) * pdim;
            for (int c = 0; c < 3; c++) {
                for (int ky = 0; ky < P; ky++) {
                    const float *row = px + (size_t)c * plane
                                          + (size_t)(gy * P + ky) * (size_t)S
                                          + (size_t)(gx * P);
                    memcpy(dst, row, (size_t)P * sizeof(float));
                    dst += P;
                }
            }
        }
    }
}

bool tc_vit_encode(const tc_vit_t *m, const float *pixels, float *out,
                   arena_t *scratch, tc_vit_tap_fn tap, void *ud)
{
    /* size_t for the same reason as in the loader; the int copies below are
     * only for the kernel signatures, which take dimensions as int. */
    const size_t T = (size_t)m->n_token, D = (size_t)m->d_model,
                 F = (size_t)m->d_ffn,   P = (size_t)m->patch;
    const int Ti = m->n_token, Di = m->d_model, Fi = m->d_ffn;
    const size_t mark = arena_mark(scratch);

#define ALLOC(n) (float *)arena_alloc(scratch, (n) * sizeof(float), ARENA_DEFAULT_ALIGN)
    float *x    = ALLOC(T * D);
    float *h    = ALLOC(T * D);
    float *q    = ALLOC(T * D);
    float *k    = ALLOC(T * D);
    float *v    = ALLOC(T * D);
    float *attn = ALLOC(T * D);
    float *ffn  = ALLOC(T * F);
    float *sm   = ALLOC(T);
#undef ALLOC
    if (sm == NULL) {
        arena_release(scratch, mark);
        return false;
    }

    /* ---- patch embed. The projection writes straight to x + D, so the CLS
     * row it leaves untouched at x[0..D) is filled in next. */
    {
        const size_t pdim = 3u * P * P;
        float *patches = (float *)arena_alloc(scratch,
                             (T - 1) * pdim * sizeof(float), ARENA_DEFAULT_ALIGN);
        if (patches == NULL) {
            arena_release(scratch, mark);
            return false;
        }
        im2patch(m, pixels, patches);
        tc_ref_linear(x + D, patches, m->patch_embd, NULL, Ti - 1, Di, (int)pdim);
        /* patches is dead now, but the arena is LIFO and x sits below it, so
         * it stays allocated until the release below. 588 KB, once. */
    }

    memcpy(x, m->cls_token, D * sizeof(float));
    tc_ref_add(x, x, m->pos_embd, T * D);

    tc_ref_layernorm(x, x, m->ln_pre_g, m->ln_pre_b, Ti, Di, m->ln_eps);
    if (tap) tap(ud, "ln_pre", x, Ti, Di);

    for (int i = 0; i < m->n_layer; i++) {
        const tc_vit_block_t *b = &m->blocks[i];

        tc_ref_layernorm(h, x, b->ln1_g, b->ln1_b, Ti, Di, m->ln_eps);

        /* The fused qkv weight is [3D, D] row-major, so its three D-row blocks
         * ARE Wq, Wk and Wv laid out exactly as separate nn.Linear weights.
         * Three calls, no gather: heads then sit contiguous within each row,
         * which is the layout tc_ref_mha documents. */
        const size_t blk = D * D;
        tc_ref_linear(q, h, b->qkv_w + 0 * blk, b->qkv_b + 0 * D, Ti, Di, Di);
        tc_ref_linear(k, h, b->qkv_w + 1 * blk, b->qkv_b + 1 * D, Ti, Di, Di);
        tc_ref_linear(v, h, b->qkv_w + 2 * blk, b->qkv_b + 2 * D, Ti, Di, Di);

        tc_ref_mha(attn, q, k, v, Ti, m->n_head, m->head_dim, sm);
        tc_ref_linear(h, attn, b->out_w, b->out_b, Ti, Di, Di);
        tc_ref_add(x, x, h, T * D);

        tc_ref_layernorm(h, x, b->ln2_g, b->ln2_b, Ti, Di, m->ln_eps);
        tc_ref_linear(ffn, h, b->up_w, b->up_b, Ti, Fi, Di);
        if (m->quick_gelu) tc_ref_gelu_quick(ffn, ffn, T * F);
        else               tc_ref_gelu(ffn, ffn, T * F);
        tc_ref_linear(h, ffn, b->down_w, b->down_b, Ti, Di, Fi);
        tc_ref_add(x, x, h, T * D);

        if (tap && i == 0)              tap(ud, "blk0", x, Ti, Di);
        if (tap && i == m->n_layer - 1) tap(ud, "blk_last", x, Ti, Di);
    }

    tc_ref_layernorm(x, x, m->ln_post_g, m->ln_post_b, Ti, Di, m->ln_eps);
    if (tap) tap(ud, "ln_post", x, Ti, Di);

    /* CLS row only. v.proj is [D, d_out] — a bare nn.Parameter used as
     * `pooled @ proj`, NOT an nn.Linear weight, so it is NOT transposed. */
    tc_ref_gemm(out, x, m->proj, 1, m->d_out, Di, false);
    if (tap) tap(ud, "embedding", out, 1, m->d_out);

    arena_release(scratch, mark);
    return true;
}
