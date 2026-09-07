/* vit.h — CLIP vision transformer, fp32 forward pass.
 *
 * The encoder half of a CLIP model: pixels in, one embedding vector out. No
 * tokenizer, no KV cache, no autoregressive loop, no sampling — every shape is
 * fixed the moment the file is opened. That is what makes it the right first
 * forward pass to write, and it exercises the compute-bound (prefill) side of
 * the engine rather than the bandwidth-bound (decode) side.
 *
 * Structure, per open_clip's VisionTransformer:
 *
 *   patch embed (conv 14x14 stride 14, no bias)  ->  256 tokens of width 1024
 *   prepend CLS, add learned position embedding
 *   ln_pre
 *   24 x  [ ln1 -> fused-qkv MHA -> attn_out -> +residual
 *           ln2 -> ffn_up -> GELU -> ffn_down -> +residual ]
 *   ln_post, take the CLS row, project to 768
 *
 * Pre-LN, bidirectional (no mask), no LayerScale, no attention pooling. Those
 * are properties of THIS checkpoint, verified against it, and the loader
 * refuses a file whose metadata says otherwise rather than quietly doing the
 * wrong arithmetic.
 */
#ifndef TC_VIT_H
#define TC_VIT_H

#include <stdbool.h>
#include <stddef.h>

#include "arena.h"
#include "gguf.h"

/* Weight layouts below are PyTorch's, used exactly as stored — see ops.h.
 * "[out, in]" means an nn.Linear weight, consumed by tc_ref_linear directly. */
typedef struct {
    const float *ln1_g, *ln1_b;
    const float *qkv_w,  *qkv_b;    /* [3D, D], [3D] — fused, split by row  */
    const float *out_w,  *out_b;    /* [D, D],  [D]                         */
    const float *ln2_g, *ln2_b;
    const float *up_w,   *up_b;     /* [F, D],  [F]                         */
    const float *down_w, *down_b;   /* [D, F],  [D]                         */
} tc_vit_block_t;

typedef struct {
    int   n_layer, d_model, n_head, head_dim, d_ffn;
    int   patch, image, grid, n_token, d_out;
    float ln_eps;
    bool  quick_gelu;               /* from the file, never assumed         */
    float mean[3], std[3];          /* preprocessing the C side must match  */

    const float *patch_embd;        /* [D, 3*P*P] — the conv as a linear    */
    const float *cls_token;         /* [D]                                  */
    const float *pos_embd;          /* [T, D]                               */
    const float *ln_pre_g,  *ln_pre_b;
    const float *ln_post_g, *ln_post_b;
    const float *proj;              /* [D, d_out] — NOT nn.Linear order     */
    tc_vit_block_t *blocks;

    gguf_t gguf;
    size_t weight_bytes;            /* arena spent materializing non-F32    */
    char   err[256];
} tc_vit_t;

/* Open `path` and resolve every weight. Weights and metadata come from `a`,
 * which must outlive the model. Any missing tensor, unexpected shape or
 * unhandled metadata value fails here with a reason in out->err — a model that
 * loads is a model whose arithmetic is fully determined. */
bool tc_vit_load(tc_vit_t *out, const char *path, arena_t *a);
void tc_vit_free(tc_vit_t *m);

/* Intermediate activations, for the golden-vector gate and for debugging a
 * divergence to the block that caused it. `rows`x`cols` is the tap's shape. */
typedef void (*tc_vit_tap_fn)(void *ud, const char *name,
                              const float *data, int rows, int cols);

/* Encode one image. `pixels` is [3, image, image] fp32, ALREADY normalized —
 * preprocessing is deliberately not done here, so a fixture can be fed in
 * unchanged and a preprocessing bug can never be mistaken for a kernel bug.
 * `out` receives d_out floats, unnormalized. `scratch` is used and released.
 * `tap` may be NULL. */
bool tc_vit_encode(const tc_vit_t *m, const float *pixels, float *out,
                   arena_t *scratch, tc_vit_tap_fn tap, void *ud);

/* Activations for one image, in bytes — what tc_vit_encode needs from scratch. */
size_t tc_vit_scratch_bytes(const tc_vit_t *m);

#endif /* TC_VIT_H */
