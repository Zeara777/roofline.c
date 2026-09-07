/* Golden-vector gate for the ViT forward pass.
 *
 * Runs the C encoder on the exact pixels PyTorch saw and compares four taps
 * against PyTorch's own activations. The taps are placed so a failure names
 * the stage that caused it: ln_pre isolates patch embed + CLS + position,
 * blk0 isolates one block, blk_last isolates the other 23, ln_post and the
 * embedding isolate pooling and the projection.
 *
 * Gate against the F32 export, not the F16 one. The fixtures were produced by
 * PyTorch from fp32 weights, so an F16 file would fold weight rounding into
 * every tolerance and leave no error budget that means "kernel bug". What F16
 * costs is a measurement of its own — tools/tc_encode.c reports it — not
 * something to hide inside a correctness threshold.
 *
 * The model is 1.2 GB and not in the repo, so this SKIPS when it is absent
 * rather than failing: a fresh clone must still go green on `make test`.
 */
#include "tt.h"
#include "arena.h"
#include "gguf.h"
#include "vit.h"

#include <math.h>
#include <stdlib.h>

#define MODEL_ENV    "ROOFLINE_VIT_MODEL"
#define MODEL_PATH   "bioclip2-vit-l14-f32.gguf"
#define FIXTURE_ENV  "ROOFLINE_VIT_FIXTURES"
#define FIXTURE_PATH "bioclip2-vit-l14-f32-fixtures.gguf"

static gguf_t FX;

static const float *fixture(const char *name, uint64_t *n_out)
{
    const gguf_tensor_t *t = gguf_tensor(&FX, name);
    if (t == NULL || t->type != GGML_TYPE_F32) {
        fprintf(stderr, "  missing or non-F32 fixture '%s'\n", name);
        exit(1);
    }
    if (n_out) *n_out = t->n_elem;
    return (const float *)t->data;
}

/* Compare against the reference and report the worst error seen.
 *
 * Both bounds are checked because either alone lies. A pure atol passes a
 * catastrophically wrong small value; a pure rtol fails on values that are
 * legitimately near zero, where fp32 has no relative precision to offer. */
static void cmp(const char *what, const float *got, const float *want_,
                size_t n, float atol, float rtol)
{
    double worst_abs = 0.0, worst_rel = 0.0, sum_sq = 0.0;
    size_t at = 0;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(got[i])) {
            fprintf(stderr, "  \033[31mFAIL\033[0m %s: non-finite at %zu\n", what, i);
            tt_fail++; tt_ran++;
            return;
        }
        double g = (double)got[i], w = (double)want_[i];
        double a = fabs(g - w), scale = fabs(w);
        if (a > worst_abs) { worst_abs = a; at = i; }
        if (scale > 1e-3 && a / scale > worst_rel) worst_rel = a / scale;
        sum_sq += w * w;
    }
    double rms = sqrt(sum_sq / (double)n);
    printf("     %-10s  max|abs| %.3e  max|rel| %.3e  (rms %.2f, worst at %zu:"
           " %.6f vs %.6f)\n",
           what, worst_abs, worst_rel, rms, at, (double)got[at], (double)want_[at]);

    tt_ran++;
    if (!(worst_abs <= (double)atol || worst_rel <= (double)rtol)) {
        tt_fail++;
        fprintf(stderr, "  \033[31mFAIL\033[0m %s: abs %.3e > %.3e AND rel %.3e > %.3e\n",
                what, worst_abs, (double)atol, worst_rel, (double)rtol);
    }
}

/* Tolerances. These are not "close enough" numbers pulled from the air: they
 * are the error a correct fp32 kernel is ENTITLED to, growing down the stack
 * because a residual stream accumulates it.
 *
 *   A dot product over K terms in fp32 accumulates ~sqrt(K)*eps relative error
 *   in the sequential order used here (eps = 1.19e-7). For K=1024 that is
 *   ~4e-6 relative, on activations whose rms grows from ~1 after ln_pre to
 *   ~10 by the last block, and PyTorch reduces in a DIFFERENT order (blocked,
 *   multithreaded), so the two disagree at exactly that scale.
 *
 * Each atol below is ~10x the error MEASURED against this fixture on arm64,
 * not a round number chosen to pass. A real kernel bug is never a factor of
 * ten out, it is a factor of everything, so 10x leaves no room to hide one.
 *
 * The measured values, for whoever has to move these later:
 *   ln_pre 3.8e-6   blk0 1.3e-5   blk_last 4.7e-4   ln_post 3.5e-4   emb 2.5e-5
 *
 * ⚠️ Cross-ISA: clang contracts a*b+c into an FMA where the hardware has one,
 * so x86-without-FMA and arm64 do not produce bit-identical results. That
 * difference is far below these bounds, but if this trips on the Linux box it
 * is evidence, not a flake — read the numbers before widening anything. */
static const struct { const char *name; float atol, rtol; } TOL[] = {
    { "ln_pre",    5e-5f, 1e-4f },
    { "blk0",      1.5e-4f, 1e-4f },
    { "blk_last",  5e-3f, 1e-3f },
    { "ln_post",   4e-3f, 1e-3f },
    { "embedding", 3e-4f, 1e-3f },
};

static void check_tap(void *ud, const char *name, const float *data, int rows, int cols)
{
    (void)ud;
    char key[64];
    snprintf(key, sizeof key, "ref.%s", name);
    uint64_t n = 0;
    const float *ref = fixture(key, &n);
    CHECK_EQ_U(n, (uint64_t)rows * (uint64_t)cols);

    for (size_t i = 0; i < sizeof TOL / sizeof TOL[0]; i++) {
        if (strcmp(TOL[i].name, name) == 0) {
            cmp(name, data, ref, (size_t)rows * (size_t)cols, TOL[i].atol, TOL[i].rtol);
            return;
        }
    }
    fprintf(stderr, "  no tolerance defined for tap '%s'\n", name);
    exit(1);
}

int main(void)
{
    const char *model = getenv(MODEL_ENV);
    const char *fxpath = getenv(FIXTURE_ENV);
    if (model == NULL)  model  = MODEL_PATH;
    if (fxpath == NULL) fxpath = FIXTURE_PATH;

    FILE *probe = fopen(model, "rb");
    if (probe == NULL) {
        printf("   \033[33mSKIP\033[0m %s not present — export it with\n"
               "        .venv/bin/python tools/export_vit_gguf.py --dtype f32 "
               "--out %s\n", model, MODEL_PATH);
        return 0;
    }
    fclose(probe);

    /* F32 weights are used straight out of the mapping, so this arena only
     * holds metadata and tensor descriptors. An F16 model materializes to fp32
     * and needs ~1.3 GB here instead — hence the generous reservation, which
     * costs nothing until it is written. */
    arena_t weights = arena_create(2ull << 30, "weights");
    arena_t scratch = arena_create(64ull << 20, "scratch");
    CHECK(arena_ok(&weights));
    CHECK(arena_ok(&scratch));

    tc_vit_t m;
    if (!tc_vit_load(&m, model, &weights)) {
        fprintf(stderr, "  \033[31mFAIL\033[0m load: %s\n", m.err);
        return 1;
    }
    CHECK_EQ_U(m.n_layer, 24);
    CHECK_EQ_U(m.d_model, 1024);
    CHECK_EQ_U(m.n_head, 16);
    CHECK_EQ_U(m.head_dim, 64);
    CHECK_EQ_U(m.d_ffn, 4096);
    CHECK_EQ_U(m.n_token, 257);
    CHECK_EQ_U(m.d_out, 768);
    CHECK(m.quick_gelu == false);      /* probed off the checkpoint, not assumed */

    arena_t fxarena = arena_create(16ull << 20, "fixtures");
    if (!gguf_open(&FX, fxpath, &fxarena)) {
        fprintf(stderr, "  \033[31mFAIL\033[0m fixtures %s: %s\n", fxpath, FX.err);
        return 1;
    }

    uint64_t npx = 0;
    const float *px = fixture("input.pixels", &npx);
    CHECK_EQ_U(npx, (uint64_t)3 * 224 * 224);

    float *emb = (float *)malloc((size_t)m.d_out * sizeof(float));
    CHECK(emb != NULL);
    if (emb == NULL) return 1;

    printf("   encoding (scalar reference kernels, one image)...\n");
    CHECK(tc_vit_encode(&m, px, emb, &scratch, check_tap, NULL));

    /* Cosine against the normalized reference. This is the number that
     * actually matters downstream: the prototype classifier compares cosines,
     * so an embedding that is off in scale but not direction still classifies
     * identically, and one that drifts in direction does not. */
    uint64_t nrm = 0;
    const float *ref_n = fixture("ref.embedding_normalized", &nrm);
    CHECK_EQ_U(nrm, (uint64_t)m.d_out);
    double dot = 0.0, mag = 0.0;
    for (int i = 0; i < m.d_out; i++) {
        dot += (double)emb[i] * (double)ref_n[i];
        mag += (double)emb[i] * (double)emb[i];
    }
    double cosine = dot / sqrt(mag);
    printf("     cosine vs PyTorch: %.9f   (norm %.6f)\n", cosine, sqrt(mag));
    CHECK(cosine >= 0.9999);

    char b[32];
    printf("   weights arena %s (%s materialized), scratch peak %s\n",
           arena_fmt_bytes(b, sizeof b, arena_used(&weights)),
           m.weight_bytes ? "f16->f32" : "zero-copy",
           arena_fmt_bytes((char[32]){0}, 32, arena_peak(&scratch)));

    free(emb);
    tc_vit_free(&m);
    gguf_close(&FX);
    arena_destroy(&weights);
    arena_destroy(&scratch);
    arena_destroy(&fxarena);
    TT_DONE();
}
