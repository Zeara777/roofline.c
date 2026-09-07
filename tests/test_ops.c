/* Golden-vector gate for the reference kernels.
 *
 * Fixtures are a GGUF file written by tools/gen_op_fixtures.py, so this reads
 * them with the loader the engine already has rather than a second parser.
 * References are computed in float64 by numpy and stored as float32; the
 * tolerances below therefore cover fp32 rounding in the C kernels and nothing
 * else. They were set from the measured errors, not guessed. */
#include "tt.h"
#include "arena.h"
#include "gguf.h"
#include "ops.h"

#include <math.h>
#include <stdint.h>

static gguf_t G;
static arena_t A;

static const float *T(const char *name, uint64_t *n_out)
{
    const gguf_tensor_t *t = gguf_tensor(&G, name);
    if (t == NULL || t->type != GGML_TYPE_F32) {
        fprintf(stderr, "  missing or non-F32 fixture tensor '%s'\n", name);
        exit(1);
    }
    if (n_out) *n_out = t->n_elem;
    return (const float *)t->data;
}

static int U(const char *key)
{
    return (int)gguf_u64_or(&G, key, 0);
}

/* Report the worst absolute and relative error, and check both. */
static void cmp(const char *what, const float *got, const float *want,
                size_t n, float atol, float rtol)
{
    double worst_abs = 0.0, worst_rel = 0.0;
    size_t at = 0;
    for (size_t i = 0; i < n; i++) {
        double a = fabs((double)got[i] - (double)want[i]);
        double scale = fabs((double)want[i]);
        double r = scale > 1e-6 ? a / scale : 0.0;
        if (a > worst_abs) { worst_abs = a; at = i; }
        if (r > worst_rel) worst_rel = r;
        if (!isfinite(got[i])) {
            fprintf(stderr, "  \033[31mFAIL\033[0m %s: non-finite at %zu\n", what, i);
            exit(1);
        }
    }
    tt_ran++;
    if (worst_abs > (double)atol && worst_rel > (double)rtol) {
        tt_fail++;
        fprintf(stderr, "  \033[31mFAIL\033[0m %-16s abs %.3g (at %zu) rel %.3g"
                        "  [atol %.g rtol %.g]\n",
                what, worst_abs, at, worst_rel, (double)atol, (double)rtol);
    } else {
        printf("   %-16s abs %.3g  rel %.3g\n", what, worst_abs, worst_rel);
    }
}

int main(void)
{
    A = arena_create(64u << 20, "test-ops");
    CHECK(arena_ok(&A));
    if (!gguf_open(&G, "tests/fixtures/ops.gguf", &A)) {
        fprintf(stderr, "  %s\n  (run: .venv/bin/python tools/gen_op_fixtures.py)\n", G.err);
        return 1;
    }

    float *out = (float *)arena_alloc_or_die(&A, (size_t)4 << 20, 64);

    /* ---- gemm ---- */
    const char *tags[] = { "small", "wide" };
    for (int i = 0; i < 2; i++) {
        char k[64];
        snprintf(k, sizeof k, "gemm.%s.M", tags[i]); int M = U(k);
        snprintf(k, sizeof k, "gemm.%s.N", tags[i]); int N = U(k);
        snprintf(k, sizeof k, "gemm.%s.K", tags[i]); int K = U(k);
        snprintf(k, sizeof k, "gemm.%s.A", tags[i]); const float *a = T(k, NULL);
        snprintf(k, sizeof k, "gemm.%s.B", tags[i]); const float *b = T(k, NULL);
        snprintf(k, sizeof k, "gemm.%s.C", tags[i]); const float *c = T(k, NULL);
        tc_ref_gemm(out, a, b, M, N, K, true);
        snprintf(k, sizeof k, "gemm %s", tags[i]);
        /* K=1024 accumulates ~1e-4 absolute in fp32; that is the kernel being
         * correct, not wrong, which is why both tolerances must be exceeded. */
        cmp(k, out, c, (size_t)M * (size_t)N, 2e-4f, 1e-5f);
    }

    /* ---- linear (with bias, PyTorch weight layout) ---- */
    {
        int M = U("linear.M"), N = U("linear.N"), K = U("linear.K");
        tc_ref_linear(out, T("linear.x", NULL), T("linear.W", NULL),
                      T("linear.b", NULL), M, N, K);
        cmp("linear", out, T("linear.y", NULL), (size_t)M * (size_t)N, 1e-5f, 1e-5f);
    }

    /* ---- layernorm (one row offset by 1e4 — the one-pass variance trap) ---- */
    {
        int rows = U("ln.rows"), cols = U("ln.cols");
        float eps = gguf_f32_or(&G, "ln.eps", 1e-5f);
        tc_ref_layernorm(out, T("ln.x", NULL), T("ln.gamma", NULL),
                         T("ln.beta", NULL), rows, cols, eps);
        cmp("layernorm", out, T("ln.out", NULL), (size_t)rows * (size_t)cols, 1e-4f, 1e-4f);
    }

    /* ---- gelu, both flavours ---- */
    {
        uint64_t n;
        const float *x = T("gelu.x", &n);
        tc_ref_gelu(out, x, n);
        cmp("gelu exact", out, T("gelu.exact", NULL), n, 1e-6f, 1e-6f);
        tc_ref_gelu_quick(out, x, n);
        cmp("gelu quick", out, T("gelu.quick", NULL), n, 1e-6f, 1e-6f);
        /* The two must genuinely differ, or a config mix-up would go unnoticed. */
        tc_ref_gelu(out, x, n);
        float *q = out + n;
        tc_ref_gelu_quick(q, x, n);
        double biggest = 0.0;
        for (uint64_t i = 0; i < n; i++) {
            double d = fabs((double)out[i] - (double)q[i]);
            if (d > biggest) biggest = d;
        }
        CHECK(biggest > 0.01);
    }

    /* ---- softmax (one row offset by 800 — naive exp overflows) ---- */
    {
        int rows = U("softmax.rows"), cols = U("softmax.cols");
        tc_ref_softmax_rows(out, T("softmax.x", NULL), rows, cols);
        cmp("softmax", out, T("softmax.out", NULL), (size_t)rows * (size_t)cols, 1e-6f, 1e-5f);
        for (int r = 0; r < rows; r++) {           /* each row must sum to 1 */
            double s = 0.0;
            for (int c = 0; c < cols; c++)
                s += (double)out[(size_t)r * (size_t)cols + (size_t)c];
            CHECK(fabs(s - 1.0) < 1e-5);
        }
    }

    /* ---- bidirectional multi-head attention ---- */
    {
        int Tn = U("mha.T"), H = U("mha.n_head"), HD = U("mha.head_dim");
        float *scratch = (float *)arena_alloc_or_die(&A, (size_t)Tn * sizeof(float), 64);
        tc_ref_mha(out, T("mha.q", NULL), T("mha.k", NULL), T("mha.v", NULL),
                   Tn, H, HD, scratch);
        cmp("mha", out, T("mha.out", NULL), (size_t)Tn * (size_t)H * (size_t)HD, 1e-5f, 1e-5f);
    }

    gguf_close(&G);
    arena_destroy(&A);
    TT_DONE();
}
