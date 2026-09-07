#include "ops.h"

#include <math.h>

void tc_ref_gemm(float *C, const float *A, const float *B,
                 int M, int N, int K, bool b_transposed)
{
    /* Dimensions arrive as int because that is what reads clearly at the call
     * site, but every index is computed in size_t: M*N on a real model
     * overflows a 32-bit int long before it overflows memory. */
    const size_t Ms = (size_t)M, Ns = (size_t)N, Ks = (size_t)K;

    for (size_t m = 0; m < Ms; m++) {
        for (size_t n = 0; n < Ns; n++) {
            float acc = 0.0f;
            if (b_transposed) {
                const float *brow = B + n * Ks;             /* B is [N,K] */
                for (size_t k = 0; k < Ks; k++) acc += A[m * Ks + k] * brow[k];
            } else {
                for (size_t k = 0; k < Ks; k++)             /* B is [K,N] */
                    acc += A[m * Ks + k] * B[k * Ns + n];
            }
            C[m * Ns + n] = acc;
        }
    }
}

void tc_ref_linear(float *y, const float *x, const float *W, const float *bias,
                   int M, int N, int K)
{
    tc_ref_gemm(y, x, W, M, N, K, true);
    if (bias != NULL)
        for (size_t m = 0; m < (size_t)M; m++)
            for (size_t n = 0; n < (size_t)N; n++) y[m * (size_t)N + n] += bias[n];
}

void tc_ref_layernorm(float *out, const float *x, const float *gamma,
                      const float *beta, int rows, int cols, float eps)
{
    for (size_t r = 0; r < (size_t)rows; r++) {
        const float *in = x + r * (size_t)cols;
        float *o = out + r * (size_t)cols;

        /* Reductions here accumulate in double, unlike matmul. The two cases
         * are not the same problem:
         *
         *   matmul accumulates over K in fp32 because an optimized SIMD kernel
         *   accumulates in fp32 lanes, and a double-accumulating reference
         *   would make every correct fast kernel look broken.
         *
         *   a norm reduces over one modest row, has no fast-kernel equivalent
         *   to be fair to, and is precision-critical: with a mean near 1e4 the
         *   fp32 sum of 64 terms already carries ~1e-3 of error, which is the
         *   entire spread of the data. ggml reaches the same conclusion — its
         *   reduction accumulator is a double.
         *
         * ViT post-LayerNorm activations carry exactly these large-offset
         * outliers, so this is the live case, not a synthetic one. */
        double sum = 0.0;
        for (int c = 0; c < cols; c++) sum += (double)in[c];
        double mean = sum / (double)cols;

        /* Two-pass. The one-pass sum-of-squares form loses catastrophically
         * when the mean is large relative to the spread. */
        double var = 0.0;
        for (int c = 0; c < cols; c++) {
            double d = (double)in[c] - mean;
            var += d * d;
        }
        var /= (double)cols;                /* biased, as PyTorch does */

        float inv = (float)(1.0 / sqrt(var + (double)eps));
        for (int c = 0; c < cols; c++) {
            float v = (float)((double)in[c] - mean) * inv;
            if (gamma) v *= gamma[c];
            if (beta)  v += beta[c];
            o[c] = v;
        }
    }
}

void tc_ref_gelu(float *out, const float *x, size_t n)
{
    const float inv_sqrt2 = 0.70710678118654752440f;
    for (size_t i = 0; i < n; i++)
        out[i] = 0.5f * x[i] * (1.0f + erff(x[i] * inv_sqrt2));
}

void tc_ref_gelu_quick(float *out, const float *x, size_t n)
{
    for (size_t i = 0; i < n; i++)
        out[i] = x[i] / (1.0f + expf(-1.702f * x[i]));
}

void tc_ref_softmax_rows(float *out, const float *x, int rows, int cols)
{
    for (size_t r = 0; r < (size_t)rows; r++) {
        const float *in = x + r * (size_t)cols;
        float *o = out + r * (size_t)cols;

        float mx = in[0];
        for (int c = 1; c < cols; c++) if (in[c] > mx) mx = in[c];

        double sum = 0.0;                   /* a reduction, as above */
        for (int c = 0; c < cols; c++) { o[c] = expf(in[c] - mx); sum += (double)o[c]; }

        float inv = (float)(1.0 / sum);
        for (int c = 0; c < cols; c++) o[c] *= inv;
    }
}

void tc_ref_mha(float *out, const float *q, const float *k, const float *v,
                int T, int n_head, int head_dim, float *scratch)
{
    const int D = n_head * head_dim;
    const float scale = 1.0f / sqrtf((float)head_dim);

    const size_t Ds = (size_t)D, hd = (size_t)head_dim;
    for (size_t h = 0; h < (size_t)n_head; h++) {
        const size_t off = h * hd;
        for (size_t t = 0; t < (size_t)T; t++) {
            const float *qh = q + t * Ds + off;

            /* scores over all positions — no causal mask, this is an encoder */
            float mx = -INFINITY;
            for (size_t s = 0; s < (size_t)T; s++) {
                const float *kh = k + s * Ds + off;
                float acc = 0.0f;
                for (size_t d = 0; d < hd; d++) acc += qh[d] * kh[d];
                acc *= scale;
                scratch[s] = acc;
                if (acc > mx) mx = acc;
            }

            double sum = 0.0;
            for (size_t s = 0; s < (size_t)T; s++) {
                scratch[s] = expf(scratch[s] - mx);
                sum += (double)scratch[s];
            }
            float inv = (float)(1.0 / sum);

            float *o = out + t * Ds + off;
            for (size_t d = 0; d < hd; d++) o[d] = 0.0f;
            for (size_t s = 0; s < (size_t)T; s++) {
                const float w = scratch[s] * inv;
                const float *vh = v + s * Ds + off;
                for (size_t d = 0; d < hd; d++) o[d] += w * vh[d];
            }
        }
    }
}

void tc_ref_add(float *out, const float *a, const float *b, size_t n)
{
    for (size_t i = 0; i < n; i++) out[i] = a[i] + b[i];
}

void tc_ref_scale(float *out, const float *x, float s, size_t n)
{
    for (size_t i = 0; i < n; i++) out[i] = x[i] * s;
}
