#include "ops.h"

#include <math.h>

void tc_ref_gemm(float *C, const float *A, const float *B,
                 int M, int N, int K, bool b_transposed)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            if (b_transposed) {
                const float *brow = B + (size_t)n * K;      /* B is [N,K] */
                for (int k = 0; k < K; k++) acc += A[(size_t)m * K + k] * brow[k];
            } else {
                for (int k = 0; k < K; k++)                 /* B is [K,N] */
                    acc += A[(size_t)m * K + k] * B[(size_t)k * N + n];
            }
            C[(size_t)m * N + n] = acc;
        }
    }
}

void tc_ref_linear(float *y, const float *x, const float *W, const float *bias,
                   int M, int N, int K)
{
    tc_ref_gemm(y, x, W, M, N, K, true);
    if (bias != NULL)
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) y[(size_t)m * N + n] += bias[n];
}

void tc_ref_layernorm(float *out, const float *x, const float *gamma,
                      const float *beta, int rows, int cols, float eps)
{
    for (int r = 0; r < rows; r++) {
        const float *in = x + (size_t)r * cols;
        float *o = out + (size_t)r * cols;

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
    for (int r = 0; r < rows; r++) {
        const float *in = x + (size_t)r * cols;
        float *o = out + (size_t)r * cols;

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

    for (int h = 0; h < n_head; h++) {
        const int off = h * head_dim;
        for (int t = 0; t < T; t++) {
            const float *qh = q + (size_t)t * D + off;

            /* scores over all positions — no causal mask, this is an encoder */
            float mx = -INFINITY;
            for (int s = 0; s < T; s++) {
                const float *kh = k + (size_t)s * D + off;
                float acc = 0.0f;
                for (int d = 0; d < head_dim; d++) acc += qh[d] * kh[d];
                acc *= scale;
                scratch[s] = acc;
                if (acc > mx) mx = acc;
            }

            double sum = 0.0;
            for (int s = 0; s < T; s++) {
                scratch[s] = expf(scratch[s] - mx);
                sum += (double)scratch[s];
            }
            float inv = (float)(1.0 / sum);

            float *o = out + (size_t)t * D + off;
            for (int d = 0; d < head_dim; d++) o[d] = 0.0f;
            for (int s = 0; s < T; s++) {
                const float w = scratch[s] * inv;
                const float *vh = v + (size_t)s * D + off;
                for (int d = 0; d < head_dim; d++) o[d] += w * vh[d];
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
