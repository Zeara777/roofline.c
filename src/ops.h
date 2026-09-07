/* ops.h — scalar fp32 reference kernels.
 *
 * These are the DEFINITION of each operation. They are never optimized, never
 * vectorized, and never threaded. Every kernel added later is checked against
 * these, not against another kernel.
 *
 * Two deliberate choices:
 *
 *   Raw float pointers with explicit dimensions, not a tensor struct. A
 *   reference implementation earns its keep by being obvious on a first read;
 *   a descriptor layer would hide the indexing, which is exactly the part a
 *   reader needs to check. The descriptor arrives when the graph needs it.
 *
 *   fp32 accumulation in a fixed sequential order. Not double — the reference
 *   must be reproducible by an optimized fp32 kernel, and a double-accumulating
 *   reference would make every fast kernel look broken. This sequential order
 *   IS the reduction order ARCHITECTURE.md requires.
 *
 * Row-major throughout. Weight layouts follow PyTorch, so an exported tensor
 * is used exactly as stored.
 */
#ifndef TC_OPS_H
#define TC_OPS_H

#include <stdbool.h>
#include <stddef.h>

/* C[M,N] = A[M,K] * B, where B is [K,N], or [N,K] when b_transposed.
 * b_transposed is the common case: a PyTorch nn.Linear weight is [out, in]. */
void tc_ref_gemm(float *C, const float *A, const float *B,
                 int M, int N, int K, bool b_transposed);

/* y[M,N] = x[M,K] * W[N,K]^T + b[N].  bias may be NULL. */
void tc_ref_linear(float *y, const float *x, const float *W, const float *bias,
                   int M, int N, int K);

/* Per row over the last axis: (x - mean) / sqrt(var + eps) * gamma + beta.
 * Variance is biased (divides by n), matching PyTorch LayerNorm. */
void tc_ref_layernorm(float *out, const float *x, const float *gamma,
                      const float *beta, int rows, int cols, float eps);

/* Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))). PyTorch nn.GELU default. */
void tc_ref_gelu(float *out, const float *x, size_t n);

/* QuickGELU: x * sigmoid(1.702 * x). OpenAI CLIP checkpoints use this one.
 * ⚠️ Which of the two a checkpoint wants is not in open_clip_config.json and
 * getting it wrong silently produces plausible, wrong embeddings. The export
 * must record it. */
void tc_ref_gelu_quick(float *out, const float *x, size_t n);

/* Row-wise softmax, max-subtracted for stability. */
void tc_ref_softmax_rows(float *out, const float *x, int rows, int cols);

/* Bidirectional multi-head attention, no mask — the ViT case.
 * q, k, v are each [T, n_head * head_dim], heads contiguous within a row,
 * which is the layout a fused qkv projection produces.
 * out is [T, n_head * head_dim]. scale is 1/sqrt(head_dim).
 * scratch must hold at least T floats. */
void tc_ref_mha(float *out, const float *q, const float *k, const float *v,
                int T, int n_head, int head_dim, float *scratch);

void tc_ref_add(float *out, const float *a, const float *b, size_t n);
void tc_ref_scale(float *out, const float *x, float s, size_t n);

#endif /* TC_OPS_H */
