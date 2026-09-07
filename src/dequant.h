/* dequant.h — ggml tensor types to fp32.
 *
 * Weights arrive in whatever type the file stores. The reference kernels take
 * fp32. This is the seam between them, and it is the seam stage 02 grows into:
 * today it handles F32 and F16, and every quantized type lands here next to
 * them with the same signature.
 *
 * Two paths, and the difference matters on an 8 GB machine:
 *
 *   F32 needs no conversion at all — the mapping already holds the bytes the
 *   kernels want, so tc_dequant_view() hands back a pointer INTO the mmap and
 *   costs nothing. A 1.2 GB fp32 model stays at zero resident bytes until a
 *   kernel touches it.
 *
 *   Everything else must be materialized. tc_dequant_view() allocates from the
 *   arena and converts once. That is a real cost (an f16 model doubles to fp32
 *   in RAM), and it is why stage 02's kernels will consume quantized blocks
 *   directly rather than going through here.
 */
#ifndef TC_DEQUANT_H
#define TC_DEQUANT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "arena.h"
#include "gguf.h"

/* IEEE binary16 -> binary32. Exact for every input: fp32 can represent every
 * f16 value including subnormals, infinities and NaN payloads. Written as bit
 * manipulation rather than _Float16 so the result does not depend on whether
 * the compiler has hardware half support. */
float tc_f16_to_f32(uint16_t h);

/* Convert n elements at `src` of ggml type `t` into `dst`. False when the type
 * has no conversion here yet — which is a missing feature, not a file error,
 * so callers report it as such. */
bool tc_dequant(float *dst, const void *src, size_t n, ggml_type_t t);

/* Fp32 view of a whole tensor: the mapping itself when it is already F32,
 * otherwise an arena-allocated conversion. NULL if the type is unsupported or
 * the arena is exhausted. `converted` (optional) reports which happened, so a
 * caller can account for the memory it just spent. */
const float *tc_dequant_view(const gguf_tensor_t *t, arena_t *a, bool *converted);

#endif /* TC_DEQUANT_H */
