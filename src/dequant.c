#include "dequant.h"

#include <string.h>

float tc_f16_to_f32(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;

    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;                       /* +/-0 */
        } else {
            /* Subnormal f16. Normalize by shifting the mantissa left until the
             * implicit bit appears, paying one exponent step per shift. fp32
             * has the range to hold every one of these as a normal number. */
            exp = 127u - 15u + 1u;
            while ((mant & 0x400u) == 0u) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FFu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        /* Inf or NaN. The payload is preserved rather than canonicalized, so a
         * NaN that reaches a kernel can still be traced back to its source. */
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }

    float out;
    memcpy(&out, &bits, sizeof out);
    return out;
}

bool tc_dequant(float *dst, const void *src, size_t n, ggml_type_t t)
{
    switch (t) {
    case GGML_TYPE_F32:
        /* memcpy, not a cast-and-loop: GGUF only guarantees the tensor data
         * section is aligned to general.alignment (32 by default), which is
         * enough for float, but the byte-copy is free and cannot be wrong. */
        memcpy(dst, src, n * sizeof(float));
        return true;

    case GGML_TYPE_F16: {
        const unsigned char *p = (const unsigned char *)src;
        for (size_t i = 0; i < n; i++) {
            uint16_t h;
            memcpy(&h, p + i * 2u, sizeof h);
            dst[i] = tc_f16_to_f32(h);
        }
        return true;
    }

    default:
        return false;   /* quantized types arrive in stage 02 */
    }
}

const float *tc_dequant_view(const gguf_tensor_t *t, arena_t *a, bool *converted)
{
    if (t == NULL || t->data == NULL) return NULL;

    if (t->type == GGML_TYPE_F32) {
        if (converted) *converted = false;
        return (const float *)t->data;
    }

    float *buf = (float *)arena_alloc(a, t->n_elem * sizeof(float), ARENA_DEFAULT_ALIGN);
    if (buf == NULL) return NULL;
    if (!tc_dequant(buf, t->data, t->n_elem, t->type)) return NULL;

    if (converted) *converted = true;
    return buf;
}
