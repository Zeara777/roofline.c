/* gguf.h — zero-copy GGUF v2/v3 reader.
 *
 * The file is mmap'd once, PROT_READ. Metadata is parsed into arena-allocated
 * descriptors whose strings and tensor payloads are *pointers into the mapping*
 * — nothing is copied. Loading a 5 GB model therefore costs a few hundred
 * microseconds and no resident memory until the weights are actually touched.
 *
 * That property is the whole reason to bother: it is what lets a quantized 3B
 * model start instantly on an 8 GB machine, and it is why llama.cpp's loader
 * looks the way it does.
 *
 * File layout (little-endian throughout):
 *
 *   magic  u32  "GGUF"
 *   version u32  2 or 3
 *   n_tensors u64
 *   n_kv      u64
 *   kv[n_kv]           key:str, type:u32, value
 *   tensors[n_tensors] name:str, n_dims:u32, dims[n_dims]:u64, type:u32, offset:u64
 *   <pad to general.alignment>
 *   tensor data
 *
 * Strings are { u64 len; char bytes[len]; } and are NOT NUL-terminated.
 */
#ifndef TC_GGUF_H
#define TC_GGUF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "arena.h"

#define GGUF_MAGIC       0x46554747u   /* "GGUF" little-endian */
#define GGUF_MAX_DIMS    4
#define GGUF_DEFAULT_ALIGNMENT 32u

/* Metadata value types. Values are the on-disk encoding — do not renumber. */
typedef enum {
    GGUF_T_UINT8   = 0,
    GGUF_T_INT8    = 1,
    GGUF_T_UINT16  = 2,
    GGUF_T_INT16   = 3,
    GGUF_T_UINT32  = 4,
    GGUF_T_INT32   = 5,
    GGUF_T_FLOAT32 = 6,
    GGUF_T_BOOL    = 7,
    GGUF_T_STRING  = 8,
    GGUF_T_ARRAY   = 9,
    GGUF_T_UINT64  = 10,
    GGUF_T_INT64   = 11,
    GGUF_T_FLOAT64 = 12,
    GGUF_T_COUNT
} gguf_type_t;

/* ggml tensor types. Only the subset this engine will actually read is given a
 * block size; everything else reports "unsupported" rather than a guessed size,
 * so a file we cannot handle fails loudly instead of computing wrong offsets. */
typedef enum {
    GGML_TYPE_F32   = 0,
    GGML_TYPE_F16   = 1,
    GGML_TYPE_Q4_0  = 2,
    GGML_TYPE_Q4_1  = 3,
    GGML_TYPE_Q5_0  = 6,
    GGML_TYPE_Q5_1  = 7,
    GGML_TYPE_Q8_0  = 8,
    GGML_TYPE_Q8_1  = 9,
    GGML_TYPE_Q2_K  = 10,
    GGML_TYPE_Q3_K  = 11,
    GGML_TYPE_Q4_K  = 12,
    GGML_TYPE_Q5_K  = 13,
    GGML_TYPE_Q6_K  = 14,
    GGML_TYPE_Q8_K  = 15,
    GGML_TYPE_I8    = 24,
    GGML_TYPE_I16   = 25,
    GGML_TYPE_I32   = 26,
    GGML_TYPE_I64   = 27,
    GGML_TYPE_F64   = 28,
    GGML_TYPE_BF16  = 30,
    GGML_TYPE_MAX   = 64
} ggml_type_t;

/* A view into the mapping. NOT NUL-terminated — print with %.*s. */
typedef struct {
    const char *ptr;
    uint64_t    len;
} gguf_str_t;

typedef struct {
    gguf_type_t  type;      /* element type                                  */
    uint64_t     len;       /* element count                                 */
    const void  *data;      /* raw elements, for fixed-width types           */
    gguf_str_t  *strs;      /* pre-indexed elements when type == GGUF_T_STRING */
} gguf_arr_t;

typedef struct {
    gguf_str_t  key;
    gguf_type_t type;
    union {
        uint8_t    u8;
        int8_t     i8;
        uint16_t   u16;
        int16_t    i16;
        uint32_t   u32;
        int32_t    i32;
        float      f32;
        uint64_t   u64;
        int64_t    i64;
        double     f64;
        bool       b;
        gguf_str_t str;
        gguf_arr_t arr;
    } v;
} gguf_kv_t;

typedef struct {
    gguf_str_t   name;
    uint32_t     n_dims;
    uint64_t     dims[GGUF_MAX_DIMS];  /* dims[0] is the fastest-varying axis */
    ggml_type_t  type;
    uint64_t     offset;               /* relative to the data section        */
    const void  *data;                 /* absolute pointer into the mapping   */
    uint64_t     n_elem;
    uint64_t     nbytes;               /* 0 when the type is unsupported      */
} gguf_tensor_t;

typedef struct {
    const unsigned char *map;
    size_t               map_size;
    int                  fd;

    uint32_t             version;
    uint32_t             alignment;
    uint64_t             n_kv;
    uint64_t             n_tensors;

    gguf_kv_t           *kv;
    gguf_tensor_t       *tensors;

    const unsigned char *data;         /* start of the tensor data section    */
    size_t               data_size;

    char                 err[256];
} gguf_t;

/* Open and fully parse `path`. Descriptors are allocated from `a`. Returns
 * false on any error, with a human-readable reason in out->err. On failure the
 * mapping is already closed; do not call gguf_close(). */
bool gguf_open(gguf_t *out, const char *path, arena_t *a);
void gguf_close(gguf_t *g);

/* Metadata lookup. `key` is a NUL-terminated C string. NULL when absent. */
const gguf_kv_t *gguf_find(const gguf_t *g, const char *key);

/* Scalar accessors with a fallback. These accept any integer type on disk and
 * widen, so a value stored as u32 still reads correctly through _u64. */
bool     gguf_get_u64(const gguf_t *g, const char *key, uint64_t *out);
bool     gguf_get_i64(const gguf_t *g, const char *key, int64_t *out);
bool     gguf_get_f32(const gguf_t *g, const char *key, float *out);
bool     gguf_get_str(const gguf_t *g, const char *key, gguf_str_t *out);
uint64_t gguf_u64_or(const gguf_t *g, const char *key, uint64_t fallback);
float    gguf_f32_or(const gguf_t *g, const char *key, float fallback);

const gguf_tensor_t *gguf_tensor(const gguf_t *g, const char *name);

/* Type metadata. block_size is elements per block, type_size bytes per block.
 * ggml_type_supported() is false for types this reader cannot size. */
const char *ggml_type_name(ggml_type_t t);
uint32_t    ggml_blck_size(ggml_type_t t);
size_t      ggml_type_size(ggml_type_t t);
bool        ggml_type_supported(ggml_type_t t);

const char *gguf_type_name(gguf_type_t t);

/* Format a gguf_str_t into a NUL-terminated buffer (truncating). Returns buf. */
char *gguf_str_cstr(char *buf, size_t buflen, gguf_str_t s);
/* Compare against a NUL-terminated C string. */
bool  gguf_str_eq(gguf_str_t s, const char *cstr);

#endif /* TC_GGUF_H */
