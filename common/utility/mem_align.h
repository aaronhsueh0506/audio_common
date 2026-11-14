/** mem_align.h - shared static-memory alignment and checked-size helpers.
 * Every X_get_mem_size / X_init pair carves its state from one caller-provided block with 16-byte-aligned bumps;
 * the block itself must be 16-byte aligned (init entry points reject a misaligned base via MEM_IS_ALIGNED16).
 * Size walks use the checked helpers below: every add/multiply/align saturates to SIZE_MAX, MEM_SIZE_INVALID(total)
 * is true iff any step overflowed, and a get_mem_size that overflows must return 0. */
#ifndef MEM_ALIGN_H
#define MEM_ALIGN_H

#include <stddef.h>
#include <stdint.h>

#ifndef ALIGN16
#define ALIGN16(x) (((size_t)(x) + 15u) & ~(size_t)15u)
#endif

#define MEM_IS_ALIGNED16(p) ((((uintptr_t)(p)) & (uintptr_t)15u) == 0u)

/* Saturating checked size arithmetic: an overflow pins the result to SIZE_MAX and every later ck_* call propagates it. */
#define MEM_SIZE_INVALID(x) ((size_t)(x) == SIZE_MAX)

static inline size_t ck_add_size(size_t a, size_t b) {
    return (b > (size_t)-1 - a) ? (size_t)-1 : a + b;
}

static inline size_t ck_mul_size(size_t a, size_t b) {
    if (a == 0 || b == 0) return 0;
    if (a > (size_t)-1 / b) return (size_t)-1;
    return a * b;
}

/* ALIGN16 with overflow check. */
static inline size_t ck_align16_size(size_t x) {
    if (x > (size_t)-1 - 15u) return (size_t)-1;
    return (x + 15u) & ~(size_t)15u;
}

/* total += ALIGN16(count * elem_size), saturating. */
static inline size_t ck_field_size(size_t total, size_t count, size_t elem_size) {
    return ck_add_size(total, ck_align16_size(ck_mul_size(count, elem_size)));
}

/* total += ALIGN16(count * elem_size) * reps, saturating (reps equal fields, each carved at its own ALIGN16 boundary). */
static inline size_t ck_field_size_reps(size_t total, size_t count, size_t elem_size,
                                        size_t reps) {
    size_t one = ck_align16_size(ck_mul_size(count, elem_size));
    return ck_add_size(total, ck_mul_size(one, reps));
}

/* Bounds-checked bump allocator over one caller block, for init paths that carve in the order get_mem_size() walks:
 * each carve takes ALIGN16(count * elem_size) bytes and returns NULL for a zero-sized request, an overflow or an exhausted block. */
typedef struct AcPoolCursor {
    uint8_t* ptr;
    size_t remaining;
} AcPoolCursor;

static inline void* ac_pool_carve(AcPoolCursor* cursor, size_t count, size_t element_size) {
    size_t raw, aligned;
    void* out;
    if (!cursor || count == 0 || element_size == 0) return NULL;
    raw = ck_mul_size(count, element_size);
    aligned = ck_align16_size(raw);
    if (MEM_SIZE_INVALID(raw) || MEM_SIZE_INVALID(aligned) ||
        aligned > cursor->remaining) return NULL;
    out = cursor->ptr;
    cursor->ptr += aligned;
    cursor->remaining -= aligned;
    return out;
}

/* Size-walk form of the same carve: reserve `count` elements of `elem` bytes at *cursor (aligned first). base == NULL only sizes;
 * with a base, *out receives base + aligned offset. Returns -1 on overflow or a NULL cursor with *cursor and *out untouched. */
static inline int ck_carve(unsigned char* base, size_t* cursor, size_t count, size_t elem, void** out) {
    size_t aligned, next;
    if (!cursor) return -1;
    aligned = ck_align16_size(*cursor);
    next = ck_add_size(aligned, ck_mul_size(count, elem));
    if (MEM_SIZE_INVALID(next)) return -1;
    if (base && out) *out = base + aligned;
    *cursor = next;
    return 0;
}

/* FNV-1a-32 descriptor folds: start from AC_FNV1A_SEED and chain; a u32 folds its four bytes least-significant first. */
#define AC_FNV1A_SEED UINT32_C(2166136261)

static inline uint32_t ac_fnv1a_u32(uint32_t h, uint32_t v) {
    int i;
    for (i = 0; i < 4; ++i) {
        h ^= (v >> (i * 8)) & 0xffu;
        h *= UINT32_C(16777619);
    }
    return h;
}

static inline uint32_t ac_fnv1a_str(uint32_t h, const char* s) {
    while (*s) {
        h ^= (uint32_t)(unsigned char)*s++;
        h *= UINT32_C(16777619);
    }
    return h;
}

/* Pool layouts contain native pointers; the hash input distinguishes ARM32 descriptors. The caller supplies its final FNV-1a hash. */
static inline uint32_t ac_memreq_abi_hash(uint32_t h) {
#if UINTPTR_MAX == UINT32_MAX
    h = ac_fnv1a_str(h, "|abi:ilp32");
#endif
    return h;
}

#endif /* MEM_ALIGN_H */
