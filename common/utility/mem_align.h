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

/* Pool layouts contain native pointers; the hash input distinguishes ARM32 descriptors. The caller supplies its final FNV-1a hash. */
static inline uint32_t ac_memreq_abi_hash(uint32_t h) {
#if UINTPTR_MAX == UINT32_MAX
    static const char token[] = "|abi:ilp32";
    size_t i;
    for (i = 0; i < sizeof(token) - 1; ++i) {
        h ^= (uint32_t)(unsigned char)token[i];
        h *= UINT32_C(16777619);
    }
#endif
    return h;
}

#endif /* MEM_ALIGN_H */
