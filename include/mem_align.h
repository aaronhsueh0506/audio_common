/**
 * mem_align.h - shared static-memory alignment + checked-size helpers
 *
 * Every module's static-memory pair (X_get_mem_size / X_init) carves its state
 * out of one caller-provided block with 16-byte-aligned pointer bumps. This is
 * the single shared definition of that alignment, so the size a query reports
 * always matches the layout an init produces, across every consumer (AEC, NR,
 * pipelines).
 *
 * The caller-provided block itself must be at least 16-byte aligned; public
 * init entry points reject a misaligned base (MEM_IS_ALIGNED16).
 *
 * Size arithmetic in get_mem_size walks must use the checked helpers below:
 * every add/multiply/align saturates to SIZE_MAX on overflow instead of
 * wrapping, and MEM_SIZE_INVALID(total) is true iff any step overflowed.
 * A get_mem_size that overflows must report failure (return 0) rather than a
 * small wrapped number that a later init would carve past.
 */
#ifndef MEM_ALIGN_H
#define MEM_ALIGN_H

#include <stddef.h>
#include <stdint.h>

#ifndef ALIGN16
#define ALIGN16(x) (((size_t)(x) + 15u) & ~(size_t)15u)
#endif

/* Base-pointer alignment check for public pool init entry points. */
#define MEM_IS_ALIGNED16(p) ((((uintptr_t)(p)) & (uintptr_t)15u) == 0u)

/* Saturating checked size arithmetic: any overflow pins the result to
 * SIZE_MAX, which every subsequent ck_* call propagates, so a single
 * MEM_SIZE_INVALID() test at the end of a get_mem_size walk catches an
 * overflow anywhere in the chain. */
#define MEM_SIZE_INVALID(x) ((size_t)(x) == SIZE_MAX)

static inline size_t ck_add_size(size_t a, size_t b) {
    return (b > (size_t)-1 - a) ? (size_t)-1 : a + b;
}

static inline size_t ck_mul_size(size_t a, size_t b) {
    if (a == 0 || b == 0) return 0;
    if (a > (size_t)-1 / b) return (size_t)-1;
    return a * b;
}

/* ALIGN16 with overflow check (ALIGN16(SIZE_MAX-14 .. SIZE_MAX) would wrap). */
static inline size_t ck_align16_size(size_t x) {
    if (x > (size_t)-1 - 15u) return (size_t)-1;
    return (x + 15u) & ~(size_t)15u;
}

/* total += ALIGN16(count * elem_size), saturating. The canonical one-liner for
 * a get_mem_size field walk. */
static inline size_t ck_field_size(size_t total, size_t count, size_t elem_size) {
    return ck_add_size(total, ck_align16_size(ck_mul_size(count, elem_size)));
}

#endif /* MEM_ALIGN_H */
