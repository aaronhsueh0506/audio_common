/**
 * mem_align.h - shared static-memory alignment helper
 *
 * Every module's static-memory pair (X_get_mem_size / X_init) carves its state
 * out of one caller-provided block with 16-byte-aligned pointer bumps. This is
 * the single shared definition of that alignment, so the size a query reports
 * always matches the layout an init produces, across every consumer (AEC, NR,
 * pipelines).
 *
 * The caller-provided block itself must be at least 16-byte aligned.
 */
#ifndef MEM_ALIGN_H
#define MEM_ALIGN_H

#include <stddef.h>

#ifndef ALIGN16
#define ALIGN16(x) (((size_t)(x) + 15u) & ~(size_t)15u)
#endif

#endif /* MEM_ALIGN_H */
