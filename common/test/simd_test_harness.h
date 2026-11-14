/*
 * simd_test_harness.h - input generation, bit-exact comparison and canary helpers for the SIMD kernel selftests.
 *
 * Include after the header under test (it needs Complex). Helpers are static inline and the generator state is
 * file-static, so each including file has its own seed stream and its own g_total_checks. A mismatch or canary
 * violation prints a diagnostic and exits 1; there is no tolerance. The including file defines EDGE_MAX_N and
 * EDGE_ARENA_LEN for its own arena size.
 */
#ifndef SIMD_TEST_HARNESS_H
#define SIMD_TEST_HARNESS_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>

/* ═══════════════════════════ input generation ═══════════════════════════
 * Deterministic LCG over raw uint32 bit patterns mapped to floats, mixed
 * with a pool of special values (drawn 25% of the time, see gen_float()).
 * NaN bit patterns from the raw-bits path are remapped to 0.0f; NaN payload
 * propagation is outside the bit-exactness contract. */

static uint32_t g_lcg = 0x9E3779B9u;

static inline uint32_t lcg_next(void) {
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return g_lcg;
}

static inline float bits_to_float(uint32_t b) {
    float f;
    memcpy(&f, &b, sizeof f);
    return f;
}

#define SPECIAL_POOL_COUNT 12
static float special_pool[SPECIAL_POOL_COUNT];

static inline void init_special_pool(void) {
    special_pool[0]  = 0.0f;
    special_pool[1]  = -0.0f;
    special_pool[2]  = FLT_MIN;
    special_pool[3]  = bits_to_float(0x00000001u); /* smallest subnormal */
    special_pool[4]  = bits_to_float(0x007FFFFFu); /* largest subnormal */
    special_pool[5]  = FLT_MAX;
    special_pool[6]  = 1.0f;
    special_pool[7]  = -1.0f;
    special_pool[8]  = 1e-30f;
    special_pool[9]  = 3e38f;
    special_pool[10] = (float)INFINITY;
    special_pool[11] = -(float)INFINITY;
}

static inline float gen_float(void) {
    uint32_t r = lcg_next();
    if ((r & 3u) == 0u) {
        uint32_t idx = (lcg_next() >> 8) % SPECIAL_POOL_COUNT;
        return special_pool[idx];
    } else {
        uint32_t bits = lcg_next();
        float f = bits_to_float(bits);
        if (f != f) f = 0.0f; /* exclude NaN */
        return f;
    }
}

static inline void gen_complex(Complex *c) {
    c->r = gen_float();
    c->i = gen_float();
}

static inline void fill_floats(float *a, int n) {
    int i;
    for (i = 0; i < n; ++i) a[i] = gen_float();
}

static inline void fill_complex(Complex *a, int n) {
    int i;
    for (i = 0; i < n; ++i) gen_complex(&a[i]);
}

/* ═══════════════════════════ mismatch reporting ═══════════════════════════
 * g_total_checks counts the individual bit-pattern comparisons performed,
 * incremented at the few functions every check goes through
 * (check_bits_or_die, the special-value asserts, the canary checks); main()
 * prints it. */

static long g_total_checks = 0;

static inline int first_diff_bits(const float *a, const float *b, int count) {
    int i;
    for (i = 0; i < count; ++i) {
        uint32_t ba, bb;
        memcpy(&ba, &a[i], sizeof ba);
        memcpy(&bb, &b[i], sizeof bb);
        if (ba != bb) return i;
    }
    return -1;
}

static inline void check_bits_or_die(const char *kernel, int n, int trial,
                               const float *simd, const float *scalar, int count) {
    int idx;
    g_total_checks += count;
    idx = first_diff_bits(simd, scalar, count);
    if (idx >= 0) {
        uint32_t gb, wb;
        memcpy(&gb, &simd[idx], sizeof gb);
        memcpy(&wb, &scalar[idx], sizeof wb);
        fprintf(stderr,
            "MISMATCH kernel=%s n=%d trial=%d idx=%d simd=0x%08x (%.9g) scalar=0x%08x (%.9g)\n",
            kernel, n, trial, idx, (unsigned)gb, (double)simd[idx], (unsigned)wb, (double)scalar[idx]);
        exit(1);
    }
}


/* ═══════════ alignment + canary helpers ═══════════ */

#define EDGE_GUARD 32           /* guaranteed guard elements each side of the
                                 * payload window, regardless of offset/n */
#define EDGE_OFFSET_MAX 15      /* max float/Complex element offset under test */
#define EDGE_CANARY_BITS 0x7fc0deadu
#define EDGE_FORM_COUNT 3

static inline float edge_canary_float(void) { return bits_to_float(EDGE_CANARY_BITS); }

static inline void *edge_aligned_alloc(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0 || p == NULL) {
        fprintf(stderr, "FATAL: posix_memalign(64, %zu) failed\n", bytes);
        exit(1);
    }
    return p;
}

static inline void edge_fill_canary_f(float *arena, int len) {
    int i;
    float c = edge_canary_float();
    for (i = 0; i < len; ++i) arena[i] = c;
}

/* Every float of arena[0, len) outside the payload window
 * [win_lo, win_lo + win_len) must still hold the canary bit pattern; an empty
 * window (n = 0) means the whole arena. Calls exit(1) with a diagnostic on
 * the first violation. */
static inline void edge_check_canary_f(const char *label, const float *arena, int len,
                                 int win_lo, int win_len) {
    int i;
    uint32_t want = EDGE_CANARY_BITS;
    int win_hi = win_lo + win_len;
    for (i = 0; i < len; ++i) {
        uint32_t got;
        if (i >= win_lo && i < win_hi) continue; /* payload window, not guarded */
        g_total_checks++;
        memcpy(&got, &arena[i], sizeof got);
        if (got != want) {
            fprintf(stderr,
                "CANARY VIOLATION %s: arena[%d]=0x%08x (want canary 0x%08x) "
                "-- out-of-bounds access, payload window=[%d,%d)\n",
                label, i, (unsigned)got, (unsigned)want, win_lo, win_hi);
            exit(1);
        }
    }
}

/* The (input, output) element offsets of the three forms. Form 2 ("both,
 * different") uses `((o+7)%15)+1`: unlike the mirror `16-o`, which equals o at
 * o = 8, it differs from o for every o in 1..15, because o + 7 == o (mod 15)
 * has no solution. */
static inline void edge_offsets_for_form(int form, int o, int *in_off, int *out_off) {
    switch (form) {
    case 0: *in_off = o; *out_off = 0; break;                   /* input offset only */
    case 1: *in_off = 0; *out_off = o; break;                   /* output offset only */
    default: *in_off = o; *out_off = ((o + 7) % 15) + 1; break;  /* both, different */
    }
}

/* ═══════════ timing ═══════════ */

static inline double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static inline void report_bench(const char *name, double ns_scalar, double ns_simd) {
    printf("kernel=%s ns_per_call_scalar=%.2f ns_per_call_simd=%.2f speedup=%.2f\n",
           name, ns_scalar, ns_simd,
           ns_simd > 0.0 ? ns_scalar / ns_simd : 0.0);
}

#endif /* SIMD_TEST_HARNESS_H */
