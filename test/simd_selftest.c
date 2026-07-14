/*
 * simd_selftest.c - bitwise correctness + microbenchmark harness for
 * include/simd_kernels.h.
 *
 * For every sk_<name> kernel, runs sk_<name>() (NEON when available, else
 * scalar) and sk_<name>_scalar() on IDENTICAL copies of randomly-generated
 * input (mixed LCG bit patterns + a curated special-value pool, NaN
 * excluded) across a matrix of n values and trials, and memcmp's the full
 * output (plus accumulator/state buffers where relevant) bit-for-bit. Any
 * mismatch prints the kernel/n/trial/index and the two bit patterns, then
 * exit(1)s immediately -- there is no tolerance here, this is the
 * bit-exactness gate itself.
 *
 * After correctness, runs a small microbenchmark per kernel (n=257,
 * ~200k reps, CLOCK_MONOTONIC) and prints a one-line summary.
 */
#include "simd_kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <float.h>
#include <time.h>

/* ═══════════════════════════════ config ═══════════════════════════════ */

#define SK_TEST_MAX_N 512
static const int N_LIST[] = {1, 3, 4, 5, 8, 9, 16, 128, 129, 160, 255, 256, 257, 512};
#define N_LIST_COUNT ((int)(sizeof(N_LIST) / sizeof(N_LIST[0])))
#define TRIALS 8

#define BENCH_N 257
#define BENCH_REPS 200000

/* ═══════════════════════════ input generation ═══════════════════════════
 * Deterministic LCG over raw uint32 bit patterns mapped to floats (25% of
 * draws), mixed with a curated special-value pool (75%... actually the
 * other way: special pool draws 25% of the time, see gen_float()). NaN
 * bit patterns from the raw-bits path are remapped to 0.0f -- NaN payload
 * propagation is explicitly out of the bit-exactness contract. */

static uint32_t g_lcg = 0x9E3779B9u;

static uint32_t lcg_next(void) {
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return g_lcg;
}

static float bits_to_float(uint32_t b) {
    float f;
    memcpy(&f, &b, sizeof f);
    return f;
}

#define SPECIAL_POOL_COUNT 12
static float special_pool[SPECIAL_POOL_COUNT];

static void init_special_pool(void) {
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

static float gen_float(void) {
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

static void gen_complex(Complex *c) {
    c->r = gen_float();
    c->i = gen_float();
}

static void fill_floats(float *a, int n) {
    int i;
    for (i = 0; i < n; ++i) a[i] = gen_float();
}

static void fill_complex(Complex *a, int n) {
    int i;
    for (i = 0; i < n; ++i) gen_complex(&a[i]);
}

/* Separate, moderate-range generator for the microbenchmarks only -- keeps
 * the timing loops away from Inf/NaN-producing accumulation artifacts so
 * the reported numbers reflect typical-case throughput. Not used by any
 * correctness check. */
static float gen_bench_float(void) {
    uint32_t r = lcg_next();
    return ((float)(r % 2000001u) / 1000000.0f) - 1.0f; /* ~[-1, 1] */
}

static void fill_bench_floats(float *a, int n) {
    int i;
    for (i = 0; i < n; ++i) a[i] = gen_bench_float();
}

static void fill_bench_complex(Complex *a, int n) {
    int i;
    for (i = 0; i < n; ++i) { a[i].r = gen_bench_float(); a[i].i = gen_bench_float(); }
}

/* ═══════════════════════════ mismatch reporting ═══════════════════════════ */

static int first_diff_bits(const float *a, const float *b, int count) {
    int i;
    for (i = 0; i < count; ++i) {
        uint32_t ba, bb;
        memcpy(&ba, &a[i], sizeof ba);
        memcpy(&bb, &b[i], sizeof bb);
        if (ba != bb) return i;
    }
    return -1;
}

static void check_bits_or_die(const char *kernel, int n, int trial,
                               const float *simd, const float *scalar, int count) {
    int idx = first_diff_bits(simd, scalar, count);
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

/* NOTE: check_scalar_bits_or_die (the single-float-return sibling of
 * check_bits_or_die) is not needed here -- it was only ever called by the
 * pairwise-sum-family kernels' tests, all of which moved to
 * AEC/c_impl/test/simd_selftest_aec.c along with their kernels. */

/* ═══════════════════════════ correctness: kernel 4 ═══════════════════════ */

static void test_ema(void) {
    float state_init[SK_TEST_MAX_N], state_scalar[SK_TEST_MAX_N], state_simd[SK_TEST_MAX_N];
    float x[SK_TEST_MAX_N];
    int ni, t;
    const float alpha = 0.9f, beta = 0.1f; /* pbfdkf.c power-EMA constants */
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(state_init, n);
            fill_floats(x, n);
            memcpy(state_scalar, state_init, (size_t)n * sizeof(float));
            memcpy(state_simd, state_init, (size_t)n * sizeof(float));
            sk_ema_f32_scalar(state_scalar, x, alpha, beta, n);
            sk_ema_f32(state_simd, x, alpha, beta, n);
            check_bits_or_die("ema_f32", n, t, state_simd, state_scalar, n);
        }
    }
    printf("PASS ema_f32\n");
}


/* ═══════════════════════════ correctness: kernel 9 ═══════════════════════ */

static void test_capply_gain(void) {
    Complex z[SK_TEST_MAX_N];
    Complex out_scalar[SK_TEST_MAX_N], out_simd[SK_TEST_MAX_N];
    float g[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_complex(z, n);
            fill_floats(g, n);
            sk_capply_gain_f32_scalar(out_scalar, z, g, n);
            sk_capply_gain_f32(out_simd, z, g, n);
            check_bits_or_die("capply_gain_f32", n, t, (const float *)out_simd, (const float *)out_scalar, 2 * n);
        }
    }
    /* dedicated in-place check: out == z */
    {
        Complex buf_scalar[SK_TEST_MAX_N], buf_simd[SK_TEST_MAX_N], orig[SK_TEST_MAX_N];
        float g2[SK_TEST_MAX_N];
        int n = SK_TEST_MAX_N;
        fill_complex(orig, n);
        fill_floats(g2, n);
        memcpy(buf_scalar, orig, (size_t)n * sizeof(Complex));
        memcpy(buf_simd, orig, (size_t)n * sizeof(Complex));
        sk_capply_gain_f32_scalar(buf_scalar, buf_scalar, g2, n);
        sk_capply_gain_f32(buf_simd, buf_simd, g2, n);
        check_bits_or_die("capply_gain_f32_inplace", n, 0, (const float *)buf_simd, (const float *)buf_scalar, 2 * n);
    }
    printf("PASS capply_gain_f32\n");
}


/* ═══════════════════════════ correctness: kernel 10 ══════════════════════ */

static void test_cadd(void) {
    Complex a[SK_TEST_MAX_N], b[SK_TEST_MAX_N];
    Complex out_scalar[SK_TEST_MAX_N], out_simd[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_complex(a, n);
            fill_complex(b, n);
            sk_cadd_f32_scalar(out_scalar, a, b, n);
            sk_cadd_f32(out_simd, a, b, n);
            check_bits_or_die("cadd_f32", n, t, (const float *)out_simd, (const float *)out_scalar, 2 * n);
        }
    }
    printf("PASS cadd_f32\n");
}


/* ═══════════════════════════ correctness: kernel 11 ══════════════════════ */

static void test_sq_scale(void) {
    float x[SK_TEST_MAX_N], out_scalar[SK_TEST_MAX_N], out_simd[SK_TEST_MAX_N];
    int ni, t;
    const float scale = 0.5f;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(x, n);
            sk_sq_scale_f32_scalar(x, scale, out_scalar, n);
            sk_sq_scale_f32(x, scale, out_simd, n);
            check_bits_or_die("sq_scale_f32", n, t, out_simd, out_scalar, n);
        }
    }
    printf("PASS sq_scale_f32\n");
}


/* ═══════════════════════════ correctness: kernel 12 ══════════════════════ */

static void test_min(void) {
    float a[SK_TEST_MAX_N], b[SK_TEST_MAX_N];
    float out_scalar[SK_TEST_MAX_N], out_simd[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(a, n);
            fill_floats(b, n);
            sk_min_f32_scalar(out_scalar, a, b, n);
            sk_min_f32(out_simd, a, b, n);
            check_bits_or_die("min_f32", n, t, out_simd, out_scalar, n);
        }
    }
    /* dedicated signed-zero tie-break check (both orderings, across lanes --
     * see the header-comment note: vminq_f32(-0,+0) != (a<b)?a:b). */
    {
        float az[8], bz[8], out_s[8], out_n[8];
        int k;
        for (k = 0; k < 8; k += 2) {
            az[k] = -0.0f; bz[k] = 0.0f;
            az[k + 1] = 0.0f; bz[k + 1] = -0.0f;
        }
        sk_min_f32_scalar(out_s, az, bz, 8);
        sk_min_f32(out_n, az, bz, 8);
        check_bits_or_die("min_f32_signed_zero", 8, 0, out_n, out_s, 8);
    }
    /* dedicated NaN check (F10): sk_min_f32's scalar ternary `(a<b)?a:b` and
     * the NEON `vcltq_f32(a,b)`-then-select both use an ORDERED '<' compare,
     * which is false whenever either operand is NaN -- so both paths fall
     * through to "return b" for every NaN case below, not just the ones
     * where b happens to be the smaller value. Covers NaN-in-a, NaN-in-b,
     * NaN-in-both, and NaN-vs-negative, all bitwise scalar-vs-NEON. */
    {
        float a[4], b[4], out_s[4], out_n[4];
        float qnan = bits_to_float(0x7FC00000u);
        float qnan_neg = bits_to_float(0xFFC00000u);
        a[0] = qnan;     b[0] = 1.0f;
        a[1] = 1.0f;     b[1] = qnan;
        a[2] = qnan;     b[2] = qnan_neg;
        a[3] = qnan;     b[3] = -1.0f;
        sk_min_f32_scalar(out_s, a, b, 4);
        sk_min_f32(out_n, a, b, 4);
        check_bits_or_die("min_f32_nan", 4, 0, out_n, out_s, 4);
    }
    printf("PASS min_f32\n");
}


static void test_clip(void) {
    float x_scalar[SK_TEST_MAX_N], x_simd[SK_TEST_MAX_N], x_init[SK_TEST_MAX_N];
    int ni, t;
    const float lo = -1.0f, hi = 1.0f;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(x_init, n);
            memcpy(x_scalar, x_init, (size_t)n * sizeof(float));
            memcpy(x_simd, x_init, (size_t)n * sizeof(float));
            sk_clip_f32_scalar(x_scalar, lo, hi, n);
            sk_clip_f32(x_simd, lo, hi, n);
            check_bits_or_die("clip_f32", n, t, x_simd, x_scalar, n);
        }
    }
    /* dedicated lo=0.0f boundary w/ -0.0f input (see header note). */
    {
        float x0_scalar[4] = { -0.0f, 0.0f, -0.0f, 0.0f };
        float x0_simd[4]   = { -0.0f, 0.0f, -0.0f, 0.0f };
        sk_clip_f32_scalar(x0_scalar, 0.0f, 1.0f, 4);
        sk_clip_f32(x0_simd, 0.0f, 1.0f, 4);
        check_bits_or_die("clip_f32_signed_zero", 4, 0, x0_simd, x0_scalar, 4);
    }
    /* dedicated NaN check (F10): sk_clip_f32's scalar `if(x<lo)...else if
     * (x>hi)...` and the NEON vcltq_f32/vcgtq_f32-then-select both use
     * ORDERED compares, which are false for a NaN input either way -- so
     * neither branch fires and the original (NaN) bit pattern passes
     * through unmodified in both paths. Verifies that "leave unchanged"
     * agreement bitwise, including a NaN sitting exactly at a `lo`/`hi`
     * bound value. */
    {
        float qnan = bits_to_float(0x7FC00000u);
        float qnan_neg = bits_to_float(0xFFC00000u);
        float x_scalar[4] = { qnan, qnan_neg, qnan, qnan };
        float x_simd[4]   = { qnan, qnan_neg, qnan, qnan };
        sk_clip_f32_scalar(x_scalar, lo, hi, 4);
        sk_clip_f32(x_simd, lo, hi, 4);
        check_bits_or_die("clip_f32_nan", 4, 0, x_simd, x_scalar, 4);
    }
    printf("PASS clip_f32\n");
}


/* ═══════════════════════════ correctness: kernel 15 ══════════════════════ */

static void test_fast_sqrt(void) {
    float x[SK_TEST_MAX_N], out_scalar[SK_TEST_MAX_N], out_simd[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(x, n);
            sk_fast_sqrt_f32_scalar(x, out_scalar, n);
            sk_fast_sqrt_f32(x, out_simd, n);
            check_bits_or_die("fast_sqrt_f32", n, t, out_simd, out_scalar, n);
        }
    }
    /* dedicated NaN check (F10 fix): sk__fast_sqrt_elem's `!(v>0.0f)` guard
     * and sk_fast_sqrt_f32's NEON `ispos = v>0` select both use an ORDERED
     * '>' compare -- false for NaN -- so every NaN lane below must land on
     * the 0.0f domain-edge branch in BOTH paths, matching fast_math.h's
     * fast_sqrt() fix. Covers a quiet NaN, a negative-signed NaN, and a
     * signaling-NaN-shaped bit pattern (exponent all-1s, nonzero mantissa,
     * quiet bit clear) mixed with ordinary negative/positive/zero values so
     * the NaN lanes sit next to non-NaN lanes within the same 4-wide vector. */
    {
        float nan_in[8], out_s[8], out_n[8];
        int i;
        nan_in[0] = bits_to_float(0x7FC00000u); /* quiet NaN */
        nan_in[1] = bits_to_float(0xFFC00000u); /* quiet NaN, sign bit set */
        nan_in[2] = bits_to_float(0x7F800001u); /* signaling NaN payload */
        nan_in[3] = bits_to_float(0xFFA00000u); /* NaN, sign bit set, other payload */
        nan_in[4] = -4.0f;                       /* ordinary negative, same vector */
        nan_in[5] = 0.0f;
        nan_in[6] = 4.0f;                        /* ordinary positive, same vector */
        nan_in[7] = bits_to_float(0x7FC00001u); /* another quiet NaN payload */
        sk_fast_sqrt_f32_scalar(nan_in, out_s, 8);
        sk_fast_sqrt_f32(nan_in, out_n, 8);
        check_bits_or_die("fast_sqrt_f32_nan", 8, 0, out_n, out_s, 8);
        /* Also assert the NaN lanes are actually the documented 0.0f, not
         * just "scalar and NEON happen to agree on some other garbage". */
        for (i = 0; i < 4; ++i) {
            if (out_s[i] != 0.0f || out_n[i] != 0.0f) {
                fprintf(stderr,
                    "fast_sqrt_f32 NaN guard FAILED: lane %d expected 0.0f, "
                    "got scalar=%.9g simd=%.9g\n",
                    i, (double)out_s[i], (double)out_n[i]);
                exit(1);
            }
        }
    }
    printf("PASS fast_sqrt_f32\n");
}

/* ═══════════════════════════════ microbench ═══════════════════════════════
 * n=257, ~200k reps, CLOCK_MONOTONIC. `g_bench_sink` (volatile) forces the
 * compiler to keep each call's result live, so the timing loop can't be
 * hoisted/eliminated as dead/invariant code. */

static volatile double g_bench_sink = 0.0;

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static void report_bench(const char *name, double ns_scalar, double ns_simd) {
    printf("kernel=%s ns_per_call_scalar=%.2f ns_per_call_simd=%.2f speedup=%.2f\n",
           name, ns_scalar, ns_simd,
           ns_simd > 0.0 ? ns_scalar / ns_simd : 0.0);
}

static void bench_ema(void) {
    float state[BENCH_N], x[BENCH_N];
    fill_bench_floats(state, BENCH_N);
    fill_bench_floats(x, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_ema_f32_scalar(state, x, 0.9f, 0.1f, BENCH_N); g_bench_sink += state[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_ema_f32(state, x, 0.9f, 0.1f, BENCH_N); g_bench_sink += state[0]; }
            {
                double t3 = now_ns();
                report_bench("ema_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}


static void bench_capply_gain(void) {
    Complex z[BENCH_N], out[BENCH_N];
    float g[BENCH_N];
    fill_bench_complex(z, BENCH_N);
    fill_bench_floats(g, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_capply_gain_f32_scalar(out, z, g, BENCH_N); g_bench_sink += out[0].r; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_capply_gain_f32(out, z, g, BENCH_N); g_bench_sink += out[0].r; }
            {
                double t3 = now_ns();
                report_bench("capply_gain_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}


static void bench_cadd(void) {
    Complex a[BENCH_N], b[BENCH_N], out[BENCH_N];
    fill_bench_complex(a, BENCH_N);
    fill_bench_complex(b, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_cadd_f32_scalar(out, a, b, BENCH_N); g_bench_sink += out[0].r; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_cadd_f32(out, a, b, BENCH_N); g_bench_sink += out[0].r; }
            {
                double t3 = now_ns();
                report_bench("cadd_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}


static void bench_sq_scale(void) {
    float x[BENCH_N], out[BENCH_N];
    fill_bench_floats(x, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_sq_scale_f32_scalar(x, 0.5f, out, BENCH_N); g_bench_sink += out[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_sq_scale_f32(x, 0.5f, out, BENCH_N); g_bench_sink += out[0]; }
            {
                double t3 = now_ns();
                report_bench("sq_scale_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}


static void bench_min(void) {
    float a[BENCH_N], b[BENCH_N], out[BENCH_N];
    fill_bench_floats(a, BENCH_N);
    fill_bench_floats(b, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_min_f32_scalar(out, a, b, BENCH_N); g_bench_sink += out[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_min_f32(out, a, b, BENCH_N); g_bench_sink += out[0]; }
            {
                double t3 = now_ns();
                report_bench("min_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}


static void bench_clip(void) {
    float x[BENCH_N];
    fill_bench_floats(x, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_clip_f32_scalar(x, -0.5f, 0.5f, BENCH_N); g_bench_sink += x[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_clip_f32(x, -0.5f, 0.5f, BENCH_N); g_bench_sink += x[0]; }
            {
                double t3 = now_ns();
                report_bench("clip_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}


static void bench_fast_sqrt(void) {
    float x[BENCH_N], out[BENCH_N];
    int i;
    fill_bench_floats(x, BENCH_N);
    for (i = 0; i < BENCH_N; ++i) x[i] = x[i] * x[i] + 1.0f; /* bias positive */
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_fast_sqrt_f32_scalar(x, out, BENCH_N); g_bench_sink += out[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_fast_sqrt_f32(x, out, BENCH_N); g_bench_sink += out[0]; }
            {
                double t3 = now_ns();
                report_bench("fast_sqrt_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}


/* ═══════════════════════════════════ main ══════════════════════════════════ */

int main(void) {
    init_special_pool();

    test_ema();
    test_capply_gain();
    test_cadd();
    test_sq_scale();
    test_min();
    test_clip();
    test_fast_sqrt();

    printf("\n--- microbenchmarks (n=%d, %d reps) ---\n", BENCH_N, BENCH_REPS);
    bench_ema();
    bench_capply_gain();
    bench_cadd();
    bench_sq_scale();
    bench_min();
    bench_clip();
    bench_fast_sqrt();

    printf("\nALL PASS (SK_HAVE_NEON=%d)\n", SK_HAVE_NEON);
    (void)g_bench_sink;
    return 0;
}
