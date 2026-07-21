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
 *
 * Re-review R07 adds test_fast_math_special_values(): explicit pinned
 * asserts for the fast_math.h special-value contract table (see that
 * header's "Special-value contract table" comment) -- fast_exp/fast_log/
 * fast_sqrt's exact documented return for NaN/+Inf/-Inf/0/negative inputs,
 * plus the NEON sk_fast_sqrt_f32 lane equivalents. These make the doc-
 * comment's table an executable regression gate instead of only prose.
 */
#include "simd_kernels.h"
#include "fast_math.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <float.h>
#include <time.h>

/* ═══════════════════════════════ config ═══════════════════════════════ */

#define SK_TEST_MAX_N 512
/* Round-3 review B05: extended to n=0 (must be a zero-read/zero-write no-op,
 * see the canary section below) plus the COMPLETE 1..17 run (every kernel's
 * 4-lane NEON/scalar-tail boundary crossed at every possible remainder, not
 * just a sparse sample) -- on top of the original hand-picked lane/leaf/
 * split boundary values, which stay for their own documented reasons. */
static const int N_LIST[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    128, 129, 160, 255, 256, 257, 512
};
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

/* ═══════════════════════════ mismatch reporting ═══════════════════════════
 * g_total_checks (round-3 review B05): a running count of individual
 * bit-pattern comparisons actually performed, incremented at each of the
 * few chokepoints every check funnels through (check_bits_or_die,
 * check_scalar_bits_or_die, the special-value asserts, and the new canary
 * checks below) -- so main()'s final printout is a real, reproducible
 * "how much did this run actually verify" number instead of a hand count,
 * and the review's requested before/after totals are just two runs of the
 * same binary at two points in this file's history. */

static long g_total_checks = 0;

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

/* NOTE: check_scalar_bits_or_die (the single-float-return sibling of
 * check_bits_or_die) is not needed here -- it was only ever called by the
 * pairwise-sum-family kernels' tests, all of which moved to
 * AEC/c_impl/test/simd_selftest_aec.c along with their kernels. */

/* ═══════════ pinned special-value contract (re-review R07) ═══════════════
 * fast_math.h's header comment documents an exact "Special-value contract
 * table" for fast_exp/fast_log/fast_sqrt (NaN/+-Inf/0/negative -> exact
 * deterministic float). These two helpers plus test_fast_math_special_values
 * below turn that table into an executable regression gate: exit(1) on any
 * deviation, no tolerance, same house style as check_bits_or_die above. */

#ifndef USE_STANDARD_MATH
/* Only used by the fast-math-path assertions below (bit-exact domain-edge
 * constants) -- USE_STANDARD_MATH's branch uses assert_f32_eq/assert_is_nan
 * instead, since libm's NaN/Inf propagation isn't a fixed bit pattern to pin
 * this precisely. #ifdef-guarded so the USE_STANDARD_MATH build doesn't warn
 * about an unused static function. */
static void assert_f32_bits(const char *label, float got, uint32_t want_bits) {
    uint32_t got_bits;
    g_total_checks++;
    memcpy(&got_bits, &got, sizeof got_bits);
    if (got_bits != want_bits) {
        fprintf(stderr,
            "SPECIAL-VALUE MISMATCH %s: got=0x%08x (%.9g) want=0x%08x\n",
            label, (unsigned)got_bits, (double)got, (unsigned)want_bits);
        exit(1);
    }
}
#endif

static void assert_is_nan(const char *label, float got) {
    g_total_checks++;
    if (!isnan(got)) {
        uint32_t got_bits;
        memcpy(&got_bits, &got, sizeof got_bits);
        fprintf(stderr,
            "SPECIAL-VALUE MISMATCH %s: got=0x%08x (%.9g), expected NaN\n",
            label, (unsigned)got_bits, (double)got);
        exit(1);
    }
}

#ifdef USE_STANDARD_MATH
/* Only used by test_fast_math_special_values's USE_STANDARD_MATH branch
 * below (the fast-math branch uses assert_f32_bits/assert_is_nan instead) --
 * #ifdef-guarded so the non-USE_STANDARD_MATH build doesn't warn about an
 * unused static function. */
static void assert_f32_eq(const char *label, float got, float want) {
    g_total_checks++;
    if (got != want) {
        fprintf(stderr, "SPECIAL-VALUE MISMATCH %s: got=%.9g want=%.9g\n",
                label, (double)got, (double)want);
        exit(1);
    }
}
#endif

/* Two build modes assert two different (each internally consistent) sets of
 * values: the default fast-math path pins the documented domain-edge
 * constants; USE_STANDARD_MATH pins ordinary libm NaN/Inf propagation
 * (fast_math.h's #else branch is a bare expf/logf/sqrtf call, so this is
 * really pinning libm's own contract) -- so the DIVERGENCE between the two
 * modes documented in fast_math.h's contract table is itself
 * regression-tested here, in both directions, not just asserted in prose. */
static void test_fast_math_special_values(void) {
    float qnan = bits_to_float(0x7FC00000u);
    float pinf = (float)INFINITY;
    float ninf = -(float)INFINITY;

#ifndef USE_STANDARD_MATH
    /* Fast-math domain-edge constants -- see fast_math.h's "Special-value
     * contract table" comment for the derivation of each bit pattern. */
    assert_f32_bits("fast_exp(NaN)",  fast_exp(qnan), 0x00000000u);
    assert_f32_bits("fast_exp(+Inf)", fast_exp(pinf), 0x4b07975eu); /* 8.8861105e+06f */
    assert_f32_bits("fast_exp(-Inf)", fast_exp(ninf), 0x00000000u);

    assert_f32_bits("fast_log(NaN)",  fast_log(qnan),  0xd01502f9u); /* -1e10f */
    assert_f32_bits("fast_log(-1)",   fast_log(-1.0f), 0xd01502f9u);
    assert_f32_bits("fast_log(0)",    fast_log(0.0f),  0xd01502f9u);
    assert_f32_bits("fast_log(+Inf)", fast_log(pinf),  0x42b17218u); /* 88.72283935546875f = 128*FM_LN2 */

    assert_f32_bits("fast_sqrt(NaN)",   fast_sqrt(qnan),  0x00000000u);
    assert_f32_bits("fast_sqrt(-1)",    fast_sqrt(-1.0f), 0x00000000u);
    assert_f32_bits("fast_sqrt(-0.0)",  fast_sqrt(-0.0f), 0x00000000u); /* sign NOT preserved */
    assert_is_nan  ("fast_sqrt(+Inf)",  fast_sqrt(pinf));
#else
    /* USE_STANDARD_MATH: bare libm calls, ordinary IEEE-754 propagation. */
    assert_is_nan("fast_exp(NaN) [libm]", fast_exp(qnan));
    assert_f32_eq("fast_exp(+Inf) [libm]", fast_exp(pinf), pinf);
    assert_f32_eq("fast_exp(-Inf) [libm]", fast_exp(ninf), 0.0f);

    assert_is_nan("fast_log(NaN) [libm]", fast_log(qnan));
    assert_is_nan("fast_log(-1) [libm]",  fast_log(-1.0f));
    assert_f32_eq("fast_log(0) [libm]",   fast_log(0.0f), ninf);
    assert_f32_eq("fast_log(+Inf) [libm]", fast_log(pinf), pinf);

    assert_is_nan("fast_sqrt(NaN) [libm]", fast_sqrt(qnan));
    assert_is_nan("fast_sqrt(-1) [libm]",  fast_sqrt(-1.0f));
    assert_f32_eq("fast_sqrt(+Inf) [libm]", fast_sqrt(pinf), pinf);
#endif

    /* NEON sk_fast_sqrt_f32 lane equivalents. No #ifdef needed for the
     * scalar-vs-NEON bit-exactness check itself -- sk__fast_sqrt_elem and
     * sk_fast_sqrt_f32's NEON body both switch on the SAME USE_STANDARD_MATH
     * flag (see simd_kernels.h), so they always agree with each other; the
     * per-lane VALUE assertions below are still fast-math-only, same as
     * the scalar checks above. */
    {
        float in[4], out_scalar[4], out_simd[4];
        in[0] = qnan; in[1] = -1.0f; in[2] = -0.0f; in[3] = pinf;
        sk_fast_sqrt_f32_scalar(in, out_scalar, 4);
        sk_fast_sqrt_f32(in, out_simd, 4);
        check_bits_or_die("fast_sqrt_f32_special_lanes", 4, 0, out_simd, out_scalar, 4);
#ifndef USE_STANDARD_MATH
        assert_f32_bits("sk_fast_sqrt_f32(NaN) scalar",   out_scalar[0], 0x00000000u);
        assert_f32_bits("sk_fast_sqrt_f32(-1) scalar",    out_scalar[1], 0x00000000u);
        assert_f32_bits("sk_fast_sqrt_f32(-0.0) scalar",  out_scalar[2], 0x00000000u);
        assert_is_nan  ("sk_fast_sqrt_f32(+Inf) scalar",  out_scalar[3]);
        assert_f32_bits("sk_fast_sqrt_f32(NaN) simd",     out_simd[0], 0x00000000u);
        assert_f32_bits("sk_fast_sqrt_f32(-1) simd",      out_simd[1], 0x00000000u);
        assert_f32_bits("sk_fast_sqrt_f32(-0.0) simd",    out_simd[2], 0x00000000u);
        assert_is_nan  ("sk_fast_sqrt_f32(+Inf) simd",    out_simd[3]);
#endif
    }

#ifdef USE_STANDARD_MATH
    printf("PASS fast_math_special_values (USE_STANDARD_MATH=1)\n");
#else
    printf("PASS fast_math_special_values (USE_STANDARD_MATH=0)\n");
#endif
}

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
        /* Also assert the NaN lanes land on the documented value for the
         * ACTIVE build mode, not just "scalar and NEON happen to agree on
         * some other garbage". Pre-R07 this unconditionally expected 0.0f,
         * which is only correct for the default fast-math path -- under
         * USE_STANDARD_MATH, sk__fast_sqrt_elem/sk_fast_sqrt_f32 are bare
         * sqrtf()/vsqrtq_f32 calls (see simd_kernels.h), so a NaN input
         * propagates to a NaN output instead; the old hardcoded check would
         * have exit(1)'d the very first time this file was ever built with
         * -DUSE_STANDARD_MATH (verified: it does, on the pre-R07 code too --
         * a latent gap, not a regression introduced here). */
        for (i = 0; i < 4; ++i) {
            g_total_checks++;
#ifndef USE_STANDARD_MATH
            if (out_s[i] != 0.0f || out_n[i] != 0.0f) {
                fprintf(stderr,
                    "fast_sqrt_f32 NaN guard FAILED: lane %d expected 0.0f, "
                    "got scalar=%.9g simd=%.9g\n",
                    i, (double)out_s[i], (double)out_n[i]);
                exit(1);
            }
#else
            if (!isnan(out_s[i]) || !isnan(out_n[i])) {
                fprintf(stderr,
                    "fast_sqrt_f32 NaN guard FAILED [USE_STANDARD_MATH]: lane %d "
                    "expected NaN, got scalar=%.9g simd=%.9g\n",
                    i, (double)out_s[i], (double)out_n[i]);
                exit(1);
            }
#endif
        }
    }
    printf("PASS fast_sqrt_f32\n");
}


/* ═══════════════════ correctness: kernels 23-27 (exp/log family) ═════════
 * s4-audio-common-sweep review: fast_math.h's exp/log/exp1_approx family had
 * zero NEON coverage anywhere. Beyond the usual scalar-vs-NEON bit-exactness
 * check every kernel in this file gets, these five ALSO cross-check
 * sk_<name>_f32_scalar's per-element output against fast_math.h's ACTUAL
 * fast_exp/fast_exp_neg/fast_log/fast_log10/exp1_approx on the SAME inputs
 * (check_matches_fastmath below) -- simd_kernels.h deliberately does not
 * #include fast_math.h and instead replicates its algorithm verbatim as a
 * private helper (see that header's Style section), so this is the gate
 * that catches any transcription drift between the two copies. */

static void check_matches_fastmath(const char *label, float (*ref_fn)(float),
                                    const float *x, const float *sk_out, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        float fm = ref_fn(x[i]);
        uint32_t fb, sb;
        g_total_checks++;
        memcpy(&fb, &fm, sizeof fb);
        memcpy(&sb, &sk_out[i], sizeof sb);
        if (fb != sb) {
            uint32_t xb;
            memcpy(&xb, &x[i], sizeof xb);
            fprintf(stderr,
                "%s vs fast_math.h MISMATCH i=%d x=%.9g(0x%08x) sk=%.9g(0x%08x) fastmath=%.9g(0x%08x)\n",
                label, i, (double)x[i], (unsigned)xb,
                (double)sk_out[i], (unsigned)sb, (double)fm, (unsigned)fb);
            exit(1);
        }
    }
}

static void test_fast_exp(void) {
    float x[SK_TEST_MAX_N], out_scalar[SK_TEST_MAX_N], out_simd[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(x, n);
            sk_fast_exp_f32_scalar(x, out_scalar, n);
            sk_fast_exp_f32(x, out_simd, n);
            check_bits_or_die("fast_exp_f32", n, t, out_simd, out_scalar, n);
            check_matches_fastmath("fast_exp_f32", fast_exp, x, out_scalar, n);
        }
    }
    printf("PASS fast_exp_f32\n");
}

static void test_fast_exp_neg(void) {
    float x[SK_TEST_MAX_N], out_scalar[SK_TEST_MAX_N], out_simd[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(x, n);
            sk_fast_exp_neg_f32_scalar(x, out_scalar, n);
            sk_fast_exp_neg_f32(x, out_simd, n);
            check_bits_or_die("fast_exp_neg_f32", n, t, out_simd, out_scalar, n);
            check_matches_fastmath("fast_exp_neg_f32", fast_exp_neg, x, out_scalar, n);
        }
    }
    printf("PASS fast_exp_neg_f32\n");
}

static void test_fast_log(void) {
    float x[SK_TEST_MAX_N], out_scalar[SK_TEST_MAX_N], out_simd[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(x, n);
            sk_fast_log_f32_scalar(x, out_scalar, n);
            sk_fast_log_f32(x, out_simd, n);
            check_bits_or_die("fast_log_f32", n, t, out_simd, out_scalar, n);
            check_matches_fastmath("fast_log_f32", fast_log, x, out_scalar, n);
        }
    }
    printf("PASS fast_log_f32\n");
}

static void test_fast_log10(void) {
    float x[SK_TEST_MAX_N], out_scalar[SK_TEST_MAX_N], out_simd[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(x, n);
            sk_fast_log10_f32_scalar(x, out_scalar, n);
            sk_fast_log10_f32(x, out_simd, n);
            check_bits_or_die("fast_log10_f32", n, t, out_simd, out_scalar, n);
            check_matches_fastmath("fast_log10_f32", fast_log10, x, out_scalar, n);
        }
    }
    printf("PASS fast_log10_f32\n");
}

static void test_exp1_approx(void) {
    float x[SK_TEST_MAX_N], out_scalar[SK_TEST_MAX_N], out_simd[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(x, n);
            sk_exp1_approx_f32_scalar(x, out_scalar, n);
            sk_exp1_approx_f32(x, out_simd, n);
            check_bits_or_die("exp1_approx_f32", n, t, out_simd, out_scalar, n);
            check_matches_fastmath("exp1_approx_f32", exp1_approx, x, out_scalar, n);
        }
    }
    printf("PASS exp1_approx_f32\n");
}

/* Dedicated domain-boundary/special-value sweep: every guard threshold these
 * five functions branch on (-16, 16, 0, 0.1, 1.0, FM_EPSILON=1e-10), plus
 * NaN/+-Inf, packed into 4-wide vectors so the boundary values sit at every
 * lane position (not just lane 0) across both the NEON body and the scalar
 * tail. Same three-way check (scalar==NEON, scalar==fast_math.h) as above. */
static void test_exp_log_boundaries(void) {
    float qnan = bits_to_float(0x7FC00000u);
    float qnan_neg = bits_to_float(0xFFC00000u);
    float pinf = (float)INFINITY;
    float ninf = -(float)INFINITY;
    float x[20] = {
        qnan, qnan_neg, pinf, ninf,
        -16.0f, 16.0f, bits_to_float(0xC1800001u) /* just below -16 */,
        bits_to_float(0x41800001u) /* just above 16 */,
        0.0f, -0.0f, 0.1f, -0.1f, 1.0f, -1.0f,
        1e-10f, 1e-30f, 3e38f, -3e38f, 0.5f, -0.5f
    };
    int n = 20;
    float out_scalar[20], out_simd[20];

    sk_fast_exp_f32_scalar(x, out_scalar, n);
    sk_fast_exp_f32(x, out_simd, n);
    check_bits_or_die("fast_exp_f32_boundary", n, 0, out_simd, out_scalar, n);
    check_matches_fastmath("fast_exp_f32_boundary", fast_exp, x, out_scalar, n);

    sk_fast_exp_neg_f32_scalar(x, out_scalar, n);
    sk_fast_exp_neg_f32(x, out_simd, n);
    check_bits_or_die("fast_exp_neg_f32_boundary", n, 0, out_simd, out_scalar, n);
    check_matches_fastmath("fast_exp_neg_f32_boundary", fast_exp_neg, x, out_scalar, n);

    sk_fast_log_f32_scalar(x, out_scalar, n);
    sk_fast_log_f32(x, out_simd, n);
    check_bits_or_die("fast_log_f32_boundary", n, 0, out_simd, out_scalar, n);
    check_matches_fastmath("fast_log_f32_boundary", fast_log, x, out_scalar, n);

    sk_fast_log10_f32_scalar(x, out_scalar, n);
    sk_fast_log10_f32(x, out_simd, n);
    check_bits_or_die("fast_log10_f32_boundary", n, 0, out_simd, out_scalar, n);
    check_matches_fastmath("fast_log10_f32_boundary", fast_log10, x, out_scalar, n);

    sk_exp1_approx_f32_scalar(x, out_scalar, n);
    sk_exp1_approx_f32(x, out_simd, n);
    check_bits_or_die("exp1_approx_f32_boundary", n, 0, out_simd, out_scalar, n);
    check_matches_fastmath("exp1_approx_f32_boundary", exp1_approx, x, out_scalar, n);

    /* n=0 must be a total no-op -- guards against a stray unconditional
     * table read/store before the loop guard. */
    sk_fast_exp_f32(x, out_simd, 0);
    sk_fast_exp_neg_f32(x, out_simd, 0);
    sk_fast_log_f32(x, out_simd, 0);
    sk_fast_log10_f32(x, out_simd, 0);
    sk_exp1_approx_f32(x, out_simd, 0);

    printf("PASS exp_log_boundaries\n");
}


/* ═══════ correctness: kernels 23/24/25/27 in-place (out==x) aliasing ═════
 * NR/c_impl's mmse_lsa_denoiser.c calculate_gain() and spp_estimator.c
 * spp_estimate()/spp_estimate_ex() chain sk_exp1_approx_f32/sk_fast_exp_f32/
 * sk_fast_log_f32 (calculate_gain) and sk_fast_exp_neg_f32 (spp_estimate)
 * back-to-back over the SAME scratch buffer (out==x) to cut per-call
 * scratch-array footprint (review-round buffer-reuse hardening) -- see the
 * "In-place (out==x) safety" paragraph in simd_kernels.h's kernels-23-27
 * section header for why this is safe (each 4-lane block is fully loaded
 * before it is stored, no cross-block state, same shape as
 * sk_capply_gain_f32's kernel-9 out==z contract). This is the dedicated
 * regression gate for that reliance: run each kernel with out==x on a copy
 * of the input and compare against the SAME kernel run with separate
 * (non-aliased) buffers, bit-for-bit -- if any of the four ever developed a
 * cross-lane dependency (e.g. a future edit added inter-block state), this
 * would catch it even though test_fast_exp() etc. above would not (those
 * always use non-overlapping buffers). sk_fast_log10_f32 (kernel 26) is
 * deliberately NOT exercised here -- no call site relies on out==x for it
 * (see simd_kernels.h's "no restrict" contract note). */

static void test_exp_log_family_inplace(void) {
    float x[SK_TEST_MAX_N];
    float out_sep[SK_TEST_MAX_N];
    float buf[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(x, n);

            sk_fast_exp_f32(x, out_sep, n);
            memcpy(buf, x, (size_t)n * sizeof(float));
            sk_fast_exp_f32(buf, buf, n);
            check_bits_or_die("fast_exp_f32_inplace", n, t, buf, out_sep, n);

            sk_fast_exp_neg_f32(x, out_sep, n);
            memcpy(buf, x, (size_t)n * sizeof(float));
            sk_fast_exp_neg_f32(buf, buf, n);
            check_bits_or_die("fast_exp_neg_f32_inplace", n, t, buf, out_sep, n);

            sk_fast_log_f32(x, out_sep, n);
            memcpy(buf, x, (size_t)n * sizeof(float));
            sk_fast_log_f32(buf, buf, n);
            check_bits_or_die("fast_log_f32_inplace", n, t, buf, out_sep, n);

            sk_exp1_approx_f32(x, out_sep, n);
            memcpy(buf, x, (size_t)n * sizeof(float));
            sk_exp1_approx_f32(buf, buf, n);
            check_bits_or_die("exp1_approx_f32_inplace", n, t, buf, out_sep, n);
        }
    }
    printf("PASS exp_log_family_inplace (out==x aliasing, kernels 23/24/25/27)\n");
}


/* ═══════════════════════════ correctness: kernel 28 ══════════════════════
 * sk_mcra_noise_update_f32 -- no fast_math.h ground truth to cross-check
 * against (this mirrors an NR/c_impl call site, not a fast_math.h function),
 * so this is a plain scalar-vs-NEON bit-exactness check, same shape as
 * test_ema() (kernel 4) since it is structurally the same per-bin-varying-
 * alpha EMA shape. */

static void test_mcra_noise_update(void) {
    float npsd_init[SK_TEST_MAX_N], npsd_scalar[SK_TEST_MAX_N], npsd_simd[SK_TEST_MAX_N];
    float spp[SK_TEST_MAX_N], power[SK_TEST_MAX_N];
    int ni, t;
    const float alpha_d = 0.95f, bb_scale = 0.7f; /* representative NR config values */
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(npsd_init, n);
            fill_floats(spp, n);
            fill_floats(power, n);
            memcpy(npsd_scalar, npsd_init, (size_t)n * sizeof(float));
            memcpy(npsd_simd, npsd_init, (size_t)n * sizeof(float));
            sk_mcra_noise_update_f32_scalar(npsd_scalar, spp, power, alpha_d, bb_scale, n);
            sk_mcra_noise_update_f32(npsd_simd, spp, power, alpha_d, bb_scale, n);
            check_bits_or_die("mcra_noise_update_f32", n, t, npsd_simd, npsd_scalar, n);
        }
    }
    /* dedicated spp/bb_scale boundary sweep: spp in {0,1} (fully-noise vs
     * fully-speech gating) and bb_scale in {0,1} (broadband-reset gate fully
     * open vs disabled), the four corner combinations of the call site's own
     * documented gating semantics. */
    {
        float npsd_s[8], npsd_n[8], spp8[8], power8[8];
        int k;
        for (k = 0; k < 8; ++k) { npsd_s[k] = npsd_n[k] = 1.0f + 0.1f * (float)k; power8[k] = 2.0f - 0.05f * (float)k; }
        spp8[0] = 0.0f; spp8[1] = 1.0f; spp8[2] = 0.0f; spp8[3] = 1.0f;
        spp8[4] = 0.0f; spp8[5] = 1.0f; spp8[6] = 0.0f; spp8[7] = 1.0f;
        sk_mcra_noise_update_f32_scalar(npsd_s, spp8, power8, alpha_d, 0.0f, 4);
        sk_mcra_noise_update_f32(npsd_n, spp8, power8, alpha_d, 0.0f, 4);
        sk_mcra_noise_update_f32_scalar(npsd_s + 4, spp8 + 4, power8 + 4, alpha_d, 1.0f, 4);
        sk_mcra_noise_update_f32(npsd_n + 4, spp8 + 4, power8 + 4, alpha_d, 1.0f, 4);
        check_bits_or_die("mcra_noise_update_f32_boundary", 8, 0, npsd_n, npsd_s, 8);
    }
    printf("PASS mcra_noise_update_f32\n");
}


/* ═══════════ alignment + canary edge-case matrix (round-3 review B05) ═════
 * Finding B05's edge-case matrix, layered on top of the per-kernel
 * correctness tests above:
 *
 *   - n=0: every test above already runs it (see the extended N_LIST). For
 *     n=0 specifically, "correct" means the kernel touches NOTHING (every
 *     loop here is a plain `for(i=0;i<n;...)`/`i+4<=n` guard, zero
 *     iterations when n==0) -- the canary buffers below turn that
 *     "touches nothing" claim into something actually checked, not just
 *     implied by code inspection.
 *   - n=1..17: also already covered by the same extended N_LIST (every
 *     kernel's 4-lane tail boundary, every possible remainder).
 *   - Unaligned float-element offsets 1..15: NOT covered above -- every
 *     existing test calls each kernel at the natural start of a plain
 *     array (offset 0). This section is new: every buffer lives in its own
 *     64-byte-aligned arena (posix_memalign), and each kernel is called at
 *     a deliberately +1..+15-FLOAT offset into it -- always 4-byte aligned
 *     (a float is always 4-byte aligned; a whole-float-element offset can
 *     never construct a misaligned float*, that would be UB and isn't what
 *     "unaligned" means for a NEON kernel test), but deliberately NOT
 *     16-byte (4-lane) or 64-byte aligned, so vld1q_f32/vst1q_f32 land on
 *     every possible sub-vector byte lane. Three forms, per the finding's
 *     ask: input-buffer offset only, output-buffer offset only, both
 *     buffers offset to DIFFERENT values (edge_offsets_for_form's "+7 mod
 *     15" derangement, never equal by construction -- see its own
 *     comment). For kernels with more than one read-only input array
 *     (cadd/min: a,b) or an extra scalar-per-bin array (capply_gain's g),
 *     only the first/primary array plays "input" for this matrix; the
 *     remaining array(s) stay at offset 0 -- a deliberate scope decision
 *     (the finding's own wording is binary: "input offset only, output
 *     offset only, both"), documented here rather than left implicit.
 *     sk_clip_f32 is a single in-place buffer (no separate output array),
 *     so its own test below sweeps just that one offset instead of the
 *     3-form matrix -- see test_clip_edge's comment.
 *   - Canary guard (the finding's own ask): every arena is entirely
 *     canary-filled (float bit pattern 0x7fc0dead) before each call, then
 *     the payload window is overwritten with real generated data (inputs)
 *     or left as canary (pure outputs, so the kernel's own write is what
 *     populates it). After the call, every element OUTSIDE the payload
 *     window must still read back as the untouched canary pattern, in
 *     EVERY arena the kernel touched (including read-only ones, as a
 *     defense-in-depth cross-check against unexpected aliasing writes).
 *     Because the window can be placed anywhere (offset 1..15) or be EMPTY
 *     (n==0), one check function catches an out-of-bounds write on either
 *     side of the payload, or literally any write at all when n==0,
 *     without a separate front/back special case.
 *
 * scalar-vs-NEON comparison is inherent to every check_bits_or_die call
 * below, same as the rest of this file -- this satisfies the review's
 * forced-scalar/forced-NEON comparison ask for this file's kernels (the
 * NE10 FFT kernel side of that same ask is covered by audio_common's
 * FFT_NE10_FORCE_C knob, exercised via `make test_ne10_force_c`, not by
 * this file). */

#define EDGE_GUARD 32           /* guaranteed guard elements each side of the
                                 * payload window, regardless of offset/n */
#define EDGE_OFFSET_MAX 15      /* max float/Complex element offset under test */
#define EDGE_MAX_N SK_TEST_MAX_N
#define EDGE_ARENA_LEN (EDGE_GUARD + EDGE_OFFSET_MAX + EDGE_MAX_N + EDGE_GUARD)
#define EDGE_CANARY_BITS 0x7fc0deadu
#define EDGE_FORM_COUNT 3

static float edge_canary_float(void) { return bits_to_float(EDGE_CANARY_BITS); }

static void *edge_aligned_alloc(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0 || p == NULL) {
        fprintf(stderr, "FATAL: posix_memalign(64, %zu) failed\n", bytes);
        exit(1);
    }
    return p;
}

static void edge_fill_canary_f(float *arena, int len) {
    int i;
    float c = edge_canary_float();
    for (i = 0; i < len; ++i) arena[i] = c;
}

/* Verifies every float in arena[0,len) OUTSIDE the payload window
 * [win_lo, win_lo+win_len) still holds the exact canary bit pattern -- an
 * empty window (win_len==0, i.e. n==0) means the ENTIRE arena must still be
 * canary, which is exactly the "n==0 performs zero reads/writes" contract
 * this section exists to check. exit(1) with a precise diagnostic on the
 * first violation, same house style as check_bits_or_die. */
static void edge_check_canary_f(const char *label, const float *arena, int len,
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

/* Complex-array counterparts: Complex is {float r; float i;} contiguous
 * (fft_wrapper.h), so a Complex arena is just a float arena with every
 * length/offset doubled -- reuses the float helpers above instead of
 * duplicating the canary logic for a second type. */
static void edge_fill_canary_c(Complex *arena, int len) {
    edge_fill_canary_f((float *)arena, len * 2);
}
static void edge_check_canary_c(const char *label, const Complex *arena, int len,
                                 int win_lo, int win_len) {
    edge_check_canary_f(label, (const float *)arena, len * 2, win_lo * 2, win_len * 2);
}

/* Derives the (input-role, output-role) element offsets for the matrix's
 * three forms. Form 2 ("both, different") uses `((o+7)%15)+1` rather than
 * the obvious mirror `16-o`: the mirror collides with o itself at the
 * midpoint (o==8 -> 16-8==8), silently degrading "both different" into
 * "both the same" for exactly one offset value. `(o+7)%15` is a fixed-
 * point-free derangement over {1..15} (o+7 == o (mod 15) requires
 * 7 == 0 (mod 15), which is false for every o), so out_off != in_off for
 * every o in 1..15 by construction, not by empirical luck. */
static void edge_offsets_for_form(int form, int o, int *in_off, int *out_off) {
    switch (form) {
    case 0: *in_off = o; *out_off = 0; break;                   /* input offset only */
    case 1: *in_off = 0; *out_off = o; break;                   /* output offset only */
    default: *in_off = o; *out_off = ((o + 7) % 15) + 1; break;  /* both, different */
    }
}

static void test_ema_edge(void) {
    float *x_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *state_ref_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *state_scalar_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *state_simd_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    const float alpha = 0.9f, beta = 0.1f;
    int ni, form, o;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (form = 0; form < EDGE_FORM_COUNT; ++form) {
            for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
                int in_off, out_off;
                edge_offsets_for_form(form, o, &in_off, &out_off);

                edge_fill_canary_f(x_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(state_ref_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(state_scalar_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(state_simd_arena, EDGE_ARENA_LEN);
                fill_floats(x_arena + in_off, n);
                fill_floats(state_ref_arena + out_off, n);
                memcpy(state_scalar_arena + out_off, state_ref_arena + out_off, (size_t)n * sizeof(float));
                memcpy(state_simd_arena + out_off, state_ref_arena + out_off, (size_t)n * sizeof(float));

                sk_ema_f32_scalar(state_scalar_arena + out_off, x_arena + in_off, alpha, beta, n);
                sk_ema_f32(state_simd_arena + out_off, x_arena + in_off, alpha, beta, n);

                check_bits_or_die("ema_f32_edge", n, form * 100 + o,
                                   state_simd_arena + out_off, state_scalar_arena + out_off, n);
                edge_check_canary_f("ema_f32_edge:x", x_arena, EDGE_ARENA_LEN, in_off, n);
                edge_check_canary_f("ema_f32_edge:state_scalar", state_scalar_arena, EDGE_ARENA_LEN, out_off, n);
                edge_check_canary_f("ema_f32_edge:state_simd", state_simd_arena, EDGE_ARENA_LEN, out_off, n);
            }
        }
    }
    free(x_arena); free(state_ref_arena); free(state_scalar_arena); free(state_simd_arena);
    printf("PASS ema_f32_edge (n=0..17+existing x offset 1..15 x 3 forms, canary-guarded)\n");
}

static void test_capply_gain_edge(void) {
    Complex *z_arena = (Complex *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(Complex));
    Complex *out_scalar_arena = (Complex *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(Complex));
    Complex *out_simd_arena = (Complex *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(Complex));
    float *g_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    int ni, form, o;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (form = 0; form < EDGE_FORM_COUNT; ++form) {
            for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
                int in_off, out_off;
                edge_offsets_for_form(form, o, &in_off, &out_off);

                edge_fill_canary_c(z_arena, EDGE_ARENA_LEN);
                edge_fill_canary_c(out_scalar_arena, EDGE_ARENA_LEN);
                edge_fill_canary_c(out_simd_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(g_arena, EDGE_ARENA_LEN);
                fill_complex(z_arena + in_off, n);
                fill_floats(g_arena, n); /* g stays at a fixed offset 0, see header note */

                sk_capply_gain_f32_scalar(out_scalar_arena + out_off, z_arena + in_off, g_arena, n);
                sk_capply_gain_f32(out_simd_arena + out_off, z_arena + in_off, g_arena, n);

                check_bits_or_die("capply_gain_f32_edge", n, form * 100 + o,
                                   (const float *)(out_simd_arena + out_off),
                                   (const float *)(out_scalar_arena + out_off), 2 * n);
                edge_check_canary_c("capply_gain_f32_edge:z", z_arena, EDGE_ARENA_LEN, in_off, n);
                edge_check_canary_c("capply_gain_f32_edge:out_scalar", out_scalar_arena, EDGE_ARENA_LEN, out_off, n);
                edge_check_canary_c("capply_gain_f32_edge:out_simd", out_simd_arena, EDGE_ARENA_LEN, out_off, n);
                edge_check_canary_f("capply_gain_f32_edge:g", g_arena, EDGE_ARENA_LEN, 0, n);
            }
        }
    }
    free(z_arena); free(out_scalar_arena); free(out_simd_arena); free(g_arena);
    printf("PASS capply_gain_f32_edge (n=0..17+existing x offset 1..15 x 3 forms, canary-guarded)\n");
}

static void test_cadd_edge(void) {
    Complex *a_arena = (Complex *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(Complex));
    Complex *b_arena = (Complex *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(Complex));
    Complex *out_scalar_arena = (Complex *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(Complex));
    Complex *out_simd_arena = (Complex *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(Complex));
    int ni, form, o;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (form = 0; form < EDGE_FORM_COUNT; ++form) {
            for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
                int in_off, out_off;
                edge_offsets_for_form(form, o, &in_off, &out_off);

                edge_fill_canary_c(a_arena, EDGE_ARENA_LEN);
                edge_fill_canary_c(b_arena, EDGE_ARENA_LEN);
                edge_fill_canary_c(out_scalar_arena, EDGE_ARENA_LEN);
                edge_fill_canary_c(out_simd_arena, EDGE_ARENA_LEN);
                fill_complex(a_arena + in_off, n);
                fill_complex(b_arena, n); /* b stays at a fixed offset 0 */

                sk_cadd_f32_scalar(out_scalar_arena + out_off, a_arena + in_off, b_arena, n);
                sk_cadd_f32(out_simd_arena + out_off, a_arena + in_off, b_arena, n);

                check_bits_or_die("cadd_f32_edge", n, form * 100 + o,
                                   (const float *)(out_simd_arena + out_off),
                                   (const float *)(out_scalar_arena + out_off), 2 * n);
                edge_check_canary_c("cadd_f32_edge:a", a_arena, EDGE_ARENA_LEN, in_off, n);
                edge_check_canary_c("cadd_f32_edge:b", b_arena, EDGE_ARENA_LEN, 0, n);
                edge_check_canary_c("cadd_f32_edge:out_scalar", out_scalar_arena, EDGE_ARENA_LEN, out_off, n);
                edge_check_canary_c("cadd_f32_edge:out_simd", out_simd_arena, EDGE_ARENA_LEN, out_off, n);
            }
        }
    }
    free(a_arena); free(b_arena); free(out_scalar_arena); free(out_simd_arena);
    printf("PASS cadd_f32_edge (n=0..17+existing x offset 1..15 x 3 forms, canary-guarded)\n");
}

static void test_sq_scale_edge(void) {
    float *x_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *out_scalar_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *out_simd_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    const float scale = 0.5f;
    int ni, form, o;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (form = 0; form < EDGE_FORM_COUNT; ++form) {
            for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
                int in_off, out_off;
                edge_offsets_for_form(form, o, &in_off, &out_off);

                edge_fill_canary_f(x_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(out_scalar_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(out_simd_arena, EDGE_ARENA_LEN);
                fill_floats(x_arena + in_off, n);

                sk_sq_scale_f32_scalar(x_arena + in_off, scale, out_scalar_arena + out_off, n);
                sk_sq_scale_f32(x_arena + in_off, scale, out_simd_arena + out_off, n);

                check_bits_or_die("sq_scale_f32_edge", n, form * 100 + o,
                                   out_simd_arena + out_off, out_scalar_arena + out_off, n);
                edge_check_canary_f("sq_scale_f32_edge:x", x_arena, EDGE_ARENA_LEN, in_off, n);
                edge_check_canary_f("sq_scale_f32_edge:out_scalar", out_scalar_arena, EDGE_ARENA_LEN, out_off, n);
                edge_check_canary_f("sq_scale_f32_edge:out_simd", out_simd_arena, EDGE_ARENA_LEN, out_off, n);
            }
        }
    }
    free(x_arena); free(out_scalar_arena); free(out_simd_arena);
    printf("PASS sq_scale_f32_edge (n=0..17+existing x offset 1..15 x 3 forms, canary-guarded)\n");
}

static void test_min_edge(void) {
    float *a_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *b_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *out_scalar_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *out_simd_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    int ni, form, o;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (form = 0; form < EDGE_FORM_COUNT; ++form) {
            for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
                int in_off, out_off;
                edge_offsets_for_form(form, o, &in_off, &out_off);

                edge_fill_canary_f(a_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(b_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(out_scalar_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(out_simd_arena, EDGE_ARENA_LEN);
                fill_floats(a_arena + in_off, n);
                fill_floats(b_arena, n);

                sk_min_f32_scalar(out_scalar_arena + out_off, a_arena + in_off, b_arena, n);
                sk_min_f32(out_simd_arena + out_off, a_arena + in_off, b_arena, n);

                check_bits_or_die("min_f32_edge", n, form * 100 + o,
                                   out_simd_arena + out_off, out_scalar_arena + out_off, n);
                edge_check_canary_f("min_f32_edge:a", a_arena, EDGE_ARENA_LEN, in_off, n);
                edge_check_canary_f("min_f32_edge:b", b_arena, EDGE_ARENA_LEN, 0, n);
                edge_check_canary_f("min_f32_edge:out_scalar", out_scalar_arena, EDGE_ARENA_LEN, out_off, n);
                edge_check_canary_f("min_f32_edge:out_simd", out_simd_arena, EDGE_ARENA_LEN, out_off, n);
            }
        }
    }
    free(a_arena); free(b_arena); free(out_scalar_arena); free(out_simd_arena);
    printf("PASS min_f32_edge (n=0..17+existing x offset 1..15 x 3 forms, canary-guarded)\n");
}

static void test_clip_edge(void) {
    float *x_ref_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *x_scalar_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *x_simd_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    const float lo = -1.0f, hi = 1.0f;
    int ni, o;
    /* sk_clip_f32 is a single in-place buffer (no separate input/output
     * array), so the 3-form input/output split used by every other kernel
     * in this section doesn't apply here -- there is exactly one buffer
     * role to offset, tested directly at each of the 15 offsets (documented
     * deviation, see this section's header comment). */
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
            edge_fill_canary_f(x_ref_arena, EDGE_ARENA_LEN);
            edge_fill_canary_f(x_scalar_arena, EDGE_ARENA_LEN);
            edge_fill_canary_f(x_simd_arena, EDGE_ARENA_LEN);
            fill_floats(x_ref_arena + o, n);
            memcpy(x_scalar_arena + o, x_ref_arena + o, (size_t)n * sizeof(float));
            memcpy(x_simd_arena + o, x_ref_arena + o, (size_t)n * sizeof(float));

            sk_clip_f32_scalar(x_scalar_arena + o, lo, hi, n);
            sk_clip_f32(x_simd_arena + o, lo, hi, n);

            check_bits_or_die("clip_f32_edge", n, o, x_simd_arena + o, x_scalar_arena + o, n);
            edge_check_canary_f("clip_f32_edge:x_scalar", x_scalar_arena, EDGE_ARENA_LEN, o, n);
            edge_check_canary_f("clip_f32_edge:x_simd", x_simd_arena, EDGE_ARENA_LEN, o, n);
        }
    }
    free(x_ref_arena); free(x_scalar_arena); free(x_simd_arena);
    printf("PASS clip_f32_edge (n=0..17+existing x offset 1..15, single in-place buffer, canary-guarded)\n");
}

static void test_fast_sqrt_edge(void) {
    float *x_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *out_scalar_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *out_simd_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    int ni, form, o;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (form = 0; form < EDGE_FORM_COUNT; ++form) {
            for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
                int in_off, out_off;
                edge_offsets_for_form(form, o, &in_off, &out_off);

                edge_fill_canary_f(x_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(out_scalar_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(out_simd_arena, EDGE_ARENA_LEN);
                fill_floats(x_arena + in_off, n);

                sk_fast_sqrt_f32_scalar(x_arena + in_off, out_scalar_arena + out_off, n);
                sk_fast_sqrt_f32(x_arena + in_off, out_simd_arena + out_off, n);

                check_bits_or_die("fast_sqrt_f32_edge", n, form * 100 + o,
                                   out_simd_arena + out_off, out_scalar_arena + out_off, n);
                edge_check_canary_f("fast_sqrt_f32_edge:x", x_arena, EDGE_ARENA_LEN, in_off, n);
                edge_check_canary_f("fast_sqrt_f32_edge:out_scalar", out_scalar_arena, EDGE_ARENA_LEN, out_off, n);
                edge_check_canary_f("fast_sqrt_f32_edge:out_simd", out_simd_arena, EDGE_ARENA_LEN, out_off, n);
            }
        }
    }
    free(x_arena); free(out_scalar_arena); free(out_simd_arena);
    printf("PASS fast_sqrt_f32_edge (n=0..17+existing x offset 1..15 x 3 forms, canary-guarded)\n");
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

static void bench_fast_exp(void) {
    float x[BENCH_N], out[BENCH_N];
    fill_bench_floats(x, BENCH_N); /* ~[-1,1], well within the fast domain */
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_fast_exp_f32_scalar(x, out, BENCH_N); g_bench_sink += out[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_fast_exp_f32(x, out, BENCH_N); g_bench_sink += out[0]; }
            {
                double t3 = now_ns();
                report_bench("fast_exp_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

static void bench_fast_exp_neg(void) {
    float x[BENCH_N], out[BENCH_N];
    int i;
    fill_bench_floats(x, BENCH_N);
    for (i = 0; i < BENCH_N; ++i) x[i] = fabsf(x[i]); /* SPP-shaped: non-negative */
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_fast_exp_neg_f32_scalar(x, out, BENCH_N); g_bench_sink += out[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_fast_exp_neg_f32(x, out, BENCH_N); g_bench_sink += out[0]; }
            {
                double t3 = now_ns();
                report_bench("fast_exp_neg_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

static void bench_fast_log(void) {
    float x[BENCH_N], out[BENCH_N];
    int i;
    fill_bench_floats(x, BENCH_N);
    for (i = 0; i < BENCH_N; ++i) x[i] = x[i] * x[i] + 1e-3f; /* positive domain */
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_fast_log_f32_scalar(x, out, BENCH_N); g_bench_sink += out[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_fast_log_f32(x, out, BENCH_N); g_bench_sink += out[0]; }
            {
                double t3 = now_ns();
                report_bench("fast_log_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

static void bench_exp1_approx(void) {
    float x[BENCH_N], out[BENCH_N];
    int i;
    fill_bench_floats(x, BENCH_N);
    for (i = 0; i < BENCH_N; ++i) x[i] = x[i] * x[i] + 1e-3f; /* SNR-ratio-shaped: positive */
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_exp1_approx_f32_scalar(x, out, BENCH_N); g_bench_sink += out[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_exp1_approx_f32(x, out, BENCH_N); g_bench_sink += out[0]; }
            {
                double t3 = now_ns();
                report_bench("exp1_approx_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

static void bench_mcra_noise_update(void) {
    float npsd[BENCH_N], spp[BENCH_N], power[BENCH_N];
    int i;
    fill_bench_floats(npsd, BENCH_N);
    fill_bench_floats(spp, BENCH_N);
    fill_bench_floats(power, BENCH_N);
    for (i = 0; i < BENCH_N; ++i) { npsd[i] = fabsf(npsd[i]) + 1e-3f; spp[i] = fabsf(spp[i]); power[i] = fabsf(power[i]); }
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_mcra_noise_update_f32_scalar(npsd, spp, power, 0.95f, 0.7f, BENCH_N); g_bench_sink += npsd[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_mcra_noise_update_f32(npsd, spp, power, 0.95f, 0.7f, BENCH_N); g_bench_sink += npsd[0]; }
            {
                double t3 = now_ns();
                report_bench("mcra_noise_update_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
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
    test_fast_math_special_values();

    printf("\n--- s4-audio-common-sweep: fast_math.h exp/log family + mcra EMA kernel ---\n");
    test_fast_exp();
    test_fast_exp_neg();
    test_fast_log();
    test_fast_log10();
    test_exp1_approx();
    test_exp_log_boundaries();
    test_exp_log_family_inplace();
    test_mcra_noise_update();

    printf("\n--- alignment + canary edge-case matrix (round-3 review B05) ---\n");
    test_ema_edge();
    test_capply_gain_edge();
    test_cadd_edge();
    test_sq_scale_edge();
    test_min_edge();
    test_clip_edge();
    test_fast_sqrt_edge();

    printf("\n--- microbenchmarks (n=%d, %d reps) ---\n", BENCH_N, BENCH_REPS);
    bench_ema();
    bench_capply_gain();
    bench_cadd();
    bench_sq_scale();
    bench_min();
    bench_clip();
    bench_fast_sqrt();
    bench_fast_exp();
    bench_fast_exp_neg();
    bench_fast_log();
    bench_exp1_approx();
    bench_mcra_noise_update();

    printf("\nALL PASS (SK_HAVE_NEON=%d)\n", SK_HAVE_NEON);
    printf("TOTAL CHECKS: %ld\n", g_total_checks);
    (void)g_bench_sink;
    return 0;
}
