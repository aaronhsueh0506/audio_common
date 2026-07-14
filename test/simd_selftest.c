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

static void check_scalar_bits_or_die(const char *kernel, int n, int trial,
                                      float simd_val, float scalar_val) {
    uint32_t gb, wb;
    memcpy(&gb, &simd_val, sizeof gb);
    memcpy(&wb, &scalar_val, sizeof wb);
    if (gb != wb) {
        fprintf(stderr,
            "MISMATCH kernel=%s n=%d trial=%d idx=0 simd=0x%08x (%.9g) scalar=0x%08x (%.9g)\n",
            kernel, n, trial, (unsigned)gb, (double)simd_val, (unsigned)wb, (double)scalar_val);
        exit(1);
    }
}

/* ═══════════════════════════ correctness: kernel 1 ═══════════════════════ */

static void test_cabs_np(void) {
    Complex z[SK_TEST_MAX_N];
    float out_scalar[SK_TEST_MAX_N], out_simd[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_complex(z, n);
            sk_cabs_np_f32_scalar(z, out_scalar, n);
            sk_cabs_np_f32(z, out_simd, n);
            check_bits_or_die("cabs_np_f32", n, t, out_simd, out_scalar, n);
        }
    }
    printf("PASS cabs_np_f32\n");
}

/* ═══════════════════════════ correctness: kernel 2 ═══════════════════════ */

static void test_cmag2_np(void) {
    Complex z[SK_TEST_MAX_N];
    float out_scalar[SK_TEST_MAX_N], out_simd[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_complex(z, n);
            sk_cmag2_np_f32_scalar(z, out_scalar, n);
            sk_cmag2_np_f32(z, out_simd, n);
            check_bits_or_die("cmag2_np_f32", n, t, out_simd, out_scalar, n);
        }
    }
    printf("PASS cmag2_np_f32\n");
}

/* ═══════════════════════════ correctness: kernel 3 ═══════════════════════ */

static void test_cmag2_np_acc(void) {
    Complex z[SK_TEST_MAX_N];
    float acc_init[SK_TEST_MAX_N], acc_scalar[SK_TEST_MAX_N], acc_simd[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_complex(z, n);
            fill_floats(acc_init, n);
            memcpy(acc_scalar, acc_init, (size_t)n * sizeof(float));
            memcpy(acc_simd, acc_init, (size_t)n * sizeof(float));
            sk_cmag2_np_acc_f32_scalar(z, acc_scalar, n);
            sk_cmag2_np_acc_f32(z, acc_simd, n);
            check_bits_or_die("cmag2_np_acc_f32", n, t, acc_simd, acc_scalar, n);
        }
    }
    printf("PASS cmag2_np_acc_f32\n");
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

/* ═══════════════════════════ correctness: kernel 5 ═══════════════════════ */

static void test_ema_cmag2(void) {
    Complex z[SK_TEST_MAX_N];
    float state_init[SK_TEST_MAX_N], state_scalar[SK_TEST_MAX_N], state_simd[SK_TEST_MAX_N];
    int ni, t;
    const float alpha = 0.9f, beta = 0.1f;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_complex(z, n);
            fill_floats(state_init, n);
            memcpy(state_scalar, state_init, (size_t)n * sizeof(float));
            memcpy(state_simd, state_init, (size_t)n * sizeof(float));
            sk_ema_cmag2_f32_scalar(state_scalar, z, alpha, beta, n);
            sk_ema_cmag2_f32(state_simd, z, alpha, beta, n);
            check_bits_or_die("ema_cmag2_f32", n, t, state_simd, state_scalar, n);
        }
    }
    printf("PASS ema_cmag2_f32\n");
}

/* ═══════════════════════════ correctness: kernel 6 ═══════════════════════ */

static void test_cmac_np(void) {
    Complex w[SK_TEST_MAX_N], x[SK_TEST_MAX_N];
    Complex acc_init[SK_TEST_MAX_N], acc_scalar[SK_TEST_MAX_N], acc_simd[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_complex(w, n);
            fill_complex(x, n);
            fill_complex(acc_init, n);
            memcpy(acc_scalar, acc_init, (size_t)n * sizeof(Complex));
            memcpy(acc_simd, acc_init, (size_t)n * sizeof(Complex));
            sk_cmac_np_f32_scalar(acc_scalar, w, x, n);
            sk_cmac_np_f32(acc_simd, w, x, n);
            check_bits_or_die("cmac_np_f32", n, t, (const float *)acc_simd, (const float *)acc_scalar, 2 * n);
        }
    }
    printf("PASS cmac_np_f32\n");
}

/* ═══════════════════════════ correctness: kernel 7 ═══════════════════════ */

static void test_wupdate_nlms(void) {
    Complex X[SK_TEST_MAX_N], err[SK_TEST_MAX_N];
    Complex W_init[SK_TEST_MAX_N], W_scalar[SK_TEST_MAX_N], W_simd[SK_TEST_MAX_N];
    float mu_eff[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_complex(X, n);
            fill_complex(err, n);
            fill_complex(W_init, n);
            fill_floats(mu_eff, n);
            memcpy(W_scalar, W_init, (size_t)n * sizeof(Complex));
            memcpy(W_simd, W_init, (size_t)n * sizeof(Complex));
            sk_wupdate_nlms_f32_scalar(W_scalar, X, err, mu_eff, n);
            sk_wupdate_nlms_f32(W_simd, X, err, mu_eff, n);
            check_bits_or_die("wupdate_nlms_f32", n, t, (const float *)W_simd, (const float *)W_scalar, 2 * n);
        }
    }
    printf("PASS wupdate_nlms_f32\n");
}

/* ═══════════════════════════ correctness: kernel 8 ═══════════════════════ */

static void test_wupdate_kf(void) {
    Complex X[SK_TEST_MAX_N], err[SK_TEST_MAX_N];
    Complex W_init[SK_TEST_MAX_N], W_scalar[SK_TEST_MAX_N], W_simd[SK_TEST_MAX_N];
    float mu[SK_TEST_MAX_N], mu_scale[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_complex(X, n);
            fill_complex(err, n);
            fill_complex(W_init, n);
            fill_floats(mu, n);
            fill_floats(mu_scale, n);
            memcpy(W_scalar, W_init, (size_t)n * sizeof(Complex));
            memcpy(W_simd, W_init, (size_t)n * sizeof(Complex));
            sk_wupdate_kf_f32_scalar(W_scalar, X, err, mu, mu_scale, n);
            sk_wupdate_kf_f32(W_simd, X, err, mu, mu_scale, n);
            check_bits_or_die("wupdate_kf_f32", n, t, (const float *)W_simd, (const float *)W_scalar, 2 * n);
        }
    }
    printf("PASS wupdate_kf_f32\n");
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
    printf("PASS clip_f32\n");
}

/* ═══════════════════════════ correctness: kernel 13 ══════════════════════ */

static void test_pairwise_sum(void) {
    float a[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(a, n);
            {
                float rs = sk_pairwise_sum_f32_scalar(a, (size_t)n);
                float rn = sk_pairwise_sum_f32(a, (size_t)n);
                check_scalar_bits_or_die("pairwise_sum_f32", n, t, rn, rs);
            }
        }
    }
    printf("PASS pairwise_sum_f32\n");
}

/* ═══════════════════════════ correctness: kernel 14 ══════════════════════ */

static void test_sum_sq_pairwise(void) {
    float a[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(a, n);
            {
                float rs = sk_sum_sq_pairwise_f32_scalar(a, (size_t)n);
                float rn = sk_sum_sq_pairwise_f32(a, (size_t)n);
                check_scalar_bits_or_die("sum_sq_pairwise_f32", n, t, rn, rs);
            }
        }
    }
    printf("PASS sum_sq_pairwise_f32\n");
}

/* ═══════════════════ correctness: kernels 21/22 (tail-fold pairwise) ══════
 * Dedicated n-list covering the leaf/split boundaries specific to these two
 * kernels' recursion (127/128/129 straddle the n<=128 leaf cutover; 960
 * exercises >=2 levels of the half-rounded-to-a-multiple-of-8 split; 7/8/9
 * straddle the small-n vs. leaf cutover that differs between kernel 21 and
 * kernel 13 -- see kernel 21's header comment). Separate, larger backing
 * buffer (960) since this exceeds SK_TEST_MAX_N (512), used only here. */

#define PW_TAILFOLD_MAX_N 960
static const int PW_TAILFOLD_N_LIST[] = {1, 7, 8, 9, 127, 128, 129, 160, 255, 256, 257, 512, 960};
#define PW_TAILFOLD_N_LIST_COUNT ((int)(sizeof(PW_TAILFOLD_N_LIST) / sizeof(PW_TAILFOLD_N_LIST[0])))

static void test_pairwise_sum_tailfold(void) {
    static float a[PW_TAILFOLD_MAX_N];
    int ni, t;
    for (ni = 0; ni < PW_TAILFOLD_N_LIST_COUNT; ++ni) {
        int n = PW_TAILFOLD_N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(a, n);
            {
                float rs = sk_pairwise_sum_tailfold_f32_scalar(a, (size_t)n);
                float rn = sk_pairwise_sum_tailfold_f32(a, (size_t)n);
                check_scalar_bits_or_die("pairwise_sum_tailfold_f32", n, t, rn, rs);
            }
        }
    }
    /* dedicated signed-zero small-n checks (see header comment: this
     * kernel's 0.0f-seeded small-n accumulator normalizes -0.0f to +0.0f,
     * a bit pattern that must still round-trip scalar==NEON identically). */
    {
        float az1[1] = { -0.0f };
        float rs = sk_pairwise_sum_tailfold_f32_scalar(az1, 1);
        float rn = sk_pairwise_sum_tailfold_f32(az1, 1);
        check_scalar_bits_or_die("pairwise_sum_tailfold_f32_negzero_n1", 1, 0, rn, rs);
    }
    {
        float az5[5] = { -0.0f, -0.0f, -0.0f, -0.0f, -0.0f };
        float rs = sk_pairwise_sum_tailfold_f32_scalar(az5, 5);
        float rn = sk_pairwise_sum_tailfold_f32(az5, 5);
        check_scalar_bits_or_die("pairwise_sum_tailfold_f32_negzero_n5", 5, 0, rn, rs);
    }
    printf("PASS pairwise_sum_tailfold_f32\n");
}

static void test_pairwise_sum_tailfold_b(void) {
    static float a[PW_TAILFOLD_MAX_N];
    int ni, t;
    for (ni = 0; ni < PW_TAILFOLD_N_LIST_COUNT; ++ni) {
        int n = PW_TAILFOLD_N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(a, n);
            {
                float rs = sk_pairwise_sum_tailfold_b_f32_scalar(a, (size_t)n);
                float rn = sk_pairwise_sum_tailfold_b_f32(a, (size_t)n);
                check_scalar_bits_or_die("pairwise_sum_tailfold_b_f32", n, t, rn, rs);
            }
        }
    }
    /* n==0 explicit-return path. */
    {
        float dummy[1] = { 1.0f };
        float rs = sk_pairwise_sum_tailfold_b_f32_scalar(dummy, 0);
        float rn = sk_pairwise_sum_tailfold_b_f32(dummy, 0);
        check_scalar_bits_or_die("pairwise_sum_tailfold_b_f32_n0", 0, 0, rn, rs);
    }
    /* dedicated signed-zero small-n checks (see header comment: this
     * kernel's a[0]-seeded small-n accumulator preserves -0.0f as-is,
     * unlike kernel 21 -- both must still be scalar==NEON internally). */
    {
        float az1[1] = { -0.0f };
        float rs = sk_pairwise_sum_tailfold_b_f32_scalar(az1, 1);
        float rn = sk_pairwise_sum_tailfold_b_f32(az1, 1);
        check_scalar_bits_or_die("pairwise_sum_tailfold_b_f32_negzero_n1", 1, 0, rn, rs);
    }
    {
        float az5[5] = { -0.0f, -0.0f, -0.0f, -0.0f, -0.0f };
        float rs = sk_pairwise_sum_tailfold_b_f32_scalar(az5, 5);
        float rn = sk_pairwise_sum_tailfold_b_f32(az5, 5);
        check_scalar_bits_or_die("pairwise_sum_tailfold_b_f32_negzero_n5", 5, 0, rn, rs);
    }
    printf("PASS pairwise_sum_tailfold_b_f32\n");
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

static void bench_cabs_np(void) {
    Complex z[BENCH_N]; float out[BENCH_N];
    fill_bench_complex(z, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_cabs_np_f32_scalar(z, out, BENCH_N); g_bench_sink += out[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_cabs_np_f32(z, out, BENCH_N); g_bench_sink += out[0]; }
            {
                double t3 = now_ns();
                report_bench("cabs_np_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

static void bench_cmag2_np(void) {
    Complex z[BENCH_N]; float out[BENCH_N];
    fill_bench_complex(z, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_cmag2_np_f32_scalar(z, out, BENCH_N); g_bench_sink += out[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_cmag2_np_f32(z, out, BENCH_N); g_bench_sink += out[0]; }
            {
                double t3 = now_ns();
                report_bench("cmag2_np_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

static void bench_cmag2_np_acc(void) {
    Complex z[BENCH_N]; float acc[BENCH_N];
    fill_bench_complex(z, BENCH_N);
    fill_bench_floats(acc, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_cmag2_np_acc_f32_scalar(z, acc, BENCH_N); g_bench_sink += acc[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_cmag2_np_acc_f32(z, acc, BENCH_N); g_bench_sink += acc[0]; }
            {
                double t3 = now_ns();
                report_bench("cmag2_np_acc_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
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

static void bench_ema_cmag2(void) {
    Complex z[BENCH_N]; float state[BENCH_N];
    fill_bench_complex(z, BENCH_N);
    fill_bench_floats(state, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_ema_cmag2_f32_scalar(state, z, 0.9f, 0.1f, BENCH_N); g_bench_sink += state[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_ema_cmag2_f32(state, z, 0.9f, 0.1f, BENCH_N); g_bench_sink += state[0]; }
            {
                double t3 = now_ns();
                report_bench("ema_cmag2_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

static void bench_cmac_np(void) {
    Complex w[BENCH_N], x[BENCH_N], acc[BENCH_N];
    fill_bench_complex(w, BENCH_N);
    fill_bench_complex(x, BENCH_N);
    fill_bench_complex(acc, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_cmac_np_f32_scalar(acc, w, x, BENCH_N); g_bench_sink += acc[0].r; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_cmac_np_f32(acc, w, x, BENCH_N); g_bench_sink += acc[0].r; }
            {
                double t3 = now_ns();
                report_bench("cmac_np_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

static void bench_wupdate_nlms(void) {
    Complex X[BENCH_N], err[BENCH_N], W[BENCH_N];
    float mu_eff[BENCH_N];
    fill_bench_complex(X, BENCH_N);
    fill_bench_complex(err, BENCH_N);
    fill_bench_complex(W, BENCH_N);
    fill_bench_floats(mu_eff, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_wupdate_nlms_f32_scalar(W, X, err, mu_eff, BENCH_N); g_bench_sink += W[0].r; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_wupdate_nlms_f32(W, X, err, mu_eff, BENCH_N); g_bench_sink += W[0].r; }
            {
                double t3 = now_ns();
                report_bench("wupdate_nlms_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

static void bench_wupdate_kf(void) {
    Complex X[BENCH_N], err[BENCH_N], W[BENCH_N];
    float mu[BENCH_N], mu_scale[BENCH_N];
    fill_bench_complex(X, BENCH_N);
    fill_bench_complex(err, BENCH_N);
    fill_bench_complex(W, BENCH_N);
    fill_bench_floats(mu, BENCH_N);
    fill_bench_floats(mu_scale, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_wupdate_kf_f32_scalar(W, X, err, mu, mu_scale, BENCH_N); g_bench_sink += W[0].r; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_wupdate_kf_f32(W, X, err, mu, mu_scale, BENCH_N); g_bench_sink += W[0].r; }
            {
                double t3 = now_ns();
                report_bench("wupdate_kf_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
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

static void bench_pairwise_sum(void) {
    float a[BENCH_N];
    fill_bench_floats(a, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) g_bench_sink += sk_pairwise_sum_f32_scalar(a, (size_t)BENCH_N);
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) g_bench_sink += sk_pairwise_sum_f32(a, (size_t)BENCH_N);
            {
                double t3 = now_ns();
                report_bench("pairwise_sum_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

static void bench_sum_sq_pairwise(void) {
    float a[BENCH_N];
    fill_bench_floats(a, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) g_bench_sink += sk_sum_sq_pairwise_f32_scalar(a, (size_t)BENCH_N);
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) g_bench_sink += sk_sum_sq_pairwise_f32(a, (size_t)BENCH_N);
            {
                double t3 = now_ns();
                report_bench("sum_sq_pairwise_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

static void bench_pairwise_sum_tailfold(void) {
    float a[BENCH_N];
    fill_bench_floats(a, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) g_bench_sink += sk_pairwise_sum_tailfold_f32_scalar(a, (size_t)BENCH_N);
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) g_bench_sink += sk_pairwise_sum_tailfold_f32(a, (size_t)BENCH_N);
            {
                double t3 = now_ns();
                report_bench("pairwise_sum_tailfold_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

static void bench_pairwise_sum_tailfold_b(void) {
    float a[BENCH_N];
    fill_bench_floats(a, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) g_bench_sink += sk_pairwise_sum_tailfold_b_f32_scalar(a, (size_t)BENCH_N);
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) g_bench_sink += sk_pairwise_sum_tailfold_b_f32(a, (size_t)BENCH_N);
            {
                double t3 = now_ns();
                report_bench("pairwise_sum_tailfold_b_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
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

/* ═══════════════════════════ correctness: kernel 16 ══════════════════════ */

static void test_coherence_ema_gate(void) {
    Complex echo[SK_TEST_MAX_N], near_spec[SK_TEST_MAX_N];
    float abs_echo[SK_TEST_MAX_N], abs_near[SK_TEST_MAX_N];
    float sye_re_init[SK_TEST_MAX_N], sye_im_init[SK_TEST_MAX_N];
    float syy_init[SK_TEST_MAX_N], see_init[SK_TEST_MAX_N];
    float sye_re_s[SK_TEST_MAX_N], sye_im_s[SK_TEST_MAX_N];
    float syy_s[SK_TEST_MAX_N], see_s[SK_TEST_MAX_N];
    float sye_re_n[SK_TEST_MAX_N], sye_im_n[SK_TEST_MAX_N];
    float syy_n[SK_TEST_MAX_N], see_n[SK_TEST_MAX_N];
    unsigned char mask_s[SK_TEST_MAX_N], mask_n[SK_TEST_MAX_N];
    int ni, t;
    const float alpha = 0.05f, threshold = 0.5f;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_complex(echo, n);
            fill_complex(near_spec, n);
            fill_floats(abs_echo, n);
            fill_floats(abs_near, n);
            fill_floats(sye_re_init, n);
            fill_floats(sye_im_init, n);
            fill_floats(syy_init, n);
            fill_floats(see_init, n);
            memcpy(sye_re_s, sye_re_init, (size_t)n * sizeof(float));
            memcpy(sye_im_s, sye_im_init, (size_t)n * sizeof(float));
            memcpy(syy_s, syy_init, (size_t)n * sizeof(float));
            memcpy(see_s, see_init, (size_t)n * sizeof(float));
            memcpy(sye_re_n, sye_re_init, (size_t)n * sizeof(float));
            memcpy(sye_im_n, sye_im_init, (size_t)n * sizeof(float));
            memcpy(syy_n, syy_init, (size_t)n * sizeof(float));
            memcpy(see_n, see_init, (size_t)n * sizeof(float));
            sk_coherence_ema_gate_f32_scalar(sye_re_s, sye_im_s, syy_s, see_s,
                                              echo, near_spec, abs_echo, abs_near,
                                              alpha, threshold, mask_s, n);
            sk_coherence_ema_gate_f32(sye_re_n, sye_im_n, syy_n, see_n,
                                       echo, near_spec, abs_echo, abs_near,
                                       alpha, threshold, mask_n, n);
            check_bits_or_die("coherence_ema_gate_f32:sye_re", n, t, sye_re_n, sye_re_s, n);
            check_bits_or_die("coherence_ema_gate_f32:sye_im", n, t, sye_im_n, sye_im_s, n);
            check_bits_or_die("coherence_ema_gate_f32:syy", n, t, syy_n, syy_s, n);
            check_bits_or_die("coherence_ema_gate_f32:see", n, t, see_n, see_s, n);
            {
                int idx;
                for (idx = 0; idx < n; ++idx) {
                    if (mask_s[idx] != mask_n[idx]) {
                        fprintf(stderr,
                            "MISMATCH kernel=coherence_ema_gate_f32:mask n=%d trial=%d idx=%d simd=%u scalar=%u\n",
                            n, t, idx, (unsigned)mask_n[idx], (unsigned)mask_s[idx]);
                        exit(1);
                    }
                }
            }
        }
    }
    printf("PASS coherence_ema_gate_f32\n");
}

/* ═══════════════════════════ correctness: kernel 17 ══════════════════════ */

static void test_ema_delta(void) {
    float state_init[SK_TEST_MAX_N], state_scalar[SK_TEST_MAX_N], state_simd[SK_TEST_MAX_N];
    float x[SK_TEST_MAX_N];
    int ni, t;
    const float alpha = 0.23156652857908377f; /* cng_y2_alpha */
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(state_init, n);
            fill_floats(x, n);
            memcpy(state_scalar, state_init, (size_t)n * sizeof(float));
            memcpy(state_simd, state_init, (size_t)n * sizeof(float));
            sk_ema_delta_f32_scalar(state_scalar, x, alpha, n);
            sk_ema_delta_f32(state_simd, x, alpha, n);
            check_bits_or_die("ema_delta_f32", n, t, state_simd, state_scalar, n);
        }
    }
    printf("PASS ema_delta_f32\n");
}

/* ═══════════════════════════ correctness: kernel 18 ══════════════════════ */

static void test_n2_track(void) {
    float n2_init[SK_TEST_MAX_N], n2_scalar[SK_TEST_MAX_N], n2_simd[SK_TEST_MAX_N];
    float y2s[SK_TEST_MAX_N];
    int ni, t;
    const float fresh = 0.9968377223398316f;
    const float retain = 0.003162277660168411f;
    const float g_up = 1.0005000750025f;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(n2_init, n);
            fill_floats(y2s, n);
            memcpy(n2_scalar, n2_init, (size_t)n * sizeof(float));
            memcpy(n2_simd, n2_init, (size_t)n * sizeof(float));
            sk_n2_track_f32_scalar(n2_scalar, y2s, fresh, retain, g_up, n);
            sk_n2_track_f32(n2_simd, y2s, fresh, retain, g_up, n);
            check_bits_or_die("n2_track_f32", n, t, n2_simd, n2_scalar, n);
        }
    }
    printf("PASS n2_track_f32\n");
}

/* ═══════════════════════════ correctness: kernel 19 ══════════════════════ */

static void test_n2_initial_track(void) {
    float n2i_init[SK_TEST_MAX_N], n2i_scalar[SK_TEST_MAX_N], n2i_simd[SK_TEST_MAX_N];
    float n2[SK_TEST_MAX_N];
    int ni, t;
    const float alpha = 0.0024981253125391234f;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(n2i_init, n);
            fill_floats(n2, n);
            memcpy(n2i_scalar, n2i_init, (size_t)n * sizeof(float));
            memcpy(n2i_simd, n2i_init, (size_t)n * sizeof(float));
            sk_n2_initial_track_f32_scalar(n2i_scalar, n2, alpha, n);
            sk_n2_initial_track_f32(n2i_simd, n2, alpha, n);
            check_bits_or_die("n2_initial_track_f32", n, t, n2i_simd, n2i_scalar, n);
        }
    }
    printf("PASS n2_initial_track_f32\n");
}

/* ═══════════════════════════ correctness: kernel 20 ══════════════════════ */

static void test_mask_zero(void) {
    float x_init[SK_TEST_MAX_N], x_scalar[SK_TEST_MAX_N], x_simd[SK_TEST_MAX_N];
    unsigned char mask[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            int i;
            fill_floats(x_init, n);
            for (i = 0; i < n; ++i) mask[i] = (unsigned char)(lcg_next() & 1u);
            memcpy(x_scalar, x_init, (size_t)n * sizeof(float));
            memcpy(x_simd, x_init, (size_t)n * sizeof(float));
            sk_mask_zero_f32_scalar(x_scalar, mask, n);
            sk_mask_zero_f32(x_simd, mask, n);
            check_bits_or_die("mask_zero_f32", n, t, x_simd, x_scalar, n);
        }
    }
    /* dedicated all-ones / all-zeros boundary check. */
    {
        float xa_init[8], xa_scalar[8], xa_simd[8];
        unsigned char mall1[8], mall0[8];
        int i;
        fill_floats(xa_init, 8);
        for (i = 0; i < 8; ++i) { mall1[i] = 1; mall0[i] = 0; }
        memcpy(xa_scalar, xa_init, sizeof(xa_init));
        memcpy(xa_simd, xa_init, sizeof(xa_init));
        sk_mask_zero_f32_scalar(xa_scalar, mall1, 8);
        sk_mask_zero_f32(xa_simd, mall1, 8);
        check_bits_or_die("mask_zero_f32_all1", 8, 0, xa_simd, xa_scalar, 8);
        memcpy(xa_scalar, xa_init, sizeof(xa_init));
        memcpy(xa_simd, xa_init, sizeof(xa_init));
        sk_mask_zero_f32_scalar(xa_scalar, mall0, 8);
        sk_mask_zero_f32(xa_simd, mall0, 8);
        check_bits_or_die("mask_zero_f32_all0", 8, 0, xa_simd, xa_scalar, 8);
    }
    printf("PASS mask_zero_f32\n");
}

static void bench_coherence_ema_gate(void) {
    Complex echo[BENCH_N], near_spec[BENCH_N];
    float abs_echo[BENCH_N], abs_near[BENCH_N];
    float sye_re[BENCH_N], sye_im[BENCH_N], syy[BENCH_N], see[BENCH_N];
    unsigned char mask[BENCH_N];
    fill_bench_complex(echo, BENCH_N);
    fill_bench_complex(near_spec, BENCH_N);
    fill_bench_floats(abs_echo, BENCH_N);
    fill_bench_floats(abs_near, BENCH_N);
    fill_bench_floats(sye_re, BENCH_N);
    fill_bench_floats(sye_im, BENCH_N);
    fill_bench_floats(syy, BENCH_N);
    fill_bench_floats(see, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) {
            sk_coherence_ema_gate_f32_scalar(sye_re, sye_im, syy, see, echo, near_spec,
                                              abs_echo, abs_near, 0.05f, 0.5f, mask, BENCH_N);
            g_bench_sink += sye_re[0];
        }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) {
                sk_coherence_ema_gate_f32(sye_re, sye_im, syy, see, echo, near_spec,
                                           abs_echo, abs_near, 0.05f, 0.5f, mask, BENCH_N);
                g_bench_sink += sye_re[0];
            }
            {
                double t3 = now_ns();
                report_bench("coherence_ema_gate_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

static void bench_ema_delta(void) {
    float state[BENCH_N], x[BENCH_N];
    fill_bench_floats(state, BENCH_N);
    fill_bench_floats(x, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_ema_delta_f32_scalar(state, x, 0.23f, BENCH_N); g_bench_sink += state[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_ema_delta_f32(state, x, 0.23f, BENCH_N); g_bench_sink += state[0]; }
            {
                double t3 = now_ns();
                report_bench("ema_delta_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

static void bench_n2_track(void) {
    float n2[BENCH_N], y2s[BENCH_N];
    fill_bench_floats(n2, BENCH_N);
    fill_bench_floats(y2s, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_n2_track_f32_scalar(n2, y2s, 0.99f, 0.003f, 1.0005f, BENCH_N); g_bench_sink += n2[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_n2_track_f32(n2, y2s, 0.99f, 0.003f, 1.0005f, BENCH_N); g_bench_sink += n2[0]; }
            {
                double t3 = now_ns();
                report_bench("n2_track_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

static void bench_n2_initial_track(void) {
    float n2i[BENCH_N], n2[BENCH_N];
    fill_bench_floats(n2i, BENCH_N);
    fill_bench_floats(n2, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_n2_initial_track_f32_scalar(n2i, n2, 0.0025f, BENCH_N); g_bench_sink += n2i[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_n2_initial_track_f32(n2i, n2, 0.0025f, BENCH_N); g_bench_sink += n2i[0]; }
            {
                double t3 = now_ns();
                report_bench("n2_initial_track_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

static void bench_mask_zero(void) {
    float x[BENCH_N];
    unsigned char mask[BENCH_N];
    int i;
    fill_bench_floats(x, BENCH_N);
    for (i = 0; i < BENCH_N; ++i) mask[i] = (unsigned char)(i & 1);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_mask_zero_f32_scalar(x, mask, BENCH_N); g_bench_sink += x[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_mask_zero_f32(x, mask, BENCH_N); g_bench_sink += x[0]; }
            {
                double t3 = now_ns();
                report_bench("mask_zero_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

/* ═══════════════════════════════════ main ══════════════════════════════════ */

int main(void) {
    init_special_pool();

    test_cabs_np();
    test_cmag2_np();
    test_cmag2_np_acc();
    test_ema();
    test_ema_cmag2();
    test_cmac_np();
    test_wupdate_nlms();
    test_wupdate_kf();
    test_capply_gain();
    test_cadd();
    test_sq_scale();
    test_min();
    test_clip();
    test_pairwise_sum();
    test_sum_sq_pairwise();
    test_pairwise_sum_tailfold();
    test_pairwise_sum_tailfold_b();
    test_fast_sqrt();
    test_coherence_ema_gate();
    test_ema_delta();
    test_n2_track();
    test_n2_initial_track();
    test_mask_zero();

    printf("\n--- microbenchmarks (n=%d, %d reps) ---\n", BENCH_N, BENCH_REPS);
    bench_cabs_np();
    bench_cmag2_np();
    bench_cmag2_np_acc();
    bench_ema();
    bench_ema_cmag2();
    bench_cmac_np();
    bench_wupdate_nlms();
    bench_wupdate_kf();
    bench_capply_gain();
    bench_cadd();
    bench_sq_scale();
    bench_min();
    bench_clip();
    bench_pairwise_sum();
    bench_sum_sq_pairwise();
    bench_pairwise_sum_tailfold();
    bench_pairwise_sum_tailfold_b();
    bench_fast_sqrt();
    bench_coherence_ema_gate();
    bench_ema_delta();
    bench_n2_track();
    bench_n2_initial_track();
    bench_mask_zero();

    printf("\nALL PASS (SK_HAVE_NEON=%d)\n", SK_HAVE_NEON);
    (void)g_bench_sink;
    return 0;
}
