/*
 * simd_nn_selftest.c - bitwise gate for include/simd_kernel_nn.h.
 *
 * Every skn_<name>() (NEON when available, else the scalar twin) runs
 * against skn_<name>_scalar() on identical inputs over n = 0..17 and the
 * DSP sizes the NN pre/post code uses, with canaries on both sides of every
 * output. Arithmetic kernels draw finite values (plus +-0, denormals and
 * +-Inf where the kernel has no Inf-Inf/0*Inf hazard); the predicate kernel
 * draws every class, NaN included. Any difference prints and exits 1.
 */
#include "simd_kernel_nn.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_N 1100
#define CANARY 16
#define TRIALS 12

static const int N_LIST[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    31, 32, 33, 63, 64, 65, 96, 127, 128, 129, 257, 513, 1024, 1025
};
#define N_COUNT ((int)(sizeof(N_LIST) / sizeof(N_LIST[0])))

static uint32_t g_lcg = 0x2545F491u;
static uint32_t next_u32(void) { g_lcg = g_lcg * 1664525u + 1013904223u; return g_lcg; }
static float from_bits(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

/* Finite values of every magnitude class; `with_inf` adds +-Inf. */
static float gen_finite(int with_inf) {
    uint32_t r = next_u32();
    switch (r % 8u) {
    case 0: return 0.0f;
    case 1: return -0.0f;
    case 2: return from_bits((next_u32() & 0x807fffffu) | 0x00000001u);   /* denormal */
    case 3: return with_inf ? ((r & 0x100u) ? INFINITY : -INFINITY) : 1.0f;
    case 4: return (float)((int32_t)next_u32()) * 1e-9f;
    default: {
        uint32_t u = next_u32();
        uint32_t e = (u >> 23) & 0xffu;
        if (e == 0xffu) u &= ~0x00800000u;       /* keep it finite */
        return from_bits(u);
    }
    }
}

static float gen_any(void) {
    uint32_t r = next_u32() % 16u;
    if (r == 0) return NAN;
    if (r == 1) return from_bits(0x7fc00001u | (next_u32() & 0x003fffffu));
    if (r == 2) return INFINITY;
    if (r == 3) return -INFINITY;
    return gen_finite(0);
}

static void fail(const char *k, int n, int t, int i, float a, float b) {
    printf("FAIL %s n=%d trial=%d index=%d: 0x%08x vs 0x%08x\n", k, n, t, i,
           (unsigned)bits(a), (unsigned)bits(b));
    exit(1);
}

static void canary_fill(float *arena, int n) {
    int i;
    for (i = 0; i < CANARY; ++i) { arena[i] = from_bits(0x7fa5a5a5u); arena[CANARY + n + i] = from_bits(0x7fa5a5a5u); }
}

static void canary_check(const char *k, const float *arena, int n) {
    int i;
    for (i = 0; i < CANARY; ++i) {
        if (bits(arena[i]) != 0x7fa5a5a5u || bits(arena[CANARY + n + i]) != 0x7fa5a5a5u) {
            printf("FAIL %s n=%d: canary overwritten\n", k, n);
            exit(1);
        }
    }
}

static void fail_int(const char *k, int n, int t, int got, int want) {
    printf("FAIL %s n=%d trial=%d: %d vs %d\n", k, n, t, got, want);
    exit(1);
}

static void cmp_arrays(const char *k, int n, int t, const float *a, const float *b) {
    int i;
    for (i = 0; i < n; ++i) if (bits(a[i]) != bits(b[i])) fail(k, n, t, i, a[i], b[i]);
}

/* A/B: the kernel's and the scalar twin's output arenas, canaried on both
 * sides; C2/C3: Complex outputs with CANARY/2 elements of slack each side. */
static float A[MAX_N + 2 * CANARY], B[MAX_N + 2 * CANARY];
static float X[MAX_N], Y[MAX_N];
static Complex C[MAX_N], C2[MAX_N + CANARY], C3[MAX_N + CANARY];

static void arm(int n) { canary_fill(A, n); canary_fill(B, n); }

static void check_out(const char *k, int n, int t) {
    canary_check(k, A, n);
    cmp_arrays(k, n, t, A + CANARY, B + CANARY);
}

static void test_all_finite(void) {
    int ni, t;
    for (ni = 0; ni < N_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS * 4; ++t) {
            int i, want, got;
            for (i = 0; i < n; ++i) X[i] = gen_finite(0);
            /* Most trials are all-finite (the production case); the rest
             * plant one or two non-finite values anywhere, including in the
             * scalar tail and at block edges. */
            if (t % 4 != 0 && n > 0) {
                X[next_u32() % (uint32_t)n] = gen_any();
                if (t % 3 == 0) X[n - 1] = (t & 1) ? INFINITY : NAN;
            }
            want = skn_all_finite_f32_scalar(X, (size_t)n);
            got = skn_all_finite_f32(X, (size_t)n);
            if (want != got) fail_int("all_finite_f32", n, t, got, want);
            for (i = 0; i < n; ++i) { C[i].r = X[i]; C[i].i = (i == n / 2 && t % 5 == 2) ? NAN : gen_finite(0); }
            want = skn_all_finite_cf32_scalar(C, (size_t)n);
            got = skn_all_finite_cf32(C, (size_t)n);
            if (want != got) fail_int("all_finite_cf32", n, t, got, want);
        }
    }
    /* every single position of a 1025 array, each non-finite class */
    {
        int n = 1025, p, cls;
        for (cls = 0; cls < 3; ++cls) {
            for (p = 0; p < n; ++p) {
                int i;
                for (i = 0; i < n; ++i) X[i] = 1.0f;
                X[p] = cls == 0 ? NAN : (cls == 1 ? INFINITY : -INFINITY);
                if (skn_all_finite_f32(X, (size_t)n) != 0) { printf("FAIL all_finite_f32 missed pos %d cls %d\n", p, cls); exit(1); }
            }
        }
        for (p = 0; p < n; ++p) X[p] = FLT_MAX;
        if (skn_all_finite_f32(X, (size_t)n) != 1) { printf("FAIL all_finite_f32 FLT_MAX\n"); exit(1); }
    }
    printf("PASS all_finite_f32 / all_finite_cf32\n");
}

static void test_fill_mul_power(void) {
    int ni, t, i;
    for (ni = 0; ni < N_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            float v = gen_any();
            arm(n);
            skn_fill_f32(A + CANARY, (size_t)n, v);
            skn_fill_f32_scalar(B + CANARY, (size_t)n, v);
            check_out("fill", n, t);

            for (i = 0; i < n; ++i) { X[i] = gen_finite(1); Y[i] = gen_finite(0); }
            arm(n);
            skn_mul_f32(A + CANARY, X, Y, (size_t)n);
            skn_mul_f32_scalar(B + CANARY, X, Y, (size_t)n);
            check_out("mul", n, t);
            /* in place, dst == a */
            memcpy(A + CANARY, X, (size_t)n * 4); memcpy(B + CANARY, X, (size_t)n * 4);
            skn_mul_f32(A + CANARY, A + CANARY, Y, (size_t)n);
            skn_mul_f32_scalar(B + CANARY, B + CANARY, Y, (size_t)n);
            cmp_arrays("mul_inplace", n, t, A + CANARY, B + CANARY);

            for (i = 0; i < n; ++i) { X[i] = gen_finite(0); Y[i] = gen_finite(0); }
            arm(n);
            /* a non-power-of-two scale too: 2^-k scaling is exact, so it
             * alone cannot see a reassociated (re^2*s + im^2*s) */
            skn_power_scale_f32(A + CANARY, X, Y, (size_t)n, (t & 1) ? 0x1p-10f : 0.3f);
            skn_power_scale_f32_scalar(B + CANARY, X, Y, (size_t)n, (t & 1) ? 0x1p-10f : 0.3f);
            check_out("power_scale", n, t);
        }
    }
    printf("PASS fill_f32 / mul_f32 / power_scale_f32\n");
}

static void test_interleave(void) {
    int ni, t, i;
    for (ni = 0; ni < N_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            float s = (t & 1) ? 0x1p-5f : 0x1p5f;
            for (i = 0; i < n; ++i) { C[i].r = gen_finite(1); C[i].i = gen_finite(1); }
            arm(n);
            memset(X, 0, sizeof X); memset(Y, 0, sizeof Y);
            skn_deinterleave_scale_cf32(C, A + CANARY, X, (size_t)n, s);
            skn_deinterleave_scale_cf32_scalar(C, B + CANARY, Y, (size_t)n, s);
            check_out("deinterleave_scale.re", n, t);
            cmp_arrays("deinterleave_scale.im", n, t, X, Y);

            /* exact moves: NaN payloads included */
            for (i = 0; i < n; ++i) { C[i].r = gen_any(); C[i].i = gen_any(); }
            skn_deinterleave_cf32(C, A + CANARY, X, (size_t)n);
            skn_deinterleave_cf32_scalar(C, B + CANARY, Y, (size_t)n);
            cmp_arrays("deinterleave.re", n, t, A + CANARY, B + CANARY);
            cmp_arrays("deinterleave.im", n, t, X, Y);
            for (i = 0; i < n; ++i) if (bits(X[i]) != bits(C[i].i)) fail("deinterleave.move", n, t, i, X[i], C[i].i);

            for (i = 0; i < n; ++i) { X[i] = gen_any(); Y[i] = gen_any(); }
            memset(C2, 0xA5, sizeof C2); memset(C3, 0xA5, sizeof C3);
            skn_interleave_cf32(X, Y, C2 + CANARY / 2, (size_t)n);
            skn_interleave_cf32_scalar(X, Y, C3 + CANARY / 2, (size_t)n);
            if (memcmp(C2, C3, sizeof C2) != 0) { printf("FAIL interleave n=%d\n", n); exit(1); }
            for (i = 0; i < n; ++i) { X[i] = gen_finite(1); Y[i] = gen_finite(1); }
            memset(C2, 0xA5, sizeof C2); memset(C3, 0xA5, sizeof C3);
            skn_interleave_scale_cf32(X, Y, C2 + CANARY / 2, (size_t)n, s);
            skn_interleave_scale_cf32_scalar(X, Y, C3 + CANARY / 2, (size_t)n, s);
            if (memcmp(C2, C3, sizeof C2) != 0) { printf("FAIL interleave_scale n=%d\n", n); exit(1); }
        }
    }
    printf("PASS (de)interleave[_scale]_cf32\n");
}

static void test_div_floor(void) {
    int ni, t, i;
    for (ni = 0; ni < N_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            for (i = 0; i < n; ++i) {
                X[i] = gen_finite(0);
                /* denominators around the floor, tiny, negative, NaN */
                switch (next_u32() % 6u) {
                case 0: Y[i] = 1e-11f; break;
                case 1: Y[i] = 0.0f; break;
                case 2: Y[i] = -gen_finite(0); break;
                case 3: Y[i] = (t % 3 == 0) ? NAN : 0.5f; break;
                default: Y[i] = fabsf(gen_finite(0)); break;
                }
            }
            arm(n);
            skn_div_floor_f32(A + CANARY, X, Y, (size_t)n, 1e-11f);
            skn_div_floor_f32_scalar(B + CANARY, X, Y, (size_t)n, 1e-11f);
            check_out("div_floor", n, t);
        }
    }
    printf("PASS div_floor_f32\n");
}

static float H[256], R0[4][256];

static void test_ring_dot(void) {
    static const int TAPS[] = {1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 16, 17, 33, 49, 97, 193};
    int ti, rows, head, t;
    for (ti = 0; ti < (int)(sizeof TAPS / sizeof TAPS[0]); ++ti) {
        int taps = TAPS[ti];
        for (rows = 1; rows <= SKN_RING_DOT_MAX_ROWS; ++rows) {
            for (t = 0; t < 3; ++t) {
                int i, r;
                const float *rp[SKN_RING_DOT_MAX_ROWS];
                for (i = 0; i < taps; ++i) H[i] = gen_finite(0) * 1e-20f + ((next_u32() & 1) ? 0.5f : -0.25f) * (float)(i % 7);
                for (r = 0; r < rows; ++r) {
                    for (i = 0; i < taps; ++i) R0[r][i] = (float)((int32_t)next_u32()) * 1e-10f;
                    rp[r] = R0[r];
                }
                for (head = 0; head < taps; ++head) {
                    float o1[SKN_RING_DOT_MAX_ROWS], o2[SKN_RING_DOT_MAX_ROWS];
                    skn_ring_dot_rows_f32(rp, rows, H, head, taps, o1);
                    skn_ring_dot_rows_f32_scalar(rp, rows, H, head, taps, o2);
                    for (r = 0; r < rows; ++r)
                        if (bits(o1[r]) != bits(o2[r])) {
                            printf("FAIL ring_dot taps=%d rows=%d head=%d row=%d: 0x%08x vs 0x%08x\n",
                                   taps, rows, head, r, (unsigned)bits(o1[r]), (unsigned)bits(o2[r]));
                            exit(1);
                        }
                }
            }
        }
    }
    printf("PASS ring_dot_rows_f32\n");
}

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static volatile int g_sink;

static void bench(void) {
    enum { N = 24576, REPS = 4000 };
    static float big[N];
    int i, r, acc = 0;
    double t0, t1, t2;
    for (i = 0; i < N; ++i) big[i] = (float)i * 1e-3f;
    t0 = now_ns();
    for (r = 0; r < REPS; ++r) { big[r % N] += 0.0f; acc += skn_all_finite_f32_scalar(big, N); }
    t1 = now_ns();
    for (r = 0; r < REPS; ++r) { big[r % N] += 0.0f; acc += skn_all_finite_f32(big, N); }
    t2 = now_ns();
    g_sink = acc;
    printf("BENCH all_finite n=%d: scalar %.2f us, kernel %.2f us\n", N,
           (t1 - t0) / REPS / 1e3, (t2 - t1) / REPS / 1e3);
}

int main(void) {
    test_all_finite();
    test_fill_mul_power();
    test_interleave();
    test_div_floor();
    test_ring_dot();
    bench();
    printf("ALL PASS simd_kernel_nn (SK_HAVE_NEON=%d)\n", SK_HAVE_NEON);
    return 0;
}
