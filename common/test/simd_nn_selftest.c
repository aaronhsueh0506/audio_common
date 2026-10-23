/*
 * simd_nn_selftest.c - bitwise gate for utility/simd_kernel_nn.h.
 *
 * Every skn_<name>() (NEON when available, else the scalar twin) runs against skn_<name>_scalar() on
 * identical inputs over n = 0..17 and the DSP sizes the NN pre/post code uses, with canaries on both
 * sides of every output. Arithmetic kernels draw finite values (plus +-0, denormals and, where safe,
 * +-Inf); the predicate kernel draws every class. ERB arithmetic counts both-NaN payload differences
 * instead of failing on them; every other difference prints and exits 1.
 */
#include "utility/simd_kernel_nn.h"

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

/* ERB sums may overflow finite inputs into NaNs. Arithmetic promises NaN
 * classification, while selection/copy tests above retain raw-bit checks. */
static unsigned long long g_erb_nan_payload_differences;
static void cmp_arithmetic_arrays(const char *k, int n, int t,
                                  const float *a, const float *b) {
    for (int i = 0; i < n; ++i) {
        if (bits(a[i]) == bits(b[i])) continue;
        if (isnan(a[i]) && isnan(b[i])) ++g_erb_nan_payload_differences;
        else fail(k, n, t, i, a[i], b[i]);
    }
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

/* Finite predicate over strided rows (rows of `width` floats, `stride`
 * floats apart; the gap between width and stride holds planted non-finite
 * values that must not matter), against a plain double loop. */
static void test_rows_all_finite(void) {
    static float buf[8 * 64];
    int round, want;
    for (round = 0; round < 4000; ++round) {
        size_t width = 1 + next_u32() % 40;
        size_t stride = width + next_u32() % 24;
        size_t rows = 1 + next_u32() % 8;
        size_t r, i;
        want = 1;
        for (i = 0; i < sizeof(buf) / sizeof(float); ++i) buf[i] = gen_finite(0);
        for (r = 0; r < rows; ++r)
            for (i = width; i < stride; ++i) buf[r * stride + i] = gen_any();
        if (round % 3 != 0) {
            r = next_u32() % rows; i = next_u32() % width;
            buf[r * stride + i] = gen_any();
        }
        for (r = 0; r < rows; ++r)
            for (i = 0; i < width; ++i)
                if (!isfinite(buf[r * stride + i])) want = 0;
        if (skn_all_finite_rows_f32(buf, rows, stride, width) != want)
            fail_int("all_finite_rows_f32", (int)(rows * stride), round, skn_all_finite_rows_f32(buf, rows, stride, width), want);
        if (skn_all_finite_rows_f32_scalar(buf, rows, stride, width) != want)
            fail_int("all_finite_rows_f32_scalar", (int)(rows * stride), round, skn_all_finite_rows_f32_scalar(buf, rows, stride, width), want);
    }
    printf("PASS all_finite_rows_f32\n");
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

static void test_mag_eps(void) {
    static const float EPS[] = { 0.0f, 1e-12f, 1e-20f, 0x1p-149f, 1.0f, -1.0f, INFINITY };
    int ni, t, i, n, off;
    for (ni = 0; ni < N_COUNT; ++ni) {
        n = N_LIST[ni];
        for (t = 0; t < TRIALS * 2; ++t) {
            float eps = EPS[t % (int)(sizeof(EPS) / sizeof(EPS[0]))];
            for (i = 0; i < n; ++i) {
                X[i] = (t % 3 == 2) ? gen_any() : gen_finite(1);
                Y[i] = (t % 3 == 2) ? gen_any() : gen_finite(1);
            }
            arm(n);
            skn_mag_eps_f32(A + CANARY, X, Y, (size_t)n, eps);
            skn_mag_eps_f32_scalar(B + CANARY, X, Y, (size_t)n, eps);
            check_out("mag_eps", n, t);
            /* in place on re and on im */
            memcpy(A + CANARY, X, (size_t)n * 4); memcpy(B + CANARY, X, (size_t)n * 4);
            skn_mag_eps_f32(A + CANARY, A + CANARY, Y, (size_t)n, eps);
            skn_mag_eps_f32_scalar(B + CANARY, B + CANARY, Y, (size_t)n, eps);
            cmp_arrays("mag_eps_inplace_re", n, t, A + CANARY, B + CANARY);
            memcpy(A + CANARY, Y, (size_t)n * 4); memcpy(B + CANARY, Y, (size_t)n * 4);
            skn_mag_eps_f32(A + CANARY, X, A + CANARY, (size_t)n, eps);
            skn_mag_eps_f32_scalar(B + CANARY, X, B + CANARY, (size_t)n, eps);
            cmp_arrays("mag_eps_inplace_im", n, t, A + CANARY, B + CANARY);
        }
    }
    /* every length 0..70 at every 4-byte misalignment of inputs and output */
    for (n = 0; n <= 70; ++n) {
        for (off = 0; off < 4; ++off) {
            for (t = 0; t < 4; ++t) {
                float eps = EPS[(n + t) % (int)(sizeof(EPS) / sizeof(EPS[0]))];
                for (i = 0; i < n + 4; ++i) { X[i] = gen_finite(1); Y[i] = gen_finite(1); }
                arm(n + 4);
                skn_mag_eps_f32(A + CANARY + off, X + off, Y + (3 - off), (size_t)n, eps);
                skn_mag_eps_f32_scalar(B + CANARY + off, X + off, Y + (3 - off), (size_t)n, eps);
                canary_check("mag_eps_align", A, n + 4);
                cmp_arrays("mag_eps_align", n, t, A + CANARY + off, B + CANARY + off);
            }
        }
    }
    printf("PASS mag_eps_f32\n");
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


/* Up to 18 taps over up to 260 bins (the shipped CCM runs 255 bins x 9 taps,
 * the deep filter 96 x 5). Coefficients are finite; the sources may carry
 * +-Inf, which both twins turn into the same NaN/Inf. */
static void test_ctaps_mac(void) {
    enum { MAX_TAPS = 18, MAX_BINS = 260 };
    static float rows_re[MAX_TAPS][MAX_BINS], rows_im[MAX_TAPS][MAX_BINS];
    static float coef[MAX_BINS * MAX_TAPS * 2];
    static float OA[2][MAX_BINS + 2 * CANARY], OB[2][MAX_BINS + 2 * CANARY];
    const int tap_list[] = {1, 2, 3, 5, 9, 18};
    const float *sr[MAX_TAPS], *si[MAX_TAPS];
    int ni, ti, t, k, i, c;
    for (ti = 0; ti < (int)(sizeof tap_list / sizeof tap_list[0]); ++ti) {
        int taps = tap_list[ti];
        for (ni = 0; ni < N_COUNT && N_LIST[ni] <= MAX_BINS; ++ni) {
            int n = N_LIST[ni];
            for (t = 0; t < TRIALS; ++t) {
                for (k = 0; k < taps; ++k) {
                    for (i = 0; i < MAX_BINS; ++i) {
                        rows_re[k][i] = gen_finite(t % 4 == 0);
                        rows_im[k][i] = gen_finite(t % 4 == 0);
                    }
                    sr[k] = rows_re[k]; si[k] = rows_im[k];
                }
                for (i = 0; i < n * taps * 2; ++i) coef[i] = gen_finite(0);
                for (c = 0; c < 2; ++c) {
                    canary_fill(OA[c], n); canary_fill(OB[c], n);
                }
                skn_ctaps_mac_f32(OA[0] + CANARY, OA[1] + CANARY, (size_t)n,
                                  (size_t)taps, sr, si, coef);
                skn_ctaps_mac_f32_scalar(OB[0] + CANARY, OB[1] + CANARY,
                                         (size_t)n, (size_t)taps, sr, si, coef);
                for (c = 0; c < 2; ++c) {
                    canary_check("ctaps_mac", OA[c], n);
                    cmp_arrays("ctaps_mac", n, t, OA[c] + CANARY, OB[c] + CANARY);
                }
            }
        }
    }
    printf("PASS ctaps_mac_f32\n");
}

/* skn_erb_forward_banded_f32 has no _scalar twin, so this checks its documented contract against a
 * dense double loop (no four-band batching, no shared prefix) that accumulates the same ascending-k
 * per-band fold into a pre-seeded erb_work, over random sparse [k0,k1) ranges including empty ones
 * and ranges touching 0 and n_bins. */
static void erb_forward_banded_ref(const float *power, const float *erb_fwd,
                                    const uint16_t *k0, const uint16_t *k1,
                                    int n_bands, float *erb_work) {
    int b, k;
    for (b = 0; b < n_bands; ++b) {
        float s = erb_work[b];
        for (k = k0[b]; k < k1[b]; ++k) s += power[k] * erb_fwd[(size_t)k * n_bands + b];
        erb_work[b] = s;
    }
}

static void test_erb_forward_banded(void) {
    static const int BANDS_LIST[] = { 4, 8, 32, 64 };
    static const int BINS_LIST[] = { 4, 16, 64, 257, 513 };
    static float erb_fwd[513 * 64];
    static uint16_t k0[64], k1[64];
    static float power[513];
    static float work_a[64], work_b[64];
    int bi, ni, t;
    for (bi = 0; bi < (int)(sizeof BANDS_LIST / sizeof BANDS_LIST[0]); ++bi) {
        int n_bands = BANDS_LIST[bi];
        for (ni = 0; ni < (int)(sizeof BINS_LIST / sizeof BINS_LIST[0]); ++ni) {
            int n_bins = BINS_LIST[ni];
            for (t = 0; t < TRIALS * 2; ++t) {
                int b, k;
                for (k = 0; k < n_bins; ++k) power[k] = gen_finite(0);
                for (k = 0; k < n_bins * n_bands; ++k) erb_fwd[k] = gen_finite(0);
                for (b = 0; b < n_bands; ++b) {
                    int lo = (int)(next_u32() % (uint32_t)(n_bins + 1));
                    int width = (int)(next_u32() % (uint32_t)(n_bins - lo + 1));
                    /* every few trials, force an edge case: empty, or
                     * touching 0 / n_bins exactly. */
                    if (t % 5 == 0) { lo = 0; width = 0; }
                    else if (t % 5 == 1) { width = n_bins - lo; }
                    k0[b] = (uint16_t)lo;
                    k1[b] = (uint16_t)(lo + width);
                    work_a[b] = work_b[b] = (t & 1) ? gen_finite(0) : 0.0f;
                }
                skn_erb_forward_banded_f32(power, erb_fwd, k0, k1, n_bands, work_a);
                erb_forward_banded_ref(power, erb_fwd, k0, k1, n_bands, work_b);
                cmp_arrays("erb_forward_banded", n_bands, t, work_a, work_b);
            }
        }
    }
    printf("PASS erb_forward_banded_f32\n");
}

/* skn_erb_expand_mask_f32{,_scalar,_neon}: the dense reference sums over EVERY band (never skipping by
 * inv_lo/inv_hi), which is what a finite gain's skip-the-zeros path must equal and what a
 * non-finite gain's whole-row fallback is by definition. */
static void erb_expand_mask_ref(const float *band_gain, const float *erb_inv,
                                 int n_bins, int n_bands, float *bin_gain) {
    int k, b;
    for (k = 0; k < n_bins; ++k) {
        float s = 0.0f;
        for (b = 0; b < n_bands; ++b) s += erb_inv[(size_t)b * n_bins + k] * band_gain[b];
        bin_gain[k] = s;
    }
}

/* First/last nonzero column of band b's row -- the same span DFN2/3's own
 * erb_spans() computes from the shipped matrix, so skipping outside it is
 * exact for a finite gain. */
static void erb_inv_spans(const float *erb_inv, int n_bins, int n_bands,
                           uint16_t *lo, uint16_t *hi) {
    int b;
    for (b = 0; b < n_bands; ++b) {
        const float *row = erb_inv + (size_t)b * n_bins;
        int first = 0, last = n_bins;
        while (first < last && row[first] == 0.0f) ++first;
        while (last > first && row[last - 1] == 0.0f) --last;
        if (first == last) first = last = 0;
        lo[b] = (uint16_t)first;
        hi[b] = (uint16_t)last;
    }
}

static void test_erb_expand_mask(void) {
    static const int BANDS_LIST[] = { 4, 22, 32, 64 };
    static const int BINS_LIST[] = { 4, 64, 192, 257, 513 };
    static float erb_inv[64 * 513];
    static float band_gain[64];
    static uint16_t lo[64], hi[64];
    static float out_a[513 + 2 * CANARY];
#if SK_HAVE_NEON
    static float out_c[513 + 2 * CANARY];
#endif
    int bi, ni, t;
    for (bi = 0; bi < (int)(sizeof BANDS_LIST / sizeof BANDS_LIST[0]); ++bi) {
        int n_bands = BANDS_LIST[bi];
        for (ni = 0; ni < (int)(sizeof BINS_LIST / sizeof BINS_LIST[0]); ++ni) {
            int n_bins = BINS_LIST[ni];
            for (t = 0; t < TRIALS * 2; ++t) {
                int i, b;
                static float ref[513];
                for (i = 0; i < n_bins * n_bands; ++i)
                    erb_inv[i] = (next_u32() % 4 == 0) ? 0.0f : gen_finite(0);
                for (b = 0; b < n_bands; ++b)
                    band_gain[b] = (t % 7 == 0) ? gen_any() : gen_finite(0);
                erb_inv_spans(erb_inv, n_bins, n_bands, lo, hi);
                erb_expand_mask_ref(band_gain, erb_inv, n_bins, n_bands, ref);

                canary_fill(out_a, n_bins);
                skn_erb_expand_mask_f32_scalar(band_gain, erb_inv, lo, hi,
                                               n_bins, n_bands, out_a + CANARY);
                canary_check("erb_expand_mask_scalar", out_a, n_bins);
                cmp_arithmetic_arrays("erb_expand_mask_scalar_vs_ref", n_bins, t,
                           out_a + CANARY, ref);

                canary_fill(out_a, n_bins);
                skn_erb_expand_mask_f32(band_gain, erb_inv, lo, hi, n_bins,
                                        n_bands, out_a + CANARY);
                canary_check("erb_expand_mask", out_a, n_bins);
                cmp_arithmetic_arrays("erb_expand_mask_vs_scalar", n_bins, t,
                           out_a + CANARY, ref);
#if SK_HAVE_NEON
                canary_fill(out_c, n_bins);
                skn_erb_expand_mask_f32_neon(band_gain, erb_inv, lo, hi,
                                            n_bins, n_bands, out_c + CANARY);
                canary_check("erb_expand_mask_neon", out_c, n_bins);
                cmp_arithmetic_arrays("erb_expand_mask_neon_vs_scalar", n_bins, t,
                           out_c + CANARY, ref);
#endif
            }
        }
    }
    printf("PASS erb_expand_mask_f32\n");
}

/* skn_erb_band_accumulate_f32_scalar: the reference uses the opposite loop nest (outer bin, inner band)
 * from the kernel's (outer band, inner bin), so it cross-checks the per-band ascending-k fold
 * independently of the kernel body. */
static void erb_band_accumulate_ref(const float *x, const float *table,
                                     int n_bins, int n_bands, float *banded) {
    int b, k;
    for (b = 0; b < n_bands; ++b) banded[b] = 0.0f;
    for (k = 0; k < n_bins; ++k) {
        float value = x[k];
        const float *row = table + (size_t)k * n_bands;
        for (b = 0; b < n_bands; ++b) banded[b] += row[b] * value;
    }
}

static void test_erb_band_accumulate(void) {
    /* 22 is RNNoise-ERB's band count and 64 is GTCRN's high-band count. */
    static const int BANDS_LIST[] = { 1, 2, 3, 4, 7, 22, 32, 64 };
    static const int BINS_LIST[] = { 1, 4, 64, 192, 257, 513 };
    static float table[513 * 64];
    static float x[513];
    static float banded_a[64 + 2 * CANARY];
    int bi, ni, t;
    for (bi = 0; bi < (int)(sizeof BANDS_LIST / sizeof BANDS_LIST[0]); ++bi) {
        int n_bands = BANDS_LIST[bi];
        for (ni = 0; ni < (int)(sizeof BINS_LIST / sizeof BINS_LIST[0]); ++ni) {
            int n_bins = BINS_LIST[ni];
            for (t = 0; t < TRIALS * 2; ++t) {
                int i;
                static float ref[64];
                for (i = 0; i < n_bins; ++i) x[i] = gen_finite(0);
                for (i = 0; i < n_bins * n_bands; ++i) table[i] = gen_finite(0);
                erb_band_accumulate_ref(x, table, n_bins, n_bands, ref);

                canary_fill(banded_a, n_bands);
                skn_erb_band_accumulate_f32_scalar(x, table, n_bins, n_bands,
                                                   banded_a + CANARY);
                canary_check("erb_band_accumulate_scalar", banded_a, n_bands);
                cmp_arithmetic_arrays("erb_band_accumulate_scalar_vs_ref", n_bands, t,
                           banded_a + CANARY, ref);
            }
        }
    }
    printf("PASS erb_band_accumulate_f32\n");
}

/* skn_erb_band_accumulate_nch_f32{,_scalar}: each channel checked against
 * the same independent-structure reference as the single-channel kernel
 * above, and n_bands chosen to exercise every tier of the NEON path's
 * block structure: a whole number of SKN_ERB_BAND_QUADS*4=16-band blocks
 * (16, 64), a block plus a partial remainder in whole quads (20), a block
 * plus a remainder needing its own scalar tail (23), quads only (4, 12),
 * quads plus a scalar tail (7, 22 -- RNNoise-ERB's band count), and
 * scalar-tail-only (1, 2, 3). */
static void test_erb_band_accumulate_nch(void) {
    static const int BANDS_LIST[] = { 1, 2, 3, 4, 7, 12, 15, 16, 20, 22, 23, 32, 64 };
    static const int BINS_LIST[] = { 1, 4, 64, 192, 257 };
    static const int NCH_LIST[] = { 1, 2, 3, 4 };
    static float table[257 * 64];
    static float x[SKN_ERB_NCH_MAX][257];
    static float ref[SKN_ERB_NCH_MAX][64];
    static float banded_a[SKN_ERB_NCH_MAX][64 + 2 * CANARY];
    static float banded_b[SKN_ERB_NCH_MAX][64 + 2 * CANARY];
    int bi, ni, ci, t, c;
    for (bi = 0; bi < (int)(sizeof BANDS_LIST / sizeof BANDS_LIST[0]); ++bi) {
        int n_bands = BANDS_LIST[bi];
        for (ni = 0; ni < (int)(sizeof BINS_LIST / sizeof BINS_LIST[0]); ++ni) {
            int n_bins = BINS_LIST[ni];
            for (ci = 0; ci < (int)(sizeof NCH_LIST / sizeof NCH_LIST[0]); ++ci) {
                int n_ch = NCH_LIST[ci];
                for (t = 0; t < TRIALS; ++t) {
                    int i;
                    const float *xp[SKN_ERB_NCH_MAX];
                    float *ap[SKN_ERB_NCH_MAX], *bp[SKN_ERB_NCH_MAX];
                    for (c = 0; c < n_ch; ++c)
                        for (i = 0; i < n_bins; ++i) x[c][i] = gen_finite(0);
                    for (i = 0; i < n_bins * n_bands; ++i) table[i] = gen_finite(0);
                    for (c = 0; c < n_ch; ++c) {
                        erb_band_accumulate_ref(x[c], table, n_bins, n_bands,
                                                ref[c]);
                        xp[c] = x[c];
                        canary_fill(banded_a[c], n_bands);
                        canary_fill(banded_b[c], n_bands);
                        ap[c] = banded_a[c] + CANARY;
                        bp[c] = banded_b[c] + CANARY;
                    }

                    skn_erb_band_accumulate_nch_f32_scalar(xp, n_ch, table,
                                                           n_bins, n_bands, ap);
                    for (c = 0; c < n_ch; ++c) {
                        canary_check("erb_band_accumulate_nch_scalar",
                                    banded_a[c], n_bands);
                        cmp_arithmetic_arrays("erb_band_accumulate_nch_scalar_vs_ref",
                                  n_bands, t, ap[c], ref[c]);
                    }

                    skn_erb_band_accumulate_nch_f32(xp, n_ch, table, n_bins,
                                                    n_bands, bp);
                    for (c = 0; c < n_ch; ++c) {
                        canary_check("erb_band_accumulate_nch", banded_b[c],
                                    n_bands);
                        cmp_arithmetic_arrays("erb_band_accumulate_nch_vs_ref", n_bands,
                                  t, bp[c], ref[c]);
                        /* n_ch == 1 must match the single-channel scalar
                         * kernel byte-for-byte (same fold). */
                        if (n_ch == 1) {
                            float single[64 + 2 * CANARY];
                            canary_fill(single, n_bands);
                            skn_erb_band_accumulate_f32_scalar(x[0], table,
                                                               n_bins, n_bands,
                                                               single + CANARY);
                            canary_check("erb_band_accumulate_nch_vs_1ch",
                                        single, n_bands);
                            cmp_arithmetic_arrays("erb_band_accumulate_nch_vs_1ch",
                                      n_bands, t, bp[0], single + CANARY);
                        }
                    }
                }
            }
        }
    }
    printf("PASS erb_band_accumulate_nch_f32\n");
}

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static volatile int g_sink;


static void bench_ctaps(int n, int taps) {
    enum { REPS = 20000, MAXB = 256, MAXT = 9 };
    static float rows[2 * MAXT][MAXB + 4], coef[MAXB * MAXT * 2];
    static float out_re[MAXB], out_im[MAXB];
    const float *sr[MAXT], *si[MAXT];
    int i, k, r;
    double t0, t1, t2;
    for (k = 0; k < taps; ++k) { sr[k] = rows[2 * k]; si[k] = rows[2 * k + 1]; }
    for (i = 0; i < 2 * MAXT * (MAXB + 4); ++i) ((float *)rows)[i] = (float)(i % 17) * 0.01f;
    for (i = 0; i < MAXB * MAXT * 2; ++i) coef[i] = (float)(i % 13) * 0.02f;
    t0 = now_ns();
    for (r = 0; r < REPS; ++r) {
        skn_ctaps_mac_f32_scalar(out_re, out_im, (size_t)n, (size_t)taps, sr, si, coef);
        g_sink += (int)out_re[r % n];
    }
    t1 = now_ns();
    for (r = 0; r < REPS; ++r) {
        skn_ctaps_mac_f32(out_re, out_im, (size_t)n, (size_t)taps, sr, si, coef);
        g_sink += (int)out_re[r % n];
    }
    t2 = now_ns();
    printf("BENCH ctaps_mac n=%d taps=%d: scalar %.3f us, kernel %.3f us\n", n, taps,
           (t1 - t0) / REPS / 1e3, (t2 - t1) / REPS / 1e3);
}

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
    bench_ctaps(255, 9);   /* DeepVQE CCM interior */
    bench_ctaps(96, 5);    /* DeepFilterNet2 deep filter */
}

int main(void) {
    test_all_finite();
    test_rows_all_finite();
    test_fill_mul_power();
    test_mag_eps();
    test_interleave();
    test_div_floor();
    test_ctaps_mac();
    test_erb_forward_banded();
    test_erb_expand_mask();
    test_erb_band_accumulate();
    test_erb_band_accumulate_nch();
    printf("ERB both-NaN payload differences: %llu (classification matched)\n",
           g_erb_nan_payload_differences);
    if (!getenv("SKN_SELFTEST_SKIP_BENCH")) bench();
    printf("ALL PASS simd_kernel_nn (SK_HAVE_NEON=%d)\n", SK_HAVE_NEON);
    return 0;
}
