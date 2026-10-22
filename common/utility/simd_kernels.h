/**
 * Shared scalar/NEON float32 DSP micro-kernels.
 *
 * For finite inputs every NEON entry point must be byte-identical to its
 * scalar reference. Ordered compare/select kernels also match scalar NaN
 * branching; pure arithmetic guarantees NaN classification but not payload
 * identity. Tests classify payload-only differences separately.
 *
 * Preserve scalar operation order and compile including translation units
 * with -ffp-contract=off. Explicit fmaf calls stay fused; ordinary mul/add
 * stays separate. Reciprocal/sqrt ESTIMATE instructions (frecpe/frsqrte)
 * and reassociation are forbidden; the correctly-rounded AArch64 FSQRT
 * (vsqrtq_f32) is not an estimate and is used directly.
 * SIMD_KERNELS_FORCE_SCALAR selects the fallback implementation.
 *
 * Exact in-place aliasing is supported only where a kernel documents it;
 * partial overlap is unsupported.
 */

#ifndef SIMD_KERNELS_H
#define SIMD_KERNELS_H

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>        /* memcpy: Complex <-> float[8] reinterpretation in
                             * the sk__cquad_* fallback */

#include "ac_complex.h"    /* Complex { float r; float i; } (interleaved AoS) */

#if defined(__ARM_NEON) && defined(__aarch64__) && !defined(SIMD_KERNELS_FORCE_SCALAR)
#include <arm_neon.h>
#define SK_HAVE_NEON 1
#else
#define SK_HAVE_NEON 0
#endif

/* Mirror of fast_math.h's FM_HAVE_HW_SQRT (this header replicates rather
 * than includes fast_math.h). Keyed on the architecture, not on SK_HAVE_NEON:
 * under SIMD_KERNELS_FORCE_SCALAR on AArch64 the scalar helper must still be
 * sqrtf so it stays bit-exact with sk_fast_sqrt_f32's vsqrtq_f32. */
#if defined(__aarch64__) || (defined(__arm__) && defined(__ARM_FP) && (__ARM_FP & 4))
#define SK_HAVE_HW_SQRT 1
#else
#define SK_HAVE_HW_SQRT 0
#endif
#if SK_HAVE_HW_SQRT || defined(__arm__)
#define SK_USE_CORRECT_SQRT 1
#else
#define SK_USE_CORRECT_SQRT 0
#endif

/* SK_STATIC_ASSERT: compile-time assert for C99, C11 and C++11. C++ spells it
 * static_assert, C11 _Static_assert; strict C99 gets a typedef of a char array
 * whose size is -1 when the condition is false. The typedef name must be
 * unique per use, hence the __LINE__ token paste. */
#if defined(__cplusplus)
#  define SK_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#  define SK_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#else
#  define SK__STATIC_ASSERT_CONCAT_(a, b) a##b
#  define SK__STATIC_ASSERT_CONCAT(a, b) SK__STATIC_ASSERT_CONCAT_(a, b)
#  define SK_STATIC_ASSERT(cond, msg) \
       typedef char SK__STATIC_ASSERT_CONCAT(sk_static_assert_line_, \
                                              __LINE__)[(cond) ? 1 : -1]
#endif

/* The sk_c*_f32 NEON kernels and their sk__cquad_load/sk__cquad_store
 * quad moves assume Complex is exactly two contiguous, unpadded floats
 * {r, i}; a layout change must fail to compile. */
SK_STATIC_ASSERT(sizeof(Complex) == 2 * sizeof(float),
               "Complex must be exactly two floats {r,i} (8 bytes, no "
               "padding) -- the sk_c*_f32 NEON kernels' "
               "quad-load/store layout depends on this");
SK_STATIC_ASSERT(offsetof(Complex, r) == 0 &&
               offsetof(Complex, i) == sizeof(float),
               "Complex.r/.i must sit at byte offsets 0/4 in that order "
               "(interleaved AoS) -- the sk_c*_f32 NEON kernels' "
               "quad-load/store assumes this exact layout");

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════ shared per-element helpers ═══════════════════
 * Used by every scalar kernel AND by every NEON kernel's scalar tail, so
 * the tail matches the fully-scalar path bit-for-bit by construction. */

/* fast_math.h fast_sqrt(), replicated rather than included. ARM targets use
 * sqrtf (hardware or libm); other targets use the bit-trick seed plus two Newton
 * iterations. Keep in sync with fast_math.h.
 *
 * USE_STANDARD_MATH makes fast_math.h use plain sqrtf; this kernel follows
 * the same flag so both behave alike in both modes (sqrtf per lane equals
 * vsqrtq_f32, correctly rounded, including sqrtf(-0) = -0 and NaN for
 * negatives). */
#ifdef USE_STANDARD_MATH
static inline float sk__fast_sqrt_elem(float v) {
    return sqrtf(v);
}
#else
/* NaN fails `v > 0.0f`, so the guard returns 0.0f for NaN too, as in
 * fast_math.h and in the NEON twin sk_fast_sqrt_f32. */
static inline float sk__fast_sqrt_elem(float v) {
    if (!(v > 0.0f)) return 0.0f;
#if SK_USE_CORRECT_SQRT
    return sqrtf(v);
#else
    {
        union { float f; uint32_t u; } fb;
        fb.f = v;
        fb.u = (fb.u >> 1) + 0x1FC00000u;
        {
            float x = fb.f;
            x = 0.5f * (x + v / x);  /* iteration 1 */
            x = 0.5f * (x + v / x);  /* iteration 2 */
            return x;
        }
    }
#endif
}
#endif /* USE_STANDARD_MATH */

/* ═══════════════════════════════ sk_ema_f32 ═══════════════════════════════
 * state[i] = alpha*state[i] + beta*x[i]: two separately rounded multiplies and
 * an add, never fused (see the FMA discipline in the header comment). Needs
 * -ffp-contract=off in the including TU; clang otherwise fuses this shape. */

static inline void sk_ema_f32_scalar(float *state, const float *x,
                                      float alpha, float beta, int n) {
    int i;
    for (i = 0; i < n; ++i) state[i] = alpha * state[i] + beta * x[i];
}

#if SK_HAVE_NEON
static inline void sk_ema_f32(float *state, const float *x,
                               float alpha, float beta, int n) {
    int i = 0;
    float32x4_t va = vdupq_n_f32(alpha), vb = vdupq_n_f32(beta);
    for (; i + 4 <= n; i += 4) {
        float32x4_t s = vld1q_f32(state + i);
        float32x4_t xv = vld1q_f32(x + i);
        float32x4_t r = vaddq_f32(vmulq_f32(va, s), vmulq_f32(vb, xv));
        vst1q_f32(state + i, r);
    }
    for (; i < n; ++i) state[i] = alpha * state[i] + beta * x[i];
}
#else
static inline void sk_ema_f32(float *state, const float *x,
                               float alpha, float beta, int n) {
    sk_ema_f32_scalar(state, x, alpha, beta, n);
}
#endif

/* ═══════════════════ Complex-quad NEON load/store ════════════════════════
 * These helpers are the only supported way to move Complex arrays through
 * NEON. GCC/Clang use a may_alias float view so strict-aliasing remains valid;
 * other compilers use a portable memcpy plus register unzip/zip fallback. */
#if SK_HAVE_NEON
static inline float32x4x2_t sk__cquad_load(const Complex *p) {
#if defined(__GNUC__) || defined(__clang__)
    typedef float sk__alias_float __attribute__((__may_alias__));
    return vld2q_f32((const sk__alias_float *)(const void *)p);
#else
    float scratch[8];
    memcpy(scratch, p, sizeof(scratch));
    {
        float32x4_t lo = vld1q_f32(scratch);
        float32x4_t hi = vld1q_f32(scratch + 4);
        return vuzpq_f32(lo, hi); /* val[0]=r's, val[1]=i's */
    }
#endif
}

static inline void sk__cquad_store(Complex *p, float32x4x2_t v) {
#if defined(__GNUC__) || defined(__clang__)
    typedef float sk__alias_float __attribute__((__may_alias__));
    vst2q_f32((sk__alias_float *)(void *)p, v);
#else
    float32x4x2_t z = vzipq_f32(v.val[0], v.val[1]); /* re-interleave r/i */
    float scratch[8];
    vst1q_f32(scratch, z.val[0]);
    vst1q_f32(scratch + 4, z.val[1]);
    memcpy(p, scratch, sizeof(scratch));
#endif
}
#endif /* SK_HAVE_NEON */

/* ═══════════════════════════════ sk_capply_gain_f32 ═══════════════════════
 * out[i] = z[i] * g[i] (real gain applied to both components). out == z is
 * supported: each iteration loads before it stores and never revisits an
 * earlier index. Partial overlap is not. */

static inline void sk_capply_gain_f32_scalar(Complex *out, const Complex *z,
                                              const float *g, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        out[i].r = z[i].r * g[i];
        out[i].i = z[i].i * g[i];
    }
}

#if SK_HAVE_NEON
static inline void sk_capply_gain_f32(Complex *out, const Complex *z,
                                       const float *g, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t zv = sk__cquad_load(z + i);
        float32x4_t gv = vld1q_f32(g + i);
        float32x4x2_t rv;
        rv.val[0] = vmulq_f32(zv.val[0], gv);
        rv.val[1] = vmulq_f32(zv.val[1], gv);
        sk__cquad_store(out + i, rv);
    }
    for (; i < n; ++i) {
        out[i].r = z[i].r * g[i];
        out[i].i = z[i].i * g[i];
    }
}
#else
static inline void sk_capply_gain_f32(Complex *out, const Complex *z,
                                       const float *g, int n) {
    sk_capply_gain_f32_scalar(out, z, g, n);
}
#endif

/* ═══════════════════════════════ sk_cadd_f32 ══════════════════════════════
 * out[i] = a[i] + b[i] (component-wise complex add). */

static inline void sk_cadd_f32_scalar(Complex *out, const Complex *a,
                                       const Complex *b, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        out[i].r = a[i].r + b[i].r;
        out[i].i = a[i].i + b[i].i;
    }
}

#if SK_HAVE_NEON
static inline void sk_cadd_f32(Complex *out, const Complex *a,
                                const Complex *b, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t av = sk__cquad_load(a + i);
        float32x4x2_t bv = sk__cquad_load(b + i);
        float32x4x2_t rv;
        rv.val[0] = vaddq_f32(av.val[0], bv.val[0]);
        rv.val[1] = vaddq_f32(av.val[1], bv.val[1]);
        sk__cquad_store(out + i, rv);
    }
    for (; i < n; ++i) {
        out[i].r = a[i].r + b[i].r;
        out[i].i = a[i].i + b[i].i;
    }
}
#else
static inline void sk_cadd_f32(Complex *out, const Complex *a,
                                const Complex *b, int n) {
    sk_cadd_f32_scalar(out, a, b, n);
}
#endif

/* ═══════════════════════════════ sk_csub_f32 ══════════════════════════════
 * out[i] = a[i] - b[i] (component-wise complex subtract). Supports out == a
 * (e.g. an accumulate-then-subtract-in-place sequence): each 4-lane block is
 * loaded in full before any part of it is stored, and no iteration revisits
 * an earlier index -- the same aliasing property sk_capply_gain_f32 above
 * documents for out == z. Partial overlap is not supported. */

static inline void sk_csub_f32_scalar(Complex *out, const Complex *a,
                                       const Complex *b, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        out[i].r = a[i].r - b[i].r;
        out[i].i = a[i].i - b[i].i;
    }
}

#if SK_HAVE_NEON
static inline void sk_csub_f32(Complex *out, const Complex *a,
                                const Complex *b, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t av = sk__cquad_load(a + i);
        float32x4x2_t bv = sk__cquad_load(b + i);
        float32x4x2_t rv;
        rv.val[0] = vsubq_f32(av.val[0], bv.val[0]);
        rv.val[1] = vsubq_f32(av.val[1], bv.val[1]);
        sk__cquad_store(out + i, rv);
    }
    for (; i < n; ++i) {
        out[i].r = a[i].r - b[i].r;
        out[i].i = a[i].i - b[i].i;
    }
}
#else
static inline void sk_csub_f32(Complex *out, const Complex *a,
                                const Complex *b, int n) {
    sk_csub_f32_scalar(out, a, b, n);
}
#endif

/* ═══════════════════════════════ sk_sq_scale_f32 ══════════════════════════
 * out[i] = (x[i]*x[i]) * scale. Two multiplies and no add: nothing to fuse. */

static inline void sk_sq_scale_f32_scalar(const float *x, float scale,
                                           float *out, int n) {
    int i;
    for (i = 0; i < n; ++i) out[i] = (x[i] * x[i]) * scale;
}

#if SK_HAVE_NEON
static inline void sk_sq_scale_f32(const float *x, float scale,
                                    float *out, int n) {
    int i = 0;
    float32x4_t sv = vdupq_n_f32(scale);
    for (; i + 4 <= n; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        float32x4_t sq = vmulq_f32(xv, xv);
        vst1q_f32(out + i, vmulq_f32(sq, sv));
    }
    for (; i < n; ++i) out[i] = (x[i] * x[i]) * scale;
}
#else
static inline void sk_sq_scale_f32(const float *x, float scale,
                                    float *out, int n) {
    sk_sq_scale_f32_scalar(x, scale, out, n);
}
#endif

/* ═══════════════════════════════ sk_min_f32, sk_clip_f32 ══════════════════
 * sk_min_f32:  out[i] = (a[i] < b[i]) ? a[i] : b[i].
 * sk_clip_f32: clip_f's branch order, low bound first then high bound.
 * Both use compare plus select (vcltq/vcgtq + vbslq) instead of
 * vminq/vmaxq, whose signed-zero tie-break differs from the scalar form. */

static inline void sk_min_f32_scalar(float *out, const float *a,
                                      const float *b, int n) {
    int i;
    for (i = 0; i < n; ++i) out[i] = (a[i] < b[i]) ? a[i] : b[i];
}

#if SK_HAVE_NEON
static inline void sk_min_f32(float *out, const float *a, const float *b, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        uint32x4_t mask = vcltq_f32(av, bv);       /* a < b, exact IEEE '<' */
        float32x4_t r = vbslq_f32(mask, av, bv);    /* mask ? a : b */
        vst1q_f32(out + i, r);
    }
    for (; i < n; ++i) out[i] = (a[i] < b[i]) ? a[i] : b[i];
}
#else
static inline void sk_min_f32(float *out, const float *a, const float *b, int n) {
    sk_min_f32_scalar(out, a, b, n);
}
#endif

static inline void sk_clip_f32_scalar(float *x, float lo, float hi, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        if (x[i] < lo) x[i] = lo;
        else if (x[i] > hi) x[i] = hi;
    }
}

#if SK_HAVE_NEON
static inline void sk_clip_f32(float *x, float lo, float hi, int n) {
    int i = 0;
    float32x4_t lov = vdupq_n_f32(lo), hiv = vdupq_n_f32(hi);
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        uint32x4_t lomask = vcltq_f32(v, lov);      /* x < lo */
        uint32x4_t himask = vcgtq_f32(v, hiv);      /* x > hi */
        float32x4_t r = vbslq_f32(lomask, lov, v);
        r = vbslq_f32(himask, hiv, r);              /* mutually exclusive
                                                      * with lomask when
                                                      * lo <= hi */
        vst1q_f32(x + i, r);
    }
    for (; i < n; ++i) {
        if (x[i] < lo) x[i] = lo;
        else if (x[i] > hi) x[i] = hi;
    }
}
#else
static inline void sk_clip_f32(float *x, float lo, float hi, int n) {
    sk_clip_f32_scalar(x, lo, hi, n);
}
#endif

/* ═══════════════════════════════ sk_fast_sqrt_f32 ═════════════════════════
 * Per-lane replica of fast_math.h fast_sqrt() (vsqrtq_f32 on AArch64). The
 * `v > 0` guard is a final vcgtq + vbslq select rather than a branch. The
 * hardware sqrt may produce NaN for discarded lanes; AArch64 FP exceptions do
 * not trap in the supported runtime configuration.
 *
 * NaN lanes: the predicate is `v > 0` (select the sqrt result when true, 0.0f
 * otherwise), not `v <= 0`. Both ordered compares are false for NaN, so only
 * the `v > 0` form sends a NaN lane to 0.0f, matching the scalar
 * sk__fast_sqrt_elem bit for bit. */

static inline void sk_fast_sqrt_f32_scalar(const float *x, float *out, int n) {
    int i;
    for (i = 0; i < n; ++i) out[i] = sk__fast_sqrt_elem(x[i]);
}

#if SK_HAVE_NEON
#ifdef USE_STANDARD_MATH
static inline void sk_fast_sqrt_f32(const float *x, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_f32(out + i, vsqrtq_f32(vld1q_f32(x + i)));
    for (; i < n; ++i) out[i] = sk__fast_sqrt_elem(x[i]);
}
#else
static inline void sk_fast_sqrt_f32(const float *x, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t xk = vsqrtq_f32(v);
        {
            /* `v > 0`, not `v <= 0`: false for NaN, which selects 0.0f. */
            uint32x4_t ispos = vcgtq_f32(v, vdupq_n_f32(0.0f));
            float32x4_t r = vbslq_f32(ispos, xk, vdupq_n_f32(0.0f));
            vst1q_f32(out + i, r);
        }
    }
    for (; i < n; ++i) out[i] = sk__fast_sqrt_elem(x[i]);
}
#endif /* USE_STANDARD_MATH */
#else
static inline void sk_fast_sqrt_f32(const float *x, float *out, int n) {
    sk_fast_sqrt_f32_scalar(x, out, n);
}
#endif

/* ═══════════════════ exp/log family ═══════════════════════════════════════
 * Vector equivalents of fast_math.h. Constants and operation order must stay
 * synchronized with that implementation. The exponential table index is
 * computed from an internally clamped [-16,16] value, keeping every gather
 * in bounds even when the final lane result is selected by a domain guard.
 * No operation is fused. Exact out==x use is supported except where a kernel
 * explicitly documents otherwise; partial overlap is unsupported.
 * USE_STANDARD_MATH falls back to scalar libm calls. */

#define SK_FM_LN2      0.693147180559945f
#define SK_FM_LOG10E   0.4342944819032518f
#define SK_FM_LN10     2.302585092994046f
#define SK_FM_EPSILON  1e-10f

/* Mirror of fast_math.h's FM_LOG_C1..C4 (degree-4 minimax ln(1+m); C2 is
 * stored positive and subtracted). */
#define SK_FM_LOG_C1   0.9972485899925232f
#define SK_FM_LOG_C2   0.46980342268943787f
#define SK_FM_LOG_C3   0.22263026237487793f
#define SK_FM_LOG_C4  -0.05692821741104126f

#define SK_EXP_TABLE_OFFSET 16
#define SK_EXP_TABLE_SIZE 33

/* Verbatim copy of fast_math.h's exp_int_table (e^n for n=-16..16). */
static const float sk_exp_int_table[SK_EXP_TABLE_SIZE] = {
    1.1253517471925912e-07f,  /* e^-16 */
    3.0590232050182579e-07f,  /* e^-15 */
    8.3152871910356788e-07f,  /* e^-14 */
    2.2603294069810542e-06f,  /* e^-13 */
    6.1442123533282097e-06f,  /* e^-12 */
    1.6701700790245659e-05f,  /* e^-11 */
    4.5399929762484854e-05f,  /* e^-10 */
    1.2340980408667956e-04f,  /* e^-9  */
    3.3546262790251185e-04f,  /* e^-8  */
    9.1188196555451624e-04f,  /* e^-7  */
    2.4787521766663585e-03f,  /* e^-6  */
    6.7379469990854670e-03f,  /* e^-5  */
    1.8315638888734180e-02f,  /* e^-4  */
    4.9787068367863944e-02f,  /* e^-3  */
    1.3533528323661270e-01f,  /* e^-2  */
    3.6787944117144233e-01f,  /* e^-1  */
    1.0000000000000000e+00f,  /* e^0   */
    2.7182818284590452e+00f,  /* e^1   */
    7.3890560989306502e+00f,  /* e^2   */
    2.0085536923187668e+01f,  /* e^3   */
    5.4598150033144236e+01f,  /* e^4   */
    1.4841315910257660e+02f,  /* e^5   */
    4.0342879349273511e+02f,  /* e^6   */
    1.0966331584284585e+03f,  /* e^7   */
    2.9809579870417283e+03f,  /* e^8   */
    8.1030839275753840e+03f,  /* e^9   */
    2.2026465794806718e+04f,  /* e^10  */
    5.9874141715197819e+04f,  /* e^11  */
    1.6275479141900392e+05f,  /* e^12  */
    4.4241339200892050e+05f,  /* e^13  */
    1.2026042841647768e+06f,  /* e^14  */
    3.2690173724721107e+06f,  /* e^15  */
    8.8861105205078726e+06f   /* e^16  */
};

#ifdef USE_STANDARD_MATH
static inline float sk__fast_exp_elem(float x)     { return expf(x); }
static inline float sk__fast_exp_neg_elem(float x) { return expf(-x); }
static inline float sk__fast_log_elem(float x)     { return logf(x); }
static inline float sk__fast_log10_elem(float x)   { return log10f(x); }
static inline float sk__exp1_approx_elem(float v) {
    if (v <= 1e-10f) v = 1e-10f;
    if (v < 0.1f) {
        return -2.31f * log10f(v) - 0.6f;
    } else if (v <= 1.0f) {
        return -1.544f * log10f(v) + 0.166f;
    } else {
        return powf(10.0f, -0.52f * v - 0.26f);
    }
}
#else
/* Verbatim from fast_math.h's fast_exp(), including the NaN-safe guard. */
static inline float sk__fast_exp_elem(float x) {
    if (!(x >= -16.0f)) return 0.0f;
    if (x > 16.0f) return 8.8861105e+06f;
    int x0 = (int)floorf(x);
    float dx = x - (float)x0;
    if (dx > 0.5f) { dx -= 1.0f; x0 += 1; }
    float exp_x0 = sk_exp_int_table[x0 + SK_EXP_TABLE_OFFSET];
    float dx2 = dx * dx;
    float exp_dx = 1.0f + dx + 0.5f * dx2 + (1.0f / 6.0f) * dx2 * dx;
    return exp_x0 * exp_dx;
}
/* Verbatim from fast_math.h's fast_exp_neg(). */
static inline float sk__fast_exp_neg_elem(float x) {
    if (x <= 0.0f) return 1.0f;
    if (x >= 16.0f) return 0.0f;
    return sk__fast_exp_elem(-x);
}
/* Verbatim from fast_math.h's fast_log(), including the NaN-safe guard. */
static inline float sk__fast_log_elem(float x) {
    if (!(x > 0.0f)) return -1e10f;
    union { float f; uint32_t i; } fb;
    fb.f = x;
    uint32_t bits = fb.i;
    int E = (int)((bits >> 23) & 0xFFu) - 127;
    fb.i = (bits & 0x007FFFFFu) | 0x3F800000u;
    float m = fb.f - 1.0f;
    float poly = SK_FM_LOG_C4;
    poly = poly * m + SK_FM_LOG_C3;
    poly = poly * m - SK_FM_LOG_C2;
    poly = poly * m + SK_FM_LOG_C1;
    float ln_1_m = poly * m;
    return (float)E * SK_FM_LN2 + ln_1_m;
}
/* Verbatim from fast_math.h's fast_log10(). */
static inline float sk__fast_log10_elem(float x) {
    return sk__fast_log_elem(x) * SK_FM_LOG10E;
}
/* Verbatim from fast_math.h's exp1_approx() in its default branch order. The
 * USE_OPTIMIZED_E1 variant only reorders the checks; its results are
 * bit-identical for every v (NaN included), so this one order covers both
 * modes. */
static inline float sk__exp1_approx_elem(float v) {
    if (v <= SK_FM_EPSILON) v = SK_FM_EPSILON;
    if (v < 0.1f) {
        return -2.31f * sk__fast_log10_elem(v) - 0.6f;
    } else if (v <= 1.0f) {
        return -1.544f * sk__fast_log10_elem(v) + 0.166f;
    } else {
        return sk__fast_exp_elem((-0.52f * v - 0.26f) * SK_FM_LN10);
    }
}
#endif /* USE_STANDARD_MATH */

static inline void sk_fast_exp_f32_scalar(const float *x, float *out, int n) {
    int i;
    for (i = 0; i < n; ++i) out[i] = sk__fast_exp_elem(x[i]);
}
static inline void sk_fast_exp_neg_f32_scalar(const float *x, float *out, int n) {
    int i;
    for (i = 0; i < n; ++i) out[i] = sk__fast_exp_neg_elem(x[i]);
}
static inline void sk_fast_log_f32_scalar(const float *x, float *out, int n) {
    int i;
    for (i = 0; i < n; ++i) out[i] = sk__fast_log_elem(x[i]);
}
static inline void sk_exp1_approx_f32_scalar(const float *x, float *out, int n) {
    int i;
    for (i = 0; i < n; ++i) out[i] = sk__exp1_approx_elem(x[i]);
}

#if SK_HAVE_NEON && !defined(USE_STANDARD_MATH)
/* Vector core of sk_fast_exp_f32, also used by sk_fast_exp_neg_f32 (on -x) and
 * by the v > 1 branch of sk_exp1_approx_f32. Safe for any input (NaN, Inf,
 * any magnitude): the table index comes from a clamped value. */
static inline float32x4_t sk__fast_exp_vec(float32x4_t x) {
    float32x4_t neg16 = vdupq_n_f32(-16.0f);
    float32x4_t pos16 = vdupq_n_f32(16.0f);
    /* The clamp only keeps the table index in bounds; the final edge select
     * uses the unclamped x, so no lane's result changes. */
    float32x4_t xc = vminq_f32(vmaxq_f32(x, neg16), pos16);
    float32x4_t floor_xc = vrndmq_f32(xc);       /* floorf; exact integral here */
    float32x4_t dx = vsubq_f32(xc, floor_xc);
    uint32x4_t adjmask = vcgtq_f32(dx, vdupq_n_f32(0.5f));
    float32x4_t dx_adj = vsubq_f32(dx, vdupq_n_f32(1.0f));
    float32x4_t x0f = vbslq_f32(adjmask, vaddq_f32(floor_xc, vdupq_n_f32(1.0f)), floor_xc);
    dx = vbslq_f32(adjmask, dx_adj, dx);

    {
        int32x4_t x0i = vaddq_s32(vcvtq_s32_f32(x0f), vdupq_n_s32(SK_EXP_TABLE_OFFSET));
        float table_lanes[4];
        /* No NEON gather: four scalar table reads. x0i is in [0, 32] for
         * every input, NaN included. */
        table_lanes[0] = sk_exp_int_table[vgetq_lane_s32(x0i, 0)];
        table_lanes[1] = sk_exp_int_table[vgetq_lane_s32(x0i, 1)];
        table_lanes[2] = sk_exp_int_table[vgetq_lane_s32(x0i, 2)];
        table_lanes[3] = sk_exp_int_table[vgetq_lane_s32(x0i, 3)];
        {
            float32x4_t exp_x0 = vld1q_f32(table_lanes);
            /* 1.0f + dx + 0.5f*dx2 + (1/6)*dx2*dx in the operation order of
             * sk__fast_exp_elem, no vfmaq_f32. */
            float32x4_t dx2 = vmulq_f32(dx, dx);
            float32x4_t term4 = vmulq_f32(vmulq_f32(vdupq_n_f32(1.0f / 6.0f), dx2), dx);
            float32x4_t term3 = vmulq_f32(vdupq_n_f32(0.5f), dx2);
            float32x4_t sum = vaddq_f32(vdupq_n_f32(1.0f), dx);
            sum = vaddq_f32(sum, term3);
            sum = vaddq_f32(sum, term4);
            {
                float32x4_t normal = vmulq_f32(exp_x0, sum);
                /* Edge selects on the unclamped x. lomask is the scalar's
                 * first check, so it is applied last (highest priority);
                 * the two masks never overlap. */
                uint32x4_t himask = vcgtq_f32(x, pos16);            /* x > 16 (finite only) */
                uint32x4_t lomask = vmvnq_u32(vcgeq_f32(x, neg16)); /* !(x>=-16): x<-16 or NaN */
                float32x4_t r = vbslq_f32(himask, vdupq_n_f32(8.8861105e+06f), normal);
                r = vbslq_f32(lomask, vdupq_n_f32(0.0f), r);
                return r;
            }
        }
    }
}

/* Supports out == x: each 4-lane block is loaded before it is stored and no
 * iteration reads a lane an earlier one wrote. */
static inline void sk_fast_exp_f32(const float *x, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(out + i, sk__fast_exp_vec(vld1q_f32(x + i)));
    for (; i < n; ++i) out[i] = sk__fast_exp_elem(x[i]);
}

/* Supports out == x, as sk_fast_exp_f32. */
static inline void sk_fast_exp_neg_f32(const float *x, float *out, int n) {
    int i = 0;
    float32x4_t zero = vdupq_n_f32(0.0f), sixteen = vdupq_n_f32(16.0f), one = vdupq_n_f32(1.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        /* if (x<=0) return 1.0f; if (x>=16) return 0.0f; return fast_exp(-x);
         * mask_le0 is the scalar's first check, so it is applied last. */
        uint32x4_t mask_le0 = vcleq_f32(xv, zero);
        uint32x4_t mask_ge16 = vcgeq_f32(xv, sixteen);
        float32x4_t inner = sk__fast_exp_vec(vnegq_f32(xv));
        float32x4_t r = vbslq_f32(mask_ge16, zero, inner);
        r = vbslq_f32(mask_le0, one, r);
        vst1q_f32(out + i, r);
    }
    for (; i < n; ++i) out[i] = sk__fast_exp_neg_elem(x[i]);
}

/* Vector core of sk_fast_log_f32, also used by the
 * v <= 1 branches of sk_exp1_approx_f32. Pure bit manipulation with no memory
 * access, so it is computed for every lane, out-of-domain ones included; it
 * replicates fast_math.h's fast_log() lane for lane. */
static inline float32x4_t sk__fast_log_vec(float32x4_t x) {
    uint32x4_t bits = vreinterpretq_u32_f32(x);
    int32x4_t exp_bits = vreinterpretq_s32_u32(vandq_u32(vshrq_n_u32(bits, 23), vdupq_n_u32(0xFFu)));
    int32x4_t E = vsubq_s32(exp_bits, vdupq_n_s32(127));
    /* Clear the exponent/sign fields, force exponent to 127 (bias), matching
     * the scalar explicit-mask implementation. */
    uint32x4_t mantissa_mask = vdupq_n_u32(0x007FFFFFu);
    uint32x4_t bias127 = vdupq_n_u32(0x3F800000u);
    uint32x4_t new_bits = vorrq_u32(vandq_u32(bits, mantissa_mask), bias127);
    float32x4_t mant_f = vreinterpretq_f32_u32(new_bits);
    float32x4_t m = vsubq_f32(mant_f, vdupq_n_f32(1.0f));
    /* Degree-4 minimax Horner sequence with separately rounded multiply and
     * add, as in sk__fast_log_elem; no vfmaq_f32. */
    float32x4_t poly = vdupq_n_f32(SK_FM_LOG_C4);
    poly = vaddq_f32(vmulq_f32(poly, m), vdupq_n_f32(SK_FM_LOG_C3));
    poly = vsubq_f32(vmulq_f32(poly, m), vdupq_n_f32(SK_FM_LOG_C2));
    poly = vaddq_f32(vmulq_f32(poly, m), vdupq_n_f32(SK_FM_LOG_C1));
    float32x4_t ln_1_m = vmulq_f32(poly, m);
    {
        float32x4_t Ef = vcvtq_f32_s32(E);
        float32x4_t result = vaddq_f32(vmulq_f32(Ef, vdupq_n_f32(SK_FM_LN2)), ln_1_m);
        uint32x4_t badmask = vmvnq_u32(vcgtq_f32(x, vdupq_n_f32(0.0f))); /* !(x>0): x<=0 or NaN */
        return vbslq_f32(badmask, vdupq_n_f32(-1e10f), result);
    }
}

/* Supports out == x, as sk_fast_exp_f32. */
static inline void sk_fast_log_f32(const float *x, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(out + i, sk__fast_log_vec(vld1q_f32(x + i)));
    for (; i < n; ++i) out[i] = sk__fast_log_elem(x[i]);
}

/* Supports out == x. Uniform four-bin regimes skip the unused E1 branch;
 * mixed regimes retain both branches and the scalar comparison order. */
static inline void sk_exp1_approx_f32(const float *x, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t eps = vdupq_n_f32(SK_FM_EPSILON);
        /* if (v<=EPS) v=EPS; false for NaN, so NaN passes through as in
         * sk__exp1_approx_elem. */
        uint32x4_t clampmask = vcleq_f32(v, eps);
        float32x4_t vc = vbslq_f32(clampmask, eps, v);
        uint32x4_t mask_lt01 = vcltq_f32(vc, vdupq_n_f32(0.1f));
        uint32x4_t mask_le1  = vcleq_f32(vc, vdupq_n_f32(1.0f));

        if (vminvq_u32(mask_le1) == UINT32_MAX) {
            float32x4_t log10v = vmulq_f32(sk__fast_log_vec(vc), vdupq_n_f32(SK_FM_LOG10E));
            float32x4_t branch1 = vsubq_f32(vmulq_f32(vdupq_n_f32(-2.31f), log10v), vdupq_n_f32(0.6f));
            float32x4_t branch2 = vaddq_f32(vmulq_f32(vdupq_n_f32(-1.544f), log10v), vdupq_n_f32(0.166f));
            vst1q_f32(out + i, vbslq_f32(mask_lt01, branch1, branch2));
            continue;
        }

        float32x4_t exparg = vmulq_f32(
            vsubq_f32(vmulq_f32(vdupq_n_f32(-0.52f), vc), vdupq_n_f32(0.26f)),
            vdupq_n_f32(SK_FM_LN10));
        float32x4_t branch3 = sk__fast_exp_vec(exparg);
        if (vmaxvq_u32(mask_le1) == 0) {
            /* NaN comparisons are false, so NaN follows branch3 as before. */
            vst1q_f32(out + i, branch3);
            continue;
        }

        float32x4_t log10v = vmulq_f32(sk__fast_log_vec(vc), vdupq_n_f32(SK_FM_LOG10E));
        float32x4_t branch1 = vsubq_f32(vmulq_f32(vdupq_n_f32(-2.31f), log10v), vdupq_n_f32(0.6f));
        float32x4_t branch2 = vaddq_f32(vmulq_f32(vdupq_n_f32(-1.544f), log10v), vdupq_n_f32(0.166f));
        float32x4_t r = vbslq_f32(mask_le1, branch2, branch3);
        vst1q_f32(out + i, vbslq_f32(mask_lt01, branch1, r));
    }
    for (; i < n; ++i) out[i] = sk__exp1_approx_elem(x[i]);
}
#else
/* USE_STANDARD_MATH or no NEON: delegate to the per-element scalar entry
 * points, for which out == x is equally safe. */
static inline void sk_fast_exp_f32(const float *x, float *out, int n) {
    sk_fast_exp_f32_scalar(x, out, n);
}
static inline void sk_fast_exp_neg_f32(const float *x, float *out, int n) {
    sk_fast_exp_neg_f32_scalar(x, out, n);
}
static inline void sk_fast_log_f32(const float *x, float *out, int n) {
    sk_fast_log_f32_scalar(x, out, n);
}
static inline void sk_exp1_approx_f32(const float *x, float *out, int n) {
    sk_exp1_approx_f32_scalar(x, out, n);
}
#endif /* SK_HAVE_NEON && !USE_STANDARD_MATH */

/* ═══════════════════════════════ sk_noise_update_f32 ══════════════════════
 * Noise-PSD tracking update whose smoothing factor depends on speech presence,
 * evaluated per bin in one pass:
 *   su = spp[i] * bb_scale                 weighted speech-presence probability
 *   a  = alpha_d + (1-alpha_d) * su        base factor lifted toward 1 as speech
 *                                          presence grows (estimate holds)
 *   noise[i] = a*noise[i] + (1-a)*power[i] recursive average toward the power
 * The source uses separate rounded multiply/add operations; do not use FMA. */

static inline void sk_noise_update_f32_scalar(float *noise_psd,
                                              const float *spp,
                                              const float *power,
                                              float alpha_d,
                                              float bb_scale,
                                              int n) {
    int i;
    for (i = 0; i < n; ++i) {
        float su = spp[i] * bb_scale;
        float tilde_alpha_d = alpha_d + (1.0f - alpha_d) * su;
        noise_psd[i] = tilde_alpha_d * noise_psd[i] + (1.0f - tilde_alpha_d) * power[i];
    }
}

#if SK_HAVE_NEON
static inline void sk_noise_update_f32(float *noise_psd,
                                       const float *spp,
                                       const float *power,
                                       float alpha_d,
                                       float bb_scale,
                                       int n) {
    int i = 0;
    float32x4_t va_d = vdupq_n_f32(alpha_d);
    float32x4_t vbb = vdupq_n_f32(bb_scale);
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t one_minus_ad = vsubq_f32(one, va_d);
    for (; i + 4 <= n; i += 4) {
        float32x4_t spp_v = vld1q_f32(spp + i);
        float32x4_t power_v = vld1q_f32(power + i);
        float32x4_t npsd_v = vld1q_f32(noise_psd + i);
        float32x4_t su = vmulq_f32(spp_v, vbb);
        float32x4_t tilde = vaddq_f32(va_d, vmulq_f32(one_minus_ad, su));
        float32x4_t one_minus_tilde = vsubq_f32(one, tilde);
        float32x4_t term1 = vmulq_f32(tilde, npsd_v);
        float32x4_t term2 = vmulq_f32(one_minus_tilde, power_v);
        vst1q_f32(noise_psd + i, vaddq_f32(term1, term2));
    }
    for (; i < n; ++i) {
        float su = spp[i] * bb_scale;
        float tilde_alpha_d = alpha_d + (1.0f - alpha_d) * su;
        noise_psd[i] = tilde_alpha_d * noise_psd[i] + (1.0f - tilde_alpha_d) * power[i];
    }
}
#else
static inline void sk_noise_update_f32(float *noise_psd,
                                       const float *spp,
                                       const float *power,
                                       float alpha_d,
                                       float bb_scale,
                                       int n) {
    sk_noise_update_f32_scalar(noise_psd, spp, power, alpha_d, bb_scale, n);
}
#endif


/* ═══════════════════════════════ sk_wola_accumulate_f32 ═══════════════════
 * acc[i] += x[i] * w[i]
 *
 * Windowed overlap-add accumulate, run once per hop over a whole FFT frame.
 * Unit stride, no reduction; the three arrays must not alias.
 *
 * Separate vmulq_f32 and vaddq_f32, not vfmaq_f32: the scalar form is a
 * multiply then an add, which rounds twice under -ffp-contract=off, and a
 * fused multiply-add would differ in the last bit. Kernels whose scalar
 * reference calls fmaf() do fuse.
 */

static inline void sk_wola_accumulate_f32_scalar(float *acc, const float *x,
                                                  const float *w, int n) {
    int i;
    for (i = 0; i < n; ++i) acc[i] += x[i] * w[i];
}

#if SK_HAVE_NEON
static inline void sk_wola_accumulate_f32(float *acc, const float *x,
                                           const float *w, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a = vld1q_f32(acc + i);
        float32x4_t xv = vld1q_f32(x + i);
        float32x4_t wv = vld1q_f32(w + i);
        /* mul then add, never vfmaq_f32. */
        vst1q_f32(acc + i, vaddq_f32(a, vmulq_f32(xv, wv)));
    }
    for (; i < n; ++i) acc[i] += x[i] * w[i];
}
#else
static inline void sk_wola_accumulate_f32(float *acc, const float *x,
                                           const float *w, int n) {
    sk_wola_accumulate_f32_scalar(acc, x, w, n);
}
#endif

/* ═══════════════════════════════ sk_clip_scale_to_s16 ═════════════════════
 * float in [-1, 1] -> int16 PCM: clip to +-1, scale by 32767, truncate.
 *
 *   sk__clip_scale_to_s16_elem()   one sample; the scalar reference
 *   sk_clip_scale_to_s16_arr()     a whole planar buffer
 *   sk_clip_scale_to_s16()         one NEON quad -> int16x4_t, for a caller
 *                                  that interleaves two channels with
 *                                  vst2_s16
 *
 * The quad form is NEON-only and has no _scalar twin (its return type needs
 * <arm_neon.h>); callers guard it with SK_HAVE_NEON and fall back to the
 * per-sample helper.
 *
 * Bit-exact against the scalar reference:
 *   - clipping is vcgtq/vcltq + vbslq_f32, the same idiom as sk_clip_f32.
 *     The signed-zero tie-break of vmaxq/vminq is unobservable here because
 *     -0.0f and +0.0f convert to the same int16;
 *   - vcvtq_s32_f32 truncates toward zero like C's float -> int conversion.
 *     vcvtnq_s32_f32 rounds to nearest and would not match;
 *   - vqmovn_s32 never saturates (the input is clipped to +-32767), so it is
 *     a plain narrow.
 *
 * wav_io.h's writer is not built on this kernel: its NR-style branch counts
 * and zeroes non-finite input and its AEC-style branch scales by 32768.0f
 * with round-half-away-from-zero.
 *
 * The high bound is tested before the low one; for non-NaN input the order
 * does not matter. NaN is not handled: it survives both clips and reaches a
 * float -> int conversion, which C leaves undefined. Callers must keep NaN
 * out of this path.
 */

#define SK_S16_FULL_SCALE 32767.0f

static inline int16_t sk__clip_scale_to_s16_elem(float x) {
    if (x > 1.0f) x = 1.0f;
    else if (x < -1.0f) x = -1.0f;
    return (int16_t)(x * SK_S16_FULL_SCALE);
}

static inline void sk_clip_scale_to_s16_arr_scalar(int16_t *out,
                                                    const float *in, int n) {
    int i;
    for (i = 0; i < n; ++i) out[i] = sk__clip_scale_to_s16_elem(in[i]);
}

#if SK_HAVE_NEON
static inline int16x4_t sk_clip_scale_to_s16(float32x4_t v) {
    const float32x4_t lo = vdupq_n_f32(-1.0f);
    const float32x4_t hi = vdupq_n_f32(1.0f);
    v = vbslq_f32(vcgtq_f32(v, hi), hi, v);      /* x > 1 ? 1 : x */
    v = vbslq_f32(vcltq_f32(v, lo), lo, v);      /* x < -1 ? -1 : x */
    /* truncate, never vcvtnq_s32_f32 */
    return vqmovn_s32(vcvtq_s32_f32(vmulq_n_f32(v, SK_S16_FULL_SCALE)));
}

static inline void sk_clip_scale_to_s16_arr(int16_t *out, const float *in,
                                             int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4)
        vst1_s16(out + i, sk_clip_scale_to_s16(vld1q_f32(in + i)));
    for (; i < n; ++i) out[i] = sk__clip_scale_to_s16_elem(in[i]);
}
#else
static inline void sk_clip_scale_to_s16_arr(int16_t *out, const float *in,
                                             int n) {
    sk_clip_scale_to_s16_arr_scalar(out, in, n);
}
#endif

/* ═══════════════════════════════ sk_cmag_f32 ══════════════════════════════
 * out[i] = x[i].r*x[i].r + x[i].i*x[i].i + eps
 *
 * Despite the name this is the squared magnitude (power) plus a floor; there
 * is no square root. eps is added after the two products are summed.
 *
 * Separate vmulq/vaddq, never vfmaq_f32: the scalar form rounds at each of
 * its two multiplies and two adds under -ffp-contract=off.
 *
 * The de-interleave uses sk__cquad_load, not a raw Complex* -> float* cast
 * into vld2q_f32: that cast violates strict aliasing in every consumer of
 * this header regardless of its flags.
 *
 * Not the same as sk_cmag2_np_f32 of simd_aec.h, which uses the
 * numpy scaled-hypot form and rounds differently.
 */

static inline void sk_cmag_f32_scalar(float *out, const Complex *x, int n,
                                       float eps) {
    int i;
    for (i = 0; i < n; ++i) out[i] = x[i].r * x[i].r + x[i].i * x[i].i + eps;
}

#if SK_HAVE_NEON
static inline void sk_cmag_f32(float *out, const Complex *x, int n,
                                float eps) {
    int i = 0;
    float32x4_t ev = vdupq_n_f32(eps);
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t ri = sk__cquad_load(x + i);   /* de-interleave */
        float32x4_t rr = vmulq_f32(ri.val[0], ri.val[0]);
        float32x4_t ii = vmulq_f32(ri.val[1], ri.val[1]);
        vst1q_f32(out + i, vaddq_f32(vaddq_f32(rr, ii), ev));
    }
    for (; i < n; ++i) out[i] = x[i].r * x[i].r + x[i].i * x[i].i + eps;
}
#else
static inline void sk_cmag_f32(float *out, const Complex *x, int n,
                                float eps) {
    sk_cmag_f32_scalar(out, x, n, eps);
}
#endif

/* ═══════════════════════════════ sk_linear_to_db_f32 ══════════════════════
 * out[i] = 10 * log10f(x[i])
 *
 * Power-domain dB (10x, not 20x), for squared magnitudes such as those of
 * sk_cmag_f32.
 *
 * There is no NEON path: the only bit-exact log10f is libm, and a dispatched
 * kernel must be bitwise identical to its scalar twin. A fast log10
 * approximation would move every value compared against a dB threshold.
 */

static inline void sk_linear_to_db_f32(float *out, const float *x, int n) {
    int i;
    for (i = 0; i < n; ++i) out[i] = 10.0f * log10f(x[i]);
}

/* ═══════════════════════════════ sk_asym_ema_f32 ══════════════════════════
 * One-pole EMA with a different coefficient per direction:
 *
 *     s[i] = (x[i] < s[i]) ? a_down*s[i] + (1-a_down)*x[i]
 *                          : a_up  *s[i] + (1-a_up)  *x[i]
 *
 * The noise-floor tracker shape: a slow a_up so the floor creeps upward, a
 * faster a_down so it drops promptly. `s` is updated IN PLACE.
 *
 * The comparison is strict `<` and selects a_down. Scalar and NEON take the
 * same branch at equality, but the two branches do not give the same value
 * there (`a*v + (1-a)*v` rounds differently for different a). A reference
 * that selects on `>` therefore differs at the tie and must not use this
 * kernel.
 *
 * (1-a) is loop-invariant, so hoisting it is bit-identical to recomputing it.
 * Separate vmulq/vaddq, never vfmaq_f32.
 */

static inline void sk_asym_ema_f32_scalar(float *s, const float *x, int n,
                                           float a_up, float a_down) {
    int i;
    for (i = 0; i < n; ++i) {
        if (x[i] < s[i]) s[i] = a_down * s[i] + (1.0f - a_down) * x[i];
        else             s[i] = a_up   * s[i] + (1.0f - a_up)   * x[i];
    }
}

#if SK_HAVE_NEON
static inline void sk_asym_ema_f32(float *s, const float *x, int n,
                                    float a_up, float a_down) {
    int i = 0;
    float32x4_t up = vdupq_n_f32(a_up);
    float32x4_t dn = vdupq_n_f32(a_down);
    float32x4_t up1 = vdupq_n_f32(1.0f - a_up);
    float32x4_t dn1 = vdupq_n_f32(1.0f - a_down);
    for (; i + 4 <= n; i += 4) {
        float32x4_t sv = vld1q_f32(s + i);
        float32x4_t xv = vld1q_f32(x + i);
        float32x4_t r_dn = vaddq_f32(vmulq_f32(dn, sv), vmulq_f32(dn1, xv));
        float32x4_t r_up = vaddq_f32(vmulq_f32(up, sv), vmulq_f32(up1, xv));
        /* x < s ? falling : rising -- compare+select, both branches computed. */
        vst1q_f32(s + i, vbslq_f32(vcltq_f32(xv, sv), r_dn, r_up));
    }
    for (; i < n; ++i) {
        if (x[i] < s[i]) s[i] = a_down * s[i] + (1.0f - a_down) * x[i];
        else             s[i] = a_up   * s[i] + (1.0f - a_up)   * x[i];
    }
}
#else
static inline void sk_asym_ema_f32(float *s, const float *x, int n,
                                    float a_up, float a_down) {
    sk_asym_ema_f32_scalar(s, x, n, a_up, a_down);
}
#endif

/* ═══════════════════════════════ sk_scale_f32 ═════════════════════════════
 * out[i] = x[i] * g, one broadcast gain over an array. Supports out == x:
 * each iteration loads before it stores and never revisits an index.
 *
 * One multiply per element: nothing to fuse, and the result is exactly
 * rounded however it is vectorised.
 */

static inline void sk_scale_f32_scalar(float *out, const float *x,
                                        float g, int n) {
    int i;
    for (i = 0; i < n; ++i) out[i] = x[i] * g;
}

#if SK_HAVE_NEON
static inline void sk_scale_f32(float *out, const float *x, float g, int n) {
    int i = 0;
    float32x4_t gv = vdupq_n_f32(g);
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(x + i), gv));
    }
    for (; i < n; ++i) out[i] = x[i] * g;
}
#else
static inline void sk_scale_f32(float *out, const float *x, float g, int n) {
    sk_scale_f32_scalar(out, x, g, n);
}
#endif

/* ═══════════════════════════════ sk_argmax_sq_f32 ═════════════════════════
 * np.argmax(h*h): FIRST index of the strongest squared element, as the plain
 * walk `if (v > best_v) { best_v = v; best = i; }` computes it (seeded at
 * h[0]*h[0]; returns 0 for n<2).
 *
 * The NEON form is a two-pass reduction: pass 1 max-reduces h[i]*h[i] across
 * four independent lane accumulators (seeded at +0.0f -- every square is
 * >= +0.0f, so the seed can never win); pass 2 finds the first element whose
 * square equals that maximum. Both passes recompute h[i]*h[i] with the same
 * plain float32 multiply the walk uses (no fma), so they see the identical
 * multiset of squares: max over a multiset is order-free, and "first index
 * equal to the max" IS the walk's first-index-of-max rule, so the returned
 * index matches the walk's for every input whose peak is finite. A
 * non-finite peak (a NaN element's square propagates through vmaxq/vmaxvq,
 * an overflowing element squares to +inf) carries no equality-match
 * guarantee and falls back to the scalar walk instead. */

static inline int sk_argmax_sq_f32_scalar(const float *h, int n) {
    int best = 0, i;
    float best_v;
    if (n < 2) return 0;
    best_v = h[0] * h[0];
    for (i = 1; i < n; ++i) {
        float v = h[i] * h[i];
        if (v > best_v) { best_v = v; best = i; }
    }
    return best;
}

#if SK_HAVE_NEON
static inline int sk_argmax_sq_f32(const float *h, int n) {
    float32x4_t m0, m1, m2, m3;
    float best_v;
    int i;
    if (n < 2) return 0;
    /* seeded at +0.0: every square is >= +0.0, so the seed can never win */
    m0 = m1 = m2 = m3 = vdupq_n_f32(0.0f);
    for (i = 0; i + 16 <= n; i += 16) {
        float32x4_t a0 = vld1q_f32(h + i),      a1 = vld1q_f32(h + i + 4);
        float32x4_t a2 = vld1q_f32(h + i + 8),  a3 = vld1q_f32(h + i + 12);
        m0 = vmaxq_f32(m0, vmulq_f32(a0, a0));
        m1 = vmaxq_f32(m1, vmulq_f32(a1, a1));
        m2 = vmaxq_f32(m2, vmulq_f32(a2, a2));
        m3 = vmaxq_f32(m3, vmulq_f32(a3, a3));
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t a0 = vld1q_f32(h + i);
        m0 = vmaxq_f32(m0, vmulq_f32(a0, a0));
    }
    best_v = vmaxvq_f32(vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3)));
    /* the ragged tail folds in with the walk's own `>` test, which (like the
     * walk) leaves a NaN square out of the peak rather than propagating it */
    for (; i < n; ++i) { float v = h[i] * h[i]; if (v > best_v) best_v = v; }
    if (isfinite(best_v)) {
        float32x4_t peak = vdupq_n_f32(best_v);
        int k;
        for (k = 0; k + 16 <= n; k += 16) {
            float32x4_t a0 = vld1q_f32(h + k),      a1 = vld1q_f32(h + k + 4);
            float32x4_t a2 = vld1q_f32(h + k + 8),  a3 = vld1q_f32(h + k + 12);
            uint32x4_t e0 = vceqq_f32(vmulq_f32(a0, a0), peak);
            uint32x4_t e1 = vceqq_f32(vmulq_f32(a1, a1), peak);
            uint32x4_t e2 = vceqq_f32(vmulq_f32(a2, a2), peak);
            uint32x4_t e3 = vceqq_f32(vmulq_f32(a3, a3), peak);
            if (vmaxvq_u32(vorrq_u32(vorrq_u32(e0, e1), vorrq_u32(e2, e3)))) {
                int j;
                for (j = k; j < k + 16; ++j)
                    if (h[j] * h[j] == best_v) return j;
            }
        }
        for (; k < n; ++k)
            if (h[k] * h[k] == best_v) return k;
    }
    /* non-finite peak; also structurally closes the function (a finite peak
     * is by construction one of the squares, so pass 2 always found it) */
    return sk_argmax_sq_f32_scalar(h, n);
}
#else
static inline int sk_argmax_sq_f32(const float *h, int n) {
    return sk_argmax_sq_f32_scalar(h, n);
}
#endif

/* ═══════════════════════════════ sk_argmax_f32 ════════════════════════════
 * np.argmax(x): FIRST index of the maximum raw value -- NOT squared (see
 * sk_argmax_sq_f32 above for that; a different value-function with a
 * different seed strategy, not interchangeable). The plain walk
 * `if (x[i] > x[best]) best = i` computes it.
 *
 * The NEON form is a two-pass reduction, gated on n>=8 and x[0] not NaN (a
 * NaN seed is never beaten by the walk, and a later NaN never wins, so the
 * walk's result for a NaN-seeded array depends on position in a way the
 * reduction does not reproduce -- that case falls straight through to the
 * walk below): pass 1 seeds its two 4-lane accumulators by LOADING x[0..3]
 * (not +0.0f -- x can be negative, unlike sk_argmax_sq_f32's squares), then
 * max-reduces the rest; pass 2 finds the first element equal to that
 * maximum. Max over a multiset is order-free and "first index equal to the
 * max" is the walk's rule, so for a finite maximum the two passes reproduce
 * the walk's index exactly; a non-finite maximum falls back to the walk
 * itself. */

static inline int sk_argmax_f32_scalar(const float *x, int n) {
    int best = 0, i;
    for (i = 1; i < n; ++i) if (x[i] > x[best]) best = i;
    return best;
}

#if SK_HAVE_NEON
static inline int sk_argmax_f32(const float *x, int n) {
    if (n >= 8 && !isnan(x[0])) {
        float32x4_t m0 = vld1q_f32(x), m1 = m0;
        float best_v;
        int i = 4;
        for (; i + 8 <= n; i += 8) {
            m0 = vmaxq_f32(m0, vld1q_f32(x + i));
            m1 = vmaxq_f32(m1, vld1q_f32(x + i + 4));
        }
        for (; i + 4 <= n; i += 4) m0 = vmaxq_f32(m0, vld1q_f32(x + i));
        best_v = vmaxvq_f32(vmaxq_f32(m0, m1));
        for (; i < n; ++i) if (x[i] > best_v) best_v = x[i];
        if (isfinite(best_v)) {
            const float32x4_t peak = vdupq_n_f32(best_v);
            int k;
            for (k = 0; k + 4 <= n; k += 4) {
                if (vmaxvq_u32(vceqq_f32(vld1q_f32(x + k), peak))) {
                    int j;
                    for (j = k; j < k + 4; ++j) if (x[j] == best_v) return j;
                }
            }
            for (; k < n; ++k) if (x[k] == best_v) return k;
        }
    }
    return sk_argmax_f32_scalar(x, n);
}
#else
static inline int sk_argmax_f32(const float *x, int n) {
    return sk_argmax_f32_scalar(x, n);
}
#endif

/* ═══════════════════════════════ sk_argmax_i32 ════════════════════════════
 * np.argmax over an int32 array: FIRST index of the maximum. Scalar only --
 * no float rounding is involved anywhere in this comparison, so there is no
 * NEON/scalar bit-exactness question and no dispatched twin. Requires n>=1
 * (reads a[0] unconditionally, as the walk it replaces always has). */

static inline int sk_argmax_i32(const int *a, int n) {
    int best = 0, i;
    int best_v = a[0];
    for (i = 1; i < n; ++i) {
        if (a[i] > best_v) { best_v = a[i]; best = i; }
    }
    return best;
}

/* ═══════════════════════════ sk_cmac_fma_f32_scalar ═══════════════════════
 * acc[i] += w[i] * x[i] (numpy complex64 multiply, FMA form):
 *   acc[k].r += fmaf(wr, xr, -(wi * xi));
 *   acc[k].i += fmaf(wr, xi,  (wi * xr));
 * The "_fma" in the name is explicit because sk_cmac_f32 (below) is the
 * UNFUSED sibling -- separate multiply/subtract/add, a different
 * value-function, not interchangeable bit-for-bit. */

static inline void sk_cmac_fma_f32_scalar(Complex *acc, const Complex *w,
                                           const Complex *x, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        float wr = w[i].r, wi = w[i].i, xr = x[i].r, xi = x[i].i;
        acc[i].r += fmaf(wr, xr, -(wi * xi));
        acc[i].i += fmaf(wr, xi,  (wi * xr));
    }
}

/* ═══════════════════════════ sk_cmac_fma_ring_f32 ══════════════════════════
 * acc[k] += w_part[k] * x_part[k] for part = 0 .. n_part-1 IN THAT ORDER,
 * where w_part = W + part*K and x_part is X_buf's row (curr_p - part) mod
 * n_part -- a partitioned echo accumulation, i.e. exactly
 * sk_cmac_fma_f32_scalar once per partition. The NEON form keeps each
 * four-bin accumulator in registers across the partitions instead of
 * storing and reloading it n_part times; every bin still sees the same fma-form product added in the
 * same partition order, so the result is bit-identical. */

static inline void sk_cmac_fma_ring_f32_scalar(Complex *acc, const Complex *W,
                                                const Complex *X_buf,
                                                int curr_p, int n_part,
                                                int K) {
    int part;
    for (part = 0; part < n_part; ++part) {
        int p_idx = curr_p - part;
        while (p_idx < 0) p_idx += n_part;
        sk_cmac_fma_f32_scalar(acc, W + (size_t)part * K,
                               X_buf + (size_t)p_idx * K, K);
    }
}

#if SK_HAVE_NEON
static inline void sk_cmac_fma_ring_f32(Complex *acc, const Complex *W,
                                         const Complex *X_buf, int curr_p,
                                         int n_part, int K) {
    int i = 0, part;
    for (; i + 4 <= K; i += 4) {
        float32x4x2_t av = sk__cquad_load(acc + i);
        int p_idx = curr_p;
        for (part = 0; part < n_part; ++part) {
            float32x4x2_t wv = sk__cquad_load(W + (size_t)part * K + i);
            float32x4x2_t xv = sk__cquad_load(X_buf + (size_t)p_idx * K + i);
            float32x4_t wr = wv.val[0], wi = wv.val[1];
            float32x4_t xr = xv.val[0], xi = xv.val[1];
            av.val[0] = vaddq_f32(av.val[0],
                vfmaq_f32(vnegq_f32(vmulq_f32(wi, xi)), wr, xr));
            av.val[1] = vaddq_f32(av.val[1],
                vfmaq_f32(vmulq_f32(wi, xr), wr, xi));
            p_idx = (p_idx == 0) ? n_part - 1 : p_idx - 1;
        }
        sk__cquad_store(acc + i, av);
    }
    for (; i < K; ++i) {
        int p_idx = curr_p;
        for (part = 0; part < n_part; ++part) {
            const Complex *w = W + (size_t)part * K + i;
            const Complex *x = X_buf + (size_t)p_idx * K + i;
            float wr = w->r, wi = w->i, xr = x->r, xi = x->i;
            acc[i].r += fmaf(wr, xr, -(wi * xi));
            acc[i].i += fmaf(wr, xi,  (wi * xr));
            p_idx = (p_idx == 0) ? n_part - 1 : p_idx - 1;
        }
    }
}
#else
static inline void sk_cmac_fma_ring_f32(Complex *acc, const Complex *W,
                                         const Complex *X_buf, int curr_p,
                                         int n_part, int K) {
    sk_cmac_fma_ring_f32_scalar(acc, W, X_buf, curr_p, n_part, K);
}
#endif

/* ═══════════════════════════════ sk_cmac_f32 ═══════════════════════════════
 * acc[i] += w[i] * x[i], the UNFUSED complex multiply-accumulate: separate
 * multiply/subtract/add under -ffp-contract=off, no fmaf anywhere --
 * distinct from sk_cmac_fma_f32_scalar above (a different value-function; the two
 * round differently and are not interchangeable). The NEON path preserves
 * that same operation sequence so SIMD=1 and SIMD=0 stay byte-identical for
 * finite inputs. */

static inline void sk_cmac_f32_scalar(Complex *acc, const Complex *w,
                                       const Complex *x, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        float wr = w[i].r;
        float wi = w[i].i;
        float xr = x[i].r;
        float xi = x[i].i;
        float real = wr * xr - wi * xi;
        float imag = wr * xi + wi * xr;
        acc[i].r += real;
        acc[i].i += imag;
    }
}

#if SK_HAVE_NEON
static inline void sk_cmac_f32(Complex *acc, const Complex *w,
                                const Complex *x, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t wv = sk__cquad_load(w + i);
        float32x4x2_t xv = sk__cquad_load(x + i);
        float32x4x2_t av = sk__cquad_load(acc + i);
        float32x4x2_t out;
        float32x4_t real = vsubq_f32(
            vmulq_f32(wv.val[0], xv.val[0]),
            vmulq_f32(wv.val[1], xv.val[1]));
        float32x4_t imag = vaddq_f32(
            vmulq_f32(wv.val[0], xv.val[1]),
            vmulq_f32(wv.val[1], xv.val[0]));
        out.val[0] = vaddq_f32(av.val[0], real);
        out.val[1] = vaddq_f32(av.val[1], imag);
        sk__cquad_store(acc + i, out);
    }
    sk_cmac_f32_scalar(acc + i, w + i, x + i, n - i);
}
#else
static inline void sk_cmac_f32(Complex *acc, const Complex *w,
                                const Complex *x, int n) {
    sk_cmac_f32_scalar(acc, w, x, n);
}
#endif

#ifdef __cplusplus
}
#endif

#endif /* SIMD_KERNELS_H */
