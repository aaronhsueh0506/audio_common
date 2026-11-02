/** Shared scalar/NEON float32 DSP micro-kernels.
 * For finite inputs every NEON entry point matches its scalar twin exactly; compare/select kernels also match scalar NaN branching,
 * pure arithmetic matches NaN classification but not payload bits. Operation order is fixed and including units are compiled with
 * -ffp-contract=off: ordinary mul/add stay separate, explicit fmaf stays fused, no frecpe/frsqrte estimates, no reassociation;
 * the correctly rounded vsqrtq_f32 is used directly. SIMD_KERNELS_FORCE_SCALAR selects the scalar twins.
 * On 32-bit ARM the same holds for finite, non-subnormal values: the vector unit flushes subnormals (see the stand-ins below).
 * In-place use (out == input) is supported only where a kernel says so; partial overlap never is. */

#ifndef SIMD_KERNELS_H
#define SIMD_KERNELS_H

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>        /* memcpy: Complex <-> float[8] fallback in sk__cquad_* */

#include "ac_complex.h"    /* Complex { float r; float i; }, interleaved */

/* The one decision point for every NEON path in the libraries, the same on AArch64 and 32-bit ARM: NEON
 * with fused multiply-add. The kernels use vfmaq_f32 to match the scalar twins' explicit fmaf bit for bit
 * (vmlaq_f32 rounds twice), and the intrinsic does not exist without FMA. AArch64 always has it (the
 * __aarch64__ term only covers a compiler that omits the macro); 32-bit ARM has it with VFPv4 or ARMv8
 * (-mcpu=cortex-a32/a53/a55/a73, -mfpu=neon-vfpv4 / neon-fp-armv8). Plain -mfpu=neon keeps the scalar
 * twins. The AArch64-only intrinsics the kernels use are supplied by the stand-ins below on 32-bit ARM. */
#if defined(__ARM_NEON) && !defined(SIMD_KERNELS_FORCE_SCALAR) && \
    (defined(__aarch64__) || defined(__ARM_FEATURE_FMA))
#include <arm_neon.h>
#define SK_HAVE_NEON 1
#if defined(__aarch64__)
#define SK_NEON_BACKEND_NAME "aarch64-neon"
#else
#define SK_NEON_BACKEND_NAME "aarch32-neon"
#endif
#else
#define SK_HAVE_NEON 0
#define SK_NEON_BACKEND_NAME "scalar"
#endif

/* AArch32 stand-ins for the AArch64-only NEON intrinsics the shared kernels use. Each is built only from the ARMv7
 * NEON subset or from one scalar operation per lane, and reproduces the AArch64 instruction's result for finite inputs
 * bit for bit, the same contract the kernels state for their scalar twins:
 *   - division and square root are the correctly rounded per-lane scalar operations (vdiv.f32, vsqrt.f32), never the
 *     reciprocal / reciprocal-square-root estimate steps;
 *   - across-lane float adds use AArch64's order, (l0 + l1) + (l2 + l3);
 *   - min/max/add reductions of integers are order-insensitive.
 * A NaN result carries the NaN class but not necessarily the payload bits, as for the other kernels. On AArch32 the
 * vector instructions flush subnormal operands and results to zero unconditionally and return the default NaN, while
 * the scalar VFP path follows FPSCR; NEON and scalar twins therefore agree there only for finite, non-subnormal values.
 * On 32-bit ARM the AArch64 intrinsic names are aliased to the sk_a32_* stand-ins below, so kernels are written once
 * against the AArch64 names. SK_A32_SHIM_HOST_TEST additionally defines the stand-ins on AArch64 (without the aliases)
 * so test/simd_a32_shim_selftest.c can compare each one with the real intrinsic. */
#if defined(__ARM_NEON) && (!defined(__aarch64__) || defined(SK_A32_SHIM_HOST_TEST))
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>

static inline float32x4_t sk_a32_vdivq_f32(float32x4_t a, float32x4_t b) {
    float32x4_t r = vdupq_n_f32(0.0f);
    r = vsetq_lane_f32(vgetq_lane_f32(a, 0) / vgetq_lane_f32(b, 0), r, 0);
    r = vsetq_lane_f32(vgetq_lane_f32(a, 1) / vgetq_lane_f32(b, 1), r, 1);
    r = vsetq_lane_f32(vgetq_lane_f32(a, 2) / vgetq_lane_f32(b, 2), r, 2);
    r = vsetq_lane_f32(vgetq_lane_f32(a, 3) / vgetq_lane_f32(b, 3), r, 3);
    return r;
}

/* Needs -fno-math-errno (the projects' FP policy) for sqrtf to compile to the instruction. */
static inline float32x4_t sk_a32_vsqrtq_f32(float32x4_t a) {
    float32x4_t r = vdupq_n_f32(0.0f);
    r = vsetq_lane_f32(sqrtf(vgetq_lane_f32(a, 0)), r, 0);
    r = vsetq_lane_f32(sqrtf(vgetq_lane_f32(a, 1)), r, 1);
    r = vsetq_lane_f32(sqrtf(vgetq_lane_f32(a, 2)), r, 2);
    r = vsetq_lane_f32(sqrtf(vgetq_lane_f32(a, 3)), r, 3);
    return r;
}

/* Round toward minus infinity per lane (a libm call). A core with the ARMv8 floating-point unit (-mfpu=neon-fp-armv8, the
 * default of -mcpu=cortex-a32/a53/a55/a73) has the native vrndmq_f32, which the build keeps; an ARMv8 core given a VFPv4 FPU
 * (-mfpu=neon-vfpv4) does not, although its arm_neon.h declares it. */
static inline float32x4_t sk_a32_vrndmq_f32(float32x4_t a) {
    float32x4_t r = vdupq_n_f32(0.0f);
    r = vsetq_lane_f32(floorf(vgetq_lane_f32(a, 0)), r, 0);
    r = vsetq_lane_f32(floorf(vgetq_lane_f32(a, 1)), r, 1);
    r = vsetq_lane_f32(floorf(vgetq_lane_f32(a, 2)), r, 2);
    r = vsetq_lane_f32(floorf(vgetq_lane_f32(a, 3)), r, 3);
    return r;
}

static inline float sk_a32_vaddvq_f32(float32x4_t v) {
    float32x2_t p = vpadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(p, p), 0);
}

static inline float sk_a32_vmaxvq_f32(float32x4_t v) {
    float32x2_t p = vpmax_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpmax_f32(p, p), 0);
}

/* [a0 + a1, a2 + a3, b0 + b1, b2 + b3] */
static inline float32x4_t sk_a32_vpaddq_f32(float32x4_t a, float32x4_t b) {
    return vcombine_f32(vpadd_f32(vget_low_f32(a), vget_high_f32(a)),
                        vpadd_f32(vget_low_f32(b), vget_high_f32(b)));
}

static inline uint32_t sk_a32_vaddvq_u32(uint32x4_t v) {
    uint32x2_t p = vpadd_u32(vget_low_u32(v), vget_high_u32(v));
    return vget_lane_u32(vpadd_u32(p, p), 0);
}

static inline uint32_t sk_a32_vmaxvq_u32(uint32x4_t v) {
    uint32x2_t p = vpmax_u32(vget_low_u32(v), vget_high_u32(v));
    return vget_lane_u32(vpmax_u32(p, p), 0);
}

static inline uint32_t sk_a32_vminvq_u32(uint32x4_t v) {
    uint32x2_t p = vpmin_u32(vget_low_u32(v), vget_high_u32(v));
    return vget_lane_u32(vpmin_u32(p, p), 0);
}

static inline uint8_t sk_a32_vminvq_u8(uint8x16_t v) {
    uint8x8_t m = vmin_u8(vget_low_u8(v), vget_high_u8(v));
    m = vpmin_u8(m, m);
    m = vpmin_u8(m, m);
    m = vpmin_u8(m, m);
    return vget_lane_u8(m, 0);
}

/* Sum of the sixteen bytes, widened (the largest sum, 4080, needs 16 bits). */
static inline uint16_t sk_a32_vaddlvq_u8(uint8x16_t v) {
    return (uint16_t)sk_a32_vaddvq_u32(vpaddlq_u16(vpaddlq_u8(v)));
}

static inline uint32x4_t sk_a32_vceqzq_s32(int32x4_t a) {
    return vceqq_s32(a, vdupq_n_s32(0));
}

/* [a0, b0, a1, b1] and [a2, b2, a3, b3] */
static inline float32x4_t sk_a32_vzip1q_f32(float32x4_t a, float32x4_t b) {
    return vzipq_f32(a, b).val[0];
}
static inline float32x4_t sk_a32_vzip2q_f32(float32x4_t a, float32x4_t b) {
    return vzipq_f32(a, b).val[1];
}

#endif /* stand-ins */

#if SK_HAVE_NEON && !defined(__aarch64__)
#define vdivq_f32   sk_a32_vdivq_f32
#define vsqrtq_f32  sk_a32_vsqrtq_f32
#define vaddvq_f32  sk_a32_vaddvq_f32
#define vmaxvq_f32  sk_a32_vmaxvq_f32
#define vpaddq_f32  sk_a32_vpaddq_f32
#define vaddvq_u32  sk_a32_vaddvq_u32
#define vmaxvq_u32  sk_a32_vmaxvq_u32
#define vminvq_u32  sk_a32_vminvq_u32
#define vminvq_u8   sk_a32_vminvq_u8
#define vaddlvq_u8  sk_a32_vaddlvq_u8
#define vceqzq_s32  sk_a32_vceqzq_s32
#define vzip1q_f32  sk_a32_vzip1q_f32
#define vzip2q_f32  sk_a32_vzip2q_f32
#if !defined(__ARM_FEATURE_NUMERIC_MAXMIN) /* defined exactly when the ARMv8 FPU (with vrintm) is present */
#define vrndmq_f32  sk_a32_vrndmq_f32
#endif
#endif

/* Keyed on the architecture, not SK_HAVE_NEON, so the scalar sqrt helper stays sqrtf on AArch64 under SIMD_KERNELS_FORCE_SCALAR (as fast_math.h's FM_HAVE_HW_SQRT). */
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

/* SK_STATIC_ASSERT: compile-time assert for C99, C11 and C++11; the C99 typedef name is made unique per use. */
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

/* The NEON Complex kernels and sk__cquad_load/store need Complex to be exactly two contiguous, unpadded floats {r, i}. */
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

/* Per-element helpers shared by every scalar kernel and every NEON kernel's scalar tail. */

/* Copy of fast_math.h fast_sqrt(): sqrtf on ARM targets, bit-trick seed plus two Newton iterations elsewhere; keep in sync.
 * USE_STANDARD_MATH switches this to plain sqrtf, as in fast_math.h. */
#ifdef USE_STANDARD_MATH
static inline float sk__fast_sqrt_elem(float v) {
    return sqrtf(v);
}
#else
/* NaN fails `v > 0.0f`, so NaN returns 0.0f (as in fast_math.h and sk_fast_sqrt_f32). */
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
            x = 0.5f * (x + v / x);
            x = 0.5f * (x + v / x);
            return x;
        }
    }
#endif
}
#endif

/* sk_ema_f32: state[i] = alpha*state[i] + beta*x[i], two separately rounded multiplies and an add (needs -ffp-contract=off). */

static inline void sk_ema_f32_scalar(float *state, const float *x,
                                      float alpha, float beta, int n) {
    int i;
    for (i = 0; i < n; ++i) state[i] = alpha * state[i] + beta * x[i];
}

#if SK_HAVE_NEON && defined(__GNUC__) && !defined(__clang__)
/* gcc on 32-bit ARM reasons that the scalar tail after the NEON loop cannot reach iteration 2^30 and warns about it
 * (an error under -Werror); the tail is correct for every n. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"
#endif
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
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
#else
static inline void sk_ema_f32(float *state, const float *x,
                               float alpha, float beta, int n) {
    sk_ema_f32_scalar(state, x, alpha, beta, n);
}
#endif

/* Complex-quad NEON load/store: the supported way to move Complex arrays through NEON. GCC/Clang use a may_alias float view; other compilers a memcpy plus unzip/zip fallback. */
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
        return vuzpq_f32(lo, hi); /* val[0] = r's, val[1] = i's */
    }
#endif
}

static inline void sk__cquad_store(Complex *p, float32x4x2_t v) {
#if defined(__GNUC__) || defined(__clang__)
    typedef float sk__alias_float __attribute__((__may_alias__));
    vst2q_f32((sk__alias_float *)(void *)p, v);
#else
    float32x4x2_t z = vzipq_f32(v.val[0], v.val[1]);
    float scratch[8];
    vst1q_f32(scratch, z.val[0]);
    vst1q_f32(scratch + 4, z.val[1]);
    memcpy(p, scratch, sizeof(scratch));
#endif
}
#endif

/* sk_capply_gain_f32: out[i] = z[i] * g[i] (real gain on both components); out == z supported. */

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

/* sk_cadd_f32: out[i] = a[i] + b[i] (component-wise). */

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

/* sk_csub_f32: out[i] = a[i] - b[i] (component-wise); out == a supported. */

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

/* sk_sq_scale_f32: out[i] = (x[i]*x[i]) * scale. */

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

/* sk_min_f32: out[i] = (a[i] < b[i]) ? a[i] : b[i]. sk_clip_f32: low bound first, then high bound.
 * Both use compare plus select (not vminq/vmaxq, whose signed-zero tie-break differs from the scalar form). */

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
        uint32x4_t mask = vcltq_f32(av, bv);
        float32x4_t r = vbslq_f32(mask, av, bv);
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
        uint32x4_t lomask = vcltq_f32(v, lov);
        uint32x4_t himask = vcgtq_f32(v, hiv);
        float32x4_t r = vbslq_f32(lomask, lov, v);
        r = vbslq_f32(himask, hiv, r);
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

/* sk_fast_sqrt_f32: per-lane fast_sqrt() (vsqrtq_f32 on AArch64) with the `v > 0` guard as a final compare-and-select.
 * `v > 0`, not `v <= 0`, so a NaN lane selects 0.0f like the scalar helper. Discarded lanes may hold NaN; FP exceptions do not trap. */

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
            uint32x4_t ispos = vcgtq_f32(v, vdupq_n_f32(0.0f));
            float32x4_t r = vbslq_f32(ispos, xk, vdupq_n_f32(0.0f));
            vst1q_f32(out + i, r);
        }
    }
    for (; i < n; ++i) out[i] = sk__fast_sqrt_elem(x[i]);
}
#endif
#else
static inline void sk_fast_sqrt_f32(const float *x, float *out, int n) {
    sk_fast_sqrt_f32_scalar(x, out, n);
}
#endif

/* exp/log family: vector forms of fast_math.h; constants and operation order must stay in sync with it. The exp table index comes from an
 * internally clamped [-16, 16] value so gathers stay in bounds. Nothing is fused. out == x is supported unless a kernel says otherwise. */

#define SK_FM_LN2      0.693147180559945f
#define SK_FM_LOG10E   0.4342944819032518f
#define SK_FM_LN10     2.302585092994046f
#define SK_FM_EPSILON  1e-10f

/* Copy of fast_math.h FM_LOG_C1..C4 (degree-4 minimax ln(1+m); C2 stored positive and subtracted). */
#define SK_FM_LOG_C1   0.9972485899925232f
#define SK_FM_LOG_C2   0.46980342268943787f
#define SK_FM_LOG_C3   0.22263026237487793f
#define SK_FM_LOG_C4  -0.05692821741104126f

#define SK_EXP_TABLE_OFFSET 16
#define SK_EXP_TABLE_SIZE 33

/* Copy of fast_math.h exp_int_table (e^n for n = -16..16). */
static const float sk_exp_int_table[SK_EXP_TABLE_SIZE] = {
    1.1253517471925912e-07f,
    3.0590232050182579e-07f,
    8.3152871910356788e-07f,
    2.2603294069810542e-06f,
    6.1442123533282097e-06f,
    1.6701700790245659e-05f,
    4.5399929762484854e-05f,
    1.2340980408667956e-04f,
    3.3546262790251185e-04f,
    9.1188196555451624e-04f,
    2.4787521766663585e-03f,
    6.7379469990854670e-03f,
    1.8315638888734180e-02f,
    4.9787068367863944e-02f,
    1.3533528323661270e-01f,
    3.6787944117144233e-01f,
    1.0000000000000000e+00f,
    2.7182818284590452e+00f,
    7.3890560989306502e+00f,
    2.0085536923187668e+01f,
    5.4598150033144236e+01f,
    1.4841315910257660e+02f,
    4.0342879349273511e+02f,
    1.0966331584284585e+03f,
    2.9809579870417283e+03f,
    8.1030839275753840e+03f,
    2.2026465794806718e+04f,
    5.9874141715197819e+04f,
    1.6275479141900392e+05f,
    4.4241339200892050e+05f,
    1.2026042841647768e+06f,
    3.2690173724721107e+06f,
    8.8861105205078726e+06f
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
/* As fast_math.h fast_exp(), including the NaN-safe guard. */
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
/* As fast_math.h fast_exp_neg(). */
static inline float sk__fast_exp_neg_elem(float x) {
    if (x <= 0.0f) return 1.0f;
    if (x >= 16.0f) return 0.0f;
    return sk__fast_exp_elem(-x);
}
/* As fast_math.h fast_log(), including the NaN-safe guard. */
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
/* As fast_math.h fast_log10(). */
static inline float sk__fast_log10_elem(float x) {
    return sk__fast_log_elem(x) * SK_FM_LOG10E;
}
/* As fast_math.h exp1_approx() in its default branch order (the USE_OPTIMIZED_E1 order gives the same values). */
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
#endif

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
/* Vector core of sk_fast_exp_f32; also used by sk_fast_exp_neg_f32 (on -x) and the v > 1 branch of sk_exp1_approx_f32. Safe for any input (NaN, Inf). */
static inline float32x4_t sk__fast_exp_vec(float32x4_t x) {
    float32x4_t neg16 = vdupq_n_f32(-16.0f);
    float32x4_t pos16 = vdupq_n_f32(16.0f);
    /* The clamp only keeps the table index in bounds; the edge selects use the unclamped x. */
    float32x4_t xc = vminq_f32(vmaxq_f32(x, neg16), pos16);
    float32x4_t floor_xc = vrndmq_f32(xc);
    float32x4_t dx = vsubq_f32(xc, floor_xc);
    uint32x4_t adjmask = vcgtq_f32(dx, vdupq_n_f32(0.5f));
    float32x4_t dx_adj = vsubq_f32(dx, vdupq_n_f32(1.0f));
    float32x4_t x0f = vbslq_f32(adjmask, vaddq_f32(floor_xc, vdupq_n_f32(1.0f)), floor_xc);
    dx = vbslq_f32(adjmask, dx_adj, dx);

    {
        int32x4_t x0i = vaddq_s32(vcvtq_s32_f32(x0f), vdupq_n_s32(SK_EXP_TABLE_OFFSET));
        float table_lanes[4];
        /* No NEON gather: four scalar table reads; x0i is in [0, 32] for every input. */
        table_lanes[0] = sk_exp_int_table[vgetq_lane_s32(x0i, 0)];
        table_lanes[1] = sk_exp_int_table[vgetq_lane_s32(x0i, 1)];
        table_lanes[2] = sk_exp_int_table[vgetq_lane_s32(x0i, 2)];
        table_lanes[3] = sk_exp_int_table[vgetq_lane_s32(x0i, 3)];
        {
            float32x4_t exp_x0 = vld1q_f32(table_lanes);
            /* 1.0f + dx + 0.5f*dx2 + (1/6)*dx2*dx in the scalar operation order, no vfmaq_f32. */
            float32x4_t dx2 = vmulq_f32(dx, dx);
            float32x4_t term4 = vmulq_f32(vmulq_f32(vdupq_n_f32(1.0f / 6.0f), dx2), dx);
            float32x4_t term3 = vmulq_f32(vdupq_n_f32(0.5f), dx2);
            float32x4_t sum = vaddq_f32(vdupq_n_f32(1.0f), dx);
            sum = vaddq_f32(sum, term3);
            sum = vaddq_f32(sum, term4);
            {
                float32x4_t normal = vmulq_f32(exp_x0, sum);
                /* Edge selects on the unclamped x; lomask is applied last (highest priority) and the two masks never overlap. */
                uint32x4_t himask = vcgtq_f32(x, pos16);
                uint32x4_t lomask = vmvnq_u32(vcgeq_f32(x, neg16)); /* !(x>=-16): x < -16 or NaN */
                float32x4_t r = vbslq_f32(himask, vdupq_n_f32(8.8861105e+06f), normal);
                r = vbslq_f32(lomask, vdupq_n_f32(0.0f), r);
                return r;
            }
        }
    }
}

/* Supports out == x. */
static inline void sk_fast_exp_f32(const float *x, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(out + i, sk__fast_exp_vec(vld1q_f32(x + i)));
    for (; i < n; ++i) out[i] = sk__fast_exp_elem(x[i]);
}

/* Supports out == x. */
static inline void sk_fast_exp_neg_f32(const float *x, float *out, int n) {
    int i = 0;
    float32x4_t zero = vdupq_n_f32(0.0f), sixteen = vdupq_n_f32(16.0f), one = vdupq_n_f32(1.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        /* if (x<=0) return 1.0f; if (x>=16) return 0.0f; return fast_exp(-x); mask_le0 is applied last. */
        uint32x4_t mask_le0 = vcleq_f32(xv, zero);
        uint32x4_t mask_ge16 = vcgeq_f32(xv, sixteen);
        float32x4_t inner = sk__fast_exp_vec(vnegq_f32(xv));
        float32x4_t r = vbslq_f32(mask_ge16, zero, inner);
        r = vbslq_f32(mask_le0, one, r);
        vst1q_f32(out + i, r);
    }
    for (; i < n; ++i) out[i] = sk__fast_exp_neg_elem(x[i]);
}

/* Vector core of sk_fast_log_f32; also used by the v <= 1 branches of sk_exp1_approx_f32. Pure bit manipulation, computed for every lane. */
static inline float32x4_t sk__fast_log_vec(float32x4_t x) {
    uint32x4_t bits = vreinterpretq_u32_f32(x);
    int32x4_t exp_bits = vreinterpretq_s32_u32(vandq_u32(vshrq_n_u32(bits, 23), vdupq_n_u32(0xFFu)));
    int32x4_t E = vsubq_s32(exp_bits, vdupq_n_s32(127));
    /* Clear the exponent and sign fields and force the exponent to 127. */
    uint32x4_t mantissa_mask = vdupq_n_u32(0x007FFFFFu);
    uint32x4_t bias127 = vdupq_n_u32(0x3F800000u);
    uint32x4_t new_bits = vorrq_u32(vandq_u32(bits, mantissa_mask), bias127);
    float32x4_t mant_f = vreinterpretq_f32_u32(new_bits);
    float32x4_t m = vsubq_f32(mant_f, vdupq_n_f32(1.0f));
    /* Degree-4 minimax Horner sequence with separately rounded multiply and add, no vfmaq_f32. */
    float32x4_t poly = vdupq_n_f32(SK_FM_LOG_C4);
    poly = vaddq_f32(vmulq_f32(poly, m), vdupq_n_f32(SK_FM_LOG_C3));
    poly = vsubq_f32(vmulq_f32(poly, m), vdupq_n_f32(SK_FM_LOG_C2));
    poly = vaddq_f32(vmulq_f32(poly, m), vdupq_n_f32(SK_FM_LOG_C1));
    float32x4_t ln_1_m = vmulq_f32(poly, m);
    {
        float32x4_t Ef = vcvtq_f32_s32(E);
        float32x4_t result = vaddq_f32(vmulq_f32(Ef, vdupq_n_f32(SK_FM_LN2)), ln_1_m);
        uint32x4_t badmask = vmvnq_u32(vcgtq_f32(x, vdupq_n_f32(0.0f)));
        return vbslq_f32(badmask, vdupq_n_f32(-1e10f), result);
    }
}

/* Supports out == x. */
static inline void sk_fast_log_f32(const float *x, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(out + i, sk__fast_log_vec(vld1q_f32(x + i)));
    for (; i < n; ++i) out[i] = sk__fast_log_elem(x[i]);
}

/* Supports out == x. Uniform four-bin regimes skip the unused E1 branch; mixed regimes keep both branches in scalar comparison order. */
static inline void sk_exp1_approx_f32(const float *x, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t eps = vdupq_n_f32(SK_FM_EPSILON);
        /* if (v<=EPS) v=EPS; NaN passes through. */
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
            /* NaN comparisons are false, so NaN follows branch 3. */
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
/* USE_STANDARD_MATH or no NEON: delegate to the per-element scalar entry points (out == x is equally safe). */
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
#endif

/* sk_noise_update_f32: noise-PSD tracking with a speech-presence-dependent smoothing factor, per bin:
 *   su = spp[i] * bb_scale; a = alpha_d + (1-alpha_d) * su; noise[i] = a*noise[i] + (1-a)*power[i].
 * Separate multiply/add roundings; no FMA. */

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


/* sk_wola_accumulate_f32: acc[i] += x[i] * w[i], unit stride; the three arrays must not alias. Separate vmulq/vaddq, not vfmaq_f32. */

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

/* sk_clip_scale_to_s16: float in [-1, 1] -> int16 PCM (clip to +-1, scale by 32767, truncate toward zero).
 *   sk__clip_scale_to_s16_elem: one sample (scalar twin); sk_clip_scale_to_s16_arr: a planar buffer;
 *   sk_clip_scale_to_s16: one NEON quad -> int16x4_t (NEON only, no _scalar twin; callers guard with SK_HAVE_NEON).
 * Clipping uses compare plus select as sk_clip_f32 (signed zeros convert identically); vcvtq_s32_f32 truncates like a C cast;
 * vqmovn_s32 never saturates after the clip. NaN is not handled and reaches an undefined float -> int conversion: callers must exclude it. */

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
    v = vbslq_f32(vcgtq_f32(v, hi), hi, v);
    v = vbslq_f32(vcltq_f32(v, lo), lo, v);
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

/* sk_cmag_f32: out[i] = x[i].r*x[i].r + x[i].i*x[i].i + eps, the squared magnitude plus a floor (no square root).
 * Separate vmulq/vaddq, never vfmaq_f32; de-interleaves through sk__cquad_load (no Complex* -> float* cast).
 * Differs from sk_cmag2_np_f32 in simd_aec.h, which uses the scaled-hypot form. */

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
        float32x4x2_t ri = sk__cquad_load(x + i);
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

/* sk_linear_to_db_f32: out[i] = 10 * log10f(x[i]), power-domain dB. Scalar only: libm log10f is the only exact choice. */

static inline void sk_linear_to_db_f32(float *out, const float *x, int n) {
    int i;
    for (i = 0; i < n; ++i) out[i] = 10.0f * log10f(x[i]);
}

/* sk_asym_ema_f32: one-pole EMA with a coefficient per direction, updating `s` in place:
 *   s[i] = (x[i] < s[i]) ? a_down*s[i] + (1-a_down)*x[i] : a_up*s[i] + (1-a_up)*x[i].
 * The comparison is strict `<` and selects a_down; an implementation that selects on `>` differs at ties. Separate vmulq/vaddq. */

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
        /* x < s ? falling : rising, compare plus select */
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

/* sk_scale_f32: out[i] = x[i] * g, one broadcast gain; out == x supported. */

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

/* sk_argmax_sq_f32: first index of the largest x[i]*x[i] (returns 0 for n < 2).
 * NEON: pass 1 max-reduces the squares over four lane accumulators seeded at +0.0f; pass 2 finds the first square equal to the maximum.
 * A non-finite peak (NaN or overflow to +inf) falls back to the scalar walk. */

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
    /* The ragged tail folds in with the walk's own `>` test, so a NaN square stays out of the peak. */
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
    /* Non-finite peak: also closes the function (a finite peak is always found by pass 2). */
    return sk_argmax_sq_f32_scalar(h, n);
}
#else
static inline int sk_argmax_sq_f32(const float *h, int n) {
    return sk_argmax_sq_f32_scalar(h, n);
}
#endif

/* sk_argmax_f32: first index of the maximum raw value (not squared). NEON: two passes, only when n >= 8 and x[0] is not NaN;
 * pass 1 seeds two 4-lane accumulators by loading x[0..3], pass 2 finds the first element equal to the maximum.
 * A NaN-seeded array or a non-finite maximum uses the scalar walk. */

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

/* sk_argmax_i32: first index of the maximum of an int32 array; scalar only; requires n >= 1. */

static inline int sk_argmax_i32(const int *a, int n) {
    int best = 0, i;
    int best_v = a[0];
    for (i = 1; i < n; ++i) {
        if (a[i] > best_v) { best_v = a[i]; best = i; }
    }
    return best;
}

/* sk_cmac_fma_f32_scalar: acc[k].r += fmaf(wr, xr, -(wi * xi)); acc[k].i += fmaf(wr, xi, wi * xr).
 * Not interchangeable with the unfused sk_cmac_f32. */

static inline void sk_cmac_fma_f32_scalar(Complex *acc, const Complex *w,
                                           const Complex *x, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        float wr = w[i].r, wi = w[i].i, xr = x[i].r, xi = x[i].i;
        acc[i].r += fmaf(wr, xr, -(wi * xi));
        acc[i].i += fmaf(wr, xi,  (wi * xr));
    }
}

/* sk_cmac_fma_ring_f32: acc[k] += w_part[k] * x_part[k] for part = 0..n_part-1 in that order, where w_part = W + part*K and x_part is
 * X_buf's row (curr_p - part) mod n_part; the sk_cmac_fma_f32_scalar form per partition. NEON keeps the accumulator in registers across partitions. */

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

/* sk_cmac_f32: acc[i] += w[i] * x[i], unfused complex multiply-accumulate (no fmaf); not interchangeable with sk_cmac_fma_f32_scalar. */

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
