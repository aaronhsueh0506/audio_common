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
 * stays separate. Reciprocal/sqrt estimates and reassociation are forbidden.
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
#include <string.h>        /* memcpy -- the legal Complex<->float[8] byte
                             * reinterpretation used by sk__cquad_load/
                             * sk__cquad_store below (see that comment) */

#include "fft_wrapper.h"   /* Complex { float r; float i; } (interleaved AoS) */

#if defined(__ARM_NEON) && defined(__aarch64__) && !defined(SIMD_KERNELS_FORCE_SCALAR)
#include <arm_neon.h>
#define SK_HAVE_NEON 1
#else
#define SK_HAVE_NEON 0
#endif

/* SK_STATIC_ASSERT -- portable compile-time assert, C99/C11/C++11 alike.
 *
 * This header documents itself as C99-compatible (the `extern "C"` guard
 * just below is for C++ consumers that still build it in C++ mode, not just
 * C11+). The bare C11 keyword `_Static_assert` is NOT available in strict
 * C99 (it is a C11 extension -- diagnosed under `-std=c99 -pedantic-errors`)
 * and, spelled with the leading underscore, is ALSO rejected under
 * `-std=c++17 -pedantic-errors` (C++'s spelling is the lowercase
 * `static_assert`, no underscore -- the identifier `_Static_assert` has no
 * special meaning to a C++ compiler at all). Route to whichever spelling the
 * including TU's language mode actually provides, with a strict-C99
 * fallback that gets equivalent enforcement strength (a hard compile
 * failure, with the condition spelled out via the array-size violation)
 * from a language feature C99 does have: a `typedef` naming a `char` array
 * whose size is `-1` (illegal) when `cond` is false. The typedef's name
 * must be unique per use site or two calls on different lines collide as a
 * redefinition, hence the __LINE__-based two-level token paste. */
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

/* The sk_c*_f32 NEON kernels below (kernels 9/10 here, and the wider family
 * in AEC/c_impl's aec_simd_kernels.h, which #includes this file and reuses
 * sk__cquad_load/sk__cquad_store directly rather than redefining them) all
 * assume `Complex` is exactly two contiguous, unpadded floats {r, i} --
 * that's the layout every sk__cquad_load/sk__cquad_store quad-load/store
 * below silently depends on (see the "Complex-quad NEON load/store" section
 * further down). Pin the assumption here, once, next to the `Complex` include,
 * so a future ABI-changing edit to that struct (an added field, reordered
 * members, explicit padding/alignment) fails to COMPILE instead of silently
 * reintroducing a misaligned/wrong-stride NEON access everywhere this header
 * is included. */
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

/* fast_math.h fast_sqrt(), replicated verbatim (bit-trick seed + 2 Newton
 * iterations) rather than #include-d, per the header-comment rationale
 * above. Keep in sync with fast_math.h if that implementation ever moves.
 *
 * USE_STANDARD_MATH: fast_math.h swaps fast_sqrt for plain sqrtf under this
 * flag (debug/parity builds). This kernel follows the SAME flag so a call
 * site converted from `fast_sqrt(...)` to sk_fast_sqrt_f32 behaves
 * identically under both build modes (sqrtf per lane == vsqrtq_f32, IEEE
 * correctly rounded, including sqrtf(-0)=-0 and NaN for negatives). */
#ifdef USE_STANDARD_MATH
static inline float sk__fast_sqrt_elem(float v) {
    return sqrtf(v);
}
#else
/* NaN: fails `v > 0.0f` (every ordered IEEE comparison with NaN is false),
 * so the negated guard returns 0.0f for NaN too -- matching fast_math.h's
 * fast_sqrt fix and this kernel's NEON twin below (sk_fast_sqrt_f32's NaN
 * lane selects the same 0.0f). Identical to `v <= 0.0f` for all finite v. */
static inline float sk__fast_sqrt_elem(float v) {
    if (!(v > 0.0f)) return 0.0f;
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
}
#endif /* USE_STANDARD_MATH */

/* ═══════════════════════════════ kernel 4 ══════════════════════════════════
 * sk_ema_f32 — state[i] = alpha*state[i] + beta*x[i]; verbatim shape of the
 * pbfdkf.c far-power EMA (`p->power[k] = a*p->power[k] + b*cmag2_np(...)`),
 * NOT collapsed to fmaf: the source computes this as two separate rounded
 * multiplies then a separate rounded add (no fmaf call at this line), so
 * this kernel must NOT fuse either — see the FMA-discipline note above.
 * Requires the including TU to build with -ffp-contract=off (verified: at
 * -O2 without it, clang auto-fuses this exact shape into fmla). */

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

/* ═══════════════════════════════ kernel 9 ══════════════════════════════════
 * sk_capply_gain_f32 — out[i] = z[i] * g[i] (real gain applied to both
 * components). Supports out == z (in-place): each iteration fully loads
 * before it stores, and iterations never revisit an earlier index, so
 * aliasing at the SAME pointer is safe (no partial-overlap aliasing is
 * supported/needed beyond that). */

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

/* ═══════════════════════════════ kernel 10 ═════════════════════════════════
 * sk_cadd_f32 — out[i] = a[i] + b[i] (component-wise complex add). */

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

/* ═══════════════════════════════ kernel 11 ═════════════════════════════════
 * sk_sq_scale_f32 — out[i] = (x[i]*x[i]) * scale (e.g. aec.c mean_sq's
 * `scratch[i] = x[i]*x[i]` per-element step, generalized with a scale).
 * Two separate multiplies, no add between them, so there is nothing for
 * -ffp-contract to fuse either way — kept as explicit vmulq/vmulq for
 * clarity and symmetry with the other "no FMA" kernels. */

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

/* ═══════════════════════════════ kernel 12 ═════════════════════════════════
 * sk_min_f32 — out[i] = min_f(a[i], b[i]) = (a[i] < b[i]) ? a[i] : b[i]
 *   (fast_math.h min_f's exact form).
 * sk_clip_f32 — clip_f's exact branch order (fast_math.h:268-272): check the
 *   LOW bound first, then the HIGH bound (equivalent to max(lo, min(hi, v))
 *   for lo <= v, non-NaN inputs, but implemented as compare+select, in that
 *   order, to avoid the vminq/vmaxq signed-zero tie-break mismatch --- see
 *   the header-comment note above).
 * Both use vcltq_f32/vcgtq_f32 + vbslq_f32 rather than vminq_f32/vmaxq_f32
 * for exactly that reason. */

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

/* ═══════════════════════════════ kernel 15 ═════════════════════════════════
 * sk_fast_sqrt_f32 — per-lane replica of fast_math.h fast_sqrt() (bit-trick
 * seed + 2 Newton iterations). The `v>0` guard is applied as a final
 * vcgtq_f32+vbslq_f32 select (matching cabs_np's "compute unconditionally,
 * then select" pattern) rather than a branch: the bit-trick/Newton steps
 * are well-defined (finite, no trap) for non-positive/negative/NaN `v` too —
 * their result is simply discarded by the select for those lanes, same
 * final bits as the scalar early return.
 *
 * NaN lanes: the select predicate is deliberately `ispos = v > 0` (selecting
 * the Newton-Raphson result `xk` when true, 0.0f otherwise) rather than
 * `nonpos = v <= 0` (selecting 0.0f when true, `xk` otherwise). Both forms
 * agree for every finite v (exactly one of `v>0`/`v<=0` is true), but they
 * disagree on NaN: vcgtq_f32/vcleq_f32 are ordered compares, so BOTH
 * `v>0` and `v<=0` are false for a NaN lane. With the nonpos form that
 * false steers the select to the `xk` (garbage) branch; with the ispos form
 * here, that same false steers it to the 0.0f branch -- matching the scalar
 * sk__fast_sqrt_elem's `!(v>0.0f)` guard (and fast_math.h's fast_sqrt)
 * bit-for-bit, including on NaN input. */

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
        uint32x4_t bits = vreinterpretq_u32_f32(v);
        uint32x4_t seed_bits = vaddq_u32(vshrq_n_u32(bits, 1),
                                         vdupq_n_u32(0x1FC00000u));
        float32x4_t xk = vreinterpretq_f32_u32(seed_bits);
        xk = vmulq_f32(vdupq_n_f32(0.5f), vaddq_f32(xk, vdivq_f32(v, xk)));
        xk = vmulq_f32(vdupq_n_f32(0.5f), vaddq_f32(xk, vdivq_f32(v, xk)));
        {
            /* v > 0 (NOT v <= 0): an ordered compare, false for NaN lanes
             * too, so NaN selects the 0.0f branch below -- see the
             * kernel-15 header comment for why this must be the ">0"
             * form and not the "<=0" form used before the NaN-safety fix. */
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

/* ═══════════════════ kernels 23-27: exp/log family ═══════════════════════
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

/* Private mirror of fast_math.h's FloatBits union (own SK-prefixed name so a
 * TU that includes BOTH headers, e.g. simd_selftest.c, never sees a
 * conflicting typedef). Relies on the mantissa-then-exponent-then-sign
 * bitfield packing every GCC/Clang target this project builds for actually
 * uses for a little-endian `unsigned int`-backed bitfield — the SAME
 * assumption fast_math.h's own fast_log() already makes; this is not a new
 * assumption introduced here. */
typedef union {
    float f;
    uint32_t i;
    struct { uint32_t mantissa : 23; uint32_t exponent : 8; uint32_t sign : 1; } parts;
} SkFloatBits;

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
/* Verbatim from fast_math.h's fast_exp() (see that function's own comment
 * for the NaN-safety rationale of the `!(x >= -16.0f)` guard form). */
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
/* Verbatim from fast_math.h's fast_log() (see that function's own comment
 * for the NaN-safety rationale of the `!(x > 0.0f)` guard form). */
static inline float sk__fast_log_elem(float x) {
    if (!(x > 0.0f)) return -1e10f;
    SkFloatBits fb;
    fb.f = x;
    int E = (int)fb.parts.exponent - 127;
    fb.parts.exponent = 127;
    float m = fb.f - 1.0f;
    float m2 = m * m;
    float m3 = m2 * m;
    float ln_1_m = m - 0.5f * m2 + (1.0f / 3.0f) * m3 - 0.25f * m2 * m2;
    return (float)E * SK_FM_LN2 + ln_1_m;
}
/* Verbatim from fast_math.h's fast_log10(). */
static inline float sk__fast_log10_elem(float x) {
    return sk__fast_log_elem(x) * SK_FM_LOG10E;
}
/* Verbatim from fast_math.h's exp1_approx() DEFAULT (non-USE_OPTIMIZED_E1)
 * branch order. fast_math.h's USE_OPTIMIZED_E1 variant reorders which check
 * runs first (v>1.0 first, single fast_log10 call reused for the v<=1.0
 * cases) but computes the IDENTICAL formula for every input with fast_log10
 * called exactly once either way — the two source variants are provably
 * bit-identical for every v, so replicating only this one branch order
 * covers both fast_math.h build modes (verified by test_exp1_approx()
 * against fast_math.h's actual exp1_approx() regardless of whether that TU
 * defines USE_OPTIMIZED_E1). */
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
static inline void sk_fast_log10_f32_scalar(const float *x, float *out, int n) {
    int i;
    for (i = 0; i < n; ++i) out[i] = sk__fast_log10_elem(x[i]);
}
static inline void sk_exp1_approx_f32_scalar(const float *x, float *out, int n) {
    int i;
    for (i = 0; i < n; ++i) out[i] = sk__exp1_approx_elem(x[i]);
}

#if SK_HAVE_NEON && !defined(USE_STANDARD_MATH)
/* Shared vector core for kernel 23 (sk_fast_exp_f32) — also reused by
 * kernel 24 (fast_exp_neg calls this on -x, mirroring fast_math.h's
 * fast_exp_neg calling fast_exp(-x)) and kernel 27 (exp1_approx's v>1.0
 * branch). Domain-safe for ANY input (NaN, ±Inf, arbitrary finite
 * magnitude) — see the memory-safety note in this section's header
 * comment. */
static inline float32x4_t sk__fast_exp_vec(float32x4_t x) {
    float32x4_t neg16 = vdupq_n_f32(-16.0f);
    float32x4_t pos16 = vdupq_n_f32(16.0f);
    /* Range-clamp ONLY for the table-index computation below (memory
     * safety) -- the final domain-edge select at the bottom uses the REAL,
     * unclamped x, so this clamp never changes the mathematical result for
     * any lane, only what "garbage" a masked-out lane's discarded
     * intermediate looks like. */
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
        /* No native NEON gather on this target -- 4 scalar table reads,
         * per this section's header comment. x0i is always in [0,32]
         * (in-bounds) for every possible input, including NaN. */
        table_lanes[0] = sk_exp_int_table[vgetq_lane_s32(x0i, 0)];
        table_lanes[1] = sk_exp_int_table[vgetq_lane_s32(x0i, 1)];
        table_lanes[2] = sk_exp_int_table[vgetq_lane_s32(x0i, 2)];
        table_lanes[3] = sk_exp_int_table[vgetq_lane_s32(x0i, 3)];
        {
            float32x4_t exp_x0 = vld1q_f32(table_lanes);
            /* 1.0f + dx + 0.5f*dx2 + (1/6)*dx2*dx, same left-to-right
             * separate-rounding op sequence as sk__fast_exp_elem -- no
             * vfmaq_f32 anywhere (see this section's FMA-discipline note). */
            float32x4_t dx2 = vmulq_f32(dx, dx);
            float32x4_t term4 = vmulq_f32(vmulq_f32(vdupq_n_f32(1.0f / 6.0f), dx2), dx);
            float32x4_t term3 = vmulq_f32(vdupq_n_f32(0.5f), dx2);
            float32x4_t sum = vaddq_f32(vdupq_n_f32(1.0f), dx);
            sum = vaddq_f32(sum, term3);
            sum = vaddq_f32(sum, term4);
            {
                float32x4_t normal = vmulq_f32(exp_x0, sum);
                /* Domain-edge select on the REAL x, priority-ordered to
                 * match the scalar guard sequence: lomask (checked first
                 * in scalar, so applied LAST/highest-priority here) can
                 * never overlap himask for any real input, but this order
                 * keeps the blend's priority explicit regardless. */
                uint32x4_t himask = vcgtq_f32(x, pos16);            /* x > 16 (finite only) */
                uint32x4_t lomask = vmvnq_u32(vcgeq_f32(x, neg16)); /* !(x>=-16): x<-16 or NaN */
                float32x4_t r = vbslq_f32(himask, vdupq_n_f32(8.8861105e+06f), normal);
                r = vbslq_f32(lomask, vdupq_n_f32(0.0f), r);
                return r;
            }
        }
    }
}

/* Supports out == x (in-place): each 4-lane block is fully loaded
 * (vld1q_f32) before it is stored (vst1q_f32), and no iteration reads a
 * lane an earlier iteration wrote -- same shape as sk_capply_gain_f32's
 * (kernel 9) documented out==z contract. See the section-header comment
 * above for the full writeup covering this kernel family. */
static inline void sk_fast_exp_f32(const float *x, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(out + i, sk__fast_exp_vec(vld1q_f32(x + i)));
    for (; i < n; ++i) out[i] = sk__fast_exp_elem(x[i]);
}

/* Supports out == x (in-place) -- same read-before-write-per-block shape as
 * sk_fast_exp_f32 above (sk__fast_exp_vec operates entirely on the
 * already-loaded `xv` register, not on memory). */
static inline void sk_fast_exp_neg_f32(const float *x, float *out, int n) {
    int i = 0;
    float32x4_t zero = vdupq_n_f32(0.0f), sixteen = vdupq_n_f32(16.0f), one = vdupq_n_f32(1.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        /* if (x<=0) return 1.0f; if (x>=16) return 0.0f; return fast_exp(-x);
         * -- mask priority mirrors the scalar if/else-if chain (mask_le0
         * checked first in scalar, so applied LAST here). */
        uint32x4_t mask_le0 = vcleq_f32(xv, zero);
        uint32x4_t mask_ge16 = vcgeq_f32(xv, sixteen);
        float32x4_t inner = sk__fast_exp_vec(vnegq_f32(xv));
        float32x4_t r = vbslq_f32(mask_ge16, zero, inner);
        r = vbslq_f32(mask_le0, one, r);
        vst1q_f32(out + i, r);
    }
    for (; i < n; ++i) out[i] = sk__fast_exp_neg_elem(x[i]);
}

/* Shared vector core for kernel 25 (sk_fast_log_f32) -- also reused by
 * kernel 26 (fast_log10 = fast_log*const) and kernel 27 (exp1_approx's
 * v<=1.0 branches). Pure bit manipulation (no memory access, no gather), so
 * unlike sk__fast_exp_vec there is no memory-safety concern computing this
 * unconditionally for every lane including out-of-domain ones -- exactly
 * the exponent-extract-and-rebias trick fast_math.h's fast_log() does,
 * replicated lane-for-lane. */
static inline float32x4_t sk__fast_log_vec(float32x4_t x) {
    uint32x4_t bits = vreinterpretq_u32_f32(x);
    int32x4_t exp_bits = vreinterpretq_s32_u32(vandq_u32(vshrq_n_u32(bits, 23), vdupq_n_u32(0xFFu)));
    int32x4_t E = vsubq_s32(exp_bits, vdupq_n_s32(127));
    /* Clear the exponent field, force it to 127 (bias) -- fb.parts.exponent
     * = 127 in the scalar union version; mant_sign_mask keeps sign+mantissa,
     * bias127 ORs in exponent=127 (0x3F800000 == 127<<23). */
    uint32x4_t mant_sign_mask = vdupq_n_u32(0x807FFFFFu);
    uint32x4_t bias127 = vdupq_n_u32(0x3F800000u);
    uint32x4_t new_bits = vorrq_u32(vandq_u32(bits, mant_sign_mask), bias127);
    float32x4_t mant_f = vreinterpretq_f32_u32(new_bits);
    float32x4_t m = vsubq_f32(mant_f, vdupq_n_f32(1.0f));
    float32x4_t m2 = vmulq_f32(m, m);
    float32x4_t m3 = vmulq_f32(m2, m);
    /* m - 0.5f*m2 + (1/3)*m3 - 0.25f*m2*m2, same left-to-right separate-
     * rounding op sequence as sk__fast_log_elem -- no vfmaq_f32 anywhere. */
    float32x4_t t1 = vsubq_f32(m, vmulq_f32(vdupq_n_f32(0.5f), m2));
    float32x4_t t2 = vaddq_f32(t1, vmulq_f32(vdupq_n_f32(1.0f / 3.0f), m3));
    float32x4_t ln_1_m = vsubq_f32(t2, vmulq_f32(vdupq_n_f32(0.25f), vmulq_f32(m2, m2)));
    {
        float32x4_t Ef = vcvtq_f32_s32(E);
        float32x4_t result = vaddq_f32(vmulq_f32(Ef, vdupq_n_f32(SK_FM_LN2)), ln_1_m);
        uint32x4_t badmask = vmvnq_u32(vcgtq_f32(x, vdupq_n_f32(0.0f))); /* !(x>0): x<=0 or NaN */
        return vbslq_f32(badmask, vdupq_n_f32(-1e10f), result);
    }
}

/* Supports out == x (in-place): sk__fast_log_vec is pure bit manipulation on
 * an already-loaded register (no memory access besides the one load/store
 * per block); same read-before-write-per-block shape as sk_fast_exp_f32
 * above. */
static inline void sk_fast_log_f32(const float *x, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(out + i, sk__fast_log_vec(vld1q_f32(x + i)));
    for (; i < n; ++i) out[i] = sk__fast_log_elem(x[i]);
}

static inline void sk_fast_log10_f32(const float *x, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t lg = sk__fast_log_vec(vld1q_f32(x + i));
        vst1q_f32(out + i, vmulq_f32(lg, vdupq_n_f32(SK_FM_LOG10E)));
    }
    for (; i < n; ++i) out[i] = sk__fast_log10_elem(x[i]);
}

/* Supports out == x (in-place): the whole 4-lane block is loaded once at
 * the top (`v = vld1q_f32(x + i)`) and only written once at the bottom
 * (`vst1q_f32(out + i, r)`), with everything in between computed from
 * registers; same shape as sk_fast_exp_f32/sk_fast_log_f32 above. */
static inline void sk_exp1_approx_f32(const float *x, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t eps = vdupq_n_f32(SK_FM_EPSILON);
        /* if (v<=EPS) v=EPS; -- false for NaN (matches scalar: NaN stays
         * NaN through this line, same as sk__exp1_approx_elem). */
        uint32x4_t clampmask = vcleq_f32(v, eps);
        float32x4_t vc = vbslq_f32(clampmask, eps, v);
        uint32x4_t mask_lt01 = vcltq_f32(vc, vdupq_n_f32(0.1f));
        uint32x4_t mask_le1  = vcleq_f32(vc, vdupq_n_f32(1.0f));

        float32x4_t log10v = vmulq_f32(sk__fast_log_vec(vc), vdupq_n_f32(SK_FM_LOG10E));

        /* -2.31f*log10v - 0.6f (branch1), -1.544f*log10v + 0.166f (branch2),
         * fast_exp((-0.52f*vc - 0.26f) * LN10) (branch3) -- each exactly the
         * scalar op order, no fusion. */
        float32x4_t branch1 = vsubq_f32(vmulq_f32(vdupq_n_f32(-2.31f), log10v), vdupq_n_f32(0.6f));
        float32x4_t branch2 = vaddq_f32(vmulq_f32(vdupq_n_f32(-1.544f), log10v), vdupq_n_f32(0.166f));
        float32x4_t exparg = vmulq_f32(
            vsubq_f32(vmulq_f32(vdupq_n_f32(-0.52f), vc), vdupq_n_f32(0.26f)),
            vdupq_n_f32(SK_FM_LN10));
        float32x4_t branch3 = sk__fast_exp_vec(exparg);

        {
            /* if/else-if priority: mask_le1 is a superset of mask_lt01 (any
             * v<0.1 also satisfies v<=1.0), so applying branch2 for the
             * whole mask_le1 region THEN overriding the narrower mask_lt01
             * region with branch1 reproduces if(v<0.1)/else-if(v<=1.0)/else
             * exactly. */
            float32x4_t r = branch3;
            r = vbslq_f32(mask_le1, branch2, r);
            r = vbslq_f32(mask_lt01, branch1, r);
            vst1q_f32(out + i, r);
        }
    }
    for (; i < n; ++i) out[i] = sk__exp1_approx_elem(x[i]);
}
#else
/* USE_STANDARD_MATH or non-NEON target: these delegate straight to the
 * *_scalar() entry points above, which are per-element `out[i]=f(x[i])`
 * with no cross-index dependency -- out == x (in-place) is safe here for
 * the identical reason as the NEON bodies above (fast_exp/fast_exp_neg/
 * fast_log/exp1_approx; fast_log10 is NOT contractually covered, see the
 * section-header comment). */
static inline void sk_fast_exp_f32(const float *x, float *out, int n) {
    sk_fast_exp_f32_scalar(x, out, n);
}
static inline void sk_fast_exp_neg_f32(const float *x, float *out, int n) {
    sk_fast_exp_neg_f32_scalar(x, out, n);
}
static inline void sk_fast_log_f32(const float *x, float *out, int n) {
    sk_fast_log_f32_scalar(x, out, n);
}
static inline void sk_fast_log10_f32(const float *x, float *out, int n) {
    sk_fast_log10_f32_scalar(x, out, n);
}
static inline void sk_exp1_approx_f32(const float *x, float *out, int n) {
    sk_exp1_approx_f32_scalar(x, out, n);
}
#endif /* SK_HAVE_NEON && !USE_STANDARD_MATH */

/* ═══════════════════════════════ kernel 28 ═════════════════════════════════
 * Fused MCRA noise update with a per-bin alpha:
 *   su = spp[i] * bb_scale
 *   a  = alpha_d + (1-alpha_d) * su
 *   noise[i] = a*noise[i] + (1-a)*power[i]
 * The source uses separate rounded multiply/add operations; do not use FMA. */

static inline void sk_mcra_noise_update_f32_scalar(float *noise_psd,
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
static inline void sk_mcra_noise_update_f32(float *noise_psd,
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
static inline void sk_mcra_noise_update_f32(float *noise_psd,
                                             const float *spp,
                                             const float *power,
                                             float alpha_d,
                                             float bb_scale,
                                             int n) {
    sk_mcra_noise_update_f32_scalar(noise_psd, spp, power, alpha_d, bb_scale, n);
}
#endif


/* ═══════════════════════════════ kernel 29 ═════════════════════════════════
 * sk_wola_accumulate_f32 — acc[i] += x[i] * w[i]
 *
 * The windowed overlap-add accumulate every AEC-side pipeline runs once per
 * hop over a whole FFT frame: the synthesis window applied to the inverse
 * transform, summed into the running OLA tail. Unit stride, no reduction, no
 * aliasing between the three arrays.
 *
 * SEPARATE vmulq_f32 + vaddq_f32, deliberately NOT vfmaq_f32. Every scalar
 * reference this kernel replaces spells the operation as a multiply followed
 * by an add, which under this project's mandatory -ffp-contract=off rounds
 * TWICE; a fused multiply-add rounds once and would differ in the last bit.
 * That is a per-kernel rule, not a project-wide one -- kernels whose own
 * scalar reference calls fmaf() do fuse, and must (see aec_simd_kernels.h).
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
        /* mul then add, never vfmaq_f32 -- see the note above. */
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

/* ═══════════════════════════════ kernel 30 ═════════════════════════════════
 * sk_clip_scale_to_s16 — float in [-1, 1] -> int16 PCM
 *
 * The last step before a float mixing chain hands samples to a PCM sink:
 * clip to +-1, scale by 32767, convert. Three entry points, because callers
 * need it in three shapes:
 *
 *   sk__clip_scale_to_s16_elem()   one sample; the scalar reference
 *   sk_clip_scale_to_s16_arr()     a whole planar buffer
 *   sk_clip_scale_to_s16()         one NEON quad -> int16x4_t, for a caller
 *                                  that interleaves two channels with
 *                                  vst2_s16 and therefore cannot hand the
 *                                  array form a unit-stride destination
 *
 * ⚠ The quad form is NEON-only and has no _scalar twin: its return type does
 * not exist without <arm_neon.h>. Callers guard it with SK_HAVE_NEON and fall
 * back to the per-sample helper, which is why that helper is part of the
 * public surface rather than an internal detail.
 *
 * Bit-exact against the scalar reference:
 *   - clipping is vcgtq/vcltq + vbslq_f32 to match sk_clip_f32 (kernel 12), so
 *     this file has one clipping idiom rather than two. ⚠ Kernel 12's reason
 *     -- vmaxq/vminq's signed-zero tie-break -- does NOT apply here, and the
 *     comment is not repeated as if it did: the conversion below maps -0.0f and
 *     +0.0f to the same int16, so the difference is unobservable through this
 *     kernel. Verified by mutation: swapping in vmaxq(lo, vminq(hi, v)) leaves
 *     the selftest passing, where the three mutations below all fail it;
 *   - vcvtq_s32_f32 truncates toward zero, which is exactly what C's
 *     float->int conversion does, so there is no rounding-mode divergence.
 *     ⚠ vcvtnq_s32_f32 rounds to nearest and would NOT match;
 *   - vqmovn_s32's saturation can never engage, the input already being
 *     clipped to +-32767, so it is a plain narrow.
 *
 * ⚠ No caller lives in these repos. The consumer is platform integration code
 * outside this tree, which open-coded the sequence; this kernel exists so that
 * code has one definition to call and a bit-exactness test behind it. Until a
 * caller lands here, only the selftest exercises it.
 *
 * Related but deliberately NOT unified: wav_io.h's writer performs the same
 * conversion in its NR-style branch, but under a different contract -- it
 * sanitises non-finite input to 0.0f and COUNTS it, and its AEC-style branch
 * scales by 32768.0f with round-half-away-from-zero instead. Folding this
 * kernel in would have to resolve those differences in wav_io.h's favour, and
 * that path is one fwrite per sample, so there is nothing to win by doing it.
 *
 * The high bound is tested before the low one. For every non-NaN input the two
 * orders agree, so the order is not load-bearing here (unlike kernel 12, whose
 * reference pins it). NaN is not
 * handled: it survives both clips and reaches a float->int conversion, which
 * C leaves undefined. A caller that can produce NaN must stop it upstream --
 * a NaN reaching a PCM sink is a defect wherever it is converted.
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
    /* truncate, never vcvtnq_s32_f32 -- see the note above. */
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

/* ═══════════════════════════════ kernel 31 ═════════════════════════════════
 * sk_cmag_f32 — out[i] = x[i].r*x[i].r + x[i].i*x[i].i + eps
 *
 * ⚠ NAME vs OPERATION: this is SQUARED magnitude (power) plus a floor, not
 * magnitude -- there is no square root. The name is the one its call site
 * already uses; renaming it would be a silent contract break for that caller,
 * so the discrepancy is recorded here instead. `eps` is the caller's floor
 * (the mask estimator passes 1e-8f) and is added AFTER the two products are
 * summed, matching the scalar reference's association exactly.
 *
 * Separate vmulq/vaddq, never vfmaq_f32: the reference spells this as two
 * multiplies then two adds, which under -ffp-contract=off rounds at each step.
 *
 * The de-interleave goes through sk__cquad_load(), never a raw
 * Complex*->float* cast fed to vld2q_f32(): that cast is a strict-aliasing
 * violation, and this is a HEADER, so it would compile into every consumer
 * regardless of that consumer's aliasing flags. The helper also carries the
 * non-GCC/clang memcpy+vuzpq fallback this kernel would otherwise lose.
 *
 * ⚠ Naming, for anyone navigating from AEC's aec_simd_kernels.h: there
 * `sk_cabs_np_*` is magnitude and `sk_cmag2_np_*` is squared magnitude, both
 * in the numpy scaled-hypot form. This kernel is the NAIVE r*r+i*i with a
 * floor, and its name predates that family. It is not a duplicate of
 * sk_cmag2_np_f32 -- different formula, different rounding.
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

/* ═══════════════════════════════ kernel 32 ═════════════════════════════════
 * sk_linear_to_db_f32 — out[i] = 10 * log10f(x[i])
 *
 * Power-domain dB (10x, not 20x): the caller feeds it the squared magnitudes
 * kernel 31 produces.
 *
 * ⚠ NO NEON PATH, AND THAT IS THE POINT. The only bit-exact source of
 * log10f() is libm, and this header's whole contract is that a dispatched
 * kernel is bitwise identical to its scalar twin. sk_fast_log10_f32 (kernel
 * 26) is a deliberate APPROXIMATION -- routing this through it would silently
 * move every value the caller then compares against a dB threshold. A caller
 * that wants the approximation should call kernel 26 directly and say so.
 * It is here rather than in the caller because the dB convention (10x, this
 * epsilon-free form, libm) is a decision worth stating once. ⚠ It does NOT yet
 * consolidate anything: the scalar `10.0f * log10f(...)` sites elsewhere in
 * this stack are single-value, not array passes, and were not changed.
 */

static inline void sk_linear_to_db_f32(float *out, const float *x, int n) {
    int i;
    for (i = 0; i < n; ++i) out[i] = 10.0f * log10f(x[i]);
}

/* ═══════════════════════════════ kernel 33 ═════════════════════════════════
 * sk_asym_ema_f32 — one-pole EMA with a different coefficient per direction:
 *
 *     s[i] = (x[i] < s[i]) ? a_down*s[i] + (1-a_down)*x[i]
 *                          : a_up  *s[i] + (1-a_up)  *x[i]
 *
 * The noise-floor tracker shape: a slow a_up so the floor creeps upward, a
 * faster a_down so it drops promptly. `s` is updated IN PLACE.
 *
 * The comparison is strict `<` and selects a_down, exactly as the scalar
 * reference spells it -- at equality both branches produce the same value, so
 * the tie only matters for reading the code, not for the result.
 *
 * (1-a) is loop-invariant and exact in binary floating point, so hoisting it
 * out of the loop is bit-identical to recomputing it per element. Separate
 * vmulq/vaddq, never vfmaq_f32, for the usual reason.
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

#ifdef __cplusplus
}
#endif

#endif /* SIMD_KERNELS_H */
