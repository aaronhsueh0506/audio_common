/**
 * NEON/scalar float32 kernels for the neural-network pre/post-processing
 * (DeepFilterNet2, Align-ULCNet, GTCRN, RNNoise-ERB, DeepVQE and the DFN2
 * rate bridge's resampler).
 *
 * Kept apart from simd_kernels.h on purpose: that header is the traditional
 * AEC/NR libraries' kernel set, and nothing added here may change what those
 * libraries compile. This header includes simd_kernels.h only to share its
 * NEON switch (SK_HAVE_NEON), its Complex layout pins and its Complex-quad
 * load/store helpers; it adds no symbol to it.
 *
 * Contract, identical to simd_kernels.h's:
 *   - Every NEON entry point is byte-identical to its _scalar twin for finite
 *     inputs; the predicate kernels (skn_all_finite_*) return the same answer
 *     for every input, NaN and Inf included.
 *   - Scalar operation order is preserved; ordinary mul/add stays unfused
 *     (vmulq_f32 + vaddq_f32, never vfmaq_f32), and every including
 *     translation unit is compiled with -ffp-contract=off.
 *   - No reciprocal/sqrt estimates, no reassociation. vdivq_f32 is the
 *     correctly rounded IEEE divide, identical to the scalar '/'.
 *   - SIMD_KERNELS_FORCE_SCALAR selects the scalar twins.
 *   - Arguments do not alias unless a kernel says so.
 *
 * test/simd_nn_selftest.c is the bit-exactness gate for every kernel here.
 */

#ifndef SIMD_KERNEL_NN_H
#define SIMD_KERNEL_NN_H

#include <math.h>
#include <stddef.h>

#include "simd_kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A float view of a Complex array for the NEON loads below. GCC/Clang get a
 * may_alias type so strict aliasing stays valid; the scalar twins read the
 * struct members and need no view. */
#if SK_HAVE_NEON
#if defined(__GNUC__) || defined(__clang__)
typedef float skn__alias_float __attribute__((__may_alias__));
#else
typedef float skn__alias_float;
#endif
#endif

/* ═══════════════════════════════ skn_all_finite ═══════════════════════════
 * 1 when every element is finite, 0 otherwise.
 *
 * The accelerator-output gates (a NaN-prefilled tensor must come back fully
 * written and finite before it may become recurrent state) scan tens of
 * thousands of floats per frame. An early-exit `if (!isfinite(x)) return 0`
 * loop does not vectorize; this one keeps the running maximum of the masked
 * exponent field over sixteen lanes per step. The masked field never exceeds
 * 0x7f800000 and reaches it only for Inf or NaN -- exactly isfinite()'s
 * false set -- so the answer is identical to the scalar loop's for every bit
 * pattern. It exits at the first 64-element block holding a non-finite
 * value. */

static inline int skn_all_finite_f32_scalar(const float *x, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) {
        if (!isfinite(x[i])) return 0;
    }
    return 1;
}

static inline int skn_all_finite_cf32_scalar(const Complex *x, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) {
        if (!isfinite(x[i].r) || !isfinite(x[i].i)) return 0;
    }
    return 1;
}

#if SK_HAVE_NEON
static inline uint32x4_t skn__exp_field(const skn__alias_float *x,
                                        uint32x4_t exp_mask) {
    return vandq_u32(vreinterpretq_u32_f32(vld1q_f32(x)), exp_mask);
}

static inline int skn__all_finite_view(const skn__alias_float *x, size_t n) {
    const uint32x4_t exp_mask = vdupq_n_u32(0x7f800000u);
    size_t i = 0;
    while (i + 64 <= n) {
        uint32x4_t top = vdupq_n_u32(0u);
        size_t end = i + 64;
        for (; i < end; i += 16) {
            top = vmaxq_u32(top, vmaxq_u32(
                vmaxq_u32(skn__exp_field(x + i, exp_mask),
                          skn__exp_field(x + i + 4, exp_mask)),
                vmaxq_u32(skn__exp_field(x + i + 8, exp_mask),
                          skn__exp_field(x + i + 12, exp_mask))));
        }
        if (vmaxvq_u32(top) == 0x7f800000u) return 0;
    }
    for (; i + 4 <= n; i += 4) {
        if (vmaxvq_u32(skn__exp_field(x + i, exp_mask)) == 0x7f800000u)
            return 0;
    }
    for (; i < n; ++i) {
        if (!isfinite((float)x[i])) return 0;
    }
    return 1;
}

static inline int skn_all_finite_f32(const float *x, size_t n) {
    return skn__all_finite_view((const skn__alias_float *)(const void *)x, n);
}

/* Complex is two contiguous floats (pinned in simd_kernels.h), so the array
 * is 2n floats and the classification does not care which is which. */
static inline int skn_all_finite_cf32(const Complex *x, size_t n) {
    return skn__all_finite_view((const skn__alias_float *)(const void *)x, 2u * n);
}
#else
static inline int skn_all_finite_f32(const float *x, size_t n) {
    return skn_all_finite_f32_scalar(x, n);
}
static inline int skn_all_finite_cf32(const Complex *x, size_t n) {
    return skn_all_finite_cf32_scalar(x, n);
}
#endif

/* ═══════════════════════════════ skn_fill_f32 ═════════════════════════════
 * x[i] = value. The NaN prefill of accelerator-writable tensors. */

static inline void skn_fill_f32_scalar(float *x, size_t n, float value) {
    size_t i;
    for (i = 0; i < n; ++i) x[i] = value;
}

#if SK_HAVE_NEON
static inline void skn_fill_f32(float *x, size_t n, float value) {
    const float32x4_t v = vdupq_n_f32(value);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        vst1q_f32(x + i, v);
        vst1q_f32(x + i + 4, v);
        vst1q_f32(x + i + 8, v);
        vst1q_f32(x + i + 12, v);
    }
    for (; i + 4 <= n; i += 4) vst1q_f32(x + i, v);
    for (; i < n; ++i) x[i] = value;
}
#else
static inline void skn_fill_f32(float *x, size_t n, float value) {
    skn_fill_f32_scalar(x, n, value);
}
#endif

/* ═══════════════════════════════ skn_mul_f32 ══════════════════════════════
 * dst[i] = a[i] * b[i]. Window application and per-bin gain application.
 * dst may be exactly a or exactly b (same-index read-then-write). */

static inline void skn_mul_f32_scalar(float *dst, const float *a,
                                      const float *b, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) dst[i] = a[i] * b[i];
}

#if SK_HAVE_NEON
static inline void skn_mul_f32(float *dst, const float *a, const float *b,
                               size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t x0 = vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        float32x4_t x1 = vmulq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        vst1q_f32(dst + i, x0);
        vst1q_f32(dst + i + 4, x1);
    }
    for (; i + 4 <= n; i += 4)
        vst1q_f32(dst + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    for (; i < n; ++i) dst[i] = a[i] * b[i];
}
#else
static inline void skn_mul_f32(float *dst, const float *a, const float *b,
                               size_t n) {
    skn_mul_f32_scalar(dst, a, b, n);
}
#endif

/* ═══════════════════════════ skn_power_scale_f32 ══════════════════════════
 * p[k] = (re[k]*re[k] + im[k]*im[k]) * scale -- the ERB feature power, in
 * that exact association (two products, their sum, then the scale). */

static inline void skn_power_scale_f32_scalar(float *p, const float *re,
                                              const float *im, size_t n,
                                              float scale) {
    size_t k;
    for (k = 0; k < n; ++k) p[k] = (re[k] * re[k] + im[k] * im[k]) * scale;
}

#if SK_HAVE_NEON
static inline void skn_power_scale_f32(float *p, const float *re,
                                       const float *im, size_t n,
                                       float scale) {
    const float32x4_t s = vdupq_n_f32(scale);
    size_t k = 0;
    for (; k + 4 <= n; k += 4) {
        float32x4_t r = vld1q_f32(re + k);
        float32x4_t i = vld1q_f32(im + k);
        vst1q_f32(p + k, vmulq_f32(vaddq_f32(vmulq_f32(r, r), vmulq_f32(i, i)), s));
    }
    for (; k < n; ++k) p[k] = (re[k] * re[k] + im[k] * im[k]) * scale;
}
#else
static inline void skn_power_scale_f32(float *p, const float *re,
                                       const float *im, size_t n,
                                       float scale) {
    skn_power_scale_f32_scalar(p, re, im, n, scale);
}
#endif

/* ══════════════════════ skn_(de)interleave[_scale]_cf32 ═══════════════════
 * Complex (AoS) <-> split re/im arrays. The _scale forms multiply each part
 * by `scale` on the way; the plain forms are exact moves (no arithmetic, so
 * NaN payloads pass through untouched). */

static inline void skn_deinterleave_cf32_scalar(const Complex *in, float *re,
                                                float *im, size_t n) {
    size_t k;
    for (k = 0; k < n; ++k) { re[k] = in[k].r; im[k] = in[k].i; }
}

static inline void skn_deinterleave_scale_cf32_scalar(const Complex *in,
                                                      float *re, float *im,
                                                      size_t n, float scale) {
    size_t k;
    for (k = 0; k < n; ++k) { re[k] = in[k].r * scale; im[k] = in[k].i * scale; }
}

static inline void skn_interleave_cf32_scalar(const float *re, const float *im,
                                              Complex *out, size_t n) {
    size_t k;
    for (k = 0; k < n; ++k) { out[k].r = re[k]; out[k].i = im[k]; }
}

static inline void skn_interleave_scale_cf32_scalar(const float *re,
                                                    const float *im,
                                                    Complex *out, size_t n,
                                                    float scale) {
    size_t k;
    for (k = 0; k < n; ++k) { out[k].r = re[k] * scale; out[k].i = im[k] * scale; }
}

#if SK_HAVE_NEON
static inline void skn_deinterleave_cf32(const Complex *in, float *re,
                                         float *im, size_t n) {
    size_t k = 0;
    for (; k + 4 <= n; k += 4) {
        float32x4x2_t v = sk__cquad_load(in + k);
        vst1q_f32(re + k, v.val[0]);
        vst1q_f32(im + k, v.val[1]);
    }
    for (; k < n; ++k) { re[k] = in[k].r; im[k] = in[k].i; }
}

static inline void skn_deinterleave_scale_cf32(const Complex *in, float *re,
                                               float *im, size_t n,
                                               float scale) {
    const float32x4_t s = vdupq_n_f32(scale);
    size_t k = 0;
    for (; k + 4 <= n; k += 4) {
        float32x4x2_t v = sk__cquad_load(in + k);
        vst1q_f32(re + k, vmulq_f32(v.val[0], s));
        vst1q_f32(im + k, vmulq_f32(v.val[1], s));
    }
    for (; k < n; ++k) { re[k] = in[k].r * scale; im[k] = in[k].i * scale; }
}

static inline void skn_interleave_cf32(const float *re, const float *im,
                                       Complex *out, size_t n) {
    size_t k = 0;
    for (; k + 4 <= n; k += 4) {
        float32x4x2_t v;
        v.val[0] = vld1q_f32(re + k);
        v.val[1] = vld1q_f32(im + k);
        sk__cquad_store(out + k, v);
    }
    for (; k < n; ++k) { out[k].r = re[k]; out[k].i = im[k]; }
}

static inline void skn_interleave_scale_cf32(const float *re, const float *im,
                                             Complex *out, size_t n,
                                             float scale) {
    const float32x4_t s = vdupq_n_f32(scale);
    size_t k = 0;
    for (; k + 4 <= n; k += 4) {
        float32x4x2_t v;
        v.val[0] = vmulq_f32(vld1q_f32(re + k), s);
        v.val[1] = vmulq_f32(vld1q_f32(im + k), s);
        sk__cquad_store(out + k, v);
    }
    for (; k < n; ++k) { out[k].r = re[k] * scale; out[k].i = im[k] * scale; }
}
#else
static inline void skn_deinterleave_cf32(const Complex *in, float *re,
                                         float *im, size_t n) {
    skn_deinterleave_cf32_scalar(in, re, im, n);
}
static inline void skn_deinterleave_scale_cf32(const Complex *in, float *re,
                                               float *im, size_t n,
                                               float scale) {
    skn_deinterleave_scale_cf32_scalar(in, re, im, n, scale);
}
static inline void skn_interleave_cf32(const float *re, const float *im,
                                       Complex *out, size_t n) {
    skn_interleave_cf32_scalar(re, im, out, n);
}
static inline void skn_interleave_scale_cf32(const float *re, const float *im,
                                             Complex *out, size_t n,
                                             float scale) {
    skn_interleave_scale_cf32_scalar(re, im, out, n, scale);
}
#endif

/* ═══════════════════════════ skn_div_floor_f32 ════════════════════════════
 * out[i] = num[i] / (den[i] > floor ? den[i] : floor) -- the WOLA envelope
 * normalisation. The select is a greater-than compare plus bit select, not
 * vmaxq_f32, so a NaN denominator takes the floor exactly as the scalar
 * ternary does. */

static inline void skn_div_floor_f32_scalar(float *out, const float *num,
                                            const float *den, size_t n,
                                            float floor_value) {
    size_t i;
    for (i = 0; i < n; ++i) {
        float e = den[i];
        out[i] = num[i] / (e > floor_value ? e : floor_value);
    }
}

#if SK_HAVE_NEON
static inline void skn_div_floor_f32(float *out, const float *num,
                                     const float *den, size_t n,
                                     float floor_value) {
    const float32x4_t f = vdupq_n_f32(floor_value);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t e = vld1q_f32(den + i);
        float32x4_t d = vbslq_f32(vcgtq_f32(e, f), e, f);
        vst1q_f32(out + i, vdivq_f32(vld1q_f32(num + i), d));
    }
    for (; i < n; ++i) {
        float e = den[i];
        out[i] = num[i] / (e > floor_value ? e : floor_value);
    }
}
#else
static inline void skn_div_floor_f32(float *out, const float *num,
                                     const float *den, size_t n,
                                     float floor_value) {
    skn_div_floor_f32_scalar(out, num, den, n, floor_value);
}
#endif

/* ═══════════════════════════ skn_ring_dot_rows_f32 ════════════════════════
 * Up to SKN_RING_DOT_MAX_ROWS polyphase FIR rows dotted against ONE
 * newest-to-oldest circular history (audio_resampler's layout: `history` has
 * `taps` slots, the newest sample at `head`, older ones at head-1, head-2,
 * ... wrapping to taps-1).
 *
 * The rows an input sample produces (up to `up` output phases share one
 * history) run in one pass: the history loads are shared and the rows'
 * accumulation chains are independent -- a single row is one serial vaddq
 * chain, which is what bounds it. Each row's operation sequence is the same
 * whether it runs alone or with others:
 *   A: 4-tap blocks into the lane accumulator while the block does not
 *      cross the ring's start (tap+4 <= taps && index >= 3);
 *   B: single taps into the scalar accumulator down to ring index 0;
 *   C: 4-tap blocks from ring index taps-1 onward;
 *   then scalar += horizontal sum of the lanes, (l0 + l1) + (l2 + l3)
 *   (vaddvq_f32's pairwise order);
 *   D: the remaining single taps into the scalar accumulator.
 * The _scalar twin spells that lane by lane, so it equals the NEON form bit
 * for bit. It is not the resampler's SIMD=0 arithmetic (a plain sequential
 * sum per row), which lives in audio_resampler.c. */

#define SKN_RING_DOT_MAX_ROWS 4

static inline void skn_ring_dot_rows_f32_scalar(const float *const *rows,
                                                int n_rows,
                                                const float *history,
                                                int head, int taps,
                                                float *out) {
    int r;
    for (r = 0; r < n_rows; ++r) {
        const float *c = rows[r];
        float lane[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float scalar = 0.0f;
        int tap = 0, index = head, j;
        while (tap + 4 <= taps && index >= 3) {
            for (j = 0; j < 4; ++j) lane[j] = lane[j] + c[tap + j] * history[index - j];
            tap += 4;
            index -= 4;
        }
        while (tap < taps && index >= 0) scalar += c[tap++] * history[index--];
        index = taps - 1;
        while (tap + 4 <= taps) {
            for (j = 0; j < 4; ++j) lane[j] = lane[j] + c[tap + j] * history[index - j];
            tap += 4;
            index -= 4;
        }
        scalar += (lane[0] + lane[1]) + (lane[2] + lane[3]);
        while (tap < taps) scalar += c[tap++] * history[index--];
        out[r] = scalar;
    }
}

#if SK_HAVE_NEON
#if defined(__GNUC__) || defined(__clang__)
#define SKN__ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define SKN__ALWAYS_INLINE inline
#endif

/* ascending load of history[index-3 .. index], reversed to newest-first */
static inline float32x4_t skn__ring_load_rev(const float *history, int index) {
    float32x4_t h = vld1q_f32(history + index - 3);
    h = vrev64q_f32(h);
    return vextq_f32(h, h, 2);
}

/* One body for every row count. It is only ever called with a literal
 * n_rows (the switch below), so after forced inlining the `n_rows > k` tests
 * fold away and each row count gets its own register-allocated loop; with a
 * run-time row count the multi-row form measured slower than one row at a
 * time. */
static SKN__ALWAYS_INLINE void skn__ring_dot_rows_impl(
    const float *const *rows, const int n_rows, const float *history,
    int head, int taps, float *out) {
    const float *c0 = rows[0];
    const float *c1 = n_rows > 1 ? rows[1] : rows[0];
    const float *c2 = n_rows > 2 ? rows[2] : rows[0];
    const float *c3 = n_rows > 3 ? rows[3] : rows[0];
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int tap = 0, index = head;
    while (tap + 4 <= taps && index >= 3) {
        float32x4_t h = skn__ring_load_rev(history, index);
        a0 = vaddq_f32(a0, vmulq_f32(vld1q_f32(c0 + tap), h));
        if (n_rows > 1) a1 = vaddq_f32(a1, vmulq_f32(vld1q_f32(c1 + tap), h));
        if (n_rows > 2) a2 = vaddq_f32(a2, vmulq_f32(vld1q_f32(c2 + tap), h));
        if (n_rows > 3) a3 = vaddq_f32(a3, vmulq_f32(vld1q_f32(c3 + tap), h));
        tap += 4;
        index -= 4;
    }
    while (tap < taps && index >= 0) {
        float h = history[index--];
        s0 += c0[tap] * h;
        if (n_rows > 1) s1 += c1[tap] * h;
        if (n_rows > 2) s2 += c2[tap] * h;
        if (n_rows > 3) s3 += c3[tap] * h;
        ++tap;
    }
    index = taps - 1;
    while (tap + 4 <= taps) {
        float32x4_t h = skn__ring_load_rev(history, index);
        a0 = vaddq_f32(a0, vmulq_f32(vld1q_f32(c0 + tap), h));
        if (n_rows > 1) a1 = vaddq_f32(a1, vmulq_f32(vld1q_f32(c1 + tap), h));
        if (n_rows > 2) a2 = vaddq_f32(a2, vmulq_f32(vld1q_f32(c2 + tap), h));
        if (n_rows > 3) a3 = vaddq_f32(a3, vmulq_f32(vld1q_f32(c3 + tap), h));
        tap += 4;
        index -= 4;
    }
    s0 += vaddvq_f32(a0);
    if (n_rows > 1) s1 += vaddvq_f32(a1);
    if (n_rows > 2) s2 += vaddvq_f32(a2);
    if (n_rows > 3) s3 += vaddvq_f32(a3);
    while (tap < taps) {
        float h = history[index--];
        s0 += c0[tap] * h;
        if (n_rows > 1) s1 += c1[tap] * h;
        if (n_rows > 2) s2 += c2[tap] * h;
        if (n_rows > 3) s3 += c3[tap] * h;
        ++tap;
    }
    out[0] = s0;
    if (n_rows > 1) out[1] = s1;
    if (n_rows > 2) out[2] = s2;
    if (n_rows > 3) out[3] = s3;
}

static inline void skn_ring_dot_rows_f32(const float *const *rows, int n_rows,
                                         const float *history, int head,
                                         int taps, float *out) {
    switch (n_rows) {
    case 1: skn__ring_dot_rows_impl(rows, 1, history, head, taps, out); break;
    case 2: skn__ring_dot_rows_impl(rows, 2, history, head, taps, out); break;
    case 3: skn__ring_dot_rows_impl(rows, 3, history, head, taps, out); break;
    case 4: skn__ring_dot_rows_impl(rows, 4, history, head, taps, out); break;
    default: break;
    }
}
#else
static inline void skn_ring_dot_rows_f32(const float *const *rows, int n_rows,
                                         const float *history, int head,
                                         int taps, float *out) {
    skn_ring_dot_rows_f32_scalar(rows, n_rows, history, head, taps, out);
}
#endif

#if SK_HAVE_NEON
#undef SKN__ALWAYS_INLINE
#endif

#ifdef __cplusplus
}
#endif

#endif /* SIMD_KERNEL_NN_H */
