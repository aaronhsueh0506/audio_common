/** NEON/scalar float32 kernels for the neural-network pre/post-processing stages.
 * Kept apart from simd_kernels.h; it shares that header's NEON switch (SK_HAVE_NEON), Complex layout checks and Complex-quad load/store helpers.
 * Contract: every NEON entry point matches its _scalar twin for finite inputs (ERB arithmetic pins NaN classification, not payload bits);
 * skn_all_finite_* agree for every input. Operation order is fixed: mul/add stay unfused (no vfmaq_f32), no reciprocal/sqrt estimates,
 * no reassociation; every including unit is compiled with -ffp-contract=off. SIMD_KERNELS_FORCE_SCALAR selects the scalar twins.
 * Arguments do not alias unless a kernel says so. */

#ifndef SIMD_KERNEL_NN_H
#define SIMD_KERNEL_NN_H

#include <math.h>
#include <stddef.h>

#include "simd_kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Float view of a Complex array for NEON loads (may_alias on GCC/Clang keeps strict aliasing valid). */
#if SK_HAVE_NEON
#if defined(__GNUC__) || defined(__clang__)
typedef float skn__alias_float __attribute__((__may_alias__));
#else
typedef float skn__alias_float;
#endif
#endif

/* skn_all_finite: 1 when every element is finite, else 0. Keeps the running maximum of the masked exponent field over sixteen lanes
 * per step (an early-exit isfinite() loop does not vectorize) and exits at the first 64-element block holding an Inf or NaN. */

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

/* Complex is two contiguous floats (checked in simd_kernels.h), so the array is 2n floats. */
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

/* skn_fill_f32: x[i] = value. */

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

/* skn_mul_f32: dst[i] = a[i] * b[i]; dst may be exactly a or exactly b. */

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

/* skn_power_scale_f32: p[k] = (re[k]*re[k] + im[k]*im[k]) * scale, in that association. */

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

/* skn_mag_eps_f32: out[k] = sqrtf((re[k]*re[k] + im[k]*im[k]) + eps), correctly rounded root; out may be exactly re or im. */

static inline void skn_mag_eps_f32_scalar(float *out, const float *re,
                                          const float *im, size_t n,
                                          float eps) {
    size_t k;
    for (k = 0; k < n; ++k) out[k] = sqrtf((re[k] * re[k] + im[k] * im[k]) + eps);
}

#if SK_HAVE_NEON
static inline void skn_mag_eps_f32(float *out, const float *re,
                                   const float *im, size_t n, float eps) {
    const float32x4_t e = vdupq_n_f32(eps);
    size_t k = 0;
    for (; k + 4 <= n; k += 4) {
        float32x4_t r = vld1q_f32(re + k);
        float32x4_t i = vld1q_f32(im + k);
        vst1q_f32(out + k, vsqrtq_f32(vaddq_f32(
            vaddq_f32(vmulq_f32(r, r), vmulq_f32(i, i)), e)));
    }
    for (; k < n; ++k) out[k] = sqrtf((re[k] * re[k] + im[k] * im[k]) + eps);
}
#else
static inline void skn_mag_eps_f32(float *out, const float *re,
                                   const float *im, size_t n, float eps) {
    skn_mag_eps_f32_scalar(out, re, im, n, eps);
}
#endif

/* skn_(de)interleave[_scale]_cf32: Complex (AoS) <-> split re/im. The _scale forms multiply by `scale`; the plain forms are exact moves. */

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

/* skn_div_floor_f32: out[i] = num[i] / (den[i] > floor ? den[i] : floor). Uses a compare plus bit select (not vmaxq_f32) so a NaN denominator takes the floor. */

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
    const size_t n4 = n & ~(size_t)3;
    size_t i = 0;
    for (; i < n4; i += 4) {
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

/* skn_ctaps_mac_f32: per-bin complex tap filter. For each bin k in [0, n), acc = sum over t = 0..taps-1 (in order) of
 *   acc.re += xr*cr - xi*ci; acc.im += xi*cr + xr*ci, with (xr, xi) = (src_re[t][k], src_im[t][k]) and (cr, ci) from `coef`
 * laid out [bin][tap][re,im] (stride 2*taps floats). One source row per tap, already offset to bin 0 of the range.
 * out_re/out_im must not alias any source or coef. Every bin takes all `taps` taps. */

static inline void skn__ctaps_mac_range(float *out_re, float *out_im,
                                        size_t k0, size_t n, size_t taps,
                                        const float *const *src_re,
                                        const float *const *src_im,
                                        const float *coef) {
    size_t k, t;
    for (k = k0; k < n; ++k) {
        float sum_re = 0.0f;
        float sum_im = 0.0f;
        for (t = 0; t < taps; ++t) {
            const float xr = src_re[t][k];
            const float xi = src_im[t][k];
            const float cr = coef[2 * (k * taps + t)];
            const float ci = coef[2 * (k * taps + t) + 1];
            sum_re += xr * cr - xi * ci;
            sum_im += xi * cr + xr * ci;
        }
        out_re[k] = sum_re;
        out_im[k] = sum_im;
    }
}

static inline void skn_ctaps_mac_f32_scalar(float *out_re, float *out_im,
                                            size_t n, size_t taps,
                                            const float *const *src_re,
                                            const float *const *src_im,
                                            const float *coef) {
    skn__ctaps_mac_range(out_re, out_im, 0, n, taps, src_re, src_im, coef);
}

#if SK_HAVE_NEON
static inline void skn_ctaps_mac_f32(float *out_re, float *out_im, size_t n,
                                     size_t taps, const float *const *src_re,
                                     const float *const *src_im,
                                     const float *coef) {
    const size_t n4 = n & ~(size_t)3;
    size_t k = 0, t;
    for (; k < n4; k += 4) {
        const float *c = coef + 2 * k * taps;
        float32x4_t sum_re = vdupq_n_f32(0.0f);
        float32x4_t sum_im = vdupq_n_f32(0.0f);
        for (t = 0; t < taps; ++t) {
            const float32x4_t xr = vld1q_f32(src_re[t] + k);
            const float32x4_t xi = vld1q_f32(src_im[t] + k);
            const float32x4x2_t cc = vuzpq_f32(
                vcombine_f32(vld1_f32(c + 2 * t),
                             vld1_f32(c + 2 * (taps + t))),
                vcombine_f32(vld1_f32(c + 2 * (2 * taps + t)),
                             vld1_f32(c + 2 * (3 * taps + t))));
            sum_re = vaddq_f32(sum_re, vsubq_f32(vmulq_f32(xr, cc.val[0]),
                                                 vmulq_f32(xi, cc.val[1])));
            sum_im = vaddq_f32(sum_im, vaddq_f32(vmulq_f32(xi, cc.val[0]),
                                                 vmulq_f32(xr, cc.val[1])));
        }
        vst1q_f32(out_re + k, sum_re);
        vst1q_f32(out_im + k, sum_im);
    }
    skn__ctaps_mac_range(out_re, out_im, k, n, taps, src_re, src_im, coef);
}
#else
static inline void skn_ctaps_mac_f32(float *out_re, float *out_im, size_t n,
                                     size_t taps, const float *const *src_re,
                                     const float *const *src_im,
                                     const float *coef) {
    skn_ctaps_mac_f32_scalar(out_re, out_im, n, taps, src_re, src_im, coef);
}
#endif

/* skn_erb_forward_banded_f32: erb_work[b] += sum over k in [k0[b], k1[b]) ascending of power[k] * erb_fwd[k * n_bands + b].
 * erb_work is NOT zeroed here. n_bands must be a multiple of 4. Four adjacent bands advance together with local accumulators.
 * A single implementation serves SIMD=0 and SIMD=1 (no _scalar twin). Every power must be finite; the caller handles non-finite power. */

static inline void skn_erb_forward_banded_f32(const float *power,
                                               const float *erb_fwd,
                                               const uint16_t *k0,
                                               const uint16_t *k1,
                                               int n_bands, float *erb_work) {
    for (int b = 0; b < n_bands; b += 4) {
        const int a0 = k0[b], a1 = k0[b + 1], a2 = k0[b + 2], a3 = k0[b + 3];
        const int n0 = k1[b] - a0, n1 = k1[b + 1] - a1;
        const int n2 = k1[b + 2] - a2, n3 = k1[b + 3] - a3;
        int common = n0 < n1 ? n0 : n1;
        const float *w0 = erb_fwd + (size_t)a0 * n_bands + b;
        const float *w1 = erb_fwd + (size_t)a1 * n_bands + b + 1;
        const float *w2 = erb_fwd + (size_t)a2 * n_bands + b + 2;
        const float *w3 = erb_fwd + (size_t)a3 * n_bands + b + 3;
        float s0 = erb_work[b], s1 = erb_work[b + 1];
        float s2 = erb_work[b + 2], s3 = erb_work[b + 3];
        if (n2 < common) common = n2;
        if (n3 < common) common = n3;
        int j;
        for (j = 0; j < common; ++j) {
            s0 += power[a0 + j] * w0[(size_t)j * n_bands];
            s1 += power[a1 + j] * w1[(size_t)j * n_bands];
            s2 += power[a2 + j] * w2[(size_t)j * n_bands];
            s3 += power[a3 + j] * w3[(size_t)j * n_bands];
        }
        for (j = common; j < n0; ++j) s0 += power[a0 + j] * w0[(size_t)j * n_bands];
        for (j = common; j < n1; ++j) s1 += power[a1 + j] * w1[(size_t)j * n_bands];
        for (j = common; j < n2; ++j) s2 += power[a2 + j] * w2[(size_t)j * n_bands];
        for (j = common; j < n3; ++j) s3 += power[a3 + j] * w3[(size_t)j * n_bands];
        erb_work[b] = s0;
        erb_work[b + 1] = s1;
        erb_work[b + 2] = s2;
        erb_work[b + 3] = s3;
    }
}

/* skn_erb_expand_mask_f32: bin_gain[k] = sum over bands b ascending of erb_inv[b * n_bins + k] * band_gain[b] (bin_gain zeroed on entry);
 * erb_inv is band-major [n_bands][n_bins] and a band covers its nonzero bins [inv_lo[b], inv_hi[b]). A non-finite band_gain takes the band's whole row.
 * The NEON step may run up to three bins past a band's span; the weights there are zero, so finite gains are unaffected. */

static inline void skn_erb_expand_mask_f32_scalar(const float *band_gain,
                                                    const float *erb_inv,
                                                    const uint16_t *inv_lo,
                                                    const uint16_t *inv_hi,
                                                    int n_bins, int n_bands,
                                                    float *bin_gain) {
    memset(bin_gain, 0, (size_t)n_bins * sizeof(float));
    for (int b = 0; b < n_bands; ++b) {
        float gain = band_gain[b];
        const float *row = erb_inv + (size_t)b * n_bins;
        int finite = isfinite(gain);
        int hi = finite ? inv_hi[b] : n_bins;
        for (int k = finite ? inv_lo[b] : 0; k < hi; ++k)
            bin_gain[k] += row[k] * gain;
    }
}

#if SK_HAVE_NEON
static inline void skn_erb_expand_mask_f32_neon(const float *band_gain,
                                                 const float *erb_inv,
                                                 const uint16_t *inv_lo,
                                                 const uint16_t *inv_hi,
                                                 int n_bins, int n_bands,
                                                 float *bin_gain) {
    memset(bin_gain, 0, (size_t)n_bins * sizeof(float));
    for (int b = 0; b < n_bands; ++b) {
        float gain = band_gain[b];
        const float *row = erb_inv + (size_t)b * n_bins;
        int finite = isfinite(gain);
        int hi = finite ? inv_hi[b] : n_bins;
        int k = finite ? inv_lo[b] : 0;
        float32x4_t g = vdupq_n_f32(gain);
        for (; k < hi && k + 4 <= n_bins; k += 4)
            vst1q_f32(bin_gain + k,
                      vaddq_f32(vld1q_f32(bin_gain + k),
                                vmulq_f32(vld1q_f32(row + k), g)));
        for (; k < hi; ++k)
            bin_gain[k] += row[k] * gain;
    }
}

static inline void skn_erb_expand_mask_f32(const float *band_gain,
                                            const float *erb_inv,
                                            const uint16_t *inv_lo,
                                            const uint16_t *inv_hi,
                                            int n_bins, int n_bands,
                                            float *bin_gain) {
    skn_erb_expand_mask_f32_neon(band_gain, erb_inv, inv_lo, inv_hi, n_bins,
                                 n_bands, bin_gain);
}
#else
static inline void skn_erb_expand_mask_f32(const float *band_gain,
                                            const float *erb_inv,
                                            const uint16_t *inv_lo,
                                            const uint16_t *inv_hi,
                                            int n_bins, int n_bands,
                                            float *bin_gain) {
    skn_erb_expand_mask_f32_scalar(band_gain, erb_inv, inv_lo, inv_hi, n_bins,
                                   n_bands, bin_gain);
}
#endif

/* banded[b] = sum_k table[k*n_bands+b]*x[k], starting at zero; table is bin-major [n_bins][n_bands]. Each band folds ascending k with
 * separate multiply/add rounding; the output is overwritten. */

static inline void skn_erb_band_accumulate_f32_scalar(const float *x,
                                                        const float *table,
                                                        int n_bins,
                                                        int n_bands,
                                                        float *banded) {
    for (int b = 0; b < n_bands; ++b) {
        float sum = 0.0f;
        for (int k = 0; k < n_bins; ++k)
            sum += table[(size_t)k * n_bands + b] * x[k];
        banded[b] = sum;
    }
}

/* Multi-channel form of the above: same ascending-k unfused fold per band and channel; each table quad is loaded once per bin and reused across channels.
 * x and banded hold n_ch pointers, n_ch in [1, SKN_ERB_NCH_MAX]. The NEON path groups 16 bands, then four, then scalar tails. */

#define SKN_ERB_NCH_MAX 4
#define SKN_ERB_BAND_QUADS 4

static inline void skn_erb_band_accumulate_nch_f32_scalar(
    const float *const *x, int n_ch, const float *table, int n_bins,
    int n_bands, float *const *banded) {
    for (int b = 0; b < n_bands; ++b) {
        float sum[SKN_ERB_NCH_MAX];
        for (int c = 0; c < n_ch; ++c) sum[c] = 0.0f;
        for (int k = 0; k < n_bins; ++k) {
            float t = table[(size_t)k * n_bands + b];
            for (int c = 0; c < n_ch; ++c) sum[c] += t * x[c][k];
        }
        for (int c = 0; c < n_ch; ++c) banded[c][b] = sum[c];
    }
}

#if SK_HAVE_NEON
/* One specialization per channel count NCH (a literal at the call site) so the inner loop bounds are compile-time constants and accumulators stay in registers. */
#define SKN_ERB_NCH_IMPL(NCH)                                               \
static inline void skn_erb_band_accumulate_nch_f32_neon_##NCH(              \
    const float *const *x, const float *table, int n_bins, int n_bands,    \
    float *const *banded) {                                                 \
    int b = 0;                                                              \
    /* Full blocks of SKN_ERB_BAND_QUADS quads (16 bands); never reads past the valid band range. */                                       \
    for (; b + 4 * SKN_ERB_BAND_QUADS <= n_bands; b += 4 * SKN_ERB_BAND_QUADS) { \
        float32x4_t sum[NCH][SKN_ERB_BAND_QUADS];                           \
        for (int c = 0; c < NCH; ++c)                                       \
            for (int q = 0; q < SKN_ERB_BAND_QUADS; ++q)                    \
                sum[c][q] = vdupq_n_f32(0.0f);                              \
        for (int k = 0; k < n_bins; ++k) {                                  \
            const float *row = table + (size_t)k * n_bands + b;             \
            for (int c = 0; c < NCH; ++c) {                                 \
                float32x4_t xv = vdupq_n_f32(x[c][k]);                      \
                for (int q = 0; q < SKN_ERB_BAND_QUADS; ++q)                \
                    sum[c][q] = vaddq_f32(                                  \
                        sum[c][q], vmulq_f32(vld1q_f32(row + 4 * q), xv)); \
            }                                                               \
        }                                                                    \
        for (int c = 0; c < NCH; ++c)                                       \
            for (int q = 0; q < SKN_ERB_BAND_QUADS; ++q)                    \
                vst1q_f32(banded[c] + b + 4 * q, sum[c][q]);                \
    }                                                                        \
    /* One quad (4 bands) at a time for what a full block cannot cover. */                                                                      \
    for (; b + 4 <= n_bands; b += 4) {                                      \
        float32x4_t sum[NCH];                                               \
        for (int c = 0; c < NCH; ++c) sum[c] = vdupq_n_f32(0.0f);           \
        for (int k = 0; k < n_bins; ++k) {                                  \
            float32x4_t tv = vld1q_f32(table + (size_t)k * n_bands + b);    \
            for (int c = 0; c < NCH; ++c)                                   \
                sum[c] = vaddq_f32(sum[c],                                  \
                                   vmulq_f32(tv, vdupq_n_f32(x[c][k])));    \
        }                                                                    \
        for (int c = 0; c < NCH; ++c) vst1q_f32(banded[c] + b, sum[c]);      \
    }                                                                        \
    /* Last fewer-than-4 bands, scalar. */       \
    for (; b < n_bands; ++b) {                                              \
        float sum[NCH];                                                     \
        for (int c = 0; c < NCH; ++c) sum[c] = 0.0f;                        \
        for (int k = 0; k < n_bins; ++k) {                                  \
            float t = table[(size_t)k * n_bands + b];                       \
            for (int c = 0; c < NCH; ++c) sum[c] += t * x[c][k];           \
        }                                                                    \
        for (int c = 0; c < NCH; ++c) banded[c][b] = sum[c];               \
    }                                                                        \
}

SKN_ERB_NCH_IMPL(1)
SKN_ERB_NCH_IMPL(2)
SKN_ERB_NCH_IMPL(3)
SKN_ERB_NCH_IMPL(4)
#undef SKN_ERB_NCH_IMPL

static inline void skn_erb_band_accumulate_nch_f32(
    const float *const *x, int n_ch, const float *table, int n_bins,
    int n_bands, float *const *banded) {
    switch (n_ch) {
    case 1: skn_erb_band_accumulate_nch_f32_neon_1(x, table, n_bins, n_bands, banded); return;
    case 2: skn_erb_band_accumulate_nch_f32_neon_2(x, table, n_bins, n_bands, banded); return;
    case 3: skn_erb_band_accumulate_nch_f32_neon_3(x, table, n_bins, n_bands, banded); return;
    default: skn_erb_band_accumulate_nch_f32_neon_4(x, table, n_bins, n_bands, banded); return;
    }
}
#else
static inline void skn_erb_band_accumulate_nch_f32(
    const float *const *x, int n_ch, const float *table, int n_bins,
    int n_bands, float *const *banded) {
    skn_erb_band_accumulate_nch_f32_scalar(x, n_ch, table, n_bins, n_bands,
                                           banded);
}
#endif

#ifdef __cplusplus
}
#endif

#endif /* SIMD_KERNEL_NN_H */
