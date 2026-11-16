/** AEC-specific scalar/NEON micro-kernels: filtering, spectral statistics, stationarity and saturation. Stateful wrappers stay in AEC;
 * generic kernels and the SK_HAVE_NEON switch come from simd_kernels.h.
 * For finite inputs every NEON entry point matches its scalar twin exactly; multi-NaN expressions may pick different NaN payloads but agree on classification.
 * Operation order is fixed: explicit fmaf stays fused, ordinary mul/add stay separate (-ffp-contract=off), no reciprocal/sqrt estimates, no reassociation.
 * Partial pointer overlap is unsupported (sk_mask_zero_f32 is in place). Complex NEON access goes through sk__cquad_load/sk__cquad_store. */

#ifndef SIMD_AEC_H
#define SIMD_AEC_H

#include "simd_kernels.h"

#include <limits.h>   /* Hold counters are plain int; saturate at INT_MIN. */

#ifdef __cplusplus
extern "C" {
#endif

/* Per-element helpers shared by the scalar kernels and the NEON scalar tails. */

/* numpy complex64 |z|: scaled hypot with an explicit fmaf (correctly rounded). */
static inline float sk__cabs_np_elem(float re, float im) {
    float ar = re < 0.0f ? -re : re;
    float ai = im < 0.0f ? -im : im;
    float larger  = ar > ai ? ar : ai;
    float smaller = ar > ai ? ai : ar;
    if (larger == 0.0f) return 0.0f;
    {
        float ratio = smaller / larger;
        return larger * sqrtf(fmaf(ratio, ratio, 1.0f));
    }
}

/* numpy complex64 magnitude squared via scaled hypot and explicit FMA. */
static inline float sk__cmag2_np_elem(float re, float im) {
    float m = sk__cabs_np_elem(re, im);
    return m * m;
}

#if SK_HAVE_NEON
/* Compare/select reproduces the scalar ternaries, including NaN signs and unordered compares (vabsq/vminq/vmaxq would change them).
 * Zero-magnitude lanes compute 0/0 internally and then select +0.0f. */
static inline float32x4_t sk__cabs_np_neon4(float32x4_t re, float32x4_t im) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    uint32x4_t re_neg = vcltq_f32(re, zero);
    uint32x4_t im_neg = vcltq_f32(im, zero);
    float32x4_t ar = vbslq_f32(re_neg, vnegq_f32(re), re);
    float32x4_t ai = vbslq_f32(im_neg, vnegq_f32(im), im);
    uint32x4_t gt = vcgtq_f32(ar, ai);
    float32x4_t larger  = vbslq_f32(gt, ar, ai);
    float32x4_t smaller = vbslq_f32(gt, ai, ar);
    float32x4_t ratio = vdivq_f32(smaller, larger);
    float32x4_t m = vmulq_f32(larger,
                       vsqrtq_f32(vfmaq_f32(vdupq_n_f32(1.0f), ratio, ratio)));
    uint32x4_t is_zero = vceqq_f32(larger, zero);
    return vbslq_f32(is_zero, zero, m);
}

static inline float32x4_t sk__cmag2_np_neon4(float32x4_t re, float32x4_t im) {
    float32x4_t m = sk__cabs_np_neon4(re, im);
    return vmulq_f32(m, m);
}
#endif

/* sk_cabs_np_f32: out[i] = numpy |z[i]|. */

static inline void sk_cabs_np_f32_scalar(const Complex *z, float *out, int n) {
    int i;
    for (i = 0; i < n; ++i) out[i] = sk__cabs_np_elem(z[i].r, z[i].i);
}

#if SK_HAVE_NEON
static inline void sk_cabs_np_f32(const Complex *z, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t v = sk__cquad_load(z + i);
        float32x4_t r = sk__cabs_np_neon4(v.val[0], v.val[1]);
        vst1q_f32(out + i, r);
    }
    for (; i < n; ++i) out[i] = sk__cabs_np_elem(z[i].r, z[i].i);
}
#else
static inline void sk_cabs_np_f32(const Complex *z, float *out, int n) {
    sk_cabs_np_f32_scalar(z, out, n);
}
#endif

/* sk_cmag2_np_f32: out[i] = numpy |z[i]|**2. */

static inline void sk_cmag2_np_f32_scalar(const Complex *z, float *out, int n) {
    int i;
    for (i = 0; i < n; ++i) out[i] = sk__cmag2_np_elem(z[i].r, z[i].i);
}

#if SK_HAVE_NEON
static inline void sk_cmag2_np_f32(const Complex *z, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t v = sk__cquad_load(z + i);
        float32x4_t m2 = sk__cmag2_np_neon4(v.val[0], v.val[1]);
        vst1q_f32(out + i, m2);
    }
    for (; i < n; ++i) out[i] = sk__cmag2_np_elem(z[i].r, z[i].i);
}
#else
static inline void sk_cmag2_np_f32(const Complex *z, float *out, int n) {
    sk_cmag2_np_f32_scalar(z, out, n);
}
#endif

/* sk_cmag2_np_acc_f32: acc[i] += numpy |z[i]|**2 (far-power accumulation). */

static inline void sk_cmag2_np_acc_f32_scalar(const Complex *z, float *acc, int n) {
    int i;
    for (i = 0; i < n; ++i) acc[i] += sk__cmag2_np_elem(z[i].r, z[i].i);
}

#if SK_HAVE_NEON
static inline void sk_cmag2_np_acc_f32(const Complex *z, float *acc, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t v = sk__cquad_load(z + i);
        float32x4_t m2 = sk__cmag2_np_neon4(v.val[0], v.val[1]);
        float32x4_t a = vld1q_f32(acc + i);
        vst1q_f32(acc + i, vaddq_f32(a, m2));
    }
    for (; i < n; ++i) acc[i] += sk__cmag2_np_elem(z[i].r, z[i].i);
}
#else
static inline void sk_cmag2_np_acc_f32(const Complex *z, float *acc, int n) {
    sk_cmag2_np_acc_f32_scalar(z, acc, n);
}
#endif

/* sk_wupdate_nlms_f32: NLMS coarse-filter W update for ONE partition, vectorized over the K bins:
 *   gr = fmaf(er, xr, -(ei * -xi)); gi = fmaf(er, -xi, ei * xr)  (err * conj(X), fused)
 *   W[k].r += mu_eff[k] * gr; W[k].i += mu_eff[k] * gi  (separate multiply then add). */

static inline void sk_wupdate_nlms_f32_scalar(Complex *W, const Complex *X,
                                               const Complex *err,
                                               const float *mu_eff, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        float er = err[i].r, ei = err[i].i;
        float xr = X[i].r, xi = X[i].i;
        float cxr = xr, cxi = -xi;
        float gr = fmaf(er, cxr, -(ei * cxi));
        float gi = fmaf(er, cxi,  (ei * cxr));
        W[i].r += mu_eff[i] * gr;
        W[i].i += mu_eff[i] * gi;
    }
}

#if SK_HAVE_NEON
static inline void sk_wupdate_nlms_f32(Complex *W, const Complex *X,
                                        const Complex *err,
                                        const float *mu_eff, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t ev = sk__cquad_load(err + i);
        float32x4_t er = ev.val[0], ei = ev.val[1];
        float32x4x2_t xv = sk__cquad_load(X + i);
        float32x4_t xr = xv.val[0], xi = xv.val[1];
        float32x4_t cxr = xr;
        float32x4_t cxi = vnegq_f32(xi);
        float32x4_t gr = vfmaq_f32(vnegq_f32(vmulq_f32(ei, cxi)), er, cxr);
        float32x4_t gi = vfmaq_f32(vmulq_f32(ei, cxr), er, cxi);
        float32x4_t mu = vld1q_f32(mu_eff + i);
        float32x4x2_t wv = sk__cquad_load(W + i);
        float32x4x2_t rv;
        rv.val[0] = vaddq_f32(wv.val[0], vmulq_f32(mu, gr));
        rv.val[1] = vaddq_f32(wv.val[1], vmulq_f32(mu, gi));
        sk__cquad_store(W + i, rv);
    }
    for (; i < n; ++i) {
        float er = err[i].r, ei = err[i].i;
        float xr = X[i].r, xi = X[i].i;
        float cxr = xr, cxi = -xi;
        float gr = fmaf(er, cxr, -(ei * cxi));
        float gi = fmaf(er, cxi,  (ei * cxr));
        W[i].r += mu_eff[i] * gr;
        W[i].i += mu_eff[i] * gi;
    }
}
#else
static inline void sk_wupdate_nlms_f32(Complex *W, const Complex *X,
                                        const Complex *err,
                                        const float *mu_eff, int n) {
    sk_wupdate_nlms_f32_scalar(W, X, err, mu_eff, n);
}
#endif

/* sk_wupdate_kf_f32: Kalman W update for ONE partition:
 *   (ksr, ksi) = mu[k] * conj(X) * mu_scale[k] (plain multiplies); W.r += fmaf(ksr, er, -(ksi * ei)) and W.i += fmaf(ksr, ei, ksi * er), each a fused product followed by a separate add
 *   (vfmaq_f32 then vaddq_f32). */

static inline void sk_wupdate_kf_f32_scalar(Complex *W, const Complex *X,
                                             const Complex *err,
                                             const float *mu,
                                             const float *mu_scale, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        float xr = X[i].r, xi = X[i].i;
        float kr = mu[i] * xr;
        float ki = -(mu[i] * xi);
        float ksr = kr * mu_scale[i];
        float ksi = ki * mu_scale[i];
        float er = err[i].r, ei = err[i].i;
        W[i].r += fmaf(ksr, er, -(ksi * ei));
        W[i].i += fmaf(ksr, ei,  (ksi * er));
    }
}

#if SK_HAVE_NEON
static inline void sk_wupdate_kf_f32(Complex *W, const Complex *X,
                                      const Complex *err,
                                      const float *mu,
                                      const float *mu_scale, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t xv = sk__cquad_load(X + i);
        float32x4_t xr = xv.val[0], xi = xv.val[1];
        float32x4_t muv = vld1q_f32(mu + i);
        float32x4_t msv = vld1q_f32(mu_scale + i);
        float32x4_t kr = vmulq_f32(muv, xr);
        float32x4_t ki = vnegq_f32(vmulq_f32(muv, xi));
        float32x4_t ksr = vmulq_f32(kr, msv);
        float32x4_t ksi = vmulq_f32(ki, msv);
        float32x4x2_t ev = sk__cquad_load(err + i);
        float32x4_t er = ev.val[0], ei = ev.val[1];
        float32x4_t dr = vfmaq_f32(vnegq_f32(vmulq_f32(ksi, ei)), ksr, er);
        float32x4_t di = vfmaq_f32(vmulq_f32(ksi, er), ksr, ei);
        float32x4x2_t wv = sk__cquad_load(W + i);
        float32x4x2_t rv;
        rv.val[0] = vaddq_f32(wv.val[0], dr);
        rv.val[1] = vaddq_f32(wv.val[1], di);
        sk__cquad_store(W + i, rv);
    }
    for (; i < n; ++i) {
        float xr = X[i].r, xi = X[i].i;
        float kr = mu[i] * xr;
        float ki = -(mu[i] * xi);
        float ksr = kr * mu_scale[i];
        float ksi = ki * mu_scale[i];
        float er = err[i].r, ei = err[i].i;
        W[i].r += fmaf(ksr, er, -(ksi * ei));
        W[i].i += fmaf(ksr, ei,  (ksi * er));
    }
}
#else
static inline void sk_wupdate_kf_f32(Complex *W, const Complex *X,
                                      const Complex *err,
                                      const float *mu,
                                      const float *mu_scale, int n) {
    sk_wupdate_kf_f32_scalar(W, X, err, mu, mu_scale, n);
}
#endif

/* sk_pairwise_sum_f32: numpy pairwise float32 sum (n <= 8 serial; n <= 128 eight-accumulator leaf; recursive split with half -= half % 8).
 * The NEON leaf holds acc[0..3] in q0 and acc[4..7] in q1 and combines them with two vpaddq_f32, in the scalar leaf's grouping. */

static inline float sk__pairwise_sum_leaf_scalar(const float *a, size_t n) {
    float acc[8];
    size_t i, j;
    for (j = 0; j < 8; ++j) acc[j] = a[j];
    for (i = 8; i + 8 <= n; i += 8)
        for (j = 0; j < 8; ++j) acc[j] += a[i + j];
    {
        float s = 0.0f, r;
        for (; i < n; ++i) s += a[i];
        r = ((acc[0] + acc[1]) + (acc[2] + acc[3]))
          + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        return r + s;
    }
}

static inline float sk_pairwise_sum_f32_scalar(const float *a, size_t n) {
    if (n <= 8) {
        float s = 0.0f;
        size_t i;
        for (i = 0; i < n; ++i) s += a[i];
        return s;
    }
    if (n <= 128) return sk__pairwise_sum_leaf_scalar(a, n);
    {
        size_t half = n / 2;
        half -= half % 8;
        return sk_pairwise_sum_f32_scalar(a, half)
             + sk_pairwise_sum_f32_scalar(a + half, n - half);
    }
}

#if SK_HAVE_NEON
static inline float sk__pairwise_sum_leaf_neon(const float *a, size_t n) {
    float32x4_t q0 = vld1q_f32(a), q1 = vld1q_f32(a + 4);
    size_t i;
    for (i = 8; i + 8 <= n; i += 8) {
        q0 = vaddq_f32(q0, vld1q_f32(a + i));
        q1 = vaddq_f32(q1, vld1q_f32(a + i + 4));
    }
    {
        float s = 0.0f, r;
        for (; i < n; ++i) s += a[i];
        {
            float32x4_t t = vpaddq_f32(q0, q1);
            float32x4_t u = vpaddq_f32(t, t);
            r = vgetq_lane_f32(u, 0) + vgetq_lane_f32(u, 1);
        }
        return r + s;
    }
}

static inline float sk_pairwise_sum_f32(const float *a, size_t n) {
    if (n <= 8) {
        float s = 0.0f;
        size_t i;
        for (i = 0; i < n; ++i) s += a[i];
        return s;
    }
    if (n <= 128) return sk__pairwise_sum_leaf_neon(a, n);
    {
        size_t half = n / 2;
        half -= half % 8;
        return sk_pairwise_sum_f32(a, half) + sk_pairwise_sum_f32(a + half, n - half);
    }
}
#else
static inline float sk_pairwise_sum_f32(const float *a, size_t n) {
    return sk_pairwise_sum_f32_scalar(a, n);
}
#endif

/* sk_sum_sq_pairwise_f32: the same tree over squared elements. */

static inline float sk__sum_sq_leaf_scalar(const float *a, size_t n) {
    float acc[8];
    size_t i, j;
    for (j = 0; j < 8; ++j) { float v = a[j]; acc[j] = v * v; }
    for (i = 8; i + 8 <= n; i += 8)
        for (j = 0; j < 8; ++j) { float v = a[i + j]; acc[j] += v * v; }
    {
        float s = 0.0f, r;
        for (; i < n; ++i) { float v = a[i]; s += v * v; }
        r = ((acc[0] + acc[1]) + (acc[2] + acc[3]))
          + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        return r + s;
    }
}

static inline float sk_sum_sq_pairwise_f32_scalar(const float *a, size_t n) {
    if (n <= 8) {
        float s = 0.0f;
        size_t i;
        for (i = 0; i < n; ++i) { float v = a[i]; s += v * v; }
        return s;
    }
    if (n <= 128) return sk__sum_sq_leaf_scalar(a, n);
    {
        size_t half = n / 2;
        half -= half % 8;
        return sk_sum_sq_pairwise_f32_scalar(a, half)
             + sk_sum_sq_pairwise_f32_scalar(a + half, n - half);
    }
}

#if SK_HAVE_NEON
static inline float sk__sum_sq_leaf_neon(const float *a, size_t n) {
    float32x4_t a0 = vld1q_f32(a), a1 = vld1q_f32(a + 4);
    float32x4_t q0 = vmulq_f32(a0, a0), q1 = vmulq_f32(a1, a1);
    size_t i;
    for (i = 8; i + 8 <= n; i += 8) {
        float32x4_t b0 = vld1q_f32(a + i), b1 = vld1q_f32(a + i + 4);
        q0 = vaddq_f32(q0, vmulq_f32(b0, b0));
        q1 = vaddq_f32(q1, vmulq_f32(b1, b1));
    }
    {
        float s = 0.0f, r;
        for (; i < n; ++i) { float v = a[i]; s += v * v; }
        {
            float32x4_t t = vpaddq_f32(q0, q1);
            float32x4_t u = vpaddq_f32(t, t);
            r = vgetq_lane_f32(u, 0) + vgetq_lane_f32(u, 1);
        }
        return r + s;
    }
}

static inline float sk_sum_sq_pairwise_f32(const float *a, size_t n) {
    if (n <= 8) {
        float s = 0.0f;
        size_t i;
        for (i = 0; i < n; ++i) { float v = a[i]; s += v * v; }
        return s;
    }
    if (n <= 128) return sk__sum_sq_leaf_neon(a, n);
    {
        size_t half = n / 2;
        half -= half % 8;
        return sk_sum_sq_pairwise_f32(a, half) + sk_sum_sq_pairwise_f32(a + half, n - half);
    }
}
#else
static inline float sk_sum_sq_pairwise_f32(const float *a, size_t n) {
    return sk_sum_sq_pairwise_f32_scalar(a, n);
}
#endif

/* sk_coherence_ema_gate_f32: post-filter coherence EMA update and the ERLE coherence-gate threshold fused into one per-bin pass.
 * Loop 1 (per k): cross term pr = er*nr + ei*ni, pi = ei*nr - er*ni (plain mul/add); EMAs of sye_re/sye_im/syy/see with weight af and (1-a) computed once.
 * Loop 2 (per k): sye2 = sye_re^2 + sye_im^2; denom = max(syy*see, 1e-30f) by compare plus select; mask[k] = (sye2/denom >= threshold).
 * Fusion is order-preserving: loop 2 at k reads only values loop 1 just wrote at the same k. */

static inline void sk__coherence_ema_gate_elem(
    float *sye_re, float *sye_im, float *syy, float *see,
    float er, float ei, float nr, float ni,
    float abs_echo, float abs_near,
    float alpha, float omaf, float threshold,
    unsigned char *mask_out) {
    float pr = er * nr + ei * ni;
    float pi = ei * nr - er * ni;
    float echo_abs2 = abs_echo * abs_echo;
    float near_abs2 = abs_near * abs_near;
    float sye2, denom, g2;
    *sye_re = omaf * (*sye_re) + alpha * pr;
    *sye_im = omaf * (*sye_im) + alpha * pi;
    *syy    = omaf * (*syy)    + alpha * echo_abs2;
    *see    = omaf * (*see)    + alpha * near_abs2;
    sye2 = (*sye_re) * (*sye_re) + (*sye_im) * (*sye_im);
    denom = (*syy) * (*see);
    if (denom < 1.0e-30f) denom = 1.0e-30f;
    g2 = sye2 / denom;
    *mask_out = (g2 >= threshold) ? (unsigned char)1 : (unsigned char)0;
}

static inline void sk_coherence_ema_gate_f32_scalar(
    float *sye_re, float *sye_im, float *syy, float *see,
    const Complex *echo, const Complex *near_spec,
    const float *abs_echo, const float *abs_near,
    float alpha, float threshold,
    unsigned char *mask, int n) {
    int i;
    float omaf = 1.0f - alpha;
    for (i = 0; i < n; ++i) {
        sk__coherence_ema_gate_elem(&sye_re[i], &sye_im[i], &syy[i], &see[i],
                                     echo[i].r, echo[i].i,
                                     near_spec[i].r, near_spec[i].i,
                                     abs_echo[i], abs_near[i],
                                     alpha, omaf, threshold, &mask[i]);
    }
}

#if SK_HAVE_NEON
static inline void sk_coherence_ema_gate_f32(
    float *sye_re, float *sye_im, float *syy, float *see,
    const Complex *echo, const Complex *near_spec,
    const float *abs_echo, const float *abs_near,
    float alpha, float threshold,
    unsigned char *mask, int n) {
    int i = 0;
    float omaf = 1.0f - alpha;
    float32x4_t va = vdupq_n_f32(alpha), vomaf = vdupq_n_f32(omaf);
    float32x4_t vfloor = vdupq_n_f32(1.0e-30f);
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t ev = sk__cquad_load(echo + i);
        float32x4_t er = ev.val[0], ei = ev.val[1];
        float32x4x2_t nv = sk__cquad_load(near_spec + i);
        float32x4_t nr = nv.val[0], ni = nv.val[1];
        float32x4_t pr = vaddq_f32(vmulq_f32(er, nr), vmulq_f32(ei, ni));
        float32x4_t pi = vsubq_f32(vmulq_f32(ei, nr), vmulq_f32(er, ni));

        float32x4_t abs_echo_v = vld1q_f32(abs_echo + i);
        float32x4_t abs_near_v = vld1q_f32(abs_near + i);
        float32x4_t echo_abs2 = vmulq_f32(abs_echo_v, abs_echo_v);
        float32x4_t near_abs2 = vmulq_f32(abs_near_v, abs_near_v);

        float32x4_t sye_re_v = vld1q_f32(sye_re + i);
        float32x4_t sye_im_v = vld1q_f32(sye_im + i);
        float32x4_t syy_v = vld1q_f32(syy + i);
        float32x4_t see_v = vld1q_f32(see + i);

        sye_re_v = vaddq_f32(vmulq_f32(vomaf, sye_re_v), vmulq_f32(va, pr));
        sye_im_v = vaddq_f32(vmulq_f32(vomaf, sye_im_v), vmulq_f32(va, pi));
        syy_v    = vaddq_f32(vmulq_f32(vomaf, syy_v), vmulq_f32(va, echo_abs2));
        see_v    = vaddq_f32(vmulq_f32(vomaf, see_v), vmulq_f32(va, near_abs2));

        vst1q_f32(sye_re + i, sye_re_v);
        vst1q_f32(sye_im + i, sye_im_v);
        vst1q_f32(syy + i, syy_v);
        vst1q_f32(see + i, see_v);

        {
            float32x4_t sye2 = vaddq_f32(vmulq_f32(sye_re_v, sye_re_v),
                                          vmulq_f32(sye_im_v, sye_im_v));
            float32x4_t denom = vmulq_f32(syy_v, see_v);
            uint32x4_t lt = vcltq_f32(denom, vfloor);
            float32x4_t g2;
            float g2_arr[4];
            int j;
            denom = vbslq_f32(lt, vfloor, denom);
            g2 = vdivq_f32(sye2, denom);
            vst1q_f32(g2_arr, g2);
            for (j = 0; j < 4; ++j)
                mask[i + j] = (g2_arr[j] >= threshold) ? (unsigned char)1
                                                        : (unsigned char)0;
        }
    }
    for (; i < n; ++i) {
        sk__coherence_ema_gate_elem(&sye_re[i], &sye_im[i], &syy[i], &see[i],
                                     echo[i].r, echo[i].i,
                                     near_spec[i].r, near_spec[i].i,
                                     abs_echo[i], abs_near[i],
                                     alpha, omaf, threshold, &mask[i]);
    }
}
#else
static inline void sk_coherence_ema_gate_f32(
    float *sye_re, float *sye_im, float *syy, float *see,
    const Complex *echo, const Complex *near_spec,
    const float *abs_echo, const float *abs_near,
    float alpha, float threshold,
    unsigned char *mask, int n) {
    sk_coherence_ema_gate_f32_scalar(sye_re, sye_im, syy, see, echo, near_spec,
                                      abs_echo, abs_near, alpha, threshold,
                                      mask, n);
}
#endif

/* sk_ema_delta_f32: state[i] = state[i] + alpha*(x[i]-state[i]) with a separate subtract, multiply and add (delta form; not interchangeable with sk_ema_f32). */

static inline void sk_ema_delta_f32_scalar(float *state, const float *x,
                                            float alpha, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        float diff = x[i] - state[i];
        state[i] = state[i] + alpha * diff;
    }
}

#if SK_HAVE_NEON
static inline void sk_ema_delta_f32(float *state, const float *x,
                                     float alpha, int n) {
    int i = 0;
    float32x4_t va = vdupq_n_f32(alpha);
    for (; i + 4 <= n; i += 4) {
        float32x4_t sv = vld1q_f32(state + i);
        float32x4_t xv = vld1q_f32(x + i);
        float32x4_t diff = vsubq_f32(xv, sv);
        float32x4_t r = vaddq_f32(sv, vmulq_f32(va, diff));
        vst1q_f32(state + i, r);
    }
    for (; i < n; ++i) {
        float diff = x[i] - state[i];
        state[i] = state[i] + alpha * diff;
    }
}
#else
static inline void sk_ema_delta_f32(float *state, const float *x,
                                     float alpha, int n) {
    sk_ema_delta_f32_scalar(state, x, alpha, n);
}
#endif

/* sk_n2_track_f32: CNG N2 tracking:
 *   track = (fresh*y2s[k] + retain*n2[k]) * g_up; up = n2[k] * g_up; n2[k] = (y2s[k] < n2[k]) ? track : up  (plain ops, exact IEEE '<'). */

static inline void sk_n2_track_f32_scalar(float *n2, const float *y2s,
                                           float fresh, float retain,
                                           float g_up, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        float track = (fresh * y2s[i] + retain * n2[i]) * g_up;
        float up = n2[i] * g_up;
        n2[i] = (y2s[i] < n2[i]) ? track : up;
    }
}

#if SK_HAVE_NEON
static inline void sk_n2_track_f32(float *n2, const float *y2s,
                                    float fresh, float retain,
                                    float g_up, int n) {
    int i = 0;
    float32x4_t vfresh = vdupq_n_f32(fresh), vretain = vdupq_n_f32(retain);
    float32x4_t vgup = vdupq_n_f32(g_up);
    for (; i + 4 <= n; i += 4) {
        float32x4_t y2v = vld1q_f32(y2s + i);
        float32x4_t n2v = vld1q_f32(n2 + i);
        float32x4_t track = vmulq_f32(
            vaddq_f32(vmulq_f32(vfresh, y2v), vmulq_f32(vretain, n2v)), vgup);
        float32x4_t up = vmulq_f32(n2v, vgup);
        uint32x4_t lt = vcltq_f32(y2v, n2v);
        float32x4_t r = vbslq_f32(lt, track, up);
        vst1q_f32(n2 + i, r);
    }
    for (; i < n; ++i) {
        float track = (fresh * y2s[i] + retain * n2[i]) * g_up;
        float up = n2[i] * g_up;
        n2[i] = (y2s[i] < n2[i]) ? track : up;
    }
}
#else
static inline void sk_n2_track_f32(float *n2, const float *y2s,
                                    float fresh, float retain,
                                    float g_up, int n) {
    sk_n2_track_f32_scalar(n2, y2s, fresh, retain, g_up, n);
}
#endif

/* sk_n2_initial_track_f32: CNG N2-initial slow tracking:
 *   slow = n2i[k] + ia*(n2[k]-n2i[k]); n2i[k] = (n2[k] > n2i[k]) ? slow : n2[k], comparing against the original n2i[k]. */

static inline void sk_n2_initial_track_f32_scalar(float *n2i, const float *n2,
                                                    float alpha, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        float old = n2i[i];
        float slow = old + alpha * (n2[i] - old);
        n2i[i] = (n2[i] > old) ? slow : n2[i];
    }
}

#if SK_HAVE_NEON
static inline void sk_n2_initial_track_f32(float *n2i, const float *n2,
                                            float alpha, int n) {
    int i = 0;
    float32x4_t va = vdupq_n_f32(alpha);
    for (; i + 4 <= n; i += 4) {
        float32x4_t oldv = vld1q_f32(n2i + i);
        float32x4_t n2v = vld1q_f32(n2 + i);
        float32x4_t diff = vsubq_f32(n2v, oldv);
        float32x4_t slow = vaddq_f32(oldv, vmulq_f32(va, diff));
        uint32x4_t gt = vcgtq_f32(n2v, oldv);
        float32x4_t r = vbslq_f32(gt, slow, n2v);
        vst1q_f32(n2i + i, r);
    }
    for (; i < n; ++i) {
        float old = n2i[i];
        float slow = old + alpha * (n2[i] - old);
        n2i[i] = (n2[i] > old) ? slow : n2[i];
    }
}
#else
static inline void sk_n2_initial_track_f32(float *n2i, const float *n2,
                                            float alpha, int n) {
    sk_n2_initial_track_f32_scalar(n2i, n2, alpha, n);
}
#endif

/* sk_mask_zero_f32: out[i] = mask[i] ? 0.0f : out[i], in place; any nonzero mask byte zeroes the element. */

static inline void sk_mask_zero_f32_scalar(float *x, const unsigned char *mask,
                                            int n) {
    int i;
    for (i = 0; i < n; ++i) if (mask[i]) x[i] = 0.0f;
}

#if SK_HAVE_NEON
static inline void sk_mask_zero_f32(float *x, const unsigned char *mask, int n) {
    int i = 0;
    float32x4_t zero = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        uint32_t m[4];
        int j;
        float32x4_t xv = vld1q_f32(x + i);
        uint32x4_t mv;
        for (j = 0; j < 4; ++j) m[j] = mask[i + j] ? 0xFFFFFFFFu : 0u;
        mv = vld1q_u32(m);
        vst1q_f32(x + i, vbslq_f32(mv, zero, xv));
    }
    for (; i < n; ++i) if (mask[i]) x[i] = 0.0f;
}
#else
static inline void sk_mask_zero_f32(float *x, const unsigned char *mask, int n) {
    sk_mask_zero_f32_scalar(x, mask, n);
}
#endif

/* sk_pairwise_sum_tailfold_f32: numpy-style pairwise float32 sum, variant A (far-end hop energy, linear-output block energies).
 * Differs from sk_pairwise_sum_f32: the n <= 128 leaf is entered directly (n == 8 takes the eight-accumulator path), and the n % 8 tail is folded
 * one element at a time into the combined root. The n < 8 path starts at 0.0f, so a lone -0.0f gives +0.0f.
 * The NEON leaf uses the same q0/q1 accumulation and vpaddq_f32 combine as sk_pairwise_sum_f32, then the scalar tail fold. */

/* gcc for 32-bit targets cannot bound n by 128 at the tail loop after inlining and reports -Waggressive-loop-optimizations; the leaf is only reached with n <= 128.
 * The pragma is scoped to this leaf and does not change code generation. */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"
#endif
static inline float sk__pairwise_sum_tailfold_leaf_scalar(const float *a, size_t n) {
    if (n < 8) {
        float s = 0.0f;
        size_t i;
        for (i = 0; i < n; ++i) s += a[i];
        return s;
    }
    {
        float acc[8];
        size_t i, j;
        float res;
        for (j = 0; j < 8; ++j) acc[j] = a[j];
        for (i = 8; i + 8 <= n; i += 8)
            for (j = 0; j < 8; ++j) acc[j] += a[i + j];
        res = ((acc[0] + acc[1]) + (acc[2] + acc[3]))
            + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

static inline float sk_pairwise_sum_tailfold_f32_scalar(const float *a, size_t n) {
    if (n <= 128) return sk__pairwise_sum_tailfold_leaf_scalar(a, n);
    {
        size_t half = n / 2;
        half -= half % 8;
        return sk_pairwise_sum_tailfold_f32_scalar(a, half)
             + sk_pairwise_sum_tailfold_f32_scalar(a + half, n - half);
    }
}

#if SK_HAVE_NEON
static inline float sk__pairwise_sum_tailfold_leaf_neon(const float *a, size_t n) {
    if (n < 8) {
        float s = 0.0f;
        size_t i;
        for (i = 0; i < n; ++i) s += a[i];
        return s;
    }
    {
        float32x4_t q0 = vld1q_f32(a), q1 = vld1q_f32(a + 4);
        size_t i;
        float res;
        for (i = 8; i + 8 <= n; i += 8) {
            q0 = vaddq_f32(q0, vld1q_f32(a + i));
            q1 = vaddq_f32(q1, vld1q_f32(a + i + 4));
        }
        {
            float32x4_t t = vpaddq_f32(q0, q1);
            float32x4_t u = vpaddq_f32(t, t);
            res = vgetq_lane_f32(u, 0) + vgetq_lane_f32(u, 1);
        }
        for (; i < n; ++i) res += a[i];
        return res;
    }
}

static inline float sk_pairwise_sum_tailfold_f32(const float *a, size_t n) {
    if (n <= 128) return sk__pairwise_sum_tailfold_leaf_neon(a, n);
    {
        size_t half = n / 2;
        half -= half % 8;
        return sk_pairwise_sum_tailfold_f32(a, half)
             + sk_pairwise_sum_tailfold_f32(a + half, n - half);
    }
}
#else
static inline float sk_pairwise_sum_tailfold_f32(const float *a, size_t n) {
    return sk_pairwise_sum_tailfold_f32_scalar(a, n);
}
#endif

/* sk_pairwise_sum_tailfold_b_f32: variant B (floor accumulation, fullband ERLE and ERL sums, reverb ratio, suppression-gain slices).
 * Same leaf, tail fold and split as variant A; differs only in the n < 8 path: n == 0 returns 0.0f and n in [1,7] starts the accumulator at a[0],
 * so a lone -0.0f is returned unchanged (variant A returns +0.0f). */

static inline float sk_pairwise_sum_tailfold_b_f32_scalar(const float *a, size_t n) {
    if (n == 0) return 0.0f;
    if (n < 8) {
        float res = a[0];
        size_t i;
        for (i = 1; i < n; ++i) res = res + a[i];
        return res;
    }
    if (n <= 128) {
        float acc[8];
        size_t i, j;
        float res;
        for (j = 0; j < 8; ++j) acc[j] = a[j];
        for (i = 8; i + 8 <= n; i += 8)
            for (j = 0; j < 8; ++j) acc[j] = acc[j] + a[i + j];
        res = ((acc[0] + acc[1]) + (acc[2] + acc[3]))
            + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        for (; i < n; ++i) res = res + a[i];
        return res;
    }
    {
        size_t n2 = n / 2;
        n2 -= n2 % 8;
        return sk_pairwise_sum_tailfold_b_f32_scalar(a, n2)
             + sk_pairwise_sum_tailfold_b_f32_scalar(a + n2, n - n2);
    }
}

#if SK_HAVE_NEON
static inline float sk_pairwise_sum_tailfold_b_f32(const float *a, size_t n) {
    if (n == 0) return 0.0f;
    if (n < 8) {
        float res = a[0];
        size_t i;
        for (i = 1; i < n; ++i) res = res + a[i];
        return res;
    }
    if (n <= 128) {
        float32x4_t q0 = vld1q_f32(a), q1 = vld1q_f32(a + 4);
        size_t i;
        float res;
        for (i = 8; i + 8 <= n; i += 8) {
            q0 = vaddq_f32(q0, vld1q_f32(a + i));
            q1 = vaddq_f32(q1, vld1q_f32(a + i + 4));
        }
        {
            float32x4_t t = vpaddq_f32(q0, q1);
            float32x4_t u = vpaddq_f32(t, t);
            res = vgetq_lane_f32(u, 0) + vgetq_lane_f32(u, 1);
        }
        for (; i < n; ++i) res = res + a[i];
        return res;
    }
    {
        size_t n2 = n / 2;
        n2 -= n2 % 8;
        return sk_pairwise_sum_tailfold_b_f32(a, n2)
             + sk_pairwise_sum_tailfold_b_f32(a + n2, n - n2);
    }
}
#else
static inline float sk_pairwise_sum_tailfold_b_f32(const float *a, size_t n) {
    return sk_pairwise_sum_tailfold_b_f32_scalar(a, n);
}
#endif

/* sk_noise_spectrum_update_f32: StationarityEstimator per-bin noise floor tracker (steady-state branch):
 *   if (pb > pn) { alpha_inc = alpha * (pn / max(pb, 1e-30f)); if (apply_mask10 && 10*pn < pb) alpha_inc *= 0.1f; noise = pn + alpha_inc*(pb - pn); }
 *   else { noise = max(pn + alpha*(pb - pn), min_noise); }
 * apply_mask10 is a call-level boolean broadcast to a mask. The maxes are compare (>) plus select, never vmaxq_f32; both branches are computed
 * and the speculative divide of a discarded lane is masked out by the final select. Inputs are finite by contract. */

static inline float sk__noise_spectrum_update_elem(float pb, float pn,
                                                     float alpha, int apply_mask10,
                                                     float min_noise) {
    if (pb > pn) {
        float denom = pb > 1e-30f ? pb : 1e-30f;
        float ratio = pn / denom;
        float alpha_inc = alpha * ratio;
        if (apply_mask10) {
            float ten_pn = 10.0f * pn;
            if (ten_pn < pb) alpha_inc = alpha_inc * 0.1f;
        }
        return pn + alpha_inc * (pb - pn);
    } else {
        float upd = pn + alpha * (pb - pn);
        return upd > min_noise ? upd : min_noise;
    }
}

static inline void sk_noise_spectrum_update_f32_scalar(float *noise, const float *spectrum,
                                                         float alpha, int apply_mask10,
                                                         float min_noise, int n) {
    int k;
    for (k = 0; k < n; ++k)
        noise[k] = sk__noise_spectrum_update_elem(spectrum[k], noise[k], alpha,
                                                   apply_mask10, min_noise);
}

#if SK_HAVE_NEON
static inline void sk_noise_spectrum_update_f32(float *noise, const float *spectrum,
                                                 float alpha, int apply_mask10,
                                                 float min_noise, int n) {
    int k = 0;
    float32x4_t valpha = vdupq_n_f32(alpha), veps = vdupq_n_f32(1.0e-30f);
    float32x4_t vten = vdupq_n_f32(10.0f), vpoint1 = vdupq_n_f32(0.1f);
    float32x4_t vminn = vdupq_n_f32(min_noise);
    uint32x4_t venable10 = apply_mask10 ? vdupq_n_u32(0xFFFFFFFFu) : vdupq_n_u32(0);
    for (; k + 4 <= n; k += 4) {
        float32x4_t vpb = vld1q_f32(spectrum + k);
        float32x4_t vpn = vld1q_f32(noise + k);
        float32x4_t vdiff = vsubq_f32(vpb, vpn); /* pb-pn, shared by both branches */
        uint32x4_t gt_eps = vcgtq_f32(vpb, veps);
        float32x4_t vdenom = vbslq_f32(gt_eps, vpb, veps);
        float32x4_t vratio = vdivq_f32(vpn, vdenom);
        float32x4_t valpha_inc = vmulq_f32(valpha, vratio);
        {
            uint32x4_t mask10 = vandq_u32(venable10, vcltq_f32(vmulq_f32(vten, vpn), vpb));
            float32x4_t scaled = vmulq_f32(valpha_inc, vpoint1);
            valpha_inc = vbslq_f32(mask10, scaled, valpha_inc);
        }
        float32x4_t vrising = vaddq_f32(vpn, vmulq_f32(valpha_inc, vdiff));
        float32x4_t vupd = vaddq_f32(vpn, vmulq_f32(valpha, vdiff));
        uint32x4_t gt_floor = vcgtq_f32(vupd, vminn);
        float32x4_t vfalling = vbslq_f32(gt_floor, vupd, vminn);
        uint32x4_t rising_sel = vcgtq_f32(vpb, vpn);
        vst1q_f32(noise + k, vbslq_f32(rising_sel, vrising, vfalling));
    }
    for (; k < n; ++k)
        noise[k] = sk__noise_spectrum_update_elem(spectrum[k], noise[k], alpha,
                                                   apply_mask10, min_noise);
}
#else
static inline void sk_noise_spectrum_update_f32(float *noise, const float *spectrum,
                                                 float alpha, int apply_mask10,
                                                 float min_noise, int n) {
    sk_noise_spectrum_update_f32_scalar(noise, spectrum, alpha, apply_mask10, min_noise, n);
}
#endif

/* sk_erl_bin_update_f32: ErlEstimator per-bin minimum-statistics update, fused into one masked pass, for k = 1..n_bins-2:
 *   if (x2[k] > x2_min) { new_erl = y2[k] / x2[k]; if (new_erl < erl[k]) { hold_counters[k-1] = hold_hops;
 *     erl[k] += 0.1f * (new_erl - erl[k]); erl[k] = max(erl[k], ERL_MIN_ERL) by compare plus select; } }
 * The caller advances erl/x2/y2 by +1 and passes hold_counters unshifted. The two conditions are separate lane masks (the second against the
 * original erl[k]) ANDed into one mask that gates both the erl and the hold_counters store. The divide is speculative for every lane;
 * a NaN or Inf new_erl never passes the ordered `<`, so it is never stored. */

static inline void sk__erl_bin_update_elem(float *erl, int *hold,
                                            float x2v, float y2v,
                                            float x2_min, int hold_hops,
                                            float min_erl) {
    if (x2v > x2_min) {
        float new_erl = y2v / x2v;
        if (new_erl < *erl) {
            float diff = new_erl - *erl;
            float delta = 0.1f * diff;
            *hold = hold_hops;
            *erl = *erl + delta;
            if (*erl < min_erl) *erl = min_erl;
        }
    }
}

static inline void sk_erl_bin_update_f32_scalar(float *erl, int *hold,
                                                  const float *x2, const float *y2,
                                                  float x2_min, int hold_hops,
                                                  float min_erl, int n) {
    int j;
    for (j = 0; j < n; ++j)
        sk__erl_bin_update_elem(&erl[j], &hold[j], x2[j], y2[j], x2_min, hold_hops, min_erl);
}

#if SK_HAVE_NEON
static inline void sk_erl_bin_update_f32(float *erl, int *hold,
                                          const float *x2, const float *y2,
                                          float x2_min, int hold_hops,
                                          float min_erl, int n) {
    int j = 0;
    float32x4_t vx2min  = vdupq_n_f32(x2_min);
    float32x4_t vminerl = vdupq_n_f32(min_erl);
    float32x4_t vpoint1 = vdupq_n_f32(0.1f);
    int32x4_t   vholdhops = vdupq_n_s32(hold_hops);
    for (; j + 4 <= n; j += 4) {
        float32x4_t vx2  = vld1q_f32(x2 + j);
        float32x4_t vy2  = vld1q_f32(y2 + j);
        float32x4_t verl = vld1q_f32(erl + j);
        int32x4_t   vhold = vld1q_s32(hold + j);

        uint32x4_t cond1 = vcgtq_f32(vx2, vx2min);
        float32x4_t vnew_erl = vdivq_f32(vy2, vx2);
        uint32x4_t cond2 = vcltq_f32(vnew_erl, verl);
        uint32x4_t mask = vandq_u32(cond1, cond2);

        float32x4_t diff     = vsubq_f32(vnew_erl, verl);
        float32x4_t delta    = vmulq_f32(vpoint1, diff);
        float32x4_t blended  = vaddq_f32(verl, delta);
        uint32x4_t floormask = vcltq_f32(blended, vminerl);
        float32x4_t floored  = vbslq_f32(floormask, vminerl, blended);

        float32x4_t erl_result  = vbslq_f32(mask, floored, verl);
        int32x4_t   hold_result = vbslq_s32(mask, vholdhops, vhold);

        vst1q_f32(erl + j, erl_result);
        vst1q_s32(hold + j, hold_result);
    }
    for (; j < n; ++j)
        sk__erl_bin_update_elem(&erl[j], &hold[j], x2[j], y2[j], x2_min, hold_hops, min_erl);
}
#else
static inline void sk_erl_bin_update_f32(float *erl, int *hold,
                                          const float *x2, const float *y2,
                                          float x2_min, int hold_hops,
                                          float min_erl, int n) {
    sk_erl_bin_update_f32_scalar(erl, hold, x2, y2, x2_min, hold_hops, min_erl, n);
}
#endif

/* Decrement every int counter, saturating at INT_MIN (a zero floor would alter raw counter values and INT_MIN avoids signed-overflow UB).
 * The NEON subtract wraps modulo 2^32 and the wrapped value is masked off for lanes already at INT_MIN. */

static inline void sk_dec1_floorintmin_s32_scalar(int *x, int n) {
    int i;
    for (i = 0; i < n; ++i)
        if (x[i] > INT_MIN) x[i] -= 1;
}

#if SK_HAVE_NEON
static inline void sk_dec1_floorintmin_s32(int *x, int n) {
    int i = 0;
    int32x4_t vintmin = vdupq_n_s32(INT_MIN);
    int32x4_t vone    = vdupq_n_s32(1);
    for (; i + 4 <= n; i += 4) {
        int32x4_t v    = vld1q_s32(x + i);
        uint32x4_t mask = vcgtq_s32(v, vintmin);
        int32x4_t  dec  = vsubq_s32(v, vone);
        int32x4_t  result = vbslq_s32(mask, dec, v);
        vst1q_s32(x + i, result);
    }
    for (; i < n; ++i)
        if (x[i] > INT_MIN) x[i] -= 1;
}
#else
static inline void sk_dec1_floorintmin_s32(int *x, int n) {
    sk_dec1_floorintmin_s32_scalar(x, n);
}
#endif

/* sk_erl_hold_expire_f32: bins whose hold counter expired double toward ERL_MAX_ERL, for k = 1..n_bins-2:
 *   if (hold_counters[k-1] <= 0) { doubled = 2.0f * erl[k]; erl[k] = (doubled < ERL_MAX_ERL) ? doubled : ERL_MAX_ERL; }
 * Same caller-offset convention as sk_erl_bin_update_f32. It must run after sk_dec1_floorintmin_s32 has updated the counters, not fused with either.
 * The clamp is compare plus select (never vminq_f32); a NaN `doubled` fails the compare on both paths and selects ERL_MAX_ERL. */

static inline void sk__erl_hold_expire_elem(float *erl, int hold, float max_erl) {
    if (hold <= 0) {
        float doubled = 2.0f * (*erl);
        *erl = (doubled < max_erl) ? doubled : max_erl;
    }
}

static inline void sk_erl_hold_expire_f32_scalar(float *erl, const int *hold,
                                                   float max_erl, int n) {
    int j;
    for (j = 0; j < n; ++j)
        sk__erl_hold_expire_elem(&erl[j], hold[j], max_erl);
}

#if SK_HAVE_NEON
static inline void sk_erl_hold_expire_f32(float *erl, const int *hold,
                                           float max_erl, int n) {
    int j = 0;
    float32x4_t vmax  = vdupq_n_f32(max_erl);
    float32x4_t vtwo  = vdupq_n_f32(2.0f);
    int32x4_t   vzero = vdupq_n_s32(0);
    for (; j + 4 <= n; j += 4) {
        float32x4_t verl  = vld1q_f32(erl + j);
        int32x4_t   vhold = vld1q_s32(hold + j);
        uint32x4_t mask = vcleq_s32(vhold, vzero);
        float32x4_t doubled = vmulq_f32(vtwo, verl);
        uint32x4_t lt = vcltq_f32(doubled, vmax);
        float32x4_t clamped = vbslq_f32(lt, doubled, vmax);
        float32x4_t result = vbslq_f32(mask, clamped, verl);
        vst1q_f32(erl + j, result);
    }
    for (; j < n; ++j)
        sk__erl_hold_expire_elem(&erl[j], hold[j], max_erl);
}
#else
static inline void sk_erl_hold_expire_f32(float *erl, const int *hold,
                                           float max_erl, int n) {
    sk_erl_hold_expire_f32_scalar(erl, hold, max_erl, n);
}
#endif

/* sk_no_audible_echo_gain_f32: per-bin soft-blend body of the no-audible-echo gain (soft_blend_enabled and soft_blend_per_bin set):
 *   enr = echo/(nearend+1); sig = clamp((enr - thr)/softness, -50, 50); ne_wb = 1/(1 + fast_exp(sig));
 *   blended threshold_k = ne_wb * nearend_k + (1 - ne_wb) * normal_k for each of the three thresholds; emr = echo/(masker+1);
 *   g = 1; if (enr > enr_tr_k && emr > emr_tr_k) g = max((enr_su_k - enr)/max(enr_su_k - enr_tr_k, 1e-30f), emr_tr_k/max(emr, 1e-30f)).
 * The exp is sk__fast_exp_vec, so the kernel exists only with SK_HAVE_NEON and not USE_STANDARD_MATH; other builds use the scalar entry point.
 * No vfmaq anywhere. All selects are compare plus select (never vminq/vmaxq), so a NaN g_lin yields g_emr as the C ternary does.
 * Both arms are computed speculatively; the discarded arm may produce Inf or NaN, which is never selected. */

static inline float sk__no_audible_echo_gain_elem(
    float nearend, float echo, float masker,
    float ne_enr_tr, float ne_enr_su, float ne_emr_tr,
    float no_enr_tr, float no_enr_su, float no_emr_tr,
    float blend_thr, float blend_softness) {
    float enr_bin = echo / (nearend + 1.0f);
    float sig = (enr_bin - blend_thr) / blend_softness;
    float ne_wb, enr_tr_k, enr_su_k, emr_tr_k, enr, emr, g;
    if (sig < -50.0f) sig = -50.0f;
    if (sig > 50.0f) sig = 50.0f;
    ne_wb = 1.0f / (1.0f + sk__fast_exp_elem(sig));
    enr_tr_k = (ne_wb * ne_enr_tr + (1.0f - ne_wb) * no_enr_tr);
    enr_su_k = (ne_wb * ne_enr_su + (1.0f - ne_wb) * no_enr_su);
    emr_tr_k = (ne_wb * ne_emr_tr + (1.0f - ne_wb) * no_emr_tr);
    enr = echo / (nearend + 1.0f);
    emr = echo / (masker + 1.0f);
    g = 1.0f;
    if (enr > enr_tr_k && emr > emr_tr_k) {
        float d_lin = enr_su_k - enr_tr_k;
        float denom_lin = d_lin < 1e-30f ? 1e-30f : d_lin;
        float g_lin = (enr_su_k - enr) / denom_lin;
        float denom_emr = emr < 1e-30f ? 1e-30f : emr;
        float g_emr = emr_tr_k / denom_emr;
        g = g_lin > g_emr ? g_lin : g_emr;
    }
    return g;
}

static inline void sk_no_audible_echo_gain_f32_scalar(
    const float *nearend, const float *echo, const float *masker,
    const float *ne_enr_tr, const float *ne_enr_su, const float *ne_emr_tr,
    const float *no_enr_tr, const float *no_enr_su, const float *no_emr_tr,
    float blend_thr, float blend_softness, float *out, int n) {
    int k;
    for (k = 0; k < n; ++k)
        out[k] = sk__no_audible_echo_gain_elem(
            nearend[k], echo[k], masker[k],
            ne_enr_tr[k], ne_enr_su[k], ne_emr_tr[k],
            no_enr_tr[k], no_enr_su[k], no_emr_tr[k],
            blend_thr, blend_softness);
}

#if SK_HAVE_NEON && !defined(USE_STANDARD_MATH)
static inline void sk_no_audible_echo_gain_f32(
    const float *nearend, const float *echo, const float *masker,
    const float *ne_enr_tr, const float *ne_enr_su, const float *ne_emr_tr,
    const float *no_enr_tr, const float *no_enr_su, const float *no_emr_tr,
    float blend_thr, float blend_softness, float *out, int n) {
    int k = 0;
    float32x4_t one   = vdupq_n_f32(1.0f);
    float32x4_t vthr  = vdupq_n_f32(blend_thr);
    float32x4_t vsoft = vdupq_n_f32(blend_softness);
    float32x4_t vlo   = vdupq_n_f32(-50.0f);
    float32x4_t vhi   = vdupq_n_f32(50.0f);
    float32x4_t vtiny = vdupq_n_f32(1e-30f);
    for (; k + 4 <= n; k += 4) {
        float32x4_t ne = vld1q_f32(nearend + k);
        float32x4_t ec = vld1q_f32(echo + k);
        float32x4_t mk = vld1q_f32(masker + k);
        float32x4_t enr_bin = vdivq_f32(ec, vaddq_f32(ne, one));
        float32x4_t sig = vdivq_f32(vsubq_f32(enr_bin, vthr), vsoft);
        sig = vbslq_f32(vcltq_f32(sig, vlo), vlo, sig);
        sig = vbslq_f32(vcgtq_f32(sig, vhi), vhi, sig);
        {
            float32x4_t ne_wb = vdivq_f32(one, vaddq_f32(one, sk__fast_exp_vec(sig)));
            float32x4_t inv   = vsubq_f32(one, ne_wb);
            float32x4_t enr_tr = vaddq_f32(vmulq_f32(ne_wb, vld1q_f32(ne_enr_tr + k)),
                                           vmulq_f32(inv,   vld1q_f32(no_enr_tr + k)));
            float32x4_t enr_su = vaddq_f32(vmulq_f32(ne_wb, vld1q_f32(ne_enr_su + k)),
                                           vmulq_f32(inv,   vld1q_f32(no_enr_su + k)));
            float32x4_t emr_tr = vaddq_f32(vmulq_f32(ne_wb, vld1q_f32(ne_emr_tr + k)),
                                           vmulq_f32(inv,   vld1q_f32(no_emr_tr + k)));
            float32x4_t enr = vdivq_f32(ec, vaddq_f32(ne, one));
            float32x4_t emr = vdivq_f32(ec, vaddq_f32(mk, one));
            uint32x4_t  cond = vandq_u32(vcgtq_f32(enr, enr_tr),
                                         vcgtq_f32(emr, emr_tr));
            float32x4_t d_lin = vsubq_f32(enr_su, enr_tr);
            float32x4_t den_l = vbslq_f32(vcltq_f32(d_lin, vtiny), vtiny, d_lin);
            float32x4_t g_lin = vdivq_f32(vsubq_f32(enr_su, enr), den_l);
            float32x4_t den_e = vbslq_f32(vcltq_f32(emr, vtiny), vtiny, emr);
            float32x4_t g_emr = vdivq_f32(emr_tr, den_e);
            float32x4_t gmax  = vbslq_f32(vcgtq_f32(g_lin, g_emr), g_lin, g_emr);
            vst1q_f32(out + k, vbslq_f32(cond, gmax, one));
        }
    }
    for (; k < n; ++k)
        out[k] = sk__no_audible_echo_gain_elem(
            nearend[k], echo[k], masker[k],
            ne_enr_tr[k], ne_enr_su[k], ne_emr_tr[k],
            no_enr_tr[k], no_enr_su[k], no_emr_tr[k],
            blend_thr, blend_softness);
}
#else
static inline void sk_no_audible_echo_gain_f32(
    const float *nearend, const float *echo, const float *masker,
    const float *ne_enr_tr, const float *ne_enr_su, const float *ne_emr_tr,
    const float *no_enr_tr, const float *no_enr_su, const float *no_emr_tr,
    float blend_thr, float blend_softness, float *out, int n) {
    sk_no_audible_echo_gain_f32_scalar(nearend, echo, masker,
                                        ne_enr_tr, ne_enr_su, ne_emr_tr,
                                        no_enr_tr, no_enr_su, no_emr_tr,
                                        blend_thr, blend_softness, out, n);
}
#endif

/* sk_max_abs_f32: largest |x[i]|: m = 0.0f; for i: a = fabsf(x[i]); if (a > m) m = a. A NaN never raises the maximum; n == 0 gives 0.0f.
 * (np.max(np.abs(x)) returns NaN on NaN input; the two agree on NaN-free input.) NEON keeps compare plus select, not vmaxq/vmaxnmq. */

static inline float sk_max_abs_f32_scalar(const float *x, int n) {
    float m = 0.0f;
    int i;
    for (i = 0; i < n; ++i) {
        float a = fabsf(x[i]);
        if (a > m) m = a;
    }
    return m;
}

#if SK_HAVE_NEON
static inline float sk_max_abs_f32(const float *x, int n) {
    float m = 0.0f;
    int i = 0;
    if (n >= 4) {
        float32x4_t m0 = vdupq_n_f32(0.0f), m1 = m0;
        for (; i + 8 <= n; i += 8) {
            float32x4_t a0 = vabsq_f32(vld1q_f32(x + i));
            float32x4_t a1 = vabsq_f32(vld1q_f32(x + i + 4));
            m0 = vbslq_f32(vcgtq_f32(a0, m0), a0, m0);
            m1 = vbslq_f32(vcgtq_f32(a1, m1), a1, m1);
        }
        for (; i + 4 <= n; i += 4) {
            float32x4_t a0 = vabsq_f32(vld1q_f32(x + i));
            m0 = vbslq_f32(vcgtq_f32(a0, m0), a0, m0);
        }
        m = vmaxvq_f32(vbslq_f32(vcgtq_f32(m1, m0), m1, m0));
    }
    for (; i < n; ++i) {
        float a = fabsf(x[i]);
        if (a > m) m = a;
    }
    return m;
}
#else
static inline float sk_max_abs_f32(const float *x, int n) {
    return sk_max_abs_f32_scalar(x, n);
}
#endif

/* sk_noise_floor_track_f32: per-bin stationary noise-floor tracker with a hold counter:
 *   if (psd[k] < floor[k]) { floor[k] = psd[k]; count[k] = 0; }
 *   else if (count[k] >= hold) { ramp = floor[k] * growth; floor[k] = (ramp > min_floor) ? ramp : min_floor; }
 *   else { count[k] += 1; }
 * NEON evaluates all arms and selects with the same compares (NaN psd or floor takes the hold/ramp arms); the speculative count + 1 wraps harmlessly. */

static inline void sk__noise_floor_track_elem(float *fl, int *cnt, float psd,
                                              float growth, float min_floor,
                                              int hold) {
    if (psd < *fl) {
        *fl = psd;
        *cnt = 0;
    } else if (*cnt >= hold) {
        float ramp = *fl * growth;
        *fl = (ramp > min_floor) ? ramp : min_floor;
    } else {
        *cnt += 1;
    }
}

static inline void sk_noise_floor_track_f32_scalar(float *fl, int *cnt, const float *psd,
                                                   float growth, float min_floor,
                                                   int hold, int n) {
    int k;
    for (k = 0; k < n; ++k)
        sk__noise_floor_track_elem(&fl[k], &cnt[k], psd[k], growth, min_floor, hold);
}

#if SK_HAVE_NEON
static inline void sk_noise_floor_track_f32(float *fl, int *cnt, const float *psd,
                                            float growth, float min_floor,
                                            int hold, int n) {
    int k = 0;
    const float32x4_t vgrowth = vdupq_n_f32(growth);
    const float32x4_t vmin = vdupq_n_f32(min_floor);
    const int32x4_t vhold = vdupq_n_s32(hold);
    const int32x4_t vone = vdupq_n_s32(1);
    for (; k + 4 <= n; k += 4) {
        float32x4_t f = vld1q_f32(fl + k);
        float32x4_t p = vld1q_f32(psd + k);
        int32x4_t c = vld1q_s32(cnt + k);
        uint32x4_t down = vcltq_f32(p, f);
        uint32x4_t expired = vcgeq_s32(c, vhold);
        float32x4_t ramp = vmulq_f32(f, vgrowth);
        float32x4_t ramped = vbslq_f32(vcgtq_f32(ramp, vmin), ramp, vmin);
        float32x4_t f_new = vbslq_f32(down, p, vbslq_f32(expired, ramped, f));
        int32x4_t c_new = vbslq_s32(down, vdupq_n_s32(0),
                                    vbslq_s32(expired, c, vaddq_s32(c, vone)));
        vst1q_f32(fl + k, f_new);
        vst1q_s32(cnt + k, c_new);
    }
    for (; k < n; ++k)
        sk__noise_floor_track_elem(&fl[k], &cnt[k], psd[k], growth, min_floor, hold);
}
#else
static inline void sk_noise_floor_track_f32(float *fl, int *cnt, const float *psd,
                                            float growth, float min_floor,
                                            int hold, int n) {
    sk_noise_floor_track_f32_scalar(fl, cnt, psd, growth, min_floor, hold, n);
}
#endif

/* sk_raise_to_f32: in-place per-bin lower bound, if (x[k] < lo[k]) x[k] = lo[k]. NaN operands and ties leave x untouched. x and lo must not overlap. */

static inline void sk_raise_to_f32_scalar(float *x, const float *lo, int n) {
    int k;
    for (k = 0; k < n; ++k)
        if (x[k] < lo[k]) x[k] = lo[k];
}

#if SK_HAVE_NEON
static inline void sk_raise_to_f32(float *x, const float *lo, int n) {
    int k = 0;
    for (; k + 4 <= n; k += 4) {
        float32x4_t v = vld1q_f32(x + k);
        float32x4_t b = vld1q_f32(lo + k);
        vst1q_f32(x + k, vbslq_f32(vcltq_f32(v, b), b, v));
    }
    for (; k < n; ++k)
        if (x[k] < lo[k]) x[k] = lo[k];
}
#else
static inline void sk_raise_to_f32(float *x, const float *lo, int n) {
    sk_raise_to_f32_scalar(x, lo, n);
}
#endif

/* sk_stationary_mask_u8: m[k] = (flags[k] != 0) && (hang[k] == 0) as a 0/1 byte; returns the number of set bins. `mask` may be NULL when only the count is wanted. */

static inline int sk__stationary_elem(unsigned char flag, int32_t hang) {
    return flag != 0 && hang == 0;
}

static inline int sk_stationary_mask_u8_scalar(const unsigned char *flags,
                                               const int32_t *hang,
                                               unsigned char *mask, int n) {
    int k, count = 0;
    for (k = 0; k < n; ++k) {
        int m = sk__stationary_elem(flags[k], hang[k]);
        if (mask) mask[k] = (unsigned char)m;
        count += m;
    }
    return count;
}

#if SK_HAVE_NEON
static inline int sk_stationary_mask_u8(const unsigned char *flags,
                                        const int32_t *hang,
                                        unsigned char *mask, int n) {
    int k = 0, count = 0;
    /* each byte lane gains at most 1 per step, so it is folded into the count before it can pass 255 */
    while (k + 16 <= n) {
        int steps = (n - k) / 16, j;
        uint8x16_t acc = vdupq_n_u8(0);
        if (steps > 255) steps = 255;
        for (j = 0; j < steps; ++j, k += 16) {
            uint8x16_t f = vld1q_u8(flags + k);
            uint16x8_t z01 = vcombine_u16(vmovn_u32(vceqzq_s32(vld1q_s32(hang + k))),
                                          vmovn_u32(vceqzq_s32(vld1q_s32(hang + k + 4))));
            uint16x8_t z23 = vcombine_u16(vmovn_u32(vceqzq_s32(vld1q_s32(hang + k + 8))),
                                          vmovn_u32(vceqzq_s32(vld1q_s32(hang + k + 12))));
            uint8x16_t z = vcombine_u8(vmovn_u16(z01), vmovn_u16(z23));
            uint8x16_t m = vshrq_n_u8(vandq_u8(vtstq_u8(f, f), z), 7);
            if (mask) vst1q_u8(mask + k, m);
            acc = vaddq_u8(acc, m);
        }
        count += (int)vaddlvq_u8(acc);
    }
    for (; k < n; ++k) {
        int m = sk__stationary_elem(flags[k], hang[k]);
        if (mask) mask[k] = (unsigned char)m;
        count += m;
    }
    return count;
}
#else
static inline int sk_stationary_mask_u8(const unsigned char *flags,
                                        const int32_t *hang,
                                        unsigned char *mask, int n) {
    return sk_stationary_mask_u8_scalar(flags, hang, mask, n);
}
#endif


/* Per-bin H-error leakage refresh; returns the number of diverged bins. The caller applies the final H-error clamp. */
static inline int sk_h_error_refresh_f32_scalar(
    float *h, const float *e2, const float *coarse, const float *erl,
    float lc, float ld, float floor, int disallow, int n) {
    int diverged = 0;
    for (int k = 0; k < n; ++k) {
        int conv = (e2[k] <= coarse[k]) || disallow;
        if (!conv) ++diverged;
        float leakage = conv ? lc : ld;
        float er = erl[k];
        if (floor > 0.0f && er < floor) er = floor;
        h[k] = h[k] + leakage * er;
    }
    return diverged;
}

static inline int sk_h_error_refresh_f32(
    float *h, const float *e2, const float *coarse, const float *erl,
    float lc, float ld, float floor, int disallow, int n) {
    int k = 0, diverged = 0;
#if SK_HAVE_NEON
    const float32x4_t vlc = vdupq_n_f32(lc), vld = vdupq_n_f32(ld);
    const float32x4_t vfloor = vdupq_n_f32(floor);
    const uint32x4_t vdis = vdupq_n_u32(disallow ? ~0u : 0u);
    uint32x4_t acc = vdupq_n_u32(0);
    const uint32x4_t one = vdupq_n_u32(1);
    for (; k + 4 <= n; k += 4) {
        uint32x4_t conv = vorrq_u32(vcleq_f32(vld1q_f32(e2 + k),
                                            vld1q_f32(coarse + k)), vdis);
        float32x4_t leakage = vbslq_f32(conv, vlc, vld);
        float32x4_t er = vld1q_f32(erl + k);
        if (floor > 0.0f)
            er = vbslq_f32(vcltq_f32(er, vfloor), vfloor, er);
        vst1q_f32(h + k, vaddq_f32(vld1q_f32(h + k), vmulq_f32(leakage, er)));
        acc = vaddq_u32(acc, vandq_u32(vmvnq_u32(conv), one));
    }
    diverged = (int)vaddvq_u32(acc);
#endif
    return diverged + sk_h_error_refresh_f32_scalar(
        h + k, e2 + k, coarse + k, erl + k, lc, ld, floor, disallow, n - k);
}

/* Accumulate power spectra and latch canonical 0/1 low-render flags. */
static inline void sk_erle_accumulate_f32_scalar(
    float *ya, float *ea, unsigned char *low,
    const float *x, const float *y, const float *e, float threshold, int n) {
    for (int k = 0; k < n; ++k) {
        ya[k] = ya[k] + y[k];
        ea[k] = ea[k] + e[k];
        if (x[k] < threshold) low[k] = 1;
    }
}

static inline void sk_erle_accumulate_f32(
    float *ya, float *ea, unsigned char *low,
    const float *x, const float *y, const float *e, float threshold, int n) {
    int k = 0;
#if SK_HAVE_NEON
    const float32x4_t thr = vdupq_n_f32(threshold);
    const uint8x8_t one = vdup_n_u8(1);
    for (; k + 8 <= n; k += 8) {
        vst1q_f32(ya + k, vaddq_f32(vld1q_f32(ya + k), vld1q_f32(y + k)));
        vst1q_f32(ya + k + 4, vaddq_f32(vld1q_f32(ya + k + 4), vld1q_f32(y + k + 4)));
        vst1q_f32(ea + k, vaddq_f32(vld1q_f32(ea + k), vld1q_f32(e + k)));
        vst1q_f32(ea + k + 4, vaddq_f32(vld1q_f32(ea + k + 4), vld1q_f32(e + k + 4)));
        uint16x8_t m16 = vcombine_u16(vmovn_u32(vcltq_f32(vld1q_f32(x + k), thr)),
                                     vmovn_u32(vcltq_f32(vld1q_f32(x + k + 4), thr)));
        uint8x8_t m8 = vand_u8(vmovn_u16(m16), one);
        vst1_u8(low + k, vorr_u8(vld1_u8(low + k), m8));
    }
#endif
    sk_erle_accumulate_f32_scalar(ya + k, ea + k, low + k,
                                 x + k, y + k, e + k, threshold, n - k);
}

/* Release ERLE hold counters and re-arm onset detection. Pass interior bins. */
static inline void sk_erle_release_f32_scalar(
    float *oc, const float *during, int32_t *hold, unsigned char *coming,
    int threshold, float decay, int n) {
    for (int k = 0; k < n; ++k) {
        hold[k] -= 1;
        if (hold[k] <= threshold) {
            if (oc[k] > during[k]) {
                float decayed = decay * oc[k];
                oc[k] = decayed > during[k] ? decayed : during[k];
            }
            if (hold[k] <= 0) {
                coming[k] = 1;
                hold[k] = 0;
            }
        }
    }
}

static inline void sk_erle_release_f32(
    float *oc, const float *during, int32_t *hold, unsigned char *coming,
    int threshold, float decay, int n) {
    int k = 0;
#if SK_HAVE_NEON
    const int32x4_t vthr = vdupq_n_s32(threshold);
    const int32x4_t vone = vdupq_n_s32(1), vzero = vdupq_n_s32(0);
    const float32x4_t vdecay = vdupq_n_f32(decay);
    const uint8x8_t one8 = vdup_n_u8(1);
    for (; k + 8 <= n; k += 8) {
        uint32x4_t re_arm[2];
        for (int g = 0; g < 2; ++g) {
            int i = k + 4 * g;
            int32x4_t hc = vsubq_s32(vld1q_s32(hold + i), vone);
            uint32x4_t below = vcleq_s32(hc, vthr);
            float32x4_t o = vld1q_f32(oc + i), d = vld1q_f32(during + i);
            float32x4_t decayed = vmulq_f32(vdecay, o);
            float32x4_t mx = vbslq_f32(vcgtq_f32(decayed, d), decayed, d);
            uint32x4_t upd = vandq_u32(below, vcgtq_f32(o, d));
            vst1q_f32(oc + i, vbslq_f32(upd, mx, o));
            re_arm[g] = vandq_u32(below, vcleq_s32(hc, vzero));
            vst1q_s32(hold + i, vbslq_s32(re_arm[g], vzero, hc));
        }
        uint8x8_t m8 = vand_u8(vmovn_u16(vcombine_u16(
            vmovn_u32(re_arm[0]), vmovn_u32(re_arm[1]))), one8);
        vst1_u8(coming + k, vorr_u8(vld1_u8(coming + k), m8));
    }
#endif
    sk_erle_release_f32_scalar(oc + k, during + k, hold + k, coming + k,
                               threshold, decay, n - k);
}

/* Each counter corresponds to psd[k+1]; psd has n+2 elements. Greater-than selects keep the scalar neighbour-NaN rule. */
static inline void sk_peak_counters_i64_scalar(
    int64_t *counters, const float *psd, float ratio, int n) {
    for (int k = 0; k < n; ++k) {
        float neighbour = psd[k] > psd[k + 2] ? psd[k] : psd[k + 2];
        counters[k] = psd[k + 1] > ratio * neighbour ? counters[k] + 1 : 0;
    }
}

static inline void sk_peak_counters_i64(
    int64_t *counters, const float *psd, float ratio, int n) {
    int k = 0;
#if SK_HAVE_NEON
    const float32x4_t vr = vdupq_n_f32(ratio);
    const int64x2_t one = vdupq_n_s64(1);
    for (; k + 4 <= n; k += 4) {
        float32x4_t c = vld1q_f32(psd + k + 1);
        float32x4_t l = vld1q_f32(psd + k), r = vld1q_f32(psd + k + 2);
        float32x4_t neighbour = vbslq_f32(vcgtq_f32(l, r), l, r);
        int32x4_t hit = vreinterpretq_s32_u32(vcgtq_f32(c, vmulq_f32(vr, neighbour)));
        int64x2_t lo = vmovl_s32(vget_low_s32(hit)), hi = vmovl_s32(vget_high_s32(hit));
        vst1q_s64(counters + k, vandq_s64(vaddq_s64(vld1q_s64(counters + k), one), lo));
        vst1q_s64(counters + k + 2, vandq_s64(vaddq_s64(vld1q_s64(counters + k + 2), one), hi));
    }
#endif
    sk_peak_counters_i64_scalar(counters + k, psd + k, ratio, n - k);
}

/* Flags are canonical 0/1; hangovers are nonnegative. */
static inline void sk_stationarity_hangover_i32_scalar(
    const unsigned char *flags, int32_t *hang, int32_t hold, int n) {
    int all = 1;
    for (int k = 0; k < n; ++k)
        if (!flags[k]) { all = 0; break; }
    for (int k = 0; k < n; ++k)
        if (!flags[k]) hang[k] = hold;
    if (all) {
        for (int k = 0; k < n; ++k) {
            int32_t v = hang[k] - 1;
            hang[k] = v > 0 ? v : 0;
        }
    }
}

static inline void sk_stationarity_hangover_i32(
    const unsigned char *flags, int32_t *hang, int32_t hold, int n) {
#if SK_HAVE_NEON
    int k = 0, all = 1;
    uint8x16_t mn = vdupq_n_u8(1);
    for (; k + 16 <= n; k += 16) mn = vminq_u8(mn, vld1q_u8(flags + k));
    if (vminvq_u8(mn) == 0) all = 0;
    for (; k < n; ++k)
        if (!flags[k]) { all = 0; break; }
    k = 0;
    const int32x4_t vhold = vdupq_n_s32(hold), vzero = vdupq_n_s32(0), vone = vdupq_n_s32(1);
    for (; k + 8 <= n; k += 8) {
        uint16x8_t f16 = vmovl_u8(vld1_u8(flags + k));
        uint32x4_t st0 = vcgtq_u32(vmovl_u16(vget_low_u16(f16)), vdupq_n_u32(0));
        uint32x4_t st1 = vcgtq_u32(vmovl_u16(vget_high_u16(f16)), vdupq_n_u32(0));
        int32x4_t h0 = vbslq_s32(st0, vld1q_s32(hang + k), vhold);
        int32x4_t h1 = vbslq_s32(st1, vld1q_s32(hang + k + 4), vhold);
        if (all) {
            h0 = vmaxq_s32(vsubq_s32(h0, vone), vzero);
            h1 = vmaxq_s32(vsubq_s32(h1, vone), vzero);
        }
        vst1q_s32(hang + k, h0);
        vst1q_s32(hang + k + 4, h1);
    }
    for (; k < n; ++k) {
        if (!flags[k]) hang[k] = hold;
        if (all) {
            int32_t v = hang[k] - 1;
            hang[k] = v > 0 ? v : 0;
        }
    }
#else
    sk_stationarity_hangover_i32_scalar(flags, hang, hold, n);
#endif
}

/* In-place three-bin logical AND of canonical 0/1 flags; endpoints copy the adjacent interior. */
static inline void sk_stationarity_smooth_u8_scalar(unsigned char *f, int n) {
    if (n >= 3) {
        unsigned char prev = f[0], cur = f[1], first = 0, last = 0;
        for (int k = 1; k <= n - 2; ++k) {
            unsigned char next = f[k + 1];
            unsigned char sm = (unsigned char)(prev && cur && next);
            if (k == 1) first = sm;
            if (k == n - 2) last = sm;
            f[k] = sm;
            prev = cur;
            cur = next;
        }
        f[0] = first;
        f[n - 1] = last;
    } else {
        for (int k = 0; k < n; ++k) f[k] = 0;
    }
}

static inline void sk_stationarity_smooth_u8(unsigned char *f, int n) {
#if SK_HAVE_NEON
    if (n >= 3) {
        unsigned char prev = f[0], cur = f[1], first = 0, last = 0;
        int k = 1;
        uint8x16_t carry = vsetq_lane_u8(f[0], vdupq_n_u8(0), 15);
        for (; k + 16 <= n - 1; k += 16) {
            uint8x16_t orig = vld1q_u8(f + k);
            uint8x16_t left = vextq_u8(carry, orig, 15), right = vld1q_u8(f + k + 1);
            uint8x16_t sm = vandq_u8(vandq_u8(left, orig), right);
            if (k == 1) first = vgetq_lane_u8(sm, 0);
            vst1q_u8(f + k, sm);
            carry = orig;
        }
        if (k > 1) {
            prev = vgetq_lane_u8(carry, 15);
            cur = f[k];
            last = f[k - 1];
        }
        for (; k <= n - 2; ++k) {
            unsigned char next = f[k + 1];
            unsigned char sm = (unsigned char)(prev && cur && next);
            if (k == 1) first = sm;
            if (k == n - 2) last = sm;
            f[k] = sm;
            prev = cur;
            cur = next;
        }
        f[0] = first;
        f[n - 1] = last;
    } else {
        for (int k = 0; k < n; ++k) f[k] = 0;
    }
#else
    sk_stationarity_smooth_u8_scalar(f, n);
#endif
}

/* Per-bin threshold: ratio * (window_hops * max(noise, 1e-30f)). */
static inline void sk_stationarity_flags_u8_scalar(
    unsigned char *flags, const float *acc, const float *noise,
    int window_hops, float ratio, int n) {
    for (int k = 0; k < n; ++k) {
        float nz = noise[k];
        float nd = nz > 1e-30f ? nz : 1e-30f;
        float noise_scaled = (float)window_hops * nd;
        float threshold = ratio * noise_scaled;
        flags[k] = acc[k] < threshold ? 1 : 0;
    }
}

static inline void sk_stationarity_flags_u8(
    unsigned char *flags, const float *acc, const float *noise,
    int window_hops, float ratio, int n) {
    int k = 0;
#if SK_HAVE_NEON
    const float32x4_t tiny = vdupq_n_f32(1e-30f);
    const float32x4_t wh = vdupq_n_f32((float)window_hops), vr = vdupq_n_f32(ratio);
    const uint8x8_t one = vdup_n_u8(1);
    for (; k + 8 <= n; k += 8) {
        uint32x4_t lt[2];
        for (int g = 0; g < 2; ++g) {
            float32x4_t nz = vld1q_f32(noise + k + 4 * g);
            float32x4_t nd = vbslq_f32(vcgtq_f32(nz, tiny), nz, tiny);
            float32x4_t threshold = vmulq_f32(vr, vmulq_f32(wh, nd));
            lt[g] = vcltq_f32(vld1q_f32(acc + k + 4 * g), threshold);
        }
        vst1_u8(flags + k, vand_u8(vmovn_u16(vcombine_u16(
            vmovn_u32(lt[0]), vmovn_u32(lt[1]))), one));
    }
#endif
    sk_stationarity_flags_u8_scalar(flags + k, acc + k, noise + k, window_hops, ratio, n - k);
}

/* sk_stationarity_window_flags_u8: flags[k] = (sum < ratio * (window_hops * max(noise[k], 1e-30f))) ? 1 : 0, where
 * sum = ((hist[k] + hist[stride + k]) + ... + hist[(rows - 1) * stride + k]) in row order, plus rev[k] last when rev is non-NULL
 * (rows >= 1). The NEON twin keeps each bin group's sum in registers over all rows; the scalar twin accumulates 128-bin blocks row by row. */
static inline void sk_stationarity_window_flags_u8_scalar(
    unsigned char *flags, const float *hist, size_t stride, int rows, const float *rev,
    const float *noise, int window_hops, float ratio, int n) {
    enum { BLOCK = 128 };   /* bins per accumulator block: rows are read contiguously; the block size does not change the sums */
    float acc[BLOCK];
    for (int base = 0; base < n; base += BLOCK) {
        int m = n - base < BLOCK ? n - base : BLOCK;
        for (int k = 0; k < m; ++k) acc[k] = hist[base + k];
        for (int r = 1; r < rows; ++r) {
            const float *row = hist + (size_t)r * stride + base;
            for (int k = 0; k < m; ++k) acc[k] = acc[k] + row[k];
        }
        if (rev)
            for (int k = 0; k < m; ++k) acc[k] = acc[k] + rev[base + k];
        sk_stationarity_flags_u8_scalar(flags + base, acc, noise + base, window_hops, ratio, m);
    }
}

static inline void sk_stationarity_window_flags_u8(
    unsigned char *flags, const float *hist, size_t stride, int rows, const float *rev,
    const float *noise, int window_hops, float ratio, int n) {
    int k = 0;
#if SK_HAVE_NEON
    const float32x4_t tiny = vdupq_n_f32(1e-30f);
    const float32x4_t wh = vdupq_n_f32((float)window_hops), vr = vdupq_n_f32(ratio);
    const uint8x8_t one = vdup_n_u8(1);
    for (; k + 8 <= n; k += 8) {
        float32x4_t a0 = vld1q_f32(hist + k), a1 = vld1q_f32(hist + k + 4);
        uint32x4_t lt[2];
        for (int r = 1; r < rows; ++r) {
            const float *row = hist + (size_t)r * stride + k;
            a0 = vaddq_f32(a0, vld1q_f32(row));
            a1 = vaddq_f32(a1, vld1q_f32(row + 4));
        }
        if (rev) {
            a0 = vaddq_f32(a0, vld1q_f32(rev + k));
            a1 = vaddq_f32(a1, vld1q_f32(rev + k + 4));
        }
        {
            float32x4_t nz0 = vld1q_f32(noise + k), nz1 = vld1q_f32(noise + k + 4);
            float32x4_t nd0 = vbslq_f32(vcgtq_f32(nz0, tiny), nz0, tiny);
            float32x4_t nd1 = vbslq_f32(vcgtq_f32(nz1, tiny), nz1, tiny);
            lt[0] = vcltq_f32(a0, vmulq_f32(vr, vmulq_f32(wh, nd0)));
            lt[1] = vcltq_f32(a1, vmulq_f32(vr, vmulq_f32(wh, nd1)));
        }
        vst1_u8(flags + k, vand_u8(vmovn_u16(vcombine_u16(vmovn_u32(lt[0]), vmovn_u32(lt[1]))), one));
    }
#endif
    sk_stationarity_window_flags_u8_scalar(flags + k, hist + k, stride, rows, rev ? rev + k : NULL,
                                            noise + k, window_hops, ratio, n - k);
}

/* sk_row_sum_f32: out[k] = ((rows[k] + rows[stride + k]) + ...) + rows[(n_rows - 1) * stride + k], rows added in order (n_rows >= 1).
 * out must not overlap rows. The NEON twin keeps 16 bins of the sum in registers over all rows. */
static inline void sk_row_sum_f32_scalar(float *out, const float *rows, size_t stride, int n_rows, int n) {
    for (int k = 0; k < n; ++k) out[k] = rows[k];
    for (int r = 1; r < n_rows; ++r) {
        const float *row = rows + (size_t)r * stride;
        for (int k = 0; k < n; ++k) out[k] = out[k] + row[k];
    }
}

static inline void sk_row_sum_f32(float *out, const float *rows, size_t stride, int n_rows, int n) {
    int k = 0;
#if SK_HAVE_NEON
    for (; k + 16 <= n; k += 16) {
        float32x4_t a0 = vld1q_f32(rows + k), a1 = vld1q_f32(rows + k + 4);
        float32x4_t a2 = vld1q_f32(rows + k + 8), a3 = vld1q_f32(rows + k + 12);
        for (int r = 1; r < n_rows; ++r) {
            const float *row = rows + (size_t)r * stride + k;
            a0 = vaddq_f32(a0, vld1q_f32(row));
            a1 = vaddq_f32(a1, vld1q_f32(row + 4));
            a2 = vaddq_f32(a2, vld1q_f32(row + 8));
            a3 = vaddq_f32(a3, vld1q_f32(row + 12));
        }
        vst1q_f32(out + k, a0); vst1q_f32(out + k + 4, a1);
        vst1q_f32(out + k + 8, a2); vst1q_f32(out + k + 12, a3);
    }
    for (; k + 4 <= n; k += 4) {
        float32x4_t a = vld1q_f32(rows + k);
        for (int r = 1; r < n_rows; ++r) a = vaddq_f32(a, vld1q_f32(rows + (size_t)r * stride + k));
        vst1q_f32(out + k, a);
    }
#endif
    sk_row_sum_f32_scalar(out + k, rows + k, stride, n_rows, n - k);
}

/* Integer counts of clipped samples and consecutive equal near-peak pairs. */
static inline void sk_saturation_counts_f32_scalar(
    const float *signal, size_t n, float threshold, float high,
    long *clips, long *consecutive) {
    long c = 0, p = 0;
    for (size_t i = 0; i < n; ++i)
        if (fabsf(signal[i]) > threshold) c++;
    for (size_t i = 1; i < n; ++i)
        if (fabsf(signal[i]) > high && fabsf(signal[i - 1]) > high &&
            fabsf(signal[i] - signal[i - 1]) < 1e-6f) p++;
    *clips = c;
    *consecutive = p;
}

static inline void sk_saturation_counts_f32(
    const float *signal, size_t n, float threshold, float high,
    long *clips, long *consecutive) {
#if SK_HAVE_NEON
    const float32x4_t thr = vdupq_n_f32(threshold), vh = vdupq_n_f32(high);
    const float32x4_t tiny = vdupq_n_f32(1e-6f);
    uint32x4_t ca = vdupq_n_u32(0), pa = vdupq_n_u32(0);
    const uint32x4_t one = vdupq_n_u32(1);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t cur = vld1q_f32(signal + i);
        ca = vaddq_u32(ca, vandq_u32(vcgtq_f32(vabsq_f32(cur), thr), one));
    }
    long c = (long)vaddvq_u32(ca);
    for (size_t k = i; k < n; ++k)
        if (fabsf(signal[k]) > threshold) c++;
    i = 1;
    for (; i + 4 <= n; i += 4) {
        float32x4_t cur = vld1q_f32(signal + i), prev = vld1q_f32(signal + i - 1);
        uint32x4_t m = vandq_u32(vandq_u32(vcgtq_f32(vabsq_f32(cur), vh),
                                         vcgtq_f32(vabsq_f32(prev), vh)),
                                vcltq_f32(vabsq_f32(vsubq_f32(cur, prev)), tiny));
        pa = vaddq_u32(pa, vandq_u32(m, one));
    }
    long p = (long)vaddvq_u32(pa);
    for (; i < n; ++i)
        if (fabsf(signal[i]) > high && fabsf(signal[i - 1]) > high &&
            fabsf(signal[i] - signal[i - 1]) < 1e-6f) p++;
    *clips = c;
    *consecutive = p;
#else
    sk_saturation_counts_f32_scalar(signal, n, threshold, high, clips, consecutive);
#endif
}

/* sk_cng_inject_f32: per bin one step of s = (s * 69069 + 1) & 0x7FFFFFFF, ix = s >> 26, then spec += ng * (amp * lut[ix]) (re) and ng * (amp * lut[(ix + 8) & 31]) (im),
 * ng = fast_sqrt(max(1 - g_res^2, 0)) times max(g_nr, nr_floor) when g_nr is non-NULL (nr_floor not NaN). The NEON twin steps four bins side by side by jump-ahead
 * mod 2^32 (only bits 0..30 feed back): the same draws and final seed for any start value. */

#define SK_CNG_LCG_A 69069u
#define SK_CNG_LCG_MASK 0x7FFFFFFFu

static inline void sk_cng_inject_f32_scalar(uint32_t *seed, Complex *spec, const float *amp,
                                             const float *g_res, const float *g_nr, float nr_floor,
                                             const float *lut, int n) {
    uint32_t s = *seed;
    int k;
    for (k = 0; k < n; ++k) {
        uint32_t ix;
        float ng, cn_re, cn_im, t;
        s = (s * SK_CNG_LCG_A + 1u) & SK_CNG_LCG_MASK;
        ix = s >> 26;
        cn_re = amp[k] * lut[ix];
        cn_im = amp[k] * lut[(ix + 8u) & 31u];
        t = 1.0f - g_res[k] * g_res[k];
        if (t < 0.0f) t = 0.0f;
        ng = sk__fast_sqrt_elem(t);
        if (g_nr) ng = ng * fmaxf(g_nr[k], nr_floor);
        spec[k].r += ng * cn_re;
        spec[k].i += ng * cn_im;
    }
    *seed = s;
}

#if SK_HAVE_NEON
static inline void sk_cng_inject_f32(uint32_t *seed, Complex *spec, const float *amp,
                                      const float *g_res, const float *g_nr, float nr_floor,
                                      const float *lut, int n) {
    /* A^1..A^4 and A^(j-1) + ... + 1 for j = 1..4, mod 2^32. */
    static const uint32_t jump_a[4] = { 69069u, 475559465u, 2801775573u, 1790562961u };
    static const uint32_t jump_c[4] = { 1u, 69070u, 475628535u, 3277404108u };
    int k = 0;
    uint32_t s = *seed;
    if (n >= 4) {
        uint32x4_t sv = vaddq_u32(vmulq_u32(vdupq_n_u32(s), vld1q_u32(jump_a)), vld1q_u32(jump_c));
        const uint32x4_t a4 = vdupq_n_u32(jump_a[3]), c4 = vdupq_n_u32(jump_c[3]);
        const uint32x4_t m31 = vdupq_n_u32(31u);
        const float32x4_t one = vdupq_n_f32(1.0f), zero = vdupq_n_f32(0.0f), vfloor = vdupq_n_f32(nr_floor);
        for (; k + 4 <= n; k += 4) {
            uint32_t ix[4];
            float32x4_t sin_v, cos_v, amp_v, g, t, ng;
            float32x4x2_t sp;
            vst1q_u32(ix, vandq_u32(vshrq_n_u32(sv, 26), m31));
            sin_v = vld1q_dup_f32(lut + ix[0]);
            sin_v = vld1q_lane_f32(lut + ix[1], sin_v, 1);
            sin_v = vld1q_lane_f32(lut + ix[2], sin_v, 2);
            sin_v = vld1q_lane_f32(lut + ix[3], sin_v, 3);
            cos_v = vld1q_dup_f32(lut + ((ix[0] + 8u) & 31u));
            cos_v = vld1q_lane_f32(lut + ((ix[1] + 8u) & 31u), cos_v, 1);
            cos_v = vld1q_lane_f32(lut + ((ix[2] + 8u) & 31u), cos_v, 2);
            cos_v = vld1q_lane_f32(lut + ((ix[3] + 8u) & 31u), cos_v, 3);
            s = vgetq_lane_u32(sv, 3);
            sv = vaddq_u32(vmulq_u32(sv, a4), c4);
            amp_v = vld1q_f32(amp + k);
            g = vld1q_f32(g_res + k);
            t = vsubq_f32(one, vmulq_f32(g, g));
            t = vbslq_f32(vcltq_f32(t, zero), zero, t);
#ifdef USE_STANDARD_MATH
            ng = vsqrtq_f32(t);
#else
            ng = vbslq_f32(vcgtq_f32(t, zero), vsqrtq_f32(t), zero);
#endif
            if (g_nr) {
                float32x4_t nr = vld1q_f32(g_nr + k);
                ng = vmulq_f32(ng, vbslq_f32(vcgtq_f32(nr, vfloor), nr, vfloor));
            }
            sp = sk__cquad_load(spec + k);
            sp.val[0] = vaddq_f32(sp.val[0], vmulq_f32(ng, vmulq_f32(amp_v, sin_v)));
            sp.val[1] = vaddq_f32(sp.val[1], vmulq_f32(ng, vmulq_f32(amp_v, cos_v)));
            sk__cquad_store(spec + k, sp);
        }
        s &= SK_CNG_LCG_MASK;
    }
    if (k < n) sk_cng_inject_f32_scalar(&s, spec + k, amp + k, g_res + k, g_nr ? g_nr + k : NULL, nr_floor, lut, n - k);
    *seed = s;
}
#else
static inline void sk_cng_inject_f32(uint32_t *seed, Complex *spec, const float *amp,
                                      const float *g_res, const float *g_nr, float nr_floor,
                                      const float *lut, int n) {
    sk_cng_inject_f32_scalar(seed, spec, amp, g_res, g_nr, nr_floor, lut, n);
}
#endif

#ifdef __cplusplus
}
#endif

#endif
