/**
 * AEC-specific scalar/NEON micro-kernels.
 *
 * Filtering, spectral statistics, stationarity and saturation kernels.
 * AEC modules retain their stateful wrappers; generic kernels and the
 * SK_HAVE_NEON switch are provided by simd_kernels.h.
 *
 * For finite inputs every NEON entry point must be byte-identical to its
 * scalar reference. Multi-NaN expressions may select different NaN
 * payloads, but must agree on NaN classification; the self-test
 * (AEC/c_impl/test/simd_selftest_aec.c) reports those separately from hard
 * failures.
 *
 * Keep the scalar operation order: explicit fmaf calls stay fused, ordinary
 * multiply/add expressions stay separate, and including translation units
 * must use -ffp-contract=off. Estimate reciprocal/sqrt instructions and
 * reassociation are not allowed.
 *
 * Partial pointer overlap is unsupported. sk_mask_zero_f32 is explicitly
 * in-place. Complex NEON access must use sk__cquad_load/sk__cquad_store from
 * simd_kernels.h so strict-aliasing remains valid.
 */

#ifndef SIMD_AEC_H
#define SIMD_AEC_H

#include "simd_kernels.h"

#include <limits.h>   /* Hold counters are plain int; saturate at INT_MIN. */

#ifdef __cplusplus
extern "C" {
#endif

/* Shared per-element helpers preserve the same operation order in scalar
 * kernels and NEON scalar tails. Generic fast-sqrt helpers stay in
 * simd_kernels.h. */

/* numpy complex64 |z| (np.abs of a complex64) — scaled hypot with an explicit
 * fmaf; correctly rounded, so identical on every IEEE-754 target. */
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
/* Compare/select reproduces the scalar ternaries, including NaN signs and
 * unordered comparisons. vabsq/vminq/vmaxq would change that contract.
 * Zero-magnitude lanes calculate 0/0 internally, then select exact +0.0f.
 * Finite inputs (including signed zero) remain byte-identical to scalar;
 * the dedicated NaN corpus checks non-finite behavior. */
static inline float32x4_t sk__cabs_np_neon4(float32x4_t re, float32x4_t im) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    uint32x4_t re_neg = vcltq_f32(re, zero);           /* re < 0.0f */
    uint32x4_t im_neg = vcltq_f32(im, zero);           /* im < 0.0f */
    float32x4_t ar = vbslq_f32(re_neg, vnegq_f32(re), re);  /* re<0?-re:re */
    float32x4_t ai = vbslq_f32(im_neg, vnegq_f32(im), im);  /* im<0?-im:im */
    uint32x4_t gt = vcgtq_f32(ar, ai);                 /* ar > ai */
    float32x4_t larger  = vbslq_f32(gt, ar, ai);       /* ar>ai?ar:ai */
    float32x4_t smaller = vbslq_f32(gt, ai, ar);       /* ar>ai?ai:ar */
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
#endif /* SK_HAVE_NEON */

/* ═══════════════════════════════ kernel 1 ══════════════════════════════════
 * sk_cabs_np_f32 — out[i] = numpy |z[i]|. */

static inline void sk_cabs_np_f32_scalar(const Complex *z, float *out, int n) {
    int i;
    for (i = 0; i < n; ++i) out[i] = sk__cabs_np_elem(z[i].r, z[i].i);
}

#if SK_HAVE_NEON
static inline void sk_cabs_np_f32(const Complex *z, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        /* sk__cquad_load (simd_kernels.h) -- memcpy-based Complex-quad
         * load, NOT a direct Complex*->float* cast fed to vld2q_f32 (that
         * cast is a strict-aliasing violation, see simd_kernels.h's
         * "Complex-quad NEON load/store" section for the full writeup). */
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

/* ═══════════════════════════════ kernel 2 ══════════════════════════════════
 * sk_cmag2_np_f32 — out[i] = numpy |z[i]|**2. */

static inline void sk_cmag2_np_f32_scalar(const Complex *z, float *out, int n) {
    int i;
    for (i = 0; i < n; ++i) out[i] = sk__cmag2_np_elem(z[i].r, z[i].i);
}

#if SK_HAVE_NEON
static inline void sk_cmag2_np_f32(const Complex *z, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        /* sk__cquad_load -- see sk_cabs_np_f32 above for the aliasing
         * rationale. */
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

/* ═══════════════════════════════ kernel 3 ══════════════════════════════════
 * sk_cmag2_np_acc_f32 — acc[i] += numpy |z[i]|**2 (the PBFDAF x2_partition_sum
 * / far-power accumulation pattern: acc[k] += cmag2_np(...)). */

static inline void sk_cmag2_np_acc_f32_scalar(const Complex *z, float *acc, int n) {
    int i;
    for (i = 0; i < n; ++i) acc[i] += sk__cmag2_np_elem(z[i].r, z[i].i);
}

#if SK_HAVE_NEON
static inline void sk_cmag2_np_acc_f32(const Complex *z, float *acc, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        /* sk__cquad_load -- see sk_cabs_np_f32 above for the aliasing
         * rationale. */
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

/* ═══════════════════════════════ kernel 7 ══════════════════════════════════
 * sk_wupdate_nlms_f32 — the PBFDAF NLMS coarse-filter W-update, per-bin, for
 * ONE partition (the outer loop-over-partitions stays scalar at the call
 * site; this kernel vectorizes over the K frequency bins of a single
 * partition). Verbatim from pbfdaf_process_with_far_cmag2() in pbfdkf.c:
 *   cxr = xr, cxi = -xi;                                  // conj(X)
 *   gr = fmaf(er, cxr, -(ei * cxi));
 *   gi = fmaf(er, cxi,  (ei * cxr));                      // grad = err*conj(X)
 *   W[k].r += mu_eff[k] * gr;                             // NOT fmaf
 *   W[k].i += mu_eff[k] * gi;                             // NOT fmaf
 * The grad computation is explicit fmaf (fused); the final mu_eff*grad
 * combine is a plain separate multiply then add (needs -ffp-contract=off). */

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
        /* sk__cquad_load/sk__cquad_store -- see sk_cabs_np_f32 above for
         * the aliasing rationale. */
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

/* ═══════════════════════════════ kernel 8 ══════════════════════════════════
 * sk_wupdate_kf_f32 — the PBFDKF Kalman W-update, per-bin, for ONE
 * partition. Verbatim from pbfdkf_update_weights() in pbfdkf.c:
 *   kr = mu[k] * xr;             ki = -(mu[k] * xi);        // K = mu*conj(X)
 *   ksr = kr * mu_scale[k];      ksi = ki * mu_scale[k];     // K *= mu_scale
 *   W[k].r += fmaf(ksr, er, -(ksi * ei));
 *   W[k].i += fmaf(ksr, ei,  (ksi * er));                   // W += K*err
 * The K/K*=mu_scale steps are plain separate multiplies (no add involved,
 * so no fusion risk there either way); the final `W += fmaf(...)` step is
 * itself two operations — the fmaf (fused, explicit) producing the K*err
 * increment, THEN a separate plain add onto the existing W[k] (the source
 * writes `+=`, not a second fmaf) — replicated here as vfmaq_f32 followed
 * by a separate vaddq_f32. */

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
        /* sk__cquad_load/sk__cquad_store -- see sk_cabs_np_f32 above for
         * the aliasing rationale. */
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

/* ═══════════════════════════════ kernel 13 ═════════════════════════════════
 * sk_pairwise_sum_f32 — numpy-1.26-bit-exact pairwise float32 sum
 * (n<=8 serial; n<=128 8-accumulator leaf; recursive split with
 * half -= half%8).
 *
 * NEON leaf math (verified by hand): q0 holds running column-sums for
 * acc[0..3], q1 for acc[4..7] (same 8-wide grouping as the scalar leaf).
 * `t = vpaddq_f32(q0,q1)` = [acc0+acc1, acc2+acc3, acc4+acc5, acc6+acc7].
 * `u = vpaddq_f32(t,t)` = [ (acc0+acc1)+(acc2+acc3), (acc4+acc5)+(acc6+acc7),
 * <repeat> ]. `u[0]+u[1]` therefore equals
 * `((acc0+acc1)+(acc2+acc3)) + ((acc4+acc5)+(acc6+acc7))` — the exact same
 * grouping as the scalar leaf's `r`. */

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

/* ═══════════════════════════════ kernel 14 ═════════════════════════════════
 * sk_sum_sq_pairwise_f32 — same tree as kernel 13, over squared elements.
 * Identical structure/split/combine order; the leaf accumulates
 * a[i+j]*a[i+j] instead of a[i+j]. */

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

/* ═══════════════════════════════ kernel 16 ═════════════════════════════════
 * sk_coherence_ema_gate_f32 — post-filter coherence Γ²(Ŷ,Y) EMA update + the
 * ERLE coh-gate threshold, FUSED into one per-bin pass. Fuses the two per-bin
 * loops (over the same [0,nb) range) of aec_post.c
 * aec_post_compute_coherence() —
 *
 *   loop 1 (per k): pr = er*nr + ei*ni;  pi = ei*nr - er*ni;   // echo *
 *                     conj(near), plain mul/add — source does NOT call
 *                     fmaf for this cross-product.
 *                   sye_re[k] = omaf*sye_re[k] + af*pr;         // plain
 *                   sye_im[k] = omaf*sye_im[k] + af*pi;         // mul/mul/add,
 *                                                                NOT fmaf
 *                   syy[k] = (1.0f-a)*syy[k] + af*abs_echo[k]^2;  // source
 *                   see[k] = (1.0f-a)*see[k] + af*abs_near[k]^2;  // recomputes
 *                     "(1.0f-a)" inline here instead of reusing the `omaf`
 *                     local — bit-identical to omaf regardless (same `a`,
 *                     same deterministic IEEE subtract), so this kernel
 *                     computes omaf once and reuses it for all four EMAs.
 *   loop 2 (per k): sye2 = sye_re[k]^2 + sye_im[k]^2;           // plain
 *                                                                mul/mul/add
 *                   denom = syy[k]*see[k];
 *                   if (denom<1e-30f) denom=1e-30f;             // compare+
 *                     select, not vmaxq (signed-zero risk per header note)
 *                   g2 = sye2/denom;
 *                   mask[k] = (g2 >= threshold) ? 1u : 0u;
 *
 * FUSION SAFETY: loop 2 at index k reads ONLY sye_re[k]/sye_im[k]/syy[k]/
 * see[k] — the values loop 1 just wrote at that SAME index k, never a
 * different index (no cross-bin term anywhere in either loop). So merging
 * into one per-k pass (update-then-gate) is order-preserving: loop 1's
 * write at k happens-before loop 2's read at k in both the original 2-loop
 * form and this fused form. */

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
        /* sk__cquad_load (read-only side of this kernel) -- see
         * sk_cabs_np_f32 above for the aliasing rationale. */
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

/* ═══════════════════════════════ kernel 17 ═════════════════════════════════
 * sk_ema_delta_f32 — state[i] = state[i] + alpha*(x[i]-state[i]) (the
 * "delta-form" EMA — distinct from kernel 4's alpha*state+beta*x shape:
 * different rounding path, NOT interchangeable bit-for-bit). Verbatim from
 * AEC/c_impl/src/aec_post.c aec_post_compute_comfort_noise()'s
 * y2_smoothed update:
 *   p->y2_smoothed[k] = p->y2_smoothed[k]
 *                     + y2a * (p->near_psd[k] - p->y2_smoothed[k]);
 * Source computes this as a separate subtract, a separate multiply, then a
 * separate add — no fmaf call at this line — so NOT fused here either (needs
 * -ffp-contract=off to stay that way, same discipline as kernel 4). */

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

/* ═══════════════════════════════ kernel 18 ═════════════════════════════════
 * sk_n2_track_f32 — the CNG N2-tracking data-dependent update. Verbatim from
 * AEC/c_impl/src/aec_post.c aec_post_compute_comfort_noise():
 *   track = (fresh*y2s[k] + retain*n2[k]) * g_up;  // mul, mul, add, mul —
 *                                                     all plain, no fmaf
 *   up    = n2[k] * g_up;
 *   n2[k] = (y2s[k] < n2[k]) ? track : up;          // exact IEEE '<'
 *                                                       compare+select
 */

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

/* ═══════════════════════════════ kernel 19 ═════════════════════════════════
 * sk_n2_initial_track_f32 — the CNG N2-initial slow-tracking data-dependent
 * update. Verbatim from AEC/c_impl/src/aec_post.c
 * aec_post_compute_comfort_noise():
 *   slow = n2i[k] + ia*(n2[k]-n2i[k]);        // plain sub/mul/add, no fmaf
 *   n2i[k] = (n2[k] > n2i[k]) ? slow : n2[k];  // exact IEEE '>' compare+select,
 *                                                 comparing against the
 *                                                 ORIGINAL n2i[k] (captured
 *                                                 before the overwrite)
 */

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

/* ═══════════════════════════════ kernel 20 ═════════════════════════════════
 * sk_mask_zero_f32 — out[i] = mask[i] ? 0.0f : out[i] (in-place, byte mask,
 * C truthiness: any nonzero byte triggers the zero — matches a plain
 * `if (mask[k]) x[k] = 0.0f;` source loop verbatim, e.g.
 * AEC/c_impl/src/aec_post.c's stationarity R²-zeroing step). */

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

/* ═══════════════════════════════ kernel 21 ═════════════════════════════════
 * sk_pairwise_sum_tailfold_f32 — numpy-style pairwise float32 sum, variant
 * "A". The numpy pairwise tree (8-accumulator leaf for n<=128, recursive
 * split above); used for the far-end hop energy and the linear-output block
 * energies.
 *
 * Two structural differences from kernel 13's sk_pairwise_sum_f32 (so this
 * is a genuinely different value-function, not a restyling):
 *   1. There is no outer n<=8 gate ahead of the n<=128 leaf. The leaf is
 *      entered directly for any n<=128 and branches internally on
 *      `n < 8` (strict). Consequence: at n==8 this kernel takes the
 *      8-accumulator tree-combine path, whereas kernel 13's OUTER n<=8
 *      check (inclusive) diverts n==8 to a plain sequential sum instead —
 *      different rounding, different bits.
 *   2. In the n in (8,128] leaf, the n%8 remainder tail is folded ONE
 *      ELEMENT AT A TIME straight into the already-combined `res` root
 *      (`res += a[i]` per leftover element), not accumulated into a
 *      separate running total and added once at the end the way kernel
 *      13's leaf does (`return r + s;`). Whenever there is more than one
 *      leftover element these two tail strategies round differently.
 *
 * The small-n (n<8) serial path here starts its accumulator at 0.0f and
 * adds every element, `s = 0.0f; for (i=0;i<n;++i) s += a[i];` — see
 * kernel 22 for the OTHER small-n convention (res = a[0], then fold from
 * i=1) used by the four other pairwise-sum call sites, which is a third
 * distinct value-function: it differs from this one only on signed-zero
 * inputs (e.g. n=1 with a[0]=-0.0f: this kernel returns
 * 0.0f + (-0.0f) = +0.0f; kernel 22 returns -0.0f unchanged), see kernel 22's
 * comment.
 *
 * NEON leaf: identical q0/q1 accumulation + vpaddq_f32×2 combine order to
 * kernel 13's leaf (bit-identical group-of-8 sums), then the same
 * one-at-a-time tail fold applied to the scalar `res` (not to q0/q1) to
 * match the scalar tree above exactly. */

/* gcc for 32-bit targets cannot bound n by 128 at the tail loop once the
 * recursion is inlined into a caller holding a fixed-size array, and reports
 * the loop as a -Waggressive-loop-optimizations error; the leaf is only
 * reached with n<=128. The pragma is scoped to this leaf and does not touch
 * code generation. */
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

/* ═══════════════════════════════ kernel 22 ═════════════════════════════════
 * sk_pairwise_sum_tailfold_b_f32 — numpy-style pairwise float32 sum, variant
 * "B". Used for the filter-analyzer floor accumulation, the fullband ERLE and
 * ERL sums, the reverb energy ratio and the suppression-gain slice sums.
 *
 * Same 8<=n<=128 leaf shape, tail-fold, and n/2-rounded-down-to-a-multiple-
 * of-8 split as kernel 21 (sk_pairwise_sum_tailfold_f32) — differs ONLY in
 * the small-n (n<8) path and the explicit n==0 case:
 *   - n==0 returns 0.0f explicitly (kernel 21 also returns 0.0f for n==0,
 *     via its 0.0f-initialized accumulator with a zero-iteration loop —
 *     same result, different code path, no observable difference).
 *   - n in [1,7]: `res = a[0]; for (i=1;i<n;++i) res = res + a[i];` — the
 *     accumulator STARTS AT a[0] (a plain copy, no addition performed for
 *     the first element), rather than starting at 0.0f and adding a[0] the
 *     way kernel 21 does. These two conventions are bit-identical for
 *     every finite/normal nonzero a[0], but diverge on signed zero: with
 *     a[0] = -0.0f and n==1, this kernel returns -0.0f (copied through
 *     untouched); kernel 21 returns 0.0f + (-0.0f) = +0.0f (IEEE-754
 *     round-to-nearest: an unlike-signed zero sum rounds to +0). For n=1 and for
 *     an all -0.0f array of length 5, kernel 21 and kernel 22 disagree with
 *     each other (each stays internally bit-identical scalar-vs-NEON), so two
 *     distinct kernels are required here, not one shared implementation. */

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

/* ═══════════════════════════════ kernel 23 ═════════════════════════════════
 * sk_noise_spectrum_update_f32 — the StationarityEstimator per-bin noise
 * floor tracker. Verbatim from
 * AEC/c_impl/src/stationarity_estimator.c noise_spectrum_update()'s steady-
 * state branch (block_counter > avg_init):
 *   if (pb > pn) {                                    // rising
 *       denom = max(pb, 1e-30f);
 *       alpha_inc = alpha * (pn / denom);
 *       if (apply_mask10 && (10.0f*pn < pb)) alpha_inc *= 0.1f;
 *       noise = pn + alpha_inc*(pb - pn);
 *   } else {                                           // falling (incl. ==)
 *       upd = pn + alpha*(pb - pn);
 *       noise = max(upd, min_noise);
 *   }
 *
 * Unlike kernels 18/19 (sk_n2_track_f32 / sk_n2_initial_track_f32), this is
 * a NESTED 2-way-inside-2-way conditional (the apply_mask10 scale-down lives
 * inside the rising branch) -- one level more complex than those two, so it
 * gets a dedicated correctness callout instead of a one-line "same shape as
 * 18/19" note:
 *   - `apply_mask10` is a call-level (not per-lane) boolean, computed once
 *     per hop by the caller (`block_counter > init_phase`) -- broadcasting
 *     it to an all-ones/all-zeros mask and ANDing with the per-lane
 *     `10*pn<pb` compare is a faithful vectorization of the scalar `if
 *     (apply_mask10) { if (...) ... }` nesting, not an approximation.
 *   - `max(pb, 1e-30f)` and the final `max(upd, min_noise)` are compare
 *     (`>`) + vbslq_f32 select, NEVER vmaxq_f32/vminq_f32 (this header's
 *     signed-zero tie-break rule) -- both source comparisons are strict
 *     `>`, so a tie (pb==1e-30f or upd==min_noise) takes the same side as
 *     the scalar ternary on both paths.
 *   - the shared `pb - pn` subtraction is textually identical in both the
 *     rising and falling branches of the SOURCE, so computing it once
 *     (`vdiff`) ahead of the two-branch compute-both-then-select is genuine
 *     CSE, not reassociation.
 *   - `vdivq_f32` (SK_HAVE_NEON == AArch64-only, see this header's own NaN-
 *     semantics section) is correctly-rounded IEEE-754 binary32 on this
 *     repo's target toolchain, matching the scalar `/` bit-for-bit --
 *     already relied on at kernel 1/16's cabs_np/coherence-gate call sites,
 *     not a new assumption for this kernel.
 * Every comparison here is over the FINITE domain per this file's NaN-
 * semantics contract (production's line of defense is WAV/API ingress
 * sanitization); a speculative rising-branch divide for an about-to-be-
 * discarded falling lane is masked out by the final vbslq_f32 select, never
 * observed. */

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
        float32x4_t vdiff = vsubq_f32(vpb, vpn); /* pb-pn: textually identical
                                                   * in BOTH scalar branches. */
        /* rising, computed unconditionally */
        uint32x4_t gt_eps = vcgtq_f32(vpb, veps);
        float32x4_t vdenom = vbslq_f32(gt_eps, vpb, veps);   /* max(pb,1e-30f) */
        float32x4_t vratio = vdivq_f32(vpn, vdenom);
        float32x4_t valpha_inc = vmulq_f32(valpha, vratio);
        {
            uint32x4_t mask10 = vandq_u32(venable10, vcltq_f32(vmulq_f32(vten, vpn), vpb));
            float32x4_t scaled = vmulq_f32(valpha_inc, vpoint1);
            valpha_inc = vbslq_f32(mask10, scaled, valpha_inc);
        }
        float32x4_t vrising = vaddq_f32(vpn, vmulq_f32(valpha_inc, vdiff));
        /* falling, computed unconditionally */
        float32x4_t vupd = vaddq_f32(vpn, vmulq_f32(valpha, vdiff));
        uint32x4_t gt_floor = vcgtq_f32(vupd, vminn);
        float32x4_t vfalling = vbslq_f32(gt_floor, vupd, vminn);  /* max(upd,min_noise) */
        /* top-level select */
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

/* ═══════════════════════════════ kernel 24 ═════════════════════════════════
 * sk_erl_bin_update_f32 — the ErlEstimator per-bin minimum-statistics update,
 * FUSED into one per-bin masked pass. Verbatim from
 * AEC/c_impl/src/erl_estimator.c erl_estimator_update()'s loop 1,
 * for k = 1..n_bins-2:
 *
 *   if (x2[k] > x2_min) {
 *       new_erl = y2[k] / x2[k];                          // float32 divide
 *       if (new_erl < erl[k]) {
 *           hold_counters[k-1] = hold_hops;                // int32 arm
 *           delta = 0.1f * (new_erl - erl[k]);             // plain sub/mul,
 *           erl[k] = erl[k] + delta;                       //   NOT fmaf
 *           if (erl[k] < ERL_MIN_ERL) erl[k] = ERL_MIN_ERL; // compare+select
 *       }
 *   }
 *
 * Caller offsets: called with erl/x2/y2 all advanced by +1 (so lane j of the
 * kernel's arrays is source index k=j+1) and hold_counters passed UNSHIFTED
 * -- hold_counters[j] already IS hold_counters[k-1] once erl/x2/y2 are
 * pre-offset by the caller (k-1 = (j+1)-1 = j), so no k-1 arithmetic happens
 * inside the kernel itself.
 *
 * FUSION SAFETY / masking: the int32 hold_counters store and the float32 erl
 * blend+floor only both happen when BOTH `x2[k]>x2_min` AND `new_erl<erl[k]`
 * hold — cond1 (x2>x2_min) and cond2 (new_erl<erl[k], against the ORIGINAL
 * erl[k], read once before any blend) are computed as two independent lane
 * masks and ANDed into a single combined mask that gates BOTH the erl store
 * and the hold_counters store identically, matching the scalar's nested `if`
 * exactly (never one store gated by cond1 alone and the other by the full
 * AND, or vice versa). The floor step is likewise computed for all lanes and
 * only becomes visible through that same top-level mask — not a second,
 * independently-gated store; matches the scalar form where the floor `if` is
 * nested INSIDE the same `if (new_erl<erl[k])` body as the hold_counters
 * write, not a sibling conditional.
 *
 * Speculative divide: `new_erl = y2[j]/x2[j]` is computed for every lane
 * unconditionally (incl. lanes that fail cond1, e.g. x2[j]==0 -> new_erl is
 * Inf or NaN) — same masked-speculative-compute pattern as kernel 23's
 * rising-branch divide: cond1 being false for that lane forces the combined
 * mask false regardless of what cond2 evaluates to on a NaN/Inf new_erl
 * (vcltq_f32 is unordered-false for NaN, exactly like scalar `<`), so the
 * speculative result never reaches erl[] or hold_counters[]. A NaN new_erl
 * specifically can NEVER be selected at all: cond2 requires new_erl<erl[k]
 * to hold, and IEEE `<` is unordered-false for a NaN operand on either side
 * -- so mask=true implies new_erl is a genuine (non-NaN) ordered value.
 * Per this header's NaN-semantics contract real inputs are finite (WAV/API
 * ingress sanitized); this is belt-and-suspenders equivalence, verified by
 * the selftest's dedicated NaN corpus, not a claim of new NaN-safety scope. */

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

        uint32x4_t cond1 = vcgtq_f32(vx2, vx2min);        /* x2>x2_min */
        float32x4_t vnew_erl = vdivq_f32(vy2, vx2);       /* speculative */
        uint32x4_t cond2 = vcltq_f32(vnew_erl, verl);     /* new_erl<erl, orig erl */
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

/* ═══════════════════════════════ kernel 25 ═════════════════════════════════
 * Decrement every int counter, saturating at INT_MIN. A zero floor would
 * alter diagnostic/raw counter values; INT_MIN preserves all practical
 * trajectories while removing signed-overflow UB in indefinitely running
 * streams. The NEON subtract is well-defined modulo 2^32 and its wrapped
 * value is masked off for lanes already at INT_MIN. */

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
        uint32x4_t mask = vcgtq_s32(v, vintmin);       /* x[i]>INT_MIN */
        int32x4_t  dec  = vsubq_s32(v, vone);          /* speculative, wraps
                                                          * (well-defined) at
                                                          * the INT_MIN lane,
                                                          * never selected */
        int32x4_t  result = vbslq_s32(mask, dec, v);   /* else unchanged */
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

/* ═══════════════════════════════ kernel 26 ═════════════════════════════════
 * sk_erl_hold_expire_f32 — bins whose hold counter has expired double toward
 * ERL_MAX_ERL. Taken from AEC/c_impl/src/erl_estimator.c
 * erl_estimator_update()'s loop 3, for k = 1..n_bins-2:
 *   if (hold_counters[k-1] <= 0) {
 *       doubled = 2.0f * erl[k];
 *       erl[k] = (doubled < ERL_MAX_ERL) ? doubled : ERL_MAX_ERL;
 *   }
 * Same caller-offset convention as kernel 24: erl advanced by +1 (lane j ==
 * source k=j+1), hold_counters passed UNSHIFTED (hold[j] == hold_counters[k-1]
 * once erl is pre-offset). hold_counters here are the values loop 2
 * (sk_dec1_floorintmin_s32) JUST decremented-or-floored -- this kernel must run
 * strictly after kernel 25's call completes, not fused with it or with
 * kernel 24.
 *
 * `hold <= 0` is an exact int32 compare (vcleq_s32, no float involved).
 * `2.0f*erl[k]` is a plain multiply (vmulq_f32), not vaddq_f32(erl,erl) --
 * bit-identical for all finite erl[k] either way, but kept as a literal
 * multiply to mirror the source token-for-token. The `<ERL_MAX_ERL` clamp is
 * compare(vcltq_f32)+select(vbslq_f32), never vminq_f32, per this header's
 * signed-zero tie-break rule (erl[k] is always >=ERL_MIN_ERL>0 in practice,
 * so the tie-break never actually engages here, but the rule is applied
 * uniformly regardless of whether a given call site can prove it moot).
 * Note this kernel's `doubled` can never surface a NaN-vs-NaN payload
 * ambiguity the way kernel 24's blend can: whenever `doubled` is NaN (e.g.
 * erl[k] itself started NaN), `doubled<ERL_MAX_ERL` is unordered-false on
 * BOTH scalar and NEON, so both deterministically select the fixed constant
 * ERL_MAX_ERL instead of the NaN — never the NaN payload itself — so this
 * kernel's selected output is always one of {unchanged erl[k] bits, doubled
 * (only ever selected when finite and < ERL_MAX_ERL), ERL_MAX_ERL}, none of
 * which is a NaN payload race; still exercised by the selftest's NaN corpus
 * for completeness, not because a divergence is expected there. */

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
        uint32x4_t mask = vcleq_s32(vhold, vzero);        /* hold<=0 */
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

/* ═══════════════════════════════ kernel 27 ═════════════════════════════════
 * sk_no_audible_echo_gain_f32 — the per-bin soft-blend body of
 * AEC/c_impl/src/suppression_gain.c gain_to_no_audible_echo(), for the
 * configuration where soft_blend_enabled and soft_blend_per_bin are both
 * set. Transcribed statement-for-statement from that loop:
 *
 *   enr_bin = echo[k] / (nearend[k] + 1.0f);
 *   sig     = (enr_bin - thr) / softness;
 *   if (sig < -50.0f) sig = -50.0f;
 *   if (sig >  50.0f) sig =  50.0f;
 *   ne_wb   = 1.0f / (1.0f + fast_exp(sig));
 *   <tab>_k = ne_wb * nearend_<tab>[k] + (1.0f - ne_wb) * normal_<tab>[k];
 *   enr = echo[k] / (nearend[k] + 1.0f);
 *   emr = echo[k] / (masker[k]  + 1.0f);
 *   g = 1.0f;
 *   if (enr > enr_tr_k && emr > emr_tr_k) {
 *       d_lin     = enr_su_k - enr_tr_k;
 *       denom_lin = d_lin < 1e-30f ? 1e-30f : d_lin;
 *       g_lin     = (enr_su_k - enr) / denom_lin;
 *       denom_emr = emr < 1e-30f ? 1e-30f : emr;
 *       g_emr     = emr_tr_k / denom_emr;
 *       g         = g_lin > g_emr ? g_lin : g_emr;
 *   }
 *   out[k] = g;
 *
 * `enr_bin` and `enr` are the same expression over the same operands and so
 * the same bits; both are written out, mirroring the source, and left to
 * the compiler to fold.
 *
 * The sigmoid's exponential is sk__fast_exp_vec (simd_kernels.h's validated
 * lane-for-lane twin of fast_math.h's fast_exp, itself domain-safe for NaN
 * and any magnitude), so this kernel exists only where that does --
 * SK_HAVE_NEON and not USE_STANDARD_MATH; every other build takes the
 * scalar entry point, which calls sk__fast_exp_elem, the equally verbatim
 * scalar twin fast_math.h's own selftest three-way-checks.
 *
 * The blend, the two ratios and the two quotients are plain
 * vmulq/vaddq/vsubq/vdivq with no vfmaq anywhere: the scalar source has no
 * fmaf call, and AArch64 FDIV is correctly rounded, so each lane rounds
 * exactly where the scalar expression does.
 *
 * Every one of the four selects is compare(vcltq/vcgtq)+select(vbslq), never
 * vminq/vmaxq, per this header's rule -- FMIN/FMAX propagate a NaN operand
 * and break +-0.0 ties toward the ordered result, while a C ternary's `<`
 * and `>` are unordered-false and therefore fall to the OTHER operand. The
 * final `g_lin > g_emr` max is where that matters most: a NaN g_lin (an
 * enr_su_k - enr of 0/0, or a NaN reaching the tables) must yield g_emr,
 * which vmaxq_f32 would not do. The two 1e-30f denominator clamps and the
 * two +-50.0f sigmoid clamps follow the same rule so the kernel is
 * unconditionally bit-exact rather than bit-exact-on-the-reachable-domain.
 *
 * `cond` computes both arms speculatively. The discarded arm can divide by
 * zero and produce an Inf or NaN, which is harmless -- no lane's selected
 * result depends on it, and no FP exception is enabled. */

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
        sig = vbslq_f32(vcltq_f32(sig, vlo), vlo, sig);   /* sig<-50 ? -50 : sig */
        sig = vbslq_f32(vcgtq_f32(sig, vhi), vhi, sig);   /* sig> 50 ?  50 : sig */
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

/* ═══════════════════════════ sk_max_abs_f32 ════════════════════════════════
 * Largest |x[i]| of n floats, the running maximum seeded at +0.0f and raised
 * only by a strictly larger magnitude:
 *   m = 0.0f; for i: { a = fabsf(x[i]); if (a > m) m = a; }
 * A NaN (quiet or signaling) never compares greater, so it never raises the
 * maximum, at any index including 0; n == 0 gives 0.0f. np.max(np.abs(x))
 * instead returns NaN when any element is NaN; the two agree on every
 * NaN-free input (an infinity gives +inf). The NEON form keeps the compare +
 * select of the scalar test (never vmaxq/vmaxnmq: the former propagates NaN,
 * the latter quiets a signaling NaN into the result). The lane accumulators
 * only ever hold values that passed the compare, so they are never NaN and
 * the final horizontal maximum is exact in any lane order; the seed and the
 * magnitudes are non-negative, so no signed zero can differ. */

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

/* ═════════════════════ sk_noise_floor_track_f32 ═══════════════════════════════
 * Per-bin tracker of a stationary noise floor with a hold counter:
 *   if (psd[k] < floor[k])        { floor[k] = psd[k]; count[k] = 0; }
 *   else if (count[k] >= hold)    { ramp = floor[k] * growth;
 *                                   floor[k] = (ramp > min_floor) ? ramp : min_floor; }
 *   else                          { count[k] += 1; }
 * The NEON form evaluates the three arms for every lane and selects with the
 * same compares: `<` and `>` are unordered-false for a NaN operand exactly as
 * in the scalar test (a NaN psd or floor takes the hold/ramp arms), the ramp
 * clamp is compare + select rather than vmaxq, and the speculative count + 1
 * wraps harmlessly in a lane that does not select it (the scalar increment
 * only runs for count < hold). */

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

/* ═══════════════════════════ sk_raise_to_f32 ═══════════════════════════════
 * In-place per-bin lower bound given by an array:
 *   if (x[k] < lo[k]) x[k] = lo[k];
 * The compare is unordered-false when either operand is NaN, so x is then left
 * as it is, and a tie leaves x untouched (a signed zero is never swapped); the
 * NEON compare + select reproduces both. `x` and `lo` must not overlap. */

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

/* ═════════════════════════ sk_stationary_mask_u8 ═════════════════════════════
 * Bins that are flagged stationary and have no hangover left:
 *   m[k] = (flags[k] != 0) && (hang[k] == 0)      -- stored as a 0/1 byte
 * Returns how many bins are set. `mask` may be NULL when only the count is
 * wanted. Pure integer logic, exact in any order. */

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
    /* each byte lane of the accumulator gains at most 1 per step, so it is
     * folded into the count before it could pass 255 */
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


/* ═══════════════ Inline filter, statistics and detector kernels ══════════ */

/* Per-bin H-error leakage refresh; returns the number of diverged bins.
 * The caller applies the final H-error clamp. */
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

/* Each counter corresponds to psd[k+1]; psd has n+2 elements.
 * Greater-than selects preserve the scalar neighbour-NaN rule. */
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

/* In-place three-bin logical AND of canonical 0/1 flags; endpoints mirror
 * the adjacent interior. */
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

#ifdef __cplusplus
}
#endif

#endif /* SIMD_AEC_H */
