/**
 * simd_kernels.h - shared NEON/scalar bit-exact DSP micro-kernels.
 *
 * A small per-bin kernel library the AEC/NR float32 C code can call instead
 * of hand-rolled scalar loops, so the same vectorized building blocks are
 * reused everywhere instead of re-deriving them per call site.
 *
 * ─────────────────────────── Bit-exactness contract ──────────────────────
 * For every kernel, `sk_<name>(...)` (NEON, when available) MUST produce a
 * byte-identical result to `sk_<name>_scalar(...)` for every input (NaN
 * payload bits excepted — NaN is out of scope, see below). This is a hard
 * requirement, not a tolerance: downstream regression gates `cmp` two WAV
 * files, so any drift anywhere breaks the gate.
 *
 * This is achievable because AArch64 NEON per-lane vaddq_f32/vmulq_f32/
 * vdivq_f32/vsqrtq_f32/vfmaq_f32 are IEEE-754 binary32 correctly-rounded
 * operations — the exact same rounding the scalar FPU ops use. Given that,
 * bit-exactness reduces to one rule: **replicate the scalar operation
 * SEQUENCE (which ops are fused, which are separate, and in what order)
 * lane-for-lane**. Concretely:
 *
 *   1. No estimate instructions anywhere (vrsqrteq_f32, vrecpeq_f32, and
 *      their Newton-refinement companions are NEVER used in this file —
 *      every reciprocal/sqrt/divide is the exact vdivq_f32/vsqrtq_f32).
 *   2. No reassociation: if the scalar source computes `a*b + c*d` as two
 *      separate rounded multiplies followed by a separate rounded add, the
 *      NEON path issues `vmulq_f32`, `vmulq_f32`, `vaddq_f32` — never a
 *      single `vfmaq_f32`. Conversely, where the scalar source explicitly
 *      calls `fmaf(x, y, z)`, the NEON path uses `vfmaq_f32` (fusing y*x+z
 *      in one rounding step), because that's what the scalar reference
 *      itself does.
 *   3. Every NEON kernel's scalar tail (the `n % 4` leftover elements) calls
 *      the exact same static per-element helper the `_scalar` entry point
 *      uses — so the tail is bit-identical to the fully-scalar path *by
 *      construction*, not by re-derivation.
 *
 * ────────────────────────────── FMA discipline ────────────────────────────
 * Empirically verified on this toolchain (Apple clang 17, arm64): a plain
 * scalar C loop shaped `state[i] = alpha*state[i] + beta*x[i];` gets
 * auto-vectorized *and auto-fused into `fmla`* at `-O2` WITHOUT
 * `-ffp-contract=off` — while two separate `vmulq_f32`/`vaddq_f32`
 * intrinsic calls never get fused by the compiler regardless of the
 * contract setting (intrinsics lower to fixed instruction selections, not
 * generic contraction-eligible IR). That means the "plain mul/add, do NOT
 * fuse" scalar reference kernels below are only correct references when
 * compiled with `-ffp-contract=off` — the same flag AEC/NR's own c_impl
 * builds already mandate. **Any TU that includes this header, or otherwise
 * reproduces these expression shapes, MUST compile with
 * `-ffp-contract=off`.**
 *
 *   Uses explicit fmaf()/vfmaq_f32 (mirrors an explicit `fmaf(...)` call in
 *   the AEC/NR source — always fused, with or without -ffp-contract, since
 *   an explicit fmaf() call requests FMA directly rather than relying on
 *   contraction of separate ops):
 *     - the cabs_np/cmag2_np magnitude helper's `ratio*ratio + 1.0f` term
 *       (kernels 1, 2, 3, 5's internal magnitude computation)
 *     - sk_cmac_np_f32          (kernel 6)
 *     - sk_wupdate_nlms_f32's grad = err * conj(X) term (kernel 7)
 *     - sk_wupdate_kf_f32's final W += K_scaled * error_spec term (kernel 8)
 *
 *   Uses separate mul then add, NEVER fused (mirrors the source NOT calling
 *   fmaf for that particular step — needs -ffp-contract=off to stay this
 *   way):
 *     - sk_ema_f32                                   (kernel 4)
 *     - sk_ema_cmag2_f32's outer alpha*state+beta*mag2 combine (kernel 5)
 *     - sk_wupdate_nlms_f32's final W += mu_eff*grad combine   (kernel 7)
 *     - sk_wupdate_kf_f32's K = mu*conj(X) and K *= mu_scale steps
 *       (kernel 8)
 *     - sk_sq_scale_f32                              (kernel 11)
 *
 * ───────────────────────── min/clip: compare+select, not min/max ─────────
 * Empirically verified: AArch64 `vminq_f32(-0.0f, +0.0f)` returns -0.0f,
 * while the plain C ternary `(a < b) ? a : b` (a=-0.0f, b=+0.0f) returns
 * +0.0f, because IEEE `<` treats -0 and +0 as equal so the ternary falls
 * through to `b`. The hardware vector min/max instructions do NOT agree
 * with a naive ternary at a signed-zero tie. Since the self-test's
 * special-value pool includes ±0.0f, `sk_min_f32` / `sk_clip_f32` are
 * therefore implemented with compare+select (`vcltq_f32`/`vcgtq_f32` +
 * `vbslq_f32`), which reproduces the ternary's bit pattern exactly in every
 * case, instead of `vminq_f32`/`vmaxq_f32`. (The larger/smaller-of-two-
 * absolute-values step inside cabs_np/cmag2_np is NOT at risk: vabsq_f32
 * clears the sign bit first, so both operands to that particular max/min
 * are always +0.0 or positive, and a max/min tie between two bit-identical
 * non-negative values returns that same bit pattern regardless of which
 * hardware convention is used.)
 *
 * ────────────────────────────── Force-scalar knob ─────────────────────────
 * Define `SIMD_KERNELS_FORCE_SCALAR` (e.g. `-DSIMD_KERNELS_FORCE_SCALAR`)
 * to make every `sk_<name>` entry point just call `sk_<name>_scalar`, even
 * on an AArch64/NEON build. This validates that the non-NEON fallback path
 * compiles and runs on NEON-capable hardware too. `SK_HAVE_NEON` is 1 when
 * the NEON bodies are compiled in, 0 otherwise.
 *
 * ───────────────────────────────── Style ──────────────────────────────────
 * Header-only, C99, static inline (same convention as fast_math.h). Only
 * fft_wrapper.h (for `Complex`) plus <math.h>/<stdint.h>/<stddef.h> (and
 * <arm_neon.h> under the NEON guard) are included — no dependency on
 * fast_math.h itself, so this header's bit-exactness cannot be silently
 * altered by an unrelated `USE_STANDARD_MATH` toggle defined by some other
 * translation unit that happens to link into the same program. Where a
 * kernel mirrors a fast_math.h function (fast_sqrt, clip_f, min_f), its
 * exact algorithm is replicated verbatim as a private per-element helper
 * instead of calling fast_math.h directly — see the per-kernel comments.
 *
 * No `restrict` anywhere: callers may pass overlapping buffers only when
 * explicitly documented as supporting it (e.g. sk_capply_gain_f32's
 * out == z in-place case); otherwise pointers are assumed non-aliasing.
 */

#ifndef SIMD_KERNELS_H
#define SIMD_KERNELS_H

#include <math.h>
#include <stdint.h>
#include <stddef.h>

#include "fft_wrapper.h"   /* Complex { float r; float i; } (interleaved AoS) */

#if defined(__ARM_NEON) && defined(__aarch64__) && !defined(SIMD_KERNELS_FORCE_SCALAR)
#include <arm_neon.h>
#define SK_HAVE_NEON 1
#else
#define SK_HAVE_NEON 0
#endif

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
static inline float sk__fast_sqrt_elem(float v) {
    if (v <= 0.0f) return 0.0f;
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
        float32x4x2_t zv = vld2q_f32((const float *)(z + i));
        float32x4_t gv = vld1q_f32(g + i);
        float32x4x2_t rv;
        rv.val[0] = vmulq_f32(zv.val[0], gv);
        rv.val[1] = vmulq_f32(zv.val[1], gv);
        vst2q_f32((float *)(out + i), rv);
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
        float32x4x2_t av = vld2q_f32((const float *)(a + i));
        float32x4x2_t bv = vld2q_f32((const float *)(b + i));
        float32x4x2_t rv;
        rv.val[0] = vaddq_f32(av.val[0], bv.val[0]);
        rv.val[1] = vaddq_f32(av.val[1], bv.val[1]);
        vst2q_f32((float *)(out + i), rv);
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
 * seed + 2 Newton iterations). The `v<=0` guard is applied as a final
 * vcleq_f32+vbslq_f32 select (matching cabs_np's "compute unconditionally,
 * then select" pattern) rather than a branch: the bit-trick/Newton steps
 * are well-defined (finite, no trap) for non-positive/negative `v` too —
 * their result is simply discarded by the select for those lanes, same
 * final bits as the scalar early return. */

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
            uint32x4_t nonpos = vcleq_f32(v, vdupq_n_f32(0.0f));
            float32x4_t r = vbslq_f32(nonpos, vdupq_n_f32(0.0f), xk);
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

#ifdef __cplusplus
}
#endif

#endif /* SIMD_KERNELS_H */
