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
 * NaN caveat, precisely: for the compare+select kernels (sk_min_f32,
 * sk_clip_f32, sk_fast_sqrt_f32, sk_fast_exp_f32, sk_fast_exp_neg_f32,
 * sk_fast_log_f32, sk_fast_log10_f32, sk_exp1_approx_f32) NaN is actually
 * deterministic and IS verified bit-exact scalar-vs-NEON (see
 * simd_selftest.c's dedicated NaN blocks) — every IEEE ordered comparison
 * (<, <=, >, >=) is false for NaN, on both the scalar FPU and NEON, so the
 * "select" always lands on the same documented branch (fast_sqrt/fast_exp/
 * fast_log: the domain-edge constant fast_math.h documents; min/clip:
 * whichever operand the ternary/mask falls through to) in both paths.
 * "NaN is out of scope" applies to the pure-arithmetic kernels (sk_ema_f32,
 * sk_cadd_f32, sk_sq_scale_f32, sk_capply_gain_f32, sk_mcra_noise_update_f32):
 * a NaN operand propagates through +/- correctly-rounded arithmetic with no
 * compare/select involved, and while AArch64 scalar and NEON FP units are
 * architecturally the same IEEE-754 hardware (so payload propagation should
 * in practice agree), this file does not assert or test that payload-bit
 * equality for those kernels — only that NaN inputs don't corrupt unrelated
 * lanes/output.
 *
 * ───────────────────── Contract summary (re-review R07) ──────────────────
 * Stated once, plainly, as the one-paragraph version of the two blocks
 * above: every kernel here is scalar-reference bit-exact for FINITE inputs,
 * full stop, no exceptions, gated by simd_selftest.c's exit(1)-on-mismatch
 * finite corpus. For a NaN input, the per-lane contract is "NaN in -> NaN
 * out, payload unspecified" for the pure-arithmetic kernels, and "matches
 * whatever branch the scalar reference's own compare+select ternary takes on
 * that NaN" for the guarded kernels (sk_min_f32/sk_clip_f32/
 * sk_fast_sqrt_f32) — which this file's kernels satisfy not just as a
 * behavioural claim but bit-for-bit, verified by simd_selftest.c's dedicated
 * NaN blocks, because every guard here is a single ordered compare with at
 * most one NaN operand in play. This header has no multi-NaN-operand
 * REDUCTION kernel (no pairwise-sum tree, no fmaf-chained accumulator that
 * folds several independent NaN-carrying inputs together) — those live in
 * AEC/c_impl's aec_simd_kernels.h, where a NaN can arrive from two different
 * operands into the same fmaf/add and scalar vs. NEON may legitimately
 * disagree on WHICH NaN's payload survives (C leaves multi-NaN payload
 * selection implementation-defined) while still both correctly producing
 * *a* NaN. aec_simd_kernels.h's own selftest (simd_selftest_aec.c) has a
 * real classified gate for exactly that "both sides say NaN, payloads
 * differ" case (bit-exact / both-NaN-payload-unspecified / HARD FAIL,
 * re-review R07) — see that file's header comment for the full contract and
 * the empirical result (100% of its historical mismatches are the in-contract
 * both-NaN case, 0% are a genuine finite-vs-NaN divergence).
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
 *     - sk_fast_exp_f32 / sk_fast_exp_neg_f32's Taylor expansion and
 *       sk_fast_log_f32 / sk_fast_log10_f32's Taylor expansion (kernels
 *       23-26) — fast_math.h's fast_exp/fast_log never call fmaf, so
 *       neither do these
 *     - sk_exp1_approx_f32's three branch formulas               (kernel 27)
 *     - sk_mcra_noise_update_f32                                  (kernel 28)
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
 * kernel mirrors a fast_math.h function (fast_sqrt, clip_f, min_f, fast_exp,
 * fast_exp_neg, fast_log, fast_log10, exp1_approx), its exact algorithm is
 * replicated verbatim as a private per-element helper instead of calling
 * fast_math.h directly — see the per-kernel comments. This DOES include
 * mirroring fast_math.h's own `USE_STANDARD_MATH` build-mode switch (a
 * separate `#ifdef USE_STANDARD_MATH` inside this header, not a dependency
 * on the other one) so a call site converted from a bare fast_math.h call to
 * the matching sk_ kernel behaves identically in both build modes.
 *
 * No `restrict` anywhere: callers may pass overlapping buffers only when
 * explicitly documented as supporting it (e.g. sk_capply_gain_f32's
 * out == z in-place case, and the exp/log family's out == x case, kernels
 * 23/24/25/27, documented at their definitions below); otherwise pointers
 * are assumed non-aliasing.
 *
 * Round-3 review B05 (extended by the NR/c_impl calculate_gain()/
 * spp_estimate() scratch-buffer-reuse review): only the alias forms actually
 * exercised by simd_selftest.c's matrix are contractually supported --
 * sk_capply_gain_f32's literal out == z (dedicated in-place check in
 * test_capply_gain()), plus sk_fast_exp_f32/sk_fast_exp_neg_f32/
 * sk_fast_log_f32/sk_exp1_approx_f32's literal out == x (dedicated in-place
 * check in test_exp_log_family_inplace()). sk_fast_log10_f32 shares the
 * identical per-block load-then-store shape but has no call site relying on
 * it yet and no dedicated in-place selftest, so out == x for it is NOT
 * contractually supported until one is added. Every other kernel's edge-case
 * matrix uses NON-overlapping buffers at every offset combination it tests;
 * partial overlap (out == z + k / out == x + k for any nonzero k) is neither
 * documented nor tested anywhere in this file and is unsupported, even if it
 * happens to work today on some input.
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
 * that's the layout every memcpy-based sk__cquad_load/sk__cquad_store
 * quad-load/store below silently depends on (see the "Complex-quad NEON
 * load/store" section further down for why this goes through memcpy +
 * vld1q_f32/vst1q_f32 + vuzpq_f32/vzipq_f32 instead of vld2q_f32/vst2q_f32
 * directly). Pin the assumption here, once, next to the `Complex` include,
 * so a future ABI-changing edit to that struct (an added field, reordered
 * members, explicit padding/alignment) fails to COMPILE instead of silently
 * reintroducing a misaligned/wrong-stride NEON access everywhere this header
 * is included. */
SK_STATIC_ASSERT(sizeof(Complex) == 2 * sizeof(float),
               "Complex must be exactly two floats {r,i} (8 bytes, no "
               "padding) -- the sk_c*_f32 NEON kernels' memcpy-based "
               "quad-load/store layout depends on this");
SK_STATIC_ASSERT(offsetof(Complex, r) == 0 &&
               offsetof(Complex, i) == sizeof(float),
               "Complex.r/.i must sit at byte offsets 0/4 in that order "
               "(interleaved AoS) -- the sk_c*_f32 NEON kernels' memcpy-based "
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

/* ═══════════════════ Complex-quad NEON load/store (legal aliasing) ═════════
 * sk__cquad_load / sk__cquad_store — the ONLY sanctioned way any kernel in
 * this file (or in AEC/c_impl's aec_simd_kernels.h, which #includes this
 * header and calls these two directly rather than redefining them) may move
 * 4 lanes of `Complex` through NEON registers.
 *
 * Why this exists: the previous code here cast a `const Complex*`/`Complex*`
 * (interleaved {r,i} AoS — see fft_wrapper.h) straight to
 * `(const float *)`/`(float *)` and handed that to vld2q_f32/vst2q_f32
 * directly. AEC/c_impl's aec_simd_kernels.h used to do that exact same cast
 * at its own many call sites too; it no longer does — that file now
 * `#include`s this header and calls sk__cquad_load/sk__cquad_store directly
 * (see above) instead of casting on its own, so it inherited this fix rather
 * than needing a separate one. Reading/writing a `Complex` object through a
 * `float` lvalue that way is a type-based-aliasing violation (C11 6.5p7):
 * every byte of the struct IS a float, but `float` is
 * not the EFFECTIVE TYPE of a `Complex` object, so the compiler is entitled
 * to assume a `Complex*` and a `float*` never alias — true UB even though it
 * happens to produce the intended codegen today, at -O2, on every toolchain
 * this project currently builds with (the exact class of exposure this
 * repo's Makefile already documents and carries a target-specific
 * -fno-strict-aliasing override for on fft_wrapper.c/fft_wrapper_ne10.c —
 * see FFT_WRAPPER_ALIAS_CFLAGS there). These two kernel FUNCTIONS live in a
 * header with no .c file of their own in audio_common, so that Makefile-level
 * fix cannot reach them; fixing the aliasing at the SOURCE instead, once,
 * here, fixes every consumer (AEC/c_impl's aec3_post.c, Audio_ALG/pipelines'
 * audio_pipeline.c, and aec_simd_kernels.h's own many call sites) without any
 * Makefile change in any of those repos.
 *
 * The legal fix, in two steps:
 *   1. memcpy 4 Complex elements (8 floats, 32 bytes) into/out of a plain
 *      `float[8]` stack scratch array. memcpy is defined over
 *      `unsigned char` and carries no notion of the source's or
 *      destination's declared type across the copy, so copying the bytes of
 *      a `Complex[4]` into a `float[8]` this way is a legal
 *      byte-reinterpretation between two unrelated types — unlike a pointer
 *      cast, which asks the compiler to treat the SAME memory as two
 *      different effective types AT ONCE.
 *   2. Deinterleave/interleave that scratch array's 8 floats using PLAIN
 *      single-register vld1q_f32/vst1q_f32 (2 each, never vld2q_f32/
 *      vst2q_f32) plus a register-only vuzpq_f32/vzipq_f32 shuffle. This
 *      step is NOT what the obvious first attempt does (see below) — it is
 *      the part that had to be discovered empirically.
 *
 * The obvious first attempt — memcpy into `float[8]`, then call
 * vld2q_f32/vst2q_f32 directly against that scratch array (the same
 * multi-register load/store used before, just now against a legally-typed
 * buffer) — was tried FIRST and rejected by the disassembly check this file's
 * bit-exactness contract demands: `objdump -d` of a standalone probe TU
 * (Apple clang 17, arm64, -O2 AND -O3, -fno-stack-protector to rule that out
 * as a confound) showed the memcpy was NOT eliminated — the compiler emits a
 * genuine load-from-source into a temporary, a real store into the scratch
 * array, THEN a separate `ld2.4s` reading that scratch array back (and the
 * mirror image on the store side: `st2.4s` into scratch, then a real
 * load-and-store out to the destination). vld2q_f32/vst2q_f32 lower to an
 * opaque multi-register memory intrinsic
 * (`llvm.aarch64.neon.ld2`/`.st2.v4f32.p0`) that LLVM's memcpy-forwarding
 * optimizations (memcpyopt/DSE/GVN) do not appear to look through — the
 * store into scratch and the intrinsic's own read of scratch never get
 * fused, so the round-trip through the stack survives at -O2 AND -O3,
 * unconditionally. That failed verification is exactly why this comment
 * exists instead of a two-line vld2q_f32(scratch)/vst2q_f32(scratch)
 * wrapper, and exactly the scenario this project's own review process
 * requires falling back from (a Makefile-level -fno-strict-aliasing escape
 * hatch, the FFT_WRAPPER_ALIAS_CFLAGS pattern) UNLESS a working alternative
 * is found — which the shape below is.
 *
 * The shape actually used swaps vld2q_f32/vst2q_f32 for vld1q_f32/
 * vst1q_f32 (ordinary single-vector loads/stores, which DO lower to plain
 * LLVM `load`/`store` IR, not an opaque intrinsic) plus a register-only
 * uzp/zip shuffle to do the deinterleave/interleave arithmetic that ld2/st2
 * would otherwise have done in the memory unit:
 *   - load:  lo = vld1q_f32(scratch), hi = vld1q_f32(scratch+4); the pair
 *     (lo, hi) is exactly [r0,i0,r1,i1] and [r2,i2,r3,i3]. vuzpq_f32(lo, hi)
 *     deinterleaves EVEN/ODD lanes across the (lo,hi) pair, giving
 *     val[0]=[r0,r1,r2,r3] (the same real vector vld2q_f32 would have
 *     produced) and val[1]=[i0,i1,i2,i3] (same imaginary vector) — verified
 *     both by hand (NEON vuzpq_f32 semantics) and by a standalone
 *     correctness probe (4-element load + round-trip store, both -O0 and
 *     -O2, values compared field-by-field).
 *   - store: the inverse, vzipq_f32(v.val[0], v.val[1]) re-interleaves the
 *     real/imag vectors back into [r0,i0,r1,i1] / [r2,i2,r3,i3] pairs, two
 *     plain vst1q_f32 calls write those into the scratch array in the
 *     correct interleaved order, then memcpy carries the final bytes out.
 * Because vld1q_f32/vst1q_f32 are ordinary loads/stores, LLVM's optimizer
 * DOES fold the memcpy away entirely on this toolchain: the verified
 * disassembly of both real kernels below (`objdump -d` of a standalone probe
 * TU including this exact header and calling sk_capply_gain_f32/sk_cadd_f32,
 * Apple clang 17/arm64, -O2) shows the 4-lane loop body loads the 32 bytes
 * DIRECTLY from the source pointer (`ldp q0,q1,[src]`), deinterleaves in
 * registers (`uzp1.4s`/`uzp2.4s`), computes, re-interleaves
 * (`zip2.4s`), and stores DIRECTLY to the destination pointer (a
 * `st2`+`str` pair) — no stack scratch traffic survives anywhere in the
 * loop. This is genuinely equivalent work to ld2/st2 (same deinterleave/
 * interleave semantics, same bit-exact numeric result — the shuffle
 * instructions only ever move bits, they perform no arithmetic), just
 * composed from instructions LLVM is willing to reason about across the
 * memcpy boundary. NOT assumed: re-verify by disassembly any time this
 * helper or its call sites change materially. */
#if SK_HAVE_NEON
static inline float32x4x2_t sk__cquad_load(const Complex *p) {
    float scratch[8];
    memcpy(scratch, p, sizeof(scratch));
    {
        float32x4_t lo = vld1q_f32(scratch);
        float32x4_t hi = vld1q_f32(scratch + 4);
        return vuzpq_f32(lo, hi); /* val[0]=r's, val[1]=i's */
    }
}

static inline void sk__cquad_store(Complex *p, float32x4x2_t v) {
    float32x4x2_t z = vzipq_f32(v.val[0], v.val[1]); /* re-interleave r/i */
    float scratch[8];
    vst1q_f32(scratch, z.val[0]);
    vst1q_f32(scratch + 4, z.val[1]);
    memcpy(p, scratch, sizeof(scratch));
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

/* ═══════════════════ kernels 23-27: fast_math.h exp/log family ════════════
 * Array-kernel NEON twins for fast_math.h's fast_exp/fast_exp_neg/fast_log/
 * fast_log10/exp1_approx — added per the s4-audio-common-sweep review
 * (fast_math.h had zero NEON coverage anywhere, and these five functions sit
 * in NR's hottest per-bin loops: spp_estimator.c's fast_exp_neg, mmse_lsa_
 * denoiser.c's calculate_gain, mcra_noise_estimator.c's per-hop hi-freq
 * flatness loop). Per this header's own "no dependency on fast_math.h" rule
 * (see the Style section above), the table/constants/algorithm are
 * replicated verbatim as private SK_-prefixed helpers below rather than
 * included from fast_math.h — keep them in sync with fast_math.h if that
 * implementation ever moves. Validated against the REAL fast_math.h
 * (fast_exp/fast_exp_neg/fast_log/fast_log10/exp1_approx) bit-for-bit over
 * tens of thousands of random + curated-boundary values by
 * test/simd_selftest.c's test_fast_exp()/test_fast_exp_neg()/test_fast_log()/
 * test_fast_log10()/test_exp1_approx().
 *
 * Memory-safety note (table gather): fast_exp's exp_int_table lookup has no
 * native NEON gather on this target, so the NEON path does 4 scalar table
 * reads assembled into a vector (see sk__fast_exp_vec below). The table
 * index is derived from an internally range-clamped copy of the input
 * (vminq_f32/vmaxq_f32 to [-16,16]) so the gather is ALWAYS in-bounds
 * (index in [0,32]) regardless of the real, unclamped input — including
 * NaN and any out-of-[-16,16]-domain finite value whose result will be
 * overridden by the domain-edge select afterward anyway (same "compute
 * unconditionally, then select" shape as sk_fast_sqrt_f32 above, kernel 15).
 * NaN specifically: vminq_f32/vmaxq_f32 (NOT vminnmq_f32/vmaxnmq_f32) are
 * ordinary NaN-propagating IEEE FMIN/FMAX, so a NaN input stays NaN through
 * the clamp and vrndmq_f32 (floor); AArch64's FCVTZS (float-to-signed-int
 * convert, `vcvtq_s32_f32`) is architecturally defined to return 0 for a NaN
 * source (ARMv8 ARM, not implementation-defined), so the resulting table
 * index for a NaN lane is exactly SK_EXP_TABLE_OFFSET (16) — safely
 * in-bounds — confirmed empirically on this toolchain, not just asserted
 * from the architecture reference. Without the clamp, a huge finite input
 * (e.g. 1e30f) would FCVTZS-saturate to INT32_MAX/INT32_MIN and then
 * overflow signed int on the `+ SK_EXP_TABLE_OFFSET` add (UB) before ever
 * indexing the table — the clamp is load-bearing for that case, confirmed
 * empirically too (see the review's validation notes). The scalar reference
 * never hits either hazard: its two domain guards (`!(x>=-16.0f)` /
 * `x>16.0f`) return early, before the `(int)floorf(x)` line, for every
 * input that isn't already in [-16,16].
 *
 * FMA discipline: fast_math.h's fast_exp/fast_log never call fmaf() at any
 * step, so none of these kernels do either — every combine below is a
 * separate vmulq_f32/vaddq_f32/vsubq_f32, matching the "separate mul then
 * add, never fused" list in the file header comment. Requires
 * -ffp-contract=off on the including TU, same as every other kernel here.
 *
 * USE_STANDARD_MATH: mirrors fast_math.h's toggle (bare expf/logf/log10f/
 * powf calls, no domain-edge constants). There is no native vectorized
 * transcendental on this target, so under USE_STANDARD_MATH the "NEON" entry
 * points below just call the scalar loop — same honest fallback shape as
 * every other `#if SK_HAVE_NEON ... #else scalar ... #endif` kernel when
 * there is genuinely nothing to vectorize.
 *
 * In-place (out==x) safety (NR/c_impl calculate_gain()/spp_estimate()
 * scratch-buffer-reuse review): sk_fast_exp_f32 (23), sk_fast_exp_neg_f32
 * (24), sk_fast_log_f32 (25), and sk_exp1_approx_f32 (27) all SUPPORT
 * out==x. Every NEON body below processes one 4-lane block per iteration by
 * fully loading it (vld1q_f32) into registers, computing entirely from
 * those registers (no further memory reads), and only THEN storing
 * (vst1q_f32) that same block back — no iteration reads a lane any other
 * iteration writes, so aliasing the output onto the exact same input
 * pointer is safe (the same reasoning as sk_capply_gain_f32's existing
 * out==z contract, kernel 9, above). The scalar tail (and the
 * USE_STANDARD_MATH / non-NEON scalar-only fallback) is the same
 * read-into-a-value-then-write-out[i] shape per element, so it agrees.
 * sk_fast_log10_f32 (26) shares the identical shape but is NOT covered by a
 * dedicated in-place selftest (no call site needs it yet) -- see the "no
 * restrict" contract note near the top of this file. This is exercised by
 * simd_selftest.c's test_exp_log_family_inplace(). Partial-offset aliasing
 * (out == x + k, k != 0) remains unsupported for all five, same as
 * everywhere else in this file. */

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
 * sk_mcra_noise_update_f32 — verbatim shape of NR/c_impl/src/
 * mcra_noise_estimator.c's per-hop, per-bin noise-PSD update (mcra_update(),
 * ~line 605-612):
 *
 *   su = spp[i] * bb_scale;
 *   tilde_alpha_d = alpha_d + (1.0f - alpha_d) * su;
 *   noise_psd[i] = tilde_alpha_d*noise_psd[i] + (1.0f-tilde_alpha_d)*power[i];
 *
 * A per-bin-VARYING-alpha EMA (tilde_alpha_d depends on spp[i], unlike
 * kernel 4's sk_ema_f32 which takes a single scalar alpha for the whole
 * call) -- added per the s4-audio-common-sweep review. Per that review's
 * independent-verifier correction, this is a FUSED, single-purpose kernel
 * matching the call site's exact shape (alpha_d, bb_scale scalars; spp,
 * power, noise_psd arrays) rather than a generic "alpha-array" primitive:
 * the fused form needs no extra n_freqs-sized scratch buffer for a
 * precomputed tilde_alpha_d array, so it doesn't grow NR's static-memory
 * pool budget the way a generic two-array-weight kernel would, matching
 * this file's existing "verbatim kernel per call site" convention (see
 * sk_ema_cmag2_f32/sk_n2_track_f32/sk_n2_initial_track_f32 in AEC's
 * aec_simd_kernels.h for the same pattern). NOT wired into NR's call site by
 * this change -- NR is a sibling repo built separately; this kernel is
 * added and validated here in isolation (test/simd_selftest.c) so it is
 * ready for that call site to adopt in NR's own review pass.
 *
 * No fmaf anywhere: the source computes `alpha_d + (1-alpha_d)*su` and
 * `tilde*noise_psd + (1-tilde)*power` as separate rounded multiplies then
 * separate rounded adds (no fmaf call at either line), so this kernel must
 * not fuse either -- same -ffp-contract=off requirement as every other
 * kernel in this file. `(1.0f - alpha_d)` is loop-invariant in the source
 * (alpha_d is a per-call scalar, re-evaluated identically every iteration)
 * so hoisting it out of the loop here changes nothing bit-wise -- same
 * `vdupq_n_f32`-once-before-the-loop pattern sk_ema_f32 (kernel 4) already
 * uses for its own scalar alpha/beta. */

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

#ifdef __cplusplus
}
#endif

#endif /* SIMD_KERNELS_H */
