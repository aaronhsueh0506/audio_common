/**
 * fast_math.h - Fast math functions for DSP
 *
 * Includes optimized implementations of:
 * - exp() using LUT + Taylor
 * - log() using IEEE 754 + Taylor
 * - sqrt() using Newton-Raphson
 * - E1(v) exponential integral approximation
 *
 * Define USE_STANDARD_MATH to use standard library functions instead
 * (for debugging/verification purposes)
 */

#ifndef FAST_MATH_H
#define FAST_MATH_H

#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// Uncomment or define USE_STANDARD_MATH for debugging
// #define USE_STANDARD_MATH

// Constants
#define FM_LN2      0.693147180559945f
#define FM_LOG10E   0.4342944819032518f  // log10(e) = 1/ln(10)
#define FM_LN10     2.302585092994046f   // ln(10)
#define FM_EPSILON  1e-10f

// ============================================================================
// Fast exp() - LUT + Taylor expansion
// ============================================================================

// e^n for n = -16 to 16 (33 entries)
#define EXP_TABLE_OFFSET 16
#define EXP_TABLE_SIZE 33

static const float exp_int_table[EXP_TABLE_SIZE] = {
    1.1253517471925912e-07f,  // e^-16
    3.0590232050182579e-07f,  // e^-15
    8.3152871910356788e-07f,  // e^-14
    2.2603294069810542e-06f,  // e^-13
    6.1442123533282097e-06f,  // e^-12
    1.6701700790245659e-05f,  // e^-11
    4.5399929762484854e-05f,  // e^-10
    1.2340980408667956e-04f,  // e^-9
    3.3546262790251185e-04f,  // e^-8
    9.1188196555451624e-04f,  // e^-7
    2.4787521766663585e-03f,  // e^-6
    6.7379469990854670e-03f,  // e^-5
    1.8315638888734180e-02f,  // e^-4
    4.9787068367863944e-02f,  // e^-3
    1.3533528323661270e-01f,  // e^-2
    3.6787944117144233e-01f,  // e^-1
    1.0000000000000000e+00f,  // e^0
    2.7182818284590452e+00f,  // e^1
    7.3890560989306502e+00f,  // e^2
    2.0085536923187668e+01f,  // e^3
    5.4598150033144236e+01f,  // e^4
    1.4841315910257660e+02f,  // e^5
    4.0342879349273511e+02f,  // e^6
    1.0966331584284585e+03f,  // e^7
    2.9809579870417283e+03f,  // e^8
    8.1030839275753840e+03f,  // e^9
    2.2026465794806718e+04f,  // e^10
    5.9874141715197819e+04f,  // e^11
    1.6275479141900392e+05f,  // e^12
    4.4241339200892050e+05f,  // e^13
    1.2026042841647768e+06f,  // e^14
    3.2690173724721107e+06f,  // e^15
    8.8861105205078726e+06f   // e^16
};

#ifdef USE_STANDARD_MATH

// Standard library implementations for debugging
static inline float fast_exp(float x) { return expf(x); }
static inline float fast_exp_neg(float x) { return expf(-x); }
static inline float fast_log(float x) { return logf(x); }
static inline float fast_log10(float x) { return log10f(x); }
static inline float fast_sqrt(float v) { return sqrtf(v); }

// E1(v) using standard math
static inline float exp1_approx(float v) {
    if (v <= 1e-10f) v = 1e-10f;
    if (v < 0.1f) {
        return -2.31f * log10f(v) - 0.6f;
    } else if (v <= 1.0f) {
        return -1.544f * log10f(v) + 0.166f;
    } else {
        return powf(10.0f, -0.52f * v - 0.26f);
    }
}

#else  // Fast math implementations

/* ────────────────────────── NaN / non-finite contract ─────────────────────
 * fast_exp/fast_log/fast_sqrt below are finite-domain approximations: their
 * Taylor/Newton-Raphson machinery is only ever validated against finite
 * inputs in-range. Each function's leading domain guard is written as
 * `if (!(x REL bound)) return EDGE;` rather than `if (x INVERSE_REL bound)
 * return EDGE;` specifically so a NaN input -- which makes EVERY IEEE
 * ordered comparison (<, <=, >, >=) false -- still takes the early-out: the
 * negated form catches "not(finite AND in-range)" instead of only "finite
 * AND out-of-range", so NaN maps to the same deterministic domain-edge
 * constant a finite out-of-range input would get, rather than falling
 * through into the fast-path bit tricks (fast_exp's `(int)floorf(x)` cast on
 * NaN is undefined behaviour in C; fast_log/fast_sqrt's IEEE-754
 * bit-reinterpret tricks are not UB on NaN but produce non-deterministic
 * finite garbage or a propagated NaN instead of the documented edge value).
 * For every FINITE input the two guard forms are logically identical
 * (De Morgan's law over an already-ordered comparison), so this change is a
 * pure NaN-safety fix with zero effect on finite (or already-handled
 * +-Inf) behaviour -- see the per-function comments and the task's
 * before/after truth tables for the exhaustive case analysis.
 *
 * USE_STANDARD_MATH (the `#ifdef` branch above) calls straight into libm
 * (expf/logf/sqrtf), which instead propagates IEEE NaN/Inf per the C
 * standard (e.g. sqrtf(NaN) == NaN, not 0.0f). That divergence between the
 * two build modes on non-finite input is intentional and out-of-contract --
 * USE_STANDARD_MATH exists for debugging/verification against a reference
 * implementation, not for bit-identical behaviour with the fast path on
 * every possible float bit pattern; only the finite-domain results need to
 * match (and are what the parity/regression gates actually check).
 *
 * ───────────────── Special-value contract table (re-review R07) ──────────
 * Plain statement of the above as a lookup table, one row per non-finite or
 * boundary input, both build modes side by side. Every "fast_*" entry below
 * is measured and pinned by test/simd_selftest.c's dedicated special-value
 * asserts (not just asserted in this comment) -- see that file's "pinned
 * special-value contract (re-review R07)" section. Bit patterns are this
 * toolchain's IEEE-754 binary32 (Apple clang 17, arm64, -ffp-contract=off);
 * they follow deterministically from the source below, not from any
 * platform-specific rounding mode.
 *
 *   fast_exp(x):
 *     x = NaN        -> 0.0f                    [bits 0x00000000]
 *                        (USE_STANDARD_MATH: expf(NaN)  = NaN)
 *     x = +Inf       -> 8.8861105e+06f           [bits 0x4b07975e]
 *                        (identical to fast_exp(x) for any finite x > 16 --
 *                        the ">16" saturation constant, NOT a separate
 *                        Inf-specific branch)
 *                        (USE_STANDARD_MATH: expf(+Inf) = +Inf; DIFFERS)
 *     x = -Inf       -> 0.0f                     [bits 0x00000000]
 *                        (same path as any finite x < -16)
 *                        (USE_STANDARD_MATH: expf(-Inf) = 0.0f; matches)
 *     x = 0.0f       -> 1.0f                     (ordinary in-range Taylor
 *                        result, not a special-cased edge -- listed for
 *                        completeness only)
 *
 *   fast_log(x):
 *     x = NaN        -> -1e10f                   [bits 0xd01502f9]
 *                        (USE_STANDARD_MATH: logf(NaN)  = NaN; DIFFERS)
 *     x = 0.0f/-0.0f -> -1e10f                    (approximate -infinity)
 *                        (USE_STANDARD_MATH: logf(0)    = -Inf; DIFFERS)
 *     x < 0          -> -1e10f
 *                        (USE_STANDARD_MATH: logf(neg)  = NaN; DIFFERS)
 *     x = +Inf       -> 88.72283935546875f        [bits 0x42b17218]
 *                        (= 128.0f * FM_LN2, computed via the SAME Taylor
 *                        fallthrough an ordinary finite x would take --
 *                        +Inf is deliberately NOT special-cased in
 *                        fast_log, see that function's own comment; the
 *                        value is deterministic finite garbage-that-looks-
 *                        like-a-real-result, pinned here precisely because
 *                        it is NOT a designed edge value)
 *                        (USE_STANDARD_MATH: logf(+Inf) = +Inf; DIFFERS)
 *
 *   fast_sqrt(v):
 *     v = NaN        -> 0.0f                     [bits 0x00000000]
 *                        (USE_STANDARD_MATH: sqrtf(NaN) = NaN)
 *     v < 0           -> 0.0f
 *                        (USE_STANDARD_MATH: sqrtf(neg) = NaN; DIFFERS)
 *     v = -0.0f       -> 0.0f (POSITIVE zero -- the domain-edge return is
 *                        the literal `0.0f`, so fast_sqrt does NOT preserve
 *                        the sign of a negative-zero input the way IEEE
 *                        sqrtf does)
 *                        (USE_STANDARD_MATH: sqrtf(-0.0) = -0.0f; DIFFERS
 *                        in sign only)
 *     v = +Inf        -> NaN [bits 0x7fc00000 on this toolchain/CPU] (falls
 *                        through the domain guard -- `+Inf > 0.0f` is true
 *                        -- into the bit-trick seed + 2 Newton iterations;
 *                        iteration 1 computes Inf/finite = Inf, iteration 2
 *                        then computes Inf/Inf = NaN; the exact NaN payload
 *                        is an IEEE-754 implementation-defined "invalid
 *                        operation" default and not asserted bit-for-bit,
 *                        only isnan() is)
 *                        (USE_STANDARD_MATH: sqrtf(+Inf) = +Inf; DIFFERS)
 *
 * Summary: fast_* is a finite-domain approximation family. Every non-finite
 * or out-of-domain input still produces a DETERMINISTIC, TESTED float (never
 * a trap, never UB, never platform-dependent aside from the one documented
 * NaN-payload exception above) -- but that value is a domain-edge constant
 * or Taylor-fallthrough artifact, not a mathematically meaningful result,
 * and it will generally NOT match USE_STANDARD_MATH's libm behaviour on the
 * same input (see the DIFFERS annotations above). The system-level defense
 * against ever exercising this table in production is ingress sanitization
 * upstream of the DSP: the hardened WAV reader replaces non-finite samples
 * with 0.0f, and config float fields are range/finite-validated at load --
 * so in practice fast_exp/fast_log/fast_sqrt are only ever called on finite
 * inputs, and this table exists to pin the OUT-of-contract behaviour as a
 * regression gate, not because callers are expected to rely on it.
 */

/**
 * Fast exp(x) using LUT + Taylor expansion
 * Range: [-16, 16], outside this range returns 0 or saturates.
 * NaN: fails `x >= -16.0f` (every IEEE comparison with NaN is false), so the
 * negated guard below returns 0.0f for NaN as well as for finite x < -16 --
 * this also closes a real out-of-bounds bug: without it, NaN falls through
 * to `(int)floorf(x)`, and converting a NaN float to int is undefined
 * behaviour in C (and could index exp_int_table out of bounds).
 */
static inline float fast_exp(float x) {
    // Handle boundaries. Written as `!(x >= -16.0f)` (not `x < -16.0f`) so
    // NaN -- for which every ordered comparison is false -- also takes this
    // early-out instead of falling through to the (int) cast below (UB on
    // NaN). Identical to `x < -16.0f` for all finite x.
    if (!(x >= -16.0f)) return 0.0f;
    if (x > 16.0f) return 8.8861105e+06f;

    // Split into integer and fractional parts
    int x0 = (int)floorf(x);
    float dx = x - (float)x0;

    // Adjust dx to [-0.5, 0.5) for better Taylor accuracy
    if (dx > 0.5f) {
        dx -= 1.0f;
        x0 += 1;
    }

    // Lookup e^x0
    float exp_x0 = exp_int_table[x0 + EXP_TABLE_OFFSET];

    // Taylor expansion for e^dx (|dx| < 0.5)
    // e^dx ≈ 1 + dx + dx²/2 + dx³/6
    float dx2 = dx * dx;
    float exp_dx = 1.0f + dx + 0.5f * dx2 + (1.0f / 6.0f) * dx2 * dx;

    return exp_x0 * exp_dx;
}

/**
 * Fast exp(-x) for x >= 0, commonly used in SPP calculation
 */
static inline float fast_exp_neg(float x) {
    if (x <= 0.0f) return 1.0f;
    if (x >= 16.0f) return 0.0f;  // e^-16 ≈ 1.1e-7
    return fast_exp(-x);
}

// ============================================================================
// Fast log() - IEEE 754 structure + Taylor
// ============================================================================

typedef union {
    float f;
    uint32_t i;
    struct {
        uint32_t mantissa : 23;
        uint32_t exponent : 8;
        uint32_t sign : 1;
    } parts;
} FloatBits;

/**
 * Fast natural log using IEEE 754 float structure
 * log(x) = (E-127) * ln(2) + ln(1+m)
 *
 * NaN: fails `x > 0.0f`, so the negated guard below returns -1e10f
 * (approximate -infinity) for NaN too, instead of falling through to the
 * exponent/mantissa bit-extraction, which for a NaN input (biased exponent
 * 255, same as +-Inf) is not UB but silently produces a large, non-obvious
 * *finite* garbage value masquerading as a real log() result. +Inf is
 * deliberately NOT special-cased here: it still falls through and gets the
 * same finite garbage-but-plausible-looking value it always has (Inf > 0.0f
 * is true, an ordered comparison) -- only NaN's behaviour changes. Ingress
 * sanitization upstream is relied on to keep +Inf from reaching this
 * function in practice; only the NaN early-out is this fix's contract.
 */
static inline float fast_log(float x) {
    // Written as `!(x > 0.0f)` (not `x <= 0.0f`) so NaN -- for which every
    // ordered comparison is false -- also takes this early-out. Identical to
    // `x <= 0.0f` for all finite x.
    if (!(x > 0.0f)) return -1e10f;  // Approximate -infinity

    FloatBits fb;
    fb.f = x;

    // Extract exponent E and mantissa
    int E = (int)fb.parts.exponent - 127;

    // Normalize mantissa to [1, 2)
    fb.parts.exponent = 127;
    float m = fb.f - 1.0f;  // m ∈ [0, 1)

    // Taylor expansion for ln(1+m)
    // ln(1+m) ≈ m - m²/2 + m³/3 - m⁴/4
    float m2 = m * m;
    float m3 = m2 * m;
    float ln_1_m = m - 0.5f * m2 + (1.0f / 3.0f) * m3 - 0.25f * m2 * m2;

    return (float)E * FM_LN2 + ln_1_m;
}

/**
 * Fast log10 using fast_log
 */
static inline float fast_log10(float x) {
    return fast_log(x) * FM_LOG10E;
}

// ============================================================================
// Fast sqrt() - Newton-Raphson iteration
// ============================================================================

/**
 * Fast sqrt using Newton-Raphson with IEEE 754 initial estimate
 * x(n+1) = 0.5 * (x(n) + v/x(n))
 *
 * NaN: fails `v > 0.0f`, so the negated guard below returns 0.0f for NaN
 * too, instead of falling through to the bit-trick seed + Newton-Raphson
 * iterations (not UB on NaN -- the seed is plain integer arithmetic on the
 * bit pattern -- but `v/x` propagates the NaN through both iterations, so
 * the old behaviour was "return NaN", not the documented domain-edge 0.0f).
 */
static inline float fast_sqrt(float v) {
    // Written as `!(v > 0.0f)` (not `v <= 0.0f`) so NaN -- for which every
    // ordered comparison is false -- also takes this early-out. Identical to
    // `v <= 0.0f` for all finite v.
    if (!(v > 0.0f)) return 0.0f;

    // Use IEEE 754 structure for initial estimate
    FloatBits fb;
    fb.f = v;
    fb.i = (fb.i >> 1) + 0x1FC00000;  // Approximate sqrt

    // Newton-Raphson iterations (2 iterations for float precision)
    float x = fb.f;
    x = 0.5f * (x + v / x);  // Iteration 1
    x = 0.5f * (x + v / x);  // Iteration 2

    return x;
}

// ============================================================================
// E1(v) Exponential Integral - Three-segment approximation
// ============================================================================

/**
 * E1(v) = ∫[v,∞] e^(-t)/t dt
 *
 * Three-segment approximation (from Python implementation):
 * - v < 0.1:   E1(v) ≈ -2.31 * log10(v) - 0.6
 * - 0.1 ≤ v ≤ 1.0: E1(v) ≈ -1.544 * log10(v) + 0.166
 * - v > 1.0:   E1(v) ≈ 10^(-0.52*v - 0.26)
 */
#ifdef USE_OPTIMIZED_E1
// Optimized: check v > 1.0 first (common case in good SNR),
// then compute log10 only once for the remaining cases
static inline float exp1_approx(float v) {
    if (v <= FM_EPSILON) v = FM_EPSILON;

    if (v > 1.0f) {
        // E1(v) ≈ 10^(-0.52*v - 0.26) = exp((-0.52*v - 0.26) * ln(10))
        return fast_exp((-0.52f * v - 0.26f) * FM_LN10);
    }

    // Single log10 calculation for v in (0, 1.0]
    float log10_v = fast_log10(v);
    if (v < 0.1f) {
        return -2.31f * log10_v - 0.6f;
    } else {
        return -1.544f * log10_v + 0.166f;
    }
}
#else
// Original version
static inline float exp1_approx(float v) {
    if (v <= FM_EPSILON) v = FM_EPSILON;

    if (v < 0.1f) {
        // E1(v) ≈ -2.31 * log10(v) - 0.6
        return -2.31f * fast_log10(v) - 0.6f;
    } else if (v <= 1.0f) {
        // E1(v) ≈ -1.544 * log10(v) + 0.166
        return -1.544f * fast_log10(v) + 0.166f;
    } else {
        // E1(v) ≈ 10^(-0.52*v - 0.26) = exp((-0.52*v - 0.26) * ln(10))
        return fast_exp((-0.52f * v - 0.26f) * FM_LN10);
    }
}
#endif

#endif  // USE_STANDARD_MATH

// ============================================================================
// Utility functions (common to both implementations)
// ============================================================================

/**
 * Clip value to range [min, max]
 */
static inline float clip_f(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

/**
 * Max of two floats
 */
static inline float max_f(float a, float b) {
    return (a > b) ? a : b;
}

/**
 * Min of two floats
 */
static inline float min_f(float a, float b) {
    return (a < b) ? a : b;
}

#ifdef __cplusplus
}
#endif

#endif // FAST_MATH_H
