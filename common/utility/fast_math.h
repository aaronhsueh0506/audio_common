/** fast_math.h - fast exp/log/sqrt/E1 approximations for DSP (LUT + Taylor exp, minimax log, ARM FSQRT sqrt with Newton fallback).
 * Define USE_STANDARD_MATH to use libm instead. */

#ifndef FAST_MATH_H
#define FAST_MATH_H

#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FM_LN2      0.693147180559945f
#define FM_LOG10E   0.4342944819032518f  // log10(e) = 1/ln(10)
#define FM_LN10     2.302585092994046f   // ln(10)
#define FM_EPSILON  1e-10f

// ARM targets use correctly-rounded sqrtf; soft-float ARM32 uses libm.
#if defined(__aarch64__) || (defined(__arm__) && defined(__ARM_FP) && (__ARM_FP & 4))
#define FM_HAVE_HW_SQRT 1
#else
#define FM_HAVE_HW_SQRT 0
#endif
#if FM_HAVE_HW_SQRT || defined(__arm__)
#define FM_USE_CORRECT_SQRT 1
#else
#define FM_USE_CORRECT_SQRT 0
#endif

// Degree-4 minimax coefficients for ln(1+m), m in [0,1): p(0) = 0 and p(1) = ln 2 exactly, so fast_log is continuous
// across binade boundaries. C2 is stored positive and subtracted. simd_kernels.h repeats them under SK_ names.
#define FM_LOG_C1   0.9972485899925232f
#define FM_LOG_C2   0.46980342268943787f
#define FM_LOG_C3   0.22263026237487793f
#define FM_LOG_C4  -0.05692821741104126f

// e^n for n = -16..16
#define EXP_TABLE_OFFSET 16
#define EXP_TABLE_SIZE 33

static const float exp_int_table[EXP_TABLE_SIZE] = {
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

static inline float fast_exp(float x) { return expf(x); }
static inline float fast_exp_neg(float x) { return expf(-x); }
static inline float fast_log(float x) { return logf(x); }
static inline float fast_log10(float x) { return log10f(x); }
static inline float fast_sqrt(float v) { return sqrtf(v); }

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

/* fast_exp / fast_log / fast_sqrt are finite-domain approximations; each guard is written `!(x REL bound)` so NaN takes the early-out.
 * Non-finite results (USE_STANDARD_MATH returns libm's IEEE results instead, so the modes differ there; pass finite values):
 *   fast_exp:  NaN -> 0.0f, +Inf -> 8.8861105e+06f (saturation), -Inf -> 0.0f
 *   fast_log:  NaN, 0, negative -> -1e10f; +Inf -> 88.72284f (ordinary range reduction)
 *   fast_sqrt: NaN, negative, -0.0f -> +0.0f; +Inf -> +Inf (ARM sqrtf) or NaN (Newton fallback, payload unspecified) */

/** Fast exp(x) via LUT + Taylor, valid on [-16, 16]; below -16 (and NaN) returns 0, above 16 saturates. */
static inline float fast_exp(float x) {
    if (!(x >= -16.0f)) return 0.0f;
    if (x > 16.0f) return 8.8861105e+06f;

    int x0 = (int)floorf(x);
    float dx = x - (float)x0;

    if (dx > 0.5f) {
        dx -= 1.0f;
        x0 += 1;
    }

    float exp_x0 = exp_int_table[x0 + EXP_TABLE_OFFSET];

    // Taylor expansion for e^dx, |dx| < 0.5.
    float dx2 = dx * dx;
    float exp_dx = 1.0f + dx + 0.5f * dx2 + (1.0f / 6.0f) * dx2 * dx;

    return exp_x0 * exp_dx;
}

/** Fast exp(-x) for x >= 0. */
static inline float fast_exp_neg(float x) {
    if (x <= 0.0f) return 1.0f;
    if (x >= 16.0f) return 0.0f;  // e^-16 ≈ 1.1e-7
    return fast_exp(-x);
}

/** Fast natural log from the IEEE 754 exponent and a minimax mantissa polynomial: log(x) = (E-127) * ln(2) + ln(1+m).
 * Accuracy is specified for positive normal binary32 inputs (subnormals are not normalised). NaN and x <= 0 return -1e10f. */
static inline float fast_log(float x) {
    if (!(x > 0.0f)) return -1e10f;

    union { float f; uint32_t i; } fb;
    fb.f = x;

    // Extract the exponent and normalise the mantissa to [1, 2) with explicit IEEE-754 masks.
    uint32_t bits = fb.i;
    int E = (int)((bits >> 23) & 0xFFu) - 127;
    fb.i = (bits & 0x007FFFFFu) | 0x3F800000u;
    float m = fb.f - 1.0f;

    // Degree-4 minimax Horner chain with separately rounded multiply and add on every target (no fmaf), so scalar and NEON agree.
    // Worst absolute error over positive normal inputs is below 1e-4.
    float poly = FM_LOG_C4;
    poly = poly * m + FM_LOG_C3;
    poly = poly * m - FM_LOG_C2;
    poly = poly * m + FM_LOG_C1;
    float ln_1_m = poly * m;

    return (float)E * FM_LN2 + ln_1_m;
}

/** Fast log10 via fast_log. */
static inline float fast_log10(float x) {
    return fast_log(x) * FM_LOG10E;
}

/** Fast sqrt. ARM targets use correctly-rounded sqrtf (FSQRT / VSQRT under -fno-math-errno, libm on soft-float ARM32).
 * Other targets use a bit-seed plus two Newton iterations (relative error below 2e-6 on normal inputs). NaN and v <= 0 return 0.0f. */
static inline float fast_sqrt(float v) {
    if (!(v > 0.0f)) return 0.0f;

#if FM_USE_CORRECT_SQRT
    return sqrtf(v);
#else
    union { float f; uint32_t i; } fb;
    fb.f = v;
    fb.i = (fb.i >> 1) + 0x1FC00000;

    float x = fb.f;
    x = 0.5f * (x + v / x);
    x = 0.5f * (x + v / x);

    return x;
#endif
}

/** E1(v) = integral from v to infinity of e^(-t)/t dt, by the low-complexity approximation of Martin, Malah, Cox & Accardi,
 * EURASIP JASP 2004, Eq. (17); E1 appears in the Ephraim-Malah MMSE-LSA gain. Keep the coefficients unchanged. */
#ifdef USE_OPTIMIZED_E1
// Upper segment first (common at high SNR), then one log10 for the other two; `!(v <= 1.0f)` sends NaN to the same fast_exp branch.
static inline float exp1_approx(float v) {
    if (v <= FM_EPSILON) v = FM_EPSILON;

    if (!(v <= 1.0f)) {
        return fast_exp((-0.52f * v - 0.26f) * FM_LN10);
    }

    float log10_v = fast_log10(v);
    if (v < 0.1f) {
        return -2.31f * log10_v - 0.6f;
    } else {
        return -1.544f * log10_v + 0.166f;
    }
}
#else
static inline float exp1_approx(float v) {
    if (v <= FM_EPSILON) v = FM_EPSILON;

    if (v < 0.1f) {
        return -2.31f * fast_log10(v) - 0.6f;
    } else if (v <= 1.0f) {
        return -1.544f * fast_log10(v) + 0.166f;
    } else {
        return fast_exp((-0.52f * v - 0.26f) * FM_LN10);
    }
}
#endif

#endif  // USE_STANDARD_MATH

/** Max of two floats. */
static inline float max_f(float a, float b) {
    return (a > b) ? a : b;
}

#ifdef __cplusplus
}
#endif

#endif // FAST_MATH_H
