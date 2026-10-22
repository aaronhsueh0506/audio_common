/* NR-specific scalar/NEON kernel bodies.
 * Dispatch stays in NR: ANR_VECTOR = SK_HAVE_NEON && !ANR_REFERENCE_KERNELS.
 * Keeping that independent reference switch lets tests compare scalar and
 * NEON kernels under the same shared FFT/backend configuration.
 *
 * Finite inputs are byte-identical to scalar twins; NaN payloads may differ,
 * but classification must match. Preserve separate multiply/add rounding
 * with -ffp-contract=off. Partial pointer overlap is unsupported unless stated.
 */

#ifndef SIMD_NR_H
#define SIMD_NR_H

#include "simd_kernels.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A build uses one twin; differential tests also call the other. */
#if defined(__GNUC__) || defined(__clang__)
#define SKR_TWIN static inline __attribute__((unused))
#else
#define SKR_TWIN static inline
#endif

/* Ordered minimum: start from values[0] and replace on a strict '<'. A NaN never
 * replaces and a NaN in values[0] is never replaced; among equal values the earliest
 * stays, so +0 and -0 keep their position-dependent identity.
 */
static inline float skr_min_ordered_scalar(const float* values, int n) {
    float minimum = values[0];
    for (int i = 1; i < n; ++i)
        if (values[i] < minimum) minimum = values[i];
    return minimum;
}

#if SK_HAVE_NEON
SKR_TWIN float32x4_t skr_vmin_lt(float32x4_t a, float32x4_t b) {
    return vbslq_f32(vcltq_f32(b, a), b, a);   /* b where b < a, else a */
}

/* Minimum through independent lane-wise running minima, seeded from the first vectors. A NaN
 * after the seed never replaces a minimum, exactly like the scalar rule, so only the seed needs
 * a NaN check. The result equals the ordered minimum whenever the seed holds no NaN and the
 * minimum is not a zero: equal non-zero floats have equal bits, so the order of comparison
 * cannot change it. Returns false (result untouched) otherwise, which needs the scalar rule.
 */
SKR_TWIN bool skr_min_ordered_fast_neon(const float* values, int n, float* result) {
    if (n < 4) return false;
    float32x4_t a0, a1, a2, a3;
    uint32x4_t seed_valid;
    int i;
    if (n >= 16) {
        a0 = vld1q_f32(values);
        a1 = vld1q_f32(values + 4);
        a2 = vld1q_f32(values + 8);
        a3 = vld1q_f32(values + 12);
        seed_valid = vandq_u32(vandq_u32(vceqq_f32(a0, a0), vceqq_f32(a1, a1)),
                               vandq_u32(vceqq_f32(a2, a2), vceqq_f32(a3, a3)));
        for (i = 16; i + 16 <= n; i += 16) {
            a0 = skr_vmin_lt(a0, vld1q_f32(values + i));
            a1 = skr_vmin_lt(a1, vld1q_f32(values + i + 4));
            a2 = skr_vmin_lt(a2, vld1q_f32(values + i + 8));
            a3 = skr_vmin_lt(a3, vld1q_f32(values + i + 12));
        }
        a0 = skr_vmin_lt(a0, a1);
        a2 = skr_vmin_lt(a2, a3);
        a0 = skr_vmin_lt(a0, a2);
    } else {
        a0 = vld1q_f32(values);
        seed_valid = vceqq_f32(a0, a0);
        i = 4;
    }
    for (; i + 4 <= n; i += 4) a0 = skr_vmin_lt(a0, vld1q_f32(values + i));
    a0 = skr_vmin_lt(a0, vextq_f32(a0, a0, 2));
    a0 = skr_vmin_lt(a0, vextq_f32(a0, a0, 1));
    float minimum = vgetq_lane_f32(a0, 0);
    for (; i < n; ++i)
        if (values[i] < minimum) minimum = values[i];
    if (vminvq_u32(seed_valid) == 0 || minimum == 0.0f) return false;
    *result = minimum;
    return true;
}

SKR_TWIN float skr_min_ordered_neon(const float* values, int n) {
    float minimum;
    if (skr_min_ordered_fast_neon(values, n, &minimum)) return minimum;
    return skr_min_ordered_scalar(values, n);
}
#endif /* SK_HAVE_NEON */

/* Keep the fused four-bin gather/scatter and deferred rescans: splitting
 * into single-bin calls reloads shared data and changes throughput. */

/* Ring write and incremental minimum for bins [0, n): store the new smoothed power S in slot
 * ring_pos of each bin's row (row k starts at ring[k * L]), keep S_min when S is a new minimum,
 * and rescan the row only when the evicted value was at most S_min * (1 + 1e-6).
 */
static inline void skr_ring_min_elem(float* s_min, float new_s, float* row,
                                     int ring_pos, bool* rescan) {
    float old_val = row[ring_pos];
    row[ring_pos] = new_s;
    *rescan = false;
    if (new_s <= *s_min) {
        *s_min = new_s;
    } else if (old_val <= *s_min * (1.0f + 1e-6f)) {
        *rescan = true;
    }
}

SKR_TWIN void skr_ring_min_scalar(float* s_min, const float* s,
                                  float* ring, int L, int ring_pos,
                                  int n) {
    for (int k = 0; k < n; k++) {
        bool rescan;
        skr_ring_min_elem(s_min + k, s[k], ring + (size_t)k * L, ring_pos,
                          &rescan);
        if (rescan) s_min[k] = skr_min_ordered_scalar(ring + (size_t)k * L, L);
    }
}

#if SK_HAVE_NEON
/* Block of bins processed before the deferred rescans; bounds the pending-bin list. */
#define SKR_RING_MIN_BLOCK 128

/* Per 4-bit lane mask: ascending numbers of the set lanes (unused entries 0) and their count. */
static const uint16_t skr_lane_list[16][4] = {
    {0, 0, 0, 0}, {0, 0, 0, 0}, {1, 0, 0, 0}, {0, 1, 0, 0},
    {2, 0, 0, 0}, {0, 2, 0, 0}, {1, 2, 0, 0}, {0, 1, 2, 0},
    {3, 0, 0, 0}, {0, 3, 0, 0}, {1, 3, 0, 0}, {0, 1, 3, 0},
    {2, 3, 0, 0}, {0, 2, 3, 0}, {1, 2, 3, 0}, {0, 1, 2, 3}};
static const uint8_t skr_lane_count[16] = {0, 1, 1, 2, 1, 2, 2, 3,
                                           1, 2, 2, 3, 2, 3, 3, 4};

/* Same result as the scalar twin (except the payload bits of a NaN): the compare and select run
 * on four bins at once, the bins that need a rescan are collected without a data-dependent
 * branch and rescanned after each block.
 */
SKR_TWIN void skr_ring_min_neon(float* s_min, const float* s, float* ring,
                                int L, int ring_pos, int n) {
    const float32x4_t growth = vdupq_n_f32(1.0f + 1e-6f);
    const uint32_t lane_bits[4] = {1u, 2u, 4u, 8u};
    const uint32x4_t weights = vld1q_u32(lane_bits);
    uint16_t pending[SKR_RING_MIN_BLOCK + 4];
    for (int base = 0; base < n; base += SKR_RING_MIN_BLOCK) {
        const int end = base + SKR_RING_MIN_BLOCK < n ? base + SKR_RING_MIN_BLOCK : n;
        int count = 0;
        int k = base;
        for (; k + 4 <= end; k += 4) {
            float* p0 = ring + (size_t)k * L + ring_pos;
            float* p1 = p0 + L;
            float* p2 = p1 + L;
            float* p3 = p2 + L;
            float32x4_t old_val = vld1q_lane_f32(p0, vdupq_n_f32(0.0f), 0);
            old_val = vld1q_lane_f32(p1, old_val, 1);
            old_val = vld1q_lane_f32(p2, old_val, 2);
            old_val = vld1q_lane_f32(p3, old_val, 3);
            float32x4_t new_s = vld1q_f32(s + k);
            vst1q_lane_f32(p0, new_s, 0);
            vst1q_lane_f32(p1, new_s, 1);
            vst1q_lane_f32(p2, new_s, 2);
            vst1q_lane_f32(p3, new_s, 3);
            float32x4_t current = vld1q_f32(s_min + k);
            uint32x4_t is_min = vcleq_f32(new_s, current);
            uint32x4_t stale = vbicq_u32(vcleq_f32(old_val, vmulq_f32(current, growth)), is_min);
            vst1q_f32(s_min + k, vbslq_f32(is_min, new_s, current));
            uint32_t bits = vaddvq_u32(vandq_u32(stale, weights));
            vst1_u16(pending + count, vadd_u16(vld1_u16(skr_lane_list[bits]),
                                               vdup_n_u16((uint16_t)k)));
            count += skr_lane_count[bits];
        }
        for (; k < end; k++) {
            bool rescan;
            skr_ring_min_elem(s_min + k, s[k], ring + (size_t)k * L, ring_pos,
                              &rescan);
            if (rescan) pending[count++] = (uint16_t)k;
        }
        for (int i = 0; i < count; i++)
            s_min[pending[i]] = skr_min_ordered_neon(ring + (size_t)pending[i] * L, L);
    }
}
#endif /* SK_HAVE_NEON */

/* Below this floor, consumer epsilons dominate the PSD and posterior gating can prevent
 * recovery, so such bins restart with an ungated update.
 */
#define SKR_NOISE_PSD_INERT 1e-20f

/* Restart inert bins with p=0 so a returning signal can update the floor. The update keeps
 * the non-fused order alpha_d*N + (1-alpha_d)*power; other bins are left unchanged.
 */
static inline void skr_restart_inert_elem(float* noise, float power, float alpha_d) {
    if (*noise < SKR_NOISE_PSD_INERT)
        *noise = alpha_d * *noise + (1.0f - alpha_d) * power;
}

SKR_TWIN void skr_restart_inert_scalar(float* noise, const float* power,
                                       float alpha_d, int n) {
    for (int k = 0; k < n; k++)
        skr_restart_inert_elem(noise + k, power[k], alpha_d);
}

#if SK_HAVE_NEON
SKR_TWIN void skr_restart_inert_neon(float* noise, const float* power,
                                     float alpha_d, int n) {
    const float32x4_t floor_psd = vdupq_n_f32(SKR_NOISE_PSD_INERT);
    const float32x4_t alpha = vdupq_n_f32(alpha_d);
    const float32x4_t one_minus = vdupq_n_f32(1.0f - alpha_d);
    int k = 0;
    for (; k + 4 <= n; k += 4) {
        float32x4_t current = vld1q_f32(noise + k);
        float32x4_t restarted = vaddq_f32(vmulq_f32(alpha, current),
                                          vmulq_f32(one_minus, vld1q_f32(power + k)));
        vst1q_f32(noise + k, vbslq_f32(vcltq_f32(current, floor_psd), restarted, current));
    }
    for (; k < n; k++)
        skr_restart_inert_elem(noise + k, power[k], alpha_d);
}
#endif /* SK_HAVE_NEON */

/* Clamp v to [1e-10,700]. Recover xi/(1+xi) directly when v underflows;
 * otherwise reuse v/(gamma+eps). Output arrays must not overlap inputs. */
static inline void skr_gain_prepare_elem(float v_in, float gamma, float xi, float* v_out,
                                         float* ratio_out) {
    float v = v_in;
    if (v < 1e-10f) v = 1e-10f;
    if (v > 700.0f) v = 700.0f;
    *v_out = v;
    if (v_in < 1e-10f) {
        *ratio_out = xi / (1.0f + xi);
    } else {
        *ratio_out = v_in / (gamma + 1e-10f);
    }
}

SKR_TWIN void skr_gain_prepare_scalar(const float* v_in, const float* gamma, const float* xi,
                                      float* v_out, float* ratio_out, int n) {
    for (int k = 0; k < n; k++)
        skr_gain_prepare_elem(v_in[k], gamma[k], xi[k], v_out + k, ratio_out + k);
}

#if SK_HAVE_NEON
SKR_TWIN void skr_gain_prepare_neon(const float* v_in, const float* gamma, const float* xi,
                                    float* v_out, float* ratio_out, int n) {
    const float32x4_t eps = vdupq_n_f32(1e-10f);
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t v_max = vdupq_n_f32(700.0f);
    int k = 0;
    for (; k + 4 <= n; k += 4) {
        float32x4_t v_raw = vld1q_f32(v_in + k);
        uint32x4_t below_eps = vcltq_f32(v_raw, eps);
        float32x4_t ratio_div = vdivq_f32(v_raw, vaddq_f32(vld1q_f32(gamma + k), eps));
        float32x4_t ratio;
        if (vmaxvq_u32(below_eps) == 0) {
            /* No lane needs the direct-xi fallback; skip its division and select. */
            ratio = ratio_div;
        } else {
            float32x4_t xi_k = vld1q_f32(xi + k);
            float32x4_t ratio_guard = vdivq_f32(xi_k, vaddq_f32(one, xi_k));
            ratio = vbslq_f32(below_eps, ratio_guard, ratio_div);
        }
        float32x4_t v = vbslq_f32(below_eps, eps, v_raw);
        v = vbslq_f32(vcgtq_f32(v, v_max), v_max, v);
        vst1q_f32(v_out + k, v);
        vst1q_f32(ratio_out + k, ratio);
    }
    for (; k < n; k++)
        skr_gain_prepare_elem(v_in[k], gamma[k], xi[k], v_out + k, ratio_out + k);
}
#endif /* SK_HAVE_NEON */

/* Final log-gain blend/store. Production calls the NEON twins under
 * ANR_VECTOR; scalar twins provide the differential reference. */
typedef struct {
    float log_g_min;
    float g_min;
    float alpha_attack;
    float alpha_decay;
    float prior_lift;
    bool  cross_band_prior;
    bool  smooth;          /* smooth against the previous log gain */
    bool  dd_from_gmmse;
} SkrFinalGainParams;

/* log_mmse: log(G_H1 + eps) in, blended and smoothed log gain out. emitted: that log gain clamped
 * to [log_g_min, 0], ready for exp(). prev_log is read only when smooth is set.
 */
static inline void skr_final_blend_elem(const SkrFinalGainParams* g, float posterior,
                                        float prev_log, float* log_mmse, float* emitted) {
    float p = posterior;
    /* Keep lift*p*(1-p) unfused and in the same order as the NEON twin. */
    if (g->cross_band_prior) p = p + g->prior_lift * p * (1.0f - p);
    float log_gain = p * *log_mmse + (1.0f - p) * g->log_g_min;
    if (g->smooth) {
        float alpha = (log_gain > prev_log) ? g->alpha_attack : g->alpha_decay;
        log_gain = alpha * prev_log + (1.0f - alpha) * log_gain;
    }
    *log_mmse = log_gain;
    *emitted = (log_gain < g->log_g_min) ? g->log_g_min
               : (log_gain > 0.0f) ? 0.0f : log_gain;
}

SKR_TWIN void skr_final_blend_scalar(const SkrFinalGainParams* g,
                                     const float* posterior, float* log_mmse,
                                     const float* saved_logs, float* emitted,
                                     int n) {
    for (int k = 0; k < n; k++)
        skr_final_blend_elem(g, posterior[k], g->smooth ? saved_logs[k] : 0.0f, log_mmse + k,
                             emitted + k);
}

/* exp_gain: exp() of the clamped log gain in, final gain out. The log gain outside
 * [log_g_min, 0] takes g_min / 1 and the matching saved log gain.
 */
static inline void skr_final_store_elem(const SkrFinalGainParams* g, float log_gain,
                                        float conditional, float power, float* exp_gain,
                                        float* saved_log, float* previous_gain,
                                        float* previous_psd) {
    float gain = *exp_gain;
    float log_save = log_gain;
    if (log_gain < g->log_g_min) {
        gain = g->g_min;
        log_save = g->log_g_min;
    } else if (log_gain > 0.0f) {
        gain = 1.0f;
        log_save = 0.0f;
    }
    *exp_gain = gain;
    *saved_log = log_save;
    *previous_gain = gain;
    float dd_gain = g->dd_from_gmmse ? conditional : gain;
    *previous_psd = dd_gain * dd_gain * power;
}

SKR_TWIN void skr_final_store_scalar(const SkrFinalGainParams* g,
                                     const float* log_gain, float* exp_gain,
                                     float* saved_logs, float* previous_gain,
                                     float* previous_psd,
                                     const float* conditional_gain,
                                     const float* power, int n) {
    for (int k = 0; k < n; k++)
        skr_final_store_elem(g, log_gain[k], conditional_gain[k], power[k], exp_gain + k,
                             saved_logs + k, previous_gain + k, previous_psd + k);
}

#if SK_HAVE_NEON
SKR_TWIN void skr_final_blend_neon(const SkrFinalGainParams* g, const float* posterior,
                                   float* log_mmse, const float* saved_logs, float* emitted,
                                   int n) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t log_min = vdupq_n_f32(g->log_g_min);
    const float32x4_t lift = vdupq_n_f32(g->prior_lift);
    const float32x4_t attack = vdupq_n_f32(g->alpha_attack);
    const float32x4_t decay = vdupq_n_f32(g->alpha_decay);
    int k = 0;
    for (; k + 4 <= n; k += 4) {
        float32x4_t p = vld1q_f32(posterior + k);
        if (g->cross_band_prior)
            p = vaddq_f32(p, vmulq_f32(vmulq_f32(lift, p), vsubq_f32(one, p)));
        float32x4_t log_gain = vaddq_f32(vmulq_f32(p, vld1q_f32(log_mmse + k)),
                                         vmulq_f32(vsubq_f32(one, p), log_min));
        if (g->smooth) {
            float32x4_t prev = vld1q_f32(saved_logs + k);
            float32x4_t alpha = vbslq_f32(vcgtq_f32(log_gain, prev), attack, decay);
            log_gain = vaddq_f32(vmulq_f32(alpha, prev),
                                 vmulq_f32(vsubq_f32(one, alpha), log_gain));
        }
        vst1q_f32(log_mmse + k, log_gain);
        float32x4_t clamped = vbslq_f32(vcgtq_f32(log_gain, zero), zero, log_gain);
        vst1q_f32(emitted + k, vbslq_f32(vcltq_f32(log_gain, log_min), log_min, clamped));
    }
    for (; k < n; k++)
        skr_final_blend_elem(g, posterior[k], g->smooth ? saved_logs[k] : 0.0f, log_mmse + k,
                             emitted + k);
}

SKR_TWIN void skr_final_store_neon(const SkrFinalGainParams* g, const float* log_gain,
                                   float* exp_gain, float* saved_logs, float* previous_gain,
                                   float* previous_psd, const float* conditional_gain,
                                   const float* power, int n) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t log_min = vdupq_n_f32(g->log_g_min);
    const float32x4_t gain_min = vdupq_n_f32(g->g_min);
    int k = 0;
    for (; k + 4 <= n; k += 4) {
        float32x4_t lg = vld1q_f32(log_gain + k);
        uint32x4_t below = vcltq_f32(lg, log_min);
        uint32x4_t above = vcgtq_f32(lg, zero);
        float32x4_t gain = vbslq_f32(below, gain_min,
                                     vbslq_f32(above, one, vld1q_f32(exp_gain + k)));
        float32x4_t log_save = vbslq_f32(below, log_min, vbslq_f32(above, zero, lg));
        vst1q_f32(exp_gain + k, gain);
        vst1q_f32(saved_logs + k, log_save);
        vst1q_f32(previous_gain + k, gain);
        float32x4_t dd = g->dd_from_gmmse ? vld1q_f32(conditional_gain + k) : gain;
        vst1q_f32(previous_psd + k, vmulq_f32(vmulq_f32(dd, dd), vld1q_f32(power + k)));
    }
    for (; k < n; k++)
        skr_final_store_elem(g, log_gain[k], conditional_gain[k], power[k], exp_gain + k,
                             saved_logs + k, previous_gain + k, previous_psd + k);
}
#endif /* SK_HAVE_NEON */

#ifdef __cplusplus
}
#endif

#endif /* SIMD_NR_H */
