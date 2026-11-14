#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "masker.h"
#include "utility/fast_math.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"

/* 1D median filter of a 0/1 mask: the median is 1 iff the count of ones c >= k - k/2 (valid for even k too). */
static void median_filter_1d(const int* in, int* out, int F, int k)
{
    int half = k / 2;
    int need = k - k / 2;    /* ones required for the median to be 1 */

    if (k <= 0) {
        memcpy(out, in, F * sizeof(int));
        return;
    }

    for (int i = 0; i < F; i++) {
        int cnt = 0;
        for (int j = -half; j <= half; j++) {
            int idx = i + j;
            if (idx < 0)   idx = 0;
            if (idx >= F)  idx = F - 1;
            cnt += in[idx];
        }
        out[i] = (cnt >= need) ? 1 : 0;
    }
}

/* masker_get_mem_size() and masker_carve() must stay in lockstep (ac_pool_carve, mem_align.h). */

size_t masker_get_mem_size(const MaskerConfig* cfg)
{
    size_t total;
    size_t F;

    if (!cfg || cfg->NFFT <= 0) return 0;
    F = (size_t)(cfg->NFFT / 2 + 1);

    total = ck_align16_size(sizeof(MaskEstimator));
    /* float[F] x5: noise_floor, noise_psd, spp_time, power_frame, energy_frame */
    total = ck_field_size_reps(total, F, sizeof(float), 5);
    /* int[F] x6: band, energy, spp, spp_bin, spp_f, mask */
    total = ck_field_size_reps(total, F, sizeof(int), 6);
    return MEM_SIZE_INVALID(total) ? 0 : total;
}

static MaskEstimator* masker_carve(void* mem, size_t need,
                                   const MaskerConfig* cfg)
{
    MaskEstimator* m;
    AcPoolCursor cursor;
    size_t F;
    int f;

    memset(mem, 0, need);
    m = (MaskEstimator*)mem;

    cursor.ptr = (uint8_t*)mem + ck_align16_size(sizeof(MaskEstimator));
    cursor.remaining = need - ck_align16_size(sizeof(MaskEstimator));

    m->cfg  = *cfg;
    m->NFFT = cfg->NFFT;
    m->sr   = cfg->sr;
    m->F    = cfg->NFFT / 2 + 1;
    F = (size_t)m->F;

    m->noise_floor = (float*)ac_pool_carve(&cursor, F, sizeof(float));
    m->noise_psd   = (float*)ac_pool_carve(&cursor, F, sizeof(float));
    m->spp_time    = (float*)ac_pool_carve(&cursor, F, sizeof(float));
    m->power_frame = (float*)ac_pool_carve(&cursor, F, sizeof(float));
    m->energy_frame = (float*)ac_pool_carve(&cursor, F, sizeof(float));

    m->band_mask    = (int*)ac_pool_carve(&cursor, F, sizeof(int));
    m->energy_mask  = (int*)ac_pool_carve(&cursor, F, sizeof(int));
    m->spp_mask     = (int*)ac_pool_carve(&cursor, F, sizeof(int));
    m->spp_mask_bin = (int*)ac_pool_carve(&cursor, F, sizeof(int));
    m->spp_mask_f   = (int*)ac_pool_carve(&cursor, F, sizeof(int));
    m->mask         = (int*)ac_pool_carve(&cursor, F, sizeof(int));

    if (!m->noise_floor || !m->noise_psd || !m->spp_time || !m->power_frame ||
        !m->energy_frame ||
        !m->band_mask || !m->energy_mask || !m->spp_mask ||
        !m->spp_mask_bin || !m->spp_mask_f || !m->mask) {
        return NULL;
    }

    for (f = 0; f < m->F; f++) {
        float freq = (float)f * m->sr / m->NFFT;
        m->band_mask[f] =
            (freq > m->cfg.low_freq && freq < m->cfg.high_freq) ? 1 : 0;
    }

    m->initialized = 0;
    return m;
}

MaskEstimator* masker_create(const MaskerConfig* cfg)
{
    MaskEstimator* m;
    void* block;
    size_t need = masker_get_mem_size(cfg);

    if (need == 0) return NULL;
    /* One 16-byte-aligned block backs the struct and every array. */
    if (posix_memalign(&block, 16, need) != 0 || !block) return NULL;

    m = masker_carve(block, need, cfg);
    if (!m) { free(block); return NULL; }
    m->owned_heap = block;
    return m;
}

MaskEstimator* masker_init(void* mem, size_t mem_size, const MaskerConfig* cfg)
{
    MaskEstimator* m;
    size_t need = masker_get_mem_size(cfg);

    if (!mem || need == 0 || !MEM_IS_ALIGNED16(mem) || mem_size < need)
        return NULL;

    m = masker_carve(mem, need, cfg);
    if (!m) return NULL;
    m->owned_heap = NULL;
    return m;
}

void masker_destroy(MaskEstimator* m)
{
    if (!m) return;
    /* Pool path (owned_heap == NULL): the caller owns the memory; nothing to free. */
    if (m->owned_heap) free(m->owned_heap);
}

/* X_ref: single-channel spectrum, e.g. X[0]. */
void masker_step(MaskEstimator* m,
                 const Complex* X_ref
                )
{
    if (!m || !X_ref ) return;

    float frame_spp_sum = 0.0f;

    sk_cmag_f32(m->power_frame, X_ref, m->F, 1e-8f);
    sk_linear_to_db_f32(m->energy_frame, m->power_frame, m->F);

    /* Noise-floor tracking (per-bin, independent of the threshold compare below). */
    if (!m->initialized) {
        memcpy(m->noise_floor, m->energy_frame,
               (size_t)m->F * sizeof(float));
    } else {
        sk_asym_ema_f32(m->noise_floor, m->energy_frame, m->F,
                        m->cfg.E_alpha_up, m->cfg.E_alpha_down);
    }

    for (int f = 0; f < m->F; f++) {
        float power = m->power_frame[f];
        float energy = m->energy_frame[f];

        {
            float thr = m->noise_floor[f] + m->cfg.margin_dB;
            m->energy_mask[f] = (energy > thr) ? 1 : 0;
        }

        if (!m->initialized)
            m->noise_psd[f] = power;

        {
            float gamma = power / (m->noise_psd[f] + 1e-12f);
            float xi = max_f(gamma - 1.0f, 0.0f);
            float Q = xi - log1pf(xi);
            float spp = 1.0f / (1.0f + expf(-Q));

            frame_spp_sum += spp;
            m->spp_mask_bin[f] = (spp > m->cfg.spp_thr) ? 1 : 0;
        }
    }

    {
        float frame_spp = frame_spp_sum / (float)m->F;

        if (frame_spp < m->cfg.spp_upd_thr) {
            for (int f = 0; f < m->F; f++) {
                float p = m->power_frame[f] + 1e-12f;
                m->noise_psd[f] =
                    m->cfg.M_alpha * m->noise_psd[f] +
                    (1.0f - m->cfg.M_alpha) * p;
            }
        }
    }

    if (m->cfg.enable_freq_smooth) {
        median_filter_1d(
            m->spp_mask_bin,
            m->spp_mask_f,
            m->F,
            m->cfg.smooth_size
        );
    } else {
        memcpy(m->spp_mask_f, m->spp_mask_bin, m->F * sizeof(int));
    }

    for (int f = 0; f < m->F; f++) {
        if (m->cfg.enable_time_smooth) {
            m->spp_time[f] =
                m->cfg.T_alpha * m->spp_time[f] +
                (1.0f - m->cfg.T_alpha) * (float)m->spp_mask_f[f];

            m->spp_mask[f] = (m->spp_time[f] > 0.5f) ? 1 : 0;
        } else {
            m->spp_time[f] = (float)m->spp_mask_f[f];
            m->spp_mask[f] = m->spp_mask_f[f];
        }
    }

    for (int f = 0; f < m->F; f++) {
        int energy_ok = m->cfg.enable_energy ? m->energy_mask[f] : 1;
        int spp_ok    = m->cfg.enable_spp    ? m->spp_mask[f]    : 1;
        int band_ok   = m->cfg.enable_band   ? m->band_mask[f]   : 1;

        m->mask[f] = (energy_ok && spp_ok && band_ok) ? 1 : 0;
    }

    m->initialized = 1;
}

const int* masker_get_mask(const MaskEstimator* m)
{
    if (!m) return NULL;
    return m->mask;
}