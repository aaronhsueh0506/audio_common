#include <stdlib.h>
#include "vad.h"
#include "utility/mem_align.h"
#include <string.h>

static int vad_median(MaskVAD* v, int raw)
{
    int k = v->cfg.median_k;

    if (k != 5 && k != 7 && k != 9)
        return raw;

    v->med_buf[v->med_pos] = raw;
    v->med_pos = (v->med_pos + 1) % k;

    int sum = 0;
    for (int i = 0; i < k; i++)
        sum += v->med_buf[i];

    return (sum >= (k / 2 + 1));
}

size_t vad_get_mem_size(void)
{
    return ck_align16_size(sizeof(MaskVAD));
}

int vad_config_is_valid(const VadConfig* cfg)
{
    if (!cfg || cfg->F <= 0) return 0;
    /* vad_median() supports only median_k 5, 7 or 9 and med_buf holds 9 entries, so other values are refused.
     * Exported because vad_get_mem_size() takes no config. */
    if (cfg->enable_median &&
        cfg->median_k != 5 && cfg->median_k != 7 && cfg->median_k != 9)
        return 0;
    return 1;
}

MaskVAD* vad_init(void* mem, size_t mem_size, const VadConfig* cfg)
{
    MaskVAD* v;
    size_t need;

    if (!mem || !vad_config_is_valid(cfg)) return NULL;
    need = vad_get_mem_size();
    if (need == 0 || !MEM_IS_ALIGNED16(mem) || mem_size < need) return NULL;

    /* Only the budgeted region is touched; trailing bytes of a larger pool are left alone. owned_heap starts NULL. */
    memset(mem, 0, need);
    v = (MaskVAD*)mem;
    v->cfg = *cfg;
    return v;
}

MaskVAD* vad_create(const VadConfig* cfg)
{
    MaskVAD* v;
    void* block;
    size_t need = vad_get_mem_size();

    /* Delegates to vad_init() so the heap path gets the same budget, alignment and zero-fill. */
    if (!cfg || cfg->F <= 0) return NULL;
    if (posix_memalign(&block, 16, need) != 0 || !block) return NULL;
    v = vad_init(block, need, cfg);
    if (!v) { free(block); return NULL; }
    v->owned_heap = block;
    return v;
}

void vad_destroy(MaskVAD* v)
{
    if (v) free(v->owned_heap);
}

static int vad_mask2raw(MaskVAD* v, const int* mask)
{
    int F = v->cfg.F;
    int count = 0;

    for (int f = 0; f < F; f++) {
        if (mask[f]) count++;
    }

    if (v->cfg.mode == 0) {
        return count > 0;
    }

    if (v->cfg.mode == 1) {
        float ratio = (float)count / (float)F;
        return ratio >= v->cfg.mask_thr;
    }

    if (v->cfg.mode == 2) {
        return count >= v->cfg.min_bins;
    }

    return count > 0;
}

static int vad_smooth(MaskVAD* v, int raw)
{
    if (raw) {
        v->state = 1;
        v->hang_cnt = 0;
    } else {
        if (v->state) {
            v->hang_cnt++;
            if (v->hang_cnt > v->cfg.hangover) {
                v->state = 0;
                v->hang_cnt = 0;
            }
        }
    }
    return v->state;
}

void vad_step(MaskVAD* v,
              const int* mask)
{
    if (!v || !mask || v->cfg.F <= 0) return;

    int vad_raw = vad_mask2raw(v, mask);
    int vad = vad_raw;

    if (v->cfg.enable_median)
        vad = vad_median(v, vad);

    if (v->cfg.enable_smooth)
        vad = vad_smooth(v, vad);

    v->vad_raw = vad_raw;
    v->vad_out = vad;
}

int vad_get_raw(const MaskVAD* v)
{
    return v ? v->vad_raw : 0;
}

int vad_get_out(const MaskVAD* v)
{
    return v ? v->vad_out : 0;
}