#ifndef VAD_H
#define VAD_H

#include <stddef.h>

typedef struct {
    int F;            // number of frequency bins

    int   mode;       // 0:any, 1:ratio, 2:count
    float mask_thr;   // ratio threshold
    int   min_bins;   // count threshold

    int   enable_median;
    int   median_k;

    int   enable_smooth;
    int   hangover;
} VadConfig;
typedef struct {

    VadConfig cfg;

    int med_buf[9];
    int med_pos;

    int state;
    int hang_cnt;

    int vad_raw;
    int vad_out;

    /* Non-NULL only on the vad_create() path: the block backing this struct, freed by vad_destroy(). */
    void* owned_heap;

} MaskVAD;


MaskVAD* vad_create(const VadConfig* cfg);

/* Caller-pool constructor: `mem` must be 16-byte aligned and `mem_size` >= vad_get_mem_size(), else NULL.
 * The caller owns the memory; vad_destroy() does not free it. */
size_t   vad_get_mem_size(void);
MaskVAD* vad_init(void* mem, size_t mem_size, const VadConfig* cfg);

/* Whether vad_init()/vad_create() accept this config; usable before sizing, since vad_get_mem_size() takes no config. */
int vad_config_is_valid(const VadConfig* cfg);

void vad_step(MaskVAD* v,
              const int* mask);

int vad_get_raw(const MaskVAD* v);
int vad_get_out(const MaskVAD* v);

void vad_destroy(MaskVAD* v);

#endif
