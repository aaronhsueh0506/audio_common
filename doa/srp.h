#ifndef SRP_H
#define SRP_H

#include <stddef.h>
#include <stdint.h>

#include "fft_wrapper.h"
#include "doa_smoother.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int M;
    float* x;
    float* y;
} ArrayGeometry;

typedef struct SRP SRP;

typedef struct {
    int M;
    int F;
    int num_angles;

    float sr;
    float NFFT;
    float c;

    /* SRP frequency band */
    float low_freq;
    float high_freq;

    /* smoothing */
    int enable_smoothing;
    int switch_consec;
    float angle_tol;

    int update_interval;

} SRP_Config;

struct SRP {
    int F;
    int num_angles;

    float* angles;
    Complex*** a_array;
    float* S_theta;

    int f_start;
    int f_end;

    int enable_smoothing;
    DOA_Smoother smoother;

    /* post-gain support */
    int* bin_best_idx;   /* length F: each bin's best angle index */

    float doa_raw;
    float doa_s;

    int update_interval;
    uint64_t frame_counter;  /* Monotonic per-frame counter. */

    float last_doa_s;

    /* Unique-pair SRP precompute: pair_steer[a][p][f] = conj(a_array[a][i][f]) * a_array[a][j][f]. */
    int num_pairs;
    int* pair_i;
    int* pair_j;
    Complex*** pair_steer;   /* [num_angles][num_pairs][F] */
    Complex** pair_phat;     /* [num_pairs][F], one frame scratch */
    float* score_scratch;         /* [F], one candidate angle */
    float* best_score;            /* [F], best candidate seen so far */

    /* NULL on the pool path (srp_init()); the caller owns the pool and srp_destroy() does not free it.
     * Set on the heap path (srp_create_from_geometry()) to the posix_memalign'd block. */
    void* owned_heap;
};

/* srp_get_mem_size() returns the total bytes; the caller allocates one 16-byte-aligned block.
 * srp_init() places the SRP at mem[0] and carves every array from it (no malloc).
 * Returns NULL if cfg/geom are invalid, mem is misaligned, or mem_size is too small. */
size_t srp_get_mem_size(const SRP_Config* cfg);
SRP*   srp_init(void* mem, size_t mem_size,
                const SRP_Config* cfg, const ArrayGeometry* geom);

/* Heap wrapper over srp_init(): allocates a 16-byte-aligned pool of srp_get_mem_size(cfg) bytes, records it in
 * SRP::owned_heap for srp_destroy(), and returns NULL (nothing leaked) on any allocation or validation failure. */
SRP* srp_create_from_geometry(
    const SRP_Config* cfg,
    const ArrayGeometry* geom
);

/* mask (optional): nonzero = use that bin in the score; NULL = all bins. Opposite polarity to bf_process_with_weights(). */
void srp(SRP* s, const Complex* const* X, const int* mask);

float srp2doa(SRP* s);

/* mask as in srp(). */
void doa_step(SRP* srp,
              const Complex* const* X,
              const int* mask,
              int vad_raw,
              int vad_out);

float doa_get_raw(const SRP* srp);
float doa_get_smooth(const SRP* srp);
void srp_reset(SRP* s);
void srp_destroy(SRP* s);

#ifdef __cplusplus
}
#endif

#endif
