#ifndef BF_H
#define BF_H

#include <stdint.h>

#include "fft_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int enable;                  /* 0: bypass BF, 1: enable BF */
    float lambda;  /* RLS forgetting factor, (0, 1] */
    float mu;  /* step scale applied to the weight update */

    int enable_fix_mode;          /* 0: auto DOA, 1: fixed beam */
    float fixed_doa_rad;          /* Fixed beam direction (rad), must be finite; used when enable_fix_mode == 1. */
    int fixed_align_notebook;     /* 1: fixed mode uses the notebook-style update with an every-hop cadence. */

    int adapt_interval;           /* Run the RLS update every N frames; <= 0 means 1. */
} BF_Config;

typedef struct {
    int enable;

    int M;
    int F;
    int num_angles;

    float lambda;
    float mu;

    int enable_fix_mode;
    float fixed_doa_rad;
    int fixed_align_notebook;

    int adapt_interval;

    Complex*** a_array;

    /* RLS state. P is P[M][M][F] (frequency contiguous for NEON). Internal; do not persist or index. */
    Complex*** P;
    Complex** wa;

    /* Persistent per-hop work area (keeps F- and M*F-sized arrays off the stack). */
    Complex* scratch;
    Complex* scratch_das;
    Complex* scratch_wu;
    Complex* scratch_spec;
    Complex* scratch_u;

    int initialized;
    int first_doa_found;
    int64_t first_doa_frame;  /* -1 until the first DOA is found. */
    float current_doa;
    uint64_t frame_idx;  /* Monotonic per-frame counter. */

    float doa_used;
    int adaptive;

    /* Lifetime count of per-bin RLS divergence recoveries; not cleared by bf_reset(). Stays at or near 0 in normal operation. */
    long bin_resets;

    /* Non-NULL only on the bf_create() path: the block backing every array above, freed by bf_destroy(). NULL on the bf_init() pool path. */
    void* owned_heap;

} BF;

/* Adapt interval BF actually runs at: bf_create() forces 1 when enable_fix_mode && fixed_align_notebook.
 * Callers deriving timing from the update cadence must use this function. */
int bf_effective_adapt_interval(
    int enable_fix_mode, int fixed_align_notebook, int adapt_interval);

BF* bf_create(int M, int F, int num_angles,
                Complex*** a_array,
                const BF_Config* cfg);

/* Static-memory pair: bf_get_mem_size(M, F) returns the bytes needed for the struct and its arrays (0 for M <= 0 or F <= 0);
 * it is independent of num_angles and cfg. bf_init() carves them from the caller's 16-byte-aligned pool with
 * bf_create()'s validation. `a_array` is borrowed in both: owned by the caller, never carved or freed by BF. */
size_t bf_get_mem_size(int M, int F);
BF*   bf_init(void* mem, size_t mem_size, int M, int F,
                int num_angles, Complex*** a_array,
                const BF_Config* cfg);

/* mask (optional): nonzero = freeze RLS adaptation on that bin; NULL = none. Opposite polarity to srp(). */
void bf_process(BF* g,
                 const Complex* const* X,
                 float doa_s,
                 int allow_adapt_in,
                 const int* mask,
                 Complex* bf_out);

/* bf_process() that also returns the pre-update effective weights: bf_out[f] = sum(effective_weights[m,f] * X[m,f])
 * (no conjugation at the call site). effective_weights may be NULL when not needed; bf_out is required. */
void bf_process_with_weights(BF* g,
                              const Complex* const* X,
                              float doa_s,
                              int allow_adapt_in,
                              const int* mask,
                              Complex* bf_out,
                              Complex* effective_weights);

void bf_reset(BF* g);
void bf_destroy(BF* g);
float bf_get_doa_used(const BF* g);
int bf_get_adaptive(const BF* g);

/* Per-hop leak factor applied to the adaptive weights during an unmasked RLS update. */
float bf_wa_leak_factor(void);

/* Lifetime count of per-bin RLS divergence recoveries (diagnostic). */
long bf_get_bin_resets(const BF* g);

/* P-diagonal clamp bounds applied after every adapted frame. */
float bf_p_diag_floor(void);
float bf_p_diag_ceil(void);

#ifdef __cplusplus
}
#endif

#endif
