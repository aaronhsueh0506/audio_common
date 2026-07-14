/**
 * hpf.h - 2nd-order Butterworth IIR High-Pass Filter
 *
 * Removes DC offset, 50/60Hz hum, and low-frequency rumble.
 * 12 dB/octave rolloff. Direct Form II transposed.
 */

#ifndef HPF_H
#define HPF_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Hpf Hpf;

/**
 * Accepted parameter domain (both hpf_create and hpf_init):
 *   - cutoff_hz must be finite and > 0
 *   - sample_rate must be > 0
 *   - cutoff_hz must stay below 0.45 * sample_rate (a fixed margin under
 *     Nyquist, not 0.5) so the bilinear-transform prewarp tan(pi*cutoff/sr)
 *     stays well-conditioned. Values outside this domain (including NaN/Inf
 *     cutoff_hz) are rejected -- both functions return NULL rather than
 *     computing undefined/degenerate coefficients.
 */

/** Create high-pass filter (malloc version). Returns NULL if the params are
 * outside the domain above or on allocation failure. */
Hpf* hpf_create(float cutoff_hz, int sample_rate);

/** Initialize HPF in pre-allocated memory (static version). `mem` must be
 * 16-byte aligned and at least hpf_get_mem_size() bytes; returns NULL
 * (without writing to `mem`) if the base is misaligned, mem_size is too
 * small, or cutoff_hz/sample_rate are outside the domain above. */
Hpf* hpf_init(void* mem, size_t mem_size, float cutoff_hz, int sample_rate);

/** Get memory required for hpf_init() */
size_t hpf_get_mem_size(void);

/** Destroy HPF (no-op if created via hpf_init) */
void hpf_destroy(Hpf* hpf);

/**
 * Process samples in-place
 *
 * @param hpf Filter handle
 * @param data Sample buffer (modified in-place)
 * @param n Number of samples
 */
void hpf_process(Hpf* hpf, float* data, int n);

/**
 * Reset filter state
 */
void hpf_reset(Hpf* hpf);

#ifdef __cplusplus
}
#endif

#endif /* HPF_H */
