/** 2nd-order Butterworth high-pass (12 dB/octave, direct form II transposed) for DC, hum and rumble removal. */

#ifndef HPF_H
#define HPF_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Hpf Hpf;

/** Accepted parameters (hpf_create and hpf_init): cutoff_hz finite and > 0, sample_rate > 0,
 * cutoff_hz < 0.45 * sample_rate. Anything else (including NaN/Inf) returns NULL. */

/** Create a filter (malloc); NULL on invalid parameters or allocation failure. */
Hpf* hpf_create(float cutoff_hz, int sample_rate);

/** Initialize in caller memory: `mem` must be 16-byte aligned and at least hpf_get_mem_size() bytes.
 * Returns NULL, without writing to `mem`, on a misaligned base, short mem_size or invalid parameters. */
Hpf* hpf_init(void* mem, size_t mem_size, float cutoff_hz, int sample_rate);

size_t hpf_get_mem_size(void);

/** Destroy the filter (no-op for hpf_init() instances). */
void hpf_destroy(Hpf* hpf);

/** Filter n samples of `data` in place. */
void hpf_process(Hpf* hpf, float* data, int n);

/** Clear the filter state. */
void hpf_reset(Hpf* hpf);

#ifdef __cplusplus
}
#endif

#endif /* HPF_H */
