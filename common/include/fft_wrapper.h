/** fft_wrapper.h - real-to-complex FFT over NE10. */

#ifndef FFT_WRAPPER_H
#define FFT_WRAPPER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
#define AC_FFT_RESTRICT __restrict
#else
#define AC_FFT_RESTRICT restrict
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Static-memory pattern used by every module (heap: _create/_destroy; static: _get_mem_size/_init/_destroy):
 *   bytes = MODULE_get_mem_size(config); m = MODULE_init(buf, bytes, config); MODULE_destroy(m);
 * buf is caller-owned and 16-byte aligned; destroy is a no-op on the static path. */
#include "ac_complex.h"

/* Opaque FFT handle. Calls on one handle must not overlap in time (shared staging block and NE10 work buffer);
 * distinct handles are independent. */
typedef struct FftHandle FftHandle;

/** Create an FFT handle on the heap. fft_size: power of 2 in [16, 8192], anything else is rejected. Returns NULL on error. */
FftHandle* fft_create(int fft_size);

/** Static-memory companions to fft_create: bytes = fft_get_mem_size(fft_size); h = fft_init(buffer, bytes, fft_size).
 * The buffer must be >= fft_get_mem_size(fft_size) bytes and 16-byte aligned. fft_get_mem_size returns 0 and fft_init NULL
 * for an fft_size outside the range above or a too-small mem_size. Nothing is allocated outside the caller's block. */
size_t     fft_get_mem_size(int fft_size);
FftHandle* fft_init(void* mem, size_t mem_size, int fft_size);

/** Destroy the handle. No-op for fft_init() handles; an fft_create() handle is freed and must be destroyed once. */
void fft_destroy(FftHandle* handle);

/** Number of frequency bins (fft_size/2 + 1). */
int fft_get_n_freqs(const FftHandle* handle);

/** Forward FFT, real -> complex. time_in [fft_size] and freq_out [n_freqs] must not overlap (restrict).
 * Use fft_forward_scratch() when the input may be given up. */
void fft_forward(FftHandle* handle, const float* AC_FFT_RESTRICT time_in,
                 Complex* AC_FFT_RESTRICT freq_out);

/** Inverse FFT, complex -> real. freq_in [n_freqs] and time_out [fft_size] must not overlap (restrict). */
void fft_inverse(FftHandle* handle, const Complex* AC_FFT_RESTRICT freq_in,
                 float* AC_FFT_RESTRICT time_out);

/** Forward FFT as fft_forward(), but the input contents are undefined after the call (skips the protective copy).
 * time_in_clobbered [fft_size], complex_out [n_freqs]. */
void fft_forward_scratch(FftHandle* handle, float* time_in_clobbered, Complex* complex_out);

/** Inverse FFT as fft_inverse(), but the input contents are undefined after the call.
 * freq_in_clobbered [n_freqs], real_out [fft_size]. */
void fft_inverse_scratch(FftHandle* handle, Complex* freq_in_clobbered, float* real_out);

/** Magnitude spectrum: magnitude[k] from freq[k], k < n_freqs. */
void fft_magnitude(const Complex* freq, float* magnitude, int n_freqs);

/** Power spectrum (magnitude^2): power[k] from freq[k], k < n_freqs. */
void fft_power(const Complex* freq, float* power, int n_freqs);

/** Phase spectrum in radians (atan2f(im, re) per bin): phase[k] from freq[k], k < n_freqs. */
void fft_phase(const Complex* freq, float* phase, int n_freqs);

/** Rebuild a spectrum: spectrum[k] = magnitude[k] * (cosf(phase[k]) + i*sinf(phase[k])), phase in radians, k < n_freqs. */
void fft_from_mag_phase(const float* magnitude, const float* phase,
                         Complex* spectrum, int n_freqs);

/** Apply gain[k] to freq[k] in place, k < n_freqs. */
void fft_apply_gain(Complex* freq, const float* gain, int n_freqs);

#ifdef __cplusplus
}
#endif

#undef AC_FFT_RESTRICT

#endif
