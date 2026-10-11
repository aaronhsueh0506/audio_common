/**
 * fft_wrapper.h - FFT wrapper interface
 *
 * Abstracts FFT implementation (KISS FFT by default)
 * Provides real-to-complex FFT for audio processing
 */

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

/* ---------------------------------------------------------------------------
 * Static memory support (embedded targets — no heap).
 *
 * Each module exposes both a heap-based API (`_create / _destroy`) and a
 * static API (`_get_mem_size / _init / _destroy`). The static API takes a
 * caller-allocated buffer and places all internal state inline using
 * pointer arithmetic with 16-byte alignment.
 *
 * Pattern (apply to every module):
 *
 *     size_t   bytes = MODULE_get_mem_size(config);
 *     void*    buf   = ... obtained from caller's static pool ...
 *     Module*  m     = MODULE_init(buf, bytes, config);
 *     ...
 *     MODULE_destroy(m);   // no-op for static path; safe for both paths
 *
 * `MODULE_get_mem_size` and `MODULE_init` must walk fields in identical
 * order to keep alignment + bytes consistent.
 * --------------------------------------------------------------------------- */
#include "mem_align.h"   /* ALIGN16 */

/* Complex number structure */
typedef struct {
    float r;  /* Real part */
    float i;  /* Imaginary part */
} Complex;

/* Opaque FFT handle */
typedef struct FftHandle FftHandle;

/**
 * Create FFT handle for given size (heap path).
 *
 * @param fft_size FFT size: a power of 2 in [16, 8192]. Both backends reject
 *                 anything else (NULL / 0).
 * @return FFT handle, or NULL on error
 */
FftHandle* fft_create(int fft_size);

/**
 * Static-memory companions to fft_create.
 *
 *   bytes = fft_get_mem_size(fft_size);
 *   h     = fft_init(buffer, bytes, fft_size);
 *
 * The buffer must be at least `fft_get_mem_size(fft_size)` bytes and
 * 16-byte aligned. Returns NULL if `mem_size` is too small, or 0 from
 * `fft_get_mem_size` (and NULL from `fft_init`) if `fft_size` is outside the
 * [16, 8192] power-of-two range documented on `fft_create` above.
 *
 * `fft_get_mem_size` covers every byte fft_init() touches, including the
 * backend's twiddle/config storage, so neither backend allocates outside the
 * caller's block (the NE10 backend uses the external-memory init described in
 * lib/ne10/VENDORED.md).
 */
size_t     fft_get_mem_size(int fft_size);
FftHandle* fft_init(void* mem, size_t mem_size, int fft_size);

/**
 * Destroy FFT handle.
 *
 * A no-op on both backends for a handle created by fft_init(): nothing was
 * allocated outside the caller's block, so any number of calls is safe. An
 * fft_create() handle is freed, and like free() must be destroyed only once.
 */
void fft_destroy(FftHandle* handle);

/**
 * Get number of frequency bins (fft_size/2 + 1)
 */
int fft_get_n_freqs(const FftHandle* handle);

/**
 * Forward FFT: real input -> complex output
 *
 * @param handle FFT handle
 * @param time_in Real input [fft_size] -- must not alias freq_out
 * @param freq_out Complex output [n_freqs] -- must not alias time_in
 *
 * `restrict` lets the compiler vectorise the copy and scale loops. The two
 * buffers must not overlap (undefined behaviour otherwise); use
 * fft_forward_scratch() when the input buffer may be given up.
 */
void fft_forward(FftHandle* handle, const float* AC_FFT_RESTRICT time_in,
                 Complex* AC_FFT_RESTRICT freq_out);

/**
 * Inverse FFT: complex input -> real output
 *
 * @param handle FFT handle
 * @param freq_in Complex input [n_freqs] -- must not alias time_out
 * @param time_out Real output [fft_size] -- must not alias freq_in
 *
 * `restrict`: as for fft_forward().
 */
void fft_inverse(FftHandle* handle, const Complex* AC_FFT_RESTRICT freq_in,
                 float* AC_FFT_RESTRICT time_out);

/**
 * Forward FFT: real input -> complex output, input clobber PERMITTED.
 *
 * Same transform as fft_forward(), but the input buffer's contents are
 * undefined after the call: a backend that would otherwise copy the input to
 * protect it from a kernel that writes through its input pointer may skip
 * that copy. Use it only when the input is not needed afterwards.
 *
 * @param handle FFT handle
 * @param time_in_clobbered Real input [fft_size]; contents undefined on return
 * @param complex_out Complex output [n_freqs]
 */
void fft_forward_scratch(FftHandle* handle, float* time_in_clobbered, Complex* complex_out);

/**
 * Inverse FFT: complex input -> real output, input clobber PERMITTED.
 *
 * Same transform as fft_inverse(), but the input buffer's contents are
 * undefined after the call (see fft_forward_scratch).
 *
 * @param handle FFT handle
 * @param freq_in_clobbered Complex input [n_freqs]; contents undefined on return
 * @param real_out Real output [fft_size]
 */
void fft_inverse_scratch(FftHandle* handle, Complex* freq_in_clobbered, float* real_out);

/**
 * Compute magnitude spectrum from complex spectrum
 *
 * @param freq Complex spectrum [n_freqs]
 * @param magnitude Output magnitude [n_freqs]
 * @param n_freqs Number of frequency bins
 */
void fft_magnitude(const Complex* freq, float* magnitude, int n_freqs);

/**
 * Compute power spectrum from complex spectrum
 *
 * @param freq Complex spectrum [n_freqs]
 * @param power Output power (magnitude^2) [n_freqs]
 * @param n_freqs Number of frequency bins
 */
void fft_power(const Complex* freq, float* power, int n_freqs);

/**
 * Compute phase spectrum from complex spectrum (atan2f(im, re) per bin).
 *
 * @param freq Complex spectrum [n_freqs]
 * @param phase Output phase (radians) [n_freqs]
 * @param n_freqs Number of frequency bins
 */
void fft_phase(const Complex* freq, float* phase, int n_freqs);

/**
 * Reconstruct a complex spectrum from magnitude + phase
 * (spectrum[k] = magnitude[k] * (cosf(phase[k]) + i*sinf(phase[k]))).
 *
 * @param magnitude Input magnitude [n_freqs]
 * @param phase Input phase (radians) [n_freqs]
 * @param spectrum Output complex spectrum [n_freqs]
 * @param n_freqs Number of frequency bins
 */
void fft_from_mag_phase(const float* magnitude, const float* phase,
                         Complex* spectrum, int n_freqs);

/**
 * Apply gain to complex spectrum (in-place)
 *
 * @param freq Complex spectrum [n_freqs]
 * @param gain Gain array [n_freqs]
 * @param n_freqs Number of frequency bins
 */
void fft_apply_gain(Complex* freq, const float* gain, int n_freqs);

#ifdef __cplusplus
}
#endif

#undef AC_FFT_RESTRICT

#endif // FFT_WRAPPER_H
