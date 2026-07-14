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
#include "mem_align.h"   /* ALIGN16 — the one shared alignment for every module */

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
 * @param fft_size FFT size (must be power of 2)
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
 * 16-byte aligned. Returns NULL if `mem_size` is too small.
 */
size_t     fft_get_mem_size(int fft_size);
FftHandle* fft_init(void* mem, size_t mem_size, int fft_size);

/**
 * Destroy FFT handle. No-op when handle was created via fft_init().
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
 * @param time_in Real input [fft_size]
 * @param freq_out Complex output [n_freqs]
 */
void fft_forward(FftHandle* handle, const float* time_in, Complex* freq_out);

/**
 * Inverse FFT: complex input -> real output
 *
 * @param handle FFT handle
 * @param freq_in Complex input [n_freqs]
 * @param time_out Real output [fft_size]
 */
void fft_inverse(FftHandle* handle, const Complex* freq_in, float* time_out);

/**
 * Forward FFT: real input -> complex output, input clobber PERMITTED.
 *
 * Same transform as fft_forward(), but the caller's input buffer contents
 * are UNDEFINED after the call: a backend that would otherwise stage a
 * private defensive copy of the input (to protect it from a kernel that
 * may write through its input pointer) is free to skip that copy and let
 * the kernel operate on `time_in_clobbered` directly. Use only when the
 * caller has no further use for the input buffer after this call.
 *
 * @param handle FFT handle
 * @param time_in_clobbered Real input [fft_size]; contents undefined on return
 * @param complex_out Complex output [n_freqs]
 */
void fft_forward_scratch(FftHandle* handle, float* time_in_clobbered, Complex* complex_out);

/**
 * Inverse FFT: complex input -> real output, input clobber PERMITTED.
 *
 * Same transform as fft_inverse(), but the caller's input buffer contents
 * are UNDEFINED after the call. See fft_forward_scratch() for the rationale.
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

#endif // FFT_WRAPPER_H
