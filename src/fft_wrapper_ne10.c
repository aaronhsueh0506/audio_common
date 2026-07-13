/**
 * fft_wrapper_ne10.c - FFT Wrapper Implementation using NE10 R2C/C2R
 *
 * ARM NEON optimized FFT for real signals.
 * Uses ne10_fft_r2c (forward) and ne10_fft_c2r (inverse).
 *
 * For portable KISS FFT version, see fft_wrapper.c
 */

#include "fft_wrapper.h"
#include "NE10.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int ne10_initialized = 0;

// Internal FFT handle structure
struct FftHandle {
    int fft_size;
    int n_freqs;                        // fft_size/2 + 1

    ne10_fft_r2c_cfg_float32_t r2c_cfg; // Single config (forward & inverse)
    ne10_float32_t*            real_buf; // [fft_size] real work buffer
    ne10_fft_cpx_float32_t*    cpx_buf;  // [n_freqs] complex work buffer

    int pool_owned;  // 1 if placed via fft_init (no free of handle/bufs in destroy)
};

/* --- construction (heap) -------------------------------------------------- */

FftHandle* fft_create(int fft_size) {
    if (fft_size <= 0 || (fft_size & (fft_size - 1)) != 0) {
        // fft_size must be power of 2
        return NULL;
    }

    // One-time NE10 initialization
    if (!ne10_initialized) {
        if (ne10_init() != NE10_OK) {
            return NULL;
        }
        ne10_initialized = 1;
    }

    FftHandle* h = (FftHandle*)calloc(1, sizeof(FftHandle));
    if (!h) return NULL;

    h->fft_size = fft_size;
    h->n_freqs = fft_size / 2 + 1;
    h->pool_owned = 0;

    // Allocate R2C config
    h->r2c_cfg = ne10_fft_alloc_r2c_float32(fft_size);
    if (!h->r2c_cfg) {
        fft_destroy(h);
        return NULL;
    }

    // Allocate work buffers
    h->real_buf = (ne10_float32_t*)calloc(fft_size, sizeof(ne10_float32_t));
    h->cpx_buf = (ne10_fft_cpx_float32_t*)calloc(h->n_freqs, sizeof(ne10_fft_cpx_float32_t));

    if (!h->real_buf || !h->cpx_buf) {
        fft_destroy(h);
        return NULL;
    }

    return h;
}

/* --- construction (static pool, caller-owned buffer) -----------------------
 * The handle and work buffers (real_buf/cpx_buf) are bump-allocated from the
 * caller's block. The NE10 R2C twiddle config is NOT pool-able: standard NE10
 * has no external-memory API for it, so ne10_fft_alloc_r2c_float32() still
 * does its own one-time internal malloc here (fft_destroy still releases it
 * unconditionally). The per-frame audio path (fft_forward/fft_inverse) stays
 * malloc-free either way. */

size_t fft_get_mem_size(int fft_size) {
    if (fft_size <= 0 || (fft_size & (fft_size - 1)) != 0) return 0;
    int n_freqs = fft_size / 2 + 1;
    size_t total = 0;
    total += ALIGN16(sizeof(FftHandle));
    total += ALIGN16((size_t)fft_size * sizeof(ne10_float32_t));        /* real_buf */
    total += ALIGN16((size_t)n_freqs * sizeof(ne10_fft_cpx_float32_t)); /* cpx_buf  */
    /* NE10 twiddle cfg uses NE10-internal malloc — deliberately NOT counted. */
    return total;
}

FftHandle* fft_init(void* mem, size_t mem_size, int fft_size) {
    if (!mem || fft_size <= 0 || (fft_size & (fft_size - 1)) != 0) return NULL;
    if (mem_size < fft_get_mem_size(fft_size)) return NULL;

    // One-time NE10 initialization
    if (!ne10_initialized) {
        if (ne10_init() != NE10_OK) return NULL;
        ne10_initialized = 1;
    }

    memset(mem, 0, fft_get_mem_size(fft_size));  /* calloc-equivalent */
    uint8_t* cursor = (uint8_t*)mem;

    FftHandle* h = (FftHandle*)cursor;
    cursor += ALIGN16(sizeof(FftHandle));

    h->fft_size = fft_size;
    h->n_freqs = fft_size / 2 + 1;
    h->pool_owned = 1;

    // R2C config — NE10 owns this allocation (one-time internal malloc).
    h->r2c_cfg = ne10_fft_alloc_r2c_float32(fft_size);
    if (!h->r2c_cfg) return NULL;

    // Work buffers from the caller's block.
    h->real_buf = (ne10_float32_t*)cursor;
    cursor += ALIGN16((size_t)fft_size * sizeof(ne10_float32_t));
    h->cpx_buf = (ne10_fft_cpx_float32_t*)cursor;
    cursor += ALIGN16((size_t)h->n_freqs * sizeof(ne10_fft_cpx_float32_t));

    return h;
}

void fft_destroy(FftHandle* h) {
    if (!h) return;

    // NE10 twiddle cfg is never pool-owned; always release it.
    if (h->r2c_cfg) ne10_fft_destroy_r2c_float32(h->r2c_cfg);

    if (h->pool_owned) return;  /* handle + work buffers live in caller's block */

    if (h->real_buf) free(h->real_buf);
    if (h->cpx_buf) free(h->cpx_buf);

    free(h);
}

int fft_get_size(const FftHandle* h) {
    return h ? h->fft_size : 0;
}

int fft_get_n_freqs(const FftHandle* h) {
    return h ? h->n_freqs : 0;
}

void fft_forward(FftHandle* h, const float* real_in, Complex* complex_out) {
    if (!h || !real_in || !complex_out) return;

    // Copy input to work buffer (NE10 R2C may modify input)
    memcpy(h->real_buf, real_in, h->fft_size * sizeof(float));

    // Forward R2C FFT: real[fft_size] -> complex[n_freqs]
    ne10_fft_r2c_1d_float32_neon(h->cpx_buf, h->real_buf, h->r2c_cfg);

    // Copy output (ne10_fft_cpx_float32_t has same layout as Complex: {float r, float i})
    memcpy(complex_out, h->cpx_buf, h->n_freqs * sizeof(Complex));
}

void fft_inverse(FftHandle* h, const Complex* complex_in, float* real_out) {
    if (!h || !complex_in || !real_out) return;

    // Copy n_freqs bins to work buffer (NE10 C2R may modify input)
    memcpy(h->cpx_buf, complex_in, h->n_freqs * sizeof(ne10_fft_cpx_float32_t));

    // Inverse C2R FFT: complex[n_freqs] -> real[fft_size]
    ne10_fft_c2r_1d_float32_neon(h->real_buf, h->cpx_buf, h->r2c_cfg);

    // NE10 C2R IFFT auto-scales by 1/N (official doc confirmed)
    memcpy(real_out, h->real_buf, h->fft_size * sizeof(float));
}

void fft_magnitude(const Complex* spectrum, float* magnitude, int n_freqs) {
    if (!spectrum || !magnitude) return;

    for (int k = 0; k < n_freqs; k++) {
        float re = spectrum[k].r;
        float im = spectrum[k].i;
        magnitude[k] = sqrtf(re * re + im * im);
    }
}

void fft_power(const Complex* spectrum, float* power, int n_freqs) {
    if (!spectrum || !power) return;

    for (int k = 0; k < n_freqs; k++) {
        float re = spectrum[k].r;
        float im = spectrum[k].i;
        power[k] = re * re + im * im;
    }
}

void fft_phase(const Complex* spectrum, float* phase, int n_freqs) {
    if (!spectrum || !phase) return;

    for (int k = 0; k < n_freqs; k++) {
        phase[k] = atan2f(spectrum[k].i, spectrum[k].r);
    }
}

void fft_from_mag_phase(const float* magnitude, const float* phase,
                        Complex* spectrum, int n_freqs) {
    if (!magnitude || !phase || !spectrum) return;

    for (int k = 0; k < n_freqs; k++) {
        spectrum[k].r = magnitude[k] * cosf(phase[k]);
        spectrum[k].i = magnitude[k] * sinf(phase[k]);
    }
}

void fft_apply_gain(Complex* spectrum, const float* gain, int n_freqs) {
    if (!spectrum || !gain) return;

    for (int k = 0; k < n_freqs; k++) {
        spectrum[k].r *= gain[k];
        spectrum[k].i *= gain[k];
    }
}
