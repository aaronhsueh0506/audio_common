/**
 * fft_wrapper.c — fft_wrapper.h implementation using KISS FFT.
 *
 * Portable backend: real-to-complex FFT through KISS FFT's complex FFT
 * (kiss_fft_scalar = float). For the ARM NEON build see fft_wrapper_ne10.c.
 *
 * Convention (numpy):
 *     fft_forward(x)  == rfft(x), unnormalised
 *     fft_inverse(X)  == irfft(X), normalised by 1/N
 * KISS computes in float32, so results match numpy's float64 FFT to about
 * 1e-5, not bit for bit.
 *
 * Static-memory API (fft_get_mem_size / fft_init): KISS places its configs
 * into a caller buffer, so the static path is fully heap-free (the FftHandle,
 * the work buffers and both KISS configs live in the caller's block).
 */
#include "fft_wrapper.h"
#include "kiss_fft.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__ARM_NEON) && defined(__aarch64__) && \
    !defined(SIMD_KERNELS_FORCE_SCALAR)
#include <arm_neon.h>
#include "simd_kernels.h"   /* sk__cquad_load/sk__cquad_store */
#endif

/* Accepted sizes: power of two in [16, 8192]. The NE10 backend accepts the
 * same set, so both backends behave alike behind fft_wrapper.h. */
static int fft_size_in_range(int fft_size) {
    return fft_size >= 16 && fft_size <= 8192 && (fft_size & (fft_size - 1)) == 0;
}

/* Internal FFT handle structure */
struct FftHandle {
    int fft_size;
    int n_freqs;            /* fft_size/2 + 1 */

    kiss_fft_cfg fft_cfg;   /* Forward FFT config */
    kiss_fft_cfg ifft_cfg;  /* Inverse FFT config */

    kiss_fft_cpx* fft_in;   /* Complex input buffer [fft_size]  */
    kiss_fft_cpx* fft_out;  /* Complex output buffer [fft_size] */

    int pool_owned;         /* 1 if placed via fft_init (no free in destroy) */
};

/* --- construction (heap) -------------------------------------------------- */

FftHandle* fft_create(int fft_size) {
    if (!fft_size_in_range(fft_size)) {
        return NULL;
    }

    FftHandle* h = (FftHandle*)calloc(1, sizeof(FftHandle));
    if (!h) return NULL;

    h->fft_size   = fft_size;
    h->n_freqs    = fft_size / 2 + 1;
    h->pool_owned = 0;

    h->fft_cfg  = kiss_fft_alloc(fft_size, 0, NULL, NULL);  /* Forward */
    h->ifft_cfg = kiss_fft_alloc(fft_size, 1, NULL, NULL);  /* Inverse */
    if (!h->fft_cfg || !h->ifft_cfg) { fft_destroy(h); return NULL; }

    h->fft_in  = (kiss_fft_cpx*)calloc(fft_size, sizeof(kiss_fft_cpx));
    h->fft_out = (kiss_fft_cpx*)calloc(fft_size, sizeof(kiss_fft_cpx));
    if (!h->fft_in || !h->fft_out) { fft_destroy(h); return NULL; }

    return h;
}

/* --- construction (static pool, heap-free) -------------------------------- */

size_t fft_get_mem_size(int fft_size) {
    if (!fft_size_in_range(fft_size)) return 0;
    size_t lf = 0, li = 0;
    kiss_fft_alloc(fft_size, 0, NULL, &lf);   /* query forward cfg size */
    kiss_fft_alloc(fft_size, 1, NULL, &li);   /* query inverse cfg size */
    size_t total = 0;
    total = ck_field_size(total, 1, sizeof(FftHandle));
    total = ck_field_size(total, (size_t)fft_size, sizeof(kiss_fft_cpx));  /* fft_in  */
    total = ck_field_size(total, (size_t)fft_size, sizeof(kiss_fft_cpx));  /* fft_out */
    total = ck_field_size(total, 1, lf);                                   /* fft_cfg  */
    total = ck_field_size(total, 1, li);                                   /* ifft_cfg */
    return MEM_SIZE_INVALID(total) ? 0 : total;
}

FftHandle* fft_init(void* mem, size_t mem_size, int fft_size) {
    if (!mem || !fft_size_in_range(fft_size)) return NULL;
    if (!MEM_IS_ALIGNED16(mem)) return NULL;
    if (mem_size == 0) return NULL;
    size_t need = fft_get_mem_size(fft_size);
    /* need == 0: the size arithmetic overflowed. */
    if (need == 0 || mem_size < need) return NULL;

    uint8_t* ptr = (uint8_t*)mem;
    FftHandle* h = (FftHandle*)ptr;
    ptr += ALIGN16(sizeof(FftHandle));
    memset(h, 0, sizeof(FftHandle));

    h->fft_size   = fft_size;
    h->n_freqs    = fft_size / 2 + 1;
    h->pool_owned = 1;

    h->fft_in = (kiss_fft_cpx*)ptr;
    ptr += ALIGN16((size_t)fft_size * sizeof(kiss_fft_cpx));
    h->fft_out = (kiss_fft_cpx*)ptr;
    ptr += ALIGN16((size_t)fft_size * sizeof(kiss_fft_cpx));
    memset(h->fft_in,  0, (size_t)fft_size * sizeof(kiss_fft_cpx));
    memset(h->fft_out, 0, (size_t)fft_size * sizeof(kiss_fft_cpx));

    size_t lf = 0;
    kiss_fft_alloc(fft_size, 0, NULL, &lf);
    h->fft_cfg = kiss_fft_alloc(fft_size, 0, ptr, &lf);
    ptr += ALIGN16(lf);

    size_t li = 0;
    kiss_fft_alloc(fft_size, 1, NULL, &li);
    h->ifft_cfg = kiss_fft_alloc(fft_size, 1, ptr, &li);
    ptr += ALIGN16(li);

    if (!h->fft_cfg || !h->ifft_cfg) return NULL;
    return h;
}

void fft_destroy(FftHandle* h) {
    if (!h) return;
    if (h->pool_owned) return;   /* pool path: caller owns the whole buffer */

    if (h->fft_cfg)  kiss_fft_free(h->fft_cfg);
    if (h->ifft_cfg) kiss_fft_free(h->ifft_cfg);
    if (h->fft_in)   free(h->fft_in);
    if (h->fft_out)  free(h->fft_out);
    free(h);
}

int fft_get_size(const FftHandle* h)    { return h ? h->fft_size : 0; }
int fft_get_n_freqs(const FftHandle* h) { return h ? h->n_freqs  : 0; }

/* --- forward: rfft -------------------------------------------------------- */

void fft_forward(FftHandle* h, const float* restrict real_in, Complex* restrict complex_out) {
    if (!h || !real_in || !complex_out) return;

    int n = h->fft_size;

    /* Staging pointers are read out of the handle once: a store through
     * h->fft_in[] could, for all the compiler can prove, alias the handle's
     * own fft_in/fft_out members, which would force a reload per element and
     * block vectorisation of the interleaved store below. */
    kiss_fft_cpx* fin  = h->fft_in;
    kiss_fft_cpx* fout = h->fft_out;

    /* Copy real input to complex buffer (imaginary = 0) */
    for (int i = 0; i < n; i++) {
        fin[i].r = real_in[i];
        fin[i].i = 0.0f;
    }

    kiss_fft(h->fft_cfg, fin, fout);

    /* Copy first n_freqs bins to output */
    for (int k = 0; k < h->n_freqs; k++) {
        complex_out[k].r = fout[k].r;
        complex_out[k].i = fout[k].i;
    }
}

/* --- inverse: irfft (1/N normalised) -------------------------------------- */

void fft_inverse(FftHandle* h, const Complex* restrict complex_in, float* restrict real_out) {
    if (!h || !complex_in || !real_out) return;

    int n = h->fft_size;
    int n_freqs = h->n_freqs;

    /* Staging pointers read once, as in fft_forward. */
    kiss_fft_cpx* fin  = h->fft_in;
    kiss_fft_cpx* fout = h->fft_out;

    /* Reconstruct full spectrum with conjugate symmetry: X[k] = conj(X[N-k]) */
    for (int k = 0; k < n_freqs; k++) {
        fin[k].r = complex_in[k].r;
        fin[k].i = complex_in[k].i;
    }
    for (int k = 1; k < n_freqs - 1; k++) {
        fin[n - k].r =  complex_in[k].r;
        fin[n - k].i = -complex_in[k].i;  /* Conjugate */
    }

    kiss_fft(h->ifft_cfg, fin, fout);

    /* KISS FFT doesn't scale, so divide by N (numpy irfft convention) */
    float scale = 1.0f / (float)n;
    for (int i = 0; i < n; i++) {
        real_out[i] = fout[i].r * scale;
    }
}

/* --- scratch variants (the input buffer's contents are undefined afterwards) --
 * The plain functions already stage the input in a private buffer and never
 * write to the caller's, so the scratch entry points simply forward. */

void fft_forward_scratch(FftHandle* h, float* time_in_clobbered, Complex* complex_out) {
    fft_forward(h, time_in_clobbered, complex_out);
}

void fft_inverse_scratch(FftHandle* h, Complex* freq_in_clobbered, float* real_out) {
    fft_inverse(h, freq_in_clobbered, real_out);
}

/* --- spectral helpers ----------------------------------------------------- */

void fft_magnitude(const Complex* spectrum, float* magnitude, int n_freqs) {
    if (!spectrum || !magnitude) return;
    int k = 0;
#if defined(__ARM_NEON) && defined(__aarch64__) && \
    !defined(SIMD_KERNELS_FORCE_SCALAR)
    /* Unlike fft_power, re*re + im*im here is two separately rounded
     * multiplies and an add (the TU builds with -ffp-contract=off), so the
     * NEON path uses vmulq/vmulq/vaddq and never vfmaq to stay bit-identical
     * to the scalar loop below. vsqrtq_f32 is correctly rounded like sqrtf. */
    for (; k + 4 <= n_freqs; k += 4) {
        float32x4x2_t v = sk__cquad_load(spectrum + k);
        float32x4_t re = v.val[0], im = v.val[1];
        float32x4_t sumsq = vaddq_f32(vmulq_f32(re, re), vmulq_f32(im, im));
        vst1q_f32(magnitude + k, vsqrtq_f32(sumsq));
    }
#endif
    for (; k < n_freqs; k++) {
        float re = spectrum[k].r, im = spectrum[k].i;
        magnitude[k] = sqrtf(re * re + im * im);
    }
}

void fft_power(const Complex* spectrum, float* power, int n_freqs) {
    if (!spectrum || !power) return;
    int k = 0;
#if defined(__ARM_NEON) && defined(__aarch64__) && \
    !defined(SIMD_KERNELS_FORCE_SCALAR)
    /* fmaf(re, re, im * im): im * im rounded on its own, then re * re fused
     * into it. The scalar tail calls fmaf explicitly and this NEON path
     * matches it lane for lane (vmulq_f32, then vfmaq_f32). */
    for (; k + 4 <= n_freqs; k += 4) {
        float32x4x2_t v = sk__cquad_load(spectrum + k);
        float32x4_t re = v.val[0], im = v.val[1];
        float32x4_t p = vfmaq_f32(vmulq_f32(im, im), re, re);
        vst1q_f32(power + k, p);
    }
#endif
    for (; k < n_freqs; k++) {
        float re = spectrum[k].r, im = spectrum[k].i;
        power[k] = fmaf(re, re, im * im);
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
    int k = 0;
#if defined(__ARM_NEON) && defined(__aarch64__) && \
    !defined(SIMD_KERNELS_FORCE_SCALAR)
    /* Multiplies only: nothing to fuse. */
    for (; k + 4 <= n_freqs; k += 4) {
        float32x4x2_t v = sk__cquad_load(spectrum + k);
        float32x4_t g = vld1q_f32(gain + k);
        float32x4x2_t r;
        r.val[0] = vmulq_f32(v.val[0], g);
        r.val[1] = vmulq_f32(v.val[1], g);
        sk__cquad_store(spectrum + k, r);
    }
#endif
    for (; k < n_freqs; k++) {
        spectrum[k].r *= gain[k];
        spectrum[k].i *= gain[k];
    }
}
