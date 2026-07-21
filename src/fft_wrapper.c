/**
 * fft_wrapper.c — fft_wrapper.h implementation using KISS FFT.
 *
 * Vendored from the NR repo (SE/NR/c_impl/src/fft_wrapper.c) so AEC and NR share
 * one portable FFT layer. Real-to-complex FFT via KISS FFT's complex FFT
 * (kiss_fft_scalar = float). For the ARM NEON build see fft_wrapper_ne10.c.
 *
 * Convention (matches the retired pocketfft backend's callers):
 *     fft_forward(x)  == rfft(x), unnormalised
 *     fft_inverse(X)  == irfft(X), normalised by 1/N
 * KISS computes in float32, so results match numpy's fp64 np.fft to ~float32
 * precision (~1e-5 e2e, a documented tolerance — NOT 0/0 bit-exact). The
 * non-FFT C logic remains bit-exact to Python.
 *
 * AEC addition over the NR original: the static-memory API (fft_get_mem_size /
 * fft_init). KISS supports cfg placement into a caller buffer, so the static
 * path is FULLY heap-free (the FftHandle, work buffers, and both kiss configs
 * all live in the caller pool).
 */
#include "fft_wrapper.h"
#include "kiss_fft.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

/* --- P0003 (re-review R05): fft_size whitelist ----------------------------
 * Mirrors fft_wrapper_ne10.c's fft_size_in_range: [16, 8192], power of two.
 * KISS's own arithmetic here already runs through the checked ck_* helpers
 * (mem_align.h) in 64-bit size_t, so it does not share the NE10 backend's
 * 32-bit-wraparound exposure -- but every real caller across AEC/NR/
 * Audio_ALG only ever derives fft_size from {256, 512, 1024}, and the two
 * backends are meant to be interchangeable behind the same fft_wrapper.h
 * contract, so this whitelist is applied here too for API symmetry (a
 * program that only ever validates against one backend should not discover
 * different accepted-input behavior when relinked against the other). */
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
        return NULL;  /* fft_size must be a power of 2 in [16, 8192] (P0003) */
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
    if (!fft_size_in_range(fft_size)) return 0;  /* P0003 */
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
    if (!mem || !fft_size_in_range(fft_size)) return NULL;  /* P0003 */
    if (!MEM_IS_ALIGNED16(mem)) return NULL;  /* F07: reject a misaligned base before any write */
    if (mem_size == 0) return NULL;
    size_t need = fft_get_mem_size(fft_size);
    /* need==0 means fft_get_mem_size's own arithmetic overflowed -- no mem_size
     * can ever satisfy that (mem_size < 0 is never true for a size_t), so it
     * must be rejected explicitly rather than falling through the compare. */
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

void fft_forward(FftHandle* h, const float* real_in, Complex* complex_out) {
    if (!h || !real_in || !complex_out) return;

    int n = h->fft_size;

    /* Copy real input to complex buffer (imaginary = 0) */
    for (int i = 0; i < n; i++) {
        h->fft_in[i].r = real_in[i];
        h->fft_in[i].i = 0.0f;
    }

    kiss_fft(h->fft_cfg, h->fft_in, h->fft_out);

    /* Copy first n_freqs bins to output */
    for (int k = 0; k < h->n_freqs; k++) {
        complex_out[k].r = h->fft_out[k].r;
        complex_out[k].i = h->fft_out[k].i;
    }
}

/* --- inverse: irfft (1/N normalised) -------------------------------------- */

void fft_inverse(FftHandle* h, const Complex* complex_in, float* real_out) {
    if (!h || !complex_in || !real_out) return;

    int n = h->fft_size;
    int n_freqs = h->n_freqs;

    /* Reconstruct full spectrum with conjugate symmetry: X[k] = conj(X[N-k]) */
    for (int k = 0; k < n_freqs; k++) {
        h->fft_in[k].r = complex_in[k].r;
        h->fft_in[k].i = complex_in[k].i;
    }
    for (int k = 1; k < n_freqs - 1; k++) {
        h->fft_in[n - k].r =  complex_in[k].r;
        h->fft_in[n - k].i = -complex_in[k].i;  /* Conjugate */
    }

    kiss_fft(h->ifft_cfg, h->fft_in, h->fft_out);

    /* KISS FFT doesn't scale, so divide by N (numpy irfft convention) */
    float scale = 1.0f / (float)n;
    for (int i = 0; i < n; i++) {
        real_out[i] = h->fft_out[i].r * scale;
    }
}

/* --- scratch variants (caller's input buffer contents left undefined) ----
 * KISS's fft_forward/fft_inverse already copy the caller's input into a
 * private buffer (h->fft_in) before touching kiss_fft's engine and never
 * write back into the caller's buffer -- the staging here is structural,
 * not a defensive copy against a clobbering kernel. So there is nothing to
 * elide: real_in/complex_in are never modified by the plain API either.
 * The _scratch entry points just forward to it; the API contract ("contents
 * undefined after the call") is satisfied trivially since it only permits
 * clobbering, it doesn't require it. */

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
#if defined(__ARM_NEON) && defined(__aarch64__)
    /* UNLIKE fft_power() just below, this TU's fft_magnitude() does NOT call
     * fmaf() anywhere -- `re*re + im*im` is a plain separately-rounded
     * multiply/multiply/add, and this TU builds with -ffp-contract=off (see
     * the Makefile's FP-contraction policy), so the compiler may not fuse it
     * either. This NEON path must therefore mirror that UNFUSED shape
     * (vmulq_f32/vmulq_f32/vaddq_f32, never vfmaq_f32) to stay bit-identical
     * to the scalar reference below -- do not copy fft_power()'s fused
     * pattern here, that would silently change this function's rounding.
     * vsqrtq_f32 is IEEE-754 correctly-rounded, matching scalar sqrtf()
     * lane-for-lane (same argument as sk_fast_sqrt_f32's USE_STANDARD_MATH
     * path in simd_kernels.h). */
    for (; k + 4 <= n_freqs; k += 4) {
        float32x4x2_t v = vld2q_f32((const float*)(spectrum + k));
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
#if defined(__ARM_NEON) && defined(__aarch64__)
    /* Verified via `objdump -d` on the CURRENT build (no -ffp-contract=off
     * on this TU): clang contracts `re*re + im*im` into
     *   fmul s1, im, im
     *   fmadd s0, re, re, s1
     * i.e. fmaf(re, re, im*im) -- im*im rounded separately, then re*re fused
     * against it. The explicit scalar form below reproduces that exactly,
     * and this NEON path mirrors the same fused/unfused shape lane-for-lane
     * (im*im via vmulq_f32, then vfmaq_f32(that, re, re)). */
    for (; k + 4 <= n_freqs; k += 4) {
        float32x4x2_t v = vld2q_f32((const float*)(spectrum + k));
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
#if defined(__ARM_NEON) && defined(__aarch64__)
    /* Pure multiplies, no add -- nothing for fp-contraction to fuse either
     * way, so this NEON path is a plain deinterleave/vmulq/reinterleave. */
    for (; k + 4 <= n_freqs; k += 4) {
        float32x4x2_t v = vld2q_f32((const float*)(spectrum + k));
        float32x4_t g = vld1q_f32(gain + k);
        float32x4x2_t r;
        r.val[0] = vmulq_f32(v.val[0], g);
        r.val[1] = vmulq_f32(v.val[1], g);
        vst2q_f32((float*)(spectrum + k), r);
    }
#endif
    for (; k < n_freqs; k++) {
        spectrum[k].r *= gain[k];
        spectrum[k].i *= gain[k];
    }
}
