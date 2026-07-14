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

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

/* This TU calls the _neon-suffixed NE10 kernels (ne10_fft_r2c_1d_float32_neon /
 * ne10_fft_c2r_1d_float32_neon) directly and unconditionally -- there is no _c
 * (scalar) fallback path and no runtime dispatch through NE10's function
 * pointers (that dispatch is set up by ne10_init(), which this TU no longer
 * calls -- see fft_create/fft_init below). So compile-time NEON availability
 * is a hard requirement, not a soft one; fail the build loudly instead of
 * emitting calls to intrinsics the target can't execute. */
#if !(defined(__ARM_NEON) && defined(__aarch64__))
#error "NE10 backend requires compile-time NEON (__ARM_NEON && __aarch64__)"
#endif

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

/* --- construction (static pool, heap-free) ---------------------------------
 * The handle, work buffers (real_buf/cpx_buf), AND the NE10 R2C/C2R twiddle
 * config are all bump-allocated from the caller's block. Standard NE10 has no
 * external-memory API for the twiddle config, so this repo vendors one
 * (lib/ne10/VENDORED.md patch P0001: ne10_fft_r2c_mem_size_float32 /
 * ne10_fft_init_r2c_float32_ext in NE10_rfft_float32.c). The heap path
 * (fft_create -> ne10_fft_alloc_r2c_float32) is now a thin malloc() wrapper
 * around that same _ext function, so heap and pool configs are bit-identical
 * by construction (one twiddle code path). fft_init is therefore fully
 * heap-free end to end: no malloc anywhere from fft_init() through
 * fft_destroy(), matching the KISS backend (fft_wrapper.c). */

size_t fft_get_mem_size(int fft_size) {
    if (fft_size <= 0 || (fft_size & (fft_size - 1)) != 0) return 0;
    int n_freqs = fft_size / 2 + 1;
    size_t total = 0;
    total = ck_field_size(total, 1, sizeof(FftHandle));
    total = ck_field_size(total, (size_t)fft_size, sizeof(ne10_float32_t));        /* real_buf */
    total = ck_field_size(total, (size_t)n_freqs, sizeof(ne10_fft_cpx_float32_t)); /* cpx_buf  */
    /* NE10 R2C/C2R twiddle config (P0001 external-memory init) — LAST field. */
    total = ck_add_size(total, ck_align16_size((size_t)ne10_fft_r2c_mem_size_float32(fft_size)));
    return MEM_SIZE_INVALID(total) ? 0 : total;
}

FftHandle* fft_init(void* mem, size_t mem_size, int fft_size) {
    if (!mem || fft_size <= 0 || (fft_size & (fft_size - 1)) != 0) return NULL;
    if (!MEM_IS_ALIGNED16(mem)) return NULL;  /* F07: reject a misaligned base before any write */
    if (mem_size == 0) return NULL;
    size_t need = fft_get_mem_size(fft_size);
    /* need==0 means fft_get_mem_size's own arithmetic overflowed -- no mem_size
     * can ever satisfy that (mem_size < 0 is never true for a size_t), so it
     * must be rejected explicitly rather than falling through the compare. */
    if (need == 0 || mem_size < need) return NULL;

    memset(mem, 0, need);  /* calloc-equivalent */
    uint8_t* cursor = (uint8_t*)mem;

    FftHandle* h = (FftHandle*)cursor;
    cursor += ALIGN16(sizeof(FftHandle));

    h->fft_size = fft_size;
    h->n_freqs = fft_size / 2 + 1;
    h->pool_owned = 1;

    // Work buffers from the caller's block.
    h->real_buf = (ne10_float32_t*)cursor;
    cursor += ALIGN16((size_t)fft_size * sizeof(ne10_float32_t));
    h->cpx_buf = (ne10_fft_cpx_float32_t*)cursor;
    cursor += ALIGN16((size_t)h->n_freqs * sizeof(ne10_fft_cpx_float32_t));

    // R2C config — carved from the same caller block (P0001 external-memory
    // init): no NE10-internal malloc on this path.
    h->r2c_cfg = ne10_fft_init_r2c_float32_ext(cursor, fft_size);
    cursor += ALIGN16((size_t)ne10_fft_r2c_mem_size_float32(fft_size));
    if (!h->r2c_cfg) return NULL;

    return h;
}

void fft_destroy(FftHandle* h) {
    if (!h) return;

    // Pool path: handle, work buffers, AND the r2c cfg all live in the
    // caller's block (P0001 external-memory init) -- nothing here was ever
    // malloc'd, so there is nothing to release, ever. Checking this BEFORE
    // touching h->r2c_cfg (unlike the old heap-cfg design) makes destroy a
    // true unconditional no-op on this path: idempotent under any number of
    // repeat calls, not just the first one (F08).
    if (h->pool_owned) return;

    if (h->r2c_cfg) { ne10_fft_destroy_r2c_float32(h->r2c_cfg); h->r2c_cfg = NULL; }
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

    // Copy input to work buffer (NE10 R2C may modify input; real_in is the
    // caller's buffer and fft_forward's contract leaves it unmodified).
    memcpy(h->real_buf, real_in, h->fft_size * sizeof(float));

    // Forward R2C FFT: real[fft_size] -> complex[n_freqs], written straight
    // into the caller's output buffer. ne10_fft_cpx_float32_t has the same
    // layout as Complex ({float r, float i}), so the kernel produces the
    // same bytes whether its destination is h->cpx_buf or complex_out --
    // the output-side staging copy through h->cpx_buf was pure overhead and
    // is elided here (h->cpx_buf is still used as the fft_inverse input
    // staging buffer below).
    ne10_fft_r2c_1d_float32_neon((ne10_fft_cpx_float32_t*)complex_out, h->real_buf, h->r2c_cfg);
}

void fft_inverse(FftHandle* h, const Complex* complex_in, float* real_out) {
    if (!h || !complex_in || !real_out) return;

    // Copy n_freqs bins to work buffer (NE10 C2R may modify input; complex_in
    // is the caller's buffer and fft_inverse's contract leaves it unmodified).
    memcpy(h->cpx_buf, complex_in, h->n_freqs * sizeof(ne10_fft_cpx_float32_t));

    // Inverse C2R FFT: complex[n_freqs] -> real[fft_size], written straight
    // into the caller's output buffer -- the output-side staging copy
    // through h->real_buf was pure overhead and is elided here (NE10 C2R
    // IFFT auto-scales by 1/N, official doc confirmed; h->real_buf is still
    // used as the fft_forward input staging buffer above).
    ne10_fft_c2r_1d_float32_neon(real_out, h->cpx_buf, h->r2c_cfg);
}

/* --- scratch variants (caller's input buffer contents left undefined) -----
 * NE10's R2C/C2R kernels are documented (and confirmed by reading
 * NE10_rfft_float32.neonintrinsic.c's ne10_fft_c2r_1d_float32_neon default
 * case: it temporarily repurposes fin[0].i to carry the Nyquist bin's real
 * part -- `fin[0].i = fin[cfg->nfft>>1].r;` ... `fin[0].i = 0.0f;` -- and
 * the r2c side is documented the same way) to possibly write through their
 * input pointer. fft_forward/fft_inverse protect the caller's buffer with a
 * staging copy into h->real_buf/h->cpx_buf for exactly that reason. These
 * _scratch entry points drop that protection: they feed the caller's own
 * buffer straight to the kernel as its input AND (per the output elision
 * above) its output is already the caller's other buffer -- so there is no
 * staging copy left at all, matching the "contents undefined after the
 * call" contract in fft_wrapper.h.
 *
 * Alignment: ne10_fft_r2c_1d_float32_neon / ne10_fft_c2r_1d_float32_neon
 * take plain `ne10_float32_t*` / `ne10_fft_cpx_float32_t*` pointers (see
 * NE10_dsp.h) with no alignment attribute -- NE10_BYTE_ALIGNMENT is only
 * used internally when carving the twiddle-factor cfg buffer, never applied
 * to the fin/fout data pointers. AArch64 NEON load/store instructions do
 * not fault on unaligned addresses, so a caller buffer that is only
 * ALIGN16-guaranteed (pools/malloc; see mem_align.h) rather than aligned to
 * some larger NE10-internal boundary is safe to pass here. */

void fft_forward_scratch(FftHandle* h, float* time_in_clobbered, Complex* complex_out) {
    if (!h || !time_in_clobbered || !complex_out) return;
    ne10_fft_r2c_1d_float32_neon((ne10_fft_cpx_float32_t*)complex_out, time_in_clobbered, h->r2c_cfg);
}

void fft_inverse_scratch(FftHandle* h, Complex* freq_in_clobbered, float* real_out) {
    if (!h || !freq_in_clobbered || !real_out) return;
    ne10_fft_c2r_1d_float32_neon(real_out, (ne10_fft_cpx_float32_t*)freq_in_clobbered, h->r2c_cfg);
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

    int k = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
    /* Verified via `objdump -d` on the CURRENT build (no -ffp-contract=off
     * on this TU): clang contracts `re*re + im*im` into
     *   fmul s1, im, im
     *   fmadd s0, re, re, s1
     * i.e. fmaf(re, re, im*im) -- im*im rounded separately, then re*re fused
     * against it (byte-identical codegen to the KISS TU's fft_power). The
     * explicit scalar form below reproduces that exactly, and this NEON
     * path mirrors the same fused/unfused shape lane-for-lane (im*im via
     * vmulq_f32, then vfmaq_f32(that, re, re)). */
    for (; k + 4 <= n_freqs; k += 4) {
        float32x4x2_t v = vld2q_f32((const float*)(spectrum + k));
        float32x4_t re = v.val[0], im = v.val[1];
        float32x4_t p = vfmaq_f32(vmulq_f32(im, im), re, re);
        vst1q_f32(power + k, p);
    }
#endif
    for (; k < n_freqs; k++) {
        float re = spectrum[k].r;
        float im = spectrum[k].i;
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
