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
#include <stddef.h>   /* offsetof */

#if defined(__ARM_NEON) && defined(__aarch64__) && \
    !defined(SIMD_KERNELS_FORCE_SCALAR)
#include <arm_neon.h>
#include "simd_kernels.h"   /* sk__cquad_load/sk__cquad_store */
#endif

/* Complex ({float r, i}) and ne10_fft_cpx_float32_t are cast to each other
 * without copying, which needs identical size, alignment and member offsets.
 * The asserts turn a layout change on either side into a build error.
 *
 * The two struct types are distinct, so the casts are layout-compatible but
 * not strict-aliasing-compatible: the Makefile builds this object alone with
 * -fno-strict-aliasing. Do not enable LTO or whole-program optimisation for
 * it: a caller inlined across the NE10 call boundary into a unit built
 * without that flag would bring the aliasing assumption back. */
_Static_assert(sizeof(Complex) == sizeof(ne10_fft_cpx_float32_t),
               "Complex/ne10_fft_cpx_float32_t size mismatch -- fft_forward/"
               "fft_inverse cast between them without copying");
_Static_assert(_Alignof(Complex) == _Alignof(ne10_fft_cpx_float32_t),
               "Complex/ne10_fft_cpx_float32_t alignment mismatch -- cast "
               "between them is unsafe");
_Static_assert(offsetof(Complex, r) == offsetof(ne10_fft_cpx_float32_t, r),
               "Complex/ne10_fft_cpx_float32_t .r offset mismatch");
_Static_assert(offsetof(Complex, i) == offsetof(ne10_fft_cpx_float32_t, i),
               "Complex/ne10_fft_cpx_float32_t .i offset mismatch");

/* SIMD_KERNELS_FORCE_SCALAR routes every call below through
 * ne10_fft_r2c_1d_float32_c / ne10_fft_c2r_1d_float32_c instead of the NEON
 * kernels. Both variants are always compiled into the archive, so it is a
 * pure dispatch switch. */
#if defined(SIMD_KERNELS_FORCE_SCALAR)
#define NE10_R2C_1D_FLOAT32 ne10_fft_r2c_1d_float32_c
#define NE10_C2R_1D_FLOAT32 ne10_fft_c2r_1d_float32_c
#else
#define NE10_R2C_1D_FLOAT32 ne10_fft_r2c_1d_float32_neon
#define NE10_C2R_1D_FLOAT32 ne10_fft_c2r_1d_float32_neon
#endif

/* The kernels are called directly, not through NE10's runtime dispatch
 * (ne10_init is never called), so compile-time NEON is required. */
#if !(defined(__ARM_NEON) && defined(__aarch64__))
#error "NE10 backend requires compile-time NEON (__ARM_NEON && __aarch64__)"
#endif

/* Accepted sizes: power of two in [16, 8192], the range the vendored NE10 size
 * query and init accept (lib/ne10/VENDORED.md); anything else is rejected
 * here (0 / NULL) before it reaches NE10's 32-bit size arithmetic. */
static int fft_size_in_range(int fft_size) {
    return fft_size >= 16 && fft_size <= 8192 && (fft_size & (fft_size - 1)) == 0;
}

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
    if (!fft_size_in_range(fft_size)) {
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
 * The handle, the work buffers (real_buf/cpx_buf) and the NE10 R2C/C2R twiddle
 * config are all carved from the caller's block. Stock NE10 cannot place its
 * config in caller memory, so the vendored copy adds
 * ne10_fft_r2c_mem_size_float32 / ne10_fft_init_r2c_float32_ext
 * (lib/ne10/VENDORED.md). The heap path wraps the same init function, so heap
 * and pool configs are bit-identical. fft_init makes no malloc call. */

size_t fft_get_mem_size(int fft_size) {
    if (!fft_size_in_range(fft_size)) return 0;
    int n_freqs = fft_size / 2 + 1;
    size_t total = 0;
    total = ck_field_size(total, 1, sizeof(FftHandle));
    total = ck_field_size(total, (size_t)fft_size, sizeof(ne10_float32_t));        /* real_buf */
    total = ck_field_size(total, (size_t)n_freqs, sizeof(ne10_fft_cpx_float32_t)); /* cpx_buf  */
    /* NE10 R2C/C2R twiddle config: last field. */
    total = ck_add_size(total, ck_align16_size((size_t)ne10_fft_r2c_mem_size_float32(fft_size)));
    return MEM_SIZE_INVALID(total) ? 0 : total;
}

FftHandle* fft_init(void* mem, size_t mem_size, int fft_size) {
    if (!mem || !fft_size_in_range(fft_size)) return NULL;
    if (!MEM_IS_ALIGNED16(mem)) return NULL;
    if (mem_size == 0) return NULL;
    size_t need = fft_get_mem_size(fft_size);
    /* need == 0: the size arithmetic overflowed. */
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

    // R2C config, carved from the same block. The init function validates the
    // budget it is given against its own size query. The config is the last
    // field, so the budget is what remains of the block, capped at the
    // amount fft_get_mem_size reserved: a very large caller block must not
    // wrap when narrowed to ne10_uint32_t.
    size_t cfg_offset   = (size_t)(cursor - (uint8_t*)mem);
    size_t cfg_remaining = mem_size - cfg_offset;
    size_t cfg_reserved  = ck_align16_size((size_t)ne10_fft_r2c_mem_size_float32(fft_size));
    size_t cfg_budget    = (cfg_remaining < cfg_reserved) ? cfg_remaining : cfg_reserved;
    h->r2c_cfg = ne10_fft_init_r2c_float32_ext(cursor, (ne10_uint32_t)cfg_budget, fft_size);
    cursor += ALIGN16((size_t)ne10_fft_r2c_mem_size_float32(fft_size));
    if (!h->r2c_cfg) return NULL;

    return h;
}

void fft_destroy(FftHandle* h) {
    if (!h) return;

    // Pool path: the handle, the work buffers and the config live in the
    // caller's block and nothing was allocated, so this is a no-op, however
    // often it is called.
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

void fft_forward(FftHandle* h, const float* restrict real_in, Complex* restrict complex_out) {
    if (!h || !real_in || !complex_out) return;

    // Copy input to work buffer (NE10 R2C may modify input; real_in is the
    // caller's buffer and fft_forward's contract leaves it unmodified).
    memcpy(h->real_buf, real_in, h->fft_size * sizeof(float));

    // Forward R2C FFT: real[fft_size] -> complex[n_freqs], written straight
    // into the caller's output buffer (same layout as Complex).
    NE10_R2C_1D_FLOAT32((ne10_fft_cpx_float32_t*)complex_out, h->real_buf, h->r2c_cfg);
}

void fft_inverse(FftHandle* h, const Complex* restrict complex_in, float* restrict real_out) {
    if (!h || !complex_in || !real_out) return;

    // Copy n_freqs bins to work buffer (NE10 C2R may modify input; complex_in
    // is the caller's buffer and fft_inverse's contract leaves it unmodified).
    memcpy(h->cpx_buf, complex_in, h->n_freqs * sizeof(ne10_fft_cpx_float32_t));

    // Inverse C2R FFT: complex[n_freqs] -> real[fft_size], written straight
    // into the caller's output buffer. NE10 C2R scales by 1/N itself.
    NE10_C2R_1D_FLOAT32(real_out, h->cpx_buf, h->r2c_cfg);
}

/* --- scratch variants (the input buffer's contents are undefined afterwards) --
 * The NE10 kernels may write through their input pointer (the C2R kernel
 * borrows fin[0].i for the Nyquist bin), which is why the plain functions
 * stage the input in a private buffer. The scratch variants pass the caller's
 * buffer to the kernel directly and skip that copy.
 *
 * The kernels take plain float / ne10_fft_cpx_float32_t pointers without an
 * alignment requirement beyond the element type, so 16-byte aligned caller
 * buffers are sufficient. */

void fft_forward_scratch(FftHandle* h, float* time_in_clobbered, Complex* complex_out) {
    if (!h || !time_in_clobbered || !complex_out) return;
    NE10_R2C_1D_FLOAT32((ne10_fft_cpx_float32_t*)complex_out, time_in_clobbered, h->r2c_cfg);
}

void fft_inverse_scratch(FftHandle* h, Complex* freq_in_clobbered, float* real_out) {
    if (!h || !freq_in_clobbered || !real_out) return;
    NE10_C2R_1D_FLOAT32(real_out, (ne10_fft_cpx_float32_t*)freq_in_clobbered, h->r2c_cfg);
}

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
        float re = spectrum[k].r;
        float im = spectrum[k].i;
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
