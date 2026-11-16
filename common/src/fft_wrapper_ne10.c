/** fft_wrapper_ne10.c - FFT wrapper over NE10 R2C/C2R. */

#include "fft_wrapper.h"
#include "NE10.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"

/* Complex and ne10_fft_cpx_float32_t are cast to each other, which needs identical size, alignment and offsets (checked at compile time below and at run time by test/roundtrip.c).
 * The casts are not strict-aliasing compatible: this object is built with -fno-strict-aliasing and must not be built with LTO. */
typedef char ne10_cpx_size_check[(sizeof(Complex) == sizeof(ne10_fft_cpx_float32_t)) ? 1 : -1];
typedef char ne10_cpx_align_check[(offsetof(struct { char c; Complex t; }, t) == offsetof(struct { char c; ne10_fft_cpx_float32_t t; }, t)) ? 1 : -1];
typedef char ne10_cpx_offset_check[(offsetof(Complex, r) == offsetof(ne10_fft_cpx_float32_t, r) && offsetof(Complex, i) == offsetof(ne10_fft_cpx_float32_t, i)) ? 1 : -1];

/* NE10 uses its ARM32/ARM64 NEON kernels, or its C kernels on a target without NEON or in a scalar build. */
#if defined(SIMD_KERNELS_FORCE_SCALAR) || !defined(__ARM_NEON)
#define NE10_R2C_1D_FLOAT32 ne10_fft_r2c_1d_float32_c
#define NE10_C2R_1D_FLOAT32 ne10_fft_c2r_1d_float32_c
#else
#define NE10_R2C_1D_FLOAT32 ne10_fft_r2c_1d_float32_neon
#define NE10_C2R_1D_FLOAT32 ne10_fft_c2r_1d_float32_neon
#endif

#if !(defined(__arm__) || defined(__aarch64__))
#error "NE10 backend requires an ARM32 or ARM64 compiler target"
#endif

/* Accepted sizes: power of two in [16, 8192]; anything else is rejected (0 / NULL) before NE10's 32-bit size arithmetic. */
static int fft_size_in_range(int fft_size) {
    return fft_size >= 16 && fft_size <= 8192 && (fft_size & (fft_size - 1)) == 0;
}

struct FftHandle {
    int fft_size;
    int n_freqs;                        // fft_size/2 + 1

    ne10_fft_r2c_cfg_float32_t r2c_cfg; // Single config (forward & inverse)

    /* Input staging for the preserving calls. Forward uses real_buf, inverse uses cpx_buf; each call overwrites its view before
     * the kernel reads it, so the two views share one block. n_freqs complex values always cover fft_size floats. */
    ne10_float32_t*            real_buf; // [fft_size] view of the shared block
    ne10_fft_cpx_float32_t*    cpx_buf;  // [n_freqs] shared block, owned (heap path)

    int pool_owned;  // 1 if placed via fft_init (no free of handle/bufs in destroy)
};

FftHandle* fft_create(int fft_size) {
    if (!fft_size_in_range(fft_size)) {
        return NULL;
    }

    FftHandle* h = (FftHandle*)calloc(1, sizeof(FftHandle));
    if (!h) return NULL;

    h->fft_size = fft_size;
    h->n_freqs = fft_size / 2 + 1;
    h->pool_owned = 0;

    h->r2c_cfg = ne10_fft_alloc_r2c_float32(fft_size);
    if (!h->r2c_cfg) {
        fft_destroy(h);
        return NULL;
    }

    h->cpx_buf = (ne10_fft_cpx_float32_t*)calloc(h->n_freqs, sizeof(ne10_fft_cpx_float32_t));
    if (!h->cpx_buf) {
        fft_destroy(h);
        return NULL;
    }
    h->real_buf = (ne10_float32_t*)(void*)h->cpx_buf;

    return h;
}

/* Static pool construction: the handle, the shared staging block and the NE10 R2C/C2R twiddle config are carved from the caller's block
 * (the vendored NE10 adds ne10_fft_r2c_mem_size_float32 / ne10_fft_init_r2c_float32_ext). fft_init makes no malloc call. */

size_t fft_get_mem_size(int fft_size) {
    if (!fft_size_in_range(fft_size)) return 0;
    int n_freqs = fft_size / 2 + 1;
    size_t total = 0;
    total = ck_field_size(total, 1, sizeof(FftHandle));
    total = ck_field_size(total, (size_t)n_freqs, sizeof(ne10_fft_cpx_float32_t));
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

    memset(mem, 0, need);
    uint8_t* cursor = (uint8_t*)mem;

    FftHandle* h = (FftHandle*)cursor;
    cursor += ALIGN16(sizeof(FftHandle));

    h->fft_size = fft_size;
    h->n_freqs = fft_size / 2 + 1;
    h->pool_owned = 1;

    h->cpx_buf = (ne10_fft_cpx_float32_t*)cursor;
    h->real_buf = (ne10_float32_t*)(void*)h->cpx_buf;
    cursor += ALIGN16((size_t)h->n_freqs * sizeof(ne10_fft_cpx_float32_t));

    // R2C config, carved last: the budget is what remains of the block, capped at what fft_get_mem_size reserved so it cannot wrap ne10_uint32_t.
    size_t cfg_offset   = (size_t)(cursor - (uint8_t*)mem);
    size_t cfg_remaining = mem_size - cfg_offset;
    size_t cfg_reserved  = ck_align16_size((size_t)ne10_fft_r2c_mem_size_float32(fft_size));
    size_t cfg_budget    = (cfg_remaining < cfg_reserved) ? cfg_remaining : cfg_reserved;
    h->r2c_cfg = ne10_fft_init_r2c_float32_ext(cursor, (ne10_uint32_t)cfg_budget, fft_size);
    if (!h->r2c_cfg) return NULL;

    return h;
}

void fft_destroy(FftHandle* h) {
    if (!h) return;

    // Pool path: nothing to free, so repeated calls are safe.
    if (h->pool_owned) return;

    if (h->r2c_cfg) { ne10_fft_destroy_r2c_float32(h->r2c_cfg); h->r2c_cfg = NULL; }
    if (h->cpx_buf) free(h->cpx_buf);

    free(h);
}

int fft_get_n_freqs(const FftHandle* h) {
    return h ? h->n_freqs : 0;
}

void fft_forward(FftHandle* h, const float* restrict real_in, Complex* restrict complex_out) {
    if (!h || !real_in || !complex_out) return;

    // Copy the input first: NE10 R2C may modify its input and fft_forward leaves the caller's buffer unmodified.
    memcpy(h->real_buf, real_in, h->fft_size * sizeof(float));

    // Forward R2C FFT, written straight into the caller's output.
    NE10_R2C_1D_FLOAT32((ne10_fft_cpx_float32_t*)complex_out, h->real_buf, h->r2c_cfg);
}

void fft_inverse(FftHandle* h, const Complex* restrict complex_in, float* restrict real_out) {
    if (!h || !complex_in || !real_out) return;

    // Copy the input first: NE10 C2R may modify its input and fft_inverse leaves the caller's buffer unmodified.
    memcpy(h->cpx_buf, complex_in, h->n_freqs * sizeof(ne10_fft_cpx_float32_t));

    // Inverse C2R FFT, written straight into the caller's output; NE10 C2R scales by 1/N itself.
    NE10_C2R_1D_FLOAT32(real_out, h->cpx_buf, h->r2c_cfg);
}

/* Scratch variants: the input buffer's contents are undefined afterwards. The C2R kernel borrows fin[0].i for the Nyquist bin,
 * which is why the plain functions stage the input; these pass the caller's buffer straight to the kernel. */

void fft_forward_scratch(FftHandle* h, float* time_in_clobbered, Complex* complex_out) {
    if (!h || !time_in_clobbered || !complex_out) return;
    NE10_R2C_1D_FLOAT32((ne10_fft_cpx_float32_t*)complex_out, time_in_clobbered, h->r2c_cfg);
}

void fft_inverse_scratch(FftHandle* h, Complex* freq_in_clobbered, float* real_out) {
    if (!h || !freq_in_clobbered || !real_out) return;
    NE10_C2R_1D_FLOAT32(real_out, (ne10_fft_cpx_float32_t*)freq_in_clobbered, h->r2c_cfg);
}

void fft_power(const Complex* spectrum, float* power, int n_freqs) {
    if (!spectrum || !power) return;

    int k = 0;
#if SK_HAVE_NEON
    /* fmaf(re, re, im * im): im * im rounded on its own, then re * re fused into it; the NEON path matches the scalar tail lane for lane. */
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
