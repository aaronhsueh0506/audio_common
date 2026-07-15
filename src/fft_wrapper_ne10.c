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
#include <stddef.h>   /* offsetof — F11 layout proof below */

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

/* --- F11: compile-time proof that Complex and ne10_fft_cpx_float32_t are
 * layout-compatible ------------------------------------------------------
 *
 * fft_forward/fft_inverse (and the _scratch variants below) cast a
 * `Complex*` (fft_wrapper.h: struct {float r, i;}) straight to a
 * `ne10_fft_cpx_float32_t*` (NE10_types.h: struct {ne10_float32_t r, i;},
 * ne10_float32_t == float) and hand it to the NE10 kernel as its
 * output/input buffer, with no element-by-element copy. That is only safe
 * if the two struct types agree on size, alignment, and per-member offset.
 * These asserts make that a build-time guarantee instead of a comment: if a
 * future NE10 update (or a change to Complex) ever breaks the match, the
 * build fails here instead of silently corrupting spectra.
 *
 * -std=gnu99 (not c11) is in effect for this TU (see the top-level
 * Makefile) but both clang and gcc accept `_Static_assert` as a portable
 * extension in gnu99 mode (verified), so this needs no negative-array-size
 * fallback.
 *
 * Residual strict-aliasing caveat (review R06): Complex and
 * ne10_fft_cpx_float32_t are still DISTINCT struct types, so this is layout
 * compatibility, not type compatibility -- the C standard does not
 * guarantee that reading through one struct type after writing through the
 * other is defined behavior (a "distinct types" strict-aliasing violation)
 * even when the layouts match byte-for-byte. This is now formally exempted
 * rather than merely argued around: the top-level Makefile applies
 * -fno-strict-aliasing as a target-specific flag to ONLY this TU's object
 * (obj/.../fft_wrapper_ne10.o), so the compiler drops the type-based-alias
 * (TBAA) assumption for every cast in this file -- sanctioned containment,
 * not a blanket -fno-strict-aliasing for the whole archive. Layout equality
 * is still independently proven by the _Static_asserts below regardless of
 * the flag; the flag only removes the aliasing assumption, it doesn't
 * change what's being asserted.
 *
 * This exemption is scoped to plain -O2 compilation: LTO/whole-program
 * optimization must NOT be enabled for this TU (or, transitively, for a
 * link that would let the compiler inline across the NE10 call boundary
 * under a shared aliasing model) -- -fno-strict-aliasing suppresses TBAA
 * for the casts textually in this file, but an LTO build that inlines
 * fft_forward/fft_inverse's callers into a TU compiled WITHOUT the flag
 * could still reintroduce the assumption at the inlined call site. If LTO
 * is ever enabled for this archive, re-verify by re-running the
 * roundtrip/selftest targets (and ideally UBSan/ASan) under -flto before
 * trusting this cast again. */
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

/* --- F11: FFT_NE10_FORCE_C build knob -------------------------------------
 * SIMD_KERNELS_FORCE_SCALAR (simd_kernels.h) forces the header-only SIMD
 * kernels to their scalar reference path, but it has no effect on NE10's
 * FFT: this TU calls the `_neon`-suffixed kernels directly (see the header
 * comment above), so there was previously no way to get an NE10-backend
 * build (correct twiddle/config layout, correct archive, etc.) that still
 * runs the FFT itself through NE10's plain-C scalar kernels for an
 * apples-to-apples NEON-vs-C comparison. Define -DFFT_NE10_FORCE_C (e.g.
 * `make BACKEND=ne10 selftest EXTRA_CFLAGS=-DFFT_NE10_FORCE_C`) to route
 * every call site below through ne10_fft_r2c_1d_float32_c /
 * ne10_fft_c2r_1d_float32_c instead. Both variants are always compiled into
 * the archive (NE10_rfft_float32.c, which defines the _c versions, is
 * already in the NE10 source list -- see the Makefile's NE10_C_SRCS -- and
 * is NOT gated out when this macro is undefined), so this is a pure
 * dispatch switch with zero effect on the archive's file list or on the
 * BACKEND=kiss build (this macro only exists in this NE10-only TU). */
#if defined(FFT_NE10_FORCE_C)
#define NE10_R2C_1D_FLOAT32 ne10_fft_r2c_1d_float32_c
#define NE10_C2R_1D_FLOAT32 ne10_fft_c2r_1d_float32_c
#else
#define NE10_R2C_1D_FLOAT32 ne10_fft_r2c_1d_float32_neon
#define NE10_C2R_1D_FLOAT32 ne10_fft_c2r_1d_float32_neon
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

/* --- P0003 (re-review R05): fft_size whitelist ----------------------------
 * All three public entry points below (fft_create/fft_get_mem_size/fft_init)
 * used to accept ANY positive power of two, including tiny ones (1/2/4/8)
 * that fall into ne10_fft_init_r2c_float32_ext's pre-P0003 nfft<16
 * silent-degenerate path, and unboundedly large ones whose byte requirement
 * this backend's underlying NE10 twiddle-config sizer computes in 32-bit
 * (ne10_uint32_t) arithmetic that can wrap for huge nfft (see
 * lib/ne10/modules/dsp/NE10_rfft_float32.c's ne10_fft_r2c_mem_size_float32,
 * vendored patch P0003). This wrapper now whitelists nfft the same way that
 * NE10-side fix does -- [16, 8192], power of two -- so a caller here gets a
 * rejection (0 / NULL) at this layer instead of ever reaching the NE10 sizer
 * with an out-of-range nfft. 8192 is 8x the largest production fft_size
 * (1024 @ 48kHz); every real caller (AEC/NR/Audio_ALG, all four sibling
 * repos) derives fft_size from a validated sample-rate/frame-size config
 * that never leaves {256, 512, 1024}. */
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
        // fft_size must be a power of 2 in [16, 8192] (P0003)
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
    if (!fft_size_in_range(fft_size)) return 0;  /* P0003 */
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
    if (!mem || !fft_size_in_range(fft_size)) return NULL;  /* P0003 */
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
    //
    // P0003: ne10_fft_init_r2c_float32_ext now takes a mem_size and validates
    // it against its own re-derived requirement (see that function). `cursor`
    // is the LAST field in this walk, so the true remaining budget is
    // `mem_size - (cursor - mem)`; that is bounded above by
    // `cfg_reserved` (the exact ALIGN16'd amount fft_get_mem_size() reserved
    // for this field) so the value handed to the _ext call — and the
    // ne10_uint32_t it gets truncated into — never depends on how much
    // larger than strictly necessary the caller's pool happens to be (a
    // caller-supplied mem_size in the billions, cast straight down to
    // ne10_uint32_t, could otherwise wrap to a small, spuriously-rejecting
    // value).
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
    NE10_R2C_1D_FLOAT32((ne10_fft_cpx_float32_t*)complex_out, h->real_buf, h->r2c_cfg);
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
    NE10_C2R_1D_FLOAT32(real_out, h->cpx_buf, h->r2c_cfg);
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
    NE10_R2C_1D_FLOAT32((ne10_fft_cpx_float32_t*)complex_out, time_in_clobbered, h->r2c_cfg);
}

void fft_inverse_scratch(FftHandle* h, Complex* freq_in_clobbered, float* real_out) {
    if (!h || !freq_in_clobbered || !real_out) return;
    NE10_C2R_1D_FLOAT32(real_out, (ne10_fft_cpx_float32_t*)freq_in_clobbered, h->r2c_cfg);
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
