/*
 * test_pool_contract.c - negative-input contract tests for the shared
 * static-memory pool APIs (fft_wrapper.h, hpf.h).
 *
 * Covers the review-remediation fixes:
 *   F07 - public pool inits (fft_init, hpf_init) must reject a misaligned
 *         base pointer (MEM_IS_ALIGNED16) BEFORE writing anything into the
 *         pool, and must reject mem_size smaller than get_mem_size()'s
 *         report (including the exact-size boundary).
 *   F14 - hpf_create/hpf_init must reject a cutoff_hz/sample_rate pair that
 *         is non-finite, non-positive, or at/above the 0.45*sample_rate
 *         margin below Nyquist.
 *
 * Backend-agnostic like roundtrip.c/simd_selftest.c: whichever FFT backend
 * (kiss or ne10) the Makefile links in, fft_wrapper.h's public API is the
 * same, so this test only exercises that public surface.
 *
 * Plain assert-and-report harness, no external framework: every failure is
 * printed and flips a global fail flag; main() reports ALL PASS / FAIL at
 * the end so one bad check doesn't hide the rest.
 */
#include "fft_wrapper.h"
#include "hpf.h"
#include "mem_align.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_fail = 1; \
    } \
} while (0)

/* Returns a 16-byte-aligned pointer at least `pad` bytes inside `*out_base`
 * (which the caller must free()), with at least `usable` bytes available
 * from the aligned pointer onward. Centralizes the "carve an aligned
 * address out of a malloc'd block, with slop for +1..+15 offset probing"
 * logic shared by every misaligned-base test below. */
static unsigned char* aligned_ptr_in(unsigned char** out_base, size_t usable, size_t pad) {
    size_t alloc_sz = usable + pad + 32;
    unsigned char* base = (unsigned char*)malloc(alloc_sz);
    if (!base) { *out_base = NULL; return NULL; }
    uintptr_t addr = (uintptr_t)base;
    uintptr_t aligned_addr = (addr + 15u) & ~(uintptr_t)15u;
    *out_base = base;
    return (unsigned char*)aligned_addr;
}

/* ═══════════════════════════ fft_wrapper pool contract ═══════════════════ */

static void test_fft_misaligned_base(void) {
    const int fft_size = 256;
    size_t sz = fft_get_mem_size(fft_size);
    CHECK(sz > 0, "fft_get_mem_size(256) must be > 0");
    if (sz == 0) return;

    unsigned char* base = NULL;
    unsigned char* aligned = aligned_ptr_in(&base, sz, 16);
    CHECK(aligned != NULL, "malloc for fft misaligned-base test");
    if (!aligned) return;

    for (int i = 1; i <= 15; i++) {
        unsigned char* mis = aligned + i;
        memset(mis, 0xA5, sz);
        FftHandle* h = fft_init(mis, sz, fft_size);
        CHECK(h == NULL, "fft_init must reject a base offset by 1..15 bytes");

        int untouched = 1;
        for (size_t j = 0; j < sz; j++) {
            if (mis[j] != 0xA5) { untouched = 0; break; }
        }
        CHECK(untouched, "fft_init must not write the pool when rejecting a misaligned base");
    }

    free(base);
}

static void test_fft_size_boundary(void) {
    const int fft_size = 256;
    size_t sz = fft_get_mem_size(fft_size);
    CHECK(sz > 0, "fft_get_mem_size(256) must be > 0");
    if (sz == 0) return;

    void* pool = NULL;
    CHECK(posix_memalign(&pool, 16, sz) == 0 && pool != NULL, "posix_memalign for fft size-boundary test");
    if (!pool) return;

    FftHandle* h = fft_init(pool, sz, fft_size);
    CHECK(h != NULL, "fft_init must accept the exact required mem_size");

    for (size_t shrink = 1; shrink <= 15 && shrink < sz; shrink++) {
        FftHandle* h2 = fft_init(pool, sz - shrink, fft_size);
        CHECK(h2 == NULL, "fft_init must reject mem_size below the required size");
    }

    FftHandle* h3 = fft_init(pool, 0, fft_size);
    CHECK(h3 == NULL, "fft_init must reject mem_size == 0");

    free(pool);
}

/* ═══════════════════════════════ hpf pool contract ═══════════════════════ */

static void test_hpf_misaligned_base(void) {
    const float cutoff = 100.0f;
    const int sr = 16000;
    size_t sz = hpf_get_mem_size();
    CHECK(sz > 0, "hpf_get_mem_size() must be > 0");
    if (sz == 0) return;

    unsigned char* base = NULL;
    unsigned char* aligned = aligned_ptr_in(&base, sz, 16);
    CHECK(aligned != NULL, "malloc for hpf misaligned-base test");
    if (!aligned) return;

    for (int i = 1; i <= 15; i++) {
        unsigned char* mis = aligned + i;
        memset(mis, 0xA5, sz);
        Hpf* h = hpf_init(mis, sz, cutoff, sr);
        CHECK(h == NULL, "hpf_init must reject a base offset by 1..15 bytes");

        int untouched = 1;
        for (size_t j = 0; j < sz; j++) {
            if (mis[j] != 0xA5) { untouched = 0; break; }
        }
        CHECK(untouched, "hpf_init must not write the pool when rejecting a misaligned base");
    }

    free(base);
}

static void test_hpf_size_boundary(void) {
    const float cutoff = 100.0f;
    const int sr = 16000;
    size_t sz = hpf_get_mem_size();
    CHECK(sz > 0, "hpf_get_mem_size() must be > 0");
    if (sz == 0) return;

    void* pool = NULL;
    CHECK(posix_memalign(&pool, 16, sz) == 0 && pool != NULL, "posix_memalign for hpf size-boundary test");
    if (!pool) return;

    Hpf* h = hpf_init(pool, sz, cutoff, sr);
    CHECK(h != NULL, "hpf_init must accept the exact required mem_size");

    for (size_t shrink = 1; shrink <= 15 && shrink < sz; shrink++) {
        Hpf* h2 = hpf_init(pool, sz - shrink, cutoff, sr);
        CHECK(h2 == NULL, "hpf_init must reject mem_size below the required size");
    }

    Hpf* h3 = hpf_init(pool, 0, cutoff, sr);
    CHECK(h3 == NULL, "hpf_init must reject mem_size == 0");

    free(pool);
}

static void test_hpf_cutoff_validation(void) {
    const int sr = 16000;
    size_t sz = hpf_get_mem_size();
    CHECK(sz > 0, "hpf_get_mem_size() must be > 0");
    if (sz == 0) return;

    void* pool = NULL;
    CHECK(posix_memalign(&pool, 16, sz) == 0 && pool != NULL, "posix_memalign for hpf cutoff-validation test");
    if (!pool) return;

    /* at/above 0.45*sample_rate rejected (the Nyquist-margin gate) */
    float at_limit = 0.45f * (float)sr;
    CHECK(hpf_init(pool, sz, at_limit, sr) == NULL, "hpf_init must reject cutoff == 0.45*sample_rate");
    CHECK(hpf_create(at_limit, sr) == NULL, "hpf_create must reject cutoff == 0.45*sample_rate");

    float above_limit = 0.49f * (float)sr;
    CHECK(hpf_init(pool, sz, above_limit, sr) == NULL, "hpf_init must reject cutoff above 0.45*sample_rate");

    float nyquist = 0.5f * (float)sr;
    CHECK(hpf_init(pool, sz, nyquist, sr) == NULL, "hpf_init must reject cutoff at Nyquist");
    CHECK(hpf_init(pool, sz, (float)sr, sr) == NULL, "hpf_init must reject cutoff == sample_rate");

    /* NaN / Inf cutoff rejected */
    CHECK(hpf_init(pool, sz, (float)NAN, sr) == NULL, "hpf_init must reject NaN cutoff");
    CHECK(hpf_init(pool, sz, (float)INFINITY, sr) == NULL, "hpf_init must reject +Inf cutoff");
    CHECK(hpf_init(pool, sz, -(float)INFINITY, sr) == NULL, "hpf_init must reject -Inf cutoff");
    CHECK(hpf_create((float)NAN, sr) == NULL, "hpf_create must reject NaN cutoff");
    CHECK(hpf_create((float)INFINITY, sr) == NULL, "hpf_create must reject +Inf cutoff");

    /* non-finite / non-positive sample_rate rejected */
    CHECK(hpf_init(pool, sz, 100.0f, 0) == NULL, "hpf_init must reject sample_rate == 0");
    CHECK(hpf_init(pool, sz, 100.0f, -16000) == NULL, "hpf_init must reject negative sample_rate");
    CHECK(hpf_create(100.0f, 0) == NULL, "hpf_create must reject sample_rate == 0");

    /* zero / negative cutoff rejected */
    CHECK(hpf_init(pool, sz, 0.0f, sr) == NULL, "hpf_init must reject cutoff == 0");
    CHECK(hpf_init(pool, sz, -100.0f, sr) == NULL, "hpf_init must reject negative cutoff");

    /* valid cutoff accepted (currently-valid input must still work) */
    Hpf* hok = hpf_init(pool, sz, 100.0f, sr);
    CHECK(hok != NULL, "hpf_init must accept a valid, well-below-margin cutoff");
    Hpf* hok2 = hpf_create(100.0f, sr);
    CHECK(hok2 != NULL, "hpf_create must accept a valid, well-below-margin cutoff");
    if (hok2) hpf_destroy(hok2);

    /* just-below-margin cutoff accepted */
    float just_below = 0.45f * (float)sr - 1.0f;
    Hpf* hok3 = hpf_init(pool, sz, just_below, sr);
    CHECK(hok3 != NULL, "hpf_init must accept cutoff just below the 0.45*sample_rate margin");

    free(pool);
}

/* ═══════════════════════════════════ main ════════════════════════════════ */

int main(void) {
    test_fft_misaligned_base();
    test_fft_size_boundary();
    test_hpf_misaligned_base();
    test_hpf_size_boundary();
    test_hpf_cutoff_validation();

    if (g_fail) {
        printf(">>> FAIL\n");
        return 1;
    }
    printf("ALL PASS\n>>> PASS\n");
    return 0;
}
