/*
 * test_fft_zero_heap.c - allocator-hook acceptance test for the static-memory
 * FFT path (review F02/F08).
 *
 * F02: the NE10 backend's static-memory path (fft_get_mem_size/fft_init) must
 * make ZERO malloc/calloc/realloc/free calls between fft_init() and
 * fft_destroy() -- not even NE10's own internal twiddle-config allocation
 * (see lib/ne10/VENDORED.md patch P0001, NE10_rfft_float32.c). KISS
 * (fft_wrapper.c) was already fully pool-carved, so this test is expected to
 * trivially pass there too -- it's backend-agnostic, like roundtrip.c /
 * test_pool_contract.c.
 *
 * F08: fft_destroy() on an fft_init() (pool-owned) handle must be a true,
 * unconditionally idempotent no-op -- calling it any number of times must
 * never double-release anything.
 *
 * The allocator-hook mechanism itself (dyld __DATA,__interpose on macOS /
 * GNU ld --wrap on Linux) lives in zero_heap_hook.c, NOT in this file --
 * on macOS it has to be built into a separate .dylib to take effect at all
 * (see that file's header comment for why, and what was empirically ruled
 * out getting here). This file only consumes the zh_* counter API and is
 * backend-agnostic, like roundtrip.c / test_pool_contract.c: whichever FFT
 * backend (kiss or ne10) the Makefile links in, fft_wrapper.h's public API
 * is the same, so this test exercises whichever one was built. KISS
 * (fft_wrapper.c) was already fully pool-carved, so this test is expected to
 * trivially pass there too.
 *
 * test_hook_actually_counts() below proves the hook mechanism actually fires
 * (a deliberate malloc()/calloc()/realloc()/free() call must be observed)
 * before any of the zero-heap assertions get to lean on it.
 */
#include "fft_wrapper.h"
#include "mem_align.h"
#include "zero_heap_hook.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_fail = 1; \
    } \
} while (0)

/* Proves the hook mechanism actually intercepts calls (not a silent no-op)
 * before any of the zero-heap assertions below get to lean on it. */
static void test_hook_actually_counts(void) {
    long mc, cc, rc, fc;

    zh_reset_counters();
    void* p = malloc(64);
    CHECK(p != NULL, "sanity malloc(64) must succeed");
    zh_get_counters(&mc, &cc, &rc, &fc);
    CHECK(mc == 1, "allocator hook must observe the deliberate malloc() call");
    free(p);
    zh_get_counters(&mc, &cc, &rc, &fc);
    CHECK(fc == 1, "allocator hook must observe the deliberate free() call");

    zh_reset_counters();
    void* q = calloc(4, 16);
    CHECK(q != NULL, "sanity calloc(4,16) must succeed");
    zh_get_counters(&mc, &cc, &rc, &fc);
    CHECK(cc == 1, "allocator hook must observe the deliberate calloc() call");
    q = realloc(q, 128);
    CHECK(q != NULL, "sanity realloc(.., 128) must succeed");
    zh_get_counters(&mc, &cc, &rc, &fc);
    CHECK(rc == 1, "allocator hook must observe the deliberate realloc() call");
    free(q);

    zh_reset_counters();
}

/* ═══════════════════════════ zero-heap sequence ═══════════════════════════ */

#define ZH_FFT_SIZE 512
#define ZH_N_FREQS  (ZH_FFT_SIZE / 2 + 1)
#define ZH_FRAMES   100
#define ZH_CYCLES   1000

static float   g_time_in[ZH_FFT_SIZE];
static Complex g_spec_buf[ZH_N_FREQS];
static float   g_time_out[ZH_FFT_SIZE];

static void fill_synthetic(float* buf, int n, int phase_offset) {
    for (int i = 0; i < n; i++) {
        buf[i] = sinf(2.0f * 3.14159265f * 7.0f * (float)(i + phase_offset) / (float)n) + 0.25f;
    }
}

/* One init -> ZH_FRAMES frames (forward+inverse+scratch variants) -> destroy
 * pass over an already-allocated pool. Returns 1 on success, 0 otherwise.
 * Does not touch the allocator counters itself -- callers arm/read them
 * around this call. */
static int run_zero_heap_pass(void* pool, size_t pool_sz) {
    FftHandle* h = fft_init(pool, pool_sz, ZH_FFT_SIZE);
    if (!h) return 0;

    for (int f = 0; f < ZH_FRAMES; f++) {
        fill_synthetic(g_time_in, ZH_FFT_SIZE, f);
        fft_forward(h, g_time_in, g_spec_buf);
        fft_inverse(h, g_spec_buf, g_time_out);

        fill_synthetic(g_time_in, ZH_FFT_SIZE, f + 1);
        fft_forward_scratch(h, g_time_in, g_spec_buf);
        fft_inverse_scratch(h, g_spec_buf, g_time_out);
    }

    fft_destroy(h);
    return 1;
}

static void test_zero_heap_init_to_destroy(void) {
    size_t sz = fft_get_mem_size(ZH_FFT_SIZE);
    CHECK(sz > 0, "fft_get_mem_size(512) must be > 0");
    if (sz == 0) return;

    void* pool = NULL;
    CHECK(posix_memalign(&pool, 16, sz) == 0 && pool != NULL, "posix_memalign for zero-heap pool");
    if (!pool) return;

    memset(pool, 0xA5, sz);  /* dirty-pool: prove fft_init doesn't rely on a fresh/zeroed block */

    zh_reset_counters();
    int ok = run_zero_heap_pass(pool, sz);
    long mc, cc, rc, fc;
    zh_get_counters(&mc, &cc, &rc, &fc);

    CHECK(ok, "fft_init must succeed on a dirty pool of the reported size");
    CHECK(mc == 0, "fft_init..fft_destroy (100 frames) must call malloc() zero times");
    CHECK(cc == 0, "fft_init..fft_destroy (100 frames) must call calloc() zero times");
    CHECK(rc == 0, "fft_init..fft_destroy (100 frames) must call realloc() zero times");
    CHECK(fc == 0, "fft_init..fft_destroy (100 frames) must call free() zero times");

    /* Repeat the whole init/destroy cycle many times on the SAME pool. */
    zh_reset_counters();
    int all_ok = 1;
    for (int i = 0; i < ZH_CYCLES; i++) {
        if (!run_zero_heap_pass(pool, sz)) { all_ok = 0; break; }
    }
    zh_get_counters(&mc, &cc, &rc, &fc);
    CHECK(all_ok, "1000x fft_init/fft_destroy cycles on the same pool must all succeed");
    CHECK(mc == 0 && cc == 0 && rc == 0 && fc == 0,
          "1000x fft_init/fft_destroy cycles on the same pool must still call the allocator zero times");

    free(pool);
}

/* F08: fft_destroy() on a pool-owned handle must be idempotent -- calling it
 * repeatedly must never touch the allocator (it has nothing left to
 * release). */
static void test_destroy_idempotent(void) {
    size_t sz = fft_get_mem_size(ZH_FFT_SIZE);
    if (sz == 0) { CHECK(0, "fft_get_mem_size(512) must be > 0"); return; }

    void* pool = NULL;
    CHECK(posix_memalign(&pool, 16, sz) == 0 && pool != NULL, "posix_memalign for idempotent-destroy pool");
    if (!pool) return;

    FftHandle* h = fft_init(pool, sz, ZH_FFT_SIZE);
    CHECK(h != NULL, "fft_init must succeed for idempotent-destroy test");

    zh_reset_counters();
    for (int i = 0; i < 1000; i++) {
        fft_destroy(h);   /* repeat on the SAME handle -- must never double-release */
    }
    CHECK(zh_counters_all_zero(), "1000x fft_destroy() on one pool-owned handle must call the allocator zero times");

    free(pool);
}

/* ══════════════════ dirty-pool vs zero-pool byte-identity ══════════════════ */

static void test_dirty_pool_matches_zero_pool(void) {
    size_t sz = fft_get_mem_size(ZH_FFT_SIZE);
    if (sz == 0) { CHECK(0, "fft_get_mem_size(512) must be > 0"); return; }

    void *pool_dirty = NULL, *pool_zero = NULL;
    CHECK(posix_memalign(&pool_dirty, 16, sz) == 0 && pool_dirty != NULL, "posix_memalign dirty pool");
    CHECK(posix_memalign(&pool_zero,  16, sz) == 0 && pool_zero  != NULL, "posix_memalign zero pool");
    if (!pool_dirty || !pool_zero) { free(pool_dirty); free(pool_zero); return; }

    memset(pool_dirty, 0xA5, sz);
    memset(pool_zero,  0x00, sz);

    fill_synthetic(g_time_in, ZH_FFT_SIZE, 0);
    float saved_in[ZH_FFT_SIZE];
    memcpy(saved_in, g_time_in, sizeof(saved_in));

    FftHandle* hd = fft_init(pool_dirty, sz, ZH_FFT_SIZE);
    CHECK(hd != NULL, "fft_init on dirty pool must succeed");
    Complex spec_dirty[ZH_N_FREQS];
    float out_dirty[ZH_FFT_SIZE];
    if (hd) {
        fft_forward(hd, saved_in, spec_dirty);
        fft_inverse(hd, spec_dirty, out_dirty);
    }

    FftHandle* hz = fft_init(pool_zero, sz, ZH_FFT_SIZE);
    CHECK(hz != NULL, "fft_init on zero-filled pool must succeed");
    Complex spec_zero[ZH_N_FREQS];
    float out_zero[ZH_FFT_SIZE];
    if (hz) {
        fft_forward(hz, saved_in, spec_zero);
        fft_inverse(hz, spec_zero, out_zero);
    }

    if (hd && hz) {
        CHECK(memcmp(spec_dirty, spec_zero, sizeof(spec_dirty)) == 0,
              "first-frame spectrum must be byte-identical: dirty-prefilled pool vs zero-filled pool");
        CHECK(memcmp(out_dirty, out_zero, sizeof(out_dirty)) == 0,
              "first-frame time-domain output must be byte-identical: dirty-prefilled pool vs zero-filled pool");
    }

    fft_destroy(hd);
    fft_destroy(hz);
    free(pool_dirty);
    free(pool_zero);
}

/* ═══════════════════════ heap vs pool bit-identity ═══════════════════════ */

static void test_heap_vs_pool_bit_identical(void) {
    fill_synthetic(g_time_in, ZH_FFT_SIZE, 3);
    float saved_in[ZH_FFT_SIZE];
    memcpy(saved_in, g_time_in, sizeof(saved_in));

    FftHandle* h_heap = fft_create(ZH_FFT_SIZE);
    CHECK(h_heap != NULL, "fft_create(512) must succeed");

    size_t sz = fft_get_mem_size(ZH_FFT_SIZE);
    void* pool = NULL;
    CHECK(posix_memalign(&pool, 16, sz) == 0 && pool != NULL, "posix_memalign for heap-vs-pool comparison");
    FftHandle* h_pool = pool ? fft_init(pool, sz, ZH_FFT_SIZE) : NULL;
    CHECK(h_pool != NULL, "fft_init(512) must succeed for heap-vs-pool comparison");

    if (h_heap && h_pool) {
        Complex spec_heap[ZH_N_FREQS], spec_pool[ZH_N_FREQS];
        float out_heap[ZH_FFT_SIZE], out_pool[ZH_FFT_SIZE];

        fft_forward(h_heap, saved_in, spec_heap);
        fft_forward(h_pool, saved_in, spec_pool);
        CHECK(memcmp(spec_heap, spec_pool, sizeof(spec_heap)) == 0,
              "forward spectrum must be byte-identical: fft_create (heap) vs fft_init (pool)");

        fft_inverse(h_heap, spec_heap, out_heap);
        fft_inverse(h_pool, spec_pool, out_pool);
        CHECK(memcmp(out_heap, out_pool, sizeof(out_heap)) == 0,
              "inverse time-domain output must be byte-identical: fft_create (heap) vs fft_init (pool)");
    }

    fft_destroy(h_heap);
    fft_destroy(h_pool);
    free(pool);
}

/* ═══════════════════════════════════ main ════════════════════════════════ */

int main(void) {
    test_hook_actually_counts();
    test_zero_heap_init_to_destroy();
    test_destroy_idempotent();
    test_dirty_pool_matches_zero_pool();
    test_heap_vs_pool_bit_identical();

    if (g_fail) {
        printf(">>> FAIL\n");
        return 1;
    }
    printf("ALL PASS\n>>> PASS\n");
    return 0;
}
