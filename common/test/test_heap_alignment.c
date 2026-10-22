/* Heap conveniences must work even when malloc guarantees only 8 bytes. */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL heap alignment at line %d\n", __LINE__); \
        exit(1); \
    } \
} while (0)

static void* offset_base;
static void* offset_ptr;
static int fail_aligned, aligned_live;

static void* test_malloc8(size_t bytes) {
    CHECK(offset_base == NULL);
    if (posix_memalign(&offset_base, 16, bytes + 16) != 0) return NULL;
    offset_ptr = (unsigned char*)offset_base + 8;
    return offset_ptr;
}
static int test_posix_memalign(void** out, size_t alignment, size_t bytes) {
    int rc;
    if (fail_aligned) return ENOMEM;
    rc = posix_memalign(out, alignment, bytes);
    if (rc == 0) ++aligned_live;
    return rc;
}
static void test_free(void* p) {
    if (p == offset_ptr && offset_ptr) {
        free(offset_base);
        offset_base = offset_ptr = NULL;
    } else if (p) {
        --aligned_live;
        CHECK(aligned_live >= 0);
        free(p);
    }
}

#define malloc test_malloc8
#define posix_memalign test_posix_memalign
#define free test_free
#include "../src/audio_resampler.c"
#undef malloc
#undef posix_memalign
#undef free

int main(void) {
    const int rates[] = {8000, 16000, 24000, 32000, 48000};
    int cells = 0;
    /* Keep the malloc fault injector live even though the fixed path uses
     * posix_memalign; reverting that path must fail the same test. */
    void* probe = test_malloc8(1);
    CHECK(probe && ((uintptr_t)probe & 15u) == 8u);
    test_free(probe);
    for (int a = 0; a < 5; ++a) for (int b = 0; b < 5; ++b)
        for (int channels = 1; channels <= 8; ++channels) {
            float input[48 * 8], heap_out[512 * 8], pool_out[512 * 8];
            int hc, hp, sc, sp;
            size_t bytes = audio_resampler_get_mem_size(rates[a], rates[b], channels);
            void* pool = NULL;
            AudioResampler* h = audio_resampler_create(rates[a], rates[b], channels);
            CHECK(h && MEM_IS_ALIGNED16(h));
            CHECK(posix_memalign(&pool, 16, bytes) == 0);
            AudioResampler* s = audio_resampler_init(pool, bytes, rates[a], rates[b], channels);
            CHECK(s);
            for (int i = 0; i < 48 * channels; ++i) input[i] = (float)(i % 31 - 15) * 0.001f;
            CHECK(audio_resampler_process(h, input, 48, heap_out, 512, &hc, &hp) == 0);
            CHECK(audio_resampler_process(s, input, 48, pool_out, 512, &sc, &sp) == 0);
            CHECK(hc == sc && hp == sp);
            CHECK(memcmp(heap_out, pool_out, (size_t)hp * (size_t)channels * sizeof(float)) == 0);
            audio_resampler_destroy(h);
            audio_resampler_destroy(s);
            free(pool);
            CHECK(aligned_live == 0 && offset_base == NULL);
            ++cells;
        }
    fail_aligned = 1;
    CHECK(audio_resampler_create(48000, 16000, 1) == NULL);
    CHECK(aligned_live == 0 && offset_base == NULL);
    printf("heap alignment: PASS %d rate/channel cells, forced malloc8, heap/static exact, allocation failure\n", cells);
    return 0;
}
