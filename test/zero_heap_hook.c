/*
 * zero_heap_hook.c - allocator call-counting hook for test_fft_zero_heap.c.
 *
 * Counts every process-wide call to malloc / calloc / realloc / free, so the
 * test can show there is no allocator call at all around fft_init() ..
 * fft_destroy() (a paired alloc and free would also count).
 *
 * Mechanism, chosen at compile time (test_hook_actually_counts() in
 * test_fft_zero_heap.c checks that the hook fires):
 *
 *   - macOS: dyld's `__DATA,__interpose` section. dyld skips the main
 *     executable's own image, so the tuples must live in a separate image:
 *     the Makefile builds this file as a small dylib that the test links
 *     against. It then intercepts the calls of every other object in the
 *     process, including the vendored NE10 code. The linker moves the
 *     section into `__DATA_CONST`; dyld matches it by section name, so that
 *     is harmless.
 *
 *   - Linux: GNU ld `--wrap=<symbol>`. It works from a plain object linked
 *     into the test binary; the Makefile passes
 *     -Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free for the
 *     test_zero_heap target only.
 *
 * This file is not part of libaudio_common.a.
 */
#include "zero_heap_hook.h"

#include <stdint.h>
#include <stdlib.h>

static volatile long g_malloc_count  = 0;
static volatile long g_calloc_count  = 0;
static volatile long g_realloc_count = 0;
static volatile long g_free_count    = 0;

void zh_reset_counters(void) {
    g_malloc_count  = 0;
    g_calloc_count  = 0;
    g_realloc_count = 0;
    g_free_count    = 0;
}

void zh_get_counters(long* malloc_count, long* calloc_count,
                      long* realloc_count, long* free_count) {
    if (malloc_count)  *malloc_count  = g_malloc_count;
    if (calloc_count)  *calloc_count  = g_calloc_count;
    if (realloc_count) *realloc_count = g_realloc_count;
    if (free_count)    *free_count    = g_free_count;
}

int zh_counters_all_zero(void) {
    return g_malloc_count == 0 && g_calloc_count == 0 &&
           g_realloc_count == 0 && g_free_count == 0;
}

#if defined(__APPLE__)
#include <malloc/malloc.h>

/* Forward to malloc_zone_* on the default zone: a different symbol from
 * malloc / free, so the hook does not intercept itself. */
static void* hook_malloc(size_t size) {
    g_malloc_count++;
    return malloc_zone_malloc(malloc_default_zone(), size);
}
static void* hook_calloc(size_t count, size_t size) {
    g_calloc_count++;
    return malloc_zone_calloc(malloc_default_zone(), count, size);
}
static void* hook_realloc(void* ptr, size_t size) {
    g_realloc_count++;
    return malloc_zone_realloc(malloc_default_zone(), ptr, size);
}
static void hook_free(void* ptr) {
    g_free_count++;
    malloc_zone_free(malloc_default_zone(), ptr);
}

typedef struct {
    const void* replacement;
    const void* replacee;
} interpose_t;

#define ZH_INTERPOSE(sym) \
    __attribute__((used)) static const interpose_t interpose_##sym \
    __attribute__((section("__DATA,__interpose"))) = { \
        (const void*)(uintptr_t)&hook_##sym, (const void*)(uintptr_t)&sym }

ZH_INTERPOSE(malloc);
ZH_INTERPOSE(calloc);
ZH_INTERPOSE(realloc);
ZH_INTERPOSE(free);

#elif defined(__linux__)

extern void* __real_malloc(size_t size);
extern void* __real_calloc(size_t count, size_t size);
extern void* __real_realloc(void* ptr, size_t size);
extern void  __real_free(void* ptr);

void* __wrap_malloc(size_t size) {
    g_malloc_count++;
    return __real_malloc(size);
}
void* __wrap_calloc(size_t count, size_t size) {
    g_calloc_count++;
    return __real_calloc(count, size);
}
void* __wrap_realloc(void* ptr, size_t size) {
    g_realloc_count++;
    return __real_realloc(ptr, size);
}
void __wrap_free(void* ptr) {
    g_free_count++;
    __real_free(ptr);
}

#else
#error "zero_heap_hook.c: no allocator-hook mechanism wired up for this platform"
#endif
