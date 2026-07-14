/*
 * zero_heap_hook.c - allocator call-counting hook for test_fft_zero_heap.c
 * (review F02/F08 acceptance test support).
 *
 * Counts every process-wide call to malloc/calloc/realloc/free so the test
 * can prove a zero-CALL window around fft_init()..fft_destroy() (not merely
 * a zero-net-bytes one, which would miss a paired alloc+free that still
 * shouldn't have happened).
 *
 * Mechanism -- chosen at compile time, and BOTH verified empirically against
 * this repo's actual toolchain before being trusted (see
 * test_hook_actually_counts() in test_fft_zero_heap.c):
 *
 *   - macOS: dyld's `__DATA,__interpose` static-interposing-tuple section.
 *     IMPORTANT, discovered while building this test: an `__interpose`
 *     section only takes effect when it lives in a *separate Mach-O image*
 *     from the code whose calls you want to intercept -- dyld's interpose
 *     pass explicitly skips the main executable's own image. An
 *     `__interpose` section embedded directly in the test's main executable
 *     was empirically confirmed (on this host, Xcode 17 / Apple clang
 *     17.0.0, macOS 25.5) to NOT fire -- the hook function was never called.
 *     Building the exact same tuples into a small dependent `.dylib`
 *     (`libzero_heap_hook.dylib`, linked normally, no DYLD_INSERT_LIBRARIES
 *     needed) DOES fire, and intercepts calls made from every other
 *     statically-linked object in the process -- including calls from deep
 *     inside the vendored NE10 code, which is exactly what this test needs.
 *     So on macOS the Makefile builds *this file* as a small dylib, and
 *     test_fft_zero_heap links against it.
 *
 *     A second wrinkle: the linker relocates the `__interpose` section into
 *     the `__DATA_CONST` segment regardless of the `__DATA,__interpose`
 *     section spec (`-Wl,-no_data_const` does not change this). That
 *     turned out to be harmless -- dyld's interpose scan matches on section
 *     name, not the segment it landed in -- so no special link flag is
 *     needed for this part.
 *
 *   - Linux: GNU ld's `--wrap=<symbol>` link-time renaming. Unlike the macOS
 *     mechanism this works from a plain object file linked directly into the
 *     test binary (no separate shared object needed) -- the Makefile passes
 *     -Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free only for
 *     the test_zero_heap target.
 *
 * This file is intentionally NOT built into libaudio_common.a -- it exists
 * only to support test_fft_zero_heap.c.
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

/* Forward to malloc_zone_* on the default zone -- a DIFFERENT symbol than
 * malloc/calloc/realloc/free, so this call is not itself re-intercepted
 * (the classic interpose self-recursion trap). */
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
