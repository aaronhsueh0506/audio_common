/*
 * zero_heap_hook.h - allocator call-counting hook for test_fft_zero_heap.c.
 *
 * See zero_heap_hook.c for the interposition mechanism (platform-specific)
 * and why it has to live in its own translation unit / shared object rather
 * than inline in the test file.
 */
#ifndef ZERO_HEAP_HOOK_H
#define ZERO_HEAP_HOOK_H

#ifdef __cplusplus
extern "C" {
#endif

void zh_reset_counters(void);
void zh_get_counters(long* malloc_count, long* calloc_count,
                      long* realloc_count, long* free_count);
int  zh_counters_all_zero(void);

#ifdef __cplusplus
}
#endif

#endif /* ZERO_HEAP_HOOK_H */
