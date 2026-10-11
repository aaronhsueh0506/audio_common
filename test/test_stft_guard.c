/*
 * test_stft_guard.c - memory-bounds test for audio_stft.h.
 *
 * Every AcStft and AcIstft is placed in a block of exactly
 * ac_stft_get_mem_size() / ac_istft_get_mem_size() bytes that ends flush
 * against a PROT_NONE page (and, in a second placement, starts flush behind
 * one), then driven through all of its entry points. A read or write outside
 * the reported size faults, and the fault is reported with the configuration
 * that caused it.
 *
 * Matrix: fft_size 16..8192 (powers of two) x center 0/1 x synthesis mode x
 * own/borrowed FFT handle x own/borrowed window x guard after/before, for
 * both objects, with non-unit scales so the staging buffers are used.
 *
 * Two probes run first: a write one byte past the block must fault on both
 * sides, which shows that the guard is effective on this system.
 *
 * POSIX only (mmap, mprotect, sigsetjmp); elsewhere the test prints SKIP and
 * succeeds.
 */
#include <stdio.h>

#if defined(__unix__) || defined(__APPLE__)

#include <math.h>
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "audio_stft.h"
#include "fft_wrapper.h"
#include "mem_align.h"

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

static sigjmp_buf g_jump;
static volatile sig_atomic_t g_armed = 0;

static void on_fault(int sig) {
    (void)sig;
    if (g_armed) siglongjmp(g_jump, 1);
    _exit(70);
}

/* A mapping of whole pages with one PROT_NONE guard page. */
typedef struct {
    unsigned char* base;
    size_t map_bytes;
    unsigned char* block;   /* first byte of the exact-size block */
} Guarded;

enum { GUARD_AFTER = 0, GUARD_BEFORE = 1 };

/* `bytes` must be a multiple of 16 so that the block start is 16-aligned
 * whichever end touches the guard page. */
static int guarded_alloc(Guarded* g, size_t bytes, int where) {
    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    size_t body = ((bytes + page - 1) / page) * page;
    unsigned char* m;
    g->map_bytes = body + page;
    m = (unsigned char*)mmap(NULL, g->map_bytes, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == (unsigned char*)MAP_FAILED) return 0;
    g->base = m;
    if (where == GUARD_AFTER) {
        if (mprotect(m + body, page, PROT_NONE) != 0) return 0;
        g->block = m + body - bytes;
    } else {
        if (mprotect(m, page, PROT_NONE) != 0) return 0;
        g->block = m + page;
    }
    return 1;
}

static void guarded_free(Guarded* g) {
    if (g->base) munmap(g->base, g->map_bytes);
    g->base = NULL;
}

/* Writes the byte just outside the block; returns 1 if it faulted. */
static int probe_faults(int where) {
    Guarded g;
    volatile unsigned char* p;
    int faulted = 0;
    if (!guarded_alloc(&g, 4096, where)) return 0;
    p = (where == GUARD_AFTER) ? g.block + 4096 : g.block - 1;
    g_armed = 1;
    if (sigsetjmp(g_jump, 1) == 0) *p = 1; else faulted = 1;
    g_armed = 0;
    guarded_free(&g);
    return faulted;
}

static uint32_t g_rng = 0x9E3779B9u;
static float rnd(void) {
    g_rng = g_rng * 1664525u + 1013904223u;
    return (float)(int32_t)g_rng * (1.0f / 2147483648.0f);
}

typedef struct {
    int n, center, ola, own_fft, own_window, where;
} Case;

static void describe(const Case* c, const char* what, char* out, size_t len) {
    snprintf(out, len, "%s N=%d center=%d ola=%s fft=%s window=%s guard=%s", what, c->n, c->center,
             c->ola ? "env_norm" : "accumulate", c->own_fft ? "own" : "borrowed",
             c->own_window ? "own" : "borrowed", c->where == GUARD_AFTER ? "after" : "before");
}

/* Drives every entry point of an AcStft. */
static void drive_stft(AcStft* s, int n) {
    const int hop = n / 2, nf = n / 2 + 1;
    float* in = (float*)calloc((size_t)n, sizeof(float));
    float* re = (float*)calloc((size_t)2 * nf, sizeof(float));
    float* im = (float*)calloc((size_t)2 * nf, sizeof(float));
    Complex* sp = (Complex*)calloc((size_t)2 * nf, sizeof(Complex));
    int p, k;
    for (p = 0; p < 6; ++p) {
        for (k = 0; k < n; ++k) in[k] = rnd();
        (void)ac_stft_push(s, in, sp);
        (void)ac_stft_push_planar(s, in + hop, re, im);
        ac_stft_frame(s, in, sp);
        ac_stft_frame_planar(s, in, re, im);
    }
    (void)ac_stft_flush(s, sp);
    (void)ac_stft_flush_planar(s, re, im);
    (void)ac_stft_window(s);
    ac_stft_reset(s);
    free(in); free(re); free(im); free(sp);
}

/* Drives every entry point of an AcIstft. */
static void drive_istft(AcIstft* s, int n) {
    const int hop = n / 2, nf = n / 2 + 1;
    float* out = (float*)calloc((size_t)hop, sizeof(float));
    float* re = (float*)calloc((size_t)nf, sizeof(float));
    float* im = (float*)calloc((size_t)nf, sizeof(float));
    Complex* sp = (Complex*)calloc((size_t)nf, sizeof(Complex));
    int p, k;
    for (p = 0; p < 6; ++p) {
        for (k = 0; k < nf; ++k) { sp[k].r = re[k] = rnd(); sp[k].i = im[k] = rnd(); }
        (void)ac_istft_push(s, sp, out);
        (void)ac_istft_push_planar(s, re, im, out);
    }
    (void)ac_istft_flush(s, out);
    (void)ac_istft_window(s);
    ac_istft_reset(s);
    free(out); free(re); free(im); free(sp);
}

/* Returns 0 on success, 1 on a fault or an unexpected NULL. */
static int run_case(const Case* c) {
    AcStftConfig cfg;
    FftHandle* shared = NULL;
    float* table = NULL;
    void* shared_mem = NULL;
    size_t sa, ss;
    Guarded ga, gs;
    AcStft* a;
    AcIstft* s;
    char label[160];
    volatile int rc = 0;

    ac_stft_config_default(&cfg, c->n);
    cfg.center = c->center;
    cfg.ola = c->ola ? AC_OLA_ENV_NORM : AC_OLA_ACCUMULATE;
    cfg.spec_scale = 0.5f;
    cfg.ispec_scale = 2.0f;
    if (!c->own_fft) {
        size_t fb = fft_get_mem_size(c->n);
        if (posix_memalign(&shared_mem, 16, ALIGN16(fb)) != 0) shared_mem = NULL;
        shared = shared_mem ? fft_init(shared_mem, fb, c->n) : NULL;
        if (!shared) { fprintf(stderr, "FAIL: shared fft setup N=%d\n", c->n); free(shared_mem); return 1; }
        cfg.fft = shared;
    }
    if (!c->own_window) {
        table = (float*)malloc((size_t)c->n * sizeof(float));
        ac_window_build(table, c->n);
        cfg.window = table;
    }
    sa = ac_stft_get_mem_size(&cfg);
    ss = ac_istft_get_mem_size(&cfg);
    if (sa == 0 || ss == 0 || (sa % 16) != 0 || (ss % 16) != 0) {
        describe(c, "FAIL: unusable size", label, sizeof(label));
        fprintf(stderr, "%s (stft %zu, istft %zu)\n", label, sa, ss);
        rc = 1;
        goto done;
    }

    memset(&ga, 0, sizeof(ga));
    memset(&gs, 0, sizeof(gs));
    if (!guarded_alloc(&ga, sa, c->where) || !guarded_alloc(&gs, ss, c->where)) {
        fprintf(stderr, "FAIL: cannot map guarded memory\n");
        rc = 1;
        goto done;
    }

    g_armed = 1;
    if (sigsetjmp(g_jump, 1) == 0) {
        a = ac_stft_init(ga.block, sa, &cfg);
        if (a) drive_stft(a, c->n);
        else rc = 1;
    } else {
        describe(c, "FAULT in AcStft", label, sizeof(label));
        fprintf(stderr, "%s (block %zu bytes)\n", label, sa);
        rc = 1;
    }
    g_armed = 0;
    if (rc == 0) {
        g_armed = 1;
        if (sigsetjmp(g_jump, 1) == 0) {
            s = ac_istft_init(gs.block, ss, &cfg);
            if (s) drive_istft(s, c->n);
            else rc = 1;
        } else {
            describe(c, "FAULT in AcIstft", label, sizeof(label));
            fprintf(stderr, "%s (block %zu bytes)\n", label, ss);
            rc = 1;
        }
        g_armed = 0;
    }
    if (rc) {
        describe(c, "FAIL", label, sizeof(label));
        fprintf(stderr, "%s\n", label);
    }
done:
    guarded_free(&ga);
    guarded_free(&gs);
    free(table);
    free(shared_mem);
    return rc;
}

int main(void) {
    struct sigaction sa;
    int n, center, ola, own_fft, own_window, where, cases = 0, failed = 0;

    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_fault;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);

    if (!probe_faults(GUARD_AFTER) || !probe_faults(GUARD_BEFORE)) {
        fprintf(stderr, "FAIL: a write next to the guard page did not fault; the test cannot detect overruns here\n");
        return 1;
    }
    for (n = 16; n <= 8192; n *= 2)
        for (center = 0; center <= 1; ++center)
            for (ola = 0; ola <= 1; ++ola)
                for (own_fft = 0; own_fft <= 1; ++own_fft)
                    for (own_window = 0; own_window <= 1; ++own_window)
                        for (where = GUARD_AFTER; where <= GUARD_BEFORE; ++where) {
                            Case c;
                            c.n = n; c.center = center; c.ola = ola;
                            c.own_fft = own_fft; c.own_window = own_window; c.where = where;
                            failed += run_case(&c);
                            cases++;
                        }
    printf("GUARD| %d cases (each: AcStft and AcIstft in an exact-size block against a guard page), %d failed\n",
           cases, failed);
    if (failed) {
        fprintf(stderr, "test_stft_guard FAILED\n");
        return 1;
    }
    printf("test_stft_guard PASS\n");
    return 0;
}

#else

int main(void) {
    printf("test_stft_guard SKIP (needs POSIX mmap/mprotect)\n");
    return 0;
}

#endif
