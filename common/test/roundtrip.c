/*
 * roundtrip.c - common FFT self-test of the public fft_wrapper API:
 *   1. forward -> inverse returns the original signal (correct 1/N scaling).
 *   2. the static-memory path (fft_get_mem_size / fft_init) is byte-identical to the heap path.
 *   3. interleaved forward/inverse calls on one handle (which share its input staging) give the
 *      results a fresh handle gives, never change a result the caller still holds, and never
 *      modify their inputs.
 */
#include "fft_wrapper.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Reference result of one call on a handle that has never run anything else. */
static void fresh_forward(int n, const float* in, Complex* out) {
    FftHandle* f = fft_create(n);
    fft_forward(f, in, out);
    fft_destroy(f);
}
static void fresh_inverse(int n, const Complex* in, float* out) {
    FftHandle* f = fft_create(n);
    fft_inverse(f, in, out);
    fft_destroy(f);
}

static void fill_test_signal(float* x, int n, unsigned seed, int with_nan) {
    for (int i = 0; i < n; i++) {
        seed = seed * 1664525u + 1013904223u;
        x[i] = (float)((int)(seed >> 8) - (1 << 23)) / (float)(1 << 23);
    }
    if (with_nan) x[seed % (unsigned)n] = NAN;  /* a quiet NaN travels through both calls too */
}

static int staging_independence(int n, int use_pool) {
    const int nf = n / 2 + 1;
    float *a = malloc(n * 4), *b = malloc(n * 4), *a0 = malloc(n * 4);
    float *ya = malloc(n * 4), *yb = malloc(n * 4), *ra = malloc(n * 4), *rb = malloc(n * 4);
    Complex *sa = malloc(nf * 8), *sb = malloc(nf * 8), *sa0 = malloc(nf * 8), *ta = malloc(nf * 8), *tb = malloc(nf * 8);
    void* pool = NULL;
    FftHandle* h;
    int ok = 1;
    if (use_pool) {
        size_t sz = fft_get_mem_size(n);
        if (posix_memalign(&pool, 16, sz) != 0) return 0;
        h = fft_init(pool, sz, n);
    } else {
        h = fft_create(n);
    }
    if (!h) return 0;
    fill_test_signal(a, n, 1u, 1); fill_test_signal(b, n, 2u, 1);
    memcpy(a0, a, n * 4);
    fresh_forward(n, a, ta); fresh_forward(n, b, tb);
    fresh_inverse(n, ta, ra); fresh_inverse(n, tb, rb);

    fft_forward(h, a, sa);                 /* sa stays held while the handle is reused */
    memcpy(sa0, sa, nf * 8);
    fft_inverse(h, sa, ya);                /* inverse on the handle that produced sa */
    fft_forward(h, b, sb);
    fft_inverse(h, sb, yb);
    ok &= !memcmp(sa, sa0, nf * 8);                        /* held result unchanged, input preserved */
    ok &= !memcmp(a, a0, n * 4);
    ok &= !memcmp(sa, ta, nf * 8) && !memcmp(sb, tb, nf * 8);   /* equal to fresh-handle results */
    ok &= !memcmp(ya, ra, n * 4) && !memcmp(yb, rb, n * 4);
    fft_inverse(h, ta, ya);                /* inverse first after a forward: no stale staging leaks in */
    ok &= !memcmp(ya, ra, n * 4);
    fft_forward(h, b, sb);                 /* and forward after an inverse */
    ok &= !memcmp(sb, tb, nf * 8);

    /* The comparisons above cannot see a defect that a fresh handle shares, so
     * a finite signal must also come back from forward -> inverse on the
     * reused handle (full Nyquist content, 1/N scaling). */
    fill_test_signal(a, n, 3u, 0);
    fft_forward(h, a, sa);
    fft_inverse(h, sa, ya);
    for (int i = 0; i < n; i++)
        if (!(fabsf(ya[i] - a[i]) < 1e-4f)) ok = 0;

    fft_destroy(h);
    free(pool);
    free(a); free(b); free(a0); free(ya); free(yb); free(ra); free(rb);
    free(sa); free(sb); free(sa0); free(ta); free(tb);
    return ok;
}

int main(void) {
    const int nfft = 512, nf = nfft / 2 + 1;
    float in[512], out_heap[512], out_static[512];
    Complex spec[257];

    for (int i = 0; i < nfft; i++)
        in[i] = sinf(2.0f * 3.14159265f * 5.0f * i / nfft) + 0.3f;

    /* Heap path */
    FftHandle* h = fft_create(nfft);
    if (!h) { printf("FAIL: fft_create\n"); return 1; }
    fft_forward(h, in, spec);
    fft_inverse(h, spec, out_heap);

    double maxerr = 0, sin_ = 0, sout = 0;
    for (int i = 0; i < nfft; i++) {
        double e = fabs(out_heap[i] - in[i]);
        if (e > maxerr) maxerr = e;
        sin_ += fabs(in[i]); sout += fabs(out_heap[i]);
    }
    printf("round-trip: ratio(sum|out|/sum|in|)=%.4f  maxerr=%.3e\n", sout / sin_, maxerr);

    /* Static path */
    size_t sz = fft_get_mem_size(nfft);
    void* pool = NULL;
    if (posix_memalign(&pool, 16, sz) != 0 || !pool) { printf("FAIL: pool\n"); return 1; }
    FftHandle* hs = fft_init(pool, sz, nfft);
    if (!hs) { printf("FAIL: fft_init\n"); return 1; }
    Complex spec2[257];
    fft_forward(hs, in, spec2);
    fft_inverse(hs, spec2, out_static);

    int spec_eq = (memcmp(spec, spec2, nf * sizeof(Complex)) == 0);
    int time_eq = (memcmp(out_heap, out_static, nfft * sizeof(float)) == 0);
    printf("static vs heap: spectrum %s, time-domain %s\n",
           spec_eq ? "byte-equal" : "DIFFER", time_eq ? "byte-equal" : "DIFFER");

    fft_destroy(h);
    fft_destroy(hs);
    free(pool);

    /* fft_magnitude against `sqrtf(re*re+im*im)` computed here under the same
     * -ffp-contract=off build, bit for bit, at n_freqs values around the
     * kernel's 4-lane boundary (0, 1, 3, 4, 5, 257). */
    int mag_ok = 1;
    {
        Complex mspec[257];
        float mag[257], mag_ref[257];
        int nvals[] = { 0, 1, 3, 4, 5, 257 };
        int vi;
        for (int i = 0; i < 257; i++) {
            mspec[i].r = sinf(0.7f * (float)i) * 3.0f - 1.0f;
            mspec[i].i = cosf(1.3f * (float)i) * 2.0f + 0.5f;
        }
        for (vi = 0; vi < (int)(sizeof(nvals) / sizeof(nvals[0])); vi++) {
            int m = nvals[vi];
            memset(mag, 0, sizeof(mag));
            memset(mag_ref, 0, sizeof(mag_ref));
            fft_magnitude(mspec, mag, m);
            for (int i = 0; i < m; i++) {
                float re = mspec[i].r, im = mspec[i].i;
                mag_ref[i] = sqrtf(re * re + im * im);
            }
            if (memcmp(mag, mag_ref, sizeof(mag)) != 0) {
                mag_ok = 0;
                printf("FAIL: fft_magnitude n_freqs=%d differs from scalar reference\n", m);
            }
        }
        /* NULL inputs are a no-op. */
        fft_magnitude(NULL, mag, 4);
        fft_magnitude(mspec, NULL, 4);
    }
    printf("fft_magnitude vs scalar reference (n_freqs=0,1,3,4,5,257): %s\n",
           mag_ok ? "byte-equal" : "DIFFER");

    int stage_ok = 1;
    {
        const int sizes[] = { 16, 64, 256, 512, 8192 };
        for (int i = 0; i < (int)(sizeof(sizes) / sizeof(sizes[0])); i++)
            for (int use_pool = 0; use_pool < 2; use_pool++)
                if (!staging_independence(sizes[i], use_pool)) {
                    stage_ok = 0;
                    printf("FAIL: interleaved calls on one handle, n=%d, %s\n", sizes[i], use_pool ? "pool" : "heap");
                }
    }
    printf("interleaved forward/inverse on one handle (n=16..8192, heap+pool): %s\n",
           stage_ok ? "independent of earlier calls" : "DIFFER");

    int ok = (maxerr < 1e-3) && spec_eq && time_eq && mag_ok && stage_ok;
    printf(ok ? ">>> PASS\n" : ">>> FAIL\n");
    return ok ? 0 : 1;
}
