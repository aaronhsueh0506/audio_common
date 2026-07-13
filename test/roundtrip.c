/*
 * roundtrip.c - audio_common FFT self-test (backend-agnostic).
 *
 * Exercises the public fft_wrapper API for whichever backend is linked
 * (KISS or NE10):
 *   1. forward -> inverse must return the original signal (correct 1/N scaling).
 *   2. the static-memory path (fft_get_mem_size / fft_init) must be
 *      byte-identical to the heap path (fft_create).
 */
#include "fft_wrapper.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

    int ok = (maxerr < 1e-3) && spec_eq && time_eq;
    printf(ok ? ">>> PASS\n" : ">>> FAIL\n");
    return ok ? 0 : 1;
}
