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

    /* fft_magnitude() NEON-path sanity (s4-audio-common-sweep review):
     * fft_magnitude has no production caller anywhere in the four-repo tree
     * (currently dead code -- see the review), so there is no existing
     * consumer regression gate covering it. Cross-check the function's
     * output bit-for-bit against the exact same `sqrtf(re*re+im*im)`
     * formula computed independently right here, in the SAME translation
     * unit under the SAME -ffp-contract=off build flag -- this directly
     * exercises whichever path (NEON body + scalar tail, or plain scalar on
     * a non-NEON build) fft_magnitude() actually took, at several n_freqs
     * values that cross the kernel's 4-lane boundary (0, 1, 3, 4, 5, 257 --
     * matching the KISS nf=257 buffer already built above). */
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
        /* NULL-input no-op guard (matches every other fft_wrapper entry
         * point's `if (!spectrum || !magnitude) return;` contract). */
        fft_magnitude(NULL, mag, 4);
        fft_magnitude(mspec, NULL, 4);
    }
    printf("fft_magnitude vs scalar reference (n_freqs=0,1,3,4,5,257): %s\n",
           mag_ok ? "byte-equal" : "DIFFER");

    int ok = (maxerr < 1e-3) && spec_eq && time_eq && mag_ok;
    printf(ok ? ">>> PASS\n" : ">>> FAIL\n");
    return ok ? 0 : 1;
}
