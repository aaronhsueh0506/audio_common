/*
 * test_ne10_c_parity.c - F11: NEON-vs-C parity data generator.
 *
 * This TU is built TWICE by the `test_ne10_force_c` Makefile target: once
 * against the normal NE10 backend (calls the `_neon` kernels) and once with
 * -DFFT_NE10_FORCE_C (fft_wrapper_ne10.c routes every call through the `_c`
 * scalar kernels instead -- see that file's FFT_NE10_FORCE_C block). Both
 * runs exercise the exact same fixed test vectors (impulse + a deterministic
 * PRNG signal) at nfft in {256, 512, 1024} and dump forward-spectrum +
 * inverse-output raw float32 bytes to the file named by argv[1]. A separate
 * tiny tool (ne10_parity_compare.c) then diffs the two dump files.
 *
 * This program also self-checks its OWN round-trip error (forward then
 * inverse must recover the input, same check as roundtrip.c) so a failure
 * here is caught before it's blamed on the other build's kernel variant.
 *
 * The PRNG is a fixed, hand-rolled LCG (not libc rand()) specifically so the
 * "random" vector is bit-identical across both builds/platforms -- libc
 * rand() is not required to produce the same sequence across libc versions
 * or even across -DFFT_NE10_FORCE_C vs. not (different translation but same
 * libc here, so it would probably be fine, but there is no reason to depend
 * on that when a 2-line LCG removes the dependency entirely).
 */
#include "fft_wrapper.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static uint32_t g_lcg_state;

static void lcg_seed(uint32_t seed) {
    g_lcg_state = seed;
}

/* Numerical Recipes LCG constants; maps to roughly [-1, 1). */
static float lcg_randf(void) {
    g_lcg_state = g_lcg_state * 1664525u + 1013904223u;
    return ((float)(g_lcg_state >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
}

static void gen_impulse(float* buf, int n) {
    memset(buf, 0, (size_t)n * sizeof(float));
    buf[0] = 1.0f;
}

static void gen_random(float* buf, int n) {
    lcg_seed(0xC0FFEEu); /* same seed regardless of nfft -> reproducible across builds */
    for (int i = 0; i < n; i++) buf[i] = lcg_randf();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <dump_file>\n", argv[0]);
        return 2;
    }

    FILE* fp = fopen(argv[1], "wb");
    if (!fp) { perror("fopen"); return 2; }

    static const int nffts[] = {256, 512, 1024};
    int ok = 1;

    for (size_t s = 0; s < sizeof(nffts) / sizeof(nffts[0]); s++) {
        const int nfft = nffts[s];
        const int nf = nfft / 2 + 1;

        float*   in   = (float*)malloc((size_t)nfft * sizeof(float));
        float*   out  = (float*)malloc((size_t)nfft * sizeof(float));
        Complex* spec = (Complex*)malloc((size_t)nf * sizeof(Complex));
        if (!in || !out || !spec) { fprintf(stderr, "FAIL: alloc\n"); ok = 0; goto next; }

        {
            FftHandle* h = fft_create(nfft);
            if (!h) { fprintf(stderr, "FAIL: fft_create(%d)\n", nfft); ok = 0; goto next; }

            for (int sig = 0; sig < 2; sig++) { /* 0 = impulse, 1 = deterministic random */
                if (sig == 0) gen_impulse(in, nfft);
                else          gen_random(in, nfft);

                fft_forward(h, in, spec);
                fft_inverse(h, spec, out);

                double maxerr = 0.0;
                for (int i = 0; i < nfft; i++) {
                    double e = fabs((double)out[i] - (double)in[i]);
                    if (e > maxerr) maxerr = e;
                }
                fprintf(stderr, "nfft=%-4d sig=%-7s own round-trip maxerr=%.3e\n",
                        nfft, sig == 0 ? "impulse" : "random", maxerr);
                if (maxerr > 1e-3) ok = 0;

                /* Raw dump: spectrum bins then time-domain output, float32 LE
                 * (host-native; both builds run on the same host/ABI here). */
                if (fwrite(spec, sizeof(Complex), (size_t)nf, fp) != (size_t)nf) { ok = 0; }
                if (fwrite(out, sizeof(float), (size_t)nfft, fp) != (size_t)nfft) { ok = 0; }
            }

            fft_destroy(h);
        }
next:
        free(in); free(out); free(spec);
    }

    fclose(fp);
    fprintf(stderr, ok ? ">>> PASS (own round-trip, dump written to %s)\n"
                       : ">>> FAIL (own round-trip)\n", argv[1]);
    return ok ? 0 : 1;
}
