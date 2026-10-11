/*
 * ne10_parity_compare.c - NEON-vs-C dump comparison tool.
 *
 * Standalone (libc only). Reads the two raw float32 dump files written by two
 * builds of test_ne10_c_parity.c (one linked against the NE10 `_neon`
 * kernels, one against the SIMD=0 `_c` kernels) and reports the maximum
 * absolute and relative difference between them.
 *
 * The dumps are not expected to be bit-identical: the two kernel sets
 * compute the same FFT with different instruction sequences, so rounding
 * accumulates differently. The tool bounds that drift.
 *
 * Usage: ne10_parity_compare <neon_dump> <c_dump>
 * Exit status: 0 if max relative diff <= NE10_PARITY_REL_TOL, else 1.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* The NEON-vs-C drift for nfft 256/512/1024 (impulse and a full-scale
 * deterministic random signal) is orders of magnitude below these
 * tolerances. They are generous on purpose: they catch a real divergence
 * (wrong kernel, corrupted twiddle table, layout error), not ulp-level
 * rounding. */
#define NE10_PARITY_REL_TOL 1e-4
#define NE10_PARITY_ABS_TOL 1e-4

static long file_size(FILE* fp) {
    if (fseek(fp, 0, SEEK_END) != 0) return -1;
    long sz = ftell(fp);
    if (fseek(fp, 0, SEEK_SET) != 0) return -1;
    return sz;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <neon_dump> <c_dump>\n", argv[0]);
        return 2;
    }

    FILE* fa = fopen(argv[1], "rb");
    FILE* fb = fopen(argv[2], "rb");
    if (!fa || !fb) { perror("fopen"); return 2; }

    long sa = file_size(fa), sb = file_size(fb);
    if (sa < 0 || sb < 0 || sa != sb || sa % (long)sizeof(float) != 0) {
        fprintf(stderr, "FAIL: dump size mismatch or unreadable (a=%ld b=%ld)\n", sa, sb);
        return 1;
    }

    size_t n = (size_t)sa / sizeof(float);
    float* a = (float*)malloc(sa);
    float* b = (float*)malloc(sb);
    if (!a || !b) { fprintf(stderr, "FAIL: alloc\n"); return 2; }

    if (fread(a, sizeof(float), n, fa) != n || fread(b, sizeof(float), n, fb) != n) {
        fprintf(stderr, "FAIL: short read\n");
        return 1;
    }
    fclose(fa); fclose(fb);

    double max_abs = 0.0, max_rel = 0.0;
    size_t max_abs_idx = 0, max_rel_idx = 0;
    int nan_or_inf = 0;

    for (size_t i = 0; i < n; i++) {
        double da = (double)a[i], db = (double)b[i];
        if (isnan(da) || isnan(db) || isinf(da) || isinf(db)) { nan_or_inf = 1; }
        double diff = fabs(da - db);
        double denom = fmax(fabs(da), fabs(db));
        double rel = denom > 1e-9 ? diff / denom : diff; /* near-zero: fall back to abs */

        if (diff > max_abs) { max_abs = diff; max_abs_idx = i; }
        if (rel > max_rel)  { max_rel = rel;  max_rel_idx = i; }
    }

    printf("F11 NE10 NEON-vs-C parity: %zu floats compared\n", n);
    printf("  max abs diff = %.6e at index %zu (neon=%.9g c=%.9g)\n",
           max_abs, max_abs_idx, (double)a[max_abs_idx], (double)b[max_abs_idx]);
    printf("  max rel diff = %.6e at index %zu (neon=%.9g c=%.9g)\n",
           max_rel, max_rel_idx, (double)a[max_rel_idx], (double)b[max_rel_idx]);

    int ok = !nan_or_inf && max_abs <= NE10_PARITY_ABS_TOL && max_rel <= NE10_PARITY_REL_TOL;
    printf(ok ? ">>> PASS (within abs<=%.1e / rel<=%.1e)\n"
              : ">>> FAIL (exceeds abs<=%.1e / rel<=%.1e)\n",
           NE10_PARITY_ABS_TOL, NE10_PARITY_REL_TOL);

    free(a); free(b);
    return ok ? 0 : 1;
}
