/*
 * test_stft.c - audio_stft.h contracts.
 *
 *   windows   ac_window_build against two independent spellings of the
 *      periodic sqrt-Hann formula at every power of two in [16, 8192].
 *   latency   the measured impulse delay of the analysis -> synthesis chain
 *      equals ac_stft_latency_samples, and the chain reconstructs its input.
 *   API contract   invalid sizes and configurations, block size and
 *      alignment checks, borrowed tables, dirty versus zeroed pools.
 *   scaled entry points   with non-unit spec_scale / ispec_scale the
 *      interleaved (Complex) and the split calls give the same bits, and both
 *      equal the unscaled transform times the scale.
 *
 * SIMD=0 / SIMD=1 comes from the build; results are only comparable within
 * one SIMD setting.
 */
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio_stft.h"
#include "fft_wrapper.h"
#include "utility/mem_align.h"

/* ----------------------------- test utilities ----------------------------- */

static int g_fail = 0;
static int g_shown = 0;

static void fail(const char* what, long where) {
    g_fail++;
    if (g_shown < 40) {
        g_shown++;
        fprintf(stderr, "FAIL: %s (index %ld)\n", what, where);
    }
}

static void expect(int cond, const char* what) {
    if (!cond) fail(what, -1);
}

/* Byte compare; reports the first differing element index. */
static int same_bytes(const void* a, const void* b, size_t n_floats, const char* what) {
    const unsigned char* x = (const unsigned char*)a;
    const unsigned char* y = (const unsigned char*)b;
    size_t i;
    for (i = 0; i < n_floats; ++i) {
        if (memcmp(x + 4 * i, y + 4 * i, 4) != 0) {
            fail(what, (long)i);
            return 0;
        }
    }
    return 1;
}

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd32(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (uint32_t)(g_rng >> 32);
}
static float rnd_unit(void) { return (float)(int32_t)rnd32() * (1.0f / 2147483648.0f); }
static float f_bits(uint32_t b) { float f; memcpy(&f, &b, sizeof(f)); return f; }

enum { KIND_UNIFORM, KIND_RANGE, KIND_SIGNED_ZERO, KIND_SPARSE, KIND_DENORM,
       KIND_SPECIAL, KIND_DC, KIND_ALT, KIND_SILENCE, KIND_NEGZERO, KIND_COUNT };

static float gen_value(int kind, int idx) {
    switch (kind) {
    case KIND_UNIFORM: return rnd_unit();
    case KIND_RANGE:   return rnd_unit() * ldexpf(1.0f, (int)(rnd32() % 61u) - 30);
    case KIND_SIGNED_ZERO: return (rnd32() & 1u) ? -0.0f : 0.0f;
    case KIND_SPARSE:  return (rnd32() % 16u) ? 0.0f : rnd_unit();
    case KIND_DENORM:  return f_bits((rnd32() & 0x007fffffu) | ((rnd32() & 1u) << 31));
    case KIND_SPECIAL: {
        static const uint32_t sp[] = { 0x7f800000u, 0xff800000u, 0x7fc00001u, 0x7fa00001u,
                                       0x80000000u, 0x00000000u, 0x7f7fffffu, 0x00800000u,
                                       0x3f800000u, 0xffc12345u };
        if (rnd32() % 24u == 0) return f_bits(sp[rnd32() % (sizeof(sp) / sizeof(sp[0]))]);
        return rnd_unit();
    }
    case KIND_DC:      return 0.5f;
    case KIND_ALT:     return (idx & 1) ? -1.0f : 1.0f;
    case KIND_NEGZERO: return -0.0f;
    default:           return 0.0f;
    }
}

static void gen_block(float* x, int n, int kind) {
    int i;
    for (i = 0; i < n; ++i) x[i] = gen_value(kind, i);
}

static void* pool_alloc(size_t bytes) {
    void* p = NULL;
    if (posix_memalign(&p, 16, bytes ? bytes : 16) != 0) return NULL;
    memset(p, 0xA5, bytes);
    return p;
}

static AcStft* make_stft(const AcStftConfig* cfg, void** keep) {
    size_t bytes = ac_stft_get_mem_size(cfg);
    void* mem = pool_alloc(bytes);
    AcStft* s = ac_stft_init(mem, bytes, cfg);
    if (!s) fail("ac_stft_init", -1);
    *keep = mem;
    return s;
}

static AcIstft* make_istft(const AcStftConfig* cfg, void** keep) {
    size_t bytes = ac_istft_get_mem_size(cfg);
    void* mem = pool_alloc(bytes);
    AcIstft* s = ac_istft_init(mem, bytes, cfg);
    if (!s) fail("ac_istft_init", -1);
    *keep = mem;
    return s;
}

/* ----------------------------- windows ----------------------------- */

#define PI_F 3.14159265358979323846f

/* Two spellings of the periodic sqrt-Hann window, w[i] = sqrt(0.5 -
 * 0.5 cos(2 pi i / n)) in float32, written the way a caller writes it; the
 * rounding differs between them only if ac_window_build departs from either. */
static void window_nested(float* w, int n) {
    int k;
    for (k = 0; k < n; ++k)
        w[k] = sqrtf(0.5f * (1.0f - cosf(2.0f * PI_F * (float)k / (float)n)));
}

static void window_flat(float* w, int n) {
    int i;
    for (i = 0; i < n; ++i)
        w[i] = sqrtf(0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)i / (float)n));
}

/* ac_window_build against both spellings, byte for byte. */
static void test_windows(void) {
    int n, nsizes = 0;
    for (n = 16; n <= 8192; n *= 2) {
        float* w  = (float*)pool_alloc((size_t)n * sizeof(float));
        float* ra = (float*)pool_alloc((size_t)n * sizeof(float));
        float* rb = (float*)pool_alloc((size_t)n * sizeof(float));
        ac_window_build(w, n);
        window_nested(ra, n);
        window_flat(rb, n);
        expect(same_bytes(w, ra, (size_t)n, "window vs nested spelling"), "window vs nested spelling");
        expect(same_bytes(w, rb, (size_t)n, "window vs flat spelling"), "window vs flat spelling");
        nsizes++;
        free(w); free(ra); free(rb);
    }
    expect(nsizes == 10, "every power-of-two size from 16 to 8192 is compared");
}

/* ----------------------------- shared helper ----------------------------- */

static void gen_spec(Complex* s, int nf, int kind) {
    int k;
    for (k = 0; k < nf; ++k) {
        s[k].r = gen_value(kind, k);
        s[k].i = gen_value(kind, k + 1);
    }
}


/* ----------------------------- latency / identity ----------------------------- */

static int run_identity(int n, int center, AcOlaMode ola, float* x, float* y, int hops) {
    const int hop = n / 2, nf = n / 2 + 1;
    AcStftConfig cfg;
    AcStft* a; AcIstft* s; void *k1, *k2;
    float *re = (float*)pool_alloc((size_t)nf * 4 * 2), *im = (float*)pool_alloc((size_t)nf * 4 * 2);
    int t, f, nfr;
    ac_stft_config_default(&cfg, n);
    cfg.center = center; cfg.ola = ola;
    a = make_stft(&cfg, &k1);
    s = make_istft(&cfg, &k2);
    memset(y, 0, (size_t)hops * hop * sizeof(float));
    for (t = 0; t < hops; ++t) {
        nfr = ac_stft_push_planar(a, x + (size_t)t * hop, re, im);
        for (f = 0; f < nfr; ++f) {
            float out[2048];
            int e = ac_istft_push_planar(s, re + f * nf, im + f * nf, out);
            if (e) memcpy(y + (size_t)t * hop, out, (size_t)e * sizeof(float));
        }
    }
    free(re); free(im); free(k1); free(k2);
    return ac_stft_latency_samples(&cfg);
}

/* Identity chain: the impulse delay equals ac_stft_latency_samples, and the
 * reconstruction error of random input stays below IDENTITY_MAX_ERR. The
 * error depends on N; the input is seeded here so that it does not depend
 * on TEST_N. */
#define IDENTITY_MAX_ERR 1e-6
static void test_latency(void) {
    static const int sizes[] = { 64, 256, 512, 1024 };
    size_t si;
    g_rng = 0x2545F4914F6CDD1Dull;
    for (si = 0; si < sizeof(sizes) / sizeof(sizes[0]); ++si) {
        const int n = sizes[si], hop = n / 2, hops = 24;
        float* x = (float*)pool_alloc((size_t)hops * hop * 4);
        float* y = (float*)pool_alloc((size_t)hops * hop * 4);
        int center;
        for (center = 0; center <= 1; ++center) {
            int lat = run_identity(n, center, center ? AC_OLA_ENV_NORM : AC_OLA_ACCUMULATE, x, y, 0);
            int i, argmax = 0, t0 = 5 * hop + 3;
            float best = 0.0f;
            double err = 0.0;
            memset(x, 0, (size_t)hops * hop * 4);
            x[t0] = 1.0f;
            run_identity(n, center, center ? AC_OLA_ENV_NORM : AC_OLA_ACCUMULATE, x, y, hops);
            for (i = 0; i < hops * hop; ++i)
                if (fabsf(y[i]) > best) { best = fabsf(y[i]); argmax = i; }
            printf("LAT| N=%d center=%d impulse_at=%d peak_at=%d delay=%d peak=%.7f api=%d\n",
                   n, center, t0, argmax, argmax - t0, best, lat);
            expect(argmax - t0 == lat, "measured impulse delay equals ac_stft_latency_samples");
            gen_block(x, hops * hop, KIND_UNIFORM);
            run_identity(n, center, center ? AC_OLA_ENV_NORM : AC_OLA_ACCUMULATE, x, y, hops);
            for (i = hop; i < hops * hop - hop; ++i) {
                double e = fabs((double)y[i + hop] - (double)x[i]);
                if (e > err) err = e;
            }
            printf("LAT| N=%d center=%d identity max_abs_error=%.3g\n", n, center, err);
            expect(err < IDENTITY_MAX_ERR, "analysis -> synthesis identity reconstructs the delayed input");
        }
        free(x); free(y);
    }
}

/* ----------------------------- API contract ----------------------------- */

static void test_api_contract(void) {
    AcStftConfig cfg;
    static const int bad_sizes[] = { 0, -16, 8, 15, 17, 100, 8193, 16384 };
    size_t i;
    void* mem;
    size_t bytes;
    FftHandle* wrong;

    for (i = 0; i < sizeof(bad_sizes) / sizeof(bad_sizes[0]); ++i) {
        ac_stft_config_default(&cfg, bad_sizes[i]);
        expect(ac_stft_get_mem_size(&cfg) == 0, "invalid fft_size: stft size 0");
        expect(ac_istft_get_mem_size(&cfg) == 0, "invalid fft_size: istft size 0");
        expect(ac_stft_latency_samples(&cfg) == -1, "invalid fft_size: latency -1");
    }
    ac_stft_config_default(&cfg, 512);
    expect(ac_stft_latency_samples(&cfg) == 256, "latency 512 -> 256");
    cfg.center = 2;  expect(ac_stft_get_mem_size(&cfg) == 0, "invalid center");
    ac_stft_config_default(&cfg, 512);
    cfg.ola = (AcOlaMode)7; expect(ac_istft_get_mem_size(&cfg) == 0, "invalid ola mode");
    expect(ac_stft_get_mem_size(NULL) == 0 && ac_stft_init(NULL, 0, NULL) == NULL, "NULL config");

    ac_stft_config_default(&cfg, 512);
    bytes = ac_stft_get_mem_size(&cfg);
    mem = pool_alloc(bytes + 16);
    expect(ac_stft_init(mem, bytes - 16, &cfg) == NULL, "short block rejected");
    expect(ac_stft_init((char*)mem + 8, bytes, &cfg) == NULL, "misaligned block rejected");
    expect(ac_stft_init(mem, bytes, &cfg) != NULL, "exact-size block accepted");
    free(mem);

    wrong = fft_create(256);
    cfg.fft = wrong;
    expect(ac_stft_get_mem_size(&cfg) == 0, "borrowed handle of another size rejected");
    fft_destroy(wrong);

    /* borrowing both tables is smaller than carving them */
    {
        AcStftConfig own, borrowed;
        FftHandle* h = fft_create(512);
        float w[512];
        ac_stft_config_default(&own, 512);
        borrowed = own; borrowed.fft = h; borrowed.window = w;
        expect(ac_stft_get_mem_size(&borrowed) < ac_stft_get_mem_size(&own), "borrowed config needs less memory");
        expect(ac_istft_get_mem_size(&borrowed) < ac_istft_get_mem_size(&own), "borrowed istft needs less memory");
        fft_destroy(h);
    }
    /* a dirty pool and a zeroed pool give identical streams */
    {
        const int n = 512, hop = 256, nf = 257;
        AcStftConfig c1, c2;
        AcStft *a, *b; AcIstft *s, *u;
        void *ka, *kb, *ks, *ku;
        size_t sa, ss;
        float in[256], ra[2 * 257], ia[2 * 257], rb[2 * 257], ib[2 * 257], oa[256], ob[256];
        int t;
        ac_stft_config_default(&c1, n); c1.center = 1; c1.ola = AC_OLA_ENV_NORM;
        c2 = c1;
        sa = ac_stft_get_mem_size(&c1); ss = ac_istft_get_mem_size(&c1);
        ka = pool_alloc(sa); kb = calloc(1, sa + 16); ks = pool_alloc(ss); ku = calloc(1, ss + 16);
        a = ac_stft_init(ka, sa, &c1); b = ac_stft_init((void*)(((uintptr_t)kb + 15u) & ~(uintptr_t)15u), sa, &c2);
        s = ac_istft_init(ks, ss, &c1); u = ac_istft_init((void*)(((uintptr_t)ku + 15u) & ~(uintptr_t)15u), ss, &c2);
        for (t = 0; t < 40; ++t) {
            int na, nb, f;
            gen_block(in, hop, KIND_UNIFORM);
            na = ac_stft_push_planar(a, in, ra, ia);
            nb = ac_stft_push_planar(b, in, rb, ib);
            expect(na == nb, "dirty vs zero pool frame count");
            same_bytes(ra, rb, (size_t)na * (size_t)nf, "dirty vs zero pool analysis re");
            for (f = 0; f < na; ++f) {
                int ea = ac_istft_push_planar(s, ra + f * nf, ia + f * nf, oa);
                int eb = ac_istft_push_planar(u, rb + f * nf, ib + f * nf, ob);
                expect(ea == eb, "dirty vs zero pool emit count");
                same_bytes(oa, ob, (size_t)ea, "dirty vs zero pool synthesis");
            }
        }
        free(ka); free(kb); free(ks); free(ku);
    }
}

/* ------------------- interleaved and planar entry points ------------------- */

/* Bit comparison of an interleaved spectrum with a split one. */
static int same_complex_planar(const Complex* c, const float* re, const float* im,
                               int n, const char* what) {
    int k;
    for (k = 0; k < n; ++k) {
        if (memcmp(&c[k].r, &re[k], sizeof(float)) != 0 ||
            memcmp(&c[k].i, &im[k], sizeof(float)) != 0) {
            fail(what, k);
            return 0;
        }
    }
    return 1;
}

/* Kinds without NaN / Inf: a non-unit scale multiplies, and NaN payloads are
 * outside the bit-exact contract of the NEON and scalar multiplies. */
static const int k_finite_kinds[] = { KIND_UNIFORM, KIND_RANGE, KIND_SIGNED_ZERO, KIND_SPARSE,
                                      KIND_DENORM, KIND_DC, KIND_ALT, KIND_SILENCE, KIND_NEGZERO };
#define N_FINITE_KINDS ((int)(sizeof(k_finite_kinds) / sizeof(k_finite_kinds[0])))

/* With non-unit spec_scale / ispec_scale the interleaved (Complex) and the
 * split entry points must give the same bits, and both must equal the
 * unscaled transform multiplied by the scale. Scales: DFN2-like 1/sqrt(N) and
 * sqrt(N), the same at N = 512, and values that are not powers of two. */
static void test_scaled_entry_points(void) {
    typedef struct { int n; float spec_scale, ispec_scale; } ScaleCase;
    const ScaleCase cases[] = {
        { 1024, 1.0f / sqrtf(1024.0f), sqrtf(1024.0f) },
        { 512,  1.0f / sqrtf(512.0f),  sqrtf(512.0f)  },
        { 256,  0.3f,  3.1f  },
        { 64,   2.0f,  0.5f  },
        { 16,  -0.7f,  1.9f  },
    };
    size_t ci;
    int checked = 0;
    for (ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
        const int n = cases[ci].n, hop = n / 2, nf = n / 2 + 1;
        int center, ola;
        for (center = 0; center <= 1; ++center) for (ola = 0; ola <= 1; ++ola) {
            AcStftConfig cs, cu;
            AcStft *ai, *ap, *au;
            AcIstft *si, *sp, *su;
            void *k1, *k2, *k3, *k4, *k5, *k6;
            Complex *spi = (Complex*)pool_alloc((size_t)2 * nf * sizeof(Complex));
            Complex *spu = (Complex*)pool_alloc((size_t)2 * nf * sizeof(Complex));
            Complex *scaled = (Complex*)pool_alloc((size_t)nf * sizeof(Complex));
            Complex *orig = (Complex*)pool_alloc((size_t)nf * sizeof(Complex));
            float *re = (float*)pool_alloc((size_t)2 * nf * sizeof(float));
            float *im = (float*)pool_alloc((size_t)2 * nf * sizeof(float));
            float *in = (float*)pool_alloc((size_t)n * sizeof(float));
            float *oi = (float*)pool_alloc((size_t)hop * sizeof(float));
            float *op = (float*)pool_alloc((size_t)hop * sizeof(float));
            float *ou = (float*)pool_alloc((size_t)hop * sizeof(float));
            char what[128];
            int t, f, ni, np, nu;
            ac_stft_config_default(&cs, n);
            cs.center = center;
            cs.ola = ola ? AC_OLA_ENV_NORM : AC_OLA_ACCUMULATE;
            cs.spec_scale = cases[ci].spec_scale;
            cs.ispec_scale = cases[ci].ispec_scale;
            cu = cs; cu.spec_scale = 1.0f; cu.ispec_scale = 1.0f;
            ai = make_stft(&cs, &k1); ap = make_stft(&cs, &k2); au = make_stft(&cu, &k3);
            si = make_istft(&cs, &k4); sp = make_istft(&cs, &k5); su = make_istft(&cu, &k6);
            if (!ai || !ap || !au || !si || !sp || !su) return;

            for (t = 0; t < 24; ++t) {
                const int kind = k_finite_kinds[t % N_FINITE_KINDS];
                /* analysis: push */
                gen_block(in, hop, kind);
                ni = ac_stft_push(ai, in, spi);
                np = ac_stft_push_planar(ap, in, re, im);
                nu = ac_stft_push(au, in, spu);
                snprintf(what, sizeof(what), "scaled N=%d center=%d push frame count", n, center);
                expect(ni == np && ni == nu, what);
                for (f = 0; f < ni && ni == np && ni == nu; ++f) {
                    int k;
                    snprintf(what, sizeof(what), "scaled N=%d center=%d push interleaved vs planar t=%d", n, center, t);
                    same_complex_planar(spi + f * nf, re + f * nf, im + f * nf, nf, what);
                    for (k = 0; k < nf; ++k) {
                        scaled[k].r = spu[f * nf + k].r * cases[ci].spec_scale;
                        scaled[k].i = spu[f * nf + k].i * cases[ci].spec_scale;
                    }
                    snprintf(what, sizeof(what), "scaled N=%d center=%d push vs unscaled * scale t=%d", n, center, t);
                    same_bytes(spi + f * nf, scaled, (size_t)nf * 2, what);
                    checked++;
                }
                /* analysis: stateless frame of n samples */
                gen_block(in, n, kind);
                ac_stft_frame(ai, in, spi);
                ac_stft_frame_planar(ap, in, re, im);
                ac_stft_frame(au, in, spu);
                snprintf(what, sizeof(what), "scaled N=%d frame interleaved vs planar t=%d", n, t);
                same_complex_planar(spi, re, im, nf, what);
                {
                    int k;
                    for (k = 0; k < nf; ++k) {
                        scaled[k].r = spu[k].r * cases[ci].spec_scale;
                        scaled[k].i = spu[k].i * cases[ci].spec_scale;
                    }
                }
                snprintf(what, sizeof(what), "scaled N=%d frame vs unscaled * scale t=%d", n, t);
                same_bytes(spi, scaled, (size_t)nf * 2, what);
                checked++;

                /* synthesis */
                gen_spec(orig, nf, kind);
                memcpy(spi, orig, (size_t)nf * sizeof(Complex));
                for (f = 0; f < nf; ++f) { re[f] = orig[f].r; im[f] = orig[f].i; }
                for (f = 0; f < nf; ++f) {
                    scaled[f].r = orig[f].r * cases[ci].ispec_scale;
                    scaled[f].i = orig[f].i * cases[ci].ispec_scale;
                }
                ni = ac_istft_push(si, spi, oi);
                np = ac_istft_push_planar(sp, re, im, op);
                nu = ac_istft_push(su, scaled, ou);
                snprintf(what, sizeof(what), "scaled N=%d ola=%d synthesis emit count", n, ola);
                expect(ni == np && ni == nu, what);
                snprintf(what, sizeof(what), "scaled N=%d ola=%d synthesis input left unchanged", n, ola);
                same_bytes(spi, orig, (size_t)nf * 2, what);
                if (ni == np && ni == nu) {
                    snprintf(what, sizeof(what), "scaled N=%d ola=%d synthesis interleaved vs planar t=%d", n, ola, t);
                    same_bytes(oi, op, (size_t)ni, what);
                    snprintf(what, sizeof(what), "scaled N=%d ola=%d synthesis vs unscaled of scaled input t=%d", n, ola, t);
                    same_bytes(oi, ou, (size_t)ni, what);
                }
                checked++;
            }
            if (center) {
                ni = ac_stft_flush(ai, spi);
                np = ac_stft_flush_planar(ap, re, im);
                nu = ac_stft_flush(au, spu);
                expect(ni == 1 && np == 1 && nu == 1, "scaled flush returns one frame");
                if (ni == 1 && np == 1 && nu == 1) {
                    int k;
                    snprintf(what, sizeof(what), "scaled N=%d flush interleaved vs planar", n);
                    same_complex_planar(spi, re, im, nf, what);
                    for (k = 0; k < nf; ++k) {
                        scaled[k].r = spu[k].r * cases[ci].spec_scale;
                        scaled[k].i = spu[k].i * cases[ci].spec_scale;
                    }
                    snprintf(what, sizeof(what), "scaled N=%d flush vs unscaled * scale", n);
                    same_bytes(spi, scaled, (size_t)nf * 2, what);
                    checked++;
                }
            }
            if (ola) {
                ni = ac_istft_flush(si, oi);
                np = ac_istft_flush(sp, op);
                nu = ac_istft_flush(su, ou);
                expect(ni == hop && np == hop && nu == hop, "scaled synthesis flush returns hop samples");
                if (ni == hop && np == hop && nu == hop) {
                    snprintf(what, sizeof(what), "scaled N=%d synthesis flush", n);
                    same_bytes(oi, op, (size_t)hop, what);
                    same_bytes(oi, ou, (size_t)hop, what);
                    checked++;
                }
            }
            free(spi); free(spu); free(scaled); free(orig); free(re); free(im); free(in);
            free(oi); free(op); free(ou);
            free(k1); free(k2); free(k3); free(k4); free(k5); free(k6);
        }
    }
    printf("SCALE| %d comparison groups (interleaved vs planar, and both vs unscaled times scale) over %d configurations\n",
           checked, (int)(sizeof(cases) / sizeof(cases[0])) * 4);
}

/* ----------------------------- main ----------------------------- */

int main(void) {
    test_windows();
    test_latency();
    test_api_contract();
    test_scaled_entry_points();
    if (g_fail) {
        fprintf(stderr, "test_stft FAILED (%d failing checks)\n", g_fail);
        return 1;
    }
    printf("test_stft PASS\n");
    return 0;
}
