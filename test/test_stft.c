/*
 * test_stft.c - audio_stft.h against reference copies of the framing code the
 * pipelines and NN pre/post stages run (files in test/stft_ref/).
 *
 *   family A  pipeline synthesis hop (accumulate, emit, shift, zero the tail):
 *      AC_OLA_ACCUMULATE must match byte for byte.
 *   family C  centered analysis and envelope-normalised synthesis:
 *      center = 1 / AC_OLA_ENV_NORM must match byte for byte.
 *   family B  normalised-spectrum NN stages (spec_scale / ispec_scale): compared and
 *      printed as "B|" lines; the checks allow only a sign-of-zero difference
 *      in the synthesis output.
 *
 * Built twice (TEST_N = 512 and 1024) because the C copy is sized by a macro.
 * SIMD=0 / SIMD=1 and the FFT backend come from the build; results are only
 * comparable within one backend.
 */
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio_stft.h"
#include "fft_wrapper.h"
#include "mem_align.h"
#include "simd_kernel_nn.h"
#include "simd_kernels.h"

#ifndef TEST_N
#define TEST_N 512
#endif

/* ----------------------------- reference harness ----------------------------- */

#define M_PI_F   3.14159265358979323846f
#define M_PI_AEC 3.14159265358979323846f

typedef struct {
    FftHandle* fft;
    float *ifft_buf, *ola, *synth_win;
    int hop, frame_sz;
    struct { uint32_t synth_us; } last_timing;
} AudioPipeline;
static uint32_t audio_pipeline_now_us(void) { return 0; }

typedef struct {
    FftHandle* fft;
    float *ifft_buffer, *ola, *synth_window;
    int hop_size, fft_size;
    struct { uint32_t synth_us; } last_timing;
} FourAecNrRes;
static uint32_t four_aec_nr_res_now_us(void) { return 0; }
static int inputs_finite(const float* x, size_t n) { return skn_all_finite_f32(x, n); }

#define ULCNET_N_FFT TEST_N
#define ULCNET_HOP   (TEST_N / 2)
#define ULCNET_BINS  (TEST_N / 2 + 1)

#include "stft_ref/ref_a.inc"

#define df_common_make_root_hann dfn2_ref_make_root_hann
#define df_common_analysis       dfn2_ref_analysis
#define df_common_synthesis      dfn2_ref_synthesis
#include "stft_ref/ref_b_dfn2.inc"
#undef df_common_make_root_hann
#undef df_common_analysis
#undef df_common_synthesis

#define df_common_make_root_hann gtcrn_ref_make_root_hann
#define df_common_analysis       gtcrn_ref_analysis
#define df_common_synthesis      gtcrn_ref_synthesis
#include "stft_ref/ref_b_gtcrn.inc"
#undef df_common_make_root_hann
#undef df_common_analysis
#undef df_common_synthesis

#include "stft_ref/ref_c.inc"

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
static uint32_t bits_f(float f) { uint32_t b; memcpy(&b, &f, sizeof(b)); return b; }

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

/* Difference statistics between a reference and the new object. */
typedef struct {
    uint64_t n;            /* elements compared                    */
    uint64_t bit_diff;     /* any bit differs                      */
    uint64_t sign_zero;    /* +0.0 vs -0.0 only                    */
    uint64_t nan_diff;     /* a NaN on one side or different payload */
    uint64_t value_diff;   /* different finite/infinite values     */
    uint32_t max_ulp;
    double   max_abs;
} Diff;

static int32_t ordered(float f) {
    int32_t i = (int32_t)bits_f(f);
    return i < 0 ? (int32_t)0x80000000 - i : i;
}

static void diff_add(Diff* d, float ref, float got) {
    d->n++;
    if (bits_f(ref) == bits_f(got)) return;
    d->bit_diff++;
    if (isnan(ref) || isnan(got)) { d->nan_diff++; return; }
    if (ref == got) { d->sign_zero++; return; }   /* only +0 / -0 compare equal with different bits */
    {
        int64_t u = (int64_t)ordered(ref) - (int64_t)ordered(got);
        uint32_t ulp = (uint32_t)(u < 0 ? -u : u);
        double a = fabs((double)ref - (double)got);
        d->value_diff++;
        if (ulp > d->max_ulp) d->max_ulp = ulp;
        if (a > d->max_abs) d->max_abs = a;   /* Inf - Inf style results are NaN and never exceed */
    }
}

static void diff_print(const char* label, const Diff* d) {
    printf("B| %-34s n=%llu bit_diff=%llu sign_zero=%llu nan_diff=%llu value_diff=%llu max_ulp=%u max_abs=%.9g\n",
           label, (unsigned long long)d->n, (unsigned long long)d->bit_diff,
           (unsigned long long)d->sign_zero, (unsigned long long)d->nan_diff,
           (unsigned long long)d->value_diff, d->max_ulp, d->max_abs);
}

static void diff_merge(Diff* a, const Diff* b) {
    a->n += b->n; a->bit_diff += b->bit_diff; a->sign_zero += b->sign_zero;
    a->nan_diff += b->nan_diff; a->value_diff += b->value_diff;
    if (b->max_ulp > a->max_ulp) a->max_ulp = b->max_ulp;
    if (b->max_abs > a->max_abs) a->max_abs = b->max_abs;
}

/* ----------------------------- windows ----------------------------- */

static void test_windows(void) {
    int n;
    Diff wb_all;
    memset(&wb_all, 0, sizeof(wb_all));
    for (n = 16; n <= 8192; n *= 2) {
        float* w   = (float*)pool_alloc((size_t)n * sizeof(float));
        float* ra  = (float*)pool_alloc((size_t)n * sizeof(float));
        float* rb  = (float*)pool_alloc((size_t)n * sizeof(float));
        float* rc  = (float*)pool_alloc((size_t)n * sizeof(float));
        float* rd  = (float*)pool_alloc((size_t)n * sizeof(float));
        float* re  = (float*)pool_alloc((size_t)n * sizeof(float));
        float* rf  = (float*)pool_alloc((size_t)n * sizeof(float));
        Diff d_dfn2, d_gtcrn, d_bridge, d_ulcnet;
        int i;
        memset(&d_dfn2, 0, sizeof(Diff)); d_gtcrn = d_bridge = d_ulcnet = d_dfn2;
        ac_window_build(w, n);
        ref_mono_window(ra, n);
        ref_4ch_window(rb, n);
        ref_pbfdkf_window(rc, n);
        expect(same_bytes(w, ra, (size_t)n, "window vs mono pipeline"), "window vs mono pipeline");
        same_bytes(w, rb, (size_t)n, "window vs 4ch pipeline");
        same_bytes(w, rc, (size_t)n, "window vs pbfdkf");
        dfn2_ref_make_root_hann(rd, n);
        gtcrn_ref_make_root_hann(re, n);
        make_root_hann(rf, n);
        for (i = 0; i < n; ++i) {
            diff_add(&d_dfn2, w[i], rd[i]);
            diff_add(&d_gtcrn, w[i], re[i]);
            diff_add(&d_bridge, w[i], rf[i]);
        }
        if (n == TEST_N) {
            float* ru = (float*)pool_alloc((size_t)n * sizeof(float));
            ulcnet_make_window(ru);
            for (i = 0; i < n; ++i) diff_add(&d_ulcnet, w[i], ru[i]);
            free(ru);
        }
        {
            char label[64];
            snprintf(label, sizeof(label), "window N=%d A-vs-dfn2", n);   diff_print(label, &d_dfn2);
            snprintf(label, sizeof(label), "window N=%d A-vs-gtcrn", n);  diff_print(label, &d_gtcrn);
            snprintf(label, sizeof(label), "window N=%d A-vs-bridge", n); diff_print(label, &d_bridge);
            if (n == TEST_N) {
                snprintf(label, sizeof(label), "window N=%d A-vs-ulcnet", n);
                diff_print(label, &d_ulcnet);
            }
        }
        diff_merge(&wb_all, &d_dfn2); diff_merge(&wb_all, &d_gtcrn);
        diff_merge(&wb_all, &d_bridge); diff_merge(&wb_all, &d_ulcnet);
        free(w); free(ra); free(rb); free(rc); free(rd); free(re); free(rf);
    }
    diff_print("window all N A-vs-B/C builders", &wb_all);
    expect(wb_all.bit_diff == 0, "A and B window builders agree bit for bit for every power-of-two N");
#ifdef MUTATE_REF_WINDOW
    {
        float w[64], r[64];
        ac_window_build(w, 64);
        ref_mono_window(r, 64);
        r[3] = nextafterf(r[3], 2.0f);
        expect(same_bytes(w, r, 64, "mutated reference window"), "mutation gate (window)");
    }
#endif
}

/* ----------------------------- family A ----------------------------- */

static void gen_spec(Complex* s, int nf, int kind) {
    int k;
    for (k = 0; k < nf; ++k) {
        s[k].r = gen_value(kind, k);
        s[k].i = gen_value(kind, k + 1);
    }
}

static void test_family_a(int n, int hops) {
    const int hop = n / 2, nf = n / 2 + 1;
    float *w_mono = (float*)pool_alloc((size_t)n * 4), *w_4ch = (float*)pool_alloc((size_t)n * 4);
    float *w_bridge = (float*)pool_alloc((size_t)n * 4);
    AudioPipeline pm; FourAecNrRes pf;
    FftHandle *fft_m = fft_create(n), *fft_f = fft_create(n), *fft_b = fft_create(n), *fft_shared = fft_create(n);
    float *ifft_b = (float*)pool_alloc((size_t)n * 4), *ola_b = (float*)pool_alloc((size_t)n * 4);
    Complex* spec = (Complex*)pool_alloc((size_t)nf * sizeof(Complex));
    float *re = (float*)pool_alloc((size_t)nf * 4), *im = (float*)pool_alloc((size_t)nf * 4);
    float *o_m = (float*)pool_alloc((size_t)hop * 4), *o_f = (float*)pool_alloc((size_t)hop * 4);
    float *o_b = (float*)pool_alloc((size_t)hop * 4);
    float *o_a = (float*)pool_alloc((size_t)hop * 4), *o_s = (float*)pool_alloc((size_t)hop * 4);
    float *o_p = (float*)pool_alloc((size_t)hop * 4);
    void *k1, *k2, *k3;
    AcStftConfig cfg_own, cfg_shared;
    AcIstft *own, *shared, *planar;
    int t, kind = 0;
    char what[96];

    ref_mono_window(w_mono, n); ref_4ch_window(w_4ch, n); make_root_hann(w_bridge, n);
    memset(&pm, 0, sizeof(pm)); memset(&pf, 0, sizeof(pf));
    pm.fft = fft_m; pm.hop = hop; pm.frame_sz = n;
    pm.ifft_buf = (float*)pool_alloc((size_t)n * 4); pm.ola = (float*)pool_alloc((size_t)n * 4); pm.synth_win = w_mono;
    pf.fft = fft_f; pf.hop_size = hop; pf.fft_size = n;
    pf.ifft_buffer = (float*)pool_alloc((size_t)n * 4); pf.ola = (float*)pool_alloc((size_t)n * 4); pf.synth_window = w_4ch;
    memset(pm.ola, 0, (size_t)n * 4); memset(pf.ola, 0, (size_t)n * 4); memset(ola_b, 0, (size_t)n * 4);

    ac_stft_config_default(&cfg_own, n);
    cfg_shared = cfg_own;
    cfg_shared.fft = fft_shared;
    cfg_shared.window = w_mono;
    own = make_istft(&cfg_own, &k1);
    shared = make_istft(&cfg_shared, &k2);
    planar = make_istft(&cfg_own, &k3);
    if (!own || !shared || !planar) return;
    expect(ac_istft_window(own) != NULL && ac_istft_window(shared) == w_mono, "window ownership");

    for (t = 0; t < hops; ++t) {
        int got, i;
        if (t % 8 == 0) kind = (int)(rnd32() % KIND_COUNT);
        gen_spec(spec, nf, kind);
        for (i = 0; i < nf; ++i) { re[i] = spec[i].r; im[i] = spec[i].i; }
        synthesize_hop(&pm, spec, o_m);
        finish_hop(&pf, spec, o_f);
        synth_hop(fft_b, spec, ifft_b, ola_b, w_bridge, n, hop, o_b);
        got = ac_istft_push(own, spec, o_a);
        expect(got == hop, "family A emits hop samples every frame");
        ac_istft_push(shared, spec, o_s);
        ac_istft_push_planar(planar, re, im, o_p);
        snprintf(what, sizeof(what), "family A mono N=%d hop %d kind %d", n, t, kind);
        same_bytes(o_m, o_a, (size_t)hop, what);
        snprintf(what, sizeof(what), "family A 4ch N=%d hop %d kind %d", n, t, kind);
        same_bytes(o_f, o_a, (size_t)hop, what);
        snprintf(what, sizeof(what), "family A bridge N=%d hop %d kind %d", n, t, kind);
        same_bytes(o_b, o_a, (size_t)hop, what);
        snprintf(what, sizeof(what), "family A shared fft+window N=%d hop %d", n, t);
        same_bytes(o_s, o_a, (size_t)hop, what);
        snprintf(what, sizeof(what), "family A planar N=%d hop %d kind %d", n, t, kind);
        same_bytes(o_p, o_a, (size_t)hop, what);
    }
    /* reset equals a fresh object */
    ac_istft_reset(own);
    memset(pm.ola, 0, (size_t)n * 4);
    for (t = 0; t < 16; ++t) {
        gen_spec(spec, nf, KIND_UNIFORM);
        synthesize_hop(&pm, spec, o_m);
        ac_istft_push(own, spec, o_a);
        same_bytes(o_m, o_a, (size_t)hop, "family A after reset");
    }
    expect(ac_istft_flush(own, o_a) == 0, "ACCUMULATE flush returns 0");
    free(w_mono); free(w_4ch); free(w_bridge); free(ifft_b); free(ola_b); free(spec); free(re); free(im);
    free(o_m); free(o_f); free(o_b); free(o_a); free(o_s); free(o_p);
    free(pm.ifft_buf); free(pm.ola); free(pf.ifft_buffer); free(pf.ola);
    fft_destroy(fft_m); fft_destroy(fft_f); fft_destroy(fft_b); fft_destroy(fft_shared);
    free(k1); free(k2); free(k3);
}

/* ----------------------------- family C ----------------------------- */

static void to_complex(Complex* c, const float* re, const float* im, int n) {
    int k;
    for (k = 0; k < n; ++k) { c[k].r = re[k]; c[k].i = im[k]; }
}

static void test_family_c_analysis(int hops, int center, int own_window) {
    enum { HOP = ULCNET_HOP, BINS = ULCNET_BINS };
    static float win[ULCNET_N_FFT];
    FftHandle* fft = fft_create(ULCNET_N_FFT);
    static UlcnetAnalysis ref;
    static float rre[2][BINS], rim[2][BINS], are[2][BINS], aim[2][BINS];
    static Complex acpx[2 * BINS], rcpx[2 * BINS];
    static float hop_in[HOP];
    AcStftConfig cfg;
    AcStft *planar, *inter;
    void *k1, *k2;
    int t, kind = 0, f;
    char what[96];

    ulcnet_make_window(win);
    ac_stft_config_default(&cfg, ULCNET_N_FFT);
    cfg.center = center;
    cfg.fft = fft;
    cfg.window = own_window ? NULL : win;
    planar = make_stft(&cfg, &k1);
    inter = make_stft(&cfg, &k2);
    if (!planar || !inter) return;
    if (own_window) same_bytes(ac_stft_window(planar), win, ULCNET_N_FFT, "own window equals the reference table");
    ulcnet_analysis_init(&ref, fft, win);

    for (t = 0; t < hops; ++t) {
        int nr, na, nb;
        if (t % 8 == 0) kind = (int)(rnd32() % KIND_COUNT);
        gen_block(hop_in, HOP, kind);
        memset(are, 0xA5, sizeof(are)); memset(aim, 0xA5, sizeof(aim));
        memset(acpx, 0xA5, sizeof(acpx));
        if (center) nr = ulcnet_analysis_push(&ref, hop_in, rre, rim);
        else nr = ulcnet_analysis_push_frame(&ref, hop_in, rre[0], rim[0]);
        na = ac_stft_push_planar(planar, hop_in, &are[0][0], &aim[0][0]);
        nb = ac_stft_push(inter, hop_in, acpx);
        snprintf(what, sizeof(what), "family C analysis frame count center=%d hop %d", center, t);
        expect(nr == na && nr == nb, what);
        for (f = 0; f < nr; ++f) {
            snprintf(what, sizeof(what), "family C analysis re center=%d hop %d frame %d kind %d", center, t, f, kind);
            same_bytes(rre[f], are[f], BINS, what);
            snprintf(what, sizeof(what), "family C analysis im center=%d hop %d frame %d kind %d", center, t, f, kind);
            same_bytes(rim[f], aim[f], BINS, what);
            to_complex(rcpx + f * BINS, rre[f], rim[f], BINS);
        }
        snprintf(what, sizeof(what), "family C analysis interleaved center=%d hop %d", center, t);
        same_bytes(rcpx, acpx, (size_t)nr * BINS * 2, what);
        if (t < 5 || t == hops - 1) {
            /* flush after a few hops and at the end, on copies of the state */
            if (center) {
                UlcnetAnalysis save = ref;
                int fr = ulcnet_analysis_flush(&save, rre, rim);
                int fa = ac_stft_flush_planar(planar, &are[0][0], &aim[0][0]);
                int fb = ac_stft_flush(inter, acpx);
                snprintf(what, sizeof(what), "family C analysis flush count hop %d", t);
                expect(fr == fa && fr == fb, what);
                if (fr == 1) {
                    same_bytes(rre[0], are[0], BINS, "family C flush re");
                    same_bytes(rim[0], aim[0], BINS, "family C flush im");
                    to_complex(rcpx, rre[0], rim[0], BINS);
                    same_bytes(rcpx, acpx, BINS * 2, "family C flush interleaved");
                }
            } else {
                expect(ac_stft_flush_planar(planar, &are[0][0], &aim[0][0]) == 0, "center=0 flush returns 0");
            }
        }
    }
    /* stateless frame vs the reference transform */
    {
        static float seg[ULCNET_N_FFT];
        gen_block(seg, ULCNET_N_FFT, KIND_UNIFORM);
        ulcnet_rfft(&ref, seg, rre[0], rim[0]);
        ac_stft_frame_planar(planar, seg, are[0], aim[0]);
        ac_stft_frame(inter, seg, acpx);
        same_bytes(rre[0], are[0], BINS, "ac_stft_frame_planar vs ulcnet_rfft re");
        same_bytes(rim[0], aim[0], BINS, "ac_stft_frame_planar vs ulcnet_rfft im");
        to_complex(rcpx, rre[0], rim[0], BINS);
        same_bytes(rcpx, acpx, BINS * 2, "ac_stft_frame vs ulcnet_rfft");
    }
    /* reset equals a fresh object */
    ac_stft_reset(planar);
    ulcnet_analysis_init(&ref, fft, win);
    for (t = 0; t < 6; ++t) {
        int nr, na;
        gen_block(hop_in, HOP, KIND_UNIFORM);
        if (center) nr = ulcnet_analysis_push(&ref, hop_in, rre, rim);
        else nr = ulcnet_analysis_push_frame(&ref, hop_in, rre[0], rim[0]);
        na = ac_stft_push_planar(planar, hop_in, &are[0][0], &aim[0][0]);
        expect(nr == na, "family C frame count after reset");
        for (f = 0; f < nr; ++f) same_bytes(rre[f], are[f], BINS, "family C re after reset");
    }
    fft_destroy(fft); free(k1); free(k2);
}

/* Flush behaviour on streams of 0..3 hops (the short-stream corner cases). */
static void test_family_c_short_flush(void) {
    enum { HOP = ULCNET_HOP, BINS = ULCNET_BINS };
    static float win[ULCNET_N_FFT], hop_in[HOP];
    static float rre[2][BINS], rim[2][BINS], are[BINS], aim[BINS];
    FftHandle* fft = fft_create(ULCNET_N_FFT);
    int pushes;
    ulcnet_make_window(win);
    for (pushes = 0; pushes <= 3; ++pushes) {
        UlcnetAnalysis ref;
        AcStftConfig cfg;
        AcStft* s;
        void* keep;
        int i, fr, fa;
        ac_stft_config_default(&cfg, ULCNET_N_FFT);
        cfg.center = 1; cfg.fft = fft; cfg.window = win;
        s = make_stft(&cfg, &keep);
        ulcnet_analysis_init(&ref, fft, win);
        for (i = 0; i < pushes; ++i) {
            float tmp[2][BINS], tmq[2][BINS];
            gen_block(hop_in, HOP, KIND_UNIFORM);
            ulcnet_analysis_push(&ref, hop_in, tmp, tmq);
            ac_stft_push_planar(s, hop_in, &tmp[0][0], &tmq[0][0]);
        }
        fr = ulcnet_analysis_flush(&ref, rre, rim);
        fa = ac_stft_flush_planar(s, are, aim);
        expect(fr == fa, "short-stream flush count");
        if (fr == 1) { same_bytes(rre[0], are, BINS, "short flush re"); same_bytes(rim[0], aim, BINS, "short flush im"); }
        free(keep);
    }
    fft_destroy(fft);
}

static void test_family_c_synthesis(int frames, int own_window) {
    enum { HOP = ULCNET_HOP, BINS = ULCNET_BINS };
    static float win[ULCNET_N_FFT];
    static float re[BINS], im[BINS], o_r[HOP], o_a[HOP], o_i[HOP], fl_r[ULCNET_N_FFT], fl_a[ULCNET_N_FFT];
    static Complex spec[BINS];
    FftHandle* fft = fft_create(ULCNET_N_FFT);
    UlcnetSynthesis* ref = (UlcnetSynthesis*)pool_alloc(sizeof(UlcnetSynthesis));
    AcStftConfig cfg;
    AcIstft *planar, *inter;
    void *k1, *k2;
    int t, kind = 0, pass;
    char what[96];

    ulcnet_make_window(win);
    if (own_window == 2) {   /* zero and tiny entries put the envelope floor in play */
        int j;
        for (j = 0; j < ULCNET_N_FFT; ++j) {
            if (j % 7 == 0) win[j] = 0.0f;
            else if (j % 11 == 0) win[j] = 1e-6f;
        }
    }
    ac_stft_config_default(&cfg, ULCNET_N_FFT);
    cfg.ola = AC_OLA_ENV_NORM;
    cfg.fft = fft;
    cfg.window = own_window == 1 ? NULL : win;
    planar = make_istft(&cfg, &k1);
    inter = make_istft(&cfg, &k2);
    if (!planar || !inter) return;
    for (pass = 0; pass < 2; ++pass) {   /* pass 1 runs after a reset */
        ulcnet_synthesis_init(ref, fft, win);
        if (pass) { ac_istft_reset(planar); ac_istft_reset(inter); }
        for (t = 0; t < frames; ++t) {
            int nr, na, nb;
            int i;
            if (t % 8 == 0) kind = (int)(rnd32() % KIND_COUNT);
            for (i = 0; i < BINS; ++i) { re[i] = gen_value(kind, i); im[i] = gen_value(kind, i + 1); }
            to_complex(spec, re, im, BINS);
            memset(o_r, 0xA5, sizeof(o_r)); memset(o_a, 0xA5, sizeof(o_a)); memset(o_i, 0xA5, sizeof(o_i));
            nr = ulcnet_synthesis_push(ref, re, im, o_r);
            na = ac_istft_push_planar(planar, re, im, o_a);
            nb = ac_istft_push(inter, spec, o_i);
            snprintf(what, sizeof(what), "family C synthesis count frame %d", t);
            expect(nr == na && nr == nb, what);
            snprintf(what, sizeof(what), "family C synthesis planar frame %d kind %d", t, kind);
            same_bytes(o_r, o_a, HOP, what);
            snprintf(what, sizeof(what), "family C synthesis interleaved frame %d kind %d", t, kind);
            same_bytes(o_r, o_i, HOP, what);
            if (t < 4) {
                int fr, fa, fb;
                memset(fl_r, 0xA5, sizeof(fl_r)); memset(fl_a, 0xA5, sizeof(fl_a));
                fr = ulcnet_synthesis_flush(ref, fl_r);
                fa = ac_istft_flush(planar, fl_a);
                fb = ac_istft_flush(inter, o_i);
                expect(fr == fa && fr == fb, "family C synthesis flush count");
                same_bytes(fl_r, fl_a, (size_t)fr, "family C synthesis flush");
                same_bytes(fl_r, o_i, (size_t)fr, "family C synthesis flush (interleaved object)");
            }
        }
        {
            int fr = ulcnet_synthesis_flush(ref, fl_r);
            int fa = ac_istft_flush(planar, fl_a);
            expect(fr == fa, "family C final flush count");
            same_bytes(fl_r, fl_a, (size_t)fr, "family C final flush");
        }
    }
    {   /* flush on an object that has seen no frame */
        void* k3;
        AcIstft* fresh = make_istft(&cfg, &k3);
        expect(ac_istft_flush(fresh, o_a) == 0, "flush with no frame returns 0");
        free(k3);
    }
    free(ref); fft_destroy(fft); free(k1); free(k2);
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

static void test_latency(void) {
    static const int sizes[] = { 64, 512, 1024 };
    size_t si;
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
            expect(err < 1e-5, "analysis -> synthesis identity reconstructs the delayed input");
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

/* ----------------------------- family B measurement ----------------------------- */

/* Scales as the shipped callers compute them. */
typedef struct {
    int n;
    float norm, inv_norm;
    int planar_spec;     /* 1: split re/im (dfn2), 0: interleaved (gtcrn) */
} BCase;

static void b_run(const BCase* bc, int hops, int signal_kind_mode, const char* label) {
    const int n = bc->n, hop = n / 2, nf = n / 2 + 1;
    float *w_ref = (float*)pool_alloc((size_t)n * 4);
    float *an_buf = (float*)pool_alloc((size_t)n * 4), *sy_buf = (float*)pool_alloc((size_t)n * 4);
    float *s_time = (float*)pool_alloc((size_t)n * 4);
    Complex* s_freq = (Complex*)pool_alloc((size_t)nf * sizeof(Complex));
    float *rre = (float*)pool_alloc((size_t)nf * 4), *rim = (float*)pool_alloc((size_t)nf * 4);
    float *are = (float*)pool_alloc((size_t)nf * 4), *aim = (float*)pool_alloc((size_t)nf * 4);
    Complex *rcx = (Complex*)pool_alloc((size_t)nf * sizeof(Complex)), *acx = (Complex*)pool_alloc((size_t)nf * sizeof(Complex));
    float *in = (float*)pool_alloc((size_t)hop * 4), *ref_out = (float*)pool_alloc((size_t)hop * 4), *ac_out = (float*)pool_alloc((size_t)hop * 4);
    float *sre = (float*)pool_alloc((size_t)nf * 4), *sim = (float*)pool_alloc((size_t)nf * 4);
    Complex* scx = (Complex*)pool_alloc((size_t)nf * sizeof(Complex));
    FftHandle* fft = fft_create(n);
    AcStftConfig cfg;
    AcStft* a; AcIstft* s; void *k1, *k2;
    Diff d_an, d_syn;
    int t, kind = 0, k;
    char name[96];

    memset(&d_an, 0, sizeof(d_an)); d_syn = d_an;
    g_rng = 0x1234567089abcdefull ^ ((uint64_t)n << 8) ^ (uint64_t)signal_kind_mode;   /* reproducible per case */
    if (bc->planar_spec) dfn2_ref_make_root_hann(w_ref, n); else gtcrn_ref_make_root_hann(w_ref, n);
    memset(an_buf, 0, (size_t)n * 4); memset(sy_buf, 0, (size_t)n * 4);
    ac_stft_config_default(&cfg, n);
    cfg.spec_scale = bc->norm; cfg.ispec_scale = bc->inv_norm;
    a = make_stft(&cfg, &k1);
    s = make_istft(&cfg, &k2);
    for (t = 0; t < hops; ++t) {
        if (t % 8 == 0) {
            /* mode 0: uniform noise only; mode 1: random structured kinds (silence, signed zeros,
             * sparse, denormal, DC...); mode 2: runs of -0.0, +0.0 and mixed zeros between noise. */
            if (signal_kind_mode == 3) {
                static const int cycle[] = { KIND_SPARSE, KIND_SIGNED_ZERO, KIND_NEGZERO, KIND_SPARSE };
                kind = cycle[(t / 8) % 4];
            } else if (signal_kind_mode == 2) {
                static const int cycle[] = { KIND_NEGZERO, KIND_SILENCE, KIND_UNIFORM, KIND_SIGNED_ZERO, KIND_UNIFORM };
                kind = cycle[(t / 8) % 5];
            } else {
                kind = signal_kind_mode ? (int)(rnd32() % KIND_COUNT) : KIND_UNIFORM;
                if (signal_kind_mode && kind == KIND_SPECIAL) kind = KIND_SILENCE;   /* NaN/Inf excluded from the numeric report */
            }
        }
        gen_block(in, hop, kind);
        if (bc->planar_spec) {
            dfn2_ref_analysis(fft, an_buf, w_ref, s_time, s_freq, in, n, hop, bc->norm, rre, rim);
            ac_stft_push_planar(a, in, are, aim);
            for (k = 0; k < nf; ++k) { diff_add(&d_an, rre[k], are[k]); diff_add(&d_an, rim[k], aim[k]); }
        } else {
            gtcrn_ref_analysis(fft, an_buf, w_ref, s_time, s_freq, in, n, hop, bc->norm, (float(*)[2])rcx);
            ac_stft_push(a, in, acx);
            for (k = 0; k < nf; ++k) { diff_add(&d_an, rcx[k].r, acx[k].r); diff_add(&d_an, rcx[k].i, acx[k].i); }
        }
        /* synthesis fed with the same family of data, independent of the analysis above */
        for (k = 0; k < nf; ++k) {
            sre[k] = gen_value(kind == KIND_SILENCE ? KIND_SIGNED_ZERO : kind, k);
            sim[k] = gen_value(kind == KIND_SILENCE ? KIND_SIGNED_ZERO : kind, k + 1);
            scx[k].r = sre[k]; scx[k].i = sim[k];
        }
        if (bc->planar_spec) {
            dfn2_ref_synthesis(fft, sy_buf, w_ref, s_time, s_freq, sre, sim, n, hop, bc->inv_norm, ref_out);
            ac_istft_push_planar(s, sre, sim, ac_out);
        } else {
            gtcrn_ref_synthesis(fft, sy_buf, w_ref, s_time, s_freq, (const float(*)[2])scx, n, hop, bc->inv_norm, ref_out);
            ac_istft_push(s, scx, ac_out);
        }
        for (k = 0; k < hop; ++k) diff_add(&d_syn, ref_out[k], ac_out[k]);
    }
    snprintf(name, sizeof(name), "%s analysis spectra", label);  diff_print(name, &d_an);
    snprintf(name, sizeof(name), "%s synthesis output", label);  diff_print(name, &d_syn);
    snprintf(name, sizeof(name), "%s samples in/out", label);
    printf("B| %-34s analysis_input_samples=%llu synthesis_output_samples=%llu\n", name,
           (unsigned long long)hops * (unsigned long long)hop, (unsigned long long)hops * (unsigned long long)hop);
    expect(d_an.bit_diff == 0, "family B analysis spectra are bit-identical to the new object");
    expect(d_syn.value_diff == 0 && d_syn.nan_diff == 0, "family B synthesis differs at most in the sign of zero");
    free(w_ref); free(an_buf); free(sy_buf); free(s_time); free(s_freq); free(rre); free(rim); free(are); free(aim);
    free(rcx); free(acx); free(in); free(ref_out); free(ac_out); free(sre); free(sim); free(scx);
    fft_destroy(fft); free(k1); free(k2);
}

static void test_family_b(void) {
    BCase dfn2 = { 1024, 0.0f, 0.0f, 1 }, gtcrn = { 512, 1.0f, 1.0f, 0 };
    dfn2.norm = 1.0f / sqrtf((float)dfn2.n);
    dfn2.inv_norm = sqrtf((float)dfn2.n);
    b_run(&dfn2, 4200, 0, "dfn2 N=1024 random");
    b_run(&dfn2, 4200, 1, "dfn2 N=1024 structured");
    b_run(&dfn2, 4200, 2, "dfn2 N=1024 zero runs");
    b_run(&dfn2, 8400, 3, "dfn2 N=1024 sparse zeros");
    b_run(&gtcrn, 8200, 0, "gtcrn N=512 random");
    b_run(&gtcrn, 8200, 1, "gtcrn N=512 structured");
    b_run(&gtcrn, 8200, 2, "gtcrn N=512 zero runs");
    b_run(&gtcrn, 16400, 3, "gtcrn N=512 sparse zeros");
}

/* ----------------------------- main ----------------------------- */

int main(void) {
    static const int sizes_a[] = { 16, 64, 256, 512, 1024, 2048 };
    size_t i;
    int own;
    printf("test_stft: TEST_N=%d\n", TEST_N);
    test_windows();
    for (i = 0; i < sizeof(sizes_a) / sizeof(sizes_a[0]); ++i)
        test_family_a(sizes_a[i], sizes_a[i] >= 1024 ? 2500 : 4000);
    for (own = 0; own <= 1; ++own) {
        test_family_c_analysis(3000, 1, own);
        test_family_c_analysis(3000, 0, own);
        test_family_c_synthesis(3000, own);
    }
    test_family_c_synthesis(3000, 2);
    test_family_c_short_flush();
    test_latency();
    test_api_contract();
    test_family_b();
    if (g_fail) {
        fprintf(stderr, "test_stft FAILED (%d failing checks)\n", g_fail);
        return 1;
    }
    printf("test_stft PASS\n");
    return 0;
}
