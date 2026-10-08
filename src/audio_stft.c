#include "audio_stft.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "mem_align.h"
#include "simd_kernel_nn.h"
#include "simd_kernels.h"

#define AC_PI_F 3.14159265358979323846f
#define AC_FFT_MIN 16
#define AC_FFT_MAX 8192
#define AC_ENV_FLOOR 1e-11f

struct AcStft {
    FftHandle*   fft;
    const float* window;
    int          n;
    int          hop;
    int          n_freqs;
    int          center;
    float        scale;
    long         hops_seen;   /* saturates at 3 */
    float*       history;     /* last n samples */
    float*       seg;         /* windowed frame; clobbered by the FFT */
    Complex*     spec;        /* FFT output staging for split outputs */
};

struct AcIstft {
    FftHandle*   fft;
    const float* window;
    int          n;
    int          hop;
    int          n_freqs;
    AcOlaMode    ola;
    float        scale;
    long         frames_seen; /* saturates at 2 */
    float*       acc;         /* overlap-add accumulator; origin = next output */
    float*       env;         /* squared-window accumulator (AC_OLA_ENV_NORM)  */
    float*       time;        /* inverse FFT output */
    Complex*     spec;        /* inverse FFT input staging */
};

typedef struct Layout {
    size_t fft_bytes;    /* 0 when borrowed */
    size_t window_bytes; /* 0 when borrowed */
    size_t total;
} Layout;

static int config_valid(const AcStftConfig* cfg) {
    int n;
    if (!cfg) return 0;
    n = cfg->fft_size;
    if (n < AC_FFT_MIN || n > AC_FFT_MAX || (n & (n - 1)) != 0) return 0;
    if (cfg->center != 0 && cfg->center != 1) return 0;
    if (cfg->ola != AC_OLA_ACCUMULATE && cfg->ola != AC_OLA_ENV_NORM) return 0;
    if (cfg->fft && fft_get_n_freqs(cfg->fft) != n / 2 + 1) return 0;
    return 1;
}

/* Field walk shared by get_mem_size and init: `arrays` is the number of
 * float[n] regions; the staging bins follow them. */
static int layout_walk(const AcStftConfig* cfg, size_t struct_bytes,
                       int arrays, Layout* lay) {
    size_t n = (size_t)cfg->fft_size;
    size_t total = ck_field_size(0, 1, struct_bytes);
    lay->fft_bytes = 0;
    lay->window_bytes = 0;
    if (!cfg->fft) {
        lay->fft_bytes = fft_get_mem_size(cfg->fft_size);
        if (lay->fft_bytes == 0) return 0;
        total = ck_field_size(total, lay->fft_bytes, 1);
    }
    if (!cfg->window) {
        lay->window_bytes = n * sizeof(float);
        total = ck_field_size(total, n, sizeof(float));
    }
    total = ck_field_size(total, n * (size_t)arrays, sizeof(float));
    total = ck_field_size(total, n / 2 + 1, sizeof(Complex));
    if (MEM_SIZE_INVALID(total)) return 0;
    lay->total = total;
    return 1;
}

typedef struct Cursor {
    uint8_t* ptr;
} Cursor;

static void* take(Cursor* c, size_t bytes) {
    void* r = c->ptr;
    c->ptr += ALIGN16(bytes);
    return r;
}

/* Carve the borrowed-or-own FFT handle and window; both walks agree with
 * layout_walk by construction. Returns 0 if the own FFT cannot be built. */
static int carve_tables(Cursor* c, const AcStftConfig* cfg, const Layout* lay,
                        FftHandle** fft, const float** window) {
    if (cfg->fft) {
        *fft = cfg->fft;
    } else {
        void* region = take(c, lay->fft_bytes);
        *fft = fft_init(region, lay->fft_bytes, cfg->fft_size);
        if (!*fft) return 0;
    }
    if (cfg->window) {
        *window = cfg->window;
    } else {
        float* w = (float*)take(c, lay->window_bytes);
        ac_window_build(w, cfg->fft_size);
        *window = w;
    }
    return 1;
}

void ac_stft_config_default(AcStftConfig* cfg, int fft_size) {
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->fft_size = fft_size;
    cfg->ola = AC_OLA_ACCUMULATE;
    cfg->spec_scale = 1.0f;
    cfg->ispec_scale = 1.0f;
}

void ac_window_build(float* w, int n) {
    int i;
    for (i = 0; i < n; ++i)
        w[i] = sqrtf(0.5f * (1.0f - cosf(2.0f * AC_PI_F * (float)i / (float)n)));
}

int ac_stft_latency_samples(const AcStftConfig* cfg) {
    return config_valid(cfg) ? cfg->fft_size / 2 : -1;
}

/* ------------------------------ analysis ------------------------------ */

size_t ac_stft_get_mem_size(const AcStftConfig* cfg) {
    Layout lay;
    if (!config_valid(cfg) || !layout_walk(cfg, sizeof(AcStft), 2, &lay)) return 0;
    return lay.total;
}

AcStft* ac_stft_init(void* mem, size_t mem_size, const AcStftConfig* cfg) {
    Layout lay;
    Cursor c;
    AcStft* s;
    size_t n;
    if (!mem || !MEM_IS_ALIGNED16(mem) || !config_valid(cfg) ||
        !layout_walk(cfg, sizeof(AcStft), 2, &lay) || mem_size < lay.total)
        return NULL;
    n = (size_t)cfg->fft_size;
    c.ptr = (uint8_t*)mem;
    s = (AcStft*)take(&c, sizeof(AcStft));
    memset(s, 0, sizeof(*s));
    if (!carve_tables(&c, cfg, &lay, &s->fft, &s->window)) return NULL;
    s->history = (float*)take(&c, n * sizeof(float));
    s->seg     = (float*)take(&c, n * sizeof(float));
    s->spec    = (Complex*)take(&c, (n / 2 + 1) * sizeof(Complex));
    s->n       = cfg->fft_size;
    s->hop     = cfg->fft_size / 2;
    s->n_freqs = cfg->fft_size / 2 + 1;
    s->center  = cfg->center;
    s->scale   = cfg->spec_scale;
    ac_stft_reset(s);
    return s;
}

void ac_stft_reset(AcStft* s) {
    if (!s) return;
    s->hops_seen = 0;
    memset(s->history, 0, (size_t)s->n * sizeof(float));
    memset(s->seg, 0, (size_t)s->n * sizeof(float));
    memset(s->spec, 0, (size_t)s->n_freqs * sizeof(Complex));
}

const float* ac_stft_window(const AcStft* s) { return s ? s->window : NULL; }

/* Window the segment into s->seg (the segment may be s->seg itself) and run
 * the forward FFT into `out`. */
static void transform(AcStft* s, const float* segment, Complex* out) {
    skn_mul_f32(s->seg, segment, s->window, (size_t)s->n);
    fft_forward_scratch(s->fft, s->seg, out);
}

static void frame_interleaved(AcStft* s, const float* segment, Complex* out) {
    int k;
    transform(s, segment, out);
    if (s->scale != 1.0f) {
        for (k = 0; k < s->n_freqs; ++k) {
            out[k].r = out[k].r * s->scale;
            out[k].i = out[k].i * s->scale;
        }
    }
}

static void frame_planar(AcStft* s, const float* segment, float* re, float* im) {
    transform(s, segment, s->spec);
    if (s->scale != 1.0f)
        skn_deinterleave_scale_cf32(s->spec, re, im, (size_t)s->n_freqs, s->scale);
    else
        skn_deinterleave_cf32(s->spec, re, im, (size_t)s->n_freqs);
}

void ac_stft_frame(AcStft* s, const float* frame, Complex* out) {
    if (!s || !frame || !out) return;
    frame_interleaved(s, frame, out);
}

void ac_stft_frame_planar(AcStft* s, const float* frame, float* re, float* im) {
    if (!s || !frame || !re || !im) return;
    frame_planar(s, frame, re, im);
}

static void history_append(AcStft* s, const float* hop_in) {
    memmove(s->history, s->history + s->hop,
            (size_t)(s->n - s->hop) * sizeof(float));
    memcpy(s->history + s->n - s->hop, hop_in, (size_t)s->hop * sizeof(float));
}

/* Reflect prefix of the first centered frame: x[hop..1] followed by x[0..hop). */
static void build_head_segment(AcStft* s) {
    int i;
    for (i = 0; i < s->hop; ++i) s->seg[i] = s->history[s->hop - i];
    memcpy(s->seg + s->hop, s->history, (size_t)s->hop * sizeof(float));
}

/* Reflect suffix after the last sample: x[L-hop..L) then x[L-2], x[L-3], ... */
static void build_tail_segment(AcStft* s) {
    int i;
    memcpy(s->seg, s->history + s->n - s->hop, (size_t)s->hop * sizeof(float));
    for (i = 0; i < s->hop; ++i) s->seg[s->hop + i] = s->history[s->n - 2 - i];
}

int ac_stft_push(AcStft* s, const float* hop_in, Complex* out) {
    if (!s || !hop_in || !out) return 0;
    history_append(s, hop_in);
    if (!s->center) {
        frame_interleaved(s, s->history, out);
        return 1;
    }
    if (s->hops_seen < 3) s->hops_seen++;
    if (s->hops_seen == 1) return 0;
    if (s->hops_seen == 2) {
        build_head_segment(s);
        frame_interleaved(s, s->seg, out);
        frame_interleaved(s, s->history, out + s->n_freqs);
        return 2;
    }
    frame_interleaved(s, s->history, out);
    return 1;
}

int ac_stft_push_planar(AcStft* s, const float* hop_in, float* re, float* im) {
    if (!s || !hop_in || !re || !im) return 0;
    history_append(s, hop_in);
    if (!s->center) {
        frame_planar(s, s->history, re, im);
        return 1;
    }
    if (s->hops_seen < 3) s->hops_seen++;
    if (s->hops_seen == 1) return 0;
    if (s->hops_seen == 2) {
        build_head_segment(s);
        frame_planar(s, s->seg, re, im);
        frame_planar(s, s->history, re + s->n_freqs, im + s->n_freqs);
        return 2;
    }
    frame_planar(s, s->history, re, im);
    return 1;
}

int ac_stft_flush(AcStft* s, Complex* out) {
    if (!s || !out || !s->center || s->hops_seen < 2) return 0;
    build_tail_segment(s);
    frame_interleaved(s, s->seg, out);
    return 1;
}

int ac_stft_flush_planar(AcStft* s, float* re, float* im) {
    if (!s || !re || !im || !s->center || s->hops_seen < 2) return 0;
    build_tail_segment(s);
    frame_planar(s, s->seg, re, im);
    return 1;
}

/* ----------------------------- synthesis ------------------------------ */

static int istft_arrays(const AcStftConfig* cfg) {
    return cfg->ola == AC_OLA_ENV_NORM ? 3 : 2;
}

size_t ac_istft_get_mem_size(const AcStftConfig* cfg) {
    Layout lay;
    if (!config_valid(cfg) ||
        !layout_walk(cfg, sizeof(AcIstft), istft_arrays(cfg), &lay))
        return 0;
    return lay.total;
}

AcIstft* ac_istft_init(void* mem, size_t mem_size, const AcStftConfig* cfg) {
    Layout lay;
    Cursor c;
    AcIstft* s;
    size_t n;
    if (!mem || !MEM_IS_ALIGNED16(mem) || !config_valid(cfg) ||
        !layout_walk(cfg, sizeof(AcIstft), istft_arrays(cfg), &lay) ||
        mem_size < lay.total)
        return NULL;
    n = (size_t)cfg->fft_size;
    c.ptr = (uint8_t*)mem;
    s = (AcIstft*)take(&c, sizeof(AcIstft));
    memset(s, 0, sizeof(*s));
    if (!carve_tables(&c, cfg, &lay, &s->fft, &s->window)) return NULL;
    s->acc = (float*)take(&c, n * sizeof(float));
    if (cfg->ola == AC_OLA_ENV_NORM) s->env = (float*)take(&c, n * sizeof(float));
    s->time = (float*)take(&c, n * sizeof(float));
    s->spec = (Complex*)take(&c, (n / 2 + 1) * sizeof(Complex));
    s->n       = cfg->fft_size;
    s->hop     = cfg->fft_size / 2;
    s->n_freqs = cfg->fft_size / 2 + 1;
    s->ola     = cfg->ola;
    s->scale   = cfg->ispec_scale;
    ac_istft_reset(s);
    return s;
}

void ac_istft_reset(AcIstft* s) {
    if (!s) return;
    s->frames_seen = 0;
    memset(s->acc, 0, (size_t)s->n * sizeof(float));
    if (s->env) memset(s->env, 0, (size_t)s->n * sizeof(float));
    memset(s->time, 0, (size_t)s->n * sizeof(float));
    memset(s->spec, 0, (size_t)s->n_freqs * sizeof(Complex));
}

const float* ac_istft_window(const AcIstft* s) { return s ? s->window : NULL; }

/* Window + overlap-add of s->time, emit, shift. Returns samples written. */
static int overlap_add(AcIstft* s, float* out) {
    const size_t keep = (size_t)(s->n - s->hop) * sizeof(float);
    const size_t step = (size_t)s->hop * sizeof(float);
    int emitted = s->hop;
    sk_wola_accumulate_f32(s->acc, s->time, s->window, s->n);
    if (s->ola == AC_OLA_ACCUMULATE) {
        memcpy(out, s->acc, step);
        memmove(s->acc, s->acc + s->hop, keep);
        memset(s->acc + (s->n - s->hop), 0, step);
        return emitted;
    }
    sk_wola_accumulate_f32(s->env, s->window, s->window, s->n);
    if (s->frames_seen < 2) s->frames_seen++;
    if (s->frames_seen > 1)
        skn_div_floor_f32(out, s->acc, s->env, (size_t)s->hop, AC_ENV_FLOOR);
    else
        emitted = 0;
    memmove(s->acc, s->acc + s->hop, keep);
    memset(s->acc + (s->n - s->hop), 0, step);
    memmove(s->env, s->env + s->hop, keep);
    memset(s->env + (s->n - s->hop), 0, step);
    return emitted;
}

int ac_istft_push(AcIstft* s, const Complex* spec, float* out) {
    int k;
    if (!s || !spec || !out) return 0;
    if (s->scale == 1.0f) {
        fft_inverse(s->fft, spec, s->time);
    } else {
        for (k = 0; k < s->n_freqs; ++k) {
            s->spec[k].r = spec[k].r * s->scale;
            s->spec[k].i = spec[k].i * s->scale;
        }
        fft_inverse_scratch(s->fft, s->spec, s->time);
    }
    return overlap_add(s, out);
}

int ac_istft_push_planar(AcIstft* s, const float* re, const float* im, float* out) {
    if (!s || !re || !im || !out) return 0;
    if (s->scale != 1.0f)
        skn_interleave_scale_cf32(re, im, s->spec, (size_t)s->n_freqs, s->scale);
    else
        skn_interleave_cf32(re, im, s->spec, (size_t)s->n_freqs);
    fft_inverse_scratch(s->fft, s->spec, s->time);
    return overlap_add(s, out);
}

int ac_istft_flush(AcIstft* s, float* out) {
    if (!s || !out || s->ola != AC_OLA_ENV_NORM || s->frames_seen == 0) return 0;
    skn_div_floor_f32(out, s->acc, s->env, (size_t)s->hop, AC_ENV_FLOOR);
    return s->hop;
}
