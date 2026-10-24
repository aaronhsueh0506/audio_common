#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "srp.h"
#include "utility/mem_align.h"
#include "spatial_complex.h"
#include "spatial_simd.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Pool-first construction: the SRP struct, then every array carved from one caller-provided block with a 16-byte-aligned bump allocator (mem_align.h). */

typedef struct PoolCursor {
    uint8_t* ptr;
    size_t remaining;
} PoolCursor;

static void* pool_carve(PoolCursor* cursor, size_t count,
                        size_t element_size) {
    size_t raw;
    size_t aligned;
    void* out;
    if (!cursor || count == 0 || element_size == 0) return NULL;
    raw = ck_mul_size(count, element_size);
    aligned = ck_align16_size(raw);
    if (MEM_SIZE_INVALID(raw) || MEM_SIZE_INVALID(aligned) ||
        aligned > cursor->remaining) return NULL;
    out = cursor->ptr;
    cursor->ptr += aligned;
    cursor->remaining -= aligned;
    return out;
}

/* Config-only validation shared by srp_get_mem_size() and srp_init(): M > 1, F > 1, num_angles > 0,
 * finite positive sr/NFFT/c, finite low_freq/high_freq. */
static int srp_validate_config(const SRP_Config* cfg) {
    if (!cfg) return 0;
    if (cfg->M <= 1 || cfg->F <= 1 || cfg->num_angles <= 0) return 0;
    if (!isfinite(cfg->sr) || cfg->sr <= 0.0f) return 0;
    if (!isfinite(cfg->NFFT) || cfg->NFFT <= 0.0f) return 0;
    if (!isfinite(cfg->low_freq) || !isfinite(cfg->high_freq)) return 0;
    if (!isfinite(cfg->c) || cfg->c <= 0.0f) return 0;
    return 1;
}

/* Geometry check (geom->M == cfg->M, x/y non-NULL); only srp_init() can run it because srp_get_mem_size() takes no geometry. */
static int srp_validate_geometry(const SRP_Config* cfg,
                                 const ArrayGeometry* geom) {
    if (!cfg || !geom) return 0;
    if (geom->M != cfg->M || !geom->x || !geom->y) return 0;
    return 1;
}

size_t srp_get_mem_size(const SRP_Config* cfg) {
    size_t a, m, f;
    size_t num_pairs;
    size_t t;

    if (!srp_validate_config(cfg)) return 0;

    a = (size_t)cfg->num_angles;
    m = (size_t)cfg->M;
    f = (size_t)cfg->F;
    /* num_pairs = M*(M-1)/2 >= 1 since M > 1; computed in size_t. */
    num_pairs = ((size_t)cfg->M * (size_t)(cfg->M - 1)) / 2;

    t = 0;
    t = ck_field_size(t, 1, sizeof(SRP));

    t = ck_field_size(t, a, sizeof(float));              /* angles */

    /* a_array: angle pointer table, mic pointer table (mid + a*M), flat data (data + (a*M+m)*F). */
    t = ck_field_size(t, a, sizeof(Complex**));
    t = ck_field_size(t, ck_mul_size(a, m), sizeof(Complex*));
    t = ck_field_size(t, ck_mul_size(ck_mul_size(a, m), f), sizeof(Complex));

    t = ck_field_size(t, a, sizeof(float));              /* S_theta */
    t = ck_field_size(t, f, sizeof(int));                /* bin_best_idx */
    t = ck_field_size(t, f, sizeof(float));              /* score_scratch */
    t = ck_field_size(t, f, sizeof(float));              /* best_score */

    t = ck_field_size(t, num_pairs, sizeof(int));        /* pair_i */
    t = ck_field_size(t, num_pairs, sizeof(int));        /* pair_j */

    /* pair_steer: same 3-level layout as a_array, [num_angles][num_pairs][F]. */
    t = ck_field_size(t, a, sizeof(Complex**));
    t = ck_field_size(t, ck_mul_size(a, num_pairs), sizeof(Complex*));
    t = ck_field_size(
        t, ck_mul_size(ck_mul_size(a, num_pairs), f), sizeof(Complex));

    /* pair_phat: 2-level [num_pairs][F]; its data is written by srp() before it is read. */
    t = ck_field_size(t, num_pairs, sizeof(Complex*));
    t = ck_field_size(t, ck_mul_size(num_pairs, f), sizeof(Complex));

    if (MEM_SIZE_INVALID(t)) return 0;
    return t;
}

SRP* srp_init(void* mem, size_t mem_size,
              const SRP_Config* cfg, const ArrayGeometry* geom) {
    size_t need;
    int M, F, A;
    size_t num_pairs_sz;
    SRP* s;
    PoolCursor cursor;

    if (!mem || !cfg) return NULL;
    if (!srp_validate_config(cfg)) return NULL;
    if (!srp_validate_geometry(cfg, geom)) return NULL;
    if (!MEM_IS_ALIGNED16(mem)) return NULL;

    need = srp_get_mem_size(cfg);
    if (need == 0 || mem_size < need) return NULL;

    M = cfg->M;
    F = cfg->F;
    A = cfg->num_angles;
    num_pairs_sz = ((size_t)M * (size_t)(M - 1)) / 2;

    /* Zero-fill: every carved field starts at 0/NULL, including owned_heap (set later by srp_create_from_geometry()). */
    memset(mem, 0, need);
    s = (SRP*)mem;

    cursor.ptr = (uint8_t*)mem + ALIGN16(sizeof(SRP));
    cursor.remaining = need - ALIGN16(sizeof(SRP));

    s->M = M;
    s->F = F;
    s->num_angles = A;

    /* Float angle grid a * 2*pi / A; the multiplication order fixes the rounding of every steering phase below. */
    s->angles = (float*)pool_carve(&cursor, (size_t)A, sizeof(float));
    if (!s->angles) return NULL;
    {
        int a;
        for (a = 0; a < A; a++) {
            s->angles[a] = a * 2.0f * M_PI / A;
        }
    }

    /* a_array[a][m][f] = exp(j*phase), phase = -2*pi*f_hz*tau, tau = -(x*cos(theta) + y*sin(theta)) / c: the plane-wave delay of mic m. */
    {
        Complex*** top;
        Complex** mid;
        Complex* data;
        int a;

        top = (Complex***)pool_carve(&cursor, (size_t)A, sizeof(Complex**));
        mid = (Complex**)pool_carve(
            &cursor, (size_t)A * (size_t)M, sizeof(Complex*));
        data = (Complex*)pool_carve(
            &cursor, (size_t)A * (size_t)M * (size_t)F, sizeof(Complex));
        if (!top || !mid || !data) return NULL;

        for (a = 0; a < A; a++) {
            float theta = s->angles[a];
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            int m;

            top[a] = mid + (size_t)a * (size_t)M;

            for (m = 0; m < M; m++) {
                Complex* slot = data + ((size_t)a * (size_t)M + (size_t)m) *
                                           (size_t)F;
                float tau = -(geom->x[m] * cos_t + geom->y[m] * sin_t) /
                            cfg->c;
                int f;

                top[a][m] = slot;

                for (f = 0; f < F; f++) {
                    float freq = (float)f * cfg->sr / cfg->NFFT;
                    float phase = -2.0f * M_PI * freq * tau;
                    slot[f].r = cosf(phase);
                    slot[f].i = sinf(phase);
                }
            }
        }
        s->a_array = top;
    }

    s->S_theta = (float*)pool_carve(&cursor, (size_t)A, sizeof(float));
    s->bin_best_idx = (int*)pool_carve(&cursor, (size_t)F, sizeof(int));
    s->score_scratch = (float*)pool_carve(&cursor, (size_t)F, sizeof(float));
    s->best_score = (float*)pool_carve(&cursor, (size_t)F, sizeof(float));
    if (!s->S_theta || !s->bin_best_idx || !s->score_scratch ||
        !s->best_score) return NULL;
    {
        int f;
        for (f = 0; f < F; f++) s->bin_best_idx[f] = -1;
    }

    s->update_interval = cfg->update_interval;
    if (s->update_interval <= 0) {
        s->update_interval = 1;
    }

    s->frame_counter = 0;
    s->last_doa_raw = NAN;
    s->last_doa_s = NAN;
    s->doa_raw = NAN;
    s->doa_s = NAN;

    s->enable_smoothing = cfg->enable_smoothing;
    s->smoother.switch_consec = cfg->switch_consec;
    s->smoother.angle_tol = cfg->angle_tol;
    s->smoother.null_value = NAN;
    s->smoother.last = NAN;
    s->smoother.pending = NAN;
    s->smoother.cnt = 0;
    s->smoother.initialized = 0;

    {
        int f_start = (int)roundf(cfg->low_freq * cfg->NFFT / cfg->sr);
        int f_end = (int)roundf(cfg->high_freq * cfg->NFFT / cfg->sr);
        if (f_start < 1) f_start = 1;
        if (f_start > F - 1) f_start = F - 1;
        if (f_end > F - 1) f_end = F - 1;
        if (f_end < f_start) f_end = f_start;
        s->f_start = f_start;
        s->f_end = f_end;
    }

    s->num_pairs = 0;
    s->pair_i = NULL;
    s->pair_j = NULL;
    s->pair_steer = NULL;
    s->pair_phat = NULL;

    {
        int num_pairs = (int)num_pairs_sz;
        int p;
        Complex*** pt_top;
        Complex** pt_mid;
        Complex* pt_data;
        Complex** phat_top;
        Complex* phat_data;

        s->num_pairs = num_pairs;
        s->pair_i = (int*)pool_carve(&cursor, num_pairs_sz, sizeof(int));
        s->pair_j = (int*)pool_carve(&cursor, num_pairs_sz, sizeof(int));
        if (!s->pair_i || !s->pair_j) return NULL;

        p = 0;
        {
            int i, j;
            for (i = 0; i < M; i++) {
                for (j = i + 1; j < M; j++) {
                    s->pair_i[p] = i;
                    s->pair_j[p] = j;
                    p++;
                }
            }
        }

        pt_top = (Complex***)pool_carve(&cursor, (size_t)A, sizeof(Complex**));
        pt_mid = (Complex**)pool_carve(
            &cursor, (size_t)A * num_pairs_sz, sizeof(Complex*));
        pt_data = (Complex*)pool_carve(
            &cursor, (size_t)A * num_pairs_sz * (size_t)F, sizeof(Complex));
        if (!pt_top || !pt_mid || !pt_data) return NULL;

        {
            int a;
            for (a = 0; a < A; a++) {
                int pp;
                pt_top[a] = pt_mid + (size_t)a * num_pairs_sz;
                for (pp = 0; pp < num_pairs; pp++) {
                    Complex* slot =
                        pt_data +
                        ((size_t)a * num_pairs_sz + (size_t)pp) * (size_t)F;
                    int i = s->pair_i[pp];
                    int j = s->pair_j[pp];
                    int f;

                    pt_top[a][pp] = slot;

                    for (f = 0; f < F; f++) {
                        /* pair_steer[a][p][f] = conj(w_i) * w_j, built once so scoring is one product per bin. */
                        slot[f] = spatial_complex_mul(
                            spatial_complex_conj(s->a_array[a][i][f]),
                            s->a_array[a][j][f]);
                    }
                }
            }
        }
        s->pair_steer = pt_top;

        phat_top = (Complex**)pool_carve(&cursor, num_pairs_sz, sizeof(Complex*));
        phat_data = (Complex*)pool_carve(
            &cursor, num_pairs_sz * (size_t)F, sizeof(Complex));
        if (!phat_top || !phat_data) return NULL;
        for (p = 0; p < num_pairs; p++) {
            phat_top[p] = phat_data + (size_t)p * (size_t)F;
            /* pair_phat data is one-frame scratch, read only within [f_start, f_end] after srp() writes it. */
        }
        s->pair_phat = phat_top;
    }

    if (cursor.remaining != 0) return NULL;

    return s;
}

SRP* srp_create_from_geometry(
    const SRP_Config* cfg,
    const ArrayGeometry* geom
)
{
    size_t need;
    void* pool = NULL;
    SRP* s;

    need = srp_get_mem_size(cfg);
    if (need == 0) return NULL;
    if (posix_memalign(&pool, 16, need) != 0 || !pool) return NULL;

    s = srp_init(pool, need, cfg, geom);
    if (!s) {
        free(pool);
        return NULL;
    }
    s->owned_heap = pool;
    return s;
}

void srp(SRP* s, const Complex* const* X, const int* mask)
{
    int F = s->F;
    int A = s->num_angles;

    for (int a = 0; a < A; a++) {
        s->S_theta[a] = 0.0f;
    }

    for (int f = 0; f < F; f++) {
        s->bin_best_idx[f] = -1;
        s->best_score[f] = -1e30f;
    }

    /* The PHAT cross spectrum does not depend on the candidate angle, so it is computed once per frame,
     * and only over [f_start, f_end], the only range the scoring loop reads. */
    for (int p = 0; p < s->num_pairs; ++p) {
        spatial_phat_cross(
            X[s->pair_i[p]] + s->f_start, X[s->pair_j[p]] + s->f_start,
            s->pair_phat[p] + s->f_start, s->f_end - s->f_start + 1);
    }

    /* Candidate-major traversal exposes contiguous bins to NEON; accumulation order is ascending in f, then p/a. */
    const int band = s->f_end - s->f_start + 1;
    for (int a = 0; a < A; ++a) {
        /* score_scratch is public: the first angle zeroes all F bins; later angles rewrite only the searched band.
         * An empty band clears fully on every angle. */
        if (a == 0 || band <= 0)
            memset(s->score_scratch, 0, (size_t)F * sizeof(float));
        else
            memset(s->score_scratch + s->f_start, 0,
                   (size_t)band * sizeof(float));
        for (int p = 0; p < s->num_pairs; ++p) {
            spatial_pair_score_accumulate(
                s->pair_phat[p] + s->f_start,
                s->pair_steer[a][p] + s->f_start,
                s->score_scratch + s->f_start,
                band);
        }
        for (int f = s->f_start; f <= s->f_end; ++f) {
            float score;
            if (mask && !mask[f]) continue;
            score = s->score_scratch[f];
            s->S_theta[a] += score;
            if (score > s->best_score[f]) {
                s->best_score[f] = score;
                s->bin_best_idx[f] = a;
            }
        }
    }
}

float srp2doa(SRP* s)
{
    int best = 0;
    float max_val = s->S_theta[0];

    for (int i = 1; i < s->num_angles; i++) {
        if (s->S_theta[i] > max_val) {
            max_val = s->S_theta[i];
            best = i;
        }
    }

    return s->angles[best];
}

void doa_step(SRP* s,
              const Complex* const* X,
              const int* mask,
              int vad_raw,
              int vad_out)
{
    if (!s || !X) return;

    int run_srp = 0;

    /* SRP runs only when raw VAD reports speech and the frame matches update_interval; other frames reuse the last DOA. */
    if (vad_raw) {
        if ((s->frame_counter % s->update_interval) == 0) {
            run_srp = 1;
        }
    }

    if (run_srp) {

        srp(s, X, mask);

        /* doa_raw: fresh SRP-PHAT result, neither smoothed nor held. */
        s->doa_raw = srp2doa(s);
        s->last_doa_raw = s->doa_raw;

        /* doa_s: pending-smoothed DOA with hold; the stable DOA for downstream modules (BF, post-gain). */
        float doa_s_new;

        if (s->enable_smoothing) {
            doa_s_new = doa_smoother_update(&s->smoother,
                                            s->doa_raw,
                                            vad_out);
        } else {
            doa_s_new = s->doa_raw;
        }

        if (!isnan(doa_s_new)) {
            s->doa_s = doa_s_new;
            s->last_doa_s = doa_s_new;
        } else {
            /* If the smoother has no valid output, keep the last valid doa_s so it is never overwritten by NAN. */
            s->doa_s = s->last_doa_s;
        }

    } else {

        /* No SRP update this frame. Do not call doa_smoother_update() with NAN here: it can reset the smoother's pending state. */
        s->doa_raw = NAN;
        s->doa_s   = s->last_doa_s;
    }

    s->frame_counter++;
}

float doa_get_raw(const SRP* s)
{
    return s ? s->doa_raw : NAN;
}

float doa_get_smooth(const SRP* s)
{
    return s ? s->doa_s : NAN;
}

void srp_reset(SRP* s)
{
    if (!s) return;
    s->frame_counter = 0;
    s->last_doa_raw = NAN;
    s->last_doa_s = NAN;
    s->doa_raw = NAN;
    s->doa_s = NAN;
    s->smoother.last = NAN;
    s->smoother.pending = NAN;
    s->smoother.cnt = 0;
    s->smoother.initialized = 0;
    if (s->S_theta) {
        memset(s->S_theta, 0, (size_t)s->num_angles * sizeof(float));
    }
    if (s->bin_best_idx) {
        for (int f = 0; f < s->F; ++f) s->bin_best_idx[f] = -1;
    }
}

void srp_destroy(SRP* s)
{
    if (!s) return;

    /* Pool path (owned_heap == NULL): nothing to free. Heap path: one free() releases the whole block. */
    if (s->owned_heap) free(s->owned_heap);
}
