#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "bf.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"
#include "spatial_simd.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* RLS numerical-hardening constants.
 * BF_WA_LEAK: wa = BF_WA_LEAK * wa + update. Its time constant, 1/(1-BF_WA_LEAK) updates (not hops), is far slower
 * than the RLS forgetting time, so it only bounds wa growth over long continuous runs. The leak is applied only
 * inside the adapt block (allow_adapt_in, adapt_interval and the per-bin mask freeze all gate it). */
#define BF_WA_LEAK 0.99999f

/* Per-bin RLS divergence guards (a whole-BF reset would discard every other bin's adaptation):
 * 1. BF_P_DIAG_FLOOR/CEIL clamp each bin's P diagonal after every update; CEIL is far above any converged value.
 * 2. If isfinite(upu_real) && upu_real > 0 fails before an update, that bin's P is reset to identity, wa to zero,
 *    and its update is skipped for the frame.
 * Without them, P = (P - gain*q) / lambda can grow to inf under sustained low excitation. */
#define BF_P_DIAG_FLOOR 1e-6f
#define BF_P_DIAG_CEIL  1e6f

int bf_effective_adapt_interval(
    int enable_fix_mode, int fixed_align_notebook, int adapt_interval)
{
    int effective = adapt_interval > 0 ? adapt_interval : 1;

    /* Fixed notebook-alignment mode updates every frame. */
    if (enable_fix_mode && fixed_align_notebook) {
        effective = 1;
    }
    return effective;
}

/* Total bytes for shape (M, F); independent of num_angles and cfg. The carve order must match bf_init(). */
size_t bf_get_mem_size(int M, int F) {
    size_t total;
    size_t fM;
    size_t mM;

    if (M <= 0 || F <= 0) return 0;

    total = ck_align16_size(sizeof(BF));

    /* P (M, M, F): frequency is the contiguous axis, so one NEON vector spans four bins. */
    total = ck_field_size(total, (size_t)M, sizeof(Complex**));
    fM = ck_mul_size((size_t)F, (size_t)M);
    mM = ck_mul_size((size_t)M, (size_t)M);
    total = ck_field_size(total, mM, sizeof(Complex*));
    total = ck_field_size(total, ck_mul_size(fM, (size_t)M), sizeof(Complex));

    total = ck_field_size(total, (size_t)M, sizeof(Complex*));
    total = ck_field_size(
        total, ck_mul_size((size_t)M, (size_t)F), sizeof(Complex));

    {
        /* (size_t)M + 3 widens before the add, so a large M cannot overflow int. */
        size_t scratch_count = ck_mul_size((size_t)M + 3, (size_t)F);
        total = ck_field_size(total, scratch_count, sizeof(Complex));
    }

    return MEM_SIZE_INVALID(total) ? 0 : total;
}

/* Caller-pool construction with bf_create()'s validation. `a_array` is borrowed: owned by the caller, never carved or freed here. */
BF* bf_init(void* mem, size_t mem_size, int M, int F,
              int num_angles, Complex*** a_array,
              const BF_Config* cfg) {
    BF* g;
    AcPoolCursor cursor;
    size_t need;
    size_t fM;
    size_t mM;

    if (!mem || !cfg || !a_array || M <= 0 || F <= 0 || num_angles <= 0 ||
        !isfinite(cfg->lambda) || cfg->lambda <= 0.0f ||
        cfg->lambda > 1.0f ||
        !isfinite(cfg->mu) ||
        (cfg->enable_fix_mode && !isfinite(cfg->fixed_doa_rad))) {
        return NULL;
    }

    need = bf_get_mem_size(M, F);
    if (need == 0 || !MEM_IS_ALIGNED16(mem) || mem_size < need) return NULL;

    /* Only the budgeted region is touched; trailing bytes of a larger pool are left alone. */
    memset(mem, 0, need);
    g = (BF*)mem;

    cursor.ptr = (uint8_t*)mem + ck_align16_size(sizeof(BF));
    cursor.remaining = need - ck_align16_size(sizeof(BF));

    g->enable = cfg->enable;
    g->M = M;
    g->F = F;
    g->num_angles = num_angles;
    g->a_array = a_array;
    g->lambda = cfg->lambda;
    g->mu = cfg->mu;
    g->enable_fix_mode = cfg->enable_fix_mode;
    g->fixed_doa_rad = cfg->fixed_doa_rad;
    g->fixed_align_notebook = cfg->fixed_align_notebook;

    g->adapt_interval = bf_effective_adapt_interval(
        g->enable_fix_mode, g->fixed_align_notebook, cfg->adapt_interval);

    fM = ck_mul_size((size_t)F, (size_t)M);
    mM = ck_mul_size((size_t)M, (size_t)M);
    if (MEM_SIZE_INVALID(fM) || MEM_SIZE_INVALID(mM)) return NULL;

    /* P: pointer tables over flat frequency planes (identity fill and state reset: bf_reset below). */
    {
        Complex*** p_top = (Complex***)ac_pool_carve(
            &cursor, (size_t)M, sizeof(Complex**));
        Complex** p_mid = (Complex**)ac_pool_carve(
            &cursor, mM, sizeof(Complex*));
        Complex* p_data = (Complex*)ac_pool_carve(
            &cursor, ck_mul_size(fM, (size_t)M), sizeof(Complex));
        if (!p_top || !p_mid || !p_data) return NULL;

        g->P = p_top;
        for (int i = 0; i < M; i++) {
            g->P[i] = p_mid + (size_t)i * (size_t)M;
            for (int j = 0; j < M; j++) {
                g->P[i][j] =
                    p_data + ((size_t)i * (size_t)M + (size_t)j) * (size_t)F;
            }
        }
    }

    {
        Complex** wa_top = (Complex**)ac_pool_carve(
            &cursor, (size_t)M, sizeof(Complex*));
        Complex* wa_data = (Complex*)ac_pool_carve(
            &cursor, fM, sizeof(Complex));
        if (!wa_top || !wa_data) return NULL;

        g->wa = wa_top;
        for (int m = 0; m < M; m++) {
            g->wa[m] = wa_data + (size_t)m * (size_t)F;
        }
    }

    {
        size_t count = ck_mul_size((size_t)M + 3, (size_t)F);
        g->scratch = (Complex*)ac_pool_carve(&cursor, count, sizeof(Complex));
        if (!g->scratch) return NULL;
        g->scratch_das = g->scratch;
        g->scratch_wu = g->scratch_das + F;
        g->scratch_spec = g->scratch_wu + F;
        g->scratch_u = g->scratch_spec + F;
    }

    /* The carves must consume exactly `need` bytes. */
    if (cursor.remaining != 0) return NULL;

    bf_reset(g);
    return g;
}

BF* bf_create(int M, int F, int num_angles,
                Complex*** a_array,
                const BF_Config* cfg)
{
    size_t need;
    void* pool = NULL;
    BF* g;

    need = bf_get_mem_size(M, F);
    if (need == 0) return NULL;

    if (posix_memalign(&pool, 16, need) != 0 || !pool) return NULL;

    g = bf_init(pool, need, M, F, num_angles, a_array, cfg);
    if (!g) {
        free(pool);
        return NULL;
    }
    g->owned_heap = pool;
    return g;
}

/* Reset one bin's RLS state to the fresh-start condition (P = identity, wa = 0). */
static void bf_reset_bin(BF* g, int f)
{
    for (int i = 0; i < g->M; i++) {
        for (int j = 0; j < g->M; j++) {
            g->P[i][j][f].r = (i == j) ? 1.0f : 0.0f;
            g->P[i][j][f].i = 0.0f;
        }
        g->wa[i][f].r = 0.0f;
        g->wa[i][f].i = 0.0f;
    }
    g->bin_resets += 1;
}

/* One-bin scalar RLS recurrence; the four-bin NEON path below applies the same state transition per lane. */
static void bf_rls_update_bin_scalar(
    BF* g,
    int f,
    const Complex* u_flat,
    const Complex* bf_spec,
    int use_notebook_update,
    int use_mu_scaling)
{
    Complex pu[g->M];
    Complex gain[g->M];
    Complex q[g->M];
    const Complex (*u)[g->F] =
        (const Complex (*)[g->F])u_flat;

    for (int i = 0; i < g->M; i++) {
        pu[i].r = 0.0f;
        pu[i].i = 0.0f;
        for (int k = 0; k < g->M; k++) {
            Complex term = spatial_complex_mul(g->P[i][k][f], u[k][f]);
            pu[i] = spatial_complex_add(pu[i], term);
        }
    }

    {
        Complex upu_c;
        float upu_real;
        float inv_upu;
        float inv_lambda;

        upu_c.r = g->lambda;
        upu_c.i = 0.0f;
        for (int i = 0; i < g->M; i++) {
            Complex term = spatial_complex_mul(
                spatial_complex_conj(u[i][f]), pu[i]);
            upu_c = spatial_complex_add(upu_c, term);
        }
        if (use_notebook_update) upu_c.r += g->lambda;
        upu_real = upu_c.r;

        if (!isfinite(upu_real) || upu_real <= 0.0f) {
            bf_reset_bin(g, f);
            return;
        }
        if (fabsf(upu_real) < 1e-12f) {
            upu_real = (upu_real >= 0.0f) ? 1e-12f : -1e-12f;
        }
        inv_upu = 1.0f / upu_real;
        inv_lambda = 1.0f / g->lambda;

        for (int i = 0; i < g->M; i++) {
            gain[i].r = pu[i].r * inv_upu;
            gain[i].i = pu[i].i * inv_upu;
        }

        for (int j = 0; j < g->M; j++) {
            q[j].r = 0.0f;
            q[j].i = 0.0f;
            for (int k = 0; k < g->M; k++) {
                Complex term = spatial_complex_mul(
                    spatial_complex_conj(u[k][f]), g->P[k][j][f]);
                q[j] = spatial_complex_add(q[j], term);
            }
        }

        for (int i = 0; i < g->M; i++) {
            for (int j = 0; j < g->M; j++) {
                Complex term = spatial_complex_mul(gain[i], q[j]);
                g->P[i][j][f].r =
                    (g->P[i][j][f].r - term.r) * inv_lambda;
                g->P[i][j][f].i =
                    (g->P[i][j][f].i - term.i) * inv_lambda;
            }
        }
    }

    for (int i = 0; i < g->M; i++) {
        for (int j = i + 1; j < g->M; j++) {
            Complex avg;
            avg.r = 0.5f * (g->P[i][j][f].r + g->P[j][i][f].r);
            avg.i = 0.5f * (g->P[i][j][f].i - g->P[j][i][f].i);
            g->P[i][j][f] = avg;
            g->P[j][i][f] = spatial_complex_conj(avg);
        }
        g->P[i][i][f].i = 0.0f;
    }

    for (int i = 0; i < g->M; i++) {
        if (g->P[i][i][f].r < BF_P_DIAG_FLOOR) {
            g->P[i][i][f].r = BF_P_DIAG_FLOOR;
        } else if (g->P[i][i][f].r > BF_P_DIAG_CEIL) {
            g->P[i][i][f].r = BF_P_DIAG_CEIL;
        }
    }

    {
        Complex bf_conj = spatial_complex_conj(bf_spec[f]);
        for (int m = 0; m < g->M; m++) {
            Complex update = spatial_complex_mul(gain[m], bf_conj);
            if (use_mu_scaling) {
                update.r *= g->mu;
                update.i *= g->mu;
            }
            g->wa[m][f] = spatial_complex_add(
                spatial_complex_scale(g->wa[m][f], BF_WA_LEAK), update);
        }
    }

    for (int m = 0; m < g->M; m++) {
        if (!isfinite(g->wa[m][f].r) || !isfinite(g->wa[m][f].i)) {
            bf_reset_bin(g, f);
            break;
        }
    }
}

#if SK_HAVE_NEON
/* Four independent bins, one per NEON lane, same accumulation order as the scalar helper;
 * no horizontal reduction, reciprocal estimate or FMA. */
static void bf_rls_update_four_neon(
    BF* g,
    int f,
    const Complex* u_flat,
    const Complex* bf_spec,
    int use_notebook_update,
    int use_mu_scaling)
{
    const Complex (*u)[g->F] =
        (const Complex (*)[g->F])u_flat;
    float32x4_t pu_r[4];
    float32x4_t pu_i[4];
    float32x4_t gain_r[4];
    float32x4_t gain_i[4];
    float32x4_t q_r[4];
    float32x4_t q_i[4];
    float32x4_t upu_r = vdupq_n_f32(g->lambda);
    float32x4_t upu_i = vdupq_n_f32(0.0f);
    float upu_lane[4];
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t inv_lambda =
        vdupq_n_f32(1.0f / g->lambda);

    for (int i = 0; i < 4; ++i) {
        pu_r[i] = zero;
        pu_i[i] = zero;
        for (int k = 0; k < 4; ++k) {
            float32x4x2_t p = sk__cquad_load(g->P[i][k] + f);
            float32x4x2_t uv = sk__cquad_load(u[k] + f);
            float32x4_t tr = vsubq_f32(
                vmulq_f32(p.val[0], uv.val[0]),
                vmulq_f32(p.val[1], uv.val[1]));
            float32x4_t ti = vaddq_f32(
                vmulq_f32(p.val[0], uv.val[1]),
                vmulq_f32(p.val[1], uv.val[0]));
            pu_r[i] = vaddq_f32(pu_r[i], tr);
            pu_i[i] = vaddq_f32(pu_i[i], ti);
        }
    }

    for (int i = 0; i < 4; ++i) {
        float32x4x2_t uv = sk__cquad_load(u[i] + f);
        float32x4_t tr = vaddq_f32(
            vmulq_f32(uv.val[0], pu_r[i]),
            vmulq_f32(uv.val[1], pu_i[i]));
        float32x4_t ti = vsubq_f32(
            vmulq_f32(uv.val[0], pu_i[i]),
            vmulq_f32(uv.val[1], pu_r[i]));
        upu_r = vaddq_f32(upu_r, tr);
        upu_i = vaddq_f32(upu_i, ti);
    }
    if (use_notebook_update) {
        upu_r = vaddq_f32(upu_r, vdupq_n_f32(g->lambda));
    }

    /* Any exceptional lane falls back to the scalar helper before any store. */
    vst1q_f32(upu_lane, upu_r);
    for (int lane = 0; lane < 4; ++lane) {
        if (!isfinite(upu_lane[lane]) || upu_lane[lane] <= 0.0f ||
            fabsf(upu_lane[lane]) < 1e-12f) {
            for (int n = 0; n < 4; ++n) {
                bf_rls_update_bin_scalar(
                    g, f + n, u_flat, bf_spec,
                    use_notebook_update, use_mu_scaling);
            }
            return;
        }
    }

    {
        float32x4_t inv_upu = vdivq_f32(vdupq_n_f32(1.0f), upu_r);
        for (int i = 0; i < 4; ++i) {
            gain_r[i] = vmulq_f32(pu_r[i], inv_upu);
            gain_i[i] = vmulq_f32(pu_i[i], inv_upu);
        }
    }

    for (int j = 0; j < 4; ++j) {
        q_r[j] = zero;
        q_i[j] = zero;
        for (int k = 0; k < 4; ++k) {
            float32x4x2_t uv = sk__cquad_load(u[k] + f);
            float32x4x2_t p = sk__cquad_load(g->P[k][j] + f);
            float32x4_t tr = vaddq_f32(
                vmulq_f32(uv.val[0], p.val[0]),
                vmulq_f32(uv.val[1], p.val[1]));
            float32x4_t ti = vsubq_f32(
                vmulq_f32(uv.val[0], p.val[1]),
                vmulq_f32(uv.val[1], p.val[0]));
            q_r[j] = vaddq_f32(q_r[j], tr);
            q_i[j] = vaddq_f32(q_i[j], ti);
        }
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float32x4x2_t p = sk__cquad_load(g->P[i][j] + f);
            float32x4_t tr = vsubq_f32(
                vmulq_f32(gain_r[i], q_r[j]),
                vmulq_f32(gain_i[i], q_i[j]));
            float32x4_t ti = vaddq_f32(
                vmulq_f32(gain_r[i], q_i[j]),
                vmulq_f32(gain_i[i], q_r[j]));
            p.val[0] = vmulq_f32(vsubq_f32(p.val[0], tr), inv_lambda);
            p.val[1] = vmulq_f32(vsubq_f32(p.val[1], ti), inv_lambda);
            sk__cquad_store(g->P[i][j] + f, p);
        }
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            float32x4x2_t pij = sk__cquad_load(g->P[i][j] + f);
            float32x4x2_t pji = sk__cquad_load(g->P[j][i] + f);
            float32x4x2_t avg;
            avg.val[0] = vmulq_n_f32(
                vaddq_f32(pij.val[0], pji.val[0]), 0.5f);
            avg.val[1] = vmulq_n_f32(
                vsubq_f32(pij.val[1], pji.val[1]), 0.5f);
            sk__cquad_store(g->P[i][j] + f, avg);
            avg.val[1] = vnegq_f32(avg.val[1]);
            sk__cquad_store(g->P[j][i] + f, avg);
        }
        {
            float32x4x2_t diag = sk__cquad_load(g->P[i][i] + f);
            uint32x4_t below =
                vcltq_f32(diag.val[0], vdupq_n_f32(BF_P_DIAG_FLOOR));
            uint32x4_t above =
                vcgtq_f32(diag.val[0], vdupq_n_f32(BF_P_DIAG_CEIL));
            diag.val[0] = vbslq_f32(
                below, vdupq_n_f32(BF_P_DIAG_FLOOR), diag.val[0]);
            diag.val[0] = vbslq_f32(
                above, vdupq_n_f32(BF_P_DIAG_CEIL), diag.val[0]);
            diag.val[1] = zero;
            sk__cquad_store(g->P[i][i] + f, diag);
        }
    }

    {
        float32x4x2_t spec = sk__cquad_load(bf_spec + f);
        const float32x4_t leak = vdupq_n_f32(BF_WA_LEAK);
        const float32x4_t mu = vdupq_n_f32(g->mu);
        for (int m = 0; m < 4; ++m) {
            float32x4x2_t wa = sk__cquad_load(g->wa[m] + f);
            float32x4_t update_r = vaddq_f32(
                vmulq_f32(gain_r[m], spec.val[0]),
                vmulq_f32(gain_i[m], spec.val[1]));
            float32x4_t update_i = vsubq_f32(
                vmulq_f32(gain_i[m], spec.val[0]),
                vmulq_f32(gain_r[m], spec.val[1]));
            if (use_mu_scaling) {
                update_r = vmulq_f32(update_r, mu);
                update_i = vmulq_f32(update_i, mu);
            }
            wa.val[0] = vaddq_f32(vmulq_f32(wa.val[0], leak), update_r);
            wa.val[1] = vaddq_f32(vmulq_f32(wa.val[1], leak), update_i);
            sk__cquad_store(g->wa[m] + f, wa);
        }
    }

    for (int lane = 0; lane < 4; ++lane) {
        for (int m = 0; m < 4; ++m) {
            if (!isfinite(g->wa[m][f + lane].r) ||
                !isfinite(g->wa[m][f + lane].i)) {
                bf_reset_bin(g, f + lane);
                break;
            }
        }
    }
}
#endif

static void bf_rls_update_bins(
    BF* g,
    const Complex* u_flat,
    const Complex* bf_spec,
    int use_notebook_update,
    int use_mu_scaling,
    int use_mask_freeze,
    const int* mask,
    int force_scalar)
{
    /* Bins whose mask bit is set are frozen (skipped) only in mask-freeze
     * mode; elsewhere the mask is ignored. */
    const int* frozen = use_mask_freeze ? mask : NULL;
    int f = 0;
#if SK_HAVE_NEON
    if (!force_scalar && g->M == 4) {
        for (; f + 4 <= g->F; f += 4) {
            int vector_eligible = 1;
            if (frozen) {
                for (int lane = 0; lane < 4; ++lane) {
                    if (frozen[f + lane]) {
                        vector_eligible = 0;
                        break;
                    }
                }
            }
            if (vector_eligible) {
                bf_rls_update_four_neon(
                    g, f, u_flat, bf_spec,
                    use_notebook_update, use_mu_scaling);
            } else {
                for (int lane = 0; lane < 4; ++lane) {
                    if (!frozen[f + lane]) {
                        bf_rls_update_bin_scalar(
                            g, f + lane, u_flat, bf_spec,
                            use_notebook_update, use_mu_scaling);
                    }
                }
            }
        }
    }
#endif
    (void)force_scalar;
    for (; f < g->F; ++f) {
        if (!(frozen && frozen[f])) {
            bf_rls_update_bin_scalar(
                g, f, u_flat, bf_spec,
                use_notebook_update, use_mu_scaling);
        }
    }
}

static int doa2index(float doa_rad, int num_angles)
{
    if (!isfinite(doa_rad)) return 0;

    float deg = doa_rad * 180.0f / M_PI;
    /* Keeps the wrap loops below short and finite for huge angles. */
    if (!isfinite(deg)) return 0;
    if (fabsf(deg) >= 1.0e6f) deg = fmodf(deg, 360.0f);

    while (deg < 0.0f)   deg += 360.0f;
    while (deg >= 360.0f) deg -= 360.0f;

    float step = 360.0f / num_angles;

    int idx = (int)roundf(deg / step);

    if (idx == num_angles) idx = 0;
    if (idx < 0) idx = 0;
    if (idx >= num_angles) idx = num_angles - 1;

    return idx;
}

/* Mic-0 passthrough frame: output = X[0], no adaptation, exported response
 * = unit weight on channel 0. */
static void bf_bypass_frame(
    BF* g,
    const Complex* const* X,
    Complex* bf_out,
    Complex* effective_weights)
{
    for (int f = 0; f < g->F; f++) {
        bf_out[f] = X[0][f];
        if (effective_weights) {
            for (int m = 0; m < g->M; ++m) {
                effective_weights[m * g->F + f].r = m == 0 ? 1.0f : 0.0f;
                effective_weights[m * g->F + f].i = 0.0f;
            }
        }
    }

    g->doa_used = NAN;
    g->adaptive = 0;
    g->frame_idx += 1;
}

static void bf_process_with_weights_impl(
    BF* g,
    const Complex* const* X,
    float doa_s,
    int allow_adapt_in,
    const int* mask,
    Complex* bf_out,
    Complex* effective_weights,
    int force_scalar_rls)
{
    float doa_use;
    int doa_idx;
    Complex** a;   // (M,F)
    const int allow_adapt = allow_adapt_in ? 1 : 0;
    int use_notebook_update = 0;
    int use_mask_freeze = 1;
    int use_mu_scaling = 1;

    if (!g->enable) {
        bf_bypass_frame(g, X, bf_out, effective_weights);
        return;
    }

    /* fixed_align_notebook=1 selects the notebook-style update and disables the mask freeze and mu scaling. */
    if (g->enable_fix_mode && g->fixed_align_notebook) {
        use_notebook_update = 1;
        use_mask_freeze = 0;
        use_mu_scaling = 0;
    }

    /* DOA policy:
     * - Auto mode: doa_s is the final smoothed and held DOA from the caller; BF holds nothing itself.
     * - doa_s == NAN means no usable DOA yet: bypass to mic 0 and do not adapt.
     * - allow_adapt_in is supplied by the caller (typically vad_out == 0). */
    if (g->enable_fix_mode) {
        doa_use = g->fixed_doa_rad;
    } else {
        if (isnan(doa_s)) {
            bf_bypass_frame(g, X, bf_out, effective_weights);
            return;
        }

        doa_use = doa_s;
    }

    g->doa_used = doa_use;
    doa_idx = doa2index(doa_use, g->num_angles);
    a = g->a_array[doa_idx];   // (M,F)

    Complex* das = g->scratch_das;
    Complex* wu = g->scratch_wu;
    Complex* bf_spec = g->scratch_spec;
    Complex (*u)[g->F] =
        (Complex (*)[g->F])g->scratch_u;

    spatial_conj_beamform(
        (const Complex* const*)a,
        (const Complex* const*)X,
        g->M, g->F, 1.0f / (float)g->M, das);

    /* Blocking output in projection form, u = x - a * (a^H x / a^H a), with a^H x = das[f] * M.
     * O(F*M) instead of building the F x M x M blocking matrix. */
    spatial_bf_projection(
        (const Complex* const*)a,
        (const Complex* const*)X,
        das, g->M, g->F, &u[0][0]);

    {
        const Complex* u_channels[g->M];
        for (int m = 0; m < g->M; ++m) u_channels[m] = u[m];
        spatial_conj_beamform(
            (const Complex* const*)g->wa,
            u_channels, g->M, g->F, 1.0f, wu);
    }

    sk_csub_f32(bf_spec, das, wu, g->F);
    memcpy(bf_out, bf_spec, (size_t)g->F * sizeof(Complex));

    /* Export the response of the current wa (before the RLS update): y = a^H x/M - wa^H (x - a(a^H x)/denom). */
    if (effective_weights) {
        spatial_bf_effective_weights(
            (const Complex* const*)a,
            (const Complex* const*)g->wa,
            g->M, g->F, effective_weights);
    }

    /* The BF output is computed every frame; adapt_interval only skips the RLS update on non-update frames. */
    int do_adapt_this_frame = allow_adapt;

    if (do_adapt_this_frame && g->adapt_interval > 1) {
        if ((g->frame_idx % g->adapt_interval) != 0) {
            do_adapt_this_frame = 0;
        }
    }

    if (do_adapt_this_frame) {
        bf_rls_update_bins(
            g, &u[0][0], bf_spec,
            use_notebook_update, use_mu_scaling,
            use_mask_freeze, mask, force_scalar_rls);

        g->adaptive = 1;
    } else {
        g->adaptive = 0;
    }

    g->frame_idx += 1;
}

void bf_process_with_weights(BF* g,
                              const Complex* const* X,
                              float doa_s,
                              int allow_adapt_in,
                              const int* mask,
                              Complex* bf_out,
                              Complex* effective_weights)
{
    bf_process_with_weights_impl(
        g, X, doa_s, allow_adapt_in, mask,
        bf_out, effective_weights, 0);
}

#ifdef BF_TESTING
/* Scalar-recurrence entry compiled only with BF_TESTING; absent from the production library. */
void bf_test_process_with_weights_scalar_rls(
    BF* g,
    const Complex* const* X,
    float doa_s,
    int allow_adapt_in,
    const int* mask,
    Complex* bf_out,
    Complex* effective_weights)
{
    bf_process_with_weights_impl(
        g, X, doa_s, allow_adapt_in, mask,
        bf_out, effective_weights, 1);
}
#endif

void bf_reset(BF* g)
{
    if (!g) return;
    for (int i = 0; i < g->M; ++i) {
        for (int j = 0; j < g->M; ++j) {
            for (int f = 0; f < g->F; ++f) {
                g->P[i][j][f].r = i == j ? 1.0f : 0.0f;
                g->P[i][j][f].i = 0.0f;
            }
        }
    }
    for (int m = 0; m < g->M; ++m) {
        memset(g->wa[m], 0, (size_t)g->F * sizeof(Complex));
    }
    g->frame_idx = 0;
    g->doa_used = NAN;
    g->adaptive = 0;
}

void bf_destroy(BF* g)
{
    if (!g) return;

    /* Pool path (owned_heap == NULL): the caller owns the memory. Heap path: one block, one free(). */
    if (g->owned_heap) free(g->owned_heap);
}

float bf_get_doa_used(const BF* g)
{
    return g ? g->doa_used : NAN;
}

int bf_get_adaptive(const BF* g)
{
    return g ? g->adaptive : 0;
}

long bf_get_bin_resets(const BF* g)
{
    return g ? g->bin_resets : 0;
}

float bf_p_diag_floor(void)
{
    return BF_P_DIAG_FLOOR;
}

float bf_p_diag_ceil(void)
{
    return BF_P_DIAG_CEIL;
}

float bf_wa_leak_factor(void)
{
    return BF_WA_LEAK;
}
