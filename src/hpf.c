/**
 * hpf.c - 2nd-order Butterworth IIR High-Pass Filter
 *
 * Bilinear transform from analog prototype.
 * Direct Form II transposed implementation.
 * Structurally mirrors the AEC Python reference's HighPassFilter formula
 * (float32, not bit-exact — see the README precision policy).
 */

#include "hpf.h"
#include "mem_align.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Hpf {
    float b0, b1, b2;
    float a1, a2;
    float z1, z2;
    int is_static;
};

/* Compute filter coefficients (shared by create and init) */
static void hpf_compute_coeffs(Hpf* h, float cutoff_hz, int sample_rate) {
    float wc = 2.0f * (float)M_PI * cutoff_hz / (float)sample_rate;
    float wc_w = tanf(wc / 2.0f);
    float k = wc_w * wc_w;
    float sqrt2 = 1.41421356237f;
    float norm = 1.0f / (1.0f + sqrt2 * wc_w + k);
    h->b0 =  norm;
    h->b1 = -2.0f * norm;
    h->b2 =  norm;
    h->a1 =  2.0f * (k - 1.0f) * norm;
    h->a2 =  (1.0f - sqrt2 * wc_w + k) * norm;
    h->z1 = 0.0f;
    h->z2 = 0.0f;
}

/* F14: shared cutoff/sample_rate domain check for hpf_create and hpf_init.
 * cutoff_hz must be finite and > 0; sample_rate (already an int -- always
 * "finite" -- is range-checked instead) must be > 0; and cutoff_hz is kept a
 * fixed margin below Nyquist (0.45*sample_rate, not 0.5) so wc/2 in
 * hpf_compute_coeffs stays well clear of tanf's pi/2 singularity. */
static int hpf_params_valid(float cutoff_hz, int sample_rate) {
    if (!isfinite(cutoff_hz) || cutoff_hz <= 0.0f) return 0;
    if (sample_rate <= 0) return 0;
    if ((double)cutoff_hz >= 0.45 * (double)sample_rate) return 0;
    return 1;
}

Hpf* hpf_create(float cutoff_hz, int sample_rate) {
    if (!hpf_params_valid(cutoff_hz, sample_rate)) return NULL;

    Hpf* h = (Hpf*)calloc(1, sizeof(Hpf));
    if (!h) return NULL;

    hpf_compute_coeffs(h, cutoff_hz, sample_rate);
    return h;
}

/* --- Static memory API --- */

size_t hpf_get_mem_size(void) {
    size_t total = ck_field_size(0, 1, sizeof(Hpf));
    return MEM_SIZE_INVALID(total) ? 0 : total;
}

Hpf* hpf_init(void* mem, size_t mem_size, float cutoff_hz, int sample_rate) {
    if (!mem) return NULL;
    if (!MEM_IS_ALIGNED16(mem)) return NULL;  /* F07: reject a misaligned base before any write */
    if (!hpf_params_valid(cutoff_hz, sample_rate)) return NULL;  /* F14 */
    if (mem_size == 0) return NULL;
    size_t need = hpf_get_mem_size();
    /* need==0 would mean hpf_get_mem_size's arithmetic overflowed -- no
     * mem_size could ever satisfy that via mem_size < need, so reject it
     * explicitly (see the identical note in fft_wrapper.c's fft_init). */
    if (need == 0 || mem_size < need) return NULL;

    Hpf* h = (Hpf*)mem;
    memset(h, 0, sizeof(Hpf));
    h->is_static = 1;
    hpf_compute_coeffs(h, cutoff_hz, sample_rate);
    return h;
}

void hpf_destroy(Hpf* hpf) {
    if (!hpf || hpf->is_static) return;
    free(hpf);
}

void hpf_process(Hpf* h, float* data, int n) {
    if (!h || !data) return;

    float b0 = h->b0, b1 = h->b1, b2 = h->b2;
    float a1 = h->a1, a2 = h->a2;
    float z1 = h->z1, z2 = h->z2;

    for (int i = 0; i < n; i++) {
        float xi = data[i];
        float yi = b0 * xi + z1;
        z1 = b1 * xi - a1 * yi + z2;
        z2 = b2 * xi - a2 * yi;
        data[i] = yi;
    }

    h->z1 = z1;
    h->z2 = z2;
}

void hpf_reset(Hpf* hpf) {
    if (hpf) {
        hpf->z1 = 0.0f;
        hpf->z2 = 0.0f;
    }
}
