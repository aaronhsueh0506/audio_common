/* hpf_f64.c — 1:1 port of the AEC repo's python/aec.py HighPassFilter.
 *
 * Python __init__ (aec.py:996-1013):
 *     wc    = 2.0 * np.pi * cutoff_hz / sample_rate
 *     wc_w  = np.tan(wc / 2.0)
 *     k     = wc_w * wc_w
 *     sqrt2 = np.sqrt(2.0)
 *     norm  = 1.0 / (1.0 + sqrt2 * wc_w + k)
 *     b0    =  norm
 *     b1    = -2.0 * norm
 *     b2    =  norm
 *     a1    =  2.0 * (k - 1.0) * norm
 *     a2    = (1.0 - sqrt2 * wc_w + k) * norm
 *     z1=z2 = 0.0
 *
 * Python process (aec.py:1015-1027):
 *     for i in range(len(x)):
 *         xi = float(x[i])
 *         yi = b0*xi + z1
 *         z1 = b1*xi - a1*yi + z2
 *         z2 = b2*xi - a2*yi
 *         out[i] = yi
 *
 * Note `out[i] = yi` truncates yi (fp64) → out dtype (np.float32 typically).
 * C mirrors this by computing yi in double and writing as float.
 */
#include "hpf_f64.h"
#include <math.h>

#ifndef M_PI_HPF_F64
#define M_PI_HPF_F64 3.14159265358979323846
#endif

void hpf_f64_init(HpfF64* h, double cutoff_hz, int sample_rate) {
    double wc    = 2.0 * M_PI_HPF_F64 * cutoff_hz / (double)sample_rate;
    double wc_w  = tan(wc / 2.0);
    double k     = wc_w * wc_w;
    double sqrt2 = sqrt(2.0);
    double norm  = 1.0 / (1.0 + sqrt2 * wc_w + k);

    h->b0 =  norm;
    h->b1 = -2.0 * norm;
    h->b2 =  norm;
    h->a1 =  2.0 * (k - 1.0) * norm;
    h->a2 = (1.0 - sqrt2 * wc_w + k) * norm;

    h->z1 = 0.0;
    h->z2 = 0.0;
}

void hpf_f64_reset(HpfF64* h) {
    h->z1 = 0.0;
    h->z2 = 0.0;
}

void hpf_f64_process(HpfF64* h, const float* x_in, float* x_out, size_t n) {
    const double b0 = h->b0, b1 = h->b1, b2 = h->b2;
    const double a1 = h->a1, a2 = h->a2;
    double z1 = h->z1, z2 = h->z2;
    for (size_t i = 0; i < n; ++i) {
        double xi = (double)x_in[i];
        double yi = b0 * xi + z1;
        z1 = b1 * xi - a1 * yi + z2;
        z2 = b2 * xi - a2 * yi;
        x_out[i] = (float)yi;
    }
    h->z1 = z1;
    h->z2 = z2;
}
