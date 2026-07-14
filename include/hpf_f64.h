/* hpf_f64.h — float64-state high-pass biquad (bit-exact AEC parity variant).
 *
 * 2nd-order Butterworth IIR HPF (bilinear transform), Direct Form II, with
 * two delay states z1/z2. This is a 1:1 port of the AEC repo's Python
 * reference (`python/aec.py` HighPassFilter): Python computes coefficients
 * in fp64 and runs the sample loop in Python `float` (fp64), so this port
 * uses `double` internally to stay bit-exact — the port rule is "match the
 * reference dtype", not "float32 everywhere".
 *
 * This is deliberately a SEPARATE module from hpf.h (the float32
 * platform HPF with the create/get_mem_size/init API): the two filters
 * differ in state precision and API shape, so they carry distinct names
 * instead of colliding on hpf_init/hpf_process/hpf_reset. Value-type
 * struct, caller-owned; no allocation anywhere.
 */
#ifndef HPF_F64_H
#define HPF_F64_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HpfF64 {
    /* Mirrors the Python __init__ field order/names exactly */
    double b0, b1, b2;
    double a1, a2;
    double z1, z2;
} HpfF64;

void hpf_f64_init(HpfF64* h, double cutoff_hz, int sample_rate);
void hpf_f64_reset(HpfF64* h);
/* Mirrors the Python `process(x)`: block of length n (in-place allowed). */
void hpf_f64_process(HpfF64* h, const float* x_in, float* x_out, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* HPF_F64_H */
