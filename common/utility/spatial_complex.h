#ifndef SPATIAL_COMPLEX_H
#define SPATIAL_COMPLEX_H

#include <math.h>
#include "fft_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

Complex spatial_complex_add(Complex a, Complex b);
Complex spatial_complex_mul(Complex a, Complex b);
Complex spatial_complex_conj(Complex a);

Complex spatial_complex_scale(Complex a, float x);

#ifdef __cplusplus
}
#endif

#endif
