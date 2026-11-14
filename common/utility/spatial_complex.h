#ifndef SPATIAL_COMPLEX_H
#define SPATIAL_COMPLEX_H

#include <math.h>
#include "fft_wrapper.h"

static inline Complex spatial_complex_add(Complex a, Complex b)
{
    Complex y;
    y.r = a.r + b.r;
    y.i = a.i + b.i;
    return y;
}

static inline Complex spatial_complex_mul(Complex a, Complex b)
{
    Complex y;
    y.r = a.r * b.r - a.i * b.i;
    y.i = a.r * b.i + a.i * b.r;
    return y;
}

static inline Complex spatial_complex_conj(Complex a)
{
    Complex y;
    y.r = a.r;
    y.i = -a.i;
    return y;
}

static inline Complex spatial_complex_scale(Complex a, float x)
{
    Complex y;
    y.r = a.r * x;
    y.i = a.i * x;
    return y;
}

#endif
