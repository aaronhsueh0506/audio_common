/** ac_complex.h - Complex type shared by the FFT wrapper and the SIMD kernel headers. */
#ifndef AC_COMPLEX_H
#define AC_COMPLEX_H

typedef struct {
    float r;  /* real */
    float i;  /* imaginary */
} Complex;

#endif
