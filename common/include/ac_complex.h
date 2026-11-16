/** ac_complex.h - Complex type shared by the FFT wrapper and the SIMD kernel headers. */
#ifndef AC_COMPLEX_H
#define AC_COMPLEX_H

#include <stddef.h>

typedef struct {
    float r;  /* real */
    float i;  /* imaginary */
} Complex;

/* The NEON complex kernels, and the pointer casts between Complex and ne10_fft_cpx_float32_t, need exactly two contiguous unpadded floats, r at 0 and i at 4.
 * C99 compile-time check (a negative array size is the failure); a typedef emits no code. */
typedef char ac_complex_layout_size_check[(sizeof(Complex) == 2 * sizeof(float)) ? 1 : -1];
typedef char ac_complex_layout_offset_check[(offsetof(Complex, r) == 0 && offsetof(Complex, i) == sizeof(float)) ? 1 : -1];

#endif
