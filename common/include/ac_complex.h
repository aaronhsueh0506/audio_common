/**
 * ac_complex.h - shared Complex type for the FFT wrapper and its consumers
 *
 * Split out of fft_wrapper.h so the public FFT API header and the internal
 * SIMD kernel headers (utility/simd_kernels.h, utility/simd_kernel_nn.h) can
 * both reference the same type without either reaching into the other's
 * layer: fft_wrapper.h includes this header directly, and so does
 * utility/simd_kernels.h (utility/simd_kernel_nn.h gets it transitively via
 * simd_kernels.h), instead of simd_kernels.h including fft_wrapper.h just
 * for this typedef.
 */
#ifndef AC_COMPLEX_H
#define AC_COMPLEX_H

/* Complex number structure */
typedef struct {
    float r;  /* Real part */
    float i;  /* Imaginary part */
} Complex;

#endif /* AC_COMPLEX_H */
