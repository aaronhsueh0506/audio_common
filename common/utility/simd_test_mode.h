/** Test helper: SK_MATCH_SCALAR_TO_VECTOR_MODE().
 *
 * The 32-bit ARM vector unit always flushes subnormals to zero and returns the default NaN, while the scalar VFP unit follows FPSCR.
 * A test that compares a NEON kernel with its scalar twin (including a twin built with SIMD_KERNELS_FORCE_SCALAR) calls this once at
 * the start of main so both sides see the same arithmetic (FPSCR.FZ: flush subnormals, FPSCR.DN: default NaN). A no-op elsewhere: on
 * AArch64 both units follow FPCR alike. No library code includes or calls it. */

#ifndef SIMD_TEST_MODE_H
#define SIMD_TEST_MODE_H

#if defined(__arm__) && defined(__ARM_FP)
static inline void sk_match_scalar_to_vector_mode(void) {
    unsigned int fpscr;
    __asm__ volatile("vmrs %0, fpscr" : "=r"(fpscr));
    fpscr |= (1u << 24) | (1u << 25);
    __asm__ volatile("vmsr fpscr, %0" : : "r"(fpscr));
}
#define SK_MATCH_SCALAR_TO_VECTOR_MODE() sk_match_scalar_to_vector_mode()
#else
#define SK_MATCH_SCALAR_TO_VECTOR_MODE() ((void)0)
#endif

#endif /* SIMD_TEST_MODE_H */
