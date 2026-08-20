/* Non-inline instantiations of header-only SIMD kernels, existing solely so
 * the floating-point-contract audit has a function body to disassemble.
 *
 * SCOPE: this carries a wrapper for exactly the kernels listed below -- today
 * that is sk_wola_accumulate_f32 alone, not all of simd_kernels.h. A kernel
 * without a wrapper here is simply not disassembly-audited; add one when a new
 * kernel's scalar/NEON rounding agreement needs to be provable from the
 * emitted code rather than from the source.
 *
 * Every kernel in simd_kernels.h is `static inline`. In a normal build it is
 * either inlined into its caller or dropped entirely, so an object file
 * carrying "the kernel" does not exist -- and a disassembly audit pointed at
 * the library would pass by finding nothing, which is exactly the shape of a
 * check that cannot fail. These wrappers give the audited kernels a real,
 * externally visible symbol whose code the audit can inspect.
 *
 * Nothing links against these at runtime; they exist for the audit and for
 * anyone wanting to read the generated code for one kernel in isolation.
 */
#include "simd_kernels.h"

void sk_audit_wola_accumulate_f32(float *acc, const float *x,
                                  const float *w, int n) {
    sk_wola_accumulate_f32(acc, x, w, n);
}
