#ifndef BF_TEST_HOOKS_H
#define BF_TEST_HOOKS_H

#include "bf.h"

/* Internal oracle for scalar-vs-dispatch state-transition tests. It exists
 * only in the BF_TESTING object and is absent from deployable libbf.a;
 * production code must call bf_process_with_weights(). */
void bf_test_process_with_weights_scalar_rls(
    BF* g,
    const Complex* const* X,
    float doa_s,
    int allow_adapt_in,
    const int* mask,
    Complex* bf_out,
    Complex* effective_weights);

#endif
