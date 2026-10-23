#ifndef BF_TEST_HOOKS_H
#define BF_TEST_HOOKS_H

#include "bf.h"

/* Scalar-recurrence entry; compiled only with BF_TESTING and absent from the production library. */
void bf_test_process_with_weights_scalar_rls(
    BF* g,
    const Complex* const* X,
    float doa_s,
    int allow_adapt_in,
    const int* mask,
    Complex* bf_out,
    Complex* effective_weights);

#endif
