#include "audio_window.h"

#include <math.h>

#define AC_PI_F 3.14159265358979323846f

void ac_window_build(float* w, int n) {
    int i;
    for (i = 0; i < n; ++i)
        w[i] = sqrtf(0.5f * (1.0f - cosf(2.0f * AC_PI_F * (float)i / (float)n)));
}
