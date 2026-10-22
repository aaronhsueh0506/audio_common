/* Periodic sqrt-Hann window construction, independent of FFT and framing. */
#ifndef AUDIO_WINDOW_H
#define AUDIO_WINDOW_H

#ifdef __cplusplus
extern "C" {
#endif

/* Fill w[0..n) with sqrtf(0.5f * (1 - cosf(2*pi*i/n))) in float32.
 * w must hold n floats when n > 0; n <= 0 leaves w untouched. */
void ac_window_build(float* w, int n);

#ifdef __cplusplus
}
#endif
#endif
