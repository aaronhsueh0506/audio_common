/**
 * Configurable input gain expressed in amplitude dB.
 *
 * The dB value is converted once by set/create/init; process() performs one
 * multiplication per sample and supports in-place operation.  No clipping is
 * applied: downstream saturation policy remains the caller's responsibility.
 */
#ifndef AUDIO_PRE_GAIN_H
#define AUDIO_PRE_GAIN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AudioPreGain AudioPreGain;

AudioPreGain* audio_pre_gain_create(float gain_db);
size_t audio_pre_gain_get_mem_size(void);
AudioPreGain* audio_pre_gain_init(void* mem, size_t mem_size, float gain_db);
void audio_pre_gain_destroy(AudioPreGain* self);

int audio_pre_gain_set_db(AudioPreGain* self, float gain_db);
float audio_pre_gain_get_db(const AudioPreGain* self);
float audio_pre_gain_get_linear(const AudioPreGain* self);

/**
 * Apply pre-gain to n_samples. input and output may be the same pointer.
 * Returns 0 on success and -1 for an invalid handle/pointer/count.
 */
int audio_pre_gain_process(const AudioPreGain* self,
                           const float* input,
                           float* output,
                           int n_samples);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_PRE_GAIN_H */
