/**
 * Stateful rational polyphase audio resampler.
 *
 * Accepted rates are deliberately restricted to the product contract:
 * 8000, 16000, 24000, 32000, and 48000 Hz.  Input and output are interleaved
 * float32 frames.  State is preserved across process() calls, so arbitrary
 * block boundaries produce the same samples as one contiguous call.
 */
#ifndef AUDIO_RESAMPLER_H
#define AUDIO_RESAMPLER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AUDIO_RESAMPLER_MAX_CHANNELS 8

typedef struct AudioResampler AudioResampler;

int audio_resampler_rate_supported(int sample_rate);

AudioResampler* audio_resampler_create(int input_rate,
                                       int output_rate,
                                       int channels);
size_t audio_resampler_get_mem_size(int input_rate,
                                    int output_rate,
                                    int channels);
AudioResampler* audio_resampler_init(void* mem,
                                     size_t mem_size,
                                     int input_rate,
                                     int output_rate,
                                     int channels);
void audio_resampler_destroy(AudioResampler* self);
void audio_resampler_reset(AudioResampler* self);

/**
 * Process as many complete input frames as output_capacity_frames permits.
 *
 * Returns 0 on success. consumed_frames and produced_frames are always
 * written on success.  For unequal rates input/output buffers must not
 * overlap. Equal-rate conversion is an exact memmove passthrough.
 */
int audio_resampler_process(AudioResampler* self,
                            const float* input,
                            int input_frames,
                            float* output,
                            int output_capacity_frames,
                            int* consumed_frames,
                            int* produced_frames);

/** Conservative output capacity for the next input_frames at current state. */
int audio_resampler_output_bound(const AudioResampler* self,
                                 int input_frames);

int audio_resampler_input_rate(const AudioResampler* self);
int audio_resampler_output_rate(const AudioResampler* self);
int audio_resampler_channels(const AudioResampler* self);
int audio_resampler_latency_input_frames(const AudioResampler* self);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_RESAMPLER_H */
