/** Stateful rational polyphase resampler for 8000, 16000, 24000, 32000 and 48000 Hz.
 * Input and output are interleaved float32 frames; state carries across process() calls, so block boundaries do not change the samples. */
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

/** Process as many complete input frames as output_capacity_frames permits.
 * `output` needs output_capacity_frames * channels floats. Returns 0 on success; consumed_frames and produced_frames are then always written.
 * consumed_frames < input_frames means the output capacity ran out; feed the remainder again.
 * For unequal rates the buffers must not overlap; equal rates are a memmove passthrough.
 * A non-finite input sample contaminates every output whose filter window contains it (NEON also reads up to 3
 * zero-padded taps, so SIMD=0 contaminates slightly fewer); guard the input. */
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

/** Group delay of the kernel in INPUT frames, rounded to the nearest frame (fractional for coprime pairs such as 32k -> 24k). */
int audio_resampler_latency_input_frames(const AudioResampler* self);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_RESAMPLER_H */
