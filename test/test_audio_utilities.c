#include "audio_pre_gain.h"
#include "audio_resampler.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CHECK(condition, message)                                      \
    do {                                                               \
        if (!(condition)) {                                            \
            fprintf(stderr, "FAIL: %s (line %d)\n", message, __LINE__); \
            return 1;                                                  \
        }                                                              \
    } while (0)

static int test_pre_gain(void)
{
    const float input[7] = {-0.5f, -0.25f, 0.0f, 0.125f,
                            0.25f, 0.5f, 1.0f};
    float output[7];
    float in_place[7];
    float pool_storage[16];
    AudioPreGain* heap = audio_pre_gain_create(6.0f);
    AudioPreGain* statik = audio_pre_gain_init(
        pool_storage, sizeof(pool_storage), 6.0f);
    CHECK(heap && statik, "pre-gain heap/static create");
    CHECK(fabsf(audio_pre_gain_get_linear(heap) - 1.9952623f) < 1e-6f,
          "6 dB uses amplitude /20 conversion");
    CHECK(audio_pre_gain_process(heap, input, output, 7) == 0,
          "out-of-place pre-gain");
    memcpy(in_place, input, sizeof(input));
    CHECK(audio_pre_gain_process(statik, in_place, in_place, 7) == 0,
          "in-place pre-gain");
    CHECK(memcmp(output, in_place, sizeof(output)) == 0,
          "heap/static and in-place pre-gain parity");
    CHECK(audio_pre_gain_set_db(heap, 0.0f) == 0,
          "set pre-gain");
    CHECK(audio_pre_gain_process(heap, input, output, 7) == 0 &&
          memcmp(input, output, sizeof(input)) == 0,
          "0 dB is exact identity");
    CHECK(audio_pre_gain_set_db(heap, NAN) != 0,
          "reject non-finite dB");
    audio_pre_gain_destroy(statik);
    audio_pre_gain_destroy(heap);
    return 0;
}

static int run_resampler(AudioResampler* resampler,
                         const float* input,
                         int input_frames,
                         int channels,
                         int chunk,
                         float* output,
                         int capacity)
{
    int input_position = 0;
    int output_position = 0;
    while (input_position < input_frames) {
        int request = input_frames - input_position;
        int consumed;
        int produced;
        if (request > chunk) request = chunk;
        CHECK(audio_resampler_process(
                  resampler,
                  input + (size_t)input_position * channels,
                  request,
                  output + (size_t)output_position * channels,
                  capacity - output_position,
                  &consumed,
                  &produced) == 0,
              "streaming resampler process");
        CHECK(consumed > 0, "output capacity must permit progress");
        input_position += consumed;
        output_position += produced;
    }
    return output_position;
}

static int test_supported_matrix(void)
{
    const int rates[] = {8000, 16000, 24000, 32000, 48000};
    for (int i = 0; i < 5; ++i) {
        CHECK(audio_resampler_rate_supported(rates[i]),
              "contract rate accepted");
        for (int j = 0; j < 5; ++j) {
            AudioResampler* value =
                audio_resampler_create(rates[i], rates[j], 2);
            CHECK(value != NULL, "all supported rate pairs construct");
            audio_resampler_destroy(value);
        }
    }
    CHECK(!audio_resampler_rate_supported(44100),
          "44.1 kHz rejected");
    CHECK(audio_resampler_create(44100, 16000, 1) == NULL,
          "unsupported input rate rejected");
    CHECK(audio_resampler_create(16000, 12000, 1) == NULL,
          "unsupported output rate rejected");
    return 0;
}

static int test_identity_and_pool(void)
{
    float input[64];
    float output[64];
    int consumed;
    int produced;
    size_t bytes = audio_resampler_get_mem_size(16000, 16000, 1);
    void* raw = malloc(bytes + 16u);
    uintptr_t aligned_value =
        ((uintptr_t)raw + 15u) & ~(uintptr_t)15u;
    AudioResampler* statik = audio_resampler_init(
        (void*)aligned_value, bytes, 16000, 16000, 1);
    CHECK(raw && statik, "static identity resampler");
    for (int i = 0; i < 64; ++i) input[i] = (float)i / 64.0f;
    CHECK(audio_resampler_process(
              statik, input, 64, output, 64, &consumed, &produced) == 0,
          "identity resample");
    CHECK(consumed == 64 && produced == 64 &&
          memcmp(input, output, sizeof(input)) == 0,
          "equal-rate resampling is bit-exact");
    CHECK(audio_resampler_init(
              (void*)(aligned_value + 1u), bytes, 16000, 16000, 1) == NULL,
          "misaligned resampler pool rejected");
    audio_resampler_destroy(statik);
    free(raw);
    return 0;
}

static int test_streaming_continuity(void)
{
    enum { INPUT_FRAMES = 4096, CHANNELS = 2, CAPACITY = 25000 };
    float* input =
        (float*)malloc(INPUT_FRAMES * CHANNELS * sizeof(float));
    float* contiguous =
        (float*)malloc(CAPACITY * CHANNELS * sizeof(float));
    float* chunked =
        (float*)malloc(CAPACITY * CHANNELS * sizeof(float));
    AudioResampler* one = audio_resampler_create(16000, 48000, CHANNELS);
    AudioResampler* many = audio_resampler_create(16000, 48000, CHANNELS);
    int one_count;
    int many_count;
    CHECK(input && contiguous && chunked && one && many,
          "allocate continuity test");
    for (int i = 0; i < INPUT_FRAMES; ++i) {
        input[i * 2] = sinf((float)i * 0.071f);
        input[i * 2 + 1] = cosf((float)i * 0.053f);
    }
    one_count = run_resampler(
        one, input, INPUT_FRAMES, CHANNELS, INPUT_FRAMES,
        contiguous, CAPACITY);
    many_count = run_resampler(
        many, input, INPUT_FRAMES, CHANNELS, 37, chunked, CAPACITY);
    CHECK(one_count == many_count,
          "block partition does not alter output count");
    CHECK(memcmp(contiguous, chunked,
                 (size_t)one_count * CHANNELS * sizeof(float)) == 0,
          "block partition is sample-identical");
    free(chunked);
    free(contiguous);
    free(input);
    audio_resampler_destroy(many);
    audio_resampler_destroy(one);
    return 0;
}

static float resampled_tone_rms(float frequency_hz)
{
    enum { INPUT_FRAMES = 48000, CAPACITY = 9000 };
    float* input = (float*)malloc(INPUT_FRAMES * sizeof(float));
    float* output = (float*)malloc(CAPACITY * sizeof(float));
    AudioResampler* resampler = audio_resampler_create(48000, 8000, 1);
    int count;
    float sum = 0.0f;
    int start;
    if (!input || !output || !resampler) return NAN;
    for (int i = 0; i < INPUT_FRAMES; ++i) {
        input[i] =
            sinf((float)(2.0 * M_PI) * frequency_hz * (float)i / 48000.0f);
    }
    count = run_resampler(
        resampler, input, INPUT_FRAMES, 1, 113, output, CAPACITY);
    start = audio_resampler_latency_input_frames(resampler) / 6 + 32;
    for (int i = start; i < count; ++i) sum += output[i] * output[i];
    sum = sqrtf(sum / (float)(count - start));
    audio_resampler_destroy(resampler);
    free(output);
    free(input);
    return sum;
}

static int test_downsample_antialias(void)
{
    float pass = resampled_tone_rms(1000.0f);
    float stop = resampled_tone_rms(10000.0f);
    CHECK(isfinite(pass) && isfinite(stop), "tone resample finite");
    CHECK(pass > 0.65f && pass < 0.75f,
          "1 kHz passband amplitude");
    CHECK(stop < pass * 0.01f,
          "48->8 kHz stopband alias suppression");
    return 0;
}

int main(void)
{
    CHECK(test_pre_gain() == 0, "pre-gain tests");
    CHECK(test_supported_matrix() == 0, "resampler rate matrix");
    CHECK(test_identity_and_pool() == 0, "resampler identity/pool");
    CHECK(test_streaming_continuity() == 0, "resampler continuity");
    CHECK(test_downsample_antialias() == 0, "resampler anti-alias");
    printf("All audio pre-gain/resampler tests passed\n");
    return 0;
}
