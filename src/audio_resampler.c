#include "audio_resampler.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "mem_align.h"

#if defined(__aarch64__) && defined(__ARM_NEON) && \
    !defined(SIMD_KERNELS_FORCE_SCALAR)
#include <arm_neon.h>
#define AUDIO_RESAMPLER_NEON 1
#else
#define AUDIO_RESAMPLER_NEON 0
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define M_PI_F ((float)M_PI)

#define AUDIO_RESAMPLER_HALF_TAPS_PER_RATIO 16

struct AudioResampler {
    int input_rate;
    int output_rate;
    int channels;
    int up;
    int down;
    int filter_length;
    int taps_per_phase;
    int history_head;
    int identity;
    int is_static;
    uint64_t input_index;
    uint64_t next_output_tick;
    float* coefficients; /* [up][taps_per_phase] */
    float* history;      /* [channels][taps_per_phase] */
    void* owned_heap;
};

typedef struct ResamplerLayout {
    int up;
    int down;
    int filter_length;
    int taps_per_phase;
    size_t bytes;
} ResamplerLayout;

static int audio_gcd(int a, int b)
{
    while (b != 0) {
        int r = a % b;
        a = b;
        b = r;
    }
    return a;
}

int audio_resampler_rate_supported(int sample_rate)
{
    return sample_rate == 8000 || sample_rate == 16000 ||
           sample_rate == 24000 || sample_rate == 32000 ||
           sample_rate == 48000;
}

static int audio_resampler_layout(int input_rate,
                                  int output_rate,
                                  int channels,
                                  ResamplerLayout* layout)
{
    int divisor;
    int ratio_max;
    int half;
    size_t bytes;
    if (!layout || !audio_resampler_rate_supported(input_rate) ||
        !audio_resampler_rate_supported(output_rate) ||
        channels <= 0 || channels > AUDIO_RESAMPLER_MAX_CHANNELS) {
        return 0;
    }
    divisor = audio_gcd(input_rate, output_rate);
    layout->up = output_rate / divisor;
    layout->down = input_rate / divisor;
    ratio_max = layout->up > layout->down ? layout->up : layout->down;
    half = AUDIO_RESAMPLER_HALF_TAPS_PER_RATIO * ratio_max;
    layout->filter_length =
        layout->up == layout->down ? 1 : 2 * half + 1;
    layout->taps_per_phase =
        (layout->filter_length + layout->up - 1) / layout->up;
    bytes = ck_field_size(0, 1, sizeof(AudioResampler));
    bytes = ck_field_size(
        bytes, (size_t)layout->up * layout->taps_per_phase, sizeof(float));
    bytes = ck_field_size(
        bytes, (size_t)channels * layout->taps_per_phase, sizeof(float));
    if (MEM_SIZE_INVALID(bytes)) return 0;
    layout->bytes = bytes;
    return 1;
}

size_t audio_resampler_get_mem_size(int input_rate,
                                    int output_rate,
                                    int channels)
{
    ResamplerLayout layout;
    return audio_resampler_layout(
               input_rate, output_rate, channels, &layout)
        ? layout.bytes : 0;
}

static void* audio_resampler_carve(uint8_t** cursor,
                                   size_t* remaining,
                                   size_t count,
                                   size_t item_size)
{
    size_t bytes = ck_align16_size(ck_mul_size(count, item_size));
    void* result;
    if (MEM_SIZE_INVALID(bytes) || bytes > *remaining) return NULL;
    result = *cursor;
    *cursor += bytes;
    *remaining -= bytes;
    return result;
}

static void audio_resampler_design(AudioResampler* self)
{
    int half = (self->filter_length - 1) / 2;
    int ratio_max = self->up > self->down ? self->up : self->down;
    float cutoff = 0.47f / (float)ratio_max;

    memset(self->coefficients, 0,
           (size_t)self->up * self->taps_per_phase * sizeof(float));
    if (self->identity) {
        self->coefficients[0] = 1.0f;
        return;
    }
    for (int n = 0; n < self->filter_length; ++n) {
        int phase = n % self->up;
        int tap = n / self->up;
        float offset = (float)(n - half);
        float ideal;
        float position =
            (float)n / (float)(self->filter_length - 1);
        float window =
            0.42f - 0.5f * cosf(2.0f * M_PI_F * position) +
            0.08f * cosf(4.0f * M_PI_F * position);
        if (n == half) {
            ideal = 2.0f * cutoff;
        } else {
            ideal = sinf(2.0f * M_PI_F * cutoff * offset) /
                    (M_PI_F * offset);
        }
        self->coefficients[phase * self->taps_per_phase + tap] =
            (float)self->up * ideal * window;
    }

    /* Pin unity DC independently for every fractional phase. */
    for (int phase = 0; phase < self->up; ++phase) {
        float sum = 0.0f;
        for (int tap = 0; tap < self->taps_per_phase; ++tap) {
            sum += self->coefficients[
                phase * self->taps_per_phase + tap];
        }
        if (fabsf(sum) > 1e-12f) {
            float inverse = 1.0f / sum;
            for (int tap = 0; tap < self->taps_per_phase; ++tap) {
                self->coefficients[
                    phase * self->taps_per_phase + tap] *= inverse;
            }
        }
    }
}

AudioResampler* audio_resampler_init(void* mem,
                                     size_t mem_size,
                                     int input_rate,
                                     int output_rate,
                                     int channels)
{
    ResamplerLayout layout;
    AudioResampler* self;
    uint8_t* cursor;
    size_t remaining;
    if (!mem || !MEM_IS_ALIGNED16(mem) ||
        !audio_resampler_layout(
            input_rate, output_rate, channels, &layout) ||
        mem_size < layout.bytes) {
        return NULL;
    }
    cursor = (uint8_t*)mem;
    remaining = mem_size;
    self = (AudioResampler*)audio_resampler_carve(
        &cursor, &remaining, 1, sizeof(*self));
    if (!self) return NULL;
    memset(self, 0, sizeof(*self));
    self->coefficients = (float*)audio_resampler_carve(
        &cursor, &remaining,
        (size_t)layout.up * layout.taps_per_phase, sizeof(float));
    self->history = (float*)audio_resampler_carve(
        &cursor, &remaining,
        (size_t)channels * layout.taps_per_phase, sizeof(float));
    if (!self->coefficients || !self->history) return NULL;
    self->input_rate = input_rate;
    self->output_rate = output_rate;
    self->channels = channels;
    self->up = layout.up;
    self->down = layout.down;
    self->filter_length = layout.filter_length;
    self->taps_per_phase = layout.taps_per_phase;
    self->identity = input_rate == output_rate;
    self->is_static = 1;
    audio_resampler_design(self);
    audio_resampler_reset(self);
    return self;
}

AudioResampler* audio_resampler_create(int input_rate,
                                       int output_rate,
                                       int channels)
{
    size_t bytes =
        audio_resampler_get_mem_size(input_rate, output_rate, channels);
    void* mem;
    AudioResampler* self;
    if (bytes == 0) return NULL;
    mem = malloc(bytes);
    if (!mem) return NULL;
    self = audio_resampler_init(
        mem, bytes, input_rate, output_rate, channels);
    if (!self) {
        free(mem);
        return NULL;
    }
    self->is_static = 0;
    self->owned_heap = mem;
    return self;
}

void audio_resampler_destroy(AudioResampler* self)
{
    void* owned;
    if (!self || self->is_static) return;
    owned = self->owned_heap;
    free(owned);
}

void audio_resampler_reset(AudioResampler* self)
{
    if (!self) return;
    memset(self->history, 0,
           (size_t)self->channels * self->taps_per_phase * sizeof(float));
    self->history_head = self->taps_per_phase - 1;
    self->input_index = 0;
    self->next_output_tick = 0;
}

static int audio_resampler_outputs_for_next_input(
    const AudioResampler* self)
{
    uint64_t tick = self->next_output_tick;
    int count = 0;
    while (tick / (uint64_t)self->up == self->input_index) {
        ++count;
        tick += (uint64_t)self->down;
    }
    return count;
}

/* Dot one polyphase FIR row against the newest-to-oldest circular history.
 * Keep the scalar implementation in the same TU so `make SIMD=0` exercises
 * exactly the same state machine and only changes this arithmetic kernel. */
static float audio_resampler_dot(const float* coefficients,
                                 const float* history,
                                 int history_head,
                                 int taps)
{
#if AUDIO_RESAMPLER_NEON
    float32x4_t vacc = vdupq_n_f32(0.0f);
    float scalar = 0.0f;
    int tap = 0;
    int index = history_head;

    /* A load is ascending in memory while the ring is consumed descending.
     * vrev64+vext reverses all four lanes without an aliasing cast. */
    while (tap + 4 <= taps && index >= 3) {
        float32x4_t h = vld1q_f32(history + index - 3);
        h = vrev64q_f32(h);
        h = vextq_f32(h, h, 2);
        vacc = vaddq_f32(
            vacc, vmulq_f32(vld1q_f32(coefficients + tap), h));
        tap += 4;
        index -= 4;
    }
    while (tap < taps && index >= 0) {
        scalar += coefficients[tap++] * history[index--];
    }
    index = taps - 1;
    while (tap + 4 <= taps) {
        float32x4_t h = vld1q_f32(history + index - 3);
        h = vrev64q_f32(h);
        h = vextq_f32(h, h, 2);
        vacc = vaddq_f32(
            vacc, vmulq_f32(vld1q_f32(coefficients + tap), h));
        tap += 4;
        index -= 4;
    }
    scalar += vaddvq_f32(vacc);
    while (tap < taps) {
        scalar += coefficients[tap++] * history[index--];
    }
    return scalar;
#else
    float sum = 0.0f;
    int history_index = history_head;
    for (int tap = 0; tap < taps; ++tap) {
        sum += coefficients[tap] * history[history_index];
        history_index -= 1;
        if (history_index < 0) history_index = taps - 1;
    }
    return sum;
#endif
}

int audio_resampler_process(AudioResampler* self,
                            const float* input,
                            int input_frames,
                            float* output,
                            int output_capacity_frames,
                            int* consumed_frames,
                            int* produced_frames)
{
    int consumed = 0;
    int produced = 0;
    if (!self || !input || !output || !consumed_frames || !produced_frames ||
        input_frames < 0 || output_capacity_frames < 0) {
        return -1;
    }
    if (self->identity) {
        int count = input_frames < output_capacity_frames
            ? input_frames : output_capacity_frames;
        memmove(output, input,
                (size_t)count * self->channels * sizeof(float));
        self->input_index += (uint64_t)count;
        self->next_output_tick += (uint64_t)count;
        *consumed_frames = count;
        *produced_frames = count;
        return 0;
    }

    while (consumed < input_frames) {
        int required = audio_resampler_outputs_for_next_input(self);
        if (required > output_capacity_frames - produced) break;

        self->history_head += 1;
        if (self->history_head == self->taps_per_phase)
            self->history_head = 0;
        for (int channel = 0; channel < self->channels; ++channel) {
            self->history[
                channel * self->taps_per_phase + self->history_head] =
                input[consumed * self->channels + channel];
        }

        while (self->next_output_tick / (uint64_t)self->up ==
               self->input_index) {
            int phase =
                (int)(self->next_output_tick % (uint64_t)self->up);
            for (int channel = 0; channel < self->channels; ++channel) {
                const float* coefficients =
                    self->coefficients + phase * self->taps_per_phase;
                const float* history =
                    self->history + channel * self->taps_per_phase;
                output[produced * self->channels + channel] =
                    audio_resampler_dot(
                        coefficients, history, self->history_head,
                        self->taps_per_phase);
            }
            produced += 1;
            self->next_output_tick += (uint64_t)self->down;
        }
        self->input_index += 1;
        consumed += 1;
    }
    *consumed_frames = consumed;
    *produced_frames = produced;
    return 0;
}

int audio_resampler_output_bound(const AudioResampler* self,
                                 int input_frames)
{
    uint64_t numerator;
    uint64_t bound;
    if (!self || input_frames < 0) return -1;
    if (self->identity) return input_frames;
    numerator = (uint64_t)input_frames * (uint64_t)self->up;
    bound = (numerator + (uint64_t)self->down - 1u) /
            (uint64_t)self->down + 1u;
    return bound > (uint64_t)INT_MAX ? -1 : (int)bound;
}

int audio_resampler_input_rate(const AudioResampler* self)
{
    return self ? self->input_rate : -1;
}

int audio_resampler_output_rate(const AudioResampler* self)
{
    return self ? self->output_rate : -1;
}

int audio_resampler_channels(const AudioResampler* self)
{
    return self ? self->channels : -1;
}

int audio_resampler_latency_input_frames(const AudioResampler* self)
{
    if (!self) return -1;
    return (self->filter_length - 1) / (2 * self->up);
}
