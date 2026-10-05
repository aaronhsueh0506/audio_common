#include "audio_resampler.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "mem_align.h"
#include "simd_kernels.h"

#if defined(__GNUC__) || defined(__clang__)
#define RS_ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define RS_ALWAYS_INLINE inline
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define M_PI_F ((float)M_PI)

#define AUDIO_RESAMPLER_HALF_TAPS_PER_RATIO 16

/* Input frames staged per pass. Every channel keeps the previous
 * (window - 1) inputs in front of the staged ones, so the dot products read
 * one contiguous oldest-to-newest window and never wrap. */
#define AUDIO_RESAMPLER_CHUNK 128

struct AudioResampler {
    int input_rate;
    int output_rate;
    int channels;
    int up;
    int down;
    int filter_length;
    int taps_per_phase;
    int window;          /* taps_per_phase rounded up to a multiple of 4 */
    int work_stride;     /* window - 1 + CHUNK, floats per channel       */
    int identity;
    int is_static;
    uint64_t input_index;
    uint64_t next_output_tick;
    /* [up][window]: each phase row time-reversed (oldest tap first) and
     * zero-padded in front, so row[j] pairs with the window sample j. */
    float* coefficients;
    float* work;         /* [channels][work_stride]: tail, then staged input */
    void* owned_heap;
};

typedef struct ResamplerLayout {
    int up;
    int down;
    int filter_length;
    int taps_per_phase;
    int window;
    int work_stride;
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
    layout->window = (layout->taps_per_phase + 3) & ~3;
    layout->work_stride = layout->window - 1 + AUDIO_RESAMPLER_CHUNK;
    bytes = ck_field_size(0, 1, sizeof(AudioResampler));
    bytes = ck_field_size(
        bytes, (size_t)layout->up * layout->window, sizeof(float));
    bytes = ck_field_size(
        bytes, (size_t)channels * layout->work_stride, sizeof(float));
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

/* Tap `tap` of phase `phase`, tap 0 being the newest input's weight. */
#define RS_COEF(self, phase, tap) \
    ((self)->coefficients[(size_t)(phase) * (self)->window + \
                          ((self)->window - 1 - (tap))])

static void audio_resampler_design(AudioResampler* self)
{
    int half = (self->filter_length - 1) / 2;
    int ratio_max = self->up > self->down ? self->up : self->down;
    float cutoff = 0.47f / (float)ratio_max;

    memset(self->coefficients, 0,
           (size_t)self->up * self->window * sizeof(float));
    if (self->identity) {
        RS_COEF(self, 0, 0) = 1.0f;
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
        RS_COEF(self, phase, tap) = (float)self->up * ideal * window;
    }

    /* Pin unity DC independently for every fractional phase. */
    for (int phase = 0; phase < self->up; ++phase) {
        float sum = 0.0f;
        for (int tap = 0; tap < self->taps_per_phase; ++tap) {
            sum += RS_COEF(self, phase, tap);
        }
        if (fabsf(sum) > 1e-12f) {
            float inverse = 1.0f / sum;
            for (int tap = 0; tap < self->taps_per_phase; ++tap) {
                RS_COEF(self, phase, tap) *= inverse;
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
        (size_t)layout.up * layout.window, sizeof(float));
    self->work = (float*)audio_resampler_carve(
        &cursor, &remaining,
        (size_t)channels * layout.work_stride, sizeof(float));
    if (!self->coefficients || !self->work) return NULL;
    self->input_rate = input_rate;
    self->output_rate = output_rate;
    self->channels = channels;
    self->up = layout.up;
    self->down = layout.down;
    self->filter_length = layout.filter_length;
    self->taps_per_phase = layout.taps_per_phase;
    self->window = layout.window;
    self->work_stride = layout.work_stride;
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
    memset(self->work, 0,
           (size_t)self->channels * self->work_stride * sizeof(float));
    self->input_index = 0;
    self->next_output_tick = 0;
}

/* One output: the window's `window` samples (oldest first) against a phase
 * row of the same length. Every output of every phase, channel and block
 * position is formed by exactly this arithmetic, so a sample's value never
 * depends on how the input was split into calls or which outputs shared a
 * pass.
 *   NEON:  four lanes, each accumulating row[j] * win[j] with an unfused
 *          multiply then add in ascending j, then the lanes summed pairwise
 *          ((l0 + l1) + (l2 + l3)).
 *   SIMD=0: one sequential sum, newest tap first -- the order the Python twin
 *          (pipelines/dfn2_rate_adapter.py) reproduces bit for bit. `first`
 *          is the oldest real tap; the zero padding in front is skipped. */
#if SK_HAVE_NEON
static RS_ALWAYS_INLINE float rs_dot1(const float* row, const float* win,
                                      int window, int first)
{
    float32x4_t acc = vdupq_n_f32(0.0f);
    (void)first;
    for (int j = 0; j < window; j += 4) {
        acc = vaddq_f32(acc, vmulq_f32(vld1q_f32(row + j),
                                       vld1q_f32(win + j)));
    }
    return vaddvq_f32(acc);
}

/* Four outputs that share one phase row, each against its own window: the
 * row loads are shared and the four accumulation chains are independent. */
static RS_ALWAYS_INLINE void rs_dot4(const float* row, const float* win0,
                                     const float* win1, const float* win2,
                                     const float* win3, int window,
                                     int first, float* out)
{
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
    (void)first;
    for (int j = 0; j < window; j += 4) {
        const float32x4_t c = vld1q_f32(row + j);
        a0 = vaddq_f32(a0, vmulq_f32(c, vld1q_f32(win0 + j)));
        a1 = vaddq_f32(a1, vmulq_f32(c, vld1q_f32(win1 + j)));
        a2 = vaddq_f32(a2, vmulq_f32(c, vld1q_f32(win2 + j)));
        a3 = vaddq_f32(a3, vmulq_f32(c, vld1q_f32(win3 + j)));
    }
    /* Pairwise, the same (l0 + l1) + (l2 + l3) as vaddvq_f32 per output. */
    vst1q_f32(out, vpaddq_f32(vpaddq_f32(a0, a1), vpaddq_f32(a2, a3)));
}
#else
static RS_ALWAYS_INLINE float rs_dot1(const float* row, const float* win,
                                      int window, int first)
{
    float sum = 0.0f;
    for (int j = window - 1; j >= first; --j) sum += row[j] * win[j];
    return sum;
}

static RS_ALWAYS_INLINE void rs_dot4(const float* row, const float* win0,
                                     const float* win1, const float* win2,
                                     const float* win3, int window,
                                     int first, float* out)
{
    out[0] = rs_dot1(row, win0, window, first);
    out[1] = rs_dot1(row, win1, window, first);
    out[2] = rs_dot1(row, win2, window, first);
    out[3] = rs_dot1(row, win3, window, first);
}
#endif

/* Integer up-sampling (down == 1): every staged input emits all `up` phases
 * and `ahead` stays 0. Four consecutive inputs run together per phase. */
static int rs_run_integer_up(const AudioResampler* self, int staged,
                             float* output, int room)
{
    const int up = self->up;
    const int window = self->window;
    const int first = window - self->taps_per_phase;
    const int channels = self->channels;
    int used = room / up;
    if (used > staged) used = staged;
    for (int channel = 0; channel < channels; ++channel) {
        const float* base = self->work + (size_t)channel * self->work_stride;
        float* out = output + channel;
        int i = 0;
        for (; i + 4 <= used; i += 4) {
            for (int phase = 0; phase < up; ++phase) {
                const float* row = self->coefficients +
                                   (size_t)phase * window;
                float v[4];
                rs_dot4(row, base + i, base + i + 1, base + i + 2,
                        base + i + 3, window, first, v);
                for (int k = 0; k < 4; ++k)
                    out[((size_t)(i + k) * up + phase) * channels] = v[k];
            }
        }
        for (; i < used; ++i) {
            for (int phase = 0; phase < up; ++phase) {
                out[((size_t)i * up + phase) * channels] = rs_dot1(
                    self->coefficients + (size_t)phase * window,
                    base + i, window, first);
            }
        }
    }
    return used;
}

/* Integer decimation (up == 1): an input emits one output when `*ahead` has
 * counted down to 0, then every `down` inputs. Outputs `down` inputs apart
 * share the single row. Returns the inputs consumed. */
static int rs_run_decimation(const AudioResampler* self, int staged,
                             int* ahead, float* output, int room,
                             int* produced)
{
    const int down = self->down;
    const int window = self->window;
    const int first = window - self->taps_per_phase;
    const int channels = self->channels;
    const int first_out = *ahead;   /* input index of the first output */
    int used = staged;
    int n_out = 0;

    if (first_out < staged) {
        n_out = (staged - first_out + down - 1) / down;
        if (n_out > room) {
            n_out = room;
            used = first_out + room * down;
        }
    }
    /* Inputs until the next output after the `used` consumed ones. */
    *ahead = first_out + n_out * down - used;
    for (int channel = 0; channel < channels; ++channel) {
        const float* base = self->work + (size_t)channel * self->work_stride;
        float* out = output + channel;
        int k = 0;
        for (; k + 4 <= n_out; k += 4) {
            const float* win = base + first_out + (size_t)k * down;
            float v[4];
            rs_dot4(self->coefficients, win, win + down, win + 2 * down,
                    win + 3 * down, window, first, v);
            for (int m = 0; m < 4; ++m)
                out[(size_t)(k + m) * channels] = v[m];
        }
        for (; k < n_out; ++k) {
            out[(size_t)k * channels] = rs_dot1(
                self->coefficients, base + first_out + (size_t)k * down,
                window, first);
        }
    }
    *produced = n_out;
    return used;
}

/* Any other rational pair: an input emits the outputs whose tick falls in its
 * own [0, up) window, one dot each. Returns the inputs consumed. */
static int rs_run_rational(const AudioResampler* self, int staged,
                           int* ahead, float* output, int room,
                           int* produced)
{
    const int up = self->up;
    const int down = self->down;
    const int window = self->window;
    const int first = window - self->taps_per_phase;
    const int channels = self->channels;
    int emitted = 0;
    int used = 0;
    while (used < staged) {
        int required = 0;
        for (int t = *ahead; t < up; t += down) ++required;
        if (required > room - emitted) break;
        for (int channel = 0; channel < channels; ++channel) {
            const float* win = self->work +
                               (size_t)channel * self->work_stride + used;
            int k = 0;
            for (int t = *ahead; t < up; t += down, ++k) {
                output[(size_t)(emitted + k) * channels + channel] =
                    rs_dot1(self->coefficients + (size_t)t * window, win,
                            window, first);
            }
        }
        emitted += required;
        *ahead += required * down - up;
        ++used;
    }
    *produced = emitted;
    return used;
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

    {
        const int up = self->up;
        const int down = self->down;
        const int channels = self->channels;
        const int tail = self->window - 1;
        /* The next output's position on the up-sampled grid relative to the
         * current input's first tick: next_output_tick - input_index * up.
         * An input emits the outputs whose tick falls in its own [0, up)
         * window, phase = that relative tick; each output moves it on by
         * `down`, each consumed input takes `up` off, so it stays in
         * [0, down) between inputs. The two 64-bit fields are written back
         * once, below. */
        int ahead = (int)(self->next_output_tick -
                          self->input_index * (uint64_t)up);

        while (consumed < input_frames) {
            int staged = input_frames - consumed;
            int used;
            int emitted = 0;
            if (staged > AUDIO_RESAMPLER_CHUNK) staged = AUDIO_RESAMPLER_CHUNK;
            for (int channel = 0; channel < channels; ++channel) {
                float* slot = self->work +
                              (size_t)channel * self->work_stride + tail;
                const float* source = input + (size_t)consumed * channels +
                                      channel;
                for (int i = 0; i < staged; ++i)
                    slot[i] = source[(size_t)i * channels];
            }
            if (down == 1) {
                used = rs_run_integer_up(
                    self, staged, output + (size_t)produced * channels,
                    output_capacity_frames - produced);
                emitted = used * up;
            } else if (up == 1) {
                used = rs_run_decimation(
                    self, staged, &ahead,
                    output + (size_t)produced * channels,
                    output_capacity_frames - produced, &emitted);
            } else {
                used = rs_run_rational(
                    self, staged, &ahead,
                    output + (size_t)produced * channels,
                    output_capacity_frames - produced, &emitted);
            }
            if (used > 0) {
                /* The newest window - 1 inputs become the next tail. */
                for (int channel = 0; channel < channels; ++channel) {
                    float* work = self->work +
                                  (size_t)channel * self->work_stride;
                    memmove(work, work + used,
                            (size_t)tail * sizeof(float));
                }
            }
            consumed += used;
            produced += emitted;
            if (used < staged) break;   /* output capacity ran out */
        }
        self->input_index += (uint64_t)consumed;
        self->next_output_tick =
            self->input_index * (uint64_t)up + (uint64_t)ahead;
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
    int up;
    if (!self) return -1;
    /* The kernel is symmetric with (filter_length - 1) / 2 taps of group
     * delay on the up-sampled grid, i.e. (filter_length - 1) / (2 * up) input
     * frames. That is exact for every rate pair whose up factor divides
     * ratio_max (8k/16k/48k in any combination); for a coprime pair such as
     * 32k -> 24k (up 3, down 4) it is fractional, so round to the nearest
     * frame instead of truncating (21.33 -> 21 either way there, but 21.5
     * would have read 21). */
    up = 2 * self->up;
    return (self->filter_length - 1 + up / 2) / up;
}
