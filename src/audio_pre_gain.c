#include "audio_pre_gain.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mem_align.h"
#include "simd_kernels.h"   /* sk_scale_f32 -- and, with it, the one shared
                             * SK_HAVE_NEON gate, replacing the private copy
                             * of that predicate this file used to carry. */

struct AudioPreGain {
    float gain_db;
    float linear_gain;
    int is_static;
};

static int audio_pre_gain_convert(float gain_db, float* linear)
{
    float value;
    if (!linear || !isfinite(gain_db)) return 0;
    value = powf(10.0f, gain_db / 20.0f);
    if (!isfinite(value) || value < 0.0f) return 0;
    *linear = value;
    return 1;
}

size_t audio_pre_gain_get_mem_size(void)
{
    size_t bytes = ck_field_size(0, 1, sizeof(AudioPreGain));
    return MEM_SIZE_INVALID(bytes) ? 0 : bytes;
}

AudioPreGain* audio_pre_gain_init(void* mem,
                                  size_t mem_size,
                                  float gain_db)
{
    AudioPreGain* self;
    float linear;
    size_t need = audio_pre_gain_get_mem_size();
    if (!mem || !MEM_IS_ALIGNED16(mem) || need == 0 || mem_size < need ||
        !audio_pre_gain_convert(gain_db, &linear)) {
        return NULL;
    }
    self = (AudioPreGain*)mem;
    memset(self, 0, sizeof(*self));
    self->gain_db = gain_db;
    self->linear_gain = linear;
    self->is_static = 1;
    return self;
}

AudioPreGain* audio_pre_gain_create(float gain_db)
{
    AudioPreGain* self;
    float linear;
    if (!audio_pre_gain_convert(gain_db, &linear)) return NULL;
    self = (AudioPreGain*)calloc(1, sizeof(*self));
    if (!self) return NULL;
    self->gain_db = gain_db;
    self->linear_gain = linear;
    return self;
}

void audio_pre_gain_destroy(AudioPreGain* self)
{
    if (!self || self->is_static) return;
    free(self);
}

int audio_pre_gain_set_db(AudioPreGain* self, float gain_db)
{
    float linear;
    if (!self || !audio_pre_gain_convert(gain_db, &linear)) return -1;
    self->gain_db = gain_db;
    self->linear_gain = linear;
    return 0;
}

float audio_pre_gain_get_db(const AudioPreGain* self)
{
    return self ? self->gain_db : NAN;
}

float audio_pre_gain_get_linear(const AudioPreGain* self)
{
    return self ? self->linear_gain : NAN;
}

int audio_pre_gain_process(const AudioPreGain* self,
                           const float* input,
                           float* output,
                           int n_samples)
{
    if (!self || !input || !output || n_samples < 0) return -1;
    sk_scale_f32(output, input, self->linear_gain, n_samples);
    return 0;
}
