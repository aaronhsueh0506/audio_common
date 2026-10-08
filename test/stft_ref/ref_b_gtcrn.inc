/* Audio_ALG/AINR/GTCRN/gtcrn_process.c: df_common_make_root_hann, df_common_analysis, df_common_synthesis */
#if defined(__aarch64__) && defined(__ARM_NEON) && \
    !defined(SIMD_KERNELS_FORCE_SCALAR)
#include <arm_neon.h>
#define DF_COMMON_HAVE_NEON 1
#else
#define DF_COMMON_HAVE_NEON 0
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline void df_common_make_root_hann(float *window, int win_len) {
    for (int i = 0; i < win_len; ++i) {
        window[i] = sqrtf(0.5f - 0.5f * cosf(
            2.0f * (float)M_PI * (float)i / (float)win_len));
    }
}

static inline void df_common_analysis(FftHandle* fft, float *analysis_buf,
                                      const float *window,
                                      float *scratch_time,
                                      Complex *scratch_freq,
                                      const float *new_samples, int n_fft,
                                      int hop, float norm,
                                      float out[][2]) {
    memmove(analysis_buf, analysis_buf + hop,
            (size_t)(n_fft - hop) * sizeof(float));
    memcpy(analysis_buf + n_fft - hop, new_samples,
           (size_t)hop * sizeof(float));
    {
        int i = 0;
#if DF_COMMON_HAVE_NEON
        for (; i + 4 <= n_fft; i += 4) {
            vst1q_f32(scratch_time + i,
                      vmulq_f32(vld1q_f32(analysis_buf + i),
                                vld1q_f32(window + i)));
        }
#endif
        for (; i < n_fft; ++i) scratch_time[i] = analysis_buf[i] * window[i];
    }
    fft_forward(fft, scratch_time, scratch_freq);
    {
        int bins = n_fft / 2 + 1;
        for (int k = 0; k < bins; ++k) {
            out[k][0] = scratch_freq[k].r * norm;
            out[k][1] = scratch_freq[k].i * norm;
        }
    }
}

static inline void df_common_synthesis(FftHandle* fft,
                                       float *synthesis_buf,
                                       const float *window,
                                       float *scratch_time,
                                       Complex *scratch_freq,
                                       const float spec[][2],
                                       int n_fft, int hop, float inv_norm,
                                       float *output) {
    int bins = n_fft / 2 + 1;
    for (int k = 0; k < bins; ++k) {
        scratch_freq[k].r = spec[k][0] * inv_norm;
        scratch_freq[k].i = spec[k][1] * inv_norm;
    }
    fft_inverse(fft, scratch_freq, scratch_time);
    {
        int i = 0;
#if DF_COMMON_HAVE_NEON
        for (; i + 4 <= n_fft; i += 4) {
            vst1q_f32(scratch_time + i,
                      vmulq_f32(vld1q_f32(scratch_time + i),
                                vld1q_f32(window + i)));
        }
        for (i = 0; i + 4 <= hop; i += 4) {
            vst1q_f32(output + i,
                      vaddq_f32(vld1q_f32(synthesis_buf + i),
                                vld1q_f32(scratch_time + i)));
        }
#else
        for (; i < n_fft; ++i) scratch_time[i] *= window[i];
#endif
#if DF_COMMON_HAVE_NEON
        for (; i < hop; ++i)
            output[i] = synthesis_buf[i] + scratch_time[i];
#else
        for (i = 0; i < hop; ++i)
            output[i] = synthesis_buf[i] + scratch_time[i];
#endif
    }
    memcpy(synthesis_buf, scratch_time + hop,
           (size_t)(n_fft - hop) * sizeof(float));
    memset(synthesis_buf + n_fft - hop, 0, (size_t)hop * sizeof(float));
}
