/**
 * audio_stft.h - sqrt-Hann STFT analysis and overlap-add synthesis, 50% overlap.
 *
 * One object pair for the framing every pipeline and NN pre/post stage needs:
 *
 *   AcStft   hop samples in  -> windowed rFFT frame(s) out
 *   AcIstft  spectrum in     -> inverse rFFT, window, overlap-add -> hop samples out
 *
 * Window: periodic sqrt-Hann, w[k] = sqrtf(0.5f * (1.0f - cosf(2.0f * pi * k / N))),
 * one formula for both objects (ac_window_build).
 *
 * Transforms go through fft_wrapper.h: the forward FFT is unnormalised and the
 * inverse includes 1/N. spec_scale / ispec_scale are the only additional
 * scalings (normalised NN spectra): spec_scale multiplies every output bin of
 * the forward FFT, ispec_scale every input bin before the inverse FFT. A scale
 * of exactly 1.0f is skipped, so the bins are exact moves (NaN payloads and
 * signed zeros included).
 *
 * Synthesis modes (AcOlaMode):
 *   AC_OLA_ACCUMULATE  acc += ifft * w; emit acc[0..hop); shift; zero the tail.
 *                      One hop out per frame in, from the first frame.
 *   AC_OLA_ENV_NORM    additionally accumulates w*w and emits acc / max(env, 1e-11f),
 *                      the same envelope normalisation as torch.istft. The first
 *                      frame lies in the trimmed half window and emits nothing.
 *
 * Analysis framing (AcStftConfig.center):
 *   0  every push yields exactly one frame, from the first push, over the last
 *      fft_size samples (zero history before the stream start).
 *   1  torch.stft(center=True, pad_mode='reflect') framing: the first push yields
 *      no frame, the second yields two (the reflect-padded frame 0 and frame 1),
 *      later pushes one each; ac_stft_flush adds the trailing reflect frame.
 *
 * Latency: an identity chain (analysis -> synthesis) in either framing delays
 * the signal by hop = fft_size / 2 samples (ac_stft_latency_samples).
 *
 * Memory: static only. size = get_mem_size(cfg); init(mem, size, cfg) with a
 * 16-byte aligned block; no heap, no destroy. The FFT handle and the window
 * table are either carved from the block (cfg.fft / cfg.window NULL) or
 * borrowed: a borrowed handle may be shared by several objects (their calls
 * are sequential, never concurrent) and a borrowed table must stay unchanged
 * and alive as long as the object. One AcStftConfig describes an AcStft, an
 * AcIstft, or both; each object reads only the fields marked for it.
 */
#ifndef AUDIO_STFT_H
#define AUDIO_STFT_H

#include <stddef.h>

#include "fft_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    AC_OLA_ACCUMULATE = 0,
    AC_OLA_ENV_NORM   = 1
} AcOlaMode;

typedef struct AcStftConfig {
    int          fft_size;     /* both: power of two in [16, 8192]; hop = fft_size / 2 */
    int          center;       /* AcStft: 0 or 1, see "Analysis framing"               */
    AcOlaMode    ola;          /* AcIstft: synthesis mode                              */
    float        spec_scale;   /* AcStft: bin multiplier; 1.0f = none                  */
    float        ispec_scale;  /* AcIstft: bin multiplier; 1.0f = none                 */
    FftHandle*   fft;          /* both: borrowed fft_size handle, or NULL to carve one */
    const float* window;       /* both: borrowed fft_size table, or NULL to carve one  */
} AcStftConfig;

/** Fill config with the defaults: center 0, AC_OLA_ACCUMULATE, scales 1.0f, own FFT and window. */
void ac_stft_config_default(AcStftConfig* cfg, int fft_size);

/** w[0..n) = periodic sqrt-Hann (formula in the file header). */
void ac_window_build(float* w, int n);

/** Samples of delay of an identity analysis -> synthesis chain; -1 for an invalid config. */
int ac_stft_latency_samples(const AcStftConfig* cfg);

/* ------------------------------ analysis ------------------------------ */

typedef struct AcStft AcStft;

/** Bytes for ac_stft_init; 0 for an invalid config. */
size_t  ac_stft_get_mem_size(const AcStftConfig* cfg);
/** NULL on an invalid config, a short or misaligned block, or a borrowed handle of another size. */
AcStft* ac_stft_init(void* mem, size_t mem_size, const AcStftConfig* cfg);
/** Clears the sample history and the frame counter; configuration and tables stay. */
void    ac_stft_reset(AcStft* s);
/** The window table in use (n floats), for sharing with an AcIstft. */
const float* ac_stft_window(const AcStft* s);

/**
 * Push hop = fft_size / 2 samples. Returns the number of frames written, 0..2;
 * frame f lands at out + f * n_freqs (n_freqs = fft_size / 2 + 1). `out` needs
 * room for 2 frames when center = 1, 1 frame otherwise.
 */
int ac_stft_push(AcStft* s, const float* hop_in, Complex* out);
/** Same as ac_stft_push with split output: frame f at re/im + f * n_freqs. */
int ac_stft_push_planar(AcStft* s, const float* hop_in, float* re, float* im);

/** Windowed rFFT of one caller-supplied fft_size-sample frame; touches no history. */
void ac_stft_frame(AcStft* s, const float* frame, Complex* out);
void ac_stft_frame_planar(AcStft* s, const float* frame, float* re, float* im);

/**
 * center = 1 only: the trailing reflect-padded frame, written to out (one
 * frame). Returns 1, or 0 when fewer than two hops were pushed or center = 0.
 * Restart with ac_stft_reset afterwards.
 */
int ac_stft_flush(AcStft* s, Complex* out);
int ac_stft_flush_planar(AcStft* s, float* re, float* im);

/* ----------------------------- synthesis ------------------------------ */

typedef struct AcIstft AcIstft;

size_t   ac_istft_get_mem_size(const AcStftConfig* cfg);
AcIstft* ac_istft_init(void* mem, size_t mem_size, const AcStftConfig* cfg);
/** Clears the overlap state and the frame counter. */
void     ac_istft_reset(AcIstft* s);
const float* ac_istft_window(const AcIstft* s);

/**
 * Inverse-transform one spectrum (n_freqs bins; the input is not modified)
 * and write the finished samples to `out`. Returns the number written: hop,
 * or 0 for the first frame of AC_OLA_ENV_NORM. `out` needs room for hop floats.
 */
int ac_istft_push(AcIstft* s, const Complex* spec, float* out);
int ac_istft_push_planar(AcIstft* s, const float* re, const float* im, float* out);

/**
 * AC_OLA_ENV_NORM only: the hop samples still held in the overlap state,
 * envelope-normalised, written to out. Returns hop, or 0 when no frame was
 * pushed or the mode is AC_OLA_ACCUMULATE (whose tail is incomplete). The
 * state is not advanced; restart with ac_istft_reset.
 */
int ac_istft_flush(AcIstft* s, float* out);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_STFT_H */
