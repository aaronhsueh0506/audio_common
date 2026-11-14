/** audio_stft.h - sqrt-Hann STFT analysis and overlap-add synthesis, 50% overlap.
 * AcStft turns hop samples into windowed rFFT frames; AcIstft turns a spectrum into hop samples
 * (inverse rFFT, window, overlap-add). Window: periodic sqrt-Hann, sqrtf(0.5f * (1 - cosf(2*pi*k/N))).
 * Transforms use fft_wrapper.h (forward unnormalised, inverse includes 1/N). spec_scale / ispec_scale multiply
 * every bin after the forward / before the inverse FFT; exactly 1.0f is skipped (bins are exact moves).
 * Synthesis (AcOlaMode):
 *   AC_OLA_ACCUMULATE  acc += ifft * w; emit acc[0..hop). One hop out per frame in.
 *   AC_OLA_ENV_NORM    also accumulates w*w and emits acc / max(env, 1e-11f), as torch.istft does;
 *                      the first frame emits nothing.
 * Analysis framing (AcStftConfig.center):
 *   0  every push yields one frame over the last fft_size samples (zero history before the start).
 *   1  torch.stft(center=True, pad_mode='reflect') framing: the first push yields no frame, the second two,
 *      later pushes one each; ac_stft_flush_planar adds the trailing reflect frame.
 * An identity chain delays the signal by hop = fft_size / 2 samples.
 * Memory: static only. size = get_mem_size(cfg); init(mem, size, cfg) takes a 16-byte-aligned block; no destroy.
 * The FFT handle and window table are carved from the block (cfg.fft / cfg.window NULL) or borrowed; a borrowed
 * handle may be shared by objects that run sequentially, and a borrowed table must outlive the object.
 * One AcStftConfig describes an AcStft, an AcIstft, or both; each reads only the fields marked for it. */
#ifndef AUDIO_STFT_H
#define AUDIO_STFT_H

#include <stddef.h>

#include "fft_wrapper.h"
#include "audio_window.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    AC_OLA_ACCUMULATE = 0,
    AC_OLA_ENV_NORM   = 1
} AcOlaMode;

typedef struct AcStftConfig {
    int          fft_size;     /* both: power of two in [16, 8192]; hop = fft_size / 2 */
    int          center;       /* AcStft: 0 or 1 (analysis framing) */
    AcOlaMode    ola;          /* AcIstft: synthesis mode */
    float        spec_scale;   /* AcStft: bin multiplier; 1.0f = none */
    float        ispec_scale;  /* AcIstft: bin multiplier; 1.0f = none */
    FftHandle*   fft;          /* both: borrowed fft_size handle, or NULL to carve one */
    const float* window;       /* both: borrowed fft_size table, or NULL to carve one */
} AcStftConfig;

/** Fill config with the defaults: center 0, AC_OLA_ACCUMULATE, scales 1.0f, own FFT and window. */
void ac_stft_config_default(AcStftConfig* cfg, int fft_size);

typedef struct AcStft AcStft;

/** Bytes for ac_stft_init; 0 for an invalid config. */
size_t  ac_stft_get_mem_size(const AcStftConfig* cfg);
/** NULL on an invalid config, a short or misaligned block, or a borrowed handle of another size. */
AcStft* ac_stft_init(void* mem, size_t mem_size, const AcStftConfig* cfg);
/** Clears the sample history and the frame counter; configuration and tables stay. */
void    ac_stft_reset(AcStft* s);
/** The window table in use (n floats), for sharing with an AcIstft. */
const float* ac_stft_window(const AcStft* s);

/** Push hop = fft_size / 2 samples. Returns the frames written (0..2); frame f lands at out + f * n_freqs
 * (n_freqs = fft_size / 2 + 1). `out` needs room for 2 frames when center = 1, else 1. */
int ac_stft_push(AcStft* s, const float* hop_in, Complex* out);
/** Same as ac_stft_push with split output: frame f at re/im + f * n_freqs. */
int ac_stft_push_planar(AcStft* s, const float* hop_in, float* re, float* im);

/** Windowed rFFT of one caller-supplied fft_size-sample frame; touches no history. */
void ac_stft_frame_planar(AcStft* s, const float* frame, float* re, float* im);

/** center = 1 only: the trailing reflect-padded frame, written to re/im. Returns 1, or 0 when fewer than two hops
 * were pushed or center = 0. Call ac_stft_reset afterwards. */
int ac_stft_flush_planar(AcStft* s, float* re, float* im);

typedef struct AcIstft AcIstft;

size_t   ac_istft_get_mem_size(const AcStftConfig* cfg);
AcIstft* ac_istft_init(void* mem, size_t mem_size, const AcStftConfig* cfg);
/** Clears the overlap state and the frame counter. */
void     ac_istft_reset(AcIstft* s);
/** Snapshot of the live synthesis config (fields of the other object keep defaults). Returns 0, or -1 for NULL. */
int ac_istft_get_config(const AcIstft* s, AcStftConfig* cfg);

/** Inverse-transform one spectrum (n_freqs bins, not modified) and write the finished samples to `out` (room for hop floats).
 * Returns the number written: hop, or 0 for the first frame of AC_OLA_ENV_NORM. */
int ac_istft_push(AcIstft* s, const Complex* spec, float* out);
int ac_istft_push_planar(AcIstft* s, const float* re, const float* im, float* out);

/** AC_OLA_ENV_NORM only: the hop samples still in the overlap state, envelope-normalised, written to out.
 * Returns hop, or 0 when no frame was pushed or the mode is AC_OLA_ACCUMULATE. The state is not advanced; call ac_istft_reset afterwards. */
int ac_istft_flush(AcIstft* s, float* out);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_STFT_H */
