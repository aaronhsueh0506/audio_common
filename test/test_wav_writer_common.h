/*
 * test_wav_writer_common.h - shared writer special-values corpus test body,
 * included by BOTH test_wav_io.c (default WAV_IO_WRITER_AEC, via the
 * canonical wav_io.h's own unset-knob default) and
 * test_wav_writer_nr_style.c (which #defines WAV_IO_WRITER_STYLE to
 * WAV_IO_WRITER_NR before including wav_io.h) -- external re-review finding
 * R01 regression coverage for the PCM16 quantizer's undefined behavior at
 * the +-1.0f boundary (AEC style specifically: `(int16_t)32768.5f` is an
 * out-of-range float-to-integer conversion, reproduced under UBSan) and the
 * unsanitized NaN/+-Inf path (both styles: NaN fails every ordered
 * comparison, so the pre-fix [-1,1] clamp silently let it through to the
 * same undefined cast).
 *
 * ONE function body covers both writer styles -- the expected int16 values
 * differ (round-half-away-from-zero + saturate vs plain truncation +
 * defensive saturate), selected at compile time via WAV_IO_WRITER_STYLE,
 * which is already defined by the time this header is included (must be
 * included AFTER "wav_io.h").
 */
#ifndef TEST_WAV_WRITER_COMMON_H
#define TEST_WAV_WRITER_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <unistd.h>

/* Returns 1 if any check failed (each failure is printed to stderr as it
 * happens), 0 if every check passed. */
static inline int test_wav_writer_special_values_corpus(void) {
    int fail = 0;
#define WCHECK(cond, msg) do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
            fail = 1; \
        } \
    } while (0)

    const int n = 9;
    float in_samples[9];
    in_samples[0] = 1.0f;                          // exactly +full-scale
    in_samples[1] = -1.0f;                          // exactly -full-scale
    in_samples[2] = nextafterf(1.0f, 0.0f);         // largest value < 1.0f
    in_samples[3] = nextafterf(-1.0f, 0.0f);        // smallest value > -1.0f
    in_samples[4] = NAN;
    in_samples[5] = INFINITY;
    in_samples[6] = -INFINITY;
    in_samples[7] = 0.5f;
    in_samples[8] = -0.5f;

#if WAV_IO_WRITER_STYLE == WAV_IO_WRITER_AEC
    /* Round-half-away-from-zero (`sample*32768.0f`, then +-0.5f before
     * truncating into a wider int32_t), THEN saturate into int16_t's
     * range. Pre-fix, +-1.0f produced (int16_t)+-32768.5f -- undefined
     * behavior; post-fix these saturate to the values a defined round
     * would give anyway. Values re-derived from the exact production
     * formula via a standalone scratch computation (not hand-guessed):
     * nextafterf(1.0f,0)*32768+0.5 truncates to 32768 pre-saturate (still
     * clamps to 32767); nextafterf(-1.0f,0) truncates to exactly -32768
     * (no saturation needed, already in range). */
    const int16_t expect[9] = {
        32767, -32768, 32767, -32768, 0, 0, 0, 16384, -16384
    };
#else
    /* Plain truncation (`sample*32767.0f`, no rounding). The saturate added
     * by the R01 fix is pure defense here -- the preceding [-1,1] clamp and
     * non-finite sanitize already bound the input, so it's never actually
     * triggered for any of these (all land in [-32767, 32767] already). */
    const int16_t expect[9] = {
        32767, -32767, 32766, -32766, 0, 0, 0, 16383, -16383
    };
#endif

    char path[64];
    snprintf(path, sizeof(path), "/tmp/audio_common_test_wav_writer_special_XXXXXX");
    int fd = mkstemp(path);
    WCHECK(fd >= 0, "setup: mkstemp for writer special-values corpus");
    if (fd < 0) return fail;
    close(fd);  // wav_open_write will fopen() the same path

    WavWriter* w = wav_open_write(path, 16000, 1);
    WCHECK(w != NULL, "wav_open_write must succeed for special-values corpus");
    if (w) {
        WCHECK(w->info.is_float == 0, "this corpus is written in PCM16 mode");
        wav_write_float(w, in_samples, n);
        WCHECK(w->nonfinite_sanitized == 3,
               "exactly 3 non-finite samples (NAN/+Inf/-Inf) must be sanitized to 0.0f");
        WCHECK(w->write_error == 0,
               "a normal-sized write to a fresh temp file must not report write_error");
        WCHECK(w->samples_written == (uint64_t)n,
               "samples_written must count all 9 samples of the corpus");
        wav_close_write(w);

        WavReader* r = wav_open_read(path);
        WCHECK(r != NULL, "wav_open_read must read back the special-values corpus file");
        if (r) {
            float out[9];
            int got = wav_read_float(r, out, n);
            WCHECK(got == n, "must read back all 9 samples");
            for (int i = 0; i < n; i++) {
                float expect_f = (float)expect[i] / 32768.0f;
                char msg[112];
                snprintf(msg, sizeof(msg),
                         "corpus sample %d must dequantize to the documented expected int16 value", i);
                WCHECK(out[i] == expect_f, msg);
            }
            wav_close_read(r);
        }
    }
    unlink(path);
#undef WCHECK
    return fail;
}

#endif // TEST_WAV_WRITER_COMMON_H
