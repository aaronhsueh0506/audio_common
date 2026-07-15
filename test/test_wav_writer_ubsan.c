/*
 * test_wav_writer_ubsan.c - UBSan probe for external re-review finding R01.
 *
 * Pre-fix, the AEC-style PCM16 quantizer in wav_write_float cast
 * `scaled +- 0.5f` straight to int16_t; for sample==+1.0f that's
 * (int16_t)32768.5f, an out-of-range float-to-integer conversion --
 * undefined behavior per C99 6.3.1.4p1. This tiny standalone TU writes
 * {1.0f, -1.0f, NAN} through wav_write_float in the default (AEC-style)
 * PCM16 mode and must run clean under -fsanitize=undefined
 * -fno-sanitize-recover=all (the Makefile's `test-wav-ubsan` target wires
 * those flags). Pre-fix, this reproducibly aborted with UBSan's
 * "outside the range of representable values" diagnostic on the +1.0f
 * sample; post-fix (int32_t round-then-saturate, and NaN sanitized to
 * 0.0f before conversion) it must complete and print PASS.
 *
 * Standalone by design (header-only, no archive link needed), same shape
 * as test_wav_io.c.
 */
#include "wav_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(void) {
    char path[64];
    snprintf(path, sizeof(path), "/tmp/audio_common_test_wav_ubsan_XXXXXX");
    int fd = mkstemp(path);
    if (fd < 0) {
        fprintf(stderr, "FAIL: mkstemp for UBSan probe file\n");
        return 1;
    }
    close(fd);

    const float samples[3] = { 1.0f, -1.0f, NAN };

    WavWriter* w = wav_open_write(path, 16000, 1);
    if (!w) {
        fprintf(stderr, "FAIL: wav_open_write for UBSan probe\n");
        unlink(path);
        return 1;
    }

    // The historically-UB call: pre-fix, sample==1.0f drove
    // (int16_t)32768.5f inside this call, which UBSan flagged and
    // -fno-sanitize-recover=all turned into an abort.
    wav_write_float(w, samples, 3);
    wav_close_write(w);

    unlink(path);
    printf("ALL PASS\n>>> PASS\n");
    return 0;
}
