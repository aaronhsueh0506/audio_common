/*
 * test_wav_writer_ubsan.c - UBSan probe for the WAV writer bounds contract.
 *
 * Writes {1.0f, -1.0f, NAN} through wav_write_float in the default (AEC
 * style) PCM16 mode and must run clean under -fsanitize=undefined
 * -fno-sanitize-recover=all (the `test-wav-ubsan` target sets the flags).
 * Converting 1.0f * 32768 + 0.5 directly to int16_t would be an out-of-range
 * float-to-integer conversion (C99 6.3.1.4p1); the writer rounds into an
 * int32_t and saturates, and replaces NaN by 0.0f first.
 *
 * Standalone and header-only, like test_wav_io.c.
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

    // 1.0f exercises the saturate, NAN the sanitize.
    wav_write_float(w, samples, 3);
    wav_close_write(w);

    unlink(path);
    printf("ALL PASS\n>>> PASS\n");
    return 0;
}
