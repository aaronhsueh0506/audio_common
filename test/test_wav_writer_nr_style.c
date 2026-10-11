/*
 * test_wav_writer_nr_style.c - runs the shared writer special-values test
 * (test_wav_writer_common.h) with WAV_IO_WRITER_NR; test_wav_io.c runs it
 * with the default WAV_IO_WRITER_AEC.
 *
 * WAV_IO_WRITER_STYLE is a compile-time setting, so this is a separate
 * translation unit built with -DWAV_IO_WRITER_STYLE=WAV_IO_WRITER_NR. The
 * reader tests are not repeated: the style does not affect the reader.
 */
#include "wav_io.h"
#include "test_wav_writer_common.h"

#include <stdio.h>

int main(void) {
    int fail = test_wav_writer_special_values_corpus();

    if (fail) {
        printf(">>> FAIL\n");
        return 1;
    }
    printf("ALL PASS\n>>> PASS\n");
    return 0;
}
