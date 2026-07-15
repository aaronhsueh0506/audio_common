/*
 * test_wav_writer_nr_style.c - runs the shared writer special-values corpus
 * (external re-review finding R01) against WAV_IO_WRITER_NR instead of the
 * default WAV_IO_WRITER_AEC that test_wav_io.c exercises.
 *
 * WAV_IO_WRITER_STYLE is a compile-time knob (see wav_io.h's file header
 * doc) -- it can't be flipped at runtime within one already-compiled TU, so
 * this is a small separate translation unit, built with
 * -DWAV_IO_WRITER_STYLE=WAV_IO_WRITER_NR, that #includes the SAME canonical
 * wav_io.h and the SAME shared test body (test_wav_writer_common.h) as
 * test_wav_io.c. This is intentionally the only thing this TU tests --
 * test_wav_io.c's exhaustive reader negative-corpus coverage is not
 * duplicated here (the writer style knob has no effect on the reader).
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
