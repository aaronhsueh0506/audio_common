/* test_tmp_path.h - mkstemp template under $TMPDIR (the Makefile points it at the build dir), else /tmp. */
#ifndef TEST_TMP_PATH_H
#define TEST_TMP_PATH_H

#include <stdio.h>
#include <stdlib.h>

#define TEST_TMP_PATH_MAX 512

/* An overflowing template is truncated, so the caller's mkstemp() fails and its setup check reports it. */
static inline void test_tmp_template(char* out, size_t cap, const char* stem) {
    const char* dir = getenv("TMPDIR");
    if (!dir || !*dir) dir = "/tmp";
    snprintf(out, cap, "%s/%s_XXXXXX", dir, stem);
}

#endif /* TEST_TMP_PATH_H */
