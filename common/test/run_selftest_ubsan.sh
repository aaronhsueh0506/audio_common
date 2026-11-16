#!/bin/sh
# run_selftest_ubsan.sh - UBSan probe for test/simd_selftest.c.
#
# Run by `make test-selftest-ubsan` (part of `make test-full`); simd_kernels.h is header-only, so no
# common archive link is required. It compiles and runs the same selftest source with
# -fsanitize=undefined -fno-sanitize-recover=all and must exit 0 with no UBSan diagnostic; any
# diagnostic is a real finding (signed overflow, UB misaligned access, ...), not noise. ASan is
# intentionally not wired here.
#
# Usage: [CC=<compiler>] [BIN_DIR=<dir>] [SELFTEST_CFLAGS=<flags>] ./test/run_selftest_ubsan.sh
# (any cwd; it cd's to the common root via its own location, so a relative BIN_DIR is under it).
# The Makefile target passes its CC, its keyed BIN_DIR and the SIMD / EXTRA_CFLAGS selection.

set -e
cd "$(dirname "$0")/.."

CC=${CC:-cc}
BIN_DIR=${BIN_DIR:-bin}
BIN="$BIN_DIR/simd_selftest_ubsan"

mkdir -p "$BIN_DIR"

echo "--- common simd_selftest UBSan build ---"
# SELFTEST_CFLAGS is deliberately unquoted: it is a flag list.
# shellcheck disable=SC2086
"$CC" -Wall -Wextra -O2 -std=gnu99 -Iinclude -I. -ffp-contract=off -fno-math-errno $SELFTEST_CFLAGS \
    -fsanitize=undefined -fno-sanitize-recover=all \
    -o "$BIN" test/simd_selftest.c -lm

echo "--- common simd_selftest UBSan run ---"
"$BIN"

echo "UBSan probe: PASS (binary exited 0, no undefined-behavior diagnostic)"
