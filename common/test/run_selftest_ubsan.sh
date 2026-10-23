#!/bin/sh
# run_selftest_ubsan.sh - UBSan probe for test/simd_selftest.c.
#
# Standalone script, not a Makefile target: simd_kernels.h is header-only, so no common archive link
# is required. It compiles and runs the same selftest source with
# -fsanitize=undefined -fno-sanitize-recover=all and must exit 0 with no UBSan diagnostic; any
# diagnostic is a real finding (signed overflow, UB misaligned access, ...), not noise. ASan is
# intentionally not wired here.
#
# Usage: ./test/run_selftest_ubsan.sh   (any cwd; it cd's to the repo root via its own location)

set -e
cd "$(dirname "$0")/.."

CC=${CC:-cc}
BIN_DIR=bin
BIN="$BIN_DIR/simd_selftest_ubsan"

mkdir -p "$BIN_DIR"

echo "--- common simd_selftest UBSan build ---"
"$CC" -Wall -Wextra -O2 -std=gnu99 -Iinclude -I. -ffp-contract=off -fno-math-errno \
    -fsanitize=undefined -fno-sanitize-recover=all \
    -o "$BIN" test/simd_selftest.c -lm

echo "--- common simd_selftest UBSan run ---"
"$BIN"

echo "UBSan probe: PASS (binary exited 0, no undefined-behavior diagnostic)"
