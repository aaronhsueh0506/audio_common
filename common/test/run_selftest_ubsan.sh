#!/bin/sh
# run_selftest_ubsan.sh - UBSan probe for test/simd_selftest.c.
#
# Standalone script, not a Makefile target: simd_kernels.h is header-only, so no
# common archive link is required (same as the simd_kernels.h half of
# `make selftest`).
#
# Compiles+runs the SAME selftest source with
# -fsanitize=undefined -fno-sanitize-recover=all (host UBSan works fine for
# a single small translation unit like this one) and must exit 0 with no
# UBSan diagnostic printed -- any diagnostic here is a real finding (signed
# overflow, misaligned access the compiler considers UB, etc.), not noise.
#
# ASan is intentionally NOT wired here: it stays a Linux-CI item (the ASan
# runtime on the macOS development host is unusable). -fsanitize=undefined only.
#
# Usage: ./test/run_selftest_ubsan.sh   (from anywhere; cd's to this repo
# root itself via the script's own location, no assumption about caller cwd)

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
