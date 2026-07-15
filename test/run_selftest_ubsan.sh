#!/bin/sh
# run_selftest_ubsan.sh - UBSan probe for test/simd_selftest.c
# (round-3 review B05, item 5).
#
# Standalone script rather than a new Makefile target: this repo's Makefile
# is being reworked concurrently by another task, so wiring a target in
# there risks a conflict; this script needs nothing from it (simd_kernels.h
# is header-only, no audio_common archive link required, same as the
# ordinary `make selftest` recipe's simd_kernels.h half).
#
# Compiles+runs the SAME selftest source with
# -fsanitize=undefined -fno-sanitize-recover=all (host UBSan works fine for
# a single small translation unit like this one) and must exit 0 with no
# UBSan diagnostic printed -- any diagnostic here is a real finding (signed
# overflow, misaligned access the compiler considers UB, etc.), not noise.
#
# ASan is intentionally NOT wired here: this development host's ASan runtime
# is broken (see project memory) -- ASan stays a Linux-CI item. -fsanitize=
# undefined only.
#
# Usage: ./test/run_selftest_ubsan.sh   (from anywhere; cd's to this repo
# root itself via the script's own location, no assumption about caller cwd)

set -e
cd "$(dirname "$0")/.."

CC=${CC:-cc}
BIN_DIR=bin
BIN="$BIN_DIR/simd_selftest_ubsan"

mkdir -p "$BIN_DIR"

echo "--- audio_common simd_selftest UBSan build ---"
"$CC" -Wall -Wextra -O2 -std=gnu99 -Iinclude -ffp-contract=off \
    -fsanitize=undefined -fno-sanitize-recover=all \
    -o "$BIN" test/simd_selftest.c -lm

echo "--- audio_common simd_selftest UBSan run ---"
"$BIN"

echo "UBSan probe: PASS (binary exited 0, no undefined-behavior diagnostic)"
