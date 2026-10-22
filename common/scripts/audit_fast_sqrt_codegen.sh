#!/usr/bin/env bash
# Verify ARM fast_sqrt codegen: hardware FSQRT/VSQRT without errno calls,
# or the correct sqrtf library path for ARM32 soft-float.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
POLICY_FLAGS="${FP_POLICY:-}"

case " $POLICY_FLAGS " in
    *" -fno-math-errno "*) ;;
    *)
        echo "FATAL: FP_POLICY must contain -fno-math-errno (got: '$POLICY_FLAGS')" >&2
        exit 1
        ;;
esac

# Compilers are word-split on purpose so a wrapper such as "ccache gcc" works.
cc_is_clang() {
    case "$($1 --version 2>/dev/null | head -1 || true)" in
        *clang*) return 0 ;;
    esac
    return 1
}
cc_machine() { $1 -dumpmachine 2>/dev/null || true; }
cc_targets_aarch64() {
    cc_is_clang "$1" && return 0
    case "$(cc_machine "$1")" in
        aarch64*linux*|arm64*linux*) return 0 ;;
    esac
    return 1
}
cc_targets_arm32() {
    cc_is_clang "$1" && return 1
    case "$(cc_machine "$1")" in
        arm*linux*gnueabi*) return 0 ;;
    esac
    return 1
}

# An armhf CC is a legitimate build setting, so CC alone never forces the
# AArch64 half to fail: an explicit AARCH64_CC must be a real AArch64 compiler,
# otherwise the first of CC, aarch64-linux-gnu-gcc, clang that can target
# AArch64 is used, and the half is reported as skipped only when none exists.
AARCH64_COMPILER=""
if [ -n "${AARCH64_CC:-}" ]; then
    if ! cc_targets_aarch64 "$AARCH64_CC"; then
        echo "FATAL: $AARCH64_CC targets '$(cc_machine "$AARCH64_CC")', not AArch64 Linux; set AARCH64_CC to an aarch64-linux compiler" >&2
        exit 1
    fi
    AARCH64_COMPILER="$AARCH64_CC"
else
    for CAND in "${CC:-}" aarch64-linux-gnu-gcc clang; do
        [ -n "$CAND" ] || continue
        command -v "${CAND%% *}" >/dev/null 2>&1 || continue
        if cc_targets_aarch64 "$CAND"; then
            AARCH64_COMPILER="$CAND"
            break
        fi
    done
fi

ARM32_COMPILER="${ARM32_CC:-}"
if [ -z "$ARM32_COMPILER" ] && [ -n "${CC:-}" ] && command -v "${CC%% *}" >/dev/null 2>&1 \
   && cc_targets_arm32 "$CC"; then
    ARM32_COMPILER="$CC"
fi
if [ -z "$ARM32_COMPILER" ] && command -v arm-linux-gnueabihf-gcc >/dev/null 2>&1; then
    ARM32_COMPILER=arm-linux-gnueabihf-gcc
fi

if [ -z "$AARCH64_COMPILER" ] && [ -z "$ARM32_COMPILER" ]; then
    echo "FATAL: no AArch64 or ARM32 compiler; set AARCH64_CC and/or ARM32_CC" >&2
    exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT INT TERM
mkdir -p "$WORK/include"

cat > "$WORK/include/stdint.h" <<'EOF'
typedef unsigned int uint32_t;
EOF
cat > "$WORK/include/math.h" <<'EOF'
float expf(float);
float floorf(float);
float logf(float);
float log10f(float);
float powf(float, float);
float sqrtf(float);
EOF
cat > "$WORK/probe.c" <<'EOF'
#include "utility/fast_math.h"
__attribute__((noinline)) float fast_sqrt_codegen_probe(float x) {
    return fast_sqrt(x);
}
EOF

TARGET_FLAGS=""
if [ -n "$AARCH64_COMPILER" ] && cc_is_clang "$AARCH64_COMPILER"; then
    TARGET_FLAGS="--target=aarch64-linux-gnu"
fi

if [ -n "$AARCH64_COMPILER" ]; then
    for CPU in cortex-a53 cortex-a73; do
        ASM="$WORK/$CPU.s"
        # POLICY_FLAGS is an internal Makefile literal containing plain,
        # whitespace-separated compiler tokens; intentional word splitting here.
        $AARCH64_COMPILER $TARGET_FLAGS -mcpu="$CPU" -std=c99 -O2 $POLICY_FLAGS \
            -nostdinc -I"$WORK/include" -I"$AC_DIR/include" -I"$AC_DIR" \
            -S "$WORK/probe.c" -o "$ASM"

        if ! grep -Eiq '(^|[[:space:]])fsqrt([[:space:]]|\.)' "$ASM"; then
            echo "FATAL: $CPU probe contains no FSQRT" >&2
            exit 1
        fi
        if grep -Eiq '(^|[^[:alnum:]_])_?sqrtf([^[:alnum:]_]|$)' "$ASM"; then
            echo "FATAL: $CPU probe still references sqrtf (errno fallback/call remains)" >&2
            exit 1
        fi
        echo "fast_sqrt codegen [$CPU]: PASS (FSQRT present, no sqrtf reference)"
    done
else
    echo "SKIP AArch64 fast_sqrt codegen: no AArch64 compiler (set AARCH64_CC)"
fi

if [ -n "$ARM32_COMPILER" ]; then
    ARM32_TARGET=""
    case "$($ARM32_COMPILER --version 2>/dev/null | head -1 || true)" in
        *clang*) ARM32_TARGET="--target=arm-linux-gnueabihf" ;;
    esac
    for CPU in cortex-a53 cortex-a73; do
        ASM="$WORK/$CPU-arm32.s"
        $ARM32_COMPILER $ARM32_TARGET -mcpu="$CPU" -mfpu=neon-vfpv4 \
            -mfloat-abi=hard -std=c99 -O2 $POLICY_FLAGS \
            -nostdinc -I"$WORK/include" -I"$AC_DIR/include" -I"$AC_DIR" \
            -S "$WORK/probe.c" -o "$ASM"
        if ! grep -Eiq '(^|[[:space:]])vsqrt[a-z]*\.f32([[:space:]]|$)' "$ASM"; then
            echo "FATAL: $CPU ARM32 probe contains no VSQRT.F32" >&2
            exit 1
        fi
        if grep -Eiq '(^|[^[:alnum:]_])_?sqrtf([^[:alnum:]_]|$)' "$ASM"; then
            echo "FATAL: $CPU ARM32 probe still references sqrtf" >&2
            exit 1
        fi
        echo "fast_sqrt codegen [$CPU ARM32]: PASS (VSQRT.F32 present, no sqrtf reference)"
    done
    ASM="$WORK/arm32-soft.s"
    $ARM32_COMPILER $ARM32_TARGET -march=armv7-a -mfloat-abi=soft \
        -std=c99 -O2 $POLICY_FLAGS -nostdinc -I"$WORK/include" \
        -I"$AC_DIR/include" -I"$AC_DIR" -S "$WORK/probe.c" -o "$ASM"
    if ! grep -Eiq '(^|[^[:alnum:]_])_?sqrtf([^[:alnum:]_]|$)' "$ASM" || \
       grep -Eq '__aeabi_fdiv' "$ASM"; then
        echo "FATAL: ARM32 soft-float probe must use sqrtf, not Newton divisions" >&2
        exit 1
    fi
    echo "fast_sqrt codegen [ARM32 soft-float]: PASS (sqrtf library call, no Newton divisions)"
else
    echo "SKIP ARM32 fast_sqrt codegen: set ARM32_CC to an ARM32 compiler"
fi
