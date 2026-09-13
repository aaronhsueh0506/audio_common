#!/usr/bin/env bash
# Verify the AArch64 fast_sqrt contract with Linux code generation: one
# hardware FSQRT and no errno-checking call/tail-call to sqrtf.

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

if [ -n "${AARCH64_CC:-}" ]; then
    CROSS_CC="$AARCH64_CC"
elif [ -n "${CC:-}" ]; then
    CROSS_CC="$CC"
elif command -v aarch64-linux-gnu-gcc >/dev/null 2>&1; then
    CROSS_CC="aarch64-linux-gnu-gcc"
elif command -v clang >/dev/null 2>&1; then
    CROSS_CC="clang"
else
    echo "FATAL: no AArch64-capable compiler; set AARCH64_CC" >&2
    exit 1
fi

CC_VERSION="$($CROSS_CC --version 2>/dev/null | head -1 || true)"
TARGET_FLAGS=""
case "$CC_VERSION" in
    *clang*) TARGET_FLAGS="--target=aarch64-linux-gnu" ;;
    *)
        MACHINE="$($CROSS_CC -dumpmachine 2>/dev/null || true)"
        case "$MACHINE" in
            aarch64*linux*|arm64*linux*) ;;
            *)
                echo "FATAL: $CROSS_CC targets '$MACHINE', not AArch64 Linux; set AARCH64_CC to an aarch64-linux compiler" >&2
                exit 1
                ;;
        esac
        ;;
esac

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
#include "fast_math.h"
__attribute__((noinline)) float fast_sqrt_codegen_probe(float x) {
    return fast_sqrt(x);
}
EOF

for CPU in cortex-a53 cortex-a73; do
    ASM="$WORK/$CPU.s"
    # POLICY_FLAGS is an internal Makefile literal containing plain,
    # whitespace-separated compiler tokens; intentional word splitting here.
    $CROSS_CC $TARGET_FLAGS -mcpu="$CPU" -std=c99 -O2 $POLICY_FLAGS \
        -nostdinc -I"$WORK/include" -I"$AC_DIR/include" \
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
