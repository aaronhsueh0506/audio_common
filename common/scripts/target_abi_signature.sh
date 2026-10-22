#!/bin/sh
# Fingerprint effective compiler targets; exclude time-dependent predefines.
set -eu
sig_kind=v2
language=c
while [ "$#" -gt 0 ]; do
    case "$1" in
        --abi) sig_kind=abi1; shift ;;
        --cxx) language=c++; shift ;;
        *) break ;;
    esac
done
if [ "$#" -eq 0 ]; then
    printf '%s\n' unavailable
    exit 1
fi
abi_fields='__SIZEOF_POINTER__ __SIZEOF_SHORT__ __SIZEOF_INT__ __SIZEOF_LONG__ __SIZEOF_LONG_LONG__ __SIZEOF_FLOAT__ __SIZEOF_DOUBLE__ __SIZEOF_LONG_DOUBLE__ __SIZEOF_WCHAR_T__ __SIZEOF_SIZE_T__ __SIZEOF_PTRDIFF_T__ __BYTE_ORDER__ __FLOAT_WORD_ORDER__ __aarch64__ __arm__ __ARM_EABI__ __ARM_PCS __ARM_PCS_AAPCS64 __ARM_PCS_VFP __SOFTFP__ __ARM_SIZEOF_MINIMAL_ENUM'
fields=$abi_fields
if [ "$sig_kind" = v2 ]; then
    fields="$fields __thumb__ __ARM_ARCH __ARM_NEON __ARM_FP __ARM_FEATURE_FMA __GNUC__ __GNUC_MINOR__ __GNUC_PATCHLEVEL__ __clang_major__ __clang_minor__ __clang_patchlevel__"
fi
# Normalize equivalent GCC/Clang spellings of ARM hard-float AAPCS.
# The marker lets us discard any text a CFLAGS-supplied `-include` header
# emits ahead of the probe (e.g. a static inline definition).
marker=SIG_PROBE_BEGIN_7f3a9c
if ! target=$(printf '%s\n' \
    "$marker" \
    '#ifndef __FLOAT_WORD_ORDER__' \
    '#define __FLOAT_WORD_ORDER__ __BYTE_ORDER__' \
    '#endif' \
    '#if defined(__ARM_PCS_VFP) && !defined(__ARM_PCS)' \
    '#define __ARM_PCS 1' \
    '#endif' "$fields" | "$@" -E -P -x "$language" - 2>/dev/null); then
    # Actual builds reject this value; clean/help can still work without CC.
    printf '%s\n' unavailable
    exit 1
fi
target=${target##*"$marker"}
target=$(printf '%s' "$target" | tr '\n\t' '  ' | sed 's/[[:space:]][[:space:]]*/ /g; s/^ //; s/ $//')
case "$target" in
    '4 '*|'8 '*) ;;
    *) printf '%s\n' unavailable; exit 1 ;;
esac
printf '%s' "$target" | cksum | awk -v kind="$sig_kind" '{print kind "-" $1 "-" $2}'
