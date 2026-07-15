#!/usr/bin/env bash
# audit_fp_contract.sh — FP-contraction policy verification (round-3 review B04).
#
# The unified policy (audio_common/NR/c_impl/AEC/c_impl/Audio_ALG/pipelines
# Makefiles, all four): EVERY TU those Makefiles compile — our own sources
# AND the vendored KISS/NE10 C and C++ TUs alike — builds with
# -ffp-contract=off, appended LAST on every compile line so nothing
# (EXTRA_CFLAGS, a BACKEND-conditional append, ...) can override it. See
# each Makefile's own "FP-contraction policy" comment block for how the flag
# is positioned and how EXTRA_CFLAGS=-Ofast/-ffast-math/-ffp-contract=<x> is
# rejected at parse time.
#
# This script is the DISASSEMBLY-level proof that the flag actually bites:
# for a fixed list of TUs that are supposed to be genuinely scalar (no
# explicit FMA-class request anywhere in their own source), it disassembles
# the keyed object this repo's own build produced and FAILS if any
# fmadd/fmsub/fnmadd/fnmsub/fmla/fmls instruction shows up — the signature
# of the COMPILER choosing, on its own, to fuse a plain `a*b+c`/`a*b-c`
# source expression into a fused multiply-accumulate, which is exactly what
# -ffp-contract=off exists to forbid.
#
# --- Scoping rationale (why some TUs are EXEMPT, not scalar-audited) -------
#
# -ffp-contract only controls whether the COMPILER may fuse a source-level
# `a*b+c` expression on its own initiative. It has no effect on — and no
# business vetoing — an EXPLICIT fusion request the source code itself
# makes: a direct call to fmaf()/fma(), or an explicit NEON fused-multiply
# intrinsic (vfmaq_f32/vfmsq_f32 and friends). Those must disassemble to a
# real fmadd/fmla either way (that's the whole point of asking for one), so
# a TU containing one is placed on the EXEMPT list below with the exact
# source-level reason, and is reported (hit count shown) but never fails
# the gate.
#
# This scoping was derived empirically, not assumed from filenames — every
# candidate object was actually disassembled and its source actually grepped
# before being classified. Two findings worth flagging explicitly, because
# they run against the naive guess:
#
#   1. fft_wrapper.c / fft_wrapper_ne10.c (audio_common's OWN FFT wrappers,
#      one per backend) are NOT purely scalar reference code, on this
#      arm64 host: fft_power()'s scalar tail loop calls fmaf(re, re, im*im)
#      EXPLICITLY (see that function's own in-file comment — it's
#      deliberately mirroring the vectorised path bit-for-bit), and both
#      files additionally carry a `#if defined(__ARM_NEON) &&
#      defined(__aarch64__)` block with explicit vfmaq_f32/vmulq_f32 calls.
#      Both are EXPLICIT fusion requests, unconditionally present regardless
#      of -ffp-contract — so both objects are EXEMPT, even though they hold
#      "our own", not vendored, code and were named in the original audit
#      brief as scalar TUs to check. (Confirmed: `grep fmaf(` finds exactly
#      these two files' fft_power(), nowhere else in the audit list.)
#
#   2. NE10_rfft_float32.neonintrinsic.c — despite its ".neonintrinsic.c"
#      name suggesting "explicit-FMA, exempt like the AEC kernel headers" —
#      contains NO vfmaq_f32/vfmsq_f32 call anywhere (`grep -oE
#      'v[a-z0-9_]*q?_f32' ` on it turns up vmulq_f32/vaddq_f32/vsubq_f32/
#      vnegq_f32 etc., used SEPARATELY, never fused). It is therefore NOT an
#      explicit-FMA TU: any fmla the compiler produced from it would be
#      exactly the auto-contraction this policy forbids, so it is SCALAR-
#      audited, not exempted. (Empirically: this object carried 36
#      fmla-class instructions built WITHOUT -ffp-contract=off, and 0 with
#      it — the single most convincing before/after data point for this
#      whole change; see the round-3 B04 report for the full before/after
#      table across every object in this list.)
#
# The AEC repo's own aec_simd_kernels.h — explicit vfmaq_f32 intrinsics
# consumed by AEC's TUs — is the same EXEMPT category as case 1 above, by
# the same rule (explicit source-level fusion request); it is not in this
# script's audit list (AEC's own scalar-vs-kernel-header split is documented
# in AEC/c_impl's own README/CLAUDE.md), but the rule is identical: an
# EXEMPT TU is one with a grep-confirmed explicit fma()/fmaf()/vfmaq_f32-
# family call in its own source, nothing else qualifies.
#
# --- Usage -------------------------------------------------------------
#   scripts/audit_fp_contract.sh [BACKEND ...]
#   scripts/audit_fp_contract.sh            # default: kiss ne10 (both)
#   scripts/audit_fp_contract.sh kiss
#   scripts/audit_fp_contract.sh ne10
#
# Env overrides: CC/CXX/AR/RANLIB forwarded to the producer builds if set;
# NR_DIR overrides the auto-detected sibling NR/c_impl checkout.
#
# Disassembler: prefers `objdump -d` (GNU binutils or the llvm-objdump
# wrapper Xcode installs at /usr/bin/objdump); falls back to `otool -tV`
# (macOS cctools) if objdump is not on PATH. Either way the mnemonic column
# is grepped case-insensitively for the fma-class instruction names.
#
# Exit status: 0 iff every SCALAR-class object disassembled clean (zero
# fma-class instructions) on every requested backend. EXEMPT-class objects
# are reported but never affect the exit status. A missing expected object
# (a repo/backend combination this script doesn't know how to build, or a
# source-list change that dropped an object this script still expects) is
# itself a hard FATAL — silently skipping a row would defeat the audit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# NR/c_impl: normal checkout has it as a sibling of audio_common
# (SE/NR/c_impl next to SE/audio_common) -- same sibling assumption NR's own
# Makefile makes in reverse (AC_DIR ?= ../../audio_common from NR/c_impl/).
# Override with NR_DIR=<path> for a different layout.
NR_DIR="${NR_DIR:-}"
if [ -z "$NR_DIR" ]; then
    for cand in "$AC_DIR/../NR/c_impl" "$AC_DIR/../../NR/c_impl"; do
        if [ -d "$cand" ]; then
            NR_DIR="$(cd "$cand" && pwd)"
            break
        fi
    done
fi
if [ -z "$NR_DIR" ] || [ ! -d "$NR_DIR" ]; then
    echo "FATAL: could not locate NR/c_impl as a sibling of $AC_DIR (override with NR_DIR=<path>)" >&2
    exit 1
fi

BACKENDS="${*:-kiss ne10}"

# A plain string (not an array): under `set -u`, bash 3.2 (macOS's system
# /bin/bash) treats expanding an EMPTY array as an unbound-variable error,
# so a string that word-splits to zero words when empty is the portable
# choice here (none of CC/CXX/AR/RANLIB need to preserve embedded spaces).
MAKE_FWD=""
[ -n "${CC:-}" ]     && MAKE_FWD="$MAKE_FWD CC=$CC"
[ -n "${CXX:-}" ]    && MAKE_FWD="$MAKE_FWD CXX=$CXX"
[ -n "${AR:-}" ]     && MAKE_FWD="$MAKE_FWD AR=$AR"
[ -n "${RANLIB:-}" ] && MAKE_FWD="$MAKE_FWD RANLIB=$RANLIB"

# --- Disassembler detection -------------------------------------------------
if command -v objdump >/dev/null 2>&1; then
    DISASM=objdump
elif command -v otool >/dev/null 2>&1; then
    DISASM=otool
else
    echo "FATAL: neither objdump nor otool found on PATH" >&2
    exit 1
fi

disas() {
    case "$DISASM" in
        objdump) objdump -d "$1" 2>/dev/null ;;
        otool)   otool -tV "$1" 2>/dev/null ;;
    esac
}

# fma-class mnemonics this policy forbids in scalar (non-EXEMPT) code:
# fmadd/fmsub/fnmadd/fnmsub (AArch64 scalar/vector fused forms) and
# fmla/fmls (AArch64 NEON vector fused multiply-accumulate/subtract, the
# form the compiler's auto-vectorizer reaches for when it fuses a
# vectorized a*b+c/a*b-c pattern -- see NE10_rfft_float32.neonintrinsic.o's
# case 2 above for a real example). Case-insensitive; both objdump's and
# otool's mnemonic columns lower-case these.
FMA_RE='fmadd|fmsub|fnmadd|fnmsub|fmla|fmls'

count_fma() {
    # grep -c always prints a count (0 on no match) but exits 1 on no
    # match; `|| true` keeps `set -e` from treating "clean" as a script
    # failure. count_fma itself never decides PASS/FAIL -- the caller does.
    disas "$1" | grep -icE "$FMA_RE" || true
}

# --- Audit list (round-3 review B04) ----------------------------------------
# One row per TU: "repo|backend-scope|object|class|note"
#   repo:  AC (audio_common) | NR
#   scope: both | kiss | ne10  (which BACKEND(s) this object is even compiled under)
#   class: SCALAR (must disassemble fma-free) | EXEMPT (reported, never fails)
# See the header comment above for the full rationale, especially the two
# non-obvious classifications (fft_wrapper*/fft_wrapper_ne10* -> EXEMPT,
# NE10_rfft_float32.neonintrinsic.o -> SCALAR despite its name).
AUDIT_ENTRIES='
AC|both|hpf.o|SCALAR|shared HPF core (src/hpf.c) -- no NEON intrinsics, no explicit fmaf/fma anywhere
AC|kiss|kiss_fft.o|SCALAR|vendored KISS FFT (lib/kiss_fft/kiss_fft.c) -- plain scalar C, no NEON, no explicit fmaf/fma
AC|kiss|fft_wrapper.o|EXEMPT|fft_power() scalar tail explicitly calls fmaf(re,re,im*im) (see that function comment) + an __ARM_NEON&&__aarch64__-guarded vfmaq_f32/vmulq_f32 block -- both deliberate explicit-fusion requests, not compiler contraction
AC|ne10|fft_wrapper_ne10.o|EXEMPT|same pattern as fft_wrapper.o: fft_power() scalar tail explicitly calls fmaf(), plus explicit vfmaq_f32/vmulq_f32 NEON intrinsics
AC|ne10|NE10_fft.o|SCALAR|vendored NE10 (lib/ne10/modules/dsp/NE10_fft.c) -- plain scalar C; its few "neon" text hits are identifier substrings (e.g. ne10_fft_alloc_c2c_float32_neon), not intrinsic calls -- confirmed by grep
AC|ne10|NE10_fft_float32.o|SCALAR|vendored NE10 c2c float32 -- plain scalar C, unreachable at runtime (kept only for link-time symbol resolution -- see the Makefile NE10 source-footprint comment), zero NEON intrinsic hits
AC|ne10|NE10_fft_int32.o|SCALAR|vendored NE10 c2c int32 -- plain scalar C, unreachable at runtime, zero NEON intrinsic hits
AC|ne10|NE10_fft_generic_float32.o|SCALAR|vendored NE10 generic (non-power-of-2) float32 -- plain scalar C, unreachable at runtime, zero NEON intrinsic hits
AC|ne10|NE10_rfft_float32.o|SCALAR|vendored NE10 R2C/C2R scalar reference implementation -- plain scalar C; its "neon" text hits are struct-field/function-name substrings (r_twiddles_neon etc.), not intrinsic calls -- confirmed by grep
AC|ne10|NE10_fft_generic_int32.o|SCALAR|vendored NE10 generic int32 (.cpp, the one C++ TU this archive compiles) -- plain scalar C++, zero NEON intrinsic hits
AC|ne10|NE10_rfft_float32.neonintrinsic.o|SCALAR|vendored NE10 R2C/C2R NEON-vectorized kernel -- despite the filename, grep confirms it calls ONLY separate vmulq_f32/vaddq_f32/vsubq_f32/vnegq_f32 (no vfmaq_f32/vfmsq_f32 anywhere): NOT an explicit-FMA TU, so audited like any other -- see the script header case 2
NR|both|mmse_lsa_denoiser.o|SCALAR|includes fast_math.h/fft_wrapper.h/mmse_lsa_types.h/mcra_noise_estimator.h/spp_estimator.h only -- no NEON intrinsics, no explicit fmaf/fma anywhere
NR|both|mcra_noise_estimator.o|SCALAR|includes fast_math.h/fft_wrapper.h only -- no NEON intrinsics, no explicit fmaf/fma anywhere
NR|both|spp_estimator.o|SCALAR|includes fast_math.h/fft_wrapper.h only -- no NEON intrinsics, no explicit fmaf/fma anywhere
'

TOTAL=0
FAILS=0
declare -a FAIL_ROWS=()

printf '%s\n' "Disassembler: $DISASM"
echo

for BACKEND in $BACKENDS; do
    echo "############################################################"
    echo "# BACKEND=$BACKEND"
    echo "############################################################"

    # Build audio_common first (NR's own two-phase dispatch needs AC_LIB
    # resolved the same way its own Makefile resolves it -- see NR's
    # Makefile "Consumer two-phase audio_common resolution" comment).
    make -C "$AC_DIR" -s --no-print-directory BACKEND="$BACKEND" $MAKE_FWD lib >/dev/null
    AC_OBJDIR="$(make -C "$AC_DIR" -s --no-print-directory BACKEND="$BACKEND" $MAKE_FWD print-obj-dir)"
    AC_LIB="$(make -C "$AC_DIR" -s --no-print-directory BACKEND="$BACKEND" $MAKE_FWD print-lib-path)"

    make -C "$NR_DIR" -s --no-print-directory BACKEND="$BACKEND" $MAKE_FWD AC_LIB="$AC_LIB" lib >/dev/null
    NR_OBJDIR="$(make -C "$NR_DIR" -s --no-print-directory BACKEND="$BACKEND" $MAKE_FWD AC_LIB="$AC_LIB" print-obj-dir)"

    printf '%-4s %-38s %-8s %6s  %s\n' "REPO" "OBJECT" "CLASS" "HITS" "VERDICT"
    printf -- '------------------------------------------------------------------------------------------------\n'

    while IFS='|' read -r repo scope obj class note; do
        [ -z "$repo" ] && continue
        case "$scope" in
            both) ;;
            "$BACKEND") ;;
            *) continue ;;
        esac

        case "$repo" in
            AC) objdir="$AC_OBJDIR" ;;
            NR) objdir="$NR_OBJDIR" ;;
            *) echo "FATAL: unknown repo tag '$repo' in AUDIT_ENTRIES" >&2; exit 1 ;;
        esac
        objpath="$objdir/$obj"

        if [ ! -f "$objpath" ]; then
            echo "FATAL: expected object missing: $objpath (repo=$repo backend=$BACKEND obj=$obj)" >&2
            echo "  -- either this backend/repo combination's source list changed, or this script's" >&2
            echo "     audit list is stale; both are coverage gaps and must be fixed, not skipped." >&2
            exit 1
        fi

        hits="$(count_fma "$objpath")"
        TOTAL=$((TOTAL + 1))

        if [ "$class" = "SCALAR" ]; then
            if [ "$hits" -eq 0 ]; then
                verdict="PASS"
            else
                verdict="FAIL"
                FAILS=$((FAILS + 1))
                FAIL_ROWS+=("$BACKEND: $repo/$obj ($hits fma-class hit(s))")
            fi
        else
            verdict="EXEMPT"
        fi

        printf '%-4s %-38s %-8s %6s  %s\n' "$repo" "$obj" "$class" "$hits" "$verdict"
        if [ "$verdict" = "FAIL" ] || [ "$class" = "EXEMPT" ]; then
            printf '     -- %s\n' "$note"
        fi
    done <<EOF
$(printf '%s' "$AUDIT_ENTRIES" | grep -v '^\s*$')
EOF
    echo
done

echo "############################################################"
echo "# Summary"
echo "############################################################"
echo "objects audited: $TOTAL   scalar-TU failures: $FAILS"
if [ "$FAILS" -gt 0 ]; then
    echo "FAIL:"
    for row in "${FAIL_ROWS[@]}"; do
        echo "  - $row"
    done
    exit 1
fi
echo "PASS: every SCALAR-class object disassembled fma-class-instruction-free on every requested backend ($BACKENDS)."
