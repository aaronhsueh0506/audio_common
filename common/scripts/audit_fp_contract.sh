#!/usr/bin/env bash
# Verify the pinned -ffp-contract=off policy by disassembling keyed objects.
# Compiler -Ofast/-ffast-math/-ffp-contract overrides are rejected by the
# producer Makefiles; NR's USE_* math approximations are separate build options.
#
# SCALAR rows fail on fmadd/fmsub/fnmadd/fnmsub/fmla/fmls instructions.
# EXEMPT rows report the count without failing: explicit fma/fmaf or fused
# NEON intrinsics request fusion even with -ffp-contract=off. In particular,
# fft_wrapper_ne10.c uses fmaf/vfmaq in fft_power; NE10's RFFT intrinsic file
# uses separate multiply/add operations and remains SCALAR-audited.
# utility/simd_aec.h contains explicit FMA kernels, but static-inline headers
# have no object of their own: audit their consumer TUs and scalar twins.
# Missing expected objects are fatal; no row is silently skipped.
#
# Usage: scripts/audit_fp_contract.sh [ne10]
# CC/CXX/AR/RANLIB are forwarded to builds; NR_DIR selects NR/c_impl.
# Uses objdump -d, falling back to otool -tV via fp_contract_lib.sh.
# Also checks fast_sqrt hardware code generation for Cortex-A53/A73.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# common lives at SE/audiolib/common; NR is the sibling SE/NR checkout.
# NR_DIR can override the search for a different workspace layout.
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

BACKENDS="${*:-ne10}"

# The same release audit also proves that the scalar AArch64 fast_sqrt path
# is a hardware instruction, not an errno-checking libm call. Read the flags
# from the production Makefile so this gate fails if its policy ever drifts.
# The probe uses Linux A53/A73 code generation even on a macOS host.
AC_FP_POLICY="$(make -C "$AC_DIR" -s --no-print-directory DEBUG=0 print-fp-policy)"
FP_POLICY="$AC_FP_POLICY" "$SCRIPT_DIR/audit_fast_sqrt_codegen.sh"

# A plain string (not an array): under `set -u`, bash 3.2 (macOS's system
# /bin/bash) treats expanding an EMPTY array as an unbound-variable error,
# so a string that word-splits to zero words when empty is the portable
# choice here (none of CC/CXX/AR/RANLIB need to preserve embedded spaces).
MAKE_FWD=""
[ -n "${CC:-}" ]     && MAKE_FWD="$MAKE_FWD CC=$CC"
[ -n "${CXX:-}" ]    && MAKE_FWD="$MAKE_FWD CXX=$CXX"
[ -n "${AR:-}" ]     && MAKE_FWD="$MAKE_FWD AR=$AR"
[ -n "${RANLIB:-}" ] && MAKE_FWD="$MAKE_FWD RANLIB=$RANLIB"

# --- Disassembler + forbidden-mnemonic set ------------------------------
# Both live in scripts/fp_contract_lib.sh so consumers further up the stack
# (Audio_ALG/pipelines) audit against the SAME set rather than a copy of it.
# Extending the mnemonic list is one edit.
. "$(dirname "$0")/fp_contract_lib.sh"
DISASM="$FP_CONTRACT_DISASM"
FMA_RE="$FP_CONTRACT_FMA_RE"

disas() { fp_contract_disas "$1"; }

count_fma() {
    # grep -c always prints a count (0 on no match) but exits 1 on no
    # match; `|| true` keeps `set -e` from treating "clean" as a script
    # failure. count_fma itself never decides PASS/FAIL -- the caller does.
    disas "$1" | grep -icE "$FMA_RE" || true
}

# --- Audit list ---------------------------------------------------------
# One row per TU: "repo|backend-scope|object|class|note"
#   repo:  AC (common) | NR
#   scope: both | ne10  (which BACKEND(s) this object is even compiled under)
#   class: SCALAR (must disassemble fma-free) | EXEMPT (reported, never fails)
# Static-inline kernels are tested against scalar twins and audited in
# consumer objects; compiling an unused wrapper would not prove the linked code.
AUDIT_ENTRIES='
AC|both|hpf.o|SCALAR|shared HPF core (src/hpf.c) -- no NEON intrinsics, no explicit fmaf/fma anywhere
AC|both|audio_resampler.o|SCALAR|streaming polyphase resampler (src/audio_resampler.c) -- its own NEON dot products (separate vmulq_f32/vaddq_f32 lane chains, never vfmaq_f32) and, under SIMD=0, a plain sequential multiply-add loop: any fmla/fmadd here is compiler contraction
AC|both|audio_window.o|SCALAR|periodic sqrt-Hann table (src/audio_window.c); separate scalar multiply/subtract operations, no explicit fmaf/fma
AC|both|audio_stft.o|SCALAR|STFT/ISTFT objects (src/audio_stft.c) -- inline simd_kernels.h / simd_kernel_nn.h kernels (separate vmulq_f32/vaddq_f32, no vfmaq_f32) and plain scalar loops for the spectrum scale; no explicit fmaf/fma, so any fmla/fmadd here is compiler contraction
AC|ne10|fft_wrapper_ne10.o|EXEMPT|fft_power() scalar tail explicitly calls fmaf(re,re,im*im) (see that function comment) + an SK_HAVE_NEON-guarded vfmaq_f32/vmulq_f32 block -- both deliberate explicit-fusion requests, not compiler contraction
AC|ne10|NE10_fft.o|SCALAR|vendored NE10 (lib/ne10/modules/dsp/NE10_fft.c) -- plain scalar C; its few "neon" text hits are identifier substrings (e.g. ne10_fft_alloc_c2c_float32_neon), not intrinsic calls -- confirmed by grep
AC|ne10|NE10_fft_float32.o|SCALAR|vendored NE10 c2c float32 -- plain scalar C, unreachable at runtime (kept only for link-time symbol resolution -- see the Makefile NE10 source-footprint comment), zero NEON intrinsic hits
AC|ne10|NE10_fft_int32.o|SCALAR|vendored NE10 c2c int32 -- plain scalar C, unreachable at runtime, zero NEON intrinsic hits
AC|ne10|NE10_fft_generic_float32.o|SCALAR|vendored NE10 generic (non-power-of-2) float32 -- plain scalar C, unreachable at runtime, zero NEON intrinsic hits
AC|ne10|NE10_rfft_float32.o|SCALAR|vendored NE10 R2C/C2R scalar reference implementation -- plain scalar C; its "neon" text hits are struct-field/function-name substrings (r_twiddles_neon etc.), not intrinsic calls -- confirmed by grep
AC|ne10|NE10_fft_generic_int32.o|SCALAR|vendored NE10 generic int32 (.cpp, the one C++ TU this archive compiles) -- plain scalar C++, zero NEON intrinsic hits
AC|ne10|NE10_rfft_float32.neonintrinsic.o|SCALAR|vendored NE10 R2C/C2R NEON-vectorized kernel -- despite the filename, grep confirms it calls ONLY separate vmulq_f32/vaddq_f32/vsubq_f32/vnegq_f32 (no vfmaq_f32/vfmsq_f32 anywhere): NOT an explicit-FMA TU, so audited like any other
NR|both|anr.o|SCALAR|inlines the simd_kernels.h gain kernels (sk_exp1_approx_f32, sk_fast_exp_f32, sk_fast_log_f32; unfused), including the batched final-gain exponential under USE_FAST_GAIN_SMOOTHING on NEON; no explicit fmaf/fma in the TU
NR|both|noise_estimator.o|SCALAR|inlines simd_nr.h ordered/ring minimum and restart kernels via local ANR_VECTOR dispatch; simd_kernels.h EMA/log/noise-update kernels are unfused; no explicit fmaf/fma
NR|both|spp_estimator.o|SCALAR|own NEON code is the decision-directed pass spp_dd_neon (vdivq_f32/vmulq_f32/vaddq_f32/vbslq_f32/vcgtq_f32 arithmetic, always unfused; stays local to this file -- takes the whole opaque SppEstimator* and writes three of its private arrays, too coupled to share); also inlines simd_kernels.h sk_fast_exp_neg_f32; no explicit fmaf/fma in the TU
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

    # Build common first; NR's two-phase dispatch needs COMMON_LIB resolved the same way.
    make -C "$AC_DIR" -s --no-print-directory BACKEND="$BACKEND" $MAKE_FWD lib >/dev/null
    AC_OBJDIR="$(make -C "$AC_DIR" -s --no-print-directory BACKEND="$BACKEND" $MAKE_FWD print-obj-dir)"
    AC_LIB="$(make -C "$AC_DIR" -s --no-print-directory BACKEND="$BACKEND" $MAKE_FWD print-lib-path)"
    make -C "$NR_DIR" -s --no-print-directory BACKEND="$BACKEND" $MAKE_FWD COMMON_LIB="$AC_LIB" lib >/dev/null
    NR_OBJDIR="$(make -C "$NR_DIR" -s --no-print-directory BACKEND="$BACKEND" $MAKE_FWD COMMON_LIB="$AC_LIB" print-obj-dir)"

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
