# fp_contract_lib.sh -- the parts of the floating-point-contract audit that
# must be identical everywhere it is run: the forbidden mnemonic set and the
# disassembler wrapper.
#
# Sourced by audio_common's own scripts/audit_fp_contract.sh and by consumers
# further up the stack (Audio_ALG/pipelines). Consumers may depend on
# audio_common; the reverse is forbidden, which is why this lives here and
# why the per-repo AUDIT LISTS stay with the objects they describe.
#
# fma-class mnemonics forbidden in scalar (non-EXEMPT) code: fmadd/fmsub/
# fnmadd/fnmsub (AArch64 scalar/vector fused forms) and fmla/fmls (the NEON
# fused multiply-accumulate the auto-vectorizer reaches for when it fuses a
# vectorized a*b+c). Case-insensitive; both objdump's and otool's mnemonic
# columns lower-case these. Extend in ONE place.
FP_CONTRACT_FMA_RE='fmadd|fmsub|fnmadd|fnmsub|fmla|fmls'

if command -v objdump >/dev/null 2>&1; then
    FP_CONTRACT_DISASM=objdump
elif command -v otool >/dev/null 2>&1; then
    FP_CONTRACT_DISASM=otool
else
    echo "FATAL: neither objdump nor otool found on PATH" >&2
    exit 1
fi

fp_contract_disas() {
    case "$FP_CONTRACT_DISASM" in
        objdump) objdump -d "$1" 2>/dev/null ;;
        otool)   otool -tV "$1" 2>/dev/null ;;
    esac
}
