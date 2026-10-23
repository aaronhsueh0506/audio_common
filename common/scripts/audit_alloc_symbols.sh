#!/bin/sh
# audit_alloc_symbols.sh - allocator-symbol provenance aid.
#
# Reports every reference to malloc/calloc/realloc/free in the symbol table of each given binary
# (static archive, object, executable or dylib) and, for a static archive, which member each reference
# comes from (`nm -A`, supported by BSD and GNU nm). Read-only: it asserts nothing and never exits
# non-zero on "found", because allocator references are often expected (the heap
# fft_create/fft_destroy path calls calloc/free; `make test_zero_heap` is the PASS/FAIL gate for the
# static path). Use it to find where an allocator call comes from. Not wired into `make`.
#
# Usage:
#   scripts/audit_alloc_symbols.sh <path-to-binary> [<path-to-binary> ...]
#
# Example (resolve the keyed path with `make print-lib-path` / `make print-bin-dir`, or use
# dist/<backend>/current/ after `make publish`):
#   scripts/audit_alloc_symbols.sh "$(make -s print-lib-path)"
#   scripts/audit_alloc_symbols.sh "$(make -s print-lib-path)" "$(make -s print-bin-dir)/roundtrip"
#
# Output: one line per (object file, symbol) reference, e.g.
#   bin/ne10-<sig>/libaudio_common.a:fft_wrapper_ne10.o:  U _calloc
#   bin/ne10-<sig>/libaudio_common.a:NE10_fft.o:          U _malloc
# "U" = undefined here (a call site); a "T"/"t" entry means the binary itself defines the symbol
# (a custom allocator or the zero_heap_hook.c interposer).

set -eu

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <path-to-binary> [<path-to-binary> ...]" >&2
    exit 2
fi

# Match malloc/calloc/realloc/free as whole symbol names, with an optional leading underscore (Mach-O
# convention on macOS/BSD nm; GNU nm on Linux has none).
ALLOC_PATTERN='^_?(malloc|calloc|realloc|free)$'

status=0
for bin in "$@"; do
    if [ ! -e "$bin" ]; then
        echo "$0: $bin: no such file" >&2
        status=1
        continue
    fi
    # -A prefixes every line with "archive:member:" for an .a, or "path:" for a plain object/executable.
    nm -A "$bin" | awk -v pat="$ALLOC_PATTERN" '
        {
            # Last two whitespace-separated fields are the nm TYPE and NAME
            # columns (e.g. "U", "_calloc"); everything before that is the
            # "file:member:" prefix nm -A prepends, which may itself contain
            # spaces in pathological paths, so do not assume a fixed split
            # point -- just check the last field against the pattern.
            name = $NF
            if (name ~ pat) print
        }
    '
done

exit "$status"
