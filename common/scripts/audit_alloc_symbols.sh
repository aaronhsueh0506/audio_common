#!/bin/sh
# audit_alloc_symbols.sh - allocator-symbol provenance aid.
#
# Given a binary (static archive .a, object .o, or linked executable/dylib),
# reports every reference to the four heap-allocator entry points
# (malloc/calloc/realloc/free) found in its symbol table, and -- when the
# input is a static archive -- WHICH archive member (object file) each
# reference comes from, via `nm -A` (`-A`/`--print-file-name` is supported
# by both macOS/BSD nm and GNU nm; see each's --help).
#
# This is a read-only inspection tool, not a test: it does not assert
# anything or exit non-zero on "found" (allocator refs are frequently
# expected -- e.g. the heap fft_create/fft_destroy path is supposed to call
# calloc/free; `make test_zero_heap` is the actual PASS/FAIL gate on the
# fft_init()..fft_destroy() static-memory path). Use this
# script instead when you need to know WHERE an allocator call is coming
# from -- e.g. auditing a full strict-build archive/executable to confirm
# no unexpected object file pulled one in, or narrowing down which .o to
# inspect after test_zero_heap already told you a call happened somewhere.
#
# Usage:
#   scripts/audit_alloc_symbols.sh <path-to-binary> [<path-to-binary> ...]
#
# Example (bin/ is keyed bin/<backend>-<config-hash>/; resolve the exact
# path for your flags with `make print-bin-dir` /
# `make print-lib-path`, or use `dist/<backend>/current/` after `make
# publish`):
#   scripts/audit_alloc_symbols.sh "$(make -s print-lib-path)"
#   scripts/audit_alloc_symbols.sh "$(make -s print-lib-path)" "$(make -s print-bin-dir)/roundtrip"
#
# Output: one line per (object file, symbol) reference found, e.g.
#   bin/ne10-<sig>/libaudio_common.a:fft_wrapper_ne10.o:  U _calloc
#   bin/ne10-<sig>/libaudio_common.a:NE10_fft.o:          U _malloc
# ("U" = undefined/referenced-not-defined here, the normal case for a call
# site; a "T"/"t" entry would mean this binary itself DEFINES that symbol,
# e.g. a custom allocator or the zero_heap_hook.c interposer -- worth a
# second look if seen somewhere unexpected).
#
# Not wired into `make` on purpose: a manual audit aid, run by hand against
# whatever binary is being audited.

set -eu

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <path-to-binary> [<path-to-binary> ...]" >&2
    exit 2
fi

# Match malloc/calloc/realloc/free as whole symbol names, tolerating an
# optional leading underscore (macOS/BSD nm's Mach-O convention) and an
# optional trailing '$' (used by some symbol decorations); GNU nm on Linux
# has no leading underscore, so the '_?' keeps this portable across both.
ALLOC_PATTERN='^_?(malloc|calloc|realloc|free)$'

status=0
for bin in "$@"; do
    if [ ! -e "$bin" ]; then
        echo "$0: $bin: no such file" >&2
        status=1
        continue
    fi
    # -A / --print-file-name: prefixes every line with "archive:member:" for
    # an .a, or just "path:" for a plain object/executable -- either way we
    # get provenance without needing separate handling for the two cases.
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
