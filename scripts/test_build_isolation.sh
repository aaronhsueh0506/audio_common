#!/usr/bin/env bash
# test_build_isolation.sh — round-3/round-4/round-5 review build-isolation
# regression suite.
#
# Exercises the CFG_SIG-keyed obj/bin directory design (audio_common's own
# Makefile, plus the two-phase AC_LIB consumer resolution in AEC/c_impl and
# NR/c_impl) against every failure mode the round-3 B01 finding described:
# mtime-staleness delivering another config's artifact, test_ne10_force_c
# overwriting the normal NE10 archive, parallel different-config builds
# co-writing, and consumers hardcoding flat paths / forwarding only BACKEND
# (config skew) -- plus, from the round-4 review, command-line override
# rejection (P1-1), fresh-archive stale-member removal (P1-4), publish v3
# content-addressing (P2-2), and the CC/CXX toolchain-coherence guard (P1-2)
# -- plus, from the round-5 review, the publish v4 redesign: a throwaway
# DIST_ROOT= knob (P1, this script's own safety fix -- see below), the
# lock-FIRST publish driver (P1), the deterministic MANIFEST.txt / append-only
# ATTEST/ split (P2), and the rename(2)-atomic `current` swap helper (P2).
#
# Scenario index:
#   S1  - A->B->A + -O0/-O3 delivered-object repro
#   S2  - same-second kiss<->ne10 switching
#   S3  - parallel differing-config builds
#   S4  - test_ne10_force_c isolation
#   S5  - consumer correctness (drives AEC)
#   S7p - producer-publish (v4 content-addressed dist/ layout; throwaway
#         DIST_ROOT=)
#   S8  - CFG_SIG collision guard
#   S9  - command-line override rejection (round-4 P1-1)
#   S10 - archive freshness / stale-member removal (round-4 P1-4)
#   S11 - publish immutability / content-addressing (round-4 P2-2 -> round-5
#         v4 semantics), in three parts:
#           S11a - in-tree republish (throwaway DIST_ROOT): byte-verified
#                  republish, ATTEST/ growth, mtimes untouched
#           S11b - MANIFEST tamper detection
#           S11c - content-change publish, exercised in a throwaway sandbox
#                  copy of the tree (never in $AC_DIR)
#   S12 - toolchain coherence guard (round-4 P1-2)
#   S13 - atomic `current` symlink swap hammer (round-5 P2)
#   S14 - lock-before-build: a concurrent publish loser builds nothing
#         (round-5 P1)
# S6 (Audio_ALG/pipelines consumer-resolution parity) is a later wave and is
# intentionally NOT covered here.
#
# Design rules (do not violate when editing this script):
#   - No `make clean` inside any scenario body: the whole point is that
#     distinct configs coexist WITHOUT ever needing a clean between them.
#   - Every path is resolved via `make -s ... print-bin-dir` / `print-obj-dir`
#     / `print-lib-path`, using the EXACT flag set under test for that call
#     — never a hand-reconstructed path guess.
#   - "Did this get rebuilt?" is always an mtime comparison, never a content
#     (sha) comparison: recompiling identical inputs with a deterministic
#     compiler produces byte-identical output, so sha equality does NOT mean
#     "not rebuilt" and sha difference is not required for "was rebuilt".
#   - "Is this the SAME delivered artifact as its own keyed object?" (the
#     actual B01 assertion — no cross-config delivery) IS a sha comparison,
#     via member_sha()/file_sha() below.
#   - (round-5 review P1) Never touch the REAL dist/: every `make ... publish`
#     call in this script passes an explicit DIST_ROOT= pointing at a
#     throwaway mktemp -d. Never `rm -rf dist`, never read dist/ without a
#     DIST_ROOT= override. See the safety-contract comment below for the
#     full rationale (this rule replaces a previous version of this script
#     that destroyed the real dist/ tree).
#   - (round-5 review P1) Never modify a git-tracked file's CONTENT (no
#     `git checkout -- ...` restore, no direct append/overwrite onto a
#     tracked file). A `touch` for an mtime-only recompile probe is fine
#     (content-neutral); a scenario that needs to exercise an actual content
#     change does it inside a throwaway sandbox copy of the tree instead.
#   - Every mktemp/mktemp -d in this script goes through new_tmpfile()/
#     new_tmpdir() below, so the single CLEANUP_DIRS trap owns all of it —
#     no scenario hand-rolls its own `rm -f`/`rm -rf` cleanup any more.
#
# Usage: ./scripts/test_build_isolation.sh   (run from audio_common/, or
# anywhere — paths are resolved relative to this script's own location).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
AEC_DIR="$(cd "$AC_DIR/../AEC/c_impl" && pwd)"
NR_DIR="$(cd "$AC_DIR/../NR/c_impl" && pwd)"

# --- round-5 review P1 safety contract --------------------------------------
# What this script writes, and ONLY this:
#   - its own scratch files/directories, created exclusively via the
#     new_tmpfile()/new_tmpdir() helpers below, every one of which is tracked
#     in CLEANUP_DIRS and removed by the single EXIT/INT/TERM trap -- even
#     when `set -e` aborts the script partway through a scenario.
#   - this repo's normal CFG_SIG-keyed obj/<backend-sig>/ and bin/<backend-sig>/
#     build output, created exactly the way any ordinary `make BACKEND=...
#     lib` invocation would create it -- nothing scenario-specific lives
#     there, and none of it is git-tracked (see .gitignore).
#   - mtime-only touches of a handful of already-tracked files (src/hpf.c,
#     include/fast_math.h, include/wav_io.h, lib/ne10/inc/NE10_dsp.h), used
#     to force a recompile probe. A `touch` changes mtime, never content.
#
# What it NEVER does, after the round-5 review P1 finding that a previous
# version of this script `rm -rf dist`'d and read/wrote the REAL dist/
# directly (destroying whatever a real `make publish` had produced there):
#   - every `make ... publish` call below passes an explicit DIST_ROOT=
#     pointing at a throwaway mktemp -d; the default dist/ is never read,
#     written, or removed by this script, full stop.
#   - no `git checkout -- ...` restore and no direct append/overwrite onto a
#     git-tracked file's content. S11c needs to exercise "publish sees new
#     source bytes" -- it does that inside a throwaway tarball copy of the
#     tree (a "sandbox" living under its own mktemp -d), never in $AC_DIR.
#
# A git-status hash of $AC_DIR is taken right now and re-checked in the
# SUMMARY section at the very end: any drift in $AC_DIR's tracked-file state
# across the whole run is reported as a hard FAIL, not a silent surprise.
CLEANUP_DIRS=""
track_tmp() { CLEANUP_DIRS="$CLEANUP_DIRS $1"; }
new_tmpdir() {
  local d
  d="$(mktemp -d)"
  track_tmp "$d"
  printf '%s\n' "$d"
}
new_tmpfile() {
  local f
  f="$(mktemp)"
  track_tmp "$f"
  printf '%s\n' "$f"
}
# shellcheck disable=SC2086  # deliberately unquoted: CLEANUP_DIRS is a
# space-separated list of paths (none of which contain spaces -- they are
# all mktemp/mktemp -d outputs) and `rm -rf` needs them as separate args.
trap 'rm -rf $CLEANUP_DIRS' EXIT INT TERM

git_status_hash() {
  ( cd "$AC_DIR" && git status --porcelain 2>/dev/null | shasum -a 256 | awk '{print $1}' )
}
GIT_STATUS_BEFORE="$(git_status_hash)"

PASS_COUNT=0
FAIL_COUNT=0
FAILURES=()

pass() { PASS_COUNT=$((PASS_COUNT + 1)); echo "  PASS: $*"; }
fail() { FAIL_COUNT=$((FAIL_COUNT + 1)); FAILURES+=("$*"); echo "  FAIL: $*" >&2; }

# ar -p <archive> <member> | sha256  -- the sha of one member's CONTENT as it
# sits inside the delivered archive (not the archive's own sha, which would
# also fold in ar's per-member timestamp/mode header fields).
member_sha() { ar -p "$1" "$2" | shasum -a 256 | awk '{print $1}'; }
file_sha()   { shasum -a 256 "$1" | awk '{print $1}'; }
mtime()      { stat -f %m "$1" 2>/dev/null || stat -c %Y "$1"; }

echo "############################################################"
echo "# S1: A->B->A + -O0/-O3 delivered-object repro"
echo "############################################################"
cd "$AC_DIR"

kiss_lib="$(make -s BACKEND=kiss print-lib-path)"
make -s BACKEND=kiss lib >/dev/null
sha_kiss_1="$(file_sha "$kiss_lib")"
mtime_kiss_1="$(mtime "$kiss_lib")"

ne10_lib="$(make -s BACKEND=ne10 print-lib-path)"
make -s BACKEND=ne10 lib >/dev/null

# back to kiss: archive must be untouched (same path, same sha, same mtime)
kiss_lib_again="$(make -s BACKEND=kiss print-lib-path)"
make -s BACKEND=kiss lib >/dev/null
sha_kiss_2="$(file_sha "$kiss_lib_again")"
mtime_kiss_2="$(mtime "$kiss_lib_again")"

[ "$kiss_lib" = "$kiss_lib_again" ] && pass "S1: kiss path stable across A->B->A" \
  || fail "S1: kiss path CHANGED across A->B->A ($kiss_lib vs $kiss_lib_again)"
[ "$sha_kiss_1" = "$sha_kiss_2" ] && [ "$mtime_kiss_1" = "$mtime_kiss_2" ] && \
  pass "S1: kiss archive sha+mtime unchanged after A->B->A (no cross-config stomp)" \
  || fail "S1: kiss archive CHANGED after A->B->A (sha $sha_kiss_1 -> $sha_kiss_2, mtime $mtime_kiss_1 -> $mtime_kiss_2)"

[ "$kiss_lib" != "$ne10_lib" ] && pass "S1: kiss/ne10 archive paths differ" \
  || fail "S1: kiss/ne10 archive paths COLLIDE ($kiss_lib)"

ar -t "$kiss_lib" | grep -q '^kiss_fft\.o$' && pass "S1: kiss archive contains kiss_fft.o" \
  || fail "S1: kiss archive missing kiss_fft.o"
ar -t "$kiss_lib" | grep -qi 'ne10' && fail "S1: kiss archive unexpectedly contains an ne10 object" \
  || pass "S1: kiss archive has no ne10 objects"
ar -t "$ne10_lib" | grep -q '^NE10_fft\.o$' && pass "S1: ne10 archive contains NE10_fft.o" \
  || fail "S1: ne10 archive missing NE10_fft.o"
ar -t "$ne10_lib" | grep -q '^kiss_fft\.o$' && fail "S1: ne10 archive unexpectedly contains kiss_fft.o" \
  || pass "S1: ne10 archive has no kiss_fft.o"

# The reviewer's literal repro: EXTRA_CFLAGS=-O0 vs -O3 must never let one
# config's compiled object be delivered under the other's archive member
# (the actual B01 bug: mtime staleness across differing flag sets).
make -s BACKEND=kiss EXTRA_CFLAGS=-O0 lib >/dev/null
lib_o0="$(make -s BACKEND=kiss EXTRA_CFLAGS=-O0 print-lib-path)"
objdir_o0="$(make -s BACKEND=kiss EXTRA_CFLAGS=-O0 print-obj-dir)"
make -s BACKEND=kiss EXTRA_CFLAGS=-O3 lib >/dev/null
lib_o3="$(make -s BACKEND=kiss EXTRA_CFLAGS=-O3 print-lib-path)"
objdir_o3="$(make -s BACKEND=kiss EXTRA_CFLAGS=-O3 print-obj-dir)"

[ "$lib_o0" != "$lib_o3" ] && pass "S1: -O0/-O3 land in different bin dirs" \
  || fail "S1: -O0/-O3 COLLIDE in the same bin dir ($lib_o0)"

sha_member_o0="$(member_sha "$lib_o0" fft_wrapper.o)"
sha_obj_o0="$(file_sha "$objdir_o0/fft_wrapper.o")"
sha_member_o3="$(member_sha "$lib_o3" fft_wrapper.o)"
sha_obj_o3="$(file_sha "$objdir_o3/fft_wrapper.o")"

if [ "$sha_member_o0" = "$sha_obj_o0" ] && [ "$sha_member_o3" = "$sha_obj_o3" ] && [ "$sha_member_o0" != "$sha_member_o3" ]; then
  pass "S1: -O0/-O3 delivered fft_wrapper.o member sha == matching keyed object sha (THE B01 assertion)"
else
  fail "S1: -O0/-O3 member/object sha mismatch (o0 member=$sha_member_o0 obj=$sha_obj_o0; o3 member=$sha_member_o3 obj=$sha_obj_o3)"
fi

echo "############################################################"
echo "# S2: same-second kiss<->ne10 switching"
echo "############################################################"
# mtime-only touch (content-neutral -- see the safety contract above): forces
# a recompile probe without ever changing what's inside src/hpf.c.
touch "$AC_DIR/src/hpf.c"
make -s BACKEND=kiss lib >/dev/null
make -s BACKEND=ne10 lib >/dev/null
kiss_lib="$(make -s BACKEND=kiss print-lib-path)";  kiss_objdir="$(make -s BACKEND=kiss print-obj-dir)"
ne10_lib="$(make -s BACKEND=ne10 print-lib-path)";  ne10_objdir="$(make -s BACKEND=ne10 print-obj-dir)"
m1="$(member_sha "$kiss_lib" hpf.o)"; o1="$(file_sha "$kiss_objdir/hpf.o")"
m2="$(member_sha "$ne10_lib" hpf.o)"; o2="$(file_sha "$ne10_objdir/hpf.o")"
[ "$m1" = "$o1" ] && [ "$m2" = "$o2" ] && \
  pass "S2: per-archive hpf.o member sha == own keyed object (same-second switch safe)" \
  || fail "S2: hpf.o member/object mismatch after same-second switch (kiss: $m1 vs $o1; ne10: $m2 vs $o2)"

echo "############################################################"
echo "# S3: parallel differing-config builds"
echo "############################################################"
S3_LOG_A="$(new_tmpfile)"; S3_LOG_B="$(new_tmpfile)"
( make -j4 -s BACKEND=kiss lib >"$S3_LOG_A" 2>&1 ) & p1=$!
( make -j4 -s BACKEND=ne10 EXTRA_CFLAGS=-DPAR_PROBE lib >"$S3_LOG_B" 2>&1 ) & p2=$!
r1=0; r2=0
wait "$p1" || r1=$?
wait "$p2" || r2=$?
if [ "$r1" -eq 0 ] && [ "$r2" -eq 0 ]; then
  pass "S3: both parallel differing-config builds exit green"
else
  fail "S3: parallel build failed (kiss exit=$r1, ne10+PAR_PROBE exit=$r2)"
  echo "--- kiss log ---" >&2; cat "$S3_LOG_A" >&2
  echo "--- ne10+PAR_PROBE log ---" >&2; cat "$S3_LOG_B" >&2
fi

kiss_lib="$(make -s BACKEND=kiss print-lib-path)";  kiss_objdir="$(make -s BACKEND=kiss print-obj-dir)"
ne10p_lib="$(make -s BACKEND=ne10 EXTRA_CFLAGS=-DPAR_PROBE print-lib-path)"
ne10p_objdir="$(make -s BACKEND=ne10 EXTRA_CFLAGS=-DPAR_PROBE print-obj-dir)"
m1="$(member_sha "$kiss_lib" hpf.o)"; o1="$(file_sha "$kiss_objdir/hpf.o")"
m2="$(member_sha "$ne10p_lib" hpf.o)"; o2="$(file_sha "$ne10p_objdir/hpf.o")"
[ "$m1" = "$o1" ] && [ "$m2" = "$o2" ] && \
  pass "S3: member==object shas hold after -j4 parallel differing-config builds" \
  || fail "S3: parallel build cross-contaminated objects (kiss: $m1 vs $o1; ne10+PROBE: $m2 vs $o2)"

echo "############################################################"
echo "# S4: test_ne10_force_c isolation (the flat-bin/ne10/ overwrite bug)"
echo "############################################################"
ne10_lib="$(make -s BACKEND=ne10 print-lib-path)"
make -s BACKEND=ne10 lib >/dev/null
sha_before="$(file_sha "$ne10_lib")"
mtime_before="$(mtime "$ne10_lib")"

S4_LOG="$(new_tmpfile)"
if make test_ne10_force_c >"$S4_LOG" 2>&1; then
  pass "S4: test_ne10_force_c exits green"
else
  fail "S4: test_ne10_force_c FAILED"
  cat "$S4_LOG" >&2
fi

sha_after="$(file_sha "$ne10_lib")"
mtime_after="$(mtime "$ne10_lib")"
[ "$sha_before" = "$sha_after" ] && [ "$mtime_before" = "$mtime_after" ] && \
  pass "S4: normal ne10 archive byte-identical + mtime-untouched after forced-C run" \
  || fail "S4: normal ne10 archive CHANGED by the forced-C sub-make (sha $sha_before -> $sha_after, mtime $mtime_before -> $mtime_after)"

forcedc_lib="$(make -s BACKEND=ne10 EXTRA_CFLAGS=-DFFT_NE10_FORCE_C print-lib-path)"
[ "$forcedc_lib" != "$ne10_lib" ] && pass "S4: forced-C archive lives at a different keyed path" \
  || fail "S4: forced-C archive path COLLIDES with the normal ne10 path"
[ -f "$forcedc_lib" ] && pass "S4: forced-C archive exists on disk" || fail "S4: forced-C archive missing"

forcedc_objdir="$(make -s BACKEND=ne10 EXTRA_CFLAGS=-DFFT_NE10_FORCE_C print-obj-dir)"
normal_objdir="$(make -s BACKEND=ne10 print-obj-dir)"
if [ -f "$forcedc_objdir/fft_wrapper_ne10.o" ] && [ -f "$normal_objdir/fft_wrapper_ne10.o" ]; then
  s1="$(file_sha "$forcedc_objdir/fft_wrapper_ne10.o")"
  s2="$(file_sha "$normal_objdir/fft_wrapper_ne10.o")"
  [ "$s1" != "$s2" ] && pass "S4: forced-C fft_wrapper_ne10.o differs from the normal build's (FFT_NE10_FORCE_C took effect)" \
    || fail "S4: forced-C object is BYTE-IDENTICAL to the normal build's (FFT_NE10_FORCE_C not taking effect?)"
else
  fail "S4: missing fft_wrapper_ne10.o in one of the two obj dirs ($forcedc_objdir or $normal_objdir)"
fi

echo "############################################################"
echo "# S5: consumer correctness (drives AEC)"
echo "############################################################"
cd "$AEC_DIR"

ac_kiss="$(make -s -C "$AC_DIR" BACKEND=kiss print-lib-path)"
make -s BACKEND=kiss all AC_LIB="$ac_kiss" >/dev/null
bindir_kiss="$(make -s BACKEND=kiss print-bin-dir AC_LIB="$ac_kiss")"
aecwav="$bindir_kiss/aec_wav"
[ -f "$aecwav" ] && pass "S5: aec_wav built" || fail "S5: aec_wav missing after initial build"
mtime_1="$(mtime "$aecwav")"

# no-op rebuild -> NOT relinked
sleep 1
make -s BACKEND=kiss all AC_LIB="$ac_kiss" >/dev/null
mtime_2="$(mtime "$aecwav")"
[ "$mtime_1" = "$mtime_2" ] && pass "S5: aec_wav NOT relinked on a no-op rebuild" \
  || fail "S5: aec_wav relinked with nothing changed"

# touch audio_common/src/hpf.c (mtime-only, content-neutral) -> hpf.o
# recompiles AND aec_wav relinks. audio_common's own CFG_SIG (a hash of its
# COMPILER INVOCATION, not file content) does not change from a source edit,
# so its archive path stays the same -- only its mtime should advance.
sleep 1
touch "$AC_DIR/src/hpf.c"
ac_objdir="$(make -s -C "$AC_DIR" BACKEND=kiss print-obj-dir)"
hpf_o="$ac_objdir/hpf.o"
mtime_hpf_before="$(mtime "$hpf_o" 2>/dev/null || echo 0)"
make -s -C "$AC_DIR" BACKEND=kiss lib >/dev/null
mtime_hpf_after="$(mtime "$hpf_o")"
[ "$mtime_hpf_after" != "$mtime_hpf_before" ] && pass "S5: hpf.c edit recompiled audio_common's hpf.o" \
  || fail "S5: hpf.o did NOT recompile after touching hpf.c"

sleep 1
make -s BACKEND=kiss all AC_LIB="$ac_kiss" >/dev/null
mtime_3="$(mtime "$aecwav")"
[ "$mtime_3" != "$mtime_1" ] && pass "S5: touching audio_common/src/hpf.c relinked aec_wav" \
  || fail "S5: aec_wav NOT relinked after audio_common's hpf.o changed"

# Header-only test 1: touch fast_math.h -> AEC objects that include it
# (aec3_post.c, suppression_gain.c) recompile.
sleep 1
touch "$AC_DIR/include/fast_math.h"
aec_objdir="$(make -s BACKEND=kiss print-obj-dir AC_LIB="$ac_kiss")"
t_before="$(mtime "$aec_objdir/aec3_post.o")"
make -s BACKEND=kiss all AC_LIB="$ac_kiss" >/dev/null
t_after="$(mtime "$aec_objdir/aec3_post.o")"
[ "$t_after" != "$t_before" ] && pass "S5: touching audio_common/include/fast_math.h recompiled AEC's aec3_post.o" \
  || fail "S5: aec3_post.o did NOT recompile after touching fast_math.h"

# Header-only test 2: touch wav_io.h -> example objs (aec_wav.o, which
# #includes example/wav_io.h -> audio_common/include/wav_io.h) recompile.
sleep 1
touch "$AC_DIR/include/wav_io.h"
t_before="$(mtime "$aec_objdir/aec_wav.o")"
make -s BACKEND=kiss all AC_LIB="$ac_kiss" >/dev/null
t_after="$(mtime "$aec_objdir/aec_wav.o")"
[ "$t_after" != "$t_before" ] && pass "S5: touching audio_common/include/wav_io.h recompiled AEC's aec_wav.o" \
  || fail "S5: aec_wav.o did NOT recompile after touching wav_io.h"

# NE10 vendored header touch -> NE10 objects recompile (audio_common level).
# NE10_dsp.h (not the top-level NE10.h umbrella -- verified via NE10_fft.o's
# own -MD -MP .d fragment that NE10_fft.c pulls in NE10_dsp.h/NE10_types.h/
# NE10_macros.h transitively, but never NE10.h itself; only THIS repo's own
# fft_wrapper_ne10.c includes the top-level NE10.h directly).
sleep 1
touch "$AC_DIR/lib/ne10/inc/NE10_dsp.h"
ac_ne10_objdir="$(make -s -C "$AC_DIR" BACKEND=ne10 print-obj-dir)"
t_before="$(mtime "$ac_ne10_objdir/NE10_fft.o" 2>/dev/null || echo 0)"
make -s -C "$AC_DIR" BACKEND=ne10 lib >/dev/null
t_after="$(mtime "$ac_ne10_objdir/NE10_fft.o")"
[ "$t_after" != "$t_before" ] && pass "S5: touching a vendored NE10 header (NE10_dsp.h) recompiled NE10_fft.o" \
  || fail "S5: NE10_fft.o did NOT recompile after touching lib/ne10/inc/NE10_dsp.h"

# A->B->A at the AEC level -> nm shows backend-appropriate FFT symbols each
# time (never a stale symbol set from the OTHER backend's link).
ac_kiss="$(make -s -C "$AC_DIR" BACKEND=kiss print-lib-path)"
make -s BACKEND=kiss all AC_LIB="$ac_kiss" >/dev/null
bd_kiss="$(make -s BACKEND=kiss print-bin-dir AC_LIB="$ac_kiss")"
ac_ne10="$(make -s -C "$AC_DIR" BACKEND=ne10 print-lib-path)"
make -s BACKEND=ne10 all AC_LIB="$ac_ne10" >/dev/null
bd_ne10="$(make -s BACKEND=ne10 print-bin-dir AC_LIB="$ac_ne10")"
ac_kiss2="$(make -s -C "$AC_DIR" BACKEND=kiss print-lib-path)"
make -s BACKEND=kiss all AC_LIB="$ac_kiss2" >/dev/null
bd_kiss2="$(make -s BACKEND=kiss print-bin-dir AC_LIB="$ac_kiss2")"

ok=1
nm "$bd_kiss/aec_wav"  2>/dev/null | grep -q 'ne10_fft_r2c_1d_float32_neon' && ok=0
nm "$bd_ne10/aec_wav"  2>/dev/null | grep -q 'ne10_fft_r2c_1d_float32_neon' || ok=0
nm "$bd_kiss2/aec_wav" 2>/dev/null | grep -q 'ne10_fft_r2c_1d_float32_neon' && ok=0
[ "$ok" -eq 1 ] && pass "S5: A->B->A at the AEC level -- nm shows backend-appropriate FFT symbols every time" \
  || fail "S5: A->B->A at the AEC level -- wrong/stale FFT symbol set in one of the three aec_wav builds"

echo "############################################################"
echo "# S7p: producer-publish (v4 layout; throwaway DIST_ROOT)"
echo "############################################################"
cd "$AC_DIR"
# round-5 review P1: throwaway DIST_ROOT under our own mktemp -d -- the real
# dist/ is never read, written, or removed by this script.
S7P_DIST="$(new_tmpdir)/dist"

S7P_LOG_FIRST="$(new_tmpfile)"
make -s BACKEND=kiss publish DIST_ROOT="$S7P_DIST" >"$S7P_LOG_FIRST" 2>&1
grep -q "(attested: attest-" "$S7P_LOG_FIRST" && pass "S7p: kiss publish success line ends with '(attested: <name>)'" \
  || fail "S7p: kiss publish success line missing '(attested: <name>)'"

kiss_current_target="$(readlink "$S7P_DIST/kiss/current" || true)"
[ -n "$kiss_current_target" ] && [ -d "$S7P_DIST/kiss/$kiss_current_target" ] && \
  pass "S7p: kiss publish -- current symlink resolves to a real release dir" \
  || fail "S7p: kiss publish -- current symlink broken or missing"

# v4 layout: the release dir is $S7P_DIST/<backend>/<cfg_sig>-<content12>,
# resolved ONLY via readlink above -- never hardcoded here. MANIFEST.txt is
# now DETERMINISTIC (config + tool identities + per-file sha256 only): no
# git_commit=/git_dirty=/date_utc= any more -- those moved to ATTEST/.
manifest_path="$S7P_DIST/kiss/current/MANIFEST.txt"
grep -q "^release_id=$kiss_current_target$" "$manifest_path" && \
  pass "S7p: kiss MANIFEST release_id= matches the resolved current target (v4 content-addressed id)" \
  || fail "S7p: kiss MANIFEST release_id= missing or does not match readlink target ($kiss_current_target)"
if grep -q "^git_commit=" "$manifest_path" || grep -q "^git_dirty=" "$manifest_path" || grep -q "^date_utc=" "$manifest_path"; then
  fail "S7p: kiss MANIFEST unexpectedly has a git_commit=/git_dirty=/date_utc= line (v4 MANIFEST must be deterministic)"
else
  pass "S7p: kiss MANIFEST has NO git_commit=/git_dirty=/date_utc= line (v4 deterministic MANIFEST)"
fi
if grep -q "^ar=" "$manifest_path" && grep -q "^ranlib=" "$manifest_path" && grep -q "^link=" "$manifest_path"; then
  pass "S7p: kiss MANIFEST has ar=/ranlib=/link= lines (round-4 P2-2 fields)"
else
  fail "S7p: kiss MANIFEST missing one of ar=/ranlib=/link= lines"
fi

# git_commit=/git_dirty=/date_utc= now live in ATTEST/, one file per publish
# event -- this first publish must have produced exactly one.
attest_files="$(find "$S7P_DIST/kiss/current/ATTEST" -name 'attest-*.txt')"
attest_n="$(printf '%s\n' "$attest_files" | grep -c . || true)"
[ "$attest_n" -eq 1 ] && pass "S7p: kiss release has exactly one ATTEST/attest-*.txt after the first publish" \
  || fail "S7p: kiss release ATTEST/ has $attest_n attest-*.txt file(s) after the first publish (expected 1)"
if grep -q "^git_commit=" "$attest_files" && grep -q "^git_dirty=" "$attest_files" && grep -q "^date_utc=" "$attest_files"; then
  pass "S7p: ATTEST file carries git_commit=/git_dirty=/date_utc="
else
  fail "S7p: ATTEST file $attest_files missing git_commit=/git_dirty=/date_utc="
fi

manifest_ok=1
while read -r sha fname; do
  [ "$fname" = "MANIFEST.txt" ] && continue
  actual="$(file_sha "$S7P_DIST/kiss/current/$fname")"
  [ "$actual" = "$sha" ] || manifest_ok=0
done < <(grep -E '^[0-9a-f]{64}  ' "$manifest_path")
[ "$manifest_ok" -eq 1 ] && pass "S7p: kiss MANIFEST shas match the files on disk" \
  || fail "S7p: kiss MANIFEST sha mismatch against files on disk"

kiss_dist_sha_before="$(file_sha "$S7P_DIST/kiss/current/libaudio_common.a")"
make -s BACKEND=ne10 publish DIST_ROOT="$S7P_DIST" >/dev/null
kiss_dist_sha_after="$(file_sha "$S7P_DIST/kiss/current/libaudio_common.a")"
[ "$kiss_dist_sha_before" = "$kiss_dist_sha_after" ] && pass "S7p: publishing ne10 left the kiss release untouched" \
  || fail "S7p: publishing ne10 CHANGED the kiss release"

# Concurrent same-backend publish: v4's lock-FIRST driver means a genuine
# race resolves to exactly one winner in practice (see S14 for a version of
# this built around a fresh, slow-to-build config that makes "exactly one
# winner" non-flaky); here, since this config is already built, a wide
# scheduling gap between the two backgrounded callers could plausibly let
# both complete in turn -- so this check stays deliberately lenient (as it
# always has been): either one caller fails fast with "already held", or
# both complete; either way `current` must end up pointing at a COMPLETE,
# self-consistent release.
S7P_LOG_A="$(new_tmpfile)"; S7P_LOG_B="$(new_tmpfile)"
( make -s BACKEND=kiss publish DIST_ROOT="$S7P_DIST" >"$S7P_LOG_A" 2>&1 ) & cp1=$!
( make -s BACKEND=kiss publish DIST_ROOT="$S7P_DIST" >"$S7P_LOG_B" 2>&1 ) & cp2=$!
cr1=0; cr2=0
wait "$cp1" || cr1=$?
wait "$cp2" || cr2=$?
if [ "$cr1" -eq 0 ] || [ "$cr2" -eq 0 ]; then
  pass "S7p: concurrent same-backend publish -- at least one caller succeeded"
else
  fail "S7p: concurrent same-backend publish -- BOTH callers failed"
  cat "$S7P_LOG_A" "$S7P_LOG_B" >&2
fi
if grep -q "already held" "$S7P_LOG_A" "$S7P_LOG_B" 2>/dev/null || { [ "$cr1" -eq 0 ] && [ "$cr2" -eq 0 ]; }; then
  pass "S7p: concurrent same-backend publish -- lock serialised (one failed fast with 'already held', or both completed in turn)"
else
  fail "S7p: concurrent same-backend publish -- no evidence of lock serialisation (both ran with no contention logged)"
fi
final_target="$(readlink "$S7P_DIST/kiss/current" || true)"
[ -n "$final_target" ] && [ -f "$S7P_DIST/kiss/$final_target/MANIFEST.txt" ] && [ -f "$S7P_DIST/kiss/$final_target/libaudio_common.a" ] && \
  pass "S7p: after concurrent publish, current points at a complete release dir" \
  || fail "S7p: after concurrent publish, current points at an incomplete/missing release dir"
manifest_ok=1
while read -r sha fname; do
  [ "$fname" = "MANIFEST.txt" ] && continue
  actual="$(file_sha "$S7P_DIST/kiss/$final_target/$fname")"
  [ "$actual" = "$sha" ] || manifest_ok=0
done < <(grep -E '^[0-9a-f]{64}  ' "$S7P_DIST/kiss/$final_target/MANIFEST.txt")
[ "$manifest_ok" -eq 1 ] && pass "S7p: final current release's MANIFEST is self-consistent" \
  || fail "S7p: final current release's MANIFEST does not match its own files"

echo "############################################################"
echo "# S8: CFG_SIG collision guard"
echo "############################################################"
cd "$AC_DIR"
kiss_objdir="$(make -s BACKEND=kiss print-obj-dir)"
manifest="$kiss_objdir/config.manifest"
backup="$(new_tmpfile)"
cp "$manifest" "$backup"
echo "CORRUPTED PAYLOAD (test_build_isolation.sh S8)" > "$manifest"
s8_log="$(new_tmpfile)"
if make BACKEND=kiss lib >"$s8_log" 2>&1; then
  fail "S8: corrupted config.manifest did NOT fail the next build (collision guard not firing)"
else
  grep -q "CFG_SIG collision" "$s8_log" && pass "S8: corrupted config.manifest correctly FAILS the next build with the collision message" \
    || fail "S8: build failed but NOT with the expected collision message"
fi
cp "$backup" "$manifest"
make -s BACKEND=kiss lib >/dev/null && pass "S8: manifest restored, build green again" \
  || fail "S8: build still failing after manifest restore"

echo "############################################################"
echo "# S9: command-line override rejection (round-4 review P1-1)"
echo "############################################################"
cd "$AC_DIR"

# Plain baseline (no BACKEND= given -- whatever this Makefile auto-detects
# from the compiler is fine; only used below to prove EXTRA_CFLAGS lands
# somewhere DIFFERENT, never compared against a hardcoded backend).
baseline_objdir="$(make -s print-obj-dir)"

# CFLAGS=/CXXFLAGS=/LDFLAGS=/FP_POLICY= set on the command line must be
# rejected at PARSE time (before any obj/bin dir is even created) with a
# message mentioning "cannot be overridden" -- these variables are internal
# (this Makefile's own +=/:= appends to them would be silently defeated by a
# command-line origin, per GNU Make semantics); EXTRA_CFLAGS/EXTRA_LDFLAGS are
# the supported hook instead.
for pair in "CFLAGS=-O3" "CXXFLAGS=-O0" "LDFLAGS=-lfoo" "FP_POLICY=-ffp-contract=fast"; do
  ov_var="${pair%%=*}"
  ov_val="${pair#*=}"
  S9_LOG="$(new_tmpfile)"
  if make "$ov_var=$ov_val" print-obj-dir >"$S9_LOG" 2>&1; then
    fail "S9: make $ov_var=$ov_val print-obj-dir unexpectedly SUCCEEDED (must be rejected at parse time)"
  else
    if grep -q "cannot be overridden" "$S9_LOG"; then
      pass "S9: make $ov_var=$ov_val print-obj-dir correctly FAILS, mentioning 'cannot be overridden'"
    else
      fail "S9: make $ov_var=$ov_val print-obj-dir failed but WITHOUT the expected 'cannot be overridden' message"
      cat "$S9_LOG" >&2
    fi
  fi
done

# EXTRA_CFLAGS is the supported hook: must still succeed, and (being folded
# into CFG_SIG) must land in a DIFFERENT keyed dir than the plain baseline.
extra_objdir="$(make -s EXTRA_CFLAGS=-DS9_PROBE print-obj-dir)"
[ -n "$extra_objdir" ] && [ "$extra_objdir" != "$baseline_objdir" ] && \
  pass "S9: EXTRA_CFLAGS=-DS9_PROBE print-obj-dir succeeds and lands in a different keyed dir than the plain baseline" \
  || fail "S9: EXTRA_CFLAGS=-DS9_PROBE print-obj-dir did NOT differ from the plain baseline ($baseline_objdir vs $extra_objdir)"

echo "############################################################"
echo "# S10: archive freshness -- stale-member removal (round-4 review P1-4)"
echo "############################################################"
cd "$AC_DIR"

make -s BACKEND=kiss lib >/dev/null
kiss_lib="$(make -s BACKEND=kiss print-lib-path)"

# Inject a foreign object directly into the delivered archive (simulating
# what an old `ar rc` onto an EXISTING archive would leave behind forever for
# a source file that's since been removed from the SRCS list).
S10_TMPDIR="$(new_tmpdir)"
cat > "$S10_TMPDIR/s10_foreign.c" <<'EOF'
int s10_foreign_symbol(void) { return 42; }
EOF
cc -c -o "$S10_TMPDIR/s10_foreign.o" "$S10_TMPDIR/s10_foreign.c"
ar r "$kiss_lib" "$S10_TMPDIR/s10_foreign.o"
ar -t "$kiss_lib" | grep -q '^s10_foreign\.o$' && pass "S10: foreign object successfully injected into archive (pre-condition)" \
  || fail "S10: failed to inject foreign object into archive (pre-condition broken -- cannot test freshness)"

# A normal rebuild (source touched, config unchanged) must produce a FRESH
# archive (rm -f $@.tmp; ar rc $@.tmp $(BE_OBJS); ranlib; mv -f) -- never
# `ar rc` onto the existing archive -- so the foreign member must be GONE
# afterward, while hpf.o (a real, current source) must still be present.
# The touch below is mtime-only (content-neutral): it forces the recompile
# without ever modifying src/hpf.c's bytes.
sleep 1
touch "$AC_DIR/src/hpf.c"
make -s BACKEND=kiss lib >/dev/null

ar -t "$kiss_lib" | grep -q '^s10_foreign\.o$' && fail "S10: stale foreign member s10_foreign.o SURVIVED a fresh archive rebuild (ar rc onto an existing archive?)" \
  || pass "S10: stale foreign member s10_foreign.o REMOVED by the fresh-archive rebuild"
ar -t "$kiss_lib" | grep -q '^hpf\.o$' && pass "S10: rebuilt archive still contains hpf.o" \
  || fail "S10: rebuilt archive missing hpf.o"

# SRCS= (the sorted source list) participates in CFG_SIG_PAYLOAD, visible
# verbatim in this config's config.manifest.
kiss_objdir="$(make -s BACKEND=kiss print-obj-dir)"
manifest="$kiss_objdir/config.manifest"
if grep -q 'SRCS=' "$manifest" && grep -q 'src/hpf\.c' "$manifest"; then
  pass "S10: config.manifest's SRCS= entry participates in CFG_SIG and lists src/hpf.c"
else
  fail "S10: config.manifest missing SRCS= or src/hpf.c reference ($manifest)"
fi

echo "############################################################"
echo "# S11: publish immutability / content-addressing"
echo "# (round-4 review P2-2 -> round-5 v4 semantics)"
echo "############################################################"
cd "$AC_DIR"

# Captured BEFORE any of S11's sub-parts run, so S11c's "the REAL repo's
# src/hpf.c is untouched" assertion works even if the user already had local
# edits to it at scenario start -- never assume a clean tree, never
# `git checkout --` to force one.
HPF_SHA_BEFORE_S11="$(file_sha "$AC_DIR/src/hpf.c")"

# dir mtime + every immediate FILE's mtime. `find -maxdepth 1 -type f`
# naturally excludes the ATTEST/ subdirectory (it's a directory, not a file)
# and everything under it -- this snapshot is deliberately blind to ATTEST/
# growing a new attestation file on every republish.
release_snapshot() {
  local dir="$1"
  { mtime "$dir"; find "$dir" -maxdepth 1 -type f | sort | while read -r f; do echo "$(basename "$f"):$(mtime "$f")"; done; }
}
count_attest() { find "$1/ATTEST" -name 'attest-*.txt' 2>/dev/null | grep -c . || true; }

echo "--- S11a: in-tree republish (throwaway DIST_ROOT) --------------------"
S11A_DIST="$(new_tmpdir)/dist"
make -s BACKEND=kiss publish DIST_ROOT="$S11A_DIST" >/dev/null
id1="$(readlink "$S11A_DIST/kiss/current" || true)"
[ -n "$id1" ] && [ -d "$S11A_DIST/kiss/$id1" ] && \
  pass "S11a: first kiss publish -- current resolves to a real release dir ($id1)" \
  || fail "S11a: first kiss publish -- current symlink broken or missing"

manifest1="$S11A_DIST/kiss/$id1/MANIFEST.txt"
if [ -f "$manifest1" ] && grep -q "^release_id=$id1$" "$manifest1" \
   && ! grep -q "^git_commit=" "$manifest1" && ! grep -q "^git_dirty=" "$manifest1" && ! grep -q "^date_utc=" "$manifest1"; then
  pass "S11a: MANIFEST.txt has release_id= and NO git_commit=/git_dirty=/date_utc= (v4 deterministic MANIFEST)"
else
  fail "S11a: MANIFEST.txt missing release_id= or unexpectedly has a git_commit=/git_dirty=/date_utc= line ($manifest1)"
fi
if grep -q "^ar=" "$manifest1" && grep -q "^ranlib=" "$manifest1" && grep -q "^link=" "$manifest1"; then
  pass "S11a: MANIFEST.txt has ar=/ranlib=/link= lines"
else
  fail "S11a: MANIFEST.txt missing one of ar=/ranlib=/link= lines"
fi

n_attest="$(count_attest "$S11A_DIST/kiss/$id1")"
[ "$n_attest" -eq 1 ] && pass "S11a: ATTEST/ has exactly 1 attest-*.txt after the first publish" \
  || fail "S11a: ATTEST/ has $n_attest attest-*.txt file(s) after the first publish (expected 1)"
attest1="$(find "$S11A_DIST/kiss/$id1/ATTEST" -name 'attest-*.txt')"
if grep -q "^git_commit=" "$attest1" && grep -q "^git_dirty=" "$attest1" && grep -q "^date_utc=" "$attest1"; then
  pass "S11a: the ATTEST file carries git_commit=/git_dirty=/date_utc="
else
  fail "S11a: ATTEST file $attest1 missing git_commit=/git_dirty=/date_utc="
fi

snap_before="$(release_snapshot "$S11A_DIST/kiss/$id1")"

# Republishing IDENTICAL content: byte-verified against the stored release,
# `current` repointed to the SAME id, artifact/MANIFEST mtimes untouched --
# only ATTEST/ gains a second file. `sleep 1` so the second attest gets a
# distinct <utcstamp> (two publishes in the same second idempotently
# overwrite the same attest filename instead of adding a second one).
sleep 1
S11A_LOG="$(new_tmpfile)"
make -s BACKEND=kiss publish DIST_ROOT="$S11A_DIST" >"$S11A_LOG" 2>&1
grep -q "byte-verified" "$S11A_LOG" && pass "S11a: identical-content republish reports 'byte-verified'" \
  || fail "S11a: identical-content republish did NOT report 'byte-verified'"

id1_again="$(readlink "$S11A_DIST/kiss/current" || true)"
[ "$id1_again" = "$id1" ] && pass "S11a: current still points at id1 after identical-content republish" \
  || fail "S11a: current CHANGED after identical-content republish ($id1 -> $id1_again)"

snap_after="$(release_snapshot "$S11A_DIST/kiss/$id1")"
[ "$snap_before" = "$snap_after" ] && pass "S11a: release dir + artifact/MANIFEST mtimes UNCHANGED by identical-content republish" \
  || fail "S11a: release dir/artifact mtimes CHANGED by identical-content republish"

n_attest2="$(count_attest "$S11A_DIST/kiss/$id1")"
[ "$n_attest2" -eq 2 ] && pass "S11a: ATTEST/ has exactly 2 attest-*.txt files after the republish" \
  || fail "S11a: ATTEST/ has $n_attest2 attest-*.txt file(s) after the republish (expected 2)"

echo "--- S11b: MANIFEST tamper detection -----------------------------------"
S11B_DIST="$(new_tmpdir)/dist"
make -s BACKEND=kiss publish DIST_ROOT="$S11B_DIST" >/dev/null
idb="$(readlink "$S11B_DIST/kiss/current" || true)"
printf 'X' >> "$S11B_DIST/kiss/$idb/MANIFEST.txt"
S11B_LOG="$(new_tmpfile)"
if make -s BACKEND=kiss publish DIST_ROOT="$S11B_DIST" >"$S11B_LOG" 2>&1; then
  fail "S11b: republish after tampering with the stored MANIFEST unexpectedly SUCCEEDED"
else
  grep -q "differs from staged content" "$S11B_LOG" && \
    pass "S11b: republish after MANIFEST tamper correctly FAILS, mentioning 'differs from staged content'" \
    || fail "S11b: republish after MANIFEST tamper failed but WITHOUT the expected message"
fi

echo "--- S11c: content-change publish, exercised in a SANDBOX --------------"
# Never in $AC_DIR: a real codegen change is needed here (a comment-only edit
# produces a byte-identical object -- and the archiver runs in deterministic
# mode, so it doesn't even pick up a new member timestamp -- so the content-
# addressed release id CORRECTLY stays the same, which is the immutability
# property under test elsewhere, not a bug). To exercise "same flags,
# different content -> new release id" without ever touching the real repo,
# this copies the tree into a throwaway sandbox and edits THAT copy.
SANDBOX="$(new_tmpdir)"
mkdir "$SANDBOX/ac"
tar -c -C "$AC_DIR" --exclude obj --exclude bin --exclude dist --exclude .git . | tar -x -C "$SANDBOX/ac"
S11C_DIST="$(new_tmpdir)/dist"

make -C "$SANDBOX/ac" -s BACKEND=kiss publish DIST_ROOT="$S11C_DIST" >/dev/null
sid1="$(readlink "$S11C_DIST/kiss/current" || true)"
[ -n "$sid1" ] && pass "S11c: sandbox publish #1 -- current resolves ($sid1)" \
  || fail "S11c: sandbox publish #1 -- current symlink broken or missing"

echo "void s11_isolation_probe(void) {}" >> "$SANDBOX/ac/src/hpf.c"
make -C "$SANDBOX/ac" -s BACKEND=kiss publish DIST_ROOT="$S11C_DIST" >/dev/null
sid2="$(readlink "$S11C_DIST/kiss/current" || true)"

[ -n "$sid2" ] && [ "$sid2" != "$sid1" ] && pass "S11c: sandbox content change -- publish produced a NEW release id ($sid1 -> $sid2)" \
  || fail "S11c: sandbox content change did NOT produce a new release id (sid1=$sid1 sid2=$sid2)"
[ -d "$S11C_DIST/kiss/$sid1" ] && [ -d "$S11C_DIST/kiss/$sid2" ] && \
  pass "S11c: both sandbox release dirs (sid1, sid2) coexist -- old release never deleted" \
  || fail "S11c: sandbox release dir $sid1 or $sid2 missing after content change"

# ids are <cfg_sig>-<content12>; same flags => same cfg_sig prefix (the part
# before the LAST '-', since content12 itself is plain lowercase hex with no
# dashes).
scfg1="${sid1%-*}"
scfg2="${sid2%-*}"
[ -n "$scfg1" ] && [ "$scfg1" = "$scfg2" ] && \
  pass "S11c: sid1/sid2 share the same cfg_sig prefix ($scfg1) -- flags unchanged, only content differed" \
  || fail "S11c: sid1/sid2 cfg_sig prefixes DIFFER ($scfg1 vs $scfg2) though only source content changed"

# The sandbox has no .git (excluded from the tar copy above), so the
# Makefile's `git rev-parse` there falls back to a placeholder value --
# deliberately NOT asserting on that field's exact contents here, since it's
# an artifact of running outside any checkout rather than something this
# isolation scenario is testing.

# The REAL repo's src/hpf.c must be untouched: compare against the shasum
# captured at the very start of S11, not `git diff --quiet` (which would
# wrongly assume the file was clean to begin with).
HPF_SHA_AFTER_S11="$(file_sha "$AC_DIR/src/hpf.c")"
[ "$HPF_SHA_BEFORE_S11" = "$HPF_SHA_AFTER_S11" ] && pass "S11c: the REAL repo's src/hpf.c is byte-identical to its pre-scenario state" \
  || fail "S11c: the REAL repo's src/hpf.c CHANGED during S11 (sandbox isolation failed)"

echo "############################################################"
echo "# S12: toolchain coherence guard (round-4 review P1-2)"
echo "############################################################"
cd "$AC_DIR"

# fake-cxx: reports a bogus -dumpmachine triple (so it can never match a
# real CC's), otherwise transparently forwards to the real c++ so a build
# using it as CXX still actually compiles/links.
S12_TMPDIR="$(new_tmpdir)"
SHIM="$S12_TMPDIR/fake-cxx"
cat > "$SHIM" <<'EOF'
#!/usr/bin/env bash
if [ "$1" = "-dumpmachine" ]; then
  echo "s12-mismatched-triple"
  exit 0
fi
exec c++ "$@"
EOF
chmod +x "$SHIM"

S12_LOG="$(new_tmpfile)"
if make BACKEND=ne10 CXX="$SHIM" lib >"$S12_LOG" 2>&1; then
  fail "S12: make BACKEND=ne10 CXX=<mismatched shim> lib unexpectedly SUCCEEDED (toolchain guard did not fire)"
else
  grep -q "different targets" "$S12_LOG" && pass "S12: BACKEND=ne10 with a mismatched CXX -dumpmachine FAILS, mentioning 'different targets'" \
    || fail "S12: BACKEND=ne10 with a mismatched CXX FAILED but WITHOUT the expected 'different targets' message"
fi

S12_LOG2="$(new_tmpfile)"
if make BACKEND=ne10 CXX="$SHIM" TOOLCHAIN_CHECK=0 lib >"$S12_LOG2" 2>&1; then
  pass "S12: TOOLCHAIN_CHECK=0 skips the guard -- BACKEND=ne10 build with the mismatched CXX shim SUCCEEDS (its own keyed dir)"
else
  fail "S12: TOOLCHAIN_CHECK=0 build with the mismatched CXX shim FAILED (guard should have been skipped)"
  cat "$S12_LOG2" >&2
fi

S12_LOG3="$(new_tmpfile)"
if make BACKEND=kiss CXX="$SHIM" lib >"$S12_LOG3" 2>&1; then
  pass "S12: BACKEND=kiss with the mismatched CXX shim SUCCEEDS (guard is ne10-only)"
else
  fail "S12: BACKEND=kiss with the mismatched CXX shim FAILED (guard should never run for kiss)"
  cat "$S12_LOG3" >&2
fi

echo "############################################################"
echo "# S13: atomic \`current\` symlink swap hammer (round-5 review P2)"
echo "############################################################"
S13_TMPDIR="$(new_tmpdir)"
cc -O2 -o "$S13_TMPDIR/swapln" "$AC_DIR/tools/atomic_symlink_swap.c"
mkdir -p "$S13_TMPDIR/d/relA" "$S13_TMPDIR/d/relB"
"$S13_TMPDIR/swapln" relA "$S13_TMPDIR/d/current"

# Background sampler: reads `current` as fast as possible. The assertion is
# on HARD misses -- a failed readlink whose IMMEDIATE re-read ALSO fails.
# rename(2) leaves no window in the link's existence, so a correct swapln
# produces zero hard misses; the old rm+mv fallback had a real unlinked
# window (and dropped its temp INTO the target dir), which is exactly what a
# hard miss would expose. A rare SINGLE-read transient failure is a
# measured macOS path-lookup-vs-rename race in the READER, not a gap in the
# link (calibrated on this host: 2000-swap hammer -> 2 single-read
# transients, 0 double-read failures, while a never-swapped control link
# sampled at the same rate showed 0 misses of any kind -- i.e. the transient
# tracks concurrent rename activity in the kernel's name lookup, not the
# helper). Soft misses are reported as INFO, never a failure.
S13_SAMPLER_LOG="$(new_tmpfile)"
(
  while true; do
    v="$(readlink "$S13_TMPDIR/d/current" 2>/dev/null || true)"
    if [ -z "$v" ]; then
      v2="$(readlink "$S13_TMPDIR/d/current" 2>/dev/null || true)"
      if [ -z "$v2" ]; then printf 'HARDMISS\n' >> "$S13_SAMPLER_LOG"; \
      else printf 'SOFTMISS\n' >> "$S13_SAMPLER_LOG"; fi
    fi
  done
) &
s13_sampler_pid=$!

# 200 alternating foreground swaps between relA and relB.
for i in $(seq 1 200); do
  if [ $((i % 2)) -eq 1 ]; then
    "$S13_TMPDIR/swapln" relA "$S13_TMPDIR/d/current"
  else
    "$S13_TMPDIR/swapln" relB "$S13_TMPDIR/d/current"
  fi
done

kill "$s13_sampler_pid" 2>/dev/null || true
wait "$s13_sampler_pid" 2>/dev/null || true

# NOTE: `grep -c` prints its 0 itself on no-match (exiting 1) -- an
# `|| echo 0` here would produce a two-line "0\n0" and break the -eq test.
hard_count="$(grep -c HARDMISS "$S13_SAMPLER_LOG" 2>/dev/null || true)"
hard_count="${hard_count:-0}"
soft_count="$(grep -c SOFTMISS "$S13_SAMPLER_LOG" 2>/dev/null || true)"
soft_count="${soft_count:-0}"
[ "$soft_count" -gt 0 ] && echo "  INFO: S13: $soft_count single-read transient(s) (measured macOS lookup-vs-rename reader race; immediate re-read succeeded every time -- see comment above)"
[ "$hard_count" -eq 0 ] && pass "S13: readlink sampler recorded ZERO hard misses across 200 alternating swaps (no missing-\`current\` window)" \
  || fail "S13: readlink sampler recorded $hard_count HARD miss(es) -- \`current\` was persistently missing/broken mid-swap"

leftover="$(ls "$S13_TMPDIR/d" 2>/dev/null | grep -cE '^current\.[0-9]+\.tmp$' || true)"
leftover="${leftover:-0}"
[ "$leftover" -eq 0 ] && pass "S13: no current.*.tmp leftovers under $S13_TMPDIR/d" \
  || fail "S13: $leftover leftover current.*.tmp temp file(s) under $S13_TMPDIR/d"

s13_final="$(readlink "$S13_TMPDIR/d/current" || true)"
if [ "$s13_final" = "relA" ] || [ "$s13_final" = "relB" ]; then
  pass "S13: final current resolves to relA or relB ($s13_final)"
else
  fail "S13: final current does NOT resolve to relA or relB ($s13_final)"
fi

echo "############################################################"
echo "# S14: lock-before-build -- a concurrent publish loser builds nothing"
echo "#       (round-5 review P1)"
echo "############################################################"
cd "$AC_DIR"
S14_DIST="$(new_tmpdir)/dist"

# A fresh, never-before-built config (BACKEND=kiss + a probe define unique to
# this scenario) -- the winner's critical section (a real compile+link+stage,
# not a no-op) takes long enough that two callers launched back-to-back
# genuinely contend for the lock, making "exactly one winner" a reliable
# assertion rather than a scheduling-dependent guess (contrast with S7p's
# already-built-config concurrency check, which stays deliberately lenient).
S14_LOG_A="$(new_tmpfile)"; S14_LOG_B="$(new_tmpfile)"
S14_RC_A="$(new_tmpfile)"; S14_RC_B="$(new_tmpfile)"
# `rc=0; cmd || rc=$?; ...` (not `cmd; echo $? >file`): under `set -e`, a
# bare failing command followed by `;` would abort this subshell BEFORE the
# echo ever ran. A command that is not the final one in a `||` list is
# exempt from triggering errexit, so this form reliably captures the make
# invocation's real exit status into its rc file even when it fails.
( rc=0; make -s BACKEND=kiss EXTRA_CFLAGS=-DS14_LOCK_PROBE publish DIST_ROOT="$S14_DIST" >"$S14_LOG_A" 2>&1 || rc=$?; echo "$rc" > "$S14_RC_A" ) &
p1=$!
( rc=0; make -s BACKEND=kiss EXTRA_CFLAGS=-DS14_LOCK_PROBE publish DIST_ROOT="$S14_DIST" >"$S14_LOG_B" 2>&1 || rc=$?; echo "$rc" > "$S14_RC_B" ) &
p2=$!
wait "$p1" || true
wait "$p2" || true
rc_a="$(cat "$S14_RC_A")"
rc_b="$(cat "$S14_RC_B")"

if { [ "$rc_a" -eq 0 ] && [ "$rc_b" -ne 0 ]; } || { [ "$rc_a" -ne 0 ] && [ "$rc_b" -eq 0 ]; }; then
  pass "S14: exactly one of the two concurrent publishers succeeded (rc_a=$rc_a rc_b=$rc_b)"
else
  fail "S14: expected exactly one winner, got rc_a=$rc_a rc_b=$rc_b"
fi

if [ "$rc_a" -eq 0 ]; then
  winner_log="$S14_LOG_A"; loser_log="$S14_LOG_B"
else
  winner_log="$S14_LOG_B"; loser_log="$S14_LOG_A"
fi

grep -q "already held" "$loser_log" && pass "S14: loser's log contains 'already held'" \
  || fail "S14: loser's log does NOT contain 'already held' ($loser_log)"
grep -q "audio_common \[kiss\]" "$loser_log" && fail "S14: loser's log unexpectedly contains 'audio_common [kiss]' -- it built the archive despite losing the lock" \
  || pass "S14: loser's log does NOT contain 'audio_common [kiss]' -- it built nothing"
grep -q "audio_common \[kiss\]" "$winner_log" && pass "S14: winner's log DOES contain 'audio_common [kiss]' -- it actually built the archive" \
  || fail "S14: winner's log missing 'audio_common [kiss]' (expected it to have built a fresh config)"

s14_target="$(readlink "$S14_DIST/kiss/current" || true)"
if [ -n "$s14_target" ] && [ -f "$S14_DIST/kiss/$s14_target/MANIFEST.txt" ] && ar -t "$S14_DIST/kiss/$s14_target/libaudio_common.a" >/dev/null 2>&1; then
  pass "S14: winner's release dir is complete (MANIFEST.txt present, libaudio_common.a readable by ar -t)"
else
  fail "S14: winner's release dir incomplete or archive unreadable ($S14_DIST/kiss/$s14_target)"
fi

stage_leftovers="$(find "$S14_DIST/kiss" -maxdepth 1 -name '.stage.*' 2>/dev/null | grep -c . || true)"
[ "$stage_leftovers" -eq 0 ] && pass "S14: no .stage.* leftovers under $S14_DIST/kiss/" \
  || fail "S14: $stage_leftovers .stage.* leftover(s) under $S14_DIST/kiss/"

# --- final safety check: this whole run must leave $AC_DIR's tracked-file --
# state bit-identical (round-5 review P1). Checked here, before the summary
# tally, so any drift shows up as a normal FAIL line rather than a silent
# discrepancy the reader has to notice on their own.
GIT_STATUS_AFTER="$(git_status_hash)"
[ "$GIT_STATUS_BEFORE" = "$GIT_STATUS_AFTER" ] && pass "repo tree drift check: \$AC_DIR git status unchanged across the whole run" \
  || fail "repo tree drift check: \$AC_DIR git status CHANGED during this run ($GIT_STATUS_BEFORE -> $GIT_STATUS_AFTER)"

echo "############################################################"
echo "SUMMARY: $PASS_COUNT passed, $FAIL_COUNT failed"
echo "############################################################"
if [ "$FAIL_COUNT" -gt 0 ]; then
  echo "Failures:" >&2
  for f in "${FAILURES[@]}"; do echo "  - $f" >&2; done
  exit 1
fi
exit 0
