#!/usr/bin/env bash
# test_build_isolation.sh — round-3/round-4 review build-isolation regression
# suite.
#
# Exercises the CFG_SIG-keyed obj/bin directory design (audio_common's own
# Makefile, plus the two-phase AC_LIB consumer resolution in AEC/c_impl and
# NR/c_impl) against every failure mode the round-3 B01 finding described:
# mtime-staleness delivering another config's artifact, test_ne10_force_c
# overwriting the normal NE10 archive, parallel different-config builds
# co-writing, and consumers hardcoding flat paths / forwarding only BACKEND
# (config skew) -- plus, from the round-4 review, command-line override
# rejection (P1-1), fresh-archive stale-member removal (P1-4), publish v3
# content-addressing (P2-2), and the CC/CXX toolchain-coherence guard (P1-2).
#
# Scenario index:
#   S1  - A->B->A + -O0/-O3 delivered-object repro
#   S2  - same-second kiss<->ne10 switching
#   S3  - parallel differing-config builds
#   S4  - test_ne10_force_c isolation
#   S5  - consumer correctness (drives AEC)
#   S7p - producer-publish (v3 content-addressed dist/ layout)
#   S8  - CFG_SIG collision guard
#   S9  - command-line override rejection (round-4 P1-1)
#   S10 - archive freshness / stale-member removal (round-4 P1-4)
#   S11 - publish immutability / content-addressing (round-4 P2-2)
#   S12 - toolchain coherence guard (round-4 P1-2)
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
#
# Usage: ./scripts/test_build_isolation.sh   (run from audio_common/, or
# anywhere — paths are resolved relative to this script's own location).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
AEC_DIR="$(cd "$AC_DIR/../AEC/c_impl" && pwd)"
NR_DIR="$(cd "$AC_DIR/../NR/c_impl" && pwd)"

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
S3_LOG_A="$(mktemp)"; S3_LOG_B="$(mktemp)"
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
rm -f "$S3_LOG_A" "$S3_LOG_B"

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

S4_LOG="$(mktemp)"
if make test_ne10_force_c >"$S4_LOG" 2>&1; then
  pass "S4: test_ne10_force_c exits green"
else
  fail "S4: test_ne10_force_c FAILED"
  cat "$S4_LOG" >&2
fi
rm -f "$S4_LOG"

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

# touch audio_common/src/hpf.c -> hpf.o recompiles AND aec_wav relinks.
# audio_common's own CFG_SIG (a hash of its COMPILER INVOCATION, not file
# content) does not change from a source edit, so its archive path stays
# the same -- only its mtime should advance.
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
echo "# S7p: producer-publish"
echo "############################################################"
cd "$AC_DIR"
make -s BACKEND=kiss publish >/dev/null
kiss_current_target="$(readlink dist/kiss/current || true)"
[ -n "$kiss_current_target" ] && [ -d "dist/kiss/$kiss_current_target" ] && \
  pass "S7p: kiss publish -- current symlink resolves to a real release dir" \
  || fail "S7p: kiss publish -- current symlink broken or missing"

# v3 layout (round-4 review P2-2): the release dir is now
# dist/<backend>/<cfg_sig>-<content12>, resolved ONLY via readlink above --
# never hardcoded here -- and MANIFEST.txt carries several new fields tying
# the on-disk release id back to the manifest content itself.
grep -q "^release_id=$kiss_current_target$" "dist/kiss/current/MANIFEST.txt" && \
  pass "S7p: kiss MANIFEST release_id= matches the resolved current target (v3 content-addressed id)" \
  || fail "S7p: kiss MANIFEST release_id= missing or does not match readlink target ($kiss_current_target)"
grep -q "^git_dirty=[01]$" "dist/kiss/current/MANIFEST.txt" && \
  pass "S7p: kiss MANIFEST has a git_dirty= line (round-4 P2-2 field)" \
  || fail "S7p: kiss MANIFEST missing a git_dirty= line"
if grep -q "^ar=" "dist/kiss/current/MANIFEST.txt" && grep -q "^ranlib=" "dist/kiss/current/MANIFEST.txt" \
   && grep -q "^link=" "dist/kiss/current/MANIFEST.txt"; then
  pass "S7p: kiss MANIFEST has ar=/ranlib=/link= lines (round-4 P2-2 fields)"
else
  fail "S7p: kiss MANIFEST missing one of ar=/ranlib=/link= lines"
fi

manifest_ok=1
while read -r sha fname; do
  [ "$fname" = "MANIFEST.txt" ] && continue
  actual="$(file_sha "dist/kiss/current/$fname")"
  [ "$actual" = "$sha" ] || manifest_ok=0
done < <(grep -E '^[0-9a-f]{64}  ' "dist/kiss/current/MANIFEST.txt")
[ "$manifest_ok" -eq 1 ] && pass "S7p: kiss MANIFEST shas match the files on disk" \
  || fail "S7p: kiss MANIFEST sha mismatch against files on disk"

kiss_dist_sha_before="$(file_sha dist/kiss/current/libaudio_common.a)"
make -s BACKEND=ne10 publish >/dev/null
kiss_dist_sha_after="$(file_sha dist/kiss/current/libaudio_common.a)"
[ "$kiss_dist_sha_before" = "$kiss_dist_sha_after" ] && pass "S7p: publishing ne10 left the kiss dist/ release untouched" \
  || fail "S7p: publishing ne10 CHANGED the kiss dist/ release"

# Concurrent same-backend publish: lock must serialise. Either one caller
# fails with the lock message (acceptable) or both succeed (retry inside
# make); either way `current` must end up pointing at a COMPLETE,
# self-consistent release.
S7P_LOG_A="$(mktemp)"; S7P_LOG_B="$(mktemp)"
( make -s BACKEND=kiss publish >"$S7P_LOG_A" 2>&1 ) & cp1=$!
( make -s BACKEND=kiss publish >"$S7P_LOG_B" 2>&1 ) & cp2=$!
cr1=0; cr2=0
wait "$cp1" || cr1=$?
wait "$cp2" || cr2=$?
if [ "$cr1" -eq 0 ] || [ "$cr2" -eq 0 ]; then
  pass "S7p: concurrent same-backend publish -- at least one caller succeeded"
else
  fail "S7p: concurrent same-backend publish -- BOTH callers failed"
  cat "$S7P_LOG_A" "$S7P_LOG_B" >&2
fi
if grep -q "publish lock" "$S7P_LOG_A" "$S7P_LOG_B" 2>/dev/null || { [ "$cr1" -eq 0 ] && [ "$cr2" -eq 0 ]; }; then
  pass "S7p: concurrent same-backend publish -- lock serialised (one waited/failed cleanly, or both completed in turn)"
else
  fail "S7p: concurrent same-backend publish -- no evidence of serialisation (both ran with no lock contention logged)"
fi
rm -f "$S7P_LOG_A" "$S7P_LOG_B"
final_target="$(readlink dist/kiss/current || true)"
[ -n "$final_target" ] && [ -f "dist/kiss/$final_target/MANIFEST.txt" ] && [ -f "dist/kiss/$final_target/libaudio_common.a" ] && \
  pass "S7p: after concurrent publish, current points at a complete release dir" \
  || fail "S7p: after concurrent publish, current points at an incomplete/missing release dir"
manifest_ok=1
while read -r sha fname; do
  [ "$fname" = "MANIFEST.txt" ] && continue
  actual="$(file_sha "dist/kiss/current/$fname")"
  [ "$actual" = "$sha" ] || manifest_ok=0
done < <(grep -E '^[0-9a-f]{64}  ' "dist/kiss/current/MANIFEST.txt")
[ "$manifest_ok" -eq 1 ] && pass "S7p: final current release's MANIFEST is self-consistent" \
  || fail "S7p: final current release's MANIFEST does not match its own files"

echo "############################################################"
echo "# S8: CFG_SIG collision guard"
echo "############################################################"
cd "$AC_DIR"
kiss_objdir="$(make -s BACKEND=kiss print-obj-dir)"
manifest="$kiss_objdir/config.manifest"
backup="$(mktemp)"
cp "$manifest" "$backup"
echo "CORRUPTED PAYLOAD (test_build_isolation.sh S8)" > "$manifest"
if make BACKEND=kiss lib >/tmp/s8.log 2>&1; then
  fail "S8: corrupted config.manifest did NOT fail the next build (collision guard not firing)"
else
  grep -q "CFG_SIG collision" /tmp/s8.log && pass "S8: corrupted config.manifest correctly FAILS the next build with the collision message" \
    || fail "S8: build failed but NOT with the expected collision message"
fi
cp "$backup" "$manifest"
rm -f "$backup" /tmp/s8.log
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
  S9_LOG="$(mktemp)"
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
  rm -f "$S9_LOG"
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
S10_TMPDIR="$(mktemp -d)"
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

rm -rf "$S10_TMPDIR"

echo "############################################################"
echo "# S11: publish immutability / content-addressing (round-4 review P2-2)"
echo "############################################################"
cd "$AC_DIR"
# Clean slate: dist/ is never committed, and this scenario's assertions
# depend on knowing exactly which release ids exist.
rm -rf dist

make -s BACKEND=kiss publish >/dev/null
id1="$(readlink dist/kiss/current || true)"
[ -n "$id1" ] && [ -d "dist/kiss/$id1" ] && \
  pass "S11: first kiss publish -- current resolves to a real release dir ($id1)" \
  || fail "S11: first kiss publish -- current symlink broken or missing"

manifest1="dist/kiss/$id1/MANIFEST.txt"
if [ -f "$manifest1" ] && grep -q "^release_id=$id1$" "$manifest1" && grep -q "^git_dirty=" "$manifest1" \
   && grep -q "^ar=" "$manifest1" && grep -q "^ranlib=" "$manifest1" && grep -q "^link=" "$manifest1"; then
  pass "S11: MANIFEST.txt has release_id=/git_dirty=/ar=/ranlib=/link= (v3 fields)"
else
  fail "S11: MANIFEST.txt missing one of release_id=/git_dirty=/ar=/ranlib=/link= ($manifest1)"
fi

mtime_id1_before="$(mtime "dist/kiss/$id1")"

# Republishing IDENTICAL content must be a no-op on the release dir itself
# (verified byte-for-byte, never re-created) -- only `current` is repointed
# (to the SAME id here, since nothing changed).
sleep 1
S11_LOG="$(mktemp)"
make -s BACKEND=kiss publish >"$S11_LOG" 2>&1
grep -q "already published" "$S11_LOG" && pass "S11: identical-content republish reports 'already published'" \
  || fail "S11: identical-content republish did NOT report 'already published'"
rm -f "$S11_LOG"

id1_again="$(readlink dist/kiss/current || true)"
[ "$id1_again" = "$id1" ] && pass "S11: current still points at id1 after identical-content republish" \
  || fail "S11: current CHANGED after identical-content republish ($id1 -> $id1_again)"

mtime_id1_after="$(mtime "dist/kiss/$id1")"
[ "$mtime_id1_before" = "$mtime_id1_after" ] && pass "S11: release dir dist/kiss/$id1 mtime UNCHANGED by identical-content republish" \
  || fail "S11: release dir dist/kiss/$id1 mtime CHANGED by identical-content republish ($mtime_id1_before -> $mtime_id1_after)"

# Content change, SAME flags: publish must mint a NEW content-addressed id,
# and the OLD release dir must survive untouched alongside it (never deleted
# or overwritten).
# A REAL codegen change, not a comment: a comment-only edit produces a
# byte-identical object (and the archiver runs in deterministic mode, so the
# archive doesn't even pick up a new member timestamp) -- the content-
# addressed release id then CORRECTLY stays the same, which is exactly the
# immutability property, not a bug. To exercise "same flags, different
# content -> new release id" the artifact bytes must actually change.
echo "void s11_isolation_probe(void) {}" >> "$AC_DIR/src/hpf.c"
make -s BACKEND=kiss publish >/dev/null
id2="$(readlink dist/kiss/current || true)"

[ -n "$id2" ] && [ "$id2" != "$id1" ] && pass "S11: content change -- publish produced a NEW release id ($id1 -> $id2)" \
  || fail "S11: content change did NOT produce a new release id (id1=$id1 id2=$id2)"
[ -d "dist/kiss/$id1" ] && [ -d "dist/kiss/$id2" ] && \
  pass "S11: both the old (id1) and new (id2) release dirs coexist -- old release never deleted" \
  || fail "S11: old release dir dist/kiss/$id1 or new dist/kiss/$id2 missing after content change"

# ids are <cfg_sig>-<content12>; same flags => same cfg_sig prefix (the part
# before the LAST '-', since content12 itself is plain lowercase hex with no
# dashes).
cfgsig1="${id1%-*}"
cfgsig2="${id2%-*}"
[ -n "$cfgsig1" ] && [ "$cfgsig1" = "$cfgsig2" ] && \
  pass "S11: id1/id2 share the same cfg_sig prefix ($cfgsig1) -- flags unchanged, only content differed" \
  || fail "S11: id1/id2 cfg_sig prefixes DIFFER ($cfgsig1 vs $cfgsig2) though only source content changed"

# Restore: leave the tree clean for whatever else is using it, and dist/ is
# never committed either way.
git checkout -- "$AC_DIR/src/hpf.c"
make -s BACKEND=kiss lib >/dev/null
rm -rf dist

echo "############################################################"
echo "# S12: toolchain coherence guard (round-4 review P1-2)"
echo "############################################################"
cd "$AC_DIR"

# fake-cxx: reports a bogus -dumpmachine triple (so it can never match a
# real CC's), otherwise transparently forwards to the real c++ so a build
# using it as CXX still actually compiles/links.
S12_TMPDIR="$(mktemp -d)"
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

S12_LOG="$(mktemp)"
if make BACKEND=ne10 CXX="$SHIM" lib >"$S12_LOG" 2>&1; then
  fail "S12: make BACKEND=ne10 CXX=<mismatched shim> lib unexpectedly SUCCEEDED (toolchain guard did not fire)"
else
  grep -q "different targets" "$S12_LOG" && pass "S12: BACKEND=ne10 with a mismatched CXX -dumpmachine FAILS, mentioning 'different targets'" \
    || fail "S12: BACKEND=ne10 with a mismatched CXX FAILED but WITHOUT the expected 'different targets' message"
fi
rm -f "$S12_LOG"

S12_LOG2="$(mktemp)"
if make BACKEND=ne10 CXX="$SHIM" TOOLCHAIN_CHECK=0 lib >"$S12_LOG2" 2>&1; then
  pass "S12: TOOLCHAIN_CHECK=0 skips the guard -- BACKEND=ne10 build with the mismatched CXX shim SUCCEEDS (its own keyed dir)"
else
  fail "S12: TOOLCHAIN_CHECK=0 build with the mismatched CXX shim FAILED (guard should have been skipped)"
  cat "$S12_LOG2" >&2
fi
rm -f "$S12_LOG2"

S12_LOG3="$(mktemp)"
if make BACKEND=kiss CXX="$SHIM" lib >"$S12_LOG3" 2>&1; then
  pass "S12: BACKEND=kiss with the mismatched CXX shim SUCCEEDS (guard is ne10-only)"
else
  fail "S12: BACKEND=kiss with the mismatched CXX shim FAILED (guard should never run for kiss)"
  cat "$S12_LOG3" >&2
fi
rm -f "$S12_LOG3"

rm -rf "$S12_TMPDIR"

echo "############################################################"
echo "SUMMARY: $PASS_COUNT passed, $FAIL_COUNT failed"
echo "############################################################"
if [ "$FAIL_COUNT" -gt 0 ]; then
  echo "Failures:" >&2
  for f in "${FAILURES[@]}"; do echo "  - $f" >&2; done
  exit 1
fi
exit 0
