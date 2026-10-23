#!/usr/bin/env bash
# test_build_isolation.sh — build-isolation suite for common.
#
# Exercises the CFG_SIG-keyed obj/bin directory design in common's Makefile and the two-phase
# COMMON_LIB consumer resolution in AEC/c_impl against the failure modes it exists to prevent: one
# configuration's objects or archive delivered under another's key, stale archive members, publish
# tampering, and unsafe flag text reaching the shell.
#
# Two normal configs are exercised side by side wherever a scenario needs "config A" and "config B":
# SIMD=1 (the default) and SIMD=0. They differ in CFG_SIG and in compiled objects.
#
# Scenario index:
#   S1   A->B->A across SIMD=1 / SIMD=0, plus the -O0/-O3 delivered-object repro
#   S2   same-second SIMD=1 <-> SIMD=0 switching (scratch clone)
#   S3   parallel differing-config builds
#   S4   test_ne10_force_c isolation
#   S5   consumer correctness, drives AEC (scratch clones of common and AEC)
#   S7p  producer publish (content-addressed dist/ layout; throwaway DIST_ROOT)
#   S8   CFG_SIG collision guard; OBJ_ROOT/BIN_ROOT is build-content neutral
#   S9   command-line override rejection (CFLAGS, CXXFLAGS, LDFLAGS, FP_POLICY)
#   S10  archive freshness: a stale member does not survive a rebuild
#   S11  publish immutability and content addressing:
#          S11a in-tree republish is byte-verified and only grows ATTEST/
#          S11b a tampered MANIFEST is detected
#          S11c a content change in a sandbox copy yields a new release id
#   S12  toolchain coherence guard (CC/CXX target mismatch) and TOOLCHAIN_CHECK=0
#   S13  atomic `current` symlink swap hammer; --excl-install no-clobber mode
#   S14  lock-before-build: a concurrent publish loser builds nothing
#   S15  `make -n`/`-q`/`-t publish` write nothing (S15b: combined flags against a stale archive)
#   S16  ATTEST uniqueness under forced same-second collisions
#   S17  dirty/untracked checkout publish policy:
#          S17a clean default, b dirty refused, c dirty + hatch, d untracked, e untracked bytes
#          change the hash, f untracked symlink, g build-relevant untracked source, h unhashable
#          path, i encoding-collision pair hashes differently, j unreadable file, k mode change,
#          l identity-less checkout refused unconditionally
#   S18  interruption safety: EXIT/INT/TERM remove the whole scratch tree
#   S19  FFT_WRAPPER_ALIAS_CFLAGS is CFG_SIG-keyed
#   S20  FFT_WRAPPER_ALIAS_CFLAGS command-line override rejection
#   S21  the same under `make -e` (environment override)
#   S22  `make -e` override rejection for FP_POLICY and CPPFLAGS
#   S23  FP-policy conflict detection covers CXXFLAGS/CPPFLAGS (S23b: quoting and
#        response-file bypasses)
#   S24  character allow-list for compiler flags
#   S25  the same for link flags (EXTRA_LDFLAGS)
#   S26  any BACKEND other than ne10 is rejected
# S6 (Audio_ALG/pipelines consumer-resolution parity) belongs to that repo's own suite.
#
# Safety contract:
#   - One scratch root per run, SCRATCH_ROOT="$(mktemp -d)", removed by a single EXIT trap. TMPDIR is
#     exported inside it so child makes' mktemp workdirs land there too. Every scratch path is a fixed
#     path under "$SCRATCH_ROOT/<scenario>/" created inline with `mkdir -p`, never inside a `$(...)`
#     (cleanup state registered in a subshell is lost).
#   - A cross-suite mutex (SUITE_LOCK under the invoking TMPDIR, the same path the Audio_ALG/pipelines
#     suite uses) refuses to run two instances at once: both assert real-tree invariants a concurrent
#     run would trample. A lock whose holder pid is dead is reclaimed once.
#   - The real obj/ and bin/ only ever see the two normal configs S1 builds. Every probe or tamper
#     config builds under a scratch OBJ_ROOT/BIN_ROOT. A snapshot of the real obj/+bin/ taken after S1
#     is re-verified byte for byte at the end.
#   - The real dist/ is never touched: every publish passes a scratch DIST_ROOT, and a sentinel digest
#     of dist/ is re-checked at the end.
#   - No git-tracked file in $AC_DIR or the AEC repo is touched, not even its mtime; mtime bumps happen
#     on scratch clones. tree_state_hash() (status --porcelain plus diff --binary HEAD) of both repos
#     must be equal at the end.
#   - No sleep. Forcing a source strictly newer than an artifact uses `touch -r <artifact> -A 01
#     <source>`, and mtime() reads fractional seconds.
#   - Interruption safe: cleanup() is the only EXIT-trap resident and INT/TERM exit through it with
#     130/143 (S18 tests this).
#
# Design rules (keep when editing this script):
#   - No `make clean` inside any scenario: distinct configs must coexist without a clean.
#   - Every path is resolved with `make -s ... print-bin-dir` / `print-obj-dir` / `print-lib-path`,
#     with the exact flag set (including OBJ_ROOT/BIN_ROOT) under test, never hand-reconstructed.
#   - "Was this rebuilt?" is an mtime comparison, never a sha comparison (recompiling identical inputs
#     gives identical bytes). "Is this the same delivered artifact as its own keyed object?" is a sha
#     comparison, via member_sha()/file_sha().
#
# Usage: ./scripts/test_build_isolation.sh   (from common/ or anywhere; paths are resolved from this
# script's own location). The AEC repo must be a sibling of audiolib and a git checkout.
#
# ISOL_INTERRUPT_PROBE (internal, S18 only): when set, this script runs as a child re-invocation of
# itself in "interruption probe" mode instead of running the suite; see the probe block below.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
AC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# AUDIOLIB_REPO_DIR is the git toplevel (audiolib/); AC_DIR (audiolib/common) is only a subdirectory, so
# adopt_worktree_clone() must be given AUDIOLIB_REPO_DIR, never AC_DIR (see its doc comment for the
# two-step pattern). AEC is a sibling of audiolib.
AUDIOLIB_REPO_DIR="$(cd "$AC_DIR/.." && pwd)"
AEC_REPO_DIR="$(cd "$AC_DIR/../../AEC" && pwd)"
AEC_DIR="$(cd "$AEC_REPO_DIR/c_impl" && pwd)"

# --- single scratch root, single trap ------------------
SCRATCH_ROOT="$(mktemp -d)"
# SUITE_LOCK_HELD must exist (as "0") before anything can exit through cleanup(), including the
# ISOL_INTERRUPT_PROBE branch below, which never reaches the mutex block that sets it to "1": `set -u`
# would fault on cleanup()'s reference otherwise.
SUITE_LOCK_HELD=0
SUITE_LOCK=""
cleanup() {
  rc=$?
  trap - EXIT INT TERM
  [ "$SUITE_LOCK_HELD" -eq 1 ] && rm -rf -- "$SUITE_LOCK" 2>/dev/null
  rm -rf -- "$SCRATCH_ROOT"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# --- S18 interruption probe: check this early, before any suite logic and before the mktemp-shim
# setup, so probe mode's `set +e` takes effect before any further command runs (a signal can arrive
# during the shim's own file writes). With ISOL_INTERRUPT_PROBE set this is a child re-invocation of
# this script, spawned by the S18 scenario below. It signals its SCRATCH_ROOT back to the parent over
# a fifo (the write doubles as the readiness signal), then either exits at once (mode "exit") or blocks
# on a second fifo nobody writes, so a parent-delivered INT/TERM is what interrupts it, as in a real
# mid-scenario interruption. Its own `mktemp -d` inherited the parent's shim via PATH, which lands this
# SCRATCH_ROOT in the parent's probe TMPDIR.
if [ -n "${ISOL_INTERRUPT_PROBE:-}" ]; then
  # `set +e` for the rest of probe mode: with errexit active, a SIGTERM/SIGINT arriving while this
  # process is blocked in a syscall (opening the fifo, or the `read` below) can surface as that
  # command's non-zero return, and errexit then runs the EXIT trap with a stale $? before bash services
  # the pending INT/TERM trap, giving the wrong exit code (flaky under load on bash 3.2). Without
  # errexit the INT/TERM traps installed above decide the exit code unconditionally.
  set +e
  mkdir -p "$SCRATCH_ROOT/probe"
  : > "$SCRATCH_ROOT/probe/canary_a"
  : > "$SCRATCH_ROOT/probe/canary_b"
  printf '%s\n' "$SCRATCH_ROOT" > "$ISOL_PROBE_FIFO"
  if [ "$ISOL_INTERRUPT_PROBE" = "exit" ]; then
    exit 0
  fi
  # Open the hold-fifo O_RDWR (never blocks on open(), since we hold both ends). Block via `wait` on a
  # background reader of that fd, not a foreground `read <&9`: a signal arriving during a foreground
  # `read` can leave the pending INT/TERM trap action un-run, and a second deferred delivery then fires
  # after cleanup() has reset the trap, killing the process mid-`rm -rf`. `wait` is bash's documented
  # interruptible blocking primitive (bash(1), SIGNALS) and does not show this.
  exec 9<>"$ISOL_PROBE_FIFO.hold"
  cat <&9 >/dev/null &
  catpid=$!
  wait "$catpid"
  exit 0
fi

# --- cross-suite mutex -----------------------------------
# Placed after the ISOL_INTERRUPT_PROBE early return (a probe child never contends for the lock its
# parent holds) and before `export TMPDIR=...` below, so `${TMPDIR:-/tmp}` still resolves to the
# invoking environment's TMPDIR. Two instances of this suite, or of the Audio_ALG/pipelines
# counterpart (same lock path), must never run concurrently: both assert real-tree invariants (S1's
# real archives, the final-guard snapshot, the dist/ sentinel, tree_state_hash()) that a concurrent
# run would trample.
SUITE_LOCK="${TMPDIR:-/tmp}/audio_dsp_isolation_suite.lock"
if mkdir "$SUITE_LOCK" 2>/dev/null; then
  printf '%s\n' "$$" > "$SUITE_LOCK/pid"
  SUITE_LOCK_HELD=1
else
  holder_pid="$(cat "$SUITE_LOCK/pid" 2>/dev/null || true)"
  reclaimed=0
  if [ -n "$holder_pid" ] && ! kill -0 "$holder_pid" 2>/dev/null; then
    # Stale lock: the recorded holder pid is confirmed dead. Reclaim once. Two reclaimers can race this
    # rm -rf/mkdir pair and only one mkdir wins; the loser must not steal the winner's fresh lock, so
    # this is a single retry (not a loop) and the loser falls through to the FATAL below.
    rm -rf -- "$SUITE_LOCK" 2>/dev/null || true
    if mkdir "$SUITE_LOCK" 2>/dev/null; then
      printf '%s\n' "$$" > "$SUITE_LOCK/pid"
      SUITE_LOCK_HELD=1
      reclaimed=1
    fi
  fi
  if [ "$reclaimed" -ne 1 ] && [ "$SUITE_LOCK_HELD" -ne 1 ]; then
    holder_pid="$(cat "$SUITE_LOCK/pid" 2>/dev/null || echo unknown)"
    echo "FATAL: another isolation suite instance is running (pid $holder_pid) -- the suites assert real-tree invariants and must not run concurrently" >&2
    exit 1
  fi
fi

mkdir "$SCRATCH_ROOT/tmp"
# Child `make`s' own mktemp workdirs (e.g. the publish recipe's `work="$(mktemp
# -d)"`) land under scratch too via this, so the single `rm -rf` above collects
# EVERYTHING this run creates, including after an interruption -- see S18.
export TMPDIR="$SCRATCH_ROOT/tmp"

# macOS: BSD `mktemp -d` with no template resolves via _CS_DARWIN_USER_TEMP_DIR before $TMPDIR
# (mktemp(1)), so exporting TMPDIR alone does not redirect a child's bare `mktemp -d` (the Makefile's
# publish recipe `work="$(mktemp -d)"`, S18's child `SCRATCH_ROOT="$(mktemp -d)"`) into scratch. A tiny
# shim ahead of the real mktemp on PATH forwards to /usr/bin/mktemp, injecting `-p "$TMPDIR"` when the
# caller gave no template or -p.
mkdir "$SCRATCH_ROOT/shimbin"
cat > "$SCRATCH_ROOT/shimbin/mktemp" <<'EOF'
#!/bin/sh
has_template=0
has_p=0
for a in "$@"; do
  case "$a" in
    -p) has_p=1 ;;
    -*) ;;
    *) has_template=1 ;;
  esac
done
if [ "$has_template" -eq 0 ] && [ "$has_p" -eq 0 ] && [ -n "${TMPDIR:-}" ]; then
  exec /usr/bin/mktemp -p "$TMPDIR" "$@"
fi
exec /usr/bin/mktemp "$@"
EOF
chmod +x "$SCRATCH_ROOT/shimbin/mktemp"
export PATH="$SCRATCH_ROOT/shimbin:$PATH"

# --- helpers -----------------------------------------------------------------
mkscratch() { mkdir -p "$SCRATCH_ROOT/$1"; }

# ar -p <archive> <member> | sha256  -- the sha of one member's CONTENT as it
# sits inside the delivered archive (not the archive's own sha, which would
# also fold in ar's per-member timestamp/mode header fields).
member_sha() { ar -p "$1" "$2" | shasum -a 256 | awk '{print $1}'; }
file_sha()   { shasum -a 256 "$1" | awk '{print $1}'; }
# members_not_matching <archive> <objdir>: prints the archive members whose content differs
# from (or is missing in) <objdir>; prints nothing when EVERY member equals its own keyed
# object. A missing archive or object is reported, never fatal under set -e.
members_not_matching() {
  local lib="$1" od="$2" m
  [ -f "$lib" ] || { printf '<no archive %s>' "$lib"; return 0; }
  while IFS= read -r m; do
    case "$m" in __.SYMDEF*) continue ;; esac
    if [ ! -f "$od/$m" ] || ! cmp -s <(ar -p "$lib" "$m") "$od/$m"; then printf '%s ' "$m"; fi
  done < <(ar -t "$lib")
}
# Fractional seconds, so two separate writes are distinguishable without a delay. The GNU fallback
# loses the fractional part.
mtime()      { stat -f '%Fm' "$1" 2>/dev/null || stat -c %Y "$1"; }

# Tree-state hash: status --porcelain ALONE misses a content
# edit to an already-dirty file, so this also folds in `diff --binary HEAD`.
tree_state_hash() {
  { git -C "$1" status --porcelain; git -C "$1" diff --binary HEAD; } 2>/dev/null | shasum -a 256 | awk '{print $1}'
}

# adopt_worktree_clone <src-repo-dir> <clone-dir>
# Clones <src-repo-dir> (read-only on the source), overlays every currently modified tracked file and
# every untracked non-ignored file, and adopts them as ONE throwaway commit inside the scratch clone
# only, so the clone has a clean git identity that still reflects today's (possibly uncommitted)
# worktree. A bare `git clone` reproduces only the last commit and would carry the old Makefile instead
# of the one under test. The clone is disposable under $SCRATCH_ROOT; its history never touches the
# source repo's .git.
#
# CONTRACT: <src-repo-dir> must be the git repository's own toplevel (enforced below):
# `git diff --name-only HEAD` emits repo-root-relative paths while `git ls-files --others` emits
# cwd-relative ones, and the overlay loop assumes both are relative to $src. Common lives in
# audiolib/common, so every call site that wants "a clone of common" clones $AUDIOLIB_REPO_DIR and
# addresses the clone's common/ subdirectory (`make -C` and `git -C` handle a non-toplevel directory):
#   FOO_ROOT="$SCRATCH_ROOT/<scenario>/audiolib_clone..."
#   adopt_worktree_clone "$AUDIOLIB_REPO_DIR" "$FOO_ROOT"
#   FOO_CLONE="$FOO_ROOT/common"
#
# Cloning the whole repo also keeps the dirty/untracked scenarios (S17 and friends) valid. The
# Makefile's tracked-dirty gate (`git status --porcelain -uno` / `git diff --binary HEAD`) is
# cwd-independent and covers the entire repo, so this script's `git -C "$dst"` assertions match it with
# no pathspec (adding `-- .` would make them disagree with the Makefile). The untracked gate
# (`git ls-files --others --exclude-standard`, also hash_untracked's input) is cwd-scoped to common/.
# The adoption commit absorbs all of today's state, so the clone is clean at HEAD (asserted below) and
# dirt a scenario injects under common/ is the only difference.
#
# The commit is conditional (`git add -A`, then commit only if `git diff --cached --quiet --exit-code`
# says something is staged): an unconditional commit exits 1 on a clean tip and `set -e` would kill
# the suite.
adopt_worktree_clone() {
  local src="$1" dst="$2" relf top
  top="$(git -C "$src" rev-parse --show-toplevel 2>/dev/null)" || {
    echo "FATAL: adopt_worktree_clone: \$src ($src) is not inside a git repository" >&2
    exit 1
  }
  [ "$top" = "$(cd "$src" && pwd)" ] || {
    echo "FATAL: adopt_worktree_clone: \$src ($src) is not its own git repo TOPLEVEL (toplevel is $top)." >&2
    echo "  Pass the repo's actual root and address a subdirectory of the CLONE afterward instead" >&2
    echo "  (see this function's doc comment for the two-step pattern)." >&2
    exit 1
  }
  git clone --no-hardlinks --quiet "$src" "$dst"
  while IFS= read -r relf; do
    [ -n "$relf" ] || continue
    if [ -e "$src/$relf" ]; then
      mkdir -p "$(dirname "$dst/$relf")"
      cp "$src/$relf" "$dst/$relf"
    else
      rm -f -- "$dst/$relf"   # tracked file deleted in the worktree
    fi
  # --no-renames: with git's default rename detection a staged rename lists only the NEW
  # path, so the old file would survive in the clone next to the new one. Untracked (not
  # ignored) files are adopted too: an in-progress rename leaves the new sources untracked.
  done < <(git -C "$src" diff --name-only --no-renames HEAD; git -C "$src" ls-files --others --exclude-standard)
  git -C "$dst" add -A
  if ! git -C "$dst" diff --cached --quiet --exit-code; then
    git -C "$dst" -c user.email=scratch@example.invalid -c user.name="scratch clone" \
      commit -q -m "scratch: adopt current worktree (disposable clone only, never touches the real repo)"
  fi
  # Loud "can this actually fail" guard, not a hopeful assumption: the clone
  # must be fully clean at this HEAD. If it is not, every downstream
  # dirty/untracked scenario (S17 and friends) would be silently polluted by
  # leftover adoption state instead of seeing only what the scenario itself
  # injects afterward.
  if [ -n "$(git -C "$dst" status --porcelain)" ]; then
    echo "FATAL: adopt_worktree_clone: $dst is NOT clean immediately after adoption (bug in the overlay/commit step above)" >&2
    git -C "$dst" status --porcelain >&2
    exit 1
  fi
}

# Real-tree final guard: a per-file "path sha256 mtime" snapshot of everything under the real obj/ and
# bin/, plus a top-level keyed-directory listing, captured in S1 once this suite's intentional
# real-tree normal-config builds (SIMD=1 + SIMD=0) are done and re-verified at the very end: every
# snapshotted file must be sha+mtime identical; a new directory is allowed (INFO/WARN), a new file
# inside an already-snapshotted directory is a FAIL.
real_tree_snapshot() {
  ( cd "$AC_DIR" && for root in obj bin; do
      [ -d "$root" ] || continue
      find "$root" -type f | LC_ALL=C sort | while IFS= read -r f; do
        printf '%s %s %s\n' "$f" "$(file_sha "$f")" "$(mtime "$f")"
      done
    done )
}
real_tree_dirs() {
  ( cd "$AC_DIR" && for root in obj bin; do
      [ -d "$root" ] || continue
      find "$root" -mindepth 1 -maxdepth 1 -type d | LC_ALL=C sort
    done )
}

# Real dist/ sentinel: absent stays absent;
# otherwise a full path list + per-file sha256 + per-file (name, mtime).
real_dist_sentinel() {
  if [ ! -e "$AC_DIR/dist" ]; then
    echo absent
    return
  fi
  ( cd "$AC_DIR" && {
      find dist -print | sort
      find dist -type f -print | sort | while read -r f; do shasum -a 256 "$f"; done
      find dist -print | sort | while read -r f; do stat -f '%N %m' "$f"; done
    } ) | shasum -a 256 | awk '{print $1}'
}

AC_STATE_BEFORE="$(tree_state_hash "$AC_DIR")"
AEC_STATE_BEFORE="$(tree_state_hash "$AEC_REPO_DIR")"
REAL_DIST_BEFORE="$(real_dist_sentinel)"

PASS_COUNT=0
FAIL_COUNT=0
FAILURES=()

pass() { PASS_COUNT=$((PASS_COUNT + 1)); echo "  PASS: $*"; }
fail() { FAIL_COUNT=$((FAIL_COUNT + 1)); FAILURES+=("$*"); echo "  FAIL: $*" >&2; }

echo "############################################################"
echo "# S1: A->B->A across SIMD=1 / SIMD=0 + -O0/-O3 delivered-object repro"
echo "############################################################"
cd "$AC_DIR"

# Config A is SIMD=1 (the default), config B is SIMD=0. These are the two normal configs the
# real obj/ and bin/ hold.
lib_a="$(make -s SIMD=1 print-lib-path)"
make -s SIMD=1 lib >/dev/null
sha_a_1="$(file_sha "$lib_a")"
mtime_a_1="$(mtime "$lib_a")"

lib_b="$(make -s SIMD=0 print-lib-path)"
make -s SIMD=0 lib >/dev/null

# Final-guard snapshot, taken now that both real-tree normal configs are built. Every later
# real-tree `make ... lib` re-targets one of them and must be a pure no-op. Re-verified at
# the very end (see "Final integrity guards").
FINAL_GUARD_DIRS_BEFORE="$(real_tree_dirs)"
FINAL_GUARD_SNAPSHOT_BEFORE="$(real_tree_snapshot)"

# back to A: the archive must be untouched (same path, same sha, same mtime)
lib_a_again="$(make -s SIMD=1 print-lib-path)"
make -s SIMD=1 lib >/dev/null
sha_a_2="$(file_sha "$lib_a_again")"
mtime_a_2="$(mtime "$lib_a_again")"

[ "$lib_a" = "$lib_a_again" ] && pass "S1: SIMD=1 path stable across A->B->A" \
  || fail "S1: SIMD=1 path CHANGED across A->B->A ($lib_a vs $lib_a_again)"
[ "$sha_a_1" = "$sha_a_2" ] && [ "$mtime_a_1" = "$mtime_a_2" ] && \
  pass "S1: SIMD=1 archive sha+mtime unchanged after A->B->A (no cross-config stomp)" \
  || fail "S1: SIMD=1 archive CHANGED after A->B->A (sha $sha_a_1 -> $sha_a_2, mtime $mtime_a_1 -> $mtime_a_2)"

[ "$lib_a" != "$lib_b" ] && pass "S1: SIMD=1/SIMD=0 archive paths differ" \
  || fail "S1: SIMD=1/SIMD=0 archive paths COLLIDE ($lib_a)"

for pair in "SIMD=1:$lib_a" "SIMD=0:$lib_b"; do
  cfg="${pair%%:*}"; lib="${pair#*:}"
  ar -t "$lib" | grep -q '^NE10_fft\.o$' && ar -t "$lib" | grep -q '^fft_wrapper_ne10\.o$' \
    && pass "S1: $cfg archive contains NE10_fft.o and fft_wrapper_ne10.o" \
    || fail "S1: $cfg archive missing NE10_fft.o or fft_wrapper_ne10.o"
done

# EXTRA_CFLAGS=-O0 vs -O3 must never let one config's compiled object be delivered under the other's
# archive member (mtime staleness across differing flag sets). These probe configs are not the normal
# configs the real obj/bin hold, so they build under a scratch OBJ_ROOT/BIN_ROOT, and every print-*
# query below uses the same scratch roots.
mkscratch s1
S1_OBJ_ROOT="$SCRATCH_ROOT/s1/obj"
S1_BIN_ROOT="$SCRATCH_ROOT/s1/bin"
make -s EXTRA_CFLAGS=-O0 OBJ_ROOT="$S1_OBJ_ROOT" BIN_ROOT="$S1_BIN_ROOT" lib >/dev/null
lib_o0="$(make -s EXTRA_CFLAGS=-O0 OBJ_ROOT="$S1_OBJ_ROOT" BIN_ROOT="$S1_BIN_ROOT" print-lib-path)"
objdir_o0="$(make -s EXTRA_CFLAGS=-O0 OBJ_ROOT="$S1_OBJ_ROOT" BIN_ROOT="$S1_BIN_ROOT" print-obj-dir)"
make -s EXTRA_CFLAGS=-O3 OBJ_ROOT="$S1_OBJ_ROOT" BIN_ROOT="$S1_BIN_ROOT" lib >/dev/null
lib_o3="$(make -s EXTRA_CFLAGS=-O3 OBJ_ROOT="$S1_OBJ_ROOT" BIN_ROOT="$S1_BIN_ROOT" print-lib-path)"
objdir_o3="$(make -s EXTRA_CFLAGS=-O3 OBJ_ROOT="$S1_OBJ_ROOT" BIN_ROOT="$S1_BIN_ROOT" print-obj-dir)"

[ "$lib_o0" != "$lib_o3" ] && pass "S1: -O0/-O3 land in different bin dirs" \
  || fail "S1: -O0/-O3 COLLIDE in the same bin dir ($lib_o0)"

sha_member_o0="$(member_sha "$lib_o0" fft_wrapper_ne10.o)"
sha_obj_o0="$(file_sha "$objdir_o0/fft_wrapper_ne10.o")"
sha_member_o3="$(member_sha "$lib_o3" fft_wrapper_ne10.o)"
sha_obj_o3="$(file_sha "$objdir_o3/fft_wrapper_ne10.o")"

if [ "$sha_member_o0" = "$sha_obj_o0" ] && [ "$sha_member_o3" = "$sha_obj_o3" ] && [ "$sha_member_o0" != "$sha_member_o3" ]; then
  pass "S1: -O0/-O3 delivered fft_wrapper_ne10.o member sha == matching keyed object sha (THE B01 assertion)"
else
  fail "S1: -O0/-O3 member/object sha mismatch (o0 member=$sha_member_o0 obj=$sha_obj_o0; o3 member=$sha_member_o3 obj=$sha_obj_o3)"
fi

echo "############################################################"
echo "# S2: same-second SIMD=1 <-> SIMD=0 switching (scratch clone)"
echo "############################################################"
# The scenario runs inside a disposable clone of $AC_DIR under scratch; the clone's own
# default-rooted obj/ and bin/ are the scratch space, so no OBJ_ROOT/BIN_ROOT override is
# needed and no real tracked file is touched.
mkscratch s2
# Two-step clone (see adopt_worktree_clone): audiolib is cloned whole and S2_CLONE points at the
# clone's common/ subdir.
S2_ROOT="$SCRATCH_ROOT/s2/audiolib_clone"
adopt_worktree_clone "$AUDIOLIB_REPO_DIR" "$S2_ROOT"
S2_CLONE="$S2_ROOT/common"
# mtime-only touch (content-neutral) of the CLONE's probed sources forces a recompile probe. The
# probed objects must have bytes that differ between SIMD=1 and SIMD=0: hpf.o compiles identically
# under both, so "member sha == own keyed object" would hold for it whatever the build did.
# audio_stft.o and fft_wrapper_ne10.o do differ. The check is two-sided: each archive's member equals
# its own keyed object AND the two configs' members differ from each other. The second half catches a
# shared object directory, where make finds the SIMD=1 object up to date and never recompiles it for
# SIMD=0.
touch "$S2_CLONE/src/audio_stft.c" "$S2_CLONE/src/fft_wrapper_ne10.c"
make -C "$S2_CLONE" -s SIMD=1 lib >/dev/null
make -C "$S2_CLONE" -s SIMD=0 lib >/dev/null
s2_lib_a="$(make -C "$S2_CLONE" -s SIMD=1 print-lib-path)";  s2_objdir_a="$(make -C "$S2_CLONE" -s SIMD=1 print-obj-dir)"
s2_lib_b="$(make -C "$S2_CLONE" -s SIMD=0 print-lib-path)";  s2_objdir_b="$(make -C "$S2_CLONE" -s SIMD=0 print-obj-dir)"
for s2_obj in audio_stft.o fft_wrapper_ne10.o; do
  m1="$(member_sha "$s2_lib_a" "$s2_obj" 2>/dev/null || echo missing)"; o1="$(file_sha "$s2_objdir_a/$s2_obj" 2>/dev/null || echo missing)"
  m2="$(member_sha "$s2_lib_b" "$s2_obj" 2>/dev/null || echo missing)"; o2="$(file_sha "$s2_objdir_b/$s2_obj" 2>/dev/null || echo missing)"
  [ "$m1" = "$o1" ] && [ "$m2" = "$o2" ] && [ "$m1" != "$m2" ] && \
    pass "S2: per-archive $s2_obj member sha == own keyed object, and SIMD=1 != SIMD=0 (same-second switch safe)" \
    || fail "S2: $s2_obj member/object mismatch or identical across configs after same-second switch (SIMD=1: $m1 vs $o1; SIMD=0: $m2 vs $o2)"
done
# The two probes above are the members known to differ between SIMD=1 and SIMD=0; a stale
# delivery of any OTHER config-sensitive member (audio_pre_gain.o, audio_resampler.o, ...)
# is only seen by checking every member against its own keyed object.
s2_bad_a="$(members_not_matching "$s2_lib_a" "$s2_objdir_a")"
s2_bad_b="$(members_not_matching "$s2_lib_b" "$s2_objdir_b")"
[ -z "$s2_bad_a" ] && [ -z "$s2_bad_b" ] && \
  pass "S2: every member of both archives equals its own keyed object" \
  || fail "S2: archive member(s) differ from their own keyed object (SIMD=1: ${s2_bad_a:-none}; SIMD=0: ${s2_bad_b:-none})"

echo "############################################################"
echo "# S3: parallel differing-config builds"
echo "############################################################"
mkscratch s3
# Both configs build concurrently under ONE scratch OBJ_ROOT/BIN_ROOT pair: differing configs sharing a
# tree must run side by side, each in its own keyed directory (separate roots could never contend, and
# a shared object directory would go unseen). The configs must also compile to different bytes --
# SIMD=0 vs SIMD=1 -- for the member-vs-own-object comparison to see a cross-config delivery:
# -DPAR_PROBE alone changes no object, and hpf.o is identical under both.
S3_OBJ_ROOT="$SCRATCH_ROOT/s3/obj"
S3_BIN_ROOT="$SCRATCH_ROOT/s3/bin"
S3_LOG_A="$SCRATCH_ROOT/s3/log_a"
S3_LOG_B="$SCRATCH_ROOT/s3/log_b"
( make -j4 -s SIMD=1 OBJ_ROOT="$S3_OBJ_ROOT" BIN_ROOT="$S3_BIN_ROOT" lib >"$S3_LOG_A" 2>&1 ) & p1=$!
( make -j4 -s SIMD=0 EXTRA_CFLAGS=-DPAR_PROBE OBJ_ROOT="$S3_OBJ_ROOT" BIN_ROOT="$S3_BIN_ROOT" lib >"$S3_LOG_B" 2>&1 ) & p2=$!
r1=0; r2=0
wait "$p1" || r1=$?
wait "$p2" || r2=$?
if [ "$r1" -eq 0 ] && [ "$r2" -eq 0 ]; then
  pass "S3: both parallel differing-config builds exit green"
else
  fail "S3: parallel build failed (SIMD=1 exit=$r1, SIMD=0 PAR_PROBE exit=$r2)"
  echo "--- SIMD=1 log ---" >&2; cat "$S3_LOG_A" >&2
  echo "--- SIMD=0 PAR_PROBE log ---" >&2; cat "$S3_LOG_B" >&2
fi

s3_lib_a="$(make -s SIMD=1 OBJ_ROOT="$S3_OBJ_ROOT" BIN_ROOT="$S3_BIN_ROOT" print-lib-path)"
s3_objdir_a="$(make -s SIMD=1 OBJ_ROOT="$S3_OBJ_ROOT" BIN_ROOT="$S3_BIN_ROOT" print-obj-dir)"
s3_lib_p="$(make -s SIMD=0 EXTRA_CFLAGS=-DPAR_PROBE OBJ_ROOT="$S3_OBJ_ROOT" BIN_ROOT="$S3_BIN_ROOT" print-lib-path)"
s3_objdir_p="$(make -s SIMD=0 EXTRA_CFLAGS=-DPAR_PROBE OBJ_ROOT="$S3_OBJ_ROOT" BIN_ROOT="$S3_BIN_ROOT" print-obj-dir)"
for s3_obj in audio_stft.o fft_wrapper_ne10.o; do
  m1="$(member_sha "$s3_lib_a" "$s3_obj" 2>/dev/null || echo missing)"; o1="$(file_sha "$s3_objdir_a/$s3_obj" 2>/dev/null || echo missing)"
  m2="$(member_sha "$s3_lib_p" "$s3_obj" 2>/dev/null || echo missing)"; o2="$(file_sha "$s3_objdir_p/$s3_obj" 2>/dev/null || echo missing)"
  [ "$m1" = "$o1" ] && [ "$m2" = "$o2" ] && [ "$m1" != "$m2" ] && \
    pass "S3: $s3_obj member==object shas hold after -j4 parallel differing-config builds, and the two configs' objects differ" \
    || fail "S3: parallel build cross-contaminated $s3_obj, or the two configs' objects are identical (SIMD=1: $m1 vs $o1; SIMD=0 PAR_PROBE: $m2 vs $o2)"
done
s3_bad_a="$(members_not_matching "$s3_lib_a" "$s3_objdir_a")"
s3_bad_p="$(members_not_matching "$s3_lib_p" "$s3_objdir_p")"
[ -z "$s3_bad_a" ] && [ -z "$s3_bad_p" ] && \
  pass "S3: every member of both parallel-built archives equals its own keyed object" \
  || fail "S3: archive member(s) differ from their own keyed object after the parallel builds (SIMD=1: ${s3_bad_a:-none}; SIMD=0 PAR_PROBE: ${s3_bad_p:-none})"

echo "############################################################"
echo "# S4: test_ne10_force_c isolation (the flat-bin/ne10/ overwrite bug)"
echo "############################################################"
ne10_lib="$(make -s BACKEND=ne10 print-lib-path)"
make -s BACKEND=ne10 lib >/dev/null
sha_before="$(file_sha "$ne10_lib")"
mtime_before="$(mtime "$ne10_lib")"

# test_ne10_force_c builds its NEON and forced-C variants with internal sub-makes, so the whole
# invocation runs under a scratch OBJ_ROOT/BIN_ROOT (forwarded to the sub-makes as ordinary
# command-line variables) and never touches the real ne10-<sig>/ directory.
mkscratch s4
S4_OBJ_ROOT="$SCRATCH_ROOT/s4/obj"
S4_BIN_ROOT="$SCRATCH_ROOT/s4/bin"
S4_LOG="$SCRATCH_ROOT/s4/log"
# env -u SIMD: the target builds the NEON and the SIMD=0 variant itself, so an inherited SIMD
# would make both variants identical.
if env -u SIMD make test_ne10_force_c OBJ_ROOT="$S4_OBJ_ROOT" BIN_ROOT="$S4_BIN_ROOT" >"$S4_LOG" 2>&1; then
  pass "S4: test_ne10_force_c exits green"
else
  fail "S4: test_ne10_force_c FAILED"
  cat "$S4_LOG" >&2
fi

sha_after="$(file_sha "$ne10_lib")"
mtime_after="$(mtime "$ne10_lib")"
[ "$sha_before" = "$sha_after" ] && [ "$mtime_before" = "$mtime_after" ] && \
  pass "S4: normal ne10 archive byte-identical + mtime-untouched after forced-C run (scratch-redirected)" \
  || fail "S4: normal ne10 archive CHANGED by the forced-C sub-make (sha $sha_before -> $sha_after, mtime $mtime_before -> $mtime_after)"

normal_lib_scratch="$(make -s BACKEND=ne10 SIMD=1 OBJ_ROOT="$S4_OBJ_ROOT" BIN_ROOT="$S4_BIN_ROOT" print-lib-path)"
forcedc_lib="$(make -s BACKEND=ne10 SIMD=0 OBJ_ROOT="$S4_OBJ_ROOT" BIN_ROOT="$S4_BIN_ROOT" print-lib-path)"
[ "$forcedc_lib" != "$normal_lib_scratch" ] && pass "S4: forced-C archive lives at a different keyed path than the normal ne10 build (both under the SAME scratch roots)" \
  || fail "S4: forced-C archive path COLLIDES with the normal ne10 path ($forcedc_lib)"
[ -f "$forcedc_lib" ] && pass "S4: forced-C archive exists on disk" || fail "S4: forced-C archive missing"

forcedc_objdir="$(make -s BACKEND=ne10 SIMD=0 OBJ_ROOT="$S4_OBJ_ROOT" BIN_ROOT="$S4_BIN_ROOT" print-obj-dir)"
normal_objdir="$(make -s BACKEND=ne10 SIMD=1 OBJ_ROOT="$S4_OBJ_ROOT" BIN_ROOT="$S4_BIN_ROOT" print-obj-dir)"
if [ -f "$forcedc_objdir/fft_wrapper_ne10.o" ] && [ -f "$normal_objdir/fft_wrapper_ne10.o" ]; then
  s1="$(file_sha "$forcedc_objdir/fft_wrapper_ne10.o")"
  s2="$(file_sha "$normal_objdir/fft_wrapper_ne10.o")"
  [ "$s1" != "$s2" ] && pass "S4: SIMD=0 fft_wrapper_ne10.o differs from the SIMD=1 build's" \
    || fail "S4: SIMD=0 object is BYTE-IDENTICAL to the SIMD=1 build's (switch did not take effect?)"
else
  fail "S4: missing fft_wrapper_ne10.o in one of the two obj dirs ($forcedc_objdir or $normal_objdir)"
fi

# The parity binaries the target actually ran must be two different programs: the NEON one
# links the NEON R2C kernel, the forced-C one does not. Archives being keyed is not enough --
# if both variants linked one unkeyed test binary, the SIMD=0 link would overwrite the NEON
# one and the parity run would compare C against C (and pass).
s4_bin_neon="$(dirname "$normal_lib_scratch")/test_ne10_c_parity"
s4_bin_c="$(dirname "$forcedc_lib")/test_ne10_c_parity"
s4_neon_n="$(nm "$s4_bin_neon" 2>/dev/null | grep -c 'ne10_fft_r2c_1d_float32_neon' || true)"
s4_c_n="$( { nm "$s4_bin_c" 2>/dev/null || echo NM_FAILED; } | grep -c 'ne10_fft_r2c_1d_float32_neon\|NM_FAILED' || true)"
[ "${s4_neon_n:-0}" -gt 0 ] && [ "${s4_c_n:-1}" -eq 0 ] && \
  pass "S4: the NEON parity binary links the NEON R2C kernel and the forced-C one does not" \
  || fail "S4: parity binaries are not the two expected variants (NEON refs in NEON binary=$s4_neon_n, in forced-C binary=$s4_c_n)"

echo "############################################################"
echo "# S5: consumer correctness (drives AEC; regression: scratch clones)"
echo "############################################################"
# S5 runs against disposable clones of both repos: a clone of audiolib (the producer, whose sources get
# mtime bumps) and a clone of the whole AEC repo (builds run in <clone>/c_impl). The clones'
# default-rooted obj/ and bin/ are the scratch space.
#
# Every make call into the AEC clone passes COMMON_DIR="$S5_AC_CLONE" explicitly alongside COMMON_LIB=:
# AEC's `$(COMMON_LIB): FORCE` rule re-invokes `$(MAKE) -C $(COMMON_DIR) ... lib`, and left to its
# wildcard default that would fall back to the real common. A command-line COMMON_DIR= does propagate
# to that sub-make, but an invocation this correctness-critical should not lean on propagation.
# COMMON_LIB= is passed on every call too: a non-dispatch target such as print-obj-dir does not
# auto-resolve it the way `all` does, and an empty COMMON_LIB computes a different CFG_SIG, so the obj
# dir keys of two calls would diverge.
mkscratch s5
# Two-step clone (see adopt_worktree_clone): audiolib is cloned whole into S5_AUDIOLIB_ROOT and
# S5_AC_CLONE points at its common/ subdir. The name "audiolib_clone" is distinct from the "audiolib"
# sibling-directory symlink below.
S5_AUDIOLIB_ROOT="$SCRATCH_ROOT/s5/audiolib_clone"
S5_AEC_CLONE="$SCRATCH_ROOT/s5/aec_clone"
adopt_worktree_clone "$AUDIOLIB_REPO_DIR" "$S5_AUDIOLIB_ROOT"
S5_AC_CLONE="$S5_AUDIOLIB_ROOT/common"
adopt_worktree_clone "$AEC_REPO_DIR" "$S5_AEC_CLONE"

# AEC's c_impl/example/wav_io.h shim finds the canonical audiolib/common/include/wav_io.h through a
# hardcoded relative __has_include path ("../../../audiolib/common/include/wav_io.h" from
# c_impl/example/), i.e. a sibling directory named "audiolib" with "common" inside it, never through an
# -I path. The AEC clone's sibling is $SCRATCH_ROOT/s5/, so an "audiolib" dir holding a "common"
# symlink to the common clone satisfies the lookup without editing a tracked source.
mkdir -p "$SCRATCH_ROOT/s5/audiolib"
ln -s "$S5_AC_CLONE" "$SCRATCH_ROOT/s5/audiolib/common"
cd "$S5_AEC_CLONE/c_impl"

# AEC's `all` auto-builds the common clone's lib via the $(COMMON_LIB): FORCE rule -- no manual
# pre-build of the common clone is needed.
ac_a="$(make -s -C "$S5_AC_CLONE" print-lib-path)"
make -s all COMMON_LIB="$ac_a" COMMON_DIR="$S5_AC_CLONE" >/dev/null
bindir_a="$(make -s print-bin-dir COMMON_LIB="$ac_a" COMMON_DIR="$S5_AC_CLONE")"
aecwav="$bindir_a/aec_wav"
[ -f "$aecwav" ] && pass "S5: aec_wav built" || fail "S5: aec_wav missing after initial build"
mtime_1="$(mtime "$aecwav")"

# Backdate aec_wav (a clone artifact, freely mutable): GNU Make 3.81 compares prerequisite and target
# mtimes truncated to whole seconds, so aec_wav's relink can silently not fire when its mtime falls in
# the same integer second as the freshly rebuilt archive. A 5-minute backdate, applied only right
# before each of the three relink-triggering builds (never before the no-op check, which must see
# aec_wav's natural mtime), removes the collision. It goes on the target side of the comparison
# because the prerequisite's timestamp is a real rebuild's output this script cannot touch.
backdate_aecwav() { touch -A -0500 "$aecwav"; }

# no-op rebuild -> NOT relinked. No sleep needed: nothing is touched, so aec_wav's mtime cannot change
# however long the two `make` calls are apart.
make -s all COMMON_LIB="$ac_a" COMMON_DIR="$S5_AC_CLONE" >/dev/null
mtime_2="$(mtime "$aecwav")"
[ "$mtime_1" = "$mtime_2" ] && pass "S5: aec_wav NOT relinked on a no-op rebuild" \
  || fail "S5: aec_wav relinked with nothing changed"

# touch the common CLONE's src/hpf.c (mtime-only, content-neutral) -> hpf.o recompiles AND aec_wav
# relinks. common's CFG_SIG hashes its compiler invocation, not file content, so the archive path
# stays the same and only its mtime advances.
#
# Deterministic strictly-newer bump instead of `sleep 1; touch`: resolve hpf.o's path first (it exists
# from earlier scenarios), then set hpf.c's mtime to hpf_o's current mtime + 1s via BSD `touch -r/-A`.
# The +1s is load-bearing: make's dependency check truncates to whole seconds on GNU Make 3.81.
#
# "Did it recompile" is verified from make's build LOG, not an mtime diff: two mtime snapshots of the
# same output taken by this script can land in the same whole second when the edit-recompile-relink
# chain runs back to back. This Makefile echoes "common [$(BACKEND)] -> $@" whenever the archive rule
# runs (a plain `@echo` is not silenced by -s), so its presence in a captured log is a timing-proof
# signal that the archive (and hence hpf.o) rebuilt.
ac_objdir="$(make -s -C "$S5_AC_CLONE" print-obj-dir)"
hpf_o="$ac_objdir/hpf.o"
# Age every file of both clones by the same 10 minutes (relative mtime order preserved, .git
# untouched). From here on the only files newer than aec_wav are the ones each step below makes newer,
# so "aec_wav relinked" can only be caused by that step's change. (Without this, backdate_aecwav alone
# makes aec_wav older than all its freshly built prerequisites and the relink checks would pass
# whether or not anything changed.)
find "$S5_AC_CLONE" "$S5_AEC_CLONE" -name .git -prune -o -type f -exec touch -A -1000 {} +
# Backdate the CLONE's archive before forcing the recompile: the +1s hpf.c bump reliably makes make
# recompile hpf.o, but the fresh hpf.o's mtime can land in the same whole second as the archive's
# prior build (everything in the clone ran back to back), and GNU Make 3.81's whole-second check then
# skips the re-archive (hpf.o rebuilt, archive stale, no echo line). Backdating the archive (a clone
# artifact) by a safe margin makes it unambiguously older.
touch -A -0500 "$ac_a"
hpf_o_mtime_before="$(mtime "$hpf_o")"
touch -r "$hpf_o" -A 01 "$S5_AC_CLONE/src/hpf.c"
S5_LOG_HPF="$SCRATCH_ROOT/s5/log_hpf"
make -s -C "$S5_AC_CLONE" lib >"$S5_LOG_HPF" 2>&1
# Two conditions: hpf.o's own (fractional-second) mtime advanced = the recompile happened (the backdate
# above would re-archive even without it, so the echo line alone would not prove this), AND the
# archive-rebuild echo line = the rebuilt object reached the delivered archive.
if [ "$(mtime "$hpf_o")" != "$hpf_o_mtime_before" ] && grep -q "^common \[ne10\] ->" "$S5_LOG_HPF"; then
  pass "S5: hpf.c edit recompiled common's hpf.o (object mtime advanced) and the archive rebuilt"
else
  fail "S5: hpf.o did NOT recompile (or the archive did not rebuild) after touching hpf.c"
fi

# Same log-based technique for "did aec_wav relink": AEC's own Makefile
# echoes "aec_wav [$(BACKEND)] -> $@" whenever ITS link rule runs.
S5_LOG_RELINK="$SCRATCH_ROOT/s5/log_relink"
# No backdate here: after the aging above, the archive just rebuilt is the only prerequisite
# of aec_wav newer than aec_wav (by about 10 minutes), so a relink proves aec_wav tracks it.
make -s all COMMON_LIB="$ac_a" COMMON_DIR="$S5_AC_CLONE" >"$S5_LOG_RELINK" 2>&1
grep -q "^aec_wav \[ne10\] ->" "$S5_LOG_RELINK" && pass "S5: touching the common clone's src/hpf.c relinked aec_wav" \
  || fail "S5: aec_wav NOT relinked after common's hpf.o changed (no relink line in the log)"

# Header-only test 1: touch fast_math.h -> AEC objects that include it recompile, then aec_wav relinks.
# Same deterministic source bump and log-based relink verification (AEC has no per-object echo, but a
# relink only happens if some object changed, and this is the only header touched in this step).
aec_objdir="$(make -s print-obj-dir COMMON_LIB="$ac_a" COMMON_DIR="$S5_AC_CLONE")"
# An AEC object that includes fast_math.h, found from the .d files rather than by name (object names
# change; a hardcoded name would abort the suite).
s5_fm_o="$({ grep -l '/fast_math\.h' "$aec_objdir"/*.d 2>/dev/null || true; } | sed -n '1s/\.d$/.o/p')"
S5_LOG_FASTMATH="$SCRATCH_ROOT/s5/log_fastmath"
if [ -f "$s5_fm_o" ]; then
  touch -r "$s5_fm_o" -A 01 "$S5_AC_CLONE/utility/fast_math.h"
  s5_fm_mtime_before="$(mtime "$s5_fm_o")"
  backdate_aecwav
  make -s all COMMON_LIB="$ac_a" COMMON_DIR="$S5_AC_CLONE" >"$S5_LOG_FASTMATH" 2>&1
  # The object's own mtime is the proof of the recompile; the relink line alone is not (the
  # archive rebuilt above is also newer than the backdated aec_wav).
  [ "$(mtime "$s5_fm_o")" != "$s5_fm_mtime_before" ] && grep -q "^aec_wav \[ne10\] ->" "$S5_LOG_FASTMATH" \
    && pass "S5: touching the common clone's utility/fast_math.h recompiled an AEC object that includes it and aec_wav relinked" \
    || fail "S5: touching fast_math.h did not recompile its AEC consumer $(basename "$s5_fm_o") (or aec_wav did not relink)"
else
  fail "S5: no AEC object in $aec_objdir lists fast_math.h in its .d file -- the header-dependency probe cannot run"
fi

# Header-only test 2: touch wav_io.h -> example objs (aec_wav.o, which
# #includes example/wav_io.h -> common/include/wav_io.h) recompile.
s5_wv_o="$({ grep -l '/include/wav_io\.h' "$aec_objdir"/*.d 2>/dev/null || true; } | sed -n '1s/\.d$/.o/p')"
S5_LOG_WAVIO="$SCRATCH_ROOT/s5/log_wavio"
if [ -f "$s5_wv_o" ]; then
  touch -r "$s5_wv_o" -A 01 "$S5_AC_CLONE/include/wav_io.h"
  s5_wv_mtime_before="$(mtime "$s5_wv_o")"
  backdate_aecwav
  make -s all COMMON_LIB="$ac_a" COMMON_DIR="$S5_AC_CLONE" >"$S5_LOG_WAVIO" 2>&1
  [ "$(mtime "$s5_wv_o")" != "$s5_wv_mtime_before" ] && grep -q "^aec_wav \[ne10\] ->" "$S5_LOG_WAVIO" \
    && pass "S5: touching the common clone's include/wav_io.h recompiled an AEC object that includes it and aec_wav relinked" \
    || fail "S5: touching wav_io.h did not recompile its AEC consumer $(basename "$s5_wv_o") (or aec_wav did not relink)"
else
  fail "S5: no AEC object in $aec_objdir lists wav_io.h in its .d file -- the header-dependency probe cannot run"
fi

# NE10 vendored header touch -> NE10 objects recompile (common level). NE10_dsp.h, not the top-level
# NE10.h umbrella: NE10_fft.o's -MD -MP .d fragment shows NE10_fft.c pulls in
# NE10_dsp.h/NE10_types.h/NE10_macros.h but never NE10.h, which only fft_wrapper_ne10.c includes
# directly. Verified from the same "common [ne10] -> ..." log line.
ac_ne10_objdir="$(make -s -C "$S5_AC_CLONE" print-obj-dir)"
S5_LOG_NE10H="$SCRATCH_ROOT/s5/log_ne10h"
if [ -f "$ac_ne10_objdir/NE10_fft.o" ]; then
  touch -r "$ac_ne10_objdir/NE10_fft.o" -A 01 "$S5_AC_CLONE/lib/ne10/inc/NE10_dsp.h"
  ne10_o_mtime_before="$(mtime "$ac_ne10_objdir/NE10_fft.o")"
  # Same reason as the hpf step: make 3.81 compares whole seconds, and the archive was last
  # rebuilt (hpf step) possibly within the second the recompile lands in. The object's own
  # mtime then proves the recompile; the echo line proves it reached the archive.
  touch -A -0500 "$ac_a"
  make -s -C "$S5_AC_CLONE" lib >"$S5_LOG_NE10H" 2>&1
  [ "$(mtime "$ac_ne10_objdir/NE10_fft.o")" != "$ne10_o_mtime_before" ] && grep -q "^common \[ne10\] ->" "$S5_LOG_NE10H" \
    && pass "S5: touching a vendored NE10 header (NE10_dsp.h) recompiled NE10_fft.o (archive rebuilt)" \
    || fail "S5: NE10_fft.o did NOT recompile after touching lib/ne10/inc/NE10_dsp.h (or the archive did not rebuild)"
else
  fail "S5: $ac_ne10_objdir/NE10_fft.o missing -- the NE10 header-dependency probe cannot run"
fi

# A->B->A at the AEC level (SIMD=1 -> SIMD=0 -> SIMD=1): nm shows the matching NE10 kernel
# set each time, never a stale symbol set from the other config's link. A SIMD=1 build links
# the NEON R2C kernel; a SIMD=0 build routes through the scalar `_c` kernel and does not pull
# the NEON member in.
ac_a="$(make -s -C "$S5_AC_CLONE" SIMD=1 print-lib-path)"
make -s SIMD=1 all COMMON_LIB="$ac_a" COMMON_DIR="$S5_AC_CLONE" >/dev/null
bd_a="$(make -s SIMD=1 print-bin-dir COMMON_LIB="$ac_a" COMMON_DIR="$S5_AC_CLONE")"
ac_b="$(make -s -C "$S5_AC_CLONE" SIMD=0 print-lib-path)"
make -s SIMD=0 all COMMON_LIB="$ac_b" COMMON_DIR="$S5_AC_CLONE" >/dev/null
bd_b="$(make -s SIMD=0 print-bin-dir COMMON_LIB="$ac_b" COMMON_DIR="$S5_AC_CLONE")"
ac_a2="$(make -s -C "$S5_AC_CLONE" SIMD=1 print-lib-path)"
make -s SIMD=1 all COMMON_LIB="$ac_a2" COMMON_DIR="$S5_AC_CLONE" >/dev/null
bd_a2="$(make -s SIMD=1 print-bin-dir COMMON_LIB="$ac_a2" COMMON_DIR="$S5_AC_CLONE")"

# grep -c (not -q): under pipefail, `nm | grep -q` can fail a match when grep exits before nm has
# finished writing (SIGPIPE on nm); grep -c reads to EOF.
# nm's own failure (unreadable file, not an object) yields -1, which fails both the "> 0" and
# the "== 0" expectation; `nm | grep -c` alone would print 0 and let the SIMD=0 check pass.
s5_neon_refs() {
  local s
  s="$(nm "$1" 2>/dev/null)" || { echo -1; return 0; }
  printf '%s\n' "$s" | grep -c 'ne10_fft_r2c_1d_float32_neon' || true
}
ok=1
[ -f "$bd_a/aec_wav" ] && [ -f "$bd_b/aec_wav" ] && [ -f "$bd_a2/aec_wav" ] || ok=0
[ "$(s5_neon_refs "$bd_a/aec_wav")"  -gt 0 ] || ok=0
[ "$(s5_neon_refs "$bd_b/aec_wav")"  -eq 0 ] || ok=0
[ "$(s5_neon_refs "$bd_a2/aec_wav")" -gt 0 ] || ok=0
[ "$ok" -eq 1 ] && pass "S5: A->B->A at the AEC level -- nm shows the matching NE10 kernel set every time" \
  || fail "S5: A->B->A at the AEC level -- wrong/stale NE10 kernel set in one of the three aec_wav builds"

echo "############################################################"
echo "# S7p: producer-publish (v4 layout; throwaway DIST_ROOT)"
echo "############################################################"
cd "$AC_DIR"
mkscratch s7p
# ALLOW_DIRTY_PUBLISH=1 and ALLOW_UNTRACKED_PUBLISH=1 ride on every publish call in this script (except
# S17's policy sub-cases, which test each knob in isolation): a dev tree is legitimately dirty and may
# carry untracked files unrelated to what each scenario tests. DIST_ROOT stays a throwaway scratch
# path: the real dist/ is never read, written, or removed.
S7P_DIST="$SCRATCH_ROOT/s7p/dist"

S7P_LOG_FIRST="$SCRATCH_ROOT/s7p/log_first"
make -s SIMD=1 publish DIST_ROOT="$S7P_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >"$S7P_LOG_FIRST" 2>&1
grep -q "(attested: attest-" "$S7P_LOG_FIRST" && pass "S7p: publish success line ends with '(attested: <name>)'" \
  || fail "S7p: publish success line missing '(attested: <name>)'"

first_release_id="$(readlink "$S7P_DIST/ne10/current" || true)"
[ -n "$first_release_id" ] && [ -d "$S7P_DIST/ne10/$first_release_id" ] && \
  pass "S7p: publish -- current symlink resolves to a real release dir" \
  || fail "S7p: publish -- current symlink broken or missing"

manifest_path="$S7P_DIST/ne10/current/MANIFEST.txt"
grep -q "^release_id=$first_release_id$" "$manifest_path" && \
  pass "S7p: MANIFEST release_id= matches the resolved current target (v4 content-addressed id)" \
  || fail "S7p: MANIFEST release_id= missing or does not match readlink target ($first_release_id)"
if grep -q "^git_commit=" "$manifest_path" || grep -q "^git_dirty=" "$manifest_path" || grep -q "^date_utc=" "$manifest_path"; then
  fail "S7p: MANIFEST unexpectedly has a git_commit=/git_dirty=/date_utc= line (v4 MANIFEST must be deterministic)"
else
  pass "S7p: MANIFEST has NO git_commit=/git_dirty=/date_utc= line (v4 deterministic MANIFEST)"
fi
if grep -q "^ar=" "$manifest_path" && grep -q "^ranlib=" "$manifest_path" && grep -q "^link=" "$manifest_path"; then
  pass "S7p: MANIFEST has ar=/ranlib=/link= lines (regression P2-2 fields)"
else
  fail "S7p: MANIFEST missing one of ar=/ranlib=/link= lines"
fi

attest_files="$(find "$S7P_DIST/ne10/current/ATTEST" -name 'attest-*.txt')"
attest_n="$(printf '%s\n' "$attest_files" | grep -c . || true)"
[ "$attest_n" -eq 1 ] && pass "S7p: release has exactly one ATTEST/attest-*.txt after the first publish" \
  || fail "S7p: release ATTEST/ has $attest_n attest-*.txt file(s) after the first publish (expected 1)"
if grep -q "^git_commit=" "$attest_files" && grep -q "^git_dirty=" "$attest_files" && grep -q "^date_utc=" "$attest_files"; then
  pass "S7p: ATTEST file carries git_commit=/git_dirty=/date_utc="
else
  fail "S7p: ATTEST file $attest_files missing git_commit=/git_dirty=/date_utc="
fi

manifest_ok=1
while read -r sha fname; do
  [ "$fname" = "MANIFEST.txt" ] && continue
  actual="$(file_sha "$S7P_DIST/ne10/current/$fname")"
  [ "$actual" = "$sha" ] || manifest_ok=0
done < <(grep -E '^[0-9a-f]{64}  ' "$manifest_path")
[ "$manifest_ok" -eq 1 ] && pass "S7p: MANIFEST shas match the files on disk" \
  || fail "S7p: MANIFEST sha mismatch against files on disk"

first_dist_sha_before="$(file_sha "$S7P_DIST/ne10/$first_release_id/libaudio_common.a")"
make -s SIMD=0 publish DIST_ROOT="$S7P_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >/dev/null
first_dist_sha_after="$(file_sha "$S7P_DIST/ne10/$first_release_id/libaudio_common.a")"
second_release_id="$(readlink "$S7P_DIST/ne10/current" || true)"
[ "$first_dist_sha_before" = "$first_dist_sha_after" ] && pass "S7p: publishing SIMD=0 left the SIMD=1 release untouched" \
  || fail "S7p: publishing SIMD=0 CHANGED the SIMD=1 release"
[ -n "$second_release_id" ] && [ "$second_release_id" != "$first_release_id" ] && [ -d "$S7P_DIST/ne10/$second_release_id" ] && \
  pass "S7p: the SIMD=0 publish is a separate release and became current ($first_release_id -> $second_release_id)" \
  || fail "S7p: the SIMD=0 publish did not produce a distinct current release ($first_release_id -> $second_release_id)"

# Concurrent same-config publish: the lock-first driver means a real race resolves to exactly one
# winner in practice (S14 uses a fresh, slow-to-build config so "exactly one winner" is not flaky).
# Here the config is already built, so a wide scheduling gap could let both callers complete in turn;
# the check stays lenient: either one caller fails fast with "already held", or both complete. Either
# way `current` must end up pointing at a complete, self-consistent release.
S7P_LOG_A="$SCRATCH_ROOT/s7p/log_a"
S7P_LOG_B="$SCRATCH_ROOT/s7p/log_b"
( make -s publish DIST_ROOT="$S7P_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >"$S7P_LOG_A" 2>&1 ) & cp1=$!
( make -s publish DIST_ROOT="$S7P_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >"$S7P_LOG_B" 2>&1 ) & cp2=$!
cr1=0; cr2=0
wait "$cp1" || cr1=$?
wait "$cp2" || cr2=$?
if [ "$cr1" -eq 0 ] || [ "$cr2" -eq 0 ]; then
  pass "S7p: concurrent same-config publish -- at least one caller succeeded"
else
  fail "S7p: concurrent same-config publish -- BOTH callers failed"
  cat "$S7P_LOG_A" "$S7P_LOG_B" >&2
fi
if grep -q "already held" "$S7P_LOG_A" "$S7P_LOG_B" 2>/dev/null || { [ "$cr1" -eq 0 ] && [ "$cr2" -eq 0 ]; }; then
  pass "S7p: concurrent same-config publish -- lock serialised (one failed fast with 'already held', or both completed in turn)"
else
  fail "S7p: concurrent same-config publish -- no evidence of lock serialisation (both ran with no contention logged)"
fi
final_target="$(readlink "$S7P_DIST/ne10/current" || true)"
[ -n "$final_target" ] && [ -f "$S7P_DIST/ne10/$final_target/MANIFEST.txt" ] && [ -f "$S7P_DIST/ne10/$final_target/libaudio_common.a" ] && \
  pass "S7p: after concurrent publish, current points at a complete release dir" \
  || fail "S7p: after concurrent publish, current points at an incomplete/missing release dir"
manifest_ok=1
while read -r sha fname; do
  [ "$fname" = "MANIFEST.txt" ] && continue
  actual="$(file_sha "$S7P_DIST/ne10/$final_target/$fname")"
  [ "$actual" = "$sha" ] || manifest_ok=0
done < <(grep -E '^[0-9a-f]{64}  ' "$S7P_DIST/ne10/$final_target/MANIFEST.txt")
[ "$manifest_ok" -eq 1 ] && pass "S7p: final current release's MANIFEST is self-consistent" \
  || fail "S7p: final current release's MANIFEST does not match its own files"

echo "############################################################"
echo "# S8: CFG_SIG collision guard"
echo "############################################################"
cd "$AC_DIR"
mkscratch s8
S8_OBJ_ROOT="$SCRATCH_ROOT/s8/obj"
S8_BIN_ROOT="$SCRATCH_ROOT/s8/bin"

# Fresh scratch build of the real worktree's normal config: exercises the current (possibly
# uncommitted) Makefile without touching the real obj/ tree.
make -s BACKEND=ne10 OBJ_ROOT="$S8_OBJ_ROOT" BIN_ROOT="$S8_BIN_ROOT" lib >/dev/null
s8_objdir="$(make -s BACKEND=ne10 OBJ_ROOT="$S8_OBJ_ROOT" BIN_ROOT="$S8_BIN_ROOT" print-obj-dir)"
s8_lib="$(make -s BACKEND=ne10 OBJ_ROOT="$S8_OBJ_ROOT" BIN_ROOT="$S8_BIN_ROOT" print-lib-path)"
manifest="$s8_objdir/config.manifest"

real_objdir="$(make -s BACKEND=ne10 print-obj-dir)"
real_lib="$(make -s BACKEND=ne10 print-lib-path)"
real_manifest="$real_objdir/config.manifest"
real_manifest_sha_before="$(file_sha "$real_manifest")"
real_manifest_mtime_before="$(mtime "$real_manifest")"

echo "CORRUPTED PAYLOAD (test_build_isolation.sh S8)" > "$manifest"
s8_log="$SCRATCH_ROOT/s8/log"
if make BACKEND=ne10 OBJ_ROOT="$S8_OBJ_ROOT" BIN_ROOT="$S8_BIN_ROOT" lib >"$s8_log" 2>&1; then
  fail "S8: corrupted config.manifest did NOT fail the next build (collision guard not firing)"
else
  grep -q "CFG_SIG collision" "$s8_log" && pass "S8: corrupted config.manifest correctly FAILS the next build with the collision message" \
    || fail "S8: build failed but NOT with the expected collision message"
fi

real_manifest_sha_after="$(file_sha "$real_manifest")"
real_manifest_mtime_after="$(mtime "$real_manifest")"
[ "$real_manifest_sha_before" = "$real_manifest_sha_after" ] && [ "$real_manifest_mtime_before" = "$real_manifest_mtime_after" ] && \
  pass "S8: the REAL keyed obj dir's config.manifest sha+mtime unchanged (tamper was scratch-only)" \
  || fail "S8: the REAL keyed obj dir's config.manifest CHANGED (scratch isolation failed)"

# Knob byte-neutrality: OBJ_ROOT/BIN_ROOT is deliberately not in CFG_SIG (a placement knob, not a
# build-content knob). Whole-archive bytes are not compared: ar embeds each member's mtime in the
# header, so archives built at different times differ byte for byte though every member is identical.
# Compared at the archive-member level: the real obj dir can carry extra test objects (test_wav_io.o,
# ... from a prior `make test_wav` in that keyed config) that the scratch `make lib` never builds, so
# globbing the real obj dir would false-FAIL. Per-member content is compared by iterating the real
# archive's own `ar -t` list with `while IFS= read -r m`, never `for m in $(ar -t ...)`: macOS BSD ar
# has a pseudo-member literally named `__.SYMDEF SORTED` whose space would word-split. `__.SYMDEF*`
# members are skipped (the name-list check (a) covers the symbol table; GNU ar does not list it, so no
# fixed member count is assumed).
neutral_ok=1
while IFS= read -r m; do
  [ -n "$m" ] || continue
  case "$m" in
    __.SYMDEF*) continue ;;
  esac
  cmp -s <(ar -p "$real_lib" "$m") <(ar -p "$s8_lib" "$m") || neutral_ok=0
done < <(ar -t "$real_lib")
[ "$neutral_ok" -eq 1 ] && pass "S8: OBJ_ROOT/BIN_ROOT is knob-neutral -- every real archive member is byte-identical in the scratch archive" \
  || fail "S8: a real archive member's content differs (or is missing) in the scratch archive -- OBJ_ROOT/BIN_ROOT is not build-content-neutral"

[ "$(ar -t "$s8_lib" | sort)" = "$(ar -t "$real_lib" | sort)" ] && \
  pass "S8: ar -t member name-lists of scratch vs real archive are identical" \
  || fail "S8: ar -t member name-lists differ between scratch and real archive"

echo "############################################################"
echo "# S9: command-line override rejection"
echo "############################################################"
cd "$AC_DIR"
mkscratch s9
S9_OBJ_ROOT="$SCRATCH_ROOT/s9/obj"
S9_BIN_ROOT="$SCRATCH_ROOT/s9/bin"

# Plain baseline, only used below to prove EXTRA_CFLAGS lands somewhere different. Both
# queries share the SAME scratch OBJ_ROOT/BIN_ROOT so the comparison isolates the CFG_SIG
# effect of EXTRA_CFLAGS, not an incidental root-path difference.
baseline_objdir="$(make -s OBJ_ROOT="$S9_OBJ_ROOT" BIN_ROOT="$S9_BIN_ROOT" print-obj-dir)"

# CFLAGS=/CXXFLAGS=/LDFLAGS=/FP_POLICY= set on the command line must be rejected at parse time (before
# any obj/bin dir is created) with a message mentioning "cannot be overridden": these variables are
# internal, and the Makefile's own +=/:= appends would be silently defeated by a command-line origin.
# EXTRA_CFLAGS/EXTRA_LDFLAGS are the supported hook.
for pair in "CFLAGS=-O3" "CXXFLAGS=-O0" "LDFLAGS=-lfoo" "FP_POLICY=-ffp-contract=fast"; do
  ov_var="${pair%%=*}"
  ov_val="${pair#*=}"
  S9_LOG="$SCRATCH_ROOT/s9/log_$ov_var"
  if make "$ov_var=$ov_val" OBJ_ROOT="$S9_OBJ_ROOT" BIN_ROOT="$S9_BIN_ROOT" print-obj-dir >"$S9_LOG" 2>&1; then
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
extra_objdir="$(make -s EXTRA_CFLAGS=-DS9_PROBE OBJ_ROOT="$S9_OBJ_ROOT" BIN_ROOT="$S9_BIN_ROOT" print-obj-dir)"
[ -n "$extra_objdir" ] && [ "$extra_objdir" != "$baseline_objdir" ] && \
  pass "S9: EXTRA_CFLAGS=-DS9_PROBE print-obj-dir succeeds and lands in a different keyed dir than the plain baseline" \
  || fail "S9: EXTRA_CFLAGS=-DS9_PROBE print-obj-dir did NOT differ from the plain baseline ($baseline_objdir vs $extra_objdir)"

echo "############################################################"
echo "# S10: archive freshness -- stale-member removal"
echo "############################################################"
cd "$AC_DIR"
mkscratch s10
S10_OBJ_ROOT="$SCRATCH_ROOT/s10/obj"
S10_BIN_ROOT="$SCRATCH_ROOT/s10/bin"

make -s BACKEND=ne10 OBJ_ROOT="$S10_OBJ_ROOT" BIN_ROOT="$S10_BIN_ROOT" lib >/dev/null
s_lib="$(make -s BACKEND=ne10 OBJ_ROOT="$S10_OBJ_ROOT" BIN_ROOT="$S10_BIN_ROOT" print-lib-path)"

real_lib="$(make -s BACKEND=ne10 print-lib-path)"
real_lib_sha_before="$(file_sha "$real_lib")"
real_lib_mtime_before="$(mtime "$real_lib")"

# Inject a foreign object directly into the SCRATCH delivered archive (what an `ar rc` onto an
# existing archive would leave behind forever for a source file since removed from SRCS).
cat > "$SCRATCH_ROOT/s10/s10_foreign.c" <<'EOF'
int s10_foreign_symbol(void) { return 42; }
EOF
cc -c -o "$SCRATCH_ROOT/s10/s10_foreign.o" "$SCRATCH_ROOT/s10/s10_foreign.c"
ar r "$s_lib" "$SCRATCH_ROOT/s10/s10_foreign.o"
ar -t "$s_lib" | grep -q '^s10_foreign\.o$' && pass "S10: foreign object successfully injected into the SCRATCH archive (pre-condition)" \
  || fail "S10: failed to inject foreign object into scratch archive (pre-condition broken -- cannot test freshness)"

# A normal rebuild (source touched, config unchanged) must produce a FRESH archive (rm -f $@.tmp;
# ar rc $@.tmp $(BE_OBJS); ranlib; mv -f), never `ar rc` onto the existing one, so the foreign member
# must be gone afterward while hpf.o (a real, current source) is still present. The re-archive is
# triggered deterministically by backdating the SCRATCH archive to a fixed date older than all its
# member .o files, so make's dependency check ($(LIB): $(BE_OBJS)) sees it stale; nothing real is
# touched.
touch -t 202001010000 "$s_lib"
make -s BACKEND=ne10 OBJ_ROOT="$S10_OBJ_ROOT" BIN_ROOT="$S10_BIN_ROOT" lib >/dev/null

# Member list captured first: `ar -t | grep -q X && fail || pass` is fail-OPEN under pipefail
# (an ar failure, or a SIGPIPE when grep exits early, reaches `|| pass`).
s10_members="$(ar -t "$s_lib" 2>/dev/null)" || s10_members="<ar -t failed>"
if [ "$s10_members" = "<ar -t failed>" ]; then
  fail "S10: cannot list the rebuilt scratch archive ($s_lib)"
elif printf '%s\n' "$s10_members" | grep -qx 's10_foreign\.o'; then
  fail "S10: stale foreign member s10_foreign.o SURVIVED a fresh archive rebuild (ar rc onto an existing archive?)"
else
  pass "S10: stale foreign member s10_foreign.o REMOVED by the fresh-archive rebuild"
fi
ar -t "$s_lib" | grep -q '^hpf\.o$' && pass "S10: rebuilt archive still contains hpf.o" \
  || fail "S10: rebuilt archive missing hpf.o"

# SRCS= (the sorted source list) participates in CFG_SIG_PAYLOAD, visible
# verbatim in this config's config.manifest.
s10_objdir="$(make -s BACKEND=ne10 OBJ_ROOT="$S10_OBJ_ROOT" BIN_ROOT="$S10_BIN_ROOT" print-obj-dir)"
manifest="$s10_objdir/config.manifest"
if grep -q 'SRCS=' "$manifest" && grep -q 'src/hpf\.c' "$manifest"; then
  pass "S10: config.manifest's SRCS= entry participates in CFG_SIG and lists src/hpf.c"
else
  fail "S10: config.manifest missing SRCS= or src/hpf.c reference ($manifest)"
fi

real_lib_sha_after="$(file_sha "$real_lib")"
real_lib_mtime_after="$(mtime "$real_lib")"
[ "$real_lib_sha_before" = "$real_lib_sha_after" ] && [ "$real_lib_mtime_before" = "$real_lib_mtime_after" ] && \
  pass "S10: the REAL archive sha+mtime unchanged across the whole scenario" \
  || fail "S10: the REAL archive CHANGED during S10 (scratch isolation failed)"

echo "############################################################"
echo "# S11: publish immutability / content-addressing"
echo "############################################################"
cd "$AC_DIR"

# Captured before any of S11's sub-parts run, so S11c's "the real repo's src/hpf.c is untouched"
# assertion holds even if the tree already had local edits to it (never assume a clean tree, never
# `git checkout --`).
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
mkscratch s11a
S11A_DIST="$SCRATCH_ROOT/s11a/dist"
make -s BACKEND=ne10 publish DIST_ROOT="$S11A_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >/dev/null
id1="$(readlink "$S11A_DIST/ne10/current" || true)"
[ -n "$id1" ] && [ -d "$S11A_DIST/ne10/$id1" ] && \
  pass "S11a: first publish -- current resolves to a real release dir ($id1)" \
  || fail "S11a: first publish -- current symlink broken or missing"

manifest1="$S11A_DIST/ne10/$id1/MANIFEST.txt"
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

n_attest="$(count_attest "$S11A_DIST/ne10/$id1")"
[ "$n_attest" -eq 1 ] && pass "S11a: ATTEST/ has exactly 1 attest-*.txt after the first publish" \
  || fail "S11a: ATTEST/ has $n_attest attest-*.txt file(s) after the first publish (expected 1)"
attest1="$(find "$S11A_DIST/ne10/$id1/ATTEST" -name 'attest-*.txt')"
attest1_base="$(basename "$attest1")"
attest1_stem="${attest1_base%.txt}"
if grep -q "^event_id=$attest1_stem$" "$attest1" && grep -q "^git_commit=" "$attest1" && grep -q "^git_dirty=" "$attest1" && grep -q "^date_utc=" "$attest1"; then
  pass "S11a: the ATTEST file's event_id= matches its own filename stem, plus git_commit=/git_dirty=/date_utc="
else
  fail "S11a: ATTEST file $attest1 missing event_id=/git_commit=/git_dirty=/date_utc=, or event_id doesn't match its filename"
fi
commit1="$(grep '^git_commit=' "$attest1" | head -1 | cut -d= -f2)"
echo "$commit1" | grep -Eq '^[0-9a-f]{40}$' && pass "S11a: git_commit= is a full 40-hex-character OID" \
  || fail "S11a: git_commit=$commit1 is not 40 hex characters"

snap_before="$(release_snapshot "$S11A_DIST/ne10/$id1")"

# No sleep: the one-event-one-file ATTEST install (--excl-install + retry-with-next-<NNN>)
# disambiguates a same-second republish via the numeric suffix, so the two publishes need not land in
# different UTC seconds (S16 stress-tests the same-second case, forcing 20 publishes into one literal
# second via ATTEST_STAMP=).
S11A_LOG="$SCRATCH_ROOT/s11a/republish.log"
make -s BACKEND=ne10 publish DIST_ROOT="$S11A_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >"$S11A_LOG" 2>&1
grep -q "byte-verified" "$S11A_LOG" && pass "S11a: identical-content republish reports 'byte-verified'" \
  || fail "S11a: identical-content republish did NOT report 'byte-verified'"

id1_again="$(readlink "$S11A_DIST/ne10/current" || true)"
[ "$id1_again" = "$id1" ] && pass "S11a: current still points at id1 after identical-content republish" \
  || fail "S11a: current CHANGED after identical-content republish ($id1 -> $id1_again)"

snap_after="$(release_snapshot "$S11A_DIST/ne10/$id1")"
[ "$snap_before" = "$snap_after" ] && pass "S11a: release dir + artifact/MANIFEST mtimes UNCHANGED by identical-content republish" \
  || fail "S11a: release dir/artifact mtimes CHANGED by identical-content republish"

n_attest2="$(count_attest "$S11A_DIST/ne10/$id1")"
[ "$n_attest2" -eq 2 ] && pass "S11a: ATTEST/ has exactly 2 attest-*.txt files after the (same-second-capable) republish" \
  || fail "S11a: ATTEST/ has $n_attest2 attest-*.txt file(s) after the republish (expected 2)"

attest2="$(find "$S11A_DIST/ne10/$id1/ATTEST" -name 'attest-*.txt' | grep -v -F "$attest1" || true)"
if [ -n "$attest2" ] && [ "$attest2" != "$attest1" ]; then
  pass "S11a: republish's ATTEST file is a NEW, distinct file from the first publish's"
else
  fail "S11a: could not identify a second distinct ATTEST file after the republish"
fi
attest2_base="$(basename "$attest2")"
attest2_stem="${attest2_base%.txt}"
grep -q "^event_id=$attest2_stem$" "$attest2" 2>/dev/null && pass "S11a: second ATTEST file's event_id= matches its own filename stem" \
  || fail "S11a: second ATTEST file's event_id= does not match its filename"

echo "--- S11b: MANIFEST tamper detection -----------------------------------"
mkscratch s11b
S11B_DIST="$SCRATCH_ROOT/s11b/dist"
make -s BACKEND=ne10 publish DIST_ROOT="$S11B_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >/dev/null
idb="$(readlink "$S11B_DIST/ne10/current" || true)"
printf 'X' >> "$S11B_DIST/ne10/$idb/MANIFEST.txt"
S11B_LOG="$SCRATCH_ROOT/s11b/log"
if make -s BACKEND=ne10 publish DIST_ROOT="$S11B_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >"$S11B_LOG" 2>&1; then
  fail "S11b: republish after tampering with the stored MANIFEST unexpectedly SUCCEEDED"
else
  grep -q "differs from staged content" "$S11B_LOG" && \
    pass "S11b: republish after MANIFEST tamper correctly FAILS, mentioning 'differs from staged content'" \
    || fail "S11b: republish after MANIFEST tamper failed but WITHOUT the expected message"
fi

echo "--- S11c: content-change publish, exercised in a SANDBOX --------------"
# Never in $AC_DIR: a real codegen change is needed (a comment-only edit yields a byte-identical
# object, and the archiver runs in deterministic mode, so the content-addressed release id correctly
# stays the same). To exercise "same flags, different content -> new release id" without touching the
# real repo, this copies the tree into a throwaway sandbox and edits that copy.
mkscratch s11c
SANDBOX="$SCRATCH_ROOT/s11c/ac"
mkdir -p "$SANDBOX"
tar -c -C "$AC_DIR" --exclude obj --exclude bin --exclude dist --exclude .git . | tar -x -C "$SANDBOX"
S11C_DIST="$SCRATCH_ROOT/s11c/dist"

# The tar copy has no .git (excluded above), and a missing git identity is an unconditional publish
# refusal, so the sandbox gets its own disposable git identity: `git init` + one commit, entirely
# inside $SCRATCH_ROOT. The content edit below then makes it ordinarily tracked-dirty, which
# ALLOW_DIRTY_PUBLISH=1 admits; ALLOW_UNTRACKED_PUBLISH=1 rides along uniformly (build byproducts under
# the sandbox's obj/bin are gitignored via the copied .gitignore), so the scenario stays about content
# addressing, not git policy.
git -C "$SANDBOX" init -q
git -C "$SANDBOX" add -A
git -C "$SANDBOX" -c user.email=scratch@example.invalid -c user.name="scratch sandbox" \
  commit -q -m "scratch: S11c sandbox baseline (disposable repo under scratch only)"
make -C "$SANDBOX" -s BACKEND=ne10 publish DIST_ROOT="$S11C_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >/dev/null
sid1="$(readlink "$S11C_DIST/ne10/current" || true)"
[ -n "$sid1" ] && pass "S11c: sandbox publish #1 -- current resolves ($sid1)" \
  || fail "S11c: sandbox publish #1 -- current symlink broken or missing"

# Resolve the sandbox's own hpf.o AND archive paths before the edit. Force hpf.c's mtime to hpf.o's
# current mtime + 1s (BSD touch -r/-A) right after appending the content change, as S5 does: GNU Make
# 3.81 truncates its prerequisite-newer-than-target check to whole seconds, and an append right after
# the sandbox's first publish can land in the same integer second as hpf.o's mtime. The same
# truncation applies between the recompiled hpf.o and the archive's prior build (make would believe
# the archive up to date and skip `ar rc`, so the staged archive stays byte-identical), so the archive
# (a sandbox artifact, freely mutable) is also backdated by a safe margin before the edit.
sandbox_lib="$(make -C "$SANDBOX" -s BACKEND=ne10 print-lib-path)"
sandbox_hpf_o="$(make -C "$SANDBOX" -s BACKEND=ne10 print-obj-dir)/hpf.o"
touch -A -0500 "$sandbox_lib"
echo "void s11_isolation_probe(void) {}" >> "$SANDBOX/src/hpf.c"
touch -r "$sandbox_hpf_o" -A 01 "$SANDBOX/src/hpf.c"
make -C "$SANDBOX" -s BACKEND=ne10 publish DIST_ROOT="$S11C_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >/dev/null
sid2="$(readlink "$S11C_DIST/ne10/current" || true)"

[ -n "$sid2" ] && [ "$sid2" != "$sid1" ] && pass "S11c: sandbox content change -- publish produced a NEW release id ($sid1 -> $sid2)" \
  || fail "S11c: sandbox content change did NOT produce a new release id (sid1=$sid1 sid2=$sid2)"
[ -d "$S11C_DIST/ne10/$sid1" ] && [ -d "$S11C_DIST/ne10/$sid2" ] && \
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

# The REAL repo's src/hpf.c must be untouched: compare against the shasum
# captured at the very start of S11, not `git diff --quiet` (which would
# wrongly assume the file was clean to begin with).
HPF_SHA_AFTER_S11="$(file_sha "$AC_DIR/src/hpf.c")"
[ "$HPF_SHA_BEFORE_S11" = "$HPF_SHA_AFTER_S11" ] && pass "S11c: the REAL repo's src/hpf.c is byte-identical to its pre-scenario state" \
  || fail "S11c: the REAL repo's src/hpf.c CHANGED during S11 (sandbox isolation failed)"

echo "############################################################"
echo "# S12: toolchain coherence guard (CC/CXX target mismatch)"
echo "############################################################"
cd "$AC_DIR"
mkscratch s12
# The three shim builds below use a scratch OBJ_ROOT/BIN_ROOT: the shim's scratch path lands in CFG_SIG
# (CXX= is in the payload and $SCRATCH_ROOT is fresh every run), so building in the real obj/bin would
# mint a new orphan keyed directory pair per run. The guard under test does not care where objects
# land.
S12_OBJ_ROOT="$SCRATCH_ROOT/s12/obj"
S12_BIN_ROOT="$SCRATCH_ROOT/s12/bin"

# fake-cxx: reports a bogus -dumpmachine triple (so it can never match a
# real CC's), otherwise transparently forwards to the real c++ so a build
# using it as CXX still actually compiles/links.
SHIM="$SCRATCH_ROOT/s12/fake-cxx"
cat > "$SHIM" <<'EOF'
#!/usr/bin/env bash
if [ "$1" = "-dumpmachine" ]; then
  echo "s12-mismatched-triple"
  exit 0
fi
exec c++ "$@"
EOF
chmod +x "$SHIM"

S12_LOG="$SCRATCH_ROOT/s12/log1"
if make BACKEND=ne10 CXX="$SHIM" OBJ_ROOT="$S12_OBJ_ROOT" BIN_ROOT="$S12_BIN_ROOT" lib >"$S12_LOG" 2>&1; then
  fail "S12: make BACKEND=ne10 CXX=<mismatched shim> lib unexpectedly SUCCEEDED (toolchain guard did not fire)"
else
  grep -q "different targets" "$S12_LOG" && pass "S12: BACKEND=ne10 with a mismatched CXX -dumpmachine FAILS, mentioning 'different targets'" \
    || fail "S12: BACKEND=ne10 with a mismatched CXX FAILED but WITHOUT the expected 'different targets' message"
fi

S12_LOG2="$SCRATCH_ROOT/s12/log2"
if make BACKEND=ne10 CXX="$SHIM" TOOLCHAIN_CHECK=0 OBJ_ROOT="$S12_OBJ_ROOT" BIN_ROOT="$S12_BIN_ROOT" lib >"$S12_LOG2" 2>&1; then
  pass "S12: TOOLCHAIN_CHECK=0 skips the guard -- BACKEND=ne10 build with the mismatched CXX shim SUCCEEDS (its own keyed dir)"
else
  fail "S12: TOOLCHAIN_CHECK=0 build with the mismatched CXX shim FAILED (guard should have been skipped)"
  cat "$S12_LOG2" >&2
fi


echo "############################################################"
echo "# S13: atomic \`current\` symlink swap hammer + --excl-install"
echo "############################################################"
mkscratch s13
cc -O2 -o "$SCRATCH_ROOT/s13/swapln" "$AC_DIR/tools/atomic_symlink_swap.c"
mkdir -p "$SCRATCH_ROOT/s13/d/relA" "$SCRATCH_ROOT/s13/d/relB"
"$SCRATCH_ROOT/s13/swapln" relA "$SCRATCH_ROOT/s13/d/current"

# Background sampler: reads `current` as fast as possible. The assertion is on HARD misses: a failed
# readlink whose immediate re-read also fails. rename(2) leaves no window in the link's existence, so
# a correct swapln produces zero hard misses (an rm+mv fallback has an unlinked window and drops its
# temp into the target dir, which a hard miss would expose). A rare single-read transient failure is a
# macOS path-lookup-vs-rename race in the reader, not a gap in the link (a never-swapped control link
# sampled at the same rate shows none); soft misses are reported as INFO, never a failure.
S13_SAMPLER_LOG="$SCRATCH_ROOT/s13/sampler.log"
(
  while true; do
    v="$(readlink "$SCRATCH_ROOT/s13/d/current" 2>/dev/null || true)"
    if [ -z "$v" ]; then
      v2="$(readlink "$SCRATCH_ROOT/s13/d/current" 2>/dev/null || true)"
      if [ -z "$v2" ]; then printf 'HARDMISS\n' >> "$S13_SAMPLER_LOG"; \
      else printf 'SOFTMISS\n' >> "$S13_SAMPLER_LOG"; fi
    fi
  done
) &
s13_sampler_pid=$!

# 200 alternating foreground swaps between relA and relB.
for i in $(seq 1 200); do
  if [ $((i % 2)) -eq 1 ]; then
    "$SCRATCH_ROOT/s13/swapln" relA "$SCRATCH_ROOT/s13/d/current"
  else
    "$SCRATCH_ROOT/s13/swapln" relB "$SCRATCH_ROOT/s13/d/current"
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

leftover="$(ls "$SCRATCH_ROOT/s13/d" 2>/dev/null | grep -cE '^current\.[0-9]+\.tmp$' || true)"
leftover="${leftover:-0}"
[ "$leftover" -eq 0 ] && pass "S13: no current.*.tmp leftovers under $SCRATCH_ROOT/s13/d" \
  || fail "S13: $leftover leftover current.*.tmp temp file(s) under $SCRATCH_ROOT/s13/d"

s13_final="$(readlink "$SCRATCH_ROOT/s13/d/current" || true)"
if [ "$s13_final" = "relA" ] || [ "$s13_final" = "relB" ]; then
  pass "S13: final current resolves to relA or relB ($s13_final)"
else
  fail "S13: final current does NOT resolve to relA or relB ($s13_final)"
fi

echo "--- S13: --excl-install no-clobber mode (regression review P2-2) --------"
echo "hello-v1" > "$SCRATCH_ROOT/s13/src_v1.txt"
echo "hello-v2-should-not-land" > "$SCRATCH_ROOT/s13/src_v2.txt"
rc1=0
"$SCRATCH_ROOT/s13/swapln" --excl-install "$SCRATCH_ROOT/s13/src_v1.txt" "$SCRATCH_ROOT/s13/dst.txt" || rc1=$?
[ "$rc1" -eq 0 ] && pass "S13: --excl-install first call rc=0 (fresh install)" \
  || fail "S13: --excl-install first call rc=$rc1 (expected 0)"
rc2=0
"$SCRATCH_ROOT/s13/swapln" --excl-install "$SCRATCH_ROOT/s13/src_v2.txt" "$SCRATCH_ROOT/s13/dst.txt" || rc2=$?
[ "$rc2" -eq 2 ] && pass "S13: --excl-install second call (same dst) rc=2 (no-clobber EEXIST)" \
  || fail "S13: --excl-install second call rc=$rc2 (expected 2)"
dst_content="$(cat "$SCRATCH_ROOT/s13/dst.txt")"
[ "$dst_content" = "hello-v1" ] && pass "S13: --excl-install did NOT overwrite existing content on rc=2" \
  || fail "S13: --excl-install dst content changed despite rc=2 (dst=$dst_content)"
leftover_tmp="$(ls "$SCRATCH_ROOT/s13" 2>/dev/null | grep -cE '^dst\.txt\.[0-9]+\.tmp$' || true)"
leftover_tmp="${leftover_tmp:-0}"
[ "$leftover_tmp" -eq 0 ] && pass "S13: no dst.txt.*.tmp leftovers after the --excl-install collision" \
  || fail "S13: $leftover_tmp leftover dst.txt.*.tmp temp file(s)"

echo "############################################################"
echo "# S14: lock-before-build -- a concurrent publish loser builds nothing"
echo "############################################################"
cd "$AC_DIR"
mkscratch s14
S14_DIST="$SCRATCH_ROOT/s14/dist"
# One shared scratch OBJ_ROOT/BIN_ROOT pair for both racers: the lock still serialises the winner's
# build (what the scenario tests), and the real obj/bin never see this throwaway -DS14_LOCK_PROBE
# config.
S14_OBJ_ROOT="$SCRATCH_ROOT/s14/obj"
S14_BIN_ROOT="$SCRATCH_ROOT/s14/bin"

# A fresh, never-built config (BACKEND=ne10 plus a probe define unique to this scenario): the winner's
# critical section (a real compile+link+stage) is long enough that two callers launched back to back
# contend for the lock, making "exactly one winner" a reliable assertion (contrast S7p's
# already-built-config check, which stays lenient).
S14_LOG_A="$SCRATCH_ROOT/s14/log_a"; S14_LOG_B="$SCRATCH_ROOT/s14/log_b"
S14_RC_A="$SCRATCH_ROOT/s14/rc_a"; S14_RC_B="$SCRATCH_ROOT/s14/rc_b"
# `rc=0; cmd || rc=$?; ...` (not `cmd; echo $? >file`): under `set -e`, a
# bare failing command followed by `;` would abort this subshell BEFORE the
# echo ever ran. A command that is not the final one in a `||` list is
# exempt from triggering errexit, so this form reliably captures the make
# invocation's real exit status into its rc file even when it fails.
( rc=0; make -s BACKEND=ne10 EXTRA_CFLAGS=-DS14_LOCK_PROBE OBJ_ROOT="$S14_OBJ_ROOT" BIN_ROOT="$S14_BIN_ROOT" publish DIST_ROOT="$S14_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >"$S14_LOG_A" 2>&1 || rc=$?; echo "$rc" > "$S14_RC_A" ) &
p1=$!
( rc=0; make -s BACKEND=ne10 EXTRA_CFLAGS=-DS14_LOCK_PROBE OBJ_ROOT="$S14_OBJ_ROOT" BIN_ROOT="$S14_BIN_ROOT" publish DIST_ROOT="$S14_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >"$S14_LOG_B" 2>&1 || rc=$?; echo "$rc" > "$S14_RC_B" ) &
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
grep -q "common \[ne10\]" "$loser_log" && fail "S14: loser's log unexpectedly contains 'common [ne10]' -- it built the archive despite losing the lock" \
  || pass "S14: loser's log does NOT contain 'common [ne10]' -- it built nothing"
grep -q "common \[ne10\]" "$winner_log" && pass "S14: winner's log DOES contain 'common [ne10]' -- it actually built the archive" \
  || fail "S14: winner's log missing 'common [ne10]' (expected it to have built a fresh config)"

s14_target="$(readlink "$S14_DIST/ne10/current" || true)"
if [ -n "$s14_target" ] && [ -f "$S14_DIST/ne10/$s14_target/MANIFEST.txt" ] && ar -t "$S14_DIST/ne10/$s14_target/libaudio_common.a" >/dev/null 2>&1; then
  pass "S14: winner's release dir is complete (MANIFEST.txt present, libaudio_common.a readable by ar -t)"
else
  fail "S14: winner's release dir incomplete or archive unreadable ($S14_DIST/ne10/$s14_target)"
fi

stage_leftovers="$(find "$S14_DIST/ne10" -maxdepth 1 -name '.stage.*' 2>/dev/null | grep -c . || true)"
[ "$stage_leftovers" -eq 0 ] && pass "S14: no .stage.* leftovers under $S14_DIST/ne10/" \
  || fail "S14: $stage_leftovers .stage.* leftover(s) under $S14_DIST/ne10/"

echo "############################################################"
echo "# S15: make -n/-q/-t publish zero-write"
echo "############################################################"
cd "$AC_DIR"
mkscratch s15
# On top of the "no paths created" checks, each of -n/-q/-t must leave an already-existing delivered
# artifact byte- and mtime-untouched: `-t` (touch mode) would bump artifact mtimes if the publish
# driver recursed under it, so the driver makes -t a one-line no-op note plus exit 0 without recursion
# (asserted below). The reference artifact is the real, default-rooted $be archive built by S1. All
# three modes are zero-write.
for cfg in SIMD=1 SIMD=0; do
  be="${cfg/=/}"   # SIMD1 / SIMD0: label for scratch paths and messages
  S15_DIST="$SCRATCH_ROOT/s15/${be}_nx"
  S15_OBJ="$SCRATCH_ROOT/s15/${be}_no"
  S15_BIN="$SCRATCH_ROOT/s15/${be}_nb"
  S15_LOG="$SCRATCH_ROOT/s15/${be}_log"

  # Ensure the normal $be lib exists (a no-op after S1's builds), then
  # record its size+mtime+sha as the zero-write reference.
  S15_REAL_LIB="$(make -s $cfg print-lib-path)"
  make -s $cfg lib >/dev/null
  s15_ref_stat="$(stat -f '%m %z' "$S15_REAL_LIB")"
  s15_ref_sha="$(file_sha "$S15_REAL_LIB")"

  rc=0
  make -n $cfg DIST_ROOT="$S15_DIST" OBJ_ROOT="$S15_OBJ" BIN_ROOT="$S15_BIN" publish >"$S15_LOG.n" 2>&1 || rc=$?
  [ "$rc" -eq 0 ] && pass "S15[$be]: make -n publish exits rc=0" || fail "S15[$be]: make -n publish exits rc=$rc (expected 0)"
  if [ ! -e "$S15_DIST" ] && [ ! -e "$S15_OBJ" ] && [ ! -e "$S15_BIN" ]; then
    pass "S15[$be]: make -n publish created NONE of DIST_ROOT/OBJ_ROOT/BIN_ROOT"
  else
    fail "S15[$be]: make -n publish left behind a path (dist=$([ -e "$S15_DIST" ] && echo yes || echo no) obj=$([ -e "$S15_OBJ" ] && echo yes || echo no) bin=$([ -e "$S15_BIN" ] && echo yes || echo no))"
  fi
  [ "$(stat -f '%m %z' "$S15_REAL_LIB")" = "$s15_ref_stat" ] && [ "$(file_sha "$S15_REAL_LIB")" = "$s15_ref_sha" ] && \
    pass "S15[$be]: make -n publish is zero-write against the existing normal $be archive (size+mtime+sha unchanged)" \
    || fail "S15[$be]: make -n publish CHANGED the normal $be archive"

  rc=0
  make -q $cfg DIST_ROOT="$S15_DIST" OBJ_ROOT="$S15_OBJ" BIN_ROOT="$S15_BIN" publish >"$S15_LOG.q" 2>&1 || rc=$?
  [ "$rc" -ne 0 ] && pass "S15[$be]: make -q publish exits NONZERO (rc=$rc)" || fail "S15[$be]: make -q publish exited 0 (expected nonzero)"
  if [ ! -e "$S15_DIST" ] && [ ! -e "$S15_OBJ" ] && [ ! -e "$S15_BIN" ]; then
    pass "S15[$be]: make -q publish created NONE of DIST_ROOT/OBJ_ROOT/BIN_ROOT"
  else
    fail "S15[$be]: make -q publish left behind a path (dist=$([ -e "$S15_DIST" ] && echo yes || echo no) obj=$([ -e "$S15_OBJ" ] && echo yes || echo no) bin=$([ -e "$S15_BIN" ] && echo yes || echo no))"
  fi
  [ "$(stat -f '%m %z' "$S15_REAL_LIB")" = "$s15_ref_stat" ] && [ "$(file_sha "$S15_REAL_LIB")" = "$s15_ref_sha" ] && \
    pass "S15[$be]: make -q publish is zero-write against the existing normal $be archive (size+mtime+sha unchanged)" \
    || fail "S15[$be]: make -q publish CHANGED the normal $be archive"

  # -t must exit 0 AND print the explicit no-op note (the driver does not recurse under -t; see the
  # Makefile's dry-run guard).
  rc=0
  make -t $cfg DIST_ROOT="$S15_DIST" OBJ_ROOT="$S15_OBJ" BIN_ROOT="$S15_BIN" publish >"$S15_LOG.t" 2>&1 || rc=$?
  [ "$rc" -eq 0 ] && pass "S15[$be]: make -t publish exits rc=0" || fail "S15[$be]: make -t publish exits rc=$rc (expected 0)"
  grep -q "no-op" "$S15_LOG.t" && pass "S15[$be]: make -t publish output contains the no-op note" \
    || fail "S15[$be]: make -t publish output missing the no-op note text"
  if [ ! -e "$S15_DIST" ] && [ ! -e "$S15_OBJ" ] && [ ! -e "$S15_BIN" ]; then
    pass "S15[$be]: make -t publish created NONE of DIST_ROOT/OBJ_ROOT/BIN_ROOT"
  else
    fail "S15[$be]: make -t publish left behind a path (dist=$([ -e "$S15_DIST" ] && echo yes || echo no) obj=$([ -e "$S15_OBJ" ] && echo yes || echo no) bin=$([ -e "$S15_BIN" ] && echo yes || echo no))"
  fi
  [ "$(stat -f '%m %z' "$S15_REAL_LIB")" = "$s15_ref_stat" ] && [ "$(file_sha "$S15_REAL_LIB")" = "$s15_ref_sha" ] && \
    pass "S15[$be]: make -t publish is zero-write against the existing normal $be archive (size+mtime+sha unchanged)" \
    || fail "S15[$be]: make -t publish CHANGED the normal $be archive (touch semantics leaked through)"
done

echo "--- S15b: COMBINED dry-run flags stay zero-write against a STALE artifact --"
# Combined `make -nt publish`: the driver checks t first, because a child make that saw both flags
# applies real touch semantics down the _publish_impl prerequisite chain. Exercised against a
# scratch-root build whose archive is deliberately backdated below its own objects: exactly the state
# a leaked -t would "fix" by touching. Scratch-side only; the real tree's artifacts are covered by the
# S1 snapshot guard and must never be backdated.
mkscratch s15b
S15B_OBJ="$SCRATCH_ROOT/s15b/obj"; S15B_BIN="$SCRATCH_ROOT/s15b/bin"; S15B_DIST="$SCRATCH_ROOT/s15b/nx"
make -s BACKEND=ne10 OBJ_ROOT="$S15B_OBJ" BIN_ROOT="$S15B_BIN" lib >/dev/null
s15b_lib="$(make -s BACKEND=ne10 OBJ_ROOT="$S15B_OBJ" BIN_ROOT="$S15B_BIN" print-lib-path)"
touch -t 202001010000 "$s15b_lib"
s15b_ref="$(stat -f '%m %z' "$s15b_lib")"
for combo in "-nt" "-n -t" "-tq" "-nq" "-nqt"; do
  rm -rf "$S15B_DIST"
  rc=0; make $combo BACKEND=ne10 OBJ_ROOT="$S15B_OBJ" BIN_ROOT="$S15B_BIN" DIST_ROOT="$S15B_DIST" publish >/dev/null 2>&1 || rc=$?
  if [ ! -e "$S15B_DIST" ] && [ "$(stat -f '%m %z' "$s15b_lib")" = "$s15b_ref" ]; then
    pass "S15b[$combo]: zero-write (no DIST_ROOT; stale scratch archive untouched; rc=$rc informational)"
  else
    fail "S15b[$combo]: WROTE something (dist=$([ -e "$S15B_DIST" ] && echo yes || echo no), archive stat now '$(stat -f '%m %z' "$s15b_lib")' vs '$s15b_ref')"
  fi
done

echo "############################################################"
echo "# S16: ATTEST uniqueness under forced same-second collisions"
echo "############################################################"
cd "$AC_DIR"
mkscratch s16
S16_DIST="$SCRATCH_ROOT/s16/dist"
make -s BACKEND=ne10 publish DIST_ROOT="$S16_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >/dev/null
s16_id="$(readlink "$S16_DIST/ne10/current" || true)"
s16_attest_dir="$S16_DIST/ne10/$s16_id/ATTEST"

s16_before_list="$(find "$s16_attest_dir" -name 'attest-*.txt' | sort)"
s16_before_snap="$SCRATCH_ROOT/s16/before_snap.txt"
: > "$s16_before_snap"
for f in $s16_before_list; do
  [ -n "$f" ] || continue
  printf '%s %s %s %s\n' "$f" "$(stat -f '%i' "$f")" "$(mtime "$f")" "$(file_sha "$f")" >> "$s16_before_snap"
done

for i in $(seq 1 20); do
  make -s BACKEND=ne10 publish DIST_ROOT="$S16_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 ATTEST_STAMP=20260715T999999Z >/dev/null
done

s16_after_list="$(find "$s16_attest_dir" -name 'attest-*.txt' | sort)"
new_files="$(comm -13 <(printf '%s\n' "$s16_before_list") <(printf '%s\n' "$s16_after_list"))"
new_count="$(printf '%s\n' "$new_files" | grep -c . || true)"
new_count="${new_count:-0}"
[ "$new_count" -eq 20 ] && pass "S16: exactly 20 new ATTEST files after 20 same-stamp publishes" \
  || fail "S16: expected 20 new ATTEST files, found $new_count"

suffixes="$(printf '%s\n' "$new_files" | while read -r f; do [ -n "$f" ] || continue; bn="$(basename "$f" .txt)"; echo "${bn##*-}"; done | sort)"
expected="$(printf '%03d\n' $(seq 1 20))"
[ "$suffixes" = "$expected" ] && pass "S16: new files' -NNN suffixes are exactly 001..020, all distinct" \
  || fail "S16: -NNN suffixes ($(printf '%s' "$suffixes" | tr '\n' ',')) do not exactly match 001..020"

unchanged_ok=1
while read -r f inode mt sha; do
  [ -n "$f" ] || continue
  if [ ! -f "$f" ]; then unchanged_ok=0; continue; fi
  [ "$(stat -f '%i' "$f")" = "$inode" ] && [ "$(mtime "$f")" = "$mt" ] && [ "$(file_sha "$f")" = "$sha" ] || unchanged_ok=0
done < "$s16_before_snap"
[ "$unchanged_ok" -eq 1 ] && pass "S16: every pre-existing ATTEST file's inode+mtime+sha is unchanged" \
  || fail "S16: a pre-existing ATTEST file's inode/mtime/sha CHANGED"

tmp_leftovers="$(find "$s16_attest_dir" -name '*.tmp' 2>/dev/null | grep -c . || true)"
tmp_leftovers="${tmp_leftovers:-0}"
[ "$tmp_leftovers" -eq 0 ] && pass "S16: no *.tmp leftovers under ATTEST/" \
  || fail "S16: $tmp_leftovers *.tmp leftover(s) under ATTEST/"

spot_ok=1
spot_n=0
for f in $new_files; do
  [ -n "$f" ] || continue
  spot_n=$((spot_n + 1))
  [ "$spot_n" -gt 3 ] && break
  stem="$(basename "$f" .txt)"
  grep -q "^event_id=$stem$" "$f" || spot_ok=0
done
if [ "$spot_ok" -eq 1 ] && [ "$spot_n" -ge 1 ]; then
  pass "S16: spot-checked $([ "$spot_n" -gt 3 ] && echo 3 || echo "$spot_n") new ATTEST file(s) -- event_id= matches filename stem"
else
  fail "S16: a spot-checked ATTEST file's event_id= did not match its filename stem"
fi

echo "############################################################"
echo "# S17: dirty-checkout publish policy"
echo "############################################################"
mkscratch s17
# S17 runs in a clone made by the shared adopt_worktree_clone() helper (clean git identity reflecting
# today's possibly-uncommitted Makefile; disposable repo under $SCRATCH_ROOT only). Two-step clone:
# audiolib is cloned whole and S17_CLONE points at the clone's common/ subdir. The adoption commit
# absorbs today's uncommitted state across all directories, so the clone is clean at HEAD and only
# what S17 dirties under common/ shows up as a diff, matching what `make -C "$S17_CLONE" publish`
# computes (its tracked-dirty gate is whole-repo-scoped, its untracked gate cwd-scoped to common/).
S17_ROOT="$SCRATCH_ROOT/s17/audiolib_clone"
adopt_worktree_clone "$AUDIOLIB_REPO_DIR" "$S17_ROOT"
S17_CLONE="$S17_ROOT/common"

S17_OBJ_ROOT="$SCRATCH_ROOT/s17/obj"
S17_BIN_ROOT="$SCRATCH_ROOT/s17/bin"
S17_DIST="$SCRATCH_ROOT/s17/dist"

# attest_from_log(): the publish recipe's success line ends with "(attested: <name>)"; extracting the
# attest filename from this invocation's own log is unambiguous whichever release dir it landed in,
# unlike `find ATTEST -name 'attest-*.txt' | head -1`: a comment-only edit to hpf.c (S17b below) yields
# byte-identical object code, so the deterministic archiver reuses the same content-addressed release
# id as the clean publish and that dir holds two attest files.
attest_from_log() {
  local log="$1" name
  name="$(grep -o '(attested: [^)]*)' "$log" | sed -e 's/^(attested: //' -e 's/)$//' | head -1)"
  [ -n "$name" ] || return 1
  printf '%s\n' "$name"
}

echo "--- S17a: clean clone, DEFAULT publish -> must succeed ----------------"
S17_LOG_A="$SCRATCH_ROOT/s17/log_a"
if make -C "$S17_CLONE" -s BACKEND=ne10 OBJ_ROOT="$S17_OBJ_ROOT" BIN_ROOT="$S17_BIN_ROOT" DIST_ROOT="$S17_DIST" publish >"$S17_LOG_A" 2>&1; then
  pass "S17a: default publish on a CLEAN clone succeeds"
else
  fail "S17a: default publish on a clean clone FAILED"
  cat "$S17_LOG_A" >&2
fi
sid_a="$(readlink "$S17_DIST/ne10/current" || true)"
attest_a_name="$(attest_from_log "$S17_LOG_A" || true)"
attest_a="$S17_DIST/ne10/$sid_a/ATTEST/$attest_a_name"
clone_head="$(git -C "$S17_CLONE" rev-parse HEAD)"
if [ -n "$attest_a_name" ] && [ -f "$attest_a" ] && grep -q "^git_dirty=0$" "$attest_a" && grep -q "^allow_dirty_publish=0$" "$attest_a" \
   && grep -q "^git_commit=$clone_head$" "$attest_a" && ! grep -q "^dirty_diff_sha256=" "$attest_a"; then
  pass "S17a: attestation shows git_dirty=0, allow_dirty_publish=0, git_commit matches clone HEAD, no dirty_diff_sha256="
else
  fail "S17a: attestation ($attest_a) does not match the expected clean-clone shape"
fi

echo "--- S17b: dirty the clone's disposable hpf.c -> default publish FAILS --"
# A disposable comment-only edit to the CLONE's own hpf.c (never the real repo's). This scenario tests
# the dirty-publish policy (attestation fields), not content addressing (S11c covers that with a real
# code change), so whether the object bytes change is irrelevant.
echo "/* s17 disposable dirty probe */" >> "$S17_CLONE/src/hpf.c"
S17_LOG_B="$SCRATCH_ROOT/s17/log_b"
if make -C "$S17_CLONE" -s BACKEND=ne10 OBJ_ROOT="$S17_OBJ_ROOT" BIN_ROOT="$S17_BIN_ROOT" DIST_ROOT="$S17_DIST" publish >"$S17_LOG_B" 2>&1; then
  fail "S17b: default publish on a DIRTY clone unexpectedly SUCCEEDED"
else
  grep -qi "publish refused" "$S17_LOG_B" && grep -qi "dirty" "$S17_LOG_B" && \
    pass "S17b: default publish on a dirty clone FAILS, mentioning 'publish refused' and 'dirty'" \
    || fail "S17b: publish failed but without the expected 'publish refused'/'dirty' wording"
fi

echo "--- S17c: same dirty clone + ALLOW_DIRTY_PUBLISH=1 -> succeeds --------"
S17_LOG_C="$SCRATCH_ROOT/s17/log_c"
if make -C "$S17_CLONE" -s BACKEND=ne10 OBJ_ROOT="$S17_OBJ_ROOT" BIN_ROOT="$S17_BIN_ROOT" DIST_ROOT="$S17_DIST" ALLOW_DIRTY_PUBLISH=1 publish >"$S17_LOG_C" 2>&1; then
  pass "S17c: dirty clone + ALLOW_DIRTY_PUBLISH=1 succeeds"
else
  fail "S17c: dirty clone + ALLOW_DIRTY_PUBLISH=1 FAILED"
  cat "$S17_LOG_C" >&2
fi
sid_c="$(readlink "$S17_DIST/ne10/current" || true)"
attest_c_name="$(attest_from_log "$S17_LOG_C" || true)"
attest_c="$S17_DIST/ne10/$sid_c/ATTEST/$attest_c_name"
expected_ddiff="$(git -C "$S17_CLONE" diff --binary HEAD | shasum -a 256 | cut -d' ' -f1)"
if [ -n "$attest_c_name" ] && [ -f "$attest_c" ] && grep -q "^git_dirty=1$" "$attest_c" && grep -q "^allow_dirty_publish=1$" "$attest_c" \
   && grep -q "^dirty_diff_sha256=$expected_ddiff$" "$attest_c" && printf '%s' "$attest_c_name" | grep -q -- "-dirty-"; then
  pass "S17c: attestation shows git_dirty=1, allow_dirty_publish=1, matching dirty_diff_sha256=, and -dirty- in the filename"
else
  fail "S17c: attestation ($attest_c) does not match the expected dirty-publish shape"
fi

# --- untracked-file policy (S17d-S17h) -------------------
# The untracked dimension is separate from the tracked-dirty one: ALLOW_DIRTY_PUBLISH covers tracked
# changes only (`git status --porcelain -uno`), ALLOW_UNTRACKED_PUBLISH covers untracked files, and
# the attestation records untracked_tree_sha256 over their exact contents. These sub-cases assert the
# policy (which knob admits what; the hash field is present and changes whenever untracked bytes or
# symlink targets change), deliberately not the hash's internal record encoding. Each runs in its own
# fresh clone (S17b/c leave $S17_CLONE tracked-dirty) with clone-local gitignored obj/ and bin/.
echo "--- S17d: untracked file -> default refused; ALLOW_DIRTY alone refused; both admit --"
S17D_ROOT="$SCRATCH_ROOT/s17/audiolib_clone_d"
adopt_worktree_clone "$AUDIOLIB_REPO_DIR" "$S17D_ROOT"
S17D_CLONE="$S17D_ROOT/common"  # two-step clone; see adopt_worktree_clone's doc comment and S17's clone above
echo "int s17d_untracked_probe;" > "$S17D_CLONE/probe.c"

S17_LOG_D1="$SCRATCH_ROOT/s17/log_d1"
if make -C "$S17D_CLONE" -s BACKEND=ne10 DIST_ROOT="$S17_DIST" publish >"$S17_LOG_D1" 2>&1; then
  fail "S17d: DEFAULT publish with an untracked file unexpectedly SUCCEEDED"
else
  grep -qi "publish refused" "$S17_LOG_D1" && grep -q "UNTRACKED" "$S17_LOG_D1" && \
    pass "S17d: default publish with an untracked file FAILS, mentioning 'publish refused' + UNTRACKED" \
    || { fail "S17d: publish failed but without the expected refusal/UNTRACKED wording"; cat "$S17_LOG_D1" >&2; }
fi

S17_LOG_D2="$SCRATCH_ROOT/s17/log_d2"
if make -C "$S17D_CLONE" -s BACKEND=ne10 DIST_ROOT="$S17_DIST" ALLOW_DIRTY_PUBLISH=1 publish >"$S17_LOG_D2" 2>&1; then
  fail "S17d: ALLOW_DIRTY_PUBLISH=1 ALONE unexpectedly admitted an untracked file (the two dimensions must be separately gated)"
else
  grep -q "UNTRACKED" "$S17_LOG_D2" && \
    pass "S17d: ALLOW_DIRTY_PUBLISH=1 alone is STILL refused (untracked file present), mentioning UNTRACKED" \
    || { fail "S17d: ALLOW_DIRTY_PUBLISH=1-alone refusal did not mention UNTRACKED"; cat "$S17_LOG_D2" >&2; }
fi

S17_LOG_D3="$SCRATCH_ROOT/s17/log_d3"
if make -C "$S17D_CLONE" -s BACKEND=ne10 DIST_ROOT="$S17_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish >"$S17_LOG_D3" 2>&1; then
  pass "S17d: ALLOW_DIRTY_PUBLISH=1 + ALLOW_UNTRACKED_PUBLISH=1 together succeed"
else
  fail "S17d: both knobs together FAILED"
  cat "$S17_LOG_D3" >&2
fi
sid_d="$(readlink "$S17_DIST/ne10/current" || true)"
attest_d_name="$(attest_from_log "$S17_LOG_D3" || true)"
attest_d="$S17_DIST/ne10/$sid_d/ATTEST/$attest_d_name"
uhash_d="$(grep '^untracked_tree_sha256=' "$attest_d" 2>/dev/null | head -1 | cut -d= -f2)"
if [ -n "$attest_d_name" ] && [ -f "$attest_d" ] && grep -q "^git_dirty=0$" "$attest_d" \
   && grep -q "^git_untracked=1$" "$attest_d" && [ -n "$uhash_d" ] \
   && grep -q "^allow_untracked_publish=1$" "$attest_d"; then
  pass "S17d: attestation shows git_dirty=0, git_untracked=1, untracked_tree_sha256 present, allow_untracked_publish=1"
else
  fail "S17d: attestation ($attest_d) does not match the expected untracked-publish shape"
fi

echo "--- S17e: same untracked path, different bytes -> untracked_tree_sha256 differs --"
echo "int s17e_untracked_probe_DIFFERENT_bytes;" > "$S17D_CLONE/probe.c"
S17_LOG_E="$SCRATCH_ROOT/s17/log_e"
if make -C "$S17D_CLONE" -s BACKEND=ne10 DIST_ROOT="$S17_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish >"$S17_LOG_E" 2>&1; then
  pass "S17e: republish with different untracked bytes succeeds"
else
  fail "S17e: republish with different untracked bytes FAILED"
  cat "$S17_LOG_E" >&2
fi
sid_e="$(readlink "$S17_DIST/ne10/current" || true)"
attest_e_name="$(attest_from_log "$S17_LOG_E" || true)"
attest_e="$S17_DIST/ne10/$sid_e/ATTEST/$attest_e_name"
uhash_e="$(grep '^untracked_tree_sha256=' "$attest_e" 2>/dev/null | head -1 | cut -d= -f2)"
[ -n "$uhash_e" ] && [ -n "$uhash_d" ] && [ "$uhash_e" != "$uhash_d" ] && \
  pass "S17e: untracked_tree_sha256 DIFFERS from S17d's after changing probe.c's bytes -- two untracked source states can no longer share a provenance record" \
  || fail "S17e: untracked_tree_sha256 did NOT differ (d=$uhash_d e=$uhash_e)"

echo "--- S17f: untracked SYMLINK, then repointed -> hash differs each time ---------"
rm -f "$S17D_CLONE/probe.c"
ln -s "s17f-target-one" "$S17D_CLONE/probe.c"
S17_LOG_F1="$SCRATCH_ROOT/s17/log_f1"
if make -C "$S17D_CLONE" -s BACKEND=ne10 DIST_ROOT="$S17_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish >"$S17_LOG_F1" 2>&1; then
  pass "S17f: publish with an untracked symlink succeeds"
else
  fail "S17f: publish with an untracked symlink FAILED"
  cat "$S17_LOG_F1" >&2
fi
sid_f1="$(readlink "$S17_DIST/ne10/current" || true)"
attest_f1_name="$(attest_from_log "$S17_LOG_F1" || true)"
attest_f1="$S17_DIST/ne10/$sid_f1/ATTEST/$attest_f1_name"
uhash_f1="$(grep '^untracked_tree_sha256=' "$attest_f1" 2>/dev/null | head -1 | cut -d= -f2)"
[ -n "$uhash_f1" ] && [ "$uhash_f1" != "$uhash_e" ] && \
  pass "S17f: symlink probe.c -- untracked_tree_sha256 present and differs from S17e's regular-file hash" \
  || fail "S17f: symlink hash missing or identical to S17e's (e=$uhash_e f1=$uhash_f1)"

# Repoint the symlink's TARGET only (same path, same type): the hash must
# change again -- the symlink's target string is part of what the
# attestation fingerprints (however the record encodes it internally).
rm -f "$S17D_CLONE/probe.c"
ln -s "s17f-target-two-repointed" "$S17D_CLONE/probe.c"
S17_LOG_F2="$SCRATCH_ROOT/s17/log_f2"
if make -C "$S17D_CLONE" -s BACKEND=ne10 DIST_ROOT="$S17_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish >"$S17_LOG_F2" 2>&1; then
  pass "S17f: republish after repointing the symlink succeeds"
else
  fail "S17f: republish after repointing the symlink FAILED"
  cat "$S17_LOG_F2" >&2
fi
sid_f2="$(readlink "$S17_DIST/ne10/current" || true)"
attest_f2_name="$(attest_from_log "$S17_LOG_F2" || true)"
attest_f2="$S17_DIST/ne10/$sid_f2/ATTEST/$attest_f2_name"
uhash_f2="$(grep '^untracked_tree_sha256=' "$attest_f2" 2>/dev/null | head -1 | cut -d= -f2)"
[ -n "$uhash_f2" ] && [ "$uhash_f2" != "$uhash_f1" ] && \
  pass "S17f: untracked_tree_sha256 changes AGAIN after repointing only the symlink's target" \
  || fail "S17f: untracked_tree_sha256 did not change after repointing the symlink (f1=$uhash_f1 f2=$uhash_f2)"

echo "--- S17g: BUILD-RELEVANT untracked source (tracked Makefile edit wires it in) --"
# A tracked one-line Makefile edit extends COMMON_SRCS with an untracked source file, so the artifact
# depends on untracked bytes the tracked diff cannot account for. Requires BOTH knobs (each alone must
# refuse, naming its own dimension), and the release must differ from the clean clone's (S17a) release
# id, with the probe object a member of the delivered archive.
S17G_ROOT="$SCRATCH_ROOT/s17/audiolib_clone_g"
adopt_worktree_clone "$AUDIOLIB_REPO_DIR" "$S17G_ROOT"
S17G_CLONE="$S17G_ROOT/common"  # two-step clone; see adopt_worktree_clone's doc comment and S17's clone above
echo "int s17g_probe_symbol(void) { return 17; }" > "$S17G_CLONE/src/s17g_probe.c"
sed -i.s17gbak '/^COMMON_SRCS[[:space:]]*=/ s|$| src/s17g_probe.c|' "$S17G_CLONE/Makefile"
rm -f "$S17G_CLONE/Makefile.s17gbak"
grep -q 's17g_probe\.c' "$S17G_CLONE/Makefile" || fail "S17g: Makefile seam edit did not land (COMMON_SRCS line changed shape?)"

S17_LOG_G1="$SCRATCH_ROOT/s17/log_g1"
if make -C "$S17G_CLONE" -s BACKEND=ne10 DIST_ROOT="$S17_DIST" ALLOW_UNTRACKED_PUBLISH=1 publish >"$S17_LOG_G1" 2>&1; then
  fail "S17g: ALLOW_UNTRACKED_PUBLISH=1 ALONE unexpectedly admitted the tracked Makefile edit"
else
  grep -qi "publish refused" "$S17_LOG_G1" && grep -qi "dirty" "$S17_LOG_G1" && \
    pass "S17g: ALLOW_UNTRACKED_PUBLISH=1 alone is refused (tracked-dirty Makefile)" \
    || { fail "S17g: untracked-knob-only refusal did not name the tracked-dirty dimension"; cat "$S17_LOG_G1" >&2; }
fi

S17_LOG_G2="$SCRATCH_ROOT/s17/log_g2"
if make -C "$S17G_CLONE" -s BACKEND=ne10 DIST_ROOT="$S17_DIST" ALLOW_DIRTY_PUBLISH=1 publish >"$S17_LOG_G2" 2>&1; then
  fail "S17g: ALLOW_DIRTY_PUBLISH=1 ALONE unexpectedly admitted the untracked build input"
else
  grep -q "UNTRACKED" "$S17_LOG_G2" && \
    pass "S17g: ALLOW_DIRTY_PUBLISH=1 alone is refused (untracked source present)" \
    || { fail "S17g: dirty-knob-only refusal did not mention UNTRACKED"; cat "$S17_LOG_G2" >&2; }
fi

S17_LOG_G3="$SCRATCH_ROOT/s17/log_g3"
if make -C "$S17G_CLONE" -s BACKEND=ne10 DIST_ROOT="$S17_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish >"$S17_LOG_G3" 2>&1; then
  pass "S17g: both knobs together succeed"
else
  fail "S17g: both knobs together FAILED"
  cat "$S17_LOG_G3" >&2
fi
sid_g="$(readlink "$S17_DIST/ne10/current" || true)"
[ -n "$sid_g" ] && [ -n "$sid_a" ] && [ "$sid_g" != "$sid_a" ] && \
  pass "S17g: release id DIFFERS from the clean clone's (S17a's $sid_a -> $sid_g) -- the untracked source changed the deliverable" \
  || fail "S17g: release id did NOT differ from the clean baseline (sid_a=$sid_a sid_g=$sid_g)"
ar -t "$S17_DIST/ne10/$sid_g/libaudio_common.a" 2>/dev/null | grep -q '^s17g_probe\.o$' && \
  pass "S17g: the published archive contains s17g_probe.o -- direct evidence the untracked source landed in the artifact" \
  || fail "S17g: published archive does not list s17g_probe.o"
attest_g_name="$(attest_from_log "$S17_LOG_G3" || true)"
attest_g="$S17_DIST/ne10/$sid_g/ATTEST/$attest_g_name"
if [ -n "$attest_g_name" ] && grep -q "^git_dirty=1$" "$attest_g" && grep -q "^git_untracked=1$" "$attest_g" \
   && grep -q "^untracked_tree_sha256=" "$attest_g" && grep -q "^dirty_diff_sha256=" "$attest_g"; then
  pass "S17g: attestation records BOTH dimensions (git_dirty=1 + dirty_diff_sha256, git_untracked=1 + untracked_tree_sha256)"
else
  fail "S17g: attestation ($attest_g) missing one of the two dimensions' records"
fi

echo "--- S17h: unhashable untracked path (embedded git checkout) -> named FATAL ----"
S17H_ROOT="$SCRATCH_ROOT/s17/audiolib_clone_h"
adopt_worktree_clone "$AUDIOLIB_REPO_DIR" "$S17H_ROOT"
S17H_CLONE="$S17H_ROOT/common"  # two-step clone; see adopt_worktree_clone's doc comment and S17's clone above
mkdir "$S17H_CLONE/foo"
git -C "$S17H_CLONE/foo" init -q
S17_LOG_H="$SCRATCH_ROOT/s17/log_h"
if make -C "$S17H_CLONE" -s BACKEND=ne10 DIST_ROOT="$S17_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish >"$S17_LOG_H" 2>&1; then
  fail "S17h: publish with an embedded git checkout unexpectedly SUCCEEDED (unhashable path admitted)"
else
  if grep -q "FATAL" "$S17_LOG_H" && grep -q "cannot be hashed" "$S17_LOG_H" && grep -q "foo" "$S17_LOG_H"; then
    pass "S17h: publish FATALs on the unhashable untracked path, naming foo/ -- even with BOTH knobs set"
  else
    fail "S17h: publish failed but without the expected unhashable-refusal naming foo/"
    cat "$S17_LOG_H" >&2
  fi
fi

echo "--- S17i: fixed-field encoding -- the classic collision pair hashes DIFFER --"
# Raw space-joined "L <path> <target>" records would encode symlink "a b"->"c" and "a"->"b c"
# identically, so two different untracked trees would share one untracked_tree_sha256. The records are
# fixed-field (every variable-length value hashed), so this pair must hash differently. Assertions stay
# format-agnostic beyond "64-hex and different".
S17I_ROOT="$SCRATCH_ROOT/s17/audiolib_clone_i"
adopt_worktree_clone "$AUDIOLIB_REPO_DIR" "$S17I_ROOT"
S17I_CLONE="$S17I_ROOT/common"  # two-step clone; see adopt_worktree_clone's doc comment and S17's clone above
S17I_OBJ="$SCRATCH_ROOT/s17/obj_i"; S17I_BIN="$SCRATCH_ROOT/s17/bin_i"; S17I_DIST="$SCRATCH_ROOT/s17/dist_i"
s17i_publish() {
  local log="$SCRATCH_ROOT/s17/log_i$1" att rel
  make -C "$S17I_CLONE" -s BACKEND=ne10 OBJ_ROOT="$S17I_OBJ" BIN_ROOT="$S17I_BIN" DIST_ROOT="$S17I_DIST" \
       ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish >"$log" 2>&1 || return 1
  att="$(attest_from_log "$log")" || return 1
  rel="$(readlink "$S17I_DIST/ne10/current")" || return 1
  grep '^untracked_tree_sha256=' "$S17I_DIST/ne10/$rel/ATTEST/$att" | cut -d= -f2
}
ln -s "c" "$S17I_CLONE/a b"
s17i_h1="$(s17i_publish a || true)"
rm "$S17I_CLONE/a b"; ln -s "b c" "$S17I_CLONE/a"
s17i_h2="$(s17i_publish b || true)"
if [ -n "$s17i_h1" ] && [ "${#s17i_h1}" -eq 64 ] && [ -n "$s17i_h2" ] && [ "$s17i_h1" != "$s17i_h2" ]; then
  pass "S17i: collision pair ('a b'->'c' vs 'a'->'b c') yields DIFFERENT untracked_tree_sha256 values"
else
  fail "S17i: collision pair hashes '$s17i_h1' vs '$s17i_h2' (must both be 64-hex and differ)"
fi

echo "--- S17j: unreadable untracked file (chmod 000) -> fail-closed refusal ------"
# A shasum failing inside a command substitution must not yield an empty content hash with rc=0 (a
# hole in the provenance): any per-entry I/O failure downgrades the entry to an X record => named
# FATAL.
echo "secret" > "$S17I_CLONE/unreadable.bin"; chmod 000 "$S17I_CLONE/unreadable.bin"
S17_LOG_J="$SCRATCH_ROOT/s17/log_j"
if make -C "$S17I_CLONE" -s BACKEND=ne10 OBJ_ROOT="$S17I_OBJ" BIN_ROOT="$S17I_BIN" DIST_ROOT="$S17I_DIST" \
     ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish >"$S17_LOG_J" 2>&1; then
  fail "S17j: publish with an unreadable untracked file unexpectedly SUCCEEDED (empty-hash fail-open)"
else
  if grep -q "cannot be hashed deterministically" "$S17_LOG_J" && grep -q "unreadable.bin" "$S17_LOG_J"; then
    pass "S17j: unreadable untracked file refused BY NAME (I/O failure -> X record, never an empty hash)"
  else
    fail "S17j: publish failed but without naming unreadable.bin in the unhashable refusal"
    cat "$S17_LOG_J" >&2
  fi
fi
chmod 644 "$S17I_CLONE/unreadable.bin"; rm "$S17I_CLONE/unreadable.bin"

echo "--- S17k: a mode change ALONE must change untracked_tree_sha256 -------------"
rm "$S17I_CLONE/a"
echo "body" > "$S17I_CLONE/probe.bin"; chmod 600 "$S17I_CLONE/probe.bin"
s17k_h1="$(s17i_publish c || true)"
chmod 755 "$S17I_CLONE/probe.bin"
s17k_h2="$(s17i_publish d || true)"
if [ -n "$s17k_h1" ] && [ "${#s17k_h1}" -eq 64 ] && [ "$s17k_h1" != "$s17k_h2" ]; then
  pass "S17k: chmod 600->755 changes untracked_tree_sha256 (F records carry the mode field)"
else
  fail "S17k: mode-change hashes '$s17k_h1' vs '$s17k_h2' (must differ)"
fi

echo "--- S17l: identity-less checkout -> UNCONDITIONAL refusal -------------------"
# An unknown commit identity sets dirty=untracked=unknown; neither ALLOW_DIRTY_PUBLISH=1 nor
# ALLOW_UNTRACKED_PUBLISH=1 may admit it: publish refuses outright.
S17L_DIR="$SCRATCH_ROOT/s17/nogit"
mkdir -p "$S17L_DIR"
tar -c -C "$AC_DIR" --exclude obj --exclude bin --exclude dist --exclude .git . | tar -x -C "$S17L_DIR"
S17_LOG_L="$SCRATCH_ROOT/s17/log_l"
if make -C "$S17L_DIR" -s BACKEND=ne10 OBJ_ROOT="$SCRATCH_ROOT/s17/obj_l" BIN_ROOT="$SCRATCH_ROOT/s17/bin_l" \
     DIST_ROOT="$SCRATCH_ROOT/s17/dist_l" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish >"$S17_LOG_L" 2>&1; then
  fail "S17l: publish from a no-git-identity tree unexpectedly SUCCEEDED (unknown slipped past the hatches)"
else
  if grep -q "no git identity" "$S17_LOG_L"; then
    pass "S17l: identity-less checkout refused UNCONDITIONALLY -- neither escape hatch admits it"
  else
    fail "S17l: publish failed but without the no-git-identity message"
    cat "$S17_LOG_L" >&2
  fi
fi

echo "############################################################"
echo "# S18: interruption-safety probe"
echo "############################################################"
mkscratch s18
cat > "$SCRATCH_ROOT/s18/sigreset_exec.c" <<'EOF'
/* Tiny launcher: resets SIGINT/SIGQUIT to their default disposition before
 * exec'ing the real command. Needed because bash, for a NON-interactive
 * script's backgrounded (`cmd &`) jobs, sets SIGINT/SIGQUIT to SIG_IGN in
 * the forked child -- and bash's own `trap` builtin refuses to install (or
 * even reset) a handler for a signal that was already SIG_IGN "upon entry
 * to the shell" (see bash(1), SIGNALS: "Signals ignored upon entry to the
 * shell cannot be trapped or reset."). A plain C signal()/execvp() is not
 * bound by that bash-specific policy, so this helper is the only reliable
 * way for this scenario's backgrounded child to actually SEE a SIGINT its
 * own `trap ... INT` can catch (verified empirically: without this, `kill
 * -INT` on a `(...) &`-launched child never invokes its INT trap at all). */
#include <signal.h>
#include <unistd.h>
int main(int argc, char** argv) {
    signal(SIGINT, SIG_DFL);
    signal(SIGQUIT, SIG_DFL);
    if (argc < 2) return 127;
    execvp(argv[1], argv + 1);
    return 127;
}
EOF
cc -O2 -o "$SCRATCH_ROOT/s18/sigreset_exec" "$SCRATCH_ROOT/s18/sigreset_exec.c"

run_s18_mode() {
  mode="$1" sig="$2" expect_rc="$3"
  d="$SCRATCH_ROOT/s18/$mode"
  mkdir -p "$d/probe_tmp"
  mkfifo "$d/ready.fifo" "$d/hold.fifo"
  ( ISOL_INTERRUPT_PROBE="$mode" ISOL_PROBE_FIFO="$d/ready.fifo" TMPDIR="$d/probe_tmp" \
    "$SCRATCH_ROOT/s18/sigreset_exec" bash "$SCRIPT_PATH" ) &
  pid=$!
  child_scratch="$(cat "$d/ready.fifo")"
  if [ -n "$sig" ]; then
    kill "-$sig" "$pid" 2>/dev/null || true
  fi
  rc=0
  wait "$pid" || rc=$?
  [ "$rc" -eq "$expect_rc" ] && pass "S18[$mode]: child exit code = $rc (expected $expect_rc)" \
    || fail "S18[$mode]: child exit code = $rc (expected $expect_rc)"
  [ ! -e "$child_scratch" ] && pass "S18[$mode]: child's own SCRATCH_ROOT no longer exists after exit" \
    || fail "S18[$mode]: child's SCRATCH_ROOT ($child_scratch) STILL EXISTS after exit"
  if [ -d "$d/probe_tmp" ] && [ -z "$(ls -A "$d/probe_tmp" 2>/dev/null)" ]; then
    pass "S18[$mode]: probe TMPDIR is empty afterward (the child's scratch root landed inside it and was fully removed)"
  else
    fail "S18[$mode]: probe TMPDIR ($d/probe_tmp) is NOT empty afterward"
  fi
}

run_s18_mode exit "" 0
run_s18_mode term TERM 143
run_s18_mode intr INT 130

echo "############################################################"
echo "# S19: FFT_WRAPPER_ALIAS_CFLAGS is CFG_SIG-keyed (build-cache-"
echo "# invalidation regression guard)"
echo "############################################################"
# Guards the build-cache gap behind -fno-strict-aliasing: a bare literal on the target-specific
# `CFLAGS +=` line of the fft_wrapper_ne10.o rule is invisible to CFG_SIG (a target-specific
# assignment does not change what $(CFLAGS) expanded to when the parse-time CFG_SIG_PAYLOAD was
# computed), so editing it would leave CFG_SIG unchanged and a keyed directory built earlier would keep
# an object compiled without it. The flag lives in the named variable $(FFT_WRAPPER_ALIAS_CFLAGS),
# folded into CFG_SIG_PAYLOAD and referenced (never spelled as a literal) on the target-specific line.
#
# The property is proved without a command-line override (rejected, see S20): one real `make lib`
# runs off the unmodified Makefile at $AC_DIR, the other off a scratch clone (adopt_worktree_clone;
# VPATH, includes and sources need the whole tree) whose Makefile copy has only the
# FFT_WRAPPER_ALIAS_CFLAGS definition line sed-patched to a different literal. Both share one scratch
# OBJ_ROOT/BIN_ROOT so the comparison isolates the CFG_SIG effect of the literal. Reverting the
# Makefile fix makes both builds land in the same keyed directory.
cd "$AC_DIR"
mkscratch s19
# Two-step clone (see adopt_worktree_clone): S19_CLONE points at the audiolib clone's common/ subdir.
S19_ROOT="$SCRATCH_ROOT/s19/audiolib_clone"
adopt_worktree_clone "$AUDIOLIB_REPO_DIR" "$S19_ROOT"
S19_CLONE="$S19_ROOT/common"

S19_CHANGED_FLAGS="-fno-strict-aliasing -fwrapv"
sed -i.s19bak "s|^FFT_WRAPPER_ALIAS_CFLAGS := -fno-strict-aliasing\$|FFT_WRAPPER_ALIAS_CFLAGS := $S19_CHANGED_FLAGS|" "$S19_CLONE/Makefile"
rm -f "$S19_CLONE/Makefile.s19bak"
grep -q "^FFT_WRAPPER_ALIAS_CFLAGS := $S19_CHANGED_FLAGS\$" "$S19_CLONE/Makefile" || \
  fail "S19: Makefile seam edit did not land (FFT_WRAPPER_ALIAS_CFLAGS definition line changed shape?)"

S19_OBJ_ROOT="$SCRATCH_ROOT/s19/obj"
S19_BIN_ROOT="$SCRATCH_ROOT/s19/bin"

s19_base_objdir="$(make -C "$AC_DIR" -s BACKEND=ne10 OBJ_ROOT="$S19_OBJ_ROOT" BIN_ROOT="$S19_BIN_ROOT" print-obj-dir)"
s19_base_lib="$(make -C "$AC_DIR" -s BACKEND=ne10 OBJ_ROOT="$S19_OBJ_ROOT" BIN_ROOT="$S19_BIN_ROOT" print-lib-path)"
if make -C "$AC_DIR" -s BACKEND=ne10 OBJ_ROOT="$S19_OBJ_ROOT" BIN_ROOT="$S19_BIN_ROOT" lib >"$SCRATCH_ROOT/s19/build_base.log" 2>&1; then
  pass "S19: baseline build (real, unmodified Makefile, default FFT_WRAPPER_ALIAS_CFLAGS) succeeds"
else
  fail "S19: baseline build (real, unmodified Makefile) FAILED"
  cat "$SCRATCH_ROOT/s19/build_base.log" >&2
fi

s19_changed_objdir="$(make -C "$S19_CLONE" -s BACKEND=ne10 OBJ_ROOT="$S19_OBJ_ROOT" BIN_ROOT="$S19_BIN_ROOT" print-obj-dir)"
s19_changed_lib="$(make -C "$S19_CLONE" -s BACKEND=ne10 OBJ_ROOT="$S19_OBJ_ROOT" BIN_ROOT="$S19_BIN_ROOT" print-lib-path)"
if make -C "$S19_CLONE" -s BACKEND=ne10 OBJ_ROOT="$S19_OBJ_ROOT" BIN_ROOT="$S19_BIN_ROOT" lib >"$SCRATCH_ROOT/s19/build_changed.log" 2>&1; then
  pass "S19: changed-FFT_WRAPPER_ALIAS_CFLAGS build (scratch clone, sed-patched Makefile literal -- NOT a command-line override) succeeds"
else
  fail "S19: changed-FFT_WRAPPER_ALIAS_CFLAGS build (scratch clone) FAILED"
  cat "$SCRATCH_ROOT/s19/build_changed.log" >&2
fi

[ -n "$s19_base_objdir" ] && [ -n "$s19_changed_objdir" ] && [ "$s19_base_objdir" != "$s19_changed_objdir" ] && \
  pass "S19: changing FFT_WRAPPER_ALIAS_CFLAGS (via the scratch clone's sed-patched Makefile) lands in a DIFFERENT keyed obj dir (CFG_SIG sees the change)" \
  || fail "S19: FFT_WRAPPER_ALIAS_CFLAGS change did NOT change the keyed obj dir ($s19_base_objdir vs $s19_changed_objdir) -- CFG_SIG is blind to this knob again"

[ -n "$s19_base_lib" ] && [ -n "$s19_changed_lib" ] && [ "$s19_base_lib" != "$s19_changed_lib" ] && \
  pass "S19: ...and the two builds' print-lib-path archives differ too" \
  || fail "S19: FFT_WRAPPER_ALIAS_CFLAGS change did NOT change print-lib-path ($s19_base_lib vs $s19_changed_lib)"

if [ -f "$s19_base_lib" ] && [ -f "$s19_changed_lib" ]; then
  pass "S19: both configs' archives actually exist on disk (real builds, not just path queries)"
else
  fail "S19: expected BOTH archives to exist (base=$s19_base_lib exists=$([ -f "$s19_base_lib" ] && echo yes || echo no); changed=$s19_changed_lib exists=$([ -f "$s19_changed_lib" ] && echo yes || echo no))"
fi

ar -t "$s19_changed_lib" 2>/dev/null | grep -q '^fft_wrapper_ne10\.o$' && \
  pass "S19: changed-config archive contains fft_wrapper_ne10.o (the object the alias flag targets)" \
  || fail "S19: changed-config archive missing fft_wrapper_ne10.o"

echo "############################################################"
echo "# S20: FFT_WRAPPER_ALIAS_CFLAGS command-line override rejection"
echo "############################################################"
# Negative-test companion to S19: FFT_WRAPPER_ALIAS_CFLAGS is a name in the same "Command-line
# override rejection" foreach as CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/FP_POLICY (S9), so each override must
# FAIL at parse time (before any obj/bin dir is created) with the same "cannot be overridden" message
# S9 checks. Two real command-line overrides under a scratch OBJ_ROOT/BIN_ROOT: an empty value (which
# silently drops -fno-strict-aliasing) and an explicit `-fstrict-aliasing`.
cd "$AC_DIR"
mkscratch s20
S20_OBJ_ROOT="$SCRATCH_ROOT/s20/obj"
S20_BIN_ROOT="$SCRATCH_ROOT/s20/bin"

for pair in "FFT_WRAPPER_ALIAS_CFLAGS=" "FFT_WRAPPER_ALIAS_CFLAGS=-fstrict-aliasing"; do
  ov_var="${pair%%=*}"
  ov_val="${pair#*=}"
  S20_LOG="$SCRATCH_ROOT/s20/log_$(printf '%s' "$ov_val" | tr -c 'A-Za-z0-9' '_')"
  if make "$ov_var=$ov_val" OBJ_ROOT="$S20_OBJ_ROOT" BIN_ROOT="$S20_BIN_ROOT" print-obj-dir >"$S20_LOG" 2>&1; then
    fail "S20: make $ov_var=$ov_val print-obj-dir unexpectedly SUCCEEDED (must be rejected at parse time)"
  else
    if grep -q "cannot be overridden" "$S20_LOG"; then
      pass "S20: make $ov_var=$ov_val print-obj-dir correctly FAILS, mentioning 'cannot be overridden'"
    else
      fail "S20: make $ov_var=$ov_val print-obj-dir failed but WITHOUT the expected 'cannot be overridden' message"
      cat "$S20_LOG" >&2
    fi
  fi
done

echo "############################################################"
echo "# S21: FFT_WRAPPER_ALIAS_CFLAGS \`make -e\` (environment-override mode)"
echo "# bypass of S20's own guard"
echo "############################################################"
# S20 covered the command line. Under `make -e` GNU Make flips a variable's origin from "environment"
# to "environment override" only when a makefile assignment to it is parsed, so the foreach must see
# the bare literal defined above it. Two cases mirror S20's, run through `env NAME=value make -e ...`
# (an environment override, not a command-line assignment): the hostile value `-fstrict-aliasing` and
# the empty value.
cd "$AC_DIR"
mkscratch s21
S21_OBJ_ROOT="$SCRATCH_ROOT/s21/obj"
S21_BIN_ROOT="$SCRATCH_ROOT/s21/bin"

for pair in "FFT_WRAPPER_ALIAS_CFLAGS=-fstrict-aliasing" "FFT_WRAPPER_ALIAS_CFLAGS="; do
  ov_var="${pair%%=*}"
  ov_val="${pair#*=}"
  S21_LOG="$SCRATCH_ROOT/s21/log_$(printf '%s' "$ov_val" | tr -c 'A-Za-z0-9' '_')"
  if env "$ov_var=$ov_val" make -e BACKEND=ne10 OBJ_ROOT="$S21_OBJ_ROOT" BIN_ROOT="$S21_BIN_ROOT" print-obj-dir >"$S21_LOG" 2>&1; then
    fail "S21: env $ov_var=$ov_val make -e print-obj-dir unexpectedly SUCCEEDED (must be rejected at parse time)"
  else
    if grep -q "cannot be overridden" "$S21_LOG"; then
      pass "S21: env $ov_var=$ov_val make -e print-obj-dir correctly FAILS, mentioning 'cannot be overridden'"
    else
      fail "S21: env $ov_var=$ov_val make -e print-obj-dir failed but WITHOUT the expected 'cannot be overridden' message"
      cat "$S21_LOG" >&2
    fi
  fi
done

# End-to-end confirmation, one level closer to reality than print-obj-dir: a real \`lib\` build must
# be rejected the same way (not just the print-* introspection targets).
S21_LIB_LOG="$SCRATCH_ROOT/s21/log_lib_e2e"
if env FFT_WRAPPER_ALIAS_CFLAGS=-fstrict-aliasing make -e BACKEND=ne10 OBJ_ROOT="$S21_OBJ_ROOT" BIN_ROOT="$S21_BIN_ROOT" lib >"$S21_LIB_LOG" 2>&1; then
  fail "S21: env FFT_WRAPPER_ALIAS_CFLAGS=-fstrict-aliasing make -e BACKEND=ne10 lib unexpectedly SUCCEEDED (the exact regression reproducer must now fail)"
else
  if grep -q "cannot be overridden" "$S21_LIB_LOG"; then
    pass "S21: the exact reported repro (env FFT_WRAPPER_ALIAS_CFLAGS=-fstrict-aliasing make -e BACKEND=ne10 lib) correctly FAILS, mentioning 'cannot be overridden'"
  else
    fail "S21: the exact reported repro failed but WITHOUT the expected 'cannot be overridden' message"
    cat "$S21_LIB_LOG" >&2
  fi
fi

echo "############################################################"
echo "# S22: make -e environment-override rejection audit for the other"
echo "# five names"
echo "############################################################"
# The other names in the same foreach (CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/FP_POLICY) are audited for the
# \`make -e\` gap. CFLAGS/CXXFLAGS/LDFLAGS are safe: each has an unconditional \`+=\` before the
# foreach, so -e's origin flip has already happened (S9 covers their command-line rejection).
#   - FP_POLICY: its bare \`:=\` is defined above the foreach, so \`env FP_POLICY=x make -e ...\` is
#     rejected instead of silently losing -ffp-contract=off.
#   - CPPFLAGS: it uses \`+=\` (an unconditional assignment attempt, like CFLAGS). A \`?=\` only
#     assigns while the origin is still "undefined", so for an environment value it is a no-op, the
#     origin flip never fires under -e, and the rejection would miss it.
# Both cases run through \`env NAME=value make -e ...\`; CPPFLAGS/FP_POLICY are not consumed by any
# compile recipe, so print-obj-dir (not a real \`lib\` build) is the right-sized check, as in S9.
cd "$AC_DIR"
mkscratch s22
S22_OBJ_ROOT="$SCRATCH_ROOT/s22/obj"
S22_BIN_ROOT="$SCRATCH_ROOT/s22/bin"

for pair in "FP_POLICY=-ffp-contract=fast" "FP_POLICY=" "CPPFLAGS=-DS22_POISON" "CPPFLAGS="; do
  ov_var="${pair%%=*}"
  ov_val="${pair#*=}"
  S22_LOG="$SCRATCH_ROOT/s22/log_${ov_var}_$(printf '%s' "$ov_val" | tr -c 'A-Za-z0-9' '_')"
  if env "$ov_var=$ov_val" make -e OBJ_ROOT="$S22_OBJ_ROOT" BIN_ROOT="$S22_BIN_ROOT" print-obj-dir >"$S22_LOG" 2>&1; then
    fail "S22: env $ov_var=$ov_val make -e print-obj-dir unexpectedly SUCCEEDED (must be rejected at parse time)"
  else
    if grep -q "cannot be overridden" "$S22_LOG"; then
      pass "S22: env $ov_var=$ov_val make -e print-obj-dir correctly FAILS, mentioning 'cannot be overridden'"
    else
      fail "S22: env $ov_var=$ov_val make -e print-obj-dir failed but WITHOUT the expected 'cannot be overridden' message"
      cat "$S22_LOG" >&2
    fi
  fi
done

echo "############################################################"
echo "# S23: FP policy conflict detection widened to CXXFLAGS/CPPFLAGS"
echo "############################################################"
# The FP-policy conflict check rejects -Ofast, -ffast-math and -ffp-contract=<anything> (each would
# re-enable the FP contraction this repo pins off). It must cover CXXFLAGS and CPPFLAGS as well as
# CFLAGS: a plain environment CXXFLAGS or CPPFLAGS is allowed to fold in (S9/S22) and never passes
# through EXTRA_CFLAGS, so `env CXXFLAGS=-Ofast make lib` would otherwise build the one C++ TU
# (NE10_fft_generic_int32.cpp, see NE10_CXXSRCS in the Makefile) with -Ofast ahead of the trailing
# -ffp-contract=off.
#
# Each case is a plain environment override (`env NAME=value make ...`, no `-e`, no command-line
# assignment) against the real `lib` target, with a scratch OBJ_ROOT/BIN_ROOT so a correctly rejected
# build never touches the real obj/bin. Three CXXFLAGS cases cover the three conflict patterns; a
# fourth is the CPPFLAGS sibling of the -Ofast case.
cd "$AC_DIR"
mkscratch s23
S23_OBJ_ROOT="$SCRATCH_ROOT/s23/obj"
S23_BIN_ROOT="$SCRATCH_ROOT/s23/bin"

for pair in "CXXFLAGS=-Ofast" "CXXFLAGS=-ffast-math" "CXXFLAGS=-ffp-contract=fast" "CPPFLAGS=-Ofast"; do
  ov_var="${pair%%=*}"
  ov_val="${pair#*=}"
  S23_LOG="$SCRATCH_ROOT/s23/log_${ov_var}_$(printf '%s' "$ov_val" | tr -c 'A-Za-z0-9' '_')"
  if env "$ov_var=$ov_val" make BACKEND=ne10 OBJ_ROOT="$S23_OBJ_ROOT" BIN_ROOT="$S23_BIN_ROOT" lib >"$S23_LOG" 2>&1; then
    fail "S23: env $ov_var=$ov_val make BACKEND=ne10 lib unexpectedly SUCCEEDED (must be rejected as an FP policy conflict)"
  else
    if grep -q "FP policy conflict" "$S23_LOG"; then
      pass "S23: env $ov_var=$ov_val make BACKEND=ne10 lib correctly FAILS, mentioning 'FP policy conflict'"
    else
      fail "S23: env $ov_var=$ov_val make BACKEND=ne10 lib failed but WITHOUT the expected 'FP policy conflict' message"
      cat "$S23_LOG" >&2
    fi
  fi
done

# Positive control, mirroring S9's EXTRA_CFLAGS control: a legitimate plain-environment CXXFLAGS value
# matching none of the three conflict patterns must still fold in normally (only the conflicting
# patterns are rejected, not CXXFLAGS overrides in general) and, being folded into CFG_SIG, must land
# in a different keyed dir than the plain baseline.
baseline_ne10_objdir="$(make -s BACKEND=ne10 OBJ_ROOT="$S23_OBJ_ROOT" BIN_ROOT="$S23_BIN_ROOT" print-obj-dir)"
wextra_ne10_objdir="$(env CXXFLAGS=-Wextra make -s BACKEND=ne10 OBJ_ROOT="$S23_OBJ_ROOT" BIN_ROOT="$S23_BIN_ROOT" print-obj-dir)"
[ -n "$wextra_ne10_objdir" ] && [ "$wextra_ne10_objdir" != "$baseline_ne10_objdir" ] && \
  pass "S23: env CXXFLAGS=-Wextra make BACKEND=ne10 print-obj-dir (non-conflicting) succeeds and lands in a different keyed dir than the plain baseline" \
  || fail "S23: env CXXFLAGS=-Wextra make BACKEND=ne10 print-obj-dir did NOT differ from the plain baseline ($baseline_ne10_objdir vs $wextra_ne10_objdir)"

S23_WEXTRA_LIB_LOG="$SCRATCH_ROOT/s23/log_wextra_lib_e2e"
if env CXXFLAGS=-Wextra make BACKEND=ne10 OBJ_ROOT="$S23_OBJ_ROOT" BIN_ROOT="$S23_BIN_ROOT" lib >"$S23_WEXTRA_LIB_LOG" 2>&1; then
  pass "S23: env CXXFLAGS=-Wextra make BACKEND=ne10 lib (non-conflicting, real e2e build) succeeds"
else
  fail "S23: env CXXFLAGS=-Wextra make BACKEND=ne10 lib (non-conflicting) unexpectedly FAILED"
  cat "$S23_WEXTRA_LIB_LOG" >&2
fi

# Whole-word matching: the conflict check uses $(filter), which splits its text into words and matches
# each whole word, so a token that merely contains a pattern as a substring is not a conflict:
#   env CXXFLAGS='-DROUND9_NOTE=-Ofastness' make -s BACKEND=ne10 print-obj-dir
# `-DROUND9_NOTE=-Ofastness` is a single harmless preprocessor macro token, not the compiler flag
# -Ofast, and must be accepted.
S23_ROUND9_NOTE_LOG="$SCRATCH_ROOT/s23/log_round9_note"
if env CXXFLAGS='-DROUND9_NOTE=-Ofastness' make -s BACKEND=ne10 OBJ_ROOT="$S23_OBJ_ROOT" BIN_ROOT="$S23_BIN_ROOT" print-obj-dir >"$S23_ROUND9_NOTE_LOG" 2>&1; then
  pass "S23: env CXXFLAGS='-DROUND9_NOTE=-Ofastness' make BACKEND=ne10 print-obj-dir (substring-only, non-conflicting) succeeds (regression false-positive fix)"
else
  fail "S23: env CXXFLAGS='-DROUND9_NOTE=-Ofastness' make BACKEND=ne10 print-obj-dir unexpectedly FAILED (regression substring false-positive regressed)"
  cat "$S23_ROUND9_NOTE_LOG" >&2
fi

echo "############################################################"
echo "# S23b: quote/escape/response-file bypass of the FP-policy"
echo "# conflict gate itself"
echo "############################################################"
# S23b: the exact-token $(filter) check matches GNU Make's own text of CFLAGS/CXXFLAGS/CPPFLAGS, but
# Make has no notion of shell quoting: a value the shell unquotes at compile-recipe time (e.g.
# EXTRA_CFLAGS="'-Ofast'", stored by Make as the 9-character text '-Ofast', quotes included) never
# equals the bare word -Ofast, yet the compiler gets a live -Ofast. Each case below is a plain
# environment EXTRA_CFLAGS override (no `-e`, no command-line assignment) against a cheap
# introspection target, under a scratch OBJ_ROOT/BIN_ROOT so a correctly rejected build never touches
# the real obj/bin. Each rejection must mention "FP policy conflict" (the marker every FP-policy
# failure emits) AND the specific forbidden construct named in the block's own $(error ...), so an
# unrelated failure is not mistaken for the check working.
cd "$AC_DIR"
mkscratch s23b
S23B_OBJ_ROOT="$SCRATCH_ROOT/s23b/obj"
S23B_BIN_ROOT="$SCRATCH_ROOT/s23b/bin"

S23B_LOG_1="$SCRATCH_ROOT/s23b/log_single_quoted"
if env EXTRA_CFLAGS="'-Ofast'" make BACKEND=ne10 OBJ_ROOT="$S23B_OBJ_ROOT" BIN_ROOT="$S23B_BIN_ROOT" print-obj-dir >"$S23B_LOG_1" 2>&1; then
  fail "S23b: env EXTRA_CFLAGS=\"'-Ofast'\" (single-quoted) print-obj-dir unexpectedly SUCCEEDED (must be rejected)"
else
  if grep -q "FP policy conflict" "$S23B_LOG_1" && grep -q "single-quote" "$S23B_LOG_1"; then
    pass "S23b: env EXTRA_CFLAGS=\"'-Ofast'\" (single-quoted) print-obj-dir correctly FAILS, identifying the single-quote character"
  else
    fail "S23b: env EXTRA_CFLAGS=\"'-Ofast'\" (single-quoted) print-obj-dir failed but did NOT identify the single-quote character specifically"
    cat "$S23B_LOG_1" >&2
  fi
fi

S23B_LOG_2="$SCRATCH_ROOT/s23b/log_double_quoted"
if env EXTRA_CFLAGS='"-ffast-math"' make BACKEND=ne10 OBJ_ROOT="$S23B_OBJ_ROOT" BIN_ROOT="$S23B_BIN_ROOT" print-obj-dir >"$S23B_LOG_2" 2>&1; then
  fail "S23b: env EXTRA_CFLAGS='\"-ffast-math\"' (double-quoted) print-obj-dir unexpectedly SUCCEEDED (must be rejected)"
else
  # The double-quote character is caught by the character allow-list (S24), not a dedicated deny-list
  # entry: the message names the offending character generically ("outside the allowed set ... found:
  # \"...\"").
  if grep -q "FP policy conflict" "$S23B_LOG_2" && grep -q "outside the allowed set" "$S23B_LOG_2" && grep -qF 'found: """' "$S23B_LOG_2"; then
    pass "S23b: env EXTRA_CFLAGS='\"-ffast-math\"' (double-quoted) print-obj-dir correctly FAILS, identifying the double-quote character"
  else
    fail "S23b: env EXTRA_CFLAGS='\"-ffast-math\"' (double-quoted) print-obj-dir failed but did NOT identify the double-quote character specifically"
    cat "$S23B_LOG_2" >&2
  fi
fi

S23B_LOG_3="$SCRATCH_ROOT/s23b/log_quote_split"
if env EXTRA_CFLAGS="-O'f'ast" make BACKEND=ne10 OBJ_ROOT="$S23B_OBJ_ROOT" BIN_ROOT="$S23B_BIN_ROOT" print-obj-dir >"$S23B_LOG_3" 2>&1; then
  fail "S23b: env EXTRA_CFLAGS=\"-O'f'ast\" (quote-split mid-token) print-obj-dir unexpectedly SUCCEEDED (must be rejected)"
else
  # The single quote is caught by the one pure-Make $(findstring) check that runs before the allow-list
  # (see the Makefile), so this message says "single-quote character" verbatim, unlike the
  # double-quote and response-file cases.
  if grep -q "FP policy conflict" "$S23B_LOG_3" && grep -q "single-quote" "$S23B_LOG_3"; then
    pass "S23b: env EXTRA_CFLAGS=\"-O'f'ast\" (quote-split mid-token) print-obj-dir correctly FAILS, identifying the single-quote character"
  else
    fail "S23b: env EXTRA_CFLAGS=\"-O'f'ast\" (quote-split mid-token) print-obj-dir failed but did NOT identify the single-quote character specifically"
    cat "$S23B_LOG_3" >&2
  fi
fi

S23B_LOG_4="$SCRATCH_ROOT/s23b/log_response_file"
if env EXTRA_CFLAGS='@flags.rsp' make BACKEND=ne10 OBJ_ROOT="$S23B_OBJ_ROOT" BIN_ROOT="$S23B_BIN_ROOT" print-obj-dir >"$S23B_LOG_4" 2>&1; then
  fail "S23b: env EXTRA_CFLAGS='@flags.rsp' (response-file) print-obj-dir unexpectedly SUCCEEDED (must be rejected)"
else
  # '@' is caught by the character allow-list (it is not in the allowed set), not a dedicated
  # response-file deny-list entry.
  if grep -q "FP policy conflict" "$S23B_LOG_4" && grep -q "outside the allowed set" "$S23B_LOG_4" && grep -qF 'found: "@"' "$S23B_LOG_4"; then
    pass "S23b: env EXTRA_CFLAGS='@flags.rsp' (response-file) print-obj-dir correctly FAILS, identifying the @-prefixed response-file token"
  else
    fail "S23b: env EXTRA_CFLAGS='@flags.rsp' (response-file) print-obj-dir failed but did NOT identify the response-file token specifically"
    cat "$S23B_LOG_4" >&2
  fi
fi

# Positive controls: harmless unquoted tokens (-DROUND9_NOTE=-Ofastness, -DTEXT=ffast-math) containing
# no forbidden character must pass: only the exact-token $(filter) check decides whether a token IS
# -Ofast/-ffast-math/-ffp-contract=<x>, and neither is. Both go through EXTRA_CFLAGS, the hook the
# quoting bypass used.
S23B_LOG_P1="$SCRATCH_ROOT/s23b/log_positive_round9_note"
if env EXTRA_CFLAGS=-DROUND9_NOTE=-Ofastness make -s BACKEND=ne10 OBJ_ROOT="$S23B_OBJ_ROOT" BIN_ROOT="$S23B_BIN_ROOT" print-obj-dir >"$S23B_LOG_P1" 2>&1; then
  pass "S23b: env EXTRA_CFLAGS=-DROUND9_NOTE=-Ofastness print-obj-dir (positive control, no forbidden characters) succeeds"
else
  fail "S23b: env EXTRA_CFLAGS=-DROUND9_NOTE=-Ofastness print-obj-dir unexpectedly FAILED (quote/escape gate false-positived on a harmless token)"
  cat "$S23B_LOG_P1" >&2
fi

S23B_LOG_P2="$SCRATCH_ROOT/s23b/log_positive_dtext"
if env EXTRA_CFLAGS=-DTEXT=ffast-math make -s BACKEND=ne10 OBJ_ROOT="$S23B_OBJ_ROOT" BIN_ROOT="$S23B_BIN_ROOT" print-obj-dir >"$S23B_LOG_P2" 2>&1; then
  pass "S23b: env EXTRA_CFLAGS=-DTEXT=ffast-math print-obj-dir (positive control, no forbidden characters) succeeds"
else
  fail "S23b: env EXTRA_CFLAGS=-DTEXT=ffast-math print-obj-dir unexpectedly FAILED (quote/escape gate false-positived on a harmless token)"
  cat "$S23B_LOG_P2" >&2
fi

echo "############################################################"
echo "# S24: FP-policy allow-list -- bypass shapes the OLD 9-item"
echo "# deny-list never caught, plus the exact-token filter's own"
echo "# distinct code path, plus the FP-policy check's OWN internal"
echo "# variables"
echo "############################################################"
# Every negative case below is a glob, tilde, redirection or process-substitution shape that the
# character-class allow-list must reject. Same discipline as S20-S23b: scratch OBJ_ROOT/BIN_ROOT so a
# correctly rejected build never touches the real obj/bin trees.
cd "$AC_DIR"
mkscratch s24
S24_OBJ_ROOT="$SCRATCH_ROOT/s24/obj"
S24_BIN_ROOT="$SCRATCH_ROOT/s24/bin"

# helper: run one EXTRA_CFLAGS negative case against the allow-list and
# assert both that it FAILS and that the specific character(s) named in
# $2 appear in the reported disallowed-character set.
s24_check_rejected() {
  local label="$1" flag_value="$2" expect_chars="$3" log
  log="$SCRATCH_ROOT/s24/log_$(printf '%s' "$label" | tr -c 'A-Za-z0-9' '_')"
  if env EXTRA_CFLAGS="$flag_value" make BACKEND=ne10 OBJ_ROOT="$S24_OBJ_ROOT" BIN_ROOT="$S24_BIN_ROOT" print-obj-dir >"$log" 2>&1; then
    fail "S24: EXTRA_CFLAGS='$flag_value' ($label) print-obj-dir unexpectedly SUCCEEDED (must be rejected)"
  elif grep -q "FP policy conflict" "$log" && grep -q "outside the allowed set" "$log" && grep -qF "found: \"$expect_chars\"" "$log"; then
    pass "S24: EXTRA_CFLAGS='$flag_value' ($label) print-obj-dir correctly FAILS, identifying \"$expect_chars\""
  else
    fail "S24: EXTRA_CFLAGS='$flag_value' ($label) print-obj-dir failed but did NOT identify \"$expect_chars\" specifically"
    cat "$log" >&2
  fi
}

s24_check_rejected "glob-star"        '-O*t'          '*'
s24_check_rejected "glob-question"    '-Ofas?'        '?'
s24_check_rejected "glob-brackets"    '-Ofas[t]'      '[]'
s24_check_rejected "tilde"            '~/pwned'       '~'
s24_check_rejected "redirect-out"     '-I>/tmp/evil'  '>'
s24_check_rejected "redirect-in"      '-I</etc/passwd' '<'
s24_check_rejected "process-subst"    '<(echo hi)'    '()<'

# Positive control on a distinct code path: a Make-native ${VAR} expansion must be rejected by the
# exact-token $(filter) check, not the character allow-list. GNU Make treats ${...} like $(...) and
# resolves it to R11_FLAG's value (-Ofast) before either FP-policy check runs, so FP_INPUT_FLAGS holds
# the all-allowed-characters token "-Ofast": the allow-list has nothing to catch and only the
# FP_CONFLICT_FLAGS $(filter) check can reject it. Literal repro:
# `env R11_FLAG=-Ofast make -n EXTRA_CFLAGS=${R11_FLAG} lib`, adapted so Make, not the invoking shell,
# resolves ${...}: R11_FLAG is `export`-ed first (under `set -u` an unset ${VAR} in the same command
# line would abort the suite), and 'EXTRA_CFLAGS=${R11_FLAG}' is single-quoted so the shell leaves the
# literal text for GNU Make, which auto-imports exported variables. -n suffices: the FP-policy checks
# run at Makefile parse time, before any target executes.
export R11_FLAG=-Ofast
S24_LOG_R11="$SCRATCH_ROOT/s24/log_r11_dollar_brace"
if make -n BACKEND=ne10 OBJ_ROOT="$S24_OBJ_ROOT" BIN_ROOT="$S24_BIN_ROOT" 'EXTRA_CFLAGS=${R11_FLAG}' lib >"$S24_LOG_R11" 2>&1; then
  fail "S24: R11_FLAG=-Ofast (exported) make -n EXTRA_CFLAGS=\${R11_FLAG} lib unexpectedly SUCCEEDED (must be rejected via the exact-token filter)"
else
  if grep -q "FP policy conflict" "$S24_LOG_R11" && grep -q -- "-Ofast" "$S24_LOG_R11" && ! grep -q "outside the allowed set" "$S24_LOG_R11"; then
    pass "S24: R11_FLAG=-Ofast (exported) make -n EXTRA_CFLAGS=\${R11_FLAG} lib correctly FAILS via the exact-token \$(filter) check (not the character allow-list)"
  else
    fail "S24: R11_FLAG=-Ofast (exported) make -n EXTRA_CFLAGS=\${R11_FLAG} lib failed but not via the expected exact-token filter path"
    cat "$S24_LOG_R11" >&2
  fi
fi
unset R11_FLAG

# The FP-policy check's own internal variables (FP_INPUT_FLAGS, SHELL_SAFE_ALLOWLIST_RC,
# FP_CONFLICT_FLAGS, FP_ALLOWED_CHARS_RE) must not be overridable from the command line or under `-e`:
# an override would silently defeat every check above while the real CFLAGS/CXXFLAGS/CPPFLAGS still
# carry the dangerous content. Each name must be rejected both as a plain command-line override and
# under `-e`. `payload` is the EXTRA_CFLAGS value the overridden check would otherwise catch:
# FP_INPUT_FLAGS/SHELL_SAFE_ALLOWLIST_RC/FP_ALLOWED_CHARS_RE gate the character allow-list, so a
# semicolon payload exercises that path; FP_CONFLICT_FLAGS gates the separate exact-token
# -Ofast/-ffast-math/-ffp-contract= filter, so it needs an all-allowed-characters payload that is
# still a real conflict ("-Ofast"), because a semicolon payload would be caught first by the untouched
# allow-list and never exercise the FP_CONFLICT_FLAGS override. (S25 repeats the
# SHELL_SAFE_INPUT_FLAGS/SHELL_SAFE_ALLOWLIST_RC pair via an EXTRA_LDFLAGS payload.)
s24_check_override_rejected() {
  local varname="$1" varvalue="$2" payload="$3" use_dash_e="$4" log
  log="$SCRATCH_ROOT/s24/log_override_$(printf '%s' "$varname" | tr -c 'A-Za-z0-9' '_')_$use_dash_e"
  if [ "$use_dash_e" = "dashe" ]; then
    if env "$varname=$varvalue" make -e BACKEND=ne10 OBJ_ROOT="$S24_OBJ_ROOT" BIN_ROOT="$S24_BIN_ROOT" EXTRA_CFLAGS="$payload" print-obj-dir >"$log" 2>&1; then
      fail "S24: env $varname=$varvalue make -e print-obj-dir unexpectedly SUCCEEDED (must be rejected: overriding $varname must not defeat the FP-policy checks)"
    elif grep -q "cannot be overridden" "$log" && grep -q "$varname" "$log"; then
      pass "S24: env $varname=$varvalue make -e print-obj-dir correctly FAILS, mentioning '$varname cannot be overridden'"
    else
      fail "S24: env $varname=$varvalue make -e print-obj-dir failed but did NOT mention '$varname cannot be overridden'"
      cat "$log" >&2
    fi
  else
    if make BACKEND=ne10 OBJ_ROOT="$S24_OBJ_ROOT" BIN_ROOT="$S24_BIN_ROOT" EXTRA_CFLAGS="$payload" "$varname=$varvalue" print-obj-dir >"$log" 2>&1; then
      fail "S24: make $varname=$varvalue print-obj-dir unexpectedly SUCCEEDED (must be rejected: overriding $varname must not defeat the FP-policy checks)"
    elif grep -q "cannot be overridden" "$log" && grep -q "$varname" "$log"; then
      pass "S24: make $varname=$varvalue print-obj-dir correctly FAILS, mentioning '$varname cannot be overridden'"
    else
      fail "S24: make $varname=$varvalue print-obj-dir failed but did NOT mention '$varname cannot be overridden'"
      cat "$log" >&2
    fi
  fi
}
s24_check_override_rejected "FP_INPUT_FLAGS"         "clean" '-O2;rm' cmdline
s24_check_override_rejected "SHELL_SAFE_ALLOWLIST_RC" "0"     '-O2;rm' cmdline
s24_check_override_rejected "FP_ALLOWED_CHARS_RE"    ".*"    '-O2;rm' cmdline
s24_check_override_rejected "FP_CONFLICT_FLAGS"      ""      '-Ofast' cmdline
s24_check_override_rejected "FP_INPUT_FLAGS"         "clean" '-O2;rm' dashe
s24_check_override_rejected "SHELL_SAFE_ALLOWLIST_RC" "0"     '-O2;rm' dashe
s24_check_override_rejected "FP_ALLOWED_CHARS_RE"    ".*"    '-O2;rm' dashe
s24_check_override_rejected "FP_CONFLICT_FLAGS"      ""      '-Ofast' dashe

echo "############################################################"
echo "# S25: link-flags (LDFLAGS/EXTRA_LDFLAGS) character-safety coverage"
echo "############################################################"
# LDFLAGS/EXTRA_LDFLAGS get the same character-class allow-list as CFLAGS: LDFLAGS is embedded in every
# link recipe ($(LINK) ... $(LDFLAGS)) just as CFLAGS is in every compile recipe. Same discipline as
# S20-S24: scratch OBJ_ROOT/BIN_ROOT so a correctly rejected build never touches the real obj/bin.
cd "$AC_DIR"
mkscratch s25
S25_OBJ_ROOT="$SCRATCH_ROOT/s25/obj"
S25_BIN_ROOT="$SCRATCH_ROOT/s25/bin"

# helper: run one EXTRA_LDFLAGS negative case against the allow-list and assert both that it FAILS and
# that the specific character(s) named in $2 appear in the reported disallowed-character set. Mirrors
# S24's s24_check_rejected, aimed at EXTRA_LDFLAGS.
s25_check_rejected() {
  local label="$1" flag_value="$2" expect_chars="$3" log
  log="$SCRATCH_ROOT/s25/log_$(printf '%s' "$label" | tr -c 'A-Za-z0-9' '_')"
  if env EXTRA_LDFLAGS="$flag_value" make BACKEND=ne10 OBJ_ROOT="$S25_OBJ_ROOT" BIN_ROOT="$S25_BIN_ROOT" print-obj-dir >"$log" 2>&1; then
    fail "S25: EXTRA_LDFLAGS='$flag_value' ($label) print-obj-dir unexpectedly SUCCEEDED (must be rejected)"
  elif grep -q "FP policy conflict" "$log" && grep -q "outside the allowed set" "$log" && grep -qF "found: \"$expect_chars\"" "$log"; then
    pass "S25: EXTRA_LDFLAGS='$flag_value' ($label) print-obj-dir correctly FAILS, identifying \"$expect_chars\""
  else
    fail "S25: EXTRA_LDFLAGS='$flag_value' ($label) print-obj-dir failed but did NOT identify \"$expect_chars\" specifically"
    cat "$log" >&2
  fi
}

# Semicolon payload via EXTRA_LDFLAGS: `make -n selftest EXTRA_LDFLAGS=';echo LINK_FLAG_INJECTION'`
# must not carry the payload verbatim after `-lm` onto the `cc -o ... $(LDFLAGS)` link line; it is
# rejected.
s25_check_rejected "injection-semicolon" ';echo LINK_FLAG_INJECTION' ';'

# Same glob/tilde/redirect/process-substitution matrix S24 runs for EXTRA_CFLAGS, repeated for
# EXTRA_LDFLAGS.
s25_check_rejected "glob-star"        '-L*t'          '*'
s25_check_rejected "glob-question"    '-Lfas?'        '?'
s25_check_rejected "glob-brackets"    '-Lfas[t]'      '[]'
s25_check_rejected "tilde"            '~/pwned'       '~'
s25_check_rejected "redirect-out"     '-lm>/tmp/evil'  '>'
s25_check_rejected "redirect-in"      '-lm</etc/passwd' '<'
s25_check_rejected "process-subst"    '<(echo hi)'    '()<'

# Positive control: a real link flag a consumer may need, in the shape the allow-list admits
# (-Wl,-rpath,dir, as this Makefile's ZERO_HEAP_LDFLAGS uses), must pass the character check AND
# actually link and run. `@loader_path` (ZERO_HEAP_LDFLAGS's macOS rpath literal) is not reused: it is
# appended directly in two target-specific recipes, never folded through LDFLAGS/EXTRA_LDFLAGS, and its
# `@` is outside the allowed set. A plain filesystem rpath path is the realistic use.
#
# "Runs selftest to completion" is judged against a reference from the same run: selftest is first run
# with no EXTRA_LDFLAGS, and the run with the link flag must report the same TOTAL CHECKS lines
# (simd_selftest prints one per pass, so the whole sorted set is compared) and the same number of ALL
# PASS lines. A link flag changes how the binaries link, never which checks run, and a pinned count
# goes stale whenever a selftest item is added or removed. Both runs use the scratch roots
# (EXTRA_LDFLAGS is in CFG_SIG, so they get separate keyed directories) and the real obj/bin see
# neither.
S25_REF_LOG="$SCRATCH_ROOT/s25/log_reference_selftest"
S25_POS_LOG="$SCRATCH_ROOT/s25/log_positive_rpath"
# Sorted TOTAL CHECKS lines of one log, newline-folded; empty when there is none.
s25_total_checks() { { grep -E '^TOTAL CHECKS: [1-9][0-9]*$' "$1" || true; } | LC_ALL=C sort | tr '\n' ' '; }
s25_all_pass_count() { grep -c '^ALL PASS' "$1" || true; }
if ! env -u EXTRA_LDFLAGS make BACKEND=ne10 OBJ_ROOT="$S25_OBJ_ROOT" BIN_ROOT="$S25_BIN_ROOT" selftest >"$S25_REF_LOG" 2>&1; then
  fail "S25: reference selftest (no EXTRA_LDFLAGS) unexpectedly FAILED -- cannot judge the positive control"
  cat "$S25_REF_LOG" >&2
elif env EXTRA_LDFLAGS='-Wl,-rpath,/usr/lib -L/usr/lib' make BACKEND=ne10 OBJ_ROOT="$S25_OBJ_ROOT" BIN_ROOT="$S25_BIN_ROOT" selftest >"$S25_POS_LOG" 2>&1; then
  s25_ref_checks="$(s25_total_checks "$S25_REF_LOG")"
  s25_pos_checks="$(s25_total_checks "$S25_POS_LOG")"
  s25_ref_passn="$(s25_all_pass_count "$S25_REF_LOG")"
  s25_pos_passn="$(s25_all_pass_count "$S25_POS_LOG")"
  if [ -n "$s25_ref_checks" ] && [ "$s25_ref_checks" = "$s25_pos_checks" ] \
     && [ "$s25_ref_passn" -ge 1 ] && [ "$s25_ref_passn" = "$s25_pos_passn" ]; then
    # The counts go in an INFO line, not the PASS text, so the PASS message stays stable
    # when a selftest item is added or removed.
    echo "  INFO: S25: reference and link-flag selftest runs both report ${s25_ref_checks}with $s25_ref_passn ALL PASS line(s)"
    pass "S25: EXTRA_LDFLAGS='-Wl,-rpath,/usr/lib -L/usr/lib' (positive control, real rpath/library-path link flag) builds AND runs selftest to completion with the same TOTAL CHECKS and ALL PASS count as the reference run"
  else
    fail "S25: EXTRA_LDFLAGS='-Wl,-rpath,/usr/lib -L/usr/lib' selftest succeeded but its TOTAL CHECKS / ALL PASS differ from the same-run reference (reference: '$s25_ref_checks' x$s25_ref_passn ALL PASS; with flag: '$s25_pos_checks' x$s25_pos_passn ALL PASS)"
    cat "$S25_POS_LOG" >&2
  fi
  # Every test binary's own result lines, in order -- not only the ones that print TOTAL CHECKS
  # / ALL PASS (roundtrip, test_stft*, test_stft_guard and test_audio_utilities print neither).
  s25_results() { grep -E '^(--- .* ---$|PASS |>>> PASS|ALL PASS|TOTAL CHECKS: |test_stft(_guard)? PASS|GUARD\| |round-trip: |static vs heap: |fft_magnitude vs )' "$1" || true; }
  if [ -n "$(s25_results "$S25_REF_LOG")" ] && [ "$(s25_results "$S25_REF_LOG")" = "$(s25_results "$S25_POS_LOG")" ]; then
    pass "S25: the link-flag selftest run prints exactly the reference run's per-test result lines (no test binary skipped)"
  else
    fail "S25: the link-flag selftest run's per-test result lines differ from the reference run's"
    diff <(s25_results "$S25_REF_LOG") <(s25_results "$S25_POS_LOG") >&2 || true
  fi
  # ...and the flag really reached the link lines: every echoed link into the flag config's bin
  # dir carries it (a flag that is validated but silently dropped would pass everything above).
  s25_pos_bindir="$(env EXTRA_LDFLAGS='-Wl,-rpath,/usr/lib -L/usr/lib' make -s BACKEND=ne10 OBJ_ROOT="$S25_OBJ_ROOT" BIN_ROOT="$S25_BIN_ROOT" print-bin-dir)"
  s25_links="$(grep -F -- "-o $s25_pos_bindir/" "$S25_POS_LOG" | grep -v -- ' -c ' || true)"
  s25_n_links="$(printf '%s' "$s25_links" | grep -c . || true)"
  s25_n_flag="$(printf '%s' "$s25_links" | grep -cF -- '-Wl,-rpath,/usr/lib -L/usr/lib' || true)"
  [ "$s25_n_links" -ge 1 ] && [ "$s25_n_flag" = "$s25_n_links" ] && \
    pass "S25: every echoed selftest link line carries the EXTRA_LDFLAGS link flag" \
    || fail "S25: EXTRA_LDFLAGS did not reach the selftest link lines ($s25_n_flag of $s25_n_links echoed link lines carry it)"
else
  fail "S25: EXTRA_LDFLAGS='-Wl,-rpath,/usr/lib -L/usr/lib' (positive control, legitimate link flag) unexpectedly FAILED"
  cat "$S25_POS_LOG" >&2
fi

# Command-line/`-e` override rejection for SHELL_SAFE_INPUT_FLAGS and SHELL_SAFE_ALLOWLIST_RC: S24
# covers both via an EXTRA_CFLAGS payload; this repeats it via EXTRA_LDFLAGS, the vector S25 exists to
# close (an override must not let an EXTRA_LDFLAGS injection payload back through).
s25_check_override_rejected() {
  local varname="$1" varvalue="$2" payload="$3" use_dash_e="$4" log
  log="$SCRATCH_ROOT/s25/log_override_$(printf '%s' "$varname" | tr -c 'A-Za-z0-9' '_')_$use_dash_e"
  if [ "$use_dash_e" = "dashe" ]; then
    if env "$varname=$varvalue" make -e BACKEND=ne10 OBJ_ROOT="$S25_OBJ_ROOT" BIN_ROOT="$S25_BIN_ROOT" EXTRA_LDFLAGS="$payload" print-obj-dir >"$log" 2>&1; then
      fail "S25: env $varname=$varvalue make -e print-obj-dir (EXTRA_LDFLAGS payload) unexpectedly SUCCEEDED (must be rejected: overriding $varname must not defeat the link-flags character-safety check)"
    elif grep -q "cannot be overridden" "$log" && grep -q "$varname" "$log"; then
      pass "S25: env $varname=$varvalue make -e print-obj-dir (EXTRA_LDFLAGS payload) correctly FAILS, mentioning '$varname cannot be overridden'"
    else
      fail "S25: env $varname=$varvalue make -e print-obj-dir (EXTRA_LDFLAGS payload) failed but did NOT mention '$varname cannot be overridden'"
      cat "$log" >&2
    fi
  else
    if make BACKEND=ne10 OBJ_ROOT="$S25_OBJ_ROOT" BIN_ROOT="$S25_BIN_ROOT" EXTRA_LDFLAGS="$payload" "$varname=$varvalue" print-obj-dir >"$log" 2>&1; then
      fail "S25: make $varname=$varvalue print-obj-dir (EXTRA_LDFLAGS payload) unexpectedly SUCCEEDED (must be rejected: overriding $varname must not defeat the link-flags character-safety check)"
    elif grep -q "cannot be overridden" "$log" && grep -q "$varname" "$log"; then
      pass "S25: make $varname=$varvalue print-obj-dir (EXTRA_LDFLAGS payload) correctly FAILS, mentioning '$varname cannot be overridden'"
    else
      fail "S25: make $varname=$varvalue print-obj-dir (EXTRA_LDFLAGS payload) failed but did NOT mention '$varname cannot be overridden'"
      cat "$log" >&2
    fi
  fi
}
s25_check_override_rejected "SHELL_SAFE_INPUT_FLAGS"  "clean" ';rm' cmdline
s25_check_override_rejected "SHELL_SAFE_ALLOWLIST_RC" "0"     ';rm' cmdline
s25_check_override_rejected "SHELL_SAFE_INPUT_FLAGS"  "clean" ';rm' dashe
s25_check_override_rejected "SHELL_SAFE_ALLOWLIST_RC" "0"     ';rm' dashe

echo "############################################################"
echo "# S26: any BACKEND other than ne10 is rejected"
echo "############################################################"
# ne10 is the only FFT backend. A removed or unknown BACKEND (kiss, an empty value, a typo)
# must fail at parse time with a clear message instead of silently building something else,
# for build targets and for the print-* query targets alike, and must create no obj/bin dir.
cd "$AC_DIR"
mkscratch s26
S26_OBJ_ROOT="$SCRATCH_ROOT/s26/obj"
S26_BIN_ROOT="$SCRATCH_ROOT/s26/bin"
for be in kiss "" nope; do
  for tgt in lib print-lib-path; do
    S26_LOG="$SCRATCH_ROOT/s26/log_${be:-empty}_$tgt"
    rc=0
    make "BACKEND=$be" OBJ_ROOT="$S26_OBJ_ROOT" BIN_ROOT="$S26_BIN_ROOT" "$tgt" >"$S26_LOG" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ] && grep -q "BACKEND must be ne10" "$S26_LOG"; then
      pass "S26: make BACKEND='$be' $tgt is rejected (rc=$rc) with the 'BACKEND must be ne10' message"
    else
      fail "S26: make BACKEND='$be' $tgt was not rejected as expected (rc=$rc)"
      cat "$S26_LOG" >&2
    fi
  done
done
[ ! -e "$S26_OBJ_ROOT" ] && [ ! -e "$S26_BIN_ROOT" ] && \
  pass "S26: the rejected invocations created no obj/bin directory" \
  || fail "S26: a rejected BACKEND invocation left an obj/bin directory behind"

echo "############################################################"
echo "# Final integrity guards"
echo "############################################################"
AC_STATE_AFTER="$(tree_state_hash "$AC_DIR")"
[ "$AC_STATE_BEFORE" = "$AC_STATE_AFTER" ] && pass "integrity: \$AC_DIR tree-state (status+diff) unchanged across the whole run" \
  || fail "integrity: \$AC_DIR tree-state CHANGED during this run"

AEC_STATE_AFTER="$(tree_state_hash "$AEC_REPO_DIR")"
[ "$AEC_STATE_BEFORE" = "$AEC_STATE_AFTER" ] && pass "integrity: AEC repo (\$AC_DIR/../AEC) tree-state (status+diff) unchanged across the whole run" \
  || fail "integrity: AEC repo tree-state CHANGED during this run"

REAL_DIST_AFTER="$(real_dist_sentinel)"
[ "$REAL_DIST_BEFORE" = "$REAL_DIST_AFTER" ] && pass "integrity: real \$AC_DIR/dist sentinel unchanged (absent stays absent, or byte+mtime identical)" \
  || fail "integrity: real \$AC_DIR/dist sentinel CHANGED during this run"

# Final-guard re-check against the snapshot taken in S1 (right after this suite's last intentional
# real-tree build). Every file that existed then must still be sha+mtime identical; a directory
# created after the snapshot is allowed but reported (WARN/INFO), a new file inside an
# already-snapshotted directory is a hard FAIL.
FINAL_GUARD_DIRS_AFTER="$(real_tree_dirs)"
FINAL_GUARD_SNAPSHOT_AFTER="$(real_tree_snapshot)"

fg_lost="$(comm -23 <(printf '%s\n' "$FINAL_GUARD_SNAPSHOT_BEFORE" | LC_ALL=C sort) <(printf '%s\n' "$FINAL_GUARD_SNAPSHOT_AFTER" | LC_ALL=C sort))"
if [ -z "$fg_lost" ]; then
  pass "final-guard: every real obj/+bin/ file snapshotted after S1's builds is still sha+mtime identical"
else
  fail "final-guard: real obj/+bin/ files changed or vanished since S1's builds"
  printf '%s\n' "$fg_lost" | sed 's/^/    changed\/lost: /' >&2
fi

fg_new_dirs="$(comm -13 <(printf '%s\n' "$FINAL_GUARD_DIRS_BEFORE" | LC_ALL=C sort) <(printf '%s\n' "$FINAL_GUARD_DIRS_AFTER" | LC_ALL=C sort) | grep . || true)"
if [ -n "$fg_new_dirs" ]; then
  echo "  WARN: final-guard: new keyed director(ies) appeared under the real obj/+bin/ during this run (allowed, but unexpected after the regression S12 fix):"
  printf '%s\n' "$fg_new_dirs" | sed 's/^/    INFO: new dir /'
fi

fg_new_files="$(comm -13 <(printf '%s\n' "$FINAL_GUARD_SNAPSHOT_BEFORE" | LC_ALL=C sort) <(printf '%s\n' "$FINAL_GUARD_SNAPSHOT_AFTER" | LC_ALL=C sort) | grep . || true)"
fg_stray_ok=1
if [ -n "$fg_new_files" ]; then
  while IFS= read -r fg_ln; do
    [ -n "$fg_ln" ] || continue
    fg_path="${fg_ln%% *}"
    fg_top="$(printf '%s' "$fg_path" | cut -d/ -f1-2)"
    printf '%s\n' "$fg_new_dirs" | grep -qxF "$fg_top" || fg_stray_ok=0
  done <<EOF_FG
$fg_new_files
EOF_FG
fi
[ "$fg_stray_ok" -eq 1 ] && pass "final-guard: no stray new file inside any already-snapshotted real obj/+bin/ directory" \
  || fail "final-guard: a NEW file appeared inside an EXISTING (already-snapshotted) real obj/+bin/ directory"

echo "############################################################"
echo "SUMMARY: $PASS_COUNT passed, $FAIL_COUNT failed"
echo "############################################################"
if [ "$FAIL_COUNT" -gt 0 ]; then
  echo "Failures:" >&2
  for f in "${FAILURES[@]}"; do echo "  - $f" >&2; done
  exit 1
fi
exit 0
