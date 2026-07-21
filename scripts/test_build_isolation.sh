#!/usr/bin/env bash
# test_build_isolation.sh — round-3/4/5/6/7 review build-isolation regression
# suite.
#
# Exercises the CFG_SIG-keyed obj/bin directory design (audio_common's own
# Makefile, plus the two-phase AC_LIB consumer resolution in AEC/c_impl and
# NR/c_impl) against every failure mode the round-3 B01 finding described,
# the round-4 review's command-line override rejection / fresh-archive
# stale-member removal / publish v3 content-addressing / CC-CXX toolchain
# guard, the round-5 review's publish v4 redesign (throwaway DIST_ROOT=,
# lock-FIRST publish driver, deterministic MANIFEST/append-only ATTEST
# split, rename(2)-atomic `current` swap), the round-6 review:
#
#   - P1: OBJ_ROOT=/BIN_ROOT= placement knobs, so every throwaway/tamper
#     scenario in THIS script can drive a scratch-directory build of the
#     real worktree without ever touching the real obj/ or bin/.
#   - P2-1: this script's OWN temp management was itself a finding -- the
#     old new_tmpdir()/new_tmpfile() registered cleanup state by appending
#     to a variable FROM INSIDE a `$(...)` command substitution, i.e. inside
#     a subshell whose variable changes never make it back to the parent
#     shell that owns the EXIT trap. An interrupted run could therefore leak
#     scratch directories the trap never knew about. Fixed by switching to a
#     single SCRATCH_ROOT directory tree (one trap, nothing to register).
#   - P2-2: ATTEST is now one-event-one-file, installed via the helper's
#     --excl-install (write-temp + link(2), atomic no-clobber) with a
#     <NNN> retry suffix on same-second/same-commit collisions.
#   - P2-3: `make -n`/`-q`/`-t publish` must have ZERO filesystem side
#     effects (no DIST_ROOT/OBJ_ROOT/BIN_ROOT paths created).
#   - P2 (dirty policy): publish FATALs by default on a dirty/no-git-identity
#     checkout; ALLOW_DIRTY_PUBLISH=1 is the recorded escape hatch.
#
# ...and now the round-7 review:
#
#   - P1a (S8): the knob-neutrality check no longer walks the real obj dir
#     wholesale with `for o in "$real_kiss_objdir"/*.o` -- a prior
#     `make test_wav` in the SAME real obj dir leaves extra test objects
#     (test_wav_io.o, ...) that the scratch build never produces, so that
#     glob false-FAILed on an otherwise-clean tip. Fixed to an
#     archive-MEMBER comparison instead (member name-list equality, already
#     present, PLUS a per-member `ar -p | cmp` loop over the real archive's
#     own member list -- read via `while IFS= read -r m` from `ar -t`,
#     never `for m in $(ar -t ...)`: macOS BSD ar's `__.SYMDEF SORTED`
#     pseudo-member has an embedded space and would word-split into two
#     bogus tokens under a bare `for`).
#   - P1b (S17): `adopt_worktree_clone()` -- factored out of what used to be
#     S17's own inline unconditional `git add -A && git commit`, which threw
#     "nothing to commit" (exit 1, `set -e` kills the suite) on a clean
#     tip with no local modifications relative to HEAD. The helper only
#     commits when `git diff --cached --quiet --exit-code` actually finds
#     staged changes, modelled on Audio_ALG/pipelines' own (correct)
#     version of this same helper.
#   - P2a: ALLOW_UNTRACKED_PUBLISH -- a SEPARATE dimension from
#     ALLOW_DIRTY_PUBLISH (which now covers TRACKED changes only, via
#     `git status --porcelain -uno`): an untracked file can change what a
#     (possibly clean) Makefile builds without leaving a trace in the
#     tracked diff, so it gets its own refusal + its own
#     untracked_tree_sha256 attestation field (sha256 over sorted
#     "L <path> <target>" / "F <mode> <path> <sha256>" records), and an
#     unhashable untracked path (an embedded git checkout, ...) is always
#     refused by name. S17d-h exercise this end to end.
#   - P2b: the last two real-tree mtime-touchers are gone -- S2 and S5 now
#     run entirely inside scratch clones (via `adopt_worktree_clone()`), and
#     S12's three toolchain-shim builds moved to a scratch OBJ_ROOT/BIN_ROOT
#     (they used to leak a fresh CXX-shim-keyed directory into the REAL obj/
#     on every single run of this suite). A cross-suite mutex (SUITE_LOCK
#     under TMPDIR) now refuses to let two instances of this suite (or the
#     Audio_ALG/pipelines counterpart, which shares the same lock path) run
#     concurrently -- both assert real-tree invariants a second concurrent
#     run would trample.
#   - P3: a final-guard snapshot of every keyed directory already under the
#     real obj/ and bin/ once this suite's own initial normal-config builds
#     finish (path + sha256 + `stat -f %Fm` per file); re-checked
#     byte-for-byte at the very end. New directories are allowed (printed as
#     INFO) -- after the S12 fix there should be none.
#
# Scenario index:
#   S1  - A->B->A + -O0/-O3 delivered-object repro
#   S2  - same-second kiss<->ne10 switching (round-7: inside a scratch clone)
#   S3  - parallel differing-config builds
#   S4  - test_ne10_force_c isolation
#   S5  - consumer correctness, drives AEC (round-7: inside scratch clones of
#         BOTH audio_common and AEC)
#   S7p - producer-publish (v4 content-addressed dist/ layout; throwaway
#         DIST_ROOT=)
#   S8  - CFG_SIG collision guard (round-6: scratch-side, real obj/
#         untouched; round-7: knob-neutrality re-done as an archive-member
#         comparison, not a real-obj-dir glob)
#   S9  - command-line override rejection (round-4 P1-1)
#   S10 - archive freshness / stale-member removal (round-4 P1-4; round-6:
#         scratch-side, deterministic backdate, real archive untouched)
#   S11 - publish immutability / content-addressing, in three parts:
#           S11a - in-tree republish (throwaway DIST_ROOT): byte-verified
#                  republish, ATTEST/ growth (round-6: no sleep -- the
#                  <NNN> suffix disambiguates a same-second republish)
#           S11b - MANIFEST tamper detection
#           S11c - content-change publish, exercised in a throwaway sandbox
#                  copy of the tree (never in $AC_DIR)
#   S12 - toolchain coherence guard (round-4 P1-2; round-7: scratch-side
#         OBJ_ROOT/BIN_ROOT, was leaking into the real obj/ every run)
#   S13 - atomic `current` symlink swap hammer + --excl-install no-clobber
#         mode (round-5 P2; round-6 adds the --excl-install checks)
#   S14 - lock-before-build: a concurrent publish loser builds nothing
#         (round-5 P1; round-6: scratch OBJ_ROOT/BIN_ROOT)
#   S15 - `make -n`/`-q`/`-t publish` zero-write assertions (round-6 P2-3;
#         round-7: pre-recorded artifact snapshot compared unchanged after
#         each of -n/-q/-t, and -t's rc=0 + no-op-note text are asserted);
#         S15b COMBINED flags (-nt/-tq/-nq/-nqt) against a deliberately
#         STALE scratch archive (t is checked FIRST in the driver -- a
#         recursing -nt used to let real touch semantics through)
#   S16 - ATTEST uniqueness under forced same-second collisions (round-6 P2-2)
#   S17 - dirty-checkout publish policy (round-6 P2), now via
#         `adopt_worktree_clone()` (round-7 P1b), PLUS (round-7 P2a) the
#         untracked-file dimension: S17d clean+untracked default-refused/
#         ALLOW_DIRTY_PUBLISH-alone-still-refused/both-succeeds; S17e
#         re-publish with different untracked bytes -> different
#         untracked_tree_sha256; S17f untracked SYMLINK, then repointed;
#         S17g a genuinely build-relevant untracked source (tracked-dirty
#         AND untracked-dirty at once -- both knobs required, content
#         differs from the clean release); S17h an unhashable untracked
#         path (embedded git checkout) -> named FATAL; S17i the classic
#         encoding-collision symlink pair must hash DIFFERENTLY; S17j an
#         unreadable (chmod 000) untracked file -> fail-closed named
#         refusal; S17k a chmod alone changes the hash; S17l an
#         identity-less checkout is refused UNCONDITIONALLY (no hatch).
#   S18 - interruption-safety probe: EXIT/INT/TERM all clean up the whole
#         scratch tree, even mid-scenario (round-6 P2-1 acceptance test)
#   S19 - FFT_WRAPPER_ALIAS_CFLAGS is CFG_SIG-keyed: build-cache-invalidation
#         regression guard for the gap a re-review found in the round-5
#         -fno-strict-aliasing fix (the flag was a bare literal on a
#         target-specific CFLAGS line, invisible to CFG_SIG_PAYLOAD, so
#         changing/removing it never forced a fresh keyed dir); asserts a
#         real `make lib` build off the unmodified Makefile and a second
#         real `make lib` build off a scratch clone whose Makefile copy has
#         had ONLY the FFT_WRAPPER_ALIAS_CFLAGS definition line sed-patched
#         to a different literal land in two different keyed obj/bin/lib
#         paths. (A later Codex review found a command-line override of this
#         same variable was itself unguarded -- see S20 -- so this scenario
#         no longer proves CFG_SIG coverage via a command-line override; the
#         sed-patched-clone shape proves the identical property without
#         relying on the now-closed hole.)
#   S20 - FFT_WRAPPER_ALIAS_CFLAGS command-line/environment-override
#         rejection (Codex review): the same class of hole S9 already closed
#         for CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/FP_POLICY existed for this
#         variable too -- `make FFT_WRAPPER_ALIAS_CFLAGS=` (empty) or
#         `make FFT_WRAPPER_ALIAS_CFLAGS=-fstrict-aliasing` on the command
#         line used to silently win over the Makefile's own definition and
#         produce a real, buildable fft_wrapper.o/fft_wrapper_ne10.o with the
#         known strict-aliasing UB unguarded; both must now be rejected at
#         parse time, mirroring S9's exact assertion style
#   S21 - `make -e` (environment-override mode) bypass of S20's own guard
#         (second Codex review): S20 proved the command-line case; a
#         SEPARATE gap let `env FFT_WRAPPER_ALIAS_CFLAGS=-fstrict-aliasing
#         make -e BACKEND=kiss lib` sail straight through with NO error,
#         because the "Command-line override rejection" foreach queried
#         $(origin FFT_WRAPPER_ALIAS_CFLAGS) BEFORE this variable's own (only)
#         assignment in the file -- under `-e`, GNU Make only flips a
#         variable's origin from "environment" to "environment override" at
#         the point a makefile assignment to it is actually parsed, so
#         querying it earlier still reports plain "environment", which
#         matches neither `command` nor `override` and the check silently
#         passed. Fixed by relocating the bare `:=` literal to directly
#         above the foreach (see the Makefile's own "Bare-literal policy
#         flags" comment). Both the exact reported repro (non-empty hostile
#         value) and the empty-value sibling (full parity with S20's two
#         cases) must now be rejected at parse time under `-e`.
#   S22 - the same `make -e` class of gap, audited across the other five
#         names S20/S9 already cover (Codex review follow-up): FP_POLICY
#         turned out to have the IDENTICAL shape as FFT_WRAPPER_ALIAS_CFLAGS
#         (its own bare `:=` also lived after the foreach) and CPPFLAGS had a
#         DIFFERENT variant (its early `?=` never actually attempts an
#         assignment once the variable is environment-original, so the
#         origin-flip trigger never fires under `-e` regardless of position)
#         -- both confirmed to let `env FP_POLICY=x make -e ...` / `env
#         CPPFLAGS=x make -e ...` sail through unguarded before this round's
#         fix (FP_POLICY: relocated bare `:=`, same as FFT_WRAPPER_ALIAS_
#         CFLAGS; CPPFLAGS: `?=` changed to `+=`). CFLAGS/CXXFLAGS/LDFLAGS
#         were confirmed ALREADY correctly rejected under `-e` (each has an
#         unconditional `+=` before the foreach) and are not re-tested here.
#   S23 - FP-policy conflict-detection widened to CXXFLAGS/CPPFLAGS (Codex
#         review): the round-3 B04 conflict-detection block (rejects
#         -Ofast/-ffast-math/-ffp-contract=<anything> so a build can never
#         silently re-enable FP contraction) used to check $(CFLAGS) alone;
#         a PLAIN ENVIRONMENT CXXFLAGS (or CPPFLAGS) -- deliberately allowed
#         to fold in normally by the command-line/`-e` rejection above,
#         since it never goes through EXTRA_CFLAGS -- sailed straight past
#         all three checks. Direct repro (confirmed BEFORE this round's
#         Makefile fix): `env CXXFLAGS=-Ofast make BACKEND=ne10 lib` built a
#         real archive with the NE10 backend's one C++ TU
#         (NE10_fft_generic_int32.cpp, the only C++ TU this Makefile
#         compiles) showing `-Ofast` on its compile line ahead of the
#         trailing `-ffp-contract=off` -- every -Ofast relaxation OTHER than
#         contraction (which the trailing flag still turns back off) live
#         and unchecked. BACKEND=ne10 (not kiss) is used throughout this
#         scenario because it is the only backend that compiles a C++ TU at
#         all -- confirmed empirically: the identical repro against
#         BACKEND=kiss never invokes a C++ compiler at all (0 `.cpp` compile
#         lines), so it would not have exercised the actual consequence of
#         the gap, only the (backend-independent) text-matching guard
#         itself. Each of the three conflict patterns is run as a PLAIN
#         environment CXXFLAGS override (no `-e`, no command-line
#         assignment) against the real `lib` target (the exact reported
#         repro), plus the CPPFLAGS sibling of the -Ofast case for
#         completeness; all four must now FAIL, mentioning "FP policy
#         conflict". A later (round-9) Codex review found the widened
#         checks themselves used $(findstring), a plain SUBSTRING search,
#         so a harmless token that merely CONTAINS one of the three
#         patterns (e.g. `-DROUND9_NOTE=-Ofastness`) was incorrectly
#         rejected too; the fix moved to $(filter), which only matches
#         whole flag words, and S23 gained a positive-case test asserting
#         that exact repro now SUCCEEDS.
#   S23b - quote/escape/response-file bypass of the FP-policy conflict gate
#          itself (Codex review): $(filter) in the Makefile's exact-token
#          check does whole-WORD matching against GNU Make's OWN text
#          representation of CFLAGS/CXXFLAGS/CPPFLAGS -- but Make has zero
#          concept of shell quoting. EXTRA_CFLAGS="'-Ofast'" is stored by
#          Make as the literal 9-character text '-Ofast' (quote characters
#          included), which never equals the bare word -Ofast, so $(filter)
#          could never match it -- yet on the actual compile recipe line,
#          handed to $(SHELL), the shell strips the quotes and the compiler
#          genuinely receives -Ofast. Confirmed-live bypass shapes (all BEFORE
#          this round's Makefile fix): EXTRA_CFLAGS="'-Ofast'" (single-
#          quoted), EXTRA_CFLAGS='"-ffast-math"' (double-quoted),
#          EXTRA_CFLAGS="-O'f'ast" (quote-split mid-token, shell concatenates
#          the pieces back into -Ofast once quotes are stripped), and
#          EXTRA_CFLAGS=@flags.rsp (a bare @file argv token many compilers
#          treat as "read more arguments from this file" -- unrelated to
#          quoting, but equally invisible to a plain-token check, and able to
#          inject anything with zero text ever appearing in CFLAGS/
#          EXTRA_CFLAGS itself). The fix adds a new rejection block, run
#          BEFORE the exact-token check, that rejects outright any of:
#          a single quote, a double quote, a backslash, a backtick, a
#          `$(` sequence, a semicolon, a pipe, an ampersand, or any whole
#          token starting with `@` -- appearing anywhere in the combined
#          CFLAGS/CXXFLAGS/CPPFLAGS text, each with its own $(error ...)
#          naming the specific construct found. All four bypass shapes
#          above must now FAIL, each mentioning the specific forbidden
#          construct (not just a generic failure); a positive control
#          (EXTRA_CFLAGS=-DROUND9_NOTE=-Ofastness, the plain unquoted token
#          from S23's own round-9 regression, plus the sibling
#          -DTEXT=ffast-math token) must still SUCCEED, since neither
#          contains any of the newly-forbidden characters.
# S6 (Audio_ALG/pipelines consumer-resolution parity) is a later wave and is
# intentionally NOT covered here.
#
# Round-7 safety contract (supersedes the round-6 version of this comment):
#   - ONE scratch root for the entire run: SCRATCH_ROOT="$(mktemp -d)",
#     removed by a single EXIT trap. Every temp file/dir this script uses is
#     a FIXED path under "$SCRATCH_ROOT/<scenario>/...", created inline with
#     `mkdir -p` -- never inside a `$(...)` command substitution (that was
#     the round-6 P2-1 bug: registering cleanup state from inside a subshell
#     silently drops it). `rm -rf "$SCRATCH_ROOT"` on exit collects
#     EVERYTHING, including anything a child `make`'s own mktemp calls
#     create, because TMPDIR is exported pointing inside the scratch root.
#   - A cross-suite mutex (SUITE_LOCK="$TMPDIR/audio_dsp_isolation_suite.lock",
#     computed from the INVOKING environment's TMPDIR, an mkdir-based lock
#     with a pid file) refuses to let two instances of this suite run
#     concurrently -- a stale lock (holder pid confirmed dead via `kill -0`)
#     is reclaimed once, with a TOCTOU-safe retry so a losing reclaimer never
#     steals the winner's fresh lock. Released in cleanup() only by the
#     process that actually acquired it.
#   - Real obj/ and bin/ (this repo's own, default-rooted) only EVER see the
#     NORMAL kiss/ne10 configs that S1/S3's real-tree assertions depend on
#     (S2 and S5, round-6's other two real-tree normal-config builders, now
#     run inside scratch clones instead -- see below). Every throwaway/probe
#     config (S1's -O0/-O3, S3's -DPAR_PROBE, S4's forced-C variant, S9's
#     EXTRA_CFLAGS probe, S12's toolchain-shim probes, S14's
#     -DS14_LOCK_PROBE) and every tamper scenario (S8, S10) builds under a
#     scratch OBJ_ROOT=/BIN_ROOT= instead. A final-guard snapshot (taken once
#     S1's initial normal-config builds finish, of every keyed directory
#     already under the real obj/+bin/ at that point) is re-verified
#     byte-for-byte identical at the very end -- the real tree gains nothing
#     this suite didn't already account for.
#   - The real dist/ is never read, written, or removed: every `make ...
#     publish` passes an explicit DIST_ROOT= under $SCRATCH_ROOT. A sentinel
#     digest of the real dist/ (absent, or a full manifest of paths + sha256
#     + mtime) is captured at the very start and re-checked at the very end.
#   - No git-tracked file, in $AC_DIR OR the AEC repo, is EVER touched --
#     not its content, not even its mtime (round-7 removes the round-6
#     allowance for a handful of mtime-only touches on real tracked sources
#     entirely). Every mtime bump this script performs to force a recompile
#     probe (hpf.c, fast_math.h, wav_io.h, NE10_dsp.h, ...) now happens
#     against a SCRATCH CLONE's copy of that source (S2, S5), or an already
#     sandbox/clone-local file (S11c, S17). A `tree_state_hash()`
#     (status --porcelain + diff --binary HEAD, not status alone -- status
#     alone misses a content edit to an ALREADY-dirty file) is captured for
#     BOTH $AC_DIR and the AEC repo at $AC_DIR/../AEC at the start and
#     re-checked at the end, asserting exact equality (not just "still
#     dirty") -- any script bug that touched a real tracked file's content
#     OR mtime-affecting metadata would still have to leave `git status
#     --porcelain`/`git diff` themselves unchanged to pass, which a content
#     change cannot do.
#   - No `sleep` anywhere. Every place the round-5 script used `sleep 1` to
#     force a source's mtime strictly past some artifact's mtime is replaced
#     with a deterministic BSD `touch -r <artifact> -A 01 <source>` bump
#     (source mtime := artifact's CURRENT mtime + 1s) -- immune to whichever
#     way whole-second wall-clock/filesystem mtime comparisons round. Where
#     no artifact-staleness relationship is being forced (a plain "did
#     nothing rebuild"/"did this relink" check), `mtime()` itself now reads
#     BSD `stat -f '%Fm'` (fractional seconds), so two genuinely different
#     real writes are distinguishable without any artificial delay.
#   - Interruption-safe by construction: cleanup() is the ONLY EXIT-trap
#     resident, INT/TERM map to the conventional 130/143 exit codes (both
#     routed back through the same cleanup()), so a `^C`/`kill` mid-scenario
#     still removes the whole scratch tree. S18 tests exactly this.
#
# Design rules (do not violate when editing this script):
#   - No `make clean` inside any scenario body: the whole point is that
#     distinct configs coexist WITHOUT ever needing a clean between them.
#   - Every path is resolved via `make -s ... print-bin-dir` / `print-obj-dir`
#     / `print-lib-path`, using the EXACT flag set (INCLUDING OBJ_ROOT=/
#     BIN_ROOT= when the build under test used them) under test for that
#     call -- never a hand-reconstructed path guess.
#   - "Did this get rebuilt?" is an mtime comparison, never a content (sha)
#     comparison: recompiling identical inputs with a deterministic compiler
#     produces byte-identical output, so sha equality does NOT mean "not
#     rebuilt" and sha difference is not required for "was rebuilt".
#   - "Is this the SAME delivered artifact as its own keyed object?" (the
#     actual B01 assertion — no cross-config delivery) IS a sha comparison,
#     via member_sha()/file_sha() below.
#
# Usage: ./scripts/test_build_isolation.sh   (run from audio_common/, or
# anywhere — paths are resolved relative to this script's own location).
#
# ISOL_INTERRUPT_PROBE (internal, used only by S18): when set, this script
# runs as a child re-invocation of itself in "interruption probe" mode
# instead of running the suite -- see the block immediately after the
# SCRATCH_ROOT/trap setup below.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
AC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
AEC_REPO_DIR="$(cd "$AC_DIR/../AEC" && pwd)"
AEC_DIR="$(cd "$AEC_REPO_DIR/c_impl" && pwd)"
NR_DIR="$(cd "$AC_DIR/../NR/c_impl" && pwd)"

# --- round-6 review P2-1: single scratch root, single trap ------------------
SCRATCH_ROOT="$(mktemp -d)"
# SUITE_LOCK_HELD must exist (as "0") before ANYTHING that could exit through
# cleanup() runs -- including the ISOL_INTERRUPT_PROBE early-return branch
# immediately below, which (being a probe CHILD, not a real suite run) never
# reaches the round-7 mutex block that would otherwise set this to "1" --
# `set -u` would otherwise fault on cleanup()'s own reference to it.
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

# --- S18 interruption probe: MUST be checked early, before any suite logic --
# (round-6 review P2-1 acceptance test), and BEFORE the mktemp-shim setup
# below: probe mode's `set +e` (see the comment inside the branch) needs to
# take effect before ANY further command runs in this process, since a
# signal can arrive during the shim's own file writes just as easily as
# during the probe's own fifo dance -- moving the whole shim setup after
# this check keeps probe mode's error-handling window as small as possible.
# When invoked with ISOL_INTERRUPT_PROBE set, this is a CHILD re-invocation
# of this very script, spawned by the S18 scenario body further down. It
# signals its own SCRATCH_ROOT back to the parent over a fifo (the write is
# also the readiness signal), then either exits immediately (mode "exit") or
# blocks forever on a second fifo that nobody ever writes to, so a
# parent-delivered INT/TERM is what interrupts it -- exactly mirroring how
# this script could be interrupted mid-scenario in real use. It does NOT
# need its own mktemp shim (below): its own `mktemp -d` for SCRATCH_ROOT
# above already inherited the PARENT's shim via PATH (set before the parent
# spawned it), which is what lands this SCRATCH_ROOT inside the parent's
# chosen probe TMPDIR in the first place.
if [ -n "${ISOL_INTERRUPT_PROBE:-}" ]; then
  # `set +e` for the rest of probe mode (round-6 review, found while
  # validating THIS script): with `errexit` still active, a SIGTERM/SIGINT
  # that arrives while this process is blocked in a syscall (opening the
  # fifo, or the `read` below) can surface as EINTR turning into that
  # command's own non-zero return -- and errexit reacts to THAT directly
  # (running the EXIT trap with the interrupted command's status) before
  # bash ever services the pending INT/TERM trap, so the child would exit
  # with the wrong code and skip straight to cleanup with a stale $?
  # (verified empirically on this host's bash 3.2 to be flaky under load:
  # which command absorbs the signal-interrupted return varies run to run).
  # Disabling errexit here removes the whole race: the INT/TERM traps
  # installed above are what decide this process's exit code, unconditionally.
  set +e
  mkdir -p "$SCRATCH_ROOT/probe"
  : > "$SCRATCH_ROOT/probe/canary_a"
  : > "$SCRATCH_ROOT/probe/canary_b"
  printf '%s\n' "$SCRATCH_ROOT" > "$ISOL_PROBE_FIFO"
  if [ "$ISOL_INTERRUPT_PROBE" = "exit" ]; then
    exit 0
  fi
  # Open the hold-fifo O_RDWR (never blocks on open(), since we hold both
  # ends ourselves). Block via `wait` on a background reader of that fd,
  # NOT a direct foreground `read <&9` -- verified empirically (bash 3.2) to
  # matter: a signal arriving while this shell is itself blocked inside a
  # foreground `read` builtin can leave the pending INT/TERM trap action
  # un-run (the interrupted `read` just falls through to whatever comes
  # next), and a SECOND, deferred delivery of that same signal then fires
  # AFTER `cleanup()` has already reset the trap to its default disposition
  # -- killing the process mid-`rm -rf` instead of letting cleanup finish.
  # `wait` is bash's OWN documented interruptible blocking primitive (see
  # bash(1), SIGNALS: "SIGINT is caught and handled so that the wait builtin
  # is interruptible") and does not exhibit this: 8/8 clean runs under both
  # SIGTERM and SIGINT in isolated repro, vs. a flaky ~1-in-3 failure rate
  # with a direct `read <&9`.
  exec 9<>"$ISOL_PROBE_FIFO.hold"
  cat <&9 >/dev/null &
  catpid=$!
  wait "$catpid"
  exit 0
fi

# --- round-7 review P2b: cross-suite mutex -----------------------------------
# Placed immediately after the ISOL_INTERRUPT_PROBE early-return above (so a
# probe CHILD -- itself a full re-invocation of this script, spawned by S18 --
# never reaches this and never contends for the lock its own PARENT already
# holds) and BEFORE `export TMPDIR=...` below (so `${TMPDIR:-/tmp}` here still
# resolves to the INVOKING environment's TMPDIR, not the scratch-local one
# this script is about to export over it). Two instances of this suite -- or
# of the Audio_ALG/pipelines counterpart, which uses this EXACT SAME lock
# path/name -- must never run concurrently: both assert real-tree invariants
# (S1's real kiss/ne10 archives, the final-guard snapshot below, the dist/
# sentinel, tree_state_hash()) that a second concurrent run's own real-tree
# activity would trample.
SUITE_LOCK="${TMPDIR:-/tmp}/audio_dsp_isolation_suite.lock"
if mkdir "$SUITE_LOCK" 2>/dev/null; then
  printf '%s\n' "$$" > "$SUITE_LOCK/pid"
  SUITE_LOCK_HELD=1
else
  holder_pid="$(cat "$SUITE_LOCK/pid" 2>/dev/null || true)"
  reclaimed=0
  if [ -n "$holder_pid" ] && ! kill -0 "$holder_pid" 2>/dev/null; then
    # Stale lock: the recorded holder pid is confirmed dead. Reclaim once --
    # TOCTOU: two reclaimers can race this exact rm -rf/mkdir pair; only one
    # mkdir wins, and the loser must NOT steal the winner's fresh lock, so
    # this is a single retry (not a loop) and the loser falls through to the
    # FATAL below on failure.
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

# macOS-specific gotcha (found while validating this rewrite): BSD `mktemp`'s
# bare `-d` / no-template form resolves via _CS_DARWIN_USER_TEMP_DIR FIRST
# (see mktemp(1): "If no arguments are passed or if only the -d flag is
# passed mktemp behaves as if -t tmp was supplied", and -t's own fallback
# order puts $TMPDIR behind that Darwin-specific dir) -- so exporting TMPDIR
# alone does NOT redirect a child process's own bare `mktemp -d` (e.g. this
# Makefile's own `publish` recipe: `work="$(mktemp -d)"`, or S18's child
# re-invocation's own `SCRATCH_ROOT="$(mktemp -d)"`) into our scratch tree.
# A tiny shim ahead of the real mktemp on PATH closes this gap: it forwards
# to the real /usr/bin/mktemp, injecting `-p "$TMPDIR"` whenever the caller
# didn't already give an explicit template or -p (verified empirically: a
# bare `TMPDIR=/some/dir mktemp -d` on this host ignores TMPDIR entirely
# without this shim).
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
# Fractional seconds (round-6: replaces whole-second `stat -f %m`) -- two
# genuinely separate real writes are distinguishable without an artificial
# delay. GNU fallback loses the fractional part (not exercised on this host).
mtime()      { stat -f '%Fm' "$1" 2>/dev/null || stat -c %Y "$1"; }

# Tree-state hash (round-6 review): status --porcelain ALONE misses a content
# edit to an already-dirty file, so this also folds in `diff --binary HEAD`.
tree_state_hash() {
  { git -C "$1" status --porcelain; git -C "$1" diff --binary HEAD; } 2>/dev/null | shasum -a 256 | awk '{print $1}'
}

# adopt_worktree_clone <src-repo-dir> <clone-dir> (round-7 review P1b) --
# clone <src-repo-dir> (fully read-only on the source -- the explicitly
# allowed exception to "no mutating git commands against real repos"), then
# overlay every currently-MODIFIED TRACKED file on top and adopt them as ONE
# throwaway commit INSIDE THE SCRATCH CLONE ONLY, so the clone gets a
# genuinely CLEAN git identity that still reflects today's (possibly
# uncommitted) worktree content. A plain `git clone` alone only reproduces
# the source's last COMMIT -- it transfers committed objects, never
# uncommitted working-tree bytes -- so a bare clone would carry the OLD
# Makefile instead of the round-7 one under test. The clone is a disposable
# git repository living entirely under $SCRATCH_ROOT, destroyed by the EXIT
# trap; its own history never touches the source repo's .git in any way (a
# distinct clone, never pushed/fetched back).
#
# Round-7 fix (P1b, the actual finding): the commit is CONDITIONAL -- `git
# add -A`, then commit only if `git diff --cached --quiet --exit-code` says
# something is actually staged. The round-6 shape committed unconditionally,
# so at a clean tip (no local modifications relative to HEAD) the empty
# commit exited 1 and `set -e` killed the whole suite. Modelled on the
# Audio_ALG/pipelines suite's (correct) version of this same helper.
adopt_worktree_clone() {
  local src="$1" dst="$2" relf
  git clone --no-hardlinks --quiet "$src" "$dst"
  while IFS= read -r relf; do
    [ -n "$relf" ] || continue
    mkdir -p "$(dirname "$dst/$relf")"
    cp "$src/$relf" "$dst/$relf"
  done < <(git -C "$src" diff --name-only HEAD)
  git -C "$dst" add -A
  if ! git -C "$dst" diff --cached --quiet --exit-code; then
    git -C "$dst" -c user.email=scratch@example.invalid -c user.name="scratch clone" \
      commit -q -m "scratch: adopt current worktree (disposable clone only, never touches the real repo)"
  fi
}

# Real-tree final guard (round-7 review P3): per-file "path sha256 mtime"
# snapshot of everything under the real obj/ and bin/, plus a separate
# top-level keyed-directory listing -- captured in S1 below, the moment this
# suite's own intentional real-tree normal-config builds (kiss + ne10) are
# done, and re-verified at the very end: every snapshotted file must still
# be sha+mtime identical; a genuinely NEW directory is allowed (INFO/WARN,
# not FAIL -- but after the round-7 S12 fix there should be none), while a
# new file inside an ALREADY-snapshotted directory is a FAIL.
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

# Real dist/ sentinel (round-6 review, standing guard): absent stays absent;
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
echo "# S1: A->B->A + -O0/-O3 delivered-object repro"
echo "############################################################"
cd "$AC_DIR"

kiss_lib="$(make -s BACKEND=kiss print-lib-path)"
make -s BACKEND=kiss lib >/dev/null
sha_kiss_1="$(file_sha "$kiss_lib")"
mtime_kiss_1="$(mtime "$kiss_lib")"

ne10_lib="$(make -s BACKEND=ne10 print-lib-path)"
make -s BACKEND=ne10 lib >/dev/null

# round-7 review P3: final-guard snapshot, taken NOW -- both of the real-tree
# normal-config builds this suite intends (kiss above, ne10 just now) are
# done; every later real-tree `make ... lib` in this suite re-targets one of
# these same two already-built configs and must be a pure no-op against
# them. Re-verified at the very end (see "Final integrity guards").
FINAL_GUARD_DIRS_BEFORE="$(real_tree_dirs)"
FINAL_GUARD_SNAPSHOT_BEFORE="$(real_tree_snapshot)"

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
#
# round-6 review P1: -O0/-O3 are throwaway probe configs, not one of the
# normal kiss/ne10 configs the rest of this repo's real obj/bin is meant to
# hold -- so this builds under a scratch OBJ_ROOT=/BIN_ROOT= instead. Every
# print-* query below is resolved against that SAME scratch root pair.
mkscratch s1
S1_OBJ_ROOT="$SCRATCH_ROOT/s1/obj"
S1_BIN_ROOT="$SCRATCH_ROOT/s1/bin"
make -s BACKEND=kiss EXTRA_CFLAGS=-O0 OBJ_ROOT="$S1_OBJ_ROOT" BIN_ROOT="$S1_BIN_ROOT" lib >/dev/null
lib_o0="$(make -s BACKEND=kiss EXTRA_CFLAGS=-O0 OBJ_ROOT="$S1_OBJ_ROOT" BIN_ROOT="$S1_BIN_ROOT" print-lib-path)"
objdir_o0="$(make -s BACKEND=kiss EXTRA_CFLAGS=-O0 OBJ_ROOT="$S1_OBJ_ROOT" BIN_ROOT="$S1_BIN_ROOT" print-obj-dir)"
make -s BACKEND=kiss EXTRA_CFLAGS=-O3 OBJ_ROOT="$S1_OBJ_ROOT" BIN_ROOT="$S1_BIN_ROOT" lib >/dev/null
lib_o3="$(make -s BACKEND=kiss EXTRA_CFLAGS=-O3 OBJ_ROOT="$S1_OBJ_ROOT" BIN_ROOT="$S1_BIN_ROOT" print-lib-path)"
objdir_o3="$(make -s BACKEND=kiss EXTRA_CFLAGS=-O3 OBJ_ROOT="$S1_OBJ_ROOT" BIN_ROOT="$S1_BIN_ROOT" print-obj-dir)"

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
echo "# S2: same-second kiss<->ne10 switching (round-7: scratch clone)"
echo "############################################################"
# round-7 review P2b: this scenario used to `touch` the REAL tree's
# src/hpf.c (mtime-only, but round-7 removes even that allowance -- no real
# tracked file is touched any more, period). The whole scenario now runs
# inside a disposable clone of $AC_DIR under scratch; the clone's own obj/
# and bin/ (default-rooted, clone-relative) ARE the scratch space, so no
# OBJ_ROOT/BIN_ROOT override is needed. Assertions are identical.
mkscratch s2
S2_CLONE="$SCRATCH_ROOT/s2/ac_clone"
adopt_worktree_clone "$AC_DIR" "$S2_CLONE"
# mtime-only touch (content-neutral) of the CLONE's hpf.c: forces a
# recompile probe without ever changing what's inside the file.
touch "$S2_CLONE/src/hpf.c"
make -C "$S2_CLONE" -s BACKEND=kiss lib >/dev/null
make -C "$S2_CLONE" -s BACKEND=ne10 lib >/dev/null
kiss_lib="$(make -C "$S2_CLONE" -s BACKEND=kiss print-lib-path)";  kiss_objdir="$(make -C "$S2_CLONE" -s BACKEND=kiss print-obj-dir)"
ne10_lib="$(make -C "$S2_CLONE" -s BACKEND=ne10 print-lib-path)";  ne10_objdir="$(make -C "$S2_CLONE" -s BACKEND=ne10 print-obj-dir)"
m1="$(member_sha "$kiss_lib" hpf.o)"; o1="$(file_sha "$kiss_objdir/hpf.o")"
m2="$(member_sha "$ne10_lib" hpf.o)"; o2="$(file_sha "$ne10_objdir/hpf.o")"
[ "$m1" = "$o1" ] && [ "$m2" = "$o2" ] && \
  pass "S2: per-archive hpf.o member sha == own keyed object (same-second switch safe)" \
  || fail "S2: hpf.o member/object mismatch after same-second switch (kiss: $m1 vs $o1; ne10: $m2 vs $o2)"

echo "############################################################"
echo "# S3: parallel differing-config builds"
echo "############################################################"
mkscratch s3
# round-6 review P1: the ne10+PAR_PROBE side is a throwaway probe config ->
# scratch OBJ_ROOT/BIN_ROOT. The plain kiss side stays in the real tree (this
# half is exactly the normal-config concurrency this scenario exists to
# verify).
S3_OBJ_ROOT="$SCRATCH_ROOT/s3/obj"
S3_BIN_ROOT="$SCRATCH_ROOT/s3/bin"
S3_LOG_A="$SCRATCH_ROOT/s3/log_a"
S3_LOG_B="$SCRATCH_ROOT/s3/log_b"
( make -j4 -s BACKEND=kiss lib >"$S3_LOG_A" 2>&1 ) & p1=$!
( make -j4 -s BACKEND=ne10 EXTRA_CFLAGS=-DPAR_PROBE OBJ_ROOT="$S3_OBJ_ROOT" BIN_ROOT="$S3_BIN_ROOT" lib >"$S3_LOG_B" 2>&1 ) & p2=$!
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
ne10p_lib="$(make -s BACKEND=ne10 EXTRA_CFLAGS=-DPAR_PROBE OBJ_ROOT="$S3_OBJ_ROOT" BIN_ROOT="$S3_BIN_ROOT" print-lib-path)"
ne10p_objdir="$(make -s BACKEND=ne10 EXTRA_CFLAGS=-DPAR_PROBE OBJ_ROOT="$S3_OBJ_ROOT" BIN_ROOT="$S3_BIN_ROOT" print-obj-dir)"
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

# round-6 review P1: test_ne10_force_c's own internal NEON+forced-C sub-makes
# are throwaway diagnostic builds, never one of the real tree's normal
# configs -- run the WHOLE invocation under a scratch OBJ_ROOT/BIN_ROOT (both
# of its internal sub-makes inherit these, being ordinary command-line
# variable overrides forwarded automatically to $(MAKE) recursive calls), so
# it can never touch the real ne10-<sig>/ directory at all, regardless of
# what it does internally.
mkscratch s4
S4_OBJ_ROOT="$SCRATCH_ROOT/s4/obj"
S4_BIN_ROOT="$SCRATCH_ROOT/s4/bin"
S4_LOG="$SCRATCH_ROOT/s4/log"
if make test_ne10_force_c OBJ_ROOT="$S4_OBJ_ROOT" BIN_ROOT="$S4_BIN_ROOT" >"$S4_LOG" 2>&1; then
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

normal_lib_scratch="$(make -s BACKEND=ne10 OBJ_ROOT="$S4_OBJ_ROOT" BIN_ROOT="$S4_BIN_ROOT" print-lib-path)"
forcedc_lib="$(make -s BACKEND=ne10 EXTRA_CFLAGS=-DFFT_NE10_FORCE_C OBJ_ROOT="$S4_OBJ_ROOT" BIN_ROOT="$S4_BIN_ROOT" print-lib-path)"
[ "$forcedc_lib" != "$normal_lib_scratch" ] && pass "S4: forced-C archive lives at a different keyed path than the normal ne10 build (both under the SAME scratch roots)" \
  || fail "S4: forced-C archive path COLLIDES with the normal ne10 path ($forcedc_lib)"
[ -f "$forcedc_lib" ] && pass "S4: forced-C archive exists on disk" || fail "S4: forced-C archive missing"

forcedc_objdir="$(make -s BACKEND=ne10 EXTRA_CFLAGS=-DFFT_NE10_FORCE_C OBJ_ROOT="$S4_OBJ_ROOT" BIN_ROOT="$S4_BIN_ROOT" print-obj-dir)"
normal_objdir="$(make -s BACKEND=ne10 OBJ_ROOT="$S4_OBJ_ROOT" BIN_ROOT="$S4_BIN_ROOT" print-obj-dir)"
if [ -f "$forcedc_objdir/fft_wrapper_ne10.o" ] && [ -f "$normal_objdir/fft_wrapper_ne10.o" ]; then
  s1="$(file_sha "$forcedc_objdir/fft_wrapper_ne10.o")"
  s2="$(file_sha "$normal_objdir/fft_wrapper_ne10.o")"
  [ "$s1" != "$s2" ] && pass "S4: forced-C fft_wrapper_ne10.o differs from the normal build's (FFT_NE10_FORCE_C took effect)" \
    || fail "S4: forced-C object is BYTE-IDENTICAL to the normal build's (FFT_NE10_FORCE_C not taking effect?)"
else
  fail "S4: missing fft_wrapper_ne10.o in one of the two obj dirs ($forcedc_objdir or $normal_objdir)"
fi

echo "############################################################"
echo "# S5: consumer correctness (drives AEC; round-7: scratch clones)"
echo "############################################################"
# round-7 review P2b: this scenario used to build in the REAL AEC repo and
# mtime-touch REAL tracked audio_common sources (hpf.c, fast_math.h,
# wav_io.h, NE10_dsp.h). It now runs against a disposable clone of EACH
# repo: an audio_common clone (the producer whose sources get the mtime
# bumps) and a clone of the whole AEC repo (builds run in <clone>/c_impl).
# The clones' own default-rooted obj/ and bin/ are the scratch space.
#
# CRITICAL invocation rule: EVERY make call into the AEC clone passes
# AC_DIR="$S5_AC_CLONE" explicitly (alongside the AC_LIB= this scenario
# already threads through) -- AEC's `$(AC_LIB): FORCE` rule re-invokes
# `$(MAKE) -C $(AC_DIR) ... lib` using that invocation's own AC_DIR, and if
# it were left to its wildcard default it would fall back to the REAL
# audio_common (reintroducing exactly the real-tree writes this round-7
# change removes). A command-line AC_DIR= does propagate to that recursive
# sub-make (command-line-origin variables travel via MAKEFLAGS), but it is
# passed explicitly on every call here anyway -- an invocation this
# correctness-critical should not lean on propagation subtleties.
mkscratch s5
S5_AC_CLONE="$SCRATCH_ROOT/s5/ac_clone"
S5_AEC_CLONE="$SCRATCH_ROOT/s5/aec_clone"
adopt_worktree_clone "$AC_DIR" "$S5_AC_CLONE"
adopt_worktree_clone "$AEC_REPO_DIR" "$S5_AEC_CLONE"

# AEC's own c_impl/example/wav_io.h shim locates the canonical
# audio_common/include/wav_io.h via a HARDCODED relative __has_include path
# ("../../../audio_common/include/wav_io.h" from c_impl/example/, i.e. it
# expects audio_common as a SIBLING of the AEC repo root itself -- never
# through any -I path this scenario could forward). The AEC clone's sibling
# here is $SCRATCH_ROOT/s5/, so a symlink named "audio_common" next to the
# clone, pointing at the audio_common clone, satisfies that lookup without
# editing any tracked source.
ln -s "$S5_AC_CLONE" "$SCRATCH_ROOT/s5/audio_common"
cd "$S5_AEC_CLONE/c_impl"

# AEC's `all` auto-builds the audio_common clone's lib via the $(AC_LIB):
# FORCE rule -- no manual pre-build of the AC clone is needed.
ac_kiss="$(make -s -C "$S5_AC_CLONE" BACKEND=kiss print-lib-path)"
make -s BACKEND=kiss all AC_LIB="$ac_kiss" AC_DIR="$S5_AC_CLONE" >/dev/null
bindir_kiss="$(make -s BACKEND=kiss print-bin-dir AC_LIB="$ac_kiss" AC_DIR="$S5_AC_CLONE")"
aecwav="$bindir_kiss/aec_wav"
[ -f "$aecwav" ] && pass "S5: aec_wav built" || fail "S5: aec_wav missing after initial build"
mtime_1="$(mtime "$aecwav")"

# Backdate aec_wav (round-6 review, found while validating THIS script;
# round-7: aec_wav is now the AEC CLONE's own build artifact, so this
# backdate no longer touches anything real at all): GNU Make
# 3.81's own prerequisite-newer-than-target check truncates to whole
# seconds, so aec_wav's relink decision (against $(AC_LIB)'s freshly
# rebuilt, real "now" mtime) can silently NOT fire if aec_wav's own existing
# mtime happens to truncate to that SAME integer second -- reproduced
# empirically: a real, later archive rebuild whose timestamp nonetheless
# landed in the same second as aec_wav's last link, so make itself declined
# to relink. A safe backdate (5 minutes), called ONLY right before each of
# the three relink-triggering builds further below (never before the no-op
# check just below, which must see aec_wav's NATURAL, untouched mtime to
# mean anything), removes the collision with certainty -- the same way
# `touch -r/-A 01` removes it for a source-vs-object comparison one line
# later in each case, just applied to the TARGET side of a target-vs-
# prerequisite comparison this script cannot touch the prerequisite's own
# real timestamp for (that prerequisite is a real rebuild's output, not our
# own touch).
backdate_aecwav() { touch -A -0500 "$aecwav"; }

# no-op rebuild -> NOT relinked. No sleep needed (round-6): nothing is
# touched here, so aec_wav's mtime cannot change regardless of how much
# wall-clock time elapses between the two `make` calls.
make -s BACKEND=kiss all AC_LIB="$ac_kiss" AC_DIR="$S5_AC_CLONE" >/dev/null
mtime_2="$(mtime "$aecwav")"
[ "$mtime_1" = "$mtime_2" ] && pass "S5: aec_wav NOT relinked on a no-op rebuild" \
  || fail "S5: aec_wav relinked with nothing changed"

# touch the audio_common CLONE's src/hpf.c (mtime-only, content-neutral) ->
# hpf.o recompiles AND aec_wav relinks. audio_common's own CFG_SIG (a hash of
# its COMPILER INVOCATION, not file content) does not change from a source
# edit, so its archive path stays the same -- only its mtime should advance.
#
# round-6 review: deterministic strictly-newer bump replaces `sleep 1; touch
# hpf.c`. Resolve hpf.o's path FIRST (it already exists from earlier
# scenarios), then set hpf.c's mtime to hpf_o's CURRENT mtime + 1s via BSD
# `touch -r/-A` -- this guarantees make ITSELF sees the source as newer than
# the object (make's own dependency check truncates to whole seconds on this
# GNU Make 3.81, so the +1s bump is load-bearing, not just extra caution).
#
# Verifying "did it actually recompile" afterward is a SEPARATE problem from
# "did make decide to": found empirically while validating this rewrite that
# comparing two mtime snapshots OF THE SAME output file, taken moments apart
# by this SCRIPT itself (not by make), can land in the very same whole
# second when the whole edit-recompile-relink chain runs fast and back-to-
# back (no human/sleep delay between steps) -- a real false-negative, not a
# hypothetical one (reproduced: audio_common's own archive rebuilt with a
# real, later timestamp that nonetheless truncated to the SAME integer
# second as its prior build, so make itself also declined to relink aec_wav
# next). So "recompiled?" is verified from make's own build LOG instead of
# an mtime diff: this Makefile still echoes "audio_common [$(BACKEND)] ->
# $@" whenever the archive rule actually runs (even under -s, since a plain
# `@echo` line's own output is never silenced -- only the recipe-echo
# itself is), so its presence/absence in a captured log is a timing-proof
# signal for "did the archive (and therefore hpf.o) actually rebuild".
ac_objdir="$(make -s -C "$S5_AC_CLONE" BACKEND=kiss print-obj-dir)"
hpf_o="$ac_objdir/hpf.o"
# Backdate the CLONE's kiss archive before forcing the recompile (round-7,
# found while validating the clone conversion -- reproduced empirically):
# the +1s hpf.c bump reliably makes make recompile hpf.o, but the fresh
# hpf.o's wall-clock "now" mtime can land in the very SAME whole second as
# the archive's own prior build -- in the clone everything above ran
# back-to-back within one second, unlike the round-6 real-tree version
# where the archive came from S1/S2 many seconds earlier -- and GNU Make
# 3.81's whole-second dependency check then skips the re-archive entirely
# (hpf.o rebuilt, archive stale, no echo line). Same fix as S11c's sandbox:
# backdate the archive (a clone artifact, freely mutable) by a safe margin
# so it is unambiguously older than whatever second the recompile lands on.
touch -A -0500 "$ac_kiss"
hpf_o_mtime_before="$(mtime "$hpf_o")"
touch -r "$hpf_o" -A 01 "$S5_AC_CLONE/src/hpf.c"
S5_LOG_HPF="$SCRATCH_ROOT/s5/log_hpf"
make -s -C "$S5_AC_CLONE" BACKEND=kiss lib >"$S5_LOG_HPF" 2>&1
# Two conditions: hpf.o's own (fractional-second) mtime advanced = the
# recompile genuinely happened (the backdate above would re-archive even
# without it, so the echo line alone would no longer prove this), AND the
# archive-rebuild echo line = the rebuilt object propagated into the
# delivered archive.
if [ "$(mtime "$hpf_o")" != "$hpf_o_mtime_before" ] && grep -q "^audio_common \[kiss\] ->" "$S5_LOG_HPF"; then
  pass "S5: hpf.c edit recompiled audio_common's hpf.o (object mtime advanced) and the archive rebuilt"
else
  fail "S5: hpf.o did NOT recompile (or the archive did not rebuild) after touching hpf.c"
fi

# Same log-based technique for "did aec_wav relink": AEC's own Makefile
# echoes "aec_wav [$(BACKEND)] -> $@" whenever ITS link rule runs.
S5_LOG_RELINK="$SCRATCH_ROOT/s5/log_relink"
backdate_aecwav
make -s BACKEND=kiss all AC_LIB="$ac_kiss" AC_DIR="$S5_AC_CLONE" >"$S5_LOG_RELINK" 2>&1
grep -q "^aec_wav \[kiss\] ->" "$S5_LOG_RELINK" && pass "S5: touching the audio_common clone's src/hpf.c relinked aec_wav" \
  || fail "S5: aec_wav NOT relinked after audio_common's hpf.o changed (no relink line in the log)"

# Header-only test 1: touch fast_math.h -> AEC objects that include it
# (aec3_post.c, suppression_gain.c) recompile, then aec_wav relinks. Same
# deterministic source bump; same log-based relink verification (AEC has no
# per-object echo, but a relink only happens if SOME object actually
# changed, and this is the only header touched in this step).
aec_objdir="$(make -s BACKEND=kiss print-obj-dir AC_LIB="$ac_kiss" AC_DIR="$S5_AC_CLONE")"
touch -r "$aec_objdir/aec3_post.o" -A 01 "$S5_AC_CLONE/include/fast_math.h"
S5_LOG_FASTMATH="$SCRATCH_ROOT/s5/log_fastmath"
backdate_aecwav
make -s BACKEND=kiss all AC_LIB="$ac_kiss" AC_DIR="$S5_AC_CLONE" >"$S5_LOG_FASTMATH" 2>&1
grep -q "^aec_wav \[kiss\] ->" "$S5_LOG_FASTMATH" && pass "S5: touching the audio_common clone's include/fast_math.h recompiled AEC's aec3_post.o (relink observed)" \
  || fail "S5: aec_wav did NOT relink after touching fast_math.h (aec3_post.o likely did not recompile)"

# Header-only test 2: touch wav_io.h -> example objs (aec_wav.o, which
# #includes example/wav_io.h -> audio_common/include/wav_io.h) recompile.
touch -r "$aec_objdir/aec_wav.o" -A 01 "$S5_AC_CLONE/include/wav_io.h"
S5_LOG_WAVIO="$SCRATCH_ROOT/s5/log_wavio"
backdate_aecwav
make -s BACKEND=kiss all AC_LIB="$ac_kiss" AC_DIR="$S5_AC_CLONE" >"$S5_LOG_WAVIO" 2>&1
grep -q "^aec_wav \[kiss\] ->" "$S5_LOG_WAVIO" && pass "S5: touching the audio_common clone's include/wav_io.h recompiled AEC's aec_wav.o (relink observed)" \
  || fail "S5: aec_wav did NOT relink after touching wav_io.h (aec_wav.o likely did not recompile)"

# NE10 vendored header touch -> NE10 objects recompile (audio_common level).
# NE10_dsp.h (not the top-level NE10.h umbrella -- verified via NE10_fft.o's
# own -MD -MP .d fragment that NE10_fft.c pulls in NE10_dsp.h/NE10_types.h/
# NE10_macros.h transitively, but never NE10.h itself; only THIS repo's own
# fft_wrapper_ne10.c includes the top-level NE10.h directly). Same log-based
# verification (audio_common's own "audio_common [ne10] -> ..." echo).
ac_ne10_objdir="$(make -s -C "$S5_AC_CLONE" BACKEND=ne10 print-obj-dir)"
if [ -f "$ac_ne10_objdir/NE10_fft.o" ]; then
  touch -r "$ac_ne10_objdir/NE10_fft.o" -A 01 "$S5_AC_CLONE/lib/ne10/inc/NE10_dsp.h"
else
  # Cold start (no prior ne10 build under this exact objdir -- in the fresh
  # clone this is the NORMAL path, since only kiss has been built here so
  # far): no reference artifact exists yet to bump from, so just touch the
  # header directly -- still mtime-only, still content-neutral, and the
  # subsequent first ne10 build of the clone must archive regardless.
  touch "$S5_AC_CLONE/lib/ne10/inc/NE10_dsp.h"
fi
S5_LOG_NE10H="$SCRATCH_ROOT/s5/log_ne10h"
make -s -C "$S5_AC_CLONE" BACKEND=ne10 lib >"$S5_LOG_NE10H" 2>&1
grep -q "^audio_common \[ne10\] ->" "$S5_LOG_NE10H" && pass "S5: touching a vendored NE10 header (NE10_dsp.h) recompiled NE10_fft.o (archive rebuilt)" \
  || fail "S5: NE10_fft.o did NOT recompile after touching lib/ne10/inc/NE10_dsp.h (no archive-rebuild line in the log)"

# A->B->A at the AEC level -> nm shows backend-appropriate FFT symbols each
# time (never a stale symbol set from the OTHER backend's link).
ac_kiss="$(make -s -C "$S5_AC_CLONE" BACKEND=kiss print-lib-path)"
make -s BACKEND=kiss all AC_LIB="$ac_kiss" AC_DIR="$S5_AC_CLONE" >/dev/null
bd_kiss="$(make -s BACKEND=kiss print-bin-dir AC_LIB="$ac_kiss" AC_DIR="$S5_AC_CLONE")"
ac_ne10="$(make -s -C "$S5_AC_CLONE" BACKEND=ne10 print-lib-path)"
make -s BACKEND=ne10 all AC_LIB="$ac_ne10" AC_DIR="$S5_AC_CLONE" >/dev/null
bd_ne10="$(make -s BACKEND=ne10 print-bin-dir AC_LIB="$ac_ne10" AC_DIR="$S5_AC_CLONE")"
ac_kiss2="$(make -s -C "$S5_AC_CLONE" BACKEND=kiss print-lib-path)"
make -s BACKEND=kiss all AC_LIB="$ac_kiss2" AC_DIR="$S5_AC_CLONE" >/dev/null
bd_kiss2="$(make -s BACKEND=kiss print-bin-dir AC_LIB="$ac_kiss2" AC_DIR="$S5_AC_CLONE")"

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
mkscratch s7p
# round-6: ALLOW_DIRTY_PUBLISH=1 on every publish call in this script -- a
# dev tree is legitimately dirty (these very review changes are
# uncommitted), and the policy itself is exercised on purpose in S17.
# round-7: ALLOW_UNTRACKED_PUBLISH=1 rides along on every one of those same
# calls (EXCEPT S17's own policy sub-cases, which deliberately test each
# knob in isolation) -- the untracked dimension is a NEW default refusal,
# and a dev tree can legitimately carry untracked files that have nothing
# to do with what each of these scenarios is actually testing. DIST_ROOT
# stays a throwaway scratch path (round-5 review P1): the real dist/ is
# never read, written, or removed.
S7P_DIST="$SCRATCH_ROOT/s7p/dist"

S7P_LOG_FIRST="$SCRATCH_ROOT/s7p/log_first"
make -s BACKEND=kiss publish DIST_ROOT="$S7P_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >"$S7P_LOG_FIRST" 2>&1
grep -q "(attested: attest-" "$S7P_LOG_FIRST" && pass "S7p: kiss publish success line ends with '(attested: <name>)'" \
  || fail "S7p: kiss publish success line missing '(attested: <name>)'"

kiss_current_target="$(readlink "$S7P_DIST/kiss/current" || true)"
[ -n "$kiss_current_target" ] && [ -d "$S7P_DIST/kiss/$kiss_current_target" ] && \
  pass "S7p: kiss publish -- current symlink resolves to a real release dir" \
  || fail "S7p: kiss publish -- current symlink broken or missing"

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
make -s BACKEND=ne10 publish DIST_ROOT="$S7P_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >/dev/null
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
S7P_LOG_A="$SCRATCH_ROOT/s7p/log_a"
S7P_LOG_B="$SCRATCH_ROOT/s7p/log_b"
( make -s BACKEND=kiss publish DIST_ROOT="$S7P_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >"$S7P_LOG_A" 2>&1 ) & cp1=$!
( make -s BACKEND=kiss publish DIST_ROOT="$S7P_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >"$S7P_LOG_B" 2>&1 ) & cp2=$!
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
echo "# S8: CFG_SIG collision guard (round-6: scratch-side)"
echo "############################################################"
cd "$AC_DIR"
mkscratch s8
S8_OBJ_ROOT="$SCRATCH_ROOT/s8/obj"
S8_BIN_ROOT="$SCRATCH_ROOT/s8/bin"

# Fresh scratch build of the REAL worktree's normal kiss config -- exercises
# the CURRENT (possibly-uncommitted) Makefile without ever touching the real
# obj/ tree (round-6 review P1: the round-5 script mutated the REAL obj dir
# here -- the actual finding this rewrite fixes).
make -s BACKEND=kiss OBJ_ROOT="$S8_OBJ_ROOT" BIN_ROOT="$S8_BIN_ROOT" lib >/dev/null
s8_objdir="$(make -s BACKEND=kiss OBJ_ROOT="$S8_OBJ_ROOT" BIN_ROOT="$S8_BIN_ROOT" print-obj-dir)"
s8_lib="$(make -s BACKEND=kiss OBJ_ROOT="$S8_OBJ_ROOT" BIN_ROOT="$S8_BIN_ROOT" print-lib-path)"
manifest="$s8_objdir/config.manifest"

real_kiss_objdir="$(make -s BACKEND=kiss print-obj-dir)"
real_kiss_lib="$(make -s BACKEND=kiss print-lib-path)"
real_manifest="$real_kiss_objdir/config.manifest"
real_manifest_sha_before="$(file_sha "$real_manifest")"
real_manifest_mtime_before="$(mtime "$real_manifest")"

echo "CORRUPTED PAYLOAD (test_build_isolation.sh S8)" > "$manifest"
s8_log="$SCRATCH_ROOT/s8/log"
if make BACKEND=kiss OBJ_ROOT="$S8_OBJ_ROOT" BIN_ROOT="$S8_BIN_ROOT" lib >"$s8_log" 2>&1; then
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

# Knob byte-neutrality: OBJ_ROOT/BIN_ROOT is deliberately NOT in CFG_SIG (a
# placement knob, not a build-content knob -- see the Makefile). We do NOT
# compare whole-archive bytes: ar embeds each member's mtime in the archive
# header, so two freshly-built archives (real vs scratch, built at different
# wall-clock times) differ byte-for-byte even though every member is
# content-identical.
#
# round-7 review P1a: the old check globbed the REAL obj dir (`for o in
# "$real_kiss_objdir"/*.o`) and demanded a scratch counterpart for every
# object found -- but the real obj dir can legitimately carry EXTRA test
# objects (test_wav_io.o, test_wav_writer_ubsan.o, ... from a prior `make
# test_wav`/`make test-wav-ubsan` in that same keyed config) that the
# scratch `make lib` here never builds, so an otherwise-clean tip that had
# once run the WAV tests false-FAILed. Compared at the ARCHIVE-MEMBER level
# instead: the archives are what `lib` DELIVERS, and their member sets are
# identical by construction of the (a) name-list check below. Per-member
# content is compared by iterating the real archive's own `ar -t` list with
# `while IFS= read -r m` -- NEVER `for m in $(ar -t ...)`: macOS BSD ar has
# a pseudo-member literally named `__.SYMDEF SORTED` whose embedded space
# would word-split into two bogus tokens under a bare `for`. `__.SYMDEF*`
# members are skipped (the symbol-table pseudo-member's presence is covered
# by the name-list check; GNU ar doesn't even list it, so no fixed member
# count is assumed).
neutral_ok=1
while IFS= read -r m; do
  [ -n "$m" ] || continue
  case "$m" in
    __.SYMDEF*) continue ;;
  esac
  cmp -s <(ar -p "$real_kiss_lib" "$m") <(ar -p "$s8_lib" "$m") || neutral_ok=0
done < <(ar -t "$real_kiss_lib")
[ "$neutral_ok" -eq 1 ] && pass "S8: OBJ_ROOT/BIN_ROOT is knob-neutral -- every real archive member is byte-identical in the scratch archive" \
  || fail "S8: a real kiss archive member's content differs (or is missing) in the scratch archive -- OBJ_ROOT/BIN_ROOT is not build-content-neutral"

[ "$(ar -t "$s8_lib" | sort)" = "$(ar -t "$real_kiss_lib" | sort)" ] && \
  pass "S8: ar -t member name-lists of scratch vs real kiss archive are identical" \
  || fail "S8: ar -t member name-lists differ between scratch and real kiss archive"

echo "############################################################"
echo "# S9: command-line override rejection (round-4 review P1-1)"
echo "############################################################"
cd "$AC_DIR"
mkscratch s9
S9_OBJ_ROOT="$SCRATCH_ROOT/s9/obj"
S9_BIN_ROOT="$SCRATCH_ROOT/s9/bin"

# Plain baseline (no BACKEND= given -- whatever this Makefile auto-detects
# from the compiler is fine; only used below to prove EXTRA_CFLAGS lands
# somewhere DIFFERENT, never compared against a hardcoded backend). Both
# queries share the SAME scratch OBJ_ROOT/BIN_ROOT so the comparison isolates
# the CFG_SIG effect of EXTRA_CFLAGS, not an incidental root-path difference.
baseline_objdir="$(make -s OBJ_ROOT="$S9_OBJ_ROOT" BIN_ROOT="$S9_BIN_ROOT" print-obj-dir)"

# CFLAGS=/CXXFLAGS=/LDFLAGS=/FP_POLICY= set on the command line must be
# rejected at PARSE time (before any obj/bin dir is even created) with a
# message mentioning "cannot be overridden" -- these variables are internal
# (this Makefile's own +=/:= appends to them would be silently defeated by a
# command-line origin, per GNU Make semantics); EXTRA_CFLAGS/EXTRA_LDFLAGS are
# the supported hook instead.
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
echo "# S10: archive freshness -- stale-member removal (round-6: scratch-side)"
echo "############################################################"
cd "$AC_DIR"
mkscratch s10
S10_OBJ_ROOT="$SCRATCH_ROOT/s10/obj"
S10_BIN_ROOT="$SCRATCH_ROOT/s10/bin"

make -s BACKEND=kiss OBJ_ROOT="$S10_OBJ_ROOT" BIN_ROOT="$S10_BIN_ROOT" lib >/dev/null
kiss_lib="$(make -s BACKEND=kiss OBJ_ROOT="$S10_OBJ_ROOT" BIN_ROOT="$S10_BIN_ROOT" print-lib-path)"

real_kiss_lib="$(make -s BACKEND=kiss print-lib-path)"
real_lib_sha_before="$(file_sha "$real_kiss_lib")"
real_lib_mtime_before="$(mtime "$real_kiss_lib")"

# Inject a foreign object directly into the SCRATCH delivered archive
# (simulating what an old `ar rc` onto an EXISTING archive would leave
# behind forever for a source file that's since been removed from SRCS).
cat > "$SCRATCH_ROOT/s10/s10_foreign.c" <<'EOF'
int s10_foreign_symbol(void) { return 42; }
EOF
cc -c -o "$SCRATCH_ROOT/s10/s10_foreign.o" "$SCRATCH_ROOT/s10/s10_foreign.c"
ar r "$kiss_lib" "$SCRATCH_ROOT/s10/s10_foreign.o"
ar -t "$kiss_lib" | grep -q '^s10_foreign\.o$' && pass "S10: foreign object successfully injected into the SCRATCH archive (pre-condition)" \
  || fail "S10: failed to inject foreign object into scratch archive (pre-condition broken -- cannot test freshness)"

# A normal rebuild (source touched, config unchanged) must produce a FRESH
# archive (rm -f $@.tmp; ar rc $@.tmp $(BE_OBJS); ranlib; mv -f) -- never
# `ar rc` onto the existing archive -- so the foreign member must be GONE
# afterward, while hpf.o (a real, current source) must still be present.
#
# round-6 review: deterministic re-archive trigger replaces `sleep 1; touch
# src/hpf.c`. Backdate the SCRATCH archive itself to a fixed date well in
# the past -- older than every one of its member .o files (all built just
# now, real "current" mtimes) -- so make's own dependency check ($(LIB):
# $(BE_OBJS)) sees the archive as stale and re-archives from scratch on the
# very next build. Touches NOTHING real: not a member .o, not any tracked
# source.
touch -t 202001010000 "$kiss_lib"
make -s BACKEND=kiss OBJ_ROOT="$S10_OBJ_ROOT" BIN_ROOT="$S10_BIN_ROOT" lib >/dev/null

ar -t "$kiss_lib" | grep -q '^s10_foreign\.o$' && fail "S10: stale foreign member s10_foreign.o SURVIVED a fresh archive rebuild (ar rc onto an existing archive?)" \
  || pass "S10: stale foreign member s10_foreign.o REMOVED by the fresh-archive rebuild"
ar -t "$kiss_lib" | grep -q '^hpf\.o$' && pass "S10: rebuilt archive still contains hpf.o" \
  || fail "S10: rebuilt archive missing hpf.o"

# SRCS= (the sorted source list) participates in CFG_SIG_PAYLOAD, visible
# verbatim in this config's config.manifest.
s10_objdir="$(make -s BACKEND=kiss OBJ_ROOT="$S10_OBJ_ROOT" BIN_ROOT="$S10_BIN_ROOT" print-obj-dir)"
manifest="$s10_objdir/config.manifest"
if grep -q 'SRCS=' "$manifest" && grep -q 'src/hpf\.c' "$manifest"; then
  pass "S10: config.manifest's SRCS= entry participates in CFG_SIG and lists src/hpf.c"
else
  fail "S10: config.manifest missing SRCS= or src/hpf.c reference ($manifest)"
fi

real_lib_sha_after="$(file_sha "$real_kiss_lib")"
real_lib_mtime_after="$(mtime "$real_kiss_lib")"
[ "$real_lib_sha_before" = "$real_lib_sha_after" ] && [ "$real_lib_mtime_before" = "$real_lib_mtime_after" ] && \
  pass "S10: the REAL kiss archive sha+mtime unchanged across the whole scenario" \
  || fail "S10: the REAL kiss archive CHANGED during S10 (scratch isolation failed)"

echo "############################################################"
echo "# S11: publish immutability / content-addressing"
echo "# (round-4 review P2-2 -> round-5 v4 semantics -> round-6 P2-2 ATTEST)"
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
mkscratch s11a
S11A_DIST="$SCRATCH_ROOT/s11a/dist"
make -s BACKEND=kiss publish DIST_ROOT="$S11A_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >/dev/null
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

snap_before="$(release_snapshot "$S11A_DIST/kiss/$id1")"

# round-6 review: no sleep. The one-event-one-file ATTEST install
# (--excl-install + retry-with-next-<NNN>) disambiguates a same-second
# republish via the numeric suffix instead of requiring a distinct
# <utcstamp> -- the round-5 version of this scenario needed `sleep 1` here
# so the two publishes landed in different UTC seconds; that dependency is
# gone (S16 stress-tests the same-second case directly, forcing 20 publishes
# into ONE literal second via ATTEST_STAMP=).
S11A_LOG="$SCRATCH_ROOT/s11a/republish.log"
make -s BACKEND=kiss publish DIST_ROOT="$S11A_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >"$S11A_LOG" 2>&1
grep -q "byte-verified" "$S11A_LOG" && pass "S11a: identical-content republish reports 'byte-verified'" \
  || fail "S11a: identical-content republish did NOT report 'byte-verified'"

id1_again="$(readlink "$S11A_DIST/kiss/current" || true)"
[ "$id1_again" = "$id1" ] && pass "S11a: current still points at id1 after identical-content republish" \
  || fail "S11a: current CHANGED after identical-content republish ($id1 -> $id1_again)"

snap_after="$(release_snapshot "$S11A_DIST/kiss/$id1")"
[ "$snap_before" = "$snap_after" ] && pass "S11a: release dir + artifact/MANIFEST mtimes UNCHANGED by identical-content republish" \
  || fail "S11a: release dir/artifact mtimes CHANGED by identical-content republish"

n_attest2="$(count_attest "$S11A_DIST/kiss/$id1")"
[ "$n_attest2" -eq 2 ] && pass "S11a: ATTEST/ has exactly 2 attest-*.txt files after the (same-second-capable) republish" \
  || fail "S11a: ATTEST/ has $n_attest2 attest-*.txt file(s) after the republish (expected 2)"

attest2="$(find "$S11A_DIST/kiss/$id1/ATTEST" -name 'attest-*.txt' | grep -v -F "$attest1" || true)"
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
make -s BACKEND=kiss publish DIST_ROOT="$S11B_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >/dev/null
idb="$(readlink "$S11B_DIST/kiss/current" || true)"
printf 'X' >> "$S11B_DIST/kiss/$idb/MANIFEST.txt"
S11B_LOG="$SCRATCH_ROOT/s11b/log"
if make -s BACKEND=kiss publish DIST_ROOT="$S11B_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >"$S11B_LOG" 2>&1; then
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
mkscratch s11c
SANDBOX="$SCRATCH_ROOT/s11c/ac"
mkdir -p "$SANDBOX"
tar -c -C "$AC_DIR" --exclude obj --exclude bin --exclude dist --exclude .git . | tar -x -C "$SANDBOX"
S11C_DIST="$SCRATCH_ROOT/s11c/dist"

# round-7: the tar copy has no .git (excluded above), and "no git identity"
# is on its way to becoming an UNCONDITIONAL publish refusal (no escape
# hatch admits it) -- so the sandbox gets its own REAL, disposable git
# identity: `git init` + one commit, entirely inside $SCRATCH_ROOT (same
# rationale as adopt_worktree_clone's throwaway history -- never touches the
# real repo's .git in any way). The content edit below then makes it
# ordinarily tracked-dirty, which ALLOW_DIRTY_PUBLISH=1 admits;
# ALLOW_UNTRACKED_PUBLISH=1 rides along uniformly (build byproducts under
# the sandbox's own obj/bin are gitignored via the copied .gitignore, but
# uniformity keeps this scenario about content-addressing, not git policy).
git -C "$SANDBOX" init -q
git -C "$SANDBOX" add -A
git -C "$SANDBOX" -c user.email=scratch@example.invalid -c user.name="scratch sandbox" \
  commit -q -m "scratch: S11c sandbox baseline (disposable repo under scratch only)"
make -C "$SANDBOX" -s BACKEND=kiss publish DIST_ROOT="$S11C_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >/dev/null
sid1="$(readlink "$S11C_DIST/kiss/current" || true)"
[ -n "$sid1" ] && pass "S11c: sandbox publish #1 -- current resolves ($sid1)" \
  || fail "S11c: sandbox publish #1 -- current symlink broken or missing"

# round-6 review: resolve the sandbox's OWN hpf.o AND archive paths before
# the edit. Force hpf.c's mtime to hpf.o's CURRENT mtime + 1s (BSD
# touch -r/-A) right after appending the real content change below, exactly
# as S5 does -- needed so make recompiles hpf.o at all (GNU Make 3.81
# truncates its prerequisite-newer-than-target check to whole seconds, and
# an append immediately after the sandbox's first publish can otherwise land
# in the very SAME integer second as hpf.o's existing mtime).
#
# That alone is NOT sufficient, though (found empirically): the SAME
# whole-second truncation applies one level up too, between the freshly
# recompiled hpf.o and the ARCHIVE's own prior build -- a real, later hpf.o
# compile whose timestamp nonetheless truncates to the SAME second as the
# archive's last build leaves make believing the archive is still up to
# date, so it never re-runs `ar rc`, and the archive (what publish actually
# stages) stays byte-identical even though hpf.o now contains a genuinely
# different symbol. So the archive itself (a sandbox artifact, not one of
# audio_common's real obj/bin under OBJ_ROOT/BIN_ROOT -- freely mutable here,
# same rationale as S5's aec_wav backdate) is ALSO backdated by a safe
# margin before the edit, guaranteeing make sees it as older than whatever
# real "now" the subsequent `ar rc` lands on.
sandbox_kiss_lib="$(make -C "$SANDBOX" -s BACKEND=kiss print-lib-path)"
sandbox_hpf_o="$(make -C "$SANDBOX" -s BACKEND=kiss print-obj-dir)/hpf.o"
touch -A -0500 "$sandbox_kiss_lib"
echo "void s11_isolation_probe(void) {}" >> "$SANDBOX/src/hpf.c"
touch -r "$sandbox_hpf_o" -A 01 "$SANDBOX/src/hpf.c"
make -C "$SANDBOX" -s BACKEND=kiss publish DIST_ROOT="$S11C_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >/dev/null
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

# The REAL repo's src/hpf.c must be untouched: compare against the shasum
# captured at the very start of S11, not `git diff --quiet` (which would
# wrongly assume the file was clean to begin with).
HPF_SHA_AFTER_S11="$(file_sha "$AC_DIR/src/hpf.c")"
[ "$HPF_SHA_BEFORE_S11" = "$HPF_SHA_AFTER_S11" ] && pass "S11c: the REAL repo's src/hpf.c is byte-identical to its pre-scenario state" \
  || fail "S11c: the REAL repo's src/hpf.c CHANGED during S11 (sandbox isolation failed)"

echo "############################################################"
echo "# S12: toolchain coherence guard (round-4 review P1-2; round-7:"
echo "#      scratch OBJ_ROOT/BIN_ROOT)"
echo "############################################################"
cd "$AC_DIR"
mkscratch s12
# round-7 review P2b: the three shim builds below used to build into the
# REAL obj/bin -- and the shim's scratch path lands in CFG_SIG (CXX= is in
# the payload, and $SCRATCH_ROOT is fresh every run), so EVERY run of this
# suite minted a brand-new orphan keyed directory pair in the real obj/+bin/
# (200+ had accumulated). All three now build under a scratch
# OBJ_ROOT/BIN_ROOT instead; the guard under test doesn't care where the
# objects land.
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

S12_LOG3="$SCRATCH_ROOT/s12/log3"
if make BACKEND=kiss CXX="$SHIM" OBJ_ROOT="$S12_OBJ_ROOT" BIN_ROOT="$S12_BIN_ROOT" lib >"$S12_LOG3" 2>&1; then
  pass "S12: BACKEND=kiss with the mismatched CXX shim SUCCEEDS (guard is ne10-only)"
else
  fail "S12: BACKEND=kiss with the mismatched CXX shim FAILED (guard should never run for kiss)"
  cat "$S12_LOG3" >&2
fi

echo "############################################################"
echo "# S13: atomic \`current\` symlink swap hammer + --excl-install"
echo "# (round-5 review P2; round-6 adds the --excl-install checks)"
echo "############################################################"
mkscratch s13
cc -O2 -o "$SCRATCH_ROOT/s13/swapln" "$AC_DIR/tools/atomic_symlink_swap.c"
mkdir -p "$SCRATCH_ROOT/s13/d/relA" "$SCRATCH_ROOT/s13/d/relB"
"$SCRATCH_ROOT/s13/swapln" relA "$SCRATCH_ROOT/s13/d/current"

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

echo "--- S13: --excl-install no-clobber mode (round-6 review P2-2) --------"
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
echo "#       (round-5 review P1; round-6: scratch OBJ_ROOT/BIN_ROOT)"
echo "############################################################"
cd "$AC_DIR"
mkscratch s14
S14_DIST="$SCRATCH_ROOT/s14/dist"
# round-6 review P1: one shared scratch OBJ_ROOT/BIN_ROOT pair for BOTH
# racers -- the lock still serialises the winner's build (this is exactly
# what the scenario tests), and the real obj/bin never see this throwaway
# -DS14_LOCK_PROBE config either way.
S14_OBJ_ROOT="$SCRATCH_ROOT/s14/obj"
S14_BIN_ROOT="$SCRATCH_ROOT/s14/bin"

# A fresh, never-before-built config (BACKEND=kiss + a probe define unique to
# this scenario) -- the winner's critical section (a real compile+link+stage,
# not a no-op) takes long enough that two callers launched back-to-back
# genuinely contend for the lock, making "exactly one winner" a reliable
# assertion rather than a scheduling-dependent guess (contrast with S7p's
# already-built-config concurrency check, which stays deliberately lenient).
S14_LOG_A="$SCRATCH_ROOT/s14/log_a"; S14_LOG_B="$SCRATCH_ROOT/s14/log_b"
S14_RC_A="$SCRATCH_ROOT/s14/rc_a"; S14_RC_B="$SCRATCH_ROOT/s14/rc_b"
# `rc=0; cmd || rc=$?; ...` (not `cmd; echo $? >file`): under `set -e`, a
# bare failing command followed by `;` would abort this subshell BEFORE the
# echo ever ran. A command that is not the final one in a `||` list is
# exempt from triggering errexit, so this form reliably captures the make
# invocation's real exit status into its rc file even when it fails.
( rc=0; make -s BACKEND=kiss EXTRA_CFLAGS=-DS14_LOCK_PROBE OBJ_ROOT="$S14_OBJ_ROOT" BIN_ROOT="$S14_BIN_ROOT" publish DIST_ROOT="$S14_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >"$S14_LOG_A" 2>&1 || rc=$?; echo "$rc" > "$S14_RC_A" ) &
p1=$!
( rc=0; make -s BACKEND=kiss EXTRA_CFLAGS=-DS14_LOCK_PROBE OBJ_ROOT="$S14_OBJ_ROOT" BIN_ROOT="$S14_BIN_ROOT" publish DIST_ROOT="$S14_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >"$S14_LOG_B" 2>&1 || rc=$?; echo "$rc" > "$S14_RC_B" ) &
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

echo "############################################################"
echo "# S15: make -n/-q/-t publish zero-write (round-6 P2-3; round-7 P3)"
echo "############################################################"
cd "$AC_DIR"
mkscratch s15
# round-7 review P3: on top of the "no paths created" checks, each of
# -n/-q/-t must also leave an ALREADY-EXISTING delivered artifact byte- and
# mtime-untouched -- `-t` (touch mode) is exactly the flag whose standard
# semantics WOULD bump artifact mtimes if the publish driver recursed under
# it (the round-6 shape did; round-7's driver makes -t a one-line no-op note
# + exit 0 without recursion, asserted below). The reference artifact is the
# REAL, default-rooted $be archive built by S1 -- the strongest thing this
# suite owns to protect. All three modes are ZERO-WRITE.
for be in kiss ne10; do
  S15_DIST="$SCRATCH_ROOT/s15/${be}_nx"
  S15_OBJ="$SCRATCH_ROOT/s15/${be}_no"
  S15_BIN="$SCRATCH_ROOT/s15/${be}_nb"
  S15_LOG="$SCRATCH_ROOT/s15/${be}_log"

  # Ensure the normal $be lib exists (a no-op after S1's builds), then
  # record its size+mtime+sha as the zero-write reference.
  S15_REAL_LIB="$(make -s BACKEND="$be" print-lib-path)"
  make -s BACKEND="$be" lib >/dev/null
  s15_ref_stat="$(stat -f '%m %z' "$S15_REAL_LIB")"
  s15_ref_sha="$(file_sha "$S15_REAL_LIB")"

  rc=0
  make -n BACKEND="$be" DIST_ROOT="$S15_DIST" OBJ_ROOT="$S15_OBJ" BIN_ROOT="$S15_BIN" publish >"$S15_LOG.n" 2>&1 || rc=$?
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
  make -q BACKEND="$be" DIST_ROOT="$S15_DIST" OBJ_ROOT="$S15_OBJ" BIN_ROOT="$S15_BIN" publish >"$S15_LOG.q" 2>&1 || rc=$?
  [ "$rc" -ne 0 ] && pass "S15[$be]: make -q publish exits NONZERO (rc=$rc)" || fail "S15[$be]: make -q publish exited 0 (expected nonzero)"
  if [ ! -e "$S15_DIST" ] && [ ! -e "$S15_OBJ" ] && [ ! -e "$S15_BIN" ]; then
    pass "S15[$be]: make -q publish created NONE of DIST_ROOT/OBJ_ROOT/BIN_ROOT"
  else
    fail "S15[$be]: make -q publish left behind a path (dist=$([ -e "$S15_DIST" ] && echo yes || echo no) obj=$([ -e "$S15_OBJ" ] && echo yes || echo no) bin=$([ -e "$S15_BIN" ] && echo yes || echo no))"
  fi
  [ "$(stat -f '%m %z' "$S15_REAL_LIB")" = "$s15_ref_stat" ] && [ "$(file_sha "$S15_REAL_LIB")" = "$s15_ref_sha" ] && \
    pass "S15[$be]: make -q publish is zero-write against the existing normal $be archive (size+mtime+sha unchanged)" \
    || fail "S15[$be]: make -q publish CHANGED the normal $be archive"

  # round-7: -t must exit 0 AND print the explicit no-op note (the driver
  # no longer recurses under -t at all -- see the Makefile's dry-run guard).
  rc=0
  make -t BACKEND="$be" DIST_ROOT="$S15_DIST" OBJ_ROOT="$S15_OBJ" BIN_ROOT="$S15_BIN" publish >"$S15_LOG.t" 2>&1 || rc=$?
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
# Round-7 follow-up: a combined `make -nt publish` used to take the -n branch
# and recurse; the child make then saw BOTH flags and GNU make applies REAL
# touch semantics down the _publish_impl prerequisite chain (reproduced) --
# the driver now checks t FIRST. Exercised against a scratch-root build whose
# archive is deliberately BACKDATED below its own objects: exactly the state
# a leaked -t would "fix" by touching. Scratch-side only -- the real tree's
# artifacts are covered by the S1 snapshot guard and must never be backdated.
mkscratch s15b
S15B_OBJ="$SCRATCH_ROOT/s15b/obj"; S15B_BIN="$SCRATCH_ROOT/s15b/bin"; S15B_DIST="$SCRATCH_ROOT/s15b/nx"
make -s BACKEND=kiss OBJ_ROOT="$S15B_OBJ" BIN_ROOT="$S15B_BIN" lib >/dev/null
s15b_lib="$(make -s BACKEND=kiss OBJ_ROOT="$S15B_OBJ" BIN_ROOT="$S15B_BIN" print-lib-path)"
touch -t 202001010000 "$s15b_lib"
s15b_ref="$(stat -f '%m %z' "$s15b_lib")"
for combo in "-nt" "-n -t" "-tq" "-nq" "-nqt"; do
  rm -rf "$S15B_DIST"
  rc=0; make $combo BACKEND=kiss OBJ_ROOT="$S15B_OBJ" BIN_ROOT="$S15B_BIN" DIST_ROOT="$S15B_DIST" publish >/dev/null 2>&1 || rc=$?
  if [ ! -e "$S15B_DIST" ] && [ "$(stat -f '%m %z' "$s15b_lib")" = "$s15b_ref" ]; then
    pass "S15b[$combo]: zero-write (no DIST_ROOT; stale scratch archive untouched; rc=$rc informational)"
  else
    fail "S15b[$combo]: WROTE something (dist=$([ -e "$S15B_DIST" ] && echo yes || echo no), archive stat now '$(stat -f '%m %z' "$s15b_lib")' vs '$s15b_ref')"
  fi
done

echo "############################################################"
echo "# S16: ATTEST uniqueness under forced same-second collisions"
echo "#       (round-6 review P2-2)"
echo "############################################################"
cd "$AC_DIR"
mkscratch s16
S16_DIST="$SCRATCH_ROOT/s16/dist"
make -s BACKEND=kiss publish DIST_ROOT="$S16_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 >/dev/null
s16_id="$(readlink "$S16_DIST/kiss/current" || true)"
s16_attest_dir="$S16_DIST/kiss/$s16_id/ATTEST"

s16_before_list="$(find "$s16_attest_dir" -name 'attest-*.txt' | sort)"
s16_before_snap="$SCRATCH_ROOT/s16/before_snap.txt"
: > "$s16_before_snap"
for f in $s16_before_list; do
  [ -n "$f" ] || continue
  printf '%s %s %s %s\n' "$f" "$(stat -f '%i' "$f")" "$(mtime "$f")" "$(file_sha "$f")" >> "$s16_before_snap"
done

for i in $(seq 1 20); do
  make -s BACKEND=kiss publish DIST_ROOT="$S16_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 ATTEST_STAMP=20260715T999999Z >/dev/null
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
echo "# S17: dirty-checkout publish policy (round-6 review P2)"
echo "############################################################"
mkscratch s17
S17_CLONE="$SCRATCH_ROOT/s17/clone"
# round-7 review P1b: what used to be S17's own inline clone+overlay+
# UNCONDITIONAL commit (which exited 1 on a clean tip -- nothing staged --
# and killed the suite under `set -e`) is now the shared
# adopt_worktree_clone() helper, whose commit is conditional. See the
# helper's own comment for the full rationale (clean git identity that
# still reflects today's possibly-uncommitted Makefile content; disposable
# repo under $SCRATCH_ROOT only).
adopt_worktree_clone "$AC_DIR" "$S17_CLONE"

S17_OBJ_ROOT="$SCRATCH_ROOT/s17/obj"
S17_BIN_ROOT="$SCRATCH_ROOT/s17/bin"
S17_DIST="$SCRATCH_ROOT/s17/dist"

# attest_from_log(): the publish recipe's own success line ends with
# "(attested: <name>)" -- extracting the exact attest filename from THIS
# invocation's own log is unambiguous regardless of which release dir it
# landed in, unlike `find ATTEST -name 'attest-*.txt' | head -1` (round-6
# review, found while validating THIS script: a comment-only edit to hpf.c
# -- see S17b below -- produces byte-identical object code, so a
# deterministic archiver correctly reuses the SAME content-addressed release
# id as the clean publish; `find | head -1` then has TWO candidate attest
# files to choose from in that one release dir, in filesystem-listing order,
# not necessarily the one THIS publish just wrote).
attest_from_log() {
  local log="$1" name
  name="$(grep -o '(attested: [^)]*)' "$log" | sed -e 's/^(attested: //' -e 's/)$//' | head -1)"
  [ -n "$name" ] || return 1
  printf '%s\n' "$name"
}

echo "--- S17a: clean clone, DEFAULT publish -> must succeed ----------------"
S17_LOG_A="$SCRATCH_ROOT/s17/log_a"
if make -C "$S17_CLONE" -s BACKEND=kiss OBJ_ROOT="$S17_OBJ_ROOT" BIN_ROOT="$S17_BIN_ROOT" DIST_ROOT="$S17_DIST" publish >"$S17_LOG_A" 2>&1; then
  pass "S17a: default publish on a CLEAN clone succeeds"
else
  fail "S17a: default publish on a clean clone FAILED"
  cat "$S17_LOG_A" >&2
fi
sid_a="$(readlink "$S17_DIST/kiss/current" || true)"
attest_a_name="$(attest_from_log "$S17_LOG_A" || true)"
attest_a="$S17_DIST/kiss/$sid_a/ATTEST/$attest_a_name"
clone_head="$(git -C "$S17_CLONE" rev-parse HEAD)"
if [ -n "$attest_a_name" ] && [ -f "$attest_a" ] && grep -q "^git_dirty=0$" "$attest_a" && grep -q "^allow_dirty_publish=0$" "$attest_a" \
   && grep -q "^git_commit=$clone_head$" "$attest_a" && ! grep -q "^dirty_diff_sha256=" "$attest_a"; then
  pass "S17a: attestation shows git_dirty=0, allow_dirty_publish=0, git_commit matches clone HEAD, no dirty_diff_sha256="
else
  fail "S17a: attestation ($attest_a) does not match the expected clean-clone shape"
fi

echo "--- S17b: dirty the clone's disposable hpf.c -> default publish FAILS --"
# A disposable edit to the CLONE's own hpf.c (never the real repo's). This
# is intentionally comment-only text -- see attest_from_log()'s comment
# above for why that's fine: this scenario tests the DIRTY-PUBLISH POLICY
# (attestation fields), not content-addressing (S11c already covers that
# with a real code change), so whether the resulting object/archive bytes
# happen to change is irrelevant here.
echo "/* s17 disposable dirty probe */" >> "$S17_CLONE/src/hpf.c"
S17_LOG_B="$SCRATCH_ROOT/s17/log_b"
if make -C "$S17_CLONE" -s BACKEND=kiss OBJ_ROOT="$S17_OBJ_ROOT" BIN_ROOT="$S17_BIN_ROOT" DIST_ROOT="$S17_DIST" publish >"$S17_LOG_B" 2>&1; then
  fail "S17b: default publish on a DIRTY clone unexpectedly SUCCEEDED"
else
  grep -qi "publish refused" "$S17_LOG_B" && grep -qi "dirty" "$S17_LOG_B" && \
    pass "S17b: default publish on a dirty clone FAILS, mentioning 'publish refused' and 'dirty'" \
    || fail "S17b: publish failed but without the expected 'publish refused'/'dirty' wording"
fi

echo "--- S17c: same dirty clone + ALLOW_DIRTY_PUBLISH=1 -> succeeds --------"
S17_LOG_C="$SCRATCH_ROOT/s17/log_c"
if make -C "$S17_CLONE" -s BACKEND=kiss OBJ_ROOT="$S17_OBJ_ROOT" BIN_ROOT="$S17_BIN_ROOT" DIST_ROOT="$S17_DIST" ALLOW_DIRTY_PUBLISH=1 publish >"$S17_LOG_C" 2>&1; then
  pass "S17c: dirty clone + ALLOW_DIRTY_PUBLISH=1 succeeds"
else
  fail "S17c: dirty clone + ALLOW_DIRTY_PUBLISH=1 FAILED"
  cat "$S17_LOG_C" >&2
fi
sid_c="$(readlink "$S17_DIST/kiss/current" || true)"
attest_c_name="$(attest_from_log "$S17_LOG_C" || true)"
attest_c="$S17_DIST/kiss/$sid_c/ATTEST/$attest_c_name"
expected_ddiff="$(git -C "$S17_CLONE" diff --binary HEAD | shasum -a 256 | cut -d' ' -f1)"
if [ -n "$attest_c_name" ] && [ -f "$attest_c" ] && grep -q "^git_dirty=1$" "$attest_c" && grep -q "^allow_dirty_publish=1$" "$attest_c" \
   && grep -q "^dirty_diff_sha256=$expected_ddiff$" "$attest_c" && printf '%s' "$attest_c_name" | grep -q -- "-dirty-"; then
  pass "S17c: attestation shows git_dirty=1, allow_dirty_publish=1, matching dirty_diff_sha256=, and -dirty- in the filename"
else
  fail "S17c: attestation ($attest_c) does not match the expected dirty-publish shape"
fi

# --- round-7 review P2a: untracked-file policy (S17d-S17h) -------------------
# The untracked dimension is SEPARATE from the tracked-dirty one:
# ALLOW_DIRTY_PUBLISH covers tracked changes only (`git status --porcelain
# -uno`), ALLOW_UNTRACKED_PUBLISH covers untracked files, and the
# attestation records untracked_tree_sha256 over the untracked files' exact
# contents. These sub-cases assert the POLICY semantics (which knob admits
# what; the hash field's presence and that it CHANGES whenever the untracked
# bytes/symlink targets change) -- deliberately NOT the hash's internal
# record encoding, which is an implementation detail the assertions here
# must survive. Each sub-case runs in its own fresh clone (S17b/c left
# $S17_CLONE tracked-dirty, and these cases need exact control of both
# dimensions); builds use each clone's own default-rooted (clone-local,
# gitignored) obj/ and bin/.
echo "--- S17d: untracked file -> default refused; ALLOW_DIRTY alone refused; both admit --"
S17D_CLONE="$SCRATCH_ROOT/s17/clone_d"
adopt_worktree_clone "$AC_DIR" "$S17D_CLONE"
echo "int s17d_untracked_probe;" > "$S17D_CLONE/probe.c"

S17_LOG_D1="$SCRATCH_ROOT/s17/log_d1"
if make -C "$S17D_CLONE" -s BACKEND=kiss DIST_ROOT="$S17_DIST" publish >"$S17_LOG_D1" 2>&1; then
  fail "S17d: DEFAULT publish with an untracked file unexpectedly SUCCEEDED"
else
  grep -qi "publish refused" "$S17_LOG_D1" && grep -q "UNTRACKED" "$S17_LOG_D1" && \
    pass "S17d: default publish with an untracked file FAILS, mentioning 'publish refused' + UNTRACKED" \
    || { fail "S17d: publish failed but without the expected refusal/UNTRACKED wording"; cat "$S17_LOG_D1" >&2; }
fi

S17_LOG_D2="$SCRATCH_ROOT/s17/log_d2"
if make -C "$S17D_CLONE" -s BACKEND=kiss DIST_ROOT="$S17_DIST" ALLOW_DIRTY_PUBLISH=1 publish >"$S17_LOG_D2" 2>&1; then
  fail "S17d: ALLOW_DIRTY_PUBLISH=1 ALONE unexpectedly admitted an untracked file (the two dimensions must be separately gated)"
else
  grep -q "UNTRACKED" "$S17_LOG_D2" && \
    pass "S17d: ALLOW_DIRTY_PUBLISH=1 alone is STILL refused (untracked file present), mentioning UNTRACKED" \
    || { fail "S17d: ALLOW_DIRTY_PUBLISH=1-alone refusal did not mention UNTRACKED"; cat "$S17_LOG_D2" >&2; }
fi

S17_LOG_D3="$SCRATCH_ROOT/s17/log_d3"
if make -C "$S17D_CLONE" -s BACKEND=kiss DIST_ROOT="$S17_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish >"$S17_LOG_D3" 2>&1; then
  pass "S17d: ALLOW_DIRTY_PUBLISH=1 + ALLOW_UNTRACKED_PUBLISH=1 together succeed"
else
  fail "S17d: both knobs together FAILED"
  cat "$S17_LOG_D3" >&2
fi
sid_d="$(readlink "$S17_DIST/kiss/current" || true)"
attest_d_name="$(attest_from_log "$S17_LOG_D3" || true)"
attest_d="$S17_DIST/kiss/$sid_d/ATTEST/$attest_d_name"
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
if make -C "$S17D_CLONE" -s BACKEND=kiss DIST_ROOT="$S17_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish >"$S17_LOG_E" 2>&1; then
  pass "S17e: republish with different untracked bytes succeeds"
else
  fail "S17e: republish with different untracked bytes FAILED"
  cat "$S17_LOG_E" >&2
fi
sid_e="$(readlink "$S17_DIST/kiss/current" || true)"
attest_e_name="$(attest_from_log "$S17_LOG_E" || true)"
attest_e="$S17_DIST/kiss/$sid_e/ATTEST/$attest_e_name"
uhash_e="$(grep '^untracked_tree_sha256=' "$attest_e" 2>/dev/null | head -1 | cut -d= -f2)"
[ -n "$uhash_e" ] && [ -n "$uhash_d" ] && [ "$uhash_e" != "$uhash_d" ] && \
  pass "S17e: untracked_tree_sha256 DIFFERS from S17d's after changing probe.c's bytes -- two untracked source states can no longer share a provenance record" \
  || fail "S17e: untracked_tree_sha256 did NOT differ (d=$uhash_d e=$uhash_e)"

echo "--- S17f: untracked SYMLINK, then repointed -> hash differs each time ---------"
rm -f "$S17D_CLONE/probe.c"
ln -s "s17f-target-one" "$S17D_CLONE/probe.c"
S17_LOG_F1="$SCRATCH_ROOT/s17/log_f1"
if make -C "$S17D_CLONE" -s BACKEND=kiss DIST_ROOT="$S17_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish >"$S17_LOG_F1" 2>&1; then
  pass "S17f: publish with an untracked symlink succeeds"
else
  fail "S17f: publish with an untracked symlink FAILED"
  cat "$S17_LOG_F1" >&2
fi
sid_f1="$(readlink "$S17_DIST/kiss/current" || true)"
attest_f1_name="$(attest_from_log "$S17_LOG_F1" || true)"
attest_f1="$S17_DIST/kiss/$sid_f1/ATTEST/$attest_f1_name"
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
if make -C "$S17D_CLONE" -s BACKEND=kiss DIST_ROOT="$S17_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish >"$S17_LOG_F2" 2>&1; then
  pass "S17f: republish after repointing the symlink succeeds"
else
  fail "S17f: republish after repointing the symlink FAILED"
  cat "$S17_LOG_F2" >&2
fi
sid_f2="$(readlink "$S17_DIST/kiss/current" || true)"
attest_f2_name="$(attest_from_log "$S17_LOG_F2" || true)"
attest_f2="$S17_DIST/kiss/$sid_f2/ATTEST/$attest_f2_name"
uhash_f2="$(grep '^untracked_tree_sha256=' "$attest_f2" 2>/dev/null | head -1 | cut -d= -f2)"
[ -n "$uhash_f2" ] && [ "$uhash_f2" != "$uhash_f1" ] && \
  pass "S17f: untracked_tree_sha256 changes AGAIN after repointing only the symlink's target" \
  || fail "S17f: untracked_tree_sha256 did not change after repointing the symlink (f1=$uhash_f1 f2=$uhash_f2)"

echo "--- S17g: BUILD-RELEVANT untracked source (tracked Makefile edit wires it in) --"
# The exact attack the reviewer described: a TRACKED one-line Makefile edit
# extends COMMON_SRCS with an UNTRACKED source file -- the artifact now
# genuinely depends on untracked bytes the tracked diff alone cannot
# account for. Requires BOTH knobs (each alone must refuse, each naming its
# own dimension), and the resulting release must differ from the clean
# clone's (S17a's) release id, with the probe object visibly a member of
# the delivered archive.
S17G_CLONE="$SCRATCH_ROOT/s17/clone_g"
adopt_worktree_clone "$AC_DIR" "$S17G_CLONE"
echo "int s17g_probe_symbol(void) { return 17; }" > "$S17G_CLONE/src/s17g_probe.c"
sed -i.s17gbak 's|^COMMON_SRCS = src/hpf.c$|COMMON_SRCS = src/hpf.c src/s17g_probe.c|' "$S17G_CLONE/Makefile"
rm -f "$S17G_CLONE/Makefile.s17gbak"
grep -q 's17g_probe\.c' "$S17G_CLONE/Makefile" || fail "S17g: Makefile seam edit did not land (COMMON_SRCS line changed shape?)"

S17_LOG_G1="$SCRATCH_ROOT/s17/log_g1"
if make -C "$S17G_CLONE" -s BACKEND=kiss DIST_ROOT="$S17_DIST" ALLOW_UNTRACKED_PUBLISH=1 publish >"$S17_LOG_G1" 2>&1; then
  fail "S17g: ALLOW_UNTRACKED_PUBLISH=1 ALONE unexpectedly admitted the tracked Makefile edit"
else
  grep -qi "publish refused" "$S17_LOG_G1" && grep -qi "dirty" "$S17_LOG_G1" && \
    pass "S17g: ALLOW_UNTRACKED_PUBLISH=1 alone is refused (tracked-dirty Makefile)" \
    || { fail "S17g: untracked-knob-only refusal did not name the tracked-dirty dimension"; cat "$S17_LOG_G1" >&2; }
fi

S17_LOG_G2="$SCRATCH_ROOT/s17/log_g2"
if make -C "$S17G_CLONE" -s BACKEND=kiss DIST_ROOT="$S17_DIST" ALLOW_DIRTY_PUBLISH=1 publish >"$S17_LOG_G2" 2>&1; then
  fail "S17g: ALLOW_DIRTY_PUBLISH=1 ALONE unexpectedly admitted the untracked build input"
else
  grep -q "UNTRACKED" "$S17_LOG_G2" && \
    pass "S17g: ALLOW_DIRTY_PUBLISH=1 alone is refused (untracked source present)" \
    || { fail "S17g: dirty-knob-only refusal did not mention UNTRACKED"; cat "$S17_LOG_G2" >&2; }
fi

S17_LOG_G3="$SCRATCH_ROOT/s17/log_g3"
if make -C "$S17G_CLONE" -s BACKEND=kiss DIST_ROOT="$S17_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish >"$S17_LOG_G3" 2>&1; then
  pass "S17g: both knobs together succeed"
else
  fail "S17g: both knobs together FAILED"
  cat "$S17_LOG_G3" >&2
fi
sid_g="$(readlink "$S17_DIST/kiss/current" || true)"
[ -n "$sid_g" ] && [ -n "$sid_a" ] && [ "$sid_g" != "$sid_a" ] && \
  pass "S17g: release id DIFFERS from the clean clone's (S17a's $sid_a -> $sid_g) -- the untracked source changed the deliverable" \
  || fail "S17g: release id did NOT differ from the clean baseline (sid_a=$sid_a sid_g=$sid_g)"
ar -t "$S17_DIST/kiss/$sid_g/libaudio_common.a" 2>/dev/null | grep -q '^s17g_probe\.o$' && \
  pass "S17g: the published archive contains s17g_probe.o -- direct evidence the untracked source landed in the artifact" \
  || fail "S17g: published archive does not list s17g_probe.o"
attest_g_name="$(attest_from_log "$S17_LOG_G3" || true)"
attest_g="$S17_DIST/kiss/$sid_g/ATTEST/$attest_g_name"
if [ -n "$attest_g_name" ] && grep -q "^git_dirty=1$" "$attest_g" && grep -q "^git_untracked=1$" "$attest_g" \
   && grep -q "^untracked_tree_sha256=" "$attest_g" && grep -q "^dirty_diff_sha256=" "$attest_g"; then
  pass "S17g: attestation records BOTH dimensions (git_dirty=1 + dirty_diff_sha256, git_untracked=1 + untracked_tree_sha256)"
else
  fail "S17g: attestation ($attest_g) missing one of the two dimensions' records"
fi

echo "--- S17h: unhashable untracked path (embedded git checkout) -> named FATAL ----"
S17H_CLONE="$SCRATCH_ROOT/s17/clone_h"
adopt_worktree_clone "$AC_DIR" "$S17H_CLONE"
mkdir "$S17H_CLONE/foo"
git -C "$S17H_CLONE/foo" init -q
S17_LOG_H="$SCRATCH_ROOT/s17/log_h"
if make -C "$S17H_CLONE" -s BACKEND=kiss DIST_ROOT="$S17_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish >"$S17_LOG_H" 2>&1; then
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
# Round-7 follow-up (reviewer-reproduced): raw space-joined "L <path> <target>"
# records encoded symlink "a b"->"c" and "a"->"b c" IDENTICALLY, so two
# different untracked trees shared one untracked_tree_sha256. The records are
# now fixed-field (every variable-length value itself hashed) -- this pair
# must therefore hash differently. Assertions stay format-agnostic beyond
# "64-hex and different".
S17I_CLONE="$SCRATCH_ROOT/s17/clone_i"
adopt_worktree_clone "$AC_DIR" "$S17I_CLONE"
S17I_OBJ="$SCRATCH_ROOT/s17/obj_i"; S17I_BIN="$SCRATCH_ROOT/s17/bin_i"; S17I_DIST="$SCRATCH_ROOT/s17/dist_i"
s17i_publish() {
  local log="$SCRATCH_ROOT/s17/log_i$1" att rel
  make -C "$S17I_CLONE" -s BACKEND=kiss OBJ_ROOT="$S17I_OBJ" BIN_ROOT="$S17I_BIN" DIST_ROOT="$S17I_DIST" \
       ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish >"$log" 2>&1 || return 1
  att="$(attest_from_log "$log")" || return 1
  rel="$(readlink "$S17I_DIST/kiss/current")" || return 1
  grep '^untracked_tree_sha256=' "$S17I_DIST/kiss/$rel/ATTEST/$att" | cut -d= -f2
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
# Round-7 follow-up (reviewer-reproduced): shasum failing inside a command
# substitution used to yield an EMPTY content hash with rc=0 -- publish
# continued with a hole in the provenance. Any per-entry I/O failure now
# downgrades the entry to an X record => named FATAL.
echo "secret" > "$S17I_CLONE/unreadable.bin"; chmod 000 "$S17I_CLONE/unreadable.bin"
S17_LOG_J="$SCRATCH_ROOT/s17/log_j"
if make -C "$S17I_CLONE" -s BACKEND=kiss OBJ_ROOT="$S17I_OBJ" BIN_ROOT="$S17I_BIN" DIST_ROOT="$S17I_DIST" \
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
# Round-7 follow-up (reviewer + /simplify independently): commit=unknown set
# dirty=untracked=unknown, ALLOW_DIRTY_PUBLISH=1 skipped the dirty gate and
# the untracked gate only rejected "=1", so a no-git tree could publish with
# no untracked enumeration at all. Unknown identity now refuses outright --
# NEITHER escape hatch admits it.
S17L_DIR="$SCRATCH_ROOT/s17/nogit"
mkdir -p "$S17L_DIR"
tar -c -C "$AC_DIR" --exclude obj --exclude bin --exclude dist --exclude .git . | tar -x -C "$S17L_DIR"
S17_LOG_L="$SCRATCH_ROOT/s17/log_l"
if make -C "$S17L_DIR" -s BACKEND=kiss OBJ_ROOT="$SCRATCH_ROOT/s17/obj_l" BIN_ROOT="$SCRATCH_ROOT/s17/bin_l" \
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
echo "# S18: interruption-safety probe (round-6 review P2-1 acceptance)"
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
# Guards exactly the gap a re-review found in the round-5 -fno-strict-
# aliasing fix: that flag used to be a bare literal on the two
# fft_wrapper.o/fft_wrapper_ne10.o target-specific `CFLAGS +=` lines
# (Makefile, just below CFG_SIG_PAYLOAD), and a target-specific variable
# assignment does NOT retroactively change what $(CFLAGS) expanded to when
# CFG_SIG_PAYLOAD (a `:=`, expanded immediately at parse time) was computed
# -- so the flag was invisible to CFG_SIG. Changing it (or removing it) left
# CFG_SIG byte-for-byte identical, `make lib` became a silent no-op, and a
# keyed obj/bin directory created before the flag existed could go on
# reusing an fft_wrapper.o compiled WITHOUT it forever. The fix extracted
# the flag into a named variable, $(FFT_WRAPPER_ALIAS_CFLAGS), defined
# ahead of CFG_SIG_PAYLOAD and folded into it, with the two target-specific
# lines now referencing the variable instead of a literal (see the
# Makefile's own "Build-cache-invalidation fix" comment right above
# CFG_SIG_PAYLOAD). This scenario is the permanent guard against that exact
# gap reopening.
#
# A later Codex review found this scenario's ORIGINAL shape was itself
# normalizing a real hole: it proved the point via a `make
# FFT_WRAPPER_ALIAS_CFLAGS=<other value>` command-line override, but this
# variable was a plain `:=` with no override-rejection -- so that same
# command line didn't just prove CFG_SIG's keying, it ALSO silently replaced
# the Makefile's own -fno-strict-aliasing value for real, and (e.g.) `make
# BACKEND=kiss FFT_WRAPPER_ALIAS_CFLAGS=` produced a real, buildable
# fft_wrapper.o shipping the known Complex*/float* strict-aliasing UB
# unguarded, in a fully valid-looking archive. The Makefile fix adds
# FFT_WRAPPER_ALIAS_CFLAGS as a 6th name to the same "Command-line override
# rejection" foreach that already covers CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/
# FP_POLICY (S9 above), so that exact command line now hard-$(error)s at
# parse time -- see S20 below, which proves the rejection directly -- and
# this scenario can no longer use it to prove CFG_SIG coverage.
#
# Rewritten instead to prove the identical CFG_SIG-keying property WITHOUT a
# command-line override: one real `make lib` runs off the unmodified
# Makefile at $AC_DIR; the other runs off a full scratch clone of the SAME
# worktree (adopt_worktree_clone -- VPATH/includes/sources need the whole
# tree, not just the Makefile) whose Makefile COPY has had ONLY the
# FFT_WRAPPER_ALIAS_CFLAGS definition line sed-patched, in place, to a
# different literal -- a real edit to that clone's own file, the same shape
# S17g already uses to edit a cloned Makefile's COMMON_SRCS line. Both
# builds share the SAME scratch OBJ_ROOT/BIN_ROOT, so the comparison below
# isolates the CFG_SIG effect of the changed literal, not an incidental
# root-path difference (same discipline S9 uses for its own baseline-vs-
# EXTRA_CFLAGS comparison). (Manually reverting the Makefile fix and
# re-running this exact scenario reproduces the original bug: both builds
# then land in the SAME keyed dir.) Runs entirely under a scratch
# OBJ_ROOT/BIN_ROOT and a scratch clone directory, like every other
# tamper/probe scenario in this suite -- the real obj/, bin/, and Makefile
# are never touched.
cd "$AC_DIR"
mkscratch s19
S19_CLONE="$SCRATCH_ROOT/s19/clone"
adopt_worktree_clone "$AC_DIR" "$S19_CLONE"

S19_CHANGED_FLAGS="-fno-strict-aliasing -fwrapv"
sed -i.s19bak "s|^FFT_WRAPPER_ALIAS_CFLAGS := -fno-strict-aliasing\$|FFT_WRAPPER_ALIAS_CFLAGS := $S19_CHANGED_FLAGS|" "$S19_CLONE/Makefile"
rm -f "$S19_CLONE/Makefile.s19bak"
grep -q "^FFT_WRAPPER_ALIAS_CFLAGS := $S19_CHANGED_FLAGS\$" "$S19_CLONE/Makefile" || \
  fail "S19: Makefile seam edit did not land (FFT_WRAPPER_ALIAS_CFLAGS definition line changed shape?)"

S19_OBJ_ROOT="$SCRATCH_ROOT/s19/obj"
S19_BIN_ROOT="$SCRATCH_ROOT/s19/bin"

s19_base_objdir="$(make -C "$AC_DIR" -s BACKEND=kiss OBJ_ROOT="$S19_OBJ_ROOT" BIN_ROOT="$S19_BIN_ROOT" print-obj-dir)"
s19_base_lib="$(make -C "$AC_DIR" -s BACKEND=kiss OBJ_ROOT="$S19_OBJ_ROOT" BIN_ROOT="$S19_BIN_ROOT" print-lib-path)"
if make -C "$AC_DIR" -s BACKEND=kiss OBJ_ROOT="$S19_OBJ_ROOT" BIN_ROOT="$S19_BIN_ROOT" lib >"$SCRATCH_ROOT/s19/build_base.log" 2>&1; then
  pass "S19: baseline build (real, unmodified Makefile, default FFT_WRAPPER_ALIAS_CFLAGS) succeeds"
else
  fail "S19: baseline build (real, unmodified Makefile) FAILED"
  cat "$SCRATCH_ROOT/s19/build_base.log" >&2
fi

s19_changed_objdir="$(make -C "$S19_CLONE" -s BACKEND=kiss OBJ_ROOT="$S19_OBJ_ROOT" BIN_ROOT="$S19_BIN_ROOT" print-obj-dir)"
s19_changed_lib="$(make -C "$S19_CLONE" -s BACKEND=kiss OBJ_ROOT="$S19_OBJ_ROOT" BIN_ROOT="$S19_BIN_ROOT" print-lib-path)"
if make -C "$S19_CLONE" -s BACKEND=kiss OBJ_ROOT="$S19_OBJ_ROOT" BIN_ROOT="$S19_BIN_ROOT" lib >"$SCRATCH_ROOT/s19/build_changed.log" 2>&1; then
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

# BACKEND=kiss is forced on both sides above, so the member name is always
# fft_wrapper.o (not the fft_wrapper_ne10.o alternative).
ar -t "$s19_changed_lib" 2>/dev/null | grep -q '^fft_wrapper\.o$' && \
  pass "S19: changed-config archive contains fft_wrapper.o (the object the alias flag targets)" \
  || fail "S19: changed-config archive missing fft_wrapper.o"

echo "############################################################"
echo "# S20: FFT_WRAPPER_ALIAS_CFLAGS command-line override rejection"
echo "# (Codex review)"
echo "############################################################"
# Direct negative-test companion to the Makefile fix S19 above now relies on:
# FFT_WRAPPER_ALIAS_CFLAGS was added as a 6th name to the SAME "Command-line
# override rejection" foreach that already covers CFLAGS/CXXFLAGS/CPPFLAGS/
# LDFLAGS/FP_POLICY (S9 above) -- reusing that exact established mechanism,
# not a parallel/different rejection style. Mirrors S9's own assertion
# shape/rigor exactly: each override must FAIL at parse time (before any
# obj/bin dir is even created), with the same "cannot be overridden" message
# S9 checks for. Two cases, both real command-line overrides, both with a
# scratch OBJ_ROOT/BIN_ROOT: an EMPTY value (silently drops
# -fno-strict-aliasing entirely -- the exact Codex finding) and an explicitly
# hostile `-fstrict-aliasing` value (re-enables strict aliasing outright).
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
echo "# bypass of S20's own guard (second Codex review)"
echo "############################################################"
# S20 above proved the COMMAND-LINE case is rejected. A SEPARATE gap let a
# \`make -e\` ENVIRONMENT override sail through with no error at all: GNU
# Make only flips $(origin FFT_WRAPPER_ALIAS_CFLAGS) from plain
# "environment" to "environment override" at the point a makefile assignment
# to that same name is actually PARSED under -e; the "Command-line override
# rejection" foreach used to query that origin BEFORE this variable's own
# (only) assignment in the file, so it always saw the pre-flip "environment"
# origin, matching neither \`command\` nor \`override\`, and silently passed.
# Direct repro (confirmed BEFORE the Makefile fix, this exact command):
#   env FFT_WRAPPER_ALIAS_CFLAGS=-fstrict-aliasing make -e BACKEND=kiss lib
# used to build a real, fully-buildable fft_wrapper.o with "-fstrict-
# aliasing" on the compile line and NO "-fno-strict-aliasing" anywhere -- the
# known Complex*/float* strict-aliasing UB shipping live. Fixed in the
# Makefile by relocating the bare FFT_WRAPPER_ALIAS_CFLAGS \`:=\` literal to
# directly above the foreach (see its "Bare-literal policy flags" comment).
# Two cases below, mirroring S20's own two cases exactly, but each run
# through \`env NAME=value make -e ...\` -- an actual ENVIRONMENT override,
# NOT a command-line one (\`make -e NAME=value\` on the command line would
# just be a command-line assignment, already covered by S20): the exact
# reported hostile non-empty value, and the empty-value sibling for full
# parity with S20.
cd "$AC_DIR"
mkscratch s21
S21_OBJ_ROOT="$SCRATCH_ROOT/s21/obj"
S21_BIN_ROOT="$SCRATCH_ROOT/s21/bin"

for pair in "FFT_WRAPPER_ALIAS_CFLAGS=-fstrict-aliasing" "FFT_WRAPPER_ALIAS_CFLAGS="; do
  ov_var="${pair%%=*}"
  ov_val="${pair#*=}"
  S21_LOG="$SCRATCH_ROOT/s21/log_$(printf '%s' "$ov_val" | tr -c 'A-Za-z0-9' '_')"
  if env "$ov_var=$ov_val" make -e BACKEND=kiss OBJ_ROOT="$S21_OBJ_ROOT" BIN_ROOT="$S21_BIN_ROOT" print-obj-dir >"$S21_LOG" 2>&1; then
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

# Direct end-to-end confirmation of the exact reported repro, one level
# closer to reality than print-obj-dir: a real \`lib\` build must ALSO be
# rejected the same way (not just the print-* introspection targets).
S21_LIB_LOG="$SCRATCH_ROOT/s21/log_lib_e2e"
if env FFT_WRAPPER_ALIAS_CFLAGS=-fstrict-aliasing make -e BACKEND=kiss OBJ_ROOT="$S21_OBJ_ROOT" BIN_ROOT="$S21_BIN_ROOT" lib >"$S21_LIB_LOG" 2>&1; then
  fail "S21: env FFT_WRAPPER_ALIAS_CFLAGS=-fstrict-aliasing make -e BACKEND=kiss lib unexpectedly SUCCEEDED (the exact Codex repro must now fail)"
else
  if grep -q "cannot be overridden" "$S21_LIB_LOG"; then
    pass "S21: the exact reported repro (env FFT_WRAPPER_ALIAS_CFLAGS=-fstrict-aliasing make -e BACKEND=kiss lib) correctly FAILS, mentioning 'cannot be overridden'"
  else
    fail "S21: the exact reported repro failed but WITHOUT the expected 'cannot be overridden' message"
    cat "$S21_LIB_LOG" >&2
  fi
fi

echo "############################################################"
echo "# S22: make -e environment-override rejection audit for the other"
echo "# five names (Codex review follow-up: FP_POLICY/CPPFLAGS also gapped)"
echo "############################################################"
# Auditing S21's fix prompted checking whether ANY of the other five names
# in the same foreach (CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/FP_POLICY) had a
# latent version of the identical \`make -e\` gap. CFLAGS/CXXFLAGS/LDFLAGS
# did NOT (each has an unconditional \`+=\` before the foreach, so -e's
# origin-flip had already happened by the time the foreach ran) -- confirmed
# empirically, not re-tested here (S9 above already covers their
# command-line rejection). FP_POLICY and CPPFLAGS DID:
#   - FP_POLICY had the IDENTICAL shape as FFT_WRAPPER_ALIAS_CFLAGS: its own
#     bare \`:=\` used to live only after the foreach (immediately ahead of
#     its \`CFLAGS/CXXFLAGS +=\` appends), so \`env FP_POLICY=x make -e ...\`
#     used to sail through -- a real build silently losing -ffp-contract=off
#     entirely. Fixed the same way: bare \`:=\` relocated above the foreach.
#   - CPPFLAGS had a DIFFERENT variant: its \`?=\` (well before the foreach)
#     only ever assigns when the variable's origin is still "undefined" --
#     given an environment-original value, that's already false, so the
#     line is a complete no-op and the origin-flip trigger (an actual
#     assignment ATTEMPT) never fires under -e, no matter how early the line
#     sits. Fixed by changing \`?=\` to \`+=\` (an unconditional attempt, same
#     shape as CFLAGS's own lines).
# Both cases below run through \`env NAME=value make -e ...\`, mirroring
# S21/S20's assertion style; CPPFLAGS/FP_POLICY are not consumed by any
# compile recipe today, so print-obj-dir (not a real \`lib\` build) is the
# right-sized check here, same as S9's own style for these
# introspection-only probes.
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
echo "# (Codex review)"
echo "############################################################"
# Direct negative-test companion to the Makefile's "FP policy conflict"
# widening: the round-3 B04 conflict-detection block (rejects
# -Ofast/-ffast-math/-ffp-contract=<anything>, since each would re-enable FP
# contraction this repo pins off) used to check $(CFLAGS) alone. A PLAIN
# ENVIRONMENT CXXFLAGS (or CPPFLAGS) -- deliberately allowed to fold in
# normally by the command-line/`-e` override rejection (S9/S22 above),
# since it never goes through EXTRA_CFLAGS at all -- smuggled every one of
# these three patterns straight past the old CFLAGS-only checks. Direct
# repro (confirmed BEFORE this round's Makefile fix, this exact command):
#   env CXXFLAGS=-Ofast make BACKEND=ne10 lib
# built a real archive with the NE10 backend's one C++ TU
# (NE10_fft_generic_int32.cpp -- see NE10_CXXSRCS in the Makefile, the only
# C++ TU this Makefile compiles) showing "-Ofast" on its compile line ahead
# of the trailing "-ffp-contract=off" -- every -Ofast relaxation OTHER than
# contraction (which the trailing flag does still turn back off) live and
# unchecked.
#
# BACKEND=ne10 (never kiss) throughout this scenario: confirmed empirically
# that this matters, not assumed -- the identical repro against BACKEND=kiss
# never invokes a C++ compiler at all (0 `.cpp` compile lines in its
# output, since VENDOR_CXXSRCS is only ever non-empty for BACKEND=ne10), so
# a kiss-backend version of this scenario would exercise only the
# (backend-independent) text-matching guard itself, never the actual
# real-world consequence the Codex review reported.
#
# Each case below is a genuine PLAIN ENVIRONMENT override (`env NAME=value
# make BACKEND=ne10 ...`, no `-e`, no command-line assignment) against the
# real `lib` target -- the exact reported repro shape, not just an
# introspection target -- with a scratch OBJ_ROOT/BIN_ROOT so a (correctly)
# rejected build never touches the real obj/bin trees. Three CXXFLAGS cases
# cover all three conflict patterns; the fourth is the CPPFLAGS sibling of
# the -Ofast case, for completeness (the fix checks CFLAGS/CXXFLAGS/CPPFLAGS
# together, so any one of the three carrying a plain environment value must
# be caught the same way).
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

# Positive control, mirroring S9's own EXTRA_CFLAGS positive-control shape:
# a LEGITIMATE plain-environment CXXFLAGS value that does NOT match any of
# the three conflict patterns must still fold in normally (the fix must
# reject only the specific conflicting patterns, not CXXFLAGS overrides in
# general) -- and, being folded into CFG_SIG, must land in a DIFFERENT
# keyed dir than the plain baseline.
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

# Round-9 Codex review: the conflict-detection guard above used to be three
# $(findstring PATTERN,TEXT) checks, and $(findstring) does a plain
# SUBSTRING search over the whole concatenated text -- so it false-positived
# on any token that merely CONTAINS one of these patterns as part of a
# larger identifier, never mind that it isn't the compiler flag at all.
# Direct repro (confirmed BEFORE this round's fix):
#   env CXXFLAGS='-DROUND9_NOTE=-Ofastness' make -s BACKEND=ne10 print-obj-dir
# used to fail with "FP policy conflict: ... contains -Ofast" even though
# `-DROUND9_NOTE=-Ofastness` is a single, entirely harmless preprocessor
# macro definition token that merely happens to contain the substring
# "-Ofast" inside a larger identifier -- not the compiler flag -Ofast. The
# fix replaced $(findstring) with $(filter), which splits its TEXT argument
# on whitespace into WORDS and matches each whole word against the given
# patterns, so a substring buried inside an unrelated token can never match.
S23_ROUND9_NOTE_LOG="$SCRATCH_ROOT/s23/log_round9_note"
if env CXXFLAGS='-DROUND9_NOTE=-Ofastness' make -s BACKEND=ne10 OBJ_ROOT="$S23_OBJ_ROOT" BIN_ROOT="$S23_BIN_ROOT" print-obj-dir >"$S23_ROUND9_NOTE_LOG" 2>&1; then
  pass "S23: env CXXFLAGS='-DROUND9_NOTE=-Ofastness' make BACKEND=ne10 print-obj-dir (substring-only, non-conflicting) succeeds (round-9 false-positive fix)"
else
  fail "S23: env CXXFLAGS='-DROUND9_NOTE=-Ofastness' make BACKEND=ne10 print-obj-dir unexpectedly FAILED (round-9 substring false-positive regressed)"
  cat "$S23_ROUND9_NOTE_LOG" >&2
fi

echo "############################################################"
echo "# S23b: quote/escape/response-file bypass of the FP-policy"
echo "# conflict gate itself (Codex review)"
echo "############################################################"
# See the header comment (search "S23b") for the full writeup. In short: the
# exact-token $(filter) check above matches GNU Make's OWN text
# representation of CFLAGS/CXXFLAGS/CPPFLAGS -- but Make has zero concept of
# shell quoting, so a value the shell will unquote at compile-recipe time
# (e.g. EXTRA_CFLAGS="'-Ofast'", stored by Make as the literal 9-character
# text '-Ofast', quote characters included) never equals the bare word
# -Ofast in Make's text and so could never be caught by $(filter) -- yet on
# the actual compile recipe line the shell strips the quotes and the
# compiler gets a real, live -Ofast. Each case below is a genuine PLAIN
# ENVIRONMENT EXTRA_CFLAGS override (no `-e`, no command-line assignment)
# against a cheap introspection target, with scratch OBJ_ROOT/BIN_ROOT so a
# (correctly) rejected build never touches the real obj/bin trees -- same
# discipline as S20/S21/S22 above. Each rejection must mention "FP policy
# conflict" (the shared marker every FP-policy failure emits) AND the
# specific forbidden construct named in the block's own $(error ...), so a
# generic/unrelated failure can never be mistaken for the fix working.
cd "$AC_DIR"
mkscratch s23b
S23B_OBJ_ROOT="$SCRATCH_ROOT/s23b/obj"
S23B_BIN_ROOT="$SCRATCH_ROOT/s23b/bin"

S23B_LOG_1="$SCRATCH_ROOT/s23b/log_single_quoted"
if env EXTRA_CFLAGS="'-Ofast'" make BACKEND=kiss OBJ_ROOT="$S23B_OBJ_ROOT" BIN_ROOT="$S23B_BIN_ROOT" print-obj-dir >"$S23B_LOG_1" 2>&1; then
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
if env EXTRA_CFLAGS='"-ffast-math"' make BACKEND=kiss OBJ_ROOT="$S23B_OBJ_ROOT" BIN_ROOT="$S23B_BIN_ROOT" print-obj-dir >"$S23B_LOG_2" 2>&1; then
  fail "S23b: env EXTRA_CFLAGS='\"-ffast-math\"' (double-quoted) print-obj-dir unexpectedly SUCCEEDED (must be rejected)"
else
  if grep -q "FP policy conflict" "$S23B_LOG_2" && grep -q "double-quote" "$S23B_LOG_2"; then
    pass "S23b: env EXTRA_CFLAGS='\"-ffast-math\"' (double-quoted) print-obj-dir correctly FAILS, identifying the double-quote character"
  else
    fail "S23b: env EXTRA_CFLAGS='\"-ffast-math\"' (double-quoted) print-obj-dir failed but did NOT identify the double-quote character specifically"
    cat "$S23B_LOG_2" >&2
  fi
fi

S23B_LOG_3="$SCRATCH_ROOT/s23b/log_quote_split"
if env EXTRA_CFLAGS="-O'f'ast" make BACKEND=kiss OBJ_ROOT="$S23B_OBJ_ROOT" BIN_ROOT="$S23B_BIN_ROOT" print-obj-dir >"$S23B_LOG_3" 2>&1; then
  fail "S23b: env EXTRA_CFLAGS=\"-O'f'ast\" (quote-split mid-token) print-obj-dir unexpectedly SUCCEEDED (must be rejected)"
else
  if grep -q "FP policy conflict" "$S23B_LOG_3" && grep -q "single-quote" "$S23B_LOG_3"; then
    pass "S23b: env EXTRA_CFLAGS=\"-O'f'ast\" (quote-split mid-token) print-obj-dir correctly FAILS, identifying the single-quote character"
  else
    fail "S23b: env EXTRA_CFLAGS=\"-O'f'ast\" (quote-split mid-token) print-obj-dir failed but did NOT identify the single-quote character specifically"
    cat "$S23B_LOG_3" >&2
  fi
fi

S23B_LOG_4="$SCRATCH_ROOT/s23b/log_response_file"
if env EXTRA_CFLAGS='@flags.rsp' make BACKEND=kiss OBJ_ROOT="$S23B_OBJ_ROOT" BIN_ROOT="$S23B_BIN_ROOT" print-obj-dir >"$S23B_LOG_4" 2>&1; then
  fail "S23b: env EXTRA_CFLAGS='@flags.rsp' (response-file) print-obj-dir unexpectedly SUCCEEDED (must be rejected)"
else
  if grep -q "FP policy conflict" "$S23B_LOG_4" && grep -q "response-file" "$S23B_LOG_4"; then
    pass "S23b: env EXTRA_CFLAGS='@flags.rsp' (response-file) print-obj-dir correctly FAILS, identifying the @-prefixed response-file token"
  else
    fail "S23b: env EXTRA_CFLAGS='@flags.rsp' (response-file) print-obj-dir failed but did NOT identify the response-file token specifically"
    cat "$S23B_LOG_4" >&2
  fi
fi

# Positive controls: both are real, harmless, UNQUOTED tokens that contain
# none of the newly-forbidden characters, so the new quote/escape/
# response-file block must leave them alone entirely -- only the unchanged
# exact-token $(filter) check further down governs whether a token IS
# -Ofast/-ffast-math/-ffp-contract=<x>, and neither of these is. The first
# is S23's own round-9 regression token (here via EXTRA_CFLAGS specifically,
# the exact hook this bypass came in through); the second is its CFLAGS-
# widening sibling from the same regression family.
S23B_LOG_P1="$SCRATCH_ROOT/s23b/log_positive_round9_note"
if env EXTRA_CFLAGS=-DROUND9_NOTE=-Ofastness make -s BACKEND=kiss OBJ_ROOT="$S23B_OBJ_ROOT" BIN_ROOT="$S23B_BIN_ROOT" print-obj-dir >"$S23B_LOG_P1" 2>&1; then
  pass "S23b: env EXTRA_CFLAGS=-DROUND9_NOTE=-Ofastness print-obj-dir (positive control, no forbidden characters) succeeds"
else
  fail "S23b: env EXTRA_CFLAGS=-DROUND9_NOTE=-Ofastness print-obj-dir unexpectedly FAILED (quote/escape gate false-positived on a harmless token)"
  cat "$S23B_LOG_P1" >&2
fi

S23B_LOG_P2="$SCRATCH_ROOT/s23b/log_positive_dtext"
if env EXTRA_CFLAGS=-DTEXT=ffast-math make -s BACKEND=kiss OBJ_ROOT="$S23B_OBJ_ROOT" BIN_ROOT="$S23B_BIN_ROOT" print-obj-dir >"$S23B_LOG_P2" 2>&1; then
  pass "S23b: env EXTRA_CFLAGS=-DTEXT=ffast-math print-obj-dir (positive control, no forbidden characters) succeeds"
else
  fail "S23b: env EXTRA_CFLAGS=-DTEXT=ffast-math print-obj-dir unexpectedly FAILED (quote/escape gate false-positived on a harmless token)"
  cat "$S23B_LOG_P2" >&2
fi

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

# round-7 review P3: final-guard re-check against the snapshot taken in S1
# (right after this suite's LAST intentional real-tree build). Every file
# that existed then must still be sha+mtime identical now; a directory
# created after the snapshot is allowed but reported (WARN/INFO -- after the
# S12 fix there should be none), while a NEW file inside an
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
  echo "  WARN: final-guard: new keyed director(ies) appeared under the real obj/+bin/ during this run (allowed, but unexpected after the round-7 S12 fix):"
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
