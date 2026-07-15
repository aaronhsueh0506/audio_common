# audio_common — shared audio DSP layer (FFT wrapper + KISS/NE10 backends + fast_math)
#
# One copy of the FFT layer for every consumer repo (AEC, NR, Audio_ALG). Consumers
# add `-I audio_common/include` and link the backend they want:
#
#   FFT backend policy:
#     make BACKEND=kiss      -> portable KISS FFT   (default; bit-reproducible reference)
#     make BACKEND=ne10      -> ARM NEON NE10 FFT   (embedded target; NE10_DSP_RFFT_SCALING on)
#
# Both backends expose the SAME public API (fft_wrapper.h): a heap path
# (fft_create/fft_destroy) and a static-memory path (fft_get_mem_size/fft_init),
# with a runtime pool_owned flag (no compile-time gate).

CC      ?= cc
CXX     ?= c++

# Archiver tools, explicit (CFG_SIG payload coverage below): a cross-compiling
# caller overriding AR/RANLIB (e.g. an arm-none-eabi- toolchain) lands in a
# different config signature automatically, same as overriding CC/CXX.
AR      ?= ar
RANLIB  ?= ranlib

# CFG_SIG payload-coverage placeholders (round-3 review B01). CPPFLAGS/
# EXTRA_LDFLAGS/NO_STDIO/DEBUG are not consumed by this particular Makefile
# today, but the CFG_SIG payload (see "Hash-keyed object directory" below)
# unconditionally includes all of them so that (a) the payload string has
# the same stable shape across all three repos sharing this build design and
# (b) a future use of any of them here automatically starts keying builds
# correctly instead of silently aliasing with a build that didn't set it.
# ?= with an explicit (possibly empty) value, not left undefined, so the
# payload string never depends on whether the variable happens to be defined.
CPPFLAGS      ?=
EXTRA_LDFLAGS ?=
NO_STDIO      ?= 0
DEBUG         ?= 0

# TOOLCHAIN_CHECK=0 skips the CC/CXX -dumpmachine target-coherence guard in
# _cfg_guard below (round-4 review P1-2). Participates in CFG_SIG, so a
# guard-skipped build can never alias a guarded one.
TOOLCHAIN_CHECK ?= 1

# DIST_ROOT (round-5 review P1): where `make publish` writes its release
# tree (default: dist/ in this directory). A legitimate override channel --
# the isolation test scripts publish into a throwaway mktemp directory so
# they can exercise publish/republish/tamper scenarios without ever touching
# (let alone deleting) the REAL dist/ releases. Deliberately NOT in CFG_SIG:
# it changes where a release is copied, never what is built.
#
# HOSTCC (round-5 review P2): compiles the tiny atomic-symlink-swap helper
# (tools/atomic_symlink_swap.c) AT PUBLISH TIME. This must be a compiler
# whose output runs on the BUILD HOST -- never $(CC), which may be a cross
# compiler targeting the board. Not in CFG_SIG: publish-time tooling only,
# never part of any built artifact.
#
# OBJ_ROOT / BIN_ROOT (round-6 review P1): where the keyed obj/bin trees
# live (defaults: obj/ and bin/ in this directory -- the default expansion
# is byte-identical to the previous hardcoded paths). Same rationale as
# DIST_ROOT: a placement knob, never a build-content knob, so deliberately
# NOT in CFG_SIG. The isolation tests use these to run tamper/injection
# scenarios (corrupt a config.manifest, inject a foreign archive member)
# against a scratch-directory build of the real worktree, so the REAL
# obj/ and bin/ artifacts are never modified -- not even transiently.
#
# ALLOW_DIRTY_PUBLISH (round-6 review P2): `make publish` REFUSES by default
# when this checkout has uncommitted tracked changes or is not a git
# checkout at all (commit unknown) -- a release whose ATTEST provenance
# cannot name the exact source state is not a board deliverable.
# ALLOW_DIRTY_PUBLISH=1 is the explicit escape hatch for dev publishes; it
# is recorded in the attestation together with a sha256 of
# `git diff --binary HEAD` so the deviation itself is traceable. Not in
# CFG_SIG: it never changes what is built, only whether publish proceeds.
#
# ATTEST_STAMP (round-6 review P2, TEST-ONLY): overrides the UTC timestamp
# embedded in the attestation FILENAME so the isolation tests can
# deterministically force same-second publish collisions and prove the
# no-clobber retry path. Never use it on a real publish -- though note a
# forged stamp is within the already-accepted threat model: attestations
# are unsigned and provide traceability, NOT authenticity against an
# attacker with filesystem access. Not in CFG_SIG or MANIFEST.
DIST_ROOT ?= dist
HOSTCC    ?= cc
OBJ_ROOT  ?= obj
BIN_ROOT  ?= bin
ALLOW_DIRTY_PUBLISH ?= 0
ATTEST_STAMP ?=

# -Werror opt-in (default off; CI can flip it on with `make WERROR=1`).
# Declared this early (rather than down by its ifeq block, where it used to
# live) because CFG_SIG below folds WERROR into its hash and needs its
# final value already resolved -- GNU Make expands variable references in a
# rule's target/prerequisite list (including a target-specific variable
# assignment's target) IMMEDIATELY when that line is parsed, not deferred
# like a recipe body, so CFG_SIG must be a fully-resolved `:=` before the
# first rule mentions anything derived from it (verified empirically: a
# forward-referenced variable in a target name silently resolves using
# whatever that variable's value was AT PARSE TIME, which can be the empty
# string if the real assignment comes later in the file).
WERROR ?= 0

# FFT backend selection:
#   - explicit `make BACKEND=kiss|ne10` always wins (policy: host/reference
#     builds -> kiss, embedded deliverable -> ne10; single main branch).
#   - otherwise AUTO-DETECT from the (possibly cross-)compiler: NE10 when the
#     target has ARM NEON (__ARM_NEON / __aarch64__ predefined), else KISS. Using
#     the compiler (not uname) makes this correct under cross-compilation.
ifeq ($(origin BACKEND),undefined)
  HAS_NEON := $(shell $(CC) -dM -E -x c /dev/null 2>/dev/null | grep -Eq '__ARM_NEON|__aarch64__' && echo yes)
  ifeq ($(HAS_NEON),yes)
    BACKEND := ne10
  else
    BACKEND := kiss
  endif
endif
# Gated on MAKECMDGOALS (round-3 review B01, parse purity for query targets):
# print-bin-dir/print-obj-dir/print-lib-path are consumed programmatically by
# other Makefiles' recipes (`ac="$$(make -s ... print-lib-path)"`) and by
# scripts; their stdout must be ONLY the path, nothing else. A bare `make
# print-lib-path` would otherwise also print this banner (parse time runs
# unconditionally, before any target is selected), corrupting the captured
# value.
ifeq ($(filter print-%,$(MAKECMDGOALS)),)
$(info audio_common: FFT backend = $(BACKEND))
endif

# gnu99 (not strict c99): the vendored NE10 headers use the GNU asm("label")
# extension. Consumers may still compile their own code with -std=c99 and link
# this archive — different TUs, different std is fine.
CFLAGS  += -Wall -Wextra -O2 -std=gnu99 -Iinclude
# -isystem (not -I) for the vendor NE10 include dirs: this dir's own headers
# (include/) are OUR code and stay -I so their diagnostics stay visible; NE10's
# headers are vendored and carry warnings we don't own, so demoting them to
# "system" headers suppresses diagnostics originating INSIDE those headers
# without touching diagnostics for our own TUs that happen to include them
# (F12 build hygiene -- see also the -isystem swap for lib/kiss_fft below).
CXXFLAGS+= -O2 -isystem lib/ne10/inc -isystem lib/ne10/common -isystem lib/ne10/modules
LDFLAGS += -lm

# EXTRA_CFLAGS hook: lets callers inject extra defines/flags (e.g.
# `make selftest EXTRA_CFLAGS=-DSIMD_KERNELS_FORCE_SCALAR`) into every
# compiled TU without editing this file.
CFLAGS  += $(EXTRA_CFLAGS)
CXXFLAGS+= $(EXTRA_CFLAGS)
LDFLAGS += $(EXTRA_LDFLAGS)

# Per-backend, per-config-signature object AND binary dir (CFG_SIG is
# computed further below, after every BACKEND-conditional CFLAGS/CXXFLAGS
# append) so a kiss build, an ne10 build, and any two builds differing only
# in EXTRA_CFLAGS/WERROR/etc. each land in their own directory automatically
# -- nothing is ever wiped to "switch" between them, and two such builds
# (e.g. one per backend) can run concurrently in this same tree without
# stomping each other's objects OR each other's final archive (round-3
# review B01: BIN_DIR used to be per-backend only, `bin/$(BACKEND)`, so an
# ne10 build after a kiss build -- or a stale mtime from a differently-flagged
# same-backend build -- could silently deliver the WRONG archive to whatever
# last happened to write that flat path; test_ne10_force_c's forced-C rebuild
# also used to overwrite the normal NE10 archive this way). Both OBJ_DIR and
# BIN_DIR are plain recursive (`=`) assignments referencing $(CFG_SIG) before
# CFG_SIG's own `:=` line below -- that's fine (and unavoidable, since BACKEND
# itself must already be resolved above): a recursive assignment just stores
# the unexpanded text, and nothing between here and CFG_SIG's `:=` assignment
# below ever expands $(OBJ_DIR)/$(BIN_DIR) (no rule mentions them yet), so by
# the time anything actually expands either variable, CFG_SIG already holds
# its final value.
OBJ_DIR = $(OBJ_ROOT)/$(BACKEND)-$(CFG_SIG)
BIN_DIR = $(BIN_ROOT)/$(BACKEND)-$(CFG_SIG)

# Backend-independent shared DSP sources (always in the archive).
COMMON_SRCS = src/hpf.c

ifeq ($(BACKEND),ne10)
  # NE10 source footprint (review F16): fft_wrapper_ne10.c only ever calls the
  # float32 R2C/C2R entry points (ne10_fft_alloc_r2c_float32 /
  # ne10_fft_r2c_mem_size_float32 / ne10_fft_init_r2c_float32_ext /
  # ne10_fft_destroy_r2c_float32 / ne10_fft_r2c_1d_float32_neon /
  # ne10_fft_c2r_1d_float32_neon) and never calls ne10_init() (see
  # fft_wrapper_ne10.c's header comment) -- so this is an explicit file list,
  # NOT a wildcard, empirically pared down to the smallest set that still
  # links cleanly (verified: `make BACKEND=ne10 clean selftest test_pool
  # test_wav test_zero_heap`, zero unresolved symbols, all four PASS).
  #   - NE10_rfft_float32.c / .neonintrinsic.c: the float32 R2C/C2R
  #     implementation itself (scalar reference + NEON kernels), including
  #     the P0001 external-memory init (VENDORED.md).
  #   - NE10_fft.c: not float32-R2C-specific, but the R2C path calls into it
  #     for ne10_factor()/ne10_fft_generate_twiddles_float32()/
  #     _transposed_float32(), and fft_wrapper_ne10.c's heap fft_destroy()
  #     calls its ne10_fft_destroy_r2c_float32() directly.
  #   - NE10_fft_float32.c, NE10_fft_int32.c, NE10_fft_generic_float32.c,
  #     NE10_fft_generic_int32.cpp: none of these float32/int32
  #     complex-to-complex (c2c) or non-power-of-2 "generic" paths are ever
  #     CALLED by this wrapper (it's real-to-complex, power-of-2-only, all
  #     the way through) -- but NE10_fft.c unconditionally defines
  #     ne10_fft_alloc_c2c_float32_neon/_int32_neon, and those reference
  #     ne10_fft_alloc_c2c_float32_c (NE10_fft_float32.c) and
  #     ne10_fft_alloc_c2c_int32_c (NE10_fft_int32.c) at file scope (not
  #     guarded by NE10_UNROLL_LEVEL or any other #if), which in turn
  #     reference the ALG_ANY/generic butterfly helpers -- confirmed
  #     empirically by dropping each file in turn and reading the linker's
  #     "Undefined symbols" list. Since ne10_fft.o is unavoidably pulled into
  #     the link (for ne10_factor/twiddle-gen above) and static-archive
  #     linking resolves an entire .o's symbol table as a unit (no
  #     function-level granularity without -ffunction-sections/--gc-sections,
  #     which is out of scope for a source-list-only trim), these four files
  #     must stay to satisfy the linker without editing any NE10 source.
  #     NE10_fft_generic_int32.cpp is why BACKEND=ne10 still needs a C++ TU
  #     (LINK=$(CXX) below) despite this trim.
  # Confirmed EXCLUDABLE (dropping each was verified to still link + pass all
  # four test targets):
  #   - NE10_init.c / NE10_init_dsp.c: ne10_init()/ne10_init_dsp() runtime
  #     dispatch setup -- dead weight now that nothing calls ne10_init().
  #   - NE10_fft_float32.neonintrinsic.c, NE10_fft_int32.neonintrinsic.c: the
  #     NEON c2c COMPUTE kernels (ne10_fft_c2c_1d_{float32,int32}_neon) --
  #     unlike the _c alloc functions above, nothing (not even NE10_fft.c's
  #     c2c NEON allocators) ever calls the NEON c2c compute entry points
  #     themselves, only the plain "_c" alloc/factor/twiddle machinery.
  #   - NE10_fft_generic_float32.neonintrinsic.cpp,
  #     NE10_fft_generic_int32.neonintrinsic.cpp: NEON-optimised generic
  #     (non-power-of-2) kernels -- same reasoning, only the "_c" generic
  #     alloc path is ever reachable from ne10_fft.o, not these.
  #   - NE10_fft_int16.c/.neonintrinsic.c: integer16 FFT, a fully separate
  #     family NE10_fft.c never touches.
  #   - NE10_fir.c/NE10_fir_init.c, NE10_iir.c/NE10_iir_init.c: FIR/IIR DSP
  #     families, unrelated to the FFT wrapper entirely.
  # No NE10 source is edited to achieve this trim -- it is purely a build
  # source-list change; every excluded file is still present under lib/ne10/
  # (see VENDORED.md) for provenance, just not compiled into this archive.
  NE10_C_SRCS  = lib/ne10/modules/dsp/NE10_fft.c \
                 lib/ne10/modules/dsp/NE10_fft_float32.c \
                 lib/ne10/modules/dsp/NE10_fft_int32.c \
                 lib/ne10/modules/dsp/NE10_fft_generic_float32.c \
                 lib/ne10/modules/dsp/NE10_rfft_float32.c \
                 lib/ne10/modules/dsp/NE10_rfft_float32.neonintrinsic.c
  NE10_CXXSRCS = lib/ne10/modules/dsp/NE10_fft_generic_int32.cpp
  # OWN_SRCS/VENDOR_SRCS split (F12 build hygiene): OWN_SRCS is code we wrote
  # and own the warnings for; VENDOR_SRCS/VENDOR_CXXSRCS is unmodified NE10.
  # BE_SRCS/BE_CXXSRCS (used by BE_OBJS below) are derived from these so the
  # archive's file list is unchanged, just re-expressed.
  OWN_SRCS       = src/fft_wrapper_ne10.c $(COMMON_SRCS)
  VENDOR_SRCS    = $(NE10_C_SRCS)
  VENDOR_CXXSRCS = $(NE10_CXXSRCS)
  # NE10_DSP_RFFT_SCALING makes c2r normalise by 1/N (verified round-trip); it is
  # default-on in NE10_fft.h — defined explicitly so no build can silently drop it.
  NE10_DEFS = -DNE10_ENABLE_DSP -DNE10_DSP_RFFT_SCALING
  # -isystem for the vendor NE10 include dirs (see the CXXFLAGS comment
  # above); NE10_DEFS stays a plain -D (a set of defines, not a search path --
  # isystem only applies to -I-style entries).
  CFLAGS  += -isystem lib/ne10/inc -isystem lib/ne10/common -isystem lib/ne10/modules $(NE10_DEFS)
  CXXFLAGS+= $(NE10_DEFS)
else
  OWN_SRCS       = src/fft_wrapper.c $(COMMON_SRCS)
  VENDOR_SRCS    = lib/kiss_fft/kiss_fft.c
  VENDOR_CXXSRCS =
  # -isystem for the vendor KISS FFT include dir (see the CXXFLAGS comment
  # above for the NE10 side of this same F12 fix).
  CFLAGS  += -isystem lib/kiss_fft
endif

# Link driver (round-4 review P1-2: declared HERE, before CFG_SIG, so it can
# participate in the payload below -- it used to live after CFG_SIG, where a
# command-line LINK= override changed the produced binaries without changing
# their keyed directory). The NE10 archive contains a C++ TU
# (NE10_fft_generic_int32.cpp), so anything linking it uses the C++ driver --
# which also supplies the correct C++ runtime library itself (libc++ on
# macOS/clang, libstdc++ on GNU/Linux gcc); the old hardcoded `-lc++` in
# LDFLAGS was a macOS-ism (duplicate of what the driver adds there, and a
# link error on GNU/Linux where the library is libstdc++) and is gone.
LINK = $(CC)
ifeq ($(BACKEND),ne10)
  LINK = $(CXX)
endif

# --- Command-line override rejection (round-4 review P1-1) ------------------
# CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/FP_POLICY are INTERNAL: GNU Make silently
# ignores every one of this file's own `+=`/`:=` assignments to a variable
# that was set on the command line (`make CFLAGS=-O3`), which would strip the
# FP-contraction policy, the backend defines/-isystem paths, and -lm from the
# build while still producing artifacts (verified: the compile line really
# loses -ffp-contract=off). The supported user hooks are EXTRA_CFLAGS /
# EXTRA_LDFLAGS (folded in above, covered by CFG_SIG, policy still appended
# after them). A plain environment CFLAGS is NOT rejected: this file's own
# assignments fold it in normally (origin becomes "file"), the policy flags
# still land last, and the value participates in CFG_SIG -- only "command
# line" and the -e "environment override" origins defeat the assignments.
$(foreach v,CFLAGS CXXFLAGS CPPFLAGS LDFLAGS FP_POLICY,\
  $(if $(findstring command,$(origin $(v)))$(findstring override,$(origin $(v))),\
    $(error $(v) cannot be overridden (origin: $(origin $(v))) -- it would silently drop this Makefile's own flag appends (FP policy, backend defines); use EXTRA_CFLAGS / EXTRA_LDFLAGS instead)))

# --- FP-contraction policy: unified across all four repos (round-3 review
# B04). Every TU THIS Makefile compiles -- our own sources AND the vendored
# KISS/NE10 C and C++ TUs alike -- builds with -ffp-contract=off, positioned
# so nothing can override it: appended here, AFTER every prior CFLAGS/
# CXXFLAGS append in this file (base flags, EXTRA_CFLAGS, and the
# BACKEND-conditional NE10_DEFS/-isystem/kiss appends just above), and BEFORE
# CFG_SIG_PAYLOAD is computed below (so it automatically participates in
# CFG_SIG -- no separate payload entry needed, same as CFLAGS/CXXFLAGS
# themselves). A caller's EXTRA_CFLAGS can add flags but can no longer
# silently re-enable contraction after this file's own flags, because
# nothing after this line ever appends to CFLAGS/CXXFLAGS again.
#
# Policy scope: contraction must be off repo-wide, not just for code we own
# -- a compiler-fused FMA in vendored scalar reference code (KISS, or NE10's
# non-neonintrinsic .c/.cpp files) is exactly the build-order-dependent
# numeric drift this flag exists to eliminate. This does NOT touch explicit
# NEON intrinsics (fft_wrapper_ne10.c's vfmaq_f32/vmulq_f32, or NE10's own
# *.neonintrinsic.c) or explicit fmaf() calls (fft_wrapper.c's/
# fft_wrapper_ne10.c's fft_power() scalar tail, deliberately mirroring the
# NEON path bit-for-bit -- see that function's own comment) -- those are
# EXPLICIT fusion requests the source code asks for regardless of this flag,
# not compiler-chosen contraction of a plain `a*b+c` expression, and
# -ffp-contract only controls the latter. See scripts/audit_fp_contract.sh
# for the disassembly-level verification (PASS/EXEMPT table distinguishing
# the two) and each repo's README/CLAUDE.md for the cross-repo policy note.
#
# Conflict detection (round-3 review B04): a caller passing
# EXTRA_CFLAGS=-Ofast/-ffast-math (both of which imply -ffp-contract=fast on
# gcc/clang) or EXTRA_CFLAGS=-ffp-contract=<anything> would either directly
# re-enable contraction or silently do so via an implied flag, despite this
# repo's pinned policy -- rejected outright rather than allowed to land.
# Checked against the fully-assembled $(CFLAGS) (already includes
# EXTRA_CFLAGS folded in above, AND any CFLAGS= command-line override, since
# a command-line-set CFLAGS is what every prior `+=` in this file appended
# to) -- BEFORE FP_POLICY is appended below and BEFORE CFG_SIG is computed,
# so a rejected build never creates an obj/bin directory. Using $(CFLAGS)
# here (not a separate raw EXTRA_CFLAGS-only check) also catches
# CXXFLAGS-only conflicts for free, since EXTRA_CFLAGS is the one hook both
# CFLAGS and CXXFLAGS fold in identically (this Makefile has no separate
# EXTRA_CXXFLAGS knob).
ifneq ($(findstring -Ofast,$(CFLAGS)),)
$(error FP policy conflict: CFLAGS/EXTRA_CFLAGS contains -Ofast; this repo pins -ffp-contract=off; remove -Ofast from EXTRA_CFLAGS)
endif
ifneq ($(findstring -ffast-math,$(CFLAGS)),)
$(error FP policy conflict: CFLAGS/EXTRA_CFLAGS contains -ffast-math; this repo pins -ffp-contract=off; remove -ffast-math from EXTRA_CFLAGS)
endif
ifneq ($(findstring -ffp-contract=,$(CFLAGS)),)
$(error FP policy conflict: CFLAGS/EXTRA_CFLAGS contains -ffp-contract=; this repo pins -ffp-contract=off; remove -ffp-contract= from EXTRA_CFLAGS)
endif

FP_POLICY := -ffp-contract=off
CFLAGS    += $(FP_POLICY)
CXXFLAGS  += $(FP_POLICY)

# --- Hash-keyed object/binary directory + full-coverage CFG_SIG ------------
# obj/$(BACKEND)-<sig> and (round-3 review B01) bin/$(BACKEND)-<sig> isolate
# every distinct build configuration from every other one -- not just
# BACKEND, but the exact compiler invocation that produces every object AND
# the exact archive that invocation produces. CFG_SIG_PAYLOAD is the single
# source string used BOTH to compute CFG_SIG (via cksum) AND, verbatim, as
# the collision-guard manifest content (see the _cfg_guard target below) --
# one string, two consumers, so the two can never drift apart.
#
# Payload coverage (USER-MANDATED, round-3 review B01): CC CXX AR RANLIB
# CFLAGS CXXFLAGS CPPFLAGS LDFLAGS EXTRA_CFLAGS EXTRA_LDFLAGS BACKEND WERROR
# NO_STDIO DEBUG -- every variable that can affect what a build produces,
# even ones this particular Makefile doesn't branch on today (CPPFLAGS/
# EXTRA_LDFLAGS/NO_STDIO/DEBUG default to an explicit empty/0 value up top so
# they always participate in the string with a stable shape). This MUST be
# defined here -- after CFLAGS/CXXFLAGS are final (every BACKEND-conditional
# append above has already run), but before the first rule below that
# mentions $(OBJ_DIR)/$(BIN_DIR)/anything derived from them (the WERROR and
# -fno-strict-aliasing target-specific-variable lines just after this, and
# the pattern rules further down) -- because GNU Make expands a rule's
# target/prerequisite list (including a target-specific variable
# assignment's target) IMMEDIATELY when that line is parsed, not deferred
# like a recipe body. A forward reference there would silently resolve
# using CFG_SIG's value at THAT point in the file (empty, if this line
# hadn't run yet), landing objects in a hash-less directory instead of the
# real one (verified empirically with a minimal reproduction).
#
# Deliberately NOT $(strip)'d: two builds differing only in incidental
# whitespace (e.g. an EXTRA_CFLAGS with a stray leading space) get different
# signatures. That's a false MISS (an unnecessary extra directory), which is
# harmless; collapsing them with $(strip) risks a false HIT (two genuinely
# different flag strings landing on the same signature by construction),
# which is the one failure mode this whole mechanism exists to prevent.
#
# There is no parse-time comparison and nothing is ever deleted to "detect"
# a config change (the previous scheme did a $(shell rm -rf ...)/$(shell
# mkdir ...)/$(shell echo ...) unconditionally while the Makefile was being
# read, which mutated the filesystem even under `make -n` and raced two
# concurrent builds with different flags/backends against each other's
# shared obj/$(BACKEND) directory). Under this scheme `make -n` touches
# nothing on disk (CFG_SIG is a pure string hash) and `make BACKEND=kiss
# -j4 & make BACKEND=ne10 -j4 & wait` (or two differing-EXTRA_CFLAGS
# builds) can run concurrently in this same tree: each lands in its own
# obj/<backend>-<sig>/ + bin/<backend>-<sig>/ directory pair and never
# touches the other's objects or archive. Directory creation (and the
# CFG_SIG collision guard) happens lazily via the _cfg_guard order-only
# prerequisite on the rules below, not at Makefile-parse time.
#
# cksum (POSIX -- present on macOS/BSD/GNU/busybox, unlike shasum which is a
# GNU coreutils extension and may be absent from slim embedded build images)
# computes the hash. On stdin it prints "<checksum> <byte-count>" with no
# filename field (verified on macOS's BSD cksum), so `cut -d' ' -f1` isolates
# just the checksum.
# Round-4 review additions to the payload: LINK (P2-1 -- the link driver
# affects every produced binary, and a command-line LINK= override is
# legitimate for exotic toolchains, so it must key the build), TOOLCHAIN_CHECK
# (the guard-skip knob must not alias a guarded build), and SRCS (P1-4 -- the
# SORTED source list, so ADDING or REMOVING a source file lands in a fresh
# keyed directory; without it, a removed .c left its stale .o AND its stale
# archive member behind under the same key, and `ar rc` on the existing
# archive would never delete that member -- see the fresh-archive recipe
# below for the other half of that fix).
CFG_SIG_PAYLOAD := CC=$(CC) CXX=$(CXX) AR=$(AR) RANLIB=$(RANLIB) LINK=$(LINK) CFLAGS=$(CFLAGS) CXXFLAGS=$(CXXFLAGS) CPPFLAGS=$(CPPFLAGS) LDFLAGS=$(LDFLAGS) EXTRA_CFLAGS=$(EXTRA_CFLAGS) EXTRA_LDFLAGS=$(EXTRA_LDFLAGS) BACKEND=$(BACKEND) WERROR=$(WERROR) NO_STDIO=$(NO_STDIO) DEBUG=$(DEBUG) TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK) SRCS=$(sort $(OWN_SRCS) $(VENDOR_SRCS) $(VENDOR_CXXSRCS))
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)

BE_SRCS    = $(OWN_SRCS) $(VENDOR_SRCS)
BE_CXXSRCS = $(VENDOR_CXXSRCS)

BE_OBJS  = $(patsubst %.c,$(OBJ_DIR)/%.o,$(notdir $(BE_SRCS))) \
           $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(BE_CXXSRCS)))
OWN_OBJS = $(patsubst %.c,$(OBJ_DIR)/%.o,$(notdir $(OWN_SRCS)))
VPATH    = src lib/kiss_fft lib/ne10/modules lib/ne10/modules/dsp

# -Werror opt-in, scoped to OUR OWN objects only (WERROR itself is declared
# near the top of this file, before CFG_SIG -- see that declaration for why).
# audio_common is the one Makefile in this project that compiles vendored
# TUs (NE10/KISS FFT) through the exact same generic pattern rule as its own
# wrapper sources (src/hpf.c, src/fft_wrapper*.c); those vendor bodies carry
# pre-existing warnings that are out of scope to fix here, so a blanket
# -Werror would fail the build on code we don't own. A target-specific
# variable applies -Werror only when building $(OWN_OBJS), leaving
# $(VENDOR_SRCS)/$(VENDOR_CXXSRCS) compiles unaffected.
ifeq ($(WERROR),1)
$(OWN_OBJS): CFLAGS += -Werror
endif

# -fno-strict-aliasing, scoped to ONLY the object for src/fft_wrapper_ne10.c
# (same target-specific-variable pattern as the -Werror opt-in above, just
# unconditional and narrower -- one specific object, not a whole class of
# objects). That TU casts `Complex*` to `ne10_fft_cpx_float32_t*` in place
# (see the "Residual strict-aliasing caveat" comment in fft_wrapper_ne10.c
# itself for the layout-equality proof and full rationale) -- two distinct
# struct types, so reading through one after writing through the other is,
# strictly, type-based-alias UB even though the layouts provably match. This
# flag removes the TBAA assumption for this one TU only, formally sanctioning
# what was previously just an -O2/no-LTO empirical argument; every other
# object in this archive (including the rest of the NE10 backend) keeps the
# default strict-aliasing behavior.
$(OBJ_DIR)/fft_wrapper_ne10.o: CFLAGS += -fno-strict-aliasing

LIB = $(BIN_DIR)/libaudio_common.a

.PHONY: all lib selftest test_pool test_wav test_wav_nr_style test-wav-ubsan test_zero_heap test_ne10_force_c _ne10_parity_bin clean publish print-bin-dir print-obj-dir print-lib-path _cfg_guard
all: lib

lib: $(LIB)

# --- CFG_SIG collision guard + directory creation (build hygiene) ----------
# _cfg_guard is the single order-only prerequisite every compile/link/archive
# rule below depends on (replacing the previous bare "| $(OBJ_DIR)"/
# "| $(BIN_DIR)" order-only prerequisites and their own tiny `@mkdir -p $@`
# rule). Being .PHONY, its recipe runs on every `make` invocation that needs
# it -- not just the first time $(OBJ_DIR)/$(BIN_DIR) are created -- and GNU
# Make builds any one target at most once per invocation even when several
# rules list it as a prerequisite, so under `make -j` it still runs exactly
# once, and (being an ordinary prerequisite in the dependency graph, order-
# only or not) make schedules it to completion before any rule that depends
# on it starts: race-free.
#
# It does two things:
#   1. mkdir -p $(OBJ_DIR) $(BIN_DIR) -- idempotent, safe if two *separate*
#      `make` processes for the very same config happen to race each other.
#   2. Writes (first time) or verifies (every subsequent time) a one-line
#      config.manifest in EACH of the two directories, containing the exact
#      CFG_SIG_PAYLOAD string that produced this build's CFG_SIG. If a
#      manifest already exists and its content does not match this build's
#      payload, two *different* configurations hashed to the same CFG_SIG
#      (a cksum collision) or the payload has a coverage gap (some flag that
#      affects the build isn't in it) -- either way, objects from an
#      unrelated config could silently be reused, so this is a hard error,
#      never a warning. The write is temp-file-then-mv (atomic rename on the
#      same filesystem), so a concurrent reader never observes a half-
#      written manifest.
#
# `make -n` only prints this recipe (nothing is created or compared under
# -n), and nothing here runs unless some real target needs it -- Makefile
# parsing itself performs no I/O.
_cfg_guard:
	@if [ "$(BACKEND)" = "ne10" ] && [ "$(TOOLCHAIN_CHECK)" = "1" ]; then \
	  cc_m="$$($(CC) -dumpmachine 2>/dev/null)"; cxx_m="$$($(CXX) -dumpmachine 2>/dev/null)"; \
	  if [ -z "$$cc_m" ] || [ "$$cc_m" != "$$cxx_m" ]; then \
	    echo "FATAL: CC/CXX compile for different targets (CC='$(CC)' -> '$$cc_m', CXX='$(CXX)' -> '$$cxx_m')." >&2; \
	    echo "  BACKEND=ne10 builds one C++ TU, so a partial cross-toolchain (only CC= passed)" >&2; \
	    echo "  would mix host C++ objects into a cross build. Pass the FULL toolchain, e.g.:" >&2; \
	    echo "    make BACKEND=ne10 CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ AR=aarch64-linux-gnu-ar RANLIB=aarch64-linux-gnu-ranlib" >&2; \
	    echo "  (TOOLCHAIN_CHECK=0 skips this check; it participates in CFG_SIG.)" >&2; \
	    exit 1; \
	  fi; \
	fi
	@mkdir -p $(OBJ_DIR) $(BIN_DIR)
	@for d in $(OBJ_DIR) $(BIN_DIR); do \
	  m="$$d/config.manifest"; \
	  if [ -f "$$m" ]; then \
	    old="$$(cat "$$m")"; \
	    if [ "$$old" != "$(CFG_SIG_PAYLOAD)" ]; then \
	      echo "FATAL: CFG_SIG collision or coverage gap in $$m" >&2; \
	      echo "  CFG_SIG=$(CFG_SIG)" >&2; \
	      echo "  stored : $$old" >&2; \
	      echo "  current: $(CFG_SIG_PAYLOAD)" >&2; \
	      exit 1; \
	    fi; \
	  else \
	    tmp="$$d/.config.manifest.tmp.$$$$"; \
	    printf '%s\n' "$(CFG_SIG_PAYLOAD)" > "$$tmp"; \
	    mv -f "$$tmp" "$$m"; \
	  fi; \
	done

# Fresh-archive discipline (round-4 review P1-4): the archive is always built
# from scratch into a temp file and renamed into place -- NEVER `ar rc` onto
# an existing archive, which only ever ADDS/REPLACES members and would leave
# a removed source's stale .o member in the delivered archive forever. The
# rename is also atomic, so a concurrent reader (another config's consumer
# resolving this same path) never observes a half-written archive. The other
# half of the source-removal fix is SRCS in CFG_SIG_PAYLOAD above (a changed
# source list keys to a fresh directory in the first place).
# One chained shell line, NOT one recipe line per step: the temp name embeds
# the shell's PID ($$$$ -> $$) so two same-config processes racing this rule
# (e.g. a plain build and a publish that started before taking its lock,
# round-5 review P2) each write their OWN temp and the final mv is an atomic
# last-writer-wins -- and each recipe LINE runs in a fresh shell with a fresh
# PID, so the steps must share one line to share one temp name.
$(LIB): $(BE_OBJS) | _cfg_guard
	t="$@.tmp.$$$$" && rm -f "$$t" && $(AR) rc "$$t" $(BE_OBJS) && $(RANLIB) "$$t" && mv -f "$$t" $@
	@echo "audio_common [$(BACKEND)] -> $@"

# -MD -MP (NOT -MMD): emits $(OBJ_DIR)/<name>.d per object, listing every
# header the TU pulled in -- INCLUDING vendored NE10/KISS headers reached via
# -isystem, which -MMD would omit. Included near the bottom of this file via
# `-include $(wildcard $(OBJ_DIR)/*.d)`; see that line's comment.
$(OBJ_DIR)/%.o: %.c | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<
$(OBJ_DIR)/%.o: %.cpp | _cfg_guard
	$(CXX) $(CXXFLAGS) -MD -MP -c -o $@ $<

selftest: $(LIB) | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $(OBJ_DIR)/roundtrip.o test/roundtrip.c
	$(LINK) -o $(BIN_DIR)/roundtrip $(OBJ_DIR)/roundtrip.o $(LIB) $(LDFLAGS)
	@echo "--- audio_common selftest [$(BACKEND)] ---"
	@$(BIN_DIR)/roundtrip
	# simd_kernels.h selftest: header-only, needs no FFT/archive objects.
	# -ffp-contract=off is NOT optional here -- verified empirically that
	# without it, this compiler auto-fuses the "plain mul/add, no FMA"
	# scalar reference loops (e.g. sk_ema_f32_scalar) into `fmla` at -O2,
	# which would then mismatch the deliberately-unfused NEON intrinics
	# path (see include/simd_kernels.h's FMA-discipline doc comment).
	$(CC) $(CFLAGS) -ffp-contract=off -MD -MP -c -o $(OBJ_DIR)/simd_selftest.o test/simd_selftest.c
	$(CC) -o $(BIN_DIR)/simd_selftest $(OBJ_DIR)/simd_selftest.o -lm
	@echo "--- audio_common SIMD kernel selftest [$(BACKEND)] ---"
	@$(BIN_DIR)/simd_selftest

# test_pool: pool-contract negative tests (F07 alignment guards, F14 HPF
# domain validation). Same shape as selftest -- links against the same $(LIB)
# archive so it exercises whichever backend (kiss/ne10) was built.
test_pool: $(LIB) | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $(OBJ_DIR)/test_pool_contract.o test/test_pool_contract.c
	$(LINK) -o $(BIN_DIR)/test_pool_contract $(OBJ_DIR)/test_pool_contract.o $(LIB) $(LDFLAGS)
	@echo "--- audio_common pool-contract test [$(BACKEND)] ---"
	@$(BIN_DIR)/test_pool_contract

# test_wav: negative-corpus tests for the shared, hardened WAV parser (F06
# remediation). Header-only like the simd_kernels.h half of `selftest` --
# wav_io.h needs no FFT/archive objects, so this doesn't depend on $(LIB)
# and links with a plain $(CC) (no NE10 C++ TU involved either way).
test_wav: | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $(OBJ_DIR)/test_wav_io.o test/test_wav_io.c
	$(CC) -o $(BIN_DIR)/test_wav_io $(OBJ_DIR)/test_wav_io.o -lm
	@echo "--- audio_common WAV I/O negative-corpus test [$(BACKEND)] ---"
	@$(BIN_DIR)/test_wav_io

# test_wav_nr_style (review R01): runs the SAME writer special-values
# corpus (test/test_wav_writer_common.h) as test_wav above, but compiled
# with WAV_IO_WRITER_STYLE forced to WAV_IO_WRITER_NR -- test_wav's own TU
# never sets that knob, so it only ever exercises the default
# WAV_IO_WRITER_AEC style (see wav_io.h's WAV_IO_WRITER_STYLE doc). Header-
# only, same shape as test_wav.
test_wav_nr_style: | _cfg_guard
	$(CC) $(CFLAGS) -DWAV_IO_WRITER_STYLE=WAV_IO_WRITER_NR -MD -MP -c -o $(OBJ_DIR)/test_wav_writer_nr_style.o test/test_wav_writer_nr_style.c
	$(CC) -o $(BIN_DIR)/test_wav_writer_nr_style $(OBJ_DIR)/test_wav_writer_nr_style.o -lm
	@echo "--- audio_common WAV writer NR-style special-values test [$(BACKEND)] ---"
	@$(BIN_DIR)/test_wav_writer_nr_style

# test-wav-ubsan (review R01): UBSan probe for wav_write_float's PCM16
# quantizer -- writes {1.0f,-1.0f,NAN} and must run clean under
# -fsanitize=undefined -fno-sanitize-recover=all. Pre-fix, +1.0f drove
# (int16_t)32768.5f (out-of-range float-to-integer conversion, UB per C99
# 6.3.1.4p1) and this aborted; post-fix (saturating int32_t conversion +
# non-finite sanitize) it must pass. Sanitizer flags are scoped to this one
# target's compile+link, same pattern as $(OWN_OBJS): CFLAGS -Werror above
# -- never applied to the main $(LIB) archive or any other test.
test-wav-ubsan: | _cfg_guard
	$(CC) $(CFLAGS) -fsanitize=undefined -fno-sanitize-recover=all -MD -MP -c -o $(OBJ_DIR)/test_wav_writer_ubsan.o test/test_wav_writer_ubsan.c
	$(CC) -fsanitize=undefined -fno-sanitize-recover=all -o $(BIN_DIR)/test_wav_writer_ubsan $(OBJ_DIR)/test_wav_writer_ubsan.o -lm
	@echo "--- audio_common WAV writer UBSan probe [$(BACKEND)] ---"
	@$(BIN_DIR)/test_wav_writer_ubsan

# test_zero_heap: allocator-hook acceptance test (review F02/F08) -- proves
# fft_init()..fft_destroy() makes zero malloc/calloc/realloc/free calls, and
# that fft_destroy() on a pool-owned handle is idempotent. Same shape as
# test_pool -- links against the same $(LIB) archive so it exercises whichever
# backend (kiss/ne10) was built (KISS was already zero-heap; NE10's twiddle
# config is now carved from the caller pool too -- see lib/ne10/VENDORED.md
# patch P0001).
#
# The allocator-hook mechanism (test/zero_heap_hook.c) is platform-specific
# and, empirically (see that file's header comment), needs different
# packaging per platform:
#   - macOS: dyld __DATA,__interpose only takes effect from a Mach-O image
#     OTHER than the main executable, so the hook is built as its own small
#     .dylib and the test links against it (rpath'd to @loader_path so it
#     resolves next to the test binary with no env var needed).
#   - Linux: GNU ld --wrap=<symbol> works from a plain object linked directly
#     into the test binary; -Wl,--wrap=... is scoped to this one target only,
#     so it never affects the main $(LIB) archive or any other test.
ifeq ($(shell uname -s),Darwin)
  ZERO_HEAP_HOOK_LIB   = $(BIN_DIR)/libzero_heap_hook.dylib
  ZERO_HEAP_LDFLAGS    = -L$(BIN_DIR) -lzero_heap_hook -Wl,-rpath,@loader_path
  ZERO_HEAP_HOOK_DEPS  = $(ZERO_HEAP_HOOK_LIB)
else
  ZERO_HEAP_HOOK_LIB   = $(OBJ_DIR)/zero_heap_hook.o
  ZERO_HEAP_LDFLAGS    = -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc -Wl,--wrap=free
  ZERO_HEAP_HOOK_DEPS  = $(ZERO_HEAP_HOOK_LIB)
endif

$(BIN_DIR)/libzero_heap_hook.dylib: test/zero_heap_hook.c test/zero_heap_hook.h | _cfg_guard
	$(CC) $(CFLAGS) -dynamiclib -o $@ test/zero_heap_hook.c

$(OBJ_DIR)/zero_heap_hook.o: test/zero_heap_hook.c test/zero_heap_hook.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ test/zero_heap_hook.c

test_zero_heap: $(LIB) $(ZERO_HEAP_HOOK_DEPS) | _cfg_guard
	# -fno-builtin is NOT optional here: without it, clang recognizes
	# malloc/calloc/realloc/free as builtin allocation functions and, when a
	# call's returned pointer never "escapes" (isn't dereferenced/printed,
	# only null-checked and freed), silently deletes the whole
	# alloc+free pair as dead code at -O2 -- verified empirically while
	# building this test (the deliberate calloc()/realloc() sanity calls in
	# test_hook_actually_counts() were compiled away entirely). That would
	# make this test pass for the wrong reason (no calls to observe, rather
	# than calls correctly observed at zero).
	$(CC) $(CFLAGS) -fno-builtin -MD -MP -c -o $(OBJ_DIR)/test_fft_zero_heap.o test/test_fft_zero_heap.c
ifeq ($(shell uname -s),Darwin)
	$(LINK) -o $(BIN_DIR)/test_fft_zero_heap $(OBJ_DIR)/test_fft_zero_heap.o $(LIB) $(LDFLAGS) $(ZERO_HEAP_LDFLAGS)
else
	$(LINK) -o $(BIN_DIR)/test_fft_zero_heap $(OBJ_DIR)/test_fft_zero_heap.o $(ZERO_HEAP_HOOK_LIB) $(LIB) $(LDFLAGS) $(ZERO_HEAP_LDFLAGS)
endif
	@echo "--- audio_common allocator-hook zero-heap test [$(BACKEND)] ---"
	@$(BIN_DIR)/test_fft_zero_heap

# test_ne10_force_c (F11): builds test/test_ne10_c_parity.c TWICE -- once
# against the normal NE10 backend (calls the `_neon`-suffixed kernels) and
# once with EXTRA_CFLAGS=-DFFT_NE10_FORCE_C (fft_wrapper_ne10.c routes every
# FFT call through NE10's `_c` scalar kernels instead -- see that file's
# FFT_NE10_FORCE_C block) -- then diffs the two runs' dumped outputs.
# NE10-specific (the FFT_NE10_FORCE_C knob only exists in
# src/fft_wrapper_ne10.c): BACKEND=ne10 is forced in both sub-makes below
# regardless of how this invocation of `make` itself resolved BACKEND, and
# nothing here touches a BACKEND=kiss build.
#
# Two full sub-`make`s (via $(MAKE) BACKEND=ne10 ...) rebuild the archive
# for each EXTRA_CFLAGS value rather than trying to build both variants in
# one invocation: EXTRA_CFLAGS is baked into $(OBJ_DIR)'s AND $(BIN_DIR)'s
# own path via CFG_SIG (see the hash-keyed directory section above), so the
# two sub-makes automatically land in two DIFFERENT bin/ne10-<sig>/
# directories -- the second sub-make's archive/objects are a real fresh
# build, not a stale reuse of the first's, with nothing needing to be wiped
# in between, and (round-3 review B01, the actual bug this rewrite fixes)
# the FORCED-C sub-make's rebuild of libaudio_common.a can never overwrite
# the normal NEON build's archive, because the two now live in different
# directories instead of the same flat bin/ne10/.
#
# Each variant's bin dir is resolved at RECIPE time (not parsed/cached) via
# `$(MAKE) ... print-bin-dir`, so this target never has to guess or
# reconstruct a keyed path itself -- it just asks the sub-make that built it.
test_ne10_force_c:
	@$(MAKE) BACKEND=ne10 _ne10_parity_bin
	@bd_neon="$$($(MAKE) -s --no-print-directory BACKEND=ne10 print-bin-dir)"; \
	 $(MAKE) BACKEND=ne10 EXTRA_CFLAGS=-DFFT_NE10_FORCE_C _ne10_parity_bin; \
	 bd_c="$$($(MAKE) -s --no-print-directory BACKEND=ne10 EXTRA_CFLAGS=-DFFT_NE10_FORCE_C print-bin-dir)"; \
	 if [ "$$bd_neon" = "$$bd_c" ]; then \
	   echo "FATAL: test_ne10_force_c: NEON and forced-C variants resolved to the SAME bin dir ($$bd_neon) -- CFG_SIG did not separate them" >&2; \
	   exit 1; \
	 fi; \
	 echo "--- audio_common NE10 NEON-vs-C parity [test_ne10_force_c] ---"; \
	 echo "  NEON     bin dir: $$bd_neon"; \
	 echo "  forced-C bin dir: $$bd_c"; \
	 "$$bd_neon/test_ne10_c_parity" "$$bd_neon/test_ne10_c_parity_neon.dump"; \
	 "$$bd_c/test_ne10_c_parity"    "$$bd_c/test_ne10_c_parity_c.dump"; \
	 $(CC) -O2 -Wall -Wextra -o "$$bd_c/ne10_parity_compare" test/ne10_parity_compare.c -lm; \
	 "$$bd_c/ne10_parity_compare" "$$bd_neon/test_ne10_c_parity_neon.dump" "$$bd_c/test_ne10_c_parity_c.dump"

# _ne10_parity_bin: internal plumbing target invoked (via $(MAKE) BACKEND=ne10
# [EXTRA_CFLAGS=...] _ne10_parity_bin) by test_ne10_force_c above; same shape
# as test_pool/test_wav/etc. Because BIN_DIR is now CFG_SIG-keyed, the two
# sub-make invocations (plain NE10 vs EXTRA_CFLAGS=-DFFT_NE10_FORCE_C) each
# land in their OWN bin/ne10-<sig>/test_ne10_c_parity -- no cp-to-rename
# dance is needed any more to keep one variant's binary from being clobbered
# by the other's rebuild (round-3 review B01).
_ne10_parity_bin: $(LIB) | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $(OBJ_DIR)/test_ne10_c_parity.o test/test_ne10_c_parity.c
	$(LINK) -o $(BIN_DIR)/test_ne10_c_parity $(OBJ_DIR)/test_ne10_c_parity.o $(LIB) $(LDFLAGS)

# --- Query targets (round-3 review B01) -------------------------------------
# Consumers (AEC/NR Makefiles, docs_smoke.sh, scripts) resolve THIS build's
# actual output paths at recipe time instead of hardcoding
# bin/<backend>/libaudio_common.a -- e.g.
# `ac="$$(make -s -C ../../audio_common BACKEND=$$BACKEND print-lib-path)"`.
# $(abspath ...) is fine here even though it prints an absolute path: the
# prohibition on absolute paths is about paths landing in COMMITTED files
# (or generated manifests), not runtime `make` stdout -- and an absolute
# path is exactly what a caller in a different, unknown-relative-depth
# directory needs to reliably -I/-L/link against. Parse-time itself never
# writes stdout (only $(info) above does, and that's gated off for these
# goals); these targets' only output is the one recipe-time @echo line.
print-bin-dir:
	@echo $(abspath $(BIN_DIR))
print-obj-dir:
	@echo $(abspath $(OBJ_DIR))
print-lib-path:
	@echo $(abspath $(LIB))

# --- publish v4: immutable, content-addressed release under $(DIST_ROOT) ---
# (round-3 B01 -> round-4 P2-2 -> round-5 P2.) Layout per backend:
#
#   $(DIST_ROOT)/<backend>/<cfg_sig>-<content12>/
#       <artifacts...>     immutable after first publish
#       MANIFEST.txt       immutable, DETERMINISTIC (config + tool identities
#                          + per-file SHA-256 only -- no commit/date, so the
#                          same release id always has the same MANIFEST bytes
#                          and republish can verify it byte-for-byte)
#       ATTEST/            append-only provenance: EXACTLY one new
#                          attest-<utc>-<commit>[-dirty]-<seq>.txt per
#                          publish EVENT (incl. idempotent republishes),
#                          carrying event_id/git_commit (FULL 40-hex OID)/
#                          git_dirty/date_utc/allow_dirty_publish -- "which
#                          checkout published this" lives here, not in
#                          MANIFEST. Round-6: installed via the helper's
#                          --excl-install mode (write-temp + link(2), the
#                          atomic no-clobber equivalent of O_CREAT|O_EXCL);
#                          a name collision (same second, same commit)
#                          regenerates the event id with the next <seq> and
#                          retries -- an existing attestation is NEVER
#                          overwritten (round-5 used mv -f, so same-second
#                          republishes silently clobbered the prior record).
#                          Attestations are UNSIGNED: they provide
#                          traceability, not authenticity -- anyone with
#                          filesystem access could forge one, so do not
#                          present ATTEST/ as tamper-proof under an attacker
#                          model (MANIFEST byte-verification is the
#                          integrity check; ATTEST is the provenance log).
#   $(DIST_ROOT)/<backend>/current -> <rel_id>   (atomic swap, see below)
#
# Republish of an existing release id verifies EVERY staged file -- the
# artifacts AND MANIFEST.txt -- byte-for-byte against the stored release
# (missing/differing file = FATAL: content-address collision or on-disk
# tampering), then appends a new attestation and repoints `current`. Nothing
# in an existing release dir is ever modified or deleted (ATTEST/ gains
# files; artifacts/MANIFEST never change after first publish).
#
# Dirty policy (round-6 review P2): publish FATALs by default when this
# checkout has uncommitted tracked changes (`git status --porcelain` after
# excluding nothing -- untracked files count here) or no git identity at
# all. ALLOW_DIRTY_PUBLISH=1 overrides; the attestation then records
# allow_dirty_publish=1 plus dirty_diff_sha256 (sha256 of
# `git diff --binary HEAD`, tracked changes only) so the deviation is
# itself traceable.
#
# `current` is swapped with tools/atomic_symlink_swap.c (built at publish
# time with $(HOSTCC), never the possibly-cross $(CC)): symlink to a
# PID-suffixed temp + rename(2), which never follows symlinks -- truly
# atomic on macOS AND GNU/Linux, with no missing-`current` window and no
# destructive fallback (round-5: `mv -fT` doesn't exist on BSD/macOS, and
# the previous rm+mv fallback both left a window and fired on ANY mv error).
#
# Locking (round-5): `publish` is a DRIVER that takes the per-backend mkdir
# lock FIRST and only then recursively builds+stages via _publish_impl -- so
# two concurrent same-config publishes can no longer race each other's
# object/archive writes during the prerequisite build (the previous shape
# built $(PUBLISH_ARTIFACTS) before its recipe ever took the lock). The
# loser fails fast having written nothing. _publish_impl is INTERNAL:
# invoking it directly skips the lock.
PUBLISH_ARTIFACTS = $(LIB)
DIST_BACKEND_DIR  = $(DIST_ROOT)/$(BACKEND)
DIST_LOCK         = $(DIST_ROOT)/.lock.$(BACKEND)

.PHONY: publish _publish_impl
# Dry-run guard (round-6 review P2-3): this recipe mentions $(MAKE), so GNU
# make executes it for real under -n/-q/-t (all three share that rule) --
# which used to mkdir $(DIST_ROOT) persistently and take/release the real
# publish lock on every dry run. The word-scan below detects those flags in
# MAKEFLAGS and branches: -n/-t exec a recursion-only path (the child make
# inherits the flag and merely prints / applies standard -t touch semantics
# to the impl chain); -q exits 1 directly (question mode's documented
# "needs updating" answer -- publish is phony, so it always would run --
# with no output). Either way: no DIST_ROOT, no lock, no stage/helper/
# MANIFEST/ATTEST writes. MAKEFLAGS parsing (verified against GNU make
# 3.81: 17 flag combinations incl. long forms): bundled no-dash short flags
# may appear as the first word ("n", "sn") OR as separate dashed words
# ("-n" after a long option); long options start with "--"; a literal "--"
# separates flags from command-line variable overrides (which can
# themselves contain n/q/t -- BACKEND=ne10 -- hence the explicit skips).
# Normal invocations keep the round-5 lock-FIRST discipline unchanged.
publish:
	+@has_n=0; has_q=0; has_t=0; \
	for w in $(MAKEFLAGS); do \
	  case "$$w" in --) break ;; --*|*=*) continue ;; -n|-q|-t) ;; -*) continue ;; esac; \
	  case "$$w" in *n*) has_n=1 ;; esac; \
	  case "$$w" in *q*) has_q=1 ;; esac; \
	  case "$$w" in *t*) has_t=1 ;; esac; \
	done; \
	if [ "$$has_n" = 1 ] || [ "$$has_t" = 1 ]; then exec $(MAKE) --no-print-directory _publish_impl; fi; \
	if [ "$$has_q" = 1 ]; then exit 1; fi; \
	mkdir -p $(DIST_ROOT); \
	if ! mkdir "$(DIST_LOCK)" 2>/dev/null; then \
	  echo "FATAL: publish lock $(DIST_LOCK) already held (concurrent 'make publish BACKEND=$(BACKEND)'?)" >&2; \
	  exit 1; \
	fi; \
	trap 'rmdir "$(DIST_LOCK)" 2>/dev/null' EXIT INT TERM; \
	$(MAKE) --no-print-directory _publish_impl

_publish_impl: $(PUBLISH_ARTIFACTS)
	@set -e; \
	work="$$(mktemp -d)"; \
	tmp="$(DIST_BACKEND_DIR)/.stage.$$$$"; \
	trap 'rm -rf "$$tmp" "$$work"' EXIT INT TERM; \
	commit="$$(git rev-parse HEAD 2>/dev/null || echo unknown)"; \
	if [ "$$commit" = "unknown" ]; then dirty=unknown; else \
	  dirty=0; [ -n "$$(git status --porcelain 2>/dev/null)" ] && dirty=1; \
	fi; \
	ddiff=""; [ "$$dirty" = "1" ] && ddiff="$$(git diff --binary HEAD 2>/dev/null | shasum -a 256 | cut -d' ' -f1)"; \
	adp=0; [ "$(ALLOW_DIRTY_PUBLISH)" = "1" ] && adp=1; \
	if [ "$$adp" != "1" ] && [ "$$dirty" != "0" ]; then \
	  if [ "$$dirty" = "unknown" ]; then \
	    echo "FATAL: publish refused -- audio_common has no git identity here (not a git checkout), so the attestation cannot name the source state." >&2; \
	  else \
	    echo "FATAL: publish refused -- audio_common working tree is dirty (uncommitted changes)." >&2; \
	  fi; \
	  echo "  Commit (or clone a clean checkout) for a board deliverable, or pass ALLOW_DIRTY_PUBLISH=1" >&2; \
	  echo "  for a dev publish -- the escape hatch is recorded in the attestation." >&2; \
	  exit 1; \
	fi; \
	dateutc="$$(date -u +%Y-%m-%dT%H:%M:%SZ)"; \
	stamp='$(ATTEST_STAMP)'; [ -n "$$stamp" ] || stamp="$$(date -u +%Y%m%dT%H%M%SZ)"; \
	$(HOSTCC) -O2 -o "$$work/swapln" tools/atomic_symlink_swap.c || { echo "FATAL: could not build the atomic symlink-swap helper with HOSTCC=$(HOSTCC) (pass HOSTCC=<host compiler>)" >&2; exit 1; }; \
	mkdir -p "$(DIST_BACKEND_DIR)"; \
	rm -rf "$$tmp"; mkdir -p "$$tmp"; \
	cp $(PUBLISH_ARTIFACTS) "$$tmp/"; \
	shas="$$( cd "$$tmp" && for f in *; do shasum -a 256 "$$f"; done )"; \
	content12="$$(printf '%s\n' "$$shas" | sort | shasum -a 256 | cut -c1-12)"; \
	rel_id="$(CFG_SIG)-$$content12"; \
	rel_dir="$(DIST_BACKEND_DIR)/$$rel_id"; \
	{ echo "repo=audio_common"; \
	  echo "release_id=$$rel_id"; \
	  echo "backend=$(BACKEND)"; \
	  echo "cfg_sig=$(CFG_SIG)"; \
	  echo "cc=$(CC)"; echo "cxx=$(CXX)"; echo "ar=$(AR)"; echo "ranlib=$(RANLIB)"; echo "link=$(LINK)"; \
	  echo "cflags=$(CFLAGS)"; echo "cxxflags=$(CXXFLAGS)"; \
	  echo "extra_cflags=$(EXTRA_CFLAGS)"; \
	  echo "werror=$(WERROR)"; echo "no_stdio=$(NO_STDIO)"; echo "debug=$(DEBUG)"; \
	  echo "$$shas"; \
	} > "$$tmp/MANIFEST.txt"; \
	suffix=""; [ "$$dirty" = "1" ] && suffix="-dirty"; \
	echo "== publish: backend-symbol audit (staged archive) =="; \
	if [ "$(BACKEND)" = "ne10" ]; then \
	  nm "$$tmp/libaudio_common.a" | grep -Eq '_?ne10_fft_r2c_1d_float32_neon' || { echo "FATAL: staged ne10 archive missing ne10_fft_r2c_1d_float32_neon" >&2; exit 1; }; \
	  ar -t "$$tmp/libaudio_common.a" | grep -q kiss_fft.o && { echo "FATAL: staged ne10 archive unexpectedly contains kiss_fft.o" >&2; exit 1; }; \
	  true; \
	else \
	  nm "$$tmp/libaudio_common.a" | grep -Eq '_?kiss_fft_alloc' || { echo "FATAL: staged kiss archive missing kiss_fft symbols" >&2; exit 1; }; \
	  ar -t "$$tmp/libaudio_common.a" | grep -qi ne10 && { echo "FATAL: staged kiss archive unexpectedly contains an ne10 object" >&2; exit 1; }; \
	  true; \
	fi; \
	if [ -d "$$rel_dir" ]; then \
	  for f in "$$tmp"/*; do \
	    bn="$$(basename "$$f")"; \
	    cmp -s "$$f" "$$rel_dir/$$bn" || { echo "FATAL: existing release $$rel_id differs from staged content for $$bn (content-address collision, or the stored release/MANIFEST was modified on disk)" >&2; exit 1; }; \
	  done; \
	  rm -rf "$$tmp"; \
	  echo "audio_common: $$rel_id already published (byte-verified, incl. MANIFEST) -- appending attestation + repointing current"; \
	else \
	  mkdir "$$tmp/ATTEST"; \
	  mv "$$tmp" "$$rel_dir"; \
	fi; \
	mkdir -p "$$rel_dir/ATTEST"; \
	n=1; \
	while :; do \
	  attest_name="attest-$$stamp-$$commit$$suffix-$$(printf '%03d' "$$n").txt"; \
	  { echo "event_id=$${attest_name%.txt}"; \
	    echo "repo=audio_common"; \
	    echo "release_id=$$rel_id"; \
	    echo "cfg_sig=$(CFG_SIG)"; \
	    echo "backend=$(BACKEND)"; \
	    echo "git_commit=$$commit"; echo "git_dirty=$$dirty"; \
	    [ -n "$$ddiff" ] && echo "dirty_diff_sha256=$$ddiff"; \
	    echo "date_utc=$$dateutc"; \
	    echo "allow_dirty_publish=$$adp"; \
	  } > "$$work/attest.txt"; \
	  rc=0; "$$work/swapln" --excl-install "$$work/attest.txt" "$$rel_dir/ATTEST/$$attest_name" || rc=$$?; \
	  [ "$$rc" -eq 0 ] && break; \
	  [ "$$rc" -eq 2 ] || { echo "FATAL: could not install attestation $$attest_name" >&2; exit 1; }; \
	  n=$$((n+1)); \
	  [ "$$n" -le 999 ] || { echo "FATAL: 999 attestation name collisions for $$stamp-$$commit (runaway?)" >&2; exit 1; }; \
	done; \
	"$$work/swapln" "$$rel_id" "$(DIST_BACKEND_DIR)/current"; \
	echo "audio_common: published $(BACKEND)/$$rel_id -> $(DIST_BACKEND_DIR)/current (attested: $$attest_name)"

clean:
	# F12 build hygiene: nuke the obj and bin roots wholesale (BOTH backends'
	# output dirs, every CFG_SIG subdirectory each), not just
	# $(OBJ_DIR)/$(BIN_DIR) for whatever BACKEND/config this invocation
	# happens to resolve to -- a bare `make clean` (no BACKEND=) used to
	# leave other configs' stale builds sitting untouched. Uses the
	# OBJ_ROOT/BIN_ROOT knobs (round-6) so `clean` with an override scrubs
	# THAT tree, never the real obj/ and bin/. dist/ is NOT removed by clean
	# (round-3 review B01): published releases are meant to survive a
	# dev-tree clean; `rm -rf dist/` is a manual, deliberate action, not
	# part of the normal edit/build/clean loop.
	rm -rf $(OBJ_ROOT) $(BIN_ROOT)

# --- Header dependency tracking (round-3 review B01, build hygiene) --------
# Every compile line above passes -MD -MP, which emits a $(OBJ_DIR)/<name>.d
# GNU-Make fragment per object, listing every header that object's TU pulled
# in -- INCLUDING vendored NE10/KISS headers reached via -isystem (-MD, not
# -MMD, which would omit -isystem/system headers and defeat "touch a
# vendored NE10 header -> its object rebuilds"). -MP additionally emits a
# phony no-prerequisites rule per header, so a deleted/renamed header doesn't
# fail the build with "No rule to make target" against a stale .d fragment.
# $(wildcard ...) makes this evaluate to nothing (not an error) before any
# objects exist yet for this CFG_SIG (a clean tree, or this config's first
# build).
-include $(wildcard $(OBJ_DIR)/*.d)
