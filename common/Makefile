# common — shared audio DSP layer (NE10 FFT wrapper + fast_math + common DSP)
#
# One copy of the FFT layer for every consumer repo (AEC, NR, Audio_ALG). Consumers add
# `-I common/include` and link the archive this Makefile builds.
#
# The one FFT backend is NE10 (NE10_DSP_RFFT_SCALING on), for ARM32/ARM64. Targets with NEON use its
# NEON FFT; SIMD=0 or targets without NEON use its scalar C FFT. Other shared SIMD kernels keep
# their own ISA guards. fft_wrapper.h has a heap path (fft_create/fft_destroy) and a static-memory
# path (fft_get_mem_size/fft_init) selected by a runtime pool_owned flag.

CC      ?= cc
CXX     ?= c++

# Archiver tools are explicit so a cross-compiling caller overriding AR/RANLIB
# (e.g. an arm-none-eabi- toolchain) lands in a different config signature, same
# as overriding CC/CXX.
AR      ?= ar
RANLIB  ?= ranlib

# CFG_SIG payload placeholders: CPPFLAGS/EXTRA_LDFLAGS/NO_STDIO/DEBUG are not consumed by this
# Makefile, but the payload always includes them (with explicit, possibly empty defaults) so its
# shape matches the other repos' Makefiles and a future use keys builds correctly.
#
# CPPFLAGS uses `+=`, not `?=`: `?=` is a no-op for an environment value, so under `make -e` its
# origin would never flip to "environment override" and the rejection below would miss it.
CPPFLAGS      +=
EXTRA_LDFLAGS ?=
NO_STDIO      ?= 0
DEBUG         ?= 0

# TOOLCHAIN_CHECK=0 skips the CC/CXX -dumpmachine target-coherence guard in
# _cfg_guard below. It participates in CFG_SIG, so a guard-skipped build never
# aliases a guarded one.
TOOLCHAIN_CHECK ?= 1

# Placement and publish knobs. None is in CFG_SIG: they change where something is written or whether
# publish proceeds, never what is built.
#   DIST_ROOT         where `make publish` writes its release tree (default dist/).
#   OBJ_ROOT/BIN_ROOT where the keyed obj/bin trees live (default obj/, bin/).
#   HOSTCC            compiles tools/atomic_symlink_swap.c at publish time; it must run on the build
#                     host, so never $(CC), which may be a cross compiler.
#   ALLOW_DIRTY_PUBLISH      publish refuses by default when tracked files have uncommitted changes
#                     or the checkout has no git identity (the attestation could not name the source
#                     state). =1 admits tracked changes and records the sha256 of
#                     `git diff --binary HEAD`; an identity-less checkout is always refused.
#   ALLOW_UNTRACKED_PUBLISH  publish also refuses when an untracked, non-ignored file exists (it can
#                     change the artifact without a trace in the tracked diff). =1 admits them and
#                     records untracked_tree_sha256, a sha256 over sorted fixed-field records
#                     ("L <sha256(path)> <sha256(target)>" for symlinks, "F <mode> <sha256(path)>
#                     <sha256(content)>" for files; every variable-length value is hashed, and a
#                     stat/readlink/shasum failure becomes a fatal "X" record). A path that is
#                     neither a regular file nor a symlink (embedded git checkout, fifo) is always
#                     refused by name.
#   ATTEST_STAMP      test-only: overrides the UTC stamp in the attestation file name so tests can
#                     force same-second collisions.
DIST_ROOT ?= dist
HOSTCC    ?= cc
OBJ_ROOT  ?= obj
BIN_ROOT  ?= bin
ALLOW_DIRTY_PUBLISH ?= 0
ALLOW_UNTRACKED_PUBLISH ?= 0
ATTEST_STAMP ?=

# -Werror opt-in (default off; `make WERROR=1`). Declared this early because CFG_SIG
# below folds WERROR into its hash: GNU Make expands variables in a rule's target and
# prerequisite list (including a target-specific assignment's target) when that line is
# parsed, not when the recipe runs, so CFG_SIG must be a fully resolved `:=` before the
# first rule that mentions anything derived from it.
WERROR ?= 0

# One project-wide switch for every optional SIMD path. SIMD=1 is the production default; SIMD=0 keeps
# the selected FFT backend but dispatches our kernels and the NE10 wrapper through scalar C.
#
# SIMD=0 and SIMD=1 are not bit-identical end to end and are not contracted to be: kernels in
# utility/simd_kernels.h are byte-exact against their scalar twins (test/simd_selftest.c), but the AEC
# matched filter deliberately diverges (four accumulators plus vfmaq_f32; fused NLMS update). Across
# configurations only finite output, the same delay winner/state and a quality tolerance are
# guaranteed.
SIMD ?= 1
ifneq ($(words $(SIMD)),1)
$(error SIMD must be exactly 0 or 1, got '$(SIMD)')
endif
ifneq ($(filter-out 0 1,$(SIMD)),)
$(error SIMD must be exactly 0 or 1, got '$(SIMD)')
endif
ifeq ($(SIMD),0)
SIMD_CFLAGS := -DSIMD_KERNELS_FORCE_SCALAR
else
SIMD_CFLAGS :=
endif

# The only FFT backend is ne10. BACKEND stays a variable because it names the keyed
# obj/bin/dist directories and the consumer Makefiles forward it; any other value is
# rejected here instead of silently building something else.
BACKEND ?= ne10
ifneq ($(BACKEND),ne10)
$(error BACKEND must be ne10 (got '$(BACKEND)'): the KISS FFT backend has been removed)
endif
# Gated on MAKECMDGOALS: print-bin-dir/print-obj-dir/print-lib-path are captured by other
# Makefiles' recipes (`ac="$$(make -s ... print-lib-path)"`) and by scripts, so their
# stdout must be only the path. Parse time runs before any target is selected, so an
# ungated banner would corrupt the captured value.
ifeq ($(filter print-%,$(MAKECMDGOALS)),)
$(info common: FFT backend = $(BACKEND))
endif

# gnu99 (not strict c99): the vendored NE10 headers use the GNU asm("label")
# extension. Consumers may still compile their own code with -std=c99 and link
# this archive; different TUs may use different standards.
CFLAGS  += -Wall -Wextra -O2 -std=gnu99 -Iinclude -I.
# -isystem (not -I) for the vendored NE10 include dirs: our own headers (include/) stay
# -I so their diagnostics remain visible, while the vendored headers carry warnings we
# do not own and are demoted to system headers.
CXXFLAGS+= -O2 -isystem lib/ne10/inc -isystem lib/ne10/common -isystem lib/ne10/modules
LDFLAGS += -lm

# EXTRA_CFLAGS lets callers inject unrelated project defines/flags into every
# compiled TU.  SIMD selection intentionally goes through `SIMD=0|1` only.
CFLAGS  += $(EXTRA_CFLAGS)
CXXFLAGS+= $(EXTRA_CFLAGS)
CFLAGS  += $(SIMD_CFLAGS)
CXXFLAGS+= $(SIMD_CFLAGS)
LDFLAGS += $(EXTRA_LDFLAGS)

# Object and binary directories are keyed by backend and CFG_SIG (computed below, after
# every CFLAGS/CXXFLAGS append), so builds that differ in any signed input -- SIMD,
# EXTRA_CFLAGS, WERROR, ... -- each get their own directory. Nothing is wiped to switch
# between them and two builds can run concurrently in one tree without touching each
# other's objects or archive. Both are recursive (`=`) assignments that reference
# CFG_SIG before its `:=` line; that is safe because nothing expands them until
# CFG_SIG holds its final value.
OBJ_DIR = $(OBJ_ROOT)/$(BACKEND)-$(CFG_SIG)
BIN_DIR = $(BIN_ROOT)/$(BACKEND)-$(CFG_SIG)

# Shared DSP sources (always in the archive).
COMMON_SRCS = src/hpf.c src/audio_resampler.c src/audio_window.c src/audio_stft.c

# NE10 source footprint: fft_wrapper_ne10.c only calls the float32 R2C/C2R entry points
# (ne10_fft_alloc_r2c_float32, ne10_fft_r2c_mem_size_float32, ne10_fft_init_r2c_float32_ext,
# ne10_fft_destroy_r2c_float32, ne10_fft_r2c_1d_float32_neon, ne10_fft_c2r_1d_float32_neon) and never
# ne10_init(), so this is an explicit list, not a wildcard: the smallest set that links without
# editing NE10 source.
#   - NE10_rfft_float32.c / .neonintrinsic.c: the R2C/C2R implementation (scalar reference + NEON),
#     including the external-memory init patch (lib/ne10/VENDORED.md).
#   - NE10_fft.c: the R2C path uses its ne10_factor() and twiddle generators; the heap fft_destroy()
#     uses its ne10_fft_destroy_r2c_float32().
#   - NE10_fft_float32.c, NE10_fft_int32.c, NE10_fft_generic_float32.c, NE10_fft_generic_int32.cpp:
#     never called, but NE10_fft.c defines ne10_fft_alloc_c2c_*_neon unconditionally and those
#     reference these files' allocators and butterfly helpers, and a static archive resolves whole .o
#     files. The .cpp is why the archive needs a C++ TU (LINK = $(CXX) below).
# Not compiled: NE10_init*.c, the NEON c2c and generic compute kernels, the int16 FFT and the FIR/IIR
# families; they stay under lib/ne10/ for provenance.
NE10_C_SRCS  = lib/ne10/modules/dsp/NE10_fft.c \
               lib/ne10/modules/dsp/NE10_fft_float32.c \
               lib/ne10/modules/dsp/NE10_fft_int32.c \
               lib/ne10/modules/dsp/NE10_fft_generic_float32.c \
               lib/ne10/modules/dsp/NE10_rfft_float32.c \
               lib/ne10/modules/dsp/NE10_rfft_float32.neonintrinsic.c
NE10_CXXSRCS = lib/ne10/modules/dsp/NE10_fft_generic_int32.cpp
# OWN_SRCS is code we wrote and own the warnings for; VENDOR_SRCS/VENDOR_CXXSRCS is
# unmodified NE10 (see the -Werror scoping below).
OWN_SRCS       = src/fft_wrapper_ne10.c $(COMMON_SRCS)
VENDOR_SRCS    = $(NE10_C_SRCS)
VENDOR_CXXSRCS = $(NE10_CXXSRCS)
# NE10_DSP_RFFT_SCALING makes c2r normalise by 1/N; it is default-on in NE10_fft.h and defined
# explicitly so no build can silently drop it.
# The vendored RFFT implementation and external-memory API use the level-1
# config layout on both ABIs (upstream otherwise defaults ARM32 to level0).
NE10_DEFS = -DNE10_ENABLE_DSP -DNE10_DSP_RFFT_SCALING -DNE10_UNROLL_LEVEL=1
CFLAGS  += -isystem lib/ne10/inc -isystem lib/ne10/common -isystem lib/ne10/modules $(NE10_DEFS)
CXXFLAGS+= $(NE10_DEFS)

# Link driver, declared before CFG_SIG so it is part of the payload (a command-line
# LINK= override changes the produced binaries, so it must change their keyed
# directory). The archive contains a C++ TU, so linking uses the C++ driver, which also
# supplies the right C++ runtime (libc++ on macOS/clang, libstdc++ on GNU/Linux gcc).
LINK = $(CXX)

# --- Policy flags and the command-line override rejection ---------------------
# FP_POLICY, FFT_WRAPPER_ALIAS_CFLAGS and FP_ALLOWED_CHARS_RE are bare `:=` literals placed ahead of the
# override-rejection foreach below. Under `make -e` a variable's origin becomes "environment override"
# only when a makefile assignment to it is parsed, so one assigned after the foreach would still look
# like plain "environment" there and an env override would slip through. Only the "command line"
# origin is position-independent. (FP_POLICY is appended to CFLAGS/CXXFLAGS further below.)
#
# FP_POLICY: keep scalar/NEON operation ordering stable (no contraction) and let
# AArch64 lower guarded sqrtf calls to a single FSQRT without an errno-checking libm
# fallback.
FP_POLICY := -ffp-contract=off -fno-math-errno
# -fno-strict-aliasing, applied to fft_wrapper_ne10.o only (see below).
FFT_WRAPPER_ALIAS_CFLAGS := -fno-strict-aliasing
# Character allow-list for the flag variables (used by the FP-policy gate below).
FP_ALLOWED_CHARS_RE := ^[A-Za-z0-9_./=,+ -]*$$

# CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS and the policy variables are internal. GNU Make ignores this file's
# own `+=`/`:=` assignments to a variable set on the command line (`make CFLAGS=-O3`), which would
# strip the FP-contraction policy, the NE10 defines and include paths, and -lm. The supported hooks
# are EXTRA_CFLAGS and EXTRA_LDFLAGS (covered by CFG_SIG, policy appended after them). A plain
# environment value is folded in by this file's own assignments; only the "command line" and
# "environment override" origins defeat them. Every internal variable must be defined early enough
# (bare `:=` or unconditional `+=`, never `?=`) that its origin is final before the foreach runs.
$(foreach v,CFLAGS CXXFLAGS CPPFLAGS LDFLAGS FP_POLICY FFT_WRAPPER_ALIAS_CFLAGS FP_ALLOWED_CHARS_RE,\
  $(if $(findstring command,$(origin $(v)))$(findstring override,$(origin $(v))),\
    $(error $(v) cannot be overridden (origin: $(origin $(v))) -- it would silently drop this Makefile's own flag appends (FP policy, backend defines); use EXTRA_CFLAGS / EXTRA_LDFLAGS instead)))

# --- FP-contraction policy ---------------------------------------------------
# Every TU this Makefile compiles, ours and vendored NE10 alike, builds with -ffp-contract=off.
# FP_POLICY is appended last, after every other CFLAGS/CXXFLAGS append and before CFG_SIG_PAYLOAD is
# computed, so it is part of CFG_SIG and EXTRA_CFLAGS cannot re-enable contraction. It does not touch
# explicit fusion requests in source (fmaf() in fft_power(), vfmaq_f32 in the NEON blocks);
# scripts/audit_fp_contract.sh verifies both classes at disassembly level.
#
# Conflict detection runs on the fully assembled CFLAGS/CXXFLAGS/CPPFLAGS (including EXTRA_CFLAGS and
# plain-environment CXXFLAGS/CPPFLAGS, which reach the NE10 C++ TU without EXTRA_CFLAGS): -Ofast,
# -ffast-math and -ffp-contract=<any> are rejected before CFG_SIG exists, so a rejected build creates
# no obj/bin directory. $(filter) matches whole words, so a harmless `-DNOTE=-Ofastness` passes.
FP_INPUT_FLAGS := $(CFLAGS) $(CXXFLAGS) $(CPPFLAGS)

# LDFLAGS is embedded in every link recipe just as CFLAGS is in every compile recipe,
# so the character-safety checks below run on the union.
SHELL_SAFE_INPUT_FLAGS := $(FP_INPUT_FLAGS) $(LDFLAGS)

# GNU Make has no notion of shell quoting: EXTRA_CFLAGS="'-Ofast'" is stored as quoted text that never
# equals the word -Ofast, yet the shell strips the quotes and the compiler receives -Ofast. A
# deny-list cannot keep up (globs, tilde, redirection, process substitution), so the flags are checked
# against a character allow-list: letters, digits, underscore, dot, slash, equals, comma, plus, minus,
# and space as the separator. That covers every flag this Makefile and its tests use (-O2, -std=gnu99,
# -isystem lib/ne10/inc, -DNE10_DSP_RFFT_SCALING, -DNAME=value, -Wl,-rpath,dir, ...).
#
# The allow-list shells out to `grep -E` with the flags single-quoted, so the single-quote check below
# stays as the one deny-list entry; it runs first, as a pure $(findstring), to make that embedding safe.
ifneq ($(findstring ',$(SHELL_SAFE_INPUT_FLAGS)),)
$(error FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/EXTRA_CFLAGS/EXTRA_LDFLAGS contains a single-quote character; this repo requires plain, unquoted, whitespace-separated compiler/linker argv tokens built only from the allowed character set (see the allow-list below) -- a single quote specifically must be rejected by this Make-native check BEFORE the allow-list's own $(shell) call can safely single-quote-embed the rest of the text)
endif

# LC_ALL=C pins bracket-expression ranges to ASCII order.
SHELL_SAFE_ALLOWLIST_RC := $(shell printf '%s' '$(SHELL_SAFE_INPUT_FLAGS)' | LC_ALL=C grep -Eq '$(FP_ALLOWED_CHARS_RE)'; echo $$?)
ifneq ($(SHELL_SAFE_ALLOWLIST_RC),0)
# Diagnostic only, runs on the failing path: delete every allowed character from the
# flags to name the offending one(s) in the error.
SHELL_SAFE_DISALLOWED_CHARS := $(shell printf '%s' '$(SHELL_SAFE_INPUT_FLAGS)' | LC_ALL=C tr -d 'A-Za-z0-9_./=,+ -' | fold -w1 | LC_ALL=C sort -u | tr -d '\n')
$(error FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/EXTRA_CFLAGS/EXTRA_LDFLAGS contains a character outside the allowed set [A-Za-z0-9_./=,+- ] (letters/digits/underscore/dot/slash/equals/comma/plus/minus, plus space as the token separator) -- disallowed character(s) found: "$(SHELL_SAFE_DISALLOWED_CHARS)"; this repo requires plain, unquoted, whitespace-separated compiler/linker argv tokens built only from that set -- remove/replace the disallowed character(s) shown above in CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/EXTRA_CFLAGS/EXTRA_LDFLAGS)
endif

FP_CONFLICT_FLAGS := $(filter -Ofast -ffast-math -ffp-contract=%,$(FP_INPUT_FLAGS))
ifneq ($(FP_CONFLICT_FLAGS),)
$(error FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS contains $(FP_CONFLICT_FLAGS); this repo pins -ffp-contract=off; remove $(FP_CONFLICT_FLAGS) from CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS)
endif

# The checks above are internal variables computed from the assembled flags, so they
# cannot be defined early like the bare literals; their command-line/`-e` override
# rejection must come after their own assignments or the origin has not flipped yet. An
# override such as `SHELL_SAFE_ALLOWLIST_RC=0` would otherwise turn every gate above into
# a no-op while the real flags stay unvalidated. SHELL_SAFE_DISALLOWED_CHARS is
# deliberately not listed: it is diagnostic text on an already failing build.
$(foreach v,FP_INPUT_FLAGS SHELL_SAFE_INPUT_FLAGS SHELL_SAFE_ALLOWLIST_RC FP_CONFLICT_FLAGS,\
  $(if $(findstring command,$(origin $(v)))$(findstring override,$(origin $(v))),\
    $(error $(v) cannot be overridden (origin: $(origin $(v))) -- it would silently defeat the FP-policy/shell-safety allow-list/conflict checks above, leaving the real CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS content unvalidated; this variable is internal to those checks and has no supported external use)))

CFLAGS    += $(FP_POLICY)
CXXFLAGS  += $(FP_POLICY)

# The vendor NEON TU includes arm_neon.h unconditionally. Probe the actual
# compiler flags after policy validation; non-NEON ARM targets need only C.
NE10_TARGET_NEON := $(shell $(CC) $(CFLAGS) -dM -E -x c /dev/null 2>/dev/null | sed -n 's/^\#define __ARM_NEON 1$$/1/p')
$(if $(findstring command,$(origin NE10_TARGET_NEON))$(findstring override,$(origin NE10_TARGET_NEON)),\
  $(error NE10_TARGET_NEON is internal; select the compiler target through CC and EXTRA_CFLAGS))
ifneq ($(NE10_TARGET_NEON),1)
NE10_C_SRCS := $(filter-out lib/ne10/modules/dsp/NE10_rfft_float32.neonintrinsic.c,$(NE10_C_SRCS))
VENDOR_SRCS := $(NE10_C_SRCS)
endif

# --- Hash-keyed object/binary directory + full-coverage CFG_SIG ------------
# obj/$(BACKEND)-<sig> and bin/$(BACKEND)-<sig> isolate every distinct build configuration.
# CFG_SIG_PAYLOAD is the one source string used both to compute CFG_SIG (via cksum) and, verbatim, as
# the collision-guard manifest (_cfg_guard below), so the two cannot drift apart.
#
# Payload coverage: every variable that can affect what a build produces (CC CXX AR RANLIB LINK CFLAGS
# CXXFLAGS CPPFLAGS LDFLAGS EXTRA_CFLAGS EXTRA_LDFLAGS BACKEND SIMD WERROR NO_STDIO DEBUG
# TOOLCHAIN_CHECK), the target signatures, and the SORTED source list (adding or removing a source
# lands in a fresh directory; `ar rc` never deletes stale members). It must be defined after
# CFLAGS/CXXFLAGS are final and before the first rule that mentions $(OBJ_DIR)/$(BIN_DIR), because
# Make expands a rule's target list when the line is parsed.
#
# Not $(strip)'d: whitespace-only differences get different signatures, a harmless extra directory.
# CFG_SIG is a pure string hash, so `make -n` touches nothing and different configs can build
# concurrently, each in its own directory (created lazily by the _cfg_guard order-only prerequisite).
# cksum is POSIX (shasum may be absent from slim images); `cut -d' ' -f1` isolates the checksum.
#
# FFT_WRAPPER_ALIAS_CFLAGS is in the payload because a target-specific `TARGET: CFLAGS += ...` line
# does not change the global $(CFLAGS) captured here: a bare literal on that line would be invisible
# to CFG_SIG, and a keyed directory built before the flag existed would reuse an object compiled
# without it. A flag added to a target-specific CFLAGS/CXXFLAGS line must follow the same shape: name
# it, add the name to this payload, reference the name (not a literal) on that line. The name is also
# rejected as a command-line/`-e` override above.
# The same CC command can select a different SDK through PATH.
# Include effective pointer/FP/ISA predefines to avoid cross-ABI cache reuse.
override CC_TARGET_SIG := $(shell sh scripts/target_abi_signature.sh $(CC) $(CPPFLAGS) $(CFLAGS))
override CXX_TARGET_SIG := $(shell sh scripts/target_abi_signature.sh --cxx $(CXX) $(CPPFLAGS) $(CXXFLAGS))
override CC_ABI_SIG := $(shell sh scripts/target_abi_signature.sh --abi $(CC) $(CPPFLAGS) $(CFLAGS))
override CXX_ABI_SIG := $(shell sh scripts/target_abi_signature.sh --abi --cxx $(CXX) $(CPPFLAGS) $(CXXFLAGS))
CFG_SIG_PAYLOAD := CC_TARGET_SIG=$(CC_TARGET_SIG) CXX_TARGET_SIG=$(CXX_TARGET_SIG) CC_ABI_SIG=$(CC_ABI_SIG) CXX_ABI_SIG=$(CXX_ABI_SIG) CC=$(CC) CXX=$(CXX) AR=$(AR) RANLIB=$(RANLIB) LINK=$(LINK) CFLAGS=$(CFLAGS) CXXFLAGS=$(CXXFLAGS) CPPFLAGS=$(CPPFLAGS) LDFLAGS=$(LDFLAGS) EXTRA_CFLAGS=$(EXTRA_CFLAGS) EXTRA_LDFLAGS=$(EXTRA_LDFLAGS) BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) NO_STDIO=$(NO_STDIO) DEBUG=$(DEBUG) TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK) SRCS=$(sort $(OWN_SRCS) $(VENDOR_SRCS) $(VENDOR_CXXSRCS)) FFT_WRAPPER_ALIAS_CFLAGS=$(FFT_WRAPPER_ALIAS_CFLAGS)
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)

BE_SRCS    = $(OWN_SRCS) $(VENDOR_SRCS)
BE_CXXSRCS = $(VENDOR_CXXSRCS)

BE_OBJS  = $(patsubst %.c,$(OBJ_DIR)/%.o,$(notdir $(BE_SRCS))) \
           $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(BE_CXXSRCS)))
OWN_OBJS = $(patsubst %.c,$(OBJ_DIR)/%.o,$(notdir $(OWN_SRCS)))
VPATH    = src lib/ne10/modules lib/ne10/modules/dsp

# -Werror opt-in, scoped to OUR OWN objects only (WERROR is declared near the top, before
# CFG_SIG). The vendored NE10 TUs go through the same generic pattern rule as our sources
# and carry warnings out of scope to fix here, so a blanket -Werror would fail the build
# on code we do not own. The target-specific variable applies -Werror only to $(OWN_OBJS).
ifeq ($(WERROR),1)
$(OWN_OBJS): CFLAGS += -Werror
endif

# -fno-strict-aliasing, scoped to the fft_wrapper_ne10.c object (same target-specific
# pattern as -Werror, unconditional and narrower). The flag string lives in
# $(FFT_WRAPPER_ALIAS_CFLAGS) and is part of CFG_SIG_PAYLOAD (see above).
#
# fft_forward/fft_inverse cast `Complex*` to `ne10_fft_cpx_float32_t*` in place. The
# layouts match (checked at compile time in fft_wrapper_ne10.c), but the types are distinct, so reading through one
# after writing through the other is a type-based-alias violation. The flag removes that
# assumption for this TU only; every other object keeps strict aliasing.
$(OBJ_DIR)/fft_wrapper_ne10.o: CFLAGS += $(FFT_WRAPPER_ALIAS_CFLAGS)

LIB = $(BIN_DIR)/libaudio_common.a

.PHONY: all lib selftest test test-full test-isolation audit-fp-contract audit-fft-wrapper-pedantic test_stft test_stft_guard test_heap_alignment test-fast-sqrt-codegen test_audio_utils test_pool test_wav test-wav-ubsan test-selftest-ubsan test_zero_heap test_ne10_force_c _ne10_parity_bin clean publish print-bin-dir print-obj-dir print-lib-path print-fp-policy _cfg_guard
all: lib

lib: $(LIB)

test-fast-sqrt-codegen:
	@FP_POLICY='$(FP_POLICY)' scripts/audit_fast_sqrt_codegen.sh

# --- CFG_SIG collision guard + directory creation ----------------------------
# _cfg_guard is the single order-only prerequisite of every compile/link/archive rule. Being .PHONY it
# runs once per invocation even under `make -j`, and does two things:
#   1. mkdir -p $(OBJ_DIR) $(BIN_DIR); idempotent, safe when two make processes race on one config.
#   2. Writes (first time) or verifies (afterwards) a one-line config.manifest in each directory
#      holding the exact CFG_SIG_PAYLOAD. A mismatch means a cksum collision or a payload that misses
#      a build input, so objects from an unrelated config could be reused: a hard error. The write is
#      temp-file-then-mv, so a reader never sees a partial manifest.
# `make -n` only prints the recipe; parsing the Makefile performs no I/O.
_cfg_guard:
	@test -n "$(CC_TARGET_SIG)" && test "$(CC_TARGET_SIG)" != unavailable || { \
	  echo "FATAL: cannot determine compiler target; CC preprocessing with the configured flags must succeed." >&2; \
	  exit 1; }
	@test -n "$(CXX_TARGET_SIG)" && test "$(CXX_TARGET_SIG)" != unavailable && \
	 test "$(CC_ABI_SIG)" != unavailable && test "$(CXX_ABI_SIG)" != unavailable || { \
	  echo "FATAL: cannot determine C/C++ target ABI; preprocessing with the configured flags must succeed." >&2; \
	  exit 1; }
	@if [ "$(TOOLCHAIN_CHECK)" = "1" ]; then \
	  cc_m="$$($(CC) -dumpmachine 2>/dev/null)"; cxx_m="$$($(CXX) -dumpmachine 2>/dev/null)"; \
	  if [ -z "$$cc_m" ] || [ "$$cc_m" != "$$cxx_m" ] || [ "$(CC_ABI_SIG)" != "$(CXX_ABI_SIG)" ]; then \
	    echo "FATAL: CC/CXX compile for different targets or ABIs (CC='$(CC)' -> '$$cc_m', CXX='$(CXX)' -> '$$cxx_m')." >&2; \
	    echo "  The archive contains one C++ TU, so a partial cross-toolchain (only CC= passed)" >&2; \
	    echo "  would mix host C++ objects into a cross build. Pass the FULL toolchain, e.g.:" >&2; \
	    echo "    make CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ AR=aarch64-linux-gnu-ar RANLIB=aarch64-linux-gnu-ranlib" >&2; \
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

# Fresh-archive discipline: the archive is built from scratch into a temp file and renamed into place
# (atomic), never `ar rc` onto an existing archive, which would keep a removed source's stale .o
# forever (SRCS in CFG_SIG_PAYLOAD covers the other half). It is one chained shell line because the
# temp name embeds the shell PID ($$$$): racing same-config processes each write their own temp and
# the final mv is last-writer-wins.
$(LIB): $(BE_OBJS) | _cfg_guard
	t="$@.tmp.$$$$" && rm -f "$$t" && $(AR) rc "$$t" $(BE_OBJS) && $(RANLIB) "$$t" && mv -f "$$t" $@
	@echo "common [$(BACKEND)] -> $@"

# -MD -MP (not -MMD) emits $(OBJ_DIR)/<name>.d per object listing every header the TU
# pulled in, including the vendored NE10 headers reached via -isystem that -MMD omits.
# They are included at the bottom of this file.
$(OBJ_DIR)/%.o: %.c | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<
$(OBJ_DIR)/%.o: %.cpp | _cfg_guard
	$(CXX) $(CXXFLAGS) -MD -MP -c -o $@ $<

selftest: $(LIB) test_stft test_stft_guard test_heap_alignment test_audio_utils | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $(OBJ_DIR)/roundtrip.o test/roundtrip.c
	$(LINK) -o $(BIN_DIR)/roundtrip $(OBJ_DIR)/roundtrip.o $(LIB) $(LDFLAGS)
	@echo "--- common selftest [$(BACKEND)] ---"
	@$(BIN_DIR)/roundtrip
	# simd_kernels.h selftest: header-only, needs no FFT/archive objects.
	# -ffp-contract=off is NOT optional here -- verified empirically that
	# without it, this compiler auto-fuses the "plain mul/add, no FMA"
	# scalar reference loops (e.g. sk_ema_f32_scalar) into `fmla` at -O2,
	# which would then mismatch the deliberately-unfused NEON intrinics
	# path (see utility/simd_kernels.h's FMA-discipline doc comment).
	# Keep strict aliasing explicit: simd_kernels.h is consumed by external
	# board builds which must not depend on this Makefile's per-TU exceptions.
	$(CC) $(CFLAGS) -ffp-contract=off -fstrict-aliasing -MD -MP -c -o $(OBJ_DIR)/simd_selftest.o test/simd_selftest.c
	$(CC) -o $(BIN_DIR)/simd_selftest $(OBJ_DIR)/simd_selftest.o $(LDFLAGS)
	@echo "--- common SIMD kernel selftest [$(BACKEND)] ---"
	@$(BIN_DIR)/simd_selftest
	# Second pass with NR's production exp1_approx branch order
	# (USE_OPTIMIZED_E1): fast_math.h's two orders must stay bit-identical,
	# NaN included, so the cross-check against the kernel runs in both modes.
	$(CC) $(CFLAGS) -DUSE_OPTIMIZED_E1 -ffp-contract=off -fstrict-aliasing -c -o $(OBJ_DIR)/simd_selftest_e1.o test/simd_selftest.c
	$(CC) -o $(BIN_DIR)/simd_selftest_e1 $(OBJ_DIR)/simd_selftest_e1.o $(LDFLAGS)
	@echo "--- common SIMD kernel selftest [$(BACKEND)] USE_OPTIMIZED_E1 ---"
	@$(BIN_DIR)/simd_selftest_e1
	# simd_kernel_nn.h selftest: the NN pre/post kernels, kept out of
	# simd_kernels.h so the traditional libraries' kernel set never moves.
	# Header-only and the same bit-exact gate, same flags.
	$(CC) $(CFLAGS) -ffp-contract=off -fstrict-aliasing -MD -MP -c -o $(OBJ_DIR)/simd_nn_selftest.o test/simd_nn_selftest.c
	$(CC) -o $(BIN_DIR)/simd_nn_selftest $(OBJ_DIR)/simd_nn_selftest.o $(LDFLAGS)
	@echo "--- common NN SIMD kernel selftest [$(BACKEND)] ---"
	@$(BIN_DIR)/simd_nn_selftest
	# simd_kernels.h AArch32 stand-ins selftest: the AArch32 stand-ins for the AArch64-only intrinsics against a C
	# reference (and, on AArch64, against the real intrinsics). Skips itself on targets without NEON.
	$(CC) $(CFLAGS) -ffp-contract=off -fstrict-aliasing -MD -MP -c -o $(OBJ_DIR)/simd_a32_shim_selftest.o test/simd_a32_shim_selftest.c
	$(CC) -o $(BIN_DIR)/simd_a32_shim_selftest $(OBJ_DIR)/simd_a32_shim_selftest.o $(LDFLAGS)
	@echo "--- common AArch32 NEON shim selftest [$(BACKEND)] ---"
	@$(BIN_DIR)/simd_a32_shim_selftest

# test_stft: audio_stft.h window, latency, API-contract and scaled-entry-point checks.
test_stft: $(LIB) | _cfg_guard
	$(CC) $(CFLAGS) -fstrict-aliasing -MD -MP -c -o $(OBJ_DIR)/test_stft.o test/test_stft.c
	$(LINK) -o $(BIN_DIR)/test_stft $(OBJ_DIR)/test_stft.o $(LIB) $(LDFLAGS)
	@echo "--- common STFT/ISTFT test [$(BACKEND)] ---"
	@$(BIN_DIR)/test_stft > $(BIN_DIR)/test_stft.log && tail -n 1 $(BIN_DIR)/test_stft.log

# test_stft_guard: AcStft and AcIstft in exact-size blocks that end against a guard
# page (POSIX only; the test prints SKIP elsewhere).
test_stft_guard: $(LIB) | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $(OBJ_DIR)/test_stft_guard.o test/test_stft_guard.c
	$(LINK) -o $(BIN_DIR)/test_stft_guard $(OBJ_DIR)/test_stft_guard.o $(LIB) $(LDFLAGS)
	@echo "--- common STFT/ISTFT guard-page test [$(BACKEND)] ---"
	@$(BIN_DIR)/test_stft_guard

test_audio_utils: $(LIB) | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $(OBJ_DIR)/test_audio_utilities.o test/test_audio_utilities.c
	$(LINK) -o $(BIN_DIR)/test_audio_utilities $(OBJ_DIR)/test_audio_utilities.o $(LIB) $(LDFLAGS)
	@echo "--- common resampler test [$(BACKEND)] ---"
	@$(BIN_DIR)/test_audio_utilities

test_heap_alignment: | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $(OBJ_DIR)/test_heap_alignment.o test/test_heap_alignment.c
	$(CC) -o $(BIN_DIR)/test_heap_alignment $(OBJ_DIR)/test_heap_alignment.o $(LDFLAGS)
	@$(BIN_DIR)/test_heap_alignment

# test_pool: pool-contract negative tests (alignment guards, HPF domain validation).
# Links against the same $(LIB) archive as selftest.
test_pool: $(LIB) | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $(OBJ_DIR)/test_pool_contract.o test/test_pool_contract.c
	$(LINK) -o $(BIN_DIR)/test_pool_contract $(OBJ_DIR)/test_pool_contract.o $(LIB) $(LDFLAGS)
	@echo "--- common pool-contract test [$(BACKEND)] ---"
	@$(BIN_DIR)/test_pool_contract

# test_wav: negative-corpus tests for the shared WAV parser. Header-only like the
# simd_kernels.h half of `selftest`: wav_io.h needs no FFT/archive objects, so this does
# not depend on $(LIB) and links with a plain $(CC). The corpus runs twice, plain and built with
# -fsanitize=undefined -fno-sanitize-recover=all: an unsaturated +-1.0f in the PCM16 quantizer would be an
# out-of-range float-to-integer conversion (UB per C99 6.3.1.4p1) that the value checks alone cannot see.
# The sanitizer flags are scoped to that one compile and link.
test_wav: | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $(OBJ_DIR)/test_wav_io.o test/test_wav_io.c
	$(CC) -o $(BIN_DIR)/test_wav_io $(OBJ_DIR)/test_wav_io.o -lm
	@echo "--- common WAV I/O negative-corpus test [$(BACKEND)] ---"
	@TMPDIR=$(abspath $(BIN_DIR)) $(BIN_DIR)/test_wav_io
	$(CC) $(CFLAGS) -fsanitize=undefined -fno-sanitize-recover=all -MD -MP -c -o $(OBJ_DIR)/test_wav_io_ubsan.o test/test_wav_io.c
	$(CC) -fsanitize=undefined -fno-sanitize-recover=all -o $(BIN_DIR)/test_wav_io_ubsan $(OBJ_DIR)/test_wav_io_ubsan.o -lm
	@echo "--- common WAV I/O corpus under UBSan [$(BACKEND)] ---"
	@TMPDIR=$(abspath $(BIN_DIR)) $(BIN_DIR)/test_wav_io_ubsan

# test-wav-ubsan: the UBSan run of the WAV corpus, which test_wav already performs.
test-wav-ubsan: test_wav

# test_zero_heap: allocator-hook acceptance test. It proves fft_init()..fft_destroy() makes no
# malloc/calloc/realloc/free call (NE10's twiddle config is carved from the caller's pool; see
# lib/ne10/VENDORED.md) and that fft_destroy() on a pool-owned handle is idempotent.
# The hook (test/zero_heap_hook.c) is packaged per platform:
#   - macOS: dyld __DATA,__interpose only works from a Mach-O image other than the main executable,
#     so the hook is its own .dylib (rpath'd to @loader_path) that the test links against.
#   - Linux: GNU ld --wrap=<symbol> works from a plain object linked into the test binary;
#     -Wl,--wrap=... is scoped to this one target.
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
	@echo "--- common allocator-hook zero-heap test [$(BACKEND)] ---"
	@$(BIN_DIR)/test_fft_zero_heap

# test_ne10_force_c: builds test/test_ne10_c_parity.c twice, once against the normal NE10 kernels
# (`_neon`) and once with SIMD=0 (fft_wrapper_ne10.c routes every FFT call through NE10's `_c` scalar
# kernels), then diffs the dumped outputs. Two full sub-makes rebuild the archive per variant: SIMD is
# part of CFG_SIG, so they land in different bin/ne10-<sig>/ directories and the forced-C rebuild
# cannot overwrite the normal archive. Each variant's bin dir is asked of its sub-make
# (`print-bin-dir`) at recipe time. Both variants pin SIMD explicitly: a command-line SIMD=0 would
# otherwise reach both sub-makes through MAKEFLAGS and collapse them into one bin dir.
test_ne10_force_c:
ifeq ($(NE10_TARGET_NEON),1)
	@$(MAKE) BACKEND=ne10 SIMD=1 _ne10_parity_bin
	@bd_neon="$$($(MAKE) -s --no-print-directory BACKEND=ne10 SIMD=1 print-bin-dir)"; \
	 $(MAKE) BACKEND=ne10 SIMD=0 _ne10_parity_bin; \
	 bd_c="$$($(MAKE) -s --no-print-directory BACKEND=ne10 SIMD=0 print-bin-dir)"; \
	 if [ "$$bd_neon" = "$$bd_c" ]; then \
	   echo "FATAL: test_ne10_force_c: NEON and forced-C variants resolved to the SAME bin dir ($$bd_neon) -- CFG_SIG did not separate them" >&2; \
	   exit 1; \
	 fi; \
	 echo "--- common NE10 NEON-vs-C parity [test_ne10_force_c] ---"; \
	 echo "  NEON     bin dir: $$bd_neon"; \
	 echo "  forced-C bin dir: $$bd_c"; \
	 "$$bd_neon/test_ne10_c_parity" "$$bd_neon/test_ne10_c_parity_neon.dump"; \
	 "$$bd_c/test_ne10_c_parity"    "$$bd_c/test_ne10_c_parity_c.dump"; \
	 $(CC) -O2 -Wall -Wextra -o "$$bd_c/ne10_parity_compare" test/ne10_parity_compare.c -lm; \
	 "$$bd_c/ne10_parity_compare" "$$bd_neon/test_ne10_c_parity_neon.dump" "$$bd_c/test_ne10_c_parity_c.dump"
else
	@echo "SKIP test_ne10_force_c: compiler target has no NEON; FFT dispatch is scalar C"
endif

# _ne10_parity_bin: internal target invoked by test_ne10_force_c. BIN_DIR is
# CFG_SIG-keyed, so the plain and SIMD=0 invocations each get their own
# test_ne10_c_parity binary.
_ne10_parity_bin: $(LIB) | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $(OBJ_DIR)/test_ne10_c_parity.o test/test_ne10_c_parity.c
	$(LINK) -o $(BIN_DIR)/test_ne10_c_parity $(OBJ_DIR)/test_ne10_c_parity.o $(LIB) $(LDFLAGS)

# --- Test tiers ---------------------------------------------------------------
# `make test` is the default tier: the unit and contract tests plus the core scenarios of the
# build-isolation suite. `make test-full` adds what the default tier leaves out because another check
# already covers it or because it only guards `make publish`: the standalone `test_ne10_force_c`,
# the FP-contract disassembly audit (`audit-fp-contract`) and the publish scenarios of the isolation suite
# (S7p, S11, S13-S18), and `test-selftest-ubsan`, the UBSan build of the SIMD kernel selftest.
# The standalone `test_stft`, `test_stft_guard`, `test_heap_alignment` and `test_audio_utils` are not listed: `selftest` runs
# them. The isolation suite needs the AEC repo as a sibling checkout and runs the same for any SIMD
# setting, because it builds both itself. Steps run one after another, so the isolation suite never
# races another target for the real obj/ and bin/ directories it snapshots.
LEAN_TESTS = selftest test_pool test_wav test_zero_heap test-fast-sqrt-codegen

test:
	@$(MAKE) --no-print-directory $(LEAN_TESTS)
	@$(MAKE) --no-print-directory test-isolation ISOL_TIER=core

test-full:
	@$(MAKE) --no-print-directory $(LEAN_TESTS) test_ne10_force_c test-selftest-ubsan audit-fp-contract audit-fft-wrapper-pedantic
	@$(MAKE) --no-print-directory test-isolation ISOL_TIER=full

# test-selftest-ubsan: simd_selftest built with -fsanitize=undefined (test/run_selftest_ubsan.sh); test-full only.
# The script gets this build's CC, keyed BIN_DIR and SIMD / EXTRA_CFLAGS selection.
test-selftest-ubsan: | _cfg_guard
	@CC="$(CC)" BIN_DIR="$(BIN_DIR)" SELFTEST_CFLAGS="$(EXTRA_CFLAGS) $(SIMD_CFLAGS)" sh test/run_selftest_ubsan.sh

# audit-fp-contract: the disassembly gate for the pinned -ffp-contract=off policy (scripts/audit_fp_contract.sh; needs NR as
# a sibling checkout). It is part of test-full only: it builds common and NR keyed objects of its own.
audit-fp-contract:
	@scripts/audit_fp_contract.sh $(BACKEND)

# audit-fft-wrapper-pedantic: fft_wrapper_ne10.c compiled with its normal flags plus -Wpedantic -Werror. Its layout typedefs must not use syntax
# newer than the library standard (a struct defined inside offsetof() is a C23 extension); the library build itself does not pass -Wpedantic.
audit-fft-wrapper-pedantic:
	@$(CC) $(CFLAGS) -Wpedantic -Werror -fsyntax-only src/fft_wrapper_ne10.c && echo "PASS audit-fft-wrapper-pedantic"

# The suite starts its own makes with explicit SIMD settings, so the caller's MAKEFLAGS must not reach
# them: a command-line SIMD=0 would otherwise override the SIMD=1 configuration it compares against.
ISOL_TIER ?= core
test-isolation:
	@env -u MAKEFLAGS -u MFLAGS -u MAKELEVEL ISOL_TIER=$(ISOL_TIER) scripts/test_build_isolation.sh

# --- Query targets -----------------------------------------------------------
# Consumers (AEC/NR Makefiles, docs_smoke.sh, scripts) resolve this build's output paths
# at recipe time instead of hardcoding them, e.g.
# `ac="$$(make -s -C ../../common print-lib-path)"`. $(abspath) is deliberate: a
# caller in a directory of unknown depth needs an absolute path to -I/-L/link against,
# and the rule against absolute paths is about committed files, not runtime stdout. The
# only output of these targets is the recipe-time @echo line ($(info) above is gated off).
print-bin-dir:
	@echo $(abspath $(BIN_DIR))
print-obj-dir:
	@echo $(abspath $(OBJ_DIR))
print-lib-path:
	@echo $(abspath $(LIB))
print-fp-policy:
	@echo $(FP_POLICY)

# --- publish: immutable, content-addressed release under $(DIST_ROOT) ---------
#
#   $(DIST_ROOT)/<backend>/<cfg_sig>-<content12>/
#       <artifacts...>     immutable after the first publish
#       MANIFEST.txt       immutable and deterministic (config, tool identities and per-file SHA-256
#                          only; no commit or date): the same release id always has the same bytes
#       ATTEST/            append-only provenance: one new attest-<utc>-<commit>[-dirty]-<seq>.txt per
#                          publish event (idempotent republishes included), recording event_id,
#                          git_commit, git_dirty, date_utc and the allow-publish flags. Installed with
#                          the helper's --excl-install mode (temp + link(2), an atomic no-clobber
#                          create); on a name collision the next <seq> is tried. Attestations are
#                          unsigned: traceability, not tamper-proofing (MANIFEST byte-verification is
#                          the integrity check).
#   $(DIST_ROOT)/<backend>/current -> <rel_id>   (atomic swap, see below)
#
# Republishing an existing release id byte-compares every staged file against the stored release (a
# difference is FATAL: content-address collision or tampering), then appends an attestation and
# repoints `current`. Nothing in an existing release directory changes except ATTEST/ growing.
#
# Dirty/untracked policy: publish is FATAL by default when the checkout (a) has uncommitted tracked
# changes, (b) contains an untracked non-ignored file, or (c) has no git identity. (c) is
# unconditional; ALLOW_DIRTY_PUBLISH and ALLOW_UNTRACKED_PUBLISH admit (a) and (b) separately, because
# the tracked-diff hash says nothing about untracked bytes.
#
# `current` is swapped with tools/atomic_symlink_swap.c, built with $(HOSTCC) (never the possibly
# cross $(CC)): a symlink to a PID-suffixed temp plus rename(2), atomic on macOS and GNU/Linux, with no
# window without a `current` (`mv -fT` does not exist on BSD/macOS).
#
# Locking: `publish` takes the per-backend mkdir lock first and only then builds and stages via
# _publish_impl, so concurrent same-config publishes cannot race each other's writes; the loser fails
# fast having written nothing. _publish_impl is internal: invoking it directly skips the lock.
PUBLISH_ARTIFACTS = $(LIB)

# Backend-symbol audit of the archive named by the shell variable $archive, shared by the
# publish recipe and `check-archive-symbols`. Only DEFINED external symbols count: the
# FFT wrapper object holds undefined references to both ne10_fft_r2c_1d_float32_c and
# ..._neon, so a bare name match would still pass after the object that defines either one
# was dropped from the archive. The C entry point is required for every target; the NEON
# one whenever the compiler target has NEON, which is when its translation unit is built
# and the wrapper dispatches to it. Kept on one logical line so it expands inside a single
# recipe shell.
ARCHIVE_SYMBOL_AUDIT = has_def() { nm -g "$$archive" 2>/dev/null | awk -v s="$$1" '$$2 ~ /^[TtDdWw]$$/ && ($$3 == s || $$3 == "_" s) { f = 1 } END { exit !f }'; }; \
	has_def ne10_fft_r2c_1d_float32_c || { echo "FATAL: ne10 archive $$archive does not define ne10_fft_r2c_1d_float32_c" >&2; exit 1; }; \
	if [ "$(NE10_TARGET_NEON)" = "1" ]; then \
	  has_def ne10_fft_r2c_1d_float32_neon || { echo "FATAL: ne10 archive $$archive does not define ne10_fft_r2c_1d_float32_neon although the compiler target has NEON" >&2; exit 1; }; \
	fi

# Standalone entry point for the same audit:
# `make check-archive-symbols AUDIT_ARCHIVE=<archive>` (default: this configuration's archive).
AUDIT_ARCHIVE ?= $(LIB)
check-archive-symbols:
	@archive="$(AUDIT_ARCHIVE)"; $(ARCHIVE_SYMBOL_AUDIT); echo "check-archive-symbols: $$archive OK"
DIST_BACKEND_DIR  = $(DIST_ROOT)/$(BACKEND)
DIST_LOCK         = $(DIST_ROOT)/.lock.$(BACKEND)

.PHONY: publish _publish_impl check-archive-symbols
# Dry-run guard: this recipe mentions $(MAKE), so GNU make executes it under -n/-q/-t, which would
# create $(DIST_ROOT) and take the real publish lock. A word scan of MAKEFLAGS branches:
#   -t  prints a note and exits 0 without recursing (touch semantics would bump the mtimes of the
#       archive and its objects). Checked first: for `-nt`, recursing for -n would hand the child both
#       flags and apply touch semantics, so any t wins over n.
#   -q  exits 1 (question mode's "needs updating"; publish is phony) with no output.
#   -n  execs a recursion-only path; the child inherits -n and only prints _publish_impl's commands.
# All three write nothing. MAKEFLAGS parsing: bundled short flags may be the first word ("n", "sn") or
# separate dashed words; long options start with "--"; a literal "--" separates flags from variable
# overrides, which can contain n/q/t (hence the skips).
publish:
	+@has_n=0; has_q=0; has_t=0; \
	for w in $(MAKEFLAGS); do \
	  case "$$w" in --) break ;; --*|*=*) continue ;; -n|-q|-t) ;; -*) continue ;; esac; \
	  case "$$w" in *n*) has_n=1 ;; esac; \
	  case "$$w" in *q*) has_q=1 ;; esac; \
	  case "$$w" in *t*) has_t=1 ;; esac; \
	done; \
	if [ "$$has_t" = 1 ]; then echo "publish: -t (touch) is a no-op here -- publish is an action, not a build product; nothing was touched"; exit 0; fi; \
	if [ "$$has_q" = 1 ]; then exit 1; fi; \
	if [ "$$has_n" = 1 ]; then exec $(MAKE) --no-print-directory _publish_impl; fi; \
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
	hash_untracked() { \
	  LC_ALL=C sort | while IFS= read -r p; do \
	    fp="$$1$$p"; \
	    ph="$$(printf '%s' "$$p" | shasum -a 256 | cut -d' ' -f1)"; \
	    [ "$${#ph}" -eq 64 ] || { printf 'X %s\n' "$$p"; continue; }; \
	    if [ -L "$$fp" ]; then \
	      tgt="$$(readlink "$$fp")" || { printf 'X %s\n' "$$p"; continue; }; \
	      th="$$(printf '%s' "$$tgt" | shasum -a 256 | cut -d' ' -f1)"; \
	      [ "$${#th}" -eq 64 ] || { printf 'X %s\n' "$$p"; continue; }; \
	      printf 'L %s %s\n' "$$ph" "$$th"; \
	    elif [ -f "$$fp" ]; then \
	      m="$$(stat -f '%Lp' "$$fp" 2>/dev/null || stat -c '%a' "$$fp" 2>/dev/null)"; \
	      ch="$$(shasum -a 256 "$$fp" 2>/dev/null | cut -d' ' -f1)"; \
	      { [ -n "$$m" ] && [ "$${#ch}" -eq 64 ]; } || { printf 'X %s\n' "$$p"; continue; }; \
	      printf 'F %s %s %s\n' "$$m" "$$ph" "$$ch"; \
	    else printf 'X %s\n' "$$p"; fi; \
	  done; \
	}; \
	commit="$$(git rev-parse HEAD 2>/dev/null || echo unknown)"; \
	if [ "$$commit" = "unknown" ]; then \
	  echo "FATAL: publish refused -- common has no git identity here (not a git checkout)." >&2; \
	  echo "  Without an identity the attestation cannot name the source state OR enumerate untracked" >&2; \
	  echo "  content, so NO escape hatch admits this (ALLOW_DIRTY_PUBLISH used to). Publish" >&2; \
	  echo "  from a real checkout or a scratch clone with a disposable commit instead." >&2; \
	  exit 1; \
	fi; \
	dirty=0; [ -n "$$(git status --porcelain -uno 2>/dev/null)" ] && dirty=1; \
	unt_list="$$(git ls-files --others --exclude-standard 2>/dev/null)"; \
	untracked=0; [ -n "$$unt_list" ] && untracked=1; \
	adp=0; [ "$(ALLOW_DIRTY_PUBLISH)" = "1" ] && adp=1; \
	aup=0; [ "$(ALLOW_UNTRACKED_PUBLISH)" = "1" ] && aup=1; \
	if [ "$$adp" != "1" ] && [ "$$dirty" = "1" ]; then \
	  echo "FATAL: publish refused -- common working tree is dirty (uncommitted TRACKED changes)." >&2; \
	  echo "  Commit (or clone a clean checkout) for a board deliverable, or pass ALLOW_DIRTY_PUBLISH=1" >&2; \
	  echo "  for a dev publish -- the escape hatch is recorded in the attestation." >&2; \
	  exit 1; \
	fi; \
	if [ "$$aup" != "1" ] && [ "$$untracked" = "1" ]; then \
	  echo "FATAL: publish refused -- common contains UNTRACKED (non-ignored) files (they can change the artifact without a trace in the tracked diff):" >&2; \
	  printf '%s\n' "$$unt_list" | head -5 | sed 's/^/    /' >&2; \
	  echo "  Track/remove them for a board deliverable, or pass ALLOW_UNTRACKED_PUBLISH=1 -- the escape" >&2; \
	  echo "  hatch is recorded in the attestation with untracked_tree_sha256 over their exact contents." >&2; \
	  exit 1; \
	fi; \
	ddiff=""; [ "$$dirty" = "1" ] && ddiff="$$(git diff --binary HEAD 2>/dev/null | shasum -a 256 | cut -d' ' -f1)"; \
	uhash=""; \
	if [ "$$untracked" = "1" ]; then \
	  printf '%s\n' "$$unt_list" | hash_untracked "./" > "$$work/untracked.records"; \
	  if grep -q '^X ' "$$work/untracked.records"; then \
	    echo "FATAL: publish refused -- an untracked path cannot be hashed deterministically (embedded git checkout / special file / I/O failure) and so cannot enter the provenance record:" >&2; \
	    grep '^X ' "$$work/untracked.records" | sed 's/^X /    /' >&2; \
	    exit 1; \
	  fi; \
	  uhash="$$(shasum -a 256 < "$$work/untracked.records" | cut -d' ' -f1)"; \
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
	{ echo "repo=common"; \
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
	archive="$$tmp/libaudio_common.a"; \
	$(ARCHIVE_SYMBOL_AUDIT); \
	if [ -d "$$rel_dir" ]; then \
	  for f in "$$tmp"/*; do \
	    bn="$$(basename "$$f")"; \
	    cmp -s "$$f" "$$rel_dir/$$bn" || { echo "FATAL: existing release $$rel_id differs from staged content for $$bn (content-address collision, or the stored release/MANIFEST was modified on disk)" >&2; exit 1; }; \
	  done; \
	  rm -rf "$$tmp"; \
	  echo "common: $$rel_id already published (byte-verified, incl. MANIFEST) -- appending attestation + repointing current"; \
	else \
	  mkdir "$$tmp/ATTEST"; \
	  mv "$$tmp" "$$rel_dir"; \
	fi; \
	mkdir -p "$$rel_dir/ATTEST"; \
	n=1; \
	while :; do \
	  attest_name="attest-$$stamp-$$commit$$suffix-$$(printf '%03d' "$$n").txt"; \
	  { echo "event_id=$${attest_name%.txt}"; \
	    echo "repo=common"; \
	    echo "release_id=$$rel_id"; \
	    echo "cfg_sig=$(CFG_SIG)"; \
	    echo "backend=$(BACKEND)"; \
	    echo "git_commit=$$commit"; echo "git_dirty=$$dirty"; \
	    [ -n "$$ddiff" ] && echo "dirty_diff_sha256=$$ddiff"; \
	    echo "git_untracked=$$untracked"; \
	    [ -n "$$uhash" ] && echo "untracked_tree_sha256=$$uhash"; \
	    echo "date_utc=$$dateutc"; \
	    echo "allow_dirty_publish=$$adp"; \
	    echo "allow_untracked_publish=$$aup"; \
	  } > "$$work/attest.txt"; \
	  rc=0; "$$work/swapln" --excl-install "$$work/attest.txt" "$$rel_dir/ATTEST/$$attest_name" || rc=$$?; \
	  [ "$$rc" -eq 0 ] && break; \
	  [ "$$rc" -eq 2 ] || { echo "FATAL: could not install attestation $$attest_name" >&2; exit 1; }; \
	  n=$$((n+1)); \
	  [ "$$n" -le 999 ] || { echo "FATAL: 999 attestation name collisions for $$stamp-$$commit (runaway?)" >&2; exit 1; }; \
	done; \
	"$$work/swapln" "$$rel_id" "$(DIST_BACKEND_DIR)/current"; \
	echo "common: published $(BACKEND)/$$rel_id -> $(DIST_BACKEND_DIR)/current (attested: $$attest_name)"

# clean removes the whole obj and bin roots, every CFG_SIG subdirectory, not just the
# config this invocation resolves to. With an OBJ_ROOT/BIN_ROOT override it scrubs that
# tree, never the real obj/ and bin/. dist/ is kept: published releases survive a clean.
clean:
	rm -rf $(OBJ_ROOT) $(BIN_ROOT)

# --- Header dependency tracking ----------------------------------------------
# Every compile line passes -MD -MP, which emits $(OBJ_DIR)/<name>.d per object listing
# every header the TU pulled in, including vendored NE10 headers reached via -isystem
# (-MD, not -MMD, so touching a vendored header rebuilds its object). -MP adds a phony rule
# per header so a deleted or renamed header does not fail the build against a stale .d.
# $(wildcard) is empty before the first build of this CFG_SIG.
-include $(wildcard $(OBJ_DIR)/*.d)
