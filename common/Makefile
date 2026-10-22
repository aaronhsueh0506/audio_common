# common — shared audio DSP layer (NE10 FFT wrapper + fast_math + common DSP)
#
# One copy of the FFT layer for every consumer repo (AEC, NR, Audio_ALG). Consumers
# add `-I common/include` and link the archive this Makefile builds.
#
# The one FFT backend is NE10 (NE10_DSP_RFFT_SCALING on), for ARM32/ARM64.
# Compiler targets with NEON use its NEON FFT; SIMD=0 or targets without NEON
# use its scalar C FFT. Other shared SIMD kernels keep their own ISA guards.
#
# fft_wrapper.h has a heap path (fft_create/fft_destroy) and a static-memory path
# (fft_get_mem_size/fft_init), with a runtime pool_owned flag (no compile-time gate).

CC      ?= cc
CXX     ?= c++

# Archiver tools are explicit so a cross-compiling caller overriding AR/RANLIB
# (e.g. an arm-none-eabi- toolchain) lands in a different config signature, same
# as overriding CC/CXX.
AR      ?= ar
RANLIB  ?= ranlib

# CFG_SIG payload placeholders: CPPFLAGS/EXTRA_LDFLAGS/NO_STDIO/DEBUG are not
# consumed by this Makefile today, but the payload always includes them so its
# shape matches the other repos' Makefiles and a future use keys builds correctly.
# They default to explicit (possibly empty) values so the payload never depends on
# whether the variable is defined.
#
# CPPFLAGS uses `+=`, not `?=`: it is also in the command-line-override rejection
# below, and `?=` is a complete no-op for an environment value, so under `make -e`
# its origin would never flip to "environment override" and the rejection would
# miss it. `+=` is an unconditional assignment attempt; appending an empty string
# leaves the value unchanged.
CPPFLAGS      +=
EXTRA_LDFLAGS ?=
NO_STDIO      ?= 0
DEBUG         ?= 0

# TOOLCHAIN_CHECK=0 skips the CC/CXX -dumpmachine target-coherence guard in
# _cfg_guard below. It participates in CFG_SIG, so a guard-skipped build never
# aliases a guarded one.
TOOLCHAIN_CHECK ?= 1

# Placement and publish knobs. None is in CFG_SIG: they change where something is
# written or whether publish proceeds, never what is built.
#   DIST_ROOT         where `make publish` writes its release tree (default dist/).
#                     The isolation tests publish into a throwaway directory.
#   OBJ_ROOT/BIN_ROOT where the keyed obj/bin trees live (default obj/, bin/). The
#                     isolation tests use scratch roots for tamper/injection
#                     scenarios so the real obj/ and bin/ are never modified.
#   HOSTCC            compiles tools/atomic_symlink_swap.c at publish time. It must
#                     produce binaries that run on the build host, so never $(CC),
#                     which may be a cross compiler.
#   ALLOW_DIRTY_PUBLISH      publish refuses by default when tracked files have
#                     uncommitted changes, or when the checkout has no git identity
#                     (the attestation could not name the source state). =1 admits
#                     tracked changes and records the sha256 of
#                     `git diff --binary HEAD` in the attestation. An identity-less
#                     checkout is refused unconditionally.
#   ALLOW_UNTRACKED_PUBLISH  publish also refuses when any untracked, non-ignored
#                     file exists: an untracked source can change the artifact
#                     without a trace in the tracked diff. =1 admits them and records
#                     untracked_tree_sha256, a sha256 over sorted fixed-field records
#                     ("L <sha256(path)> <sha256(target)>" for symlinks,
#                     "F <mode> <sha256(path)> <sha256(content)>" for files). Every
#                     variable-length value is hashed so records concatenate
#                     unambiguously, and a stat/readlink/shasum failure becomes an
#                     "X" record that is fatal (fail closed). A path that is neither a
#                     regular file nor a symlink (embedded git checkout, fifo) is
#                     always refused by name.
#   ATTEST_STAMP      TEST-ONLY: overrides the UTC stamp in the attestation file name
#                     so the tests can force same-second collisions. Attestations are
#                     unsigned: traceability, not authenticity.
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

# One project-wide switch for every optional SIMD path.  SIMD=1 is the
# production default; SIMD=0 keeps the selected FFT backend/configuration but
# dispatches both our kernels and the NE10 wrapper through their scalar C
# implementations.  `SIMD` is the only public build-time switch.
#
# SIMD=0 and SIMD=1 are NOT bit-identical end to end, and are not contracted
# to be.  Individual kernels in utility/simd_kernels.h ARE byte-exact against
# their scalar twins (test/simd_selftest.c proves it per kernel), but the AEC
# matched filter deliberately diverges: its dot product uses four accumulators
# plus vfmaq_f32 (reordered summation AND fused rounding) and its NLMS update
# fuses where the scalar spells a separate multiply and add.  The
# contract is per-configuration: a change must be byte-exact against the same
# configuration before it, and across configurations only finite output, the
# same delay winner/state and a quality tolerance are guaranteed.
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
COMMON_SRCS = src/hpf.c src/audio_pre_gain.c src/audio_resampler.c src/audio_window.c src/audio_stft.c

# NE10 source footprint: fft_wrapper_ne10.c only calls the float32 R2C/C2R entry
# points (ne10_fft_alloc_r2c_float32, ne10_fft_r2c_mem_size_float32,
# ne10_fft_init_r2c_float32_ext, ne10_fft_destroy_r2c_float32,
# ne10_fft_r2c_1d_float32_neon, ne10_fft_c2r_1d_float32_neon) and never calls
# ne10_init(), so this is an explicit file list, not a wildcard: the smallest set
# that links cleanly without editing any NE10 source.
#   - NE10_rfft_float32.c / .neonintrinsic.c: the float32 R2C/C2R implementation
#     (scalar reference + NEON kernels), including the external-memory init patch
#     (lib/ne10/VENDORED.md).
#   - NE10_fft.c: the R2C path calls its ne10_factor() and twiddle generators, and
#     the wrapper's heap fft_destroy() calls its ne10_fft_destroy_r2c_float32().
#   - NE10_fft_float32.c, NE10_fft_int32.c, NE10_fft_generic_float32.c,
#     NE10_fft_generic_int32.cpp: the wrapper never calls their complex-to-complex
#     or non-power-of-2 paths, but NE10_fft.c defines ne10_fft_alloc_c2c_*_neon
#     unconditionally and those reference the *_c allocators and generic butterfly
#     helpers in these files. A static archive resolves a whole .o at a time, so
#     they must stay to satisfy the linker. NE10_fft_generic_int32.cpp is why the
#     archive still needs a C++ TU (LINK = $(CXX) below).
# Not compiled (nothing reachable from NE10_fft.o needs them): NE10_init*.c, the NEON
# c2c compute kernels, the NEON generic kernels, the int16 FFT, and the FIR/IIR
# families. They stay under lib/ne10/ for provenance.
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
# NE10_DSP_RFFT_SCALING makes c2r normalise by 1/N (round-trip verified); it is
# default-on in NE10_fft.h and defined explicitly so no build can silently drop it.
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
# FP_POLICY, FFT_WRAPPER_ALIAS_CFLAGS and FP_ALLOWED_CHARS_RE are bare, dependency-free
# `:=` literals defined here, ahead of the override-rejection foreach below. Under
# `make -e` GNU Make flips a variable's origin to "environment override" only when a
# makefile assignment to it is parsed, so a variable assigned after the foreach would
# still report plain "environment" there and an `env FP_POLICY=x make -e ...` override
# would sail through, silently dropping -ffp-contract=off. Only plain "command line"
# origin is position-independent. (FP_POLICY is appended to CFLAGS/CXXFLAGS further
# below, after every other append; CFG_SIG_PAYLOAD reads these values.)
#
# FP_POLICY: keep scalar/NEON operation ordering stable (no contraction) and let
# AArch64 lower guarded sqrtf calls to a single FSQRT without an errno-checking libm
# fallback.
FP_POLICY := -ffp-contract=off -fno-math-errno
# -fno-strict-aliasing, applied to fft_wrapper_ne10.o only (see below).
FFT_WRAPPER_ALIAS_CFLAGS := -fno-strict-aliasing
# Character allow-list for the flag variables (used by the FP-policy gate below).
FP_ALLOWED_CHARS_RE := ^[A-Za-z0-9_./=,+ -]*$$

# CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS and the policy variables are INTERNAL. GNU Make
# silently ignores this file's own `+=`/`:=` assignments to a variable set on the command
# line (`make CFLAGS=-O3`), which would strip the FP-contraction policy, the NE10 defines
# and include paths, and -lm from a still-buildable archive. The supported hooks are
# EXTRA_CFLAGS and EXTRA_LDFLAGS (folded in above, covered by CFG_SIG, policy appended
# after them). A plain environment value is not rejected: this file's own assignments
# fold it in (origin becomes "file") and it participates in CFG_SIG. Only the "command
# line" and "environment override" origins defeat the assignments. Every internal
# variable must be defined early enough (bare `:=` or unconditional `+=`, never `?=`)
# that its origin is final before this foreach runs.
$(foreach v,CFLAGS CXXFLAGS CPPFLAGS LDFLAGS FP_POLICY FFT_WRAPPER_ALIAS_CFLAGS FP_ALLOWED_CHARS_RE,\
  $(if $(findstring command,$(origin $(v)))$(findstring override,$(origin $(v))),\
    $(error $(v) cannot be overridden (origin: $(origin $(v))) -- it would silently drop this Makefile's own flag appends (FP policy, backend defines); use EXTRA_CFLAGS / EXTRA_LDFLAGS instead)))

# --- FP-contraction policy ---------------------------------------------------
# Every TU this Makefile compiles, ours and vendored NE10 alike, builds with
# -ffp-contract=off. FP_POLICY is appended last, after every other CFLAGS/CXXFLAGS
# append (base flags, EXTRA_CFLAGS, SIMD, NE10 defines) and before CFG_SIG_PAYLOAD is
# computed, so it participates in CFG_SIG and a caller's EXTRA_CFLAGS cannot re-enable
# contraction after it. Compiler-chosen fusion of a plain `a*b+c` is exactly the
# build-dependent numeric drift this flag removes. It does not touch explicit fusion
# requests in source (the fmaf() call in fft_power(), vfmaq_f32 in the NEON blocks);
# scripts/audit_fp_contract.sh verifies both classes at disassembly level.
#
# Conflict detection runs on the fully assembled CFLAGS/CXXFLAGS/CPPFLAGS (including
# EXTRA_CFLAGS and any plain-environment CXXFLAGS/CPPFLAGS, which reach the NE10 C++ TU
# without going through EXTRA_CFLAGS): -Ofast, -ffast-math and -ffp-contract=<any> are
# rejected, before FP_POLICY is appended and before CFG_SIG exists, so a rejected build
# never creates an obj/bin directory. $(filter) matches whole words, so a harmless
# `-DNOTE=-Ofastness` is not a false positive the way a $(findstring) substring match was.
FP_INPUT_FLAGS := $(CFLAGS) $(CXXFLAGS) $(CPPFLAGS)

# LDFLAGS is embedded in every link recipe just as CFLAGS is in every compile recipe,
# so the character-safety checks below run on the union.
SHELL_SAFE_INPUT_FLAGS := $(FP_INPUT_FLAGS) $(LDFLAGS)

# GNU Make has no notion of shell quoting: EXTRA_CFLAGS="'-Ofast'" is stored as the
# quoted text, never equals the word -Ofast, yet the shell strips the quotes and the
# compiler receives -Ofast. A deny-list of such constructs cannot keep up (globs, tilde,
# redirection and process substitution all got through), so the flags are validated
# against a character ALLOW-list instead: letters, digits, underscore, dot, slash,
# equals, comma, plus, minus, and space as the token separator. That covers every flag
# this Makefile and its tests use (-O2, -std=gnu99, -isystem lib/ne10/inc,
# -DNE10_DSP_RFFT_SCALING, -DNAME=value, -Wl,-rpath,dir, ...).
#
# The allow-list shells out to `grep -E` with the flags SINGLE-QUOTED in the command.
# Inside single quotes every character is literal except the single quote itself, so the
# single-quote check below is the one deny-list entry that stays: it must run first, as a
# pure $(findstring), to make that embedding safe.
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
# obj/$(BACKEND)-<sig> and bin/$(BACKEND)-<sig> isolate every distinct build
# configuration from every other one: the exact compiler invocation that produces every
# object and the exact archive that results. CFG_SIG_PAYLOAD is the one source string
# used both to compute CFG_SIG (via cksum) and, verbatim, as the collision-guard
# manifest content (_cfg_guard below), so the two cannot drift apart.
#
# Payload coverage: every variable that can affect what a build produces (CC CXX AR
# RANLIB LINK CFLAGS CXXFLAGS CPPFLAGS LDFLAGS EXTRA_CFLAGS EXTRA_LDFLAGS BACKEND SIMD
# WERROR NO_STDIO DEBUG TOOLCHAIN_CHECK), even ones this Makefile does not branch on, plus
# the SORTED source list: adding or removing a source lands in a fresh keyed directory,
# where a stale .o and a stale archive member (`ar rc` never deletes members) would
# otherwise survive under the same key. It must be defined after CFLAGS/CXXFLAGS are final
# and before the first rule that mentions $(OBJ_DIR)/$(BIN_DIR), because GNU Make expands
# a rule's target and prerequisite list when the line is parsed; a forward reference would
# resolve with an empty CFG_SIG and land objects in a hash-less directory.
#
# Deliberately not $(strip)'d: whitespace-only differences get different signatures. That
# is a harmless false miss (an extra directory); collapsing them risks a false hit, the
# one failure this mechanism exists to prevent.
#
# Nothing is compared or deleted at parse time: CFG_SIG is a pure string hash, so
# `make -n` touches nothing on disk and builds with different flags (for example SIMD=0
# and SIMD=1) can run concurrently in one tree, each in its own obj/<backend>-<sig>/ and
# bin/<backend>-<sig>/. Directories are created lazily by the _cfg_guard order-only
# prerequisite.
#
# cksum is POSIX (shasum is a GNU coreutils extension and may be absent from slim
# embedded build images); on stdin it prints "<checksum> <byte-count>", so `cut -d' ' -f1`
# isolates the checksum.
#
# FFT_WRAPPER_ALIAS_CFLAGS is in the payload because a target-specific
# `TARGET: CFLAGS += ...` line does not change the global $(CFLAGS) already captured
# here: with a bare literal on that line, editing the flag left CFG_SIG unchanged and a
# keyed directory built before the flag existed kept reusing an object compiled without
# it. Any future flag added to a target-specific CFLAGS/CXXFLAGS line must follow the same
# shape: name it, add the name to this payload, reference the name (not a literal) on the
# target-specific line. The name is also rejected as a command-line/`-e` override above,
# since a keyed directory alone would not stop a build that ships without the flag.
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
# layouts match (static-asserted), but the types are distinct, so reading through one
# after writing through the other is a type-based-alias violation. The flag removes that
# assumption for this TU only; every other object keeps strict aliasing.
$(OBJ_DIR)/fft_wrapper_ne10.o: CFLAGS += $(FFT_WRAPPER_ALIAS_CFLAGS)

LIB = $(BIN_DIR)/libaudio_common.a

.PHONY: all lib selftest test_stft test_stft_guard test_heap_alignment test-fast-sqrt-codegen test_audio_utils test_pool test_wav test_wav_nr_style test-wav-ubsan test_zero_heap test_ne10_force_c _ne10_parity_bin clean publish print-bin-dir print-obj-dir print-lib-path print-fp-policy _cfg_guard
all: lib

lib: $(LIB)

test-fast-sqrt-codegen:
	@FP_POLICY='$(FP_POLICY)' scripts/audit_fast_sqrt_codegen.sh

# --- CFG_SIG collision guard + directory creation ----------------------------
# _cfg_guard is the single order-only prerequisite of every compile/link/archive rule.
# Being .PHONY it runs on every invocation that needs it, and once even under `make -j`
# (a target is built at most once and scheduled before its dependents). It does two things:
#   1. mkdir -p $(OBJ_DIR) $(BIN_DIR); idempotent, safe if two make processes for the
#      same config race.
#   2. Writes (first time) or verifies (afterwards) a one-line config.manifest in each of
#      the two directories holding the exact CFG_SIG_PAYLOAD. A mismatch means two
#      different configurations hashed to the same CFG_SIG (a cksum collision) or the
#      payload misses an input that affects the build; either way objects from an
#      unrelated config could be reused, so it is a hard error. The write is
#      temp-file-then-mv (atomic rename), so a concurrent reader never sees a partial
#      manifest.
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

# Fresh-archive discipline: the archive is built from scratch into a temp file and
# renamed into place, never `ar rc` onto an existing archive, which only adds or
# replaces members and would keep a removed source's stale .o forever. The rename is
# atomic, so a concurrent reader never sees a half-written archive. The other half of the
# source-removal fix is SRCS in CFG_SIG_PAYLOAD. It is one chained shell line because the
# temp name embeds the shell PID ($$$$): two same-config processes racing this rule each
# write their own temp and the final mv is an atomic last-writer-wins, and each recipe
# line runs in a fresh shell with a fresh PID.
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

selftest: $(LIB) test_stft test_stft_guard test_heap_alignment | _cfg_guard
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
	$(CC) $(CFLAGS) -MD -MP -c -o $(OBJ_DIR)/test_audio_utilities.o test/test_audio_utilities.c
	$(LINK) -o $(BIN_DIR)/test_audio_utilities $(OBJ_DIR)/test_audio_utilities.o $(LIB) $(LDFLAGS)
	@echo "--- common pre-gain/resampler test [$(BACKEND)] ---"
	@$(BIN_DIR)/test_audio_utilities

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
	@echo "--- common pre-gain/resampler test [$(BACKEND)] ---"
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
# not depend on $(LIB) and links with a plain $(CC).
test_wav: | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $(OBJ_DIR)/test_wav_io.o test/test_wav_io.c
	$(CC) -o $(BIN_DIR)/test_wav_io $(OBJ_DIR)/test_wav_io.o -lm
	@echo "--- common WAV I/O negative-corpus test [$(BACKEND)] ---"
	@$(BIN_DIR)/test_wav_io

# test_wav_nr_style: runs the same writer special-values corpus
# (test/test_wav_writer_common.h) as test_wav, compiled with WAV_IO_WRITER_STYLE forced
# to WAV_IO_WRITER_NR; test_wav itself only exercises the default WAV_IO_WRITER_AEC
# style (see wav_io.h). Header-only, same shape as test_wav.
test_wav_nr_style: | _cfg_guard
	$(CC) $(CFLAGS) -DWAV_IO_WRITER_STYLE=WAV_IO_WRITER_NR -MD -MP -c -o $(OBJ_DIR)/test_wav_writer_nr_style.o test/test_wav_writer_nr_style.c
	$(CC) -o $(BIN_DIR)/test_wav_writer_nr_style $(OBJ_DIR)/test_wav_writer_nr_style.o -lm
	@echo "--- common WAV writer NR-style special-values test [$(BACKEND)] ---"
	@$(BIN_DIR)/test_wav_writer_nr_style

# test-wav-ubsan: UBSan probe for wav_write_float's PCM16 quantizer. It writes
# {1.0f,-1.0f,NAN} and must run clean under -fsanitize=undefined
# -fno-sanitize-recover=all (an unsaturated +1.0f would be an out-of-range
# float-to-integer conversion, UB per C99 6.3.1.4p1). The sanitizer flags are scoped to
# this target's compile and link, never the main archive or another test.
test-wav-ubsan: | _cfg_guard
	$(CC) $(CFLAGS) -fsanitize=undefined -fno-sanitize-recover=all -MD -MP -c -o $(OBJ_DIR)/test_wav_writer_ubsan.o test/test_wav_writer_ubsan.c
	$(CC) -fsanitize=undefined -fno-sanitize-recover=all -o $(BIN_DIR)/test_wav_writer_ubsan $(OBJ_DIR)/test_wav_writer_ubsan.o -lm
	@echo "--- common WAV writer UBSan probe [$(BACKEND)] ---"
	@$(BIN_DIR)/test_wav_writer_ubsan

# test_zero_heap: allocator-hook acceptance test. It proves fft_init()..fft_destroy()
# makes zero malloc/calloc/realloc/free calls (NE10's twiddle config is carved from the
# caller's pool; see lib/ne10/VENDORED.md patch P0001) and that fft_destroy() on a
# pool-owned handle is idempotent.
#
# The allocator hook (test/zero_heap_hook.c) needs different packaging per platform:
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
	@echo "--- common allocator-hook zero-heap test [$(BACKEND)] ---"
	@$(BIN_DIR)/test_fft_zero_heap

# test_ne10_force_c: builds test/test_ne10_c_parity.c twice, once against the normal NE10
# kernels (`_neon` suffix) and once with SIMD=0 (fft_wrapper_ne10.c routes every FFT call
# through NE10's `_c` scalar kernels), then diffs the two runs' dumped outputs.
#
# Two full sub-makes rebuild the archive per variant: SIMD is part of CFG_SIG, so they
# land in two different bin/ne10-<sig>/ directories and the forced-C rebuild can never
# overwrite the normal archive. Each variant's bin dir is asked of the sub-make that built
# it (`print-bin-dir`) at recipe time rather than reconstructed here.
test_ne10_force_c:
ifeq ($(NE10_TARGET_NEON),1)
	@$(MAKE) BACKEND=ne10 _ne10_parity_bin
	@bd_neon="$$($(MAKE) -s --no-print-directory BACKEND=ne10 print-bin-dir)"; \
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
#       MANIFEST.txt       immutable and deterministic (config, tool identities and
#                          per-file SHA-256 only; no commit or date), so the same release
#                          id always has the same bytes and a republish can verify it
#       ATTEST/            append-only provenance: exactly one new
#                          attest-<utc>-<commit>[-dirty]-<seq>.txt per publish event
#                          (idempotent republishes included), recording event_id,
#                          git_commit (full OID), git_dirty, date_utc and the
#                          allow-publish flags. It is installed with the helper's
#                          --excl-install mode (write-temp + link(2), an atomic no-clobber
#                          create); on a name collision (same second, same commit) the
#                          next <seq> is tried, so an existing attestation is never
#                          overwritten. Attestations are unsigned: traceability, not
#                          tamper-proofing (MANIFEST byte-verification is the integrity
#                          check, ATTEST the provenance log).
#   $(DIST_ROOT)/<backend>/current -> <rel_id>   (atomic swap, see below)
#
# Republishing an existing release id byte-compares every staged file, artifacts and
# MANIFEST.txt, against the stored release (a difference is FATAL: content-address
# collision or on-disk tampering), then appends an attestation and repoints `current`.
# Nothing in an existing release directory is modified or deleted except ATTEST/ growing.
#
# Dirty/untracked policy: publish is FATAL by default when the checkout (a) has
# uncommitted tracked changes, (b) contains any untracked non-ignored file, or (c) has no
# git identity at all. (c) is unconditional. The escape hatches for (a) and (b) are
# ALLOW_DIRTY_PUBLISH and ALLOW_UNTRACKED_PUBLISH (encoding described with the knobs at
# the top); they are separate because the tracked-diff hash says nothing about untracked
# bytes, so two different untracked source states must not share a provenance record.
#
# `current` is swapped with tools/atomic_symlink_swap.c, built at publish time with
# $(HOSTCC) (never the possibly-cross $(CC)): a symlink to a PID-suffixed temp plus
# rename(2), atomic on macOS and GNU/Linux, with no window without a `current` and no
# destructive fallback (`mv -fT` does not exist on BSD/macOS).
#
# Locking: `publish` is a driver that takes the per-backend mkdir lock first and only then
# builds and stages via _publish_impl, so two concurrent same-config publishes cannot
# race each other's object/archive writes; the loser fails fast having written nothing.
# _publish_impl is internal: invoking it directly skips the lock.
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
# Dry-run guard: this recipe mentions $(MAKE), so GNU make executes it for real under
# -n/-q/-t, which would create $(DIST_ROOT) and take the real publish lock. The word scan
# below detects those flags in MAKEFLAGS and branches:
#   -t  prints a note and exits 0 without recursing: touch semantics would bump the mtimes
#       of the archive's objects and the archive itself. publish is an action, not a build
#       product, so touch mode is an explicit no-op. Checked first: in a combined `-nt`
#       invocation, recursing for -n would hand the child both flags and make would then
#       really apply touch semantics, so any t wins over n.
#   -q  exits 1 (question mode's "needs updating" answer; publish is phony) with no output.
#   -n  execs a recursion-only path; the child inherits -n and only prints
#       _publish_impl's commands.
# All three write nothing: no DIST_ROOT, no lock, no stage/MANIFEST/ATTEST, no mtime
# changes. MAKEFLAGS parsing: bundled short flags may be the first word ("n", "sn") or
# separate dashed words; long options start with "--"; a literal "--" separates flags from
# command-line variable overrides, which can themselves contain n/q/t (hence the skips).
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
