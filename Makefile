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
$(info audio_common: FFT backend = $(BACKEND))

CXX     ?= c++
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

# Per-backend, per-config-signature object dir (CFG_SIG is computed further
# below, after every BACKEND-conditional CFLAGS/CXXFLAGS append) so a kiss
# build, an ne10 build, and any two builds differing only in EXTRA_CFLAGS/
# WERROR each land in their own directory automatically -- nothing is ever
# wiped to "switch" between them, and two such builds (e.g. one per backend)
# can run concurrently in this same tree without stomping each other's
# objects. BIN_DIR stays per-backend only: the final archive path doesn't
# encode EXTRA_CFLAGS/WERROR (it's a last-write-wins whole, same as before).
OBJ_DIR = obj/$(BACKEND)-$(CFG_SIG)
BIN_DIR = bin/$(BACKEND)

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
  #     (LINK=$(CXX)/-lc++ below) despite this trim.
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
  LDFLAGS += -lc++
else
  OWN_SRCS       = src/fft_wrapper.c $(COMMON_SRCS)
  VENDOR_SRCS    = lib/kiss_fft/kiss_fft.c
  VENDOR_CXXSRCS =
  # -isystem for the vendor KISS FFT include dir (see the CXXFLAGS comment
  # above for the NE10 side of this same F12 fix).
  CFLAGS  += -isystem lib/kiss_fft
endif

# --- Hash-keyed object directory (build hygiene) ----------------------------
# obj/$(BACKEND) alone isolates a kiss build from an ne10 build, but
# EXTRA_CFLAGS (and WERROR) were NOT keyed within a given backend -- a build
# with one EXTRA_CFLAGS value left its .o's sitting in obj/$(BACKEND), and a
# later build with a different EXTRA_CFLAGS silently reused/mixed them.
# CFG_SIG hashes the exact compiler invocation that produces every object in
# this build (CFLAGS is now fully assembled -- every BACKEND-conditional
# append above has already run), and OBJ_DIR (defined near the top of this
# file) embeds CFG_SIG directly in its path -- every distinct (BACKEND,
# CFLAGS, EXTRA_CFLAGS, WERROR) combination gets its own directory,
# automatically.
#
# This MUST be defined here -- after CFLAGS is final, but before the first
# rule below that mentions $(OBJ_DIR)/anything derived from it (the WERROR
# and -fno-strict-aliasing target-specific-variable lines just after this,
# and the pattern rules further down) -- because GNU Make expands a rule's
# target/prerequisite list (including a target-specific variable
# assignment's target) IMMEDIATELY when that line is parsed, not deferred
# like a recipe body. A forward reference there would silently resolve
# using CFG_SIG's value at THAT point in the file (empty, if this line
# hadn't run yet), landing objects in a hash-less directory instead of the
# real one (verified empirically with a minimal reproduction).
#
# There is no parse-time comparison and nothing is ever deleted to "detect"
# a config change (the previous scheme did a $(shell rm -rf ...)/$(shell
# mkdir ...)/$(shell echo ...) unconditionally while the Makefile was being
# read, which mutated the filesystem even under `make -n`, and raced two
# concurrent builds with different flags/backends against each other's
# shared obj/$(BACKEND) directory). Under this scheme `make -n` touches
# nothing on disk (CFG_SIG is a pure string hash) and `make BACKEND=kiss
# -j4 & make BACKEND=ne10 -j4 & wait` (or two differing-EXTRA_CFLAGS
# builds) can run concurrently in this same tree: each lands in its own
# obj/<backend>-<sig>/ directory and never touches the other's objects.
# OBJ_DIR's own creation happens lazily via the order-only prerequisite on
# the pattern rules below ($(OBJ_DIR)/%.o: %.c | $(OBJ_DIR)), not at
# Makefile-parse time.
#
# cksum (POSIX -- present on macOS/BSD/GNU/busybox, unlike shasum which is a
# GNU coreutils extension and may be absent from slim embedded build images)
# computes the hash. On stdin it prints "<checksum> <byte-count>" with no
# filename field (verified on macOS's BSD cksum), so `cut -d' ' -f1` isolates
# just the checksum.
CFG_SIG := $(shell printf '%s' "$(CC) $(CFLAGS) $(EXTRA_CFLAGS) BACKEND=$(BACKEND) WERROR=$(WERROR)" | cksum | cut -d' ' -f1)

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

.PHONY: all lib selftest test_pool test_wav test_wav_nr_style test-wav-ubsan test_zero_heap test_ne10_force_c _ne10_parity_bin clean
all: lib

lib: $(LIB)

$(LIB): $(BE_OBJS) | $(BIN_DIR)
	ar rcs $@ $(BE_OBJS)
	@echo "audio_common [$(BACKEND)] -> $@"

$(OBJ_DIR)/%.o: %.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<
$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

LINK = $(CC)
ifeq ($(BACKEND),ne10)
  LINK = $(CXX)   # NE10 archive contains a C++ TU -> link with the C++ driver
endif

selftest: $(LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) -c -o $(OBJ_DIR)/roundtrip.o test/roundtrip.c
	$(LINK) -o $(BIN_DIR)/roundtrip $(OBJ_DIR)/roundtrip.o $(LIB) $(LDFLAGS)
	@echo "--- audio_common selftest [$(BACKEND)] ---"
	@$(BIN_DIR)/roundtrip
	# simd_kernels.h selftest: header-only, needs no FFT/archive objects.
	# -ffp-contract=off is NOT optional here -- verified empirically that
	# without it, this compiler auto-fuses the "plain mul/add, no FMA"
	# scalar reference loops (e.g. sk_ema_f32_scalar) into `fmla` at -O2,
	# which would then mismatch the deliberately-unfused NEON intrinics
	# path (see include/simd_kernels.h's FMA-discipline doc comment).
	$(CC) $(CFLAGS) -ffp-contract=off -c -o $(OBJ_DIR)/simd_selftest.o test/simd_selftest.c
	$(CC) -o $(BIN_DIR)/simd_selftest $(OBJ_DIR)/simd_selftest.o -lm
	@echo "--- audio_common SIMD kernel selftest [$(BACKEND)] ---"
	@$(BIN_DIR)/simd_selftest

# test_pool: pool-contract negative tests (F07 alignment guards, F14 HPF
# domain validation). Same shape as selftest -- links against the same $(LIB)
# archive so it exercises whichever backend (kiss/ne10) was built.
test_pool: $(LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) -c -o $(OBJ_DIR)/test_pool_contract.o test/test_pool_contract.c
	$(LINK) -o $(BIN_DIR)/test_pool_contract $(OBJ_DIR)/test_pool_contract.o $(LIB) $(LDFLAGS)
	@echo "--- audio_common pool-contract test [$(BACKEND)] ---"
	@$(BIN_DIR)/test_pool_contract

# test_wav: negative-corpus tests for the shared, hardened WAV parser (F06
# remediation). Header-only like the simd_kernels.h half of `selftest` --
# wav_io.h needs no FFT/archive objects, so this doesn't depend on $(LIB)
# and links with a plain $(CC) (no NE10 C++ TU involved either way).
test_wav: | $(OBJ_DIR) $(BIN_DIR)
	$(CC) $(CFLAGS) -c -o $(OBJ_DIR)/test_wav_io.o test/test_wav_io.c
	$(CC) -o $(BIN_DIR)/test_wav_io $(OBJ_DIR)/test_wav_io.o -lm
	@echo "--- audio_common WAV I/O negative-corpus test [$(BACKEND)] ---"
	@$(BIN_DIR)/test_wav_io

# test_wav_nr_style (review R01): runs the SAME writer special-values
# corpus (test/test_wav_writer_common.h) as test_wav above, but compiled
# with WAV_IO_WRITER_STYLE forced to WAV_IO_WRITER_NR -- test_wav's own TU
# never sets that knob, so it only ever exercises the default
# WAV_IO_WRITER_AEC style (see wav_io.h's WAV_IO_WRITER_STYLE doc). Header-
# only, same shape as test_wav.
test_wav_nr_style: | $(OBJ_DIR) $(BIN_DIR)
	$(CC) $(CFLAGS) -DWAV_IO_WRITER_STYLE=WAV_IO_WRITER_NR -c -o $(OBJ_DIR)/test_wav_writer_nr_style.o test/test_wav_writer_nr_style.c
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
test-wav-ubsan: | $(OBJ_DIR) $(BIN_DIR)
	$(CC) $(CFLAGS) -fsanitize=undefined -fno-sanitize-recover=all -c -o $(OBJ_DIR)/test_wav_writer_ubsan.o test/test_wav_writer_ubsan.c
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

$(BIN_DIR)/libzero_heap_hook.dylib: test/zero_heap_hook.c test/zero_heap_hook.h | $(BIN_DIR)
	$(CC) $(CFLAGS) -dynamiclib -o $@ test/zero_heap_hook.c

$(OBJ_DIR)/zero_heap_hook.o: test/zero_heap_hook.c test/zero_heap_hook.h | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ test/zero_heap_hook.c

test_zero_heap: $(LIB) $(ZERO_HEAP_HOOK_DEPS) | $(BIN_DIR)
	# -fno-builtin is NOT optional here: without it, clang recognizes
	# malloc/calloc/realloc/free as builtin allocation functions and, when a
	# call's returned pointer never "escapes" (isn't dereferenced/printed,
	# only null-checked and freed), silently deletes the whole
	# alloc+free pair as dead code at -O2 -- verified empirically while
	# building this test (the deliberate calloc()/realloc() sanity calls in
	# test_hook_actually_counts() were compiled away entirely). That would
	# make this test pass for the wrong reason (no calls to observe, rather
	# than calls correctly observed at zero).
	$(CC) $(CFLAGS) -fno-builtin -c -o $(OBJ_DIR)/test_fft_zero_heap.o test/test_fft_zero_heap.c
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
# one invocation: EXTRA_CFLAGS is baked into $(OBJ_DIR)'s own path via
# CFG_SIG (see the hash-keyed object directory comment above), so the two
# sub-makes automatically land in two DIFFERENT obj/ne10-<sig>/ directories
# -- the second sub-make's objects are a real fresh compile, not a stale
# reuse of the first's, with nothing needing to be wiped in between.
#
# The renamed binaries and dump files all live under bin/ne10/, which
# neither sub-make ever removes (BIN_DIR is a last-write-wins whole, not
# CFG_SIG-keyed -- see the hash-keyed object directory comment above), so
# the first (NEON) build's artifacts survive the second (forced-C)
# sub-make's rebuild only because this recipe cp's them to a variant-
# specific name in between (bin/ne10/test_ne10_c_parity itself IS
# overwritten by the second sub-make).
test_ne10_force_c:
	@$(MAKE) BACKEND=ne10 _ne10_parity_bin
	@cp bin/ne10/test_ne10_c_parity bin/ne10/test_ne10_c_parity_neon
	@$(MAKE) BACKEND=ne10 EXTRA_CFLAGS=-DFFT_NE10_FORCE_C _ne10_parity_bin
	@cp bin/ne10/test_ne10_c_parity bin/ne10/test_ne10_c_parity_c
	@echo "--- audio_common NE10 NEON-vs-C parity [test_ne10_force_c] ---"
	@bin/ne10/test_ne10_c_parity_neon bin/ne10/test_ne10_c_parity_neon.dump
	@bin/ne10/test_ne10_c_parity_c    bin/ne10/test_ne10_c_parity_c.dump
	@$(CC) -O2 -Wall -Wextra -o bin/ne10/ne10_parity_compare test/ne10_parity_compare.c -lm
	@bin/ne10/ne10_parity_compare bin/ne10/test_ne10_c_parity_neon.dump bin/ne10/test_ne10_c_parity_c.dump

# _ne10_parity_bin: internal plumbing target invoked (via $(MAKE) BACKEND=ne10
# [EXTRA_CFLAGS=...] _ne10_parity_bin) by test_ne10_force_c above; same shape
# as test_pool/test_wav/etc. but not meant to be run directly since it always
# overwrites the SAME $(BIN_DIR)/test_ne10_c_parity path on each of the two
# sub-make calls (that's why test_ne10_force_c cp's it to a variant-specific
# name immediately after each sub-make returns).
_ne10_parity_bin: $(LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) -c -o $(OBJ_DIR)/test_ne10_c_parity.o test/test_ne10_c_parity.c
	$(LINK) -o $(BIN_DIR)/test_ne10_c_parity $(OBJ_DIR)/test_ne10_c_parity.o $(LIB) $(LDFLAGS)

$(OBJ_DIR) $(BIN_DIR):
	@mkdir -p $@

clean:
	# F12 build hygiene: nuke obj/ and bin/ wholesale (BOTH backends' output
	# dirs), not just $(OBJ_DIR)/$(BIN_DIR) for whatever BACKEND this
	# invocation happens to resolve to -- a bare `make clean` (no BACKEND=)
	# used to leave the other backend's stale build sitting untouched.
	rm -rf obj bin
