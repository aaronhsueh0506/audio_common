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
CXXFLAGS+= -O2 -Ilib/ne10/inc -Ilib/ne10/common -Ilib/ne10/modules
LDFLAGS += -lm

# EXTRA_CFLAGS hook: lets callers inject extra defines/flags (e.g.
# `make selftest EXTRA_CFLAGS=-DSIMD_KERNELS_FORCE_SCALAR`) into every
# compiled TU without editing this file.
CFLAGS  += $(EXTRA_CFLAGS)
CXXFLAGS+= $(EXTRA_CFLAGS)

# Per-backend output dirs so a kiss build and an ne10 build never stomp each
# other's objects/archive (consumers may build both in one tree).
OBJ_DIR = obj/$(BACKEND)
BIN_DIR = bin/$(BACKEND)

# Backend-independent shared DSP sources (always in the archive).
COMMON_SRCS = src/hpf.c

ifeq ($(BACKEND),ne10)
  # Whole, UNMODIFIED NE10 DSP module (C reference + NEON-intrinsic variants, all
  # fft/fir/iir + init). No NE10 source edited, no stubs — ne10_init is scoped to
  # DSP only via NE10's own official NE10_ENABLE_DSP flag (its other sub-inits are
  # #if-guarded), and on macOS/arm64 it just sets NEON-available without reading
  # /proc/cpuinfo. Both the CPU (_c) and NEON (_neon) kernels compile in.
  NE10_C_SRCS  = $(wildcard lib/ne10/modules/*.c lib/ne10/modules/dsp/*.c)
  NE10_CXXSRCS = $(wildcard lib/ne10/modules/dsp/*.cpp)
  BE_SRCS  = src/fft_wrapper_ne10.c $(COMMON_SRCS) $(NE10_C_SRCS)
  BE_CXXSRCS = $(NE10_CXXSRCS)
  # NE10_DSP_RFFT_SCALING makes c2r normalise by 1/N (verified round-trip); it is
  # default-on in NE10_fft.h — defined explicitly so no build can silently drop it.
  NE10_DEFS = -DNE10_ENABLE_DSP -DNE10_DSP_RFFT_SCALING
  CFLAGS  += -Ilib/ne10/inc -Ilib/ne10/common -Ilib/ne10/modules $(NE10_DEFS)
  CXXFLAGS+= $(NE10_DEFS)
  LDFLAGS += -lc++
else
  BE_SRCS  = src/fft_wrapper.c $(COMMON_SRCS) \
             lib/kiss_fft/kiss_fft.c
  BE_CXXSRCS =
  CFLAGS  += -Ilib/kiss_fft
endif

BE_OBJS  = $(patsubst %.c,$(OBJ_DIR)/%.o,$(notdir $(BE_SRCS))) \
           $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(BE_CXXSRCS)))
VPATH    = src lib/kiss_fft lib/ne10/modules lib/ne10/modules/dsp

LIB = $(BIN_DIR)/libaudio_common.a

.PHONY: all lib selftest test_pool test_wav test_zero_heap clean
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

$(OBJ_DIR) $(BIN_DIR):
	@mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
