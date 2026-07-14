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
#   - explicit `make BACKEND=kiss|ne10` always wins (consumers pin it per policy:
#     main branch -> kiss, static-memory branch -> ne10).
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

# Per-backend output dirs so a kiss build and an ne10 build never stomp each
# other's objects/archive (consumers may build both in one tree).
OBJ_DIR = obj/$(BACKEND)
BIN_DIR = bin/$(BACKEND)

# Backend-independent shared DSP sources (always in the archive).
COMMON_SRCS = src/hpf.c src/hpf_f64.c

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

.PHONY: all lib selftest clean
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

$(OBJ_DIR) $(BIN_DIR):
	@mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
