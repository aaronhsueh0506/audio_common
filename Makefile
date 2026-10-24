# Reusable spatial DSP libraries used by Audio_ALG pipelines.
#
#   make SIMD=0|1
#
# BACKEND (ne10, the only FFT backend) follows common's build identity even though these
# frequency-domain algorithms run no FFT themselves; their public API uses common's
# backend-neutral Complex type. SIMD selects the shared scalar/NEON spatial kernels.
# common/ is a subdirectory of this repo, so COMMON_DIR is a fixed relative path.

CC ?= cc
CXX ?= c++
AR ?= ar
RANLIB ?= ranlib

COMMON_DIR := common
BACKEND = ne10
SIMD ?= 1
WERROR = 0
NO_STDIO = 0
DEBUG = 0
TOOLCHAIN_CHECK = 1
EXTRA_CFLAGS =
EXTRA_LDFLAGS =
CPPFLAGS +=
CXXFLAGS +=

ifneq ($(BACKEND),ne10)
$(error BACKEND must be ne10 (got '$(BACKEND)'): ne10 is the only FFT backend)
endif

define require_bool
ifneq ($$(words $$($(1))),1)
$$(error $(1) must be exactly 0 or 1, got '$$($(1))')
endif
ifneq ($$(filter-out 0 1,$$($(1))),)
$$(error $(1) must be exactly 0 or 1, got '$$($(1))')
endif
endef
$(eval $(call require_bool,SIMD))
$(eval $(call require_bool,WERROR))
$(eval $(call require_bool,NO_STDIO))
$(eval $(call require_bool,DEBUG))
$(eval $(call require_bool,TOOLCHAIN_CHECK))

ifneq ($(findstring ;,$(EXTRA_CFLAGS) $(EXTRA_LDFLAGS)),)
$(error EXTRA_CFLAGS/EXTRA_LDFLAGS must not contain ';')
endif
ifneq ($(filter -Ofast -ffast-math -funsafe-math-optimizations -ffp-contract=%,$(CFLAGS) $(CPPFLAGS) $(EXTRA_CFLAGS)),)
$(error fast-math and caller-supplied -ffp-contract flags are unsupported)
endif

ifeq ($(SIMD),0)
SIMD_CFLAGS := -DSIMD_KERNELS_FORCE_SCALAR
else
SIMD_CFLAGS :=
endif

ifeq ($(DEBUG),1)
OPT_CFLAGS := -O0 -g
else
OPT_CFLAGS := -O2
endif

ifeq ($(WERROR),1)
WERROR_CFLAGS := -Werror
else
WERROR_CFLAGS :=
endif

override CFLAGS := -Wall -Wextra $(OPT_CFLAGS) -std=gnu99 \
                   -I$(COMMON_DIR)/utility -Idoa -Ibf -IVAD -I$(COMMON_DIR)/include -I$(COMMON_DIR) \
                   $(CPPFLAGS) $(EXTRA_CFLAGS) $(SIMD_CFLAGS) \
                   $(WERROR_CFLAGS) -ffp-contract=off -fno-math-errno

SRCS = common/utility/spatial_complex.c common/utility/spatial_simd.c \
       doa/doa_smoother.c doa/srp.c bf/bf.c \
       VAD/masker.c VAD/vad.c VAD/vad_api.c
# The same CC command can select a different SDK through PATH.
# Include effective pointer/FP/ISA predefines to avoid cross-ABI cache reuse.
override CC_TARGET_SIG := $(shell sh $(COMMON_DIR)/scripts/target_abi_signature.sh $(CC) $(CPPFLAGS) $(CFLAGS))
CFG_SIG_PAYLOAD := CC_TARGET_SIG=$(CC_TARGET_SIG) CC=$(CC) CXX=$(CXX) AR=$(AR) RANLIB=$(RANLIB) CFLAGS=$(CFLAGS) CXXFLAGS=$(CXXFLAGS) CPPFLAGS=$(CPPFLAGS) EXTRA_CFLAGS=$(EXTRA_CFLAGS) EXTRA_LDFLAGS=$(EXTRA_LDFLAGS) BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) NO_STDIO=$(NO_STDIO) DEBUG=$(DEBUG) TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK) SRCS=$(sort $(SRCS))
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)

OBJ_ROOT ?= obj
BIN_ROOT ?= bin
OBJ_DIR = $(OBJ_ROOT)/$(BACKEND)-$(CFG_SIG)
BIN_DIR = $(BIN_ROOT)/$(BACKEND)-$(CFG_SIG)

SPATIAL_COMMON_LIB = $(BIN_DIR)/libspatial_common.a
DOA_LIB = $(BIN_DIR)/libdoa.a
BF_LIB = $(BIN_DIR)/libbf.a
VAD_LIB = $(BIN_DIR)/libvad.a

# Every deployable archive in one place: `all`, `libs` and the NO_STDIO audit all read this, so a
# new archive cannot join the build while escaping the gate.
ARCHIVES = $(SPATIAL_COMMON_LIB) $(DOA_LIB) $(BF_LIB) $(VAD_LIB)

# The archives above need only common's headers; the test binary links, so it needs the archive itself.
#
# common keys its output directory on many variables, not just BACKEND/SIMD (DEBUG=1 and NO_STDIO=1
# each land in a different hash dir). Forwarding a subset would resolve and link the wrong archive
# with no diagnostic, so every knob this Makefile accepts is forwarded. A caller that already holds
# the resolved path (the 4ch pipeline Makefile) supplies COMMON_LIB directly. CFLAGS/CXXFLAGS are
# not forwarded: common refuses them on the command line because overriding would drop its
# FP-policy and backend appends.
COMMON_FWD = BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) \
         CC='$(CC)' CXX='$(CXX)' AR='$(AR)' RANLIB='$(RANLIB)' \
         EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=$(NO_STDIO) \
         TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK) DEBUG=$(DEBUG) \
         $(if $(filter command line,$(origin OBJ_ROOT)),OBJ_ROOT='$(abspath $(OBJ_ROOT))') \
         $(if $(filter command line,$(origin BIN_ROOT)),BIN_ROOT='$(abspath $(BIN_ROOT))')
COMMON_LIB ?= $(shell $(MAKE) -s --no-print-directory -C $(COMMON_DIR) $(COMMON_FWD) print-lib-path)
ifeq ($(strip $(COMMON_LIB)),)
$(error cannot resolve the common archive; pass COMMON_LIB= explicitly)
endif

# print-lib-path only echoes; without this rule nothing would ever build the archive the test links.
$(COMMON_LIB): FORCE
	+$(MAKE) -C $(COMMON_DIR) $(COMMON_FWD) lib
	@test -f '$@' || { echo "FATAL: common did not produce $@" >&2; exit 1; }

SPATIAL_COMMON_OBJS = $(OBJ_DIR)/spatial_complex.o $(OBJ_DIR)/spatial_simd.o
DOA_OBJS = $(OBJ_DIR)/doa_smoother.o $(OBJ_DIR)/srp.o
BF_OBJS = $(OBJ_DIR)/bf.o
VAD_OBJS = $(OBJ_DIR)/masker.o $(OBJ_DIR)/vad.o $(OBJ_DIR)/vad_api.o

.DEFAULT_GOAL := all
.PHONY: all libs lib libspatial_common.a libdoa.a libbf.a libvad.a \
        clean bench-masker \
        test test_spatial_third_party audit-no-bf-test-symbols \
        print-bin-dir print-obj-dir print-spatial-common-lib-path \
        print-doa-lib-path print-bf-lib-path print-vad-lib-path \
        audit-no-stdio _cfg_guard

all libs lib: $(ARCHIVES)
libspatial_common.a: $(SPATIAL_COMMON_LIB)
libdoa.a: $(DOA_LIB)
libbf.a: $(BF_LIB)
libvad.a: $(VAD_LIB)

FORCE:
.PHONY: FORCE

_cfg_guard:
	@test -n "$(CC_TARGET_SIG)" && test "$(CC_TARGET_SIG)" != unavailable || { \
	  echo "FATAL: cannot determine compiler target; CC preprocessing with the configured flags must succeed." >&2; \
	  exit 1; }
	@if [ "$(TOOLCHAIN_CHECK)" = "1" ]; then \
	  cc_m="$$($(CC) -dumpmachine 2>/dev/null)"; cxx_m="$$($(CXX) -dumpmachine 2>/dev/null)"; \
	  if [ -z "$$cc_m" ] || [ "$$cc_m" != "$$cxx_m" ]; then \
	    echo "FATAL: CC/CXX target mismatch (CC=$$cc_m, CXX=$$cxx_m)" >&2; exit 1; \
	  fi; \
	fi
	@mkdir -p "$(OBJ_DIR)" "$(BIN_DIR)"
	@if [ -f "$(OBJ_DIR)/config.manifest" ]; then \
	  test "$$(cat "$(OBJ_DIR)/config.manifest")" = "$(CFG_SIG_PAYLOAD)" || { \
	    echo "FATAL: spatial CFG_SIG collision in $(OBJ_DIR)" >&2; exit 1; }; \
	else \
	  tmp="$(OBJ_DIR)/.config.manifest.tmp.$$$$"; \
	  printf '%s\n' "$(CFG_SIG_PAYLOAD)" > "$$tmp"; \
	  mv -f "$$tmp" "$(OBJ_DIR)/config.manifest"; \
	fi

$(OBJ_DIR)/spatial_complex.o: common/utility/spatial_complex.c common/utility/spatial_complex.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/spatial_simd.o: common/utility/spatial_simd.c common/utility/spatial_simd.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/doa_smoother.o: doa/doa_smoother.c doa/doa_smoother.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/srp.o: doa/srp.c doa/srp.h common/utility/spatial_complex.h common/utility/spatial_simd.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/masker.o: VAD/masker.c VAD/masker.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/vad.o: VAD/vad.c VAD/vad.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/vad_api.o: VAD/vad_api.c VAD/vad_api.h VAD/masker.h VAD/vad.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/bf.o: bf/bf.c bf/bf.h common/utility/spatial_complex.h common/utility/spatial_simd.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/test_spatial_third_party.o: tests/test_spatial_third_party.c \
        bf/bf.h bf/bf_test_hooks.h VAD/vad_api.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

# The BF_TESTING build of bf.c, compiled ONLY here. audit-no-bf-test-symbols
# below proves the deployable archive never picks it up.
$(OBJ_DIR)/bf_test.o: bf/bf.c bf/bf.h common/utility/spatial_complex.h \
        common/utility/spatial_simd.h | _cfg_guard
	$(CC) $(CFLAGS) -DBF_TESTING=1 -MD -MP -c -o $@ $<

$(BIN_DIR)/test_spatial_third_party: $(OBJ_DIR)/test_spatial_third_party.o \
        $(OBJ_DIR)/bf_test.o $(DOA_LIB) $(SPATIAL_COMMON_LIB) $(VAD_LIB) \
        $(COMMON_LIB) | _cfg_guard
	$(CC) $(CFLAGS) -o $@ $(OBJ_DIR)/test_spatial_third_party.o \
		$(OBJ_DIR)/bf_test.o $(DOA_LIB) $(SPATIAL_COMMON_LIB) $(VAD_LIB) \
		$(COMMON_LIB) -lm

# Diagnostic only, never part of `test`: it reports timings, so it has no pass/fail and would make
# the gate depend on machine load. Run it on the target (see tests/bench_masker.c).
$(OBJ_DIR)/bench_masker.o: tests/bench_masker.c VAD/masker.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(BIN_DIR)/bench_masker: $(OBJ_DIR)/bench_masker.o $(VAD_LIB) $(COMMON_LIB) \
        | _cfg_guard
	$(CC) $(CFLAGS) -o $@ $(OBJ_DIR)/bench_masker.o $(VAD_LIB) $(COMMON_LIB) -lm

# BENCH_NFFT defaults to the integrator's grid (256/128 @ 16 kHz -> F=129): stage proportions move with F.
BENCH_NFFT ?= 256
bench-masker: $(BIN_DIR)/bench_masker
	"$(abspath $(BIN_DIR))/bench_masker" $(BENCH_NFFT)

audit-no-bf-test-symbols: $(BF_LIB)
	@if nm "$(BF_LIB)" | grep -q 'bf_test_'; then \
	  echo "FAIL: test-only symbol present in deployable $(BF_LIB)" >&2; \
	  exit 1; \
	fi
	@echo "audit-no-bf-test-symbols: clean"

test: audit-no-bf-test-symbols $(BIN_DIR)/test_spatial_third_party
	"$(abspath $(BIN_DIR))/test_spatial_third_party"

test_spatial_third_party: $(BIN_DIR)/test_spatial_third_party

$(SPATIAL_COMMON_LIB): $(SPATIAL_COMMON_OBJS) | _cfg_guard
	t="$@.tmp.$$$$" && rm -f "$$t" && $(AR) rcs "$$t" $^ && $(RANLIB) "$$t" && mv -f "$$t" $@
	@echo "spatial-common [$(BACKEND), SIMD=$(SIMD)] -> $@"

$(DOA_LIB): $(DOA_OBJS) | _cfg_guard
	t="$@.tmp.$$$$" && rm -f "$$t" && $(AR) rcs "$$t" $^ && $(RANLIB) "$$t" && mv -f "$$t" $@
	@echo "doa [$(BACKEND), SIMD=$(SIMD)] -> $@"

$(BF_LIB): $(BF_OBJS) | _cfg_guard
	t="$@.tmp.$$$$" && rm -f "$$t" && $(AR) rcs "$$t" $^ && $(RANLIB) "$$t" && mv -f "$$t" $@
	@echo "bf [$(BACKEND), SIMD=$(SIMD)] -> $@"

$(VAD_LIB): $(VAD_OBJS) | _cfg_guard
	t="$@.tmp.$$$$" && rm -f "$$t" && $(AR) rcs "$$t" $^ && $(RANLIB) "$$t" && mv -f "$$t" $@
	@echo "vad [$(BACKEND), SIMD=$(SIMD)] -> $@"

print-bin-dir:
	@echo $(abspath $(BIN_DIR))
print-obj-dir:
	@echo $(abspath $(OBJ_DIR))
print-spatial-common-lib-path:
	@echo $(abspath $(SPATIAL_COMMON_LIB))
print-doa-lib-path:
	@echo $(abspath $(DOA_LIB))
print-bf-lib-path:
	@echo $(abspath $(BF_LIB))
print-vad-lib-path:
	@echo $(abspath $(VAD_LIB))

audit-no-stdio: libs
	@for lib in $(ARCHIVES); do \
	  hits=$$(nm -A "$$lib" | awk '{ n=$$NF; if (n ~ /^_?(fprintf|vfprintf|printf|puts|fputs|stderr|__stderrp|fputc|fwrite)$$/) print }'); \
	  if [ -n "$$hits" ]; then echo "FAIL: stdio symbols in $$lib"; echo "$$hits"; exit 1; fi; \
	done
	@echo "spatial libraries: NO_STDIO audit PASS"

clean:
	rm -rf "$(OBJ_ROOT)" "$(BIN_ROOT)"

-include $(wildcard $(OBJ_DIR)/*.d)
