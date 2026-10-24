# common — shared audio DSP layer

One copy of the shared DSP code for every consumer repo (AEC, NR, Audio_ALG):

| component | files | notes |
|---|---|---|
| FFT wrapper | `include/fft_wrapper.h`, `src/fft_wrapper_ne10.c` | real-to-complex FFT over NE10; heap and static-memory paths. ARM32/ARM64 compiler-target dispatch; NEON or scalar C FFT |
| NE10 | `lib/ne10/` | vendored DSP subset with documented local patches; see `lib/ne10/VENDORED.md` |
| fast_math | `utility/fast_math.h` | header-only LUT/Taylor exp, minimax log, ARM hardware sqrt, and paper-pinned Martin-2004 E1 approximation |
| simd_kernels | `utility/simd_kernels.h` | header-only **bit-exact** NEON/scalar per-bin kernels (complex magnitude, complex MAC / filter W-updates, EMA, gain apply, min/clip, pairwise sums, `fast_sqrt`, `fast_exp`/`fast_exp_neg`/`fast_log`/`fast_log10`/`exp1_approx`, noise-PSD update with a per-bin smoothing factor (`sk_noise_update_f32`), windowed overlap-add accumulate, float->int16 PCM clip/scale/convert). Every array-form kernel has an always-compiled scalar twin; the one exception is the single-quad `sk_clip_scale_to_s16` (returns `int16x4_t`), which callers guard with `SK_HAVE_NEON`. The AArch64 NEON bodies replicate the scalar operation order lane for lane (strict FMA discipline, no estimate instructions, correctly rounded `vsqrtq_f32`). `SIMD=0` selects every scalar fallback; consumer TUs must build with `-ffp-contract=off`. `make selftest` runs the bitwise NEON-vs-scalar check, gates the minimax log's accuracy against libm, and cross-checks the exp/log family bit for bit against `fast_math.h`. |
| simd_kernel_nn | `utility/simd_kernel_nn.h` | header-only **bit-exact** NEON/scalar kernels for the neural-network pre/post-processing (DFN2, Align-ULCNet, GTCRN, DeepVQE): branch-free all-finite scan, fill, elementwise multiply, feature power, Complex (de)interleave with optional scale, floored WOLA divide, magnitude with eps inside the root (`skn_mag_eps_f32`), and the per-bin complex tap filter (`skn_ctaps_mac_f32`, shared by DeepVQE's CCM and DeepFilterNet2's deep filter). Separate from `simd_kernels.h`; it includes that header only for the NEON switch and the Complex-quad helpers. Same contract (scalar twins, `SIMD=0`, `-ffp-contract=off`); `make selftest` also runs `test/simd_nn_selftest.c`. |
| window | `include/audio_window.h`, `src/audio_window.c` | periodic sqrt-Hann in float32 via `ac_window_build`; no FFT or framing dependency. `audio_stft.h` includes this API for existing callers. |
| STFT/ISTFT | `include/audio_stft.h`, `src/audio_stft.c` | one analysis object (`AcStft`) and one synthesis object (`AcIstft`) for 50%-overlap sqrt-Hann framing: accumulate-and-emit overlap-add or envelope-normalised overlap-add with reflect-padded centered analysis, optional spectrum scales for normalised NN spectra, shared or carved FFT handle and window, static memory only. `make selftest` runs `test/test_stft.c` (window formulas, latency and identity reconstruction, API contract, scaled entry points) |
| HPF | `include/hpf.h`, `src/hpf.c` | biquad high-pass (f32, DF2-transposed), `hpf_create`/`hpf_get_mem_size`+`hpf_init` API — shared by platform code AND AEC's mic path |
| resampler | `include/audio_resampler.h`, `src/audio_resampler.c` | streaming rational polyphase float32 resampler; only 8/16/24/32/48 kHz, 1–8 interleaved channels, heap/static lifecycle, block-boundary-identical state (a sample never depends on how the input was split; NEON and `SIMD=0` sum each output in different orders, so they agree to float rounding, not bit for bit) |

## Build

```
make               # build the archive (NE10 FFT backend)
make SIMD=0        # force every optional SIMD path to scalar (NE10 scalar C kernels)
make selftest      # round-trip + static==heap byte-equality check
make test_audio_utils  # all supported resample-rate pairs
make test          # default test tier (see Tests)
make test-full     # default tier plus the publish checks and the standalone NEON-vs-C FFT parity
```

Output: `bin/ne10-<config-hash>/libaudio_common.a` (keyed by the exact
compiler-flag signature — two configs, e.g. differing only in `SIMD` or
`EXTRA_CFLAGS`, never share a directory or stomp each other's archive). Run
`make print-lib-path` (same flags as your build) to get this build's exact
archive path, or `make publish` for a stable `dist/ne10/current/` handoff path.
Consumers add `-I<here>/include` and `-I<here>` (for `utility/...` headers), then link the archive.

## Tests

`make test` is the default tier. It runs `selftest`, `test_pool`, `test_wav`, `test_wav_nr_style`,
`test_zero_heap` and `test-fast-sqrt-codegen`, then the core scenarios of the build-isolation suite
(`scripts/test_build_isolation.sh`, `make test-isolation`). `selftest` already runs `test_stft`,
`test_stft_guard`, `test_heap_alignment` and `test_audio_utils`; the targets stay available for running one
of them alone. `test_wav` runs the WAV corpus twice, plain and built with `-fsanitize=undefined`.
`test-wav-ubsan` is the same run under another name.

`make test-full` runs the same unit tests plus `test_ne10_force_c` (NEON-vs-scalar FFT parity within a
tolerance; the isolation suite's S4 also runs it) and the full isolation suite, which adds the `make publish`
scenarios (content addressing, attestation, locking, dirty-tree policy, interruption safety). Run it before
cutting a release or changing `publish`.

- The isolation suite needs the AEC repo as a sibling git checkout and does not depend on `SIMD=`: it builds
  both SIMD configurations itself, so `make SIMD=0 test` and `make test` run it identically. `ISOL_TIER=core|full`
  selects the scenarios when it is run directly (`make test-isolation ISOL_TIER=full`).
- `simd_selftest` and `simd_nn_selftest` assert bit-exactness and print no timing by default. Set
  `SK_SELFTEST_BENCH=1` to add their report-only microbenchmarks, e.g. `SK_SELFTEST_BENCH=1 make selftest`.
- `test/run_selftest_ubsan.sh` runs `simd_selftest` under UBSan; it is not part of either tier.

## Resampling

`AudioResampler` accepts only 8000, 16000, 24000, 32000, and 48000 Hz.
Samples are interleaved float32 frames and the object must remain alive across
streaming blocks:

```c
AudioResampler *rs = audio_resampler_create(48000, 16000, 4);
int capacity = audio_resampler_output_bound(rs, input_frames);
int consumed, produced;
audio_resampler_process(
    rs, input_4ch, input_frames, output_4ch, capacity,
    &consumed, &produced);
audio_resampler_destroy(rs);
```

Use the `get_mem_size`/`init` APIs for caller-owned aligned memory. Unequal-rate
input and output must not overlap; equal-rate conversion is an exact `memmove`
pass-through. `audio_resampler_reset()` clears streaming history, so do not
call it at ordinary chunk boundaries.

FFT backend: NE10 is the only backend. `BACKEND=ne10` is the default and the only
accepted value (any other value stops the build with an error).
The compiler target selects ARM32 or ARM64 automatically. With `SIMD=1`,
NE10 uses NEON when `__ARM_NEON` is available; otherwise it uses its scalar C
FFT. `SIMD=0` always selects C. ARM32 uses the level-1 RFFT configuration,
matching the existing ARM64 algorithm. Non-ARM targets remain unsupported.

Shared DSP/spatial SIMD kernels require AArch64 instructions and retain their
scalar fallbacks on ARM32, including an ARM32 NEON FFT build. ARM32 targets
use correctly-rounded `sqrtf` (VSQRT on VFP, libm on soft-float), matching the ARM64 sqrt policy rather
than the portable Newton approximation. Keep the same dispatch/ABI for byte
regressions; cross-ABI or NEON/C FFT results need numerical tolerance.

For example, set the full ARM32 toolchain with
`CC=arm-linux-gnueabihf-gcc CXX=arm-linux-gnueabihf-g++ AR=arm-linux-gnueabihf-ar RANLIB=arm-linux-gnueabihf-ranlib`.
Add `EXTRA_CFLAGS=-mfpu=neon-vfpv4` when that target supports NEON (use
`-mfpu=neon-fp-armv8` for AArch32 ARMv8 cores). Do not use plain `-mfpu=neon`:
it has no hardware fused multiply-add, so every explicit `fmaf()` (the PBFDKF
gradient and weight update in `simd_aec.h`, the complex multiply-accumulate in
`simd_kernels.h`, the FFT power spectrum in `fft_wrapper_ne10.c`, `aec_post.c`)
becomes a libm call. The bits are unchanged, the cost is not. No source edit is
needed when switching compiler targets. Query static-pool sizes/descriptors
again for each ABI; pointers change native layouts.

ARMv7 Advanced SIMD flushes subnormal inputs and results to zero and returns the
default NaN instead of propagating a payload. On ARM32 `SIMD=1` and `SIMD=0` can
therefore differ for subnormal or NaN-payload values. Only the NE10 NEON FFT and, on cores with fused FMA (`-mfpu=neon-vfpv4`), the AEC delay
matched-filter kernels are affected; every other kernel is plain C on ARM32 and follows the
VFP rules in both modes.

## FP-contraction policy

`-ffp-contract=off -fno-math-errno` is one policy across the four repos
(`common`, `NR/c_impl`, `AEC/c_impl`, `Audio_ALG/pipelines`): every TU they
compile, including the vendored NE10 C and C++ TUs, uses these flags, appended
last to `CFLAGS`/`CXXFLAGS`. Each Makefile rejects, at parse time, a
`CFLAGS`/`EXTRA_CFLAGS` containing `-Ofast`, `-ffast-math` or
`-ffp-contract=<anything>`:

```
$ make EXTRA_CFLAGS=-ffast-math
*** FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS contains -ffast-math ...  Stop.
```

`scripts/audit_fp_contract.sh [ne10]` is the disassembly-level check. It builds
the `ne10` config, disassembles a fixed list of expected-scalar objects (this
repo's `hpf.o`, `audio_resampler.o`, `audio_window.o`,
`audio_stft.o`, the NE10 objects, and NR's `anr.o`, `noise_estimator.o`,
`spp_estimator.o`), and fails on any fmadd/fmsub/fnmadd/fnmsub/fmla/fmls, the
signature of the compiler fusing a plain `a*b+c`. `fft_wrapper_ne10.o` requests
fusion explicitly (`fmaf` in `fft_power()`, a `vfmaq_f32` block) and is
reported, never failed. The audit first compiles `fast_sqrt` for Cortex-A53 and
Cortex-A73 Linux and requires an `FSQRT` instruction with no remaining `sqrtf`
reference, which is why `-fno-math-errno` is part of the policy.

## API conventions

- FFT, HPF and resampling expose a heap path (`X_create`/`X_destroy`) and a static-memory
  path (`X_get_mem_size(...)` + `X_init(void* mem, size_t mem_size, ...)`), with a
  runtime `pool_owned`/`is_static` flag — no compile-time gate. `X_destroy` is
  safe on both paths (frees nothing pool-owned). Window generation needs no state;
  STFT/ISTFT objects are static-memory-only.
- FFT convention: `fft_forward` = unnormalised rfft; `fft_inverse` = irfft × 1/N.
  NE10's c2r auto-scales **only because** `NE10_DSP_RFFT_SCALING` is defined
  (default-on in NE10, pinned explicitly in the Makefile — round-trip verified).
  Never build NE10 without it.
- Heap and static-memory (pool) paths are byte-equal (allocation is numerically
  transparent). The NEON and `SIMD=0` (scalar C kernel) paths agree to float
  rounding, not bit for bit (`make test_ne10_force_c` compares them within a
  tolerance); results are only bit-comparable within one `SIMD` setting.

## Precision and parity

Audio buffers, spectra and shared SIMD kernels use float32. AEC/NR retain
selected double bookkeeping and validation expressions to preserve their
Python reference behavior; do not narrow those mechanically.

Python/C validation has two layers: standard-math C against Python, then
shipping C against standard-math C within its approved approximation budget.
C goldens separately protect output within each build configuration. SIMD
kernel twins preserve their documented operation order and NaN contract;
FFT and resampler SIMD modes may have floating-point rounding differences.
