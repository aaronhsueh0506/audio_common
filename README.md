# audio_common — shared audio DSP layer

One copy of the shared DSP code for every consumer repo (AEC, NR, Audio_ALG):

| component | files | notes |
|---|---|---|
| FFT wrapper | `include/fft_wrapper.h`, `src/fft_wrapper.c` (KISS), `src/fft_wrapper_ne10.c` (NE10) | one public API, two backends |
| KISS FFT | `lib/kiss_fft/` | portable reference backend |
| NE10 | `lib/ne10/` | vendored DSP subset with documented local patches; see `lib/ne10/VENDORED.md` |
| fast_math | `include/fast_math.h` | header-only LUT/Taylor approximations |
| simd_kernels | `include/simd_kernels.h` | header-only **bit-exact** NEON/scalar per-bin kernels (complex magnitude, complex MAC / filter W-updates, EMA, gain apply, min/clip, pairwise sums, fast_sqrt, fast_exp/fast_exp_neg/fast_log/fast_log10/exp1_approx, mcra per-bin-varying-alpha noise update). Every kernel has an always-compiled scalar twin; the AArch64 NEON body replicates the scalar op sequence lane-for-lane (strict FMA discipline, no estimate instructions). Use the Makefile's `SIMD=0` switch to select every scalar fallback; consumer TUs must build with `-ffp-contract=off` (see the header's doc comment). `make selftest` runs the bitwise NEON-vs-scalar check (the exp/log family is additionally cross-checked bit-for-bit against fast_math.h's own implementation, which this header privately replicates rather than includes) |
| HPF | `include/hpf.h`, `src/hpf.c` | biquad high-pass (f32, DF2-transposed), `hpf_create`/`hpf_get_mem_size`+`hpf_init` API — shared by platform code AND AEC's mic path |
| pre-gain | `include/audio_pre_gain.h`, `src/audio_pre_gain.c` | amplitude-dB (`10^(dB/20)`) input gain, heap/static lifecycle, in-place-safe, AArch64 NEON/scalar |
| resampler | `include/audio_resampler.h`, `src/audio_resampler.c` | streaming rational polyphase float32 resampler; only 8/16/24/32/48 kHz, 1–8 interleaved channels, heap/static lifecycle, block-boundary-identical state |

## Build

```
make               # backend auto-detected from the compiler (ARM NEON -> ne10, else kiss)
make BACKEND=kiss  # force portable KISS backend
make BACKEND=ne10  # force NE10 backend (ARM NEON)
make SIMD=0        # keep the backend/grid but force every optional SIMD path to scalar
make selftest      # round-trip + static==heap byte-equality check
make test_audio_utils  # pre-gain + all supported resample-rate pairs
```

Output: `bin/<backend>-<config-hash>/libaudio_common.a` (keyed
by backend AND the exact compiler-flag signature, not just backend — two
configs, e.g. differing only in `EXTRA_CFLAGS`, never share a directory or
stomp each other's archive). Run `make print-lib-path` (same flags as your
build) to get this build's exact archive path, or `make publish` for a stable
`dist/<backend>/current/` handoff path. Consumers add `-I<here>/include` and
link the archive for their chosen backend.

## Pre-gain and resampling

`AudioPreGain` takes amplitude dB, converts it once, and applies no clipping:

```c
AudioPreGain *gain = audio_pre_gain_create(-6.0f);
audio_pre_gain_process(gain, input, output, sample_count); /* in-place is OK */
audio_pre_gain_destroy(gain);
```

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

Backend policy: desktop/CI builds use KISS (bit-reproducible reference); embedded
builds pass `BACKEND=ne10`. Backend is a build knob, not a branch property — every
consumer repo is single-branch (`main`).

## FP-contraction policy

`-ffp-contract=off` is a **unified policy spanning all four repos**
(`audio_common`, `NR/c_impl`, `AEC/c_impl`, `Audio_ALG/pipelines`): every
translation unit any of their Makefiles compile — each repo's own sources
*and* the vendored KISS/NE10 C and C++ TUs alike — builds with this flag,
positioned so nothing can override it. In each Makefile the flag is the
LAST token appended to `CFLAGS`/`CXXFLAGS` (after `EXTRA_CFLAGS`, after any
BACKEND-conditional append, after `WERROR`/`NO_STDIO`), and each Makefile
rejects, at parse time, an `EXTRA_CFLAGS` (or `CFLAGS=` override) containing
`-Ofast`, `-ffast-math`, or `-ffp-contract=<anything>`:

```
$ make EXTRA_CFLAGS=-ffast-math
Makefile:252: *** FP policy conflict: CFLAGS/EXTRA_CFLAGS contains -ffast-math; this repo pins -ffp-contract=off; remove -ffast-math from EXTRA_CFLAGS.  Stop.
```

`scripts/audit_fp_contract.sh` is the disassembly-level proof: it builds the
`kiss`/`ne10` configs, disassembles a fixed list of TUs expected to be
genuinely scalar (this repo's `hpf.o`/`kiss_fft.o`/the NE10 scalar-C
objects, plus NR's three core objects), and fails if any fmadd/fmsub/
fnmadd/fnmsub/fmla/fmls instruction shows up — the signature of the
compiler choosing, on its own, to fuse a plain `a*b+c` expression, which is
exactly what this flag forbids. It does NOT flag TUs that request fusion
**explicitly** — `fft_wrapper.c`/`fft_wrapper_ne10.c`'s `fft_power()` scalar
tail calls `fmaf()` directly (see that function's own comment) and both
files carry an `__ARM_NEON`-guarded explicit `vfmaq_f32` block; those are
EXEMPT (reported, never failed) with the exact source-level reason — see
the script's own header comment for the full rationale, including a
non-obvious finding (`NE10_rfft_float32.neonintrinsic.o`, despite its name,
uses no fused intrinsic anywhere and is audited like any other scalar TU).
Run it with `scripts/audit_fp_contract.sh kiss ne10`.

## API conventions

- Every module exposes a heap path (`X_create`/`X_destroy`) and a static-memory
  path (`X_get_mem_size(...)` + `X_init(void* mem, size_t mem_size, ...)`), with a
  runtime `pool_owned`/`is_static` flag — no compile-time gate. `X_destroy` is
  safe on both paths (frees nothing pool-owned).
- FFT convention: `fft_forward` = unnormalised rfft; `fft_inverse` = irfft × 1/N.
  KISS divides manually; NE10's c2r auto-scales **only because**
  `NE10_DSP_RFFT_SCALING` is defined (default-on in NE10, pinned explicitly in the
  Makefile — round-trip verified). Never build NE10 without it.
- NE10 output is NOT bit-identical to KISS (different FFT implementation). All
  bit-exactness / parity-vs-Python guarantees are KISS-only; NE10-heap vs
  NE10-static must still be byte-equal (allocation is numerically transparent).

## Precision policy — float32 everywhere (f32 campaign, 2026-07-15)

All four repos (AEC / NR / Audio_ALG / audio_common) run float32 end-to-end on
their production paths: per-sample loops, per-bin loops, scalar bookkeeping,
and both FFT backends. The former fp64 sites (numpy-parity scalar mirrors, the
delay decimator biquads, the fp64 HPF variant) were converted in the staged f32
campaign — Python bit-exact parity is retired; the regression anchors are
C-goldens plus AECMOS/soak gates against the `fp64-baseline` tags. Only
exception: vendored kiss_fft computes its twiddles in double at plan time
(init-only, host reference backend; the NE10 embedded backend is double-free
including init). Rule for new code: f32 everywhere in production sources,
`f`-suffix every literal (`0.95f` — an unsuffixed literal silently promotes
the expression to double), float libm variants (`sqrtf`/`powf`/`expf`), and no
`double` declarations.
