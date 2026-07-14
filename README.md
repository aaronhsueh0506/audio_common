# audio_common — shared audio DSP layer

One copy of the shared DSP code for every consumer repo (AEC, NR, Audio_ALG):

| component | files | notes |
|---|---|---|
| FFT wrapper | `include/fft_wrapper.h`, `src/fft_wrapper.c` (KISS), `src/fft_wrapper_ne10.c` (NE10) | one public API, two backends |
| KISS FFT | `lib/kiss_fft/` | portable reference backend |
| NE10 | `lib/ne10/` | whole, **unmodified** NE10 DSP module (C + NEON kernels) |
| fast_math | `include/fast_math.h` | header-only LUT/Taylor approximations |
| simd_kernels | `include/simd_kernels.h` | header-only **bit-exact** NEON/scalar per-bin kernels (complex magnitude, complex MAC / filter W-updates, EMA, gain apply, min/clip, pairwise sums, fast_sqrt). Every kernel has an always-compiled scalar twin; the AArch64 NEON body replicates the scalar op sequence lane-for-lane (strict FMA discipline, no estimate instructions). `SIMD_KERNELS_FORCE_SCALAR` forces the fallback; consumer TUs must build with `-ffp-contract=off` (see the header's doc comment). `make selftest` runs the bitwise NEON-vs-scalar check |
| HPF | `include/hpf.h`, `src/hpf.c` | biquad high-pass (f32, DF2-transposed), `hpf_create`/`hpf_get_mem_size`+`hpf_init` API — shared by platform code AND AEC's mic path |

## Build

```
make               # backend auto-detected from the compiler (ARM NEON -> ne10, else kiss)
make BACKEND=kiss  # force portable KISS backend
make BACKEND=ne10  # force NE10 backend (ARM NEON)
make selftest      # round-trip + static==heap byte-equality check
```

Output: `bin/<backend>/libaudio_common.a` (per-backend dirs — the two builds coexist).
Consumers add `-I<here>/include` and link the archive for their chosen backend.

Backend policy: desktop/CI builds use KISS (bit-reproducible reference); embedded
builds pass `BACKEND=ne10`. Backend is a build knob, not a branch property — every
consumer repo is single-branch (`main`).

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
