# audio_common — shared audio DSP layer

One copy of the shared DSP code for every consumer repo (AEC, NR, Audio_ALG):

| component | files | notes |
|---|---|---|
| FFT wrapper | `include/fft_wrapper.h`, `src/fft_wrapper.c` (KISS), `src/fft_wrapper_ne10.c` (NE10) | one public API, two backends |
| KISS FFT | `lib/kiss_fft/` | portable reference backend |
| NE10 | `lib/ne10/` | whole, **unmodified** NE10 DSP module (C + NEON kernels) |
| fast_math | `include/fast_math.h` | header-only LUT/Taylor approximations |
| HPF (f32) | `include/hpf.h`, `src/hpf.c` | biquad high-pass, DF2-transposed, `hpf_create`/`hpf_get_mem_size`+`hpf_init` API |
| HPF (f64) | `include/hpf_f64.h`, `src/hpf_f64.c` | fp64-state Direct Form II biquad, bit-exact port of the AEC Python reference; value-type caller-owned struct (`hpf_f64_*`) — used by AEC's mic path |

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

## Precision policy (f32 vs f64) — adjudicated 2026-07-15, don't reopen without new evidence

Everything hot is already float32: both FFT backends (`kiss_fft_scalar` = float;
NE10 is f32-only), every per-bin loop in AEC/NR, and the AEC matched-filter dot
products (f32 + NEON). A full four-repo audit found **zero** `double` on any
per-sample or per-bin production hot path in NR / audio_common / the Audio_ALG
pipelines (sole exception: `hpf_f64`, ~1.4e5 flops/s, parity-mandated), and in
AEC only (a) per-hop scalar bookkeeping that textually mirrors numpy fp64
promotion rules — converting it breaks the documented bit-exact Python↔C
contract — and (b) the delay-estimator decimator biquads (~1.15M flops/s).
Measured on an out-of-order arm64 core, a 4-stage biquad cascade in fp64 runs
within 1.08× of f32 (IIR recurrences don't vectorise), so converting (b) would
recover on the order of 0.001% of a core while forcing a delay-chain golden
regen + AECMOS re-sample and risking f32 state-quantisation drift in the
highest-Q anti-alias section (pole radius ≈ 0.985). Rule of thumb: new code
uses f32 in loops over samples/bins; scalar fp64 is fine where it mirrors a
Python reference or costs O(per-hop); any f32 conversion of existing fp64 needs
a measured hotspot first, then the same gate discipline as the matched-filter
conversion.

## Why two HPFs (collision resolved)

AEC's mic-path HPF is a **bit-exact fp64 port** of its Python reference — its state
precision and value-type API are part of the Python↔C parity contract, so it cannot
adopt the f32 platform HPF without changing output. It used to live inside the AEC
repo under colliding symbol names (`hpf_init`/`hpf_process`/`hpf_reset` with a
different signature — a latent duplicate-symbol / ABI-mismatch hazard); it now lives
here as `hpf_f64_*`, so both filters link side by side safely. Pick by need:
`hpf.h` (f32, heap/pool API) for platform code; `hpf_f64.h` for anything that must
match the AEC reference bit-for-bit. The fp64 cost is negligible (~6 scalar fp64
ops/sample on one path; Cortex-A/AArch64 has hardware fp64).
