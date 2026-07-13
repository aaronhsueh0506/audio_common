# audio_common — shared audio DSP layer

One copy of the shared DSP code for every consumer repo (AEC, NR, Audio_ALG):

| component | files | notes |
|---|---|---|
| FFT wrapper | `include/fft_wrapper.h`, `src/fft_wrapper.c` (KISS), `src/fft_wrapper_ne10.c` (NE10) | one public API, two backends |
| KISS FFT | `lib/kiss_fft/` | portable reference backend |
| NE10 | `lib/ne10/` | whole, **unmodified** NE10 DSP module (C + NEON kernels) |
| fast_math | `include/fast_math.h` | header-only LUT/Taylor approximations |
| HPF | `include/hpf.h`, `src/hpf.c` | biquad high-pass |

## Build

```
make               # backend auto-detected from the compiler (ARM NEON -> ne10, else kiss)
make BACKEND=kiss  # force portable KISS backend
make BACKEND=ne10  # force NE10 backend (ARM NEON)
make selftest      # round-trip + static==heap byte-equality check
```

Output: `bin/<backend>/libaudio_common.a` (per-backend dirs — the two builds coexist).
Consumers add `-I<here>/include` and link the archive for their chosen backend.

Backend policy: `main` branches build against KISS (bit-reproducible reference);
`feature/static-memory` (embedded deliverable) builds against NE10.

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

## ⚠ Known collision to resolve before linking hpf anywhere

AEC has its OWN internal `src/hpf.c` (embedded-struct API: `hpf_init(Hpf*, cutoff_hz,
sample_rate)`, `hpf_process`) whose symbol names collide with this library's standalone
hpf (`hpf_init(void* mem, size_t, ...)` — different signature). A binary that pulls hpf
members from BOTH archives fails with duplicate symbols (loud, at link time). Until the
two are reconciled (AEC migrating to this hpf, or a rename), do NOT call audio_common's
hpf from anything that also links libaec — the AEC mic-path HPF is internal to
aec_process and needs nothing from here.
