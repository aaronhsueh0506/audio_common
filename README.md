# audiolib

Reusable DSP libraries shared by AEC, NR and Audio_ALG pipelines.

- `common/`: FFT, window/STFT/WOLA, fast math, SIMD kernels, HPF,
  resampling, WAV I/O and static-memory utilities. See
  [`common/README.md`](common/README.md).
- `doa/`, `bf/`, `VAD/`: direction estimation (SRP-PHAT with a DOA smoother),
  beamforming and single-channel VAD; shared spatial kernels live in
  `common/utility/`.
- `tests/`: spatial equivalence tests and masker benchmark.

`make SIMD=0|1` builds `libspatial_common.a`, `libdoa.a`, `libbf.a` and
`libvad.a`. `make -C common` separately builds `libaudio_common.a`.
`make test` builds its required common archive, runs spatial equivalence
and checks that deployable beamformer archives contain no test hooks.
`make bench-masker` prints per-stage masker timings (report only), and
`make audit-no-stdio` checks the four spatial archives for stdio symbols.

Consumers use the sibling checkout `SE/audiolib`. The pipeline's revision
is recorded in [`audiolib.pin`](../Audio_ALG/pipelines/audiolib.pin);
`make -C ../Audio_ALG/pipelines check-pins` verifies it.

## Documentation

| Topic | Where |
|---|---|
| common: call order, examples, sequence diagrams | `common/docs/html/audio_common.html` |
| common: data flow, I/O and state tables, latency | `common/docs/html/audio_common_detail.html` |
| common: integrator API summary | `common/docs/html/audio_common_c_api.html` |
| common: external vs internal parameters, public vs internal functions | `common/docs/html/audio_common_params.html` |
| common: per-function manual | [`common/docs/c_user_manual_zh_TW.md`](common/docs/c_user_manual_zh_TW.md) (index; per-module chapters in the `c_user_manual_zh_TW_0*.md` part files next to it) |
| VAD + SRP + BF chain: call order, sequence diagram, memory | `docs/html/spatial_chain.html` |
| SRP-PHAT / DOA smoother | `docs/html/doa_srp.html` |
| Beamformer | `docs/html/bf.html` |
| VAD (masker + mask VAD) | `docs/html/vad.html` |

The pages are plain HTML (no scripts, no external resources); open them from
the checkout. Each page lists its source files and tests.

## Spatial modules at a glance

All three work on frequency-domain input (`const Complex *const *X`, one array
of `F = NFFT/2+1` bins per microphone) and do not run an FFT. Link
`libdoa.a libbf.a libvad.a libspatial_common.a -lm`; `libaudio_common.a` is not
needed (the headers of `common/` are). Every object has a `*_get_mem_size` /
`*_init` caller-pool path (16-byte aligned pool; a size of 0 or a NULL return
means the configuration was refused) and a heap constructor.

Per-hop order: `vad_api_process` -> `doa_step` -> `bf_process_with_weights`.
The mask from the VAD selects the bins SRP scores and freezes the same bins in
the beamformer (the two headers state opposite polarities for the same
non-zero value). `bf_init` borrows `srp->a_array`, so destroy the beamformer
before the SRP object.

| Module | External (a third party may set; fixed at init) | Internal (do not open) |
|---|---|---|
| SRP-PHAT (`SRP_Config`) | `num_angles`, `low_freq`/`high_freq`, `enable_smoothing`, `switch_consec`, `angle_tol`, `update_interval`, `c`, the array geometry | angle grid `a*2pi/A`, use of all microphone pairs, PHAT epsilon, `f_start >= 1` clamp, no 2pi wrap-around in the smoother, scratch and state fields |
| BF (`BF_Config`) | `enable`, `lambda` (per RLS update), `mu`, `enable_fix_mode`/`fixed_doa_rad`, `adapt_interval` | `fixed_align_notebook`, `BF_WA_LEAK`, `BF_P_DIAG_FLOOR/CEIL`, RLS guards, `bin_resets` |
| VAD (`VADApiConfig`) | `VadConfig` mode/threshold/median/hangover, masker band and `margin_dB`/`spp_thr`, the enable switches | per-hop EMA coefficients (`E_alpha_*`, `M_alpha`, `T_alpha`), `spp_upd_thr`, numeric floors |

The reasoning, ranges, source lines and the full function lists (what an
integrator calls versus what is internal) are in the pages above. The VAD has
no product default configuration and no in-tree pipeline instantiates it.

Change history is in [`CHANGELOG.md`](CHANGELOG.md) and
[`common/CHANGELOG.md`](common/CHANGELOG.md).
