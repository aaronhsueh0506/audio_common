# audiolib

Reusable DSP libraries shared by AEC, NR and Audio_ALG pipelines.

- `common/`: FFT, window/STFT/WOLA, fast math, SIMD kernels, HPF,
  resampling, WAV I/O and static-memory utilities. See
  [`common/README.md`](common/README.md).
- `doa/`, `bf/`, `VAD/`: direction estimation, beamforming and
  single-channel VAD; shared spatial kernels live in `common/utility/`.
- `tests/`: spatial equivalence tests and masker benchmark.

`make SIMD=0|1` builds `libspatial_common.a`, `libdoa.a`, `libbf.a` and
`libvad.a`. `make -C common` separately builds `libaudio_common.a`.
`make test` builds its required common archive, runs spatial equivalence
and checks that deployable beamformer archives contain no test hooks.

Consumers use the sibling checkout `SE/audiolib`. The pipeline's revision
is recorded in [`audiolib.pin`](../Audio_ALG/pipelines/audiolib.pin);
`make -C ../Audio_ALG/pipelines check-pins` verifies it.

Change history is in [`CHANGELOG.md`](CHANGELOG.md) and
[`common/CHANGELOG.md`](common/CHANGELOG.md).
