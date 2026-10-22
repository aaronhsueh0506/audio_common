# Changelog

All notable changes to `audiolib`. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This file is new: `audiolib` is the renamed `audio_common` repo (see
[`common/CHANGELOG.md`](common/CHANGELOG.md) for that component's own
history up to and including the rename), merged with the SRP-PHAT/DOA
smoother, beamformer and VAD sources that used to vendor under
`Audio_ALG/pipelines/third_party/`. This file records changes to `audiolib`
as a whole from this point on.

## [Unreleased]

### Changed
- Renamed the repo from `audio_common` to `audiolib`; its prior content
  (FFT backend, fast_math, SIMD kernels, HPF, resampler, STFT/WOLA, WAV I/O)
  now lives at `common/` inside this checkout (`git mv`, history preserved).
- Merged `Audio_ALG/pipelines/third_party/{doa,bf,VAD}` into this checkout
  (`doa/`, `bf/`, `VAD/`) and `third_party/utility/{complex,spatial_simd}.*`
  into `common/utility/` (new sibling directory under `common/`), renaming
  `complex.{c,h}` to `spatial_complex.{c,h}` (guard `COMPLEX_UTILS_H` ->
  `SPATIAL_COMPLEX_H`; the `spatial_complex_*` symbols were already
  namespaced and are unchanged). `third_party/gain/` stayed behind at
  `Audio_ALG/pipelines/third_party/gain/` -- it is not linked into any
  shipped binary, so it was out of scope for this merge.
- Added a top-level `Makefile` (trimmed from the former
  `third_party/Makefile`): builds `libspatial_common.a`/`libdoa.a`/
  `libbf.a`/`libvad.a` and this checkout's `test_spatial_third_party`;
  the former `third_party/Makefile`'s `gain`-related targets and
  `AC_DIR`/`AC_DIR_ABS`/`AC_FWD`/`AC_LIB` cross-repo resolution dropped out
  (`common/` is now a same-repo sibling, so `COMMON_DIR := common` resolves
  with no wildcard search and no forwarded build knobs).
- `tests/test_spatial_third_party.c` is the PHAT/BF/SRP/VAD half of the
  former combined `third_party/tests/test_spatial_third_party.c`; the gain
  half (plus its own copy of the shared `check_pool_contract` helper) moved
  to `Audio_ALG/pipelines/third_party/gain/tests/test_gain.c`.
- Consumers (`AEC`, `NR`, `Audio_ALG/pipelines`, `Audio_ALG/pipelines/
  4ch_aec_bf_nr_res`, and `AINR`/`AIAEC`) updated their `AC_DIR`/`AC_LIB`/
  `AC_FWD` Makefile variables to `COMMON_DIR`/`COMMON_LIB`/`COMMON_FWD`
  (`Audio_ALG/pipelines/4ch_aec_bf_nr_res/Makefile`'s former
  `THIRD_PARTY_DIR` became `AUDIOLIB_DIR`) and their hardcoded
  `../../audio_common` family of relative paths to the new `audiolib`
  location (an extra `common/` segment where they resolve into what is now
  this checkout's `common/` subdirectory). `Audio_ALG/pipelines/
  audio_common.pin` was renamed to `audiolib.pin`
  (`AUDIO_COMMON_COMMIT` -> `AUDIOLIB_COMMIT`); the pinned commit value is
  unchanged (the rename is an uncommitted working-tree move, not a new
  commit).
- Archive filenames are unchanged (`libaudio_common.a` still comes out of
  `common/`'s build) -- only the directory path producing them moved.

### Removed
- The never-enabled compile-time switches `SRP_USE_UNIQUE_PAIRS` (srp.c; no
  build ever enabled the full-pair SRP-PHAT branch) and `BF_USE_PROJECTION_BLOCKING` (bf.c/bf.h; the full blocking-matrix branch
  and its `scratch_b` field, likewise never enabled), with the three helpers only they used:
  `spatial_complex_div_real`, `spatial_complex_abs`, `spatial_complex_abs2`.
  The default behaviour is unchanged.
