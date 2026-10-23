# Changelog

All notable changes to `audiolib`. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This file is new: `audiolib` is the renamed `audio_common` repo (see
[`common/CHANGELOG.md`](common/CHANGELOG.md) for that component's own
history up to and including the rename), merged with the SRP-PHAT/DOA
smoother, beamformer and VAD sources that used to vendor under
`Audio_ALG/pipelines/third_party/`. This file records changes to `audiolib`
as a whole from this point on.

## [Unreleased] — 2026-10-23 — Input-validation fixes; sources and docs condensed

### Fixed
- `srp_init`: a search band whose `low_freq` is at or above Nyquist is clamped to the last bin
  (`f_start <= F - 1`) instead of indexing past the F-sized buffers.
- `bf` and `srp` angle-to-index conversion: non-finite angles return 0 / -1, and very large angles are
  reduced with `fmodf` first; the `while (deg >= 360)` loops no longer spin on +-Inf or |angle| above 2^31.
  `bf_init` rejects a non-finite `fixed_doa_rad` when fixed mode is on. Ordinary angles are unchanged.
- `Makefile`: `OBJ_ROOT` / `BIN_ROOT` given on the command line are forwarded to `common/` as absolute paths, so
  the archive path the rule checks is the one the sub-make builds.

### Changed
- `bf.h` and `srp.h` state the `mask` polarity: `srp()` uses bins where the mask is non-zero, `bf_process` freezes
  adaptation on them.
- Small de-duplications in `bf.c`, `spatial_simd.c`, `srp.c` and `masker.c` (shared bypass block, one NEON-tail
  helper); results are unchanged. Source comments and `common/docs` are shortened to contracts, and statements that no
  longer matched the code (include flags, selftest total 37,045,791) are corrected.

### Added
- `tests/test_spatial_third_party.c`: degenerate-band and non-finite-angle cases.

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
- `srp()` clears all `F` bins of `score_scratch` only for the first angle and
  just the searched band for later ones (an empty band keeps the full clear),
  removing unused stores; outputs and the public scratch contents are
  unchanged.

### Removed
- The never-enabled compile-time switches `SRP_USE_UNIQUE_PAIRS` (srp.c; no
  build ever enabled the full-pair SRP-PHAT branch) and `BF_USE_PROJECTION_BLOCKING` (bf.c/bf.h; the full blocking-matrix branch
  and its `scratch_b` field, likewise never enabled), with the three helpers only they used:
  `spatial_complex_div_real`, `spatial_complex_abs`, `spatial_complex_abs2`.
  The default behaviour is unchanged.

### Source comments condensed
Long explanatory comments in `bf/`, `doa/` and `VAD/` were cut to short contracts; the design notes
that existed only in those comments are recorded here.
- BF RLS hardening: a real recording failed the pipeline at frames ~2800-3300 (20-27 s in) on every grid,
  scalar and SIMD, because `P = (P - gain*q) / lambda` grows without bound under sustained low excitation
  and reached inf. Two per-bin guards remain: the `BF_P_DIAG_FLOOR/CEIL` clamp after every update, and an
  `isfinite(upu_real) && upu_real > 0` check that resets only that bin's `P`/`wa`. `BF_WA_LEAK` has a time
  constant of 100000 updates (not hops), far slower than the RLS forgetting time.
- BF keeps a persistent per-hop work area (avoids a ~29 KiB stack spike at 48 kHz / 1024 FFT) and uses the
  projection form `u = x - a * (a^H x / a^H a)` for the blocking output, O(F*M) instead of O(F*M*M).
  `frame_idx`/`first_doa_frame` are 64-bit so a long-running counter cannot overflow (a 32-bit counter wraps
  after ~199-265 days at typical hop rates).
- SRP computes the PHAT cross spectrum once per frame and only over the searched band (about 144 of 513
  bins at 48 kHz / 1024 for a ~300-7000 Hz band, ~3.5x fewer in that loop); bins outside the band are never
  read. The pool path builds angles and steering with the same formula as the heap path.
- VAD median filter on the binary mask is a majority test, `ones >= k - k/2` (`ones > k/2` agrees only for
  odd k). `vad_api` refuses configs that would silently misbehave: `vad_cfg.F` above the mask length
  (out-of-bounds read), even `smooth_size` (window widens to k+1), `smooth_size <= 0` (pass-through),
  `median_k` outside 5/7/9 (`med_buf` holds 9).
- The heap constructors of BF, SRP, VAD and the masker share the pool path's single carve, so both entry
  points have one layout and one free.
