# Changelog

All notable changes to audio_common. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The README states the
current behaviour only; how it got there is recorded here.

## [Unreleased] — 2026-10-23 — FFT handle shares its input staging

- The preserving `fft_forward`/`fft_inverse` stage their input in one block per handle instead of
  two (they never run at once and hand nothing back in it), so `fft_get_mem_size` drops by
  `4 * fft_size` bytes (256: 8,176 -> 7,152; 512: 15,600 -> 13,552; 1024: 30,448 -> 26,352 on
  AArch64; 8,128 -> 7,104, 15,552 -> 13,504, 30,400 -> 26,304 on ARM32) and the heap path makes
  one allocation fewer. Outputs are bit-identical; `ac_stft`/`ac_istft` pools with their own FFT
  shrink by the same amount.

## [Unreleased] — 2026-10-23 — `ck_field_size_reps()` is shared

- `mem_align.h` gains `ck_field_size_reps()` (checked `total += ALIGN16(count * elem_size) * reps`),
  replacing AEC's two identical local copies. The gcc-for-32-bit-ARM false
  `-Waggressive-loop-optimizations` error in the scalar twin of `sk_pairwise_sum_tailfold_f32` is
  silenced by a gcc-only diagnostic pragma around its leaf (no code generation change).

## [Unreleased] — 2026-10-23 — test_stft checks contracts only

### Removed
- `test/stft_ref/` (verbatim copies of the former pipeline and neural pre/post framing code) and the
  families of `test_stft` that compared `audio_stft.h` with them, together with the `HASH|` and `B|`
  report lines. `test_stft` keeps the window formulas (two independent spellings, every power of two
  from 16 to 8192), latency and identity reconstruction, API contract and the scaled
  interleaved/split entry points, and is built once instead of twice.

### Fixed
- `make selftest` links the SIMD selftests with `-lm` after the objects, so it links on ARM32
  toolchains that resolve libraries left to right. `run_selftest_ubsan.sh` compiles again (it
  lacked `-I.`). `test-fast-sqrt-codegen` no longer needs `AARCH64_CC` under an ARM32 `CC`: it
  runs the ARM32 half with that `CC`, uses any AArch64 compiler it finds for the other half, and
  reports the half it cannot run.

## [Unreleased] — 2026-10-22 — ARM32 flags and publish symbol audit

### Fixed
- The publish backend-symbol audit again requires the NEON FFT entry point
  (`ne10_fft_r2c_1d_float32_neon`) whenever the compiler target has NEON, besides the
  C one. Both must be DEFINED in the archive: the FFT wrapper object holds undefined
  references to each, so a bare name match passed even with the defining object
  missing. The audit is also available as `make check-archive-symbols
  AUDIT_ARCHIVE=<archive>`.

### Documented
- ARM32 NEON builds use `EXTRA_CFLAGS=-mfpu=neon-vfpv4` (`-mfpu=neon-fp-armv8` on
  AArch32 ARMv8), not plain `-mfpu=neon`, whose lack of hardware FMA turns every
  explicit `fmaf()` into a libm call (same bits, much slower).
- ARMv7 Advanced SIMD flushes subnormals to zero and returns the default NaN, so
  `SIMD=1` and `SIMD=0` can differ on ARM32 for subnormal or NaN-payload values; only
  the NE10 NEON FFT and, on cores with fused FMA, the AEC delay matched filter are affected.

## [Unreleased] — 2026-10-22 — ARM32 target support

### Changed
- Select NE10 ARM32/ARM64 NEON or C FFT from compiler capabilities and the
  existing SIMD switch. Pin the level-1 RFFT config on both ABIs and omit
  the NEON TU on targets without NEON; BACKEND remains ne10.
- ARM32 fast sqrt uses correctly-rounded sqrtf (hardware or software libm); shared AArch64-only
  vector kernels retain scalar ARM32 fallbacks. ARM64 arithmetic is unchanged.
- Align resampler heap pools explicitly to 16 bytes, including on ARM32
  allocators that only guarantee 8-byte alignment.
- Add a 32-bit ABI marker helper for composed memory descriptor hashes;
  existing ARM64 hashes remain unchanged.
- Keep NE10 opt-in AArch64 inline assembly gated on its actual ISA, selecting
  the existing intrinsic fallback on ARM32 (vendor patch P0004).
- Bound NN div-floor/complex-tap vector loops by the rounded-down size,
  avoiding unsigned wrap and GCC constant-tail diagnostics without changing
  per-bin arithmetic. Keep ERB selftest NaN classification separate from
  payload counts; finite output and selection/copy checks remain strict.
- Make pool-size assertions ABI-specific; add forced-8-byte-malloc and
  allocation-failure coverage for resampler heap construction.

## [Unreleased] — 2026-10-21 — E1 branch skipping

### Changed
- E1's AArch64 kernel skips the unused log or exp branch when all four bins
  take the same branch. Mixed blocks, scalar tails and standard-math builds
  retain their arithmetic; output values remain byte-identical under the
  existing strict FP policy. Floating-point exception flags are not preserved.
- Added E1 block-regime coverage for boundaries, NaNs, tails, canaries and
  exact in-place operation.

## [Unreleased] — 2026-10-21 — Shared periodic sqrt-Hann window

### Changed
- Extracted `ac_window_build` unchanged into standalone `audio_window.h` /
  `src/audio_window.c`. `audio_stft.h` includes the new header, preserving
  existing callers. AEC, NR runners and pipeline fixtures use the shared
  periodic sqrt-Hann construction; independent frozen test formulas remain.
- Added the window object to the archive source list and scalar FP-contract
  audit. Window table values remain byte-identical; the generated RNNoise header only
  updates its standalone regeneration command.

## [Unreleased]

### Removed
- KISS FFT backend: `lib/kiss_fft/` (vendored sources, license and provenance
  notes) and `src/fft_wrapper.c`. NE10 is now the only FFT backend.
- `BACKEND=kiss`, and the Makefile's automatic fall-back to KISS on a compiler
  target without ARM NEON. The `$(CC) -dM -E` NEON probe is gone with it.
- `include/` `src/` `test/` no longer contain the word "kiss" (the vendored
  `lib/ne10/` sources still carry their upstream KISS-FFT-style notices).

### Changed
- `Makefile`: `BACKEND` defaults to `ne10`; any other value (`kiss`, an empty
  value, a typo) stops the build at parse time with
  `BACKEND must be ne10 (got '...')`, for build targets and `print-*` queries
  alike. `BACKEND` stays a variable because it names the `obj/`, `bin/` and
  `dist/` directories and the consumer Makefiles forward it.
- `Makefile`: the NE10 source list, flags, link driver and `CFG_SIG` payload are
  unchanged for ne10 configs, so existing `obj/ne10-<sig>/` and
  `bin/ne10-<sig>/` directory names stay the same (SIMD=1 `597631494`,
  SIMD=0 `3784739529` on this host). Only the removed KISS configs and the
  `src/fft_wrapper.c` entry in the source list went away.
- `Makefile`: the `publish` backend-symbol audit checks the NE10 symbol only;
  the toolchain-coherence guard no longer tests `BACKEND`; the banner no longer
  advertises `BACKEND=kiss`. Over-long review-history comments were condensed
  to the reason each mechanism exists.
- `scripts/audit_fp_contract.sh`: default backend list is `ne10`; the two
  KISS-only rows (`kiss_fft.o`, `fft_wrapper.o`) are removed. The audit itself
  (disassembly scan of the scalar-class objects, EXEMPT handling of explicit
  fusion) is unchanged.
- `scripts/test_build_isolation.sh`: every scenario that used `BACKEND=kiss` as
  its default configuration now uses the default ne10 build. The scenarios that
  needed two distinct configurations (S1, S2, S5, S7p, S15) use `SIMD=1` /
  `SIMD=0` instead of kiss / ne10. S12's "kiss skips the toolchain guard" case
  is dropped; new S26 asserts that `BACKEND=kiss`, an empty value and an unknown
  value are rejected and create no `obj/`/`bin/` directory.
  `adopt_worktree_clone()` now deletes in the clone a tracked file that was
  deleted in the worktree (it used to abort on `cp`). S4 runs
  `test_ne10_force_c` with `SIMD` unset in its environment (the target builds
  both variants itself). S25's positive control compared the selftest check
  count with a stale hard-coded number; it now runs a reference `selftest` in
  the same invocation and requires the flagged run to print the same
  `TOTAL CHECKS` lines, the same number of `ALL PASS` lines and the same
  per-test result lines, and every selftest link line to carry the flag.
  S2 and S3 probe `audio_stft.o` and `fft_wrapper_ne10.o` (the members that
  differ between `SIMD=1` and `SIMD=0`; `hpf.o` is identical in both) and check
  every archive member against its own keyed object; S3 builds a `SIMD=1` and a
  `SIMD=0 -DPAR_PROBE` configuration in parallel under one scratch root. S4
  also requires the NEON parity binary to link the NEON R2C kernel and the
  forced-C one not to. S5 ages both clones before the relink steps so that a
  relink can only be caused by the step under test, finds the AEC object that
  includes `fast_math.h` / `wav_io.h` from its `.d` file instead of by name
  (fails instead of aborting when there is none), and `adopt_worktree_clone()`
  also adopts untracked files and staged renames. S10's membership check can no
  longer pass when `ar` fails.
- `scripts/audit_alloc_symbols.sh`: usage examples use the ne10 archive.
- `lib/ne10/LICENSE`, `lib/ne10/VENDORED.md`: no longer point at the removed
  `lib/kiss_fft/` files. The full Kiss FFT license text remains in the header of
  each NE10 source file that carries it.
- `test/test_pool_contract.c`: the NE10 pool-size checks are unconditional
  (`NE10_ENABLE_DSP` is always defined).
- `README.md`, `docs/c_user_manual_zh_TW.md`, `docs/html/`: describe the single
  NE10 backend; KISS rows, columns and measurements are removed. The README
  precision-policy section keeps the conclusion only.

### Notes
- A compiler target without compile-time AArch64 NEON (for example x86_64)
  cannot build this archive any more: `src/fft_wrapper_ne10.c` stops with
  `#error "NE10 backend requires compile-time NEON (__ARM_NEON && __aarch64__)"`.
  Before this change such hosts silently selected KISS. `SIMD=0` is not a
  substitute: it still needs an AArch64 NEON compiler and only switches
  dispatch to NE10's scalar C kernels.
- Verification (arm64, `BACKEND=ne10`, SIMD=0 and SIMD=1, against the tree
  before this change): the 12 archive members and all library objects are
  byte-identical; `selftest` prints the same `TOTAL CHECKS: 37073253`;
  `test_stft_512/1024` logs and `test_ne10_c_parity` dumps are byte-identical.
  `test_pool_contract.o` and `test_fft_zero_heap.o` differ only in `__LINE__`
  immediates because their comments were shortened.
  `scripts/test_build_isolation.sh` passes 225/225 on a clean snapshot with the
  AEC checkout at its committed state; the tree before this change passed 212
  and failed 1 (S25's stale check count). Mutation runs that break an
  invariant of S1-S5, S25 and S26 (shared object directory, a `SIMD` setting
  that has no effect, a stale or foreign archive member, a flag that is
  validated but never linked, a flattened parity binary) make the script fail.
- Precision-policy history moved out of the README: the 2026-07-15 f32 campaign
  converted the former fp64 sites (numpy-parity scalar mirrors, the delay
  decimator biquads, the fp64 HPF variant) to float32 and retired Python
  bit-exact parity; KISS was the one place that still computed twiddles in
  double (at plan time), which no longer exists.
