# Vendored: NE10

- Upstream: https://github.com/projectNe10/Ne10
- Imported commit: `1f059a764d0e1bc2481c0055c0e71538470baa83` (2018-11-15, version 0.9.10)
- Import date into this tree: unknown, before 2026-06
- Upstream status: unmaintained since 2018 (no activity since the imported commit)

## Local patches

Zero local patches prior to this entry.

| ID     | Summary                                          | Files touched                                                     | Rationale |
|--------|---------------------------------------------------|---------------------------------------------------------------------|-----------|
| P0001  | External-memory r2c config init                    | `modules/dsp/NE10_rfft_float32.c`, `inc/NE10_dsp.h`                  | The strict caller-pool build (`fft_wrapper_ne10.c`'s static-memory path) must not call `malloc()` anywhere between init and destroy. `ne10_fft_alloc_r2c_float32` did one internal `NE10_MALLOC` with no way to place the R2C/C2R config into caller-owned memory. P0001 splits it into `ne10_fft_r2c_mem_size_float32` (exact size query) + `ne10_fft_init_r2c_float32_ext` (carve + twiddle-generate over caller-supplied memory, never frees it) and rewrites `ne10_fft_alloc_r2c_float32` as a thin `malloc()` + call into the `_ext` function. One twiddle code path -> heap and pool configs are bit-identical by construction. `ne10_fft_destroy_r2c_float32` (bare `free()`) is unchanged; the pool caller never calls it on a pool-placed config. |
| P0002  | Guard `NE10_DSP_RFFT_SCALING` against redefinition | `modules/dsp/NE10_fft.h`                                             | `audio_common/Makefile`'s `BACKEND=ne10` build passes `-DNE10_ENABLE_DSP -DNE10_DSP_RFFT_SCALING` on the command line explicitly (so no build can silently drop the flag), but this header also unconditionally `#define`s the same macro (line ~76), producing a `-Wmacro-redefined` warning on every NE10 TU that transitively includes it (13 occurrences on a clean `BACKEND=ne10` build, one per compiled `.c`/`.cpp`). P0002 wraps the header's `#define` in `#ifndef NE10_DSP_RFFT_SCALING ... #endif` so the command-line definition wins without a second, textually-different (`1` vs empty) redefinition. Purely a warning-hygiene fix -- behavior is unchanged either way (`NE10_DSP_RFFT_SCALING`'s c2r 1/N auto-scaling was already verified correct with the macro defined via either the command line or the header; only the "is it defined at all" state matters to the code that guards on it). No other NE10 source touched. |

## Vendored-file manifest (review F15)

Every file under `lib/ne10/` (excluding this `VENDORED.md` itself), with its
SHA-256 as currently checked in. **These hashes are of the file as it sits in
this tree today** — i.e. they include any local patch listed in the table
above (currently: P0001's edits to `NE10_rfft_float32.c` and `NE10_dsp.h`, and
if a later patch touches other files, those too). This manifest does not
separately carry a "pristine upstream" hash per file; provenance back to the
imported commit is the `Imported commit` line above plus the Local patches
table's `Files touched` column — re-diffing against a fresh checkout of that
commit is how you'd reconstruct the pristine hash for any one file if needed.
Generated with `find lib/ne10 -type f ! -name VENDORED.md | sort | xargs
shasum -a 256`.

Build-membership column legend:
- **compiled** — currently a source file the `BACKEND=ne10` Makefile build
  compiles into `libaudio_common.a` (see the Makefile's `NE10_C_SRCS` /
  `NE10_CXXSRCS`, review F16).
- **header-only** — never compiled itself; pulled in via `#include` by one or
  more of the compiled sources (or by another header in this list).
- **vendored, not compiled** — present in the tree (whole, unmodified copy of
  the upstream import for that file) but excluded from the `BACKEND=ne10`
  build by the F16 source-list trim below; kept for provenance/completeness
  with the upstream import, not because anything in this repo still needs it.

### Compiled into `libaudio_common.a` (BACKEND=ne10)

| File | SHA-256 |
|------|---------|
| `modules/dsp/NE10_fft.c` | `ef665fc9792150d715ae9e897affc3278551f587bb6bdbcefaf5fdf60d0eab51` |
| `modules/dsp/NE10_fft_float32.c` | `5067205fc4ec1fd584caa9d14be40243f19c3e6f6ba88660d335017a87a344ab` |
| `modules/dsp/NE10_fft_int32.c` | `b6f7cd5619b931691a97e81b7e44d1282f7f04265b18e142345492302437af91` |
| `modules/dsp/NE10_fft_generic_float32.c` | `0d2790c16ac26d0efa294ff4c1a01923dd8d48ea5d4c93c4ab59e6bb7101a399` |
| `modules/dsp/NE10_fft_generic_int32.cpp` | `02ba6c951eeafcab58bf1f830511debab3f4fd75cb5be12d8fa35b17847faa46` |
| `modules/dsp/NE10_rfft_float32.c` (carries P0001) | `5bf6d655d5d2d24fd0b7590e43153c5714ed70fd10c534db09097da39f012bfe` |
| `modules/dsp/NE10_rfft_float32.neonintrinsic.c` | `3506f7a84fc512c36960547dc77e9375f5f28267ffed181a94d51686105e8441` |

`NE10_fft_float32.c`/`NE10_fft_int32.c`/`NE10_fft_generic_float32.c`/
`NE10_fft_generic_int32.cpp` are compiled in even though this wrapper never
*calls* their complex-to-complex / generic-length entry points — `NE10_fft.c`
unconditionally references their `_c` alloc functions, so they must link even
though they never execute for this wrapper's inputs; see the Makefile
comment above `NE10_C_SRCS` for the full empirical trail.

### Header-only (included by the compiled sources above, never compiled directly)

| File | SHA-256 |
|------|---------|
| `common/NE10_mask_table.h` | `470074c1e66d3bc081418c7d7ea465ff94902ff0bb78b3d1f08d77f466db7de4` |
| `common/factor.h` | `76ca4ab897e356c018c8a2bbbc22d63bd90219544010e60d547c65855731765c` |
| `common/macros.h` | `7ebd5524636855c348fb30fe5385a4e5ab7315bb84213691b78a138b9c6ca536` |
| `common/versionheader.h` | `26074a49e1fca81f93e3ef08e1c879b0d94f9092da3975ce77bf5f2bc80796c1` |
| `inc/NE10.h` | `c7834fa2f1c588ff9715117ec6bb5ecb24ffa89783034ea6b3e6b222b3ed91d2` |
| `inc/NE10_dsp.h` (carries P0001) | `1f0b2598040a226acdc26a1b79cb7cc0f5fc9ccb10c66bfacfdacd3110751aaf` |
| `inc/NE10_imgproc.h` | `2adb7e64ff127e5dc940231b02664ae12d40c3aa8cfa0bafd10e7abafc4a4f61` |
| `inc/NE10_init.h` | `afc3e6cc252fb7edacf81e18d957ee1fa5cfe412bade9ab2d657e770d5455aed` |
| `inc/NE10_macros.h` | `11efb68c57eeee69cb4094976d2024dd3422c2d6e5e556876515287b49ffac85` |
| `inc/NE10_math.h` | `6d5258171c399a557ea17bdb78719344cf779d3b0faa9646aefbc63e54650019` |
| `inc/NE10_physics.h` | `6fe3a03e24a8237fc0fe115774114a66cc463c082f737a5cc5781a51c6616c67` |
| `inc/NE10_types.h` | `4d6c792e80a24086b4031a6bb815fbc69976931144de544834cd5fc02a0bf8d5` |
| `modules/dsp/NE10_fft.h` | `402d17c8ab79cf2268af8f1180760c08870b4760910d9d19d08358856fe61549` |
| `modules/dsp/NE10_fft.neonintrinsic.h` | `182ad7c8b40d2619982c6d768e7d70a9b8f1f70c340a1330d80c06a12188a4aa` |
| `modules/dsp/NE10_fft_bfly.h` | `30f6ce5e98663d0ef2eb0a8434e90edf03c88e9db5b23626454aeadc71dd1ea0` |
| `modules/dsp/NE10_fft_common_varibles.h` | `f08d3ec2b742b12bf985fd8a6992401b776b3f05051a5b16172eb0d3ca85b761` |
| `modules/dsp/NE10_fft_cplx_ops.h` | `a1acd4afec0e2c9c2082904217cb75cbc509e69e71df6c7723bb2052d81bbec1` |
| `modules/dsp/NE10_fft_debug_macro.h` | `63f502a791b31c79401da731783322e1b9e16b3f8c35c4afb2ca583ac580e777` |
| `modules/dsp/NE10_fft_generic_float32.h` | `851487bd696d41bfcec5b3b499487d0d575d37b349790947d17ed90e386ead5e` |
| `modules/dsp/NE10_fft_generic_int32.h` | `985142603e86dffe4e964cfbaef6315183cc5ba78e2898aea212e9741126b907` |
| `modules/dsp/NE10_fft_generic_int32.neonintrinsic.h` | `e6a12e64d2849035dd25b4084f5f0c0189df20bb01e2b6bbb2aaea06ab83b69a` |

Note: `modules/dsp/NE10_fft.h` is where P0002 (see the Local patches table
above, if present) guards `NE10_DSP_RFFT_SCALING` against redefinition — its
hash here reflects that patch once it lands.

### Vendored, not compiled (F16 source-list trim — present for provenance only)

| File | SHA-256 |
|------|---------|
| `modules/NE10_init.c` | `4b28183c31b35e309aad2e917ac50bb288cfef457fb5983b978272e9e6f7e1c4` |
| `modules/dsp/NE10_fft_float32.neonintrinsic.c` | `05ae7347a0185f0c1f48a059930b80e94ff8c2e8faefdc17309d1b8e0f5d2472` |
| `modules/dsp/NE10_fft_int16.c` | `d9a7ef4e35256894a4ad8d32a3bfc94b6a98ddd5d79479919ea3fdadba0eaed5` |
| `modules/dsp/NE10_fft_int16.neonintrinsic.c` | `5cb8ddaec1b237874ccc75ed17fd8d7df25823dd1a76aee0dc13948f7d8b7f24` |
| `modules/dsp/NE10_fft_int32.neonintrinsic.c` | `91a0dc8459e70bf5c9ec50d0d74bf59f2c671683e9d31b4e5aa614482381f2e7` |
| `modules/dsp/NE10_fft_generic_float32.neonintrinsic.cpp` | `5692555e3f1bbed12534fe0a1e60d61a5ab158c3d4db37ab1958e724e9ce5920` |
| `modules/dsp/NE10_fft_generic_int32.neonintrinsic.cpp` | `324fd03ed39d618a9b90e634eb73e0eb2df66c14dd3798b86d242d346dedea1c` |
| `modules/dsp/NE10_fir.c` | `85ddaace603728075444b57fc04fe28b20e3939c89b6a1efeb7d0074d3935764` |
| `modules/dsp/NE10_fir_init.c` | `17a9590296d844649b4ca88e3b5724076649ef8c973a1451a531f72dc5a255af` |
| `modules/dsp/NE10_iir.c` | `c052cc28578402df409021a19ecb5bab2107fa4b73094a3a51e3fef508d4f3ce` |
| `modules/dsp/NE10_iir_init.c` | `7105bf577992073eb17aaa9cd9522a5b52601baabdc33f34e3ae40ed7133d02b` |
| `modules/dsp/NE10_init_dsp.c` | `58e70cbb71e51f9bf036f10ef589aec19e0de9f5b3028bcfd87f1b22cae827f9` |

Verified excludable by rebuilding `BACKEND=ne10` with each dropped (in
combination) and confirming a clean link plus a full PASS on all four test
targets (`selftest`, `test_pool`, `test_wav`, `test_zero_heap`) — see the
Makefile comment above `NE10_C_SRCS` for the per-file reasoning.

## License

See `LICENSE` in this directory (BSD-3-Clause, ARM copyright, extracted
verbatim from this codebase's own vendored NE10 source headers).
