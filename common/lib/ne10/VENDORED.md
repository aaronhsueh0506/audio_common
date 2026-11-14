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
| P0002  | Guard `NE10_DSP_RFFT_SCALING` against redefinition | `modules/dsp/NE10_fft.h`                                             | `common/Makefile`'s `BACKEND=ne10` build passes `-DNE10_ENABLE_DSP -DNE10_DSP_RFFT_SCALING` on the command line explicitly (so no build can silently drop the flag), but this header also unconditionally `#define`s the same macro (line ~76), producing a `-Wmacro-redefined` warning on every NE10 TU that transitively includes it (13 occurrences on a clean `BACKEND=ne10` build, one per compiled `.c`/`.cpp`). P0002 wraps the header's `#define` in `#ifndef NE10_DSP_RFFT_SCALING ... #endif` so the command-line definition wins without a second, textually-different (`1` vs empty) redefinition. Purely a warning-hygiene fix -- behavior is unchanged either way (`NE10_DSP_RFFT_SCALING`'s c2r 1/N auto-scaling was already verified correct with the macro defined via either the command line or the header; only the "is it defined at all" state matters to the code that guards on it). No other NE10 source touched. |
| P0003  | External-memory size/init boundary hardening | `modules/dsp/NE10_rfft_float32.c`, `inc/NE10_dsp.h`                  | Two boundary holes in P0001's external-memory API: (1) `ne10_fft_r2c_mem_size_float32` computed its byte total in `ne10_uint32_t` (32-bit) arithmetic with no bound on `nfft`, so a huge `nfft` silently wrapped to a small, deceptively "valid" nonzero result instead of failing (measured: `nfft=2^28` wrapped to 1,342,177,968; `nfft=2^30` to 1,073,742,512) -- a caller's own 64-bit checked-size arithmetic downstream (e.g. `fft_wrapper_ne10.c`'s `ck_*` helpers) can only catch an overflow that happens on its side of the call, not one that already happened inside this 32-bit function before the result was ever returned. (2) `ne10_fft_init_r2c_float32_ext` took no `mem_size` at all, so it had no way to confirm a caller-supplied pool was actually big enough for the `nfft` it was asked to initialise -- and its `if (nfft<16) return st;` early-out silently returned a non-NULL config whose `r_factors`/`r_twiddles`/`r_twiddles_neon`/etc. were never populated for `nfft` in `[1, 15]`. P0003 fixes both: `ne10_fft_r2c_mem_size_float32` now rejects (returns 0) any `nfft` outside `[16, 8192]` or not a power of two -- within that whitelist the true byte requirement (~21*8192+688 =~ 173 KB at the top end) cannot overflow 32 bits, eliminating the wraparound structurally rather than by a runtime overflow check on the arithmetic itself; `ne10_fft_init_r2c_float32_ext` gained a `mem_size` parameter, validates `mem != NULL` and `mem_size >= ne10_fft_r2c_mem_size_float32(nfft)` (which also subsumes the range check, since an out-of-range `nfft` makes that call return 0), and returns `NULL` instead of a partially-initialised `st` for any `nfft` it won't fully initialise -- the old `nfft<16` silent-degenerate path is deleted, not just relocated. `ne10_fft_alloc_r2c_float32` (the malloc() wrapper) now passes the `memneeded` it already computed through as the new `mem_size` argument. Byte-neutral for every previously-valid, in-whitelist `nfft` (256/512/1024/8192 measured identical before/after); the common-side callers (`fft_wrapper_ne10.c`'s `fft_get_mem_size`/`fft_create`/`fft_init`) gained the same `[16, 8192]` power-of-two whitelist at their own public entry points, one layer up from this vendored fix. **P0003 amendment (factorization guard)**: `ne10_fft_init_r2c_float32_ext`'s two internal `ne10_factor()` calls (one for `nfft`, one for `nfft/4`) still had a boundary hole of the same shape the rest of P0003 closed -- the first call's return value was discarded outright, and the second's was checked but non-fatal (`return st` on `NE10_ERR`, handing back the same partially-initialised, non-NULL config P0003's `nfft<16` fix had already eliminated one layer up). Both calls are now checked, and either returning `NE10_ERR` now returns `NULL` instead of `st`, restoring the init-fully-or-NULL contract for this path too. Defensively unreachable given the `[16, 8192]` power-of-two whitelist already enforced above this point in the function (`ne10_factor` only fails on a NULL buffer, `n<=0`, or `stage_num>21` -- none reachable for `nfft` or `nfft/4` anywhere in that whitelist, so every in-range nfft still factors cleanly and this is byte-neutral for all of them) -- kept as a real check rather than an assert so the contract holds even if the whitelist bounds ever change. |
| P0004 | Gate optional inline assembly by ISA | `modules/dsp/NE10_rfft_float32.neonintrinsic.c`, `modules/dsp/NE10_fft_generic_float32.neonintrinsic.cpp`, `modules/dsp/NE10_fft_generic_int32.neonintrinsic.h` | When `NE10_INLINE_ASM_OPT` is defined, use its AArch64 assembly only on AArch64; ARM32 uses the existing intrinsic fallback. The normal AArch64 branch and arithmetic are unchanged. Common pins `NE10_UNROLL_LEVEL=1` on both ABIs because its vendored RFFT/ext-memory API uses that config layout. |
| P0005 | Comment-only text updates (commit 2ac9971) | `modules/dsp/NE10_fft.c`, `modules/dsp/NE10_fft_float32.c`, `modules/dsp/NE10_fft_int32.c`, `modules/dsp/NE10_fft_generic_float32.c`, `modules/dsp/NE10_rfft_float32.c`, `modules/dsp/NE10_fft_cplx_ops.h`, `modules/dsp/NE10_fft_generic_float32.h`, `modules/dsp/NE10_fft_generic_int32.h`, `modules/dsp/NE10_fft_int16.c`, `modules/dsp/NE10_fft.h` | In the nine `.c`/`.h` files other than `NE10_fft.h`, the license-block comment `/* license of Kiss FFT */` became `/* Additional copyright notice. */` (one line per file); in `NE10_fft.h` the comment that names the Makefile became `(common/Makefile)` instead of `(audio_common/Makefile)`. Comments only -- no behavior difference. These edits are why the manifest hashes of these ten files differ from the pre-2ac9971 values. |

## Vendored-file manifest (review F15)

Every file under `lib/ne10/` (excluding this `VENDORED.md` itself and
`LICENSE`, see the License section), with its SHA-256 as currently checked in. **These hashes are of the file as it sits in
this tree today** — i.e. they include any local patch listed in the table
above, including the ISA guards in P0004 and the comment edits in P0005. This manifest does not
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
- **vendored, not compiled** — present in the tree (the upstream import for
  that file, plus any local patch listed in the Local patches table, e.g. the
  P0004 ISA guards and the P0005 comment edits; not necessarily byte-identical
  to upstream) but excluded from the `BACKEND=ne10` build by the F16
  source-list trim below, or a header that no compiled source includes; kept
  for provenance/completeness with the upstream import, not because anything in
  this repo still needs it.

### Compiled into `libaudio_common.a` (BACKEND=ne10)

`NE10_rfft_float32.neonintrinsic.c` is included only when the compiler
target defines `__ARM_NEON`; non-NEON ARM targets use the scalar RFFT TU.

| File | SHA-256 |
|------|---------|
| `modules/dsp/NE10_fft.c` | `0981262ea84ab6abc54fdde230cdb919c59fcee14bd7dd439aa51b3a34868ec4` |
| `modules/dsp/NE10_fft_float32.c` | `9e255cd5dae314e2ebd1a2c3e7db04ab6778a4fbfcb33b6d8a757dbb99365ab9` |
| `modules/dsp/NE10_fft_int32.c` | `2e5c66369be22d9b7805e863e7f473c9f0a498bf571e367c5d2373348a8e077b` |
| `modules/dsp/NE10_fft_generic_float32.c` | `cd4db16c9da4dc5493f7e80381247be444f115e522611b60d4b45d9a48ad62ab` |
| `modules/dsp/NE10_fft_generic_int32.cpp` | `02ba6c951eeafcab58bf1f830511debab3f4fd75cb5be12d8fa35b17847faa46` |
| `modules/dsp/NE10_rfft_float32.c` (carries P0001, P0003 + amendment) | `fc39afda03362addc1d3af597f8aae6fc396146a69375bf9680f48cea28d8a49` |
| `modules/dsp/NE10_rfft_float32.neonintrinsic.c` | `736fa5288348f122ba28580f1ca7b3389cad0f136da6b80f4a39cdfe10ca2538` |

`NE10_fft_float32.c`/`NE10_fft_int32.c`/`NE10_fft_generic_float32.c`/
`NE10_fft_generic_int32.cpp` are compiled in even though this wrapper never
*calls* their complex-to-complex / generic-length entry points — `NE10_fft.c`
unconditionally references their `_c` alloc functions, so they must link even
though they never execute for this wrapper's inputs; see the Makefile
comment above `NE10_C_SRCS` for the full empirical trail.

### Header-only (included by the compiled sources above, never compiled directly)

| File | SHA-256 |
|------|---------|
| `inc/NE10.h` | `c7834fa2f1c588ff9715117ec6bb5ecb24ffa89783034ea6b3e6b222b3ed91d2` |
| `inc/NE10_dsp.h` (carries P0001, P0003) | `076ec0136cb0b749ca080cf5182f188e6ac9ec617f239ac967fc5a3ebd1cef85` |
| `inc/NE10_imgproc.h` | `2adb7e64ff127e5dc940231b02664ae12d40c3aa8cfa0bafd10e7abafc4a4f61` |
| `inc/NE10_init.h` | `afc3e6cc252fb7edacf81e18d957ee1fa5cfe412bade9ab2d657e770d5455aed` |
| `inc/NE10_macros.h` | `11efb68c57eeee69cb4094976d2024dd3422c2d6e5e556876515287b49ffac85` |
| `inc/NE10_math.h` | `6d5258171c399a557ea17bdb78719344cf779d3b0faa9646aefbc63e54650019` |
| `inc/NE10_physics.h` | `6fe3a03e24a8237fc0fe115774114a66cc463c082f737a5cc5781a51c6616c67` |
| `inc/NE10_types.h` | `4d6c792e80a24086b4031a6bb815fbc69976931144de544834cd5fc02a0bf8d5` |
| `modules/dsp/NE10_fft.h` | `55b9f517048508e9364f117268217df2b4a820de2f4f9c0b1f50f67dc9071183` |
| `modules/dsp/NE10_fft.neonintrinsic.h` | `182ad7c8b40d2619982c6d768e7d70a9b8f1f70c340a1330d80c06a12188a4aa` |
| `modules/dsp/NE10_fft_bfly.h` | `30f6ce5e98663d0ef2eb0a8434e90edf03c88e9db5b23626454aeadc71dd1ea0` |
| `modules/dsp/NE10_fft_common_varibles.h` | `f08d3ec2b742b12bf985fd8a6992401b776b3f05051a5b16172eb0d3ca85b761` |
| `modules/dsp/NE10_fft_cplx_ops.h` | `0428247c52c76975f147d740f42b95f0ce7332f803bc9369aa3a5afc80fece1a` |
| `modules/dsp/NE10_fft_debug_macro.h` | `63f502a791b31c79401da731783322e1b9e16b3f8c35c4afb2ca583ac580e777` |
| `modules/dsp/NE10_fft_generic_float32.h` | `2c8b2d4d568e0ac09af882f3a0a58269cc639e25ae7af46bbbc952345807ebb0` |
| `modules/dsp/NE10_fft_generic_int32.h` | `5108f6d4d1a2eff2fb4b275271be9b7d868793c5d48ddce07860d99b5e429308` |

Note: `modules/dsp/NE10_fft.h` is where P0002 (see the Local patches table
above, if present) guards `NE10_DSP_RFFT_SCALING` against redefinition — its
hash here reflects that patch once it lands.

### Vendored, not compiled (F16 source-list trim — present for provenance only)

| File | SHA-256 |
|------|---------|
| `modules/NE10_init.c` | `4b28183c31b35e309aad2e917ac50bb288cfef457fb5983b978272e9e6f7e1c4` |
| `modules/dsp/NE10_fft_float32.neonintrinsic.c` | `05ae7347a0185f0c1f48a059930b80e94ff8c2e8faefdc17309d1b8e0f5d2472` |
| `modules/dsp/NE10_fft_int16.c` | `07cb39edcda832799fed2da393588d4d17ea9a3975eb3f466abaadec9f99d548` |
| `modules/dsp/NE10_fft_int16.neonintrinsic.c` | `5cb8ddaec1b237874ccc75ed17fd8d7df25823dd1a76aee0dc13948f7d8b7f24` |
| `modules/dsp/NE10_fft_int32.neonintrinsic.c` | `91a0dc8459e70bf5c9ec50d0d74bf59f2c671683e9d31b4e5aa614482381f2e7` |
| `modules/dsp/NE10_fft_generic_float32.neonintrinsic.cpp` | `e2c2bd077f488b664a7566353ed7472dd1a5dae1c5f2fab065ac84ed329c0ad5` |
| `modules/dsp/NE10_fft_generic_int32.neonintrinsic.cpp` | `324fd03ed39d618a9b90e634eb73e0eb2df66c14dd3798b86d242d346dedea1c` |
| `modules/dsp/NE10_fir.c` | `85ddaace603728075444b57fc04fe28b20e3939c89b6a1efeb7d0074d3935764` |
| `modules/dsp/NE10_fir_init.c` | `17a9590296d844649b4ca88e3b5724076649ef8c973a1451a531f72dc5a255af` |
| `modules/dsp/NE10_iir.c` | `c052cc28578402df409021a19ecb5bab2107fa4b73094a3a51e3fef508d4f3ce` |
| `modules/dsp/NE10_iir_init.c` | `7105bf577992073eb17aaa9cd9522a5b52601baabdc33f34e3ae40ed7133d02b` |
| `common/NE10_mask_table.h` | `470074c1e66d3bc081418c7d7ea465ff94902ff0bb78b3d1f08d77f466db7de4` |
| `common/factor.h` | `76ca4ab897e356c018c8a2bbbc22d63bd90219544010e60d547c65855731765c` |
| `common/macros.h` | `7ebd5524636855c348fb30fe5385a4e5ab7315bb84213691b78a138b9c6ca536` |
| `common/versionheader.h` | `26074a49e1fca81f93e3ef08e1c879b0d94f9092da3975ce77bf5f2bc80796c1` |
| `modules/dsp/NE10_fft_generic_int32.neonintrinsic.h` | `cfdebbd98622230f947792c3a3ed8114e491b85a2afde70a567dcccdeb106c9f` |
| `modules/dsp/NE10_init_dsp.c` | `58e70cbb71e51f9bf036f10ef589aec19e0de9f5b3028bcfd87f1b22cae827f9` |

Verified excludable by rebuilding `BACKEND=ne10` with each dropped (in
combination) and confirming a clean link plus a full PASS on all four test
targets (`selftest`, `test_pool`, `test_wav`, `test_zero_heap`) — see the
Makefile comment above `NE10_C_SRCS` for the per-file reasoning.

Five of the rows above are headers rather than sources, and no compiled source includes them (checked against the
compiler dependency files of the `SIMD=1` and `SIMD=0` builds): `common/NE10_mask_table.h`, `common/factor.h`,
`common/macros.h`, `common/versionheader.h` and `modules/dsp/NE10_fft_generic_int32.neonintrinsic.h`.
`common/factor.h` is reached only through `common/macros.h`, which nothing includes; `common/NE10_mask_table.h` and
`common/versionheader.h` have no includer; the `neonintrinsic.h` is included only by the uncompiled
`NE10_fft_generic_int32.neonintrinsic.cpp`.

## License

See `LICENSE` in this directory (BSD-3-Clause, ARM copyright, extracted
verbatim from this codebase's own vendored NE10 source headers).
