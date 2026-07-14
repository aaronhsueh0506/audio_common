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

## TODO

- A later stage adds the full vendored-file manifest (path + hash) for provenance/audit.
