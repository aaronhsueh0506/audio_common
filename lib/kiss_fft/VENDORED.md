# Vendored: KISS FFT

- Upstream: https://github.com/mborgerding/kissfft
- Imported commit: **unknown** — not recorded anywhere in this tree (no
  submodule, no import-log entry, no version macro in the vendored headers
  themselves). The per-file SPDX headers below (`SPDX-License-Identifier:
  BSD-3-Clause` + a `kiss_fft_log.h` companion header) are consistent with a
  post-~2020 kissfft snapshot (older kissfft releases used a plain BSD-style
  comment block with no SPDX tag and no separate log header), but that is an
  inference from file shape, not a verified commit hash.
- Import date into this tree: unknown, before 2026-06 (same as this repo's
  first commit — see `lib/ne10/VENDORED.md` for the sibling NE10 import,
  which has the same "unknown, before 2026-06" gap).
- Upstream status: actively maintained (unlike NE10) as of this writing.

## Local modifications: unknown-but-tracked-from-now

There is no historical record (no diff, no patch log, no prior VENDORED.md)
of whether any of these files were hand-edited after import. From this point
forward, any local modification to a file under `lib/kiss_fft/` MUST be
recorded here the same way `lib/ne10/VENDORED.md` records NE10's P0001/P0002:
an ID, the summary, the files touched, and the rationale. Nothing has been
added below yet because the audit that produced this file made no source
edits (see the "Vendored-file manifest" section for what a fresh `shasum -a
256` over every file reads today; that's the baseline any future local patch
diffs against).

## Vendored-file manifest (review F15)

Every file under `lib/kiss_fft/`, with its SHA-256 as currently checked in.
Generated with `find lib/kiss_fft -type f | sort | xargs shasum -a 256`.

| File | SHA-256 | Bytes | Build role |
|------|---------|-------|------------|
| `kiss_fft.c` | `84154815c4e734bdc986fae5f2301212b1ed4dd91e005bc88680218d4bc3bae3` | 11970 | **compiled** (`BACKEND=kiss`'s only KISS `.c` source — see the Makefile's `BE_SRCS`) |
| `kiss_fft.h` | `9d6c7ddf520932c0aad669c86c064a24e571e594681be9f1491c14c9c9c75693` | 4330 | header-only, included by `kiss_fft.c` and by `audio_common/src/fft_wrapper.c` |
| `_kiss_fft_guts.h` | `dd1731891d0fff0e297198847de39216c5c8b9677ea4fe53160d1f6c74218d40` | 4763 | header-only, included by `kiss_fft.c` |
| `kiss_fft_log.h` | `f954cb6890ec999f7fbc80cdd1c8c0194bbaeb0f1c27e11bd23450f53871cf8c` | 971 | header-only, included by `_kiss_fft_guts.h` |

### `kiss_fftr.c` / `kiss_fftr.h` were corrupted placeholders — REMOVED

Both files are exactly 14 bytes and their entire content, verbatim, is:

```
404: Not Found
```

Removed from the tree at the same commit that added this manifest: they
were the artifact of a broken fetch, never compiled, never referenced by
any build (the KISS wrapper implements r2c/c2r directly on the complex
`kiss_fft()`).

Byte-identical between the two (same hash). This is almost certainly the
body of an HTTP 404 response saved to disk in place of the real
`kiss_fftr.c`/`kiss_fftr.h` from upstream (kissfft's actual real-only FFT
wrapper, normally hundreds of lines) — i.e. whatever fetched this vendor
drop got a broken URL or a since-moved upstream path for these two files
specifically, and nobody noticed because **the `BACKEND=kiss` Makefile build
never references `kiss_fftr` at all** — `audio_common/src/fft_wrapper.c`
implements real-to-complex FFT directly on top of the plain complex
`kiss_fft()` engine (zero-padding the imaginary input, discarding the
conjugate-symmetric half of the output — see that file's own header
comment), so these two broken files have had zero effect on any build or
test to date. They are flagged here rather than silently left in place:
anyone who tries to `#include "kiss_fftr.h"` expecting the real KISS FFT
real-only API will get nonsense, and anyone re-vendoring/updating KISS FFT
in the future should either fetch real replacements for these two files or
delete them outright (they are currently dead weight either way).

## License

See `LICENSE` in this directory (BSD-3-Clause, Mark Borgerding copyright,
extracted verbatim from this codebase's own vendored `kiss_fft.c` header).
Note the upstream header's "See COPYING file for more information" line:
**no `COPYING` file was ever vendored into this tree** alongside these
sources — another provenance gap from the same unrecorded import as the
missing commit hash above. The full license text is reproduced in `LICENSE`
from the header comment itself, which is self-sufficient (BSD-3-Clause is a
short, complete license; the missing `COPYING` file upstream is understood
to carry the same text, not additional terms), so this is a documentation
gap, not a licensing uncertainty.
