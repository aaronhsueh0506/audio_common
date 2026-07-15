/**
 * wav_io.h - Hardened, shared WAV file I/O (F06 remediation)
 *
 * THE single canonical implementation. Before this fix, AEC/c_impl/example
 * and NR/c_impl/example each carried their own byte-identical-reader copy
 * (plus two more Audio_ALG submodule mirrors of those). Both readers shared
 * the same unsafe parsing:
 *   (1) the fmt chunk was blindly read as 16 fixed bytes even when its
 *       declared chunk_sz was smaller (desyncing the rest of the parse);
 *   (2) audio_format/channels/bits_per_sample were never validated --
 *       bits_per_sample==0 or channels==0 reached a divide in the
 *       data-chunk sample-count arithmetic;
 *   (3) an odd-sized chunk's RIFF even-alignment pad byte was never
 *       skipped, silently losing sync with every chunk after it;
 *   (4) a chunk's declared chunk_sz was never bounds-checked against how
 *       many bytes were actually left in the file, so a hostile/corrupt
 *       header could fseek past EOF or claim a data chunk far larger than
 *       the file itself, and num_samples (an int) could silently wrap for
 *       a ~4 GB+ data chunk.
 * This file fixes all four, in one reviewed place, for every consumer.
 *
 * Reader hardening (wav_open_read):
 *   - The file size is queried once via fseek(SEEK_END)/ftell right after
 *     open (wav_io_file_size), and every chunk_sz seen afterward (fmt
 *     search, data search, and the fmt chunk's own "extra format bytes"
 *     tail) is bounds-checked against it before being trusted to seek or
 *     size anything (wav_io_skip_chunk / the inline checks in the fmt and
 *     data branches). A chunk that claims to reach past the end of the
 *     file is rejected (NULL), never silently truncated or seeked past.
 *   - The fmt chunk must declare chunk_sz >= 16 (a malformed/truncated fmt
 *     chunk is rejected, instead of reading 16 fixed bytes off a shorter
 *     declared chunk).
 *   - Exactly two (audio_format, bits_per_sample) combinations are
 *     accepted: (1, 16) = PCM16, or (3, 32) = IEEE float32. Every other
 *     combination (bits==0/1/7/24, 32-bit int PCM, ADPCM/A-law/mu-law/
 *     WAVE_FORMAT_EXTENSIBLE, etc.) is rejected at open time. This is also
 *     what eliminates the old bytes_per_sample==0 divide-by-zero (bits==0)
 *     and the channels==0 divide-by-zero (channels is separately bounded
 *     below) in the data-chunk sample-count arithmetic.
 *   - channels must be in [1, 8]; sample_rate must be in [1, 384000].
 *   - block_align, when nonzero in the file, must equal
 *     channels*bytes_per_sample; byte_rate, when nonzero, must equal
 *     sample_rate*that canonical block_align. A conforming writer gets
 *     both of these right for free, so a mismatch is treated as a
 *     corrupt/hostile header (rejected), not a benign quirk to warn about
 *     and continue past.
 *   - every chunk skip (both "find fmt" and "find data" search loops, plus
 *     the fmt chunk's own trailing "extra format bytes" region) now
 *     advances one extra byte when the chunk's declared size is odd --
 *     the RIFF even-alignment pad the old parser never accounted for.
 *   - num_samples is computed in uint64_t (chunk_sz / bytes_per_sample /
 *     channels) and rejected if it would not fit in the (int)
 *     WavInfo.num_samples field, instead of silently wrapping.
 *   - every new AND pre-existing fseek/fread is checked; any failure
 *     takes the existing goto-error path (fclose the file, free the
 *     reader, return NULL) -- never a crash, never partial state.
 *
 * Float ingress sanitize (wav_read_float): a float32 (audio_format==3)
 * source sample that is NaN or +/-Inf is replaced with 0.0f, and the
 * WavReader's `nonfinite_sanitized` counter is incremented, so a caller
 * can detect and log that a file needed cleanup. This is a SANITIZE, not
 * a reject: unlike a malformed header (cheap for any conforming writer to
 * get right, so a mismatch is suspicious), a non-finite sample is a normal
 * hazard of upstream float32 processing (e.g. a filter under test
 * diverging) and failing an entire read over one bad sample would be
 * worse than the hazard itself. The PCM16 conversion arithmetic
 * (`(float)sample / 32768.0f`) is untouched, character-for-character
 * identical to the pre-hardening version -- integer samples can never be
 * non-finite, so no sanitize logic applies on that path.
 *
 * Writer: ONE implementation. Both repos' pre-existing, bit-for-bit
 * different output behaviors are preserved via the WAV_IO_WRITER_STYLE
 * compile-time knob (each repo's thin example/wav_io.h shim sets it
 * before including this file -- see AEC's / NR's copies):
 *   WAV_IO_WRITER_AEC (default when unset) - PCM16 by default;
 *       AEC_OUT_FLOAT=1 env var switches to raw, unquantized IEEE
 *       float32 output (a test-only path for C-vs-Python correlation);
 *       PCM16 quantization is round-half-away-from-zero
 *       (`sample*32768.0f`, then +-0.5f before truncating), saturated
 *       into int16_t range (see R01 fix below).
 *   WAV_IO_WRITER_NR - always PCM16, `sample*32767.0f` truncation (no
 *       rounding, no float32 path) -- NR's historical behavior, also
 *       saturated into int16_t range.
 *
 * Writer hardening (external re-review finding R01, PCM16 quantization +
 * RIFF-size arithmetic -- fixed in place, not preserved as a "quirk"):
 *   - Undefined behavior at the +-1.0f boundary: the pre-fix AEC-style
 *     quantizer cast `scaled +- 0.5f` directly to int16_t; for
 *     sample==+1.0f that's (int16_t)32768.5f, an out-of-range
 *     float-to-integer conversion -- undefined behavior per C99 6.3.1.4p1,
 *     reproduced under UBSan. Fixed by rounding into a wider int32_t
 *     accumulator first, THEN saturating into [-32768, 32767] before the
 *     final int16_t narrowing. The float rounding arithmetic itself is
 *     unchanged (still `sample*32768.0f`, then +-0.5f before truncating):
 *     verified bit-identical over this repo's regression corpus and its
 *     byte-exact anchor renders (re-review round-3, B08). That is NOT the
 *     same claim as "bit-identical for every |sample| < 1 input" -- a value
 *     arbitrarily close to +-1.0f (e.g. nextafterf(1.0f, 0.0f), |sample| <
 *     1 by definition) can still push the `scaled +- 0.5f` accumulator past
 *     +-32768 before saturation, and pre-fix that was undefined behavior
 *     too (the same out-of-range float-to-int16_t cast as the exact +-1.0f
 *     case) -- so there was never a defined pre-fix byte value at that
 *     boundary for this fix to have stayed identical to; no byte contract
 *     existed there. Post-fix, that boundary region is fully defined for
 *     the first time (saturates to +-32767/+-32768 like any other
 *     out-of-range magnitude, +1.0f -> 32767, -1.0f -> -32768, matching
 *     what a defined round would give anyway) instead of invoking UB. The
 *     NR-style quantizer gets the same int32_t-then-saturate treatment for
 *     consistency, though it's pure defense there -- the pre-existing
 *     [-1,1] clamp plus the non-finite sanitize below already bound its
 *     input, so its output can never actually need saturating.
 *   - Non-finite (NaN/+-Inf) PCM16-path source samples: previously reached
 *     the quantizer's ordered comparisons unchanged (NaN fails every
 *     ordered comparison, so the [-1,1] clamp was a silent no-op for it)
 *     and then hit the same float-to-int UB as the +-1.0f case. Now
 *     sanitized to 0.0f before quantization, mirroring wav_read_float's
 *     ingress sanitize (see WavWriter.nonfinite_sanitized above). The
 *     float-mode write path is deliberately UNCHANGED -- still a raw,
 *     unsanitized bit-exact passthrough (see wav_write_float): that path
 *     exists specifically for C-vs-Python numerical correlation, where
 *     silently zeroing a NaN/Inf would hide exactly the divergence it's
 *     meant to surface; wav_read_float's own ingress sanitize is the
 *     defense for a non-finite sample actually reaching a consumer.
 *   - RIFF outer chunk_size: previously computed from a hardcoded 2
 *     bytes/sample regardless of writer mode, silently under-reporting the
 *     outer RIFF size specifically for AEC's float32 output (4
 *     bytes/sample) -- the "data" sub-chunk's OWN size field was always
 *     correct (it already used sample_bytes). Both the outer RIFF
 *     chunk_size and the data sub-chunk size are now computed from the
 *     same sample_bytes-derived value. PCM16 output is a no-op change (2
 *     == sample_bytes there, both before and after this fix); AEC's
 *     float32 output path is the only case where the on-disk header bytes
 *     differ from a pre-fix build. Any byte-exact regression golden built
 *     from AEC_OUT_FLOAT=1 output predating this fix needs re-baking
 *     against the corrected header.
 *   - samples_written widened int -> uint64_t: a long capture could
 *     overflow a signed int counter before ever reaching the close-time
 *     uint64_t size arithmetic below. A short fwrite() inside
 *     wav_write_float now also sets WavWriter.write_error instead of
 *     silently under-counting.
 * wav_close_write's header size fields (RIFF chunk_size, byte_rate,
 * block_align, data-chunk size) are computed in uint64_t and range
 * checked before being narrowed to their on-disk uint32_t/uint16_t
 * fields; a size that would overflow abandons the file (fclose + free)
 * instead of writing a silently-truncated header.
 *
 * gnu99 (not strict C99): both consumer Makefiles already build with
 * -std=gnu99, so this file may use GNU/POSIX extensions if ever needed;
 * today it only relies on __has_include, a preprocessor feature (not a
 * language extension) supported by every compiler either consumer repo
 * targets.
 */
#ifndef WAV_IO_H
#define WAV_IO_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

/* Writer-style selector. A consumer's thin shim (AEC's / NR's
 * example/wav_io.h) defines WAV_IO_WRITER_STYLE to one of these BEFORE
 * including this file; if left unset (e.g. this header included directly,
 * as audio_common's own test does), WAV_IO_WRITER_AEC is the default --
 * matching how this file behaved when it only lived in AEC's tree. */
#define WAV_IO_WRITER_AEC 1
#define WAV_IO_WRITER_NR  2
#ifndef WAV_IO_WRITER_STYLE
#define WAV_IO_WRITER_STYLE WAV_IO_WRITER_AEC
#endif

// WAV file information
typedef struct {
    int sample_rate;
    int channels;
    int bits_per_sample;
    int num_samples;
    int is_float;
} WavInfo;

// WAV reader handle
typedef struct {
    FILE* fp;
    WavInfo info;
    long data_start;
    int samples_read;
    /* Count of float32 source samples that were NaN/+-Inf and were
     * replaced with 0.0f by wav_read_float (see file header doc). Always
     * 0 for a PCM16 source -- integer samples can't be non-finite. */
    uint64_t nonfinite_sanitized;
} WavReader;

// WAV writer handle
typedef struct {
    FILE* fp;
    WavInfo info;
    long data_start;
    uint64_t samples_written;
    /* Count of PCM16-path source samples that were NaN/+-Inf and were
     * replaced with 0.0f before int16 quantization (mirrors WavReader's
     * nonfinite_sanitized on the read side -- see the Writer hardening doc
     * above). The FLOAT-mode write path (WAV_IO_WRITER_AEC +
     * AEC_OUT_FLOAT=1) does NOT sanitize -- it is a raw bit-exact
     * passthrough for C-vs-Python correlation tooling, and wav_read_float's
     * ingress sanitize is the actual defense for that path. Always 0 in
     * float mode. */
    uint64_t nonfinite_sanitized;
    /* Set to 1 the first time an fwrite() inside wav_write_float writes
     * fewer elements than requested (e.g. ENOSPC). samples_written is not
     * incremented for that short write (it already reflects exactly what
     * made it to disk so far), and no further samples from that same
     * wav_write_float() call are attempted. wav_close_write()/
     * wav_finalize_write() still finalize a header from the true
     * samples_written value; this flag exists purely so a caller that cares
     * can notice the failure -- either directly (struct-field read, same
     * convention as nonfinite_sanitized) or, since B08, folded automatically
     * into wav_finalize_write()'s return value. */
    int write_error;
} WavWriter;

// ============================================================================
// WAV Reader
// ============================================================================

/* Returns the total size of an already-open file without disturbing the
 * caller's current file position, or -1 on error. Called once at the top
 * of wav_open_read so every chunk_sz encountered afterward can be
 * bounds-checked against "how many bytes are actually left in the file"
 * instead of being trusted at face value. */
static inline long wav_io_file_size(FILE* fp) {
    long cur = ftell(fp);
    if (cur < 0) return -1;
    if (fseek(fp, 0, SEEK_END) != 0) return -1;
    long sz = ftell(fp);
    if (fseek(fp, cur, SEEK_SET) != 0) return -1;
    return sz;
}

/* Skips exactly chunk_sz bytes of a sub-chunk's data, PLUS the RIFF
 * even-alignment pad byte if chunk_sz is odd (the byte every pre-hardening
 * copy of this parser silently forgot). Rejects -- returns 0, seeks
 * nothing -- if doing so would move past the end of the file. Must be
 * called with fp positioned exactly at the first byte of the chunk's
 * declared data (i.e. immediately after its 4-byte id + 4-byte size). */
static inline int wav_io_skip_chunk(FILE* fp, long file_size, uint32_t chunk_sz) {
    long cur = ftell(fp);
    if (cur < 0 || file_size < 0) return 0;
    uint32_t pad = (chunk_sz & 1u) ? 1u : 0u;
    uint64_t end64 = (uint64_t)cur + (uint64_t)chunk_sz + (uint64_t)pad;
    if (end64 > (uint64_t)file_size) return 0;
    if (chunk_sz != 0u || pad != 0u) {
        if (fseek(fp, (long)chunk_sz + (long)pad, SEEK_CUR) != 0) return 0;
    }
    return 1;
}

/**
 * Open WAV file for reading
 *
 * @param path Path to WAV file
 * @return Reader handle, or NULL on error
 */
static inline WavReader* wav_open_read(const char* path) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return NULL;

    long file_size = wav_io_file_size(fp);
    if (file_size < 0) { fclose(fp); return NULL; }

    WavReader* r = (WavReader*)calloc(1, sizeof(WavReader));
    if (!r) {
        fclose(fp);
        return NULL;
    }
    r->fp = fp;

    // Read RIFF header
    char riff[4], wave[4];
    uint32_t chunk_size;

    if (fread(riff, 1, 4, fp) != 4) goto error;
    if (fread(&chunk_size, 4, 1, fp) != 1) goto error;
    if (fread(wave, 1, 4, fp) != 4) goto error;

    if (memcmp(riff, "RIFF", 4) != 0 || memcmp(wave, "WAVE", 4) != 0) {
        goto error;
    }

    int bytes_per_sample = 0;  // set once the fmt chunk validates below

    // Find fmt chunk
    while (1) {
        char chunk_id[4];
        uint32_t chunk_sz;
        if (fread(chunk_id, 1, 4, fp) != 4) goto error;
        if (fread(&chunk_sz, 4, 1, fp) != 1) goto error;

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            if (chunk_sz < 16) goto error;  // malformed/truncated fmt chunk

            long fmt_data_pos = ftell(fp);
            if (fmt_data_pos < 0) goto error;
            {
                uint32_t pad = (chunk_sz & 1u) ? 1u : 0u;
                uint64_t end64 = (uint64_t)fmt_data_pos + (uint64_t)chunk_sz + (uint64_t)pad;
                if (end64 > (uint64_t)file_size) goto error;
            }

            uint16_t audio_format, num_channels, block_align, bits_per_sample;
            uint32_t sample_rate, byte_rate;

            if (fread(&audio_format, 2, 1, fp) != 1) goto error;
            if (fread(&num_channels, 2, 1, fp) != 1) goto error;
            if (fread(&sample_rate, 4, 1, fp) != 1) goto error;
            if (fread(&byte_rate, 4, 1, fp) != 1) goto error;
            if (fread(&block_align, 2, 1, fp) != 1) goto error;
            if (fread(&bits_per_sample, 2, 1, fp) != 1) goto error;

            // Accept ONLY PCM16 or IEEE float32 (see file header doc). This
            // also rules out bits_per_sample==0 (the old divide-by-zero).
            int is_float;
            if (audio_format == 1 && bits_per_sample == 16) {
                is_float = 0;
            } else if (audio_format == 3 && bits_per_sample == 32) {
                is_float = 1;
            } else {
                goto error;
            }

            if (num_channels < 1 || num_channels > 8) goto error;
            if (sample_rate < 1 || sample_rate > 384000u) goto error;

            bytes_per_sample = bits_per_sample / 8;  // exact: 2 or 4
            uint32_t canonical_block_align = (uint32_t)num_channels * (uint32_t)bytes_per_sample;
            if (block_align != 0 && (uint32_t)block_align != canonical_block_align) goto error;
            if (byte_rate != 0 && byte_rate != sample_rate * canonical_block_align) goto error;

            r->info.sample_rate = (int)sample_rate;
            r->info.channels = (int)num_channels;
            r->info.bits_per_sample = (int)bits_per_sample;
            r->info.is_float = is_float;

            // Consume any extra format bytes beyond the 16 fixed fields,
            // plus the RIFF pad byte if chunk_sz is odd. Bounds already
            // verified above.
            {
                uint32_t remaining = chunk_sz - 16;
                uint32_t pad = (chunk_sz & 1u) ? 1u : 0u;
                if (remaining != 0u || pad != 0u) {
                    if (fseek(fp, (long)remaining + (long)pad, SEEK_CUR) != 0) goto error;
                }
            }
            break;
        } else {
            if (!wav_io_skip_chunk(fp, file_size, chunk_sz)) goto error;
        }
    }

    // Find data chunk
    while (1) {
        char chunk_id[4];
        uint32_t chunk_sz;
        if (fread(chunk_id, 1, 4, fp) != 4) goto error;
        if (fread(&chunk_sz, 4, 1, fp) != 1) goto error;

        if (memcmp(chunk_id, "data", 4) == 0) {
            long data_pos = ftell(fp);
            if (data_pos < 0) goto error;
            if ((uint64_t)data_pos + (uint64_t)chunk_sz > (uint64_t)file_size) goto error;

            uint64_t num_samples64 = (uint64_t)chunk_sz / (uint64_t)bytes_per_sample / (uint64_t)r->info.channels;
            if (num_samples64 > (uint64_t)INT_MAX) goto error;

            r->info.num_samples = (int)num_samples64;
            r->data_start = data_pos;
            break;
        } else {
            if (!wav_io_skip_chunk(fp, file_size, chunk_sz)) goto error;
        }
    }

    return r;

error:
    fclose(fp);
    free(r);
    return NULL;
}

/**
 * Read samples as float (mono, first channel only)
 *
 * @param r Reader handle
 * @param buf Output buffer
 * @param n Number of samples to read
 * @return Number of samples actually read
 */
static inline int wav_read_float(WavReader* r, float* buf, int n) {
    if (!r || !buf) return 0;

    int read_count = 0;
    int channels = r->info.channels;
    int bits = r->info.bits_per_sample;

    for (int i = 0; i < n && r->samples_read < r->info.num_samples; i++) {
        if (bits == 16) {
            int16_t sample;
            if (fread(&sample, 2, 1, r->fp) != 1) break;
            buf[i] = (float)sample / 32768.0f;
            // Skip other channels
            if (channels > 1) {
                fseek(r->fp, 2 * (channels - 1), SEEK_CUR);
            }
        } else if (bits == 32 && r->info.is_float) {
            float sample;
            if (fread(&sample, 4, 1, r->fp) != 1) break;
            if (!isfinite(sample)) {
                sample = 0.0f;
                r->nonfinite_sanitized++;
            }
            buf[i] = sample;
            if (channels > 1) {
                fseek(r->fp, 4 * (channels - 1), SEEK_CUR);
            }
        } else if (bits == 32) {
            // 32-bit integer PCM: unreachable via the hardened
            // wav_open_read above (only PCM16 / float32 are accepted at
            // open time). Kept only as a defensive fallback for a caller
            // that hand-builds a WavReader/WavInfo outside wav_open_read.
            // Integer samples can never be non-finite -- no sanitize.
            int32_t sample;
            if (fread(&sample, 4, 1, r->fp) != 1) break;
            buf[i] = (float)sample / 2147483648.0f;
            if (channels > 1) {
                fseek(r->fp, 4 * (channels - 1), SEEK_CUR);
            }
        } else {
            break;  // Unsupported format
        }
        r->samples_read++;
        read_count++;
    }

    return read_count;
}

/**
 * Close WAV reader
 */
static inline void wav_close_read(WavReader* r) {
    if (r) {
        if (r->fp) fclose(r->fp);
        free(r);
    }
}

// ============================================================================
// WAV Writer
// ============================================================================

/**
 * Open WAV file for writing (16-bit PCM mono, or AEC-style float32 -- see
 * WAV_IO_WRITER_STYLE in the file header doc)
 *
 * @param path Path to output WAV file
 * @param sample_rate Sample rate
 * @param channels Number of channels (usually 1)
 * @return Writer handle, or NULL on error
 */
static inline WavWriter* wav_open_write(const char* path, int sample_rate, int channels) {
    FILE* fp = fopen(path, "wb");
    if (!fp) return NULL;

    WavWriter* w = (WavWriter*)calloc(1, sizeof(WavWriter));
    if (!w) {
        fclose(fp);
        return NULL;
    }
    w->fp = fp;
    w->info.sample_rate = sample_rate;
    w->info.channels = channels;
#if WAV_IO_WRITER_STYLE == WAV_IO_WRITER_AEC
    /* Default PCM16. Set AEC_OUT_FLOAT=1 to write IEEE float32 (for testing) */
    const char* float_env = getenv("AEC_OUT_FLOAT");
    w->info.is_float = (float_env && float_env[0] == '1') ? 1 : 0;
    w->info.bits_per_sample = w->info.is_float ? 32 : 16;
#else
    w->info.is_float = 0;
    w->info.bits_per_sample = 16;
#endif

    // Write placeholder header (will update at close). Checked like every
    // other open-time failure in this function (B08): a short write here
    // (e.g. ENOSPC on a full disk) would otherwise leave `w` claiming
    // data_start==44 over a file that doesn't actually have 44 header bytes
    // on disk yet, silently desyncing every wav_write_float() call after it.
    uint8_t header[44] = {0};
    if (fwrite(header, 1, 44, fp) != 44) {
        fclose(fp);
        free(w);
        return NULL;
    }
    w->data_start = 44;

    return w;
}

/**
 * Write samples from float buffer
 *
 * @param w Writer handle
 * @param buf Input buffer (float samples)
 * @param n Number of samples to write
 */
static inline void wav_write_float(WavWriter* w, const float* buf, int n) {
    if (!w || !buf) return;
    // Guards the float-mode fwrite() below: a negative n would convert to
    // a huge size_t element count when passed as fwrite's nmemb argument.
    if (n <= 0) return;

#if WAV_IO_WRITER_STYLE == WAV_IO_WRITER_AEC
    if (w->info.is_float) {
        /* IEEE float32 (no quantization, for C-vs-Python correlation
         * tests). Deliberately raw bit-exact passthrough, including any
         * NaN/Inf payload -- see the Writer hardening doc in the file
         * header comment for why this path does NOT sanitize. */
        size_t written = fwrite(buf, sizeof(float), (size_t)n, w->fp);
        w->samples_written += written;
        if (written != (size_t)n) w->write_error = 1;
        return;
    }
#endif

    for (int i = 0; i < n; i++) {
        float sample = buf[i];

        // Sanitize policy mirrors wav_read_float's ingress sanitize (see
        // WavWriter.nonfinite_sanitized / file header doc): a non-finite
        // PCM16-path sample is replaced with 0.0f before it can reach the
        // int16 conversion below, where a NaN/Inf would otherwise silently
        // pass the ordered [-1,1] clamp comparisons and hit the same
        // undefined float-to-int conversion as the +-1.0f boundary case.
        if (!isfinite(sample)) {
            sample = 0.0f;
            w->nonfinite_sanitized++;
        }

        // Clamp to [-1, 1]
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;

#if WAV_IO_WRITER_STYLE == WAV_IO_WRITER_AEC
        // Round-half-away-from-zero into a wider (int32_t) accumulator,
        // THEN saturate into int16_t's representable range. The float
        // rounding arithmetic itself is untouched (still `sample*32768.0f`,
        // still +-0.5f before truncating) -- verified bit-identical over
        // this repo's regression corpus/anchors, NOT a universal "every
        // |sample| < 1" claim (a value near +-1.0f can still round the
        // accumulator past +-32768 pre-saturate, and pre-fix THAT was
        // undefined behavior too, so no byte contract existed there either
        // -- see the Writer hardening doc in the file header comment for
        // the full writeup). Only the +1.0f / -1.0f boundary itself changes
        // behavior here: a float-to-int16_t cast of +-32768.5f (out of
        // int16_t's range -- undefined behavior per C99 6.3.1.4p1,
        // reproduced under UBSan) becomes a defined saturate at
        // 32767 / -32768.
        float scaled = sample * 32768.0f;
        int32_t v = (int32_t)(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
        if (v > 32767) v = 32767;
        if (v < -32768) v = -32768;
        int16_t s16 = (int16_t)v;
#else
        // NR style: plain truncation, no rounding. The saturate here is
        // pure defense -- the preceding clamp already bounds `sample` to
        // [-1,1] and the non-finite sanitize above already removed
        // NaN/Inf, so `v` can only ever land in [-32767, 32767] for any
        // real input -- but it keeps the int16_t narrowing itself always
        // well-defined regardless.
        int32_t v = (int32_t)(sample * 32767.0f);
        if (v > 32767) v = 32767;
        if (v < -32768) v = -32768;
        int16_t s16 = (int16_t)v;
#endif
        size_t written = fwrite(&s16, 2, 1, w->fp);
        if (written != 1) {
            // Short write: flag it and stop attempting further samples in
            // this call rather than keep spinning through failing fwrites.
            w->write_error = 1;
            break;
        }
        w->samples_written++;
    }
}

/**
 * Finalizes a WAV writer's header and closes the underlying file (B08).
 *
 * Does exactly what wav_close_write() has always done -- seek to the start,
 * write the real RIFF/fmt/data header now that samples_written is known,
 * close the stream, free the writer -- but CHECKS every fseek/fwrite/fclose
 * return value here, and folds in any write_error already flagged by an
 * earlier short write inside wav_write_float(). The header WRITES
 * THEMSELVES are byte-for-byte identical to before on every success path
 * (same fields, same values, same order) -- only the observability of a
 * failure changes.
 *
 * Ownership: `w` is ALWAYS consumed by this call, on success or failure --
 * the stream is closed and the writer struct freed either way (best-effort:
 * if a fwrite partway through the header fails, this still attempts to
 * fclose+free rather than leak the handle; if fclose itself then also fails,
 * that's still reflected in the return value, but `w` is freed regardless).
 * Do not call this (or wav_close_write()) again on the same pointer, and do
 * not dereference `w` afterward either way.
 *
 * @param w Writer handle (may be NULL; returns -1 immediately, no-op)
 * @return 0 if the header was written and the file closed with no error
 *         observed anywhere in this call AND no write_error was already
 *         flagged by an earlier wav_write_float() call on this writer;
 *         -1 otherwise (any fseek/fwrite/fclose failure here, the
 *         pathological-size abandon path, a NULL writer, or a pre-existing
 *         write_error).
 */
static inline int wav_finalize_write(WavWriter* w) {
    if (!w) return -1;

    int ok = (w->write_error == 0);

    int sample_bytes = w->info.is_float ? 4 : 2;

    /* R01 fix: the outer RIFF chunk_size now uses the SAME sample_bytes-
     * derived size as the "data" sub-chunk (data_sz64 below), instead of a
     * hardcoded 2 bytes/sample. Before this fix, the outer RIFF size was
     * silently under-reported specifically in AEC's float32 output mode (4
     * bytes/sample) -- the "data" sub-chunk's OWN size field was always
     * correct even before this fix. PCM16 output (both writer styles) is a
     * no-op change here: 2 == sample_bytes there already. See the Writer
     * hardening doc in the file header comment for the full writeup. */
    uint64_t data_sz64 = w->samples_written * (uint64_t)w->info.channels * (uint64_t)sample_bytes;
    /* This multiply happens BEFORE the >0xFFFFFFFF range check just below,
     * but it can't realistically overflow its own uint64_t regardless: every
     * wav_write_float() call increments samples_written by at most an `int`
     * argument's worth of samples per call (INT_MAX), and channels/
     * sample_bytes are each tiny (channels <=8 by convention in this
     * codebase's readers, sample_bytes is 2 or 4) -- reaching anywhere near
     * UINT64_MAX would require on the order of 2^61 wav_write_float() calls.
     * The range check below is what actually matters: it's what rejects any
     * data_sz64 that would silently truncate on narrowing to the real
     * on-disk uint32_t/uint16_t header fields, not this multiply. */
    uint64_t file_size64 = 36ull + data_sz64;
    uint64_t byte_rate64 = (uint64_t)w->info.sample_rate * (uint64_t)w->info.channels * (uint64_t)sample_bytes;
    uint64_t block_align64 = (uint64_t)w->info.channels * (uint64_t)sample_bytes;

    if (file_size64 > 0xFFFFFFFFull || data_sz64 > 0xFFFFFFFFull ||
        byte_rate64 > 0xFFFFFFFFull || block_align64 > 0xFFFFull) {
        // Pathological size (never hit by any real capture) -- would
        // silently truncate a WAV header field. Abandon the file instead
        // of writing a corrupt header.
        if (fclose(w->fp) != 0) ok = 0;
        free(w);
        return ok ? 0 : -1;
    }

    // Seek to beginning and write proper header
    if (fseek(w->fp, 0, SEEK_SET) != 0) ok = 0;

    // RIFF header
    if (fwrite("RIFF", 1, 4, w->fp) != 4) ok = 0;
    uint32_t chunk_size = (uint32_t)file_size64;
    if (fwrite(&chunk_size, 4, 1, w->fp) != 1) ok = 0;
    if (fwrite("WAVE", 1, 4, w->fp) != 4) ok = 0;

    // fmt chunk
    if (fwrite("fmt ", 1, 4, w->fp) != 4) ok = 0;
    uint32_t subchunk1_size = 16;
    if (fwrite(&subchunk1_size, 4, 1, w->fp) != 1) ok = 0;
    uint16_t audio_format = w->info.is_float ? 3 : 1;  // 3 = IEEE float, 1 = PCM
    if (fwrite(&audio_format, 2, 1, w->fp) != 1) ok = 0;
    uint16_t num_channels = (uint16_t)w->info.channels;
    if (fwrite(&num_channels, 2, 1, w->fp) != 1) ok = 0;
    uint32_t sample_rate = (uint32_t)w->info.sample_rate;
    if (fwrite(&sample_rate, 4, 1, w->fp) != 1) ok = 0;
    uint32_t byte_rate = (uint32_t)byte_rate64;
    if (fwrite(&byte_rate, 4, 1, w->fp) != 1) ok = 0;
    uint16_t block_align = (uint16_t)block_align64;
    if (fwrite(&block_align, 2, 1, w->fp) != 1) ok = 0;
    uint16_t bits_per_sample = (uint16_t)(sample_bytes * 8);
    if (fwrite(&bits_per_sample, 2, 1, w->fp) != 1) ok = 0;

    // data chunk header
    if (fwrite("data", 1, 4, w->fp) != 4) ok = 0;
    uint32_t data_sz = (uint32_t)data_sz64;
    if (fwrite(&data_sz, 4, 1, w->fp) != 1) ok = 0;

    if (fclose(w->fp) != 0) ok = 0;
    free(w);
    return ok ? 0 : -1;
}

/**
 * Close WAV writer and finalize header.
 *
 * Compat wrapper around wav_finalize_write() (B08) that discards the
 * success/failure result, preserving this function's original `void`
 * signature for existing callers. Prefer wav_finalize_write() directly in
 * new code: it surfaces exactly the same finalize failure (a bad
 * fseek/fwrite/fclose here, or an earlier short write flagged during
 * wav_write_float()) that this wrapper silently swallows. Ownership is
 * identical either way -- `w` is always consumed.
 */
static inline void wav_close_write(WavWriter* w) {
    (void)wav_finalize_write(w);
}

#endif // WAV_IO_H
