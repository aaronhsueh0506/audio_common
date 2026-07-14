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
 *       (`sample*32768.0f`, then +-0.5f before truncating).
 *   WAV_IO_WRITER_NR - always PCM16, `sample*32767.0f` truncation (no
 *       rounding, no float32 path) -- NR's historical behavior.
 * wav_close_write's header size fields (RIFF chunk_size, byte_rate,
 * block_align, data-chunk size) are computed in uint64_t and range
 * checked before being narrowed to their on-disk uint32_t/uint16_t
 * fields; a size that would overflow abandons the file (fclose + free)
 * instead of writing a silently-truncated header. For every size actually
 * reachable by either repo's callers today this is a no-op -- see the
 * in-function comment on the pre-existing (and deliberately preserved)
 * "RIFF-level chunk_size uses a hardcoded 2 bytes/sample" quirk that
 * under-reports the outer RIFF size specifically in AEC's float32 output
 * mode (the "data" sub-chunk's own size field is, and always was,
 * correct). Not in scope to fix here -- preserving it byte-for-bit is the
 * F06 contract; AEC_OUT_FLOAT is a test-only path and no consumer reads
 * the outer RIFF size instead of the data chunk's own.
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
    int samples_written;
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

    // Write placeholder header (will update at close)
    uint8_t header[44] = {0};
    fwrite(header, 1, 44, fp);
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

#if WAV_IO_WRITER_STYLE == WAV_IO_WRITER_AEC
    if (w->info.is_float) {
        /* IEEE float32 (no quantization, for C-vs-Python correlation tests) */
        fwrite(buf, sizeof(float), n, w->fp);
        w->samples_written += n;
        return;
    }
#endif

    for (int i = 0; i < n; i++) {
        float sample = buf[i];
        // Clamp to [-1, 1]
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;

#if WAV_IO_WRITER_STYLE == WAV_IO_WRITER_AEC
        float scaled = sample * 32768.0f;
        int16_t s16 = (int16_t)(scaled >= 0 ? scaled + 0.5f : scaled - 0.5f);
#else
        int16_t s16 = (int16_t)(sample * 32767.0f);
#endif
        fwrite(&s16, 2, 1, w->fp);
        w->samples_written++;
    }
}

/**
 * Close WAV writer and finalize header
 */
static inline void wav_close_write(WavWriter* w) {
    if (!w) return;

    int sample_bytes = w->info.is_float ? 4 : 2;

    /* Preserves the pre-hardening arithmetic verbatim for BOTH writer
     * styles: the RIFF-level chunk_size has always been computed here
     * from a hardcoded 2 bytes/sample -- a pre-existing quirk that
     * under-reports the outer RIFF size in AEC's float32 output mode (the
     * "data" sub-chunk's OWN size field below is, and always was, correct
     * via sample_bytes). NR's writer style never sets is_float, so the
     * quirk never manifests there (2 == sample_bytes always). Kept
     * byte-for-byte; not in scope to fix as part of F06. */
    uint64_t data_size64 = (uint64_t)w->samples_written * 2u * (uint64_t)w->info.channels;
    uint64_t file_size64 = 36ull + data_size64;
    uint64_t data_sz64 = (uint64_t)w->samples_written * (uint64_t)w->info.channels * (uint64_t)sample_bytes;
    uint64_t byte_rate64 = (uint64_t)w->info.sample_rate * (uint64_t)w->info.channels * (uint64_t)sample_bytes;
    uint64_t block_align64 = (uint64_t)w->info.channels * (uint64_t)sample_bytes;

    if (data_size64 > 0xFFFFFFFFull || file_size64 > 0xFFFFFFFFull ||
        data_sz64 > 0xFFFFFFFFull || byte_rate64 > 0xFFFFFFFFull ||
        block_align64 > 0xFFFFull) {
        // Pathological size (never hit by any real capture) -- would
        // silently truncate a WAV header field. Abandon the file instead
        // of writing a corrupt header.
        fclose(w->fp);
        free(w);
        return;
    }

    // Seek to beginning and write proper header
    fseek(w->fp, 0, SEEK_SET);

    // RIFF header
    fwrite("RIFF", 1, 4, w->fp);
    uint32_t chunk_size = (uint32_t)file_size64;
    fwrite(&chunk_size, 4, 1, w->fp);
    fwrite("WAVE", 1, 4, w->fp);

    // fmt chunk
    fwrite("fmt ", 1, 4, w->fp);
    uint32_t subchunk1_size = 16;
    fwrite(&subchunk1_size, 4, 1, w->fp);
    uint16_t audio_format = w->info.is_float ? 3 : 1;  // 3 = IEEE float, 1 = PCM
    fwrite(&audio_format, 2, 1, w->fp);
    uint16_t num_channels = (uint16_t)w->info.channels;
    fwrite(&num_channels, 2, 1, w->fp);
    uint32_t sample_rate = (uint32_t)w->info.sample_rate;
    fwrite(&sample_rate, 4, 1, w->fp);
    uint32_t byte_rate = (uint32_t)byte_rate64;
    fwrite(&byte_rate, 4, 1, w->fp);
    uint16_t block_align = (uint16_t)block_align64;
    fwrite(&block_align, 2, 1, w->fp);
    uint16_t bits_per_sample = (uint16_t)(sample_bytes * 8);
    fwrite(&bits_per_sample, 2, 1, w->fp);

    // data chunk header
    fwrite("data", 1, 4, w->fp);
    uint32_t data_sz = (uint32_t)data_sz64;
    fwrite(&data_sz, 4, 1, w->fp);

    fclose(w->fp);
    free(w);
}

#endif // WAV_IO_H
