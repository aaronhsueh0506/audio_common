/**
 * Shared WAV reader/writer used by AEC, NR and Audio_ALG.
 *
 * Reader: PCM16 and IEEE-float32, 1..8 channels. RIFF chunk bounds, format
 * metadata and arithmetic are validated; non-finite float input is replaced
 * with zero and counted in WavReader.nonfinite_sanitized.
 *
 * Writer behavior is selected with WAV_IO_WRITER_STYLE. AEC style defaults
 * to PCM16 (rounded) and supports raw float32 with AEC_OUT_FLOAT=1; NR style
 * writes PCM16 truncated toward zero. PCM16 output is finite-sanitized and
 * saturated to int16. Float32 output remains an unsanitized diagnostic path.
 */
#ifndef WAV_IO_H
#define WAV_IO_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

/* Writer-style selector. A consumer defines WAV_IO_WRITER_STYLE to one of
 * these before including this file; unset means WAV_IO_WRITER_AEC. */
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
    /* Count of float32 source samples that were NaN/+-Inf and were replaced
     * with 0.0f by wav_read_float. Always 0 for a PCM16 source. */
    uint64_t nonfinite_sanitized;
} WavReader;

// WAV writer handle
typedef struct {
    FILE* fp;
    WavInfo info;
    long data_start;
    uint64_t samples_written;
    /* Count of PCM16-path source samples that were NaN/+-Inf and were
     * replaced with 0.0f before int16 quantization. The float32 mode
     * (WAV_IO_WRITER_AEC with AEC_OUT_FLOAT=1) writes samples unchanged and
     * leaves this at 0. */
    uint64_t nonfinite_sanitized;
    /* Set to 1 the first time an fwrite inside wav_write_float writes fewer
     * elements than requested (e.g. ENOSPC). samples_written counts only what
     * reached the file and no further samples of that call are attempted.
     * wav_finalize_write finalizes the header from samples_written and
     * reports the error in its return value. */
    int write_error;
} WavWriter;

// ============================================================================
// WAV Reader
// ============================================================================

/* Size of an open file without changing its position, or -1 on error. Every
 * chunk size read by wav_open_read is bounds-checked against it. */
static inline long wav_io_file_size(FILE* fp) {
    long cur = ftell(fp);
    if (cur < 0) return -1;
    if (fseek(fp, 0, SEEK_END) != 0) return -1;
    long sz = ftell(fp);
    if (fseek(fp, cur, SEEK_SET) != 0) return -1;
    return sz;
}

/* Skips chunk_sz bytes of a sub-chunk's data plus the RIFF pad byte when
 * chunk_sz is odd. Returns 0 and seeks nothing if that would move past the
 * end of the file. fp must be at the first data byte of the chunk (right
 * after its 4-byte id and 4-byte size). */
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

            // Only PCM16 and IEEE float32 are accepted; this also rules out
            // bits_per_sample == 0.
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

            // Consume extra format bytes beyond the 16 fixed fields plus the
            // RIFF pad byte; the bounds were checked above.
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
            // 32-bit integer PCM: not reachable through wav_open_read, which
            // accepts only PCM16 and float32; kept for a hand-built
            // WavReader. Integer samples are always finite.
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
 * Open WAV file for writing (16-bit PCM, or float32 in the AEC style; see
 * WAV_IO_WRITER_STYLE)
 *
 * @param path Path to output WAV file
 * @param sample_rate Sample rate
 * @param channels Number of channels (usually 1)
 * @return Writer handle, or NULL on error: invalid arguments (NULL path,
 *         sample_rate <= 0, channels <= 0) or header fields that would not
 *         fit the WAV format (block_align > 0xFFFF or byte_rate > 0xFFFFFFFF,
 *         the limits wav_finalize_write enforces)
 */
static inline WavWriter* wav_open_write(const char* path, int sample_rate, int channels) {
    // Invalid arguments fail before the filesystem is touched.
    if (!path || sample_rate <= 0 || channels <= 0) return NULL;

    // The sample format is decided before fopen so that the header range
    // checks below run before the output file is created or truncated.
    int is_float;
    int bits_per_sample;
#if WAV_IO_WRITER_STYLE == WAV_IO_WRITER_AEC
    /* Default PCM16. Set AEC_OUT_FLOAT=1 to write IEEE float32 (for testing) */
    const char* float_env = getenv("AEC_OUT_FLOAT");
    is_float = (float_env && float_env[0] == '1') ? 1 : 0;
    bits_per_sample = is_float ? 32 : 16;
#else
    is_float = 0;
    bits_per_sample = 16;
#endif
    int sample_bytes = is_float ? 4 : 2;

    // The same block_align / byte_rate limits wav_finalize_write enforces,
    // so a writer that could never finalize a valid header is not created.
    uint64_t block_align64 = (uint64_t)channels * (uint64_t)sample_bytes;
    uint64_t byte_rate64 = (uint64_t)sample_rate * block_align64;
    if (block_align64 > 0xFFFFull || byte_rate64 > 0xFFFFFFFFull) return NULL;

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
    w->info.is_float = is_float;
    w->info.bits_per_sample = bits_per_sample;

    // Placeholder header, rewritten at close. A short write here (e.g. a full
    // disk) must fail the open: data_start would otherwise point past bytes
    // that are not in the file.
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
    // A negative n would become a huge size_t count in fwrite below.
    if (n <= 0) return;

#if WAV_IO_WRITER_STYLE == WAV_IO_WRITER_AEC
    if (w->info.is_float) {
        /* IEEE float32 written unchanged, NaN/Inf payloads included. */
        size_t written = fwrite(buf, sizeof(float), (size_t)n, w->fp);
        w->samples_written += written;
        if (written != (size_t)n) w->write_error = 1;
        return;
    }
#endif

    for (int i = 0; i < n; i++) {
        float sample = buf[i];

        // As in wav_read_float, a non-finite sample becomes 0.0f. Otherwise
        // NaN would pass the ordered clamp below and reach an undefined
        // float-to-int conversion.
        if (!isfinite(sample)) {
            sample = 0.0f;
            w->nonfinite_sanitized++;
        }

        // Clamp to [-1, 1]
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;

#if WAV_IO_WRITER_STYLE == WAV_IO_WRITER_AEC
        // Round half away from zero into an int32_t, then saturate to the
        // int16_t range. Converting +-32768.5f straight to int16_t would be
        // undefined behaviour (C99 6.3.1.4p1).
        float scaled = sample * 32768.0f;
        int32_t v = (int32_t)(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
        if (v > 32767) v = 32767;
        if (v < -32768) v = -32768;
        int16_t s16 = (int16_t)v;
#else
        // NR style: truncation, no rounding. The clamp and the sanitize above
        // already bound v to [-32767, 32767]; the saturate keeps the int16_t
        // narrowing defined regardless.
        int32_t v = (int32_t)(sample * 32767.0f);
        if (v > 32767) v = 32767;
        if (v < -32768) v = -32768;
        int16_t s16 = (int16_t)v;
#endif
        size_t written = fwrite(&s16, 2, 1, w->fp);
        if (written != 1) {
            // Short write: flag it and stop for this call.
            w->write_error = 1;
            break;
        }
        w->samples_written++;
    }
}

/**
 * Finalizes a WAV writer's header and closes the file.
 *
 * Seeks to the start, writes the RIFF/fmt/data header now that samples_written
 * is known, closes the stream and frees the writer. Every fseek / fwrite /
 * fclose result is checked, and a write_error flagged by an earlier short
 * write in wav_write_float is included in the return value.
 *
 * Ownership: `w` is always consumed, on success or failure; the stream is
 * closed and the writer freed even when a write fails. Do not use `w` again.
 *
 * @param w Writer handle (may be NULL; returns -1)
 * @return 0 if the header was written, the file closed without error and no
 *         write_error was flagged; -1 otherwise (an fseek / fwrite / fclose
 *         failure, a size too large for the WAV fields, a NULL writer, or a
 *         flagged write_error)
 */
static inline int wav_finalize_write(WavWriter* w) {
    if (!w) return -1;

    int ok = (w->write_error == 0);

    int sample_bytes = w->info.is_float ? 4 : 2;

    /* Checked multiply: it guards samples_written * per_sample against uint64_t
     * wraparound. per_sample == 0 (a hand-built WavInfo with channels == 0) is
     * abandoned the same way instead of dividing by zero. */
    uint64_t per_sample = (uint64_t)w->info.channels * (uint64_t)sample_bytes;
    if (per_sample == 0 || w->samples_written > (UINT64_MAX - 36ull) / per_sample) {
        // The size arithmetic below would wrap: abandon the file. The
        // placeholder header stays on disk, so this reports failure.
        fclose(w->fp);
        free(w);
        return -1;
    }

    /* The RIFF chunk size and the "data" chunk size both use the actual
     * bytes per sample (2 for PCM16, 4 for float32). */
    uint64_t data_sz64 = w->samples_written * per_sample;
    uint64_t file_size64 = 36ull + data_sz64;
    uint64_t byte_rate64 = (uint64_t)w->info.sample_rate * per_sample;
    uint64_t block_align64 = per_sample;

    if (file_size64 > 0xFFFFFFFFull || data_sz64 > 0xFFFFFFFFull ||
        byte_rate64 > 0xFFFFFFFFull || block_align64 > 0xFFFFull) {
        // A header field would be truncated: abandon the file rather than
        // write a corrupt header. Only the placeholder header is on disk, so
        // this reports failure whatever fclose returns.
        fclose(w->fp);
        free(w);
        return -1;
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
 * Wrapper around wav_finalize_write() that discards its result. Prefer
 * wav_finalize_write() to see a failed finalize. `w` is always consumed.
 */
static inline void wav_close_write(WavWriter* w) {
    (void)wav_finalize_write(w);
}

#endif // WAV_IO_H
