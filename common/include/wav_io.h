/** Shared WAV reader/writer.
 * Reader: PCM16 and IEEE-float32, 1..8 channels; RIFF chunk bounds and format fields are validated; non-finite float
 * input becomes 0 and is counted in WavReader.nonfinite_sanitized.
 * Writer style is selected by WAV_IO_WRITER_STYLE: AEC style writes rounded PCM16 (raw float32 with AEC_OUT_FLOAT=1),
 * NR style writes PCM16 truncated toward zero. PCM16 output is finite-sanitized and saturated; float32 output is unsanitized. */
#ifndef WAV_IO_H
#define WAV_IO_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

/* Writer-style selector: define WAV_IO_WRITER_STYLE to one of these before including; unset means WAV_IO_WRITER_AEC. */
#define WAV_IO_WRITER_AEC 1
#define WAV_IO_WRITER_NR  2
#ifndef WAV_IO_WRITER_STYLE
#define WAV_IO_WRITER_STYLE WAV_IO_WRITER_AEC
#endif

typedef struct {
    int sample_rate;
    int channels;
    int bits_per_sample;
    int num_samples;
    int is_float;
} WavInfo;

typedef struct {
    FILE* fp;
    WavInfo info;
    long data_start;
    int samples_read;
    /* Float32 source samples that were NaN/Inf and replaced with 0.0f by wav_read_float (0 for PCM16 sources). */
    uint64_t nonfinite_sanitized;
} WavReader;

typedef struct {
    FILE* fp;
    WavInfo info;
    long data_start;
    uint64_t samples_written;
    /* PCM16-path samples that were NaN/Inf and replaced with 0.0f before quantization (float32 mode leaves this at 0). */
    uint64_t nonfinite_sanitized;
    /* Set to 1 when an fwrite in wav_write_float writes fewer elements than requested (e.g. ENOSPC);
     * wav_finalize_write reports it in its return value. */
    int write_error;
} WavWriter;

/* Size of an open file without changing its position, or -1 on error. */
static inline long wav_io_file_size(FILE* fp) {
    long cur = ftell(fp);
    if (cur < 0) return -1;
    if (fseek(fp, 0, SEEK_END) != 0) return -1;
    long sz = ftell(fp);
    if (fseek(fp, cur, SEEK_SET) != 0) return -1;
    return sz;
}

/* Skip chunk_sz bytes of sub-chunk data plus the RIFF pad byte when chunk_sz is odd; returns 0 and seeks nothing
 * if that would pass the end of the file. fp must be at the first data byte of the chunk. */
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

/** Open a WAV file for reading; returns the reader handle, or NULL on error. */
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

    char riff[4], wave[4];
    uint32_t chunk_size;

    if (fread(riff, 1, 4, fp) != 4) goto error;
    if (fread(&chunk_size, 4, 1, fp) != 1) goto error;
    if (fread(wave, 1, 4, fp) != 4) goto error;

    if (memcmp(riff, "RIFF", 4) != 0 || memcmp(wave, "WAVE", 4) != 0) {
        goto error;
    }

    int bytes_per_sample = 0;

    while (1) {
        char chunk_id[4];
        uint32_t chunk_sz;
        if (fread(chunk_id, 1, 4, fp) != 4) goto error;
        if (fread(&chunk_sz, 4, 1, fp) != 1) goto error;

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            if (chunk_sz < 16) goto error;

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

            // Only PCM16 and IEEE float32 are accepted.
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

            bytes_per_sample = bits_per_sample / 8;
            uint32_t canonical_block_align = (uint32_t)num_channels * (uint32_t)bytes_per_sample;
            if (block_align != 0 && (uint32_t)block_align != canonical_block_align) goto error;
            if (byte_rate != 0 && byte_rate != sample_rate * canonical_block_align) goto error;

            r->info.sample_rate = (int)sample_rate;
            r->info.channels = (int)num_channels;
            r->info.bits_per_sample = (int)bits_per_sample;
            r->info.is_float = is_float;

            // Consume extra format bytes beyond the 16 fixed fields plus the RIFF pad byte.
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

/** Read up to n samples as float (mono: first channel only); returns the number actually read. */
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
            // 32-bit integer PCM is not reachable through wav_open_read; kept for a hand-built WavReader.
            int32_t sample;
            if (fread(&sample, 4, 1, r->fp) != 1) break;
            buf[i] = (float)sample / 2147483648.0f;
            if (channels > 1) {
                fseek(r->fp, 4 * (channels - 1), SEEK_CUR);
            }
        } else {
            break;
        }
        r->samples_read++;
        read_count++;
    }

    return read_count;
}

/** Close the reader. */
static inline void wav_close_read(WavReader* r) {
    if (r) {
        if (r->fp) fclose(r->fp);
        free(r);
    }
}

/** Open a WAV file for writing (16-bit PCM, or float32 in the AEC style). Returns NULL on invalid arguments
 * (NULL path, sample_rate <= 0, channels <= 0) or when block_align > 0xFFFF or byte_rate > 0xFFFFFFFF. */
static inline WavWriter* wav_open_write(const char* path, int sample_rate, int channels) {
    if (!path || sample_rate <= 0 || channels <= 0) return NULL;

    // The sample format is decided before fopen so header range checks run before the file is created or truncated.
    int is_float;
    int bits_per_sample;
#if WAV_IO_WRITER_STYLE == WAV_IO_WRITER_AEC
    /* Default PCM16; AEC_OUT_FLOAT=1 writes IEEE float32. */
    const char* float_env = getenv("AEC_OUT_FLOAT");
    is_float = (float_env && float_env[0] == '1') ? 1 : 0;
    bits_per_sample = is_float ? 32 : 16;
#else
    is_float = 0;
    bits_per_sample = 16;
#endif
    int sample_bytes = is_float ? 4 : 2;

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

    // Placeholder header, rewritten at close; a short write here fails the open.
    uint8_t header[44] = {0};
    if (fwrite(header, 1, 44, fp) != 44) {
        fclose(fp);
        free(w);
        return NULL;
    }
    w->data_start = 44;

    return w;
}

/** Write n float samples. */
static inline void wav_write_float(WavWriter* w, const float* buf, int n) {
    if (!w || !buf) return;
    // A negative n would become a huge size_t count in fwrite.
    if (n <= 0) return;

#if WAV_IO_WRITER_STYLE == WAV_IO_WRITER_AEC
    if (w->info.is_float) {
        /* IEEE float32 is written unchanged, NaN/Inf payloads included. */
        size_t written = fwrite(buf, sizeof(float), (size_t)n, w->fp);
        w->samples_written += written;
        if (written != (size_t)n) w->write_error = 1;
        return;
    }
#endif

    for (int i = 0; i < n; i++) {
        float sample = buf[i];

        // A non-finite sample becomes 0.0f (as in wav_read_float); otherwise NaN would reach an undefined float-to-int conversion.
        if (!isfinite(sample)) {
            sample = 0.0f;
            w->nonfinite_sanitized++;
        }

        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;

#if WAV_IO_WRITER_STYLE == WAV_IO_WRITER_AEC
        // Round half away from zero into int32_t, then saturate to int16_t (a direct +-32768.5f conversion is undefined).
        float scaled = sample * 32768.0f;
        int32_t v = (int32_t)(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
        if (v > 32767) v = 32767;
        if (v < -32768) v = -32768;
        int16_t s16 = (int16_t)v;
#else
        // NR style: truncation, no rounding.
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

/** Finalize the header and close the file; the writer `w` is always consumed (success or failure) and must not be used again.
 * Returns 0 on success; -1 on a NULL writer, an fseek/fwrite/fclose failure, sizes too large for the WAV fields,
 * or a write_error flagged by an earlier short write. */
static inline int wav_finalize_write(WavWriter* w) {
    if (!w) return -1;

    int ok = (w->write_error == 0);

    int sample_bytes = w->info.is_float ? 4 : 2;

    /* Checked multiply guarding samples_written * per_sample against uint64_t wraparound; per_sample == 0 is abandoned too. */
    uint64_t per_sample = (uint64_t)w->info.channels * (uint64_t)sample_bytes;
    if (per_sample == 0 || w->samples_written > (UINT64_MAX - 36ull) / per_sample) {
        // The size arithmetic would wrap: abandon the file (reports failure).
        fclose(w->fp);
        free(w);
        return -1;
    }

    /* RIFF and data chunk sizes use the actual bytes per sample (2 for PCM16, 4 for float32). */
    uint64_t data_sz64 = w->samples_written * per_sample;
    uint64_t file_size64 = 36ull + data_sz64;
    uint64_t byte_rate64 = (uint64_t)w->info.sample_rate * per_sample;
    uint64_t block_align64 = per_sample;

    if (file_size64 > 0xFFFFFFFFull || data_sz64 > 0xFFFFFFFFull ||
        byte_rate64 > 0xFFFFFFFFull || block_align64 > 0xFFFFull) {
        // A header field would be truncated: abandon the file rather than write a corrupt header.
        fclose(w->fp);
        free(w);
        return -1;
    }

    if (fseek(w->fp, 0, SEEK_SET) != 0) ok = 0;

    if (fwrite("RIFF", 1, 4, w->fp) != 4) ok = 0;
    uint32_t chunk_size = (uint32_t)file_size64;
    if (fwrite(&chunk_size, 4, 1, w->fp) != 1) ok = 0;
    if (fwrite("WAVE", 1, 4, w->fp) != 4) ok = 0;

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

    if (fwrite("data", 1, 4, w->fp) != 4) ok = 0;
    uint32_t data_sz = (uint32_t)data_sz64;
    if (fwrite(&data_sz, 4, 1, w->fp) != 1) ok = 0;

    /* fwrite can report a full count on a stream whose write already failed (glibc cookie streams in unbuffered mode,
     * where fclose then also returns 0); the sticky stream error flag is the reliable signal. */
    if (ferror(w->fp)) ok = 0;
    if (fclose(w->fp) != 0) ok = 0;
    free(w);
    return ok ? 0 : -1;
}

/** Close the writer and finalize the header; wraps wav_finalize_write() and discards its result. `w` is always consumed. */
static inline void wav_close_write(WavWriter* w) {
    (void)wav_finalize_write(w);
}

#endif // WAV_IO_H
