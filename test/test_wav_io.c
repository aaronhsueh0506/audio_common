/*
 * test_wav_io.c - negative-corpus tests for the hardened, shared WAV parser
 * (F06 remediation: unsafe duplicated WAV parser).
 *
 * Exercises audio_common/include/wav_io.h directly (this test does not set
 * WAV_IO_WRITER_STYLE, so it gets the default WAV_IO_WRITER_AEC writer --
 * the writer style knob isn't reader-observable and isn't what's under
 * test here).
 *
 * Every malformed input below must make wav_open_read() return NULL
 * without crashing (no OOB read, no divide-by-zero, no fseek/fread on a
 * freed/NULL FILE*). Every well-formed input must parse exactly as it did
 * before the F06 hardening pass.
 *
 * Plain assert-and-report harness, same shape as test_pool_contract.c: no
 * external framework, every failure prints and flips a global fail flag,
 * main() reports ALL PASS / FAIL at the end so one bad check doesn't hide
 * the rest.
 */
#include "wav_io.h"
#include "test_wav_writer_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <unistd.h>
#include <sys/resource.h>
#include <signal.h>

static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_fail = 1; \
    } \
} while (0)

/* ═══════════════════════════ raw-byte WAV builder ═════════════════════════
 * Minimal growable byte buffer + little-endian field pushers, so each test
 * can hand-construct (and deliberately corrupt) a WAV header byte-for-byte
 * -- exactly the kind of malformed input a real file or a hostile input
 * could produce, which no "write a valid WAV, then edit WavInfo" API could
 * express. Little-endian is the native byte order on every platform this
 * code targets (same assumption wav_io.h itself already makes by fread'ing
 * multi-byte fields directly into typed variables).
 */
typedef struct {
    uint8_t* buf;
    size_t len;
    size_t cap;
} ByteBuf;

static void bb_init(ByteBuf* b) { b->buf = NULL; b->len = 0; b->cap = 0; }

static void bb_push(ByteBuf* b, const void* data, size_t n) {
    if (b->len + n > b->cap) {
        size_t new_cap = b->cap ? b->cap * 2 : 256;
        while (new_cap < b->len + n) new_cap *= 2;
        b->buf = (uint8_t*)realloc(b->buf, new_cap);
        b->cap = new_cap;
    }
    memcpy(b->buf + b->len, data, n);
    b->len += n;
}

static void bb_str4(ByteBuf* b, const char* s4) { bb_push(b, s4, 4); }
static void bb_u8(ByteBuf* b, uint8_t v)   { bb_push(b, &v, 1); }
static void bb_u16(ByteBuf* b, uint16_t v) { bb_push(b, &v, 2); }
static void bb_u32(ByteBuf* b, uint32_t v) { bb_push(b, &v, 4); }
static void bb_free(ByteBuf* b) { free(b->buf); b->buf = NULL; b->len = b->cap = 0; }

/* Appends a standard 16-byte fmt sub-chunk (id + size + the 16 fixed
 * fields), letting every field be overridden for negative tests. */
static void bb_fmt_chunk(ByteBuf* b, uint32_t fmt_chunk_sz, uint16_t audio_format,
                          uint16_t channels, uint32_t sample_rate, uint32_t byte_rate,
                          uint16_t block_align, uint16_t bits_per_sample) {
    bb_str4(b, "fmt ");
    bb_u32(b, fmt_chunk_sz);
    bb_u16(b, audio_format);
    bb_u16(b, channels);
    bb_u32(b, sample_rate);
    bb_u32(b, byte_rate);
    bb_u16(b, block_align);
    bb_u16(b, bits_per_sample);
}

/* Appends "RIFF" + riff_size + "WAVE". riff_size isn't validated by
 * wav_open_read (see file header doc / wav_io.h's own doc comment) so its
 * exact value doesn't matter for these tests; a plausible placeholder
 * (0xFFFFFFFF-safe small value) is fine. */
static void bb_riff_header(ByteBuf* b, uint32_t riff_size) {
    bb_str4(b, "RIFF");
    bb_u32(b, riff_size);
    bb_str4(b, "WAVE");
}

/* Writes a ByteBuf to a fresh, uniquely-named temp file and returns the
 * heap-allocated path (caller must unlink() + free() it). Returns NULL on
 * failure. */
static char* write_temp_wav(const ByteBuf* b) {
    char* path = (char*)malloc(64);
    if (!path) return NULL;
    snprintf(path, 64, "/tmp/audio_common_test_wav_io_XXXXXX");
    int fd = mkstemp(path);
    if (fd < 0) { free(path); return NULL; }
    FILE* fp = fdopen(fd, "wb");
    if (!fp) { close(fd); unlink(path); free(path); return NULL; }
    size_t written = fwrite(b->buf, 1, b->len, fp);
    fclose(fp);
    if (written != b->len) { unlink(path); free(path); return NULL; }
    return path;
}

static void cleanup_temp(char* path) {
    if (path) { unlink(path); free(path); }
}

/* Builds a plain, well-formed PCM16 mono WAV (canonical block_align/byte_rate,
 * no extra chunks) with `n` int16 samples. */
static ByteBuf build_valid_pcm16(int sample_rate, const int16_t* samples, int n) {
    ByteBuf b; bb_init(&b);
    uint32_t data_sz = (uint32_t)(n * 2);
    bb_riff_header(&b, 36 + data_sz);
    bb_fmt_chunk(&b, 16, 1, 1, (uint32_t)sample_rate, (uint32_t)sample_rate * 2, 2, 16);
    bb_str4(&b, "data");
    bb_u32(&b, data_sz);
    for (int i = 0; i < n; i++) bb_u16(&b, (uint16_t)samples[i]);
    return b;
}

/* ═══════════════════════════════ reject tests ═════════════════════════════ */

static void test_reject_truncated_header(void) {
    ByteBuf b; bb_init(&b);
    bb_str4(&b, "RIF");  /* not even a full 4-byte "RIFF" id */
    char* path = write_temp_wav(&b);
    CHECK(path != NULL, "setup: write truncated-header temp file");
    if (path) {
        WavReader* r = wav_open_read(path);
        CHECK(r == NULL, "wav_open_read must reject a file truncated inside the RIFF id");
        if (r) wav_close_read(r);
    }
    cleanup_temp(path);
    bb_free(&b);

    /* Truncated further in: valid RIFF/WAVE, but cut off mid-fmt-chunk. */
    ByteBuf b2; bb_init(&b2);
    bb_riff_header(&b2, 100);
    bb_str4(&b2, "fmt ");
    bb_u32(&b2, 16);
    bb_u16(&b2, 1);  /* audio_format only -- nothing else follows */
    char* path2 = write_temp_wav(&b2);
    CHECK(path2 != NULL, "setup: write truncated-fmt temp file");
    if (path2) {
        WavReader* r = wav_open_read(path2);
        CHECK(r == NULL, "wav_open_read must reject a file truncated mid-fmt-chunk");
        if (r) wav_close_read(r);
    }
    cleanup_temp(path2);
    bb_free(&b2);
}

static void test_reject_fmt_size_12(void) {
    ByteBuf b; bb_init(&b);
    bb_riff_header(&b, 100);
    /* fmt chunk_sz=12 (< 16), but the buffer still has the following 16
     * bytes there anyway -- this is exactly the pre-hardening desync
     * scenario: the old parser would happily fread() 16 bytes regardless
     * of the declared (too-small) chunk_sz. */
    bb_fmt_chunk(&b, 12, 1, 1, 16000, 32000, 2, 16);
    bb_str4(&b, "data");
    bb_u32(&b, 4);
    bb_u16(&b, 0); bb_u16(&b, 0);
    char* path = write_temp_wav(&b);
    CHECK(path != NULL, "setup: write fmt-size-12 temp file");
    if (path) {
        WavReader* r = wav_open_read(path);
        CHECK(r == NULL, "wav_open_read must reject a fmt chunk_sz < 16");
        if (r) wav_close_read(r);
    }
    cleanup_temp(path);
    bb_free(&b);
}

static void test_reject_channels_0(void) {
    ByteBuf b; bb_init(&b);
    bb_riff_header(&b, 100);
    bb_fmt_chunk(&b, 16, 1, 0 /* channels */, 16000, 0, 0, 16);
    bb_str4(&b, "data");
    bb_u32(&b, 4);
    bb_u16(&b, 0); bb_u16(&b, 0);
    char* path = write_temp_wav(&b);
    CHECK(path != NULL, "setup: write channels-0 temp file");
    if (path) {
        WavReader* r = wav_open_read(path);
        CHECK(r == NULL, "wav_open_read must reject channels == 0 (old divide-by-zero)");
        if (r) wav_close_read(r);
    }
    cleanup_temp(path);
    bb_free(&b);
}

static void test_reject_channels_too_many(void) {
    ByteBuf b; bb_init(&b);
    bb_riff_header(&b, 100);
    bb_fmt_chunk(&b, 16, 1, 9 /* channels, out of [1,8] */, 16000, 16000 * 9 * 2, 9 * 2, 16);
    bb_str4(&b, "data");
    bb_u32(&b, 4);
    bb_u16(&b, 0); bb_u16(&b, 0);
    char* path = write_temp_wav(&b);
    CHECK(path != NULL, "setup: write channels-9 temp file");
    if (path) {
        WavReader* r = wav_open_read(path);
        CHECK(r == NULL, "wav_open_read must reject channels > 8");
        if (r) wav_close_read(r);
    }
    cleanup_temp(path);
    bb_free(&b);
}

static void test_reject_bits_0_1_7(void) {
    const uint16_t bad_bits[3] = { 0, 1, 7 };
    for (int k = 0; k < 3; k++) {
        ByteBuf b; bb_init(&b);
        bb_riff_header(&b, 100);
        bb_fmt_chunk(&b, 16, 1, 1, 16000, 16000, 2, bad_bits[k]);
        bb_str4(&b, "data");
        bb_u32(&b, 4);
        bb_u32(&b, 0);
        char* path = write_temp_wav(&b);
        char msg[128];
        snprintf(msg, sizeof(msg), "setup: write bits=%d temp file", bad_bits[k]);
        CHECK(path != NULL, msg);
        if (path) {
            WavReader* r = wav_open_read(path);
            snprintf(msg, sizeof(msg), "wav_open_read must reject bits_per_sample=%d (old divide-by-zero for bits<8)", bad_bits[k]);
            CHECK(r == NULL, msg);
            if (r) wav_close_read(r);
        }
        cleanup_temp(path);
        bb_free(&b);
    }
}

static void test_reject_bits_24_pcm(void) {
    ByteBuf b; bb_init(&b);
    bb_riff_header(&b, 100);
    bb_fmt_chunk(&b, 16, 1 /* PCM */, 1, 16000, 16000 * 3, 3, 24 /* not an accepted combo */);
    bb_str4(&b, "data");
    bb_u32(&b, 3);
    bb_u8(&b, 0); bb_u8(&b, 0); bb_u8(&b, 0);
    char* path = write_temp_wav(&b);
    CHECK(path != NULL, "setup: write 24-bit-PCM temp file");
    if (path) {
        WavReader* r = wav_open_read(path);
        CHECK(r == NULL, "wav_open_read must reject 24-bit PCM (only PCM16/float32 accepted)");
        if (r) wav_close_read(r);
    }
    cleanup_temp(path);
    bb_free(&b);
}

static void test_reject_bits_32_int_pcm(void) {
    /* format==1 (PCM) with bits==32 is a real (if rare) WAV variant the
     * pre-hardening reader's wav_read_float() had code for, but the F06
     * accept-list narrows wav_open_read() to exactly (1,16) and (3,32) --
     * see wav_io.h's file header doc. */
    ByteBuf b; bb_init(&b);
    bb_riff_header(&b, 100);
    bb_fmt_chunk(&b, 16, 1, 1, 16000, 16000 * 4, 4, 32);
    bb_str4(&b, "data");
    bb_u32(&b, 4);
    bb_u32(&b, 0);
    char* path = write_temp_wav(&b);
    CHECK(path != NULL, "setup: write 32-bit-int-PCM temp file");
    if (path) {
        WavReader* r = wav_open_read(path);
        CHECK(r == NULL, "wav_open_read must reject 32-bit integer PCM (format=1,bits=32)");
        if (r) wav_close_read(r);
    }
    cleanup_temp(path);
    bb_free(&b);
}

static void test_reject_sample_rate_bounds(void) {
    /* sample_rate == 0 */
    {
        ByteBuf b; bb_init(&b);
        bb_riff_header(&b, 100);
        bb_fmt_chunk(&b, 16, 1, 1, 0, 0, 2, 16);
        bb_str4(&b, "data"); bb_u32(&b, 2); bb_u16(&b, 0);
        char* path = write_temp_wav(&b);
        CHECK(path != NULL, "setup: write sample_rate=0 temp file");
        if (path) {
            WavReader* r = wav_open_read(path);
            CHECK(r == NULL, "wav_open_read must reject sample_rate == 0");
            if (r) wav_close_read(r);
        }
        cleanup_temp(path);
        bb_free(&b);
    }
    /* sample_rate > 384000 */
    {
        ByteBuf b; bb_init(&b);
        bb_riff_header(&b, 100);
        bb_fmt_chunk(&b, 16, 1, 1, 400000, 800000, 2, 16);
        bb_str4(&b, "data"); bb_u32(&b, 2); bb_u16(&b, 0);
        char* path = write_temp_wav(&b);
        CHECK(path != NULL, "setup: write sample_rate=400000 temp file");
        if (path) {
            WavReader* r = wav_open_read(path);
            CHECK(r == NULL, "wav_open_read must reject sample_rate > 384000");
            if (r) wav_close_read(r);
        }
        cleanup_temp(path);
        bb_free(&b);
    }
}

static void test_reject_block_align_mismatch(void) {
    ByteBuf b; bb_init(&b);
    bb_riff_header(&b, 100);
    /* channels=1, bits=16 -> canonical block_align is 2, declare 4 instead */
    bb_fmt_chunk(&b, 16, 1, 1, 16000, 16000 * 4, 4, 16);
    bb_str4(&b, "data"); bb_u32(&b, 2); bb_u16(&b, 0);
    char* path = write_temp_wav(&b);
    CHECK(path != NULL, "setup: write block_align-mismatch temp file");
    if (path) {
        WavReader* r = wav_open_read(path);
        CHECK(r == NULL, "wav_open_read must reject block_align != channels*bytes_per_sample");
        if (r) wav_close_read(r);
    }
    cleanup_temp(path);
    bb_free(&b);
}

static void test_reject_byte_rate_mismatch(void) {
    ByteBuf b; bb_init(&b);
    bb_riff_header(&b, 100);
    /* channels=1, bits=16, sample_rate=16000 -> canonical byte_rate is
     * 32000, declare something else. */
    bb_fmt_chunk(&b, 16, 1, 1, 16000, 99999, 2, 16);
    bb_str4(&b, "data"); bb_u32(&b, 2); bb_u16(&b, 0);
    char* path = write_temp_wav(&b);
    CHECK(path != NULL, "setup: write byte_rate-mismatch temp file");
    if (path) {
        WavReader* r = wav_open_read(path);
        CHECK(r == NULL, "wav_open_read must reject byte_rate != sample_rate*block_align");
        if (r) wav_close_read(r);
    }
    cleanup_temp(path);
    bb_free(&b);
}

static void test_reject_data_chunk_size_exceeds_file(void) {
    ByteBuf b; bb_init(&b);
    bb_riff_header(&b, 100);
    bb_fmt_chunk(&b, 16, 1, 1, 16000, 32000, 2, 16);
    bb_str4(&b, "data");
    bb_u32(&b, 1000000u);  /* claims 1MB of data ... */
    bb_u16(&b, 0); bb_u16(&b, 0);  /* ... but the file only has 4 bytes here */
    char* path = write_temp_wav(&b);
    CHECK(path != NULL, "setup: write oversized-data-chunk temp file");
    if (path) {
        WavReader* r = wav_open_read(path);
        CHECK(r == NULL, "wav_open_read must reject a data chunk_sz larger than the remaining file");
        if (r) wav_close_read(r);
    }
    cleanup_temp(path);
    bb_free(&b);
}

static void test_reject_oversized_unknown_chunk(void) {
    /* A non-fmt, non-data chunk claiming to be larger than the file: must
     * be rejected (not silently seeked past EOF) while searching for fmt. */
    ByteBuf b; bb_init(&b);
    bb_riff_header(&b, 100);
    bb_str4(&b, "JUNK");
    bb_u32(&b, 5000000u);
    bb_u8(&b, 0);  /* only one byte actually present */
    char* path = write_temp_wav(&b);
    CHECK(path != NULL, "setup: write oversized-unknown-chunk temp file");
    if (path) {
        WavReader* r = wav_open_read(path);
        CHECK(r == NULL, "wav_open_read must reject an unknown chunk claiming to extend past EOF");
        if (r) wav_close_read(r);
    }
    cleanup_temp(path);
    bb_free(&b);
}

/* ═══════════════════════════════ accept tests ═════════════════════════════ */

static void test_accept_odd_junk_chunk_with_pad(void) {
    /* A 3-byte "JUNK" chunk before "data" -- odd chunk_sz means a RIFF
     * pad byte follows it before the next chunk id. The pre-hardening
     * parser never accounted for that pad byte and would desync. */
    const int16_t samples[4] = { 100, -200, 32767, -32768 };
    ByteBuf b; bb_init(&b);
    bb_riff_header(&b, 100);
    bb_fmt_chunk(&b, 16, 1, 1, 16000, 32000, 2, 16);
    bb_str4(&b, "JUNK");
    bb_u32(&b, 3);
    bb_u8(&b, 0xAA); bb_u8(&b, 0xBB); bb_u8(&b, 0xCC);
    bb_u8(&b, 0x00);  /* RIFF even-alignment pad byte for the odd chunk_sz=3 */
    bb_str4(&b, "data");
    bb_u32(&b, 8);
    for (int i = 0; i < 4; i++) bb_u16(&b, (uint16_t)samples[i]);
    char* path = write_temp_wav(&b);
    CHECK(path != NULL, "setup: write odd-junk-chunk-with-pad temp file");
    if (path) {
        WavReader* r = wav_open_read(path);
        CHECK(r != NULL, "wav_open_read must accept a valid file with an odd-sized junk chunk + pad before data");
        if (r) {
            CHECK(r->info.num_samples == 4, "num_samples must be 4 after correctly skipping the odd junk chunk + pad");
            float buf[4];
            int n = wav_read_float(r, buf, 4);
            CHECK(n == 4, "must read all 4 samples after the odd junk chunk");
            for (int i = 0; i < 4; i++) {
                float expect = (float)samples[i] / 32768.0f;
                CHECK(buf[i] == expect, "sample value must match after odd-junk-chunk pad handling");
            }
            wav_close_read(r);
        }
    }
    cleanup_temp(path);
    bb_free(&b);
}

static void test_accept_data_size_not_multiple_of_block_align(void) {
    /* data chunk_sz = 2*3 + 1 = 7 bytes (block_align=2, mono 16-bit): one
     * trailing partial byte. Must not crash; num_samples must floor to 3,
     * and reading must stop cleanly at 3 samples. */
    ByteBuf b; bb_init(&b);
    bb_riff_header(&b, 100);
    bb_fmt_chunk(&b, 16, 1, 1, 16000, 32000, 2, 16);
    bb_str4(&b, "data");
    bb_u32(&b, 7);
    bb_u16(&b, 10); bb_u16(&b, 20); bb_u16(&b, 30);
    bb_u8(&b, 0x7F);  /* trailing partial sample byte */
    char* path = write_temp_wav(&b);
    CHECK(path != NULL, "setup: write non-block-aligned data-size temp file");
    if (path) {
        WavReader* r = wav_open_read(path);
        CHECK(r != NULL, "wav_open_read must accept a data chunk_sz not a multiple of block_align");
        if (r) {
            CHECK(r->info.num_samples == 3, "num_samples must floor-divide when the trailing bytes are a partial sample");
            float buf[8];
            int n = wav_read_float(r, buf, 8);
            CHECK(n == 3, "wav_read_float must stop at the floor-divided sample count, not crash on the trailing byte");
            wav_close_read(r);
        }
    }
    cleanup_temp(path);
    bb_free(&b);
}

static void test_float_nan_inf_sanitized(void) {
    ByteBuf b; bb_init(&b);
    float samples[5];
    samples[0] = 0.25f;
    samples[1] = NAN;
    samples[2] = -0.5f;
    samples[3] = INFINITY;
    samples[4] = -INFINITY;
    uint32_t data_sz = (uint32_t)(5 * sizeof(float));
    bb_riff_header(&b, 36 + data_sz);
    bb_fmt_chunk(&b, 16, 3 /* IEEE float */, 1, 48000, 48000 * 4, 4, 32);
    bb_str4(&b, "data");
    bb_u32(&b, data_sz);
    for (int i = 0; i < 5; i++) bb_push(&b, &samples[i], sizeof(float));
    char* path = write_temp_wav(&b);
    CHECK(path != NULL, "setup: write float32 NaN/Inf temp file");
    if (path) {
        WavReader* r = wav_open_read(path);
        CHECK(r != NULL, "wav_open_read must accept a well-formed float32 file containing NaN/Inf");
        if (r) {
            CHECK(r->info.is_float == 1, "is_float must be true for a format==3 file");
            float buf[5];
            int n = wav_read_float(r, buf, 5);
            CHECK(n == 5, "must read all 5 float samples");
            CHECK(buf[0] == 0.25f, "finite sample before the NaN must be untouched");
            CHECK(buf[1] == 0.0f, "NaN sample must be sanitized to 0.0f");
            CHECK(buf[2] == -0.5f, "finite sample between non-finite ones must be untouched");
            CHECK(buf[3] == 0.0f, "+Inf sample must be sanitized to 0.0f");
            CHECK(buf[4] == 0.0f, "-Inf sample must be sanitized to 0.0f");
            CHECK(r->nonfinite_sanitized == 3, "nonfinite_sanitized must count exactly the 3 NaN/Inf samples");
            wav_close_read(r);
        }
    }
    cleanup_temp(path);
    bb_free(&b);
}

static void test_valid_pcm16_parses_as_before(void) {
    const int16_t samples[6] = { 0, 1, -1, 32767, -32768, 12345 };
    ByteBuf b = build_valid_pcm16(44100, samples, 6);
    char* path = write_temp_wav(&b);
    CHECK(path != NULL, "setup: write plain valid PCM16 temp file");
    if (path) {
        WavReader* r = wav_open_read(path);
        CHECK(r != NULL, "wav_open_read must accept a plain valid PCM16 mono file");
        if (r) {
            CHECK(r->info.sample_rate == 44100, "sample_rate must parse correctly");
            CHECK(r->info.channels == 1, "channels must parse correctly");
            CHECK(r->info.bits_per_sample == 16, "bits_per_sample must parse correctly");
            CHECK(r->info.is_float == 0, "is_float must be false for PCM16");
            CHECK(r->info.num_samples == 6, "num_samples must parse correctly");
            CHECK(r->nonfinite_sanitized == 0, "PCM16 must never sanitize anything");
            float buf[6];
            int n = wav_read_float(r, buf, 6);
            CHECK(n == 6, "must read all 6 samples");
            for (int i = 0; i < 6; i++) {
                float expect = (float)samples[i] / 32768.0f;
                CHECK(buf[i] == expect, "PCM16->float conversion must match the untouched (sample/32768.0f) formula");
            }
            wav_close_read(r);
        }
    }
    cleanup_temp(path);
    bb_free(&b);
}

/* ═══════════════════════════ writer round-trip test ═══════════════════════ */

static void test_write_read_roundtrip_byte_exact(void) {
    /* Values chosen to round exactly through *32768.0f round-half-away
     * quantization (the default WAV_IO_WRITER_AEC style, since this test
     * doesn't set WAV_IO_WRITER_STYLE) with no ambiguity at the .5 boundary.
     * NOTE: deliberately NOT testing exactly +1.0f here -- that hits a
     * pre-existing (not introduced by F06) latent UB in this quantizer:
     * scaled = 1.0f*32768.0f = 32768.0f, +0.5f = 32768.5f, and int16_t's
     * range is asymmetric ([-32768, 32767]), so casting 32768.5f to
     * int16_t discards a fractional part whose integral part (32768)
     * doesn't fit -- undefined behavior per C99 6.3.1.4p1. It's preserved
     * byte-for-byte (per the F06 writer-preservation contract) rather than
     * fixed here, but a *test* asserting a specific outcome from UB isn't a
     * meaningful conformance check (two call sites of the same UB
     * expression are free to differ, and were observed to in practice). */
    const float in_samples[6] = { 0.0f, 0.5f, -0.5f, 0.999969482421875f /* 32767/32768 */, -1.0f, 0.99993896484375f /* 32766/32768 */ };
    char path[64];
    snprintf(path, sizeof(path), "/tmp/audio_common_test_wav_io_wr_XXXXXX");
    int fd = mkstemp(path);
    CHECK(fd >= 0, "setup: mkstemp for write/read round-trip");
    if (fd >= 0) close(fd);  /* wav_open_write will fopen() the same path */

    WavWriter* w = wav_open_write(path, 16000, 1);
    CHECK(w != NULL, "wav_open_write must succeed");
    if (w) {
        wav_write_float(w, in_samples, 6);
        wav_close_write(w);

        WavReader* r = wav_open_read(path);
        CHECK(r != NULL, "wav_open_read must successfully read back the just-written file");
        if (r) {
            CHECK(r->info.sample_rate == 16000, "round-trip sample_rate must match");
            CHECK(r->info.channels == 1, "round-trip channels must match");
            CHECK(r->info.bits_per_sample == 16, "round-trip bits_per_sample must be 16 (default writer style)");
            CHECK(r->info.num_samples == 6, "round-trip num_samples must match");

            float out_samples[6];
            int n = wav_read_float(r, out_samples, 6);
            CHECK(n == 6, "round-trip must read back all 6 samples");

            /* Recompute the exact expected int16 encoding (round-half-away
             * from zero, matching wav_write_float's WAV_IO_WRITER_AEC path)
             * and compare the read-back float to that exact re-derived value
             * -- i.e. byte-exact through the quantization step, not just
             * "close". */
            for (int i = 0; i < 6; i++) {
                float s = in_samples[i];
                if (s > 1.0f) s = 1.0f;
                if (s < -1.0f) s = -1.0f;
                float scaled = s * 32768.0f;
                int16_t expect_s16 = (int16_t)(scaled >= 0 ? scaled + 0.5f : scaled - 0.5f);
                float expect = (float)expect_s16 / 32768.0f;
                CHECK(out_samples[i] == expect, "round-tripped sample must be byte-exact vs the re-derived int16 encoding");
            }
            wav_close_read(r);
        }
    }
    unlink(path);
}

/* ═════════════════════ writer special-values corpus (R01) ═════════════════
 * The shared corpus/expected-value logic lives in test_wav_writer_common.h
 * (included above) so it can run unchanged against BOTH writer styles --
 * this TU gets WAV_IO_WRITER_AEC (the default, since it doesn't set
 * WAV_IO_WRITER_STYLE, same as every other test in this file). The NR-style
 * run of the exact same corpus lives in test_wav_writer_nr_style.c.
 */
static void test_writer_special_values_corpus_aec_style(void) {
    if (test_wav_writer_special_values_corpus()) g_fail = 1;
}

/* Float mode (WAV_IO_WRITER_AEC + AEC_OUT_FLOAT=1) is a raw, unquantized
 * bit-exact passthrough -- unlike the PCM16 path, it deliberately does NOT
 * sanitize NaN/Inf (see wav_io.h's Writer hardening doc: that would hide
 * exactly the divergence this test-only path exists to surface). Verifying
 * "bit-exact, including the NaN's raw payload" therefore means reading the
 * on-disk bytes directly, NOT through wav_read_float (which sanitizes NaN/
 * Inf to 0.0f on ingress and would hide the very thing under test here).
 * This also asserts the R01 RIFF-size fix: the outer RIFF chunk_size
 * (bytes 4-7) must now equal file_size-8, and the data sub-chunk size
 * (bytes 40-43) must equal n*sizeof(float) -- both were already correct
 * for PCM16 output, but the outer size was wrong for float32 pre-fix.
 */
static void test_write_float_mode_bitexact_and_riff_size(void) {
    setenv("AEC_OUT_FLOAT", "1", 1);

    const int n = 9;
    float in_samples[9];
    in_samples[0] = 1.0f;
    in_samples[1] = -1.0f;
    in_samples[2] = nextafterf(1.0f, 0.0f);
    in_samples[3] = nextafterf(-1.0f, 0.0f);
    /* A NaN with a deliberately non-canonical payload bit pattern: if the
     * float write path (or a compiler/libc) silently canonicalized this
     * (unlike the PCM16 path's explicit, intentional sanitize-to-0.0f), the
     * raw on-disk bytes would differ from this exact pattern. */
    uint32_t custom_nan_bits = 0x7FC00123u;
    float custom_nan;
    memcpy(&custom_nan, &custom_nan_bits, sizeof(float));
    in_samples[4] = custom_nan;
    in_samples[5] = INFINITY;
    in_samples[6] = -INFINITY;
    in_samples[7] = 0.5f;
    in_samples[8] = -0.5f;

    char path[64];
    snprintf(path, sizeof(path), "/tmp/audio_common_test_wav_writer_f32_XXXXXX");
    int fd = mkstemp(path);
    CHECK(fd >= 0, "setup: mkstemp for float-mode bit-exact/RIFF-size test");
    if (fd >= 0) close(fd);

    WavWriter* w = wav_open_write(path, 48000, 1);
    CHECK(w != NULL, "wav_open_write must succeed (AEC_OUT_FLOAT=1)");
    if (w) {
        CHECK(w->info.is_float == 1, "AEC_OUT_FLOAT=1 must select float32 output");
        wav_write_float(w, in_samples, n);
        CHECK(w->write_error == 0, "a normal-sized float-mode write must not report write_error");
        CHECK(w->samples_written == (uint64_t)n, "float-mode samples_written must count all 9 samples");
        CHECK(w->nonfinite_sanitized == 0, "float mode must never sanitize -- it's a raw passthrough");
        wav_close_write(w);

        FILE* fp = fopen(path, "rb");
        CHECK(fp != NULL, "must be able to reopen the float-mode file for raw byte inspection");
        if (fp) {
            uint8_t header[44];
            size_t got_hdr = fread(header, 1, 44, fp);
            CHECK(got_hdr == 44, "must read the full 44-byte header");

            uint32_t riff_chunk_size, data_chunk_size;
            memcpy(&riff_chunk_size, &header[4], 4);
            memcpy(&data_chunk_size, &header[40], 4);

            long file_size = wav_io_file_size(fp);
            CHECK(file_size >= 44, "sanity: file must be at least header-sized");
            CHECK((uint64_t)riff_chunk_size == (uint64_t)file_size - 8,
                  "R01 fix: outer RIFF chunk_size (bytes 4-7) must equal file_size-8 in float32 mode");
            CHECK((uint64_t)data_chunk_size == (uint64_t)n * sizeof(float),
                  "data sub-chunk size (bytes 40-43) must equal n*sizeof(float)");

            float raw_samples[9];
            size_t got_data = fread(raw_samples, sizeof(float), n, fp);
            CHECK(got_data == (size_t)n, "must read back all 9 raw float samples");
            for (int i = 0; i < n; i++) {
                uint32_t a, b;
                memcpy(&a, &in_samples[i], 4);
                memcpy(&b, &raw_samples[i], 4);
                char msg[112];
                snprintf(msg, sizeof(msg),
                         "float-mode sample %d must be bit-exact passthrough (incl. NaN payload)", i);
                CHECK(a == b, msg);
            }
            fclose(fp);
        }
    }
    unlink(path);
    unsetenv("AEC_OUT_FLOAT");
}

/* n<=0 must be a no-op, not undersized/negative-count UB: the float-mode
 * fwrite() call passes n straight through as fwrite's nmemb argument, and a
 * negative int converted to size_t there would be a huge element count. */
static void test_write_zero_and_negative_n_is_noop(void) {
    const float dummy[1] = { 0.25f };

    char path[64];
    snprintf(path, sizeof(path), "/tmp/audio_common_test_wav_writer_nzero_XXXXXX");
    int fd = mkstemp(path);
    CHECK(fd >= 0, "setup: mkstemp for n<=0 no-op test");
    if (fd >= 0) close(fd);

    WavWriter* w = wav_open_write(path, 16000, 1);
    CHECK(w != NULL, "wav_open_write must succeed for n<=0 no-op test");
    if (w) {
        wav_write_float(w, dummy, 0);
        CHECK(w->samples_written == 0, "n==0 must write nothing");
        wav_write_float(w, dummy, -5);
        CHECK(w->samples_written == 0, "n<0 must write nothing (and must not crash)");
        CHECK(w->write_error == 0, "n<=0 no-op calls must not raise write_error");
        wav_close_write(w);
    }
    unlink(path);
}

/* Forces a genuine short fwrite() inside wav_write_float's PCM16 loop (via
 * RLIMIT_FSIZE) and checks write_error gets set, samples_written stops
 * exactly at the true on-disk count, and the process survives (SIGXFSZ is
 * explicitly ignored first -- its default disposition is to terminate the
 * process, which would defeat observing fwrite's short-write return here).
 *
 * Two things must happen in a specific order for this to actually exercise
 * the code path instead of quietly no-oping:
 *   1. The rlimit is applied only AFTER wav_open_write() returns, so the
 *      44-byte placeholder header (written inside wav_open_write, and not
 *      itself return-value-checked -- out of scope for this fix) isn't
 *      affected by it.
 *   2. w->fp is switched to unbuffered (setvbuf _IONBF) before the rlimit
 *      is applied and before any sample is written. Per-sample fwrite calls
 *      here are only 2 bytes each -- with the default fully-buffered mode,
 *      those just memcpy into libc's internal buffer and every fwrite call
 *      would report success regardless of the rlimit, with the eventual
 *      failure only surfacing (unobserved) at the buffer's first real
 *      flush. Unbuffered mode makes each 2-byte fwrite hit write(2)
 *      directly, so the 3rd sample's write can actually be observed
 *      failing exactly when the limit is crossed.
 */
static void test_write_error_on_short_write(void) {
    char path[64];
    snprintf(path, sizeof(path), "/tmp/audio_common_test_wav_writer_shortwr_XXXXXX");
    int fd = mkstemp(path);
    CHECK(fd >= 0, "setup: mkstemp for short-write test");
    if (fd < 0) return;
    close(fd);

    void (*old_handler)(int) = signal(SIGXFSZ, SIG_IGN);

    struct rlimit old_lim;
    int have_old_lim = (getrlimit(RLIMIT_FSIZE, &old_lim) == 0);

    const float samples[4] = { 0.1f, 0.2f, 0.3f, 0.4f };
    WavWriter* w = wav_open_write(path, 16000, 1);
    CHECK(w != NULL, "wav_open_write must succeed (no rlimit applied yet)");
    if (w) {
        setvbuf(w->fp, NULL, _IONBF, 0);

        // 44-byte header (already flushed above via the setvbuf-triggered
        // buffer switch) + exactly 2 PCM16 samples (4 bytes) fit; the 3rd
        // sample's fwrite must come back short once the limit is hit. Only
        // rlim_cur is lowered -- rlim_max is left at its original value (a
        // non-privileged process generally cannot raise rlim_max back up
        // once lowered, so this keeps the restore-on-exit below possible).
        struct rlimit lim;
        lim.rlim_cur = 48;
        lim.rlim_max = have_old_lim ? old_lim.rlim_max : RLIM_INFINITY;
        int lim_ok = (setrlimit(RLIMIT_FSIZE, &lim) == 0);
        CHECK(lim_ok, "setup: setrlimit(RLIMIT_FSIZE) must succeed for short-write test");

        if (lim_ok) {
            wav_write_float(w, samples, 4);
            CHECK(w->write_error == 1, "a write past the RLIMIT_FSIZE limit must set write_error");
            CHECK(w->samples_written == 2,
                  "samples_written must stop exactly at the count that actually made it to disk");
            if (have_old_lim) setrlimit(RLIMIT_FSIZE, &old_lim);
            // wav_close_write still finalizes a header from the true
            // samples_written -- must not crash even though the file is
            // right at its size limit (the header is an in-place overwrite
            // of bytes already on disk, not a growth past the old limit).
            wav_close_write(w);
        } else {
            wav_close_write(w);
        }
    }

    signal(SIGXFSZ, old_handler);
    unlink(path);
}

/* ═══════════════════════════════════ main ════════════════════════════════ */

int main(void) {
    test_reject_truncated_header();
    test_reject_fmt_size_12();
    test_reject_channels_0();
    test_reject_channels_too_many();
    test_reject_bits_0_1_7();
    test_reject_bits_24_pcm();
    test_reject_bits_32_int_pcm();
    test_reject_sample_rate_bounds();
    test_reject_block_align_mismatch();
    test_reject_byte_rate_mismatch();
    test_reject_data_chunk_size_exceeds_file();
    test_reject_oversized_unknown_chunk();

    test_accept_odd_junk_chunk_with_pad();
    test_accept_data_size_not_multiple_of_block_align();
    test_float_nan_inf_sanitized();
    test_valid_pcm16_parses_as_before();

    test_write_read_roundtrip_byte_exact();

    test_writer_special_values_corpus_aec_style();
    test_write_float_mode_bitexact_and_riff_size();
    test_write_zero_and_negative_n_is_noop();
    test_write_error_on_short_write();

    if (g_fail) {
        printf(">>> FAIL\n");
        return 1;
    }
    printf("ALL PASS\n>>> PASS\n");
    return 0;
}
