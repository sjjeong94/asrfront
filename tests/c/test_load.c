/* Audio loading tests on hand-crafted byte buffers (no framework). Comparisons against ffmpeg
 * itself are in tests/python/test_load.py. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "asrfront.h"
#include "audio.h"
#include "resample.h"

static int failures = 0;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);               \
            failures++;                                                                            \
        }                                                                                          \
    } while (0)

/* --- tiny WAV writer ------------------------------------------------------------------------ */

typedef struct {
    uint8_t b[4096];
    size_t n;
} buf_t;

static void put(buf_t *w, const void *p, size_t n) {
    memcpy(w->b + w->n, p, n);
    w->n += n;
}
static void put16(buf_t *w, unsigned v) {
    uint8_t b[2] = {(uint8_t)v, (uint8_t)(v >> 8)};
    put(w, b, 2);
}
static void put32(buf_t *w, uint32_t v) {
    uint8_t b[4] = {(uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24)};
    put(w, b, 4);
}

/* RIFF header + fmt chunk (PCM / float / extensible when mask != 0). */
static void wav_head(buf_t *w, unsigned tag, unsigned ch, uint32_t sr, unsigned bits,
                     uint32_t mask) {
    w->n = 0;
    put(w, "RIFF", 4);
    put32(w, 0); /* RIFF size is ignored */
    put(w, "WAVE", 4);
    put(w, "fmt ", 4);
    put32(w, mask ? 40 : 16);
    put16(w, mask ? 0xFFFE : tag);
    put16(w, ch);
    put32(w, sr);
    put32(w, sr * ch * bits / 8);
    put16(w, ch * bits / 8);
    put16(w, bits);
    if (mask) {
        put16(w, 22);
        put16(w, bits);
        put32(w, mask);
        put16(w, tag);
        put(w, "\x00\x00\x00\x00\x10\x00\x80\x00\x00\xAA\x00\x38\x9B\x71", 14);
    }
}

static int load(const buf_t *w, af_raw_mode raw, float **out, size_t *n) {
    return af_load_memory(w->b, w->n, raw, out, n);
}

static void test_wav_s16(void) {
    buf_t w;
    wav_head(&w, 1, 1, 16000, 16, 0);
    put(&w, "LIST", 4); /* unknown chunk with odd size + pad byte before data */
    put32(&w, 3);
    put(&w, "abc\0", 4);
    put(&w, "data", 4);
    put32(&w, 8);
    put16(&w, 0);
    put16(&w, 0x4000);
    put16(&w, 0x8000);
    put16(&w, 0x7FFF);
    float *a = NULL;
    size_t n = 0;
    CHECK(load(&w, AF_RAW_NEVER, &a, &n) == AF_OK && n == 4);
    if (a && n == 4)
        CHECK(a[0] == 0.0f && a[1] == 0.5f && a[2] == -1.0f && a[3] == 32767.0f / 32768.0f);
    af_audio_free(a);

    /* data size 0 / 0xFFFFFFFF / too large: read to the end, partial frame dropped */
    static const uint32_t sizes[] = {0, 0xFFFFFFFFu, 1000};
    for (int i = 0; i < 3; i++) {
        wav_head(&w, 1, 1, 16000, 16, 0);
        put(&w, "data", 4);
        put32(&w, sizes[i]);
        put16(&w, 1);
        put16(&w, 2);
        put(&w, "\x03", 1);
        CHECK(load(&w, AF_RAW_NEVER, &a, &n) == AF_OK && n == 2);
        af_audio_free(a);
    }
}

static void test_wav_formats(void) {
    buf_t w;
    float *a = NULL;
    size_t n = 0;

    wav_head(&w, 1, 1, 16000, 8, 0); /* u8 */
    put(&w, "data", 4);
    put32(&w, 2);
    put(&w, "\x80\xC0", 2);
    CHECK(load(&w, AF_RAW_AUTO, &a, &n) == AF_OK && n == 2 && a[0] == 0.0f && a[1] == 0.5f);
    af_audio_free(a);

    wav_head(&w, 1, 1, 16000, 24, 0); /* s24 -> s32 >> 16 = s24 >> 8 (floor) */
    put(&w, "data", 4);
    put32(&w, 6);
    put(&w, "\xFF\x7F\x00\x01\x00\x80", 6); /* 0x007FFF -> 127, 0x800001 -> -32768 */
    CHECK(load(&w, AF_RAW_AUTO, &a, &n) == AF_OK && n == 2);
    if (a && n == 2)
        CHECK(a[0] == 127.0f / 32768.0f && a[1] == -1.0f);
    af_audio_free(a);

    wav_head(&w, 3, 1, 16000, 32, 0); /* f32: round half to even, saturate */
    put(&w, "data", 4);
    put32(&w, 12);
    float f[3] = {2.5f / 32768.0f, 3.5f / 32768.0f, 2.0f};
    for (int i = 0; i < 3; i++) {
        uint32_t u;
        memcpy(&u, &f[i], 4);
        put32(&w, u);
    }
    CHECK(load(&w, AF_RAW_AUTO, &a, &n) == AF_OK && n == 3);
    if (a && n == 3)
        CHECK(a[0] == 2.0f / 32768.0f && a[1] == 4.0f / 32768.0f && a[2] == 32767.0f / 32768.0f);
    af_audio_free(a);

    /* extensible stereo s16: (L + R + 1) >> 1 */
    wav_head(&w, 1, 2, 16000, 16, AF_CH_FL | AF_CH_FR);
    put(&w, "data", 4);
    put32(&w, 8);
    put16(&w, 1);
    put16(&w, 0);
    put16(&w, 0xFFFF); /* -1 */
    put16(&w, 0xFFFE); /* -2 */
    CHECK(load(&w, AF_RAW_AUTO, &a, &n) == AF_OK && n == 2);
    if (a && n == 2)
        CHECK(a[0] == 1.0f / 32768.0f && a[1] == -1.0f / 32768.0f);
    af_audio_free(a);

    /* RF64: data size taken from ds64 */
    w.n = 0;
    put(&w, "RF64", 4);
    put32(&w, 0xFFFFFFFFu);
    put(&w, "WAVE", 4);
    put(&w, "ds64", 4);
    put32(&w, 28);
    put32(&w, 0), put32(&w, 0); /* riff size */
    put32(&w, 4), put32(&w, 0); /* data size */
    put32(&w, 2), put32(&w, 0); /* sample count */
    put32(&w, 0);               /* table length */
    buf_t h;
    wav_head(&h, 1, 1, 16000, 16, 0);
    put(&w, h.b + 12, h.n - 12);
    put(&w, "data", 4);
    put32(&w, 0xFFFFFFFFu);
    put16(&w, 5);
    put16(&w, 6);
    put16(&w, 7); /* beyond the ds64 data size */
    CHECK(load(&w, AF_RAW_NEVER, &a, &n) == AF_OK && n == 2);
    af_audio_free(a);
}

static void test_wav_errors(void) {
    buf_t w;
    float *a = (float *)&w; /* must be reset to NULL on error */
    size_t n = 7;

    wav_head(&w, 1, 1, 16000, 16, 0); /* no data chunk */
    CHECK(load(&w, AF_RAW_NEVER, &a, &n) == AF_ECORRUPT && a == NULL && n == 0);

    w.n = 0; /* data before fmt */
    put(&w, "RIFF\0\0\0\0WAVEdata\2\0\0\0\0\0", 22);
    CHECK(load(&w, AF_RAW_NEVER, &a, &n) == AF_ECORRUPT);

    wav_head(&w, 1, 0, 16000, 16, 0); /* zero channels */
    put(&w, "data\0\0\0\0", 8);
    CHECK(load(&w, AF_RAW_NEVER, &a, &n) == AF_ECORRUPT);

    wav_head(&w, 1, 1, 16000, 12, 0); /* odd bit depth */
    put(&w, "data\0\0\0\0", 8);
    CHECK(load(&w, AF_RAW_NEVER, &a, &n) == AF_EUNSUPPORTED);

    wav_head(&w, 0x55, 1, 16000, 16, 0); /* MP3-in-WAV codec tag */
    put(&w, "data\0\0\0\0", 8);
    CHECK(load(&w, AF_RAW_NEVER, &a, &n) == AF_EUNSUPPORTED);

    wav_head(&w, 1, 1, 500, 16, 0); /* sample rate outside the resampler's range */
    put(&w, "data\2\0\0\0\0\0", 10);
    CHECK(load(&w, AF_RAW_NEVER, &a, &n) == AF_EUNSUPPORTED);

    wav_head(&w, 1, 1, 48000, 16, 0); /* resampled: 300 -> ceil((300 - 1) / 3) = 100 */
    put(&w, "data", 4);
    put32(&w, 600);
    for (int i = 0; i < 300; i++)
        put16(&w, 1000);
    CHECK(load(&w, AF_RAW_NEVER, &a, &n) == AF_OK && n == 100);
    if (a && n == 100) /* DC is preserved by phase 0 */
        CHECK(a[50] == 1000.0f / 32768.0f);
    af_audio_free(a);

    wav_head(&w, 1, 1, 16000, 16, 0); /* chunk claiming more than the file */
    put(&w, "LIST\xFF\xFF\x00\x00", 8);
    CHECK(load(&w, AF_RAW_NEVER, &a, &n) == AF_ECORRUPT);

    /* every truncation of a valid file fails cleanly or decodes a prefix */
    wav_head(&w, 1, 2, 16000, 16, 0);
    put(&w, "data\x10\0\0\0", 8);
    for (int i = 0; i < 16; i++)
        put(&w, "\x11", 1);
    size_t full = w.n;
    for (size_t len = 0; len < full; len++) {
        int rc = af_load_memory(w.b, len, AF_RAW_NEVER, &a, &n);
        CHECK(rc == AF_OK || rc == AF_ECORRUPT || rc == AF_EUNSUPPORTED);
        af_audio_free(a);
    }
}

static void test_limit(void) {
    float *a = NULL;
    size_t n = 0;
    buf_t w;
    wav_head(&w, 1, 2, 16000, 16, 0);
    put(&w, "data\x20\0\0\0", 8);
    for (int i = 0; i < 32; i++)
        put(&w, "\x11", 1); /* 8 stereo frames */
    CHECK(af_load_memory_ex(w.b, w.n, AF_RAW_NEVER, 8, &a, &n, NULL, 0) == AF_OK && n == 8);
    af_audio_free(a);
    CHECK(af_load_memory_ex(w.b, w.n, AF_RAW_NEVER, 0, &a, &n, NULL, 0) == AF_OK && n == 8);
    af_audio_free(a);
    CHECK(af_load_memory_ex(w.b, w.n, AF_RAW_NEVER, 7, &a, &n, NULL, 0) == AF_ETOOLONG && !a &&
          n == 0);
    CHECK(af_load_memory_ex(w.b, 44 + 28, AF_RAW_NEVER, 7, &a, &n, NULL, 0) == AF_OK && n == 7);
    af_audio_free(a);
    const uint8_t pcm[6] = {0};
    CHECK(af_load_memory_ex(pcm, 6, AF_RAW_FORCE, 2, &a, &n, NULL, 0) == AF_ETOOLONG);
    CHECK(strcmp(af_strerror(AF_ETOOLONG), "audio longer than the limit") == 0);

    /* af_resample_max_in is the exact inverse of af_resampler_out_len */
    static const int rates[] = {1000,  8000,  11025, 22050, 32000,
                                32001, 44100, 48000, 96000, 384000};
    for (size_t i = 0; i < sizeof rates / sizeof rates[0]; i++) {
        af_resampler *r = NULL;
        CHECK(af_resampler_create(rates[i], 16000, &r) == AF_OK);
        if (!r)
            continue;
        for (size_t lim = 0; lim < 500; lim++) {
            size_t m = af_resample_max_in(rates[i], 16000, lim);
            CHECK(af_resampler_out_len(r, m) <= lim && af_resampler_out_len(r, m + 1) > lim);
        }
        af_resampler_destroy(r);
    }
    CHECK(af_resample_max_in(384000, 16000, SIZE_MAX / 2) == SIZE_MAX); /* no overflow */
}

/* af_load_memory_ex's error description: what is not supported, or which container is broken */
static void expect_err(const void *b, size_t n, af_raw_mode raw, int want_rc, const char *want) {
    float *a = NULL;
    size_t len = 0;
    char err[128];
    int rc = af_load_memory_ex(b, n, raw, 0, &a, &len, err, sizeof err);
    if (rc != want_rc || strcmp(err, want) != 0) {
        fprintf(stderr, "error description: rc %d, \"%s\"; want rc %d, \"%s\"\n", rc, err, want_rc,
                want);
        failures++;
    }
    af_audio_free(a);
}

static void test_error_details(void) {
    buf_t w;
    wav_head(&w, 0x55, 2, 44100, 0, 0);
    put(&w, "data\x04\0\0\0\0\0\0\0", 12);
    expect_err(w.b, w.n, AF_RAW_NEVER, AF_EUNSUPPORTED, "WAV: format tag 0x0055 (MP3)");
    wav_head(&w, 1, 1, 16000, 12, 0);
    put(&w, "data\x04\0\0\0\0\0\0\0", 12);
    expect_err(w.b, w.n, AF_RAW_NEVER, AF_EUNSUPPORTED, "WAV: 12-bit PCM");
    wav_head(&w, 1, 1, 500, 16, 0);
    put(&w, "data\x04\0\0\0\0\0\0\0", 12);
    expect_err(w.b, w.n, AF_RAW_NEVER, AF_EUNSUPPORTED,
               "WAV: sample rate 500 Hz (1000 to 384000 supported)");
    wav_head(&w, 1, 10, 16000, 16, 0);
    put(&w, "data\x14\0\0\0", 8);
    for (int i = 0; i < 20; i++)
        put(&w, "\0", 1);
    expect_err(w.b, w.n, AF_RAW_NEVER, AF_EUNSUPPORTED,
               "WAV: 10 channels without a channel mask (at most 8)");
    expect_err(w.b, 30, AF_RAW_NEVER, AF_ECORRUPT, "WAV"); /* truncated header */
    expect_err("not audio", 9, AF_RAW_NEVER, AF_EUNSUPPORTED, "no known container header");
    expect_err("\0\0\0 ftypM4A ", 12, AF_RAW_NEVER, AF_EUNSUPPORTED, "MP4 / M4A container");
    expect_err("OggS\0\2", 6, AF_RAW_NEVER, AF_EUNSUPPORTED, "Ogg (Vorbis / Opus) container");
    /* bytes input: a known unsupported container is an error, not raw PCM noise */
    expect_err("\0\0\0 ftypM4A ", 12, AF_RAW_AUTO, AF_EUNSUPPORTED, "MP4 / M4A container");
    expect_err("RIFF\0\0\0\0AVI LIST", 16, AF_RAW_AUTO, AF_EUNSUPPORTED, "AVI container");
    const uint8_t adts[16] = {0xFF, 0xF1, 0x50, 0x80, 0x01, 0x00, 0xFC, 0,
                              0xFF, 0xF1, 0x50, 0x80, 0x01, 0x00, 0xFC, 0}; /* two 8-byte frames */
    expect_err(adts, sizeof adts, AF_RAW_AUTO, AF_EUNSUPPORTED, "AAC (ADTS) container");
    expect_err(adts, 8, AF_RAW_AUTO, AF_OK, ""); /* one sync word alone may be PCM */
    expect_err("fLaC\0\0\0\x22", 8, AF_RAW_AUTO, AF_ECORRUPT, "FLAC");
    /* bytes from the file are sanitized to printable ASCII */
    uint8_t caf[64] = "caff\0\1\0\0desc\0\0\0\0\0\0\0\x20";
    memcpy(caf + 20, "\x40\xCF\x40\0\0\0\0\0", 8); /* 16000.0 */
    memcpy(caf + 28, "\x01\xFFxy", 4);             /* format */
    caf[47] = 1;                                   /* channels */
    expect_err(caf, sizeof caf, AF_RAW_NEVER, AF_EUNSUPPORTED, "CAF: format '??xy'");
    /* success and other codes leave it empty */
    wav_head(&w, 1, 1, 16000, 16, 0);
    put(&w, "data\x04\0\0\0\0\0\0\0", 12);
    expect_err(w.b, w.n, AF_RAW_NEVER, AF_OK, "");
}

static void test_raw_and_detection(void) {
    float *a = NULL;
    size_t n = 0;
    const uint8_t pcm[5] = {0x00, 0x40, 0xFF, 0xFF, 0x12};
    CHECK(af_load_memory(pcm, 5, AF_RAW_FORCE, &a, &n) == AF_OK && n == 2);
    if (a && n == 2)
        CHECK(a[0] == 0.5f && a[1] == -1.0f / 32768.0f);
    af_audio_free(a);
    CHECK(af_load_memory(pcm, 5, AF_RAW_NEVER, &a, &n) == AF_EUNSUPPORTED);
    CHECK(af_load_memory(NULL, 0, AF_RAW_AUTO, &a, &n) == AF_OK && n == 0 && a != NULL);
    af_audio_free(a);
    CHECK(af_load_memory(NULL, 0, AF_RAW_NEVER, &a, &n) == AF_EUNSUPPORTED);
    CHECK(af_load_memory(pcm, 5, (af_raw_mode)7, &a, &n) == AF_EINVAL);

    /* raw PCM that looks like an MPEG sync (int16 -1 = FF FF, and FF FB 90 64 runs) stays raw */
    uint8_t ones[4096];
    memset(ones, 0xFF, sizeof ones);
    CHECK(af_load_memory(ones, sizeof ones, AF_RAW_AUTO, &a, &n) == AF_OK && n == 2048);
    af_audio_free(a);
    uint8_t fake[4096] = {0};
    memcpy(fake, "\xFF\xFB\x90\x64", 4); /* one valid Layer III header, no second frame */
    CHECK(af_load_memory(fake, sizeof fake, AF_RAW_AUTO, &a, &n) == AF_OK && n == 2048);
    af_audio_free(a);
    /* two consecutive valid frames (128 kbps, 44.1 kHz: 417 bytes) -> decoded as MP3, not raw */
    memcpy(fake + 417, "\xFF\xFB\x90\x64", 4);
    int rc = af_load_memory(fake, sizeof fake, AF_RAW_AUTO, &a, &n);
    CHECK((rc == AF_OK && n != 2048) || rc == AF_ECORRUPT);
    af_audio_free(a);
    /* FLAC magic with a broken stream: an error, not raw PCM */
    CHECK(af_load_memory("fLaC\0\0\0\x22", 8, AF_RAW_AUTO, &a, &n) == AF_ECORRUPT);
    CHECK(af_load_memory("fLaC\0\0\0\x22", 8, AF_RAW_FORCE, &a, &n) == AF_OK && n == 4);
    af_audio_free(a);
}

static void test_layouts(void) {
    uint32_t lay[AF_MAX_CHANNELS];
    double w[AF_MAX_CHANNELS];
    CHECK(af_default_layout(9, lay) == AF_EUNSUPPORTED);
    CHECK(af_default_layout(6, lay) == AF_OK && lay[3] == AF_CH_LFE);
    af_downmix_weights(lay, 6, w);
    double sum = 0;
    for (int i = 0; i < 6; i++)
        sum += w[i];
    CHECK(sum > 0.999999 && sum < 1.000001 && w[3] == 0.0);
    CHECK(af_mask_layout(AF_CH_FL | AF_CH_FR, 3, lay) == AF_EUNSUPPORTED);
    CHECK(af_mask_layout(AF_CH_FL | AF_CH_FR | AF_CH_LFE, 2, lay) == AF_EUNSUPPORTED);
    CHECK(af_mask_layout(AF_CH_BL | AF_CH_FC, 2, lay) == AF_OK && lay[0] == AF_CH_FC);
}

/* --- other containers -------------------------------------------------------------------- */

static void put16be(buf_t *w, unsigned v) {
    uint8_t b[2] = {(uint8_t)(v >> 8), (uint8_t)v};
    put(w, b, 2);
}
static void put32be(buf_t *w, uint32_t v) {
    uint8_t b[4] = {(uint8_t)(v >> 24), (uint8_t)(v >> 16), (uint8_t)(v >> 8), (uint8_t)v};
    put(w, b, 4);
}
static void put64be(buf_t *w, uint64_t v) {
    put32be(w, (uint32_t)(v >> 32));
    put32be(w, (uint32_t)v);
}

/* every truncation of a valid file fails cleanly or decodes at most the full length */
static void check_truncations(const buf_t *w) {
    float *a = NULL;
    size_t n = 0, full = 0;
    CHECK(af_load_memory(w->b, w->n, AF_RAW_NEVER, &a, &full) == AF_OK);
    af_audio_free(a);
    for (size_t len = 0; len < w->n; len++) {
        int rc = af_load_memory(w->b, len, AF_RAW_NEVER, &a, &n);
        CHECK(rc == AF_OK || rc == AF_ECORRUPT || rc == AF_EUNSUPPORTED);
        CHECK(rc != AF_OK || n <= full);
        af_audio_free(a);
    }
}

/* AIFF / AIFF-C at 16 kHz: COMM (+ AIFC compression type), then SSND with `n` body bytes */
static void aiff_head(buf_t *w, const char *comp, unsigned ch, unsigned bits, uint32_t n) {
    w->n = 0;
    put(w, "FORM", 4);
    put32be(w, 0);
    put(w, comp ? "AIFC" : "AIFF", 4);
    put(w, "COMM", 4);
    put32be(w, comp ? 24 : 18);
    put16be(w, ch);
    put32be(w, 0); /* frame count: not used */
    put16be(w, bits);
    put(w, "\x40\x0C\xFA\x00\x00\x00\x00\x00\x00\x00", 10); /* 16000.0 */
    if (comp) {
        put(w, comp, 4);
        put16be(w, 0); /* empty name */
    }
    put(w, "SSND", 4);
    put32be(w, 8 + n);
    put32be(w, 0); /* offset */
    put32be(w, 0); /* block size */
}

static void test_aiff(void) {
    buf_t w;
    float *a = NULL;
    size_t n = 0;
    aiff_head(&w, NULL, 1, 16, 4);
    put16be(&w, 0x4000);
    put16be(&w, 0x8000);
    CHECK(load(&w, AF_RAW_AUTO, &a, &n) == AF_OK && n == 2);
    if (a && n == 2)
        CHECK(a[0] == 0.5f && a[1] == -1.0f);
    af_audio_free(a);
    check_truncations(&w);

    aiff_head(&w, "sowt", 1, 16, 2); /* little-endian */
    put16(&w, 0x4000);
    CHECK(load(&w, AF_RAW_NEVER, &a, &n) == AF_OK && n == 1 && a[0] == 0.5f);
    af_audio_free(a);

    aiff_head(&w, NULL, 1, 8, 2); /* 8-bit AIFF is signed */
    put(&w, "\x40\xC0", 2);
    CHECK(load(&w, AF_RAW_NEVER, &a, &n) == AF_OK && n == 2);
    if (a && n == 2)
        CHECK(a[0] == 0.5f && a[1] == -0.5f);
    af_audio_free(a);

    /* QuickTime IMA: header predictor 0x1000 (step index 0), nibble 4 adds step 7 + 7/8 */
    aiff_head(&w, "ima4", 1, 16, 34);
    put16be(&w, 0x1000);
    put(&w, "\x04", 1);
    for (int i = 0; i < 31; i++)
        put(&w, "\x00", 1);
    CHECK(load(&w, AF_RAW_NEVER, &a, &n) == AF_OK && n == 64);
    if (a && n == 64)
        CHECK(a[0] == (4096.0f + 7.0f) / 32768.0f);
    af_audio_free(a);
    check_truncations(&w);

    aiff_head(&w, "QDM2", 1, 16, 2);
    put16be(&w, 0);
    CHECK(load(&w, AF_RAW_NEVER, &a, &n) == AF_EUNSUPPORTED);

    aiff_head(&w, NULL, 1, 16, 2);
    w.n -= 12;
    put(&w, "SSND\0\0\0\0", 8); /* SSND size 0 */
    CHECK(load(&w, AF_RAW_NEVER, &a, &n) == AF_ECORRUPT);
}

static void caf_head(buf_t *w, const char *format, uint32_t flags, unsigned ch, unsigned bits,
                     int64_t data_size) {
    w->n = 0;
    put(w, "caff\0\1\0\0desc", 12);
    put64be(w, 32);
    put(w, "\x40\xCF\x40\x00\x00\x00\x00\x00", 8); /* 16000.0 */
    put(w, format, 4);
    put32be(w, flags);
    put32be(w, ch * ((bits + 7) / 8)); /* bytes per packet */
    put32be(w, 1);                     /* frames per packet */
    put32be(w, ch);
    put32be(w, bits);
    put(w, "data", 4);
    put64be(w, (uint64_t)data_size);
    put32be(w, 0); /* edit count */
}

static void test_caf(void) {
    buf_t w;
    float *a = NULL;
    size_t n = 0;
    caf_head(&w, "lpcm", 2, 2, 16, 4 + 8); /* little-endian stereo s16 */
    put16(&w, 0x4000), put16(&w, 0x4000);
    put16(&w, 0xC000), put16(&w, 0xC000);
    CHECK(load(&w, AF_RAW_AUTO, &a, &n) == AF_OK && n == 2);
    if (a && n == 2)
        CHECK(a[0] == 0.5f && a[1] == -0.5f);
    af_audio_free(a);
    check_truncations(&w);

    caf_head(&w, "lpcm", 1, 1, 32, -1); /* big-endian float, streamed (size -1) */
    put32be(&w, 0x3F000000);            /* 0.5f */
    CHECK(load(&w, AF_RAW_NEVER, &a, &n) == AF_OK && n == 1 && a[0] == 0.5f);
    af_audio_free(a);

    caf_head(&w, "aac ", 0, 1, 16, -1);
    CHECK(load(&w, AF_RAW_NEVER, &a, &n) == AF_EUNSUPPORTED);
    memcpy(w.b + 20 + 8, "lpcm", 4); /* desc body at 20: rate, format, flags, ... */
    memset(w.b + 20 + 16, 0, 4);     /* bytes per packet 0 */
    CHECK(load(&w, AF_RAW_NEVER, &a, &n) == AF_ECORRUPT);
}

static void test_sphere(void) {
    buf_t w;
    float *a = NULL;
    size_t n = 0;
    static const char head[] = "NIST_1A\n   1024\nchannel_count -i 1\nsample_rate -i 16000\n"
                               "sample_n_bytes -i 2\nsample_byte_format -s2 10\n"
                               "sample_coding -s3 pcm\nend_head\n";
    w.n = 0;
    put(&w, head, sizeof head - 1);
    while (w.n < 1024)
        put(&w, " ", 1);
    put16be(&w, 0x4000);
    put16be(&w, 0xC000);
    CHECK(load(&w, AF_RAW_AUTO, &a, &n) == AF_OK && n == 2);
    if (a && n == 2)
        CHECK(a[0] == 0.5f && a[1] == -0.5f);
    af_audio_free(a);
    check_truncations(&w);

    memcpy(strstr((char *)w.b, "-s3 pcm"), "-s3 xyz", 7);
    CHECK(load(&w, AF_RAW_NEVER, &a, &n) == AF_EUNSUPPORTED);
}

static void test_w64(void) {
    static const uint8_t tail[12] = {0xF3, 0xAC, 0xD3, 0x11, 0x8C, 0xD1,
                                     0x00, 0xC0, 0x4F, 0x8E, 0xDB, 0x8A};
    buf_t w, h;
    float *a = NULL;
    size_t n = 0;
    w.n = 0;
    put(&w, af_w64_riff_guid, 16);
    put32(&w, 0), put32(&w, 0);
    put(&w, "wave", 4), put(&w, tail, 12);
    put(&w, "fmt ", 4), put(&w, tail, 12);
    put32(&w, 24 + 16), put32(&w, 0);
    wav_head(&h, 1, 1, 16000, 16, 0);
    put(&w, h.b + 20, 16); /* the WAVEFORMAT fields */
    put(&w, "data", 4), put(&w, tail, 12);
    put32(&w, 24 + 4), put32(&w, 0);
    put16(&w, 0x4000);
    put16(&w, 0xC000);
    CHECK(load(&w, AF_RAW_AUTO, &a, &n) == AF_OK && n == 2);
    if (a && n == 2)
        CHECK(a[0] == 0.5f && a[1] == -0.5f);
    af_audio_free(a);
    check_truncations(&w);
}

static void test_adpcm(void) {
    buf_t w;
    float *a = NULL;
    size_t n = 0;
    /* IMA ADPCM WAV, one 8-byte block: header (predictor 100, step index 0) + 8 nibbles.
     * Nibble 7: ((2 * 7 + 1) * 7) >> 3 = 13 */
    wav_head(&w, 0x11, 1, 16000, 4, 0);
    w.b[32] = 8; /* block align */
    w.b[33] = 0;
    put(&w, "data", 4);
    put32(&w, 8);
    put16(&w, 100);
    put(&w, "\0\0\x07\0\0\0", 6);
    CHECK(load(&w, AF_RAW_NEVER, &a, &n) == AF_OK && n == 9);
    if (a && n == 9)
        CHECK(a[0] == 100.0f / 32768.0f && a[1] == 113.0f / 32768.0f);
    af_audio_free(a);
    check_truncations(&w);
    w.b[w.n - 6] = 89; /* step index out of range */
    CHECK(load(&w, AF_RAW_NEVER, &a, &n) == AF_ECORRUPT);

    /* MS ADPCM, one 9-byte block: predictor 3 (192, 64), delta 16, sample1 1000, sample2 -1;
     * (1000 * 192 - 1 * 64) / 256 = 749.75 rounds toward zero, + 1 * 16 */
    wav_head(&w, 0x02, 1, 16000, 4, 0);
    w.b[32] = 9;
    w.b[33] = 0;
    put(&w, "data", 4);
    put32(&w, 9);
    put(&w, "\x03", 1);
    put16(&w, 16);
    put16(&w, 1000);
    put16(&w, 0xFFFF);
    put(&w, "\x10", 1);
    CHECK(load(&w, AF_RAW_NEVER, &a, &n) == AF_OK && n == 4);
    if (a && n == 4)
        CHECK(a[0] == -1.0f / 32768.0f && a[1] == 1000.0f / 32768.0f && a[2] == 765.0f / 32768.0f);
    af_audio_free(a);
    check_truncations(&w);
}

static void test_coreaudio_layouts(void) {
    uint8_t chan[12 + 3 * 20] = {0};
    uint32_t lay[AF_MAX_CHANNELS];
    chan[1] = 108, chan[3] = 4; /* kAudioChannelLayoutTag_Quadraphonic */
    CHECK(af_coreaudio_layout(chan, 12, 4, lay) == AF_OK && lay[2] == AF_CH_SL);
    CHECK(af_coreaudio_layout(chan, 12, 3, lay) == AF_OK && lay[2] == AF_CH_LFE); /* count */
    CHECK(af_coreaudio_layout(chan, 11, 4, lay) == AF_ECORRUPT);
    memset(chan, 0, sizeof chan);
    chan[1] = 1, chan[7] = 0x07; /* channel bitmap FL FR FC */
    CHECK(af_coreaudio_layout(chan, 12, 3, lay) == AF_OK && lay[2] == AF_CH_FC);
    memset(chan, 0, sizeof chan);
    chan[11] = 3; /* descriptions: C, L, (label 38: Lt -> no position) */
    chan[12 + 3] = 3, chan[32 + 3] = 1, chan[52 + 3] = 38;
    CHECK(af_coreaudio_layout(chan, sizeof chan, 3, lay) == AF_OK && lay[0] == AF_CH_FC &&
          lay[1] == AF_CH_FL && lay[2] == 0);
    CHECK(af_coreaudio_layout(chan, sizeof chan - 1, 3, lay) == AF_OK && lay[2] == AF_CH_LFE);
}

static void test_file(void) {
    float *a = NULL;
    size_t n = 0;
    CHECK(af_load_file("/nonexistent/x.wav", &a, &n) == AF_EIO);
    CHECK(af_load_file(NULL, &a, &n) == AF_EINVAL);
}

static void test_resampler_cache(void) {
    /* Reused while cached; private when every slot is in use or the bank is too large. */
    af_resampler *a = NULL, *b = NULL;
    CHECK(af_resampler_acquire(44100, 16000, AF_BANK_F32, &a) == AF_OK && a);
    CHECK(af_resampler_acquire(44100, 16000, AF_BANK_F32, &b) == AF_OK && b == a);
    af_resampler_release(b);
    af_resampler_release(a);
    CHECK(af_resampler_acquire(44100, 16000, AF_BANK_F32, &b) == AF_OK &&
          b == a); /* idle, still cached */
    af_resampler_release(b);

    const int busy_rates[4] = {11025, 22050, 32000, 48000};
    af_resampler *busy[4];
    for (int i = 0; i < 4; i++)
        CHECK(af_resampler_acquire(busy_rates[i], 16000, AF_BANK_F32, &busy[i]) == AF_OK);
    CHECK(af_resampler_acquire(44100, 16000, AF_BANK_F32, &a) ==
          AF_OK); /* evicted: all slots busy */
    CHECK(af_resampler_acquire(44100, 16000, AF_BANK_F32, &b) == AF_OK && b != a);
    CHECK(af_resampler_taps(a) == 92 && af_resampler_taps(b) == 92);
    af_resampler_release(a);
    af_resampler_release(b);
    af_resampler *again = NULL;
    CHECK(af_resampler_acquire(48000, 16000, AF_BANK_F32, &again) == AF_OK && again == busy[3]);
    af_resampler_release(again);
    for (int i = 0; i < 4; i++)
        af_resampler_release(busy[i]);
    CHECK(af_resampler_acquire(44100, 16000, AF_BANK_F32, &a) == AF_OK); /* evicts an idle entry */
    CHECK(af_resampler_acquire(44100, 16000, AF_BANK_F32, &b) == AF_OK && b == a);
    af_resampler_release(a);
    af_resampler_release(b);

    /* interpolated mode at 44101 Hz: only the float bank (0.4 MB) counts, so it is cached */
    CHECK(af_resampler_acquire(44101, 16000, AF_BANK_F32, &a) == AF_OK && !af_resampler_exact(a));
    CHECK(af_resampler_acquire(44101, 16000, AF_BANK_F32, &b) == AF_OK && b == a);
    af_resampler_release(b);
    /* another path's bank: a separate resampler, then cached as well */
    CHECK(af_resampler_acquire(44101, 16000, AF_BANK_S16, &b) == AF_OK && b != a);
    af_resampler_release(b);
    af_resampler *c = NULL;
    CHECK(af_resampler_acquire(44101, 16000, AF_BANK_S16, &c) == AF_OK && c == b);
    af_resampler_release(c);
    af_resampler_release(a);

    /* interpolated mode at 383999 Hz: ~3.2 MB of float filters, never cached */
    CHECK(af_resampler_acquire(383999, 16000, AF_BANK_F32, &a) == AF_OK && !af_resampler_exact(a));
    CHECK(af_resampler_acquire(383999, 16000, AF_BANK_F32, &b) == AF_OK && b != a);
    af_resampler_release(a);
    af_resampler_release(b);

    CHECK(af_resampler_acquire(999, 16000, AF_BANK_F32, &a) == AF_EUNSUPPORTED && a == NULL);
    CHECK(af_resampler_acquire(44100, 16000, AF_BANK_F32, NULL) == AF_EINVAL);
    CHECK(af_resampler_acquire(44100, 16000, 0, &a) == AF_EINVAL && a == NULL);
    CHECK(af_resampler_acquire(44100, 16000, 8, &a) == AF_EINVAL && a == NULL);
}

static void test_resampler_limits(void) {
    /* Filters longer than the edge buffers (low output rates from high input rates) are
     * refused instead of overflowing them; every input rate works for 16 kHz output. */
    af_resampler *r = NULL;
    CHECK(af_resampler_create(384000, 1000, &r) == AF_EUNSUPPORTED && r == NULL);
    CHECK(af_resampler_create(384000, 15000, &r) == AF_EUNSUPPORTED && r == NULL);
    CHECK(af_resampler_create(384000, 16000, &r) == AF_OK && af_resampler_taps(r) == 792);
    float x[2000] = {0}, y[100];
    af_resample_f32(r, x, 2000, y); /* edge outputs use the whole buffer */
    af_resampler_destroy(r);
    CHECK(af_resampler_create(383999, 16000, &r) == AF_OK && !af_resampler_exact(r));
    af_resample_f32(r, x, 2000, y);
    double xd[2000] = {0}, yd[100];
    af_resample_f64(r, xd, 2000, yd);
    int32_t xq[2000] = {0}, yq[100];
    af_resample_s16(r, xq, 2000, yq);
    af_resampler_destroy(r);

    /* a resampler with one bank computes the same as one with all of them */
    af_resampler *all = NULL, *f32 = NULL;
    CHECK(af_resampler_create(44101, 16000, &all) == AF_OK);
    CHECK(af_resampler_create_banks(44101, 16000, AF_BANK_F32, &f32) == AF_OK);
    float s[4000], ya[1500], yb[1500];
    for (int i = 0; i < 4000; i++)
        s[i] = (float)((i * 7919 % 2001) - 1000) / 1000.0f;
    CHECK(af_resampler_out_len(all, 4000) <= 1500);
    af_resample_f32(all, s, 4000, ya);
    af_resample_f32(f32, s, 4000, yb);
    CHECK(memcmp(ya, yb, af_resampler_out_len(all, 4000) * sizeof(float)) == 0);
    af_resampler_destroy(all);
    af_resampler_destroy(f32);
}

int main(void) {
    test_wav_s16();
    test_wav_formats();
    test_wav_errors();
    test_raw_and_detection();
    test_limit();
    test_error_details();
    test_layouts();
    test_aiff();
    test_caf();
    test_sphere();
    test_w64();
    test_adpcm();
    test_coreaudio_layouts();
    test_file();
    test_resampler_cache();
    test_resampler_limits();
    if (failures) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("test_load: all passed\n");
    return 0;
}
