/* Audio loading tests on hand-crafted byte buffers (no framework). Comparisons against ffmpeg
 * itself are in tests/python/test_load.py. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "asrfront.h"
#include "audio.h"

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

    wav_head(&w, 1, 1, 44100, 16, 0); /* needs resampling (M9.3) */
    put(&w, "data\2\0\0\0\0\0", 10);
    CHECK(load(&w, AF_RAW_NEVER, &a, &n) == AF_EUNSUPPORTED);

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
    /* two consecutive valid frames (128 kbps, 44.1 kHz: 417 bytes) -> MP3 (not decoded yet) */
    memcpy(fake + 417, "\xFF\xFB\x90\x64", 4);
    CHECK(af_load_memory(fake, sizeof fake, AF_RAW_AUTO, &a, &n) == AF_EUNSUPPORTED);
    CHECK(af_load_memory("fLaC\0\0\0\x22", 8, AF_RAW_AUTO, &a, &n) == AF_EUNSUPPORTED);
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

static void test_file(void) {
    float *a = NULL;
    size_t n = 0;
    CHECK(af_load_file("/nonexistent/x.wav", &a, &n) == AF_EIO);
    CHECK(af_load_file(NULL, &a, &n) == AF_EINVAL);
}

int main(void) {
    test_wav_s16();
    test_wav_formats();
    test_wav_errors();
    test_raw_and_detection();
    test_layouts();
    test_file();
    if (failures) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("test_load: all passed\n");
    return 0;
}
