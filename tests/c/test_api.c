/* Minimal C API tests (no framework). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "asrfront.h"

static int failures = 0;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);               \
            failures++;                                                                            \
        }                                                                                          \
    } while (0)

static void test_whisper(void) {
    af_whisper_config cfg;
    af_whisper *h = NULL;

    af_whisper_default_config(&cfg);
    CHECK(af_whisper_create(&cfg, &h) == AF_OK && h);
    CHECK(af_whisper_out_frames(h, 0) == 3000);
    CHECK(af_whisper_out_frames(h, 1000000) == 3000);
    af_whisper_destroy(h);

    cfg.pad_or_trim = 0;
    CHECK(af_whisper_create(&cfg, &h) == AF_OK);
    CHECK(af_whisper_out_frames(h, 480000) == 3000);
    CHECK(af_whisper_out_frames(h, 16000) == 100);
    CHECK(af_whisper_out_frames(h, 159) == 0);
    af_whisper_destroy(h);

    cfg.n_mels = 64;
    CHECK(af_whisper_create(&cfg, &h) == AF_EINVAL && h == NULL);
    CHECK(af_whisper_create(NULL, &h) == AF_EINVAL);
}

/* pad_or_trim skips frames that read only zero padding: a short input must give exactly
 * the output of the same input zero-padded to 30 s by the caller (nothing skipped). */
static void test_whisper_silence_skip(void) {
    static const size_t lens[] = {0,      1,      100,    159,    160,    161,    199,   200,
                                  201,    399,    400,    401,    2361,   2400,   16000, 80000,
                                  479000, 479758, 479800, 479959, 479960, 479999, 480000};
    const size_t n_virt = 480000;
    float *audio = malloc(n_virt * sizeof(float)), *padded = malloc(n_virt * sizeof(float));
    float *a = malloc(128 * 3000 * sizeof(float)), *b = malloc(128 * 3000 * sizeof(float));
    for (size_t i = 0; i < n_virt; i++)
        audio[i] =
            0.3f * sinf(0.01f * (float)i) + 0.01f * (float)((int)(i * 7919 % 1000) - 500) / 500.0f;
    for (int n_mels = 80; n_mels <= 128; n_mels += 48) {
        af_whisper_config cfg;
        af_whisper *h = NULL;
        af_whisper_default_config(&cfg);
        cfg.n_mels = n_mels;
        CHECK(af_whisper_create(&cfg, &h) == AF_OK);
        for (size_t i = 0; i < sizeof(lens) / sizeof(lens[0]); i++) {
            const size_t n = lens[i];
            memcpy(padded, audio, n * sizeof(float));
            memset(padded + n, 0, (n_virt - n) * sizeof(float));
            CHECK(af_whisper_log_mel(h, audio, n, a) == AF_OK);
            CHECK(af_whisper_log_mel(h, padded, n_virt, b) == AF_OK);
            if (memcmp(a, b, (size_t)n_mels * 3000 * sizeof(float)) != 0) {
                fprintf(stderr, "silence skip mismatch: n_mels=%d n=%zu\n", n_mels, n);
                failures++;
            }
        }
        af_whisper_destroy(h);
    }
    free(audio);
    free(padded);
    free(a);
    free(b);
}

/* af_whisper_log_mel_padded: the same output as an explicitly zero-padded input (no skipped
 * blocks) for paddings around a frame, a hop and the reflection length, where skipped blocks
 * meet frames that reflect back into the input. */
static void test_whisper_padded(void) {
    static const size_t lens[] = {201, 400, 1000, 1601, 16000, 16123};
    static const size_t extra[] = {0,   1,   40,  159, 160,  161,  199,  200,  201,   239,
                                   240, 241, 360, 361, 1280, 1281, 1500, 2000, 16000, 50000};
    const size_t cap = 16123 + 50000;
    float *audio = malloc(cap * sizeof(float)), *padded = malloc(cap * sizeof(float));
    float *a = malloc(128 * (cap / 160) * sizeof(float)),
          *b = malloc(128 * (cap / 160) * sizeof(float));
    for (size_t i = 0; i < cap; i++)
        audio[i] =
            0.3f * sinf(0.013f * (float)i) + 0.01f * (float)((int)(i * 7919 % 1000) - 500) / 500.0f;
    af_whisper_config cfg;
    af_whisper *h = NULL;
    af_whisper_default_config(&cfg);
    cfg.n_mels = 128;
    cfg.pad_or_trim = 0;
    CHECK(af_whisper_create(&cfg, &h) == AF_OK);
    for (size_t i = 0; i < sizeof(lens) / sizeof(lens[0]); i++) {
        for (size_t j = 0; j < sizeof(extra) / sizeof(extra[0]); j++) {
            const size_t n = lens[i], n_padded = n + extra[j];
            memcpy(padded, audio, n * sizeof(float));
            memset(padded + n, 0, (n_padded - n) * sizeof(float));
            CHECK(af_whisper_log_mel_padded(h, audio, n, n_padded, a) == AF_OK);
            CHECK(af_whisper_log_mel(h, padded, n_padded, b) == AF_OK);
            if (memcmp(a, b, 128 * (n_padded / 160) * sizeof(float)) != 0) {
                fprintf(stderr, "padded mismatch: n=%zu n_padded=%zu\n", n, n_padded);
                failures++;
            }
        }
    }
    /* tiny inputs in a long padding: all but the first frames are silent */
    CHECK(af_whisper_log_mel_padded(h, audio, 0, 16000, a) == AF_OK);
    CHECK(af_whisper_log_mel_padded(h, audio, 1, 16000, a) == AF_OK);
    CHECK(af_whisper_log_mel_padded(h, audio, 10, 10, a) == AF_EINVAL);     /* <= 200 */
    CHECK(af_whisper_log_mel_padded(h, audio, 1001, 1000, a) == AF_EINVAL); /* longer */
    CHECK(af_whisper_log_mel_padded(h, NULL, 10, 1000, a) == AF_EINVAL);
    af_whisper_destroy(h);
    af_whisper_default_config(&cfg); /* pad_or_trim handles have their own padding */
    CHECK(af_whisper_create(&cfg, &h) == AF_OK);
    CHECK(af_whisper_log_mel_padded(h, audio, 1000, 2000, a) == AF_EINVAL);
    af_whisper_destroy(h);
    free(audio);
    free(padded);
    free(a);
    free(b);
}

static void test_sensevoice(void) {
    af_sensevoice_config cfg;
    af_sensevoice *h = NULL;

    af_sensevoice_default_config(&cfg);
    CHECK(af_sensevoice_create(&cfg, &h) == AF_OK && h);
    CHECK(af_sensevoice_feat_dim(h) == 560);
    CHECK(af_sensevoice_raw_frames(h, 399) == 0);
    CHECK(af_sensevoice_raw_frames(h, 400) == 1);
    CHECK(af_sensevoice_raw_frames(h, 16000) == 98);
    CHECK(af_sensevoice_out_frames(h, 16000) == 17); /* ceil(98 / 6) */
    CHECK(af_sensevoice_out_frames(h, 0) == 0);
    af_sensevoice_destroy(h);

    /* CMVN vectors are copied: caller buffers may be freed after create. */
    float *shift = calloc(560, sizeof(float));
    float *scale = calloc(560, sizeof(float));
    cfg.cmvn_shift = shift;
    cfg.cmvn_scale = scale;
    CHECK(af_sensevoice_create(&cfg, &h) == AF_OK);
    free(shift);
    free(scale);
    af_sensevoice_destroy(h);

    cfg.cmvn_scale = NULL; /* shift without scale */
    CHECK(af_sensevoice_create(&cfg, &h) == AF_EINVAL && h == NULL);

    af_sensevoice_default_config(&cfg);
    cfg.lfr_n = 0;
    CHECK(af_sensevoice_create(&cfg, &h) == AF_EINVAL);
}

/* af_whisper_log_mel_ld: each row as af_whisper_log_mel's, the gap after it untouched. Covers
 * the plain and the silence-skipping normalization (pad_or_trim with a short input). */
static void test_whisper_ld(void) {
    const size_t n = 16123, gap = 5;
    float *audio = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++)
        audio[i] = 0.3f * sinf(0.013f * (float)i);
    for (int pad = 0; pad < 2; pad++) {
        af_whisper_config cfg;
        af_whisper *h = NULL;
        af_whisper_default_config(&cfg);
        cfg.pad_or_trim = pad;
        CHECK(af_whisper_create(&cfg, &h) == AF_OK);
        const size_t frames = af_whisper_out_frames(h, n), ld = frames + gap;
        float *a = malloc(80 * frames * sizeof(float)), *b = malloc(80 * ld * sizeof(float));
        for (size_t i = 0; i < 80 * ld; i++)
            b[i] = 7.0f;
        CHECK(af_whisper_log_mel(h, audio, n, a) == AF_OK);
        CHECK(af_whisper_log_mel_ld(h, audio, n, b, ld) == AF_OK);
        for (size_t m = 0; m < 80; m++) {
            CHECK(memcmp(a + m * frames, b + m * ld, frames * sizeof(float)) == 0);
            for (size_t t = frames; t < ld; t++)
                CHECK(b[m * ld + t] == 7.0f);
        }
        CHECK(af_whisper_log_mel_ld(h, audio, n, b, frames - 1) == AF_EINVAL);
        af_whisper_destroy(h);
        free(a);
        free(b);
    }
    free(audio);
}

static void test_gemma4(void) {
    CHECK(af_gemma4_out_frames(0) == 0 && af_gemma4_out_frames(160) == 0);
    CHECK(af_gemma4_out_frames(161) == 1 && af_gemma4_out_frames(16000) == 99);
    CHECK(af_gemma4_valid_frames(0) == 0 && af_gemma4_valid_frames(160) == 0);
    CHECK(af_gemma4_valid_frames(161) == 1 && af_gemma4_valid_frames(16001) == 100);
    const size_t n = 16123, frames = af_gemma4_out_frames(16128) + 3; /* 3 extra: zeros */
    float *audio = malloc(n * sizeof(float)), *out = malloc(frames * 128 * sizeof(float));
    for (size_t i = 0; i < n; i++)
        audio[i] = 0.3f * sinf(0.013f * (float)i);
    for (size_t i = 0; i < frames * 128; i++)
        out[i] = 7.0f;
    af_gemma4 *h = NULL;
    CHECK(af_gemma4_create(&h) == AF_OK);
    CHECK(af_gemma4_log_mel(h, audio, n, frames, out) == AF_OK);
    const size_t valid = af_gemma4_valid_frames(n);
    int finite = 1, zeros = 1;
    for (size_t i = 0; i < valid * 128; i++)
        finite &= isfinite(out[i]) && out[i] != 7.0f && out[i] > -7.0f;
    for (size_t i = valid * 128; i < frames * 128; i++)
        zeros &= out[i] == 0.0f;
    CHECK(finite && zeros);
    CHECK(af_gemma4_log_mel(h, audio, n, 2, out) == AF_OK); /* fewer frames than valid */
    CHECK(af_gemma4_log_mel(h, NULL, 0, 0, NULL) == AF_OK);
    CHECK(af_gemma4_log_mel(h, NULL, 10, 1, out) == AF_EINVAL);
    CHECK(af_gemma4_log_mel(NULL, audio, n, 1, out) == AF_EINVAL);
    CHECK(af_gemma4_create(NULL) == AF_EINVAL);
    af_gemma4_destroy(h);
    af_gemma4_destroy(NULL);
    free(audio);
    free(out);
}

static void test_misc(void) {
    CHECK(strcmp(af_version(), ASRFRONT_VERSION) == 0);
    CHECK(strcmp(af_strerror(AF_EINVAL), "invalid argument") == 0);
}

int main(void) {
    test_whisper();
    test_whisper_silence_skip();
    test_whisper_padded();
    test_whisper_ld();
    test_sensevoice();
    test_gemma4();
    test_misc();
    if (failures) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("all C API tests passed\n");
    return 0;
}
