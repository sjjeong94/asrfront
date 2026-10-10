/* Minimal C API tests (no framework). */
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

static void test_misc(void) {
    CHECK(strcmp(af_version(), ASRFRONT_VERSION) == 0);
    CHECK(strcmp(af_strerror(AF_EINVAL), "invalid argument") == 0);
}

int main(void) {
    test_whisper();
    test_sensevoice();
    test_misc();
    if (failures) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("all C API tests passed\n");
    return 0;
}
