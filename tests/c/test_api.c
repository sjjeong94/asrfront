/* Minimal C API tests (no framework). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "asrfront.h"
#include "kernels.h"

extern const af_kernels af_kernels_generic, af_kernels_scalar;

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

/* channel_last: exactly the transpose of the default layout, for the plain and the
 * silence-skipping normalization, rows spaced apart (_ld, gap untouched) and _padded. */
static void test_whisper_channel_last(void) {
    static const size_t lens[] = {201, 16123, 479959, 480000, 500000};
    const size_t gap = 3;
    float *audio = malloc(500000 * sizeof(float));
    float *a = malloc(128 * 3125 * sizeof(float)), *b = malloc(3125 * (128 + gap) * sizeof(float));
    for (size_t i = 0; i < 500000; i++)
        audio[i] =
            0.3f * sinf(0.013f * (float)i) + 0.01f * (float)((int)(i * 7919 % 1000) - 500) / 500.0f;
    for (int n_mels = 80; n_mels <= 128; n_mels += 48) {
        for (int pad = 0; pad < 2; pad++) {
            af_whisper_config cfg;
            af_whisper *cf = NULL, *cl = NULL;
            af_whisper_default_config(&cfg);
            CHECK(cfg.channel_last == 0);
            cfg.n_mels = n_mels;
            cfg.pad_or_trim = pad;
            CHECK(af_whisper_create(&cfg, &cf) == AF_OK);
            cfg.channel_last = 1;
            CHECK(af_whisper_create(&cfg, &cl) == AF_OK);
            for (size_t i = 0; i < sizeof(lens) / sizeof(lens[0]); i++) {
                const size_t n = lens[i], frames = af_whisper_out_frames(cf, n),
                             ld = (size_t)n_mels + gap;
                CHECK(af_whisper_log_mel(cf, audio, n, a) == AF_OK);
                CHECK(af_whisper_log_mel(cl, audio, n, b) == AF_OK);
                int same = 1;
                for (size_t t = 0; t < frames; t++)
                    for (int m = 0; m < n_mels; m++)
                        same &= memcmp(&a[(size_t)m * frames + t], &b[t * n_mels + m], 4) == 0;
                CHECK(same);
                for (size_t j = 0; j < frames * ld; j++)
                    b[j] = 7.0f;
                CHECK(af_whisper_log_mel_ld(cl, audio, n, b, ld) == AF_OK);
                same = 1;
                for (size_t t = 0; t < frames; t++) {
                    for (int m = 0; m < n_mels; m++)
                        same &= memcmp(&a[(size_t)m * frames + t], &b[t * ld + m], 4) == 0;
                    for (size_t m = (size_t)n_mels; m < ld; m++)
                        same &= b[t * ld + m] == 7.0f;
                }
                CHECK(same);
                CHECK(af_whisper_log_mel_ld(cl, audio, n, b, (size_t)n_mels - 1) == AF_EINVAL);
            }
            if (!pad) { /* a short item of a padded batch skips silent frames too */
                const size_t n = 1000, n_padded = 500000, frames = n_padded / 160;
                CHECK(af_whisper_log_mel_padded(cf, audio, n, n_padded, a) == AF_OK);
                CHECK(af_whisper_log_mel_padded(cl, audio, n, n_padded, b) == AF_OK);
                int same = 1;
                for (size_t t = 0; t < frames; t++)
                    for (int m = 0; m < n_mels; m++)
                        same &= memcmp(&a[(size_t)m * frames + t], &b[t * n_mels + m], 4) == 0;
                CHECK(same);
            }
            af_whisper_destroy(cf);
            af_whisper_destroy(cl);
        }
    }
    free(audio);
    free(a);
    free(b);
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

/* af_gemma4_log_mel_channel_first: exactly the transpose of af_gemma4_log_mel, padding frames
 * (zeros) included. */
static void test_gemma4_channel_first(void) {
    static const size_t lens[] = {0, 160, 161, 1000, 16000, 16123};
    float *audio = malloc(16123 * sizeof(float));
    float *a = malloc(128 * 110 * sizeof(float)), *b = malloc(128 * 110 * sizeof(float));
    for (size_t i = 0; i < 16123; i++)
        audio[i] = 0.3f * sinf(0.017f * (float)i);
    af_gemma4 *h = NULL;
    CHECK(af_gemma4_create(&h) == AF_OK);
    for (size_t i = 0; i < sizeof(lens) / sizeof(lens[0]); i++) {
        const size_t n = lens[i];
        const size_t frames = af_gemma4_out_frames((n + 127) / 128 * 128) + 3; /* + zero frames */
        for (size_t j = 0; j < 128 * frames; j++)
            b[j] = 7.0f;
        CHECK(af_gemma4_log_mel(h, audio, n, frames, a) == AF_OK);
        CHECK(af_gemma4_log_mel_channel_first(h, audio, n, frames, b) == AF_OK);
        int same = 1;
        for (size_t t = 0; t < frames; t++)
            for (size_t m = 0; m < 128; m++)
                same &= memcmp(&a[t * 128 + m], &b[m * frames + t], 4) == 0;
        CHECK(same);
    }
    CHECK(af_gemma4_log_mel_channel_first(NULL, audio, 10, 1, b) == AF_EINVAL);
    af_gemma4_destroy(h);
    free(audio);
    free(a);
    free(b);
}

/* soa_to_rows and all_finite: every kernel table gives the plain loop's result. */
static void test_kernels_extra(void) {
    const af_kernels *tables[] = {&af_kernels_scalar, &af_kernels_generic, af_kernels_select()};
    float soa[131 * AF_VL], out[AF_VL * 140];
    for (int i = 0; i < 131 * AF_VL; i++)
        soa[i] = (float)i + 0.5f;
    float *x = malloc(5000 * sizeof(float));
    for (size_t k = 0; k < sizeof(tables) / sizeof(tables[0]); k++) {
        const af_kernels *t = tables[k];
        static const int ns[] = {1, 7, 8, 23, 80, 128, 131};
        for (size_t j = 0; j < sizeof(ns) / sizeof(ns[0]); j++)
            for (int count = 1; count <= AF_VL; count++) {
                const int n = ns[j];
                const size_t ld = (size_t)n + 9;
                for (size_t i = 0; i < sizeof(out) / sizeof(out[0]); i++)
                    out[i] = -1.0f;
                t->soa_to_rows(soa, n, count, out, ld);
                int ok = 1;
                for (int l = 0; l < AF_VL; l++)
                    for (size_t i = 0; i < ld && l * ld + i < sizeof(out) / sizeof(out[0]); i++) {
                        const float want = l < count && i < (size_t)n
                                               ? soa[i * AF_VL + (size_t)l]
                                               : -1.0f; /* outside the rows: untouched */
                        ok &= out[(size_t)l * ld + i] == want;
                    }
                CHECK(ok);
            }
        for (size_t i = 0; i < 5000; i++)
            x[i] = (float)i * 1e30f; /* large but finite */
        CHECK(t->all_finite(x, 5000) && t->all_finite(x, 0));
        static const size_t at[] = {0, 7, 1023, 1024, 1025, 4095, 4999};
        const float bad[] = {NAN, INFINITY, -INFINITY};
        for (size_t i = 0; i < sizeof(at) / sizeof(at[0]); i++)
            for (int v = 0; v < 3; v++) {
                const float keep = x[at[i]];
                x[at[i]] = bad[v];
                CHECK(!t->all_finite(x, 5000));
                CHECK(t->all_finite(x, at[i])); /* stops just before it */
                x[at[i]] = keep;
            }
    }
    x[4999] = NAN;
    CHECK(!af_all_finite(x, 5000) && af_all_finite(x, 4999));
    free(x);
}

static void test_misc(void) {
    CHECK(strcmp(af_version(), ASRFRONT_VERSION) == 0);
    CHECK(strcmp(af_strerror(AF_EINVAL), "invalid argument") == 0);
}

/* Streaming fbank: the rows of pushes of random sizes and the finish are af_sensevoice_fbank of
 * the whole input, bit for bit. */
static void test_sensevoice_stream(void) {
    static const int lfr[][2] = {{7, 6}, {1, 1}, {1, 6}, {11, 1}, {4, 9}, {7, 6}};
    const size_t lengths[] = {0, 399, 400, 401, 560, 1100, 16000, 37777};
    unsigned rng = 12345;
    float *audio = malloc(37777 * sizeof(float));
    for (size_t i = 0; i < 37777; i++) {
        rng = rng * 1103515245u + 12345u;
        audio[i] = (float)((int)(rng >> 9) % 20001 - 10000) / 32768.0f;
    }
    for (size_t c = 0; c < sizeof lfr / sizeof lfr[0]; c++) {
        af_sensevoice_config cfg;
        af_sensevoice_default_config(&cfg);
        cfg.lfr_m = lfr[c][0];
        cfg.lfr_n = lfr[c][1];
        const size_t dim = (size_t)cfg.n_mels * (size_t)cfg.lfr_m;
        float *shift = NULL, *scale = NULL;
        if (c == 0) { /* dither */
            cfg.dither = 1.0f;
            cfg.seed = 7;
        } else if (c == 5) { /* CMVN */
            shift = malloc(dim * sizeof(float));
            scale = malloc(dim * sizeof(float));
            for (size_t d = 0; d < dim; d++) {
                shift[d] = -(float)d * 0.01f;
                scale[d] = 0.5f + (float)d * 0.001f;
            }
            cfg.cmvn_shift = shift;
            cfg.cmvn_scale = scale;
        }
        af_sensevoice *h = NULL;
        af_sensevoice_stream *s = NULL;
        CHECK(af_sensevoice_create(&cfg, &h) == AF_OK);
        CHECK(af_sensevoice_stream_create(&cfg, &s) == AF_OK);
        if (!h || !s)
            return;
        CHECK(af_sensevoice_feat_dim(af_sensevoice_stream_frontend(s)) == dim);
        for (size_t li = 0; li < sizeof lengths / sizeof lengths[0]; li++) {
            const size_t n = lengths[li], rows = af_sensevoice_out_frames(h, n);
            float *want = malloc((rows + 1) * dim * sizeof(float));
            float *got = malloc((rows + 1) * dim * sizeof(float));
            CHECK(af_sensevoice_fbank(h, audio, n, want) == AF_OK);
            for (int rep = 0; rep < 3; rep++) { /* also: a finished stream starts over */
                size_t pos = 0, k = 0;
                while (pos < n) {
                    rng = rng * 1103515245u + 12345u;
                    size_t step = rep == 0 ? 1 + (rng >> 8) % 700 : rep == 1 ? n : 160;
                    step = step < n - pos ? step : n - pos;
                    const size_t expect = af_sensevoice_stream_push_rows(s, step);
                    size_t got_rows = (size_t)-1;
                    CHECK(af_sensevoice_stream_push(s, audio + pos, step, got + k * dim,
                                                    &got_rows) == AF_OK);
                    CHECK(got_rows == expect);
                    k += got_rows;
                    pos += step;
                }
                const size_t expect = af_sensevoice_stream_finish_rows(s);
                size_t got_rows = (size_t)-1;
                CHECK(af_sensevoice_stream_finish(s, got + k * dim, &got_rows) == AF_OK);
                CHECK(got_rows == expect);
                k += got_rows;
                CHECK(k == rows);
                CHECK(k != rows || memcmp(got, want, rows * dim * sizeof(float)) == 0);
            }
            free(want);
            free(got);
        }
        size_t r = 0;
        CHECK(af_sensevoice_stream_push(s, NULL, 1, NULL, &r) == AF_EINVAL);
        CHECK(af_sensevoice_stream_push(s, audio, 0, NULL, &r) == AF_OK && r == 0);
        CHECK(af_sensevoice_stream_push(s, audio, 2000, NULL, &r) == AF_EINVAL); /* rows due */
        af_sensevoice_stream_reset(s);
        CHECK(af_sensevoice_stream_finish_rows(s) == 0);
        af_sensevoice_stream_destroy(s);
        af_sensevoice_destroy(h);
        free(shift);
        free(scale);
    }
    af_sensevoice_stream *s = NULL;
    CHECK(af_sensevoice_stream_create(NULL, &s) == AF_EINVAL && s == NULL);
    af_sensevoice_stream_destroy(NULL);
    free(audio);
}

/* af_lfm3_audio_*: frame counts, the two layouts (exact transposes, rows ld apart with the
 * gaps untouched), frames past the input reading zeros, argument checks. */
static void test_lfm3_audio(void) {
    CHECK(af_lfm3_audio_out_frames(0) == 1 && af_lfm3_audio_out_frames(400) == 1);
    CHECK(af_lfm3_audio_out_frames(401) == 2 && af_lfm3_audio_out_frames(560) == 2);
    CHECK(af_lfm3_audio_out_frames(561) == 3 && af_lfm3_audio_out_frames(16000) == 99);
    CHECK(af_lfm3_audio_complete_frames(399) == 0 && af_lfm3_audio_complete_frames(400) == 1);
    CHECK(af_lfm3_audio_complete_frames(559) == 1 && af_lfm3_audio_complete_frames(560) == 2);
    const size_t n = 16123, frames = af_lfm3_audio_out_frames(n) + 3, ld = frames + 5;
    float *audio = malloc(n * sizeof(float));
    float *a = malloc(128 * ld * sizeof(float)), *b = malloc(frames * 128 * sizeof(float));
    for (size_t i = 0; i < n; i++)
        audio[i] = 0.3f * sinf(0.013f * (float)i);
    for (size_t i = 0; i < 128 * ld; i++)
        a[i] = 7.0f;
    af_lfm3_audio *h = NULL;
    CHECK(af_lfm3_audio_create(&h) == AF_OK);
    CHECK(af_lfm3_audio_log_mel(h, audio, n, frames, a, ld) == AF_OK);
    CHECK(af_lfm3_audio_log_mel_channel_last(h, audio, n, frames, b, 128) == AF_OK);
    int same = 1, gaps = 1, range = 1;
    for (size_t m = 0; m < 128; m++) {
        for (size_t t = 0; t < frames; t++) {
            same &= memcmp(&a[m * ld + t], &b[t * 128 + m], 4) == 0;
            range &= a[m * ld + t] >= -2.0f && a[m * ld + t] < 2.0f;
        }
        for (size_t t = frames; t < ld; t++)
            gaps &= a[m * ld + t] == 7.0f;
    }
    CHECK(same && gaps && range);
    /* the 3 frames past the padded input read only zeros: the floor, (-12 + 4) / 4 */
    CHECK(b[(frames - 1) * 128] == -2.0f && b[(frames - 1) * 128 + 127] == -2.0f);
    CHECK(af_lfm3_audio_log_mel(h, NULL, 0, 1, a, 1) == AF_OK && a[0] == -2.0f);
    CHECK(af_lfm3_audio_log_mel(h, NULL, 0, 0, NULL, 0) == AF_OK);
    CHECK(af_lfm3_audio_log_mel(h, audio, n, frames, a, frames - 1) == AF_EINVAL);
    CHECK(af_lfm3_audio_log_mel_channel_last(h, audio, n, frames, b, 127) == AF_EINVAL);
    CHECK(af_lfm3_audio_log_mel(h, NULL, 10, 1, a, 1) == AF_EINVAL);
    CHECK(af_lfm3_audio_log_mel(NULL, audio, n, 1, a, 1) == AF_EINVAL);
    CHECK(af_lfm3_audio_create(NULL) == AF_EINVAL);
    af_lfm3_audio_destroy(h);
    af_lfm3_audio_destroy(NULL);
    free(audio);
    free(a);
    free(b);
}

int main(void) {
    test_whisper();
    test_whisper_silence_skip();
    test_whisper_padded();
    test_whisper_ld();
    test_whisper_channel_last();
    test_sensevoice();
    test_sensevoice_stream();
    test_gemma4();
    test_gemma4_channel_first();
    test_lfm3_audio();
    test_kernels_extra();
    test_misc();
    if (failures) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("all C API tests passed\n");
    return 0;
}
