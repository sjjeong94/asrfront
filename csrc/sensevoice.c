#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "asrfront.h"
#include "fft.h"
#include "kernels.h"
#include "melbank.h"

#define KALDI_FRAME_LENGTH 400 /* 25 ms @ 16 kHz */
#define KALDI_FRAME_SHIFT 160  /* 10 ms @ 16 kHz */
#define KALDI_PADDED 512       /* round_to_power_of_two */
#define KALDI_N_BINS (KALDI_PADDED / 2 + 1)
#define KALDI_PREEMPH 0.97f
#define KALDI_LOW_FREQ 20.0
#define KALDI_HIGH_FREQ 0.0  /* nyquist */
#define INPUT_SCALE 32768.0f /* FunASR: waveform * (1 << 15) */

struct af_sensevoice {
    af_sensevoice_config cfg; /* cmvn pointers point into cmvn below */
    float *cmvn;              /* [shift | scale], each n_mels * lfr_m; NULL if unused */
    const af_kernels *k;
    af_rfft *fft;
    af_melbank *mel;
    unsigned long long rng;
    float window[KALDI_FRAME_LENGTH];
    /* AF_VL-frame SoA work buffers */
    float frames[KALDI_FRAME_LENGTH * AF_VL];
    float re[KALDI_PADDED / 2 * AF_VL], im[KALDI_PADDED / 2 * AF_VL];
    float scr_re[KALDI_PADDED / 2 * AF_VL], scr_im[KALDI_PADDED / 2 * AF_VL];
    float power[KALDI_N_BINS * AF_VL];
    float *melv;    /* n_mels * AF_VL */
    float *raw;     /* grow-only buffer for raw fbank before LFR */
    size_t raw_cap; /* in floats */
};

void af_sensevoice_default_config(af_sensevoice_config *cfg) {
    cfg->n_mels = 80;
    cfg->lfr_m = 7;
    cfg->lfr_n = 6;
    cfg->dither = 0.0f;
    cfg->seed = 0;
    cfg->cmvn_shift = NULL;
    cfg->cmvn_scale = NULL;
    cfg->n_threads = 1;
}

int af_sensevoice_create(const af_sensevoice_config *cfg, af_sensevoice **out) {
    if (!cfg || !out)
        return AF_EINVAL;
    *out = NULL;
    if (cfg->n_mels <= 0 || cfg->lfr_m <= 0 || cfg->lfr_n <= 0 || !(cfg->dither >= 0.0f) ||
        cfg->n_threads < 0 || (cfg->cmvn_shift == NULL) != (cfg->cmvn_scale == NULL))
        return AF_EINVAL;

    af_sensevoice *h = calloc(1, sizeof(*h));
    if (!h)
        return AF_ENOMEM;
    h->cfg = *cfg;

    if (cfg->cmvn_shift) {
        size_t dim = (size_t)cfg->n_mels * (size_t)cfg->lfr_m;
        h->cmvn = malloc(2 * dim * sizeof(float));
        if (!h->cmvn) {
            free(h);
            return AF_ENOMEM;
        }
        memcpy(h->cmvn, cfg->cmvn_shift, dim * sizeof(float));
        memcpy(h->cmvn + dim, cfg->cmvn_scale, dim * sizeof(float));
        h->cfg.cmvn_shift = h->cmvn;
        h->cfg.cmvn_scale = h->cmvn + dim;
    }

    h->k = af_kernels_select();
    h->melv = malloc((size_t)cfg->n_mels * AF_VL * sizeof(float));
    int rc = h->melv ? af_rfft_create(KALDI_PADDED, &h->fft) : AF_ENOMEM;
    if (rc == AF_OK)
        rc = af_melbank_kaldi(AF_SAMPLE_RATE, KALDI_PADDED, cfg->n_mels, KALDI_LOW_FREQ,
                              KALDI_HIGH_FREQ, &h->mel);
    if (rc != AF_OK) {
        af_sensevoice_destroy(h);
        return rc;
    }
    /* torch.hamming_window(400, periodic=False, alpha=0.54, beta=0.46) */
    for (int i = 0; i < KALDI_FRAME_LENGTH; i++)
        h->window[i] = (float)(0.54 - 0.46 * cos(6.283185307179586476925286766559 * i /
                                                 (KALDI_FRAME_LENGTH - 1)));
    *out = h;
    return AF_OK;
}

void af_sensevoice_destroy(af_sensevoice *h) {
    if (!h)
        return;
    af_rfft_destroy(h->fft);
    af_melbank_destroy(h->mel);
    free(h->melv);
    free(h->raw);
    free(h->cmvn);
    free(h);
}

size_t af_sensevoice_raw_frames(const af_sensevoice *h, size_t n_samples) {
    (void)h;
    if (n_samples < KALDI_FRAME_LENGTH)
        return 0;
    return 1 + (n_samples - KALDI_FRAME_LENGTH) / KALDI_FRAME_SHIFT;
}

size_t af_sensevoice_out_frames(const af_sensevoice *h, size_t n_samples) {
    size_t t = af_sensevoice_raw_frames(h, n_samples);
    size_t n = (size_t)h->cfg.lfr_n;
    return (t + n - 1) / n;
}

size_t af_sensevoice_feat_dim(const af_sensevoice *h) {
    return (size_t)h->cfg.n_mels * (size_t)h->cfg.lfr_m;
}

/* ---------------------------------------------------------------------------------- */
/* Dither RNG: splitmix64 seeding, xorshift64*, Box-Muller                            */
/* ---------------------------------------------------------------------------------- */

static unsigned long long splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

static double rng_uniform(unsigned long long *s) { /* (0, 1] */
    *s ^= *s >> 12;
    *s ^= *s << 25;
    *s ^= *s >> 27;
    unsigned long long r = *s * 0x2545F4914F6CDD1Dull;
    return ((double)(r >> 11) + 1.0) / 9007199254740992.0;
}

/* x[i * stride] += dither * N(0, 1), i < n */
static void add_dither(unsigned long long *s, float *x, int n, int stride, float dither) {
    for (int i = 0; i < n; i += 2) {
        double r = sqrt(-2.0 * log(rng_uniform(s)));
        double a = 6.283185307179586476925286766559 * rng_uniform(s);
        x[(size_t)i * stride] += dither * (float)(r * cos(a));
        if (i + 1 < n)
            x[(size_t)(i + 1) * stride] += dither * (float)(r * sin(a));
    }
}

/* ---------------------------------------------------------------------------------- */
/* Kaldi fbank                                                                         */
/* ---------------------------------------------------------------------------------- */

/* Log-mel energies of frames t0 .. t0 + count - 1 into out[(t - t0) * n_mels + m]. */
static void fbank_block(af_sensevoice *h, const float *audio, size_t t0, int count, float *out) {
    const af_kernels *k = h->k;
    const int n_mels = h->cfg.n_mels;
    float *f = h->frames;

    /* Gather frames into lanes, scaled to int16 range. Dither is drawn lane by lane, in
     * frame order, so the RNG sequence matches frame-by-frame processing. Unused lanes of
     * the last block repeat lane 0. */
    const float *src[AF_VL];
    for (int l = 0; l < AF_VL; l++)
        src[l] = audio + (t0 + (size_t)(l < count ? l : 0)) * KALDI_FRAME_SHIFT;
    k->gather_scaled(src, KALDI_FRAME_LENGTH, INPUT_SCALE, f);
    if (h->cfg.dither > 0.0f)
        for (int l = 0; l < count; l++)
            add_dither(&h->rng, f + l, KALDI_FRAME_LENGTH, AF_VL, h->cfg.dither);

    k->kaldi_prep(f, KALDI_FRAME_LENGTH, h->window, KALDI_PREEMPH, KALDI_PADDED, h->re, h->im);
    k->rfft_power(h->fft, h->re, h->im, h->scr_re, h->scr_im, h->power);
    k->mel(h->mel, h->power, h->melv);
    k->log_floor(h->melv, (size_t)n_mels * AF_VL, FLT_EPSILON);
    for (int l = 0; l < count; l++)
        for (int m = 0; m < n_mels; m++)
            out[(size_t)l * n_mels + m] = h->melv[(size_t)m * AF_VL + l];
}

static void fbank_raw(af_sensevoice *h, const float *audio, size_t n_frames, float *out) {
    h->rng = splitmix64(h->cfg.seed); /* reseed per call: same input -> same output */
    for (size_t t0 = 0; t0 < n_frames; t0 += AF_VL) {
        const int count = n_frames - t0 < AF_VL ? (int)(n_frames - t0) : AF_VL;
        fbank_block(h, audio, t0, count, out + t0 * (size_t)h->cfg.n_mels);
    }
}

int af_sensevoice_fbank_raw(af_sensevoice *h, const float *audio, size_t n_samples, float *out) {
    if (!h || !out || (!audio && n_samples))
        return AF_EINVAL;
    fbank_raw(h, audio, af_sensevoice_raw_frames(h, n_samples), out);
    return AF_OK;
}

/*
 * LFR (FunASR apply_lfr): output row i concatenates raw frames
 * i * lfr_n - left + j, j < lfr_m, with left = (lfr_m - 1) / 2; indices before the
 * start repeat frame 0 and indices past the end repeat the last frame.
 * CMVN (FunASR apply_cmvn): (x + shift) * scale.
 */
int af_sensevoice_fbank(af_sensevoice *h, const float *audio, size_t n_samples, float *out) {
    if (!h || !out || (!audio && n_samples))
        return AF_EINVAL;
    const size_t n_raw = af_sensevoice_raw_frames(h, n_samples);
    const size_t n_out = af_sensevoice_out_frames(h, n_samples);
    if (n_raw == 0)
        return AF_OK;

    const size_t n_mels = (size_t)h->cfg.n_mels;
    const size_t need = n_raw * n_mels;
    if (need > h->raw_cap) {
        float *buf = realloc(h->raw, need * sizeof(float));
        if (!buf)
            return AF_ENOMEM;
        h->raw = buf;
        h->raw_cap = need;
    }
    fbank_raw(h, audio, n_raw, h->raw);

    const long m = h->cfg.lfr_m, n = h->cfg.lfr_n, left = (m - 1) / 2;
    const long last = (long)n_raw - 1;
    const size_t dim = n_mels * (size_t)m;
    const float *shift = h->cfg.cmvn_shift, *scale = h->cfg.cmvn_scale;
    for (size_t i = 0; i < n_out; i++) {
        float *row = out + i * dim;
        for (long j = 0; j < m; j++) {
            long src = (long)i * n - left + j;
            src = src < 0 ? 0 : (src > last ? last : src);
            memcpy(row + (size_t)j * n_mels, h->raw + (size_t)src * n_mels, n_mels * sizeof(float));
        }
        if (shift)
            for (size_t d = 0; d < dim; d++)
                row[d] = (row[d] + shift[d]) * scale[d];
    }
    return AF_OK;
}
