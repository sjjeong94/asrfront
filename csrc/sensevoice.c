#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "asrfront.h"
#include "fft.h"
#include "fpmode.h"
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
#define LFR_CHUNK 64         /* raw frames computed between LFR passes (a multiple of AF_VL) */

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
    float *melv; /* n_mels * AF_VL */
    /* Raw fbank frames for LFR: frame t at row t % ring_frames. LFR row i reads frames up to
     * lfr_m - 1 apart and frames are computed LFR_CHUNK at a time, so lfr_m + LFR_CHUNK - 1
     * frames (rounded up to whole blocks) are always enough, whatever the input length. */
    float *ring;
    size_t ring_frames;
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
    h->ring_frames = ((size_t)cfg->lfr_m + LFR_CHUNK + AF_VL - 2) / AF_VL * AF_VL;
    const size_t ring_floats = h->ring_frames * (size_t)cfg->n_mels;
    h->ring = ring_floats / h->ring_frames == (size_t)cfg->n_mels &&
                      ring_floats < SIZE_MAX / sizeof(float)
                  ? malloc(ring_floats * sizeof(float))
                  : NULL;
    int rc = h->melv && h->ring ? af_rfft_create(KALDI_PADDED, &h->fft) : AF_ENOMEM;
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
    free(h->ring);
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
    k->soa_to_rows(h->melv, n_mels, count, out, (size_t)n_mels);
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
    const unsigned fp = af_ftz_enter();
    fbank_raw(h, audio, af_sensevoice_raw_frames(h, n_samples), out);
    af_ftz_leave(fp);
    return AF_OK;
}

/* Raw frames [*done, end) of the signal whose frame t starts at audio + (t - base) * shift,
 * into the ring, in blocks of AF_VL. A block also stops at the end of the ring (a stream's
 * frames need not start at a multiple of AF_VL); frames do not depend on their block. */
static void compute_frames(af_sensevoice *h, const float *audio, size_t base, size_t *done,
                           size_t end) {
    const size_t n_mels = (size_t)h->cfg.n_mels;
    while (*done < end) {
        size_t count = end - *done < AF_VL ? end - *done : AF_VL;
        const size_t room = h->ring_frames - *done % h->ring_frames;
        count = count < room ? count : room;
        fbank_block(h, audio, *done - base, (int)count, h->ring + *done % h->ring_frames * n_mels);
        *done += count;
    }
}

/* Last raw frame LFR row i reads (before clamping to the last frame). */
static ptrdiff_t row_last_frame(const af_sensevoice *h, size_t i) {
    const ptrdiff_t m = h->cfg.lfr_m, n = h->cfg.lfr_n, left = (m - 1) / 2;
    return (ptrdiff_t)i * n - left + m - 1;
}

/* Compute raw frames up to the last one LFR row i reads (clamped to `last`), LFR_CHUNK at a
 * time (FFT and LFR passes keep their data in cache), from the n_raw frames available. Frames
 * below the row's first one are not read again, so the ring never overwrites a needed frame. */
static void frames_for_row(af_sensevoice *h, const float *audio, size_t base, size_t *done,
                           size_t n_raw, size_t i, ptrdiff_t last) {
    ptrdiff_t hi = row_last_frame(h, i);
    hi = hi > last ? last : hi;
    while ((ptrdiff_t)*done <= hi)
        compute_frames(h, audio, base, done, n_raw - *done < LFR_CHUNK ? n_raw : *done + LFR_CHUNK);
}

/* LFR (+ CMVN) row i from the ring, raw frame indices clamped to [0, last]. */
static void lfr_row(const af_sensevoice *h, size_t i, ptrdiff_t last, float *row) {
    const size_t n_mels = (size_t)h->cfg.n_mels;
    const ptrdiff_t m = h->cfg.lfr_m, n = h->cfg.lfr_n, left = (m - 1) / 2;
    const float *shift = h->cfg.cmvn_shift, *scale = h->cfg.cmvn_scale;
    for (ptrdiff_t j = 0; j < m; j++) {
        ptrdiff_t src = (ptrdiff_t)i * n - left + j;
        src = src < 0 ? 0 : (src > last ? last : src);
        const size_t offset = (size_t)j * n_mels;
        const float *raw = h->ring + (size_t)src % h->ring_frames * n_mels;
        if (shift) {
            /* Copy and normalize together, preserving (x + shift) * scale. */
            for (size_t d = 0; d < n_mels; d++)
                row[offset + d] = (raw[d] + shift[offset + d]) * scale[offset + d];
        } else {
            memcpy(row + offset, raw, n_mels * sizeof(float));
        }
    }
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
    const ptrdiff_t last = (ptrdiff_t)n_raw - 1;
    const size_t dim = af_sensevoice_feat_dim(h);
    h->rng = splitmix64(h->cfg.seed); /* as in fbank_raw(): every frame, in order */
    size_t done = 0;                  /* raw frames computed; a multiple of AF_VL until the end */
    const unsigned fp = af_ftz_enter();
    for (size_t i = 0; i < n_out; i++) {
        frames_for_row(h, audio, 0, &done, n_raw, i, last);
        lfr_row(h, i, last, out + i * dim);
    }
    af_ftz_leave(fp);
    return AF_OK;
}

/* ---------------------------------------------------------------------------------- */
/* Streaming                                                                          */
/* ---------------------------------------------------------------------------------- */

/* Input taken per step of af_sensevoice_stream_push: bounds the sample buffer. */
#define STREAM_STEP (LFR_CHUNK * KALDI_FRAME_SHIFT)

struct af_sensevoice_stream {
    af_sensevoice *h; /* config, kernels, ring of raw frames, dither RNG */
    float *buf;       /* samples from the start of frame `base` on */
    size_t buf_len, buf_cap;
    size_t base;      /* first frame whose samples buf holds (= done after each step) */
    size_t n_samples; /* pushed since the start */
    size_t done;      /* raw frames computed */
    size_t next_row;  /* LFR rows emitted */
};

int af_sensevoice_stream_create(const af_sensevoice_config *cfg, af_sensevoice_stream **out) {
    if (!out)
        return AF_EINVAL;
    *out = NULL;
    af_sensevoice_stream *s = calloc(1, sizeof(*s));
    if (!s)
        return AF_ENOMEM;
    int rc = af_sensevoice_create(cfg, &s->h);
    if (rc != AF_OK) {
        free(s);
        return rc;
    }
    /* frames not yet computed span at most lfr_n + lfr_m frames, plus one step of input */
    s->buf_cap = ((size_t)s->h->cfg.lfr_n + (size_t)s->h->cfg.lfr_m + 2) * KALDI_FRAME_SHIFT +
                 KALDI_FRAME_LENGTH + STREAM_STEP;
    s->buf = malloc(s->buf_cap * sizeof(float));
    if (!s->buf) {
        af_sensevoice_stream_destroy(s);
        return AF_ENOMEM;
    }
    af_sensevoice_stream_reset(s);
    *out = s;
    return AF_OK;
}

void af_sensevoice_stream_destroy(af_sensevoice_stream *s) {
    if (!s)
        return;
    af_sensevoice_destroy(s->h);
    free(s->buf);
    free(s);
}

void af_sensevoice_stream_reset(af_sensevoice_stream *s) {
    s->buf_len = s->base = s->n_samples = s->done = s->next_row = 0;
    s->h->rng = splitmix64(s->h->cfg.seed);
}

const af_sensevoice *af_sensevoice_stream_frontend(const af_sensevoice_stream *s) {
    return s->h;
}

/* LFR rows that read only frames among the first n_raw (none past the end, so they are final). */
static size_t rows_ready(const af_sensevoice *h, size_t n_raw) {
    const size_t m = (size_t)h->cfg.lfr_m, n = (size_t)h->cfg.lfr_n, left = (m - 1) / 2;
    return n_raw + left < m ? 0 : (n_raw + left - m) / n + 1;
}

size_t af_sensevoice_stream_push_rows(const af_sensevoice_stream *s, size_t n_samples) {
    const size_t total = n_samples > SIZE_MAX - s->n_samples ? SIZE_MAX : s->n_samples + n_samples;
    return rows_ready(s->h, af_sensevoice_raw_frames(s->h, total)) - s->next_row;
}

size_t af_sensevoice_stream_finish_rows(const af_sensevoice_stream *s) {
    return af_sensevoice_out_frames(s->h, s->n_samples) - s->next_row;
}

/* Emit the rows that the frames of the samples so far make final (end = 0), or all remaining
 * rows, the last frame repeated past the end (end = 1). */
static size_t emit_rows(af_sensevoice_stream *s, int end, float *out) {
    af_sensevoice *h = s->h;
    const size_t n_raw = af_sensevoice_raw_frames(h, s->n_samples);
    const size_t n_rows = end ? af_sensevoice_out_frames(h, s->n_samples) : rows_ready(h, n_raw);
    const ptrdiff_t last = (ptrdiff_t)n_raw - 1; /* rows_ready() rows never reach it */
    const size_t dim = af_sensevoice_feat_dim(h);
    size_t k = 0;
    const unsigned fp = af_ftz_enter();
    for (; s->next_row < n_rows; s->next_row++, k++) {
        frames_for_row(h, s->buf, s->base, &s->done, n_raw, s->next_row, last);
        lfr_row(h, s->next_row, last, out + k * dim);
    }
    af_ftz_leave(fp);
    return k;
}

int af_sensevoice_stream_push(af_sensevoice_stream *s, const float *audio, size_t n_samples,
                              float *out, size_t *n_rows) {
    if (!s || !n_rows || (!audio && n_samples))
        return AF_EINVAL;
    *n_rows = 0;
    if (af_sensevoice_stream_push_rows(s, n_samples) > 0 && !out)
        return AF_EINVAL;
    if (n_samples > SIZE_MAX - s->n_samples)
        return AF_EINVAL;
    const size_t dim = af_sensevoice_feat_dim(s->h);
    while (n_samples > 0) {
        /* Drop the samples of computed frames: the next frame starts at sample done * shift. */
        const size_t drop = (s->done - s->base) * KALDI_FRAME_SHIFT;
        if (drop > 0) {
            s->buf_len = drop < s->buf_len ? s->buf_len - drop : 0;
            memmove(s->buf, s->buf + drop, s->buf_len * sizeof(float));
            s->base = s->done;
        }
        const size_t take = n_samples < STREAM_STEP ? n_samples : STREAM_STEP;
        memcpy(s->buf + s->buf_len, audio, take * sizeof(float));
        s->buf_len += take;
        s->n_samples += take;
        audio += take;
        n_samples -= take;
        const size_t k = emit_rows(s, 0, out ? out + *n_rows * dim : NULL);
        *n_rows += k;
    }
    return AF_OK;
}

int af_sensevoice_stream_finish(af_sensevoice_stream *s, float *out, size_t *n_rows) {
    if (!s || !n_rows)
        return AF_EINVAL;
    *n_rows = 0;
    if (af_sensevoice_stream_finish_rows(s) > 0 && !out)
        return AF_EINVAL;
    *n_rows = emit_rows(s, 1, out);
    af_sensevoice_stream_reset(s);
    return AF_OK;
}
