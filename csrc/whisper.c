#include <math.h>
#include <stdlib.h>

#include "asrfront.h"
#include "fft.h"
#include "kernels.h"
#include "melbank.h"

#define WHISPER_N_FFT 400
#define WHISPER_HOP 160
#define WHISPER_PAD (WHISPER_N_FFT / 2)                    /* center=True reflect padding */
#define WHISPER_HALF (WHISPER_N_FFT / 2)                   /* packed complex size: 200 */
#define WHISPER_N_BINS (WHISPER_N_FFT / 2 + 1)             /* 201 */
#define WHISPER_N_SAMPLES (30 * AF_SAMPLE_RATE)            /* 480000 */
#define WHISPER_N_FRAMES (WHISPER_N_SAMPLES / WHISPER_HOP) /* 3000 */
#define WHISPER_MAX_MELS 128

struct af_whisper {
    af_whisper_config cfg;
    const af_kernels *k;
    af_rfft *fft;
    af_melbank *mel;
    float window[WHISPER_N_FFT];
    float edge[AF_VL][WHISPER_N_FFT]; /* reflect/zero-padded frames near the edges */
    /* AF_VL-frame SoA work buffers */
    float re[WHISPER_HALF * AF_VL], im[WHISPER_HALF * AF_VL];
    float scr_re[WHISPER_HALF * AF_VL], scr_im[WHISPER_HALF * AF_VL];
    float power[WHISPER_N_BINS * AF_VL];
    float melv[WHISPER_MAX_MELS * AF_VL];
    float silence; /* log-mel of an all-zero frame (every mel bin), before normalization */
};

void af_whisper_default_config(af_whisper_config *cfg) {
    cfg->n_mels = 80;
    cfg->pad_or_trim = 1;
    cfg->n_threads = 1;
    cfg->channel_last = 0;
}

int af_whisper_create(const af_whisper_config *cfg, af_whisper **out) {
    if (!cfg || !out)
        return AF_EINVAL;
    *out = NULL;
    if ((cfg->n_mels != 80 && cfg->n_mels != 128) || cfg->n_threads < 0)
        return AF_EINVAL;

    af_whisper *h = calloc(1, sizeof(*h));
    if (!h)
        return AF_ENOMEM;
    h->cfg = *cfg;
    h->k = af_kernels_select();
    int rc = af_rfft_create(WHISPER_N_FFT, &h->fft);
    if (rc == AF_OK)
        rc = af_melbank_slaney(AF_SAMPLE_RATE, WHISPER_N_FFT, cfg->n_mels, 0.0,
                               AF_SAMPLE_RATE / 2.0, &h->mel);
    if (rc != AF_OK) {
        af_whisper_destroy(h);
        return rc;
    }
    /* torch.hann_window(400), periodic */
    for (int i = 0; i < WHISPER_N_FFT; i++)
        h->window[i] =
            (float)(0.5 - 0.5 * cos(6.283185307179586476925286766559 * i / WHISPER_N_FFT));
    /* An all-zero frame has zero power and mel energies, so its output is log10_floor(0).
     * Computed with the same kernel, so skipped frames match computed ones bit for bit. */
    float zero[AF_VL] = {0};
    h->k->log10_floor(zero, AF_VL, 1e-10f);
    h->silence = zero[0];
    *out = h;
    return AF_OK;
}

void af_whisper_destroy(af_whisper *h) {
    if (!h)
        return;
    af_rfft_destroy(h->fft);
    af_melbank_destroy(h->mel);
    free(h);
}

size_t af_whisper_out_frames(const af_whisper *h, size_t n_samples) {
    if (h->cfg.pad_or_trim)
        return WHISPER_N_FRAMES;
    /* center=True STFT yields n/hop + 1 frames; whisper drops the last one. */
    return n_samples / WHISPER_HOP;
}

/*
 * Start of frame t of the reflect-padded virtual signal v, where v[i] = audio[i] for
 * i < n_in and 0 for n_in <= i < n_virt (zero padding). Frame t covers
 * v[t * hop - pad, t * hop + pad); indices outside [0, n_virt) are reflected. Interior
 * frames point into `audio`; edge frames are materialized in `edge`.
 */
static const float *frame_src(const float *audio, size_t n_in, size_t n_virt, size_t t,
                              float *edge) {
    const ptrdiff_t i0 = (ptrdiff_t)(t * WHISPER_HOP) - WHISPER_PAD;
    if (i0 >= 0 && (size_t)i0 + WHISPER_N_FFT <= n_in)
        return audio + i0;
    const ptrdiff_t last = (ptrdiff_t)n_virt - 1;
    for (int k = 0; k < WHISPER_N_FFT; k++) {
        ptrdiff_t i = i0 + k;
        if (i < 0)
            i = -i;
        else if (i > last)
            i = 2 * last - i;
        edge[k] = (size_t)i < n_in ? audio[i] : 0.0f;
    }
    return edge;
}

/* Whether frames t0 .. t1 read only zero padding (indices n_in .. n_virt - 1): the first one
 * starts at or after n_in, and the samples the last one reflects back from past n_virt - 1
 * (the furthest back is 2 * (n_virt - 1) - (t1 * hop + pad - 1)) are padding too. With 30 s of
 * pad_or_trim only frame 2999 reflects, to 479959, past any silent frame's start; with a
 * padding of a few samples the reflection reaches back into the input. */
static int frames_silent(size_t n_in, size_t n_virt, size_t t0, size_t t1) {
    if (t0 * WHISPER_HOP < n_in + WHISPER_PAD)
        return 0;
    const size_t end = t1 * WHISPER_HOP + WHISPER_PAD; /* one past the last index read */
    return end <= n_virt || 2 * (n_virt - 1) - (end - 1) >= n_in;
}

/* whisper_normalize over n_rows rows of row_len floats, ld apart. */
static void normalize_rows(const af_kernels *k, float *out, size_t n_rows, size_t row_len,
                           size_t ld, float gmax) {
    if (ld == row_len) {
        k->whisper_normalize(out, n_rows * row_len, gmax);
        return;
    }
    for (size_t r = 0; r < n_rows; r++)
        k->whisper_normalize(out + r * ld, row_len, gmax);
}

/* Frames of the input virtually zero-padded to n_virt samples (n_in <= n_virt, 200 < n_virt):
 * n_virt / hop frames of the reflect-padded centered STFT, the last one dropped. Mel row m
 * goes to out + m * ld (ld >= the frame count); with channel_last, frame t goes to
 * out + t * ld (ld >= n_mels). */
static void log_mel_virtual(af_whisper *h, const float *audio, size_t n_in, size_t n_virt,
                            float *out, size_t ld) {
    const af_kernels *k = h->k;
    const size_t n_frames = n_virt / WHISPER_HOP;
    const int n_mels = h->cfg.n_mels;
    const int channel_last = h->cfg.channel_last;
    /* Short inputs padded to 30 s (or to a long batch item): most blocks are silence, with a
     * known output. They form one run of frames [s0, s1) (frames_silent() holds from some
     * block on, until the reflection at the end reaches back into the input; a silent block
     * that would not extend the run is simply computed), written once, already normalized,
     * after the max is known. */
    size_t s0 = n_frames, s1 = n_frames;
    float gmax = -3.402823466e+38f;

    for (size_t t0 = 0; t0 < n_frames; t0 += AF_VL) {
        const int count = n_frames - t0 < AF_VL ? (int)(n_frames - t0) : AF_VL;
        if (n_in < n_virt && (s0 == n_frames || s1 == t0) &&
            frames_silent(n_in, n_virt, t0, t0 + (size_t)count - 1)) {
            if (s0 == n_frames)
                s0 = t0;
            s1 = t0 + (size_t)count;
            continue;
        }
        const float *src[AF_VL];
        src[0] = frame_src(audio, n_in, n_virt, t0, h->edge[0]);
        for (int l = 1; l < AF_VL; l++) /* unused lanes of the last block repeat lane 0 */
            src[l] = l < count ? frame_src(audio, n_in, n_virt, t0 + l, h->edge[l]) : src[0];
        k->gather_packed(src, WHISPER_N_FFT, h->window, h->re, h->im);
        k->rfft_power(h->fft, h->re, h->im, h->scr_re, h->scr_im, h->power);
        k->mel(h->mel, h->power, h->melv);
        k->log10_floor(h->melv, (size_t)n_mels * AF_VL, 1e-10f);
        /* Unused lanes hold copies of lane 0, so they do not change the max. */
        gmax = k->max_f32(h->melv, (size_t)n_mels * AF_VL, gmax);
        if (channel_last) { /* SoA lane l is output frame t0 + l */
            for (int l = 0; l < count; l++) {
                float *dst = out + (t0 + (size_t)l) * ld;
                for (int m = 0; m < n_mels; m++)
                    dst[m] = h->melv[(size_t)m * AF_VL + l];
            }
            continue;
        }
        for (int m = 0; m < n_mels; m++) { /* SoA rows map to contiguous output runs */
            float *dst = out + (size_t)m * ld + t0;
            const float *row = h->melv + (size_t)m * AF_VL;
            for (int l = 0; l < count; l++)
                dst[l] = row[l];
        }
    }

    /* Dynamic range compression against the global max, then scale. */
    if (s0 == n_frames) {
        if (channel_last)
            normalize_rows(k, out, n_frames, (size_t)n_mels, ld, gmax);
        else
            normalize_rows(k, out, (size_t)n_mels, n_frames, ld, gmax);
        return;
    }
    gmax = h->silence > gmax ? h->silence : gmax;
    float silence = h->silence; /* normalized by the same kernel as the computed frames */
    k->whisper_normalize(&silence, 1, gmax);
    if (channel_last) {
        normalize_rows(k, out, s0, (size_t)n_mels, ld, gmax);
        for (size_t t = s0; t < s1; t++)
            for (int m = 0; m < n_mels; m++)
                out[t * ld + (size_t)m] = silence;
        normalize_rows(k, out + s1 * ld, n_frames - s1, (size_t)n_mels, ld, gmax);
        return;
    }
    for (int m = 0; m < n_mels; m++) {
        float *row = out + (size_t)m * ld;
        k->whisper_normalize(row, s0, gmax);
        for (size_t t = s0; t < s1; t++)
            row[t] = silence;
        k->whisper_normalize(row + s1, n_frames - s1, gmax);
    }
}

/* Elements between the rows of a dense output: frames, or mels with channel_last. */
static size_t dense_ld(const af_whisper *h, size_t n_frames) {
    return h->cfg.channel_last ? (size_t)h->cfg.n_mels : n_frames;
}

int af_whisper_log_mel(af_whisper *h, const float *audio, size_t n_samples, float *out) {
    if (!h)
        return AF_EINVAL;
    return af_whisper_log_mel_ld(h, audio, n_samples, out,
                                 dense_ld(h, af_whisper_out_frames(h, n_samples)));
}

int af_whisper_log_mel_ld(af_whisper *h, const float *audio, size_t n_samples, float *out,
                          size_t ld) {
    if (!h || !out || (!audio && n_samples) ||
        ld < dense_ld(h, af_whisper_out_frames(h, n_samples)))
        return AF_EINVAL;
    if (h->cfg.pad_or_trim) {
        const size_t n_in = n_samples < WHISPER_N_SAMPLES ? n_samples : WHISPER_N_SAMPLES;
        log_mel_virtual(h, audio, n_in, WHISPER_N_SAMPLES, out, ld);
        return AF_OK;
    }
    /* Reflect padding needs pad < length (torch raises otherwise). */
    if (n_samples <= WHISPER_PAD)
        return AF_EINVAL;
    log_mel_virtual(h, audio, n_samples, n_samples, out, ld);
    return AF_OK;
}

int af_whisper_log_mel_padded(af_whisper *h, const float *audio, size_t n_samples, size_t n_padded,
                              float *out) {
    if (!h || !out || (!audio && n_samples) || h->cfg.pad_or_trim || n_samples > n_padded ||
        n_padded <= WHISPER_PAD)
        return AF_EINVAL;
    log_mel_virtual(h, audio, n_samples, n_padded, out, dense_ld(h, n_padded / WHISPER_HOP));
    return AF_OK;
}
