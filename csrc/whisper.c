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
};

void af_whisper_default_config(af_whisper_config *cfg) {
    cfg->n_mels = 80;
    cfg->pad_or_trim = 1;
    cfg->n_threads = 1;
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
 * i < n_in and 0 for n_in <= i < n_virt (pad_or_trim zero padding). Frame t covers
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

int af_whisper_log_mel(af_whisper *h, const float *audio, size_t n_samples, float *out) {
    if (!h || !out || (!audio && n_samples))
        return AF_EINVAL;

    size_t n_virt, n_in;
    if (h->cfg.pad_or_trim) {
        n_virt = WHISPER_N_SAMPLES;
        n_in = n_samples < n_virt ? n_samples : n_virt;
    } else {
        /* Reflect padding needs pad < length (torch raises otherwise). */
        if (n_samples <= WHISPER_PAD)
            return AF_EINVAL;
        n_virt = n_in = n_samples;
    }

    const af_kernels *k = h->k;
    const size_t n_frames = af_whisper_out_frames(h, n_samples);
    const int n_mels = h->cfg.n_mels;

    for (size_t t0 = 0; t0 < n_frames; t0 += AF_VL) {
        const int count = n_frames - t0 < AF_VL ? (int)(n_frames - t0) : AF_VL;
        const float *src[AF_VL];
        for (int l = 0; l < AF_VL; l++) /* unused lanes of the last block repeat lane 0 */
            src[l] = l < count ? frame_src(audio, n_in, n_virt, t0 + l, h->edge[l]) : src[0];
        k->gather_packed(src, WHISPER_N_FFT, h->window, h->re, h->im);
        k->rfft_power(h->fft, h->re, h->im, h->scr_re, h->scr_im, h->power);
        k->mel(h->mel, h->power, h->melv);
        k->log10_floor(h->melv, (size_t)n_mels * AF_VL, 1e-10f);
        for (int m = 0; m < n_mels; m++) { /* SoA rows map to contiguous output runs */
            float *dst = out + (size_t)m * n_frames + t0;
            const float *row = h->melv + (size_t)m * AF_VL;
            for (int l = 0; l < count; l++)
                dst[l] = row[l];
        }
    }

    /* Dynamic range compression against the global max, then scale. */
    k->whisper_normalize(out, (size_t)n_mels * n_frames);
    return AF_OK;
}
