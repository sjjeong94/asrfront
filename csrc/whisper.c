#include <math.h>
#include <stdlib.h>

#include "asrfront.h"
#include "fft.h"
#include "melbank.h"

#define WHISPER_N_FFT 400
#define WHISPER_HOP 160
#define WHISPER_PAD (WHISPER_N_FFT / 2)                    /* center=True reflect padding */
#define WHISPER_N_BINS (WHISPER_N_FFT / 2 + 1)             /* 201 */
#define WHISPER_N_SAMPLES (30 * AF_SAMPLE_RATE)            /* 480000 */
#define WHISPER_N_FRAMES (WHISPER_N_SAMPLES / WHISPER_HOP) /* 3000 */
#define WHISPER_MAX_MELS 128

struct af_whisper {
    af_whisper_config cfg;
    af_rfft *fft;
    af_melbank *mel;
    float window[WHISPER_N_FFT];
    float frame[WHISPER_N_FFT];
    float power[WHISPER_N_BINS];
    float melv[WHISPER_MAX_MELS];
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
 * Windowed frame t of the reflect-padded virtual signal v, where v[i] = audio[i] for
 * i < n_in and 0 for n_in <= i < n_virt (pad_or_trim zero padding). Frame t covers
 * v[t * hop - pad, t * hop + pad); indices outside [0, n_virt) are reflected.
 */
static void load_frame(af_whisper *h, const float *audio, size_t n_in, size_t n_virt, size_t t) {
    const long i0 = (long)(t * WHISPER_HOP) - WHISPER_PAD;
    if (i0 >= 0 && (size_t)i0 + WHISPER_N_FFT <= n_in) {
        const float *src = audio + i0;
        for (int k = 0; k < WHISPER_N_FFT; k++)
            h->frame[k] = src[k] * h->window[k];
        return;
    }
    const long last = (long)n_virt - 1;
    for (int k = 0; k < WHISPER_N_FFT; k++) {
        long i = i0 + k;
        if (i < 0)
            i = -i;
        else if (i > last)
            i = 2 * last - i;
        const float v = (size_t)i < n_in ? audio[i] : 0.0f;
        h->frame[k] = v * h->window[k];
    }
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

    const size_t n_frames = af_whisper_out_frames(h, n_samples);
    const int n_mels = h->cfg.n_mels;
    float gmax = -INFINITY;

    for (size_t t = 0; t < n_frames; t++) {
        load_frame(h, audio, n_in, n_virt, t);
        af_rfft_power(h->fft, h->frame, h->power);
        af_melbank_apply(h->mel, h->power, h->melv);
        for (int m = 0; m < n_mels; m++) {
            const float v = log10f(fmaxf(h->melv[m], 1e-10f));
            out[(size_t)m * n_frames + t] = v;
            gmax = fmaxf(gmax, v);
        }
    }

    /* Dynamic range compression against the global max, then scale. */
    const float floor_v = gmax - 8.0f;
    const size_t total = (size_t)n_mels * n_frames;
    for (size_t i = 0; i < total; i++)
        out[i] = (fmaxf(out[i], floor_v) + 4.0f) * 0.25f;
    return AF_OK;
}
