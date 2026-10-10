#include <stdlib.h>
#include <string.h>

#include "asrfront.h"

#define KALDI_FRAME_LENGTH 400 /* 25 ms @ 16 kHz */
#define KALDI_FRAME_SHIFT 160  /* 10 ms @ 16 kHz */

struct af_sensevoice {
    af_sensevoice_config cfg; /* cmvn pointers point into cmvn below */
    float *cmvn;              /* [shift | scale], each n_mels * lfr_m; NULL if unused */
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
    *out = h;
    return AF_OK;
}

void af_sensevoice_destroy(af_sensevoice *h) {
    if (!h)
        return;
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

int af_sensevoice_fbank(af_sensevoice *h, const float *audio, size_t n_samples, float *out) {
    if (!h || !out || (!audio && n_samples))
        return AF_EINVAL;
    return AF_ENOTIMPL;
}

int af_sensevoice_fbank_raw(af_sensevoice *h, const float *audio, size_t n_samples, float *out) {
    if (!h || !out || (!audio && n_samples))
        return AF_EINVAL;
    return AF_ENOTIMPL;
}
