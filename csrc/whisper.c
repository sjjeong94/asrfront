#include <stdlib.h>

#include "asrfront.h"

#define WHISPER_N_FFT 400
#define WHISPER_HOP 160
#define WHISPER_N_SAMPLES (30 * AF_SAMPLE_RATE)        /* 480000 */
#define WHISPER_N_FRAMES (WHISPER_N_SAMPLES / WHISPER_HOP) /* 3000 */

struct af_whisper {
    af_whisper_config cfg;
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
    *out = h;
    return AF_OK;
}

void af_whisper_destroy(af_whisper *h) { free(h); }

size_t af_whisper_out_frames(const af_whisper *h, size_t n_samples) {
    if (h->cfg.pad_or_trim)
        return WHISPER_N_FRAMES;
    /* center=True STFT yields n/hop + 1 frames; whisper drops the last one. */
    return n_samples / WHISPER_HOP;
}

int af_whisper_log_mel(af_whisper *h, const float *audio, size_t n_samples, float *out) {
    if (!h || !out || (!audio && n_samples))
        return AF_EINVAL;
    return AF_ENOTIMPL;
}
