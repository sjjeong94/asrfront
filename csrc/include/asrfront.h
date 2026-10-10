/*
 * asrfront - dependency-free ASR frontends (Whisper log-mel, SenseVoice fbank).
 *
 * Usage pattern: create a handle once (precomputes windows, FFT plans, filter
 * banks and scratch space), call the compute function as many times as needed,
 * then destroy it. Handles are NOT thread-safe; use one handle per thread.
 *
 * All functions return AF_OK (0) on success or a negative AF_E* error code.
 * Output buffers are allocated by the caller; query their size first with the
 * matching *_out_frames() function.
 */
#ifndef ASRFRONT_H
#define ASRFRONT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ASRFRONT_VERSION "0.1.0"
#define AF_SAMPLE_RATE 16000

enum {
    AF_OK = 0,
    AF_EINVAL = -1,   /* invalid argument / config */
    AF_ENOMEM = -2,   /* allocation failure */
    AF_ENOTIMPL = -3, /* not implemented yet */
};

const char *af_version(void);
const char *af_strerror(int err);

/* ------------------------------------------------------------------------ */
/* Whisper log-mel spectrogram                                              */
/* ------------------------------------------------------------------------ */

typedef struct af_whisper af_whisper;

typedef struct af_whisper_config {
    int n_mels;      /* 80 | 128 */
    int pad_or_trim; /* nonzero: pad/trim input to 30 s -> 3000 frames */
    int n_threads;   /* 0: auto, 1: single-threaded */
} af_whisper_config;

void af_whisper_default_config(af_whisper_config *cfg);
int af_whisper_create(const af_whisper_config *cfg, af_whisper **out);
void af_whisper_destroy(af_whisper *h);

/* Number of output frames for an input of n_samples. */
size_t af_whisper_out_frames(const af_whisper *h, size_t n_samples);

/* out: (n_mels, out_frames) float32, row-major. */
int af_whisper_log_mel(af_whisper *h, const float *audio, size_t n_samples, float *out);

/* ------------------------------------------------------------------------ */
/* SenseVoice fbank (kaldi fbank + LFR + CMVN, as in FunASR WavFrontend)    */
/* ------------------------------------------------------------------------ */

typedef struct af_sensevoice af_sensevoice;

typedef struct af_sensevoice_config {
    int n_mels;               /* 80 */
    int lfr_m;                /* 7 */
    int lfr_n;                /* 6 */
    float dither;             /* 0.0 */
    unsigned long long seed;  /* dither RNG seed */
    const float *cmvn_shift;  /* NULL: no CMVN. Length n_mels * lfr_m. Copied. */
    const float *cmvn_scale;  /* must be NULL iff cmvn_shift is NULL. Copied. */
    int n_threads;            /* 0: auto, 1: single-threaded */
} af_sensevoice_config;

void af_sensevoice_default_config(af_sensevoice_config *cfg);
int af_sensevoice_create(const af_sensevoice_config *cfg, af_sensevoice **out);
void af_sensevoice_destroy(af_sensevoice *h);

/* Raw kaldi fbank frame count T (snip_edges=True). */
size_t af_sensevoice_raw_frames(const af_sensevoice *h, size_t n_samples);
/* LFR output frame count T' = ceil(T / lfr_n). */
size_t af_sensevoice_out_frames(const af_sensevoice *h, size_t n_samples);
/* Feature dimension of af_sensevoice_fbank output: n_mels * lfr_m. */
size_t af_sensevoice_feat_dim(const af_sensevoice *h);

/* audio: float waveform in [-1, 1] (scaled by 32768 internally).
 * out: (out_frames, n_mels * lfr_m) float32. */
int af_sensevoice_fbank(af_sensevoice *h, const float *audio, size_t n_samples, float *out);

/* Raw kaldi fbank without LFR/CMVN. out: (raw_frames, n_mels) float32. */
int af_sensevoice_fbank_raw(af_sensevoice *h, const float *audio, size_t n_samples, float *out);

#ifdef __cplusplus
}
#endif

#endif /* ASRFRONT_H */
