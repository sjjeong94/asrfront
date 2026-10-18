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

#define ASRFRONT_VERSION "0.2.0"
#define AF_SAMPLE_RATE 16000

enum {
    AF_OK = 0,
    AF_EINVAL = -1,       /* invalid argument / config */
    AF_ENOMEM = -2,       /* allocation failure */
    AF_ENOTIMPL = -3,     /* not implemented yet */
    AF_EUNSUPPORTED = -4, /* unsupported file format / codec / sample rate */
    AF_EIO = -5,          /* file could not be opened or read */
    AF_ECORRUPT = -6,     /* malformed or truncated file */
};

const char *af_version(void);
const char *af_strerror(int err);

/* SIMD kernel variant used by newly created handles: "avx2", "generic" or "scalar".
 * ASRFRONT_KERNELS=generic|scalar forces the baseline / plain-C build; all variants
 * produce identical results. */
const char *af_kernel_isa(void);

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

/* The same over the input followed by zeros up to n_padded samples, as when it is padded to
 * the longest item of a batch before the STFT (transformers' WhisperFeatureExtractor with
 * padding=True, used by Qwen3-ASR). Handles with pad_or_trim = 0 only; n_samples <= n_padded
 * and n_padded > 200. out: (n_mels, n_padded / 160). */
int af_whisper_log_mel_padded(af_whisper *h, const float *audio, size_t n_samples, size_t n_padded,
                              float *out);

/* ------------------------------------------------------------------------ */
/* SenseVoice fbank (kaldi fbank + LFR + CMVN, as in FunASR WavFrontend)    */
/* ------------------------------------------------------------------------ */

typedef struct af_sensevoice af_sensevoice;

typedef struct af_sensevoice_config {
    int n_mels;              /* 80 */
    int lfr_m;               /* 7 */
    int lfr_n;               /* 6 */
    float dither;            /* 0.0 */
    unsigned long long seed; /* dither RNG seed */
    const float *cmvn_shift; /* NULL: no CMVN. Length n_mels * lfr_m. Copied. */
    const float *cmvn_scale; /* must be NULL iff cmvn_shift is NULL. Copied. */
    int n_threads;           /* 0: auto, 1: single-threaded */
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

/* ------------------------------------------------------------------------ */
/* Audio loading (16 kHz mono float32, as whisper's ffmpeg-based load_audio) */
/* ------------------------------------------------------------------------ */

/* How headerless (raw) PCM is treated. Raw PCM is always s16le, 16 kHz, mono. */
typedef enum {
    AF_RAW_AUTO = 0,  /* container if its header is recognized, else raw PCM (bytes input) */
    AF_RAW_NEVER = 1, /* container only, else AF_EUNSUPPORTED (path without .pcm/.raw) */
    AF_RAW_FORCE = 2  /* always raw PCM (.pcm/.raw path) */
} af_raw_mode;

/* Decode an in-memory audio file (WAV, Wave64, AIFF, CAF, NIST SPHERE, FLAC, MP3; raw PCM
 * per `raw`) to 16 kHz mono float32,
 * matching `ffmpeg -i FILE -f s16le -ac 1 -ar 16000` / 32768 (see docs/load.md).
 * On success *out is allocated by the library (release with af_audio_free) and
 * *n_samples is set; on error *out is NULL. */
int af_load_memory(const void *buf, size_t size, af_raw_mode raw, float **out, size_t *n_samples);

/* Same for a file (UTF-8 path). `.pcm` / `.raw` extensions select AF_RAW_FORCE,
 * anything else AF_RAW_NEVER. */
int af_load_file(const char *path_utf8, float **out, size_t *n_samples);

void af_audio_free(float *p);

#ifdef __cplusplus
}
#endif

#endif /* ASRFRONT_H */
