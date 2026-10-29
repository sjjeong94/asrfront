/*
 * asrfront - dependency-free ASR frontends (Whisper log-mel, SenseVoice fbank, Gemma 4 and
 * LFM3-Audio audio log-mel).
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

#define ASRFRONT_VERSION "0.2.3"
#define AF_SAMPLE_RATE 16000

enum {
    AF_OK = 0,
    AF_EINVAL = -1,       /* invalid argument / config */
    AF_ENOMEM = -2,       /* allocation failure */
    AF_ENOTIMPL = -3,     /* not implemented yet */
    AF_EUNSUPPORTED = -4, /* unsupported file format / codec / sample rate */
    AF_EIO = -5,          /* file could not be opened or read */
    AF_ECORRUPT = -6,     /* malformed or truncated file */
    AF_ETOOLONG = -7,     /* audio longer than the caller's limit */
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
    int n_mels;       /* 80 | 128 */
    int pad_or_trim;  /* nonzero: pad/trim input to 30 s -> 3000 frames */
    int n_threads;    /* reserved, must be >= 0; currently unused (the core is single-threaded) */
    int channel_last; /* nonzero: outputs are (out_frames, n_mels) instead of (n_mels, ...) */
} af_whisper_config;

void af_whisper_default_config(af_whisper_config *cfg);
int af_whisper_create(const af_whisper_config *cfg, af_whisper **out);
void af_whisper_destroy(af_whisper *h);

/* Number of output frames for an input of n_samples. */
size_t af_whisper_out_frames(const af_whisper *h, size_t n_samples);

/* out: (n_mels, out_frames) float32, row-major; (out_frames, n_mels) with channel_last. */
int af_whisper_log_mel(af_whisper *h, const float *audio, size_t n_samples, float *out);

/* The same with mel row m at out + m * ld (ld >= out_frames; elements after a row's
 * out_frames are not touched), e.g. into a zero-padded batch (n_mels, T >= out_frames).
 * With channel_last, frame t is at out + t * ld (ld >= n_mels). */
int af_whisper_log_mel_ld(af_whisper *h, const float *audio, size_t n_samples, float *out,
                          size_t ld);

/* The same over the input followed by zeros up to n_padded samples, as when it is padded to
 * the longest item of a batch before the STFT (transformers' WhisperFeatureExtractor with
 * padding=True, used by Qwen3-ASR). Handles with pad_or_trim = 0 only; n_samples <= n_padded
 * and n_padded > 200. out: (n_mels, n_padded / 160), or its transpose with channel_last. */
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
    int n_threads;           /* reserved, must be >= 0; currently unused (single-threaded) */
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

/* Streaming af_sensevoice_fbank: audio is pushed in pieces of any size, and every LFR row is
 * returned as soon as no later sample can change it. The rows of all pushes and the finish
 * are, bit for bit, af_sensevoice_fbank of the concatenated audio (dither included: the RNG is
 * seeded at creation and after each finish). Memory does not grow with the input. Not
 * thread-safe. */
typedef struct af_sensevoice_stream af_sensevoice_stream;

int af_sensevoice_stream_create(const af_sensevoice_config *cfg, af_sensevoice_stream **out);
void af_sensevoice_stream_destroy(af_sensevoice_stream *s);
/* Drop the pushed audio and start a new utterance. */
void af_sensevoice_stream_reset(af_sensevoice_stream *s);
/* The configuration's frontend, for af_sensevoice_feat_dim() (not for computing). */
const af_sensevoice *af_sensevoice_stream_frontend(const af_sensevoice_stream *s);

/* Rows that pushing n_samples more samples returns (size its output). */
size_t af_sensevoice_stream_push_rows(const af_sensevoice_stream *s, size_t n_samples);
/* Append audio; out: (af_sensevoice_stream_push_rows(s, n_samples), feat_dim) float32, of
 * which *n_rows rows (that count) are written. */
int af_sensevoice_stream_push(af_sensevoice_stream *s, const float *audio, size_t n_samples,
                              float *out, size_t *n_rows);
/* Rows af_sensevoice_stream_finish returns. */
size_t af_sensevoice_stream_finish_rows(const af_sensevoice_stream *s);
/* End of the utterance: the remaining rows (their last frames repeat the final one, as in
 * af_sensevoice_fbank), then a reset. out: (af_sensevoice_stream_finish_rows(s), feat_dim). */
int af_sensevoice_stream_finish(af_sensevoice_stream *s, float *out, size_t *n_rows);

/* ------------------------------------------------------------------------ */
/* Gemma 4 audio log-mel (transformers Gemma4AudioFeatureExtractor)         */
/* ------------------------------------------------------------------------ */

/* Gemma 4 E2B / E4B: 128 HTK mel bins of the magnitude spectrum, log(mel + 0.001), frames of
 * 20 ms every 10 ms after 160 zeros of padding (see docs/gemma4.md). Truncating the input
 * to 30 s and padding a batch to a multiple of 128 samples, as transformers does, is left to
 * the caller: it only decides the frame count. */
typedef struct af_gemma4 af_gemma4;

int af_gemma4_create(af_gemma4 **out);
void af_gemma4_destroy(af_gemma4 *h);

/* Frames for an input (zero-)padded to n_padded samples. */
size_t af_gemma4_out_frames(size_t n_padded);
/* Of those, the frames that read no padding (the feature mask): (n_samples - 1) / 160. */
size_t af_gemma4_valid_frames(size_t n_samples);

/* out: (n_frames, 128) float32, row-major: the valid frames of the input (at most n_frames),
 * then zeros, as transformers masks the padding's frames. */
int af_gemma4_log_mel(af_gemma4 *h, const float *audio, size_t n_samples, size_t n_frames,
                      float *out);
/* The same transposed: out (128, n_frames). */
int af_gemma4_log_mel_channel_first(af_gemma4 *h, const float *audio, size_t n_samples,
                                    size_t n_frames, float *out);

/* ------------------------------------------------------------------------ */
/* LFM3-Audio log-mel (LiquidAI Lfm3AudioFeatureExtractor)                  */
/* ------------------------------------------------------------------------ */

/* LFM3-Audio-2B: 128 Slaney mel bins of the power spectrum of 25 ms Hann windows every 10 ms
 * without centering, (max(log10(mel), -12) + 4) / 4 (see docs/lfm3_audio.md). Frames are
 * independent of each other: frame t reads samples [160 t, 160 t + 400) of the input followed
 * by zeros. */
typedef struct af_lfm3_audio af_lfm3_audio;

int af_lfm3_audio_create(af_lfm3_audio **out);
void af_lfm3_audio_destroy(af_lfm3_audio *h);

/* Frames of the extractor's output: the input right-padded with zeros to whole windows,
 * 1 + ceil((n_samples - 400) / 160), and 1 for n_samples <= 400. */
size_t af_lfm3_audio_out_frames(size_t n_samples);
/* Frames that read no padding (0 for n_samples < 400), as its streaming mode emits them. */
size_t af_lfm3_audio_complete_frames(size_t n_samples);

/* n_frames frames of the input (any number; frames past it read zeros). out: (128, n_frames)
 * float32 with mel row m at out + m * ld (ld >= n_frames; elements after a row's n_frames
 * are not touched), e.g. into a zero-padded batch (128, T >= n_frames). */
int af_lfm3_audio_log_mel(af_lfm3_audio *h, const float *audio, size_t n_samples, size_t n_frames,
                          float *out, size_t ld);
/* The same transposed: frame t at out + t * ld (ld >= 128). */
int af_lfm3_audio_log_mel_channel_last(af_lfm3_audio *h, const float *audio, size_t n_samples,
                                       size_t n_frames, float *out, size_t ld);

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

/* As af_load_memory, with two additions. err (may be NULL) receives a NUL-terminated ASCII
 * description of an AF_EUNSUPPORTED / AF_ECORRUPT failure, such as "WAV: format tag 0x0055
 * (MP3)" or "FLAC" when only the container is known (empty otherwise; 128 bytes are enough).
 * And it fails with AF_ETOOLONG when the output would have more than
 * max_samples samples (0: no limit). Checked from the decoded frame count before the samples
 * are allocated (PCM and ADPCM codecs) or while decoding (FLAC, MP3, stopping one frame past
 * the limit), so a small file that would decode to gigabytes fails early. */
int af_load_memory_ex(const void *buf, size_t size, af_raw_mode raw, size_t max_samples,
                      float **out, size_t *n_samples, char *err, size_t err_size);

/* Same for a file (UTF-8 path). `.pcm` / `.raw` extensions select AF_RAW_FORCE,
 * anything else AF_RAW_NEVER. */
int af_load_file(const char *path_utf8, float **out, size_t *n_samples);
/* af_load_file with af_load_memory_ex's max_samples and err. With max_samples, a WAV, Wave64,
 * NIST SPHERE or raw PCM file is checked as it is read and fails with AF_ETOOLONG as soon as the
 * part read is too long, without reading the rest. */
int af_load_file_ex(const char *path_utf8, size_t max_samples, float **out, size_t *n_samples,
                    char *err, size_t err_size);

/* What af_load_memory would decode, found without decoding or allocating the samples. */
typedef struct af_audio_info {
    const char *container; /* "WAV", "Wave64", "AIFF", "CAF", "NIST SPHERE", "FLAC", "MP3" or
                            * "raw PCM" (static string) */
    int sample_rate;       /* of the file, Hz */
    int channels;
    size_t n_frames;  /* decoded frames at sample_rate */
    size_t n_samples; /* af_load_memory's output length (16 kHz mono) */
} af_audio_info;

/* Parse an in-memory audio file as af_load_memory_ex would, without its samples: the result
 * and errors are those of af_load_memory_ex(buf, size, raw, max_samples, ...) (AF_ETOOLONG
 * included), except that ADPCM block headers are not checked and memory is not exhausted.
 * Uncompressed and ADPCM formats take time independent of the length; FLAC and MP3 are
 * decoded (through a small buffer, stopping past max_samples) to count their frames. */
int af_probe_memory(const void *buf, size_t size, af_raw_mode raw, size_t max_samples,
                    af_audio_info *info, char *err, size_t err_size);

void af_audio_free(float *p);

#ifdef __cplusplus
}
#endif

#endif /* ASRFRONT_H */
