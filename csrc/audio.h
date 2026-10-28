/* Internal audio loading pieces (see docs/load.md). Not part of the public API. */
#ifndef ASRFRONT_AUDIO_H
#define ASRFRONT_AUDIO_H

#include <stddef.h>
#include <stdint.h>

#include "asrfront.h"

/* Decoder output sample formats (they select ffmpeg's arithmetic path). */
typedef enum {
    AF_SFMT_U8,
    AF_SFMT_S16,
    AF_SFMT_S32, /* 24-bit audio is stored in the top 24 bits */
    AF_SFMT_F32,
    AF_SFMT_F64,
} af_sample_fmt;

/* Channel positions: the WAVE_FORMAT_EXTENSIBLE channel-mask bits. */
enum {
    AF_CH_FL = 1u << 0,
    AF_CH_FR = 1u << 1,
    AF_CH_FC = 1u << 2,
    AF_CH_LFE = 1u << 3,
    AF_CH_BL = 1u << 4,
    AF_CH_BR = 1u << 5,
    AF_CH_FLC = 1u << 6,
    AF_CH_FRC = 1u << 7,
    AF_CH_BC = 1u << 8,
    AF_CH_SL = 1u << 9,
    AF_CH_SR = 1u << 10,
    AF_CH_TC = 1u << 11,
    AF_CH_TFL = 1u << 12,
    AF_CH_TFC = 1u << 13,
    AF_CH_TFR = 1u << 14,
    AF_CH_TBL = 1u << 15,
    AF_CH_TBC = 1u << 16,
    AF_CH_TBR = 1u << 17,
};

#define AF_MAX_CHANNELS 18 /* one per mask bit */

/* Decoded audio: interleaved samples in `fmt`, native endianness, owned unless `borrowed`. */
typedef struct af_decoded {
    af_sample_fmt fmt;
    int channels;
    int sample_rate;
    uint32_t layout[AF_MAX_CHANNELS]; /* position bit per channel */
    size_t n_frames;
    void *data;
    /* data points into the caller's input buffer (af_pcm_decode_frames on native-layout
     * samples): read-only, valid only while that buffer is, not freed by af_decoded_free */
    int borrowed;
    /* set by the caller before decoding: stop once n_frames is known, without decoding or
     * allocating the samples (af_probe_memory); the decoder then returns AF_PROBED */
    int probe;
    size_t limit;    /* the caller's limit in 16 kHz samples (0: none); see af_frame_limit */
    char detail[96]; /* why decoding failed, if known (af_fail) */
} af_decoded;

/* Record why decoding failed (printf-style, e.g. "format tag 0x0055"; bytes outside printable
 * ASCII become '?') and return rc. */
int af_fail(af_decoded *d, int rc, const char *fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 3, 4)))
#endif
    ;

/* Internal success code of a decoder in probe mode: n_frames, channels and sample_rate are
 * set, data is NULL. Never returned by the public API. */
#define AF_PROBED 1

/* Clear *d for a decoder (keeping d->probe) and set its limit. */
void af_decoded_init(af_decoded *d, size_t limit);
void af_decoded_free(af_decoded *d);
/* Most frames at d->sample_rate that resample to at most d->limit samples at 16 kHz
 * (SIZE_MAX if there is no limit). */
size_t af_frame_limit(const af_decoded *d);
/* Allocate d->data for d->n_frames * d->channels samples of `fmt` and set d->fmt;
 * AF_ETOOLONG if d->n_frames exceeds af_frame_limit(d). In probe mode nothing is allocated
 * and it returns AF_PROBED, which callers pass on as they do errors. */
int af_decoded_alloc(af_decoded *d, af_sample_fmt fmt);

/* Coded sample formats of the uncompressed codecs (pcm.c). */
typedef enum {
    AF_PCM_U8,
    AF_PCM_S8,
    AF_PCM_S16LE,
    AF_PCM_S16BE,
    AF_PCM_S24LE,
    AF_PCM_S24BE,
    AF_PCM_S32LE,
    AF_PCM_S32BE,
    AF_PCM_F32LE,
    AF_PCM_F32BE,
    AF_PCM_F64LE,
    AF_PCM_F64BE,
    AF_PCM_ALAW,
    AF_PCM_MULAW,
} af_pcm_codec;

int af_pcm_bytes(af_pcm_codec c); /* bytes per sample */
/* Convert d->n_frames * d->channels coded samples at src into d->data (sets d->fmt). */
int af_pcm_convert(af_decoded *d, af_pcm_codec c, const uint8_t *src);
/* As af_pcm_convert for all whole frames in `size` bytes (sets d->n_frames). Samples that
 * are already in their decoded layout (u8, and little-endian s16 / s32 / f32 / f64 on a
 * little-endian host, suitably aligned) are not copied: d->data borrows src (d->borrowed).
 * On error d->data is freed. */
int af_pcm_decode_frames(af_decoded *d, af_pcm_codec c, const uint8_t *src, size_t size);

/* ADPCM codecs (adpcm.c), decoded to s16 with ffmpeg's arithmetic. d->channels must be set;
 * these set d->fmt, d->n_frames and d->data. A partial last block decodes the whole sample
 * groups it contains. */
int af_adpcm_ima_wav(af_decoded *d, const uint8_t *src, size_t size, size_t block_align);
int af_adpcm_ms(af_decoded *d, const uint8_t *src, size_t size, size_t block_align);
int af_adpcm_ima_qt(af_decoded *d, const uint8_t *src, size_t size); /* 34-byte packets */

/* Default layout for a channel count (ffmpeg's), AF_EUNSUPPORTED above 8 channels. */
int af_default_layout(int channels, uint32_t *layout);
/* Layout from a channel mask (lowest set bits first); AF_EUNSUPPORTED if the mask does not
 * name exactly `channels` positions. */
int af_mask_layout(uint32_t mask, int channels, uint32_t *layout);
/* Layout from a CoreAudio AudioChannelLayout (CAF 'chan' / AIFF 'CHAN' chunk body). */
int af_coreaudio_layout(const uint8_t *body, size_t size, int channels, uint32_t *layout);

extern const uint8_t af_w64_riff_guid[16]; /* first 16 bytes of a Wave64 file */

/* Container decoders. On success *out owns its samples. `limit` as in af_decoded. */
int af_wav_decode(const uint8_t *buf, size_t size, size_t limit,
                  af_decoded *out); /* RIFF / RF64 / BW64 */
int af_w64_decode(const uint8_t *buf, size_t size, size_t limit, af_decoded *out); /* Sony Wave64 */
int af_aiff_decode(const uint8_t *buf, size_t size, size_t limit,
                   af_decoded *out); /* AIFF / AIFF-C */
int af_caf_decode(const uint8_t *buf, size_t size, size_t limit, af_decoded *out);
int af_sphere_decode(const uint8_t *buf, size_t size, size_t limit,
                     af_decoded *out); /* NIST SPHERE */
int af_pcm_decode_raw(const uint8_t *buf, size_t size, size_t limit,
                      af_decoded *out); /* s16le 16k mono */
int af_flac_decode(const uint8_t *buf, size_t size, size_t limit, af_decoded *out);
int af_mp3_decode(const uint8_t *buf, size_t size, size_t limit, af_decoded *out);

/* Whether the first `size` bytes of a file already show that af_load_memory_ex(file, raw,
 * max_samples) fails with AF_ETOOLONG: 1 if so (and then so does the load of these bytes alone),
 * 0 if not (yet: more of the file may tell), -1 if no prefix can tell (no limit, or a container
 * other than WAV, Wave64, NIST SPHERE and forced raw PCM; FLAC and MP3 would be decoded). For
 * readers of large files: af_load_file_ex and the Python binding. */
int af_prefix_too_long(const uint8_t *buf, size_t size, af_raw_mode raw, size_t max_samples);
/* Where af_load_file_ex first calls it (then at twice that, and so on). */
#define AF_PREFIX_CHECK_BYTES ((size_t)1 << 20)

/* Mono downmix weights (normalized when their sum exceeds 1). */
void af_downmix_weights(const uint32_t *layout, int channels, double *w);

/* Decoded audio -> mono float32 at sr_out (int16 / 32768): resample every channel, then
 * downmix, with ffmpeg's arithmetic. *out is malloc'ed (n_out samples). */
int af_audio_to_mono(const af_decoded *d, int sr_out, float **out, size_t *n_out);

#endif
