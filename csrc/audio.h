/* Internal audio loading pieces (see docs/load.md). Not part of the public API. */
#ifndef ASRFRONT_AUDIO_H
#define ASRFRONT_AUDIO_H

#include <stddef.h>
#include <stdint.h>

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

/* Decoded audio: interleaved samples in `fmt`, native endianness, owned. */
typedef struct af_decoded {
    af_sample_fmt fmt;
    int channels;
    int sample_rate;
    uint32_t layout[AF_MAX_CHANNELS]; /* position bit per channel */
    size_t n_frames;
    void *data;
} af_decoded;

void af_decoded_free(af_decoded *d);

/* Default layout for a channel count (ffmpeg's), AF_EUNSUPPORTED above 8 channels. */
int af_default_layout(int channels, uint32_t *layout);
/* Layout from a channel mask (lowest set bits first); AF_EUNSUPPORTED if the mask does not
 * name exactly `channels` positions. */
int af_mask_layout(uint32_t mask, int channels, uint32_t *layout);

/* Container decoders. On success *out owns its samples. */
int af_wav_decode(const uint8_t *buf, size_t size, af_decoded *out);
int af_pcm_decode_raw(const uint8_t *buf, size_t size, af_decoded *out); /* s16le 16k mono */

/* Mono downmix weights (normalized when their sum exceeds 1). */
void af_downmix_weights(const uint32_t *layout, int channels, double *w);

/* Decoded audio -> mono float32 at sr_out (int16 / 32768): resample every channel, then
 * downmix, with ffmpeg's arithmetic. *out is malloc'ed (n_out samples). */
int af_audio_to_mono(const af_decoded *d, int sr_out, float **out, size_t *n_out);

#endif
