/* Uncompressed sample codecs shared by the containers (PCM of any byte order, IEEE float,
 * G.711), and headerless raw PCM. */
#include <stdlib.h>
#include <string.h>

#include "asrfront.h"
#include "audio.h"

static uint16_t rd16le(const uint8_t *p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}

static uint16_t rd16be(const uint8_t *p) {
    return (uint16_t)((p[0] << 8) | p[1]);
}

static uint32_t rd32le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint32_t rd32be(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

/* G.711 expansion to 16-bit linear (ITU-T G.711 segment/quantization tables). */
static int16_t alaw_to_s16(uint8_t a) {
    a ^= 0x55;
    int seg = (a >> 4) & 7;
    int t = (a & 0x0F) << 4;
    if (seg == 0)
        t += 8;
    else
        t = (t + 0x108) << (seg - 1);
    return (int16_t)((a & 0x80) ? t : -t);
}

static int16_t mulaw_to_s16(uint8_t u) {
    u = (uint8_t)~u;
    int t = (((u & 0x0F) << 3) + 0x84) << ((u >> 4) & 7);
    return (int16_t)((u & 0x80) ? (0x84 - t) : (t - 0x84));
}

void af_decoded_free(af_decoded *d) {
    if (d) {
        free(d->data);
        d->data = NULL;
    }
}

int af_decoded_alloc(af_decoded *d, af_sample_fmt fmt) {
    static const size_t elem_size[] = {1, 2, 4, 4, 8}; /* by af_sample_fmt */
    size_t es = elem_size[fmt];
    size_t n = d->n_frames * (size_t)d->channels;
    if (d->channels > 0 && n / (size_t)d->channels != d->n_frames)
        return AF_ECORRUPT;
    if (n > SIZE_MAX / es)
        return AF_ENOMEM;
    d->fmt = fmt;
    d->data = malloc(n * es > 0 ? n * es : 1);
    return d->data ? AF_OK : AF_ENOMEM;
}

int af_pcm_bytes(af_pcm_codec c) {
    switch (c) {
    case AF_PCM_S16LE:
    case AF_PCM_S16BE:
        return 2;
    case AF_PCM_S24LE:
    case AF_PCM_S24BE:
        return 3;
    case AF_PCM_S32LE:
    case AF_PCM_S32BE:
    case AF_PCM_F32LE:
    case AF_PCM_F32BE:
        return 4;
    case AF_PCM_F64LE:
    case AF_PCM_F64BE:
        return 8;
    default: /* U8, S8, A-law, mu-law */
        return 1;
    }
}

int af_pcm_convert(af_decoded *d, af_pcm_codec c, const uint8_t *src) {
    size_t n = d->n_frames * (size_t)d->channels;
    int rc;
    switch (c) {
    case AF_PCM_U8:
    case AF_PCM_S8: { /* ffmpeg decodes signed 8-bit to u8 as well */
        if ((rc = af_decoded_alloc(d, AF_SFMT_U8)) != AF_OK)
            return rc;
        uint8_t *o = d->data;
        const uint8_t flip = c == AF_PCM_S8 ? 0x80 : 0;
        for (size_t i = 0; i < n; i++)
            o[i] = src[i] ^ flip;
        return AF_OK;
    }
    case AF_PCM_S16LE:
    case AF_PCM_S16BE: {
        if ((rc = af_decoded_alloc(d, AF_SFMT_S16)) != AF_OK)
            return rc;
        int16_t *o = d->data;
        if (c == AF_PCM_S16LE)
            for (size_t i = 0; i < n; i++)
                o[i] = (int16_t)rd16le(src + 2 * i);
        else
            for (size_t i = 0; i < n; i++)
                o[i] = (int16_t)rd16be(src + 2 * i);
        return AF_OK;
    }
    case AF_PCM_S24LE:
    case AF_PCM_S24BE: /* the sample goes to the top bits of an s32 */
    case AF_PCM_S32LE:
    case AF_PCM_S32BE: {
        if ((rc = af_decoded_alloc(d, AF_SFMT_S32)) != AF_OK)
            return rc;
        int32_t *o = d->data;
        if (c == AF_PCM_S24LE) {
            for (size_t i = 0; i < n; i++) {
                const uint8_t *p = src + 3 * i;
                o[i] = (int32_t)(((uint32_t)p[0] << 8) | ((uint32_t)p[1] << 16) |
                                 ((uint32_t)p[2] << 24));
            }
        } else if (c == AF_PCM_S24BE) {
            for (size_t i = 0; i < n; i++) {
                const uint8_t *p = src + 3 * i;
                o[i] = (int32_t)(((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
                                 ((uint32_t)p[2] << 8));
            }
        } else if (c == AF_PCM_S32LE) {
            for (size_t i = 0; i < n; i++)
                o[i] = (int32_t)rd32le(src + 4 * i);
        } else {
            for (size_t i = 0; i < n; i++)
                o[i] = (int32_t)rd32be(src + 4 * i);
        }
        return AF_OK;
    }
    case AF_PCM_F32LE:
    case AF_PCM_F32BE: {
        if ((rc = af_decoded_alloc(d, AF_SFMT_F32)) != AF_OK)
            return rc;
        float *o = d->data;
        for (size_t i = 0; i < n; i++) {
            uint32_t u = c == AF_PCM_F32LE ? rd32le(src + 4 * i) : rd32be(src + 4 * i);
            memcpy(&o[i], &u, 4);
        }
        return AF_OK;
    }
    case AF_PCM_F64LE:
    case AF_PCM_F64BE: {
        if ((rc = af_decoded_alloc(d, AF_SFMT_F64)) != AF_OK)
            return rc;
        double *o = d->data;
        for (size_t i = 0; i < n; i++) {
            const uint8_t *p = src + 8 * i;
            uint64_t u = c == AF_PCM_F64LE ? (uint64_t)rd32le(p) | ((uint64_t)rd32le(p + 4) << 32)
                                           : ((uint64_t)rd32be(p) << 32) | (uint64_t)rd32be(p + 4);
            memcpy(&o[i], &u, 8);
        }
        return AF_OK;
    }
    case AF_PCM_ALAW:
    case AF_PCM_MULAW: {
        if ((rc = af_decoded_alloc(d, AF_SFMT_S16)) != AF_OK)
            return rc;
        int16_t *o = d->data;
        for (size_t i = 0; i < n; i++)
            o[i] = c == AF_PCM_ALAW ? alaw_to_s16(src[i]) : mulaw_to_s16(src[i]);
        return AF_OK;
    }
    }
    return AF_EUNSUPPORTED;
}

int af_pcm_decode_frames(af_decoded *d, af_pcm_codec c, const uint8_t *src, size_t size) {
    size_t frame = (size_t)af_pcm_bytes(c) * (size_t)d->channels;
    d->n_frames = size / frame; /* a partial last frame is dropped */
    int rc = af_pcm_convert(d, c, src);
    if (rc != AF_OK)
        af_decoded_free(d);
    return rc;
}

int af_pcm_decode_raw(const uint8_t *buf, size_t size, af_decoded *out) {
    memset(out, 0, sizeof(*out));
    out->channels = 1;
    out->sample_rate = AF_SAMPLE_RATE;
    out->layout[0] = AF_CH_FC;
    return af_pcm_decode_frames(out, AF_PCM_S16LE, buf, size); /* odd byte dropped */
}
