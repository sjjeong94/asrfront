/* WAV (RIFF / RF64 / BW64) and raw PCM decoding. */
#include <stdlib.h>
#include <string.h>

#include "asrfront.h"
#include "audio.h"

enum {
    WAV_TAG_PCM = 0x0001,
    WAV_TAG_FLOAT = 0x0003,
    WAV_TAG_ALAW = 0x0006,
    WAV_TAG_MULAW = 0x0007,
    WAV_TAG_EXTENSIBLE = 0xFFFE,
};

static uint16_t rd16(const uint8_t *p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}

static uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t rd64(const uint8_t *p) {
    return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32);
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

static int alloc_samples(af_decoded *d, size_t elem_size) {
    size_t n = d->n_frames * (size_t)d->channels;
    if (d->channels > 0 && n / (size_t)d->channels != d->n_frames)
        return AF_ECORRUPT;
    if (n > SIZE_MAX / elem_size)
        return AF_ENOMEM;
    d->data = malloc(n * elem_size > 0 ? n * elem_size : 1);
    return d->data ? AF_OK : AF_ENOMEM;
}

/* Convert `n` little-endian coded samples into d->data. */
static int convert(af_decoded *d, int tag, int bytes, const uint8_t *src) {
    size_t n = d->n_frames * (size_t)d->channels;
    int rc;
    switch (tag) {
    case WAV_TAG_PCM:
        if (bytes == 1) {
            d->fmt = AF_SFMT_U8;
            if ((rc = alloc_samples(d, 1)) != AF_OK)
                return rc;
            memcpy(d->data, src, n);
        } else if (bytes == 2) {
            d->fmt = AF_SFMT_S16;
            if ((rc = alloc_samples(d, 2)) != AF_OK)
                return rc;
            int16_t *o = d->data;
            for (size_t i = 0; i < n; i++)
                o[i] = (int16_t)rd16(src + 2 * i);
        } else { /* 3 or 4 bytes: the sample goes to the top bits of an s32 */
            d->fmt = AF_SFMT_S32;
            if ((rc = alloc_samples(d, 4)) != AF_OK)
                return rc;
            int32_t *o = d->data;
            if (bytes == 3) {
                for (size_t i = 0; i < n; i++) {
                    const uint8_t *p = src + 3 * i;
                    o[i] = (int32_t)(((uint32_t)p[0] << 8) | ((uint32_t)p[1] << 16) |
                                     ((uint32_t)p[2] << 24));
                }
            } else {
                for (size_t i = 0; i < n; i++)
                    o[i] = (int32_t)rd32(src + 4 * i);
            }
        }
        return AF_OK;
    case WAV_TAG_FLOAT:
        if (bytes == 4) {
            d->fmt = AF_SFMT_F32;
            if ((rc = alloc_samples(d, 4)) != AF_OK)
                return rc;
            float *o = d->data;
            for (size_t i = 0; i < n; i++) {
                uint32_t u = rd32(src + 4 * i);
                memcpy(&o[i], &u, 4);
            }
        } else {
            d->fmt = AF_SFMT_F64;
            if ((rc = alloc_samples(d, 8)) != AF_OK)
                return rc;
            double *o = d->data;
            for (size_t i = 0; i < n; i++) {
                uint64_t u = rd64(src + 8 * i);
                memcpy(&o[i], &u, 8);
            }
        }
        return AF_OK;
    case WAV_TAG_ALAW:
    case WAV_TAG_MULAW: {
        d->fmt = AF_SFMT_S16;
        if ((rc = alloc_samples(d, 2)) != AF_OK)
            return rc;
        int16_t *o = d->data;
        for (size_t i = 0; i < n; i++)
            o[i] = tag == WAV_TAG_ALAW ? alaw_to_s16(src[i]) : mulaw_to_s16(src[i]);
        return AF_OK;
    }
    default:
        return AF_EUNSUPPORTED;
    }
}

int af_wav_decode(const uint8_t *buf, size_t size, af_decoded *out) {
    memset(out, 0, sizeof(*out));
    if (size < 12 || memcmp(buf + 8, "WAVE", 4) != 0)
        return AF_ECORRUPT;
    int rf64 = memcmp(buf, "RF64", 4) == 0 || memcmp(buf, "BW64", 4) == 0;
    if (!rf64 && memcmp(buf, "RIFF", 4) != 0)
        return AF_ECORRUPT;

    int have_fmt = 0, tag = 0, channels = 0, bits = 0;
    uint32_t sample_rate = 0, mask = 0;
    uint64_t ds64_data_size = 0;
    int have_ds64 = 0;
    size_t pos = 12;
    for (;;) {
        if (size - pos < 8)
            return AF_ECORRUPT; /* no data chunk */
        const uint8_t *ck = buf + pos;
        uint32_t ck_size = rd32(ck + 4);
        const uint8_t *body = ck + 8;
        size_t avail = size - pos - 8;

        if (memcmp(ck, "data", 4) == 0) {
            if (!have_fmt)
                return AF_ECORRUPT;
            uint64_t data_size = ck_size;
            if (rf64 && have_ds64 && ck_size == 0xFFFFFFFFu)
                data_size = ds64_data_size;
            /* streamed / unfinished / truncated files: read to the end of the buffer */
            if (data_size == 0 || data_size == 0xFFFFFFFFu || data_size > avail)
                data_size = avail;
            int bytes = (bits + 7) / 8;
            size_t frame = (size_t)bytes * (size_t)channels;
            out->channels = channels;
            out->sample_rate = (int)sample_rate;
            out->n_frames = (size_t)data_size / frame;
            int rc = mask ? af_mask_layout(mask, channels, out->layout)
                          : af_default_layout(channels, out->layout);
            if (rc == AF_EUNSUPPORTED && mask)
                rc = af_default_layout(channels, out->layout);
            if (rc != AF_OK)
                return rc;
            rc = convert(out, tag, bytes, body);
            if (rc != AF_OK)
                af_decoded_free(out);
            return rc;
        }

        if (ck_size > avail)
            return AF_ECORRUPT;
        if (memcmp(ck, "fmt ", 4) == 0) {
            if (ck_size < 16)
                return AF_ECORRUPT;
            tag = rd16(body);
            channels = rd16(body + 2);
            sample_rate = rd32(body + 4);
            bits = rd16(body + 14);
            if (tag == WAV_TAG_EXTENSIBLE) {
                if (ck_size < 40 || rd16(body + 16) < 22)
                    return AF_ECORRUPT;
                mask = rd32(body + 20);
                tag = rd16(body + 24); /* first two bytes of the sub-format GUID */
            }
            if (channels <= 0 || sample_rate == 0 || sample_rate > 0x7FFFFFFF)
                return AF_ECORRUPT;
            if (channels > AF_MAX_CHANNELS)
                return AF_EUNSUPPORTED;
            int ok =
                (tag == WAV_TAG_PCM && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) ||
                (tag == WAV_TAG_FLOAT && (bits == 32 || bits == 64)) ||
                ((tag == WAV_TAG_ALAW || tag == WAV_TAG_MULAW) && bits == 8);
            if (!ok)
                return AF_EUNSUPPORTED;
            have_fmt = 1;
        } else if (memcmp(ck, "ds64", 4) == 0 && rf64) {
            if (ck_size < 16)
                return AF_ECORRUPT;
            ds64_data_size = rd64(body + 8);
            have_ds64 = 1;
        }
        pos += 8 + (size_t)ck_size + (ck_size & 1); /* chunks are word aligned */
        if (pos > size)
            return AF_ECORRUPT;
    }
}

int af_pcm_decode_raw(const uint8_t *buf, size_t size, af_decoded *out) {
    memset(out, 0, sizeof(*out));
    out->channels = 1;
    out->sample_rate = AF_SAMPLE_RATE;
    out->layout[0] = AF_CH_FC;
    out->n_frames = size / 2; /* a trailing odd byte is dropped */
    return convert(out, WAV_TAG_PCM, 2, buf);
}
