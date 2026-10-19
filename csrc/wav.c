/* WAV (RIFF / RF64 / BW64) and Sony Wave64 decoding. */
#include <stdlib.h>
#include <string.h>

#include "asrfront.h"
#include "audio.h"

enum {
    WAV_TAG_PCM = 0x0001,
    WAV_TAG_MS_ADPCM = 0x0002,
    WAV_TAG_FLOAT = 0x0003,
    WAV_TAG_ALAW = 0x0006,
    WAV_TAG_MULAW = 0x0007,
    WAV_TAG_IMA_ADPCM = 0x0011,
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

/* The parts of a WAVEFORMATEX / WAVEFORMATEXTENSIBLE ("fmt " chunk) that decoding needs. */
typedef struct {
    int tag, channels, bits;
    uint32_t sample_rate, mask, block_align;
} wav_fmt;

static int parse_fmt(const uint8_t *body, uint64_t size, wav_fmt *f) {
    if (size < 16)
        return AF_ECORRUPT;
    f->tag = rd16(body);
    f->channels = rd16(body + 2);
    f->sample_rate = rd32(body + 4);
    f->block_align = rd16(body + 12);
    f->bits = rd16(body + 14);
    f->mask = 0;
    if (f->tag == WAV_TAG_EXTENSIBLE) {
        if (size < 40 || rd16(body + 16) < 22)
            return AF_ECORRUPT;
        f->mask = rd32(body + 20);
        f->tag = rd16(body + 24); /* first two bytes of the sub-format GUID */
    }
    if (f->channels <= 0 || f->sample_rate == 0 || f->sample_rate > 0x7FFFFFFF)
        return AF_ECORRUPT;
    if (f->channels > AF_MAX_CHANNELS)
        return AF_EUNSUPPORTED;
    const uint32_t ch = (uint32_t)f->channels;
    switch (f->tag) {
    case WAV_TAG_PCM:
        return f->bits == 8 || f->bits == 16 || f->bits == 24 || f->bits == 32 ? AF_OK
                                                                               : AF_EUNSUPPORTED;
    case WAV_TAG_FLOAT:
        return f->bits == 32 || f->bits == 64 ? AF_OK : AF_EUNSUPPORTED;
    case WAV_TAG_ALAW:
    case WAV_TAG_MULAW:
        return f->bits == 8 ? AF_OK : AF_EUNSUPPORTED;
    case WAV_TAG_IMA_ADPCM: /* per channel: a 4-byte header, then groups of 4 bytes */
        if (f->bits != 4)
            return AF_EUNSUPPORTED; /* 2/3/5-bit IMA variants */
        return f->block_align >= 4 * ch && (f->block_align - 4 * ch) % (4 * ch) == 0 ? AF_OK
                                                                                     : AF_ECORRUPT;
    case WAV_TAG_MS_ADPCM: /* ffmpeg decodes it for one or two channels only */
        if (ch > 2)
            return AF_EUNSUPPORTED;
        return f->block_align >= 7 * ch ? AF_OK : AF_ECORRUPT;
    default:
        return AF_EUNSUPPORTED;
    }
}

/* Decode `size` bytes of sample data (whole frames / blocks, partial ones as available). */
static int decode_data(const wav_fmt *f, const uint8_t *data, size_t size, af_decoded *out) {
    out->channels = f->channels;
    out->sample_rate = (int)f->sample_rate;
    int rc = f->mask ? af_mask_layout(f->mask, f->channels, out->layout)
                     : af_default_layout(f->channels, out->layout);
    if (rc == AF_EUNSUPPORTED && f->mask)
        rc = af_default_layout(f->channels, out->layout);
    if (rc != AF_OK)
        return rc;
    switch (f->tag) {
    case WAV_TAG_IMA_ADPCM:
        return af_adpcm_ima_wav(out, data, size, f->block_align);
    case WAV_TAG_MS_ADPCM:
        return af_adpcm_ms(out, data, size, f->block_align);
    case WAV_TAG_FLOAT:
        return af_pcm_decode_frames(out, f->bits == 32 ? AF_PCM_F32LE : AF_PCM_F64LE, data, size);
    case WAV_TAG_ALAW:
        return af_pcm_decode_frames(out, AF_PCM_ALAW, data, size);
    case WAV_TAG_MULAW:
        return af_pcm_decode_frames(out, AF_PCM_MULAW, data, size);
    default: {
        static const af_pcm_codec pcm[4] = {AF_PCM_U8, AF_PCM_S16LE, AF_PCM_S24LE, AF_PCM_S32LE};
        return af_pcm_decode_frames(out, pcm[f->bits / 8 - 1], data, size);
    }
    }
}

int af_wav_decode(const uint8_t *buf, size_t size, size_t limit, af_decoded *out) {
    memset(out, 0, sizeof(*out));
    out->limit = limit;
    if (size < 12 || memcmp(buf + 8, "WAVE", 4) != 0)
        return AF_ECORRUPT;
    int rf64 = memcmp(buf, "RF64", 4) == 0 || memcmp(buf, "BW64", 4) == 0;
    if (!rf64 && memcmp(buf, "RIFF", 4) != 0)
        return AF_ECORRUPT;

    wav_fmt fmt;
    int have_fmt = 0;
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
            return decode_data(&fmt, body, (size_t)data_size, out);
        }

        if (ck_size > avail)
            return AF_ECORRUPT;
        if (memcmp(ck, "fmt ", 4) == 0) {
            int rc = parse_fmt(body, ck_size, &fmt);
            if (rc != AF_OK)
                return rc;
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

/* Wave64: RIFF with GUID chunk ids and 64-bit sizes that include the 24-byte chunk header;
 * chunks are 8-byte aligned. */
static const uint8_t W64_GUID_TAIL[12] = {0xF3, 0xAC, 0xD3, 0x11, 0x8C, 0xD1,
                                          0x00, 0xC0, 0x4F, 0x8E, 0xDB, 0x8A};
const uint8_t af_w64_riff_guid[16] = {'r',  'i',  'f',  'f',  0x2E, 0x91, 0xCF, 0x11,
                                      0xA5, 0xD6, 0x28, 0xDB, 0x04, 0xC1, 0x00, 0x00};

static int w64_id(const uint8_t *guid, const char *fourcc) {
    return memcmp(guid, fourcc, 4) == 0 && memcmp(guid + 4, W64_GUID_TAIL, 12) == 0;
}

int af_w64_decode(const uint8_t *buf, size_t size, size_t limit, af_decoded *out) {
    memset(out, 0, sizeof(*out));
    out->limit = limit;
    if (size < 40 || memcmp(buf, af_w64_riff_guid, 16) != 0 || !w64_id(buf + 24, "wave"))
        return AF_ECORRUPT;
    wav_fmt fmt;
    int have_fmt = 0;
    size_t pos = 40;
    for (;;) {
        if (size - pos < 24)
            return AF_ECORRUPT; /* no data chunk */
        const uint8_t *ck = buf + pos;
        uint64_t ck_size = rd64(ck + 16); /* including the header */
        const uint8_t *body = ck + 24;
        size_t avail = size - pos - 24;
        if (ck_size <= 24 || ck_size > INT64_MAX)
            return AF_ECORRUPT;
        uint64_t body_size = ck_size - 24;

        if (w64_id(ck, "data")) {
            if (!have_fmt)
                return AF_ECORRUPT;
            return decode_data(&fmt, body, body_size > avail ? avail : (size_t)body_size, out);
        }
        if (body_size > avail)
            return AF_ECORRUPT;
        if (w64_id(ck, "fmt ")) {
            int rc = parse_fmt(body, body_size, &fmt);
            if (rc != AF_OK)
                return rc;
            have_fmt = 1;
        }
        uint64_t next = ((uint64_t)pos + ck_size + 7) & ~(uint64_t)7;
        if (next > size)
            return AF_ECORRUPT;
        pos = (size_t)next;
    }
}
