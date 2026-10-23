/* AIFF and AIFF-C decoding. */
#include <string.h>

#include "asrfront.h"
#include "audio.h"

static uint32_t rd32be(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

/* 80-bit IEEE extended sample rate, rounded to the nearest integer (halves up); 0 if it is not
 * a positive number below 2^31. */
static uint32_t ext80_rate(const uint8_t *p) {
    int exp = ((p[0] & 0x7F) << 8) | p[1];
    uint64_t mant = ((uint64_t)rd32be(p + 2) << 32) | rd32be(p + 6);
    int shift = exp - 16383 - 63; /* value = mant * 2^shift */
    if ((p[0] & 0x80) || mant == 0 || shift >= 0 || shift <= -64)
        return 0;
    uint64_t r = (mant >> -shift) + ((mant >> (-shift - 1)) & 1);
    return r <= 0x7FFFFFFF ? (uint32_t)r : 0;
}

static int has_id(const uint8_t *p, const char *a, const char *b) {
    return memcmp(p, a, 4) == 0 || memcmp(p, b, 4) == 0;
}

/* Sample codec from the bit depth (AIFF, and AIFF-C 'NONE' / 'twos'): rounded up to bytes. */
static int pcm_by_bits(int bits, af_pcm_codec *c) {
    if (bits <= 8)
        *c = AF_PCM_S8;
    else if (bits <= 16)
        *c = AF_PCM_S16BE;
    else if (bits <= 24)
        *c = AF_PCM_S24BE;
    else if (bits <= 32)
        *c = AF_PCM_S32BE;
    else
        return AF_EUNSUPPORTED;
    return AF_OK;
}

int af_aiff_decode(const uint8_t *buf, size_t size, size_t limit, af_decoded *out) {
    af_decoded_init(out, limit);
    if (size < 12 || memcmp(buf, "FORM", 4) != 0)
        return AF_ECORRUPT;
    int aifc = memcmp(buf + 8, "AIFC", 4) == 0;
    if (!aifc && memcmp(buf + 8, "AIFF", 4) != 0)
        return AF_ECORRUPT;

    const uint8_t *comm = NULL, *chan = NULL, *data = NULL;
    size_t comm_size = 0, chan_size = 0, data_size = 0;
    /* Chunks in any order (COMM may follow SSND); stop at an SSND that runs to the end. */
    size_t pos = 12;
    while (size - pos >= 8) {
        const uint8_t *ck = buf + pos;
        uint32_t ck_size = rd32be(ck + 4);
        const uint8_t *body = ck + 8;
        size_t avail = size - pos - 8;
        if (memcmp(ck, "SSND", 4) == 0 && !data) {
            if (ck_size < 8 || avail < 8)
                return AF_ECORRUPT;
            int to_end = ck_size > avail; /* also 0xFFFFFFFF: streamed */
            size_t n = to_end ? avail : ck_size;
            uint32_t offset = rd32be(body); /* bytes to skip before the samples */
            if (offset > n - 8)
                return AF_ECORRUPT;
            data = body + 8 + offset;
            data_size = n - 8 - offset;
            if (to_end)
                break;
        } else if (ck_size > avail) {
            if (data && comm)
                break; /* damaged trailing chunk */
            return AF_ECORRUPT;
        } else if (memcmp(ck, "COMM", 4) == 0) {
            comm = body;
            comm_size = ck_size;
        } else if (memcmp(ck, "CHAN", 4) == 0) {
            chan = body;
            chan_size = ck_size;
        }
        pos += 8 + (size_t)ck_size + (ck_size & 1); /* chunks are word aligned */
        if (pos > size)
            break;
    }
    if (!comm || !data || comm_size < (aifc ? 22u : 18u))
        return AF_ECORRUPT;

    int channels = (int16_t)(uint16_t)((comm[0] << 8) | comm[1]);
    int bits = (comm[6] << 8) | comm[7]; /* comm[2..5]: frame count, not used (ffmpeg reads
                                          * the whole SSND chunk) */
    uint32_t rate = ext80_rate(comm + 8);
    if (channels <= 0 || rate == 0)
        return AF_ECORRUPT;
    if (channels > AF_MAX_CHANNELS)
        return af_fail(out, AF_EUNSUPPORTED, "%d channels (at most %d)", channels, AF_MAX_CHANNELS);
    out->channels = channels;
    out->sample_rate = (int)rate;
    int rc = chan ? af_coreaudio_layout(chan, chan_size, channels, out->layout)
                  : af_default_layout(channels, out->layout);
    if (rc != AF_OK)
        return rc;

    af_pcm_codec codec;
    const uint8_t *comp = comm + 18;
    if (!aifc || has_id(comp, "NONE", "none") || memcmp(comp, "twos", 4) == 0) {
        rc = pcm_by_bits(bits, &codec);
        if (rc != AF_OK)
            return af_fail(out, rc, "%d-bit PCM", bits);
    } else if (memcmp(comp, "sowt", 4) == 0) {
        codec = AF_PCM_S16LE;
    } else if (memcmp(comp, "raw ", 4) == 0) {
        codec = AF_PCM_U8;
    } else if (memcmp(comp, "in24", 4) == 0) {
        codec = AF_PCM_S24BE;
    } else if (memcmp(comp, "in32", 4) == 0) {
        codec = AF_PCM_S32BE;
    } else if (has_id(comp, "fl32", "FL32")) {
        codec = AF_PCM_F32BE;
    } else if (has_id(comp, "fl64", "FL64")) {
        codec = AF_PCM_F64BE;
    } else if (has_id(comp, "alaw", "ALAW")) {
        codec = AF_PCM_ALAW;
    } else if (has_id(comp, "ulaw", "ULAW")) {
        codec = AF_PCM_MULAW;
    } else if (memcmp(comp, "ima4", 4) == 0) {
        return af_adpcm_ima_qt(out, data, data_size);
    } else {
        return af_fail(out, AF_EUNSUPPORTED, "compression '%.4s'", (const char *)comp);
    }
    return af_pcm_decode_frames(out, codec, data, data_size);
}
