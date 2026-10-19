/* Apple Core Audio Format (CAF) decoding, and CoreAudio channel layouts (also used by AIFF). */
#include <string.h>

#include "asrfront.h"
#include "audio.h"

static uint32_t rd32be(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

static uint64_t rd64be(const uint8_t *p) {
    return ((uint64_t)rd32be(p) << 32) | rd32be(p + 4);
}

/* Position of a CoreAudio channel label (AudioChannelLabel), 0 for labels that get no weight
 * in the downmix (LFE2, wide, surround direct, matrix Lt/Rt, ...). */
static uint32_t label_position(uint32_t label) {
    switch (label) {
    case 1:
        return AF_CH_FL;
    case 2:
        return AF_CH_FR;
    case 3:
        return AF_CH_FC;
    case 4:
        return AF_CH_LFE;
    case 5: /* "left surround": ffmpeg's side left */
        return AF_CH_SL;
    case 6:
        return AF_CH_SR;
    case 7:
        return AF_CH_FLC;
    case 8:
        return AF_CH_FRC;
    case 9:
        return AF_CH_BC;
    case 12:
        return AF_CH_TC;
    case 13:
        return AF_CH_TFL;
    case 14:
        return AF_CH_TFC;
    case 15:
        return AF_CH_TFR;
    case 16:
        return AF_CH_TBL;
    case 17:
        return AF_CH_TBC;
    case 18:
        return AF_CH_TBR;
    case 33: /* rear surround */
        return AF_CH_BL;
    case 34:
        return AF_CH_BR;
    default:
        return 0;
    }
}

/* Predefined layout tags whose downmix differs from the default layout for their channel
 * count, as far as ffmpeg can downmix them at all (it rejects layouts that are not in its
 * native channel order, e.g. kAudioChannelLayoutTag_MPEG_7_1_C, which then fall back to the
 * default layout here). The tag's low 16 bits are the channel count. */
static const struct {
    uint16_t tag;
    uint32_t pos[8];
} ca_layouts[] = {
    {108, {AF_CH_FL, AF_CH_FR, AF_CH_SL, AF_CH_SR}}, /* Quadraphonic */
    /* Cube */
    {112, {AF_CH_FL, AF_CH_FR, AF_CH_BL, AF_CH_BR, AF_CH_TFL, AF_CH_TFR, AF_CH_TBL, AF_CH_TBR}},
    {113, {AF_CH_FL, AF_CH_FR, AF_CH_FC}},                                      /* MPEG_3_0_A */
    {130, {AF_CH_FL, AF_CH_FR, AF_CH_FC, AF_CH_LFE, AF_CH_SL, AF_CH_SR, 0, 0}}, /* SMPTE_DTV */
    {131, {AF_CH_FL, AF_CH_FR, AF_CH_BC}},                                      /* ITU_2_1 */
    {132, {AF_CH_FL, AF_CH_FR, AF_CH_SL, AF_CH_SR}},                            /* ITU_2_2 */
    {134, {AF_CH_FL, AF_CH_FR, AF_CH_LFE, AF_CH_BC}},                           /* DVD_5 */
    {135, {AF_CH_FL, AF_CH_FR, AF_CH_LFE, AF_CH_SL, AF_CH_SR}},                 /* DVD_6 */
    {136, {AF_CH_FL, AF_CH_FR, AF_CH_FC, AF_CH_LFE}},                           /* DVD_10 */
    {137, {AF_CH_FL, AF_CH_FR, AF_CH_FC, AF_CH_LFE, AF_CH_BC}},                 /* DVD_11 */
    {149, {AF_CH_FC, AF_CH_LFE}},                                               /* AC3_1_0_1 */
};

int af_coreaudio_layout(const uint8_t *body, size_t size, int channels, uint32_t *layout) {
    if (size < 12)
        return AF_ECORRUPT; /* ffmpeg rejects the whole file */
    uint32_t tag = rd32be(body), bitmap = rd32be(body + 4), n_desc = rd32be(body + 8);
    if (tag == 0) { /* kAudioChannelLayoutTag_UseChannelDescriptions: one label each */
        if (n_desc < (uint32_t)channels || (size - 12) / 20 < (size_t)channels)
            return af_default_layout(channels, layout);
        for (int c = 0; c < channels; c++)
            layout[c] = label_position(rd32be(body + 12 + 20 * (size_t)c));
        return AF_OK;
    }
    if (tag == 0x10000) { /* kAudioChannelLayoutTag_UseChannelBitmap: a WAVE channel mask */
        if (af_mask_layout(bitmap, channels, layout) == AF_OK)
            return AF_OK;
    } else if ((int)(tag & 0xFFFF) == channels) {
        for (size_t i = 0; i < sizeof ca_layouts / sizeof ca_layouts[0]; i++) {
            if (ca_layouts[i].tag == tag >> 16) {
                memcpy(layout, ca_layouts[i].pos, (size_t)channels * sizeof(uint32_t));
                return AF_OK;
            }
        }
    }
    return af_default_layout(channels, layout);
}

int af_caf_decode(const uint8_t *buf, size_t size, size_t limit, af_decoded *out) {
    memset(out, 0, sizeof(*out));
    out->limit = limit;
    /* file header: 'caff', version 1, flags; then 'desc' must be the first chunk */
    if (size < 8 + 12 + 32 || memcmp(buf, "caff", 4) != 0 || buf[4] != 0 || buf[5] != 1 ||
        memcmp(buf + 8, "desc", 4) != 0)
        return AF_ECORRUPT;
    uint64_t desc_size = rd64be(buf + 12);
    if (desc_size < 32 || desc_size > size - 20)
        return AF_ECORRUPT;
    const uint8_t *desc = buf + 20;
    uint64_t rate_bits = rd64be(desc);
    double rate;
    memcpy(&rate, &rate_bits, sizeof rate);
    const uint8_t *format = desc + 8;
    uint32_t flags = rd32be(desc + 12), bytes_per_packet = rd32be(desc + 16);
    uint32_t frames_per_packet = rd32be(desc + 20), channels = rd32be(desc + 24);
    uint32_t bits = rd32be(desc + 28);
    if (!(rate >= 1.0) || channels == 0 || channels > 0x7FFFFFFF)
        return AF_ECORRUPT;
    if (channels > AF_MAX_CHANNELS)
        return af_fail(out, AF_EUNSUPPORTED, "%u channels (at most %d)", (unsigned)channels,
                       AF_MAX_CHANNELS);
    out->channels = (int)channels;
    /* truncated to an integer, like ffmpeg; anything this large is rejected by the resampler */
    out->sample_rate = rate < 2147483647.0 ? (int)rate : 2147483647;

    int ima4 = 0;
    af_pcm_codec codec = AF_PCM_S16BE;
    if (memcmp(format, "lpcm", 4) == 0) {
        int le = (flags & 2) != 0; /* kCAFLinearPCMFormatFlagIsLittleEndian */
        if (bytes_per_packet == 0 || frames_per_packet == 0)
            return AF_ECORRUPT;
        /* The sample size comes from the bit depth alone (bytes_per_packet is not used);
         * 8-bit samples are always signed. */
        if (flags & 1) { /* kCAFLinearPCMFormatFlagIsFloat */
            if (bits == 32)
                codec = le ? AF_PCM_F32LE : AF_PCM_F32BE;
            else if (bits == 64)
                codec = le ? AF_PCM_F64LE : AF_PCM_F64BE;
            else
                return af_fail(out, AF_EUNSUPPORTED, "%u-bit float", (unsigned)bits);
        } else if (bits >= 1 && bits <= 8) {
            codec = AF_PCM_S8;
        } else if (bits >= 9 && bits <= 16) {
            codec = le ? AF_PCM_S16LE : AF_PCM_S16BE;
        } else if (bits >= 17 && bits <= 24) {
            codec = le ? AF_PCM_S24LE : AF_PCM_S24BE;
        } else if (bits >= 25 && bits <= 32) {
            codec = le ? AF_PCM_S32LE : AF_PCM_S32BE;
        } else {
            return af_fail(out, AF_EUNSUPPORTED, "%u-bit PCM", (unsigned)bits);
        }
    } else if (memcmp(format, "alaw", 4) == 0) {
        codec = AF_PCM_ALAW;
    } else if (memcmp(format, "ulaw", 4) == 0) {
        codec = AF_PCM_MULAW;
    } else if (memcmp(format, "ima4", 4) == 0) {
        if (bytes_per_packet != 34 * channels || frames_per_packet != 64)
            return AF_ECORRUPT;
        ima4 = 1;
    } else {
        return af_fail(out, AF_EUNSUPPORTED, "format '%.4s'", (const char *)format);
    }

    const uint8_t *chan = NULL, *data = NULL;
    size_t chan_size = 0, data_size = 0;
    size_t pos = 20 + (size_t)desc_size;
    while (size - pos >= 12) {
        const uint8_t *ck = buf + pos;
        uint64_t ck_size = rd64be(ck + 4);
        const uint8_t *body = ck + 12;
        size_t avail = size - pos - 12;
        if (memcmp(ck, "data", 4) == 0) {
            if (avail < 4)
                return AF_ECORRUPT;
            /* 4 bytes of edit count, then the samples. A size of -1 (streamed), one that
             * leaves no samples, or one beyond the file means "to the end of the file". */
            int64_t n = (int64_t)ck_size - 4;
            data = body + 4;
            if (n <= 0 || (uint64_t)n > avail - 4) {
                data_size = avail - 4;
                break;
            }
            data_size = (size_t)n; /* a later data chunk replaces this one, as in ffmpeg */
        } else if (ck_size > avail) {
            if (data)
                break; /* damaged trailing chunk */
            return AF_ECORRUPT;
        } else if (memcmp(ck, "chan", 4) == 0) {
            chan = body;
            chan_size = (size_t)ck_size;
        }
        pos += 12 + (size_t)ck_size;
    }
    if (!data)
        return AF_ECORRUPT;
    int rc = chan ? af_coreaudio_layout(chan, chan_size, (int)channels, out->layout)
                  : af_default_layout((int)channels, out->layout);
    if (rc != AF_OK)
        return rc;
    if (ima4)
        return af_adpcm_ima_qt(out, data, data_size);
    return af_pcm_decode_frames(out, codec, data, data_size);
}
