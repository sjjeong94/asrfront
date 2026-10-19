/* FLAC decoding via the vendored dr_flac (csrc/third_party). */
#include <stdlib.h>
#include <string.h>

#include "asrfront.h"
#include "audio.h"
#include "dr_flac.h"

/* FLAC's channel assignments (also ffmpeg's layouts for FLAC). */
static const uint32_t flac_layouts[8][8] = {
    {AF_CH_FC},
    {AF_CH_FL, AF_CH_FR},
    {AF_CH_FL, AF_CH_FR, AF_CH_FC},
    {AF_CH_FL, AF_CH_FR, AF_CH_BL, AF_CH_BR},
    {AF_CH_FL, AF_CH_FR, AF_CH_FC, AF_CH_BL, AF_CH_BR},
    {AF_CH_FL, AF_CH_FR, AF_CH_FC, AF_CH_LFE, AF_CH_BL, AF_CH_BR},
    {AF_CH_FL, AF_CH_FR, AF_CH_FC, AF_CH_LFE, AF_CH_BC, AF_CH_SL, AF_CH_SR},
    {AF_CH_FL, AF_CH_FR, AF_CH_FC, AF_CH_LFE, AF_CH_BL, AF_CH_BR, AF_CH_SL, AF_CH_SR},
};

/* ffmpeg stores layouts that differ from FLAC's assignment in a Vorbis comment
 * WAVEFORMATEXTENSIBLE_CHANNEL_MASK=0x... and applies it when decoding. */
static void on_meta(void *user, drflac_metadata *m) {
    static const char key[] = "WAVEFORMATEXTENSIBLE_CHANNEL_MASK=";
    const size_t klen = sizeof key - 1;
    if (m->type != DRFLAC_METADATA_BLOCK_TYPE_VORBIS_COMMENT)
        return;
    drflac_vorbis_comment_iterator it;
    drflac_init_vorbis_comment_iterator(&it, m->data.vorbis_comment.commentCount,
                                        m->data.vorbis_comment.pComments);
    const char *c;
    drflac_uint32 len;
    while ((c = drflac_next_vorbis_comment(&it, &len)) != NULL) {
        if (len <= klen || len > klen + 32)
            continue;
        size_t i = 0;
        while (i < klen && (c[i] & ~0x20) == (key[i] & ~0x20)) /* ASCII case-insensitive */
            i++;
        if (i < klen)
            continue;
        char val[40];
        memcpy(val, c + klen, len - klen);
        val[len - klen] = 0;
        char *end;
        unsigned long mask = strtoul(val, &end, 0);
        if (*end == 0 && mask > 0 && mask <= 0xFFFFFFFFul)
            *(uint32_t *)user = (uint32_t)mask;
    }
}

int af_flac_decode(const uint8_t *buf, size_t size, size_t limit, af_decoded *out) {
    memset(out, 0, sizeof(*out));
    out->limit = limit;
    uint32_t mask = 0;
    drflac *f = drflac_open_memory_with_metadata(buf, size, on_meta, &mask, NULL);
    if (!f)
        return AF_ECORRUPT;
    int rc = AF_OK;
    const int ch = f->channels, bps = f->bitsPerSample;
    if (ch < 1 || ch > 8 || f->sampleRate == 0 || f->sampleRate > 0x7FFFFFFF || bps < 4 ||
        bps > 32) {
        drflac_close(f);
        return ch > 8 ? AF_EUNSUPPORTED : AF_ECORRUPT;
    }
    out->channels = ch;
    out->sample_rate = (int)f->sampleRate;
    /* Frames past the caller's limit are an error: decoding one more than the limit tells. */
    const size_t lim = af_frame_limit(out), lim_cap = lim < SIZE_MAX ? lim + 1 : SIZE_MAX;
    /* Decode as left-justified s32, growing the buffer: the header's total may be 0 (unknown)
     * or larger than what a truncated file holds. It is a hint only, never trusted beyond the
     * input size: a 10 kB file claiming 2^28 frames must not allocate gigabytes up front. */
    size_t cap = 65536 + size; /* frames; enough for any common compression ratio */
    /* One frame more than the header's total, so that a truthful total ends with a short read
     * (n < cap) instead of growing the full buffer just before EOF. */
    if (f->totalPCMFrameCount > 0 && f->totalPCMFrameCount < cap)
        cap = (size_t)f->totalPCMFrameCount + 1;
    cap = cap < lim_cap ? cap : lim_cap;
    size_t n = 0;
    int32_t *pcm = malloc(cap * (size_t)ch * sizeof(int32_t));
    while (pcm && n <= lim) {
        if (n == cap) {
            size_t ncap = cap * 2 < lim_cap ? cap * 2 : lim_cap;
            int32_t *p = ncap > cap ? realloc(pcm, ncap * (size_t)ch * sizeof(int32_t)) : NULL;
            if (!p) {
                free(pcm);
                pcm = NULL;
                break;
            }
            pcm = p;
            cap = ncap;
        }
        drflac_uint64 got = drflac_read_pcm_frames_s32(f, cap - n, pcm + n * (size_t)ch);
        if (got == 0)
            break;
        n += (size_t)got;
    }
    out->n_frames = n;
    if (!mask || af_mask_layout(mask, ch, out->layout) != AF_OK)
        memcpy(out->layout, flac_layouts[ch - 1], (size_t)ch * sizeof(uint32_t));
    drflac_close(f);
    if (!pcm)
        return AF_ENOMEM;
    if (n > lim) {
        free(pcm);
        return AF_ETOOLONG;
    }

    if (bps <= 16) { /* ffmpeg decodes up to 16 bits as s16 (left-justified) */
        int16_t *s = malloc(n * (size_t)ch * sizeof(int16_t) + 1);
        if (!s)
            rc = AF_ENOMEM;
        else
            for (size_t i = 0; i < n * (size_t)ch; i++)
                s[i] = (int16_t)(pcm[i] / 65536); /* exact: the low 16 bits are zero */
        free(pcm);
        out->fmt = AF_SFMT_S16;
        out->data = s;
    } else {
        out->fmt = AF_SFMT_S32;
        out->data = pcm;
    }
    return rc;
}
