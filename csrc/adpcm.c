/* IMA ADPCM (WAV and QuickTime flavors) and Microsoft ADPCM, decoded to s16 like ffmpeg
 * (docs/load.md, "ADPCM"). */
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "asrfront.h"
#include "audio.h"

static const int16_t ima_steps[89] = {
    7,     8,     9,     10,    11,    12,    13,    14,    16,    17,    19,    21,    23,
    25,    28,    31,    34,    37,    41,    45,    50,    55,    60,    66,    73,    80,
    88,    97,    107,   118,   130,   143,   157,   173,   190,   209,   230,   253,   279,
    307,   337,   371,   408,   449,   494,   544,   598,   658,   724,   796,   876,   963,
    1060,  1166,  1282,  1411,  1552,  1707,  1878,  2066,  2272,  2499,  2749,  3024,  3327,
    3660,  4026,  4428,  4871,  5358,  5894,  6484,  7132,  7845,  8630,  9493,  10442, 11487,
    12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
};

static const int8_t ima_index_adjust[8] = {-1, -1, -1, -1, 2, 4, 6, 8};

typedef struct {
    int predictor, index;
} ima_state;

static int16_t clip16(int v) {
    return (int16_t)(v > INT16_MAX ? INT16_MAX : v < INT16_MIN ? INT16_MIN : v);
}

static void ima_adapt(ima_state *s, int nibble, int diff) {
    s->predictor = clip16(nibble & 8 ? s->predictor - diff : s->predictor + diff);
    s->index += ima_index_adjust[nibble & 7];
    s->index = s->index < 0 ? 0 : s->index > 88 ? 88 : s->index;
}

/* WAV flavor: the difference as one product, ((2 * magnitude + 1) * step) / 8. */
static int16_t ima_wav_expand(ima_state *s, int nibble) {
    int step = ima_steps[s->index];
    ima_adapt(s, nibble, ((2 * (nibble & 7) + 1) * step) >> 3);
    return (int16_t)s->predictor;
}

/* QuickTime flavor: the difference as the sum of shifted steps (the reference algorithm). */
static int16_t ima_qt_expand(ima_state *s, int nibble) {
    int step = ima_steps[s->index];
    int diff = step >> 3;
    if (nibble & 4)
        diff += step;
    if (nibble & 2)
        diff += step >> 1;
    if (nibble & 1)
        diff += step >> 2;
    ima_adapt(s, nibble, diff);
    return (int16_t)s->predictor;
}

static int alloc_s16(af_decoded *d, size_t n_frames) {
    d->n_frames = n_frames;
    return af_decoded_alloc(d, AF_SFMT_S16);
}

int af_adpcm_ima_wav(af_decoded *d, const uint8_t *src, size_t size, size_t block_align) {
    const size_t ch = (size_t)d->channels, head = 4 * ch;
    /* per channel and block: 1 sample from the header, 8 per 4-byte group */
    const size_t full = block_align / head - 1; /* groups in a whole block */
    const size_t n_blocks = size / block_align, rest = size % block_align;
    size_t frames = n_blocks * (1 + 8 * full);
    if (rest >= head)
        frames += 1 + 8 * ((rest - head) / head);
    int rc = alloc_s16(d, frames);
    if (rc != AF_OK)
        return rc;
    int16_t *o = d->data;
    for (size_t b = 0; b <= n_blocks; b++) {
        const uint8_t *blk = src + b * block_align;
        size_t len = b < n_blocks ? block_align : rest;
        if (len < head)
            break;
        ima_state st[AF_MAX_CHANNELS];
        for (size_t c = 0; c < ch; c++) {
            const uint8_t *h = blk + 4 * c;
            st[c].predictor = (int16_t)(h[0] | (h[1] << 8));
            st[c].index = h[2];
            if (st[c].index > 88 || h[3] != 0) {
                af_decoded_free(d);
                return AF_ECORRUPT;
            }
            o[c] = (int16_t)st[c].predictor;
        }
        o += ch;
        size_t groups = (len - head) / head;
        for (size_t g = 0; g < groups; g++) {
            const uint8_t *p = blk + head + g * head;
            for (size_t c = 0; c < ch; c++) {
                for (int k = 0; k < 4; k++) {
                    uint8_t v = p[4 * c + (size_t)k];
                    o[(size_t)(2 * k) * ch + c] = ima_wav_expand(&st[c], v & 0x0F);
                    o[(size_t)(2 * k + 1) * ch + c] = ima_wav_expand(&st[c], v >> 4);
                }
            }
            o += 8 * ch;
        }
    }
    return AF_OK;
}

int af_adpcm_ima_qt(af_decoded *d, const uint8_t *src, size_t size) {
    const size_t ch = (size_t)d->channels, frame = 34 * ch;
    const size_t n = size / frame; /* a partial last packet is dropped */
    int rc = alloc_s16(d, n * 64);
    if (rc != AF_OK)
        return rc;
    int16_t *o = d->data;
    ima_state st[AF_MAX_CHANNELS] = {{0, 0}};
    for (size_t f = 0; f < n; f++) {
        for (size_t c = 0; c < ch; c++) {
            const uint8_t *p = src + f * frame + 34 * c;
            int header = (p[0] << 8) | p[1];
            int predictor = (int16_t)(uint16_t)(header & 0xFF80);
            int index = header & 0x7F;
            if (index > 88) {
                af_decoded_free(d);
                return AF_ECORRUPT;
            }
            /* The decoder state carries over between packets unless the header disagrees
             * with it (other step index, or predictor more than 0x7F away). */
            int dp = predictor - st[c].predictor;
            if (st[c].index != index || dp > 0x7F || dp < -0x7F) {
                st[c].predictor = predictor;
                st[c].index = index;
            }
            for (int k = 0; k < 32; k++) {
                o[(size_t)(2 * k) * ch + c] = ima_qt_expand(&st[c], p[2 + k] & 0x0F);
                o[(size_t)(2 * k + 1) * ch + c] = ima_qt_expand(&st[c], p[2 + k] >> 4);
            }
        }
        o += 64 * ch;
    }
    return AF_OK;
}

static const int ms_adaptation[16] = {230, 230, 230, 230, 307, 409, 512, 614,
                                      768, 614, 512, 409, 307, 230, 230, 230};
/* The standard predictor coefficients; ffmpeg ignores the ones stored in the fmt chunk. */
static const int ms_coef1[7] = {256, 512, 0, 192, 240, 460, 392};
static const int ms_coef2[7] = {0, -256, 0, 64, 0, -208, -232};

static int16_t rd16s(const uint8_t *p) {
    return (int16_t)(uint16_t)(p[0] | (p[1] << 8));
}

int af_adpcm_ms(af_decoded *d, const uint8_t *src, size_t size, size_t block_align) {
    const size_t ch = (size_t)d->channels, head = 7 * ch;
    /* per channel and block: 2 samples from the header, then one nibble each */
    const size_t n_blocks = size / block_align, rest = size % block_align;
    size_t frames = n_blocks * (2 + (block_align - head) * 2 / ch);
    if (rest >= head)
        frames += 2 + (rest - head) * 2 / ch;
    int rc = alloc_s16(d, frames);
    if (rc != AF_OK)
        return rc;
    int16_t *o = d->data;
    for (size_t b = 0; b <= n_blocks; b++) {
        const uint8_t *blk = src + b * block_align;
        size_t len = b < n_blocks ? block_align : rest;
        if (len < head)
            break;
        int pred[2], delta[2], s1[2], s2[2];
        for (size_t c = 0; c < ch; c++) {
            pred[c] = blk[c];
            if (pred[c] > 6) {
                af_decoded_free(d);
                return AF_ECORRUPT;
            }
            delta[c] = rd16s(blk + ch + 2 * c);
            s1[c] = rd16s(blk + 3 * ch + 2 * c);
            s2[c] = rd16s(blk + 5 * ch + 2 * c);
            o[c] = (int16_t)s2[c];
            o[ch + c] = (int16_t)s1[c];
        }
        o += 2 * ch;
        size_t c = 0;
        for (size_t i = head; i < len; i++) {
            for (int half = 0; half < 2; half++) {
                int nibble = half ? blk[i] & 0x0F : blk[i] >> 4;
                int signed_nibble = nibble & 8 ? nibble - 16 : nibble;
                /* the prediction is rounded toward zero (a division, not a shift) */
                int64_t p =
                    ((int64_t)s1[c] * ms_coef1[pred[c]] + (int64_t)s2[c] * ms_coef2[pred[c]]) /
                        256 +
                    (int64_t)signed_nibble * delta[c];
                int16_t v = (int16_t)(p > INT16_MAX ? INT16_MAX : p < INT16_MIN ? INT16_MIN : p);
                s2[c] = s1[c];
                s1[c] = v;
                *o++ = v;
                int64_t nd = (int64_t)ms_adaptation[nibble] * delta[c] / 256;
                nd = nd < 16 ? 16 : nd; /* also catches a negative delta from the header */
                delta[c] = (int)(nd > INT_MAX / 768 ? INT_MAX / 768 : nd);
                c = ch == 2 ? c ^ 1 : 0;
            }
        }
    }
    return AF_OK;
}
