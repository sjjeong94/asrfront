/* Channel layouts, mono downmix and conversion to s16, with ffmpeg's arithmetic
 * (docs/load.md, "Pipeline overview" and "Downmix"). */
#include <math.h>
#include <string.h>

#include "asrfront.h"
#include "audio.h"

int af_default_layout(int channels, uint32_t *layout) {
    static const uint32_t defaults[8][8] = {
        {AF_CH_FC},
        {AF_CH_FL, AF_CH_FR},
        {AF_CH_FL, AF_CH_FR, AF_CH_LFE},
        {AF_CH_FL, AF_CH_FR, AF_CH_FC, AF_CH_BC},
        {AF_CH_FL, AF_CH_FR, AF_CH_FC, AF_CH_BL, AF_CH_BR},
        {AF_CH_FL, AF_CH_FR, AF_CH_FC, AF_CH_LFE, AF_CH_BL, AF_CH_BR},
        {AF_CH_FL, AF_CH_FR, AF_CH_FC, AF_CH_LFE, AF_CH_BC, AF_CH_SL, AF_CH_SR},
        {AF_CH_FL, AF_CH_FR, AF_CH_FC, AF_CH_LFE, AF_CH_BL, AF_CH_BR, AF_CH_SL, AF_CH_SR},
    };
    if (channels < 1 || channels > 8)
        return AF_EUNSUPPORTED;
    memcpy(layout, defaults[channels - 1], (size_t)channels * sizeof(uint32_t));
    return AF_OK;
}

int af_mask_layout(uint32_t mask, int channels, uint32_t *layout) {
    int n = 0;
    for (int bit = 0; bit < AF_MAX_CHANNELS; bit++) {
        if (mask & (1u << bit)) {
            if (n == channels)
                return AF_EUNSUPPORTED;
            layout[n++] = 1u << bit;
        }
    }
    return n == channels ? AF_OK : AF_EUNSUPPORTED;
}

static double position_weight(uint32_t pos) {
    const double sqrt_half = 0.70710678118654752440;
    switch (pos) {
    case AF_CH_FC:
        return 1.0;
    case AF_CH_FL:
    case AF_CH_FR:
    case AF_CH_FLC:
    case AF_CH_FRC:
    case AF_CH_TFL:
    case AF_CH_TFR:
        return sqrt_half;
    case AF_CH_BL:
    case AF_CH_BR:
    case AF_CH_SL:
    case AF_CH_SR:
    case AF_CH_BC:
        return 0.5;
    default: /* LFE, top, ... */
        return 0.0;
    }
}

void af_downmix_weights(const uint32_t *layout, int channels, double *w) {
    if (channels == 1) {
        w[0] = 1.0;
        return;
    }
    double sum = 0.0;
    for (int c = 0; c < channels; c++) {
        w[c] = position_weight(layout[c]);
        sum += fabs(w[c]);
    }
    if (sum > 1.0)
        for (int c = 0; c < channels; c++)
            w[c] /= sum;
}

/* floor(v / 2^bits) without relying on the implementation-defined >> of negatives */
static int64_t floor_shift(int64_t v, int bits) {
    return v >= 0 ? v >> bits : -((-v + ((int64_t)1 << bits) - 1) >> bits);
}

static int16_t wrap16(int64_t v) {
    int32_t w = (int32_t)((uint32_t)v & 0xFFFFu);
    return (int16_t)(w >= 32768 ? w - 65536 : w);
}

static int16_t clip16(int64_t v) {
    return (int16_t)(v > INT16_MAX ? INT16_MAX : v < INT16_MIN ? INT16_MIN : v);
}

/* float -> s16, x already scaled by 32768: round half to even, saturate. Out of int32 range or
 * NaN gives -32768, like ffmpeg's SIMD conversion (its scalar tail differs; docs/load.md). */
static int16_t sat16f(float x) {
    float r = nearbyintf(x);
    if (!(r >= -2147483648.0f && r < 2147483648.0f))
        return INT16_MIN;
    return clip16((int64_t)r);
}

/* double -> s16 as ffmpeg's scalar conversion: a 64-bit lrint (out of range / NaN gives
 * INT64_MIN) truncated to 32 bits, then saturated. */
static int16_t sat16d(double x) {
    double r = nearbyint(x);
    int64_t l = (r >= -9223372036854775808.0 && r < 9223372036854775808.0) ? (int64_t)r : INT64_MIN;
    uint32_t lo = (uint32_t)((uint64_t)l & 0xFFFFFFFFu);
    int64_t t = lo >= 0x80000000u ? (int64_t)lo - 0x100000000LL : (int64_t)lo;
    return clip16(t);
}

static float s16_to_float(int16_t s) {
    return (float)s / 32768.0f;
}

static int32_t int_sample(const af_decoded *d, size_t i) {
    if (d->fmt == AF_SFMT_U8)
        return ((int32_t)((const uint8_t *)d->data)[i] - 128) * 256;
    return ((const int16_t *)d->data)[i];
}

static float float_sample(const af_decoded *d, size_t i) {
    switch (d->fmt) {
    case AF_SFMT_S16:
        return (float)((const int16_t *)d->data)[i] / 32768.0f;
    case AF_SFMT_S32:
        return (float)((const int32_t *)d->data)[i] * (1.0f / 2147483648.0f);
    default:
        return ((const float *)d->data)[i];
    }
}

int af_audio_to_mono(const af_decoded *d, float *out) {
    const int ch = d->channels;
    const size_t n = d->n_frames;
    double w[AF_MAX_CHANNELS];
    af_downmix_weights(d->layout, ch, w);

    if (d->fmt == AF_SFMT_U8 || d->fmt == AF_SFMT_S16) {
        /* int16 path: Q15 weights, rounded shift, wraps instead of saturating */
        int64_t q[AF_MAX_CHANNELS];
        for (int c = 0; c < ch; c++)
            q[c] = (int64_t)nearbyint(w[c] * 32768.0);
        for (size_t i = 0; i < n; i++) {
            int64_t v;
            if (ch == 1) {
                v = int_sample(d, i);
            } else {
                int64_t acc = 16384;
                for (int c = 0; c < ch; c++)
                    acc += (int64_t)int_sample(d, i * (size_t)ch + (size_t)c) * q[c];
                v = floor_shift(acc, 15);
            }
            out[i] = s16_to_float(wrap16(v));
        }
    } else if (d->fmt == AF_SFMT_S32 && ch == 1) {
        const int32_t *x = d->data;
        for (size_t i = 0; i < n; i++)
            out[i] = s16_to_float((int16_t)floor_shift(x[i], 16));
    } else if (d->fmt == AF_SFMT_F64) {
        const double *x = d->data;
        for (size_t i = 0; i < n; i++) {
            double acc;
            if (ch == 1) {
                acc = x[i];
            } else {
                acc = 0.0;
                for (int c = 0; c < ch; c++) {
                    double p = x[i * (size_t)ch + (size_t)c] * w[c];
                    acc = acc + p;
                }
            }
            out[i] = s16_to_float(sat16d(acc * 32768.0));
        }
    } else { /* float path: s32 multichannel, f32 */
        float wf[AF_MAX_CHANNELS];
        for (int c = 0; c < ch; c++)
            wf[c] = (float)w[c];
        for (size_t i = 0; i < n; i++) {
            float acc;
            if (ch == 1) {
                acc = float_sample(d, i);
            } else {
                acc = 0.0f;
                for (int c = 0; c < ch; c++) {
                    float p = float_sample(d, i * (size_t)ch + (size_t)c) * wf[c];
                    acc = acc + p; /* separate statements: no FMA contraction */
                }
            }
            out[i] = s16_to_float(sat16f(acc * 32768.0f));
        }
    }
    return AF_OK;
}
