/* Channel layouts, mono downmix and conversion to s16, with ffmpeg's arithmetic
 * (docs/load.md, "Pipeline overview" and "Downmix"). */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "asrfront.h"
#include "audio.h"
#include "resample.h"

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
    if (!(x >= -2147483648.0f && x < 2147483648.0f)) /* NaN, or rint(x) outside int32 */
        return INT16_MIN;
    if (x >= 32767.0f)
        return INT16_MAX;
    if (x <= -32768.0f)
        return INT16_MIN;
    /* round half to even without libm (nearbyintf is a call on baseline x86-64): exact for
     * |x| < 2^22 in the default rounding mode; -ffp-contract=off keeps the two steps apart */
    float r = (x + 12582912.0f) - 12582912.0f;
    return (int16_t)r;
}

/* double -> s16 as ffmpeg's scalar conversion: a 64-bit lrint (out of range / NaN gives
 * INT64_MIN) truncated to 32 bits, then saturated. */
static int16_t sat16d(double x) {
    if (x > -32768.0 && x < 32767.0) { /* common case: magic-number rint, exact here */
        double r = (x + 6755399441055744.0) - 6755399441055744.0;
        return (int16_t)r;
    }
    double r = nearbyint(x);
    int64_t l = (r >= -9223372036854775808.0 && r < 9223372036854775808.0) ? (int64_t)r : INT64_MIN;
    uint32_t lo = (uint32_t)((uint64_t)l & 0xFFFFFFFFu);
    int64_t t = lo >= 0x80000000u ? (int64_t)lo - 0x100000000LL : (int64_t)lo;
    return clip16(t);
}

static float s16_to_float(int16_t s) {
    return (float)s / 32768.0f;
}

/* ffmpeg's arithmetic path (docs/load.md, "Internal sample format by decoder output"). */
typedef enum { PATH_INT16, PATH_S32_SHIFT, PATH_FLOAT, PATH_DOUBLE } path_t;

static path_t select_path(const af_decoded *d, int resample) {
    if (d->fmt == AF_SFMT_U8 || (d->fmt == AF_SFMT_S16 && !resample))
        return PATH_INT16;
    if (d->fmt == AF_SFMT_S32 && d->channels == 1 && !resample)
        return PATH_S32_SHIFT;
    return d->fmt == AF_SFMT_F64 ? PATH_DOUBLE : PATH_FLOAT;
}

static const size_t elem_size[] = {sizeof(int32_t), sizeof(int32_t), sizeof(float), sizeof(double)};

/* Channel c of d in the path's working type, into dst (n_frames elements). */
/* Channel c of d in the path's working type, into dst (n_frames elements). One loop per
 * (path, format) pair so the inner loops are branch-free. */
#define EXTRACT_LOOP(TYPE_IN, TYPE_OUT, EXPR)                                                      \
    do {                                                                                           \
        const TYPE_IN *src_ = (const TYPE_IN *)d->data + c;                                        \
        TYPE_OUT *dst_ = dst;                                                                      \
        for (size_t i = 0; i < n; i++) {                                                           \
            TYPE_IN v = src_[i * ch];                                                              \
            dst_[i] = (EXPR);                                                                      \
        }                                                                                          \
    } while (0)

static void extract(const af_decoded *d, path_t path, int c, void *dst) {
    const size_t n = d->n_frames, ch = (size_t)d->channels;
    switch (path) {
    case PATH_INT16:
        if (d->fmt == AF_SFMT_U8)
            EXTRACT_LOOP(uint8_t, int32_t, ((int32_t)v - 128) * 256);
        else
            EXTRACT_LOOP(int16_t, int32_t, (int32_t)v);
        break;
    case PATH_S32_SHIFT:
        EXTRACT_LOOP(int32_t, int32_t, (int32_t)floor_shift(v, 16));
        break;
    case PATH_FLOAT:
        if (d->fmt == AF_SFMT_S16)
            EXTRACT_LOOP(int16_t, float, (float)v / 32768.0f);
        else if (d->fmt == AF_SFMT_S32)
            EXTRACT_LOOP(int32_t, float, (float)v * (1.0f / 2147483648.0f));
        else
            EXTRACT_LOOP(float, float, v);
        break;
    case PATH_DOUBLE:
        EXTRACT_LOOP(double, double, v);
        break;
    }
}

/* Mix planar channels (n samples each) to mono float32 = s16 / 32768. */
static void mix(path_t path, void *const *chan, int ch, const double *w, size_t n, float *out) {
    if (path == PATH_INT16 || path == PATH_S32_SHIFT) {
        /* Q15 weights, rounded shift, wraps instead of saturating */
        int64_t q[AF_MAX_CHANNELS];
        for (int c = 0; c < ch; c++)
            q[c] = (int64_t)nearbyint(w[c] * 32768.0);
        for (size_t i = 0; i < n; i++) {
            int64_t v;
            if (ch == 1) {
                v = ((const int32_t *)chan[0])[i];
            } else {
                int64_t acc = 16384;
                for (int c = 0; c < ch; c++)
                    acc += (int64_t)((const int32_t *)chan[c])[i] * q[c];
                v = floor_shift(acc, 15);
            }
            out[i] = s16_to_float(wrap16(v));
        }
    } else if (path == PATH_DOUBLE) {
        for (size_t i = 0; i < n; i++) {
            double acc = ((const double *)chan[0])[i];
            if (ch > 1) {
                acc = 0.0;
                for (int c = 0; c < ch; c++) {
                    double p = ((const double *)chan[c])[i] * w[c];
                    acc = acc + p;
                }
            }
            out[i] = s16_to_float(sat16d(acc * 32768.0));
        }
    } else {
        float wf[AF_MAX_CHANNELS];
        for (int c = 0; c < ch; c++)
            wf[c] = (float)w[c];
        for (size_t i = 0; i < n; i++) {
            float acc = ((const float *)chan[0])[i];
            if (ch > 1) {
                acc = 0.0f;
                for (int c = 0; c < ch; c++) {
                    float p = ((const float *)chan[c])[i] * wf[c];
                    acc = acc + p; /* separate statements: no FMA contraction */
                }
            }
            out[i] = s16_to_float(sat16f(acc * 32768.0f));
        }
    }
}

/* Mono without resampling: extract() and mix() fused into one pass straight into the float
 * output, element for element the same expressions (mix() of one channel is the identity
 * before the conversion). */
#define MONO_LOOP(TYPE_IN, EXPR)                                                                   \
    do {                                                                                           \
        const TYPE_IN *src_ = (const TYPE_IN *)d->data;                                            \
        for (size_t i = 0; i < n; i++) {                                                           \
            TYPE_IN v = src_[i];                                                                   \
            out[i] = (EXPR);                                                                       \
        }                                                                                          \
    } while (0)

static void mono_direct(const af_decoded *d, path_t path, float *out) {
    const size_t n = d->n_frames;
    switch (path) {
    case PATH_INT16:
        if (d->fmt == AF_SFMT_U8)
            MONO_LOOP(uint8_t, s16_to_float(wrap16(((int32_t)v - 128) * 256)));
        else
            MONO_LOOP(int16_t, s16_to_float(wrap16(v)));
        break;
    case PATH_S32_SHIFT:
        MONO_LOOP(int32_t, s16_to_float(wrap16(floor_shift(v, 16))));
        break;
    case PATH_FLOAT: /* only f32 reaches here: s16 / s32 mono take the integer paths */
        MONO_LOOP(float, s16_to_float(sat16f(v * 32768.0f)));
        break;
    case PATH_DOUBLE:
        MONO_LOOP(double, s16_to_float(sat16d(v * 32768.0)));
        break;
    }
}

int af_audio_to_mono(const af_decoded *d, int sr_out, float **out, size_t *n_out) {
    const int ch = d->channels;
    const size_t n = d->n_frames;
    const int resample = d->sample_rate != sr_out;
    const path_t path = select_path(d, resample);
    af_resampler *rs = NULL;
    int rc = AF_OK;
    *out = NULL;
    *n_out = 0;
    if (ch == 1 && !resample) {
        float *res = malloc(n > 0 ? n * sizeof(float) : 1);
        if (!res)
            return AF_ENOMEM;
        mono_direct(d, path, res);
        *out = res;
        *n_out = n;
        return AF_OK;
    }
    if (resample && (rc = af_resampler_create(d->sample_rate, sr_out, &rs)) != AF_OK)
        return rc;
    size_t m = resample ? af_resampler_out_len(rs, n) : n;
    const size_t es = elem_size[path];

    void *chan[AF_MAX_CHANNELS] = {0};
    void *tmp = NULL;
    if (resample && !(tmp = malloc(n > 0 ? n * es : 1)))
        rc = AF_ENOMEM;
    for (int c = 0; c < ch && rc == AF_OK; c++) {
        if (!(chan[c] = malloc(m > 0 ? m * es : 1))) {
            rc = AF_ENOMEM;
            break;
        }
        if (!resample) {
            extract(d, path, c, chan[c]);
            continue;
        }
        extract(d, path, c, tmp);
        if (path == PATH_FLOAT)
            af_resample_f32(rs, tmp, n, chan[c]);
        else if (path == PATH_DOUBLE)
            af_resample_f64(rs, tmp, n, chan[c]);
        else
            af_resample_s16(rs, tmp, n, chan[c]);
    }
    if (rc == AF_OK) {
        double w[AF_MAX_CHANNELS];
        af_downmix_weights(d->layout, ch, w);
        /* The output reuses channel 0's buffer (one large allocation and its page faults
         * less): mix() reads every channel at i before writing out[i], and float elements
         * are never larger than the working type. */
        float *res = chan[0];
        mix(path, chan, ch, w, m, res);
        if (es > sizeof(float) && m > 0) {
            float *shrunk = realloc(res, m * sizeof(float));
            res = shrunk ? shrunk : res;
        }
        chan[0] = NULL;
        *out = res;
        *n_out = m;
    }
    for (int c = 0; c < ch; c++)
        free(chan[c]);
    free(tmp);
    af_resampler_destroy(rs);
    return rc;
}
