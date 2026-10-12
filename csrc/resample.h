/* Internal: ffmpeg-compatible polyphase resampler (docs/load.md, "Resampling").
 * Written from the behavior spec, not from ffmpeg's code. One-shot (whole signal). */
#ifndef ASRFRONT_RESAMPLE_H
#define ASRFRONT_RESAMPLE_H

#include <stddef.h>
#include <stdint.h>

#define AF_RESAMPLE_MIN_RATE 1000
#define AF_RESAMPLE_MAX_RATE 384000

typedef struct af_resampler af_resampler;

/* AF_EUNSUPPORTED for rates outside [AF_RESAMPLE_MIN_RATE, AF_RESAMPLE_MAX_RATE]. */
int af_resampler_create(int sr_in, int sr_out, af_resampler **out);
void af_resampler_destroy(af_resampler *r);

int af_resampler_taps(const af_resampler *r);   /* N */
int af_resampler_phases(const af_resampler *r); /* filters in the bank, excluding the extra one */
int af_resampler_exact(const af_resampler *r);  /* 1: exact rational phases */
/* Filter bank row m (0 <= m <= phases), N float64 taps (testing). */
const double *af_resampler_bank(const af_resampler *r, int m);

/* Number of output samples for n input samples. */
size_t af_resampler_out_len(const af_resampler *r, size_t n);

/* y must hold af_resampler_out_len(r, n) samples. */
void af_resample_f32(const af_resampler *r, const float *x, size_t n, float *y);
void af_resample_f64(const af_resampler *r, const double *x, size_t n, double *y);
/* Outputs [k0, k1) of af_resample_f32() over a signal of n samples, of which x holds samples
 * [x0, x0 + len): [x0, x0 + len) must contain the span af_resampler_span() gives. y receives
 * k1 - k0 samples, bit for bit those of af_resample_f32(). */
void af_resample_f32_range(const af_resampler *r, const float *x, size_t x0, size_t n, size_t k0,
                           size_t k1, float *y);
/* Input samples [*lo, *hi) that outputs [k0, k1) read (mirrored edges included). */
void af_resampler_span(const af_resampler *r, size_t n, size_t k0, size_t k1, size_t *lo,
                       size_t *hi);
/* int16 path: s16-range integers in, saturated s16-range integers out. */
void af_resample_s16(const af_resampler *r, const int32_t *x, size_t n, int32_t *y);

#endif
