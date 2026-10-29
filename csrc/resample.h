/* Internal: ffmpeg-compatible polyphase resampler (docs/load.md, "Resampling").
 * Written from the behavior spec, not from ffmpeg's code. One-shot (whole signal). */
#ifndef ASRFRONT_RESAMPLE_H
#define ASRFRONT_RESAMPLE_H

#include <stddef.h>
#include <stdint.h>

#define AF_RESAMPLE_MIN_RATE 1000
#define AF_RESAMPLE_MAX_RATE 384000

typedef struct af_resampler af_resampler;

/* Filter banks a resampler holds, one per arithmetic path: af_resample_f32 / _f32_range need
 * AF_BANK_F32, af_resample_f64 and af_resampler_bank AF_BANK_F64, af_resample_s16
 * AF_BANK_S16. */
enum { AF_BANK_F32 = 1, AF_BANK_F64 = 2, AF_BANK_S16 = 4, AF_BANK_ALL = 7 };

/* With every bank. AF_EUNSUPPORTED for rates outside [AF_RESAMPLE_MIN_RATE,
 * AF_RESAMPLE_MAX_RATE], and for filters longer than the resampler supports (output rates
 * below 16 kHz from the highest input rates). */
int af_resampler_create(int sr_in, int sr_out, af_resampler **out);
/* The same with only the banks in `banks` (a nonzero mask of AF_BANK_*). */
int af_resampler_create_banks(int sr_in, int sr_out, int banks, af_resampler **out);
void af_resampler_destroy(af_resampler *r);

/* af_resampler_create_banks, shared through a small process-wide cache (a resampler is
 * read-only once created, so one serves any number of threads): repeated loads at a rate skip
 * building the filter bank. Thread-safe. Release with af_resampler_release(), never destroy. */
int af_resampler_acquire(int sr_in, int sr_out, int banks, af_resampler **out);
void af_resampler_release(af_resampler *r);

int af_resampler_taps(const af_resampler *r);   /* N */
int af_resampler_phases(const af_resampler *r); /* filters in the bank, excluding the extra one */
int af_resampler_exact(const af_resampler *r);  /* 1: exact rational phases */
/* Filter bank row m (0 <= m <= phases), N float64 taps (testing; needs AF_BANK_F64). */
const double *af_resampler_bank(const af_resampler *r, int m);

/* Number of output samples for n input samples. */
size_t af_resampler_out_len(const af_resampler *r, size_t n);
/* The same without a resampler (rates > 0). */
size_t af_resample_len(int sr_in, int sr_out, size_t n);
/* AF_OK if af_resampler_create would accept the rates (no allocation), else AF_EUNSUPPORTED. */
int af_resample_check(int sr_in, int sr_out);
/* The largest n whose af_resampler_out_len() from sr_in to sr_out is at most max_out
 * (SIZE_MAX if it does not fit). Needs no resampler. */
size_t af_resample_max_in(int sr_in, int sr_out, size_t max_out);

/* y must hold af_resampler_out_len(r, n) samples. */
void af_resample_f32(const af_resampler *r, const float *x, size_t n, float *y);
void af_resample_f64(const af_resampler *r, const double *x, size_t n, double *y);
/* Outputs [k0, k1) of af_resample_f32() over a signal of n samples, of which x holds samples
 * [x0, x0 + len): [x0, x0 + len) must contain the span af_resampler_span() gives. y receives
 * k1 - k0 samples, bit for bit those of af_resample_f32(). */
void af_resample_f32_range(const af_resampler *r, const float *x, size_t x0, size_t n, size_t k0,
                           size_t k1, float *y);
/* The same for af_resample_f64() and af_resample_s16(). */
void af_resample_f64_range(const af_resampler *r, const double *x, size_t x0, size_t n, size_t k0,
                           size_t k1, double *y);
void af_resample_s16_range(const af_resampler *r, const int32_t *x, size_t x0, size_t n, size_t k0,
                           size_t k1, int32_t *y);
/* Input samples [*lo, *hi) that outputs [k0, k1) read (mirrored edges included), by any of the
 * _range functions. */
void af_resampler_span(const af_resampler *r, size_t n, size_t k0, size_t k1, size_t *lo,
                       size_t *hi);
/* int16 path: s16-range integers in, saturated s16-range integers out. */
void af_resample_s16(const af_resampler *r, const int32_t *x, size_t n, int32_t *y);

#endif
