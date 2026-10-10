/* ffmpeg-compatible polyphase resampler, written from docs/load.md ("Resampling"):
 * Kaiser-windowed sinc, phase-0 normalization, exact rational or 1024 interpolated phases,
 * reflected edges, and the float path's 8-accumulator FMA dot product. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "asrfront.h"
#include "resample.h"

#define FILTER_SIZE 32
#define CUTOFF 0.97
#define KAISER_BETA 9.0
#define MAX_EXACT_PHASES 1024
#define LANES 8

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct af_resampler {
    int sr_in, sr_out;
    uint64_t p, q; /* sr_in / sr_out = p / q, lowest terms */
    int n_taps;    /* N */
    int width;     /* N rounded up to a multiple of LANES (float bank row stride) */
    int phases;
    int exact;
    double *bank;   /* (phases + 1) x n_taps */
    float *bankf;   /* (phases + 1) x width, zero padded */
    int32_t *bankq; /* (phases + 1) x n_taps, Q15 */
};

static uint64_t gcd_u64(uint64_t a, uint64_t b) {
    while (b) {
        uint64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

/* Modified Bessel function of the first kind, order 0 (power series; x <= ~10 here). */
static double bessel_i0(double x) {
    double q = x * x / 4.0, term = 1.0, sum = 1.0;
    for (int k = 1; k < 64; k++) {
        term *= q / ((double)k * (double)k);
        sum += term;
        if (term < sum * 1e-18)
            break;
    }
    return sum;
}

/* Unnormalized taps for x[b + r], r = -N/2+1 .. N/2, at output position b + frac. */
static void kaiser_sinc(double frac, int n_taps, double fc, double *h) {
    double half = n_taps / 2.0;
    for (int i = 0; i < n_taps; i++) {
        double r = (double)(i - n_taps / 2 + 1);
        double d = frac - r;
        double y = fc * d;
        double s = 1.0;
        if (y != 0.0) {
            double x = M_PI * y;
            s = sin(x) / x;
        }
        double w = d / half;
        double win = 0.0;
        if (fabs(w) <= 1.0) {
            double t = 1.0 - w * w;
            win = bessel_i0(KAISER_BETA * sqrt(t > 0.0 ? t : 0.0));
        }
        h[i] = s * win;
    }
}

int af_resampler_create(int sr_in, int sr_out, af_resampler **out) {
    if (!out)
        return AF_EINVAL;
    *out = NULL;
    if (sr_in < AF_RESAMPLE_MIN_RATE || sr_in > AF_RESAMPLE_MAX_RATE ||
        sr_out < AF_RESAMPLE_MIN_RATE || sr_out > AF_RESAMPLE_MAX_RATE)
        return AF_EUNSUPPORTED;
    af_resampler *r = calloc(1, sizeof(*r));
    if (!r)
        return AF_ENOMEM;
    r->sr_in = sr_in;
    r->sr_out = sr_out;
    uint64_t g = gcd_u64((uint64_t)sr_in, (uint64_t)sr_out);
    r->p = (uint64_t)sr_in / g;
    r->q = (uint64_t)sr_out / g;

    double fc = CUTOFF * sr_out / sr_in;
    if (fc > 1.0)
        fc = 1.0;
    int n_taps = (int)ceil(FILTER_SIZE / fc);
    n_taps += n_taps % 2;
    r->n_taps = n_taps;
    r->width = (n_taps + LANES - 1) / LANES * LANES;
    r->exact = r->q <= MAX_EXACT_PHASES;
    r->phases = r->exact ? (int)r->q : MAX_EXACT_PHASES;

    size_t rows = (size_t)r->phases + 1;
    r->bank = malloc(rows * (size_t)n_taps * sizeof(double));
    r->bankf = calloc(rows * (size_t)r->width, sizeof(float));
    r->bankq = malloc(rows * (size_t)n_taps * sizeof(int32_t));
    if (!r->bank || !r->bankf || !r->bankq) {
        af_resampler_destroy(r);
        return AF_ENOMEM;
    }
    for (int m = 0; m < r->phases; m++)
        kaiser_sinc((double)m / r->phases, n_taps, fc, r->bank + (size_t)m * n_taps);
    double norm = 0.0;
    for (int i = 0; i < n_taps; i++)
        norm += r->bank[i];
    for (size_t i = 0; i < (size_t)r->phases * n_taps; i++)
        r->bank[i] /= norm;
    /* extra row: phase 0 shifted right by one input sample */
    double *last = r->bank + (size_t)r->phases * n_taps;
    for (int i = n_taps - 1; i > 0; i--)
        last[i] = r->bank[i - 1];
    last[0] = 0.0;

    for (size_t m = 0; m < rows; m++) {
        for (int i = 0; i < n_taps; i++) {
            double h = r->bank[m * n_taps + i];
            r->bankf[m * r->width + i] = (float)h;
            double v = nearbyint(h * 32768.0); /* stored as int16: 1.0 -> 32767 */
            r->bankq[m * n_taps + i] = (int32_t)(v > 32767.0    ? 32767.0
                                                 : v < -32768.0 ? -32768.0
                                                                : v);
        }
    }
    *out = r;
    return AF_OK;
}

void af_resampler_destroy(af_resampler *r) {
    if (!r)
        return;
    free(r->bank);
    free(r->bankf);
    free(r->bankq);
    free(r);
}

int af_resampler_taps(const af_resampler *r) {
    return r->n_taps;
}
int af_resampler_phases(const af_resampler *r) {
    return r->phases;
}
int af_resampler_exact(const af_resampler *r) {
    return r->exact;
}
const double *af_resampler_bank(const af_resampler *r, int m) {
    return r->bank + (size_t)m * r->n_taps;
}

size_t af_resampler_out_len(const af_resampler *r, size_t n) {
    if (r->sr_in == r->sr_out)
        return n;
    if (n == 0)
        return 0;
    /* docs/load.md "Output length": ffmpeg's exact count is not reproducible */
    uint64_t m = r->sr_in > 2 * r->sr_out ? (uint64_t)n - 1 : (uint64_t)n;
    return (size_t)((m * r->q + r->p - 1) / r->p);
}

/* Position of output k: base sample b, filter row m and interpolation numerator fr / q. */
typedef struct {
    ptrdiff_t b;
    int m;
    uint64_t fr;
} pos_t;

static pos_t position(const af_resampler *r, size_t k) {
    pos_t s;
    uint64_t num = (uint64_t)k * r->p;
    uint64_t rem = num % r->q;
    s.b = (ptrdiff_t)(num / r->q);
    if (r->exact) {
        s.m = (int)rem;
        s.fr = 0;
    } else {
        uint64_t fp = rem * MAX_EXACT_PHASES;
        s.m = (int)(fp / r->q);
        s.fr = fp - (uint64_t)s.m * r->q;
    }
    return s;
}

/* Input index with mirrored edges: x[-j] = x[j], x[n-1+j] = x[n-j]. */
static size_t mirror(ptrdiff_t i, size_t n) {
    ptrdiff_t nn = (ptrdiff_t)n;
    for (int it = 0; it < 4 && (i < 0 || i > nn - 1); it++) {
        if (i < 0)
            i = -i;
        if (i > nn - 1)
            i = 2 * nn - 1 - i;
    }
    return (size_t)(i < 0 ? 0 : i > nn - 1 ? nn - 1 : i);
}

/* Gather the N inputs of output position b into win (mirrored at the edges). */
#define GATHER(win, x, n, b, n_taps)                                                               \
    do {                                                                                           \
        ptrdiff_t first_ = (b) - (n_taps) / 2 + 1;                                                 \
        if (first_ >= 0 && first_ + (n_taps) <= (ptrdiff_t)(n)) {                                  \
            (win) = (x) + first_;                                                                  \
        } else {                                                                                   \
            for (int i_ = 0; i_ < (n_taps); i_++)                                                  \
                edge_[i_] = (x)[mirror(first_ + i_, (n))];                                         \
            (win) = edge_;                                                                         \
        }                                                                                          \
    } while (0)

/* float32 dot product: 8 FMA accumulators, then (a0+a4, a1+a5, a2+a6, a3+a7), pairwise. */
static float dot8_fma(const float *h, const float *x, int n_taps) {
    float acc[LANES] = {0};
    for (int i = 0; i < n_taps; i++)
        acc[i % LANES] = fmaf(h[i], x[i], acc[i % LANES]);
    float s0 = acc[0] + acc[4], s1 = acc[1] + acc[5], s2 = acc[2] + acc[6], s3 = acc[3] + acc[7];
    float t0 = s0 + s2, t1 = s1 + s3;
    return t0 + t1;
}

void af_resample_f32(const af_resampler *r, const float *x, size_t n, float *y) {
    size_t n_out = af_resampler_out_len(r, n);
    const int nt = r->n_taps;
    float edge_[AF_RESAMPLE_MAX_RATE / 485 + 2]; /* N <= ceil(32 * sr / (0.97 * 16000)) */
    for (size_t k = 0; k < n_out; k++) {
        pos_t s = position(r, k);
        const float *win;
        GATHER(win, x, n, s.b, nt);
        float v1 = dot8_fma(r->bankf + (size_t)s.m * r->width, win, nt);
        if (r->exact) {
            y[k] = v1;
        } else {
            float v2 = dot8_fma(r->bankf + (size_t)(s.m + 1) * r->width, win, nt);
            float a = (float)((double)s.fr / (double)r->q);
            float d = v2 - v1;
            float t = d * a;
            y[k] = v1 + t;
        }
    }
}

void af_resample_f64(const af_resampler *r, const double *x, size_t n, double *y) {
    size_t n_out = af_resampler_out_len(r, n);
    const int nt = r->n_taps;
    double edge_[AF_RESAMPLE_MAX_RATE / 485 + 2];
    for (size_t k = 0; k < n_out; k++) {
        pos_t s = position(r, k);
        const double *win;
        GATHER(win, x, n, s.b, nt);
        const double *h1 = r->bank + (size_t)s.m * nt;
        double acc = 0.0;
        if (r->exact) {
            for (int i = 0; i < nt; i++) {
                double p = h1[i] * win[i];
                acc = acc + p;
            }
        } else {
            const double *h2 = h1 + nt;
            double a = (double)s.fr / (double)r->q;
            for (int i = 0; i < nt; i++) {
                double h = h1[i] + (h2[i] - h1[i]) * a;
                double p = h * win[i];
                acc = acc + p;
            }
        }
        y[k] = acc;
    }
}

static int64_t floor_div(int64_t a, int64_t b) { /* b > 0 */
    int64_t q = a / b;
    return (a % b != 0 && a < 0) ? q - 1 : q;
}

void af_resample_s16(const af_resampler *r, const int32_t *x, size_t n, int32_t *y) {
    size_t n_out = af_resampler_out_len(r, n);
    const int nt = r->n_taps;
    int32_t edge_[AF_RESAMPLE_MAX_RATE / 485 + 2];
    for (size_t k = 0; k < n_out; k++) {
        pos_t s = position(r, k);
        const int32_t *win;
        GATHER(win, x, n, s.b, nt);
        const int32_t *h1 = r->bankq + (size_t)s.m * nt;
        int64_t acc = 0;
        for (int i = 0; i < nt; i++)
            acc += (int64_t)h1[i] * win[i];
        if (!r->exact) { /* interpolate the accumulators (<= 1 LSB vs. ffmpeg, docs/load.md) */
            const int32_t *h2 = h1 + nt;
            int64_t acc2 = 0;
            for (int i = 0; i < nt; i++)
                acc2 += (int64_t)h2[i] * win[i];
            acc += floor_div((acc2 - acc) * (int64_t)s.fr, (int64_t)r->q);
        }
        int64_t v = floor_div(acc + 16384, 32768);
        y[k] = (int32_t)(v > 32767 ? 32767 : v < -32768 ? -32768 : v);
    }
}
