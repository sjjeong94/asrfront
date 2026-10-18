/* ffmpeg-compatible polyphase resampler, written from docs/load.md ("Resampling"):
 * Kaiser-windowed sinc, phase-0 normalization, exact rational or 1024 interpolated phases,
 * reflected edges, and the float path's 8-accumulator FMA dot product. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

#include "asrfront.h"
#include "kernels.h"
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
    double *bank;           /* (phases + 1) x n_taps */
    float *bankf;           /* (phases + 1) x width, zero padded */
    int32_t *bankq;         /* (phases + 1) x n_taps, Q15 */
    const af_kernels *kern; /* float path kernels */
    /* af_resampler_acquire(): users, and whether the cache holds it (both under cache_lock) */
    unsigned refs;
    int cached;
    unsigned long long used; /* LRU stamp */
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
    r->kern = af_kernels_select(); /* once: the selection runs cpuid (slow in VMs) */
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

/* ---------------------------------------------------------------------------------- */
/* Shared resamplers (af_resampler_acquire)                                           */
/* ---------------------------------------------------------------------------------- */

/* Creating the bank costs ~0.5 ms at 44.1 kHz (160 phases), more than resampling a 2 s
 * clip, so loads keep a few resamplers. Bounded: CACHE_SLOTS entries of at most
 * CACHE_MAX_BYTES of filter bank each; larger banks (interpolated mode at high rates,
 * up to ~13 MB) are created per call. */
#define CACHE_SLOTS 4
#define CACHE_MAX_BYTES ((size_t)1 << 20)

static af_resampler *cache[CACHE_SLOTS];
static unsigned long long cache_clock;

#ifdef _WIN32
static SRWLOCK cache_lock = SRWLOCK_INIT;
static void cache_init(void) {}
static void lock(void) {
    AcquireSRWLockExclusive(&cache_lock);
}
static void unlock(void) {
    ReleaseSRWLockExclusive(&cache_lock);
}
#else
static pthread_mutex_t cache_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t cache_once = PTHREAD_ONCE_INIT;
static void lock(void) {
    pthread_mutex_lock(&cache_lock);
}
static void unlock(void) {
    pthread_mutex_unlock(&cache_lock);
}
/* Held across fork(), so that a child never inherits it locked by a thread it does not have.
 * The forking thread owns it on both sides and unlocks it in each. */
static void register_fork_handlers(void) {
    pthread_atfork(lock, unlock, unlock);
}
static void cache_init(void) {
    pthread_once(&cache_once, register_fork_handlers);
}
#endif

static size_t bank_bytes(const af_resampler *r) {
    return ((size_t)r->phases + 1) * ((size_t)r->n_taps * (sizeof(double) + sizeof(int32_t)) +
                                      (size_t)r->width * sizeof(float));
}

/* Under cache_lock: the cached resampler for these rates, marked as used once more. */
static af_resampler *cache_take(int sr_in, int sr_out) {
    for (int i = 0; i < CACHE_SLOTS; i++) {
        af_resampler *r = cache[i];
        if (r && r->sr_in == sr_in && r->sr_out == sr_out) {
            r->refs++;
            r->used = ++cache_clock;
            return r;
        }
    }
    return NULL;
}

int af_resampler_acquire(int sr_in, int sr_out, af_resampler **out) {
    if (!out)
        return AF_EINVAL;
    *out = NULL;
    cache_init();
    lock();
    af_resampler *r = cache_take(sr_in, sr_out);
    unlock();
    if (r) {
        *out = r;
        return AF_OK;
    }
    int rc = af_resampler_create(sr_in, sr_out, &r); /* outside the lock: slow */
    if (rc != AF_OK)
        return rc;
    r->refs = 1;
    af_resampler *evicted = NULL;
    if (bank_bytes(r) <= CACHE_MAX_BYTES) {
        lock();
        af_resampler *other = cache_take(sr_in, sr_out); /* created meanwhile by another thread */
        if (other) {
            unlock();
            af_resampler_destroy(r);
            *out = other;
            return AF_OK;
        }
        int slot = -1; /* an empty slot, else the least recently used idle entry */
        for (int i = 0; i < CACHE_SLOTS && (slot < 0 || cache[slot]); i++)
            if (!cache[i] ||
                (cache[i]->refs == 0 && (slot < 0 || cache[i]->used < cache[slot]->used)))
                slot = i;
        if (slot >= 0) { /* else every entry is in use: r stays private */
            evicted = cache[slot];
            cache[slot] = r;
            r->cached = 1;
            r->used = ++cache_clock;
        }
        unlock();
    }
    af_resampler_destroy(evicted); /* idle, and no longer reachable through the cache */
    *out = r;
    return AF_OK;
}

void af_resampler_release(af_resampler *r) {
    if (!r)
        return;
    lock();
    const int last = --r->refs == 0 && !r->cached;
    unlock();
    if (last)
        af_resampler_destroy(r);
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

/* Position of output k: base sample b, filter row m and interpolation numerator fr / q.
 * Stepped incrementally (t_k = k * p / q) to avoid two 64-bit divisions per output. */
typedef struct {
    ptrdiff_t b;
    int m;
    uint64_t fr;
    uint64_t rem; /* k * p mod q */
    uint64_t step_int, step_rem;
} pos_t;

static void pos_init(const af_resampler *r, pos_t *s) {
    memset(s, 0, sizeof(*s));
    s->step_int = r->p / r->q;
    s->step_rem = r->p % r->q;
}

static void pos_phase(const af_resampler *r, pos_t *s) {
    if (r->exact) {
        s->m = (int)s->rem;
    } else {
        uint64_t fp = s->rem * MAX_EXACT_PHASES;
        s->m = (int)(fp / r->q);
        s->fr = fp - (uint64_t)s->m * r->q;
    }
}

static void pos_next(const af_resampler *r, pos_t *s) {
    s->b += (ptrdiff_t)s->step_int;
    s->rem += s->step_rem;
    if (s->rem >= r->q) {
        s->rem -= r->q;
        s->b++;
    }
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

/* First input sample read by output k (before mirroring): b_k - N/2 + 1. */
static ptrdiff_t first_read(const af_resampler *r, size_t k) {
    return (ptrdiff_t)((uint64_t)k * r->p / r->q) - r->n_taps / 2 + 1;
}

void af_resampler_span(const af_resampler *r, size_t n, size_t k0, size_t k1, size_t *lo,
                       size_t *hi) {
    const ptrdiff_t nn = (ptrdiff_t)n, wd = r->width;
    if (k1 <= k0 || nn < 2 * wd + 2) { /* nothing, or short input: mirrors may bounce */
        *lo = 0;
        *hi = n;
        return;
    }
    ptrdiff_t a = first_read(r, k0), b = first_read(r, k1 - 1) + wd;
    if (a < 0) { /* x[-j] = x[j], j <= N/2 - 1 */
        a = 0;
        b = b > wd ? b : wd;
    }
    if (b > nn) { /* x[n-1+j] = x[n-1-j], j <= width */
        b = nn;
        a = a < nn - 1 - wd ? a : nn - 1 - wd;
    }
    *lo = (size_t)a;
    *hi = (size_t)(b > a ? b : a);
}

void af_resample_f32_range(const af_resampler *r, const float *x, size_t x0, size_t n, size_t k0,
                           size_t k1, float *y) {
    const int nt = r->n_taps, wd = r->width;
    const af_kernels *kern = r->kern;
    /* N <= ceil(32 * sr / (0.97 * 16000)), rounded up to a multiple of 8 */
    float edge_[AF_RESAMPLE_MAX_RATE / 485 + 16] = {0};
    /* The kernel reads `width` samples from first = b - N/2 + 1 (the taps past N are zero).
     * first grows with k, so the interior outputs (0 <= first, first + width <= n) are one
     * range [k_lo, k_hi): b_k = floor(k p / q) >= N/2 - 1 and b_k <= n + N/2 - 1 - width. */
    size_t k_lo = k1, k_hi = 0;
    const ptrdiff_t b_min = nt / 2 - 1, b_max = (ptrdiff_t)n + nt / 2 - 1 - wd;
    if (r->exact && b_max >= b_min) {
        k_lo = (size_t)(((uint64_t)b_min * r->q + r->p - 1) / r->p);
        k_hi = (size_t)((((uint64_t)b_max + 1) * r->q - 1) / r->p + 1);
        k_lo = k_lo > k0 ? k_lo : k0;
        k_hi = k_hi < k1 ? k_hi : k1;
    }
    pos_t s;
    pos_init(r, &s);
    const uint64_t t0 = (uint64_t)k0 * r->p;
    s.b = (ptrdiff_t)(t0 / r->q);
    s.rem = t0 % r->q;
    for (size_t k = k0; k < k1; k++, pos_next(r, &s)) {
        if (k == k_lo && k_lo < k_hi) {
            kern->resample_run(r->bankf, wd, x + (s.b - nt / 2 + 1 - (ptrdiff_t)x0), (int)s.rem,
                               (int)s.step_int, (int)s.step_rem, (int)r->q, k_hi - k_lo,
                               y + (k_lo - k0));
            /* jump to output k_hi - 1; the loop's pos_next() moves to k_hi */
            k = k_hi - 1;
            const uint64_t t = (uint64_t)k * r->p;
            s.b = (ptrdiff_t)(t / r->q);
            s.rem = t % r->q;
            continue;
        }
        pos_phase(r, &s);
        const float *win;
        ptrdiff_t first = s.b - nt / 2 + 1;
        if (first >= 0 && first + wd <= (ptrdiff_t)n) {
            win = x + (first - (ptrdiff_t)x0);
        } else {
            for (int i = 0; i < nt; i++)
                edge_[i] = x[mirror(first + i, n) - x0];
            win = edge_; /* edge_[nt .. wd) stay zero */
        }
        const float *h1 = r->bankf + (size_t)s.m * wd;
        float v1 = kern->resample_dot(h1, win, wd);
        if (r->exact) {
            y[k - k0] = v1;
        } else {
            float v2 = kern->resample_dot(h1 + wd, win, wd);
            float a = (float)((double)s.fr / (double)r->q);
            float d = v2 - v1;
            float t = d * a;
            y[k - k0] = v1 + t;
        }
    }
}

void af_resample_f32(const af_resampler *r, const float *x, size_t n, float *y) {
    af_resample_f32_range(r, x, 0, n, 0, af_resampler_out_len(r, n), y);
}

void af_resample_f64(const af_resampler *r, const double *x, size_t n, double *y) {
    size_t n_out = af_resampler_out_len(r, n);
    const int nt = r->n_taps;
    double edge_[AF_RESAMPLE_MAX_RATE / 485 + 2];
    pos_t s;
    pos_init(r, &s);
    for (size_t k = 0; k < n_out; k++, pos_next(r, &s)) {
        pos_phase(r, &s);
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
    pos_t s;
    pos_init(r, &s);
    for (size_t k = 0; k < n_out; k++, pos_next(r, &s)) {
        pos_phase(r, &s);
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
