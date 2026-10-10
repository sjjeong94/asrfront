#include "fft.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "asrfront.h"

#define TWO_PI 6.283185307179586476925286766559

/*
 * Stockham stage (Govindaraju et al., "High Performance Discrete Fourier Transforms
 * on Graphics Processors"). With ns = product of the radices of earlier stages,
 * stride = m / radix and t = b * ns + j (j < ns):
 *
 *   v[k]  = x[t + k * stride] * exp(-2*pi*i * j * k / (ns * radix))
 *   v     = DFT_radix(v)
 *   y[b * ns * radix + j + k * ns] = v[k]
 *
 * which leaves the output in natural order after the last stage.
 */
static inline af_cpx cmul(af_cpx a, af_cpx b) {
    af_cpx r = {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
    return r;
}

static af_cpx expi(long num, long den) {
    /* exp(-2*pi*i * num / den) in double, num reduced to keep the argument small. */
    double a = -TWO_PI * (double)(num % den) / (double)den;
    af_cpx r = {(float)cos(a), (float)sin(a)};
    return r;
}

/* ---------------------------------------------------------------------------------- */
/* Butterflies (forward transform, exp(-i...))                                        */
/* ---------------------------------------------------------------------------------- */

static void pass2(int ns, int nb, int stride, const af_cpx *tw, const af_cpx *x, af_cpx *y) {
    for (int b = 0; b < nb; b++) {
        for (int j = 0; j < ns; j++) {
            const int t = b * ns + j;
            const af_cpx *w = tw + j;
            af_cpx a0 = x[t];
            af_cpx a1 = cmul(x[t + stride], w[0]);
            af_cpx *o = y + b * ns * 2 + j;
            o[0].re = a0.re + a1.re;
            o[0].im = a0.im + a1.im;
            o[ns].re = a0.re - a1.re;
            o[ns].im = a0.im - a1.im;
        }
    }
}

static void pass3(int ns, int nb, int stride, const af_cpx *tw, const af_cpx *x, af_cpx *y) {
    const float c = -0.5f;                                   /* cos(2*pi/3) */
    const float s = 0.866025403784438646763723170752936183f; /* sin(2*pi/3) */
    for (int b = 0; b < nb; b++) {
        for (int j = 0; j < ns; j++) {
            const int t = b * ns + j;
            const af_cpx *w = tw + 2 * j;
            af_cpx a0 = x[t];
            af_cpx a1 = cmul(x[t + stride], w[0]);
            af_cpx a2 = cmul(x[t + 2 * stride], w[1]);
            af_cpx t1 = {a1.re + a2.re, a1.im + a2.im};
            af_cpx t2 = {a1.re - a2.re, a1.im - a2.im};
            af_cpx m1 = {a0.re + c * t1.re, a0.im + c * t1.im};
            af_cpx m2 = {s * t2.im, -s * t2.re}; /* -i * s * t2 */
            af_cpx *o = y + b * ns * 3 + j;
            o[0].re = a0.re + t1.re;
            o[0].im = a0.im + t1.im;
            o[ns].re = m1.re + m2.re;
            o[ns].im = m1.im + m2.im;
            o[2 * ns].re = m1.re - m2.re;
            o[2 * ns].im = m1.im - m2.im;
        }
    }
}

static void pass4(int ns, int nb, int stride, const af_cpx *tw, const af_cpx *x, af_cpx *y) {
    for (int b = 0; b < nb; b++) {
        for (int j = 0; j < ns; j++) {
            const int t = b * ns + j;
            const af_cpx *w = tw + 3 * j;
            af_cpx a0 = x[t];
            af_cpx a1 = cmul(x[t + stride], w[0]);
            af_cpx a2 = cmul(x[t + 2 * stride], w[1]);
            af_cpx a3 = cmul(x[t + 3 * stride], w[2]);
            af_cpx t0 = {a0.re + a2.re, a0.im + a2.im};
            af_cpx t1 = {a0.re - a2.re, a0.im - a2.im};
            af_cpx t2 = {a1.re + a3.re, a1.im + a3.im};
            af_cpx t3 = {a1.re - a3.re, a1.im - a3.im};
            af_cpx *o = y + b * ns * 4 + j;
            o[0].re = t0.re + t2.re;
            o[0].im = t0.im + t2.im;
            o[ns].re = t1.re + t3.im; /* t1 - i * t3 */
            o[ns].im = t1.im - t3.re;
            o[2 * ns].re = t0.re - t2.re;
            o[2 * ns].im = t0.im - t2.im;
            o[3 * ns].re = t1.re - t3.im; /* t1 + i * t3 */
            o[3 * ns].im = t1.im + t3.re;
        }
    }
}

static void pass5(int ns, int nb, int stride, const af_cpx *tw, const af_cpx *x, af_cpx *y) {
    const float c1 = 0.309016994374947424102293417182819059f;  /* cos(2*pi/5) */
    const float c2 = -0.809016994374947424102293417182819059f; /* cos(4*pi/5) */
    const float s1 = 0.951056516295153572116439333379382143f;  /* sin(2*pi/5) */
    const float s2 = 0.587785252292473129168705954639072769f;  /* sin(4*pi/5) */
    for (int b = 0; b < nb; b++) {
        for (int j = 0; j < ns; j++) {
            const int t = b * ns + j;
            const af_cpx *w = tw + 4 * j;
            af_cpx a0 = x[t];
            af_cpx a1 = cmul(x[t + stride], w[0]);
            af_cpx a2 = cmul(x[t + 2 * stride], w[1]);
            af_cpx a3 = cmul(x[t + 3 * stride], w[2]);
            af_cpx a4 = cmul(x[t + 4 * stride], w[3]);
            af_cpx t1 = {a1.re + a4.re, a1.im + a4.im};
            af_cpx t2 = {a2.re + a3.re, a2.im + a3.im};
            af_cpx t3 = {a1.re - a4.re, a1.im - a4.im};
            af_cpx t4 = {a2.re - a3.re, a2.im - a3.im};
            af_cpx m1 = {a0.re + c1 * t1.re + c2 * t2.re, a0.im + c1 * t1.im + c2 * t2.im};
            af_cpx m2 = {a0.re + c2 * t1.re + c1 * t2.re, a0.im + c2 * t1.im + c1 * t2.im};
            /* n1 = -i * (s1 * t3 + s2 * t4), n2 = -i * (s2 * t3 - s1 * t4) */
            af_cpx n1 = {s1 * t3.im + s2 * t4.im, -(s1 * t3.re + s2 * t4.re)};
            af_cpx n2 = {s2 * t3.im - s1 * t4.im, -(s2 * t3.re - s1 * t4.re)};
            af_cpx *o = y + b * ns * 5 + j;
            o[0].re = a0.re + t1.re + t2.re;
            o[0].im = a0.im + t1.im + t2.im;
            o[ns].re = m1.re + n1.re;
            o[ns].im = m1.im + n1.im;
            o[2 * ns].re = m2.re + n2.re;
            o[2 * ns].im = m2.im + n2.im;
            o[3 * ns].re = m2.re - n2.re;
            o[3 * ns].im = m2.im - n2.im;
            o[4 * ns].re = m1.re - n1.re;
            o[4 * ns].im = m1.im - n1.im;
        }
    }
}

/* ---------------------------------------------------------------------------------- */
/* Plan                                                                               */
/* ---------------------------------------------------------------------------------- */

static int factorize(int m, int *radices) {
    int count = 0;
    static const int order[] = {4, 2, 3, 5};
    for (int i = 0; i < 4; i++) {
        while (m % order[i] == 0) {
            if (count == AF_FFT_MAX_STAGES)
                return -1;
            radices[count++] = order[i];
            m /= order[i];
        }
    }
    return m == 1 ? count : -1;
}

int af_rfft_create(int n, af_rfft **out) {
    int radices[AF_FFT_MAX_STAGES];
    if (!out)
        return AF_EINVAL;
    *out = NULL;
    if (n < 2 || n % 2 != 0)
        return AF_EINVAL;
    const int m = n / 2;
    const int n_stages = factorize(m, radices);
    if (n_stages < 0)
        return AF_EINVAL;

    size_t n_tw = 0;
    for (int s = 0, ns = 1; s < n_stages; ns *= radices[s], s++)
        n_tw += (size_t)ns * (size_t)(radices[s] - 1);

    /* One allocation: stage twiddles | post twiddles | buf0 | buf1 | spec */
    size_t n_cpx = n_tw + (size_t)(m + 1) + 2 * (size_t)m + (size_t)(m + 1);
    af_rfft *p = calloc(1, sizeof(*p));
    af_cpx *mem = malloc(n_cpx * sizeof(af_cpx));
    if (!p || !mem) {
        free(p);
        free(mem);
        return AF_ENOMEM;
    }
    p->n = n;
    p->m = m;
    p->n_stages = n_stages;
    p->mem = mem;

    af_cpx *tw = mem;
    for (int s = 0, ns = 1; s < n_stages; ns *= radices[s], s++) {
        const int r = radices[s];
        p->stages[s].radix = r;
        p->stages[s].ns = ns;
        p->stages[s].tw = tw;
        for (int j = 0; j < ns; j++)
            for (int k = 1; k < r; k++)
                *tw++ = expi((long)j * k, (long)ns * r);
    }
    p->post_tw = tw;
    for (int k = 0; k <= m; k++)
        p->post_tw[k] = expi(k, n);
    p->buf[0] = p->post_tw + (m + 1);
    p->buf[1] = p->buf[0] + m;
    p->spec = p->buf[1] + m;

    *out = p;
    return AF_OK;
}

void af_rfft_destroy(af_rfft *p) {
    if (!p)
        return;
    free(p->mem);
    free(p);
}

int af_rfft_size(const af_rfft *p) {
    return p->n;
}

/* Complex FFT of buf[0] (size m); returns the buffer holding the result. */
static const af_cpx *cfft(af_rfft *p) {
    af_cpx *x = p->buf[0], *y = p->buf[1];
    for (int s = 0; s < p->n_stages; s++) {
        const af_fft_stage *st = &p->stages[s];
        const int stride = p->m / st->radix;
        const int nb = stride / st->ns;
        switch (st->radix) {
        case 2:
            pass2(st->ns, nb, stride, st->tw, x, y);
            break;
        case 3:
            pass3(st->ns, nb, stride, st->tw, x, y);
            break;
        case 4:
            pass4(st->ns, nb, stride, st->tw, x, y);
            break;
        default:
            pass5(st->ns, nb, stride, st->tw, x, y);
            break;
        }
        af_cpx *tmp = x;
        x = y;
        y = tmp;
    }
    return x;
}

void af_rfft_forward(af_rfft *p, const float *in, af_cpx *out) {
    const int m = p->m;
    /* Pack z[i] = x[2i] + i * x[2i + 1]. */
    memcpy(p->buf[0], in, (size_t)p->n * sizeof(float));
    const af_cpx *z = cfft(p);

    /* X[k] = E[k] + W^k O[k], E = (Z[k] + conj Z[m-k]) / 2, O = -i (Z[k] - conj Z[m-k]) / 2 */
    out[0].re = z[0].re + z[0].im;
    out[0].im = 0.0f;
    out[m].re = z[0].re - z[0].im;
    out[m].im = 0.0f;
    for (int k = 1; k < m; k++) {
        const af_cpx zk = z[k], zc = z[m - k];
        const float er = 0.5f * (zk.re + zc.re), ei = 0.5f * (zk.im - zc.im);
        const float or_ = 0.5f * (zk.im + zc.im), oi = -0.5f * (zk.re - zc.re);
        const af_cpx w = p->post_tw[k];
        out[k].re = er + (w.re * or_ - w.im * oi);
        out[k].im = ei + (w.re * oi + w.im * or_);
    }
}

void af_rfft_power(af_rfft *p, const float *in, float *out) {
    af_rfft_forward(p, in, p->spec);
    for (int k = 0; k <= p->m; k++)
        out[k] = p->spec[k].re * p->spec[k].re + p->spec[k].im * p->spec[k].im;
}
