/*
 * Compiled once per ISA with -DAF_ISA=<name> (see CMakeLists.txt); defines the table
 * af_kernels_<name>. Everything here must stay free of FMA contraction and
 * reassociation so the variants agree bit for bit.
 *
 * Kernels work on AF_VL lanes (frames) in chunks of the native vector width W: a GCC /
 * Clang vector-extension type of 8 floats with AVX, 4 otherwise (SSE2, NEON). Other
 * compilers (MSVC) use W = 1 (plain floats), which they may auto-vectorize; defining
 * AF_NO_VECTOR_EXT forces that path (the "scalar" table, built everywhere for tests).
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "kernels.h"

#ifndef AF_ISA
#define AF_ISA generic
#endif
#define AF_CAT_(a, b) a##_##b
#define AF_CAT(a, b) AF_CAT_(a, b)
#define AF_STR_(a) #a
#define AF_STR(a) AF_STR_(a)

#define VL AF_VL

/* ---------------------------------------------------------------------------------- */
/* Native vectors                                                                     */
/* ---------------------------------------------------------------------------------- */

#if (defined(__GNUC__) || defined(__clang__)) && !defined(AF_NO_VECTOR_EXT)
#define AF_VEC_EXT 1
#if defined(__AVX__)
#define W 8
#else
#define W 4
#endif
typedef float vn __attribute__((vector_size(W * sizeof(float))));
typedef int32_t vni __attribute__((vector_size(W * sizeof(int32_t))));
#define VZERO ((vn){0})
#define VINLINE static inline __attribute__((always_inline))
#else
#define W 1
typedef float vn;
#define VZERO 0.0f
#define VINLINE static inline
#endif

VINLINE vn vld(const float *p) {
    vn v;
    memcpy(&v, p, sizeof(v));
    return v;
}
VINLINE void vst(float *p, vn v) {
    memcpy(p, &v, sizeof(v));
}

/* (ar + i ai) * w, same operation order as cmul() in fft.c */
#define CMUL_R(ar, ai, w) ((ar) * (w).re - (ai) * (w).im)
#define CMUL_I(ar, ai, w) ((ar) * (w).im + (ai) * (w).re)

/* ---------------------------------------------------------------------------------- */
/* Stockham FFT passes on split SoA buffers (same formulas as fft.c)                  */
/* ---------------------------------------------------------------------------------- */

#if defined(AF_VEC_EXT) && defined(__aarch64__)
#define FFT_INLINE VINLINE
#else
#define FFT_INLINE static
#endif

FFT_INLINE void pass2(int ns, int nb, int stride, const af_cpx *tw, const float *xr,
                      const float *xi, float *yr, float *yi) {
    const size_t st = (size_t)stride * VL, o = (size_t)ns * VL;
    for (int b = 0; b < nb; b++) {
        for (int j = 0; j < ns; j++) {
            const size_t in = (size_t)(b * ns + j) * VL, out = (size_t)(b * ns * 2 + j) * VL;
            const af_cpx w1 = tw[j];
            for (int c = 0; c < VL; c += W) {
                const vn a0r = vld(xr + in + c), a0i = vld(xi + in + c);
                const vn x1r = vld(xr + in + st + c), x1i = vld(xi + in + st + c);
                const vn a1r = CMUL_R(x1r, x1i, w1), a1i = CMUL_I(x1r, x1i, w1);
                vst(yr + out + c, a0r + a1r);
                vst(yi + out + c, a0i + a1i);
                vst(yr + out + o + c, a0r - a1r);
                vst(yi + out + o + c, a0i - a1i);
            }
        }
    }
}

static void pass3(int ns, int nb, int stride, const af_cpx *tw, const float *xr, const float *xi,
                  float *yr, float *yi) {
    const float cc = -0.5f;
    const float s = 0.866025403784438646763723170752936183f;
    const size_t st = (size_t)stride * VL, o = (size_t)ns * VL;
    for (int b = 0; b < nb; b++) {
        for (int j = 0; j < ns; j++) {
            const size_t in = (size_t)(b * ns + j) * VL, out = (size_t)(b * ns * 3 + j) * VL;
            const af_cpx w1 = tw[2 * j], w2 = tw[2 * j + 1];
            for (int c = 0; c < VL; c += W) {
                const vn a0r = vld(xr + in + c), a0i = vld(xi + in + c);
                const vn x1r = vld(xr + in + st + c), x1i = vld(xi + in + st + c);
                const vn x2r = vld(xr + in + 2 * st + c), x2i = vld(xi + in + 2 * st + c);
                const vn a1r = CMUL_R(x1r, x1i, w1), a1i = CMUL_I(x1r, x1i, w1);
                const vn a2r = CMUL_R(x2r, x2i, w2), a2i = CMUL_I(x2r, x2i, w2);
                const vn t1r = a1r + a2r, t1i = a1i + a2i;
                const vn t2r = a1r - a2r, t2i = a1i - a2i;
                const vn m1r = a0r + cc * t1r, m1i = a0i + cc * t1i;
                const vn m2r = s * t2i, m2i = -s * t2r; /* -i * s * t2 */
                vst(yr + out + c, a0r + t1r);
                vst(yi + out + c, a0i + t1i);
                vst(yr + out + o + c, m1r + m2r);
                vst(yi + out + o + c, m1i + m2i);
                vst(yr + out + 2 * o + c, m1r - m2r);
                vst(yi + out + 2 * o + c, m1i - m2i);
            }
        }
    }
}

FFT_INLINE void pass4(int ns, int nb, int stride, const af_cpx *tw, const float *xr,
                      const float *xi, float *yr, float *yi) {
    const size_t st = (size_t)stride * VL, o = (size_t)ns * VL;
    for (int b = 0; b < nb; b++) {
        for (int j = 0; j < ns; j++) {
            const size_t in = (size_t)(b * ns + j) * VL, out = (size_t)(b * ns * 4 + j) * VL;
            const af_cpx w1 = tw[3 * j], w2 = tw[3 * j + 1], w3 = tw[3 * j + 2];
            for (int c = 0; c < VL; c += W) {
                const vn a0r = vld(xr + in + c), a0i = vld(xi + in + c);
                const vn x1r = vld(xr + in + st + c), x1i = vld(xi + in + st + c);
                const vn x2r = vld(xr + in + 2 * st + c), x2i = vld(xi + in + 2 * st + c);
                const vn x3r = vld(xr + in + 3 * st + c), x3i = vld(xi + in + 3 * st + c);
                const vn a1r = CMUL_R(x1r, x1i, w1), a1i = CMUL_I(x1r, x1i, w1);
                const vn a2r = CMUL_R(x2r, x2i, w2), a2i = CMUL_I(x2r, x2i, w2);
                const vn a3r = CMUL_R(x3r, x3i, w3), a3i = CMUL_I(x3r, x3i, w3);
                const vn t0r = a0r + a2r, t0i = a0i + a2i;
                const vn t1r = a0r - a2r, t1i = a0i - a2i;
                const vn t2r = a1r + a3r, t2i = a1i + a3i;
                const vn t3r = a1r - a3r, t3i = a1i - a3i;
                vst(yr + out + c, t0r + t2r);
                vst(yi + out + c, t0i + t2i);
                vst(yr + out + o + c, t1r + t3i); /* t1 - i * t3 */
                vst(yi + out + o + c, t1i - t3r);
                vst(yr + out + 2 * o + c, t0r - t2r);
                vst(yi + out + 2 * o + c, t0i - t2i);
                vst(yr + out + 3 * o + c, t1r - t3i); /* t1 + i * t3 */
                vst(yi + out + 3 * o + c, t1i + t3r);
            }
        }
    }
}

FFT_INLINE void pass5(int ns, int nb, int stride, const af_cpx *tw, const float *xr,
                      const float *xi, float *yr, float *yi) {
    const float c1 = 0.309016994374947424102293417182819059f;
    const float c2 = -0.809016994374947424102293417182819059f;
    const float s1 = 0.951056516295153572116439333379382143f;
    const float s2 = 0.587785252292473129168705954639072769f;
    const size_t st = (size_t)stride * VL, o = (size_t)ns * VL;
    for (int b = 0; b < nb; b++) {
        for (int j = 0; j < ns; j++) {
            const size_t in = (size_t)(b * ns + j) * VL, out = (size_t)(b * ns * 5 + j) * VL;
            const af_cpx w1 = tw[4 * j], w2 = tw[4 * j + 1], w3 = tw[4 * j + 2], w4 = tw[4 * j + 3];
            for (int c = 0; c < VL; c += W) {
                const vn a0r = vld(xr + in + c), a0i = vld(xi + in + c);
                const vn x1r = vld(xr + in + st + c), x1i = vld(xi + in + st + c);
                const vn x2r = vld(xr + in + 2 * st + c), x2i = vld(xi + in + 2 * st + c);
                const vn x3r = vld(xr + in + 3 * st + c), x3i = vld(xi + in + 3 * st + c);
                const vn x4r = vld(xr + in + 4 * st + c), x4i = vld(xi + in + 4 * st + c);
                const vn a1r = CMUL_R(x1r, x1i, w1), a1i = CMUL_I(x1r, x1i, w1);
                const vn a2r = CMUL_R(x2r, x2i, w2), a2i = CMUL_I(x2r, x2i, w2);
                const vn a3r = CMUL_R(x3r, x3i, w3), a3i = CMUL_I(x3r, x3i, w3);
                const vn a4r = CMUL_R(x4r, x4i, w4), a4i = CMUL_I(x4r, x4i, w4);
                const vn t1r = a1r + a4r, t1i = a1i + a4i;
                const vn t2r = a2r + a3r, t2i = a2i + a3i;
                const vn t3r = a1r - a4r, t3i = a1i - a4i;
                const vn t4r = a2r - a3r, t4i = a2i - a3i;
                const vn m1r = a0r + c1 * t1r + c2 * t2r, m1i = a0i + c1 * t1i + c2 * t2i;
                const vn m2r = a0r + c2 * t1r + c1 * t2r, m2i = a0i + c2 * t1i + c1 * t2i;
                /* n1 = -i (s1 t3 + s2 t4), n2 = -i (s2 t3 - s1 t4) */
                const vn n1r = s1 * t3i + s2 * t4i, n1i = -(s1 * t3r + s2 * t4r);
                const vn n2r = s2 * t3i - s1 * t4i, n2i = -(s2 * t3r - s1 * t4r);
                vst(yr + out + c, a0r + t1r + t2r);
                vst(yi + out + c, a0i + t1i + t2i);
                vst(yr + out + o + c, m1r + n1r);
                vst(yi + out + o + c, m1i + n1i);
                vst(yr + out + 2 * o + c, m2r + n2r);
                vst(yi + out + 2 * o + c, m2i + n2i);
                vst(yr + out + 3 * o + c, m2r - n2r);
                vst(yi + out + 3 * o + c, m2i - n2i);
                vst(yr + out + 4 * o + c, m1r - n1r);
                vst(yi + out + 4 * o + c, m1i - n1i);
            }
        }
    }
}

static void rfft_power(const af_rfft *p, float *re, float *im, float *scr_re, float *scr_im,
                       float *power) {
    const int m = p->m;
    float *xr = re, *xi = im, *yr = scr_re, *yi = scr_im;
#if defined(AF_VEC_EXT) && defined(__aarch64__)
    /* Fixed frontend sizes: same butterflies and operation order, with constant
     * stage dimensions so the compiler can specialize indexing and loops.
     * These schedules match fft.c factorize(): 4/2/5/5 and 4/4/4/4. */
    if (m == 200) {
        pass4(1, 50, 50, p->stages[0].tw, re, im, scr_re, scr_im);
        pass2(4, 25, 100, p->stages[1].tw, scr_re, scr_im, re, im);
        pass5(8, 5, 40, p->stages[2].tw, re, im, scr_re, scr_im);
        pass5(40, 1, 40, p->stages[3].tw, scr_re, scr_im, re, im);
    } else if (m == 256) {
        pass4(1, 64, 64, p->stages[0].tw, re, im, scr_re, scr_im);
        pass4(4, 16, 64, p->stages[1].tw, scr_re, scr_im, re, im);
        pass4(16, 4, 64, p->stages[2].tw, re, im, scr_re, scr_im);
        pass4(64, 1, 64, p->stages[3].tw, scr_re, scr_im, re, im);
    } else
#endif
        for (int s = 0; s < p->n_stages; s++) {
            const af_fft_stage *st = &p->stages[s];
            const int stride = m / st->radix;
            const int nb = stride / st->ns;
            switch (st->radix) {
            case 2:
                pass2(st->ns, nb, stride, st->tw, xr, xi, yr, yi);
                break;
            case 3:
                pass3(st->ns, nb, stride, st->tw, xr, xi, yr, yi);
                break;
            case 4:
                pass4(st->ns, nb, stride, st->tw, xr, xi, yr, yi);
                break;
            default:
                pass5(st->ns, nb, stride, st->tw, xr, xi, yr, yi);
                break;
            }
            float *tr = xr, *ti = xi;
            xr = yr;
            xi = yi;
            yr = tr;
            yi = ti;
        }

    /* Split post-processing fused with |X|^2 (see af_rfft_forward). */
    for (int c = 0; c < VL; c += W) {
        const vn z0r = vld(xr + c), z0i = vld(xi + c);
        const vn dc = z0r + z0i, ny = z0r - z0i;
        vst(power + c, dc * dc);
        vst(power + (size_t)m * VL + c, ny * ny);
    }
#if defined(AF_VEC_EXT) && defined(__aarch64__)
    /* Load each conjugate pair once. Use the original twiddle and subtraction order
     * for each bin, including k == m-k, so the scalar result is preserved exactly. */
    for (int k = 1; k <= m / 2; k++) {
        const af_cpx w = p->post_tw[k], wc = p->post_tw[m - k];
        const size_t ik = (size_t)k * VL, ic = (size_t)(m - k) * VL;
        for (int c = 0; c < VL; c += W) {
            const vn zkr = vld(xr + ik + c), zki = vld(xi + ik + c);
            const vn zcr = vld(xr + ic + c), zci = vld(xi + ic + c);
            const vn er = 0.5f * (zkr + zcr), or_ = 0.5f * (zki + zci);
            const vn ei = 0.5f * (zki - zci), oi = -0.5f * (zkr - zcr);
            const vn eic = 0.5f * (zci - zki), oic = -0.5f * (zcr - zkr);
            const vn Xr = er + (w.re * or_ - w.im * oi);
            const vn Xi = ei + (w.re * oi + w.im * or_);
            const vn Xcr = er + (wc.re * or_ - wc.im * oic);
            const vn Xci = eic + (wc.re * oic + wc.im * or_);
            vst(power + ik + c, Xr * Xr + Xi * Xi);
            vst(power + ic + c, Xcr * Xcr + Xci * Xci);
        }
    }
#else
    for (int k = 1; k < m; k++) {
        const af_cpx w = p->post_tw[k];
        const size_t ik = (size_t)k * VL, ic = (size_t)(m - k) * VL;
        for (int c = 0; c < VL; c += W) {
            const vn zkr = vld(xr + ik + c), zki = vld(xi + ik + c);
            const vn zcr = vld(xr + ic + c), zci = vld(xi + ic + c);
            const vn er = 0.5f * (zkr + zcr), ei = 0.5f * (zki - zci);
            const vn or_ = 0.5f * (zki + zci), oi = -0.5f * (zkr - zcr);
            const vn Xr = er + (w.re * or_ - w.im * oi);
            const vn Xi = ei + (w.re * oi + w.im * or_);
            vst(power + ik + c, Xr * Xr + Xi * Xi);
        }
    }
#endif
}

/* ---------------------------------------------------------------------------------- */
/* Frame gathers (AoS rows -> SoA lanes)                                              */
/* ---------------------------------------------------------------------------------- */

#if defined(AF_VEC_EXT)
#if defined(__has_builtin)
#if __has_builtin(__builtin_shufflevector)
#define AF_SHUF(a, b, ...) __builtin_shufflevector(a, b, __VA_ARGS__)
#endif
#endif
#ifndef AF_SHUF
#define AF_SHUF(a, b, ...) __builtin_shuffle(a, b, (vni){__VA_ARGS__})
#endif
#endif

#if defined(AF_VEC_EXT) && W == 8 && VL == 8
/* r[l][j] -> c[j][l] (unpack / shuffle / 128-bit permute pattern) */
VINLINE void transpose8(const vn r[8], vn c[8]) {
    const vn t0 = AF_SHUF(r[0], r[1], 0, 8, 1, 9, 4, 12, 5, 13);
    const vn t1 = AF_SHUF(r[0], r[1], 2, 10, 3, 11, 6, 14, 7, 15);
    const vn t2 = AF_SHUF(r[2], r[3], 0, 8, 1, 9, 4, 12, 5, 13);
    const vn t3 = AF_SHUF(r[2], r[3], 2, 10, 3, 11, 6, 14, 7, 15);
    const vn t4 = AF_SHUF(r[4], r[5], 0, 8, 1, 9, 4, 12, 5, 13);
    const vn t5 = AF_SHUF(r[4], r[5], 2, 10, 3, 11, 6, 14, 7, 15);
    const vn t6 = AF_SHUF(r[6], r[7], 0, 8, 1, 9, 4, 12, 5, 13);
    const vn t7 = AF_SHUF(r[6], r[7], 2, 10, 3, 11, 6, 14, 7, 15);
    const vn u0 = AF_SHUF(t0, t2, 0, 1, 8, 9, 4, 5, 12, 13);
    const vn u1 = AF_SHUF(t0, t2, 2, 3, 10, 11, 6, 7, 14, 15);
    const vn u2 = AF_SHUF(t1, t3, 0, 1, 8, 9, 4, 5, 12, 13);
    const vn u3 = AF_SHUF(t1, t3, 2, 3, 10, 11, 6, 7, 14, 15);
    const vn u4 = AF_SHUF(t4, t6, 0, 1, 8, 9, 4, 5, 12, 13);
    const vn u5 = AF_SHUF(t4, t6, 2, 3, 10, 11, 6, 7, 14, 15);
    const vn u6 = AF_SHUF(t5, t7, 0, 1, 8, 9, 4, 5, 12, 13);
    const vn u7 = AF_SHUF(t5, t7, 2, 3, 10, 11, 6, 7, 14, 15);
    c[0] = AF_SHUF(u0, u4, 0, 1, 2, 3, 8, 9, 10, 11);
    c[1] = AF_SHUF(u1, u5, 0, 1, 2, 3, 8, 9, 10, 11);
    c[2] = AF_SHUF(u2, u6, 0, 1, 2, 3, 8, 9, 10, 11);
    c[3] = AF_SHUF(u3, u7, 0, 1, 2, 3, 8, 9, 10, 11);
    c[4] = AF_SHUF(u0, u4, 4, 5, 6, 7, 12, 13, 14, 15);
    c[5] = AF_SHUF(u1, u5, 4, 5, 6, 7, 12, 13, 14, 15);
    c[6] = AF_SHUF(u2, u6, 4, 5, 6, 7, 12, 13, 14, 15);
    c[7] = AF_SHUF(u3, u7, 4, 5, 6, 7, 12, 13, 14, 15);
}

static void gather_packed(const float *const *src, int len, const float *win, float *re,
                          float *im) {
    for (int k0 = 0; k0 < len; k0 += 8) {
        vn r[8], c[8];
        for (int l = 0; l < 8; l++)
            r[l] = vld(src[l] + k0);
        transpose8(r, c);
        for (int j = 0; j < 8; j += 2) {
            vst(re + (size_t)((k0 + j) >> 1) * VL, c[j] * win[k0 + j]);
            vst(im + (size_t)((k0 + j) >> 1) * VL, c[j + 1] * win[k0 + j + 1]);
        }
    }
}

static void gather_scaled(const float *const *src, int len, float scale, float *out) {
    for (int k0 = 0; k0 < len; k0 += 8) {
        vn r[8], c[8];
        for (int l = 0; l < 8; l++)
            r[l] = vld(src[l] + k0);
        transpose8(r, c);
        for (int j = 0; j < 8; j++)
            vst(out + (size_t)(k0 + j) * VL, c[j] * scale);
    }
}
#elif defined(AF_VEC_EXT) && defined(__aarch64__) && W == 4 && VL == 8
/* Transpose four frames at a time using contiguous loads on NEON. The two
 * groups fill the low/high halves of each eight-frame SoA row. */
VINLINE void transpose4(const vn r[4], vn c[4]) {
    const vn t0 = AF_SHUF(r[0], r[1], 0, 4, 1, 5);
    const vn t1 = AF_SHUF(r[0], r[1], 2, 6, 3, 7);
    const vn t2 = AF_SHUF(r[2], r[3], 0, 4, 1, 5);
    const vn t3 = AF_SHUF(r[2], r[3], 2, 6, 3, 7);
    c[0] = AF_SHUF(t0, t2, 0, 1, 4, 5);
    c[1] = AF_SHUF(t0, t2, 2, 3, 6, 7);
    c[2] = AF_SHUF(t1, t3, 0, 1, 4, 5);
    c[3] = AF_SHUF(t1, t3, 2, 3, 6, 7);
}

static void gather_packed(const float *const *src, int len, const float *win, float *re,
                          float *im) {
    for (int k = 0; k < len; k += 4)
        for (int l = 0; l < VL; l += 4) {
            vn r[4], c[4];
            for (int j = 0; j < 4; j++)
                r[j] = vld(src[l + j] + k);
            transpose4(r, c);
            for (int j = 0; j < 4; j += 2) {
                vst(re + (size_t)((k + j) / 2) * VL + l, c[j] * win[k + j]);
                vst(im + (size_t)((k + j) / 2) * VL + l, c[j + 1] * win[k + j + 1]);
            }
        }
}

/* The compiler's scalar gather is faster for this scaled layout on M1. */
static void gather_scaled(const float *const *src, int len, float scale, float *out) {
    for (int k = 0; k < len; k++)
        for (int l = 0; l < VL; l++)
            out[(size_t)k * VL + l] = src[l][k] * scale;
}
#else
static void gather_packed(const float *const *src, int len, const float *win, float *re,
                          float *im) {
    for (int k = 0; k < len; k++) {
        float *dst = (k & 1 ? im : re) + (size_t)(k >> 1) * VL;
        for (int l = 0; l < VL; l++)
            dst[l] = src[l][k] * win[k];
    }
}

static void gather_scaled(const float *const *src, int len, float scale, float *out) {
    for (int k = 0; k < len; k++)
        for (int l = 0; l < VL; l++)
            out[(size_t)k * VL + l] = src[l][k] * scale;
}
#endif

/* ---------------------------------------------------------------------------------- */
/* Mel                                                                                */
/* ---------------------------------------------------------------------------------- */

static void mel(const af_melbank *mb, const float *power, float *out) {
    for (int m = 0; m < mb->n_mels; m++) {
        const float *w = mb->w + mb->offset[m];
        const float *p = power + (size_t)mb->start[m] * VL;
        for (int c = 0; c < VL; c += W) {
            vn acc = VZERO;
            for (int k = 0; k < mb->len[m]; k++)
                acc = acc + w[k] * vld(p + (size_t)k * VL + c);
            vst(out + (size_t)m * VL + c, acc);
        }
    }
}

/* ---------------------------------------------------------------------------------- */
/* Logarithm (Cephes logf: frexp + degree-9 polynomial, ~1 ulp), branch-free          */
/* ---------------------------------------------------------------------------------- */

#define LOG_SQRTHF 0.707106781186547524f
#define LOG_P0 7.0376836292e-2f
#define LOG_P1 -1.1514610310e-1f
#define LOG_P2 1.1676998740e-1f
#define LOG_P3 -1.2420140846e-1f
#define LOG_P4 1.4249322787e-1f
#define LOG_P5 -1.6668057665e-1f
#define LOG_P6 2.0000714765e-1f
#define LOG_P7 -2.4999993993e-1f
#define LOG_P8 3.3333331174e-1f
#define LOG_Q1 -2.12194440e-4f
#define LOG_Q2 0.693359375f

#if defined(AF_VEC_EXT)
static void log_floor_scaled(float *x, size_t n, float add, float floor, float scale) {
    const vn vfloor = VZERO + floor;
    for (size_t i = 0; i < n; i += W) { /* n is a multiple of AF_VL */
        const vn a = vld(x + i) + add;
        const vni keep = a > vfloor; /* false for NaN -> floor */
        const vn v = (vn)((keep & (vni)a) | (~keep & (vni)vfloor));
        const vni u = (vni)v;
        vni e = ((u >> 23) & 0xff) - 126; /* v = m * 2^e, m in [0.5, 1) */
        vn m = (vn)((u & 0x007fffff) | 0x3f000000);
        const vni lt = m < LOG_SQRTHF;
        e += lt;                            /* lt is -1 where true */
        m = (m + (vn)(lt & (vni)m)) - 1.0f; /* (lt ? m + m : m) - 1 */
        const vn z = m * m;
        vn y = VZERO + LOG_P0;
        y = y * m + LOG_P1;
        y = y * m + LOG_P2;
        y = y * m + LOG_P3;
        y = y * m + LOG_P4;
        y = y * m + LOG_P5;
        y = y * m + LOG_P6;
        y = y * m + LOG_P7;
        y = y * m + LOG_P8;
        y = y * m * z;
        const vn fe = __builtin_convertvector(e, vn);
        y += LOG_Q1 * fe;
        y += -0.5f * z;
        vn r = m + y;
        r += LOG_Q2 * fe;
        vst(x + i, r * scale);
    }
}
#else
static void log_floor_scaled(float *x, size_t n, float add, float floor, float scale) {
    for (size_t i = 0; i < n; i++) {
        const float a = x[i] + add;
        const float v = a > floor ? a : floor;
        uint32_t u;
        memcpy(&u, &v, sizeof(u));
        int e = (int)((u >> 23) & 0xffu) - 126;
        u = (u & 0x007fffffu) | 0x3f000000u;
        float m;
        memcpy(&m, &u, sizeof(m));
        const int lt = m < LOG_SQRTHF;
        e -= lt;
        m = (m + (lt ? m : 0.0f)) - 1.0f;
        const float z = m * m;
        float y = LOG_P0;
        y = y * m + LOG_P1;
        y = y * m + LOG_P2;
        y = y * m + LOG_P3;
        y = y * m + LOG_P4;
        y = y * m + LOG_P5;
        y = y * m + LOG_P6;
        y = y * m + LOG_P7;
        y = y * m + LOG_P8;
        y = y * m * z;
        const float fe = (float)e;
        y += LOG_Q1 * fe;
        y += -0.5f * z;
        float r = m + y;
        r += LOG_Q2 * fe;
        x[i] = r * scale;
    }
}
#endif

/* (x + add) > floor ? x + add : floor, then the log scaled; add = 0 leaves x unchanged (but
 * -0, whose log is the floor's either way). */
static void log10_floor(float *x, size_t n, float floor) {
    log_floor_scaled(x, n, 0.0f, floor, 0.434294481903251827651128918916605082f);
}

static void log_floor(float *x, size_t n, float floor) {
    log_floor_scaled(x, n, 0.0f, floor, 1.0f);
}

static void log_add(float *x, size_t n, float add) {
    log_floor_scaled(x, n, add, add, 1.0f);
}

/* sqrtf is correctly rounded, so every build agrees; the kernels are compiled with
 * -fno-math-errno (no errno for negative inputs, which do not occur), so this vectorizes. */
static void sqrt_f32(float *x, size_t n) {
    for (size_t i = 0; i < n; i++)
        x[i] = sqrtf(x[i]);
}

/* ---------------------------------------------------------------------------------- */
/* Kaldi frame preprocessing                                                          */
/* ---------------------------------------------------------------------------------- */

static void kaldi_prep(const float *frames, int len, const float *window, float coef, int padded,
                       float *re, float *im) {
    /* Per-lane sequential double sums (same order as a per-frame loop); the lane loop is
     * innermost so the 8 independent chains run in parallel. */
    double sum[VL];
    float mean[VL];
    for (int l = 0; l < VL; l++)
        sum[l] = 0.0;
    for (int k = 0; k < len; k++)
        for (int l = 0; l < VL; l++)
            sum[l] += frames[(size_t)k * VL + l];
    for (int l = 0; l < VL; l++)
        mean[l] = (float)(sum[l] / len);
    for (int c = 0; c < VL; c += W) {
        const vn mu = vld(mean + c);
        vn prev = vld(frames + c) - mu; /* y[0] = x[0] - coef * x[0] */
        int k = 0;
        for (; k < len; k++) {
            const vn g = vld(frames + (size_t)k * VL + c) - mu;
            vst((k & 1 ? im : re) + (size_t)(k >> 1) * VL + c, (g - coef * prev) * window[k]);
            prev = g;
        }
        for (; k < padded; k++)
            vst((k & 1 ? im : re) + (size_t)(k >> 1) * VL + c, VZERO);
    }
}

/* ---------------------------------------------------------------------------------- */
/* Whisper normalization                                                              */
/* ---------------------------------------------------------------------------------- */

/* max(init, x[0..n)) over finite floats. Max is order-independent, so the vector reduction
 * matches a sequential one exactly. */
static float max_f32(const float *x, size_t n, float init) {
    float gmax = init;
    size_t i = 0;
#if defined(AF_VEC_EXT)
    vn vmax = VZERO + gmax;
    for (; i + W <= n; i += W) {
        const vn a = vld(x + i);
        const vni gt = a > vmax;
        vmax = (vn)((gt & (vni)a) | (~gt & (vni)vmax));
    }
    float lanes[W];
    memcpy(lanes, &vmax, sizeof(lanes));
    for (int l = 0; l < W; l++)
        gmax = lanes[l] > gmax ? lanes[l] : gmax;
#endif
    for (; i < n; i++)
        gmax = x[i] > gmax ? x[i] : gmax;
    return gmax;
}

/* x = (max(x, gmax - 8) + 4) / 4 over n floats. */
static void whisper_normalize(float *x, size_t n, float gmax) {
    const float floor_v = gmax - 8.0f;
    for (size_t i = 0; i < n; i++)
        x[i] = ((x[i] > floor_v ? x[i] : floor_v) + 4.0f) * 0.25f;
}

/* ---------------------------------------------------------------------------------- */
/* Resampler dot product (docs/load.md, float path)                                   */
/* ---------------------------------------------------------------------------------- */

/* 8 accumulators fed with fused multiply-adds, then (a0+a4, a1+a5, a2+a6, a3+a7) and
 * pairwise sums: ffmpeg's AVX2/FMA3 kernel. fmaf() is correctly rounded, so this is the same
 * in every build; with -mfma (the avx2 table) the lane loop becomes one vfmadd per 8 taps. */
#if defined(__AVX__) && defined(__FMA__)
#include <immintrin.h>
VINLINE float resample_dot_inl(const float *h, const float *x, int width) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < width; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(h + i), _mm256_loadu_ps(x + i), acc);
    __m128 s =
        _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1)); /* a_l+a_l+4 */
    __m128 t = _mm_add_ps(s, _mm_movehl_ps(s, s));                              /* s0+s2, s1+s3 */
    return _mm_cvtss_f32(_mm_add_ss(t, _mm_shuffle_ps(t, t, 1)));               /* t0 + t1 */
}
#elif defined(AF_VEC_EXT) && defined(__aarch64__)
#include <arm_neon.h>
VINLINE float resample_dot_inl(const float *h, const float *x, int width) {
    /* Keep the eight FMA chains and reduction tree of the scalar / AVX2 paths.
     * Explicit 128-bit vectors avoid Clang splitting them into four 64-bit FMAs. */
    float32x4_t a = vdupq_n_f32(0.0f), b = vdupq_n_f32(0.0f);
    for (int i = 0; i < width; i += 8) {
        a = vfmaq_f32(a, vld1q_f32(h + i), vld1q_f32(x + i));
        b = vfmaq_f32(b, vld1q_f32(h + i + 4), vld1q_f32(x + i + 4));
    }
    const float32x4_t s = vaddq_f32(a, b);
    const float32x2_t t = vadd_f32(vget_low_f32(s), vget_high_f32(s));
    return vget_lane_f32(t, 0) + vget_lane_f32(t, 1);
}
#else
VINLINE float resample_dot_inl(const float *h, const float *x, int width) {
    float acc[8] = {0};
    for (int i = 0; i < width; i += 8)
        for (int l = 0; l < 8; l++)
            acc[l] = fmaf(h[i + l], x[i + l], acc[l]);
    float s0 = acc[0] + acc[4], s1 = acc[1] + acc[5], s2 = acc[2] + acc[6], s3 = acc[3] + acc[7];
    float t0 = s0 + s2, t1 = s1 + s3;
    return t0 + t1;
}
#endif

static float resample_dot(const float *h, const float *x, int width) {
    return resample_dot_inl(h, x, width);
}

static void resample_run(const float *bank, int width, const float *x, int m, int step_int,
                         int step_rem, int q, size_t count, float *y) {
    for (size_t k = 0; k < count; k++) {
        y[k] = resample_dot_inl(bank + (size_t)m * width, x, width);
        x += step_int;
        m += step_rem;
        if (m >= q) {
            m -= q;
            x++;
        }
    }
}

/* ---------------------------------------------------------------------------------- */
/* Output transposes and input checks                                                 */
/* ---------------------------------------------------------------------------------- */

#if defined(AF_VEC_EXT) && W == 8 && VL == 8
static void soa_to_rows(const float *soa, int n, int count, float *out, size_t ld) {
    int i0 = 0;
    for (; i0 + 8 <= n; i0 += 8) { /* 8 x 8 tiles: c[l] is lane l's elements i0 .. i0 + 7 */
        vn r[8], c[8];
        for (int j = 0; j < 8; j++)
            r[j] = vld(soa + (size_t)(i0 + j) * VL);
        transpose8(r, c);
        for (int l = 0; l < count; l++)
            vst(out + (size_t)l * ld + (size_t)i0, c[l]);
    }
    for (int l = 0; l < count; l++)
        for (int i = i0; i < n; i++)
            out[(size_t)l * ld + (size_t)i] = soa[(size_t)i * VL + (size_t)l];
}
#else
static void soa_to_rows(const float *soa, int n, int count, float *out, size_t ld) {
    for (int l = 0; l < count; l++)
        for (int i = 0; i < n; i++)
            out[(size_t)l * ld + (size_t)i] = soa[(size_t)i * VL + (size_t)l];
}
#endif

/* Blocks of a fixed size so that the inner loop vectorizes; a block with a NaN / inf ends the
 * scan. */
static int all_finite(const float *x, size_t n) {
    for (size_t i0 = 0; i0 < n; i0 += 1024) {
        const size_t len = n - i0 < 1024 ? n - i0 : 1024;
        uint32_t bad = 0;
        for (size_t i = 0; i < len; i++) {
            uint32_t u;
            memcpy(&u, x + i0 + i, sizeof(u));
            bad |= (u & 0x7F800000u) == 0x7F800000u;
        }
        if (bad)
            return 0;
    }
    return 1;
}

const af_kernels AF_CAT(af_kernels, AF_ISA) = {
    AF_STR(AF_ISA), rfft_power, gather_packed, gather_scaled,     mel,          log10_floor,
    log_floor,      kaldi_prep, max_f32,       whisper_normalize, resample_dot, resample_run,
    log_add,        sqrt_f32,   soa_to_rows,   all_finite,
};
