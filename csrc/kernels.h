/*
 * SIMD kernels, processing AF_VL frames at once in a structure-of-arrays layout:
 * element i of frame (lane) l lives at buf[i * AF_VL + l]. The inner loops run over
 * lanes, so they vectorize with any SIMD width.
 *
 * kernels_impl.c is compiled once per ISA (baseline, and AVX2 on x86-64); the table
 * is picked at runtime. The kernels use no FMA (except explicit fmaf() in resample_dot) and
 * the same operation order in every build, so all variants produce bit-identical results.
 */
#ifndef ASRFRONT_KERNELS_H
#define ASRFRONT_KERNELS_H

#include <stddef.h>

#include "fft.h"
#include "melbank.h"

#define AF_VL 8

typedef struct af_kernels {
    const char *name;

    /* Power spectrum of AF_VL real frames of size p->n. Input: the packed complex
     * sequence z[i] = x[2i] + i x[2i+1], i < m = n/2, as split re/im SoA buffers of
     * m * AF_VL floats. re/im/scr_re/scr_im are clobbered. power: (m + 1) * AF_VL. */
    void (*rfft_power)(const af_rfft *p, float *re, float *im, float *scr_re, float *scr_im,
                       float *power);

    /* Transposing frame gathers. src[l] is the first sample of lane l's frame; len must
     * be a multiple of 8.
     * gather_packed: x = src[l][k] * win[k]; even k -> re[(k/2) * AF_VL + l], odd -> im.
     * gather_scaled: out[k * AF_VL + l] = src[l][k] * scale. */
    void (*gather_packed)(const float *const *src, int len, const float *win, float *re, float *im);
    void (*gather_scaled)(const float *const *src, int len, float scale, float *out);

    /* mel[m * AF_VL + l] = sum_k w[m][k] * power[k * AF_VL + l]. */
    void (*mel)(const af_melbank *mb, const float *power, float *mel);

    /* In place: x = log10(max(x, floor)) / ln(max(x, floor)) over n floats. floor must
     * be a positive normal number. */
    void (*log10_floor)(float *x, size_t n, float floor);
    void (*log_floor)(float *x, size_t n, float floor);

    /* Kaldi frame preprocessing for AF_VL frames: frames is (len, AF_VL) SoA. Removes
     * the per-frame mean (double accumulation), applies pre-emphasis
     * y[k] = x[k] - coef * x[k-1] (y[0] = x[0] - coef * x[0]) and the window, zero-pads
     * to `padded` and packs even/odd samples into re/im ((padded / 2) * AF_VL each). */
    void (*kaldi_prep)(const float *frames, int len, const float *window, float coef, int padded,
                       float *re, float *im);

    /* Whisper dynamic range + scale: x = (max(x, max(x) - 8) + 4) / 4, in place. */
    void (*whisper_normalize)(float *x, size_t n);

    /* Resampler float path: sum h[i] * x[i] over i < width (a multiple of 8) with 8 FMA
     * accumulators and a fixed reduction tree (docs/load.md). The only kernel that uses FMA,
     * through fmaf(), which is correctly rounded: identical in every build. */
    float (*resample_dot)(const float *h, const float *x, int width);
    /* Four independent resample_dot()s at once (hides the FMA latency): out[j] =
     * resample_dot(h[j], x[j], width), bit for bit. */
    void (*resample_dot4)(const float *const *h, const float *const *x, int width, float *out);
} af_kernels;

/* Best table for this CPU. ASRFRONT_KERNELS=generic forces the baseline build,
 * ASRFRONT_KERNELS=scalar the plain-C build without vector extensions (the code path
 * MSVC compiles; used to test it on every platform). */
const af_kernels *af_kernels_select(void);

#endif /* ASRFRONT_KERNELS_H */
