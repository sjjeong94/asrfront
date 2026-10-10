#include "melbank.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "asrfront.h"

/* Build the sparse bank from dense float weights (n_mels, n_bins). */
static int from_dense(const float *dense, int n_mels, int n_bins, af_melbank **out) {
    af_melbank *mb = calloc(1, sizeof(*mb));
    if (!mb)
        return AF_ENOMEM;
    mb->n_mels = n_mels;
    mb->n_bins = n_bins;
    mb->start = malloc(3 * (size_t)n_mels * sizeof(int));
    if (!mb->start) {
        free(mb);
        return AF_ENOMEM;
    }
    mb->len = mb->start + n_mels;
    mb->offset = mb->len + n_mels;

    size_t total = 0;
    for (int m = 0; m < n_mels; m++) {
        const float *row = dense + (size_t)m * n_bins;
        int lo = 0, hi = n_bins;
        while (lo < n_bins && row[lo] == 0.0f)
            lo++;
        while (hi > lo && row[hi - 1] == 0.0f)
            hi--;
        mb->start[m] = lo;
        mb->len[m] = hi - lo;
        mb->offset[m] = (int)total;
        total += (size_t)(hi - lo);
    }
    mb->w = malloc((total ? total : 1) * sizeof(float));
    if (!mb->w) {
        af_melbank_destroy(mb);
        return AF_ENOMEM;
    }
    for (int m = 0; m < n_mels; m++)
        memcpy(mb->w + mb->offset[m], dense + (size_t)m * n_bins + mb->start[m],
               (size_t)mb->len[m] * sizeof(float));
    *out = mb;
    return AF_OK;
}

/* Slaney mel scale (librosa htk=False). */
#define SLANEY_F_SP (200.0 / 3.0)
#define SLANEY_MIN_LOG_HZ 1000.0
#define SLANEY_MIN_LOG_MEL (SLANEY_MIN_LOG_HZ / SLANEY_F_SP)

static double slaney_logstep(void) {
    return log(6.4) / 27.0;
}

static double hz_to_mel_slaney(double f) {
    if (f >= SLANEY_MIN_LOG_HZ)
        return SLANEY_MIN_LOG_MEL + log(f / SLANEY_MIN_LOG_HZ) / slaney_logstep();
    return f / SLANEY_F_SP;
}

static double mel_to_hz_slaney(double m) {
    if (m >= SLANEY_MIN_LOG_MEL)
        return SLANEY_MIN_LOG_HZ * exp(slaney_logstep() * (m - SLANEY_MIN_LOG_MEL));
    return SLANEY_F_SP * m;
}

int af_melbank_slaney(int sr, int n_fft, int n_mels, double fmin, double fmax, af_melbank **out) {
    if (!out)
        return AF_EINVAL;
    *out = NULL;
    if (sr <= 0 || n_fft <= 0 || n_mels <= 0 || fmin < 0 || fmax <= fmin)
        return AF_EINVAL;
    const int n_bins = n_fft / 2 + 1;
    const int n_pts = n_mels + 2;
    double *mel_f = malloc((size_t)n_pts * sizeof(double));
    float *dense = malloc((size_t)n_mels * n_bins * sizeof(float));
    if (!mel_f || !dense) {
        free(mel_f);
        free(dense);
        return AF_ENOMEM;
    }

    /* np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2), last point exact. */
    const double m0 = hz_to_mel_slaney(fmin), m1 = hz_to_mel_slaney(fmax);
    const double step = (m1 - m0) / (n_pts - 1);
    for (int i = 0; i < n_pts; i++)
        mel_f[i] = mel_to_hz_slaney(i == n_pts - 1 ? m1 : i * step + m0);

    for (int m = 0; m < n_mels; m++) {
        const double fdiff_lo = mel_f[m + 1] - mel_f[m];
        const double fdiff_hi = mel_f[m + 2] - mel_f[m + 1];
        const double enorm = 2.0 / (mel_f[m + 2] - mel_f[m]);
        for (int k = 0; k < n_bins; k++) {
            const double fft_f = (double)k * sr / n_fft; /* np.fft.rfftfreq */
            const double lower = -(mel_f[m] - fft_f) / fdiff_lo;
            const double upper = (mel_f[m + 2] - fft_f) / fdiff_hi;
            double tri = lower < upper ? lower : upper;
            tri = tri > 0.0 ? tri : 0.0;
            /* librosa stores the triangle in float32, then scales by enorm in float64. */
            dense[(size_t)m * n_bins + k] = (float)((double)(float)tri * enorm);
        }
    }
    int rc = from_dense(dense, n_mels, n_bins, out);
    free(mel_f);
    free(dense);
    return rc;
}

static double kaldi_mel(double f) {
    return 1127.0 * log(1.0 + f / 700.0);
}

int af_melbank_kaldi(int sr, int padded, int n_mels, double low_freq, double high_freq,
                     af_melbank **out) {
    if (!out)
        return AF_EINVAL;
    *out = NULL;
    const double nyquist = 0.5 * sr;
    if (high_freq <= 0.0)
        high_freq += nyquist;
    if (sr <= 0 || padded < 2 || padded % 2 != 0 || n_mels <= 0 || low_freq < 0.0 ||
        high_freq <= low_freq || high_freq > nyquist)
        return AF_EINVAL;

    const int n_fft_bins = padded / 2;
    const int n_bins = n_fft_bins + 1; /* plus the (always zero) Nyquist bin */
    float *dense = calloc((size_t)n_mels * n_bins, sizeof(float));
    if (!dense)
        return AF_ENOMEM;

    const double fft_bin_width = (double)sr / padded;
    const double mel_low = kaldi_mel(low_freq), mel_high = kaldi_mel(high_freq);
    const double mel_delta = (mel_high - mel_low) / (n_mels + 1);
    for (int m = 0; m < n_mels; m++) {
        const double left = mel_low + m * mel_delta;
        const double center = mel_low + (m + 1.0) * mel_delta;
        const double right = mel_low + (m + 2.0) * mel_delta;
        for (int k = 0; k < n_fft_bins; k++) {
            const double mel = kaldi_mel(fft_bin_width * k);
            const double up = (mel - left) / (center - left);
            const double down = (right - mel) / (right - center);
            double w = up < down ? up : down;
            dense[(size_t)m * n_bins + k] = (float)(w > 0.0 ? w : 0.0);
        }
    }
    int rc = from_dense(dense, n_mels, n_bins, out);
    free(dense);
    return rc;
}

void af_melbank_destroy(af_melbank *mb) {
    if (!mb)
        return;
    free(mb->start);
    free(mb->w);
    free(mb);
}

int af_melbank_n_mels(const af_melbank *mb) {
    return mb->n_mels;
}
int af_melbank_n_bins(const af_melbank *mb) {
    return mb->n_bins;
}

void af_melbank_apply(const af_melbank *mb, const float *power, float *mel) {
    for (int m = 0; m < mb->n_mels; m++) {
        const float *w = mb->w + mb->offset[m];
        const float *p = power + mb->start[m];
        const int len = mb->len[m];
        float acc = 0.0f;
        for (int k = 0; k < len; k++)
            acc += w[k] * p[k];
        mel[m] = acc;
    }
}

void af_melbank_to_dense(const af_melbank *mb, float *out) {
    memset(out, 0, (size_t)mb->n_mels * mb->n_bins * sizeof(float));
    for (int m = 0; m < mb->n_mels; m++)
        memcpy(out + (size_t)m * mb->n_bins + mb->start[m], mb->w + mb->offset[m],
               (size_t)mb->len[m] * sizeof(float));
}
