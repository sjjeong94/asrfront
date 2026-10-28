#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "asrfront.h"
#include "fft.h"
#include "kernels.h"
#include "melbank.h"

#define LFM3_N_FFT 400                   /* 25 ms window, no centering */
#define LFM3_HOP 160                     /* 10 ms */
#define LFM3_HALF (LFM3_N_FFT / 2)       /* packed complex size: 200 */
#define LFM3_N_BINS (LFM3_N_FFT / 2 + 1) /* 201 */
#define LFM3_N_MELS 128
#define LFM3_MEL_FLOOR 1e-12f /* log10(mel) is clamped to -12 */

struct af_lfm3_audio {
    const af_kernels *k;
    af_rfft *fft;
    af_melbank *mel;
    float window[LFM3_N_FFT];
    float edge[AF_VL][LFM3_N_FFT]; /* frames reaching past the input: zero-padded */
    /* AF_VL-frame SoA work buffers */
    float re[LFM3_HALF * AF_VL], im[LFM3_HALF * AF_VL];
    float scr_re[LFM3_HALF * AF_VL], scr_im[LFM3_HALF * AF_VL];
    float power[LFM3_N_BINS * AF_VL];
    float melv[LFM3_N_MELS * AF_VL];
};

int af_lfm3_audio_create(af_lfm3_audio **out) {
    if (!out)
        return AF_EINVAL;
    *out = NULL;
    af_lfm3_audio *h = calloc(1, sizeof(*h));
    if (!h)
        return AF_ENOMEM;
    h->k = af_kernels_select();
    int rc = af_rfft_create(LFM3_N_FFT, &h->fft);
    if (rc == AF_OK)
        rc = af_melbank_slaney(AF_SAMPLE_RATE, LFM3_N_FFT, LFM3_N_MELS, 0.0, AF_SAMPLE_RATE / 2.0,
                               &h->mel);
    if (rc != AF_OK) {
        af_lfm3_audio_destroy(h);
        return rc;
    }
    /* torch.hann_window(400), periodic */
    for (int i = 0; i < LFM3_N_FFT; i++)
        h->window[i] = (float)(0.5 - 0.5 * cos(6.283185307179586476925286766559 * i / LFM3_N_FFT));
    *out = h;
    return AF_OK;
}

void af_lfm3_audio_destroy(af_lfm3_audio *h) {
    if (!h)
        return;
    af_rfft_destroy(h->fft);
    af_melbank_destroy(h->mel);
    free(h);
}

size_t af_lfm3_audio_out_frames(size_t n_samples) {
    /* right-padded to at least one window, then to whole hops: 1 + ceil((n - 400) / 160) */
    return n_samples <= LFM3_N_FFT ? 1 : (n_samples - LFM3_N_FFT + LFM3_HOP - 1) / LFM3_HOP + 1;
}

size_t af_lfm3_audio_complete_frames(size_t n_samples) {
    return n_samples < LFM3_N_FFT ? 0 : (n_samples - LFM3_N_FFT) / LFM3_HOP + 1;
}

/* Frame t covers samples [t * hop, t * hop + 400) of the input followed by zeros. */
static const float *frame_src(const float *audio, size_t n, size_t t, float *edge) {
    const size_t i0 = t * LFM3_HOP;
    if (i0 + LFM3_N_FFT <= n)
        return audio + i0;
    const size_t have = i0 < n ? n - i0 : 0;
    if (have)
        memcpy(edge, audio + i0, have * sizeof(float));
    memset(edge + have, 0, (LFM3_N_FFT - have) * sizeof(float));
    return edge;
}

static int log_mel(af_lfm3_audio *h, const float *audio, size_t n_samples, size_t n_frames,
                   float *out, size_t ld, int channel_last) {
    if (!h || (!out && n_frames) || (!audio && n_samples) ||
        ld < (channel_last ? (size_t)LFM3_N_MELS : n_frames))
        return AF_EINVAL;
    if (n_frames > SIZE_MAX / LFM3_HOP) /* frame starts must not overflow */
        return AF_EINVAL;
    const af_kernels *k = h->k;
    for (size_t t0 = 0; t0 < n_frames; t0 += AF_VL) {
        const int count = n_frames - t0 < AF_VL ? (int)(n_frames - t0) : AF_VL;
        const float *src[AF_VL];
        src[0] = frame_src(audio, n_samples, t0, h->edge[0]);
        for (int l = 1; l < AF_VL; l++) /* unused lanes of the last block repeat lane 0 */
            src[l] = l < count ? frame_src(audio, n_samples, t0 + l, h->edge[l]) : src[0];
        k->gather_packed(src, LFM3_N_FFT, h->window, h->re, h->im);
        k->rfft_power(h->fft, h->re, h->im, h->scr_re, h->scr_im, h->power);
        k->mel(h->mel, h->power, h->melv);
        k->log10_floor(h->melv, (size_t)LFM3_N_MELS * AF_VL, LFM3_MEL_FLOOR);
        /* (max(x, -12) + 4) / 4: Whisper's normalization with a fixed max of -4 */
        k->whisper_normalize(h->melv, (size_t)LFM3_N_MELS * AF_VL, -4.0f);
        if (channel_last) { /* SoA lane l is output frame t0 + l */
            k->soa_to_rows(h->melv, LFM3_N_MELS, count, out + t0 * ld, ld);
            continue;
        }
        for (int m = 0; m < LFM3_N_MELS; m++) { /* SoA rows map to contiguous output runs */
            float *dst = out + (size_t)m * ld + t0;
            const float *row = h->melv + (size_t)m * AF_VL;
            for (int l = 0; l < count; l++)
                dst[l] = row[l];
        }
    }
    return AF_OK;
}

int af_lfm3_audio_log_mel(af_lfm3_audio *h, const float *audio, size_t n_samples, size_t n_frames,
                          float *out, size_t ld) {
    return log_mel(h, audio, n_samples, n_frames, out, ld, 0);
}

int af_lfm3_audio_log_mel_channel_last(af_lfm3_audio *h, const float *audio, size_t n_samples,
                                       size_t n_frames, float *out, size_t ld) {
    return log_mel(h, audio, n_samples, n_frames, out, ld, 1);
}
