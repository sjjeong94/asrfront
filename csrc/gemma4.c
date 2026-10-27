#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "asrfront.h"
#include "fft.h"
#include "kernels.h"
#include "melbank.h"

#define G4_FRAME 320               /* 20 ms */
#define G4_HOP 160                 /* 10 ms */
#define G4_PAD (G4_FRAME / 2)      /* semicausal padding: zeros before the first sample */
#define G4_FFT 512                 /* frame zero-padded to the next power of two */
#define G4_HALF (G4_FFT / 2)       /* packed complex size: 256 */
#define G4_N_BINS (G4_FFT / 2 + 1) /* 257 */
#define G4_N_MELS 128
#define G4_MEL_FLOOR 1e-3f

struct af_gemma4 {
    const af_kernels *k;
    af_rfft *fft;
    af_melbank *mel;
    float window[G4_FRAME];
    float edge[G4_FRAME]; /* frame 0: G4_PAD zeros, then the first G4_PAD samples */
    /* AF_VL-frame SoA work buffers */
    float re[G4_HALF * AF_VL], im[G4_HALF * AF_VL];
    float scr_re[G4_HALF * AF_VL], scr_im[G4_HALF * AF_VL];
    float power[G4_N_BINS * AF_VL];
    float melv[G4_N_MELS * AF_VL];
};

int af_gemma4_create(af_gemma4 **out) {
    if (!out)
        return AF_EINVAL;
    *out = NULL;
    af_gemma4 *h = calloc(1, sizeof(*h));
    if (!h)
        return AF_ENOMEM;
    h->k = af_kernels_select();
    int rc = af_rfft_create(G4_FFT, &h->fft);
    if (rc == AF_OK)
        rc = af_melbank_htk(AF_SAMPLE_RATE, G4_FFT, G4_N_MELS, 0.0, AF_SAMPLE_RATE / 2.0, &h->mel);
    if (rc != AF_OK) {
        af_gemma4_destroy(h);
        return rc;
    }
    /* transformers' window_function(320): np.hanning(321)[:-1] (periodic), stored as float32 */
    for (int i = 0; i < G4_FRAME; i++)
        h->window[i] = (float)(0.5 - 0.5 * cos(6.283185307179586476925286766559 * i / G4_FRAME));
    *out = h;
    return AF_OK;
}

void af_gemma4_destroy(af_gemma4 *h) {
    if (!h)
        return;
    af_rfft_destroy(h->fft);
    af_melbank_destroy(h->mel);
    free(h);
}

size_t af_gemma4_out_frames(size_t n_padded) {
    /* frames of G4_FRAME + 1 samples (the last one unused) over G4_PAD zeros + the input */
    return n_padded + G4_PAD < G4_FRAME + 1 ? 0 : (n_padded + G4_PAD - (G4_FRAME + 1)) / G4_HOP + 1;
}

size_t af_gemma4_valid_frames(size_t n_samples) {
    /* frame t is valid if sample t * hop + G4_PAD (its unused last one) is input, not padding */
    return n_samples == 0 ? 0 : (n_samples - 1) / G4_HOP;
}

/* Log-mel energies of frames t0 .. t0 + count - 1 (all valid) into h->melv (SoA). Frame t
 * covers input samples [t * hop - G4_PAD, t * hop + G4_PAD); only frame 0 reaches the padding. */
static void block(af_gemma4 *h, const float *audio, size_t t0, int count) {
    const af_kernels *k = h->k;
    const float *src[AF_VL];
    for (int l = 0; l < AF_VL; l++) { /* unused lanes of the last block repeat lane 0 */
        const size_t t = t0 + (size_t)(l < count ? l : 0);
        src[l] = t == 0 ? h->edge : audio + t * G4_HOP - G4_PAD;
    }
    k->gather_packed(src, G4_FRAME, h->window, h->re, h->im);
    /* zero padding to G4_FFT (rfft_power clobbers the buffers, so on every block) */
    memset(h->re + (G4_FRAME / 2) * AF_VL, 0, (G4_HALF - G4_FRAME / 2) * AF_VL * sizeof(float));
    memset(h->im + (G4_FRAME / 2) * AF_VL, 0, (G4_HALF - G4_FRAME / 2) * AF_VL * sizeof(float));
    k->rfft_power(h->fft, h->re, h->im, h->scr_re, h->scr_im, h->power);
    k->sqrt_f32(h->power, (size_t)G4_N_BINS * AF_VL); /* magnitude */
    k->mel(h->mel, h->power, h->melv);
    k->log_add(h->melv, (size_t)G4_N_MELS * AF_VL, G4_MEL_FLOOR);
}

/* out: (n_frames, 128), or (128, n_frames) without channel_last. */
static int log_mel(af_gemma4 *h, const float *audio, size_t n_samples, size_t n_frames,
                   int channel_last, float *out) {
    if (!h || (!out && n_frames) || (!audio && n_samples))
        return AF_EINVAL;
    if (n_frames > SIZE_MAX / (G4_N_MELS * sizeof(float))) /* no buffer can be that large */
        return AF_EINVAL;
    size_t valid = af_gemma4_valid_frames(n_samples);
    valid = valid < n_frames ? valid : n_frames;
    if (valid > 0) { /* then n_samples > G4_PAD */
        memset(h->edge, 0, G4_PAD * sizeof(float));
        memcpy(h->edge + G4_PAD, audio, G4_PAD * sizeof(float));
    }
    for (size_t t0 = 0; t0 < valid; t0 += AF_VL) {
        const int count = valid - t0 < AF_VL ? (int)(valid - t0) : AF_VL;
        block(h, audio, t0, count);
        if (channel_last) {
            h->k->soa_to_rows(h->melv, G4_N_MELS, count, out + t0 * G4_N_MELS, G4_N_MELS);
            continue;
        }
        for (int m = 0; m < G4_N_MELS; m++) { /* SoA rows map to contiguous output runs */
            float *dst = out + (size_t)m * n_frames + t0;
            const float *row = h->melv + (size_t)m * AF_VL;
            for (int l = 0; l < count; l++)
                dst[l] = row[l];
        }
    }
    if (n_frames == valid) /* the padding's frames, if any, are masked to zero */
        return AF_OK;
    if (channel_last) {
        memset(out + valid * G4_N_MELS, 0, (n_frames - valid) * G4_N_MELS * sizeof(float));
        return AF_OK;
    }
    for (int m = 0; m < G4_N_MELS; m++)
        memset(out + (size_t)m * n_frames + valid, 0, (n_frames - valid) * sizeof(float));
    return AF_OK;
}

int af_gemma4_log_mel(af_gemma4 *h, const float *audio, size_t n_samples, size_t n_frames,
                      float *out) {
    return log_mel(h, audio, n_samples, n_frames, 1, out);
}

int af_gemma4_log_mel_channel_first(af_gemma4 *h, const float *audio, size_t n_samples,
                                    size_t n_frames, float *out) {
    return log_mel(h, audio, n_samples, n_frames, 0, out);
}
