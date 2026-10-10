/*
 * Internal mel filter banks, stored sparsely: each filter keeps only its contiguous
 * non-zero range of FFT bins.
 */
#ifndef ASRFRONT_MELBANK_H
#define ASRFRONT_MELBANK_H

/* Layout shared with the SIMD kernels (kernels_impl.c). */
typedef struct af_melbank {
    int n_mels;
    int n_bins;
    int *start;  /* n_mels: first non-zero bin */
    int *len;    /* n_mels: number of bins */
    int *offset; /* n_mels: offset into w */
    float *w;    /* concatenated weights */
} af_melbank;

/* librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax) with htk=False, norm='slaney',
 * reproducing librosa's float32 rounding (as in whisper's mel_filters.npz). */
int af_melbank_slaney(int sr, int n_fft, int n_mels, double fmin, double fmax, af_melbank **out);

/* torchaudio.compliance.kaldi.get_mel_banks (no VTLN) for an FFT of size `padded`:
 * Kaldi mel scale 1127 ln(1 + f/700), triangles between low_freq and high_freq
 * (high_freq <= 0 means nyquist + high_freq), no normalization, zero Nyquist bin.
 * Computed in double (torchaudio uses float32). */
int af_melbank_kaldi(int sr, int padded, int n_mels, double low_freq, double high_freq,
                     af_melbank **out);

void af_melbank_destroy(af_melbank *mb);

int af_melbank_n_mels(const af_melbank *mb);
int af_melbank_n_bins(const af_melbank *mb);

/* mel[m] = sum_k w[m][k] * power[k]. power: n_bins, mel: n_mels. */
void af_melbank_apply(const af_melbank *mb, const float *power, float *mel);

/* Dense (n_mels, n_bins) copy of the weights, for tests. */
void af_melbank_to_dense(const af_melbank *mb, float *out);

#endif /* ASRFRONT_MELBANK_H */
