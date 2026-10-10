/*
 * Internal real FFT (not part of the public API).
 *
 * A real FFT of even size n is computed as a complex FFT of size n/2 followed by a
 * split post-processing step. The complex FFT is an iterative Stockham autosort FFT
 * with radix-2/3/4/5 butterflies, so n/2 must factor into 2, 3 and 5
 * (whisper: 400 -> 200 = 4*2*5*5, kaldi: 512 -> 256 = 4^4).
 *
 * Twiddles are generated in double precision and stored as float; all arithmetic is
 * float32. A plan owns its scratch buffers and is therefore not thread-safe.
 */
#ifndef ASRFRONT_FFT_H
#define ASRFRONT_FFT_H

typedef struct af_cpx {
    float re, im;
} af_cpx;

typedef struct af_rfft af_rfft;

/* AF_EINVAL if n is not even/positive or n/2 has a prime factor other than 2, 3, 5. */
int af_rfft_create(int n, af_rfft **out);
void af_rfft_destroy(af_rfft *p);
int af_rfft_size(const af_rfft *p);

/* in: n real samples. out: n/2 + 1 complex bins (X[0] and X[n/2] have im == 0). */
void af_rfft_forward(af_rfft *p, const float *in, af_cpx *out);

/* in: n real samples. out: n/2 + 1 power values |X[k]|^2. */
void af_rfft_power(af_rfft *p, const float *in, float *out);

#endif /* ASRFRONT_FFT_H */
