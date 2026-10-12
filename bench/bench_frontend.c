/* Frontend throughput (30 s input) plus a per-stage breakdown of the 8-frame kernels. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "asrfront.h"
#include "fft.h"
#include "kernels.h"
#include "melbank.h"

#define N_SAMPLES 480000
#define REPS 50

static double now_ns(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

#define BEST_MS(expr)                                                                              \
    ({                                                                                             \
        double best_ = 1e30;                                                                       \
        for (int r_ = 0; r_ < REPS; r_++) {                                                        \
            double t0_ = now_ns();                                                                 \
            expr;                                                                                  \
            double dt_ = now_ns() - t0_;                                                           \
            best_ = dt_ < best_ ? dt_ : best_;                                                     \
        }                                                                                          \
        best_ / 1e6;                                                                               \
    })

int main(void) {
    float *audio = malloc(N_SAMPLES * sizeof(float));
    for (int i = 0; i < N_SAMPLES; i++)
        audio[i] = 0.3f * sinf(0.01f * i) +
                   0.01f * (float)((int)(((long long)i * 7919) % 1000) - 500) / 500.0f;
    printf("kernels: %s\n", af_kernel_isa());

    af_whisper_config wc;
    af_whisper_default_config(&wc);
    af_whisper *w;
    af_whisper_create(&wc, &w);
    float *wout = malloc(80 * 3000 * sizeof(float));
    printf("whisper log_mel 30 s:        %6.3f ms\n",
           BEST_MS(af_whisper_log_mel(w, audio, N_SAMPLES, wout)));

    af_sensevoice_config sc;
    af_sensevoice_default_config(&sc);
    af_sensevoice *s;
    af_sensevoice_create(&sc, &s);
    float *sout = malloc(af_sensevoice_out_frames(s, N_SAMPLES) * 560 * sizeof(float));
    printf("sensevoice fbank 30 s:       %6.3f ms\n",
           BEST_MS(af_sensevoice_fbank(s, audio, N_SAMPLES, sout)));

    float shift[560], scale[560];
    for (int i = 0; i < 560; i++) {
        shift[i] = -10.0f;
        scale[i] = 0.3f;
    }
    sc.cmvn_shift = shift;
    sc.cmvn_scale = scale;
    af_sensevoice *s_cmvn;
    af_sensevoice_create(&sc, &s_cmvn);
    printf("sensevoice fbank + CMVN:     %6.3f ms\n",
           BEST_MS(af_sensevoice_fbank(s_cmvn, audio, N_SAMPLES, sout)));
    af_sensevoice_destroy(s_cmvn);

    /* Stage breakdown for 3000 frames = 375 blocks of 8 */
    const af_kernels *k = af_kernels_select();
    const int blocks = 3000 / AF_VL;
    for (int n = 400; n <= 512; n += 112) {
        af_rfft *p;
        af_rfft_create(n, &p);
        af_melbank *mb;
        if (n == 400)
            af_melbank_slaney(16000, 400, 80, 0.0, 8000.0, &mb);
        else
            af_melbank_kaldi(16000, 512, 80, 20.0, 0.0, &mb);
        const int m = n / 2;
        float *re = malloc(m * AF_VL * sizeof(float)), *im = malloc(m * AF_VL * sizeof(float));
        float *sr = malloc(m * AF_VL * sizeof(float)), *si = malloc(m * AF_VL * sizeof(float));
        float *pw = malloc((m + 1) * AF_VL * sizeof(float)),
              *mel = malloc(80 * AF_VL * sizeof(float));
        for (int i = 0; i < m * AF_VL; i++)
            re[i] = im[i] = audio[i];
        double t_fft = BEST_MS(for (int b = 0; b < blocks; b++) {
            re[0] = (float)b;
            k->rfft_power(p, re, im, sr, si, pw);
        });
        double t_mel = BEST_MS(for (int b = 0; b < blocks; b++) k->mel(mb, pw, mel));
        double t_log = BEST_MS(for (int b = 0; b < blocks; b++) {
                           k->mel(mb, pw, mel);
                           k->log10_floor(mel, 80 * AF_VL, 1e-10f);
                       }) -
                       t_mel;
        printf("n=%d per 3000 frames: rfft+power %.3f ms, mel %.3f ms, log %.3f ms\n", n, t_fft,
               t_mel, t_log);
        free(re), free(im), free(sr), free(si), free(pw), free(mel);
        af_melbank_destroy(mb);
        af_rfft_destroy(p);
    }
    return 0;
}
