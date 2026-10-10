/* Real FFT throughput: ns per transform for the frontend sizes. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "asrfront.h"
#include "fft.h"

static double now_ns(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static void bench(int n, int iters) {
    af_rfft *p = NULL;
    if (af_rfft_create(n, &p) != AF_OK) {
        fprintf(stderr, "create failed for n=%d\n", n);
        exit(1);
    }
    float *x = malloc(n * sizeof(float));
    float *pw = malloc((n / 2 + 1) * sizeof(float));
    for (int i = 0; i < n; i++)
        x[i] = (float)((i * 7919) % 1000) / 1000.0f - 0.5f;

    volatile float sink = 0.0f;
    for (int i = 0; i < iters / 10; i++) { /* warm-up */
        af_rfft_power(p, x, pw);
        sink += pw[1];
    }
    double best = 1e30;
    for (int rep = 0; rep < 5; rep++) {
        double t0 = now_ns();
        for (int i = 0; i < iters; i++) {
            x[0] = (float)i; /* defeat hoisting */
            af_rfft_power(p, x, pw);
            sink += pw[1];
        }
        double dt = (now_ns() - t0) / iters;
        best = dt < best ? dt : best;
    }
    printf("rfft+power n=%4d: %8.1f ns/frame  (%6.3f ms per 3000 frames = 30 s)\n", n, best,
           best * 3000 / 1e6);
    (void)sink;
    free(x);
    free(pw);
    af_rfft_destroy(p);
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 200000;
    bench(400, iters);
    bench(512, iters);
    return 0;
}
