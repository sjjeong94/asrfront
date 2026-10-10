/* Real FFT vs. a double-precision naive DFT. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "asrfront.h"
#include "fft.h"
#include "kernels.h"

#define TWO_PI 6.283185307179586476925286766559

static int failures = 0;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);               \
            failures++;                                                                            \
        }                                                                                          \
    } while (0)

static unsigned long long rng_state = 0x9E3779B97F4A7C15ull;

static float uniform(void) { /* xorshift64*, uniform in [-1, 1) */
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    unsigned long long r = rng_state * 0x2545F4914F6CDD1Dull;
    return (float)((double)(r >> 11) / 9007199254740992.0 * 2.0 - 1.0);
}

static void naive_dft(const float *x, int n, double *re, double *im) {
    for (int k = 0; k <= n / 2; k++) {
        double sr = 0.0, si = 0.0;
        for (int t = 0; t < n; t++) {
            double a = -TWO_PI * (double)(((long)k * t) % n) / n;
            sr += x[t] * cos(a);
            si += x[t] * sin(a);
        }
        re[k] = sr;
        im[k] = si;
    }
}

/* max |X - X_ref| / max |X_ref| for one input; also checks af_rfft_power. */
static double check_input(af_rfft *p, const float *x, int n) {
    const int nb = n / 2 + 1;
    double *re = malloc(nb * sizeof(double)), *im = malloc(nb * sizeof(double));
    af_cpx *X = malloc(nb * sizeof(af_cpx));
    float *pw = malloc(nb * sizeof(float));

    naive_dft(x, n, re, im);
    af_rfft_forward(p, x, X);
    af_rfft_power(p, x, pw);

    double max_ref = 0.0, max_err = 0.0, max_perr = 0.0, max_pref = 0.0;
    for (int k = 0; k < nb; k++) {
        double mag = sqrt(re[k] * re[k] + im[k] * im[k]);
        double err = hypot(X[k].re - re[k], X[k].im - im[k]);
        double pref = re[k] * re[k] + im[k] * im[k];
        max_ref = mag > max_ref ? mag : max_ref;
        max_err = err > max_err ? err : max_err;
        max_pref = pref > max_pref ? pref : max_pref;
        double perr = fabs(pw[k] - pref);
        max_perr = perr > max_perr ? perr : max_perr;
    }
    CHECK(X[0].im == 0.0f && X[n / 2].im == 0.0f);
    double rel = max_ref > 0 ? max_err / max_ref : max_err;
    double prel = max_pref > 0 ? max_perr / max_pref : max_perr;
    CHECK(prel < 2e-6);

    free(re);
    free(im);
    free(X);
    free(pw);
    return rel;
}

static void test_size(int n, double tol) {
    af_rfft *p = NULL;
    if (af_rfft_create(n, &p) != AF_OK || !p) {
        fprintf(stderr, "create failed for n=%d\n", n);
        failures++;
        return;
    }
    CHECK(af_rfft_size(p) == n);
    float *x = malloc(n * sizeof(float));
    double worst = 0.0;

    for (int trial = 0; trial < 5; trial++) { /* random */
        for (int i = 0; i < n; i++)
            x[i] = uniform();
        double r = check_input(p, x, n);
        worst = r > worst ? r : worst;
    }
    for (int i = 0; i < n; i++) /* impulse at 1 */
        x[i] = (i == 1) ? 1.0f : 0.0f;
    double r = check_input(p, x, n);
    worst = r > worst ? r : worst;
    for (int i = 0; i < n; i++) /* constant (DC only) */
        x[i] = 0.25f;
    r = check_input(p, x, n);
    worst = r > worst ? r : worst;
    for (int i = 0; i < n; i++) /* cosine on bin n/2 - 1 plus large DC */
        x[i] = 100.0f + (float)cos(TWO_PI * (n / 2 - 1) * i / n);
    r = check_input(p, x, n);
    worst = r > worst ? r : worst;

    printf("n=%5d  max rel err %.3e\n", n, worst);
    if (!(worst < tol)) {
        fprintf(stderr, "n=%d: rel err %.3e >= %.1e\n", n, worst, tol);
        failures++;
    }
    free(x);
    af_rfft_destroy(p);
}

/* The AF_VL-lane SoA kernel must reproduce the scalar power spectrum bit for bit. */
static void test_soa_kernel(const af_kernels *k, int n) {
    af_rfft *p = NULL;
    if (af_rfft_create(n, &p) != AF_OK) {
        failures++;
        return;
    }
    const int m = n / 2;
    float *x = malloc((size_t)n * AF_VL * sizeof(float));
    float *re = malloc((size_t)m * AF_VL * sizeof(float)),
          *im = malloc((size_t)m * AF_VL * sizeof(float));
    float *sr = malloc((size_t)m * AF_VL * sizeof(float)),
          *si = malloc((size_t)m * AF_VL * sizeof(float));
    float *pw = malloc((size_t)(m + 1) * AF_VL * sizeof(float));
    float *ref = malloc((size_t)(m + 1) * sizeof(float));
    for (int i = 0; i < n * AF_VL; i++)
        x[i] = uniform();
    for (int l = 0; l < AF_VL; l++)
        for (int i = 0; i < m; i++) {
            re[i * AF_VL + l] = x[l * n + 2 * i];
            im[i * AF_VL + l] = x[l * n + 2 * i + 1];
        }
    k->rfft_power(p, re, im, sr, si, pw);
    int mismatches = 0;
    for (int l = 0; l < AF_VL; l++) {
        af_rfft_power(p, x + l * n, ref);
        for (int i = 0; i <= m; i++)
            mismatches += pw[i * AF_VL + l] != ref[i];
    }
    if (mismatches) {
        fprintf(stderr, "%s kernel n=%d: %d bins differ from scalar FFT\n", k->name, n, mismatches);
        failures++;
    }
    free(x), free(re), free(im), free(sr), free(si), free(pw), free(ref);
    af_rfft_destroy(p);
}

static void test_invalid_sizes(void) {
    static const int bad[] = {0, -2, 1, 3, 14, 22, 26, 98};
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        af_rfft *p = (af_rfft *)&failures; /* must be reset to NULL */
        CHECK(af_rfft_create(bad[i], &p) == AF_EINVAL && p == NULL);
    }
    CHECK(af_rfft_create(400, NULL) == AF_EINVAL);
}

int main(void) {
    static const int sizes[] = {2,   4,   6,   8,   10,  12,   16,   18,   20,
                                30,  50,  64,  90,  96,  100,  128,  200,  250,
                                256, 320, 400, 480, 512, 1024, 1000, 1200, 2048};
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++)
        test_size(sizes[i], 1e-6);
    test_invalid_sizes();

    extern const af_kernels af_kernels_generic, af_kernels_scalar;
    const af_kernels *tables[] = {&af_kernels_generic, &af_kernels_scalar, af_kernels_select()};
    for (int t = 0; t < 3; t++)
        for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++)
            if (sizes[i] >= 4) /* the SoA kernel needs at least one butterfly stage */
                test_soa_kernel(tables[t], sizes[i]);
    printf("SoA kernels (%s, %s, %s) match the scalar FFT bit for bit\n", tables[0]->name,
           tables[1]->name, tables[2]->name);
    if (failures) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("all FFT tests passed\n");
    return 0;
}
