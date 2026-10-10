#include "kernels.h"

#include <stdlib.h>
#include <string.h>

extern const af_kernels af_kernels_generic;
extern const af_kernels af_kernels_scalar;
#ifdef AF_HAVE_AVX2
extern const af_kernels af_kernels_avx2;
#endif

#ifdef AF_HAVE_AVX2
#if defined(_MSC_VER)
#include <immintrin.h>
#include <intrin.h>
static int cpu_has_avx2(void) {
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7)
        return 0;
    __cpuid(r, 1);
    const int osxsave = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1;
    if (!osxsave || !avx || (_xgetbv(0) & 6) != 6) /* OS saves XMM and YMM state */
        return 0;
    const int fma = (r[2] >> 12) & 1;
    __cpuidex(r, 7, 0);
    return fma && ((r[1] >> 5) & 1);
}
#else
static int cpu_has_avx2(void) { /* AVX2 and FMA3: the avx2 table is built with -mfma */
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
#endif
#endif

const af_kernels *af_kernels_select(void) {
    const char *force = getenv("ASRFRONT_KERNELS");
    if (force && strcmp(force, "generic") == 0)
        return &af_kernels_generic;
    if (force && strcmp(force, "scalar") == 0)
        return &af_kernels_scalar;
#ifdef AF_HAVE_AVX2
    if (cpu_has_avx2())
        return &af_kernels_avx2;
#endif
    return &af_kernels_generic;
}
