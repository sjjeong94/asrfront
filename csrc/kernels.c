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
/* cpuid directly instead of __builtin_cpu_supports, which needs compiler-rt's __cpu_model
 * (not linked by every toolchain, e.g. for macOS). */
#include <cpuid.h>
static int cpu_has_avx2(void) { /* AVX2 and FMA3: the avx2 table is built with -mfma */
    unsigned a, b, c, d, xlo, xhi;
    if (!__get_cpuid(1, &a, &b, &c, &d))
        return 0;
    const unsigned osxsave = (c >> 27) & 1, avx = (c >> 28) & 1, fma = (c >> 12) & 1;
    if (!osxsave || !avx || !fma)
        return 0;
    __asm__ volatile("xgetbv" : "=a"(xlo), "=d"(xhi) : "c"(0));
    (void)xhi;
    if ((xlo & 6) != 6) /* OS saves XMM and YMM state */
        return 0;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d))
        return 0;
    return (b >> 5) & 1;
}
#endif
#endif

#ifdef AF_HAVE_AVX2
/* cpuid traps to the hypervisor in many VMs (microseconds), and every handle and resampler
 * selects its kernels: query the CPU once. */
static int have_avx2;
#ifdef _WIN32
#include <windows.h>
static INIT_ONCE avx2_once = INIT_ONCE_STATIC_INIT;
static BOOL CALLBACK detect_avx2(PINIT_ONCE once, PVOID param, PVOID *ctx) {
    (void)once, (void)param, (void)ctx;
    have_avx2 = cpu_has_avx2();
    return TRUE;
}
static int avx2_available(void) {
    InitOnceExecuteOnce(&avx2_once, detect_avx2, NULL, NULL);
    return have_avx2;
}
#else
#include <pthread.h>
static pthread_once_t avx2_once = PTHREAD_ONCE_INIT;
static void detect_avx2(void) {
    have_avx2 = cpu_has_avx2();
}
static int avx2_available(void) {
    pthread_once(&avx2_once, detect_avx2);
    return have_avx2;
}
#endif
#endif

const af_kernels *af_kernels_select(void) {
    /* read on every call: tests switch variants at run time */
    const char *force = getenv("ASRFRONT_KERNELS");
    if (force && strcmp(force, "generic") == 0)
        return &af_kernels_generic;
    if (force && strcmp(force, "scalar") == 0)
        return &af_kernels_scalar;
#ifdef AF_HAVE_AVX2
    if (avx2_available())
        return &af_kernels_avx2;
#endif
    return &af_kernels_generic;
}
