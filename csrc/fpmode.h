/* Internal: scoped flush-to-zero for the frontends (not part of the public API). */
#ifndef ASRFRONT_FPMODE_H
#define ASRFRONT_FPMODE_H

/* Flush-to-zero and denormals-are-zero for the duration of a frontend call (x86 SSE; elsewhere
 * a no-op). Tiny inputs (|x| ≲ 1e-17) make FFT / mel intermediates subnormal, which x86 handles
 * in microcode, up to 16x slower. Values that small are far below every log floor (whisper /
 * LFM3 1e-10 / 1e-12, kaldi FLT_EPSILON, Gemma 4's + 1e-3), so the outputs are bit for bit the
 * same either way. Not for the resampler, which must match ffmpeg (no FTZ). Usage:
 * `unsigned m = af_ftz_enter(); ...; af_ftz_leave(m);` */
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#include <xmmintrin.h>
#if defined(__x86_64__) || defined(_M_X64)
#define AF_FTZ_BITS 0x8040u /* FTZ | DAZ (every x86-64 CPU has DAZ) */
#else
#define AF_FTZ_BITS 0x8000u /* FTZ only: DAZ is missing on early SSE CPUs */
#endif
static inline unsigned af_ftz_enter(void) {
    const unsigned csr = _mm_getcsr();
    if ((csr & AF_FTZ_BITS) != AF_FTZ_BITS)
        _mm_setcsr(csr | AF_FTZ_BITS);
    return csr;
}
static inline void af_ftz_leave(unsigned csr) {
    if ((csr & AF_FTZ_BITS) != AF_FTZ_BITS)
        _mm_setcsr(csr);
}
#else
static inline unsigned af_ftz_enter(void) {
    return 0;
}
static inline void af_ftz_leave(unsigned csr) {
    (void)csr;
}
#endif

#endif /* ASRFRONT_FPMODE_H */
