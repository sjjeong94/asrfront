/* Fuzz target for af_load_memory (libFuzzer entry point; tests/fuzz/README.md).
 *
 * Besides memory errors (ASan / UBSan), it checks the output contract: every sample is an
 * int16 value / 32768, so it lies in [-1, 32767/32768] and is an exact multiple of 2^-15.
 * A decoded file is loaded again with af_load_memory_ex at its own length (same output) and
 * one sample less (AF_ETOOLONG). */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "asrfront.h"

static void check(int ok) {
    if (!ok)
        abort();
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    /* AF_RAW_NEVER exercises every container parser; AF_RAW_AUTO adds the raw PCM fallback,
     * which matters only for the detection logic, so it runs on a cheap subset. */
    static const af_raw_mode modes[2] = {AF_RAW_NEVER, AF_RAW_AUTO};
    for (int m = 0; m < (size < 64 ? 2 : 1); m++) {
        float *out = (float *)&out; /* must be reset by the library */
        size_t n = (size_t)-1;
        int rc = af_load_memory(data, size, modes[m], &out, &n);
        if (rc != AF_OK) {
            check(out == NULL && n == 0);
            check(rc == AF_ECORRUPT || rc == AF_EUNSUPPORTED || rc == AF_ENOMEM);
            continue;
        }
        check(out != NULL);
        for (size_t i = 0; i < n; i++) {
            float v = out[i] * 32768.0f;
            check(v >= -32768.0f && v <= 32767.0f && v == (float)(int32_t)v);
        }
        if (m == 0 && n > 1) {
            float *lim = NULL;
            size_t n_lim = 0;
            rc = af_load_memory_ex(data, size, modes[m], n, &lim, &n_lim);
            check(rc == AF_ENOMEM || (rc == AF_OK && n_lim == n && memcmp(lim, out, n * 4) == 0));
            af_audio_free(lim);
            rc = af_load_memory_ex(data, size, modes[m], n - 1, &lim, &n_lim);
            check((rc == AF_ETOOLONG || rc == AF_ENOMEM) && lim == NULL && n_lim == 0);
        }
        af_audio_free(out);
    }
    return 0;
}
