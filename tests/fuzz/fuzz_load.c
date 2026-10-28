/* Fuzz target for af_load_memory (libFuzzer entry point; tests/fuzz/README.md).
 *
 * Besides memory errors (ASan / UBSan), it checks the output contract: every sample is an
 * int16 value / 32768, so it lies in [-1, 32767/32768] and is an exact multiple of 2^-15.
 * A decoded file is loaded again with af_load_memory_ex at its own length (same output) and
 * one sample less (AF_ETOOLONG). The error description is printable ASCII, set exactly for
 * AF_EUNSUPPORTED / AF_ECORRUPT, and af_probe_memory agrees with the load. Prefixes of a decoded
 * file are never judged too long for its own length (af_prefix_too_long). */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "asrfront.h"
#include "audio.h" /* internal: af_prefix_too_long */

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
        char err[128];
        memset(err, 'x', sizeof err);
        int rc = af_load_memory_ex(data, size, modes[m], 0, &out, &n, err, sizeof err);
        const char *nul = memchr(err, 0, sizeof err);
        check(nul != NULL);
        const size_t err_len = (size_t)(nul - err);
        for (size_t i = 0; i < err_len; i++)
            check(err[i] >= 0x20 && err[i] <= 0x7E);
        check((err_len > 0) == (rc == AF_ECORRUPT || rc == AF_EUNSUPPORTED));

        /* af_probe_memory agrees with the load: same output length, same error and message
         * (it skips ADPCM block headers, so where they are corrupt it may accept the file or
         * reject its sample rate: AF_ECORRUPT -> OK / AF_EUNSUPPORTED) */
        af_audio_info info;
        char perr[128];
        int prc = af_probe_memory(data, size, modes[m], 0, &info, perr, sizeof perr);
        if (rc == AF_OK) {
            check(prc == AF_OK && info.n_samples == n && info.container != NULL);
            if (n > 1) /* max_samples 0 means no limit */
                check(af_probe_memory(data, size, modes[m], n - 1, &info, NULL, 0) == AF_ETOOLONG);
        } else if (rc == AF_EUNSUPPORTED || (rc == AF_ECORRUPT && prc == AF_ECORRUPT)) {
            check(prc == rc && strcmp(perr, err) == 0);
        } else if (rc == AF_ECORRUPT) {
            check(prc == AF_OK || prc == AF_EUNSUPPORTED);
        }

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
            rc = af_load_memory_ex(data, size, modes[m], n, &lim, &n_lim, NULL, 0);
            check(rc == AF_ENOMEM || (rc == AF_OK && n_lim == n && memcmp(lim, out, n * 4) == 0));
            af_audio_free(lim);
            rc = af_load_memory_ex(data, size, modes[m], n - 1, &lim, &n_lim, NULL, 0);
            check((rc == AF_ETOOLONG || rc == AF_ENOMEM) && lim == NULL && n_lim == 0);
        }
        /* af_prefix_too_long never rejects a prefix of a file that loads within the limit, and
         * a prefix it rejects fails to load as it says (af_load_file_ex relies on both) */
        if (m == 0 && n > 0) {
            for (size_t p = size / 2; p < size; p += (size - p + 1) / 2) {
                check(af_prefix_too_long(data, p, AF_RAW_NEVER, n) != 1);
                if (af_prefix_too_long(data, p, AF_RAW_NEVER, 1) == 1) {
                    float *pre = NULL;
                    size_t n_pre = 0;
                    rc = af_load_memory_ex(data, p, AF_RAW_NEVER, 1, &pre, &n_pre, NULL, 0);
                    check(rc == AF_ETOOLONG && pre == NULL);
                }
            }
        }
        af_audio_free(out);
    }
    return 0;
}
