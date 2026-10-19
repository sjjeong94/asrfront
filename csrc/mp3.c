/* MP3 decoding via the vendored dr_mp3 (csrc/third_party). Encoder delay and padding from a
 * LAME/Xing header are removed by dr_mp3, as ffmpeg does. */
#include <stdlib.h>
#include <string.h>

#include "asrfront.h"
#include "audio.h"
#include "dr_mp3.h"

int af_mp3_decode(const uint8_t *buf, size_t size, size_t limit, af_decoded *out) {
    memset(out, 0, sizeof(*out));
    out->limit = limit;
    drmp3 mp3;
    if (!drmp3_init_memory(&mp3, buf, size, NULL))
        return AF_ECORRUPT;
    const int ch = (int)mp3.channels;
    if (ch < 1 || ch > 2 || mp3.sampleRate == 0) {
        drmp3_uninit(&mp3);
        return AF_ECORRUPT;
    }
    out->channels = ch;
    out->sample_rate = (int)mp3.sampleRate;
    /* Frames past the caller's limit are an error: decoding one more than the limit tells. */
    const size_t lim = af_frame_limit(out), lim_cap = lim < SIZE_MAX ? lim + 1 : SIZE_MAX;
    size_t cap = 65536 < lim_cap ? 65536 : lim_cap, n = 0;
    float *pcm = malloc(cap * (size_t)ch * sizeof(float));
    while (pcm && n <= lim) {
        if (n == cap) {
            size_t ncap = cap * 2 < lim_cap ? cap * 2 : lim_cap;
            float *p = ncap > cap ? realloc(pcm, ncap * (size_t)ch * sizeof(float)) : NULL;
            if (!p) {
                free(pcm);
                pcm = NULL;
                break;
            }
            pcm = p;
            cap = ncap;
        }
        drmp3_uint64 got = drmp3_read_pcm_frames_f32(&mp3, cap - n, pcm + n * (size_t)ch);
        if (got == 0)
            break;
        n += (size_t)got;
    }
    out->fmt = AF_SFMT_F32;
    out->n_frames = n;
    af_default_layout(ch, out->layout);
    out->data = pcm;
    drmp3_uninit(&mp3);
    if (!pcm)
        return AF_ENOMEM;
    if (n > lim) {
        af_decoded_free(out);
        return AF_ETOOLONG;
    }
    if (n == 0) {
        af_decoded_free(out);
        return AF_ECORRUPT; /* no decodable frame */
    }
    return AF_OK;
}
