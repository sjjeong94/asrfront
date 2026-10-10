/* af_load_*: format detection, decoding and conversion to 16 kHz mono float32. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "asrfront.h"
#include "audio.h"

typedef enum { FMT_UNKNOWN, FMT_WAV, FMT_FLAC, FMT_MP3 } container;

/* Size of an ID3v2 tag at buf (0 if none). */
static size_t id3v2_size(const uint8_t *buf, size_t size) {
    if (size < 10 || memcmp(buf, "ID3", 3) != 0)
        return 0;
    if ((buf[6] | buf[7] | buf[8] | buf[9]) & 0x80)
        return 0; /* sizes are 7-bit "syncsafe" */
    size_t n = ((size_t)buf[6] << 21) | ((size_t)buf[7] << 14) | ((size_t)buf[8] << 7) | buf[9];
    n += 10 + ((buf[5] & 0x10) ? 10 : 0); /* header (+ footer) */
    return n;
}

/* Length of the MPEG audio Layer III frame starting at p, 0 if p is not a valid header.
 * *key receives the fields that must stay constant between frames. */
static size_t mp3_frame_len(const uint8_t *p, uint32_t *key) {
    static const int kbps[2][15] = {
        {0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320}, /* MPEG-1 */
        {0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160},     /* MPEG-2/2.5 */
    };
    static const int rates[3] = {44100, 48000, 32000};
    if (p[0] != 0xFF || (p[1] & 0xE0) != 0xE0)
        return 0;
    int version = (p[1] >> 3) & 3; /* 0: 2.5, 1: reserved, 2: 2, 3: 1 */
    int layer = (p[1] >> 1) & 3;   /* 1: Layer III */
    int br_idx = p[2] >> 4, sr_idx = (p[2] >> 2) & 3, pad = (p[2] >> 1) & 1;
    if (version == 1 || layer != 1 || br_idx == 0 || br_idx == 15 || sr_idx == 3)
        return 0;
    int sr = rates[sr_idx] >> (version == 3 ? 0 : version == 2 ? 1 : 2);
    int kb = kbps[version == 3 ? 0 : 1][br_idx];
    *key = ((uint32_t)p[1] << 8) | (uint32_t)(p[2] & 0x0C); /* version, layer, rate */
    return (size_t)((version == 3 ? 144000 : 72000) * kb / sr + pad);
}

static container detect(const uint8_t *buf, size_t size) {
    if (size >= 12 && memcmp(buf + 8, "WAVE", 4) == 0 &&
        (memcmp(buf, "RIFF", 4) == 0 || memcmp(buf, "RF64", 4) == 0 || memcmp(buf, "BW64", 4) == 0))
        return FMT_WAV;
    size_t off = id3v2_size(buf, size);
    if (off > size)
        return FMT_MP3; /* truncated ID3 tag: let the decoder complain */
    if (size - off >= 4 && memcmp(buf + off, "fLaC", 4) == 0)
        return FMT_FLAC;
    /* MPEG audio: a valid Layer III header followed by another consistent one, so that raw
     * PCM (where e.g. int16 -1 is FF FF) is not mistaken for MP3. */
    if (size - off >= 4) {
        uint32_t k1, k2;
        size_t len = mp3_frame_len(buf + off, &k1);
        if (len >= 4) {
            size_t next = off + len;
            if (next == size && off > 0)
                return FMT_MP3; /* single frame after an ID3 tag */
            if (next <= size - 4 && mp3_frame_len(buf + next, &k2) && k1 == k2)
                return FMT_MP3;
        }
    }
    return FMT_UNKNOWN;
}

int af_load_memory(const void *buf_, size_t size, af_raw_mode raw, float **out, size_t *n_samples) {
    if (!out || !n_samples || (!buf_ && size > 0))
        return AF_EINVAL;
    *out = NULL;
    *n_samples = 0;
    if (raw != AF_RAW_AUTO && raw != AF_RAW_NEVER && raw != AF_RAW_FORCE)
        return AF_EINVAL;
    const uint8_t *buf = buf_ ? buf_ : (const uint8_t *)"";

    af_decoded d;
    int rc;
    container fmt = raw == AF_RAW_FORCE ? FMT_UNKNOWN : detect(buf, size);
    switch (fmt) {
    case FMT_WAV:
        rc = af_wav_decode(buf, size, &d);
        break;
    case FMT_FLAC:
        rc = af_flac_decode(buf, size, &d);
        break;
    case FMT_MP3:
        rc = af_mp3_decode(buf, size, &d);
        break;
    default:
        if (raw == AF_RAW_NEVER)
            return AF_EUNSUPPORTED;
        rc = af_pcm_decode_raw(buf, size, &d);
        break;
    }
    if (rc != AF_OK)
        return rc;

    rc = af_audio_to_mono(&d, AF_SAMPLE_RATE, out, n_samples);
    af_decoded_free(&d);
    return rc;
}

static int has_raw_extension(const char *path) {
    const char *dot = strrchr(path, '.');
    const char *sep = strrchr(path, '/');
#ifdef _WIN32
    const char *bsep = strrchr(path, '\\');
    if (bsep && (!sep || bsep > sep))
        sep = bsep;
#endif
    if (!dot || (sep && dot < sep))
        return 0;
    char ext[5] = {0};
    for (int i = 0; i < 4 && dot[1 + i]; i++)
        ext[i] = (char)(dot[1 + i] | 0x20); /* ASCII lower case */
    return dot[1] && strlen(dot + 1) == 3 && (strcmp(ext, "pcm") == 0 || strcmp(ext, "raw") == 0);
}

static FILE *open_utf8(const char *path) {
#ifdef _WIN32
    int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, NULL, 0);
    if (n <= 0)
        return NULL;
    wchar_t *w = malloc((size_t)n * sizeof(wchar_t));
    if (!w)
        return NULL;
    MultiByteToWideChar(CP_UTF8, 0, path, -1, w, n);
    FILE *f = _wfopen(w, L"rb");
    free(w);
    return f;
#else
    return fopen(path, "rb");
#endif
}

int af_load_file(const char *path_utf8, float **out, size_t *n_samples) {
    if (!path_utf8 || !out || !n_samples)
        return AF_EINVAL;
    *out = NULL;
    *n_samples = 0;
    FILE *f = open_utf8(path_utf8);
    if (!f)
        return AF_EIO;
    uint8_t *buf = NULL;
    size_t size = 0, cap = 0;
    int rc = AF_OK;
    for (;;) {
        if (size == cap) {
            size_t ncap = cap ? cap * 2 : (size_t)1 << 20;
            uint8_t *nb = ncap > cap ? realloc(buf, ncap) : NULL;
            if (!nb) {
                rc = AF_ENOMEM;
                break;
            }
            buf = nb;
            cap = ncap;
        }
        size_t got = fread(buf + size, 1, cap - size, f);
        size += got;
        if (got == 0) {
            if (ferror(f))
                rc = AF_EIO;
            break;
        }
    }
    fclose(f);
    if (rc == AF_OK)
        rc = af_load_memory(buf, size, has_raw_extension(path_utf8) ? AF_RAW_FORCE : AF_RAW_NEVER,
                            out, n_samples);
    free(buf);
    return rc;
}

void af_audio_free(float *p) {
    free(p);
}
