/* af_load_*: format detection, decoding and conversion to 16 kHz mono float32. */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L /* fileno */
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "asrfront.h"
#include "audio.h"
#include "resample.h"

typedef enum {
    FMT_UNKNOWN,
    FMT_WAV,
    FMT_W64,
    FMT_AIFF,
    FMT_CAF,
    FMT_SPHERE,
    FMT_FLAC,
    FMT_MP3,
} container;

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

/* MPEG audio after leading junk (zero padding, garbage between an ID3 tag and the first frame;
 * ffmpeg resyncs past it) is searched for this far. */
#define MP3_SCAN_BYTES 65536

/* Offset of the first of three consecutive consistent Layer III frames starting in
 * [from, from + MP3_SCAN_BYTES), or of two that end exactly at the end of the file; SIZE_MAX
 * if there is none. One more frame than at the start of the file, since a match anywhere in
 * the window is accepted and raw PCM must still never be taken for MP3. */
static size_t mp3_find(const uint8_t *buf, size_t size, size_t from) {
    const size_t end = size - from > MP3_SCAN_BYTES ? from + MP3_SCAN_BYTES : size;
    for (size_t pos = from; pos < end && size - pos >= 4; pos++) {
        const uint8_t *ff = memchr(buf + pos, 0xFF, end - pos);
        if (!ff)
            break;
        pos = (size_t)(ff - buf);
        if (size - pos < 4)
            break;
        uint32_t k1, k;
        size_t len = mp3_frame_len(buf + pos, &k1), next = pos + len;
        int n = 1;
        while (len >= 4 && n < 3 && next <= size - 4 && (len = mp3_frame_len(buf + next, &k)) &&
               k == k1) {
            next += len;
            n++;
        }
        if (n == 3 || (n == 2 && next == size))
            return pos;
    }
    return SIZE_MAX;
}

/* *mp3_off: where the MPEG audio decoder starts (past leading junk, see mp3_find). */
static container detect(const uint8_t *buf, size_t size, size_t *mp3_off) {
    *mp3_off = 0;
    if (size >= 12 && memcmp(buf + 8, "WAVE", 4) == 0 &&
        (memcmp(buf, "RIFF", 4) == 0 || memcmp(buf, "RF64", 4) == 0 || memcmp(buf, "BW64", 4) == 0))
        return FMT_WAV;
    if (size >= 16 && memcmp(buf, af_w64_riff_guid, 16) == 0)
        return FMT_W64;
    if (size >= 12 && memcmp(buf, "FORM", 4) == 0 &&
        (memcmp(buf + 8, "AIFF", 4) == 0 || memcmp(buf + 8, "AIFC", 4) == 0))
        return FMT_AIFF;
    if (size >= 4 && memcmp(buf, "caff", 4) == 0)
        return FMT_CAF;
    if (size >= 8 && memcmp(buf, "NIST_1A\n", 8) == 0)
        return FMT_SPHERE;
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
    const size_t at = mp3_find(buf, size, off);
    if (at != SIZE_MAX) {
        *mp3_off = at;
        return FMT_MP3;
    }
    return FMT_UNKNOWN;
}

int af_load_memory(const void *buf, size_t size, af_raw_mode raw, float **out, size_t *n_samples) {
    return af_load_memory_ex(buf, size, raw, 0, out, n_samples, NULL, 0);
}

static const char *const container_names[] = {
    "raw PCM", "WAV", "Wave64", "AIFF", "CAF", "NIST SPHERE", "FLAC", "MP3",
};

/* AAC in ADTS framing: a frame header whose length leads to a second one, so that raw PCM
 * starting with the bytes FF F0 / FF F1 / FF F8 / FF F9 is not taken for AAC. */
static int adts_frames(const uint8_t *buf, size_t size) {
    if (size < 7 || buf[0] != 0xFF || (buf[1] & 0xF6) != 0xF0)
        return 0;
    size_t len = ((size_t)(buf[3] & 0x03) << 11) | ((size_t)buf[4] << 3) | (buf[5] >> 5);
    return len >= 7 && len <= size - 2 && buf[len] == 0xFF && (buf[len + 1] & 0xF6) == 0xF0;
}

/* Common containers that are not decoded, named in the error instead of "no known header".
 * Also checked for bytes input (AF_RAW_AUTO, `strict`) before falling back to raw PCM, so only
 * signatures that headerless s16le audio practically never starts with; there an ADTS sync
 * word must be confirmed by a second frame. */
static const char *unsupported_container(const uint8_t *buf, size_t size, int strict) {
    if (size >= 8 && memcmp(buf + 4, "ftyp", 4) == 0)
        return "MP4 / M4A";
    if (size >= 4 && memcmp(buf, "OggS", 4) == 0)
        return "Ogg (Vorbis / Opus)";
    if (size >= 4 && memcmp(buf, "\x1A\x45\xDF\xA3", 4) == 0)
        return "Matroska / WebM";
    if (size >= 4 && memcmp(buf, "\x30\x26\xB2\x75", 4) == 0)
        return "ASF / WMA";
    if (size >= 4 && memcmp(buf, "RIFX", 4) == 0)
        return "big-endian RIFX WAV";
    if (size >= 12 && memcmp(buf, "RIFF", 4) == 0)
        return memcmp(buf + 8, "AVI ", 4) == 0 ? "AVI" : "RIFF (not WAVE)";
    if (size >= 6 && memcmp(buf, "#!AMR", 5) == 0)
        return "AMR";
    if (strict ? adts_frames(buf, size) : size >= 2 && buf[0] == 0xFF && (buf[1] & 0xF6) == 0xF0)
        return "AAC (ADTS)";
    return NULL;
}

/* "<container>: <why>" for a failed load into err (container alone if the reason is unknown). */
static void describe(int rc, container fmt, const uint8_t *buf, size_t size, int raw_never,
                     const af_decoded *d, char *err, size_t err_size) {
    if (!err || err_size == 0)
        return;
    const char *name = container_names[fmt];
    if (raw_never) {
        const char *other = unsupported_container(buf, size, 0);
        if (other)
            snprintf(err, err_size, "%s container", other);
        else
            snprintf(err, err_size, "no known container header");
    } else if (d->detail[0]) {
        snprintf(err, err_size, "%s: %s", name, d->detail);
    } else if (rc == AF_EUNSUPPORTED && d->channels > 8) { /* no layout for the channel count */
        snprintf(err, err_size, "%s: %d channels without a channel mask (at most 8)", name,
                 d->channels);
    } else if (rc == AF_EUNSUPPORTED && d->sample_rate > 0 &&
               (d->sample_rate < AF_RESAMPLE_MIN_RATE || d->sample_rate > AF_RESAMPLE_MAX_RATE)) {
        snprintf(err, err_size, "%s: sample rate %d Hz (%d to %d supported)", name, d->sample_rate,
                 AF_RESAMPLE_MIN_RATE, AF_RESAMPLE_MAX_RATE);
    } else {
        snprintf(err, err_size, "%s", name);
    }
}

/* Detect the container and decode it into *d (d->probe: only its frame count). Returns AF_OK
 * (AF_PROBED when probing) or an error; *fmt receives the container. On an unsupported
 * container err is already filled in (*described). */
static int decode(const uint8_t *buf, size_t size, af_raw_mode raw, size_t max_samples,
                  af_decoded *d, container *fmt_out, int *described, char *err, size_t err_size) {
    int rc;
    size_t mp3_off = 0;
    container fmt = raw == AF_RAW_FORCE ? FMT_UNKNOWN : detect(buf, size, &mp3_off);
    *fmt_out = fmt;
    *described = 0;
    switch (fmt) {
    case FMT_WAV:
        rc = af_wav_decode(buf, size, max_samples, d);
        break;
    case FMT_W64:
        rc = af_w64_decode(buf, size, max_samples, d);
        break;
    case FMT_AIFF:
        rc = af_aiff_decode(buf, size, max_samples, d);
        break;
    case FMT_CAF:
        rc = af_caf_decode(buf, size, max_samples, d);
        break;
    case FMT_SPHERE:
        rc = af_sphere_decode(buf, size, max_samples, d);
        break;
    case FMT_FLAC:
        rc = af_flac_decode(buf, size, max_samples, d);
        break;
    case FMT_MP3:
        rc = af_mp3_decode(buf + mp3_off, size - mp3_off, max_samples, d);
        break;
    default:
        /* bytes of a known but unsupported container would decode as raw PCM noise */
        if (raw == AF_RAW_NEVER || (raw == AF_RAW_AUTO && unsupported_container(buf, size, 1))) {
            describe(AF_EUNSUPPORTED, fmt, buf, size, 1, d, err, err_size);
            *described = 1;
            return AF_EUNSUPPORTED;
        }
        rc = af_pcm_decode_raw(buf, size, max_samples, d);
        break;
    }
    return rc;
}

/* The argument checks shared by af_load_memory_ex and af_probe_memory. */
static int check_args(const void *buf, size_t size, af_raw_mode raw, char *err, size_t err_size) {
    if (err && err_size > 0)
        err[0] = 0;
    if (!buf && size > 0)
        return AF_EINVAL;
    if (raw != AF_RAW_AUTO && raw != AF_RAW_NEVER && raw != AF_RAW_FORCE)
        return AF_EINVAL;
    return AF_OK;
}

int af_load_memory_ex(const void *buf_, size_t size, af_raw_mode raw, size_t max_samples,
                      float **out, size_t *n_samples, char *err, size_t err_size) {
    if (err && err_size > 0)
        err[0] = 0;
    if (!out || !n_samples)
        return AF_EINVAL;
    *out = NULL;
    *n_samples = 0;
    int rc = check_args(buf_, size, raw, err, err_size);
    if (rc != AF_OK)
        return rc;
    const uint8_t *buf = buf_ ? buf_ : (const uint8_t *)"";

    af_decoded d;
    memset(&d, 0, sizeof d);
    container fmt;
    int described;
    rc = decode(buf, size, raw, max_samples, &d, &fmt, &described, err, err_size);
    if (described)
        return rc;
    if (rc == AF_OK) {
        rc = af_audio_to_mono(&d, AF_SAMPLE_RATE, out, n_samples);
        af_decoded_free(&d);
    }
    if (rc == AF_EUNSUPPORTED || rc == AF_ECORRUPT)
        describe(rc, fmt, buf, size, 0, &d, err, err_size);
    return rc;
}

int af_probe_memory(const void *buf_, size_t size, af_raw_mode raw, size_t max_samples,
                    af_audio_info *info, char *err, size_t err_size) {
    if (err && err_size > 0)
        err[0] = 0;
    if (!info)
        return AF_EINVAL;
    memset(info, 0, sizeof *info);
    int rc = check_args(buf_, size, raw, err, err_size);
    if (rc != AF_OK)
        return rc;
    const uint8_t *buf = buf_ ? buf_ : (const uint8_t *)"";

    af_decoded d;
    memset(&d, 0, sizeof d);
    d.probe = 1;
    container fmt;
    int described;
    rc = decode(buf, size, raw, max_samples, &d, &fmt, &described, err, err_size);
    if (described)
        return rc;
    af_decoded_free(&d); /* nothing to free in probe mode, unless a decoder ignored it */
    if (rc == AF_PROBED || rc == AF_OK)
        rc = d.sample_rate == AF_SAMPLE_RATE ? AF_OK
                                             : af_resample_check(d.sample_rate, AF_SAMPLE_RATE);
    if (rc == AF_OK) {
        info->container = container_names[fmt];
        info->sample_rate = d.sample_rate;
        info->channels = d.channels;
        info->n_frames = d.n_frames;
        info->n_samples = af_resample_len(d.sample_rate, AF_SAMPLE_RATE, d.n_frames);
    } else if (rc == AF_EUNSUPPORTED || rc == AF_ECORRUPT) {
        describe(rc, fmt, buf, size, 0, &d, err, err_size);
    }
    return rc;
}

int af_prefix_too_long(const uint8_t *buf, size_t size, af_raw_mode raw, size_t max_samples) {
    if (max_samples == 0 || (!buf && size > 0))
        return -1;
    if (!buf)
        buf = (const uint8_t *)"";
    /* Only containers whose frame count cannot shrink as the file goes on: the samples run from
     * one header (all before them) to min(their stated size, the end of the input). A WAV ends
     * its parsing at the first data chunk; AIFF and CAF read chunks after the samples, where a
     * later sound / data or COMM chunk would replace what the prefix shows. Raw PCM only when
     * forced: with AF_RAW_AUTO, MP3 found further on would change the container. */
    size_t mp3_off;
    const container fmt = raw == AF_RAW_FORCE ? FMT_UNKNOWN : detect(buf, size, &mp3_off);
    if (!(fmt == FMT_WAV || fmt == FMT_W64 || fmt == FMT_SPHERE ||
          (fmt == FMT_UNKNOWN && raw == AF_RAW_FORCE)))
        return -1;
    af_decoded d;
    memset(&d, 0, sizeof d);
    d.probe = 1;
    container fmt_out;
    int described;
    const int rc = decode(buf, size, raw, max_samples, &d, &fmt_out, &described, NULL, 0);
    af_decoded_free(&d);
    return rc == AF_ETOOLONG ? 1 : 0;
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

/* Size of a regular file, 0 if unknown (not a regular file, or too large for a size_t). */
static size_t regular_file_size(FILE *f) {
#ifdef _WIN32
    struct _stat64 st;
    if (_fstat64(_fileno(f), &st) != 0 || !(st.st_mode & _S_IFREG))
        return 0;
#else
    struct stat st;
    if (fstat(fileno(f), &st) != 0 || !S_ISREG(st.st_mode))
        return 0;
#endif
    return st.st_size > 0 && (unsigned long long)st.st_size < SIZE_MAX ? (size_t)st.st_size : 0;
}

int af_load_file(const char *path_utf8, float **out, size_t *n_samples) {
    return af_load_file_ex(path_utf8, 0, out, n_samples, NULL, 0);
}

int af_load_file_ex(const char *path_utf8, size_t max_samples, float **out, size_t *n_samples,
                    char *err, size_t err_size) {
    if (err && err_size > 0)
        err[0] = 0;
    if (!path_utf8 || !out || !n_samples)
        return AF_EINVAL;
    *out = NULL;
    *n_samples = 0;
    FILE *f = open_utf8(path_utf8);
    if (!f)
        return AF_EIO;
    const af_raw_mode raw = has_raw_extension(path_utf8) ? AF_RAW_FORCE : AF_RAW_NEVER;
    /* A regular file is read into one buffer of its size (one byte more, to see it end); other
     * files, and one that grows meanwhile, into a doubling buffer. */
    const size_t expect = regular_file_size(f);
    uint8_t *buf = NULL;
    size_t size = 0, cap = 0;
    /* With a limit, af_prefix_too_long() looks at what has been read at 1 MiB, 2 MiB, 4 MiB, ...
     * (until it says the container cannot tell), so a long WAV is rejected without reading it
     * all. The load then runs on that prefix, which fails as the whole file would. */
    size_t check = max_samples ? AF_PREFIX_CHECK_BYTES : SIZE_MAX;
    int rc = AF_OK;
    for (;;) {
        if (size == cap) {
            size_t ncap = cap ? cap * 2 : expect ? expect + 1 : (size_t)1 << 20;
            uint8_t *nb = ncap > cap ? realloc(buf, ncap) : NULL;
            if (!nb) {
                rc = AF_ENOMEM;
                break;
            }
            buf = nb;
            cap = ncap;
        }
        const size_t want = cap - size < check - size ? cap - size : check - size;
        const size_t got = fread(buf + size, 1, want, f);
        size += got;
        if (got < want) {
            if (ferror(f))
                rc = AF_EIO;
            break;
        }
        if (size == check) {
            const int known = af_prefix_too_long(buf, size, raw, max_samples);
            if (known > 0)
                break;
            check = known < 0 || check > SIZE_MAX / 2 ? SIZE_MAX : check * 2;
        }
    }
    fclose(f);
    if (rc == AF_OK)
        rc = af_load_memory_ex(buf, size, raw, max_samples, out, n_samples, err, err_size);
    free(buf);
    return rc;
}

void af_audio_free(float *p) {
    free(p);
}
