/* NIST SPHERE decoding (uncompressed PCM and G.711; not the shorten-compressed variant). */
#include <string.h>

#include "asrfront.h"
#include "audio.h"

static int is_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' || c == '\f';
}

/* The n-th (0-based) whitespace-separated token of line [p, end) into tok (truncated);
 * 0 if the line has fewer tokens. */
static int token(const uint8_t *p, const uint8_t *end, int n, char *tok, size_t cap) {
    for (int i = 0;; i++) {
        while (p < end && is_space(*p))
            p++;
        if (p == end)
            return 0;
        const uint8_t *s = p;
        while (p < end && !is_space(*p))
            p++;
        if (i == n) {
            size_t len = (size_t)(p - s) < cap - 1 ? (size_t)(p - s) : cap - 1;
            memcpy(tok, s, len);
            tok[len] = '\0';
            return 1;
        }
    }
}

/* Leading decimal integer of s (optional sign); 0 if there are no digits. Saturates. */
static int parse_int(const char *s, long *v) {
    int neg = *s == '-';
    if (*s == '-' || *s == '+')
        s++;
    if (*s < '0' || *s > '9')
        return 0;
    long x = 0;
    for (; *s >= '0' && *s <= '9'; s++)
        x = x < 100000000L ? x * 10 + (*s - '0') : x;
    *v = neg ? -x : x;
    return 1;
}

static int lower_eq(const char *s, const char *lit) {
    for (; *s && *lit; s++, lit++)
        if ((*s | 0x20) != *lit)
            return 0;
    return *s == *lit;
}

/* "NIST_1A\n", the header size on the second line, then "key -type value" lines up to
 * "end_head". Keys are matched by prefix and the last occurrence wins (as in ffmpeg). */
int af_sphere_decode(const uint8_t *buf, size_t size, size_t limit, af_decoded *out) {
    memset(out, 0, sizeof(*out));
    out->limit = limit;
    if (size < 8 || memcmp(buf, "NIST_1A\n", 8) != 0)
        return AF_ECORRUPT;
    long header_size = 0, channels = 0, n_bytes = 0, rate = 0;
    char coding[32] = "pcm", byte_format[32] = "", tok[32];
    int have_format = 0, have_size = 0, ended = 0;
    const uint8_t *p = buf + 8, *end = buf + size;
    while (p < end && !ended) {
        const uint8_t *eol = memchr(p, '\n', (size_t)(end - p));
        const uint8_t *line_end = eol ? eol : end;
        size_t len = (size_t)(line_end - p);
        if (!have_size) {
            have_size = token(p, line_end, 0, tok, sizeof tok) && parse_int(tok, &header_size);
            if (!have_size || header_size <= 0)
                return AF_ECORRUPT;
        } else if (len >= 8 && memcmp(p, "end_head", 8) == 0) {
            ended = 1;
        } else if (token(p, line_end, 2, tok, sizeof tok)) {
            static const char *const keys[] = {"channel_count", "sample_n_bytes", "sample_rate",
                                               "sample_byte_format", "sample_coding"};
            long *ints[] = {&channels, &n_bytes, &rate};
            for (int k = 0; k < 5; k++) {
                size_t kl = strlen(keys[k]);
                if (len < kl || memcmp(p, keys[k], kl) != 0)
                    continue;
                if (k < 3) {
                    long v;
                    if (parse_int(tok, &v))
                        *ints[k] = v;
                } else if (k == 3) {
                    memcpy(byte_format, tok, sizeof tok);
                    have_format = 1;
                } else {
                    memcpy(coding, tok, sizeof tok);
                }
                break;
            }
        }
        if (!eol)
            break;
        p = eol + 1;
    }
    if (!ended || (size_t)header_size > size)
        return AF_ECORRUPT;
    if (channels <= 0 || rate <= 0)
        return AF_ECORRUPT;
    if (channels > AF_MAX_CHANNELS)
        return af_fail(out, AF_EUNSUPPORTED, "%ld channels (at most %d)", channels,
                       AF_MAX_CHANNELS);

    af_pcm_codec codec;
    if (lower_eq(coding, "pcm")) {
        if (n_bytes == 1) {
            codec = AF_PCM_S8;
        } else if (n_bytes == 2) {
            if (!have_format || strcmp(byte_format, "01") == 0 || strcmp(byte_format, "1") == 0)
                codec = AF_PCM_S16LE;
            else if (strcmp(byte_format, "10") == 0)
                codec = AF_PCM_S16BE;
            else
                return af_fail(out, AF_EUNSUPPORTED, "sample_byte_format %s", byte_format);
        } else if (n_bytes <= 0) {
            return AF_ECORRUPT;
        } else { /* 3/4 bytes: not in ffmpeg */
            return af_fail(out, AF_EUNSUPPORTED, "%ld-byte PCM", n_bytes);
        }
    } else if (lower_eq(coding, "ulaw") || lower_eq(coding, "mu-law")) {
        codec = AF_PCM_MULAW; /* one byte per sample whatever sample_n_bytes says */
    } else if (lower_eq(coding, "alaw")) {
        codec = AF_PCM_ALAW;
    } else { /* e.g. "pcm,embedded-shorten-v2.00" */
        return af_fail(out, AF_EUNSUPPORTED, "sample_coding %s", coding);
    }

    out->channels = (int)channels;
    out->sample_rate = (int)rate;
    int rc = af_default_layout(out->channels, out->layout);
    if (rc != AF_OK)
        return rc;
    /* sample_count is not used: the samples run to the end of the file */
    return af_pcm_decode_frames(out, codec, buf + header_size, size - (size_t)header_size);
}
