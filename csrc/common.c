#include "asrfront.h"

const char *af_version(void) { return ASRFRONT_VERSION; }

const char *af_strerror(int err) {
    switch (err) {
    case AF_OK:
        return "success";
    case AF_EINVAL:
        return "invalid argument";
    case AF_ENOMEM:
        return "out of memory";
    case AF_ENOTIMPL:
        return "not implemented";
    default:
        return "unknown error";
    }
}
