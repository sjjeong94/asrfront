#include "asrfront.h"
#include "kernels.h"

const char *af_version(void) {
    return ASRFRONT_VERSION;
}

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
    case AF_EUNSUPPORTED:
        return "unsupported audio format";
    case AF_EIO:
        return "cannot read file";
    case AF_ECORRUPT:
        return "malformed or truncated audio file";
    default:
        return "unknown error";
    }
}

const char *af_kernel_isa(void) {
    return af_kernels_select()->name;
}
