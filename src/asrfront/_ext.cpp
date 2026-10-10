// nanobind binding for the asrfront C core. Thin RAII wrappers around the C
// handles; argument normalization (dtype, contiguity) is done in Python.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

#include <cstring>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>

#include "asrfront.h"
extern "C" {
#include "melbank.h" // internal; exposed for tests only
}

namespace nb = nanobind;
using namespace nb::literals;

using Audio = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Vec = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Out2D = nb::ndarray<nb::numpy, float, nb::ndim<2>, nb::c_contig>;
using OutArg = std::optional<nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu>>;

static void check(int rc) {
    switch (rc) {
    case AF_OK:
        return;
    case AF_EINVAL:
        throw nb::value_error(af_strerror(rc));
    case AF_ENOMEM:
        throw std::bad_alloc();
    case AF_ENOTIMPL:
        PyErr_SetString(PyExc_NotImplementedError, "asrfront: not implemented yet");
        throw nb::python_error();
    default:
        throw std::runtime_error(af_strerror(rc));
    }
}

// Allocate a (rows, cols) float32 buffer, fill it with fn(buf) without the
// GIL, and hand ownership to a numpy array (zero-copy).
template <typename F> static Out2D compute_2d(size_t rows, size_t cols, F &&fn) {
    float *buf = new float[rows * cols > 0 ? rows * cols : 1];
    nb::capsule owner(buf, [](void *p) noexcept { delete[] static_cast<float *>(p); });
    int rc;
    {
        nb::gil_scoped_release release;
        rc = fn(buf);
    }
    check(rc);
    return Out2D(buf, {rows, cols}, owner);
}

// Either fill a new array (out is None) or write into the caller's C-contiguous,
// writable float32 buffer of exactly (rows, cols) and return None.
template <typename F>
static nb::object compute_into(size_t rows, size_t cols, OutArg &out, F &&fn) {
    if (!out)
        return nb::cast(compute_2d(rows, cols, fn));
    if (out->shape(0) != rows || out->shape(1) != cols)
        throw nb::value_error(("out has shape (" + std::to_string(out->shape(0)) + ", " +
                               std::to_string(out->shape(1)) + "), expected (" +
                               std::to_string(rows) + ", " + std::to_string(cols) + ")")
                                  .c_str());
    int rc;
    {
        nb::gil_scoped_release release;
        rc = fn(out->data());
    }
    check(rc);
    return nb::none();
}

struct Whisper {
    af_whisper *h = nullptr;
    int n_mels;

    Whisper(int n_mels_, bool pad_or_trim, int n_threads) : n_mels(n_mels_) {
        af_whisper_config cfg;
        af_whisper_default_config(&cfg);
        cfg.n_mels = n_mels_;
        cfg.pad_or_trim = pad_or_trim;
        cfg.n_threads = n_threads;
        check(af_whisper_create(&cfg, &h));
    }
    ~Whisper() { af_whisper_destroy(h); }
    Whisper(const Whisper &) = delete;
    Whisper &operator=(const Whisper &) = delete;

    nb::object log_mel(Audio audio, OutArg out) {
        size_t n = audio.shape(0);
        size_t frames = af_whisper_out_frames(h, n);
        return compute_into((size_t)n_mels, frames, out,
                            [&](float *o) { return af_whisper_log_mel(h, audio.data(), n, o); });
    }
};

struct SenseVoice {
    af_sensevoice *h = nullptr;
    int n_mels;

    SenseVoice(int n_mels, int lfr_m, int lfr_n, float dither, unsigned long long seed,
               std::optional<Vec> cmvn_shift, std::optional<Vec> cmvn_scale, int n_threads)
        : n_mels(n_mels) {
        af_sensevoice_config cfg;
        af_sensevoice_default_config(&cfg);
        cfg.n_mels = n_mels;
        cfg.lfr_m = lfr_m;
        cfg.lfr_n = lfr_n;
        cfg.dither = dither;
        cfg.seed = seed;
        cfg.n_threads = n_threads;
        if (cmvn_shift.has_value() != cmvn_scale.has_value())
            throw nb::value_error("cmvn_shift and cmvn_scale must be given together");
        if (cmvn_shift) {
            size_t dim = (size_t)n_mels * (size_t)lfr_m;
            if (cmvn_shift->shape(0) != dim || cmvn_scale->shape(0) != dim)
                throw nb::value_error("cmvn vectors must have length n_mels * lfr_m");
            cfg.cmvn_shift = cmvn_shift->data();
            cfg.cmvn_scale = cmvn_scale->data();
        }
        check(af_sensevoice_create(&cfg, &h));
    }
    ~SenseVoice() { af_sensevoice_destroy(h); }
    SenseVoice(const SenseVoice &) = delete;
    SenseVoice &operator=(const SenseVoice &) = delete;

    nb::object fbank(Audio audio, OutArg out) {
        size_t n = audio.shape(0);
        return compute_into(af_sensevoice_out_frames(h, n), af_sensevoice_feat_dim(h), out,
                            [&](float *o) { return af_sensevoice_fbank(h, audio.data(), n, o); });
    }

    nb::object fbank_raw(Audio audio, OutArg out) {
        size_t n = audio.shape(0);
        return compute_into(af_sensevoice_raw_frames(h, n), (size_t)n_mels, out, [&](float *o) {
            return af_sensevoice_fbank_raw(h, audio.data(), n, o);
        });
    }
};

NB_MODULE(_ext, m) {
    m.doc() = "asrfront native extension";
    m.attr("__version__") = af_version();
    m.def("kernel_isa", &af_kernel_isa,
          "SIMD kernel variant used by newly created frontends ('avx2' or 'generic'). "
          "ASRFRONT_KERNELS=generic forces the baseline build; results are identical.");

    m.def(
        "_whisper_mel_filters",
        [](int n_mels) {
            af_melbank *mb = nullptr;
            check(af_melbank_slaney(AF_SAMPLE_RATE, 400, n_mels, 0.0, AF_SAMPLE_RATE / 2.0, &mb));
            size_t rows = (size_t)af_melbank_n_mels(mb), cols = (size_t)af_melbank_n_bins(mb);
            Out2D res = compute_2d(rows, cols, [&](float *out) {
                af_melbank_to_dense(mb, out);
                return AF_OK;
            });
            af_melbank_destroy(mb);
            return res;
        },
        "n_mels"_a, "Dense (n_mels, 201) slaney filters used by the whisper frontend (testing).");

    m.def(
        "_kaldi_mel_banks",
        [](int n_mels) {
            af_melbank *mb = nullptr;
            check(af_melbank_kaldi(AF_SAMPLE_RATE, 512, n_mels, 20.0, 0.0, &mb));
            size_t rows = (size_t)af_melbank_n_mels(mb), cols = (size_t)af_melbank_n_bins(mb);
            Out2D res = compute_2d(rows, cols, [&](float *out) {
                af_melbank_to_dense(mb, out);
                return AF_OK;
            });
            af_melbank_destroy(mb);
            return res;
        },
        "n_mels"_a, "Dense (n_mels, 257) kaldi filters used by the sensevoice frontend (testing).");

    nb::class_<Whisper>(m, "WhisperFrontend")
        .def(nb::init<int, bool, int>(), "n_mels"_a = 80, "pad_or_trim"_a = true, "n_threads"_a = 1)
        .def_ro("n_mels", &Whisper::n_mels)
        .def(
            "out_frames", [](const Whisper &w, size_t n) { return af_whisper_out_frames(w.h, n); },
            "n_samples"_a)
        .def("log_mel", &Whisper::log_mel, "audio"_a,
             nb::arg("out").none().noconvert() = nb::none(),
             "Log-mel spectrogram (n_mels, out_frames). With `out`, writes into it and returns "
             "None.");

    nb::class_<SenseVoice>(m, "SenseVoiceFrontend")
        .def(nb::init<int, int, int, float, unsigned long long, std::optional<Vec>,
                      std::optional<Vec>, int>(),
             "n_mels"_a = 80, "lfr_m"_a = 7, "lfr_n"_a = 6, "dither"_a = 0.0f, "seed"_a = 0,
             "cmvn_shift"_a = nb::none(), "cmvn_scale"_a = nb::none(), "n_threads"_a = 1)
        .def_prop_ro("feat_dim", [](const SenseVoice &s) { return af_sensevoice_feat_dim(s.h); })
        .def(
            "raw_frames",
            [](const SenseVoice &s, size_t n) { return af_sensevoice_raw_frames(s.h, n); },
            "n_samples"_a)
        .def(
            "out_frames",
            [](const SenseVoice &s, size_t n) { return af_sensevoice_out_frames(s.h, n); },
            "n_samples"_a)
        .def_ro("n_mels", &SenseVoice::n_mels)
        .def("fbank", &SenseVoice::fbank, "audio"_a, nb::arg("out").none().noconvert() = nb::none(),
             "LFR (+CMVN) features (out_frames, feat_dim). With `out`, writes into it and returns "
             "None.")
        .def("fbank_raw", &SenseVoice::fbank_raw, "audio"_a,
             nb::arg("out").none().noconvert() = nb::none(),
             "Raw kaldi fbank (raw_frames, n_mels). With `out`, writes into it and returns None.");
}
