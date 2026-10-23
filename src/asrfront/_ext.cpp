// nanobind binding for the asrfront C core. Thin RAII wrappers around the C
// handles; argument normalization (dtype, contiguity) is done in Python.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>

#include "asrfront.h"
extern "C" {
#include "melbank.h"  // internal; exposed for tests only
#include "resample.h" // internal; exposed for tests only
}

namespace nb = nanobind;
using namespace nb::literals;

using Audio = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Vec = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Out2D = nb::ndarray<nb::numpy, float, nb::ndim<2>, nb::c_contig>;
using OutArg = std::optional<nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu>>;
// Rows may be spaced apart (e.g. feats[i, :, :n] of a padded batch); elements contiguous.
using OutRows = std::optional<nb::ndarray<float, nb::ndim<2>, nb::device::cpu>>;
using Bytes = nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Out1D = nb::ndarray<nb::numpy, float, nb::ndim<1>, nb::c_contig>;
using OutC = nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Rows = nb::ndarray<float, nb::ndim<2>, nb::device::cpu>;

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
    case AF_EUNSUPPORTED:
    case AF_ECORRUPT:
    case AF_ETOOLONG:
        throw nb::value_error(af_strerror(rc));
    case AF_EIO:
        PyErr_SetString(PyExc_OSError, af_strerror(rc));
        throw nb::python_error();
    default:
        throw std::runtime_error(af_strerror(rc));
    }
}

// Allocate a (rows, cols) float32 buffer, fill it with fn(buf) without the
// GIL, and hand ownership to a numpy array (zero-copy).
template <typename F> static Out2D compute_2d(size_t rows, size_t cols, F &&fn) {
    if (cols && rows > SIZE_MAX / sizeof(float) / cols)
        throw nb::value_error(("output of shape (" + std::to_string(rows) + ", " +
                               std::to_string(cols) + ") is too large")
                                  .c_str());
    std::unique_ptr<float[]> mem(new float[rows * cols > 0 ? rows * cols : 1]);
    float *buf = mem.get();
    nb::capsule owner(buf, [](void *p) noexcept { delete[] static_cast<float *>(p); });
    mem.release(); // the capsule owns it now (its constructor may throw before)
    int rc;
    {
        nb::gil_scoped_release release;
        rc = fn(buf);
    }
    check(rc);
    return Out2D(buf, {rows, cols}, owner);
}

template <typename A> static void check_shape(const A &out, size_t rows, size_t cols) {
    if (out.shape(0) != rows || out.shape(1) != cols)
        throw nb::value_error(("out has shape (" + std::to_string(out.shape(0)) + ", " +
                               std::to_string(out.shape(1)) + "), expected (" +
                               std::to_string(rows) + ", " + std::to_string(cols) + ")")
                                  .c_str());
}

// Row stride (elements) of an output of `frames` columns whose rows are contiguous and apart.
static size_t row_stride(const Rows &out, size_t frames) {
    const int64_t ld = out.stride(0);
    if ((frames > 1 && out.stride(1) != 1) || ld < (int64_t)frames)
        throw nb::value_error("out must have contiguous rows (e.g. a slice a[:, :n] of a "
                              "C-contiguous array)");
    return (size_t)ld;
}

// Whether no sample is NaN or inf (exponent bits all set), in blocks that vectorize.
static bool all_finite(const float *x, size_t n) {
    for (size_t i0 = 0; i0 < n; i0 += 1024) {
        const size_t end = n - i0 < 1024 ? n : i0 + 1024;
        unsigned bad = 0;
        for (size_t i = i0; i < end; i++) {
            uint32_t u;
            std::memcpy(&u, x + i, sizeof u);
            bad |= (u & 0x7F800000u) == 0x7F800000u;
        }
        if (bad)
            return false;
    }
    return true;
}

// The C handles are not thread-safe and the methods run without the GIL, so concurrent calls
// on one object would silently share its scratch buffers: refuse them instead. Held by every
// method that computes, for the whole call.
class Busy {
    std::atomic<bool> &flag;

  public:
    explicit Busy(std::atomic<bool> &f) : flag(f) {
        if (flag.exchange(true, std::memory_order_acquire))
            throw std::runtime_error("asrfront: a frontend object was used by two threads at "
                                     "once; use one object per thread");
    }
    ~Busy() { flag.store(false, std::memory_order_release); }
    Busy(const Busy &) = delete;
    Busy &operator=(const Busy &) = delete;
};

// Batch chunks (the *_many methods): one GIL release for all items, since per-item releases
// (and the NaN / inf check in numpy) make worker threads queue for the GIL when items are
// short. fn(i) returns an AF_* code; the first failure stops the chunk. Returns the index of
// the first item with a NaN / inf sample, which stops the chunk before it is computed, or -1.
template <typename F> static int64_t run_many(const std::vector<Audio> &audios, F &&fn) {
    int rc = AF_OK;
    int64_t bad = -1;
    {
        nb::gil_scoped_release release;
        for (size_t i = 0; i < audios.size() && rc == AF_OK; i++) {
            if (!all_finite(audios[i].data(), audios[i].shape(0))) {
                bad = (int64_t)i;
                break;
            }
            rc = fn(i);
        }
    }
    check(rc);
    return bad;
}

static void check_count(size_t audios, size_t outs) {
    if (audios != outs)
        throw nb::value_error("audios and outs must have the same length");
}

// Either fill a new array (out is None) or write into the caller's C-contiguous,
// writable float32 buffer of exactly (rows, cols) and return None.
template <typename F>
static nb::object compute_into(size_t rows, size_t cols, OutArg &out, F &&fn) {
    if (!out)
        return nb::cast(compute_2d(rows, cols, fn));
    check_shape(*out, rows, cols);
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
    std::atomic<bool> busy{false};

    Whisper(int n_mels_, bool pad_or_trim) : n_mels(n_mels_) {
        af_whisper_config cfg;
        af_whisper_default_config(&cfg);
        cfg.n_mels = n_mels_;
        cfg.pad_or_trim = pad_or_trim;
        check(af_whisper_create(&cfg, &h));
    }
    ~Whisper() { af_whisper_destroy(h); }
    Whisper(const Whisper &) = delete;
    Whisper &operator=(const Whisper &) = delete;

    nb::object log_mel(Audio audio, OutRows out) {
        Busy guard(busy);
        size_t n = audio.shape(0);
        size_t frames = af_whisper_out_frames(h, n);
        if (!out)
            return nb::cast(compute_2d((size_t)n_mels, frames, [&](float *o) {
                return af_whisper_log_mel(h, audio.data(), n, o);
            }));
        check_shape(*out, (size_t)n_mels, frames);
        const size_t ld = row_stride(*out, frames);
        int rc;
        {
            nb::gil_scoped_release release;
            rc = af_whisper_log_mel_ld(h, audio.data(), n, out->data(), ld);
        }
        check(rc);
        return nb::none();
    }

    int64_t log_mel_many(const std::vector<Audio> &audios, const std::vector<Rows> &outs) {
        Busy guard(busy);
        check_count(audios.size(), outs.size());
        std::vector<size_t> ld(outs.size());
        for (size_t i = 0; i < outs.size(); i++) {
            const size_t frames = af_whisper_out_frames(h, audios[i].shape(0));
            check_shape(outs[i], (size_t)n_mels, frames);
            ld[i] = row_stride(outs[i], frames);
        }
        return run_many(audios, [&](size_t i) {
            return af_whisper_log_mel_ld(h, audios[i].data(), audios[i].shape(0), outs[i].data(),
                                         ld[i]);
        });
    }

    nb::object log_mel_padded(Audio audio, size_t n_padded, OutArg out) {
        Busy guard(busy);
        size_t n = audio.shape(0);
        return compute_into((size_t)n_mels, n_padded / 160, out, [&](float *o) {
            return af_whisper_log_mel_padded(h, audio.data(), n, n_padded, o);
        });
    }

    int64_t log_mel_padded_many(const std::vector<Audio> &audios, size_t n_padded,
                                const std::vector<OutC> &outs) {
        Busy guard(busy);
        check_count(audios.size(), outs.size());
        for (const OutC &o : outs)
            check_shape(o, (size_t)n_mels, n_padded / 160);
        return run_many(audios, [&](size_t i) {
            return af_whisper_log_mel_padded(h, audios[i].data(), audios[i].shape(0), n_padded,
                                             outs[i].data());
        });
    }
};

struct SenseVoice {
    af_sensevoice *h = nullptr;
    int n_mels;
    std::atomic<bool> busy{false};

    SenseVoice(int n_mels, int lfr_m, int lfr_n, float dither, unsigned long long seed,
               std::optional<Vec> cmvn_shift, std::optional<Vec> cmvn_scale)
        : n_mels(n_mels) {
        af_sensevoice_config cfg;
        af_sensevoice_default_config(&cfg);
        cfg.n_mels = n_mels;
        cfg.lfr_m = lfr_m;
        cfg.lfr_n = lfr_n;
        cfg.dither = dither;
        cfg.seed = seed;
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
        Busy guard(busy);
        size_t n = audio.shape(0);
        return compute_into(af_sensevoice_out_frames(h, n), af_sensevoice_feat_dim(h), out,
                            [&](float *o) { return af_sensevoice_fbank(h, audio.data(), n, o); });
    }

    int64_t fbank_many(const std::vector<Audio> &audios, const std::vector<OutC> &outs) {
        Busy guard(busy);
        check_count(audios.size(), outs.size());
        for (size_t i = 0; i < outs.size(); i++)
            check_shape(outs[i], af_sensevoice_out_frames(h, audios[i].shape(0)),
                        af_sensevoice_feat_dim(h));
        return run_many(audios, [&](size_t i) {
            return af_sensevoice_fbank(h, audios[i].data(), audios[i].shape(0), outs[i].data());
        });
    }

    nb::object fbank_raw(Audio audio, OutArg out) {
        Busy guard(busy);
        size_t n = audio.shape(0);
        return compute_into(af_sensevoice_raw_frames(h, n), (size_t)n_mels, out, [&](float *o) {
            return af_sensevoice_fbank_raw(h, audio.data(), n, o);
        });
    }
};

struct SenseVoiceStream {
    af_sensevoice_stream *s = nullptr;
    std::atomic<bool> busy{false};

    SenseVoiceStream(int n_mels, int lfr_m, int lfr_n, float dither, unsigned long long seed,
                     std::optional<Vec> cmvn_shift, std::optional<Vec> cmvn_scale) {
        af_sensevoice_config cfg;
        af_sensevoice_default_config(&cfg);
        cfg.n_mels = n_mels;
        cfg.lfr_m = lfr_m;
        cfg.lfr_n = lfr_n;
        cfg.dither = dither;
        cfg.seed = seed;
        if (cmvn_shift.has_value() != cmvn_scale.has_value())
            throw nb::value_error("cmvn_shift and cmvn_scale must be given together");
        if (cmvn_shift) {
            size_t dim = (size_t)n_mels * (size_t)lfr_m;
            if (cmvn_shift->shape(0) != dim || cmvn_scale->shape(0) != dim)
                throw nb::value_error("cmvn vectors must have length n_mels * lfr_m");
            cfg.cmvn_shift = cmvn_shift->data();
            cfg.cmvn_scale = cmvn_scale->data();
        }
        check(af_sensevoice_stream_create(&cfg, &s));
    }
    ~SenseVoiceStream() { af_sensevoice_stream_destroy(s); }
    SenseVoiceStream(const SenseVoiceStream &) = delete;
    SenseVoiceStream &operator=(const SenseVoiceStream &) = delete;

    size_t feat_dim() const { return af_sensevoice_feat_dim(af_sensevoice_stream_frontend(s)); }

    Out2D push(Audio audio) {
        Busy guard(busy);
        const size_t n = audio.shape(0);
        return compute_2d(af_sensevoice_stream_push_rows(s, n), feat_dim(), [&](float *o) {
            size_t rows;
            return af_sensevoice_stream_push(s, audio.data(), n, o, &rows);
        });
    }

    Out2D finish() {
        Busy guard(busy);
        return compute_2d(af_sensevoice_stream_finish_rows(s), feat_dim(), [&](float *o) {
            size_t rows;
            return af_sensevoice_stream_finish(s, o, &rows);
        });
    }

    void reset() {
        Busy guard(busy);
        af_sensevoice_stream_reset(s);
    }
};

struct Gemma4 {
    af_gemma4 *h = nullptr;
    std::atomic<bool> busy{false};

    Gemma4() { check(af_gemma4_create(&h)); }
    ~Gemma4() { af_gemma4_destroy(h); }
    Gemma4(const Gemma4 &) = delete;
    Gemma4 &operator=(const Gemma4 &) = delete;

    nb::object log_mel(Audio audio, size_t n_frames, OutArg out) {
        Busy guard(busy);
        size_t n = audio.shape(0);
        return compute_into(n_frames, 128, out, [&](float *o) {
            return af_gemma4_log_mel(h, audio.data(), n, n_frames, o);
        });
    }

    int64_t log_mel_many(const std::vector<Audio> &audios, size_t n_frames,
                         const std::vector<OutC> &outs) {
        Busy guard(busy);
        check_count(audios.size(), outs.size());
        for (const OutC &o : outs)
            check_shape(o, n_frames, 128);
        return run_many(audios, [&](size_t i) {
            return af_gemma4_log_mel(h, audios[i].data(), audios[i].shape(0), n_frames,
                                     outs[i].data());
        });
    }
};

NB_MODULE(_ext, m) {
    m.doc() = "asrfront native extension";
    m.attr("__version__") = af_version();
    m.def("kernel_isa", &af_kernel_isa,
          "SIMD kernel variant used by newly created frontends ('avx2', 'generic' or "
          "'scalar'). ASRFRONT_KERNELS=generic|scalar forces the baseline / plain-C build; "
          "results are identical.");

    m.def(
        "all_finite",
        [](Audio a) {
            const size_t n = a.shape(0);
            if (n < (1u << 16)) // short: cheaper than giving up and retaking the GIL
                return all_finite(a.data(), n);
            nb::gil_scoped_release release;
            return all_finite(a.data(), n);
        },
        nb::arg("audio").noconvert(),
        "Whether no sample of a contiguous float32 array is NaN or inf (without the GIL).");

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

    m.def(
        "load_memory",
        [](Bytes data, int raw, size_t max_samples) {
            float *buf = nullptr;
            size_t n = 0;
            char err[128];
            int rc;
            {
                nb::gil_scoped_release release;
                rc = af_load_memory_ex(data.data(), data.shape(0), (af_raw_mode)raw, max_samples,
                                       &buf, &n, err, sizeof err);
            }
            if (rc != AF_OK && err[0])
                throw nb::value_error((std::string(af_strerror(rc)) + " (" + err + ")").c_str());
            check(rc);
            nb::capsule owner(buf,
                              [](void *p) noexcept { af_audio_free(static_cast<float *>(p)); });
            return Out1D(buf, {n}, owner);
        },
        "data"_a, "raw"_a, "max_samples"_a = 0,
        "Decode an in-memory audio file to 16 kHz mono float32. raw: 0 auto, 1 never, 2 force "
        "(raw = headerless s16le 16 kHz mono). max_samples > 0: ValueError if the audio is "
        "longer than max_samples / 16000 s.");

    m.def(
        "probe_memory",
        [](Bytes data, int raw, size_t max_samples) {
            af_audio_info info;
            char err[128];
            int rc;
            {
                nb::gil_scoped_release release;
                rc = af_probe_memory(data.data(), data.shape(0), (af_raw_mode)raw, max_samples,
                                     &info, err, sizeof err);
            }
            if (rc != AF_OK && err[0])
                throw nb::value_error((std::string(af_strerror(rc)) + " (" + err + ")").c_str());
            check(rc);
            return nb::make_tuple(info.container, info.sample_rate, info.channels, info.n_frames,
                                  info.n_samples);
        },
        "data"_a, "raw"_a, "max_samples"_a = 0,
        "What load_memory would decode, without decoding: (container, sample_rate, channels, "
        "n_frames, n_samples). Raises as load_memory.");

    m.def(
        "_resample",
        [](nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu, nb::ndim<1>> x, int sr_in,
           int sr_out) {
            af_resampler *r = nullptr;
            check(af_resampler_create(sr_in, sr_out, &r));
            std::unique_ptr<af_resampler, void (*)(af_resampler *)> guard(r, af_resampler_destroy);
            size_t n = x.shape(0), m = af_resampler_out_len(r, n);
            nb::dlpack::dtype dt = x.dtype();
            if (dt == nb::dtype<float>()) {
                std::unique_ptr<float[]> mem(new float[m > 0 ? m : 1]);
                float *y = mem.get();
                nb::capsule owner(y, [](void *p) noexcept { delete[] static_cast<float *>(p); });
                mem.release();
                af_resample_f32(r, static_cast<const float *>(x.data()), n, y);
                return nb::cast(nb::ndarray<nb::numpy, float, nb::ndim<1>>(y, {m}, owner));
            }
            if (dt == nb::dtype<double>()) {
                std::unique_ptr<double[]> mem(new double[m > 0 ? m : 1]);
                double *y = mem.get();
                nb::capsule owner(y, [](void *p) noexcept { delete[] static_cast<double *>(p); });
                mem.release();
                af_resample_f64(r, static_cast<const double *>(x.data()), n, y);
                return nb::cast(nb::ndarray<nb::numpy, double, nb::ndim<1>>(y, {m}, owner));
            }
            if (dt == nb::dtype<int32_t>()) {
                std::unique_ptr<int32_t[]> mem(new int32_t[m > 0 ? m : 1]);
                int32_t *y = mem.get();
                nb::capsule owner(y, [](void *p) noexcept { delete[] static_cast<int32_t *>(p); });
                mem.release();
                af_resample_s16(r, static_cast<const int32_t *>(x.data()), n, y);
                return nb::cast(nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(y, {m}, owner));
            }
            throw nb::type_error("x must be float32 (float path), float64 or int32 (int16 path)");
        },
        "x"_a, "sr_in"_a, "sr_out"_a = AF_SAMPLE_RATE, "Resample one channel (testing).");

    m.def(
        "_resample_bank",
        [](int sr_in, int sr_out) {
            af_resampler *r = nullptr;
            check(af_resampler_create(sr_in, sr_out, &r));
            std::unique_ptr<af_resampler, void (*)(af_resampler *)> guard(r, af_resampler_destroy);
            size_t rows = (size_t)af_resampler_phases(r) + 1, cols = (size_t)af_resampler_taps(r);
            std::unique_ptr<double[]> mem(new double[rows * cols]);
            double *b = mem.get();
            nb::capsule owner(b, [](void *p) noexcept { delete[] static_cast<double *>(p); });
            mem.release();
            for (size_t i = 0; i < rows; i++)
                std::memcpy(b + i * cols, af_resampler_bank(r, (int)i), cols * sizeof(double));
            return nb::ndarray<nb::numpy, double, nb::ndim<2>>(b, {rows, cols}, owner);
        },
        "sr_in"_a, "sr_out"_a = AF_SAMPLE_RATE, "(phases + 1, N) float64 filter bank (testing).");

    nb::class_<Whisper>(m, "WhisperFrontend")
        .def(nb::init<int, bool>(), "n_mels"_a = 80, "pad_or_trim"_a = true)
        .def_ro("n_mels", &Whisper::n_mels)
        .def(
            "out_frames", [](const Whisper &w, size_t n) { return af_whisper_out_frames(w.h, n); },
            "n_samples"_a)
        .def("log_mel", &Whisper::log_mel, "audio"_a,
             nb::arg("out").none().noconvert() = nb::none(),
             "Log-mel spectrogram (n_mels, out_frames). With `out` (float32, contiguous rows, "
             "e.g. a[:, :n] of a C-contiguous array), writes into it and returns None.")
        .def("log_mel_padded", &Whisper::log_mel_padded, "audio"_a, "n_padded"_a,
             nb::arg("out").none().noconvert() = nb::none(),
             "Log-mel spectrogram of `audio` followed by zeros up to `n_padded` samples, "
             "(n_mels, n_padded // 160). Needs pad_or_trim=False.")
        .def("log_mel_many", &Whisper::log_mel_many, nb::arg("audios").noconvert(),
             nb::arg("outs").noconvert(),
             "log_mel(audios[i], out=outs[i]) for every i with one GIL release (batch chunks). "
             "Stops at the first item with a NaN / inf sample and returns its index (else -1).")
        .def("log_mel_padded_many", &Whisper::log_mel_padded_many, nb::arg("audios").noconvert(),
             "n_padded"_a, nb::arg("outs").noconvert(),
             "log_mel_padded(audios[i], n_padded, out=outs[i]) for every i with one GIL release; "
             "returns as log_mel_many.");

    nb::class_<Gemma4>(m, "Gemma4Frontend")
        .def(nb::init<>())
        .def_static("out_frames", &af_gemma4_out_frames, "n_padded"_a,
                    "Frames for an input padded to n_padded samples.")
        .def_static("valid_frames", &af_gemma4_valid_frames, "n_samples"_a,
                    "Frames that read no padding (the feature mask sum).")
        .def("log_mel", &Gemma4::log_mel, "audio"_a, "n_frames"_a,
             nb::arg("out").none().noconvert() = nb::none(),
             "Log-mel features (n_frames, 128): the valid frames of `audio`, then zeros. With "
             "`out`, writes into it and returns None.")
        .def("log_mel_many", &Gemma4::log_mel_many, nb::arg("audios").noconvert(), "n_frames"_a,
             nb::arg("outs").noconvert(),
             "log_mel(audios[i], n_frames, out=outs[i]) for every i with one GIL release; "
             "returns as WhisperFrontend.log_mel_many.");

    nb::class_<SenseVoiceStream>(m, "SenseVoiceStream")
        .def(nb::init<int, int, int, float, unsigned long long, std::optional<Vec>,
                      std::optional<Vec>>(),
             "n_mels"_a = 80, "lfr_m"_a = 7, "lfr_n"_a = 6, "dither"_a = 0.0f, "seed"_a = 0,
             "cmvn_shift"_a = nb::none(), "cmvn_scale"_a = nb::none())
        .def_prop_ro("feat_dim", &SenseVoiceStream::feat_dim)
        .def("push", &SenseVoiceStream::push, "audio"_a,
             "Append audio; returns the LFR (+CMVN) rows it completes, (k, feat_dim).")
        .def("finish", &SenseVoiceStream::finish,
             "The remaining rows of the utterance, then a reset.")
        .def("reset", &SenseVoiceStream::reset, "Drop the pushed audio.");

    nb::class_<SenseVoice>(m, "SenseVoiceFrontend")
        .def(nb::init<int, int, int, float, unsigned long long, std::optional<Vec>,
                      std::optional<Vec>>(),
             "n_mels"_a = 80, "lfr_m"_a = 7, "lfr_n"_a = 6, "dither"_a = 0.0f, "seed"_a = 0,
             "cmvn_shift"_a = nb::none(), "cmvn_scale"_a = nb::none())
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
             "Raw kaldi fbank (raw_frames, n_mels). With `out`, writes into it and returns None.")
        .def("fbank_many", &SenseVoice::fbank_many, nb::arg("audios").noconvert(),
             nb::arg("outs").noconvert(),
             "fbank(audios[i], out=outs[i]) for every i with one GIL release (batch chunks); "
             "returns as WhisperFrontend.log_mel_many.");
}
