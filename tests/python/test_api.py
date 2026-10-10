"""Public API surface: input handling, validation, batch API, low-level ``out=``."""

from __future__ import annotations

import numpy as np
import pytest

import asrfront
from asrfront import _ext
from asrfront.sensevoice import fbank, fbank_batch, fbank_raw
from asrfront.whisper import log_mel, log_mel_batch

import reference as ref
from helpers import assert_close


def test_version():
    assert asrfront.__version__ == "0.1.0"


# ---------------------------------------------------------------------------------------------
# Low-level extension
# ---------------------------------------------------------------------------------------------


def test_whisper_out_frames():
    assert _ext.WhisperFrontend().out_frames(16000) == 3000
    assert _ext.WhisperFrontend(pad_or_trim=False).out_frames(16000) == 100
    with pytest.raises(ValueError):
        _ext.WhisperFrontend(n_mels=64)


def test_sensevoice_out_frames():
    fe = _ext.SenseVoiceFrontend()
    assert fe.feat_dim == 560
    assert fe.raw_frames(16000) == 98
    assert fe.out_frames(16000) == 17


def test_sensevoice_cmvn_length_checked():
    with pytest.raises(ValueError):
        _ext.SenseVoiceFrontend(
            cmvn_shift=np.zeros(10, np.float32), cmvn_scale=np.ones(10, np.float32)
        )


def test_out_argument(sigs):
    audio = sigs["speech"]
    fe = _ext.WhisperFrontend()
    out = np.full((80, 3000), np.nan, np.float32)
    assert fe.log_mel(audio, out=out) is None
    np.testing.assert_array_equal(out, fe.log_mel(audio))

    sv = _ext.SenseVoiceFrontend()
    out = np.empty((sv.out_frames(audio.size), sv.feat_dim), np.float32)
    assert sv.fbank(audio, out=out) is None
    np.testing.assert_array_equal(out, sv.fbank(audio))


@pytest.mark.parametrize(
    "out,exc",
    [
        (np.empty((80, 2999), np.float32), ValueError),  # wrong shape
        (np.empty((80, 3000), np.float64), TypeError),  # wrong dtype: never converted
        (np.empty((3000, 80), np.float32).T, TypeError),  # not C-contiguous
        (np.empty((80, 3000), np.float32), TypeError),  # read-only (set below)
    ],
)
def test_out_argument_rejected(sigs, out, exc):
    if out.dtype == np.float32 and out.flags.c_contiguous and out.shape == (80, 3000):
        out.flags.writeable = False
    with pytest.raises(exc):
        _ext.WhisperFrontend().log_mel(sigs["speech"], out=out)


# ---------------------------------------------------------------------------------------------
# Input normalization and validation
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fn", [log_mel, fbank, fbank_raw])
def test_rejects_multichannel(fn):
    with pytest.raises(ValueError, match=r"1-D mono waveform.*mean\(axis=0\)"):
        fn(np.zeros((2, 16000), np.float32))
    with pytest.raises(ValueError, match="1-D mono"):
        fn(np.float32(0.5))


@pytest.mark.parametrize("fn", [log_mel, fbank, fbank_raw])
@pytest.mark.parametrize("dtype", [np.int32, np.uint8, np.int64])
def test_rejects_ambiguous_integer_dtypes(fn, dtype):
    with pytest.raises(TypeError, match="only int16 PCM"):
        fn(np.zeros(16000, dtype))


@pytest.mark.parametrize("fn", [log_mel, fbank, fbank_raw])
@pytest.mark.parametrize("dtype", [np.complex64, np.bool_, object])
def test_rejects_other_dtypes(fn, dtype):
    with pytest.raises(TypeError):
        fn(np.zeros(16000, dtype))


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_float_dtypes(sigs, dtype):
    audio = sigs["speech"].astype(dtype)
    expected = ref.whisper_log_mel(audio.astype(np.float64))
    assert_close(log_mel(audio), expected, 1e-4, str(dtype))


def test_torch_tensor_input(sigs):
    torch = pytest.importorskip("torch")
    audio = sigs["speech"]
    t = torch.from_numpy(audio.copy())
    np.testing.assert_array_equal(log_mel(t), log_mel(audio))
    np.testing.assert_array_equal(fbank(t), fbank(audio))


@pytest.mark.parametrize("n_mels", [64, 0, True, "80"])
def test_whisper_n_mels_validation(n_mels):
    with pytest.raises(ValueError, match="80 or 128"):
        log_mel(np.zeros(16000, np.float32), n_mels=n_mels)


@pytest.mark.parametrize(
    "kwargs,exc,match",
    [
        ({"n_mels": 0}, ValueError, "n_mels"),
        ({"n_mels": 80.0}, TypeError, "n_mels must be an int"),
        ({"lfr_m": 0}, ValueError, "lfr_m"),
        ({"lfr_n": -1}, ValueError, "lfr_n"),
        ({"dither": -1.0}, ValueError, "dither"),
        ({"dither": float("nan")}, ValueError, "dither"),
        ({"dither": "x"}, TypeError, "dither"),
        ({"seed": -1}, ValueError, "seed"),
        ({"seed": 2**64}, ValueError, "seed"),
        ({"cmvn": 5}, TypeError, "cmvn"),
    ],
)
def test_sensevoice_param_validation(kwargs, exc, match):
    with pytest.raises(exc, match=match):
        fbank(np.zeros(16000, np.float32), **kwargs)


def test_seed_full_range(sigs):
    a = fbank_raw(sigs["speech"], dither=1.0, seed=2**64 - 1)
    b = fbank_raw(sigs["speech"], dither=1.0, seed=2**64 - 1)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_threads", [1, 4, None])
def test_log_mel_batch(sigs, n_threads):
    names = ["speech", "chirp", "short_201", "silence", "speech_35s", "white_noise"]
    audios = [sigs[n] for n in names]
    out = log_mel_batch(audios, n_threads=n_threads)
    assert isinstance(out, np.ndarray) and out.shape == (len(names), 80, 3000)
    for i, a in enumerate(audios):
        np.testing.assert_array_equal(out[i], log_mel(a))


def test_log_mel_batch_unpadded_and_128(sigs):
    audios = [sigs["speech"], sigs["short_1000"], sigs["odd_length"]]
    out = log_mel_batch(audios, n_mels=128, pad_or_trim=False, n_threads=2)
    assert isinstance(out, list) and [o.shape for o in out] == [(128, 1100), (128, 6), (128, 100)]
    for o, a in zip(out, audios):
        np.testing.assert_array_equal(o, log_mel(a, n_mels=128, pad_or_trim=False))


def test_log_mel_batch_2d_array_and_empty(sigs):
    batch = np.stack([sigs["sine_440"][:16000], sigs["silence"]])
    out = log_mel_batch(batch)
    np.testing.assert_array_equal(out[0], log_mel(batch[0]))
    assert log_mel_batch([]).shape == (0, 80, 3000)


def test_log_mel_batch_errors(sigs):
    with pytest.raises(ValueError, match=r"audios\[1\] needs more than 200"):
        log_mel_batch([sigs["speech"], sigs["short_201"][:100]], pad_or_trim=False)
    with pytest.raises(TypeError, match=r"audios\[0\] has integer dtype"):
        log_mel_batch([np.zeros(10, np.int32)])
    with pytest.raises(TypeError, match="sequence"):
        log_mel_batch(5)
    with pytest.raises(ValueError, match="n_threads"):
        log_mel_batch([sigs["speech"]], n_threads=0)


@pytest.mark.parametrize("n_threads", [1, 3, None])
def test_fbank_batch(sigs, n_threads):
    names = ["speech", "short_399", "chirp", "short_1000", "speech_35s"]
    audios = [sigs[n] for n in names]
    cmvn = (np.full(560, -10, np.float32), np.full(560, 0.3, np.float32))
    feats, lengths = fbank_batch(audios, cmvn=cmvn, n_threads=n_threads)
    singles = [fbank(a, cmvn=cmvn) for a in audios]
    assert lengths.dtype == np.int32
    assert lengths.tolist() == [s.shape[0] for s in singles]
    assert feats.shape == (len(audios), max(lengths), 560)
    for i, s in enumerate(singles):
        np.testing.assert_array_equal(feats[i, : lengths[i]], s)
        assert not feats[i, lengths[i] :].any()  # zero padding


def test_fbank_batch_with_path_and_empty(sigs, tmp_path):
    path = tmp_path / "am.mvn"
    vals = " ".join(["0.5"] * 560)
    path.write_text(
        f"<AddShift> 560 560\n<LearnRateCoef> 0 [ {vals} ]\n"
        f"<Rescale> 560 560\n<LearnRateCoef> 0 [ {vals} ]\n"
    )
    feats, lengths = fbank_batch([sigs["speech"]], cmvn=path)
    np.testing.assert_array_equal(feats[0], fbank(sigs["speech"], cmvn=path))
    feats, lengths = fbank_batch([])
    assert feats.shape == (0, 0, 560) and lengths.shape == (0,)
