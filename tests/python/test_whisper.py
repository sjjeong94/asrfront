"""asrfront.whisper vs. the float64 reference."""

from __future__ import annotations

import threading

import numpy as np
import pytest

import asrfront
from asrfront import _ext

import reference as ref
from audio_signals import DATA_DIR, SIGNAL_NAMES
from helpers import assert_close

ATOL = 1e-4
MEL_NPZ = DATA_DIR / "whisper_mel_filters.npz"


def _too_short(audio: np.ndarray, pad_or_trim: bool) -> bool:
    return not pad_or_trim and audio.shape[0] <= ref.WHISPER_N_FFT // 2


@pytest.mark.parametrize("n_mels", [80, 128])
def test_mel_filters_match_whisper_assets(n_mels):
    actual = _ext._whisper_mel_filters(n_mels)
    expected = ref.load_whisper_mel_filters(MEL_NPZ, n_mels)
    assert actual.dtype == np.float32 and actual.shape == expected.shape
    # Generated in double and rounded like librosa: identical up to the odd 1-ulp libm diff.
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=0)
    assert np.mean(actual == expected) > 0.99


@pytest.mark.parametrize("pad_or_trim", [True, False])
@pytest.mark.parametrize("n_mels", [80, 128])
@pytest.mark.parametrize("name", SIGNAL_NAMES)
def test_log_mel_matches_reference(sigs, name, n_mels, pad_or_trim):
    audio = sigs[name]
    if _too_short(audio, pad_or_trim):
        with pytest.raises(ValueError, match="more than 200 samples"):
            asrfront.whisper.log_mel(audio, n_mels=n_mels, pad_or_trim=pad_or_trim)
        return
    actual = asrfront.whisper.log_mel(audio, n_mels=n_mels, pad_or_trim=pad_or_trim)
    expected = ref.whisper_log_mel(audio, n_mels, pad_or_trim)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    assert_close(actual, expected, ATOL, name)


@pytest.mark.parametrize("pad_or_trim", [True, False])
@pytest.mark.parametrize("name", ["speech", "speech_35s", "chirp", "white_noise"])
def test_log_mel_matches_whisper_torch(sigs, name, pad_or_trim):
    pytest.importorskip("torch")
    audio = sigs[name]
    actual = asrfront.whisper.log_mel(audio, pad_or_trim=pad_or_trim)
    expected = ref.whisper_log_mel_torch(audio, 80, pad_or_trim, MEL_NPZ)
    assert_close(actual, expected, ATOL, name)


def test_default_shape(sigs):
    assert asrfront.whisper.log_mel(sigs["speech"]).shape == (80, 3000)
    assert asrfront.whisper.log_mel(sigs["speech"], n_mels=128).shape == (128, 3000)
    assert asrfront.whisper.log_mel(sigs["speech"], pad_or_trim=False).shape == (80, 1100)


def test_empty_input_with_padding():
    out = asrfront.whisper.log_mel(np.zeros(0, np.float32))
    assert_close(out, ref.whisper_log_mel(np.zeros(0)), ATOL, "empty")


def test_int16_input_is_scaled(sigs):
    audio = sigs["speech"]
    pcm = np.round(audio * 32768).clip(-32768, 32767).astype(np.int16)
    expected = ref.whisper_log_mel(pcm.astype(np.float64) / 32768.0)
    assert_close(asrfront.whisper.log_mel(pcm), expected, ATOL, "int16")


def test_float64_and_strided_input(sigs):
    audio = sigs["speech"]
    expected = asrfront.whisper.log_mel(audio)
    assert_close(asrfront.whisper.log_mel(audio.astype(np.float64)), expected, 1e-6, "float64")
    strided = np.repeat(audio, 2)[::2]
    assert not strided.flags.c_contiguous
    assert_close(asrfront.whisper.log_mel(strided), expected, 0, "strided")


def test_list_input():
    audio = [0.0, 0.5, -0.5] * 1000
    assert_close(
        asrfront.whisper.log_mel(audio), ref.whisper_log_mel(np.array(audio)), ATOL, "list"
    )


def test_deterministic_and_reusable(sigs):
    fe = _ext.WhisperFrontend()
    a = fe.log_mel(sigs["speech"])
    fe.log_mel(sigs["white_noise"])  # scratch state must not leak between calls
    b = fe.log_mel(sigs["speech"])
    np.testing.assert_array_equal(a, b)


def test_concurrent_threads(sigs):
    names = ["speech", "chirp", "white_noise", "sine_440"]
    expected = {n: asrfront.whisper.log_mel(sigs[n]) for n in names}
    errors = []

    def worker(name):
        for _ in range(5):
            if not np.array_equal(asrfront.whisper.log_mel(sigs[name]), expected[name]):
                errors.append(name)

    threads = [threading.Thread(target=worker, args=(n,)) for n in names * 2]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
