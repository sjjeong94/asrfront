"""asrfront.whisper vs. the float64 reference."""

from __future__ import annotations

import numpy as np
import pytest

import asrfront

import reference as ref
from audio_signals import SIGNAL_NAMES
from helpers import assert_close

ATOL = 1e-4

# Remove once M4 lands; strict so an unexpected pass flags the stale marker.
pending_m4 = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="M4")


@pending_m4
@pytest.mark.parametrize("pad_or_trim", [True, False])
@pytest.mark.parametrize("n_mels", [80, 128])
@pytest.mark.parametrize("name", SIGNAL_NAMES)
def test_log_mel_matches_reference(sigs, name, n_mels, pad_or_trim):
    audio = sigs[name]
    if not pad_or_trim and audio.shape[0] <= ref.WHISPER_N_FFT // 2:
        pytest.skip("too short for reflect padding (behavior defined in M4)")
    actual = asrfront.whisper.log_mel(audio, n_mels=n_mels, pad_or_trim=pad_or_trim)
    expected = ref.whisper_log_mel(audio, n_mels, pad_or_trim)
    assert actual.dtype == np.float32
    assert_close(actual, expected, ATOL, name)


@pending_m4
def test_int16_input_is_scaled(sigs):
    audio = sigs["speech"]
    pcm = np.round(audio * 32768).clip(-32768, 32767).astype(np.int16)
    expected = ref.whisper_log_mel(pcm.astype(np.float64) / 32768.0)
    assert_close(asrfront.whisper.log_mel(pcm), expected, ATOL, "int16")
