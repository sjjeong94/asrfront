"""asrfront.sensevoice vs. the float64 reference."""

from __future__ import annotations

import numpy as np
import pytest

import asrfront

import reference as ref
from audio_signals import SIGNAL_NAMES
from helpers import assert_close

ATOL = 1e-3  # log domain

# Remove once M5 lands; strict so an unexpected pass flags the stale marker.
pending_m5 = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="M5")


def _synthetic_cmvn(dim: int = 560) -> tuple[np.ndarray, np.ndarray]:
    """Shift/scale in the same range as SenseVoiceSmall's am.mvn (mean ~ -10, 1/std ~ 0.3)."""
    rng = np.random.default_rng(0)
    shift = (-10.0 + rng.standard_normal(dim)).astype(np.float32)
    scale = (0.3 + 0.05 * rng.random(dim)).astype(np.float32)
    return shift, scale


@pending_m5
@pytest.mark.parametrize("name", SIGNAL_NAMES)
def test_fbank_raw_matches_reference(sigs, name):
    audio = sigs[name]
    actual = asrfront.sensevoice.fbank_raw(audio)
    assert actual.dtype == np.float32
    assert_close(actual, ref.kaldi_fbank(audio), ATOL, name)


@pending_m5
@pytest.mark.parametrize("name", SIGNAL_NAMES)
def test_fbank_without_cmvn_matches_reference(sigs, name):
    audio = sigs[name]
    assert_close(asrfront.sensevoice.fbank(audio), ref.sensevoice_fbank(audio), ATOL, name)


@pending_m5
@pytest.mark.parametrize("name", SIGNAL_NAMES)
def test_fbank_with_cmvn_matches_reference(sigs, name):
    audio = sigs[name]
    cmvn = _synthetic_cmvn()
    actual = asrfront.sensevoice.fbank(audio, cmvn=cmvn)
    # CMVN scales by ~0.35, so the log-domain tolerance carries over.
    assert_close(actual, ref.sensevoice_fbank(audio, cmvn), ATOL, name)
