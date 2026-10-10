"""SIMD kernel variants (baseline vs. AVX2) must agree bit for bit."""

from __future__ import annotations

import numpy as np
import pytest

from asrfront import _ext

from audio_signals import SIGNAL_NAMES


def test_kernel_isa_reported():
    assert _ext.kernel_isa() in ("avx2", "generic")


def _frontends():
    return (
        _ext.WhisperFrontend(),
        _ext.WhisperFrontend(n_mels=128, pad_or_trim=False),
        _ext.SenseVoiceFrontend(),
        _ext.SenseVoiceFrontend(dither=1.0, seed=3),
    )


def _run(fes, audio):
    w80, w128, sv, svd = fes
    out = [w80.log_mel(audio), sv.fbank(audio), sv.fbank_raw(audio), svd.fbank_raw(audio)]
    if audio.size > 200:
        out.append(w128.log_mel(audio))
    return out


@pytest.mark.parametrize("name", SIGNAL_NAMES)
def test_generic_matches_default(sigs, name, monkeypatch):
    if _ext.kernel_isa() == "generic":
        pytest.skip("no SIMD variant on this CPU/build")
    default = _run(_frontends(), sigs[name])
    monkeypatch.setenv("ASRFRONT_KERNELS", "generic")
    generic = _run(_frontends(), sigs[name])
    for a, b in zip(default, generic):
        np.testing.assert_array_equal(a, b)
