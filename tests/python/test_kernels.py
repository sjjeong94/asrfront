"""SIMD kernel variants (baseline vs. AVX2) must agree bit for bit."""

from __future__ import annotations

import numpy as np
import pytest

from asrfront import _ext

from audio_signals import SIGNAL_NAMES


def test_kernel_isa_reported(monkeypatch):
    assert _ext.kernel_isa() in ("avx2", "generic", "scalar")
    for isa in ("generic", "scalar"):
        monkeypatch.setenv("ASRFRONT_KERNELS", isa)
        assert _ext.kernel_isa() == isa


def _frontends():
    return (
        _ext.WhisperFrontend(),
        _ext.WhisperFrontend(n_mels=128, pad_or_trim=False),
        _ext.SenseVoiceFrontend(),
        _ext.SenseVoiceFrontend(dither=1.0, seed=3),
        _ext.Gemma4Frontend(),
    )


def _run(fes, audio):
    w80, w128, sv, svd, g4 = fes
    out = [w80.log_mel(audio), sv.fbank(audio), sv.fbank_raw(audio), svd.fbank_raw(audio)]
    out.append(g4.log_mel(audio, g4.out_frames(audio.size + 127)))
    if audio.size > 200:
        out.append(w128.log_mel(audio))
    out.append(w128.log_mel_padded(audio, audio.size + 333))  # qwen3_asr hf_padding batches
    return out


@pytest.mark.parametrize("isa", ["generic", "scalar"])
@pytest.mark.parametrize("name", SIGNAL_NAMES)
def test_variant_matches_default(sigs, name, isa, monkeypatch):
    if _ext.kernel_isa() == isa:
        pytest.skip(f"{isa} is already the default on this CPU/build")
    default = _run(_frontends(), sigs[name])
    monkeypatch.setenv("ASRFRONT_KERNELS", isa)
    other = _run(_frontends(), sigs[name])
    for a, b in zip(default, other):
        np.testing.assert_array_equal(a, b)
