"""Validate the float64 reference implementations in ``reference.py``.

Comparisons against the original float32 libraries (torch / torchaudio) need the ``ref``
dependency group (``uv sync --group ref``) and are skipped otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

from asrfront import _ext

import reference as ref
from audio_signals import DATA_DIR, SIGNAL_NAMES
from helpers import assert_close

MEL_NPZ = DATA_DIR / "whisper_mel_filters.npz"


# ---------------------------------------------------------------------------------------------
# Whisper
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_mels", [80, 128])
def test_slaney_filters_match_whisper_assets(n_mels):
    expected = ref.load_whisper_mel_filters(MEL_NPZ, n_mels)
    actual = ref.slaney_mel_filters(n_mels=n_mels)
    # The npz was produced by librosa in float32: agree to float32 rounding.
    assert_close(actual, expected, atol=1e-8, what=f"mel_{n_mels}")
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)


def test_hann_window_matches_torch():
    torch = pytest.importorskip("torch")
    assert_close(ref.hann_window_periodic(400), torch.hann_window(400).numpy(), 1e-6, "hann")


@pytest.mark.parametrize("pad_or_trim", [True, False])
@pytest.mark.parametrize("n_mels", [80, 128])
@pytest.mark.parametrize("name", SIGNAL_NAMES)
def test_whisper_reference_matches_torch(sigs, name, n_mels, pad_or_trim):
    pytest.importorskip("torch")
    audio = sigs[name]
    if not pad_or_trim and audio.shape[0] <= ref.WHISPER_N_FFT // 2:
        with pytest.raises(ValueError):
            ref.whisper_log_mel(audio, n_mels, pad_or_trim)
        return
    expected = ref.whisper_log_mel_torch(audio, n_mels, pad_or_trim, MEL_NPZ)
    actual = ref.whisper_log_mel(audio, n_mels, pad_or_trim)
    assert_close(actual, expected, atol=1e-4, what=name)


# ---------------------------------------------------------------------------------------------
# Kaldi fbank
# ---------------------------------------------------------------------------------------------


def test_hamming_window_matches_torch():
    torch = pytest.importorskip("torch")
    expected = torch.hamming_window(400, periodic=False, alpha=0.54, beta=0.46).numpy()
    assert_close(ref.hamming_window_symmetric(400), expected, 1e-6, "hamming")


def _torchaudio_banks() -> np.ndarray:
    from torchaudio.compliance import kaldi

    banks, _ = kaldi.get_mel_banks(80, 512, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0)
    return np.pad(banks.numpy(), ((0, 0), (0, 1)))


def test_kaldi_mel_banks_match_torchaudio():
    pytest.importorskip("torchaudio")
    # torchaudio builds the banks in float32 (mel values ~2800 lose ~1e-4 absolute), so the
    # float64 reference differs by up to ~1.4e-5.
    assert_close(ref.kaldi_mel_banks(80), _torchaudio_banks(), 3e-5, "kaldi mel banks")


def _torchaudio_fbank(audio: np.ndarray, dtype) -> np.ndarray:
    """Exactly what FunASR's WavFrontend calls (dither=0 for deterministic inference)."""
    import torch
    from torchaudio.compliance import kaldi

    waveform = torch.from_numpy(audio.astype(dtype) * dtype(1 << 15)).unsqueeze(0)
    return kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        energy_floor=0.0,
        window_type="hamming",
        sample_frequency=16000,
        snip_edges=True,
    ).numpy()


@pytest.mark.parametrize("name", SIGNAL_NAMES)
def test_kaldi_fbank_reference_is_exact_port(sigs, name):
    """With torchaudio's own filter bank and float64 input, the algorithms agree exactly."""
    pytest.importorskip("torchaudio")
    audio = sigs[name]
    actual = ref.kaldi_fbank_stages(audio, banks=_torchaudio_banks().astype(np.float64))["out"]
    if audio.shape[0] < ref.KALDI_FRAME_LENGTH:
        assert actual.shape == (0, 80)
        return
    assert_close(actual, _torchaudio_fbank(audio, np.float64), atol=1e-6, what=name)


@pytest.mark.parametrize("name", SIGNAL_NAMES)
def test_kaldi_fbank_reference_matches_torchaudio(sigs, name):
    """End to end vs. FunASR's float32 call. The residual (worst: chirp, ~1.3e-3 on
    low-energy bins) comes from torchaudio's float32 filter bank and FFT."""
    pytest.importorskip("torchaudio")
    audio = sigs[name]
    actual = ref.kaldi_fbank(audio)
    if audio.shape[0] < ref.KALDI_FRAME_LENGTH:
        assert actual.shape == (0, 80)
        return
    assert_close(actual, _torchaudio_fbank(audio, np.float32), atol=2e-3, what=name)


# ---------------------------------------------------------------------------------------------
# LFR / CMVN
# ---------------------------------------------------------------------------------------------


def test_apply_lfr_layout():
    t, dim = 14, 2
    x = np.arange(t * dim, dtype=np.float64).reshape(t, dim)  # row i == [2i, 2i+1]
    y = ref.apply_lfr(x, lfr_m=7, lfr_n=6)
    assert y.shape == (3, 14)  # ceil(14 / 6)
    rows = lambda *ix: np.concatenate([x[i] for i in ix])  # noqa: E731
    # 3 copies of frame 0 on the left, then frames 0..3.
    np.testing.assert_array_equal(y[0], rows(0, 0, 0, 0, 1, 2, 3))
    np.testing.assert_array_equal(y[1], rows(3, 4, 5, 6, 7, 8, 9))
    # Last LFR frame runs past the end: padded with the final frame.
    np.testing.assert_array_equal(y[2], rows(9, 10, 11, 12, 13, 13, 13))


def test_apply_lfr_empty():
    assert ref.apply_lfr(np.zeros((0, 80))).shape == (0, 560)


def test_load_cmvn(tmp_path):
    shift = np.linspace(-10, 10, 560, dtype=np.float32)
    scale = np.linspace(0.1, 2, 560, dtype=np.float32)
    fmt = lambda v: " ".join(f"{x:.6f}" for x in v)  # noqa: E731
    text = (
        "<Nnet>\n"
        "<Splice> 560 560\n"
        "[ 0 ]\n"
        "<AddShift> 560 560\n"
        f"<LearnRateCoef> 0 [ {fmt(shift)} ]\n"
        "<Rescale> 560 560\n"
        f"<LearnRateCoef> 0 [ {fmt(scale)} ]\n"
        "</Nnet>\n"
    )
    path = tmp_path / "am.mvn"
    path.write_text(text)
    got_shift, got_scale = ref.load_cmvn(path)
    assert_close(got_shift, shift, 1e-5, "shift")
    assert_close(got_scale, scale, 1e-6, "scale")


# ---------------------------------------------------------------------------------------------
# Frame-count helpers of the C core agree with the reference shapes
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", SIGNAL_NAMES)
def test_c_frame_counts_match_reference(sigs, name):
    audio = sigs[name]
    n = audio.shape[0]

    assert _ext.WhisperFrontend(pad_or_trim=True).out_frames(n) == 3000
    if n > ref.WHISPER_N_FFT // 2:
        frames = ref.whisper_log_mel(audio, pad_or_trim=False).shape[1]
        assert _ext.WhisperFrontend(pad_or_trim=False).out_frames(n) == frames

    sv = _ext.SenseVoiceFrontend()
    raw = ref.kaldi_fbank(audio)
    assert sv.raw_frames(n) == raw.shape[0]
    assert sv.out_frames(n) == ref.apply_lfr(raw).shape[0]
    assert sv.feat_dim == 560
