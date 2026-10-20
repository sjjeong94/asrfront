"""asrfront.gemma4 vs. the float64 reference and transformers' Gemma4AudioFeatureExtractor."""

from __future__ import annotations

import numpy as np
import pytest

from asrfront import gemma4

import reference as ref
from audio_signals import SIGNAL_NAMES

# Float32 FFT round-off is ~1e-6 of a frame's peak magnitude in every bin; in bins far below the
# 0.001 mel floor, log(mel + 0.001) turns that into up to ~1e-2 (around log(0.001) = -6.9).
# Above the floor the features are as close as the other frontends'.
ATOL = 1e-2
ATOL_ABOVE_FLOOR = 5e-4  # where the reference's mel energy is >= 0.01


def _noise(n: int, seed: int) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(n) * 0.1).astype(np.float32)


def _frames(n_longest: int) -> int:
    padded = -(-n_longest // 128) * 128
    return max(0, (padded - 161) // 160 + 1)


# Lengths around the 128-sample padding, the 160-sample hop and the 161-sample first frame,
# empty items, and items past 30 s.
BATCHES = [
    (16000,),
    (16000, 16001),
    (0,),
    (1, 128, 129, 160),
    (161, 162, 320, 321, 481),
    (16000, 300, 0, 16159, 16161),
    (480000, 1234),
    (500000, 16000),
]


def _check_against_reference(actual: np.ndarray, audio: np.ndarray, name: str) -> None:
    st = ref.gemma4_log_mel_stages(audio[: gemma4.MAX_SAMPLES])
    expected = st["out"]
    assert actual.dtype == np.float32 and actual.shape == expected.shape, name
    if actual.size == 0:
        return
    err = np.abs(actual - expected)
    assert err.max() <= ATOL, f"{name}: max abs error {err.max():.2e}"
    above = st["mel"] >= 1e-2
    if above.any():
        worst = err[above].max()
        assert worst <= ATOL_ABOVE_FLOOR, f"{name}: max abs error above the floor {worst:.2e}"


# ---------------------------------------------------------------------------------------------
# Single items
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", SIGNAL_NAMES)
def test_log_mel_matches_reference(sigs, name):
    audio = sigs[name]
    actual = gemma4.log_mel(audio)
    n = min(audio.shape[0], gemma4.MAX_SAMPLES)
    assert actual.shape == (_frames(n), 128)
    _check_against_reference(actual, audio, name)


@pytest.mark.parametrize("n", [0, 1, 128, 129, 160, 161, 162, 320, 321, 16000, 16001, 16123])
def test_lengths_and_padding_frames(n):
    audio = _noise(n, n)
    out = gemma4.log_mel(audio)
    valid = gemma4.num_frames(n)
    assert out.shape == (_frames(n), 128)
    assert valid == (max(n - 1, 0)) // 160 and valid <= out.shape[0]
    assert not out[valid:].any()  # the padding's frames are masked to zero
    assert (out[:valid] < 0).any() or valid == 0
    _check_against_reference(out, audio, f"n={n}")


def test_truncation(sigs):
    long = sigs["speech_35s"]
    assert long.shape[0] > gemma4.MAX_SAMPLES
    out = gemma4.log_mel(long)
    np.testing.assert_array_equal(out, gemma4.log_mel(long[: gemma4.MAX_SAMPLES]))
    assert out.shape == (2999, 128) and gemma4.num_frames(long.shape[0]) == 2999
    full = gemma4.log_mel(long, truncation=False)
    assert full.shape == (_frames(long.shape[0]), 128)
    assert gemma4.num_frames(long.shape[0], truncation=False) == (long.shape[0] - 1) // 160
    np.testing.assert_array_equal(full[:2999], out)  # causal: earlier frames do not change
    with_nan = long.astype(np.float64)
    with_nan[gemma4.MAX_SAMPLES] = np.nan  # dropped by the truncation: not an error
    np.testing.assert_array_equal(gemma4.log_mel(with_nan), out)
    with pytest.raises(ValueError, match="NaN or inf"):
        gemma4.log_mel(with_nan, truncation=False)


def test_input_types(sigs):
    audio = sigs["speech"][:16000]
    expected = gemma4.log_mel(audio)
    np.testing.assert_array_equal(gemma4.log_mel(audio.astype(np.float64)), expected)
    np.testing.assert_array_equal(gemma4.log_mel(audio.tolist()), expected)
    pcm = (audio * 32767).astype(np.int16)
    np.testing.assert_array_equal(gemma4.log_mel(pcm), gemma4.log_mel(pcm / np.float32(32768)))
    with pytest.raises(ValueError, match="1-D"):
        gemma4.log_mel(np.zeros((2, 1000), np.float32))
    with pytest.raises(TypeError, match="integer dtype"):
        gemma4.log_mel(np.zeros(1000, np.int32))
    with pytest.raises(ValueError, match="NaN or inf.*index 7"):
        bad = audio.copy()
        bad[7] = np.inf
        gemma4.log_mel(bad)


def test_window_and_filters():
    """The reference's stages agree with transformers' formulas (checked independently of the C
    code): HTK filters with one empty lowest filter, and the semicausal framing."""
    filters = ref.htk_mel_filters()
    assert filters.shape == (128, 257)
    assert (filters.max(axis=1) == 0).sum() == 1  # transformers warns about it too
    st = ref.gemma4_log_mel_stages(np.arange(1, 641, dtype=np.float64))
    win = ref.hann_window_periodic(320)
    np.testing.assert_allclose(st["frames"][0][160:] / win[160:], np.arange(1, 161))
    assert not st["frames"][0][:160].any()  # 160 zeros before the first sample


# ---------------------------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("lens", BATCHES, ids=lambda b: "-".join(map(str, b)))
@pytest.mark.parametrize("n_threads", [1, 3])
def test_batch(lens, n_threads):
    audios = [_noise(n, i) for i, n in enumerate(lens)]
    feats, lengths = gemma4.log_mel_batch(audios, n_threads=n_threads)
    n_longest = min(max(lens), gemma4.MAX_SAMPLES)
    T = _frames(n_longest)
    assert feats.shape == (len(lens), T, 128) and feats.dtype == np.float32
    assert lengths.dtype == np.int32
    assert lengths.tolist() == [gemma4.num_frames(n) for n in lens]
    for i, a in enumerate(audios):
        single = gemma4.log_mel(a)
        valid = lengths[i]
        # the valid frames read no padding: the same as alone, whatever the batch
        np.testing.assert_array_equal(feats[i, :valid], single[:valid])
        assert not feats[i, valid:].any()
        expected = ref.gemma4_log_mel(a[: gemma4.MAX_SAMPLES], -(-n_longest // 128) * 128)
        assert np.abs(feats[i] - expected).max(initial=0.0) <= ATOL


def test_batch_empty_and_2d():
    feats, lengths = gemma4.log_mel_batch([])
    assert feats.shape == (0, 0, 128) and lengths.shape == (0,)
    batch = np.stack([_noise(8000, 0), _noise(8000, 1)])
    feats, lengths = gemma4.log_mel_batch(batch)
    np.testing.assert_array_equal(feats[1], gemma4.log_mel(batch[1]))


def test_batch_errors():
    with pytest.raises(ValueError, match=r"audios\[1\] contains NaN or inf"):
        gemma4.log_mel_batch([_noise(1000, 0), np.full(1000, np.nan, np.float32)])
    with pytest.raises(TypeError, match=r"audios\[0\] has integer dtype"):
        gemma4.log_mel_batch([np.zeros(10, np.int32)])
    with pytest.raises(ValueError, match="n_threads"):
        gemma4.log_mel_batch([_noise(1000, 0)], n_threads=0)


# ---------------------------------------------------------------------------------------------
# Token counts
# ---------------------------------------------------------------------------------------------


def _processor_tokens(n_samples: int) -> int:
    """Gemma4Processor._compute_audio_num_tokens (transformers 5.x): mel framing, two stride-2
    convolutions (kernel 3, padding 1), at most audio_seq_length = 750."""
    t = (n_samples + 160 - 321) // 160 + 1
    if t <= 0:
        return 0
    for _ in range(2):
        t = (t + 2 - 3) // 2 + 1
    return min(t, 750)


def _mask_tokens(mask: np.ndarray) -> int:
    """Gemma4Processor.replace_audio_token: the feature mask subsampled twice."""
    for _ in range(2):
        t_out = (len(mask) + 2 - 3) // 2 + 1
        mask = mask[::2][:t_out]
    return int(mask.sum())


@pytest.mark.parametrize("n", [0, 1, 160, 161, 320, 321, 481, 640, 641, 16000, 123457, 480000])
def test_num_audio_tokens(n):
    frames = gemma4.num_frames(n)
    T = _frames(n)
    mask = np.arange(T) < frames
    assert gemma4.num_audio_tokens(frames) == _processor_tokens(n) == _mask_tokens(mask)


def test_num_audio_tokens_arrays_and_errors():
    out = gemma4.num_audio_tokens(np.array([0, 1, 4, 5, 2999]))
    assert out.tolist() == [0, 1, 1, 2, 750]
    assert gemma4.num_audio_tokens(2999) == 750 and isinstance(gemma4.num_audio_tokens(9), int)
    with pytest.raises(ValueError):
        gemma4.num_audio_tokens(-1)
    with pytest.raises(TypeError):
        gemma4.num_audio_tokens(np.array([1.5]))


# ---------------------------------------------------------------------------------------------
# Against transformers (needs transformers >= 5.5, which the `qwen` group's pin excludes)
# ---------------------------------------------------------------------------------------------


def _transformers():
    pytest.importorskip("transformers.models.gemma4.feature_extraction_gemma4")


@pytest.mark.parametrize("name", SIGNAL_NAMES)
def test_log_mel_matches_transformers(sigs, name):
    _transformers()
    audio = sigs[name]
    feats, mask = ref.gemma4_transformers([audio])
    actual = gemma4.log_mel(audio)
    assert actual.shape == feats[0].shape and gemma4.num_frames(audio.shape[0]) == mask[0]
    assert np.abs(actual - feats[0]).max(initial=0.0) <= ATOL


@pytest.mark.parametrize("lens", BATCHES, ids=lambda b: "-".join(map(str, b)))
def test_batch_matches_transformers(lens):
    _transformers()
    audios = [_noise(n, i) for i, n in enumerate(lens)]
    expected, mask = ref.gemma4_transformers(audios)
    feats, lengths = gemma4.log_mel_batch(audios)
    assert feats.shape == expected.shape and lengths.tolist() == mask.tolist()
    assert np.abs(feats - expected).max(initial=0.0) <= ATOL
