"""asrfront.lfm3_audio vs. the float64 reference and a torch port of Lfm3AudioFeatureExtractor."""

from __future__ import annotations

import numpy as np
import pytest

from asrfront import lfm3_audio

import reference as ref
from audio_signals import SIGNAL_NAMES

# The -12 floor of log10(mel) is ~120 dB below full scale, so bins far below a frame's peak are
# not floored as in Whisper (max - 8) and show the float32 FFT's round-off (~1e-7 of the peak
# magnitude in every bin): up to ~0.1 where the energy is 1e-10 of the frame's. torch's float32
# STFT differs from float64 as much. Within 80 dB of the peak the features are as close as the
# other frontends'.
ATOL = 0.15
ATOL_NEAR_PEAK = 2e-4  # where the reference's mel energy is >= 1e-8 of the frame's largest


def _noise(n: int, seed: int) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(n) * 0.1).astype(np.float32)


def _check(actual: np.ndarray, audio: np.ndarray, name: str, expected=None) -> None:
    st = ref.lfm3_audio_log_mel_stages(audio)
    expected = st["out"] if expected is None else expected
    assert actual.dtype == np.float32 and actual.shape == expected.shape, name
    err = np.abs(actual - expected)
    assert err.max() <= ATOL, f"{name}: max abs error {err.max():.2e}"
    mel = st["mel"]
    near = mel >= 1e-8 * mel.max(axis=0, keepdims=True)
    if near.any():
        worst = err[near].max()
        assert worst <= ATOL_NEAR_PEAK, f"{name}: max abs error near the peak {worst:.2e}"


# Lengths around the 400-sample window and the 160-sample hop, empty items, long items.
BATCHES = [
    (16000,),
    (16000, 16001),
    (0,),
    (1, 399, 400, 401),
    (559, 560, 561, 720, 721),
    (16000, 300, 0, 16159, 16161),
    (480000, 1234),
]


# ---------------------------------------------------------------------------------------------
# Single items
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", SIGNAL_NAMES)
def test_log_mel_matches_reference(sigs, name):
    audio = sigs[name]
    actual = lfm3_audio.log_mel(audio)
    assert actual.shape == (128, ref.lfm3_audio_num_frames(audio.shape[0]))
    _check(actual, audio, name)


@pytest.mark.parametrize("n", [0, 1, 159, 160, 399, 400, 401, 559, 560, 561, 16000, 16123])
def test_lengths(n):
    audio = _noise(n, n)
    out = lfm3_audio.log_mel(audio)
    assert lfm3_audio.num_frames(n) == ref.lfm3_audio_num_frames(n) == out.shape[1]
    assert out.shape[1] == (1 if n <= 400 else -(-(n - 400) // 160) + 1)
    _check(out, audio, f"n={n}")


def test_silence_and_floor():
    out = lfm3_audio.log_mel(np.zeros(1000, np.float32))
    np.testing.assert_array_equal(out, np.full((128, 5), -2.0, np.float32))  # (-12 + 4) / 4
    loud = lfm3_audio.log_mel(_noise(4000, 0))
    assert loud.min() >= -2.0 and loud.max() < 2.0


def test_channel_last(sigs):
    for n in [0, 400, 1000, 16123]:
        audio = sigs["speech"][:n]
        cl = lfm3_audio.log_mel(audio, channel_last=True)
        assert cl.flags.c_contiguous and cl.shape == (lfm3_audio.num_frames(n), 128)
        np.testing.assert_array_equal(cl, lfm3_audio.log_mel(audio).T)


def test_frames_are_independent(sigs):
    """Frame t reads only samples [160 t, 160 t + 400): a prefix gives the same frames."""
    audio = sigs["speech"]
    full = lfm3_audio.log_mel(audio)
    for n in [400, 560, 16000, 16001]:
        part = lfm3_audio.log_mel(audio[:n])
        complete = (n - 400) // 160 + 1
        np.testing.assert_array_equal(part[:, :complete], full[:, :complete])


def test_input_types(sigs):
    audio = sigs["speech"][:16000]
    expected = lfm3_audio.log_mel(audio)
    np.testing.assert_array_equal(lfm3_audio.log_mel(audio.astype(np.float64)), expected)
    np.testing.assert_array_equal(lfm3_audio.log_mel(audio.tolist()), expected)
    pcm = (audio * 32767).astype(np.int16)
    np.testing.assert_array_equal(
        lfm3_audio.log_mel(pcm), lfm3_audio.log_mel(pcm / np.float32(32768))
    )
    with pytest.raises(ValueError, match="1-D"):
        lfm3_audio.log_mel(np.zeros((2, 1000), np.float32))
    with pytest.raises(TypeError, match="integer dtype"):
        lfm3_audio.log_mel(np.zeros(1000, np.int32))
    with pytest.raises(ValueError, match="NaN or inf.*index 7"):
        bad = audio.copy()
        bad[7] = np.inf
        lfm3_audio.log_mel(bad)
    with pytest.raises(ValueError):
        lfm3_audio.num_frames(-1)


# ---------------------------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("lens", BATCHES, ids=lambda b: "-".join(map(str, b)))
@pytest.mark.parametrize("n_threads", [1, 3])
def test_batch(lens, n_threads):
    audios = [_noise(n, i) for i, n in enumerate(lens)]
    feats, lengths = lfm3_audio.log_mel_batch(audios, n_threads=n_threads)
    T = max(ref.lfm3_audio_num_frames(n) for n in lens)
    assert feats.shape == (len(lens), 128, T) and feats.dtype == np.float32
    assert lengths.dtype == np.int32
    assert lengths.tolist() == [lfm3_audio.num_frames(n) for n in lens]
    for i, a in enumerate(audios):
        np.testing.assert_array_equal(feats[i, :, : lengths[i]], lfm3_audio.log_mel(a))
        assert not feats[i, :, lengths[i] :].any()


def test_batch_empty_and_2d():
    feats, lengths = lfm3_audio.log_mel_batch([])
    assert feats.shape == (0, 128, 0) and lengths.shape == (0,)
    batch = np.stack([_noise(8000, 0), _noise(8000, 1)])
    feats, lengths = lfm3_audio.log_mel_batch(batch)
    np.testing.assert_array_equal(feats[1], lfm3_audio.log_mel(batch[1]))


def test_batch_errors():
    with pytest.raises(ValueError, match=r"audios\[1\] contains NaN or inf"):
        lfm3_audio.log_mel_batch([_noise(1000, 0), np.full(1000, np.nan, np.float32)])
    with pytest.raises(TypeError, match=r"audios\[0\] has integer dtype"):
        lfm3_audio.log_mel_batch([np.zeros(10, np.int32)])
    with pytest.raises(ValueError, match="n_threads"):
        lfm3_audio.log_mel_batch([_noise(1000, 0)], n_threads=0)


# ---------------------------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 399, 400, 401, 560, 16000, 16123])
@pytest.mark.parametrize("chunk", [1, 159, 160, 400, 1000, 100000])
@pytest.mark.parametrize("channel_last", [False, True])
def test_stream_matches_log_mel(n, chunk, channel_last):
    audio = _noise(n, 7)
    stream = lfm3_audio.LogMelStream(channel_last=channel_last)
    axis = 0 if channel_last else 1
    for _ in range(2):  # finish() starts the stream over
        parts = [stream.push(audio[i : i + chunk]) for i in range(0, n, chunk)]
        parts.append(stream.finish())
        out = np.concatenate(parts, axis=axis)
        np.testing.assert_array_equal(out, lfm3_audio.log_mel(audio, channel_last=channel_last))


def test_stream_emits_frames_once_complete():
    stream = lfm3_audio.LogMelStream()
    assert stream.push(_noise(399, 0)).shape == (128, 0)
    assert stream.push(_noise(1, 1)).shape == (128, 1)  # 400 samples: the first window
    assert stream.push(_noise(159, 2)).shape == (128, 0)
    assert stream.push(_noise(1, 3)).shape == (128, 1)
    assert stream.finish().shape == (128, 0)  # 560 samples: exactly 2 frames, no padding
    assert stream.finish().shape == (128, 1)  # an empty utterance: one frame, as log_mel


def test_stream_errors_and_reset():
    stream = lfm3_audio.LogMelStream()
    stream.push(_noise(300, 0))
    with pytest.raises(ValueError, match="NaN or inf"):
        stream.push(np.full(10, np.nan, np.float32))
    audio = _noise(1000, 1)
    stream.reset()
    out = np.concatenate([stream.push(audio), stream.finish()], axis=1)
    np.testing.assert_array_equal(out, lfm3_audio.log_mel(audio))


# ---------------------------------------------------------------------------------------------
# Token counts
# ---------------------------------------------------------------------------------------------


def test_num_audio_tokens():
    out = lfm3_audio.num_audio_tokens(np.array([0, 1, 8, 9, 16, 99]))
    assert out.tolist() == [0, 1, 1, 2, 2, 13]
    assert lfm3_audio.num_audio_tokens(lfm3_audio.num_frames(16000)) == 13
    assert isinstance(lfm3_audio.num_audio_tokens(np.int32(9)), int)
    with pytest.raises(ValueError):
        lfm3_audio.num_audio_tokens(-1)
    with pytest.raises(TypeError):
        lfm3_audio.num_audio_tokens(np.array([1.5]))


# ---------------------------------------------------------------------------------------------
# Against the torch port of the extractor (the `ref` group)
# ---------------------------------------------------------------------------------------------


def _torch():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")


@pytest.mark.parametrize("name", SIGNAL_NAMES)
def test_log_mel_matches_torch(sigs, name):
    _torch()
    audio = sigs[name]
    feats, lengths = ref.lfm3_audio_torch([audio])
    assert lengths[0] == lfm3_audio.num_frames(audio.shape[0])
    _check(lfm3_audio.log_mel(audio), audio, name, feats[0])


@pytest.mark.parametrize("lens", BATCHES, ids=lambda b: "-".join(map(str, b)))
def test_batch_matches_torch(lens):
    _torch()
    audios = [_noise(n, i) for i, n in enumerate(lens)]
    expected, mask = ref.lfm3_audio_torch(audios)
    feats, lengths = lfm3_audio.log_mel_batch(audios)
    assert feats.shape == expected.shape and lengths.tolist() == mask.tolist()
    assert np.abs(feats - expected).max(initial=0.0) <= ATOL
