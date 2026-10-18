"""asrfront.qwen3_asr vs. the float64 reference and transformers' WhisperFeatureExtractor."""

from __future__ import annotations

import numpy as np
import pytest

import asrfront
from asrfront import qwen3_asr

import reference as ref
from audio_signals import SIGNAL_NAMES
from helpers import assert_close

ATOL = 1e-4


def _noise(n: int, seed: int) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(n) * 0.1).astype(np.float32)


# Batches around the frame / hop / reflection boundaries: equal lengths, one sample apart,
# paddings shorter than the 200-sample reflection, tiny and empty items next to long ones.
BATCHES = [
    (16000,),
    (16000, 16000),
    (16159, 16161),
    (16000, 16001, 16040, 16199, 16200, 16201, 16360),
    (1000, 16000),
    (16050, 16000, 300),
    (0, 1, 199, 201, 8000),
    (480000, 1234),
]


# ---------------------------------------------------------------------------------------------
# Single items
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", SIGNAL_NAMES)
def test_log_mel_matches_reference(sigs, name):
    audio = sigs[name]
    if audio.shape[0] <= 200:
        with pytest.raises(ValueError, match="more than 200 samples"):
            qwen3_asr.log_mel(audio)
        return
    actual = qwen3_asr.log_mel(audio)
    assert actual.dtype == np.float32 and actual.shape == (128, audio.shape[0] // 160)
    assert_close(actual, ref.whisper_log_mel(audio, 128, pad_or_trim=False), ATOL, name)


@pytest.mark.parametrize("name", ["speech", "chirp", "short_201", "odd_length"])
def test_log_mel_is_whisper_128_without_pad_or_trim(sigs, name):
    audio = sigs[name]
    expected = asrfront.whisper.log_mel(audio, n_mels=128, pad_or_trim=False)
    np.testing.assert_array_equal(qwen3_asr.log_mel(audio), expected)


# ---------------------------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("lens", [b for b in BATCHES if min(b) > 200])
@pytest.mark.parametrize("n_threads", [1, 4])
def test_batch_items_are_independent(lens, n_threads):
    audios = [_noise(n, i) for i, n in enumerate(lens)]
    feats, lengths = qwen3_asr.log_mel_batch(audios, n_threads=n_threads)
    assert feats.shape == (len(lens), 128, max(n // 160 for n in lens))
    assert lengths.dtype == np.int32 and list(lengths) == [n // 160 for n in lens]
    for a, f, n in zip(audios, feats, lengths):
        np.testing.assert_array_equal(f[:, :n], qwen3_asr.log_mel(a))
        assert not f[:, n:].any()


@pytest.mark.parametrize("lens", BATCHES)
def test_batch_hf_padding_matches_reference(lens):
    audios = [_noise(n, i) for i, n in enumerate(lens)]
    feats, lengths = qwen3_asr.log_mel_batch(audios, hf_padding=True)
    longest = max(lens)
    assert feats.shape == (len(lens), 128, longest // 160)
    assert list(lengths) == [min(-(-n // 160), longest // 160) for n in lens]
    for a, f in zip(audios, feats):
        assert_close(f, ref.whisper_log_mel_padded(a, 128, longest), ATOL, f"{len(a)}/{lens}")
        if len(a) == longest:  # the longest item is not padded at all
            np.testing.assert_array_equal(f, qwen3_asr.log_mel(a))


def test_batch_hf_padding_can_differ_from_single_items():
    # The second item's last frames see the first one's zero padding, not its reflection.
    audios = [_noise(16500, 0), _noise(16000, 1)]
    feats, lengths = qwen3_asr.log_mel_batch(audios, hf_padding=True)
    single = qwen3_asr.log_mel(audios[1])
    np.testing.assert_array_equal(feats[1, :, :98], single[:, :98])
    assert not np.array_equal(feats[1, :, :100], single)
    assert lengths[1] == 100


def test_batch_input_forms_and_errors(sigs):
    speech = sigs["speech"]
    feats, lengths = qwen3_asr.log_mel_batch(np.stack([speech, speech]))
    assert feats.shape[0] == 2 and (lengths == speech.shape[0] // 160).all()
    for hf_padding in (False, True):
        feats, lengths = qwen3_asr.log_mel_batch([], hf_padding=hf_padding)
        assert feats.shape == (0, 128, 0) and lengths.shape == (0,)
    with pytest.raises(ValueError, match=r"audios\[1\] needs more than 200"):
        qwen3_asr.log_mel_batch([speech, speech[:200]])
    with pytest.raises(ValueError, match="longest item needs more than 200"):
        qwen3_asr.log_mel_batch([speech[:100], speech[:200]], hf_padding=True)
    with pytest.raises(ValueError, match=r"audios\[0\] contains NaN"):
        qwen3_asr.log_mel_batch([np.full(1000, np.nan, np.float32)])


@pytest.mark.parametrize("hf_padding", [False, True])
def test_batch_out_buffer(hf_padding):
    audios = [_noise(n, i) for i, n in enumerate((16050, 480000, 300, 16000))]
    expected, lengths = qwen3_asr.log_mel_batch(audios, hf_padding=hf_padding)
    out = np.full(expected.shape, np.nan, np.float32)  # every value must be overwritten
    for n_threads in (1, 3):
        feats, lengths_out = qwen3_asr.log_mel_batch(
            audios, n_threads=n_threads, hf_padding=hf_padding, out=out
        )
        assert feats is out
        np.testing.assert_array_equal(feats, expected)
        np.testing.assert_array_equal(lengths_out, lengths)
        out[...] = np.nan


def test_batch_out_errors():
    audios = [_noise(16000, 0), _noise(8000, 1)]
    shape = (2, 128, 100)
    with pytest.raises(TypeError, match="float32 numpy array, got dtype float64"):
        qwen3_asr.log_mel_batch(audios, out=np.zeros(shape))
    with pytest.raises(TypeError, match="got list"):
        qwen3_asr.log_mel_batch(audios, out=[])
    with pytest.raises(
        ValueError, match=r"out has shape \(2, 128, 101\), expected \(2, 128, 100\)"
    ):
        qwen3_asr.log_mel_batch(audios, out=np.zeros((2, 128, 101), np.float32))
    with pytest.raises(ValueError, match="C-contiguous"):
        qwen3_asr.log_mel_batch(audios, out=np.zeros((2, 128, 200), np.float32)[:, :, ::2])
    read_only = np.zeros(shape, np.float32)
    read_only.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        qwen3_asr.log_mel_batch(audios, out=read_only)
    feats, _ = qwen3_asr.log_mel_batch([], out=np.zeros((0, 128, 0), np.float32))
    assert feats.shape == (0, 128, 0)


def test_log_mel_padded_low_level_errors():
    fe = asrfront._ext.WhisperFrontend(n_mels=128, pad_or_trim=False)
    audio = _noise(1000, 0)
    with pytest.raises(ValueError):
        fe.log_mel_padded(audio, 999)  # shorter than the audio
    with pytest.raises(ValueError):
        fe.log_mel_padded(audio[:10], 200)  # reflect padding needs more than 200 samples
    with pytest.raises(ValueError):
        asrfront._ext.WhisperFrontend(n_mels=128).log_mel_padded(audio, 2000)  # pad_or_trim


# ---------------------------------------------------------------------------------------------
# Against transformers (the qwen-asr pipeline)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["speech", "speech_35s", "chirp", "multi_tone", "short_201"])
def test_log_mel_matches_transformers(sigs, name):
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    audio = sigs[name]
    feats, mask = ref.qwen3_asr_transformers([audio])
    actual = qwen3_asr.log_mel(audio)
    assert feats.shape[1:] == actual.shape and mask[0] == actual.shape[1]
    assert_close(actual, feats[0], ATOL, name)


@pytest.mark.parametrize("lens", BATCHES)
def test_batch_matches_transformers(lens):
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    audios = [_noise(n, i) for i, n in enumerate(lens)]
    expected, mask = ref.qwen3_asr_transformers(audios)
    feats, lengths = qwen3_asr.log_mel_batch(audios, hf_padding=True)
    assert feats.shape == expected.shape
    np.testing.assert_array_equal(lengths, mask)
    assert_close(feats, expected, ATOL, str(lens))


# ---------------------------------------------------------------------------------------------
# Audio tokens
# ---------------------------------------------------------------------------------------------


def _published_output_lengths(n: np.ndarray) -> np.ndarray:
    """``_get_feat_extract_output_lengths`` of Qwen3-ASR's processor, as published."""
    leave = n % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (n // 100) * 13


def test_num_audio_tokens():
    n = np.arange(0, 20000)
    np.testing.assert_array_equal(qwen3_asr.num_audio_tokens(n), _published_output_lengths(n))
    assert qwen3_asr.num_audio_tokens(3000) == 390 and isinstance(
        qwen3_asr.num_audio_tokens(np.int32(7)), int
    )
    assert qwen3_asr.num_audio_tokens(np.array([[0, 100]], np.int32)).tolist() == [[0, 13]]
    with pytest.raises(ValueError):
        qwen3_asr.num_audio_tokens(-1)
    with pytest.raises(ValueError):
        qwen3_asr.num_audio_tokens([5, -1])
    with pytest.raises(TypeError):
        qwen3_asr.num_audio_tokens([1.5])
    with pytest.raises(TypeError):
        qwen3_asr.num_audio_tokens(True)
