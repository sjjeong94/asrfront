"""LFM3-Audio (LiquidAI/LFM3-Audio-2B) audio input features.

The model's ``Lfm3AudioFeatureExtractor`` computes a Whisper-like log-mel spectrogram: 25 ms
periodic Hann windows every 10 ms **without centering**, the power spectrum, 128 Slaney mel
filters, ``max(log10(mel), -12)`` (a fixed floor instead of Whisper's "max - 8") and
``(x + 4) / 4``. Every clip is right-padded with zeros to whole windows, so no samples are
dropped, and is computed on its own: a batch only zero-pads the features.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, overload

import numpy as np
import numpy.typing as npt

from . import _ext
from ._audio import as_audio, cached_frontend, check_audio_list, check_int, run_batch

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
ENCODER_STRIDE = 8  # feature frames per encoder token

__all__ = [
    "log_mel",
    "log_mel_batch",
    "num_frames",
    "num_audio_tokens",
    "LogMelStream",
    "SAMPLE_RATE",
    "N_FFT",
    "HOP_LENGTH",
    "N_MELS",
    "ENCODER_STRIDE",
]


def _frontend() -> _ext.Lfm3AudioFrontend:
    return cached_frontend(("lfm3_audio",), _ext.Lfm3AudioFrontend)


def num_frames(n_samples: int) -> int:
    """Feature frames of an input of ``n_samples`` samples, right-padded to whole windows:
    ``1 + ceil((n - 400) / 160)``, and 1 for ``n <= 400`` (the extractor's ``num_frames``)."""
    return int(_ext.Lfm3AudioFrontend.out_frames(check_int("n_samples", n_samples, 0)))


def log_mel(audio: Any, channel_last: bool = False) -> np.ndarray:
    """Compute LFM3-Audio's input features for one waveform.

    Equivalent to ``Lfm3AudioFeatureExtractor(...)(audio)["input_features"][0]`` with the
    LFM3-Audio-2B checkpoint's settings.

    Args:
        audio: 16 kHz mono waveform: float array in [-1, 1], int16 PCM, list, or CPU
            torch tensor. Any length, including empty (one frame of zeros' features).
        channel_last: return ``(T, 128)`` (time first, C-contiguous) instead, written in that
            layout by the native code.

    Returns:
        float32 array ``(128, T)``, ``T = num_frames(len(audio))``, or its transpose with
        ``channel_last``.

    Raises:
        ValueError: ``audio`` is not 1-D or contains NaN / inf.
        TypeError: ``audio`` has an unsupported dtype.
    """
    audio = as_audio(audio)
    n_frames = _ext.Lfm3AudioFrontend.out_frames(audio.shape[0])
    return _frontend().log_mel(audio, n_frames, channel_last=bool(channel_last))


def log_mel_batch(
    audios: Sequence[Any] | np.ndarray, n_threads: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute :func:`log_mel` for many waveforms in parallel, padded to a common length.

    Equivalent to one ``Lfm3AudioFeatureExtractor`` call on the batch: every item's features
    are computed on their own and zero-padded (``padding_value=0``) to the most frames.

    Args:
        audios: sequence of 1-D waveforms (any lengths), or a 2-D ``(B, N)`` array.
        n_threads: at most this many worker threads (fewer for little audio); ``None``: all
            available cores.

    Returns:
        ``(features, lengths)``: float32 ``(B, 128, T)`` with ``log_mel(audios[i])`` in
        ``features[i, :, :lengths[i]]`` and zeros after it, and int32 ``(B,)`` frames per item
        (the ``feature_attention_mask`` sums; the mask is ``np.arange(T) < lengths[:, None]``).
    """
    items = check_audio_list(audios)
    lengths = np.array(
        [_ext.Lfm3AudioFrontend.out_frames(a.shape[0]) for a in items], dtype=np.int32
    )
    n_frames = int(lengths.max()) if items else 0
    feats = np.zeros((len(items), N_MELS, n_frames), np.float32)  # zeros after each item

    def work(chunk: range, audios: list[np.ndarray]) -> int:
        # rows of feats[i], no copy
        return _frontend().log_mel_many(audios, [feats[i, :, : lengths[i]] for i in chunk])

    run_batch(work, items, n_threads)
    return feats, lengths


def _tokens(n: Any) -> Any:  # ints or int64 arrays
    return -(-n // ENCODER_STRIDE)


@overload
def num_audio_tokens(n_frames: int) -> int: ...
@overload
def num_audio_tokens(n_frames: npt.ArrayLike) -> np.ndarray: ...
def num_audio_tokens(n_frames: Any) -> Any:
    """Number of audio tokens the encoder produces from ``n_frames`` feature frames.

    Three stride-2 convolutions turn every 8 frames into one token: ``ceil(n / 8)``, as
    ``Lfm3AudioProcessor`` counts the audio placeholder tokens of a prompt.

    Args:
        n_frames: frame count(s), e.g. the ``lengths`` of :func:`log_mel_batch` or
            :func:`num_frames`.
    """
    if isinstance(n_frames, (int, np.integer)) and not isinstance(n_frames, bool):
        return int(_tokens(check_int("n_frames", n_frames, 0)))
    arr = np.asarray(n_frames)
    if arr.dtype.kind not in "iu":
        raise TypeError(f"n_frames must be integers, got dtype {arr.dtype}")
    arr = arr.astype(np.int64)
    if (arr < 0).any():
        raise ValueError("n_frames must be >= 0")
    return _tokens(arr)


class LogMelStream:
    """Streaming :func:`log_mel`: push audio in pieces, get each frame once its window is full.

    The frames returned by all :meth:`push` calls and the :meth:`finish` of an utterance are
    exactly ``log_mel(concatenated audio)``, however the audio is split (the extractor's
    ``stream()``, with ``is_final=True`` on the last chunk). A frame is returned as soon as its
    25 ms window has arrived; :meth:`finish` zero-pads the last window. At most 399 samples are
    kept between pushes. Not thread-safe: use one object per stream.

    Example::

        stream = lfm3_audio.LogMelStream()
        for chunk in microphone_chunks():  # 16 kHz mono, any sizes
            frames = stream.push(chunk)  # (128, k), k >= 0
            ...
        frames = stream.finish()  # the last frames; the stream starts over

    Args:
        channel_last: return ``(k, 128)`` frames instead of ``(128, k)``.
    """

    def __init__(self, channel_last: bool = False) -> None:
        self._channel_last = bool(channel_last)
        self.reset()

    def reset(self) -> None:
        """Drop the audio pushed since the last :meth:`finish` and start over."""
        self._tail = np.zeros(0, np.float32)  # samples from the next frame's start on
        self._n_samples = 0
        self._n_frames = 0

    def _frames(self, audio: np.ndarray, count: int) -> np.ndarray:
        return _frontend().log_mel(audio, count, channel_last=self._channel_last)

    def push(self, audio: Any) -> np.ndarray:
        """Append 16 kHz mono audio (as :func:`log_mel` takes it); returns the float32
        ``(128, k)`` frames it completes (``k`` may be 0). A chunk with NaN / inf raises
        ``ValueError`` and is not appended."""
        chunk = as_audio(audio)
        buf = np.concatenate((self._tail, chunk)) if self._tail.size else chunk
        count = int(_ext.Lfm3AudioFrontend.complete_frames(buf.shape[0]))
        out = self._frames(buf, count)
        self._tail = buf[count * HOP_LENGTH :].copy()
        self._n_samples += chunk.shape[0]
        self._n_frames += count
        return out

    def finish(self) -> np.ndarray:
        """End the utterance: the remaining frames, the last one zero-padded (at least one frame
        for the utterance, as :func:`log_mel` gives), then the stream starts over."""
        count = int(_ext.Lfm3AudioFrontend.out_frames(self._n_samples)) - self._n_frames
        out = self._frames(self._tail, count)
        self.reset()
        return out
