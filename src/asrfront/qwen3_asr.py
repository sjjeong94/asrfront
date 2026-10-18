"""Qwen3-ASR (0.6B / 1.7B) input features.

The models use transformers' ``WhisperFeatureExtractor`` with 128 mel bins, called with
``padding=True`` and ``truncation=False``: Whisper's log-mel spectrogram over the whole input,
without padding or trimming to 30 s.
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

__all__ = [
    "log_mel",
    "log_mel_batch",
    "num_audio_tokens",
    "SAMPLE_RATE",
    "N_FFT",
    "HOP_LENGTH",
    "N_MELS",
]

_MIN_SAMPLES = N_FFT // 2 + 1  # the centered STFT's reflect padding needs more than 200


def _frontend() -> _ext.WhisperFrontend:
    return cached_frontend(
        ("whisper", N_MELS, False),  # the same handle as whisper.log_mel(n_mels=128, ...)
        lambda: _ext.WhisperFrontend(n_mels=N_MELS, pad_or_trim=False),
    )


def _check_length(n: int, name: str) -> None:
    if n < _MIN_SAMPLES:
        raise ValueError(f"{name} needs more than {N_FFT // 2} samples, got {n}")


def log_mel(audio: Any) -> np.ndarray:
    """Compute Qwen3-ASR's input features for one waveform.

    Equivalent to ``WhisperFeatureExtractor(feature_size=128)(audio, padding=True,
    truncation=False)`` for a single input (also what the vLLM backend computes per request).

    Args:
        audio: 16 kHz mono waveform: float array in [-1, 1], int16 PCM, list, or CPU
            torch tensor.

    Returns:
        float32 array of shape ``(128, len(audio) // 160)``.

    Raises:
        ValueError: ``audio`` is not 1-D, contains NaN / inf, or has at most 200 samples.
        TypeError: ``audio`` has an unsupported dtype.
    """
    audio = as_audio(audio)
    _check_length(audio.shape[0], "audio")
    return _frontend().log_mel(audio)


def _check_out(out: Any, shape: tuple[int, int, int]) -> np.ndarray:
    if not isinstance(out, np.ndarray) or out.dtype != np.float32:
        what = f"dtype {out.dtype}" if isinstance(out, np.ndarray) else type(out).__name__
        raise TypeError(f"out must be a float32 numpy array, got {what}")
    if out.shape != shape:
        raise ValueError(f"out has shape {out.shape}, expected {shape}")
    if not out.flags.c_contiguous or not out.flags.writeable:
        raise ValueError("out must be C-contiguous and writable")
    return out


def log_mel_batch(
    audios: Sequence[Any] | np.ndarray,
    n_threads: int | None = None,
    hf_padding: bool = False,
    out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute :func:`log_mel` for many waveforms in parallel, padded to a common length.

    Args:
        audios: sequence of 1-D waveforms (any lengths), or a 2-D ``(B, N)`` array.
        n_threads: worker threads; ``None`` uses all available cores.
        hf_padding: reproduce the transformers backend's batches, where every waveform is
            zero-padded to the longest one *before* the STFT: the last frames of shorter items
            then see zeros instead of the reflection, so an item's features depend on the
            batch it is in. By default every item is computed on its own (as by
            :func:`log_mel` and the vLLM backend).
        out: optional C-contiguous, writable float32 array of the exact ``(B, 128, T)`` shape
            (see Returns) to write the features into, e.g. a buffer reused across calls: a new
            array of that size costs about as much to touch for the first time as computing
            the features in parallel. It is returned as ``features``; its whole contents are
            overwritten (including the zeros after each item). If an error is raised after
            the input checks (NaN / inf samples), its contents are unspecified.

    Returns:
        ``(features, lengths)``: float32 ``(B, 128, T)`` and int32 ``(B,)`` valid frames per
        item (the ``feature_attention_mask`` sums).

        - default: item ``i`` is ``log_mel(audios[i])`` in ``features[i, :, :lengths[i]]``,
          ``lengths[i] = len_i // 160``, followed by zeros; ``T = max(lengths)``.
        - ``hf_padding=True``: ``T = L // 160`` for the longest length ``L``, every frame
          computed, ``lengths[i] = min(ceil(len_i / 160), T)``.
    """
    items = check_audio_list(audios)
    fe = _frontend()
    sizes = [a.shape[0] for a in items]

    if hf_padding:
        n_padded = max(sizes, default=0)
        if items:
            _check_length(n_padded, "the longest item")
        n_frames = n_padded // HOP_LENGTH
        lengths = np.array([min(-(-n // HOP_LENGTH), n_frames) for n in sizes], np.int32)
        shape = (len(items), N_MELS, n_frames)
        feats = np.empty(shape, np.float32) if out is None else _check_out(out, shape)

        def work(i: int, audio: np.ndarray) -> None:
            _frontend().log_mel_padded(audio, n_padded, out=feats[i])

    else:
        for i, n in enumerate(sizes):
            _check_length(n, f"audios[{i}]")
        lengths = np.array([fe.out_frames(n) for n in sizes], dtype=np.int32)
        n_frames = int(lengths.max()) if items else 0
        shape = (len(items), N_MELS, n_frames)
        # a new array is already zero after each item; a caller's buffer is cleared per item
        feats = np.zeros(shape, np.float32) if out is None else _check_out(out, shape)

        def work(i: int, audio: np.ndarray) -> None:
            n = lengths[i]
            if n == n_frames:  # writes straight into the (contiguous) row
                _frontend().log_mel(audio, out=feats[i])
                return
            feats[i, :, :n] = _frontend().log_mel(audio)
            if out is not None:
                feats[i, :, n:] = 0.0

    run_batch(work, items, n_threads)
    return feats, lengths


def _tokens(n: Any) -> Any:  # ints or int64 arrays
    r = n % 100
    for _ in range(3):
        r = -(-r // 2)  # a stride-2 convolution: ceil(r / 2)
    return n // 100 * 13 + r


@overload
def num_audio_tokens(n_frames: int) -> int: ...
@overload
def num_audio_tokens(n_frames: npt.ArrayLike) -> np.ndarray: ...
def num_audio_tokens(n_frames: Any) -> Any:
    """Number of audio tokens the encoder produces from ``n_frames`` feature frames.

    The encoder works on windows of 100 frames, each reduced by three stride-2 convolutions
    to 13 tokens; a last partial window of ``r`` frames gives ``ceil(ceil(ceil(r / 2) / 2) /
    2)`` tokens. Qwen3-ASR's prompt holds this many audio placeholder tokens per input.

    Args:
        n_frames: frame count(s), e.g. the ``lengths`` of :func:`log_mel_batch`.
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
