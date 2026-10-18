"""Whisper log-mel spectrogram frontend (``whisper.audio.log_mel_spectrogram``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from . import _ext
from ._audio import as_audio, as_audio_list, cached_frontend, run_parallel

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000

__all__ = [
    "log_mel",
    "log_mel_batch",
    "SAMPLE_RATE",
    "N_FFT",
    "HOP_LENGTH",
    "N_SAMPLES",
    "N_FRAMES",
]


def _check_n_mels(n_mels: Any) -> int:
    if isinstance(n_mels, bool) or n_mels not in (80, 128):
        raise ValueError(f"n_mels must be 80 or 128 (large-v3), got {n_mels!r}")
    return int(n_mels)


def _check_length(audio: np.ndarray, pad_or_trim: bool, name: str = "audio") -> None:
    if not pad_or_trim and audio.shape[0] <= N_FFT // 2:
        raise ValueError(
            f"{name} needs more than {N_FFT // 2} samples when pad_or_trim=False, "
            f"got {audio.shape[0]}"
        )


def _frontend(n_mels: int, pad_or_trim: bool) -> _ext.WhisperFrontend:
    return cached_frontend(
        ("whisper", n_mels, pad_or_trim),
        lambda: _ext.WhisperFrontend(n_mels=n_mels, pad_or_trim=pad_or_trim),
    )


def log_mel(audio: Any, n_mels: int = 80, pad_or_trim: bool = True) -> np.ndarray:
    """Compute Whisper's log-mel spectrogram.

    Equivalent to ``whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), n_mels)``
    (or without ``pad_or_trim``), computed in native code.

    Args:
        audio: 16 kHz mono waveform: float array in [-1, 1], int16 PCM, list, or CPU
            torch tensor.
        n_mels: 80, or 128 for large-v3.
        pad_or_trim: pad/trim the input to 30 s (3000 frames) like ``whisper.pad_or_trim``.

    Returns:
        float32 array of shape ``(n_mels, n_frames)``; ``n_frames`` is 3000 with
        ``pad_or_trim``, otherwise ``len(audio) // 160``.

    Raises:
        ValueError: ``audio`` is not 1-D or contains NaN / inf, ``n_mels`` is unsupported, or
            ``pad_or_trim=False`` and ``audio`` has at most 200 samples (the reflect
            padding of the centered STFT needs more; whisper/torch fail too).
        TypeError: ``audio`` has an unsupported dtype (e.g. int32, complex).
    """
    n_mels = _check_n_mels(n_mels)
    pad_or_trim = bool(pad_or_trim)
    audio = as_audio(audio)
    _check_length(audio, pad_or_trim)
    return _frontend(n_mels, pad_or_trim).log_mel(audio)


def log_mel_batch(
    audios: Sequence[Any] | np.ndarray,
    n_mels: int = 80,
    pad_or_trim: bool = True,
    n_threads: int | None = None,
) -> np.ndarray | list[np.ndarray]:
    """Compute :func:`log_mel` for many waveforms in parallel.

    Args:
        audios: sequence of 1-D waveforms (any lengths), or a 2-D ``(B, N)`` array.
        n_mels, pad_or_trim: as in :func:`log_mel`.
        n_threads: worker threads; ``None`` uses all available cores.

    Returns:
        With ``pad_or_trim`` (the default): one float32 array ``(B, n_mels, 3000)``.
        Otherwise a list of ``(n_mels, len_i // 160)`` arrays, since lengths differ.
    """
    n_mels = _check_n_mels(n_mels)
    pad_or_trim = bool(pad_or_trim)
    items = as_audio_list(audios)
    for i, a in enumerate(items):
        _check_length(a, pad_or_trim, name=f"audios[{i}]")

    if pad_or_trim:
        out = np.empty((len(items), n_mels, N_FRAMES), dtype=np.float32)
        targets = list(out)
    else:
        targets = [np.empty((n_mels, a.shape[0] // HOP_LENGTH), np.float32) for a in items]

    def work(i: int) -> None:
        _frontend(n_mels, pad_or_trim).log_mel(items[i], out=targets[i])

    run_parallel(work, len(items), n_threads)
    return out if pad_or_trim else targets
