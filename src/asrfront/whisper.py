"""Whisper log-mel spectrogram frontend."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _ext
from ._audio import as_audio, cached_frontend

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000

__all__ = ["log_mel", "SAMPLE_RATE", "N_FFT", "HOP_LENGTH", "N_SAMPLES", "N_FRAMES"]


def log_mel(audio: Any, n_mels: int = 80, pad_or_trim: bool = True) -> np.ndarray:
    """Compute Whisper's log-mel spectrogram.

    Args:
        audio: 16 kHz mono waveform (float in [-1, 1], or int16).
        n_mels: 80, or 128 for large-v3.
        pad_or_trim: pad/trim the input to 30 s (3000 frames) like ``whisper.pad_or_trim``.

    Returns:
        float32 array of shape ``(n_mels, n_frames)``.
    """
    fe = cached_frontend(
        ("whisper", n_mels, bool(pad_or_trim)),
        lambda: _ext.WhisperFrontend(n_mels=n_mels, pad_or_trim=bool(pad_or_trim)),
    )
    return fe.log_mel(as_audio(audio))
