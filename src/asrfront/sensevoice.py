"""SenseVoice (FunASR WavFrontend) fbank frontend: kaldi fbank + LFR + CMVN."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _ext
from ._audio import as_audio, cached_frontend

SAMPLE_RATE = 16000

Cmvn = tuple[np.ndarray, np.ndarray]

__all__ = ["fbank", "fbank_raw", "SAMPLE_RATE"]


def _frontend(n_mels: int, lfr_m: int, lfr_n: int, cmvn: Cmvn | None) -> _ext.SenseVoiceFrontend:
    if cmvn is None:
        shift = scale = None
        cmvn_key: Any = None
    else:
        shift, scale = (np.ascontiguousarray(v, dtype=np.float32).ravel() for v in cmvn)
        cmvn_key = (shift.tobytes(), scale.tobytes())
    return cached_frontend(
        ("sensevoice", n_mels, lfr_m, lfr_n, cmvn_key),
        lambda: _ext.SenseVoiceFrontend(
            n_mels=n_mels, lfr_m=lfr_m, lfr_n=lfr_n, cmvn_shift=shift, cmvn_scale=scale
        ),
    )


def fbank(
    audio: Any,
    cmvn: Cmvn | None = None,
    n_mels: int = 80,
    lfr_m: int = 7,
    lfr_n: int = 6,
) -> np.ndarray:
    """Compute SenseVoice input features.

    Args:
        audio: 16 kHz mono waveform (float in [-1, 1], or int16).
        cmvn: optional ``(shift, scale)`` vectors of length ``n_mels * lfr_m``
            (``<AddShift>`` / ``<Rescale>`` from the model's ``am.mvn``).

    Returns:
        float32 array of shape ``(ceil(T / lfr_n), n_mels * lfr_m)``.
    """
    return _frontend(n_mels, lfr_m, lfr_n, cmvn).fbank(as_audio(audio))


def fbank_raw(audio: Any, n_mels: int = 80) -> np.ndarray:
    """Kaldi-compatible log-mel fbank without LFR/CMVN. Shape ``(T, n_mels)``."""
    return _frontend(n_mels, 7, 6, None).fbank_raw(as_audio(audio))
