from __future__ import annotations

import threading
from typing import Any, Callable, TypeVar

import numpy as np

T = TypeVar("T")

_local = threading.local()


def as_audio(audio: Any) -> np.ndarray:
    """Normalize input to a contiguous 1-D float32 waveform in [-1, 1].

    int16 input is scaled by 1/32768; other dtypes are cast to float32.
    """
    a = np.asarray(audio)
    if a.ndim != 1:
        raise ValueError(f"audio must be 1-D (mono), got shape {a.shape}")
    if a.dtype == np.int16:
        a = a.astype(np.float32) / np.float32(32768.0)
    return np.ascontiguousarray(a, dtype=np.float32)


def cached_frontend(key: tuple, factory: Callable[[], T]) -> T:
    """Per-thread cache of native frontend handles (handles are not thread-safe)."""
    cache = _local.__dict__.setdefault("cache", {})
    h = cache.get(key)
    if h is None:
        h = cache[key] = factory()
    return h
