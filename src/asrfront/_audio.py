from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar, cast

import numpy as np

T = TypeVar("T")

_local = threading.local()


def as_audio(audio: Any, name: str = "audio") -> np.ndarray:
    """Normalize input to a contiguous 1-D float32 waveform in [-1, 1].

    Accepts anything ``np.asarray`` understands (lists, numpy arrays, CPU torch tensors).
    int16 is treated as PCM and scaled by 1/32768; float dtypes are cast to float32.
    """
    a = np.asarray(audio)
    if a.ndim != 1:
        hint = ""
        if a.ndim == 2:
            hint = (
                "; for multi-channel audio mix down first (e.g. audio.mean(axis=0)), "
                "for a batch use the *_batch function"
            )
        raise ValueError(f"{name} must be a 1-D mono waveform, got shape {a.shape}{hint}")
    kind = a.dtype.kind
    if a.dtype == np.int16:
        return a.astype(np.float32) / np.float32(32768.0)
    if kind == "f":
        return np.ascontiguousarray(a, dtype=np.float32)
    if kind in "iu":
        raise TypeError(
            f"{name} has integer dtype {a.dtype}; only int16 PCM is scaled automatically. "
            f"Convert to float in [-1, 1] first (e.g. audio / {2 ** (8 * a.itemsize - 1)})"
        )
    raise TypeError(f"{name} must be a float or int16 array, got dtype {a.dtype}")


def as_audio_list(audios: Any) -> list[np.ndarray]:
    if isinstance(audios, np.ndarray) and audios.ndim == 2:
        audios = list(audios)
    try:
        items = list(audios)
    except TypeError:
        raise TypeError(
            f"audios must be a sequence of 1-D waveforms, got {type(audios).__name__}"
        ) from None
    return [as_audio(a, name=f"audios[{i}]") for i, a in enumerate(items)]


def check_int(name: str, value: Any, lo: int, hi: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    v = int(value)
    if v < lo or (hi is not None and v > hi):
        bound = f">= {lo}" if hi is None else f"in [{lo}, {hi}]"
        raise ValueError(f"{name} must be {bound}, got {v}")
    return v


def resolve_threads(n_threads: int | None, n_items: int) -> int:
    if n_threads is None:
        try:
            n_threads = len(os.sched_getaffinity(0))
        except AttributeError:  # not available on macOS / Windows
            n_threads = os.cpu_count() or 1
    n = check_int("n_threads", n_threads, 1)
    return max(1, min(n, n_items))


def run_parallel(fn: Callable[[int], None], n_items: int, n_threads: int | None) -> None:
    """Run fn(0..n_items-1) on a thread pool. The native calls release the GIL."""
    workers = resolve_threads(n_threads, n_items)
    if workers == 1:
        for i in range(n_items):
            fn(i)
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="asrfront") as ex:
        for _ in ex.map(fn, range(n_items)):  # re-raises the first worker exception
            pass


def cached_frontend(key: tuple[Any, ...], factory: Callable[[], T]) -> T:
    """Per-thread cache of native frontend handles (handles are not thread-safe)."""
    cache = _local.__dict__.setdefault("cache", {})
    h = cache.get(key)
    if h is None:
        h = cache[key] = factory()
    return cast(T, h)
