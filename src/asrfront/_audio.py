from __future__ import annotations

import os
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Callable, TypeVar, cast

import numpy as np

T = TypeVar("T")

_local = threading.local()

# Native handles kept per thread (see cached_frontend). Small: a call rarely alternates
# between more than a few configurations, and each handle owns its scratch buffers.
_CACHE_SIZE = 8

# Shared worker pool for the *_batch functions, created on first use and grown on demand.
# Persistent so that a batch call neither starts threads nor rebuilds the workers' cached
# handles.
_pool: ThreadPoolExecutor | None = None
_pool_size = 0
_pool_lock = threading.Lock()


def _reset_pool() -> None:  # in a forked child the parent's worker threads do not exist
    global _pool, _pool_size, _pool_lock
    _pool, _pool_size, _pool_lock = None, 0, threading.Lock()


if hasattr(os, "register_at_fork"):  # not on Windows
    os.register_at_fork(after_in_child=_reset_pool)


def as_audio(audio: Any, name: str = "audio") -> np.ndarray:
    """Normalize input to a contiguous 1-D float32 waveform in [-1, 1].

    Accepts anything ``np.asarray`` understands (lists, numpy arrays, CPU torch tensors).
    int16 is treated as PCM and scaled by 1/32768; float dtypes are cast to float32.
    NaN / inf (also float64 values beyond the float32 range) raise ValueError: the native
    frontends would silently turn them into finite, meaningless features.
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
        with np.errstate(over="ignore"):  # out-of-range values become inf, rejected below
            out = np.ascontiguousarray(a, dtype=np.float32)
        if not np.isfinite(out).all():
            bad = int(np.flatnonzero(~np.isfinite(out))[0])
            raise ValueError(
                f"{name} contains NaN or inf (first at index {bad}, value {a[bad]!r}); "
                "only finite samples are supported"
            )
        return out
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


def _available_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not available on macOS / Windows
        return os.cpu_count() or 1


def resolve_threads(n_threads: int | None, n_items: int) -> int:
    if n_threads is None:
        n_threads = _available_cores()
    n = check_int("n_threads", n_threads, 1)
    return max(1, min(n, n_items))


def _mark_worker() -> None:
    _local.in_pool = True


def _executor(workers: int) -> ThreadPoolExecutor:
    global _pool, _pool_size
    with _pool_lock:
        if _pool is None or _pool_size < workers:
            old = _pool
            _pool_size = max(workers, _available_cores())
            _pool = ThreadPoolExecutor(
                max_workers=_pool_size, thread_name_prefix="asrfront", initializer=_mark_worker
            )
            if old is not None:
                old.shutdown(wait=False)  # its queued work still runs
        return _pool


def run_parallel(fn: Callable[[int], None], n_items: int, n_threads: int | None) -> None:
    """Run fn(0..n_items-1) on `n_threads` threads of the shared pool. The native calls
    release the GIL."""
    workers = resolve_threads(n_threads, n_items)
    # In a pool worker (fn itself calls a *_batch function), waiting on the pool could
    # deadlock: run inline.
    if workers == 1 or getattr(_local, "in_pool", False):
        for i in range(n_items):
            fn(i)
        return
    # `workers` tasks pull indices from a shared iterator, so a pool larger than n_threads
    # still runs at most n_threads items at once.
    items = iter(range(n_items))
    lock = threading.Lock()
    failed = threading.Event()

    def drain() -> None:
        while not failed.is_set():
            with lock:
                i = next(items, None)
            if i is None:
                return
            try:
                fn(i)
            except BaseException:
                failed.set()  # the other tasks stop taking items
                raise

    ex = _executor(workers)
    futures: list[Future[None]] = [ex.submit(drain) for _ in range(workers)]
    try:
        wait(futures)  # all done before returning: workers write into caller-visible buffers
    except BaseException:  # e.g. KeyboardInterrupt: do not finish the batch in the background
        failed.set()
        raise
    for f in futures:
        f.result()  # re-raises the first worker exception


def cached_frontend(key: tuple[Any, ...], factory: Callable[[], T]) -> T:
    """Per-thread LRU cache of native frontend handles (handles are not thread-safe)."""
    cache: OrderedDict[tuple[Any, ...], Any] | None = getattr(_local, "cache", None)
    if cache is None:
        cache = _local.cache = OrderedDict()
    h = cache.get(key)
    if h is None:
        h = cache[key] = factory()
        if len(cache) > _CACHE_SIZE:
            cache.popitem(last=False)
    else:
        cache.move_to_end(key)
    return cast(T, h)
