"""Audio file loading: the 16 kHz mono float32 input Whisper and SenseVoiceSmall expect."""

from __future__ import annotations

import math
import os
from typing import Union

import numpy as np

from . import _ext

SAMPLE_RATE = 16000

_RAW_AUTO, _RAW_NEVER, _RAW_FORCE = 0, 1, 2
_RAW_EXTENSIONS = (".pcm", ".raw")
_TOO_LONG = "audio longer than the limit"  # af_strerror(AF_ETOOLONG)
_SUPPORTED = "WAV, Wave64, AIFF, CAF, NIST SPHERE, FLAC, MP3, or raw PCM (s16le, 16 kHz, mono)"

Source = Union[str, "os.PathLike[str]", bytes, bytearray, memoryview, np.ndarray]


def load(source: Source, max_duration: float | None = None) -> np.ndarray:
    """Load audio as Whisper / SenseVoiceSmall input.

    Args:
        source: a file path, or the file's contents as bytes (``bytes``, ``bytearray``,
            ``memoryview`` or a 1-D uint8 array). Supported: WAV / RF64 and Wave64 (PCM,
            float, A-law, mu-law, IMA and Microsoft ADPCM), AIFF / AIFF-C and CAF (PCM, float,
            A-law, mu-law, IMA4), NIST SPHERE (PCM, A-law, mu-law), FLAC, MP3, and raw PCM
            (s16le, 16 kHz, mono) given as a ``.pcm`` / ``.raw`` path or as bytes without a
            recognized container header. The format is detected from the content.
        max_duration: if given, the longest audio in seconds to accept: longer audio raises
            ``ValueError`` before most of it is decoded (for uncompressed and ADPCM formats
            before any sample is), so a small upload that would decode to gigabytes fails
            cheaply. The result then has at most ``max_duration * 16000`` samples.

    Returns:
        1-D float32 array, 16 kHz mono, matching ``whisper.load_audio(path)`` (ffmpeg's
        decoding, resampling and downmix, rounded through int16): bit-exact for all formats
        but MP3 at common sample rates, within 1/32768 for MP3 (different decoder). See
        docs/load.md for the details.
    """
    limit = _max_samples(max_duration)
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        raw = _RAW_FORCE if path.lower().endswith(_RAW_EXTENSIONS) else _RAW_NEVER
        # read() rather than mmap: a file truncated by another process while mapped would
        # crash the interpreter with SIGBUS
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return _decode(data, raw, limit, repr(path), max_duration)
    return _decode(
        _as_bytes(source), _RAW_AUTO, limit, f"{type(source).__name__} input", max_duration
    )


def _max_samples(max_duration: object) -> int:
    """max_duration in seconds -> the native limit in samples (0: none)."""
    if max_duration is None:
        return 0
    if isinstance(max_duration, bool) or not isinstance(max_duration, (int, float, np.number)):
        raise TypeError(f"max_duration must be a number, got {type(max_duration).__name__}")
    d = float(max_duration)
    if not (math.isfinite(d) and d > 0):
        raise ValueError(f"max_duration must be finite and > 0, got {max_duration!r}")
    # at least one sample (0 means no limit), at most what size_t holds; + 1e-6 so that
    # n / 16000 seconds, which may round to just below n samples, still allows n
    limit = 2**32 - 1 if _SIZE_T_32 else 2**63
    if d >= limit / SAMPLE_RATE:  # cap seconds before multiplying: a finite d can overflow
        return limit
    return max(1, min(math.floor(d * SAMPLE_RATE + 1e-6), limit))


_SIZE_T_32 = np.dtype(np.uintp).itemsize == 4


def _decode(data: np.ndarray, raw: int, limit: int, what: str, max_duration: object) -> np.ndarray:
    try:
        return _ext.load_memory(data, raw, limit)
    except ValueError as e:
        if str(e).startswith(_TOO_LONG):
            raise ValueError(
                f"{what}: audio is longer than max_duration={max_duration} s"
            ) from None
        raise ValueError(f"{what}: {e}; supported: {_SUPPORTED}") from None


def _as_bytes(source: object) -> np.ndarray:
    if isinstance(source, np.ndarray):
        if source.dtype != np.uint8 or source.ndim != 1:
            raise TypeError(
                f"array input must be 1-D uint8 file contents, got {source.dtype} {source.shape}"
            )
        return np.ascontiguousarray(source)
    try:
        view = memoryview(source)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(
            "source must be a path (str / os.PathLike) or bytes-like file contents, "
            f"got {type(source).__name__}"
        ) from None
    if not view.contiguous:
        raise ValueError("bytes-like source must be contiguous")
    return np.frombuffer(view.cast("B"), dtype=np.uint8)
