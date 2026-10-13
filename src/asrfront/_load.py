"""Audio file loading: the 16 kHz mono float32 input Whisper and SenseVoiceSmall expect."""

from __future__ import annotations

import os
from typing import Union

import numpy as np

from . import _ext

SAMPLE_RATE = 16000

_RAW_AUTO, _RAW_NEVER, _RAW_FORCE = 0, 1, 2
_RAW_EXTENSIONS = (".pcm", ".raw")
_SUPPORTED = "WAV, Wave64, AIFF, CAF, NIST SPHERE, FLAC, MP3, or raw PCM (s16le, 16 kHz, mono)"

Source = Union[str, "os.PathLike[str]", bytes, bytearray, memoryview, np.ndarray]


def load(source: Source) -> np.ndarray:
    """Load audio as Whisper / SenseVoiceSmall input.

    Args:
        source: a file path, or the file's contents as bytes (``bytes``, ``bytearray``,
            ``memoryview`` or a 1-D uint8 array). Supported: WAV / RF64 and Wave64 (PCM,
            float, A-law, mu-law, IMA and Microsoft ADPCM), AIFF / AIFF-C and CAF (PCM, float,
            A-law, mu-law, IMA4), NIST SPHERE (PCM, A-law, mu-law), FLAC, MP3, and raw PCM
            (s16le, 16 kHz, mono) given as a ``.pcm`` / ``.raw`` path or as bytes without a
            recognized container header. The format is detected from the content.

    Returns:
        1-D float32 array, 16 kHz mono, matching ``whisper.load_audio(path)`` (ffmpeg's
        decoding, resampling and downmix, rounded through int16): bit-exact for all formats
        but MP3 at common sample rates, within 1/32768 for MP3 (different decoder). See
        docs/load.md for the details.
    """
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        raw = _RAW_FORCE if path.lower().endswith(_RAW_EXTENSIONS) else _RAW_NEVER
        # read() rather than mmap: a file truncated by another process while mapped would
        # crash the interpreter with SIGBUS
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return _decode(data, raw, repr(path))
    return _decode(_as_bytes(source), _RAW_AUTO, f"{type(source).__name__} input")


def _decode(data: np.ndarray, raw: int, what: str) -> np.ndarray:
    try:
        return _ext.load_memory(data, raw)
    except ValueError as e:
        raise ValueError(f"{what}: {e} (supported: {_SUPPORTED})") from None


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
