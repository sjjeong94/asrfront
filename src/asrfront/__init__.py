"""asrfront: super-fast, dependency-free frontends for ASR models."""

from . import gemma4, qwen3_asr, sensevoice, whisper
from ._ext import __version__
from ._load import SAMPLE_RATE, AudioInfo, info, load

__all__ = [
    "whisper",
    "sensevoice",
    "qwen3_asr",
    "gemma4",
    "load",
    "info",
    "AudioInfo",
    "SAMPLE_RATE",
    "__version__",
]
