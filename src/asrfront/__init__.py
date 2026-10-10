"""asrfront: super-fast, dependency-free frontends for ASR models."""

from . import sensevoice, whisper
from ._ext import __version__
from ._load import SAMPLE_RATE, load

__all__ = ["whisper", "sensevoice", "load", "SAMPLE_RATE", "__version__"]
