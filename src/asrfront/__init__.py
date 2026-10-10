"""asrfront: super-fast, dependency-free frontends for ASR models."""

from . import sensevoice, whisper
from ._ext import __version__

__all__ = ["whisper", "sensevoice", "__version__"]
