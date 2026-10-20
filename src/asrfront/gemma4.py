"""Gemma 4 (E2B / E4B) audio input features.

Both models use transformers' ``Gemma4AudioFeatureExtractor`` with the same settings: 128 HTK
mel bins of the magnitude spectrum, ``log(mel + 0.001)``, 20 ms frames every 10 ms with
"semicausal" padding (160 zeros before the first sample), inputs truncated to 30 s and a batch
zero-padded to its longest item, rounded up to a multiple of 128 samples; the frames that read
padding are masked to zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, overload

import numpy as np
import numpy.typing as npt

from . import _ext
from ._audio import (
    cached_frontend,
    check_audio,
    check_audio_list,
    check_int,
    convert_audio,
    run_batch,
)

SAMPLE_RATE = 16000
FRAME_LENGTH = 320
HOP_LENGTH = 160
N_FFT = 512
N_MELS = 128
MAX_SAMPLES = 30 * SAMPLE_RATE  # transformers truncates longer inputs (max_length=480000)
PAD_TO_MULTIPLE_OF = 128

__all__ = [
    "log_mel",
    "log_mel_batch",
    "num_frames",
    "num_audio_tokens",
    "SAMPLE_RATE",
    "FRAME_LENGTH",
    "HOP_LENGTH",
    "N_FFT",
    "N_MELS",
    "MAX_SAMPLES",
]


def _frontend() -> _ext.Gemma4Frontend:
    return cached_frontend(("gemma4",), _ext.Gemma4Frontend)


def _out_frames(n_longest: int) -> int:
    """Frames of a batch whose longest item has n_longest samples (padded to a multiple of 128)."""
    padded = -(-n_longest // PAD_TO_MULTIPLE_OF) * PAD_TO_MULTIPLE_OF
    return int(_ext.Gemma4Frontend.out_frames(padded))


def num_frames(n_samples: int, truncation: bool = True) -> int:
    """Valid feature frames of an input of ``n_samples`` samples: the ``input_features_mask``
    sum, ``(n - 1) // 160`` (``n`` capped at 30 s with ``truncation``). The other frames of the
    features are padding (zeros)."""
    n = check_int("n_samples", n_samples, 0)
    if truncation:
        n = min(n, MAX_SAMPLES)
    return int(_ext.Gemma4Frontend.valid_frames(n))


def log_mel(audio: Any, truncation: bool = True) -> np.ndarray:
    """Compute Gemma 4's audio input features for one waveform.

    Equivalent to ``Gemma4AudioFeatureExtractor(...)([audio])["input_features"][0]`` with the
    E2B / E4B checkpoints' settings, as ``Gemma4Processor`` calls it.

    Args:
        audio: 16 kHz mono waveform: float array in [-1, 1], int16 PCM, list, or CPU
            torch tensor.
        truncation: use only the first 30 s, as transformers does by default (samples past it
            are neither converted nor checked). ``False`` computes the whole input.

    Returns:
        float32 array ``(T, 128)``, time first like transformers' ``input_features``:
        ``T = (L - 1) // 160`` frames for the input zero-padded to ``L``, the next multiple of
        128 samples. The first :func:`num_frames` frames are the features, the rest (at most
        one) are zeros.

    Raises:
        ValueError: ``audio`` is not 1-D or contains NaN / inf.
        TypeError: ``audio`` has an unsupported dtype.
    """
    audio = check_audio(audio)
    if truncation:
        audio = audio[:MAX_SAMPLES]
    audio = convert_audio(audio)
    return _frontend().log_mel(audio, _out_frames(audio.shape[0]))


def log_mel_batch(
    audios: Sequence[Any] | np.ndarray,
    n_threads: int | None = None,
    truncation: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute :func:`log_mel` for many waveforms in parallel, padded to a common length.

    Equivalent to one ``Gemma4AudioFeatureExtractor`` call on the batch: every waveform is
    zero-padded to the longest one (rounded up to a multiple of 128 samples). An item's
    features do not depend on the batch (its valid frames read no padding); only the number
    of zero frames after them does.

    Args:
        audios: sequence of 1-D waveforms (any lengths), or a 2-D ``(B, N)`` array.
        n_threads: at most this many worker threads (fewer for little audio); ``None``: all
            available cores.
        truncation: as in :func:`log_mel`.

    Returns:
        ``(features, lengths)``: float32 ``(B, T, 128)``, zeros after each item's valid
        frames, and int32 ``(B,)`` valid frames per item (the ``input_features_mask`` sums;
        the mask is ``np.arange(T) < lengths[:, None]``).
    """
    items = check_audio_list(audios)
    if truncation:
        items = [a[:MAX_SAMPLES] for a in items]
    sizes = [a.shape[0] for a in items]
    n_frames = _out_frames(max(sizes, default=0))
    lengths = np.array([_ext.Gemma4Frontend.valid_frames(n) for n in sizes], np.int32)
    feats = np.empty((len(items), n_frames, N_MELS), np.float32)  # every element is written

    def work(chunk: range, audios: list[np.ndarray]) -> int:
        return _frontend().log_mel_many(audios, n_frames, [feats[i] for i in chunk])

    run_batch(work, items, n_threads)
    return feats, lengths


def _tokens(n: Any) -> Any:  # ints or int64 arrays
    for _ in range(2):
        n = -(-n // 2)  # a stride-2 convolution (kernel 3, padding 1): ceil(n / 2)
    return n


@overload
def num_audio_tokens(n_frames: int) -> int: ...
@overload
def num_audio_tokens(n_frames: npt.ArrayLike) -> np.ndarray: ...
def num_audio_tokens(n_frames: Any) -> Any:
    """Number of audio soft tokens the encoder produces from ``n_frames`` valid frames.

    Two stride-2 convolutions subsample the frames: ``ceil(ceil(n / 2) / 2)``, as
    ``Gemma4Processor`` counts the audio placeholder tokens of a prompt (at most 750 for
    30 s).

    Args:
        n_frames: valid frame count(s), e.g. the ``lengths`` of :func:`log_mel_batch` or
            :func:`num_frames`.
    """
    if isinstance(n_frames, (int, np.integer)) and not isinstance(n_frames, bool):
        return int(_tokens(check_int("n_frames", n_frames, 0)))
    arr = np.asarray(n_frames)
    if arr.dtype.kind not in "iu":
        raise TypeError(f"n_frames must be integers, got dtype {arr.dtype}")
    arr = arr.astype(np.int64)
    if (arr < 0).any():
        raise ValueError("n_frames must be >= 0")
    return _tokens(arr)
