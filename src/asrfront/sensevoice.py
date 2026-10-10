"""SenseVoice (FunASR WavFrontend) fbank frontend: kaldi fbank + LFR + CMVN."""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Union

import numpy as np

from . import _ext
from ._audio import as_audio, cached_frontend

SAMPLE_RATE = 16000

Cmvn = tuple[np.ndarray, np.ndarray]
CmvnLike = Union[Cmvn, str, os.PathLike]

__all__ = ["fbank", "fbank_raw", "load_cmvn", "SAMPLE_RATE"]


def load_cmvn(path: str | os.PathLike) -> Cmvn:
    """Read ``(shift, scale)`` from a Kaldi nnet ``am.mvn`` file (FunASR format).

    The file holds an ``<AddShift>`` and a ``<Rescale>`` component, each followed by a
    ``<LearnRateCoef> 0 [ v1 v2 ... ]`` line; features are normalized as
    ``(x + shift) * scale``.
    """
    p = Path(path)
    st = p.stat()
    shift, scale = _load_cmvn_cached(str(p.resolve()), st.st_mtime_ns, st.st_size)
    return shift.copy(), scale.copy()


@functools.lru_cache(maxsize=16)
def _load_cmvn_cached(path: str, _mtime_ns: int, _size: int) -> Cmvn:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    vectors: dict[str, np.ndarray] = {}
    for i, line in enumerate(lines):
        tok = line.split()
        if not tok or tok[0] not in ("<AddShift>", "<Rescale>") or i + 1 >= len(lines):
            continue
        nxt = lines[i + 1].split()
        if len(nxt) < 4 or nxt[0] != "<LearnRateCoef>" or nxt[2] != "[" or nxt[-1] != "]":
            raise ValueError(f"{path}:{i + 2}: expected '<LearnRateCoef> x [ ... ]' after {tok[0]}")
        vectors[tok[0]] = np.array(nxt[3:-1], dtype=np.float32)
    missing = {"<AddShift>", "<Rescale>"} - vectors.keys()
    if missing:
        raise ValueError(f"{path}: missing {', '.join(sorted(missing))}")
    shift, scale = vectors["<AddShift>"], vectors["<Rescale>"]
    if shift.shape != scale.shape:
        raise ValueError(f"{path}: <AddShift> has {shift.size} values, <Rescale> {scale.size}")
    return shift, scale


def _frontend(
    n_mels: int, lfr_m: int, lfr_n: int, cmvn: CmvnLike | None, dither: float, seed: int
) -> _ext.SenseVoiceFrontend:
    if cmvn is None:
        shift = scale = None
        cmvn_key: Any = None
    else:
        if isinstance(cmvn, (str, os.PathLike)):
            cmvn = load_cmvn(cmvn)
        shift, scale = (np.ascontiguousarray(v, dtype=np.float32).ravel() for v in cmvn)
        dim = n_mels * lfr_m
        if shift.size < dim or scale.size < dim:
            raise ValueError(
                f"cmvn vectors have {shift.size}/{scale.size} values, need n_mels * lfr_m = {dim}"
            )
        shift, scale = shift[:dim], scale[:dim]  # FunASR uses the first `dim` entries
        cmvn_key = (shift.tobytes(), scale.tobytes())
    return cached_frontend(
        ("sensevoice", n_mels, lfr_m, lfr_n, cmvn_key, float(dither), int(seed)),
        lambda: _ext.SenseVoiceFrontend(
            n_mels=n_mels,
            lfr_m=lfr_m,
            lfr_n=lfr_n,
            dither=dither,
            seed=seed,
            cmvn_shift=shift,
            cmvn_scale=scale,
        ),
    )


def fbank(
    audio: Any,
    cmvn: CmvnLike | None = None,
    n_mels: int = 80,
    lfr_m: int = 7,
    lfr_n: int = 6,
    dither: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Compute SenseVoice input features (FunASR ``WavFrontend``).

    Args:
        audio: 16 kHz mono waveform (float in [-1, 1], or int16).
        cmvn: path to the model's ``am.mvn``, or ``(shift, scale)`` vectors of length
            ``n_mels * lfr_m``. ``None`` skips CMVN.
        n_mels, lfr_m, lfr_n: fbank bins and low-frame-rate stacking (stack ``lfr_m``
            frames every ``lfr_n`` frames).
        dither: Gaussian dither on the int16-scale waveform (kaldi ``dither``). 0 for
            deterministic inference.
        seed: dither RNG seed. The RNG is reseeded on every call, so identical inputs
            give identical outputs.

    Returns:
        float32 array of shape ``(ceil(T / lfr_n), n_mels * lfr_m)``, where
        ``T = 1 + (len(audio) - 400) // 160`` (0 if shorter than 400 samples).
    """
    return _frontend(n_mels, lfr_m, lfr_n, cmvn, dither, seed).fbank(as_audio(audio))


def fbank_raw(audio: Any, n_mels: int = 80, dither: float = 0.0, seed: int = 0) -> np.ndarray:
    """Kaldi-compatible log-mel fbank without LFR/CMVN. Shape ``(T, n_mels)``."""
    return _frontend(n_mels, 1, 1, None, dither, seed).fbank_raw(as_audio(audio))
