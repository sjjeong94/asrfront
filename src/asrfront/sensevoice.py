"""SenseVoice (FunASR WavFrontend) fbank frontend: kaldi fbank + LFR + CMVN."""

from __future__ import annotations

import functools
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import numpy as np

from . import _ext
from ._audio import as_audio, as_audio_list, cached_frontend, check_int, run_parallel

SAMPLE_RATE = 16000

Cmvn = tuple[np.ndarray, np.ndarray]
CmvnLike = Union[Cmvn, str, "os.PathLike[str]"]

__all__ = ["fbank", "fbank_batch", "fbank_raw", "load_cmvn", "SAMPLE_RATE"]


def load_cmvn(path: str | os.PathLike[str]) -> Cmvn:
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


def _resolve_cmvn(cmvn: CmvnLike | None, dim: int) -> Cmvn | None:
    if cmvn is None:
        return None
    if isinstance(cmvn, (str, os.PathLike)):
        cmvn = load_cmvn(cmvn)
    try:
        shift, scale = cmvn
    except (TypeError, ValueError):
        raise TypeError("cmvn must be a path to am.mvn or a (shift, scale) pair") from None
    shift = np.ascontiguousarray(shift, dtype=np.float32).ravel()
    scale = np.ascontiguousarray(scale, dtype=np.float32).ravel()
    if shift.size < dim or scale.size < dim:
        raise ValueError(
            f"cmvn vectors have {shift.size}/{scale.size} values, need n_mels * lfr_m = {dim}"
        )
    shift, scale = shift[:dim], scale[:dim]  # FunASR uses the first `dim` entries
    for name, v in (("shift", shift), ("scale", scale)):
        if not np.isfinite(v).all():
            bad = int(np.flatnonzero(~np.isfinite(v))[0])
            raise ValueError(f"cmvn {name} contains NaN or inf (first at index {bad})")
    return shift, scale


def _frontend(
    n_mels: int, lfr_m: int, lfr_n: int, cmvn: Cmvn | None, dither: float, seed: int
) -> _ext.SenseVoiceFrontend:
    """``cmvn`` must already be resolved by :func:`_resolve_cmvn`."""
    if dither == 0.0:
        seed = 0  # the RNG is unused: one cached handle for every seed
    shift, scale = cmvn if cmvn is not None else (None, None)
    cmvn_key = None if cmvn is None else (cmvn[0].tobytes(), cmvn[1].tobytes())
    return cached_frontend(
        ("sensevoice", n_mels, lfr_m, lfr_n, cmvn_key, dither, seed),
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


def _check_params(
    n_mels: Any, lfr_m: Any, lfr_n: Any, dither: Any, seed: Any
) -> tuple[int, int, int, float, int]:
    n_mels = check_int("n_mels", n_mels, 1, 1024)
    lfr_m = check_int("lfr_m", lfr_m, 1, 1024)
    lfr_n = check_int("lfr_n", lfr_n, 1, 1024)
    seed = check_int("seed", seed, 0, 2**64 - 1)
    try:
        dither = float(dither)
    except (TypeError, ValueError):
        raise TypeError(f"dither must be a number, got {type(dither).__name__}") from None
    if not (math.isfinite(dither) and dither >= 0):
        raise ValueError(f"dither must be finite and >= 0, got {dither}")
    return n_mels, lfr_m, lfr_n, dither, seed


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
    n_mels, lfr_m, lfr_n, dither, seed = _check_params(n_mels, lfr_m, lfr_n, dither, seed)
    audio = as_audio(audio)
    cmvn_v = _resolve_cmvn(cmvn, n_mels * lfr_m)
    return _frontend(n_mels, lfr_m, lfr_n, cmvn_v, dither, seed).fbank(audio)


def fbank_batch(
    audios: Sequence[Any] | np.ndarray,
    cmvn: CmvnLike | None = None,
    n_mels: int = 80,
    lfr_m: int = 7,
    lfr_n: int = 6,
    dither: float = 0.0,
    seed: int = 0,
    n_threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute :func:`fbank` for many waveforms in parallel, padded like FunASR.

    Args:
        audios: sequence of 1-D waveforms (any lengths), or a 2-D ``(B, N)`` array.
        cmvn, n_mels, lfr_m, lfr_n, dither, seed: as in :func:`fbank`; every item is
            processed exactly as an individual :func:`fbank` call would.
        n_threads: worker threads; ``None`` uses all available cores.

    Returns:
        ``(feats, lengths)``: float32 ``(B, max_len, n_mels * lfr_m)`` zero-padded after
        each item's ``lengths[i]`` frames (FunASR ``WavFrontend`` output), and int32
        ``(B,)`` lengths.
    """
    n_mels, lfr_m, lfr_n, dither, seed = _check_params(n_mels, lfr_m, lfr_n, dither, seed)
    items = as_audio_list(audios)
    cmvn_v = _resolve_cmvn(cmvn, n_mels * lfr_m)  # load am.mvn once, not per worker

    fe = _frontend(n_mels, lfr_m, lfr_n, cmvn_v, dither, seed)
    lengths = np.array([fe.out_frames(a.shape[0]) for a in items], dtype=np.int32)
    max_len = int(lengths.max()) if len(items) else 0
    feats = np.zeros((len(items), max_len, fe.feat_dim), dtype=np.float32)

    def work(i: int) -> None:
        fe_i = _frontend(n_mels, lfr_m, lfr_n, cmvn_v, dither, seed)
        fe_i.fbank(items[i], out=feats[i, : lengths[i]])

    run_parallel(work, len(items), n_threads)
    return feats, lengths


def fbank_raw(audio: Any, n_mels: int = 80, dither: float = 0.0, seed: int = 0) -> np.ndarray:
    """Kaldi-compatible log-mel fbank without LFR/CMVN. Shape ``(T, n_mels)``.

    Same as ``torchaudio.compliance.kaldi.fbank(audio * 32768, num_mel_bins=n_mels,
    window_type="hamming", dither=dither, energy_floor=0)``.
    """
    n_mels, _, _, dither, seed = _check_params(n_mels, 1, 1, dither, seed)
    return _frontend(n_mels, 1, 1, None, dither, seed).fbank_raw(as_audio(audio))
