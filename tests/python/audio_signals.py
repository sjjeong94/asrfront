"""Deterministic test signals (16 kHz mono float32 in [-1, 1])."""

from __future__ import annotations

import functools
import wave
from pathlib import Path

import numpy as np

SR = 16000
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def read_wav_int16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == SR
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.int16)


def _t(seconds: float) -> np.ndarray:
    return np.arange(int(seconds * SR)) / SR


@functools.cache
def signals() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1234)
    speech = read_wav_int16(DATA_DIR / "jfk.wav").astype(np.float32) / 32768.0

    t = _t(5.0)
    chirp = 0.5 * np.sin(2 * np.pi * (20 * t + (7999 - 20) / (2 * 5.0) * t**2))
    impulses = np.zeros(int(2.0 * SR))
    impulses[:: SR // 4] = 0.9

    sigs = {
        "silence": np.zeros(SR),
        "sine_440": 0.5 * np.sin(2 * np.pi * 440 * _t(3.0)),
        "multi_tone": sum(0.2 * np.sin(2 * np.pi * f * _t(3.0)) for f in (100, 1000, 7900)),
        "chirp": chirp,
        "white_noise": np.clip(0.1 * rng.standard_normal(5 * SR), -1, 1),
        "near_silent": 1e-6 * rng.standard_normal(2 * SR),
        "impulses": impulses,
        "clipped": np.clip(3.0 * np.sin(2 * np.pi * 300 * _t(2.0)), -1, 1),
        "dc_offset": 0.3 + 0.01 * rng.standard_normal(2 * SR),
        "speech": speech,
        "speech_35s": np.tile(speech, 4)[: 35 * SR],  # longer than whisper's 30 s window
        "short_201": 0.1 * rng.standard_normal(201),
        "short_399": 0.1 * rng.standard_normal(399),
        "short_400": 0.1 * rng.standard_normal(400),
        "short_1000": 0.1 * rng.standard_normal(1000),
        "odd_length": 0.1 * rng.standard_normal(16000 + 123),
    }
    out = {}
    for k, v in sigs.items():
        a = np.ascontiguousarray(v, dtype=np.float32)
        a.flags.writeable = False
        out[k] = a
    return out


SIGNAL_NAMES = list(signals())
