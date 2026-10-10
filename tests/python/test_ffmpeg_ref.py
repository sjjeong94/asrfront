"""Validate the ffmpeg load-pipeline reference (``reference.py``, docs/load.md) against ffmpeg.

Needs the ``load`` dependency group (``uv sync --group load``: imageio-ffmpeg with a pinned
ffmpeg 7.0.2 binary); skipped otherwise. ffmpeg's float arithmetic depends on the CPU:
bit-exactness of the float path is only asserted when ffmpeg runs on x86-64 with AVX2 + FMA3
(the CPU the spec was measured on); elsewhere (e.g. aarch64 / Apple silicon) the float path may
differ by 1 LSB. See helpers.ffmpeg_is_reference_build.
"""

from __future__ import annotations

import subprocess
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import reference as ref
from helpers import ffmpeg_is_reference_build

imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

RAW_FMT = {"u8": "u8", "s16": "s16le", "s32": "s32le", "f32": "f32le", "f64": "f64le"}
COMMON_RATES = [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000]
# Rates where ref.ffmpeg_out_len matches ffmpeg exactly (docs/load.md, "Output length").
EXACT_LENGTH_RATES = {8000, 11025, 12000, 16000, 24000, 32000, 48000, 96000}


FLOAT_PATH_EXACT = ffmpeg_is_reference_build()


def ffmpeg_raw(samples: np.ndarray, sample_fmt: str, sr: int, layout: str | None = None):
    """Run whisper's ffmpeg command on raw samples (n, ch) fed through a pipe."""
    x = np.ascontiguousarray(samples)
    ch = 1 if x.ndim == 1 else x.shape[1]
    lay = ["-ch_layout", layout] if layout else ["-ac", str(ch)]
    cmd = [
        FFMPEG,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "0",
        "-f",
        RAW_FMT[sample_fmt],
        "-ar",
        str(sr),
        *lay,
        "-i",
        "-",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-",
    ]
    out = subprocess.run(cmd, input=x.tobytes(), capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16)


def ffmpeg_file(path: Path) -> np.ndarray:
    """Exactly whisper.audio.load_audio's command line, as int16."""
    cmd = [
        FFMPEG,
        "-nostdin",
        "-threads",
        "0",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-",
    ]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.int16)


def make_samples(rng: np.random.Generator, sample_fmt: str, n: int, ch: int) -> np.ndarray:
    """Speech-like level noise + a sine, with a few clipped peaks."""
    t = np.arange(n)[:, None] / 16000.0
    sig = 0.25 * rng.standard_normal((n, ch)) + 0.3 * np.sin(
        2 * np.pi * 440 * t * (1 + np.arange(ch))
    )
    sig[rng.integers(0, n, 20)] = 1.5  # out-of-range peaks
    sig = np.clip(sig, -1.0, 1.0 - 2.0**-31) if sample_fmt != "f32" and sample_fmt != "f64" else sig
    if sample_fmt == "u8":
        return np.clip(np.rint(sig * 128 + 128), 0, 255).astype(np.uint8)
    if sample_fmt == "s16":
        return np.clip(np.rint(sig * 32768), -32768, 32767).astype(np.int16)
    if sample_fmt == "s32":
        return np.floor(sig * 2.0**31).astype(np.int32)
    if sample_fmt == "f32":
        return sig.astype(np.float32)
    return sig


def check(
    actual: np.ndarray,
    expected: np.ndarray,
    sr: int,
    float_path: bool,
    what: str,
    force_tol: int | None = None,
):
    """Common samples bit-exact (or <= 1 LSB where the spec allows), length per spec."""
    n = min(len(actual), len(expected))
    if sr in EXACT_LENGTH_RATES:
        assert len(actual) == len(expected), f"{what}: length {len(actual)} != {len(expected)}"
    else:
        assert abs(len(actual) - len(expected)) <= 1, f"{what}: {len(actual)} vs {len(expected)}"
    diff = np.abs(actual[:n].astype(np.int64) - expected[:n].astype(np.int64))
    tol = 0 if (FLOAT_PATH_EXACT or not float_path) else 1
    if force_tol is not None:
        tol = force_tol
    bad = np.nonzero(diff > tol)[0]
    assert bad.size == 0, (
        f"{what}: {bad.size}/{n} samples differ (max {diff.max()}), first at {bad[0]}: "
        f"{actual[bad[0]]} vs {expected[bad[0]]}"
    )


def uses_float_path(sample_fmt: str, sr: int, ch: int = 1) -> bool:
    """Whether ffmpeg converts through float32 (docs/load.md, internal sample formats)."""
    if sample_fmt == "f32":
        return True
    if sample_fmt == "s32":
        return sr != 16000 or ch > 1
    return sample_fmt == "s16" and sr != 16000


# ---------------------------------------------------------------------------------------------
# Conversion and downmix at 16 kHz (no resampling)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ch", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("sample_fmt", ref.FFMPEG_SAMPLE_FORMATS)
def test_convert_and_downmix(sample_fmt, ch):
    rng = np.random.default_rng(ch)
    x = make_samples(rng, sample_fmt, 20000, ch)
    # f64 with >= 5 channels: <= 1 LSB in rare near-tie samples (docs/load.md, Downmix)
    loose = sample_fmt == "f64" and ch >= 5
    check(
        ref.ffmpeg_load_ref(x, sample_fmt, 16000),
        ffmpeg_raw(x, sample_fmt, 16000),
        16000,
        uses_float_path(sample_fmt, 16000, ch),
        f"{sample_fmt} x{ch}",
        force_tol=1 if loose else None,
    )


@pytest.mark.parametrize(
    "layout",
    [
        "FL+FR+FC",
        "FL+FR+BC",
        "FL+FR+BL+BR",
        "FL+FR+SL+SR",
        "FL+FR+FLC+FRC",
        "FL+FR+FC+LFE+SL+SR",
        "FL+FR+FC+BC+SL+SR",
        "FL+FR+TC+TFL+TFR",
        "FL",
        "FR",
    ],
)
@pytest.mark.parametrize("sample_fmt,sr", [("s16", 16000), ("f32", 16000), ("s16", 44100)])
def test_named_layout_downmix(layout, sample_fmt, sr):
    names = layout.split("+")
    rng = np.random.default_rng(len(names))
    x = make_samples(rng, sample_fmt, 12000, len(names))
    check(
        ref.ffmpeg_load_ref(x, sample_fmt, sr, layout=names),
        ffmpeg_raw(x, sample_fmt, sr, layout=layout),
        sr,
        uses_float_path(sample_fmt, sr, len(names)),
        layout,
    )


# ---------------------------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("sr", COMMON_RATES)
@pytest.mark.parametrize("sample_fmt", ["s16", "f32"])
def test_resample_common_rates(sample_fmt, sr):
    rng = np.random.default_rng(sr)
    x = make_samples(rng, sample_fmt, sr * 2 // 3 + 17, 1)
    check(
        ref.ffmpeg_load_ref(x, sample_fmt, sr),
        ffmpeg_raw(x, sample_fmt, sr),
        sr,
        uses_float_path(sample_fmt, sr),
        f"{sample_fmt} {sr}",
    )


@pytest.mark.parametrize("sr", [44100, 48000, 22050, 8000])
@pytest.mark.parametrize("sample_fmt", ["u8", "s32", "f64"])
def test_resample_other_formats(sample_fmt, sr):
    rng = np.random.default_rng(sr + 1)
    x = make_samples(rng, sample_fmt, sr // 2 + 3, 1)
    check(
        ref.ffmpeg_load_ref(x, sample_fmt, sr),
        ffmpeg_raw(x, sample_fmt, sr),
        sr,
        uses_float_path(sample_fmt, sr),
        f"{sample_fmt} {sr}",
    )


@pytest.mark.parametrize("ch", [2, 6])
@pytest.mark.parametrize("sample_fmt", ref.FFMPEG_SAMPLE_FORMATS)
def test_resample_multichannel(sample_fmt, ch):
    sr = 44100
    rng = np.random.default_rng(ch * 7)
    x = make_samples(rng, sample_fmt, 15000, ch)
    check(
        ref.ffmpeg_load_ref(x, sample_fmt, sr),
        ffmpeg_raw(x, sample_fmt, sr),
        sr,
        uses_float_path(sample_fmt, sr, ch),
        f"{sample_fmt} x{ch} {sr}",
    )


@pytest.mark.parametrize("sr", [16494, 44056])
@pytest.mark.parametrize("sample_fmt", ["s16", "f64"])
def test_resample_interpolated_mode(sample_fmt, sr):
    """Non-standard rates (> 1024 phases): float path <= 1 LSB, double path exact."""
    rng = np.random.default_rng(sr)
    x = make_samples(rng, sample_fmt, 20000, 1)
    actual, expected = ref.ffmpeg_load_ref(x, sample_fmt, sr), ffmpeg_raw(x, sample_fmt, sr)
    n = min(len(actual), len(expected))
    assert abs(len(actual) - len(expected)) <= 1
    diff = np.abs(actual[:n].astype(np.int64) - expected[:n])
    assert diff.max() <= (1 if sample_fmt == "s16" else 0)


def test_filter_bank_matches_impulse_response():
    """Every tap of every phase, read from ffmpeg's double path with impulses (24 kHz)."""
    sr = 24000
    bank = ref.ffmpeg_filter_bank(sr)
    prm = ref.ffmpeg_resample_params(sr)
    r0 = -(prm["n_taps"] // 2) + 1
    ratio = Fraction(sr, 16000)
    for p in (1500, 1501, 1502):
        x = np.zeros(3000)
        x[p] = 1.0
        cmd = [
            FFMPEG,
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "f64le",
            "-ar",
            str(sr),
            "-ac",
            "1",
            "-i",
            "-",
            "-f",
            "f64le",
            "-acodec",
            "pcm_f64le",
            "-ar",
            "16000",
            "-",
        ]
        y = np.frombuffer(
            subprocess.run(cmd, input=x.tobytes(), capture_output=True, check=True).stdout,
            np.float64,
        )
        for k in range(900, 1100):
            t = k * ratio
            b = t.numerator // t.denominator
            row = (t - b) * prm["phases"]
            assert row.denominator == 1
            i = p - b - r0
            want = bank[int(row), i] if 0 <= i < prm["n_taps"] else 0.0
            assert abs(y[k] - want) < 1e-15, (p, k)


@pytest.mark.parametrize("sr", [8000, 22050, 44100, 48000])
def test_output_length_rule(sr):
    rng = np.random.default_rng(sr)
    ns = rng.integers(200, 50000, 40)
    diffs = []
    for n in ns:
        got = len(ffmpeg_raw(np.zeros(int(n), np.int16), "s16", sr))
        diffs.append(ref.ffmpeg_out_len(int(n), sr) - got)
    diffs = np.array(diffs)
    if sr in EXACT_LENGTH_RATES:
        assert np.all(diffs == 0)
    else:
        assert np.all(np.abs(diffs) <= 1)
        assert np.mean(diffs != 0) < 0.15


def test_wav_file_matches_raw(tmp_path):
    """Container input (whisper's exact command line) == raw pipe input."""
    sr = 44100
    x = make_samples(np.random.default_rng(0), "s16", 30000, 2)
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(x.tobytes())
    expected = ffmpeg_file(path)
    np.testing.assert_array_equal(expected, ffmpeg_raw(x, "s16", sr))
    check(ref.ffmpeg_load_ref(x, "s16", sr), expected, sr, True, "wav")


@pytest.mark.parametrize("sr,sample_fmt", [(44100, "s16"), (48000, "s32"), (22050, "f32")])
def test_speech_end_to_end(tmp_path, sr, sample_fmt):
    """11 s of real speech (jfk) as a stereo file at another rate, through whisper's command."""
    from audio_signals import DATA_DIR

    src = DATA_DIR / "jfk.wav"
    codec = {"s16": "pcm_s16le", "s32": "pcm_s24le", "f32": "pcm_f32le"}[sample_fmt]
    path = tmp_path / "speech.wav"
    subprocess.run(
        [
            FFMPEG,
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-ar",
            str(sr),
            "-ac",
            "2",
            "-acodec",
            codec,
            str(path),
        ],
        check=True,
    )
    # decoded samples as ffmpeg's decoder delivers them (s24 WAV -> s32 with the sample on top)
    dec_fmt = {"s16": "s16le", "s32": "s32le", "f32": "f32le"}[sample_fmt]
    cmd = [FFMPEG, "-nostdin", "-loglevel", "error", "-i", str(path), "-f", dec_fmt, "-"]
    dtype = {"s16": np.int16, "s32": np.int32, "f32": np.float32}[sample_fmt]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype).reshape(-1, 2)
    check(ref.ffmpeg_load_ref(x, sample_fmt, sr), ffmpeg_file(path), sr, True, path.name)
