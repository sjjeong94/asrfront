"""C resampler (csrc/resample.c) vs. the reference in reference.py (docs/load.md).

The reference itself is checked against ffmpeg in test_ffmpeg_ref.py; here the C code must
match it bit for bit, on every path and rate, without needing ffmpeg.
"""

from __future__ import annotations

import numpy as np
import pytest

from asrfront import _ext

import reference as ref

EXACT_RATES = [8000, 11025, 12000, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 192000]
INTERP_RATES = [16494, 44056, 15800 + 7]
ALL_RATES = EXACT_RATES + INTERP_RATES + [16200, 15800, 1000, 384000]


def _signal(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 16000
    x = 0.3 * rng.standard_normal(n) + 0.5 * np.sin(2 * np.pi * 997 * t)
    x[rng.integers(0, n, 10)] = 1.0  # full-scale peaks: filter overshoot must saturate
    x[rng.integers(0, n, 10)] = -1.0
    return x


@pytest.mark.parametrize("sr", ALL_RATES)
def test_filter_bank(sr):
    c, r = _ext._resample_bank(sr), ref.ffmpeg_filter_bank(sr)
    assert c.shape == r.shape
    np.testing.assert_allclose(c, r, rtol=0, atol=2e-15)
    np.testing.assert_array_equal(c.astype(np.float32), r.astype(np.float32))


@pytest.mark.parametrize("sr", ALL_RATES)
@pytest.mark.parametrize("path", ["float", "double", "int16"])
def test_resample_matches_reference(sr, path):
    x = _signal(max(3000, sr // 4), sr)
    if path == "float":
        x = x.astype(np.float32)
    elif path == "int16":
        x = np.clip(np.rint(x * 32767), -32768, 32767).astype(np.int32)
    got = _ext._resample(x, sr)
    want = ref.ffmpeg_resample(x, sr, path)
    assert len(got) == len(want) == ref.ffmpeg_out_len(len(x), sr)
    if path == "double":  # summation order may differ in the last bits
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-13)
    else:
        np.testing.assert_array_equal(got, want.astype(got.dtype))


@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 31, 50, 99, 100, 101, 257])
@pytest.mark.parametrize("sr", [8000, 44100, 48000, 96000])
def test_short_inputs(sr, n):
    """Inputs shorter than the filter: repeated mirroring, no out-of-bounds access."""
    x = _signal(max(n, 1), n)[:n].astype(np.float32)
    got = _ext._resample(x, sr)
    np.testing.assert_array_equal(got, ref.ffmpeg_resample(x, sr, "float"))


def test_out_len():
    for sr in [8000, 22050, 32000, 32001, 44100, 48000]:
        for n in [0, 1, 2, 10, 1000, 123457]:
            x = np.zeros(n, np.float32)
            assert len(_ext._resample(x, sr)) == ref.ffmpeg_out_len(n, sr), (sr, n)


def test_same_rate_is_filtered_like_reference():
    """sr_in == sr_out still filters (the loader never calls the resampler at 16 kHz)."""
    x = _signal(1000, 0).astype(np.float32)
    np.testing.assert_array_equal(_ext._resample(x, 16000), ref.ffmpeg_resample(x, 16000, "float"))


def test_unsupported_rates():
    for sr in [0, 999, 384001]:
        with pytest.raises(ValueError):
            _ext._resample(np.zeros(10, np.float32), sr)
    with pytest.raises(TypeError):
        _ext._resample(np.zeros(10, np.int16), 44100)
