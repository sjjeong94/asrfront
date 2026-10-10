from __future__ import annotations

import os
import platform
from pathlib import Path

import numpy as np
import pytest


def assert_close(actual: np.ndarray, expected: np.ndarray, atol: float, what: str = "") -> None:
    """Shape-checked absolute-error comparison with an informative failure message."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape, f"{what}: shape {actual.shape} != {expected.shape}"
    if actual.size == 0:
        return
    err = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    max_err = float(err.max())
    if max_err > atol:
        idx = np.unravel_index(int(err.argmax()), err.shape)
        pytest.fail(
            f"{what}: max abs err {max_err:.3e} > {atol:.1e} at {idx} "
            f"(actual={actual[idx]!r}, expected={expected[idx]!r}, "
            f"mean abs err={float(err.mean()):.3e})"
        )


def ffmpeg_is_reference_build() -> bool:
    """Whether ffmpeg's float arithmetic is that of the build docs/load.md was measured on.

    That is ffmpeg on x86-64 with AVX2 + FMA3: there asrfront must match the float path bit for
    bit. Elsewhere (aarch64, x86 without FMA3, other OSes where the CPU flags are not checked)
    ffmpeg's float resampling kernel and its float -> s16 rounding (round half up on aarch64)
    differ by up to 1 LSB, and the tests allow that on the float path. The int16 and double
    paths stay exact everywhere. ASRFRONT_FFMPEG_EXACT=1 / 0 overrides the detection.
    """
    env = os.environ.get("ASRFRONT_FFMPEG_EXACT")
    if env in ("0", "1"):
        return env == "1"
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return False
    try:
        flags = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    return " avx2" in flags and " fma" in flags
