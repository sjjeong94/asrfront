from __future__ import annotations

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
