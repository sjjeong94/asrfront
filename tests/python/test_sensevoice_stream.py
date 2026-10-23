"""sensevoice.FbankStream: the same rows as sensevoice.fbank, however the audio is split."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from asrfront import sensevoice


def _chunks(x: np.ndarray, rng: np.random.Generator, max_size: int) -> list[np.ndarray]:
    cuts = np.cumsum(rng.integers(0, max_size, size=len(x) // max(1, max_size // 2) + 2))
    return np.split(x, cuts[cuts < len(x)])


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"lfr_m": 1, "lfr_n": 1},
        {"lfr_m": 3, "lfr_n": 8},
        {"dither": 1.0, "seed": 3},
        {"n_mels": 40, "lfr_m": 5, "lfr_n": 2},
    ],
)
@pytest.mark.parametrize("n", [0, 399, 400, 7000, 48000])
def test_stream_matches_fbank(kwargs, n):
    rng = np.random.default_rng(n)
    x = (rng.standard_normal(n) * 0.1).astype(np.float32)
    want = sensevoice.fbank(x, **kwargs)
    stream = sensevoice.FbankStream(**kwargs)
    for max_size in (1, 160, 1000, n + 1):  # also: a finished stream starts over
        parts = [stream.push(c) for c in _chunks(x, rng, max_size)] if n else []
        got = np.concatenate([*parts, stream.finish()])
        assert got.dtype == np.float32 and got.shape == (len(want), stream.feat_dim)
        np.testing.assert_array_equal(got, want)


def test_stream_cmvn_and_int16():
    dim = 80 * 7
    cmvn = (np.linspace(-1, 1, dim, dtype=np.float32), np.linspace(0.5, 2, dim, dtype=np.float32))
    x = (np.random.default_rng(0).standard_normal(16000) * 3000).astype(np.int16)
    stream = sensevoice.FbankStream(cmvn=cmvn)
    got = np.concatenate([stream.push(x[:5000]), stream.push(x[5000:]), stream.finish()])
    np.testing.assert_array_equal(got, sensevoice.fbank(x, cmvn=cmvn))


def test_stream_latency():
    """Rows come out as soon as they are final: row i needs frames up to 6 i + 3, which the
    first 60 (i + 1) ms of audio hold, so every 60 ms push completes one row."""
    stream = sensevoice.FbankStream()
    counts = [len(stream.push(np.zeros(960, np.float32))) for _ in range(20)]  # 60 ms each
    assert counts == [1] * 20


def test_stream_errors_and_reset():
    stream = sensevoice.FbankStream()
    stream.push(np.zeros(4000, np.float32))
    with pytest.raises(ValueError, match="NaN"):
        stream.push(np.array([0.0, np.nan], np.float32))
    stream.reset()
    assert stream.finish().shape == (0, 560)
    with pytest.raises(ValueError):
        sensevoice.FbankStream(lfr_m=0)
    with pytest.raises(ValueError):
        stream.push(np.zeros((2, 10), np.float32))


def test_stream_concurrent_use_refused():
    stream = sensevoice.FbankStream()
    x = np.zeros(16000 * 60, np.float32)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            for _ in range(5):
                stream.push(x)
        except RuntimeError as e:
            errors.append(e)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all("two threads" in str(e) for e in errors)
