"""asrfront.sensevoice vs. the float64 reference."""

from __future__ import annotations

import os
import threading

import numpy as np
import pytest

import asrfront
from asrfront import _ext

import reference as ref
from audio_signals import SIGNAL_NAMES
from helpers import assert_close

ATOL = 1e-3  # log domain


def _synthetic_cmvn(dim: int = 560) -> tuple[np.ndarray, np.ndarray]:
    """Shift/scale in the same range as SenseVoiceSmall's am.mvn (mean ~ -10, 1/std ~ 0.3)."""
    rng = np.random.default_rng(0)
    shift = (-10.0 + rng.standard_normal(dim)).astype(np.float32)
    scale = (0.3 + 0.05 * rng.random(dim)).astype(np.float32)
    return shift, scale


def _write_am_mvn(path, shift, scale) -> None:
    fmt = lambda v: " ".join(repr(float(x)) for x in v)  # noqa: E731
    path.write_text(
        "<Nnet>\n"
        f"<Splice> {shift.size} {shift.size}\n"
        "[ 0 ]\n"
        f"<AddShift> {shift.size} {shift.size}\n"
        f"<LearnRateCoef> 0 [ {fmt(shift)} ]\n"
        f"<Rescale> {scale.size} {scale.size}\n"
        f"<LearnRateCoef> 0 [ {fmt(scale)} ]\n"
        "</Nnet>\n"
    )


# ---------------------------------------------------------------------------------------------
# Accuracy vs. reference
# ---------------------------------------------------------------------------------------------


def test_mel_banks_match_reference():
    assert_close(_ext._kaldi_mel_banks(80), ref.kaldi_mel_banks(80), 1e-7, "kaldi banks")


@pytest.mark.parametrize("name", SIGNAL_NAMES)
def test_fbank_raw_matches_reference(sigs, name):
    audio = sigs[name]
    actual = asrfront.sensevoice.fbank_raw(audio)
    assert actual.dtype == np.float32
    assert_close(actual, ref.kaldi_fbank(audio), ATOL, name)


@pytest.mark.parametrize("name", SIGNAL_NAMES)
def test_fbank_without_cmvn_matches_reference(sigs, name):
    audio = sigs[name]
    assert_close(asrfront.sensevoice.fbank(audio), ref.sensevoice_fbank(audio), ATOL, name)


@pytest.mark.parametrize("name", SIGNAL_NAMES)
def test_fbank_with_cmvn_matches_reference(sigs, name):
    audio = sigs[name]
    cmvn = _synthetic_cmvn()
    actual = asrfront.sensevoice.fbank(audio, cmvn=cmvn)
    # CMVN scales by ~0.35, so the log-domain tolerance carries over.
    assert_close(actual, ref.sensevoice_fbank(audio, cmvn), ATOL, name)


@pytest.mark.parametrize("lfr_m,lfr_n", [(1, 1), (5, 3), (7, 6), (3, 7), (8, 2)])
def test_lfr_params(sigs, lfr_m, lfr_n):
    audio = sigs["speech"]
    actual = asrfront.sensevoice.fbank(audio, lfr_m=lfr_m, lfr_n=lfr_n)
    expected = ref.sensevoice_fbank(audio, lfr_m=lfr_m, lfr_n=lfr_n)
    assert_close(actual, expected, ATOL, f"lfr {lfr_m}/{lfr_n}")


def _lfr_from_raw(raw: np.ndarray, lfr_m: int, lfr_n: int) -> np.ndarray:
    """FunASR apply_lfr on a raw fbank, in NumPy (exact: it only copies)."""
    left = (lfr_m - 1) // 2
    rows = -(-raw.shape[0] // lfr_n)
    idx = np.arange(rows)[:, None] * lfr_n - left + np.arange(lfr_m)
    return raw[np.clip(idx, 0, raw.shape[0] - 1)].reshape(rows, -1)


# Raw frames are kept in a ring of lfr_m + 63 frames: rows far apart (lfr_n > 64), wide rows
# (lfr_m > 64), lengths around the 64-frame chunks, and dither (RNG order).
@pytest.mark.parametrize(
    "lfr_m,lfr_n", [(7, 6), (1, 1), (1, 100), (3, 200), (100, 1), (70, 70), (130, 9)]
)
@pytest.mark.parametrize("n_raw", [1, 63, 64, 65, 129, 1000])
def test_lfr_ring_matches_raw(lfr_m, lfr_n, n_raw):
    rng = np.random.default_rng(n_raw)
    audio = (rng.standard_normal(400 + 160 * (n_raw - 1) + 37) * 0.1).astype(np.float32)
    sv = asrfront.sensevoice
    for dither in (0.0, 1.0):
        raw = sv.fbank_raw(audio, dither=dither, seed=3)
        assert raw.shape[0] == n_raw
        got = sv.fbank(audio, lfr_m=lfr_m, lfr_n=lfr_n, dither=dither, seed=3)
        np.testing.assert_array_equal(got, _lfr_from_raw(raw, lfr_m, lfr_n))


@pytest.mark.parametrize("n_samples", [0, 399, 400, 1000, 16001])
@pytest.mark.parametrize("n_mels,lfr_m,lfr_n", [(80, 7, 6), (128, 8, 2), (80, 3, 7), (80, 1, 1)])
def test_cmvn_preserves_separate_float32_operations(n_samples, n_mels, lfr_m, lfr_n):
    """Fusing the LFR copy with CMVN must retain exact float32 results at edges."""
    audio = np.random.default_rng(42).uniform(-0.5, 0.5, n_samples).astype(np.float32)
    shift, scale = _synthetic_cmvn(n_mels * lfr_m)
    params = dict(n_mels=n_mels, lfr_m=lfr_m, lfr_n=lfr_n)
    unnormalized = asrfront.sensevoice.fbank(audio, **params)
    actual = asrfront.sensevoice.fbank(audio, cmvn=(shift, scale), **params)
    np.testing.assert_array_equal(actual, (unnormalized + shift) * scale)


@pytest.mark.parametrize("name", ["speech", "chirp", "white_noise", "dc_offset"])
def test_fbank_matches_torchaudio(sigs, name):
    """End to end vs. FunASR's float32 torchaudio call (see test_reference for why 2e-3)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    from torchaudio.compliance import kaldi

    audio = sigs[name]
    waveform = torch.from_numpy(audio * np.float32(1 << 15)).unsqueeze(0)
    expected = kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        energy_floor=0.0,
        window_type="hamming",
        sample_frequency=16000,
        snip_edges=True,
    ).numpy()
    assert_close(asrfront.sensevoice.fbank_raw(audio), expected, 2e-3, name)


# ---------------------------------------------------------------------------------------------
# Shapes and inputs
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n,t_raw,t_lfr", [(0, 0, 0), (399, 0, 0), (400, 1, 1), (16000, 98, 17)])
def test_shapes(n, t_raw, t_lfr):
    audio = np.zeros(n, np.float32)
    assert asrfront.sensevoice.fbank_raw(audio).shape == (t_raw, 80)
    assert asrfront.sensevoice.fbank(audio).shape == (t_lfr, 560)


def test_int16_input_is_scaled(sigs):
    pcm = np.round(sigs["speech"] * 32768).clip(-32768, 32767).astype(np.int16)
    expected = ref.sensevoice_fbank(pcm.astype(np.float64) / 32768.0)
    assert_close(asrfront.sensevoice.fbank(pcm), expected, ATOL, "int16")


def test_cmvn_validation(sigs):
    shift, scale = _synthetic_cmvn(100)
    with pytest.raises(ValueError, match="need n_mels"):
        asrfront.sensevoice.fbank(sigs["speech"], cmvn=(shift, scale))
    # Longer vectors: only the first n_mels * lfr_m entries are used (as in FunASR).
    long_cmvn = _synthetic_cmvn(600)
    expected = ref.sensevoice_fbank(sigs["speech"], long_cmvn)
    assert_close(asrfront.sensevoice.fbank(sigs["speech"], cmvn=long_cmvn), expected, ATOL, "")


# ---------------------------------------------------------------------------------------------
# am.mvn loading
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["shift", "scale"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_cmvn_rejects_non_finite(sigs, which, value):
    shift, scale = _synthetic_cmvn()
    (shift if which == "shift" else scale)[17] = value
    with pytest.raises(ValueError, match=f"cmvn {which} contains NaN or inf.*index 17"):
        asrfront.sensevoice.fbank(sigs["speech"], cmvn=(shift, scale))
    shift[:] = 0  # entries past n_mels * lfr_m are not used, so not checked
    scale[:] = 1
    long = (np.append(shift, value), np.append(scale, value))
    asrfront.sensevoice.fbank(sigs["speech"], cmvn=long)


def test_load_cmvn_matches_reference_parser(tmp_path):
    shift, scale = _synthetic_cmvn()
    path = tmp_path / "am.mvn"
    _write_am_mvn(path, shift, scale)
    got = asrfront.sensevoice.load_cmvn(path)
    want = ref.load_cmvn(path)
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])
    np.testing.assert_array_equal(got[0], shift)


def test_cmvn_path_argument(sigs, tmp_path):
    shift, scale = _synthetic_cmvn()
    path = tmp_path / "am.mvn"
    _write_am_mvn(path, shift, scale)
    by_path = asrfront.sensevoice.fbank(sigs["speech"], cmvn=path)
    by_str = asrfront.sensevoice.fbank(sigs["speech"], cmvn=str(path))
    by_arrays = asrfront.sensevoice.fbank(sigs["speech"], cmvn=(shift, scale))
    np.testing.assert_array_equal(by_path, by_arrays)
    np.testing.assert_array_equal(by_str, by_arrays)


def test_load_cmvn_sees_file_changes(tmp_path):
    path = tmp_path / "am.mvn"
    _write_am_mvn(path, np.zeros(4, np.float32), np.ones(4, np.float32))
    assert asrfront.sensevoice.load_cmvn(path)[0].tolist() == [0, 0, 0, 0]
    _write_am_mvn(path, np.full(5, 2, np.float32), np.ones(5, np.float32))
    assert asrfront.sensevoice.load_cmvn(path)[0].tolist() == [2, 2, 2, 2, 2]


def test_fbank_cmvn_path_sees_file_changes(sigs, tmp_path):
    path = tmp_path / "am.mvn"
    first, second = _synthetic_cmvn(), (np.zeros(560, np.float32), np.ones(560, np.float32))
    for shift, scale in (first, second, first):
        _write_am_mvn(path, shift, scale)
        os.utime(path, ns=(0, path.stat().st_mtime_ns + 10**9))  # new mtime even on coarse clocks
        np.testing.assert_array_equal(
            asrfront.sensevoice.fbank(sigs["speech"], cmvn=path),
            asrfront.sensevoice.fbank(sigs["speech"], cmvn=(shift, scale)),
        )


def test_cmvn_caches_see_replacement_with_same_mtime_and_size(sigs, tmp_path):
    path = tmp_path / "am.mvn"
    shift, scale = np.zeros(560, np.float32), np.ones(560, np.float32)
    _write_am_mvn(path, shift, scale)
    before = path.stat()
    np.testing.assert_array_equal(asrfront.sensevoice.load_cmvn(path)[0], shift)
    asrfront.sensevoice.fbank(sigs["speech"], cmvn=path)  # warm both caches

    replacement = tmp_path / "replacement.mvn"
    shift = np.full(560, 2, np.float32)
    _write_am_mvn(replacement, shift, scale)
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    replacement.replace(path)
    after = path.stat()
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    assert after.st_ino != before.st_ino

    np.testing.assert_array_equal(asrfront.sensevoice.load_cmvn(path)[0], shift)
    expected = asrfront.sensevoice.fbank(sigs["speech"], cmvn=(shift, scale))
    np.testing.assert_array_equal(asrfront.sensevoice.fbank(sigs["speech"], cmvn=path), expected)
    feats, _ = asrfront.sensevoice.fbank_batch([sigs["speech"]] * 2, cmvn=path, n_threads=2)
    for item in feats:
        np.testing.assert_array_equal(item, expected)


@pytest.mark.parametrize(
    "text,match",
    [
        ("<Nnet>\n</Nnet>\n", "missing"),
        ("<AddShift> 2 2\n<LearnRateCoef> 0 [ 1 2 ]\n", "missing <Rescale>"),
        ("<AddShift> 2 2\n<LearnRateCoef> 0 1 2\n<Rescale> 2 2\n", "expected"),
        (
            "<AddShift> 2 2\n<LearnRateCoef> 0 [ 1 2 ]\n"
            "<Rescale> 3 3\n<LearnRateCoef> 0 [ 1 2 3 ]\n",
            "<AddShift> has 2",
        ),
    ],
)
def test_load_cmvn_errors(tmp_path, text, match):
    path = tmp_path / "bad.mvn"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        asrfront.sensevoice.load_cmvn(path)


@pytest.mark.skipif(
    "ASRFRONT_SENSEVOICE_MVN" not in os.environ,
    reason="set ASRFRONT_SENSEVOICE_MVN to SenseVoiceSmall's am.mvn",
)
def test_real_am_mvn(sigs):
    path = os.environ["ASRFRONT_SENSEVOICE_MVN"]
    shift, scale = asrfront.sensevoice.load_cmvn(path)
    want = ref.load_cmvn(path)
    np.testing.assert_array_equal(shift, want[0])
    np.testing.assert_array_equal(scale, want[1])
    assert shift.shape == (560,)
    expected = ref.sensevoice_fbank(sigs["speech"], (shift, scale))
    assert_close(asrfront.sensevoice.fbank(sigs["speech"], cmvn=path), expected, ATOL, "")


# ---------------------------------------------------------------------------------------------
# Dither, determinism, threads
# ---------------------------------------------------------------------------------------------


def test_dither_is_seeded_and_small(sigs):
    audio = sigs["speech"]
    clean = asrfront.sensevoice.fbank_raw(audio)
    a = asrfront.sensevoice.fbank_raw(audio, dither=1.0, seed=1)
    b = asrfront.sensevoice.fbank_raw(audio, dither=1.0, seed=1)
    c = asrfront.sensevoice.fbank_raw(audio, dither=1.0, seed=2)
    np.testing.assert_array_equal(a, b)  # reseeded per call
    assert not np.array_equal(a, c)
    assert not np.array_equal(a, clean)
    # Unit-variance noise on an int16-scale signal barely moves speech energies.
    assert np.median(np.abs(a - clean)) < 0.05


def test_dither_noise_floor(sigs):
    """On silence, dither=1 is unit white noise: E[mel] = banks @ (sum(win^2) * |H_preemph|^2)."""
    out = asrfront.sensevoice.fbank_raw(sigs["silence"], dither=1.0, seed=0)
    win = ref.hamming_window_symmetric(400)
    k = np.arange(257)
    preemph_gain = np.abs(1 - 0.97 * np.exp(-2j * np.pi * k / 512)) ** 2
    expected = ref.kaldi_mel_banks(80) @ (np.sum(win**2) * preemph_gain)
    # Compare in the linear domain (log of a noisy power is biased low); skip the lowest
    # bins where per-frame DC removal eats into the noise.
    ratio = np.exp(out.astype(np.float64)).mean(axis=0)[10:] / expected[10:]
    assert np.all(np.abs(ratio - 1) < 0.3), ratio
    assert abs(float(np.mean(ratio)) - 1) < 0.05


def test_deterministic_and_reusable(sigs):
    fe = _ext.SenseVoiceFrontend()
    a = fe.fbank(sigs["speech"])
    fe.fbank(sigs["speech_35s"])  # grows the internal buffer
    fe.fbank(sigs["short_1000"])
    b = fe.fbank(sigs["speech"])
    np.testing.assert_array_equal(a, b)


def test_concurrent_threads(sigs):
    names = ["speech", "chirp", "white_noise", "sine_440"]
    expected = {n: asrfront.sensevoice.fbank(sigs[n]) for n in names}
    errors = []

    def worker(name):
        for _ in range(5):
            if not np.array_equal(asrfront.sensevoice.fbank(sigs[name]), expected[name]):
                errors.append(name)

    threads = [threading.Thread(target=worker, args=(n,)) for n in names * 2]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
