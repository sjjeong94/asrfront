"""Public API surface: input handling, validation, batch API, low-level ``out=``."""

from __future__ import annotations

import functools
import platform

import numpy as np
import pytest

import asrfront
from asrfront import _ext
from asrfront.sensevoice import fbank, fbank_batch, fbank_raw
from asrfront.whisper import log_mel, log_mel_batch

import reference as ref
from helpers import assert_close


def test_version():
    from importlib.metadata import version

    assert asrfront.__version__ == version("asrfront")


# ---------------------------------------------------------------------------------------------
# Low-level extension
# ---------------------------------------------------------------------------------------------


def test_whisper_out_frames():
    assert _ext.WhisperFrontend().out_frames(16000) == 3000
    assert _ext.WhisperFrontend(pad_or_trim=False).out_frames(16000) == 100
    with pytest.raises(ValueError):
        _ext.WhisperFrontend(n_mels=64)


def test_sensevoice_out_frames():
    fe = _ext.SenseVoiceFrontend()
    assert fe.feat_dim == 560
    assert fe.raw_frames(16000) == 98
    assert fe.out_frames(16000) == 17


def test_sensevoice_cmvn_length_checked():
    with pytest.raises(ValueError):
        _ext.SenseVoiceFrontend(
            cmvn_shift=np.zeros(10, np.float32), cmvn_scale=np.ones(10, np.float32)
        )


def test_out_argument(sigs):
    audio = sigs["speech"]
    fe = _ext.WhisperFrontend()
    out = np.full((80, 3000), np.nan, np.float32)
    assert fe.log_mel(audio, out=out) is None
    np.testing.assert_array_equal(out, fe.log_mel(audio))

    sv = _ext.SenseVoiceFrontend()
    out = np.empty((sv.out_frames(audio.size), sv.feat_dim), np.float32)
    assert sv.fbank(audio, out=out) is None
    np.testing.assert_array_equal(out, sv.fbank(audio))


@pytest.mark.parametrize("pad_or_trim,n", [(False, 16000), (False, 333), (True, 16000)])
def test_out_argument_row_stride(sigs, pad_or_trim, n):
    """log_mel writes into rows spaced apart (a[:, :frames]) and leaves the rest alone."""
    audio = sigs["speech"][:n]
    fe = _ext.WhisperFrontend(pad_or_trim=pad_or_trim)
    frames = fe.out_frames(n)
    big = np.full((80, frames + 7), 5.0, np.float32)
    assert fe.log_mel(audio, out=big[:, :frames]) is None
    np.testing.assert_array_equal(big[:, :frames], fe.log_mel(audio))
    assert (big[:, frames:] == 5.0).all()

    sv = _ext.SenseVoiceFrontend()  # other outputs still need C-contiguous arrays
    rows = np.empty((sv.out_frames(16000), sv.feat_dim + 1), np.float32)[:, :-1]
    with pytest.raises(TypeError):
        sv.fbank(sigs["speech"][:16000], out=rows)


@pytest.mark.parametrize(
    "out,exc",
    [
        (np.empty((80, 2999), np.float32), ValueError),  # wrong shape
        (np.empty((80, 3000), np.float64), TypeError),  # wrong dtype: never converted
        (np.empty((3000, 80), np.float32).T, ValueError),  # rows not contiguous
        (
            np.lib.stride_tricks.as_strided(
                np.empty(80 * 3000, np.float32), (80, 3000), (4 * 2999, 4)
            ),
            ValueError,
        ),  # overlapping rows
        (np.empty((80, 3000), np.float32), TypeError),  # read-only (set below)
    ],
)
def test_out_argument_rejected(sigs, out, exc):
    if out.dtype == np.float32 and out.flags.c_contiguous and out.shape == (80, 3000):
        out.flags.writeable = False
    with pytest.raises(exc):
        _ext.WhisperFrontend().log_mel(sigs["speech"], out=out)


# ---------------------------------------------------------------------------------------------
# Input normalization and validation
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fn", [log_mel, fbank, fbank_raw])
def test_rejects_multichannel(fn):
    with pytest.raises(ValueError, match=r"1-D mono waveform.*mean\(axis=0\)"):
        fn(np.zeros((2, 16000), np.float32))
    with pytest.raises(ValueError, match="1-D mono"):
        fn(np.float32(0.5))


@pytest.mark.parametrize("fn", [log_mel, fbank, fbank_raw])
@pytest.mark.parametrize("dtype", [np.int32, np.uint8, np.int64])
def test_rejects_ambiguous_integer_dtypes(fn, dtype):
    with pytest.raises(TypeError, match="only int16 PCM"):
        fn(np.zeros(16000, dtype))


@pytest.mark.parametrize("fn", [log_mel, fbank, fbank_raw])
@pytest.mark.parametrize("dtype", [np.complex64, np.bool_, object])
def test_rejects_other_dtypes(fn, dtype):
    with pytest.raises(TypeError):
        fn(np.zeros(16000, dtype))


@pytest.mark.parametrize("fn", [log_mel, fbank, fbank_raw])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1e40])  # 1e40: inf as float32
def test_rejects_non_finite(fn, value):
    audio = np.zeros(16000)
    audio[1234] = value
    with pytest.raises(ValueError, match="NaN or inf.*index 1234"):
        fn(audio)


def test_pad_or_trim_uses_only_the_first_30_s():
    audio = np.random.default_rng(0).standard_normal(3 * 480000) * 0.1  # float64, 90 s
    expected = log_mel(audio[:480000].astype(np.float32))
    audio[480000] = np.nan  # dropped by pad_or_trim, as in whisper: not an error
    np.testing.assert_array_equal(log_mel(audio), expected)
    np.testing.assert_array_equal(log_mel_batch([audio, audio])[1], expected)
    with pytest.raises(ValueError, match="NaN or inf"):
        log_mel(audio, pad_or_trim=False)
    audio[479999] = np.inf
    with pytest.raises(ValueError, match="index 479999"):
        log_mel(audio)
    with pytest.raises(ValueError, match=r"audios\[0\] .*index 479999"):
        log_mel_batch([audio])


def test_batch_rejects_non_finite():
    items = [np.zeros(16000, np.float32), np.full(16000, np.nan, np.float32)]
    with pytest.raises(ValueError, match=r"audios\[1\] contains NaN or inf"):
        log_mel_batch(items)
    with pytest.raises(ValueError, match=r"audios\[1\] contains NaN or inf"):
        fbank_batch(items)


@pytest.mark.parametrize("n_threads", [1, 4])
def test_batch_non_finite_error_is_the_lowest_index(n_threads):
    # The NaN / inf check runs in the workers; the error is still the first bad item's.
    items = [np.zeros(16000, np.float32) for _ in range(12)]
    for i in (9, 3, 7):
        items[i] = items[i].copy()
        items[i][5] = np.nan
    items[5] = np.zeros(16000)  # float64, converted in a worker
    items[5][77] = 1e40
    hf = functools.partial(asrfront.qwen3_asr.log_mel_batch, hf_padding=True)
    for fn in (log_mel_batch, fbank_batch, asrfront.qwen3_asr.log_mel_batch, hf):
        with pytest.raises(ValueError, match=r"audios\[3\] contains NaN or inf \(first at index 5"):
            fn(items, n_threads=n_threads)
        with pytest.raises(ValueError, match=r"audios\[4\] .*index 77, value .*1e\+40"):
            fn(items[:3] + items[4:7], n_threads=n_threads)  # item 5 is now 4


def test_batch_shape_and_dtype_errors_come_first():
    # Checked before any work, so they win over a NaN in an earlier item.
    items = [np.full(16000, np.nan, np.float32), np.zeros(16000, np.int32)]
    for fn in (log_mel_batch, fbank_batch, asrfront.qwen3_asr.log_mel_batch):
        with pytest.raises(TypeError, match=r"audios\[1\] has integer dtype int32"):
            fn(items)


def test_big_endian_int16(sigs):
    pcm = (sigs["speech"] * 32767).astype(np.int16)
    np.testing.assert_array_equal(log_mel(pcm.astype(">i2")), log_mel(pcm))
    np.testing.assert_array_equal(fbank(pcm.astype(">i2")), fbank(pcm))


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_float_dtypes(sigs, dtype):
    audio = sigs["speech"].astype(dtype)
    expected = ref.whisper_log_mel(audio.astype(np.float64))
    assert_close(log_mel(audio), expected, 1e-4, str(dtype))


def test_torch_tensor_input(sigs):
    torch = pytest.importorskip("torch")
    audio = sigs["speech"]
    t = torch.from_numpy(audio.copy())
    np.testing.assert_array_equal(log_mel(t), log_mel(audio))
    np.testing.assert_array_equal(fbank(t), fbank(audio))


@pytest.mark.parametrize("n_mels", [64, 0, True, "80"])
def test_whisper_n_mels_validation(n_mels):
    with pytest.raises(ValueError, match="80 or 128"):
        log_mel(np.zeros(16000, np.float32), n_mels=n_mels)


@pytest.mark.parametrize(
    "kwargs,exc,match",
    [
        ({"n_mels": 0}, ValueError, "n_mels"),
        ({"n_mels": 80.0}, TypeError, "n_mels must be an int"),
        ({"lfr_m": 0}, ValueError, "lfr_m"),
        ({"lfr_n": -1}, ValueError, "lfr_n"),
        ({"dither": -1.0}, ValueError, "dither"),
        ({"dither": float("nan")}, ValueError, "dither"),
        ({"dither": "x"}, TypeError, "dither"),
        ({"seed": -1}, ValueError, "seed"),
        ({"seed": 2**64}, ValueError, "seed"),
        ({"cmvn": 5}, TypeError, "cmvn"),
    ],
)
def test_sensevoice_param_validation(kwargs, exc, match):
    with pytest.raises(exc, match=match):
        fbank(np.zeros(16000, np.float32), **kwargs)


def test_seed_full_range(sigs):
    a = fbank_raw(sigs["speech"], dither=1.0, seed=2**64 - 1)
    b = fbank_raw(sigs["speech"], dither=1.0, seed=2**64 - 1)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_threads", [1, 4, None])
def test_log_mel_batch(sigs, n_threads):
    names = ["speech", "chirp", "short_201", "silence", "speech_35s", "white_noise"]
    audios = [sigs[n] for n in names]
    out = log_mel_batch(audios, n_threads=n_threads)
    assert isinstance(out, np.ndarray) and out.shape == (len(names), 80, 3000)
    for i, a in enumerate(audios):
        np.testing.assert_array_equal(out[i], log_mel(a))


@pytest.mark.parametrize("pad_or_trim", [True, False])
def test_log_mel_channel_last(sigs, pad_or_trim):
    audio = sigs["speech"]
    mel = log_mel(audio, n_mels=128, pad_or_trim=pad_or_trim)
    out = log_mel(audio, n_mels=128, pad_or_trim=pad_or_trim, channel_last=True)
    assert out.shape == mel.shape[::-1] and out.flags.c_contiguous
    np.testing.assert_array_equal(out, mel.T)


@pytest.mark.parametrize(
    "fn, default",
    [
        (asrfront.qwen3_asr.log_mel, False),
        (asrfront.gemma4.log_mel, True),
        (fbank, True),
        (fbank_raw, True),
    ],
)
def test_channel_last_other_frontends(sigs, fn, default):
    audio = sigs["speech"]
    feat = fn(audio)
    np.testing.assert_array_equal(fn(audio, channel_last=default), feat)
    out = fn(audio, channel_last=not default)
    assert out.shape == feat.shape[::-1] and out.flags.c_contiguous
    np.testing.assert_array_equal(out, feat.T)


def test_log_mel_batch_unpadded_and_128(sigs):
    audios = [sigs["speech"], sigs["short_1000"], sigs["odd_length"]]
    out = log_mel_batch(audios, n_mels=128, pad_or_trim=False, n_threads=2)
    assert isinstance(out, list) and [o.shape for o in out] == [(128, 1100), (128, 6), (128, 100)]
    for o, a in zip(out, audios):
        np.testing.assert_array_equal(o, log_mel(a, n_mels=128, pad_or_trim=False))


def test_log_mel_batch_2d_array_and_empty(sigs):
    batch = np.stack([sigs["sine_440"][:16000], sigs["silence"]])
    out = log_mel_batch(batch)
    np.testing.assert_array_equal(out[0], log_mel(batch[0]))
    assert log_mel_batch([]).shape == (0, 80, 3000)


def test_log_mel_batch_errors(sigs):
    with pytest.raises(ValueError, match=r"audios\[1\] needs more than 200"):
        log_mel_batch([sigs["speech"], sigs["short_201"][:100]], pad_or_trim=False)
    with pytest.raises(TypeError, match=r"audios\[0\] has integer dtype"):
        log_mel_batch([np.zeros(10, np.int32)])
    with pytest.raises(TypeError, match="sequence"):
        log_mel_batch(5)
    with pytest.raises(ValueError, match="n_threads"):
        log_mel_batch([sigs["speech"]], n_threads=0)


@pytest.mark.parametrize("n_threads", [1, 3, None])
def test_fbank_batch(sigs, n_threads):
    names = ["speech", "short_399", "chirp", "short_1000", "speech_35s"]
    audios = [sigs[n] for n in names]
    cmvn = (np.full(560, -10, np.float32), np.full(560, 0.3, np.float32))
    feats, lengths = fbank_batch(audios, cmvn=cmvn, n_threads=n_threads)
    singles = [fbank(a, cmvn=cmvn) for a in audios]
    assert lengths.dtype == np.int32
    assert lengths.tolist() == [s.shape[0] for s in singles]
    assert feats.shape == (len(audios), max(lengths), 560)
    for i, s in enumerate(singles):
        np.testing.assert_array_equal(feats[i, : lengths[i]], s)
        assert not feats[i, lengths[i] :].any()  # zero padding


def test_fbank_batch_with_path_and_empty(sigs, tmp_path):
    path = tmp_path / "am.mvn"
    vals = " ".join(["0.5"] * 560)
    path.write_text(
        f"<AddShift> 560 560\n<LearnRateCoef> 0 [ {vals} ]\n"
        f"<Rescale> 560 560\n<LearnRateCoef> 0 [ {vals} ]\n"
    )
    feats, lengths = fbank_batch([sigs["speech"]], cmvn=path)
    np.testing.assert_array_equal(feats[0], fbank(sigs["speech"], cmvn=path))
    feats, lengths = fbank_batch([])
    assert feats.shape == (0, 0, 560) and lengths.shape == (0,)


# ---------------------------------------------------------------------------------------------
# Worker pool and handle cache (asrfront._audio)
# ---------------------------------------------------------------------------------------------


def test_batch_reuses_pool_and_handles(sigs):
    from asrfront import _audio

    audios = [sigs["speech"]] * 4
    log_mel_batch(audios, n_threads=2)
    pool = _audio._pool
    log_mel_batch(audios, n_threads=2)
    assert _audio._pool is pool  # no new threads per call


def test_batch_with_more_threads_than_cores(sigs):
    import threading
    import time

    from asrfront import _audio

    log_mel_batch([sigs["speech"]] * 2, n_threads=2)
    pool, before = _audio._pool, threading.active_count()
    more = _audio._available_cores() + 3
    out = log_mel_batch([sigs["speech"]] * more, n_threads=more)
    np.testing.assert_array_equal(out[-1], log_mel(sigs["speech"]))
    assert _audio._pool is pool and pool._max_workers == _audio._available_cores()
    for _ in range(100):  # the extra pool's threads exit
        if threading.active_count() <= before:
            break
        time.sleep(0.01)
    assert threading.active_count() <= before


def test_run_parallel_limits_concurrency_and_raises():
    import threading
    import time

    from asrfront._audio import run_parallel

    active, peak, lock = [0], [0], threading.Lock()

    def work(i):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.002)
        with lock:
            active[0] -= 1

    run_parallel(work, 40, 3)
    assert peak[0] <= 3

    done = []

    def fail(i):
        if i == 5:
            raise RuntimeError("boom")
        done.append(i)

    with pytest.raises(RuntimeError, match="boom"):
        run_parallel(fail, 1000, 4)
    assert len(done) < 999  # the other workers stopped taking items


@pytest.mark.parametrize("own_pool", [False, True])
def test_run_parallel_waits_for_workers_on_interrupt(monkeypatch, own_pool):
    import threading

    from asrfront import _audio

    monkeypatch.setattr(_audio, "_available_cores", lambda: 1 if own_pool else 2)
    started = threading.Barrier(3)
    interrupted = threading.Event()
    release = threading.Event()
    returned = threading.Event()
    done = []
    observed = []

    def work(i):
        started.wait(timeout=5)
        assert release.wait(5)
        done.append(i)

    def interrupted_wait(fs):
        started.wait(timeout=5)  # both workers are writing into caller-owned output
        interrupted.set()
        raise KeyboardInterrupt

    def run():
        try:
            _audio.run_parallel(work, 100, 2)
        except KeyboardInterrupt:
            observed.append(sorted(done))
        finally:
            returned.set()

    monkeypatch.setattr(_audio, "wait", interrupted_wait)
    caller = threading.Thread(target=run)
    caller.start()
    try:
        assert interrupted.wait(5)
        assert not returned.wait(0.1), "returned while workers can still modify output"
    finally:
        release.set()
        caller.join(timeout=5)
    assert not caller.is_alive()
    assert observed == [[0, 1]]
    assert sorted(done) == [0, 1]  # queued items were stopped


def test_free_threaded_build_keeps_the_gil_disabled():
    import os
    import subprocess
    import sys
    import sysconfig

    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        pytest.skip("not a free-threaded CPython")
    env = {k: v for k, v in os.environ.items() if k != "PYTHON_GIL"}
    code = "import sys, asrfront; assert not sys._is_gil_enabled()"
    r = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("warm", [False, True])
def test_batch_at_interpreter_shutdown(warm):
    # concurrent.futures refuses new work before atexit handlers run: items run inline.
    import subprocess
    import sys

    code = f"""
import atexit
import numpy as np
from asrfront import _audio, sensevoice, whisper
_audio._MIN_COST_PER_THREAD = 1  # threads even for these small inputs
x = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
if {warm}:
    whisper.log_mel_batch([x] * 4, n_threads=4)
def at_exit():
    out = whisper.log_mel_batch([x] * 8, n_threads=4)
    assert np.array_equal(out[7], whisper.log_mel(x))
    feats, _ = sensevoice.fbank_batch([x] * 8, n_threads=64)  # own pool, also refused
    assert np.array_equal(feats[7], sensevoice.fbank(x))
    print("ok")
atexit.register(at_exit)
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def test_plan_chunks(monkeypatch):
    from asrfront import _audio

    monkeypatch.setattr(_audio, "_MIN_COST_PER_THREAD", 512000)
    monkeypatch.setattr(_audio, "_ITEM_COST", 2000)
    # little work stays on one thread, in one chunk, whatever n_threads says
    assert _audio.plan_chunks([1600] * 256, None) == ([range(256)], 1)
    assert _audio.plan_chunks([1600] * 256, 16) == ([range(256)], 1)
    assert _audio.plan_chunks([], None) == ([], 1)
    # enough work: workers by total cost, ~4 contiguous chunks each, covering every item once
    chunks, workers = _audio.plan_chunks([160000] * 64, 4)
    assert workers == 4 and len(chunks) == 16
    assert [i for c in chunks for i in c] == list(range(64))
    # unequal items: chunks of about equal cost, not of equal item counts
    costs = [10**6] * 4 + [1000] * 400
    chunks, workers = _audio.plan_chunks(costs, 2)
    assert workers == 2 and [i for c in chunks for i in c] == list(range(404))
    assert len(chunks[0]) == 1 and len(chunks[-1]) > 100
    with pytest.raises(ValueError, match="n_threads"):
        _audio.plan_chunks([1], 0)


def test_batch_many_chunk_errors(sigs):
    fe = _ext.WhisperFrontend(pad_or_trim=False)
    a = sigs["speech"][:16000]
    with pytest.raises(ValueError, match="same length"):
        fe.log_mel_many([a, a], [np.zeros((80, 100), np.float32)])
    with pytest.raises(ValueError, match=r"expected \(80, 100\)"):
        fe.log_mel_many([a], [np.zeros((80, 99), np.float32)])
    with pytest.raises(TypeError):  # never a converted copy that would drop the results
        fe.log_mel_many([a], [np.zeros((80, 100), np.float64)])
    with pytest.raises(TypeError):
        fe.log_mel_many([a.astype(np.float64)], [np.zeros((80, 100), np.float32)])
    sv = _ext.SenseVoiceFrontend()
    with pytest.raises(ValueError, match="expected"):
        sv.fbank_many([a], [np.zeros((3, 560), np.float32)])
    out = np.zeros((2, 80, 100), np.float32)
    fe.log_mel_many([a, a[::-1].copy()], list(out))
    np.testing.assert_array_equal(out[1], fe.log_mel(a[::-1].copy()))


def test_run_parallel_nested_does_not_deadlock(sigs):
    from asrfront._audio import run_parallel

    outs = [None] * 8
    audio = sigs["speech"]

    def work(i):  # a batch call from inside a pool worker runs inline
        outs[i] = log_mel_batch([audio, audio], n_threads=2)

    run_parallel(work, len(outs), 8)
    for o in outs:
        np.testing.assert_array_equal(o[0], log_mel(audio))


def test_handle_cache_is_bounded(sigs):
    from asrfront import _audio

    audio = sigs["speech"][:16000]
    first = fbank(audio, dither=1.0, seed=0)
    for seed in range(3 * _audio._CACHE_SIZE):
        fbank(audio, dither=1.0, seed=seed)
    assert len(_audio._local.cache) <= _audio._CACHE_SIZE
    # seed 0's handle was evicted; the rebuilt one gives the same result
    np.testing.assert_array_equal(fbank(audio, dither=1.0, seed=0), first)


def test_seed_ignored_without_dither(sigs):
    from asrfront import _audio

    audio = sigs["speech"][:16000]
    fbank(audio, seed=1)
    n = len(_audio._local.cache)
    np.testing.assert_array_equal(fbank(audio, seed=2), fbank(audio, seed=1))
    assert len(_audio._local.cache) == n  # one handle for every seed


@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="needs fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_batch_after_fork(sigs):
    import os

    log_mel_batch([sigs["speech"]] * 2, n_threads=2)  # parent pool exists
    pid = os.fork()
    if pid == 0:  # child: the parent's pool threads are gone; a new pool must be used
        try:
            out = log_mel_batch([sigs["speech"]] * 2, n_threads=2)
            ok = np.array_equal(out[0], log_mel(sigs["speech"]))
        except BaseException:
            ok = False
        os._exit(0 if ok else 1)
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


# ---------------------------------------------------------------------------------------------
# Subnormal intermediates (fpmode.h)
# ---------------------------------------------------------------------------------------------


def _frontend_calls():
    from asrfront import gemma4, lfm3_audio, qwen3_asr, sensevoice

    return {
        "whisper": lambda a: log_mel(a, pad_or_trim=False),
        "qwen3_asr": qwen3_asr.log_mel,
        "sensevoice": sensevoice.fbank,
        "gemma4": gemma4.log_mel,
        "lfm3_audio": lfm3_audio.log_mel,
    }


@pytest.mark.parametrize("name", list(_frontend_calls()))
def test_fp_mode_restored(name):
    """The frontends flush subnormals only while they run: numpy's float32 arithmetic in the
    calling thread still produces subnormals afterwards."""
    tiny = np.full(16000, 1e-23, np.float32)
    _frontend_calls()[name](tiny)
    assert np.float32(1e-30) * np.float32(1e-10) != 0  # 1e-40: subnormal, not flushed


@pytest.mark.skipif(
    platform.machine().lower() not in ("x86_64", "amd64"), reason="flushing is x86-only"
)
@pytest.mark.parametrize("name", list(_frontend_calls()))
def test_tiny_input_not_slow(name):
    """Audio of tiny amplitude makes FFT / mel intermediates subnormal, which x86 computes in
    microcode (8-16x slower before the frontends flushed them)."""
    import time

    fn = _frontend_calls()[name]
    noise = np.random.default_rng(0).standard_normal(16000 * 10).astype(np.float32)

    def best(a: np.ndarray) -> float:
        times = []
        for _ in range(5):
            t = time.perf_counter()
            fn(a)
            times.append(time.perf_counter() - t)
        return min(times)

    normal = best(noise * np.float32(0.1))
    for amp in (1e-20, 1e-25, 1e-36):
        assert best(noise * np.float32(amp)) < 3 * normal, amp
