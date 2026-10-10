"""Compare asrfront against reference frontends.

    uv run --group ref --group bench python bench/compare.py [--json out.json]

Sections:
  1. 30 s, single thread: asrfront vs. openai-whisper (torch), librosa, FunASR WavFrontend
     (torchaudio), torchaudio kaldi.fbank, kaldi-native-fbank
  2. 30 s, reference libraries with all threads (asrfront stays single-threaded)
  3. input-length sweep (1 / 5 / 10 / 30 / 120 s)
  4. batch throughput vs. number of threads

Times are the best of several runs (``best_ms``); libraries that are not installed are
skipped. Reference implementations live in tests/python/reference.py.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "python"))

import asrfront  # noqa: E402
from asrfront import _ext  # noqa: E402

import reference as ref  # noqa: E402
from audio_signals import signals  # noqa: E402

MEL_NPZ = ROOT / "tests" / "data" / "whisper_mel_filters.npz"
SR = 16000
CMVN = (np.full(560, -10, np.float32), np.full(560, 0.3, np.float32))
RESULTS: dict[str, Any] = {}


def best_ms(fn: Callable[[], object], reps: int = 20, min_time: float = 0.5) -> float:
    fn()
    times = []
    start = time.perf_counter()
    while len(times) < reps or time.perf_counter() - start < min_time:
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
        if len(times) >= 1000:
            break
    return min(times) * 1e3


def row(section: str, name: str, ms: float, base: float | None) -> None:
    RESULTS.setdefault(section, {})[name] = ms
    speed = "" if base is None else f"{ms / base:7.1f}x"
    print(f"  {name:46s} {ms:9.2f} ms {speed}")


def set_torch_threads(n: int) -> None:
    try:
        import torch

        torch.set_num_threads(n)
    except ImportError:
        pass


# ---------------------------------------------------------------------------------------------
# Reference implementations (None when the library is missing)
# ---------------------------------------------------------------------------------------------


def whisper_torch() -> Callable[[np.ndarray, bool], np.ndarray] | None:
    try:
        import torch  # noqa: F401
    except ImportError:
        return None
    return lambda a, pad: ref.whisper_log_mel_torch(a, 80, pad, MEL_NPZ)


def whisper_librosa() -> Callable[[np.ndarray, bool], np.ndarray] | None:
    try:
        import librosa
    except ImportError:
        return None
    filters = ref.load_whisper_mel_filters(MEL_NPZ, 80)

    def fn(a: np.ndarray, pad: bool) -> np.ndarray:
        x = ref.whisper_pad_or_trim(a) if pad else a
        s = np.abs(librosa.stft(x, n_fft=400, hop_length=160, window="hann")) ** 2
        log = np.log10(np.maximum(filters @ s[:, :-1], 1e-10))
        return (np.maximum(log, log.max() - 8.0) + 4.0) / 4.0

    return fn


def torchaudio_fbank() -> Callable[[np.ndarray], Any] | None:
    try:
        import torch
        from torchaudio.compliance import kaldi
    except ImportError:
        return None

    def fn(a: np.ndarray) -> Any:
        w = torch.from_numpy(a * np.float32(32768)).unsqueeze(0)
        return kaldi.fbank(w, num_mel_bins=80, dither=0.0, energy_floor=0.0, window_type="hamming")

    return fn


def funasr_wavfrontend() -> Callable[[np.ndarray], np.ndarray] | None:
    """FunASR WavFrontend.forward for one utterance (kaldi.fbank + apply_lfr + apply_cmvn)."""
    fbank = torchaudio_fbank()
    if fbank is None:
        return None
    import torch

    shift, scale = (torch.from_numpy(v) for v in CMVN)

    def fn(a: np.ndarray) -> np.ndarray:
        m = fbank(a)
        t = m.shape[0]
        t_lfr = -(-t // 6)
        m = torch.vstack((m[0].repeat(3, 1), m))
        t += 3
        rows = []
        for i in range(t_lfr):
            if 7 <= t - i * 6:
                rows.append(m[i * 6 : i * 6 + 7].reshape(1, -1))
            else:
                f = m[i * 6 :].reshape(-1)
                for _ in range(7 - (t - i * 6)):
                    f = torch.hstack((f, m[-1]))
                rows.append(f.reshape(1, -1))
        return ((torch.vstack(rows) + shift) * scale).numpy()

    return fn


def kaldi_native_fbank() -> Callable[[np.ndarray], np.ndarray] | None:
    try:
        import kaldi_native_fbank as knf
    except ImportError:
        return None
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.samp_freq = SR
    opts.mel_opts.num_bins = 80
    opts.energy_floor = 0.0

    def fn(a: np.ndarray) -> np.ndarray:
        fb = knf.OnlineFbank(opts)
        fb.accept_waveform(SR, (a * np.float32(32768)).tolist())
        fb.input_finished()
        return np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)])

    return fn


# ---------------------------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------------------------


def bench_single(audio: np.ndarray) -> None:
    set_torch_threads(1)
    sec = "whisper_30s_1thread"
    print("\n[1] Whisper log-mel, 30 s -> (80, 3000), single thread")
    ours = best_ms(lambda: asrfront.whisper.log_mel(audio))
    row(sec, f"asrfront ({_ext.kernel_isa()})", ours, None)
    if (f := whisper_torch()) is not None:
        ms = best_ms(lambda: f(audio, True))
        row(sec, "openai-whisper log_mel_spectrogram (torch)", ms, ours)
    if (f := whisper_librosa()) is not None:
        row(sec, "librosa stft + whisper mel (numpy)", best_ms(lambda: f(audio, True)), ours)

    sec = "sensevoice_30s_1thread"
    print("\n[1] SenseVoice features, 30 s -> (500, 560), single thread")
    ours = best_ms(lambda: asrfront.sensevoice.fbank(audio, cmvn=CMVN))
    row(sec, f"asrfront ({_ext.kernel_isa()})", ours, None)
    if (f := funasr_wavfrontend()) is not None:
        row(sec, "FunASR WavFrontend (torchaudio + LFR/CMVN)", best_ms(lambda: f(audio)), ours)

    sec = "kaldi_fbank_30s_1thread"
    print("\n[1] kaldi fbank only, 30 s -> (2998, 80), single thread")
    raw = best_ms(lambda: asrfront.sensevoice.fbank_raw(audio))
    row(sec, "asrfront fbank_raw", raw, None)
    if (f := torchaudio_fbank()) is not None:
        row(sec, "torchaudio.compliance.kaldi.fbank", best_ms(lambda: f(audio)), raw)
    if (f := kaldi_native_fbank()) is not None:
        row(sec, "kaldi-native-fbank (Python API)", best_ms(lambda: f(audio)), raw)


def bench_multithread_refs(audio: np.ndarray, threads: int) -> None:
    set_torch_threads(threads)
    sec = f"refs_30s_{threads}threads"
    print(f"\n[2] 30 s, reference libraries with {threads} threads (asrfront: 1 thread)")
    ours_w = best_ms(lambda: asrfront.whisper.log_mel(audio))
    ours_s = best_ms(lambda: asrfront.sensevoice.fbank(audio, cmvn=CMVN))
    row(sec, "asrfront whisper.log_mel (1 thread)", ours_w, None)
    if (f := whisper_torch()) is not None:
        row(sec, "openai-whisper (torch)", best_ms(lambda: f(audio, True)), ours_w)
    row(sec, "asrfront sensevoice.fbank (1 thread)", ours_s, None)
    if (f := funasr_wavfrontend()) is not None:
        row(sec, "FunASR WavFrontend (torchaudio)", best_ms(lambda: f(audio)), ours_s)
    set_torch_threads(1)


def bench_lengths(speech: np.ndarray) -> None:
    set_torch_threads(1)
    print("\n[3] Input length sweep, single thread (whisper without pad_or_trim)")
    long = np.tile(speech, 12)
    wt, fa = whisper_torch(), funasr_wavfrontend()
    for sec_len in (1, 5, 10, 30, 120):
        a = np.ascontiguousarray(long[: sec_len * SR])
        sec = f"length_{sec_len}s"
        print(f"  -- {sec_len} s")
        w = best_ms(lambda a=a: asrfront.whisper.log_mel(a, pad_or_trim=False))
        row(sec, "asrfront whisper.log_mel", w, None)
        if wt is not None:
            row(sec, "openai-whisper (torch)", best_ms(lambda a=a: wt(a, False)), w)
        s = best_ms(lambda a=a: asrfront.sensevoice.fbank(a, cmvn=CMVN))
        row(sec, "asrfront sensevoice.fbank", s, None)
        if fa is not None:
            row(sec, "FunASR WavFrontend (torchaudio)", best_ms(lambda a=a: fa(a)), s)


def bench_batch(audio: np.ndarray, batch: int) -> None:
    audios = [audio] * batch
    n_cpu = os.cpu_count() or 1
    counts = sorted({1, 2, 4, 8, n_cpu})
    print(f"\n[4] Batch of {batch} x 30 s, time per batch vs. n_threads")
    for name, fn in (
        ("whisper.log_mel_batch", lambda n: asrfront.whisper.log_mel_batch(audios, n_threads=n)),
        ("sensevoice.fbank_batch", lambda n: asrfront.sensevoice.fbank_batch(audios, n_threads=n)),
    ):
        sec = f"batch{batch}_{name}"
        base = None
        for n in counts:
            ms = best_ms(lambda fn=fn, n=n: fn(n), reps=5)
            base = base or ms
            RESULTS.setdefault(sec, {})[str(n)] = ms
            print(f"  {name:24s} {n:3d} threads {ms:8.2f} ms  ({base / ms:4.1f}x)")


def environment() -> dict[str, str]:
    import importlib.metadata as md

    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "kernels": _ext.kernel_isa(),
        "cpus": str(os.cpu_count()),
    }
    for pkg in ("asrfront", "numpy", "torch", "torchaudio", "librosa", "kaldi-native-fbank"):
        try:
            env[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--json", type=Path, help="also write the results as JSON")
    args = ap.parse_args()
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")

    env = environment()
    print(", ".join(f"{k} {v}" for k, v in env.items()))
    print("(right column: time relative to asrfront; higher = asrfront faster)")
    sigs = signals()
    audio = np.ascontiguousarray(sigs["speech_35s"][: 30 * SR])
    bench_single(audio)
    bench_multithread_refs(audio, os.cpu_count() or 1)
    bench_lengths(np.asarray(sigs["speech"]))
    bench_batch(audio, args.batch)
    if args.json:
        args.json.write_text(json.dumps({"env": env, "results": RESULTS}, indent=1))


if __name__ == "__main__":
    main()
