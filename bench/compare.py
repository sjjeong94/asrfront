"""Compare asrfront against reference frontends on 30 s of audio.

    uv run --group ref --group bench python bench/compare.py [--threads N] [--batch B]

Single-input timings are single-threaded (torch/numba set to 1 thread) and report the best
of several runs. Libraries that are not installed are skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "python"))

import asrfront  # noqa: E402
from asrfront import _ext  # noqa: E402

import reference as ref  # noqa: E402
from audio_signals import signals  # noqa: E402

MEL_NPZ = ROOT / "tests" / "data" / "whisper_mel_filters.npz"


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


def row(name: str, ms: float, base: float | None) -> None:
    speed = "" if base is None else f"{ms / base:7.1f}x"
    print(f"  {name:44s} {ms:8.2f} ms {speed}")


def bench_whisper(audio: np.ndarray) -> None:
    print("\nWhisper log-mel (80 x 3000, 30 s)")
    ours = best_ms(lambda: asrfront.whisper.log_mel(audio))
    row(f"asrfront ({_ext.kernel_isa()})", ours, None)
    try:
        import torch

        torch.set_num_threads(1)
        row(
            "openai-whisper log_mel_spectrogram (torch)",
            best_ms(lambda: ref.whisper_log_mel_torch(audio, 80, True, MEL_NPZ)),
            ours,
        )
    except ImportError:
        print("  (torch not installed: uv run --group ref ...)")
    try:
        import librosa

        filters = ref.load_whisper_mel_filters(MEL_NPZ, 80)

        def librosa_log_mel() -> np.ndarray:
            x = ref.whisper_pad_or_trim(audio)
            s = np.abs(librosa.stft(x, n_fft=400, hop_length=160, window="hann")) ** 2
            log = np.log10(np.maximum(filters @ s[:, :-1], 1e-10))
            return (np.maximum(log, log.max() - 8.0) + 4.0) / 4.0

        row("librosa stft + whisper mel (numpy)", best_ms(librosa_log_mel), ours)
    except ImportError:
        print("  (librosa not installed: uv run --group bench ...)")


def bench_sensevoice(audio: np.ndarray) -> None:
    print("\nSenseVoice fbank (kaldi fbank + LFR 7/6 + CMVN -> 500 x 560, 30 s)")
    cmvn = (np.full(560, -10, np.float32), np.full(560, 0.3, np.float32))
    ours = best_ms(lambda: asrfront.sensevoice.fbank(audio, cmvn=cmvn))
    row(f"asrfront ({_ext.kernel_isa()})", ours, None)
    raw = best_ms(lambda: asrfront.sensevoice.fbank_raw(audio))
    row("asrfront fbank_raw (kaldi fbank only)", raw, None)
    try:
        import torch
        from torchaudio.compliance import kaldi

        torch.set_num_threads(1)
        shift, scale = (torch.from_numpy(v) for v in cmvn)

        def funasr_style() -> np.ndarray:
            w = torch.from_numpy(audio * np.float32(32768)).unsqueeze(0)
            m = kaldi.fbank(w, num_mel_bins=80, dither=0.0, energy_floor=0.0, window_type="hamming")
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

        row("FunASR WavFrontend (torchaudio + LFR/CMVN)", best_ms(funasr_style), ours)
        row(
            "torchaudio.compliance.kaldi.fbank only",
            best_ms(
                lambda: kaldi.fbank(
                    torch.from_numpy(audio * np.float32(32768)).unsqueeze(0),
                    num_mel_bins=80,
                    dither=0.0,
                    energy_floor=0.0,
                    window_type="hamming",
                )
            ),
            raw,
        )
    except ImportError:
        print("  (torch/torchaudio not installed: uv run --group ref ...)")
    try:
        import kaldi_native_fbank as knf

        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0
        opts.frame_opts.window_type = "hamming"
        opts.frame_opts.samp_freq = 16000
        opts.mel_opts.num_bins = 80
        opts.energy_floor = 0.0
        scaled = (audio * np.float32(32768)).tolist()

        def knf_fbank() -> np.ndarray:
            fb = knf.OnlineFbank(opts)
            fb.accept_waveform(16000, scaled)
            fb.input_finished()
            return np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)])

        row("kaldi-native-fbank (C++) fbank only", best_ms(knf_fbank), raw)
    except ImportError:
        print("  (kaldi-native-fbank not installed: uv run --group bench ...)")


def bench_batch(audio: np.ndarray, batch: int, threads: int | None) -> None:
    audios = [audio] * batch
    print(f"\nBatch of {batch} x 30 s, n_threads={threads or 'all'}")
    one = best_ms(lambda: asrfront.whisper.log_mel_batch(audios, n_threads=1), reps=3)
    many = best_ms(lambda: asrfront.whisper.log_mel_batch(audios, n_threads=threads), reps=3)
    print(
        f"  whisper.log_mel_batch      1 thread {one:7.2f} ms -> {many:7.2f} ms ({one / many:.1f}x)"
    )
    one = best_ms(lambda: asrfront.sensevoice.fbank_batch(audios, n_threads=1), reps=3)
    many = best_ms(lambda: asrfront.sensevoice.fbank_batch(audios, n_threads=threads), reps=3)
    print(
        f"  sensevoice.fbank_batch     1 thread {one:7.2f} ms -> {many:7.2f} ms ({one / many:.1f}x)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")

    audio = np.ascontiguousarray(signals()["speech_35s"][: 30 * 16000])
    print(f"asrfront {asrfront.__version__}, kernels: {_ext.kernel_isa()}, cpus: {os.cpu_count()}")
    print("(right column: time relative to asrfront; higher = asrfront faster)")
    bench_whisper(audio)
    bench_sensevoice(audio)
    bench_batch(audio, args.batch, args.threads)


if __name__ == "__main__":
    main()
