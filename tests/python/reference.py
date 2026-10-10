"""Float64 numpy reference implementations ("ground truth") for asrfront.

- Whisper: port of ``whisper/audio.py`` from openai/whisper (MIT) with the mel filters of
  ``librosa.filters.mel`` (ISC).
- SenseVoice: port of ``torchaudio.compliance.kaldi.fbank`` (BSD-2-Clause) plus
  ``apply_lfr`` / ``apply_cmvn`` / ``load_cmvn`` from FunASR's ``WavFrontend`` (MIT).

Every stage is exposed separately so intermediate values (window, filters, power spectrum,
...) can be compared when debugging the C implementation. Everything is computed in float64;
the original libraries compute in float32, and ``test_reference.py`` checks the two agree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------------------------
# Whisper
# ---------------------------------------------------------------------------------------------

WHISPER_N_FFT = 400
WHISPER_HOP = 160
WHISPER_N_SAMPLES = 30 * SAMPLE_RATE  # 480000


def whisper_pad_or_trim(audio: np.ndarray, length: int = WHISPER_N_SAMPLES) -> np.ndarray:
    if audio.shape[0] > length:
        return audio[:length]
    return np.pad(audio, (0, length - audio.shape[0]))


def hann_window_periodic(n: int) -> np.ndarray:
    """``torch.hann_window(n)`` (periodic=True)."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def _hz_to_mel_slaney(f: np.ndarray) -> np.ndarray:
    f = np.asanyarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_part = min_log_mel + np.log(np.maximum(f, min_log_hz) / min_log_hz) / logstep
    return np.where(f >= min_log_hz, log_part, mels)


def _mel_to_hz_slaney(m: np.ndarray) -> np.ndarray:
    m = np.asanyarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)


def slaney_mel_filters(
    sr: int = SAMPLE_RATE,
    n_fft: int = WHISPER_N_FFT,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """``librosa.filters.mel(sr, n_fft, n_mels)`` (htk=False, norm='slaney') in float64.

    Returns ``(n_mels, n_fft // 2 + 1)``.
    """
    if fmax is None:
        fmax = sr / 2.0
    fftfreqs = np.linspace(0, sr / 2.0, 1 + n_fft // 2)
    mel_f = _mel_to_hz_slaney(
        np.linspace(_hz_to_mel_slaney(fmin), _hz_to_mel_slaney(fmax), n_mels + 2)
    )
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    weights = np.zeros((n_mels, 1 + n_fft // 2))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    return weights * enorm[:, None]


def whisper_stft_power(audio: np.ndarray) -> np.ndarray:
    """``|torch.stft(audio, 400, 160, hann, center=True, reflect)|**2`` minus the last frame.

    Returns ``(n_fft // 2 + 1, n_samples // hop)``.
    """
    pad = WHISPER_N_FFT // 2
    if audio.shape[0] <= pad:
        # torch's reflect padding requires pad < input length.
        raise ValueError(f"whisper STFT needs more than {pad} samples, got {audio.shape[0]}")
    x = np.pad(audio.astype(np.float64), (pad, pad), mode="reflect")
    n_frames = 1 + (x.shape[0] - WHISPER_N_FFT) // WHISPER_HOP
    idx = np.arange(WHISPER_N_FFT)[None, :] + WHISPER_HOP * np.arange(n_frames)[:, None]
    frames = x[idx] * hann_window_periodic(WHISPER_N_FFT)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    return power[:-1].T


def whisper_log_mel_stages(
    audio: np.ndarray,
    n_mels: int = 80,
    pad_or_trim: bool = True,
    filters: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    audio = np.asarray(audio, dtype=np.float64)
    if pad_or_trim:
        audio = whisper_pad_or_trim(audio)
    if filters is None:
        filters = slaney_mel_filters(n_mels=n_mels)
    power = whisper_stft_power(audio)
    mel = filters @ power
    log = np.log10(np.maximum(mel, 1e-10))
    clamped = np.maximum(log, log.max() - 8.0)
    out = (clamped + 4.0) / 4.0
    return {"audio": audio, "power": power, "filters": filters, "mel": mel, "log": log, "out": out}


def whisper_log_mel(audio: np.ndarray, n_mels: int = 80, pad_or_trim: bool = True) -> np.ndarray:
    """Whisper log-mel spectrogram, ``(n_mels, n_frames)``."""
    return whisper_log_mel_stages(audio, n_mels, pad_or_trim)["out"]


def load_whisper_mel_filters(path: str | Path, n_mels: int) -> np.ndarray:
    """Load ``mel_{n_mels}`` from whisper's ``assets/mel_filters.npz`` (float32)."""
    with np.load(path, allow_pickle=False) as f:
        return f[f"mel_{n_mels}"]


# ---------------------------------------------------------------------------------------------
# Kaldi fbank (torchaudio.compliance.kaldi.fbank) as configured by FunASR WavFrontend
# ---------------------------------------------------------------------------------------------

KALDI_FRAME_LENGTH = 400  # 25 ms
KALDI_FRAME_SHIFT = 160  # 10 ms
KALDI_PADDED = 512  # round_to_power_of_two
KALDI_PREEMPH = 0.97
FLT_EPSILON = float(np.finfo(np.float32).eps)  # torch.finfo(torch.float).eps


def hamming_window_symmetric(n: int) -> np.ndarray:
    """``torch.hamming_window(n, periodic=False, alpha=0.54, beta=0.46)``."""
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))


def _kaldi_mel(f: np.ndarray) -> np.ndarray:
    return 1127.0 * np.log(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def kaldi_mel_banks(
    n_mels: int = 80,
    padded: int = KALDI_PADDED,
    sr: int = SAMPLE_RATE,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
) -> np.ndarray:
    """``torchaudio.compliance.kaldi.get_mel_banks`` (no VTLN) plus the zero Nyquist column.

    Returns ``(n_mels, padded // 2 + 1)``.
    """
    nyquist = 0.5 * sr
    if high_freq <= 0.0:
        high_freq += nyquist
    num_fft_bins = padded // 2
    fft_bin_width = sr / padded
    mel_low = _kaldi_mel(low_freq)
    mel_high = _kaldi_mel(high_freq)
    mel_delta = (mel_high - mel_low) / (n_mels + 1)

    b = np.arange(n_mels, dtype=np.float64)[:, None]
    left = mel_low + b * mel_delta
    center = mel_low + (b + 1.0) * mel_delta
    right = mel_low + (b + 2.0) * mel_delta
    mel = _kaldi_mel(fft_bin_width * np.arange(num_fft_bins))[None, :]
    up = (mel - left) / (center - left)
    down = (right - mel) / (right - center)
    banks = np.maximum(0.0, np.minimum(up, down))
    return np.pad(banks, ((0, 0), (0, 1)))


def kaldi_num_frames(n_samples: int) -> int:
    if n_samples < KALDI_FRAME_LENGTH:
        return 0
    return 1 + (n_samples - KALDI_FRAME_LENGTH) // KALDI_FRAME_SHIFT


def kaldi_fbank_stages(
    audio: np.ndarray,
    n_mels: int = 80,
    scale: float = 32768.0,
    banks: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Kaldi fbank with FunASR's settings: hamming, dither=0, snip_edges, no energy.

    ``audio`` is a float waveform in [-1, 1]; it is multiplied by ``scale`` first
    (FunASR's ``waveform * (1 << 15)``). ``banks`` overrides the mel filter bank.
    """
    x = np.asarray(audio, dtype=np.float64) * scale
    n_frames = kaldi_num_frames(x.shape[0])
    idx = np.arange(KALDI_FRAME_LENGTH)[None, :] + KALDI_FRAME_SHIFT * np.arange(n_frames)[:, None]
    frames = x[idx].reshape(n_frames, KALDI_FRAME_LENGTH)

    frames = frames - frames.mean(axis=1, keepdims=True)  # remove_dc_offset
    prev = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)  # replicate-pad by one
    frames = frames - KALDI_PREEMPH * prev
    frames = frames * hamming_window_symmetric(KALDI_FRAME_LENGTH)
    padded = np.pad(frames, ((0, 0), (0, KALDI_PADDED - KALDI_FRAME_LENGTH)))

    power = np.abs(np.fft.rfft(padded, axis=1)) ** 2
    if banks is None:
        banks = kaldi_mel_banks(n_mels)
    mel = power @ banks.T
    out = np.log(np.maximum(mel, FLT_EPSILON))
    return {"frames": frames, "power": power, "banks": banks, "mel": mel, "out": out}


def kaldi_fbank(audio: np.ndarray, n_mels: int = 80) -> np.ndarray:
    """Raw kaldi log-mel fbank, ``(T, n_mels)``."""
    return kaldi_fbank_stages(audio, n_mels)["out"]


# ---------------------------------------------------------------------------------------------
# FunASR LFR / CMVN
# ---------------------------------------------------------------------------------------------


def apply_lfr(inputs: np.ndarray, lfr_m: int = 7, lfr_n: int = 6) -> np.ndarray:
    """FunASR ``apply_lfr``: stack ``lfr_m`` frames every ``lfr_n`` frames."""
    t, dim = inputs.shape
    if t == 0:
        return np.zeros((0, dim * lfr_m), dtype=inputs.dtype)
    t_lfr = int(np.ceil(t / lfr_n))
    left = (lfr_m - 1) // 2
    inputs = np.vstack([np.repeat(inputs[:1], left, axis=0), inputs])
    t += left
    out = []
    for i in range(t_lfr):
        if lfr_m <= t - i * lfr_n:
            out.append(inputs[i * lfr_n : i * lfr_n + lfr_m].reshape(-1))
        else:  # last LFR frame: pad with the final input frame
            num_padding = lfr_m - (t - i * lfr_n)
            frame = inputs[i * lfr_n :].reshape(-1)
            out.append(np.concatenate([frame] + [inputs[-1]] * num_padding))
    return np.vstack(out)


def apply_cmvn(inputs: np.ndarray, shift: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """FunASR ``apply_cmvn``: ``(x + shift) * scale``."""
    dim = inputs.shape[1]
    return (inputs + shift[:dim]) * scale[:dim]


def load_cmvn(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """FunASR ``load_cmvn``: parse ``<AddShift>`` / ``<Rescale>`` from a Kaldi nnet ``am.mvn``."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    shift: list[str] = []
    scale: list[str] = []
    for i, line in enumerate(lines):
        item = line.split()
        if not item:
            continue
        if item[0] in ("<AddShift>", "<Rescale>"):
            nxt = lines[i + 1].split()
            if nxt and nxt[0] == "<LearnRateCoef>":
                values = nxt[3 : len(nxt) - 1]
                if item[0] == "<AddShift>":
                    shift = values
                else:
                    scale = values
    return np.array(shift, dtype=np.float32), np.array(scale, dtype=np.float32)


def sensevoice_fbank(
    audio: np.ndarray,
    cmvn: tuple[np.ndarray, np.ndarray] | None = None,
    n_mels: int = 80,
    lfr_m: int = 7,
    lfr_n: int = 6,
) -> np.ndarray:
    """FunASR ``WavFrontend`` features, ``(ceil(T / lfr_n), n_mels * lfr_m)``."""
    feats = apply_lfr(kaldi_fbank(audio, n_mels), lfr_m, lfr_n)
    if cmvn is not None:
        feats = apply_cmvn(feats, np.asarray(cmvn[0], np.float64), np.asarray(cmvn[1], np.float64))
    return feats
